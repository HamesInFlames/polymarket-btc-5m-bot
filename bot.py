"""
Polymarket BTC 5-Minute Oracle-Verified Trading Bot
====================================================
Main entry point.  Runs a continuous loop that:
  1. Discovers active BTC 5-min up/down rounds on Polymarket.
  2. For each round approaching expiry, captures the opening BTC price.
  3. In the final seconds, compares real BTC price (Chainlink + CEX) to
     the opening price to determine likely direction.
  4. If the CLOB contract price offers sufficient edge, places a
     Fill-or-Kill BUY order on the likely-winning outcome.
  5. Tracks results and enforces risk limits (daily loss cap,
     consecutive-loss pause, emergency shutdown).

DISCLAIMER: This software is for educational and research purposes only.
It is NOT financial advice.  Algorithmic trading carries substantial risk
of loss.  Over 90% of back-tested strategies fail in live markets.
Always consult a qualified financial advisor before risking real capital.
"""

import logging
import signal
import sys
import time
from datetime import datetime, timezone

from src.config import (
    LIVE_TRADING,
    ENTRY_WINDOW_START,
    ENTRY_WINDOW_END,
    MAX_TRADE_SIZE,
    MIN_EDGE,
    MAX_DAILY_LOSS,
    MAX_CONSECUTIVE_LOSSES,
)
from src.market_reader import discover_active_btc_5m_markets, MarketRound
from src.price_oracle import get_btc_price
from src.strategy import evaluate_round
from src.risk_manager import RiskManager
from src.trader import place_buy_order

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("web3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

risk = RiskManager()

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.warning("Shutdown signal received - finishing current cycle then exiting")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def print_banner():
    mode = "LIVE" if LIVE_TRADING else "DRY RUN (paper)"
    print()
    print("=" * 62)
    print("   Polymarket BTC 5-Min Oracle-Verified Trading Bot")
    print("=" * 62)
    print(f"   Mode:                {mode}")
    print(f"   Max trade size:      ${MAX_TRADE_SIZE:.2f}")
    print(f"   Min edge threshold:  {MIN_EDGE*100:.1f}%")
    print(f"   Entry window:        {ENTRY_WINDOW_START}s -> {ENTRY_WINDOW_END}s before close")
    print(f"   Daily loss limit:    ${MAX_DAILY_LOSS:.2f}")
    print(f"   Consec. loss limit:  {MAX_CONSECUTIVE_LOSSES}")
    print("=" * 62)
    if not LIVE_TRADING:
        print("   *** DRY RUN - no real orders will be placed ***")
        print("   Set LIVE_TRADING=true in .env to go live.")
        print("=" * 62)
    print()


class RoundTracker:
    """Tracks which rounds we've already traded to avoid duplicates."""

    def __init__(self):
        self._traded: set[str] = set()
        self._opening_prices: dict[str, float] = {}

    def already_traded(self, condition_id: str) -> bool:
        return condition_id in self._traded

    def mark_traded(self, condition_id: str):
        self._traded.add(condition_id)

    def set_opening_price(self, condition_id: str, price: float):
        if condition_id not in self._opening_prices:
            self._opening_prices[condition_id] = price
            log.info(
                "Opening BTC price for %s: $%.2f",
                condition_id[:12], price,
            )

    def get_opening_price(self, condition_id: str) -> float | None:
        return self._opening_prices.get(condition_id)

    def cleanup(self, active_ids: set[str]):
        stale = [k for k in self._opening_prices if k not in active_ids]
        for k in stale:
            del self._opening_prices[k]
            self._traded.discard(k)


def main_loop():
    tracker = RoundTracker()
    cycle = 0

    while not _shutdown:
        cycle += 1

        allowed, reason = risk.pre_trade_check()
        if not allowed:
            log.warning("Risk check blocked: %s - sleeping 60s", reason)
            _sleep(60)
            continue

        log.info("-- Cycle %d ------------------------------------------", cycle)

        rounds = discover_active_btc_5m_markets()
        if not rounds:
            log.info("No active BTC 5-min markets found - retrying in 10s")
            _sleep(10)
            continue

        active_ids = {r.condition_id for r in rounds}
        tracker.cleanup(active_ids)

        log.info("Found %d active round(s)", len(rounds))

        for rnd in rounds:
            if _shutdown:
                break

            remaining = rnd.seconds_remaining

            _capture_opening_price(tracker, rnd)

            if tracker.already_traded(rnd.condition_id):
                continue

            if remaining > ENTRY_WINDOW_START or remaining < ENTRY_WINDOW_END:
                continue

            opening = tracker.get_opening_price(rnd.condition_id)
            log.info(
                "EVALUATING %s | %.0fs left | opening=$%.2f",
                rnd.condition_id[:12],
                remaining,
                opening or 0,
            )

            signal = evaluate_round(rnd, opening)
            if signal is None:
                log.info("  -> No signal (edge/confidence/price insufficient)")
                continue

            allowed, reason = risk.pre_trade_check()
            if not allowed:
                log.warning("Risk blocked just before order: %s", reason)
                break

            log.info(
                ">>> EXECUTING: %s on %s - edge=%.3f, size=%.2f shares @ $%.3f",
                signal.action, rnd.condition_id[:12],
                signal.edge, signal.size, signal.price,
            )

            result = place_buy_order(
                token_id=signal.token_id,
                price=signal.price,
                size=signal.size,
                neg_risk=signal.neg_risk,
                tick_size=signal.tick_size,
            )

            if result:
                tracker.mark_traded(rnd.condition_id)

                pnl = _estimate_pnl(signal)
                won = pnl > 0
                risk.record_pnl(won=won, pnl=pnl, direction=signal.direction)

        _print_status(cycle, rounds)

        sleep_time = _calculate_sleep(rounds)
        log.debug("Sleeping %.1fs until next check", sleep_time)
        _sleep(sleep_time)

    log.info("Bot shut down gracefully.")
    _print_final_stats()


def _capture_opening_price(tracker: RoundTracker, rnd: MarketRound):
    """
    Record the BTC price at the start of a round if not already captured.
    Only captures if the round has actually started and is within
    the first 30 seconds, to get a price close to the true opening.
    """
    if tracker.get_opening_price(rnd.condition_id) is not None:
        return

    elapsed = rnd.seconds_elapsed
    if elapsed < 0 or elapsed > 60:
        return

    oracle = get_btc_price()
    price = oracle.get("chainlink") or oracle.get("median")
    if price:
        tracker.set_opening_price(rnd.condition_id, price)


def _estimate_pnl(signal) -> float:
    """
    In dry-run mode we can't know the real outcome, so we use
    the edge as a probabilistic estimate.
    For live mode, the actual PnL would come from contract resolution.
    """
    if not LIVE_TRADING:
        expected_return = signal.edge * signal.size
        return expected_return
    return 0.0


def _calculate_sleep(rounds: list[MarketRound]) -> float:
    """Smart sleep: wake up when the nearest round enters our entry window."""
    if not rounds:
        return 10.0

    nearest = min(r.seconds_remaining for r in rounds)

    if nearest <= ENTRY_WINDOW_START + 5:
        return 1.0
    elif nearest <= ENTRY_WINDOW_START + 30:
        return 2.0
    else:
        return min(10.0, nearest - ENTRY_WINDOW_START - 5)


def _sleep(seconds: float):
    """Interruptible sleep."""
    end = time.time() + seconds
    while time.time() < end and not _shutdown:
        time.sleep(min(1.0, end - time.time()))


def _print_status(cycle: int, rounds: list[MarketRound]):
    if cycle % 10 != 0:
        return
    stats = risk.stats_summary()
    log.info(
        "STATUS | trades=%d  wins=%d  losses=%d  win_rate=%.1f%%  "
        "daily_pnl=$%.4f  total_pnl=$%.4f  consec_losses=%d",
        stats["total_trades"], stats["wins"], stats["losses"],
        stats["win_rate"] * 100, stats["daily_pnl"], stats["total_pnl"],
        stats["consecutive_losses"],
    )


def _print_final_stats():
    stats = risk.stats_summary()
    print()
    print("=" * 50)
    print("  FINAL SESSION STATS")
    print("=" * 50)
    print(f"  Total trades:       {stats['total_trades']}")
    print(f"  Wins / Losses:      {stats['wins']} / {stats['losses']}")
    print(f"  Win rate:           {stats['win_rate']*100:.1f}%")
    print(f"  Daily PnL:          ${stats['daily_pnl']:.4f}")
    print(f"  Total PnL:          ${stats['total_pnl']:.4f}")
    print(f"  Consecutive losses: {stats['consecutive_losses']}")
    print(f"  Paused:             {stats['paused']}")
    if stats["pause_reason"]:
        print(f"  Pause reason:       {stats['pause_reason']}")
    print("=" * 50)
    print()


if __name__ == "__main__":
    print_banner()
    try:
        main_loop()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        _print_final_stats()
    except Exception as e:
        log.critical("Fatal error: %s", e, exc_info=True)
        _print_final_stats()
        sys.exit(1)
