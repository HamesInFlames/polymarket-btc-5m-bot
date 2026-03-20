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

from src.config import (
    LIVE_TRADING,
    ENTRY_WINDOW_START,
    ENTRY_WINDOW_END,
    MIN_EDGE,
    KELLY_MULTIPLIER,
)
from src.market_reader import discover_active_btc_5m_markets, MarketRound
from src.price_oracle import get_btc_price
from src.strategy import evaluate_round, get_bankroll, record_bet_result
from src.risk_manager import RiskManager
from src.trader import place_buy_order
from src.bot_state import state as dashboard, TradeEntry, install_log_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("web3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

install_log_handler()

risk = RiskManager()

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.warning("Shutdown signal received - finishing current cycle then exiting")
    _shutdown = True


import threading as _threading
if _threading.current_thread() is _threading.main_thread():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def print_banner():
    mode = "LIVE" if LIVE_TRADING else "DRY RUN (paper)"
    br = get_bankroll()
    print()
    print("=" * 62)
    print("   Polymarket BTC 5-Min UNRESTRICTED Kelly Bot")
    print("=" * 62)
    print(f"   Mode:                {mode}")
    print(f"   Bankroll:            ${br.current_balance:.2f} (started ${br.starting_balance:.2f})")
    print(f"   Kelly multiplier:    {KELLY_MULTIPLIER:.0%} Kelly (FULL)")
    print(f"   Bet sizing:          UNRESTRICTED (Kelly decides)")
    print(f"   Profit reinvestment: 100% (max compounding)")
    print(f"   Trade limits:        NONE (runs forever)")
    print(f"   Min edge threshold:  {MIN_EDGE*100:.1f}%")
    print(f"   Entry window:        {ENTRY_WINDOW_START}s -> {ENTRY_WINDOW_END}s before close")
    if br.num_bets > 0:
        print(f"   Lifetime bets:       {br.num_bets} ({br.win_rate*100:.1f}% win rate)")
        print(f"   Lifetime P&L:        ${br.total_profit:+.2f} ({br.roi*100:+.1f}% ROI)")
        print(f"   Current drawdown:    {br.drawdown*100:.1f}%")
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
    trade_count = 0

    mode = "LIVE" if LIVE_TRADING else "DRY RUN"
    dashboard.update_bot_status(True, 0, 0, 0, mode)

    while not _shutdown:
        cycle += 1
        dashboard.update_bot_status(True, cycle, trade_count, 0, mode)

        risk.pre_trade_check()
        dashboard.update_risk_stats(risk.stats_summary())

        log.info("-- Cycle %d  [trades: %d] -------------------------",
                 cycle, trade_count)

        try:
            oracle = get_btc_price()
            dashboard.update_btc_price(oracle)
        except Exception:
            pass

        try:
            rounds = discover_active_btc_5m_markets()
        except Exception as e:
            dashboard.set_error(f"Market discovery failed: {e}")
            rounds = []

        if not rounds:
            log.info("No active BTC 5-min markets found - retrying in 10s")
            dashboard.clear_rounds()
            _sleep(10)
            continue

        dashboard.update_rounds(rounds)
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

            log.info(
                ">>> EXECUTING: %s on %s | P(win)=%.3f edge=%.4f | "
                "Kelly=%.4f -> $%.2f (%.1f contracts @ $%.3f) | EV=$%.4f",
                signal.action, rnd.condition_id[:12],
                signal.confidence, signal.edge,
                signal.kelly_fraction, signal.bet_dollars,
                signal.size, signal.price, signal.expected_profit,
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
                trade_count += 1

                pnl = _estimate_pnl(signal)
                won = pnl > 0

                if won:
                    record_bet_result(True, signal.bet_dollars, signal.size * 1.0)
                else:
                    record_bet_result(False, signal.bet_dollars)

                risk.record_pnl(
                    won=won, pnl=pnl, direction=signal.direction,
                    bet_dollars=signal.bet_dollars,
                    kelly_fraction=signal.kelly_fraction,
                    win_prob=signal.confidence,
                )

                br = get_bankroll()
                dashboard.add_trade(TradeEntry(
                    timestamp=time.time(),
                    direction=signal.direction,
                    action=signal.action,
                    price=signal.price,
                    size=signal.size,
                    edge=signal.edge,
                    confidence=signal.confidence,
                    pnl=pnl,
                    won=won,
                    btc_price=signal.btc_current,
                    condition_id=rnd.condition_id[:16],
                    reason=signal.reason,
                ))
                dashboard.update_risk_stats(risk.stats_summary())
                dashboard.add_log(
                    "INFO",
                    f"Trade #{trade_count}: {signal.action} "
                    f"edge={signal.edge:.4f} kelly={signal.kelly_fraction:.4f} "
                    f"bet=${signal.bet_dollars:.2f} pnl=${pnl:.4f} "
                    f"bankroll=${br.current_balance:.2f}",
                )

                log.info(
                    "Trade #%d | PnL=$%.4f | Bankroll=$%.2f | "
                    "Drawdown=%.1f%% | Total PnL=$%.4f",
                    trade_count, pnl,
                    br.current_balance, br.drawdown * 100,
                    risk.total_pnl(),
                )

        _print_status(cycle, rounds)
        dashboard.update_risk_stats(risk.stats_summary())

        sleep_time = _calculate_sleep(rounds)
        log.debug("Sleeping %.1fs until next check", sleep_time)
        _sleep(sleep_time)

    dashboard.update_bot_status(False, cycle, trade_count, 0, mode)
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
    dashboard.update_btc_price(oracle)
    price = oracle.get("chainlink") or oracle.get("median")
    if price:
        tracker.set_opening_price(rnd.condition_id, price)


def _estimate_pnl(signal) -> float:
    """
    In dry-run mode we simulate the outcome probabilistically.
    Each bet either wins (profit = contracts * (1 - price)) or
    loses (loss = bet_dollars). We use the estimated win_prob
    to determine the EV, but for tracking we pick the expected value.
    For live mode, the actual PnL comes from contract resolution.
    """
    if not LIVE_TRADING:
        win_profit = signal.size * (1.0 - signal.price)
        lose_cost = signal.bet_dollars
        ev = signal.confidence * win_profit - (1.0 - signal.confidence) * lose_cost
        return ev
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
    br = get_bankroll()
    print()
    print("=" * 58)
    print("  FINAL SESSION STATS")
    print("=" * 58)
    print(f"  Total trades:       {stats['total_trades']}")
    print(f"  Wins / Losses:      {stats['wins']} / {stats['losses']}")
    print(f"  Win rate:           {stats['win_rate']*100:.1f}%")
    print(f"  Profit factor:      {stats['profit_factor']:.2f}")
    print(f"  Avg edge:           {stats['avg_edge']*100:.2f}%")
    print(f"  Avg Kelly fraction: {stats['avg_kelly']*100:.2f}%")
    print(f"  Daily PnL:          ${stats['daily_pnl']:.4f}")
    print(f"  Total PnL:          ${stats['total_pnl']:.4f}")
    print(f"  Consecutive losses: {stats['consecutive_losses']}")
    print("  " + "-" * 40)
    print(f"  Bankroll:           ${br.current_balance:.2f}")
    print(f"  Starting balance:   ${br.starting_balance:.2f}")
    print(f"  Peak balance:       ${br.peak_balance:.2f}")
    print(f"  Lifetime P&L:       ${br.total_profit:+.2f}")
    print(f"  ROI:                {br.roi*100:+.1f}%")
    print(f"  Drawdown:           {br.drawdown*100:.1f}%")
    print(f"  Total wagered:      ${br.total_wagered:.2f}")
    print(f"  Paused:             {stats['paused']}")
    if stats["pause_reason"]:
        print(f"  Pause reason:       {stats['pause_reason']}")
    print("=" * 58)
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
