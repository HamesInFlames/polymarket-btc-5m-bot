"""
Polymarket BTC 5-Minute Oracle-Verified Trading Bot
====================================================
Main entry point.  Runs a continuous loop that:
  1. Discovers active BTC 5-min up/down rounds on Polymarket.
  2. Captures the opening BTC price for each round (from market
     description first, then Chainlink as close to round start
     as possible).
  3. In the final minutes, compares real BTC price (Chainlink + CEX)
     to the opening price to determine likely direction.
  4. If the CLOB contract price offers sufficient edge, places a
     Fill-And-Kill BUY order on the likely-winning outcome.
  5. After the round ends, checks the ACTUAL resolution from
     Polymarket's Gamma API (or Chainlink fallback) — no random
     simulation — and updates bankroll from real outcomes.

DISCLAIMER: This software is for educational and research purposes only.
It is NOT financial advice.  Algorithmic trading carries substantial risk
of loss.  Always consult a qualified financial advisor before risking
real capital.
"""

import logging
import signal
import sys
import time
from dataclasses import dataclass, field

from src.config import (
    LIVE_TRADING,
    ENTRY_WINDOW_START,
    ENTRY_WINDOW_END,
    MIN_EDGE,
    KELLY_MULTIPLIER,
)
from src.market_reader import (
    discover_active_btc_5m_markets,
    MarketRound,
    check_round_resolution,
    extract_reference_price,
)
from src.price_oracle import get_btc_price
from src.strategy import evaluate_round, get_bankroll, record_bet_result
from src.risk_manager import RiskManager
from src.trader import place_buy_order, FillResult
from src.fees import effective_fee_rate
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

RESOLUTION_WAIT_SECONDS = 20
RESOLUTION_TIMEOUT_SECONDS = 600
OPENING_PRICE_MAX_AGE = 30


def _handle_signal(signum, frame):
    global _shutdown
    log.warning("Shutdown signal received - finishing current cycle then exiting")
    _shutdown = True


import threading as _threading
if _threading.current_thread() is _threading.main_thread():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


@dataclass
class PendingTrade:
    """A trade waiting for its round to resolve."""
    condition_id: str
    event_slug: str
    direction: str
    action: str
    token_id: str
    entry_price: float
    filled_size: float
    bet_dollars: float
    edge: float
    confidence: float
    kelly_fraction: float
    fee_rate_pct: float
    effective_price: float
    btc_opening: float
    btc_at_entry: float
    entry_time: float
    round_end_timestamp: int
    is_live: bool
    resolution_attempts: int = 0


def print_banner():
    mode = "LIVE" if LIVE_TRADING else "DRY RUN (paper — real resolution)"
    br = get_bankroll()
    print()
    print("=" * 62)
    print("   Polymarket BTC 5-Min Oracle-Verified Bot")
    print("=" * 62)
    print(f"   Mode:                {mode}")
    print(f"   Bankroll:            ${br.current_balance:.2f} (started ${br.starting_balance:.2f})")
    print(f"   Kelly multiplier:    {KELLY_MULTIPLIER:.0%} Kelly (quarter)")
    print(f"   Bet sizing:          Kelly-optimal (capped at {KELLY_MULTIPLIER:.0%})")
    print(f"   Profit reinvestment: 100% (compounding)")
    print(f"   Min edge threshold:  {MIN_EDGE*100:.1f}% (after Polymarket fees)")
    print(f"   Crypto fee at 50c:   ~1.56% (auto-fetched per token)")
    print(f"   Entry window:        {ENTRY_WINDOW_START}s -> {ENTRY_WINDOW_END}s before close")
    print(f"   Resolution:          REAL (Polymarket Gamma API + Chainlink)")
    if br.num_bets > 0:
        print(f"   Lifetime bets:       {br.num_bets} ({br.win_rate*100:.1f}% win rate)")
        print(f"   Lifetime P&L:        ${br.total_profit:+.2f} ({br.roi*100:+.1f}% ROI)")
        print(f"   Current drawdown:    {br.drawdown*100:.1f}%")
    print("=" * 62)
    if not LIVE_TRADING:
        print("   *** DRY RUN — orders are simulated but outcomes are REAL ***")
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

    def cleanup(self, active_ids: set[str], pending_ids: set[str]):
        """Remove stale entries, but keep any that have pending trades."""
        keep = active_ids | pending_ids
        stale = [k for k in self._opening_prices if k not in keep]
        for k in stale:
            del self._opening_prices[k]
            self._traded.discard(k)


def main_loop():
    tracker = RoundTracker()
    pending_trades: list[PendingTrade] = []
    cycle = 0
    trade_count = 0

    mode = "LIVE" if LIVE_TRADING else "DRY RUN"
    dashboard.update_bot_status(True, 0, 0, 0, mode)

    while not _shutdown:
        cycle += 1
        dashboard.update_bot_status(True, cycle, trade_count, 0, mode)

        risk.pre_trade_check()
        dashboard.update_risk_stats(risk.stats_summary())

        log.info(
            "-- Cycle %d  [trades: %d | pending: %d] -------------------------",
            cycle, trade_count, len(pending_trades),
        )

        # ── Phase 1: resolve pending trades whose rounds have ended ──
        _resolve_pending_trades(pending_trades, tracker)

        # ── Phase 2: fetch current BTC price ──
        try:
            oracle = get_btc_price()
            dashboard.update_btc_price(oracle)
        except Exception:
            pass

        # ── Phase 3: discover active rounds ──
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
        pending_ids = {p.condition_id for p in pending_trades}
        tracker.cleanup(active_ids, pending_ids)

        log.info("Found %d active round(s)", len(rounds))

        # ── Phase 4: capture opening prices & trade eligible rounds ──
        for rnd in rounds:
            if _shutdown:
                break

            _capture_opening_price(tracker, rnd)

            if tracker.already_traded(rnd.condition_id):
                continue

            remaining = rnd.seconds_remaining
            if remaining > ENTRY_WINDOW_START or remaining < ENTRY_WINDOW_END:
                continue

            opening = tracker.get_opening_price(rnd.condition_id)
            log.info(
                "EVALUATING %s | %.0fs left | opening=$%.2f",
                rnd.condition_id[:12],
                remaining,
                opening or 0,
            )

            sig = evaluate_round(rnd, opening)
            if sig is None:
                log.info("  -> No signal (edge/confidence/price insufficient)")
                continue

            log.info(
                ">>> EXECUTING: %s on %s | P(win)=%.3f edge=%.4f (fee-adj) | "
                "price=$%.3f eff=$%.3f fee=%.2f%% | "
                "Kelly=%.4f -> $%.2f (%.1f contracts @ $%.3f) | EV=$%.4f",
                sig.action, rnd.condition_id[:12],
                sig.confidence, sig.edge,
                sig.price, sig.effective_price, sig.fee_rate_pct,
                sig.kelly_fraction, sig.bet_dollars,
                sig.size, sig.price, sig.expected_profit,
            )

            fill = place_buy_order(
                token_id=sig.token_id,
                price=sig.price,
                size=sig.size,
                neg_risk=sig.neg_risk,
                tick_size=sig.tick_size,
                order_type="FAK",
                min_order_size=sig.min_order_size,
            )

            if fill.success:
                tracker.mark_traded(rnd.condition_id)
                trade_count += 1

                # bet_dollars = raw USDC paid (contracts * price).
                # The fee is taken in shares (fewer shares received),
                # NOT as extra USDC cost. Accounted for in _record_resolution
                # by reducing effective_contracts.
                actual_bet = fill.filled_size * fill.avg_price

                pending = PendingTrade(
                    condition_id=rnd.condition_id,
                    event_slug=rnd.event_slug,
                    direction=sig.direction,
                    action=sig.action,
                    token_id=sig.token_id,
                    entry_price=fill.avg_price,
                    filled_size=fill.filled_size,
                    bet_dollars=actual_bet,
                    edge=sig.edge,
                    confidence=sig.confidence,
                    kelly_fraction=sig.kelly_fraction,
                    fee_rate_pct=sig.fee_rate_pct,
                    effective_price=sig.effective_price,
                    btc_opening=sig.btc_opening,
                    btc_at_entry=sig.btc_current,
                    entry_time=time.time(),
                    round_end_timestamp=rnd.end_timestamp,
                    is_live=fill.is_live,
                )
                pending_trades.append(pending)

                dashboard.add_trade(TradeEntry(
                    timestamp=time.time(),
                    direction=sig.direction,
                    action=sig.action,
                    price=fill.avg_price,
                    size=fill.filled_size,
                    edge=sig.edge,
                    confidence=sig.confidence,
                    pnl=0.0,
                    won=False,
                    btc_price=sig.btc_current,
                    condition_id=rnd.condition_id[:16],
                    reason=sig.reason,
                    status="pending",
                ))

                if fill.filled_size < fill.requested_size:
                    log.warning(
                        "Partial fill: %.1f / %.1f contracts (%.0f%%)",
                        fill.filled_size, fill.requested_size,
                        fill.filled_size / fill.requested_size * 100,
                    )

                dashboard.add_log(
                    "INFO",
                    f"Trade #{trade_count}: {sig.action} "
                    f"edge={sig.edge:.4f} kelly={sig.kelly_fraction:.4f} "
                    f"filled={fill.filled_size:.1f}/{fill.requested_size:.1f} "
                    f"@ ${fill.avg_price:.3f} -> PENDING resolution",
                )
            else:
                log.warning(
                    "Order failed: %s (size=%.1f @ $%.3f)",
                    fill.status, fill.requested_size, sig.price,
                )

        _print_status(cycle, rounds, pending_trades)
        dashboard.update_risk_stats(risk.stats_summary())

        sleep_time = _calculate_sleep(rounds, pending_trades)
        log.debug("Sleeping %.1fs until next check", sleep_time)
        _sleep(sleep_time)

    # ── Shutdown: try to resolve remaining trades ──
    if pending_trades:
        log.info("Resolving %d pending trades before shutdown...", len(pending_trades))
        for _ in range(30):
            _resolve_pending_trades(pending_trades, tracker)
            if not pending_trades:
                break
            _sleep(2)

    dashboard.update_bot_status(False, cycle, trade_count, 0, mode)
    log.info("Bot shut down gracefully.")
    _print_final_stats(pending_trades)


# ── Resolution ────────────────────────────────────────────────

def _resolve_pending_trades(
    pending_trades: list[PendingTrade],
    tracker: RoundTracker,
):
    """
    Check each pending trade for resolution.

    Primary: Polymarket Gamma API (check_round_resolution) — gives the
    authoritative on-chain result.
    Fallback: If the Gamma API doesn't show resolution yet but the round
    ended >60s ago, use Chainlink price vs opening price.
    """
    if not pending_trades:
        return

    now = time.time()
    resolved: list[PendingTrade] = []

    for trade in pending_trades:
        if now < trade.round_end_timestamp + RESOLUTION_WAIT_SECONDS:
            continue

        trade.resolution_attempts += 1

        outcome = check_round_resolution(trade.event_slug)

        if outcome is None and now > trade.round_end_timestamp + 300:
            log.warning(
                "Gamma API not resolving %s after 5min — trying Chainlink fallback "
                "(UNRELIABLE: compares current price, not close price)",
                trade.condition_id[:12],
            )
            outcome = _chainlink_fallback_resolution(trade)

        if outcome is None:
            if now > trade.round_end_timestamp + RESOLUTION_TIMEOUT_SECONDS:
                log.warning(
                    "Resolution timeout for %s after %d attempts — "
                    "marking as unresolved loss",
                    trade.condition_id[:12], trade.resolution_attempts,
                )
                _record_resolution(trade, won=False, outcome="timeout")
                resolved.append(trade)
            else:
                if trade.resolution_attempts <= 3 or trade.resolution_attempts % 10 == 0:
                    log.info(
                        "Waiting for resolution: %s (attempt %d, %.0fs since end)",
                        trade.condition_id[:12],
                        trade.resolution_attempts,
                        now - trade.round_end_timestamp,
                    )
            continue

        won = outcome == trade.direction
        _record_resolution(trade, won=won, outcome=outcome)
        resolved.append(trade)

    for trade in resolved:
        pending_trades.remove(trade)


def _chainlink_fallback_resolution(trade: PendingTrade) -> str | None:
    """
    If the Gamma API hasn't reported resolution, compare the current
    Chainlink price to the opening price as a fallback.
    """
    try:
        oracle = get_btc_price()
        chainlink = oracle.get("chainlink")
        if chainlink is None:
            chainlink = oracle.get("median")
        if chainlink is None or trade.btc_opening <= 0:
            return None

        if chainlink > trade.btc_opening:
            return "up"
        elif chainlink < trade.btc_opening:
            return "down"
        return None
    except Exception:
        return None


def _record_resolution(trade: PendingTrade, won: bool, outcome: str):
    """Update bankroll, risk manager, and dashboard with the actual result."""
    fee_rate = effective_fee_rate(trade.entry_price)
    effective_contracts = trade.filled_size * (1.0 - fee_rate)

    if won:
        payout = effective_contracts * 1.0
        pnl = payout - trade.bet_dollars
    else:
        pnl = -trade.bet_dollars

    if won:
        record_bet_result(True, trade.bet_dollars, payout)
    else:
        record_bet_result(False, trade.bet_dollars)

    risk.record_pnl(
        won=won, pnl=pnl, direction=trade.direction,
        bet_dollars=trade.bet_dollars,
        kelly_fraction=trade.kelly_fraction,
        win_prob=trade.confidence,
    )

    br = get_bankroll()

    dashboard.resolve_trade(trade.condition_id[:16], won=won, pnl=pnl)
    dashboard.update_risk_stats(risk.stats_summary())

    result_str = "WIN" if won else "LOSS"
    source = "Gamma API" if outcome in ("up", "down") else f"fallback ({outcome})"
    log.info(
        "RESOLVED %s: %s %s | outcome=%s (via %s) | "
        "PnL=$%.4f | Bankroll=$%.2f | Drawdown=%.1f%%",
        trade.condition_id[:12], result_str, trade.action,
        outcome, source, pnl, br.current_balance, br.drawdown * 100,
    )
    dashboard.add_log(
        "INFO",
        f"RESOLVED: {trade.action} -> {result_str} (actual={outcome}) "
        f"pnl=${pnl:.4f} bankroll=${br.current_balance:.2f}",
    )


# ── Opening price capture ─────────────────────────────────────

def _capture_opening_price(tracker: RoundTracker, rnd: MarketRound):
    """
    Record the BTC reference price for this round.

    Priority:
    1. Extract from market description/question (Polymarket's own ref price)
    2. Chainlink on-chain price (captured within first 15s of round)
    """
    if tracker.get_opening_price(rnd.condition_id) is not None:
        return

    ref = extract_reference_price(rnd.question) or extract_reference_price(rnd.description)
    if ref is not None:
        tracker.set_opening_price(rnd.condition_id, ref)
        log.info(
            "Reference price from market text for %s: $%.2f",
            rnd.condition_id[:12], ref,
        )
        return

    elapsed = rnd.seconds_elapsed
    if elapsed < 0 or elapsed > OPENING_PRICE_MAX_AGE:
        return

    oracle = get_btc_price()
    dashboard.update_btc_price(oracle)
    price = oracle.get("chainlink") or oracle.get("median")
    if price:
        tracker.set_opening_price(rnd.condition_id, price)


# ── Sleep / status helpers ────────────────────────────────────

def _calculate_sleep(
    rounds: list[MarketRound],
    pending_trades: list[PendingTrade],
) -> float:
    """Smart sleep: wake up when the nearest round enters our entry window
    or when a pending trade might be ready for resolution."""
    min_sleep = 10.0

    if rounds:
        nearest = min(r.seconds_remaining for r in rounds)
        if nearest <= ENTRY_WINDOW_START + 5:
            min_sleep = min(min_sleep, 1.0)
        elif nearest <= ENTRY_WINDOW_START + 30:
            min_sleep = min(min_sleep, 2.0)
        else:
            min_sleep = min(min_sleep, nearest - ENTRY_WINDOW_START - 5)

    if pending_trades:
        now = time.time()
        for pt in pending_trades:
            time_until_check = (pt.round_end_timestamp + RESOLUTION_WAIT_SECONDS) - now
            if time_until_check <= 0:
                return 1.0
            min_sleep = min(min_sleep, time_until_check + 1)

    return max(1.0, min_sleep)


def _sleep(seconds: float):
    """Interruptible sleep."""
    end = time.time() + seconds
    while time.time() < end and not _shutdown:
        time.sleep(min(1.0, end - time.time()))


def _print_status(cycle: int, rounds: list[MarketRound], pending: list[PendingTrade]):
    if cycle % 10 != 0:
        return
    stats = risk.stats_summary()
    log.info(
        "STATUS | trades=%d  wins=%d  losses=%d  win_rate=%.1f%%  "
        "daily_pnl=$%.4f  total_pnl=$%.4f  pending=%d  consec_losses=%d",
        stats["total_trades"], stats["wins"], stats["losses"],
        stats["win_rate"] * 100, stats["daily_pnl"], stats["total_pnl"],
        len(pending), stats["consecutive_losses"],
    )


def _print_final_stats(pending_trades: list[PendingTrade] | None = None):
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
    if pending_trades:
        print(f"  Unresolved trades:  {len(pending_trades)}")
    print("  " + "-" * 40)
    print(f"  Bankroll:           ${br.current_balance:.2f}")
    print(f"  Starting balance:   ${br.starting_balance:.2f}")
    print(f"  Peak balance:       ${br.peak_balance:.2f}")
    print(f"  Lifetime P&L:       ${br.total_profit:+.2f}")
    print(f"  ROI:                {br.roi*100:+.1f}%")
    print(f"  Drawdown:           {br.drawdown*100:.1f}%")
    print(f"  Total wagered:      ${br.total_wagered:.2f}")
    print(f"  Fee model:          Polymarket crypto (max 1.56% at 50c)")
    print(f"  Resolution:         REAL (Gamma API + Chainlink)")
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
