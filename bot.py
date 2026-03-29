"""
Polymarket BTC 5-Minute Oracle-Verified Trading Bot (v3)
========================================================
v3 fixes (senior quant review):
  - FIXED: PnL calculation now uses trade_economics() single source of
    truth — no more double-counting fees on wins.
  - FIXED: Bankroll sync respects pending positions — Kelly no longer
    undersizes when USDC is locked in outcome tokens.
  - FIXED: Bot runs CONTINUOUSLY instead of exiting after one trade.
    Set MAX_TRADES_PER_SESSION in .env to limit (0 = unlimited).
  - FIXED: Recent outcome tracking feeds streak feature into model.
  - FIXED: _refresh_wallet passes has_pending flag.

DISCLAIMER: Educational and research purposes only. Not financial advice.
"""

import logging, signal, sys, time, os
from dataclasses import dataclass

from src.config import (
    LIVE_TRADING, ENTRY_WINDOW_START, ENTRY_WINDOW_END, MIN_EDGE,
    KELLY_MULTIPLIER, RUIN_TARGET, RUIN_LEVEL, MAX_DRAWDOWN_HALT,
)
from src.market_reader import (
    discover_active_btc_5m_markets, MarketRound,
    check_round_resolution, extract_reference_price,
)
from src.price_oracle import get_btc_price
from src.strategy import (
    evaluate_round, get_bankroll, record_bet_result, sync_bankroll_to_balance,
    record_outcome, add_pending_cost, clear_pending_cost,
)
from src.risk_manager import RiskManager
from src.trader import place_buy_order, FillResult
from src.fees import trade_economics
from src.bot_state import state as dashboard, TradeEntry, install_log_handler
from src.redeemer import (
    fetch_wallet_balances, redeem_winning_position_blocking,
    sweep_unredeemed_positions,
)
from src.geoblock import (
    assert_not_geoblocked, is_clob_geoblocked, clob_geoblock_status,
    probe_clob_trading, clear_clob_geoblock, signal_clob_geoblock,
    SAFE_COUNTRIES,
)
from src.ws_client import get_ws_client
from src.data_api import reconcile_bankroll
from src.kelly import risk_of_ruin, optimal_kelly_for_ruin

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
RESOLUTION_WAIT_SECONDS = 10
RESOLUTION_TIMEOUT_SECONDS = 300
OPENING_PRICE_MAX_AGE = 30
_force_wallet_refresh = False
_won_conditions: list[tuple[str, bool]] = []
WALLET_REFRESH_SECONDS = 15
MAX_TRADES_PER_SESSION = int(os.getenv("MAX_TRADES_PER_SESSION", "0"))


def _handle_signal(signum, frame):
    global _shutdown
    log.warning("Shutdown signal received")
    _shutdown = True

import threading as _threading
if _threading.current_thread() is _threading.main_thread():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


@dataclass
class PendingTrade:
    condition_id: str
    event_slug: str
    direction: str
    action: str
    token_id: str
    entry_price: float
    filled_size: float
    gross_cost: float
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
    neg_risk: bool = False
    resolution_attempts: int = 0
    ruin_prob: float = 0.0
    quality_grade: str = ""
    features: dict = None


def print_banner():
    mode = "LIVE" if LIVE_TRADING else "DRY RUN"
    br = get_bankroll()
    opt_mult = optimal_kelly_for_ruin(0.55, 0.50, RUIN_TARGET, RUIN_LEVEL)
    print()
    print("=" * 62)
    print("   Polymarket BTC 5-Min Oracle-Verified Bot (v3)")
    print("=" * 62)
    print(f"   Mode:                {mode}")
    print(f"   Bankroll:            ${br.current_balance:.2f} (started ${br.starting_balance:.2f})")
    print(f"   Kelly multiplier:    {KELLY_MULTIPLIER:.0%} (ruin-calibrated)")
    print(f"   Ruin target:         {RUIN_TARGET:.0%} at {RUIN_LEVEL:.0%}")
    print(f"   Min edge:            {MIN_EDGE*100:.1f}%")
    print(f"   Entry window:        {ENTRY_WINDOW_START}s -> {ENTRY_WINDOW_END}s")
    print(f"   Max DD halt:         {MAX_DRAWDOWN_HALT*100:.0f}%")
    print(f"   Max trades/session:  {'unlimited' if MAX_TRADES_PER_SESSION == 0 else MAX_TRADES_PER_SESSION}")
    if br.num_bets > 0:
        print(f"   Lifetime:            {br.num_bets} bets ({br.win_rate*100:.1f}% WR) P&L=${br.total_profit:+.2f}")
    print("=" * 62)
    print()


class RoundTracker:
    def __init__(self):
        self._traded: set[str] = set()
        self._opening_prices: dict[str, float] = {}

    def already_traded(self, cid: str) -> bool: return cid in self._traded
    def mark_traded(self, cid: str): self._traded.add(cid)

    def set_opening_price(self, cid: str, price: float):
        if cid not in self._opening_prices:
            self._opening_prices[cid] = price
            log.info("Opening BTC price for %s: $%.2f", cid[:12], price)

    def get_opening_price(self, cid: str) -> float | None:
        return self._opening_prices.get(cid)

    def cleanup(self, active_ids: set[str], pending_ids: set[str]):
        keep = active_ids | pending_ids
        stale = [k for k in self._opening_prices if k not in keep]
        for k in stale:
            del self._opening_prices[k]
            self._traded.discard(k)


def _refresh_wallet(has_pending: bool = False):
    try:
        wb = fetch_wallet_balances()
        sync_bankroll_to_balance(wb["usdc"], has_pending=has_pending)
        br = get_bankroll()
        dashboard.update_wallet(wb["address"], wb["usdc"], wb["pol"], bankroll=br.current_balance)
        dashboard.update_bankroll_meta(
            starting=br.starting_balance, peak=br.peak_balance,
            num_bets=br.num_bets, num_wins=br.num_wins,
            total_wagered=br.total_wagered, max_drawdown=br.max_drawdown,
        )
        return True
    except Exception as e:
        log.debug("Wallet fetch failed: %s", e)
        return False


def _redeem_all_and_refresh():
    redeemed = 0
    if _won_conditions:
        log.info("Redeeming %d winning condition(s)...", len(_won_conditions))
        for cid, neg_risk in _won_conditions:
            try:
                tx = redeem_winning_position_blocking(cid, neg_risk=neg_risk)
                if tx:
                    log.info("Redeemed %s: tx=%s", cid[:12], tx)
                    redeemed += 1
            except Exception as e:
                log.warning("Redeem failed %s: %s", cid[:12], e)
            time.sleep(2)
        _won_conditions.clear()

    try:
        time.sleep(3)
        n = sweep_unredeemed_positions()
        if n: redeemed += n
    except Exception as e:
        log.warning("Sweep failed: %s", e)

    log.info("Redeemed %d position(s)", redeemed)
    _refresh_wallet(has_pending=False)


def main_loop():
    global _force_wallet_refresh
    tracker = RoundTracker()
    pending_trades: list[PendingTrade] = []
    cycle = 0
    trade_count = 0

    mode = "LIVE" if LIVE_TRADING else "DRY RUN"
    dashboard.update_bot_status(True, 0, 0, 0, mode)

    ws = get_ws_client()
    ws.start()

    _ws_resolutions: list[tuple[str, str]] = []
    _ws_res_lock = __import__("threading").Lock()

    def _on_ws_resolution(cid, outcome):
        with _ws_res_lock:
            _ws_resolutions.append((cid, outcome))

    ws.set_resolution_callback(_on_ws_resolution)
    _last_wallet_refresh = 0.0

    while not _shutdown:
        cycle += 1
        dashboard.update_bot_status(True, cycle, trade_count, MAX_TRADES_PER_SESSION, mode)

        allowed, risk_reason = risk.pre_trade_check()
        kelly_scale = risk.kelly_scale_factor()
        dashboard.update_risk_stats(risk.stats_summary())

        now = time.time()
        has_pending = len(pending_trades) > 0
        if _force_wallet_refresh or (now - _last_wallet_refresh) >= WALLET_REFRESH_SECONDS:
            _force_wallet_refresh = False
            _last_wallet_refresh = now
            _refresh_wallet(has_pending=has_pending)

        # Process WS resolutions
        with _ws_res_lock:
            ws_events = list(_ws_resolutions)
            _ws_resolutions.clear()
        if ws_events:
            _apply_ws_resolutions(ws_events, pending_trades, tracker)

        # Resolve pending trades
        had_pending = len(pending_trades)
        _resolve_pending_trades(pending_trades, tracker)
        if len(pending_trades) < had_pending:
            _force_wallet_refresh = True

        # Redeem wins when no more pending
        if _won_conditions and not pending_trades and LIVE_TRADING:
            _redeem_all_and_refresh()

        # Check engine restart
        from src.http_client import is_engine_restart_window, engine_restart_status
        es = engine_restart_status()
        if es["active"] or is_engine_restart_window():
            _sleep(10)
            continue

        # Check geoblock
        if is_clob_geoblocked():
            _sleep(min(30, clob_geoblock_status()["seconds_until_recheck"] + 1))
            continue
        elif clob_geoblock_status()["blocked"]:
            if probe_clob_trading():
                clear_clob_geoblock()
            else:
                signal_clob_geoblock()
                _sleep(30)
                continue

        # Check trade limit
        if MAX_TRADES_PER_SESSION > 0 and trade_count >= MAX_TRADES_PER_SESSION:
            if not pending_trades:
                break
            _sleep(5)
            continue

        # Discover rounds
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
            dashboard.clear_rounds()
            _sleep(10)
            continue

        dashboard.update_rounds(rounds)
        active_ids = {r.condition_id for r in rounds}
        pending_ids = {p.condition_id for p in pending_trades}
        tracker.cleanup(active_ids, pending_ids)

        ws_tokens = []
        for r in rounds:
            ws_tokens.extend([r.up_token_id, r.down_token_id])
        if ws_tokens:
            ws.subscribe(ws_tokens)

        # Evaluate and trade
        for rnd in rounds:
            if _shutdown: break
            _capture_opening_price(tracker, rnd)
            if tracker.already_traded(rnd.condition_id): continue
            if not allowed: continue

            remaining = rnd.seconds_remaining
            if remaining > ENTRY_WINDOW_START or remaining < ENTRY_WINDOW_END:
                continue

            opening = tracker.get_opening_price(rnd.condition_id)
            sig = evaluate_round(rnd, opening)
            if sig is None: continue

            if kelly_scale < 1.0 and sig.bet_dollars > 0:
                sig.bet_dollars *= kelly_scale
                sig.size *= kelly_scale

            log.info(
                ">>> EXECUTING: %s [%s] | P(win)=%.3f edge=%.4f | $%.2f (%d contracts)",
                sig.action, sig.quality_grade, sig.confidence, sig.edge,
                sig.bet_dollars, sig.size,
            )

            fill = place_buy_order(
                token_id=sig.token_id, price=sig.price, size=sig.size,
                neg_risk=sig.neg_risk, tick_size=sig.tick_size,
                order_type="FAK", min_order_size=sig.min_order_size,
            )

            if fill.success:
                tracker.mark_traded(rnd.condition_id)
                trade_count += 1
                gross_cost = fill.filled_size * fill.avg_price
                add_pending_cost(gross_cost)

                pending_trades.append(PendingTrade(
                    condition_id=rnd.condition_id, event_slug=rnd.event_slug,
                    direction=sig.direction, action=sig.action,
                    token_id=sig.token_id, entry_price=fill.avg_price,
                    filled_size=fill.filled_size, gross_cost=gross_cost,
                    bet_dollars=gross_cost, edge=sig.edge,
                    confidence=sig.confidence, kelly_fraction=sig.kelly_fraction,
                    fee_rate_pct=sig.fee_rate_pct, effective_price=sig.effective_price,
                    btc_opening=sig.btc_opening, btc_at_entry=sig.btc_current,
                    entry_time=time.time(), round_end_timestamp=rnd.end_timestamp,
                    is_live=fill.is_live, neg_risk=sig.neg_risk,
                    ruin_prob=sig.ruin_prob, quality_grade=sig.quality_grade,
                    features=sig.features,
                ))

                dashboard.add_trade(TradeEntry(
                    timestamp=time.time(), direction=sig.direction,
                    action=sig.action, price=fill.avg_price,
                    size=fill.filled_size, edge=sig.edge,
                    confidence=sig.confidence, pnl=0.0, won=False,
                    btc_price=sig.btc_current,
                    condition_id=rnd.condition_id[:16],
                    reason=sig.reason, status="pending",
                ))

        # Second resolve pass
        if pending_trades:
            pre = len(pending_trades)
            _resolve_pending_trades(pending_trades, tracker)
            if len(pending_trades) < pre:
                _force_wallet_refresh = True

        if _force_wallet_refresh:
            _force_wallet_refresh = False
            _last_wallet_refresh = time.time()
            _refresh_wallet(has_pending=len(pending_trades) > 0)

        sleep_time = _calculate_sleep(rounds, pending_trades)
        _sleep(sleep_time)

    # Shutdown
    if pending_trades:
        log.info("Resolving %d pending trades...", len(pending_trades))
        for _ in range(30):
            _resolve_pending_trades(pending_trades, tracker)
            if not pending_trades: break
            _sleep(2)
        if not pending_trades and LIVE_TRADING:
            _redeem_all_and_refresh()

    ws.stop()
    dashboard.update_bot_status(False, cycle, trade_count, MAX_TRADES_PER_SESSION, mode)
    _print_final_stats(pending_trades)


# ── Resolution (FIXED PnL) ───────────────────────────────────

def _resolve_pending_trades(pending_trades, tracker):
    if not pending_trades: return
    now = time.time()
    resolved = []

    for trade in pending_trades:
        if now < trade.round_end_timestamp + RESOLUTION_WAIT_SECONDS:
            continue
        trade.resolution_attempts += 1

        outcome = check_round_resolution(trade.event_slug)
        if outcome is None and now > trade.round_end_timestamp + 120:
            outcome = _chainlink_fallback(trade)
        if outcome is None:
            if now > trade.round_end_timestamp + RESOLUTION_TIMEOUT_SECONDS:
                _record_resolution(trade, won=False, outcome="timeout")
                resolved.append(trade)
            continue

        won = outcome == trade.direction
        _record_resolution(trade, won=won, outcome=outcome)
        resolved.append(trade)

    for t in resolved:
        pending_trades.remove(t)


def _apply_ws_resolutions(ws_events, pending_trades, tracker):
    resolved = []
    for cid, winning in ws_events:
        outcome = winning.lower()
        if outcome not in ("up", "down"): continue
        for trade in pending_trades:
            if trade.condition_id == cid or cid.endswith(trade.condition_id):
                won = outcome == trade.direction
                _record_resolution(trade, won=won, outcome=outcome)
                resolved.append(trade)
                break
    for t in resolved:
        pending_trades.remove(t)


def _chainlink_fallback(trade):
    try:
        oracle = get_btc_price()
        cl = oracle.get("chainlink") or oracle.get("median")
        if cl is None or trade.btc_opening <= 0: return None
        return "up" if cl > trade.btc_opening else "down" if cl < trade.btc_opening else None
    except Exception:
        return None


def _record_resolution(trade, won, outcome):
    """FIX #2: Use trade_economics() for correct PnL."""
    econ = trade_economics(trade.entry_price, trade.filled_size, won)
    pnl = econ["pnl"]

    record_outcome(outcome)
    clear_pending_cost(trade.gross_cost)

    if won:
        record_bet_result(True, trade.gross_cost, econ["payout"])
        _won_conditions.append((trade.condition_id, trade.neg_risk))
    else:
        record_bet_result(False, trade.gross_cost)

    risk.record_pnl(
        won=won, pnl=pnl, direction=trade.direction,
        bet_dollars=trade.gross_cost, kelly_fraction=trade.kelly_fraction,
        win_prob=trade.confidence, ruin_prob=trade.ruin_prob,
        quality_grade=trade.quality_grade, features=trade.features,
    )

    br = get_bankroll()
    dashboard.resolve_trade(trade.condition_id[:16], won=won, pnl=pnl)
    dashboard.update_risk_stats(risk.stats_summary())

    log.info(
        "RESOLVED %s: %s [%s] | gross=$%.4f payout=$%.4f PnL=$%.4f | BR=$%.2f",
        trade.condition_id[:12], "WIN" if won else "LOSS",
        trade.quality_grade, trade.gross_cost, econ["payout"], pnl,
        br.current_balance,
    )


def _capture_opening_price(tracker, rnd):
    if tracker.get_opening_price(rnd.condition_id) is not None: return
    ref = extract_reference_price(rnd.question) or extract_reference_price(rnd.description)
    if ref is not None:
        tracker.set_opening_price(rnd.condition_id, ref)
        return
    elapsed = rnd.seconds_elapsed
    if elapsed < 0 or elapsed > OPENING_PRICE_MAX_AGE: return
    oracle = get_btc_price()
    dashboard.update_btc_price(oracle)
    price = oracle.get("chainlink") or oracle.get("median")
    if price: tracker.set_opening_price(rnd.condition_id, price)


def _calculate_sleep(rounds, pending_trades):
    min_sleep = 8.0
    if _force_wallet_refresh: return 1.0
    if rounds:
        nearest = min(r.seconds_remaining for r in rounds)
        if nearest <= ENTRY_WINDOW_START + 5: min_sleep = min(min_sleep, 1.0)
        elif nearest <= ENTRY_WINDOW_START + 30: min_sleep = min(min_sleep, 2.0)
        else: min_sleep = min(min_sleep, nearest - ENTRY_WINDOW_START - 5)
    if pending_trades:
        now = time.time()
        for pt in pending_trades:
            t = (pt.round_end_timestamp + RESOLUTION_WAIT_SECONDS) - now
            if t <= 0: return 1.0
            min_sleep = min(min_sleep, max(1.0, t))
    return max(1.0, min_sleep)


def _sleep(seconds):
    end = time.time() + seconds
    while time.time() < end and not _shutdown:
        time.sleep(min(1.0, end - time.time()))


def _print_final_stats(pending=None):
    stats = risk.stats_summary()
    br = get_bankroll()
    print()
    print("=" * 62)
    print("  FINAL SESSION STATS (v3)")
    print("=" * 62)
    print(f"  Trades: {stats['total_trades']}  W/L: {stats['wins']}/{stats['losses']}  WR: {stats['win_rate']*100:.1f}%")
    print(f"  PF: {stats['profit_factor']:.2f}  Sharpe: {stats['sharpe_ratio']:.2f}  Expect: ${stats['expectancy']:.4f}")
    print(f"  Total PnL: ${stats['total_pnl']:.4f}  Bankroll: ${br.current_balance:.2f}")
    print(f"  MaxDD: {br.max_drawdown*100:.1f}%  ROI: {br.roi*100:+.1f}%")
    print("=" * 62)
    print()


if __name__ == "__main__":
    print_banner()
    try:
        geo = assert_not_geoblocked()
        print(f"   Geoblock: PASSED ({geo['ip']} | {geo['region']})")
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    try:
        wb = fetch_wallet_balances()
        sync_bankroll_to_balance(wb["usdc"])
        br = get_bankroll()
        dashboard.update_wallet(wb["address"], wb["usdc"], wb["pol"], bankroll=br.current_balance)
        print(f"   Wallet: {wb['address'][:8]}...{wb['address'][-6:]}  USDC: ${wb['usdc']:.4f}")
    except Exception as e:
        log.warning("Wallet fetch failed: %s", e)

    try:
        recon = reconcile_bankroll()
        print(f"   Positions: {recon.get('open_positions', '?')} open")
    except Exception:
        pass

    print()
    try:
        main_loop()
    except KeyboardInterrupt:
        _print_final_stats()
    except Exception as e:
        log.critical("Fatal: %s", e, exc_info=True)
        _print_final_stats()
        sys.exit(1)
