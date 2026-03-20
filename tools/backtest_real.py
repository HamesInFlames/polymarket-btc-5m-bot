"""
Historical Backtest with FULLY REAL Data
=========================================
Everything in this backtest is real:

  1. BTC PRICES — fetched from Binance 1-minute klines (historical).
     For each round we get the actual BTC price at round start (opening)
     and at the bot's evaluation time (~2 min before close).

  2. OUTCOMES — fetched from Polymarket's Gamma API.  Each round's winner
     (Up / Down) is the actual on-chain settlement.

  3. STRATEGY DECISIONS — the bot compares real BTC price at evaluation
     time to real opening price, exactly like it would live.  If BTC is
     above opening → direction = "up", below → "down".

  4. FEES & SIZING — real Polymarket fee formula, real Kelly criterion.

The ONLY estimation is entry price (order book), since historical
order book snapshots are not available.

Usage:
    python tools/backtest_real.py
    python tools/backtest_real.py --hours 48 --bankroll 250 --save
"""

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GAMMA_HOST
from src.fees import effective_fee_rate
from src.kelly import kelly_criterion, Bankroll, save_bankroll, BANKROLL_FILE
from src.market_reader import ROUND_DURATION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest")

GAMMA_DELAY = 0.12
BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"
EVAL_OFFSET = 120  # evaluate 2 min before close


# ── BTC price history from Binance ────────────────────────────

def fetch_btc_price_history(start_ts: int, end_ts: int) -> dict[int, float]:
    """
    Fetch 1-minute BTC/USDT close prices from Binance.
    Returns {minute_timestamp: close_price}.
    """
    prices: dict[int, float] = {}
    chunk_size = 1000
    cursor = start_ts * 1000

    log.info("Fetching BTC price history from Binance (%d hours)...",
             (end_ts - start_ts) // 3600)

    while cursor < end_ts * 1000:
        try:
            resp = requests.get(BINANCE_KLINE_URL, params={
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": int(cursor),
                "endTime": int(end_ts * 1000),
                "limit": chunk_size,
            }, timeout=15)

            if resp.status_code == 429:
                log.warning("Binance rate limited — waiting 10s")
                time.sleep(10)
                continue

            resp.raise_for_status()
            candles = resp.json()

            if not candles:
                break

            for c in candles:
                minute_ts = int(c[0]) // 1000
                close_price = float(c[4])
                prices[minute_ts] = close_price

            cursor = candles[-1][0] + 60000
            time.sleep(0.2)

        except Exception as e:
            log.warning("Binance fetch error: %s — retrying", e)
            time.sleep(2)

    log.info("Got %d minutes of BTC price data", len(prices))
    return prices


def lookup_price(prices: dict[int, float], target_ts: int) -> float | None:
    """Find the closest price to target_ts (within 120 seconds)."""
    minute = target_ts - (target_ts % 60)

    for offset in [0, -60, 60, -120, 120]:
        p = prices.get(minute + offset)
        if p is not None:
            return p
    return None


# ── Polymarket round fetching ─────────────────────────────────

@dataclass
class ResolvedRound:
    slug: str
    condition_id: str
    question: str
    start_ts: int
    end_ts: int
    outcome: str


@dataclass
class BacktestTrade:
    round_slug: str
    condition_id: str
    timestamp: float
    direction: str
    outcome: str
    won: bool
    entry_price: float
    contracts: float
    bet_dollars: float
    fee_rate_pct: float
    pnl: float
    edge: float
    kelly_fraction: float
    btc_opening: float
    btc_at_eval: float
    btc_move_pct: float
    win_prob_est: float
    seconds_before_close: int


def fetch_resolved_rounds(hours_back: int) -> list[ResolvedRound]:
    now = int(time.time())
    start = now - (hours_back * 3600)
    start_base = start - (start % ROUND_DURATION)

    rounds = []
    total = 0
    expected = (now - start_base) // ROUND_DURATION

    log.info("Scanning ~%d Polymarket rounds from last %d hours...",
             expected, hours_back)

    for ts in range(start_base, now - ROUND_DURATION, ROUND_DURATION):
        slug = f"btc-updown-5m-{ts}"
        total += 1

        try:
            resp = requests.get(
                f"{GAMMA_HOST}/events/slug/{slug}", timeout=10,
            )
            if resp.status_code == 429:
                time.sleep(5)
                resp = requests.get(
                    f"{GAMMA_HOST}/events/slug/{slug}", timeout=10,
                )
            if resp.status_code != 200:
                time.sleep(GAMMA_DELAY)
                continue

            event = resp.json()
            markets = event.get("markets", [])
            if not markets:
                continue

            m = markets[0]
            op = m.get("outcomePrices")
            if isinstance(op, str):
                op = json.loads(op)
            if not op or len(op) < 2:
                continue

            outcomes = m.get("outcomes")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if not outcomes:
                outcomes = ["Up", "Down"]

            up_idx, down_idx = 0, 1
            for i, o in enumerate(outcomes):
                lo = str(o).lower()
                if "up" in lo:
                    up_idx = i
                elif "down" in lo:
                    down_idx = i

            try:
                up_f = float(op[up_idx])
                dn_f = float(op[down_idx])
            except (ValueError, IndexError):
                continue

            if up_f > 0.9:
                outcome = "up"
            elif dn_f > 0.9:
                outcome = "down"
            else:
                continue

            end_ts = ts + ROUND_DURATION
            end_date = m.get("endDate", "")
            if end_date:
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    end_ts = int(dt.timestamp())
                except ValueError:
                    pass

            rounds.append(ResolvedRound(
                slug=slug,
                condition_id=m.get("conditionId", "")[:16],
                question=m.get("question", ""),
                start_ts=ts,
                end_ts=end_ts,
                outcome=outcome,
            ))

        except Exception as e:
            log.debug("Error %s: %s", slug, e)
            continue

        time.sleep(GAMMA_DELAY)
        if total % 50 == 0:
            log.info("  checked %d / ~%d, found %d resolved...",
                     total, expected, len(rounds))

    log.info("Scan: %d checked, %d resolved", total, len(rounds))
    return sorted(rounds, key=lambda r: r.start_ts)


# ── Strategy replay ───────────────────────────────────────────

def replay_strategy(
    rounds: list[ResolvedRound],
    btc_prices: dict[int, float],
    starting_bankroll: float,
    kelly_mult: float,
    min_edge: float,
    confidence_floor: float,
    max_bet: float = 50.0,
    entry_window_start: int = 180,
    entry_window_end: int = 5,
) -> tuple[list[BacktestTrade], Bankroll]:
    """
    Replay the strategy with REAL data:
      - Real BTC prices from Binance (opening + evaluation time)
      - Real outcomes from Polymarket
      - Real fee & Kelly calculations

    For each round:
      1. Look up BTC price at round start → opening price
      2. Look up BTC price at (end - EVAL_OFFSET) → evaluation price
      3. Compare → direction = "up" if eval > opening, else "down"
      4. Estimate win probability from the size & consistency of the move
      5. Apply fee-adjusted Kelly sizing
      6. Check against REAL outcome from Polymarket
    """
    br = Bankroll(
        starting_balance=starting_bankroll,
        current_balance=starting_bankroll,
        peak_balance=starting_bankroll,
    )
    trades: list[BacktestTrade] = []
    skipped_no_price = 0
    skipped_no_move = 0
    skipped_no_edge = 0

    for rnd in rounds:
        if br.current_balance < 1.0:
            log.warning("Bankroll depleted — stopping")
            break

        opening_price = lookup_price(btc_prices, rnd.start_ts)
        eval_ts = rnd.end_ts - EVAL_OFFSET
        eval_price = lookup_price(btc_prices, eval_ts)

        if opening_price is None or eval_price is None:
            skipped_no_price += 1
            continue

        delta = eval_price - opening_price
        pct_move = abs(delta) / opening_price

        if pct_move < 0.00005:
            skipped_no_move += 1
            continue

        direction = "up" if delta > 0 else "down"
        win_prob = _estimate_win_prob(pct_move, EVAL_OFFSET)

        if win_prob < confidence_floor:
            skipped_no_edge += 1
            continue

        entry_price = _estimate_entry_price(win_prob)
        fee_rate = effective_fee_rate(entry_price)
        eff_price = entry_price / (1.0 - fee_rate) if fee_rate < 1.0 else entry_price

        edge = win_prob - eff_price
        if edge < min_edge:
            skipped_no_edge += 1
            continue

        rec = kelly_criterion(
            win_prob=win_prob,
            contract_price=eff_price,
            bankroll=br.current_balance,
            kelly_multiplier=kelly_mult,
            min_bet_dollars=0.50,
        )
        if not rec.should_bet:
            skipped_no_edge += 1
            continue

        bet_dollars = min(rec.bet_dollars, max_bet)
        contracts = bet_dollars / eff_price

        won = direction == rnd.outcome
        effective_contracts = contracts * (1.0 - fee_rate)

        if won:
            payout = effective_contracts * 1.0
            pnl = payout - bet_dollars
            br.record_win(bet_dollars, payout)
        else:
            pnl = -bet_dollars
            br.record_loss(bet_dollars)

        trades.append(BacktestTrade(
            round_slug=rnd.slug,
            condition_id=rnd.condition_id,
            timestamp=rnd.end_ts,
            direction=direction,
            outcome=rnd.outcome,
            won=won,
            entry_price=round(entry_price, 3),
            contracts=round(contracts, 2),
            bet_dollars=round(bet_dollars, 4),
            fee_rate_pct=round(fee_rate * 100, 3),
            pnl=round(pnl, 4),
            edge=round(edge, 4),
            kelly_fraction=round(rec.adj_kelly_fraction, 4),
            btc_opening=round(opening_price, 2),
            btc_at_eval=round(eval_price, 2),
            btc_move_pct=round(pct_move * 100, 4),
            win_prob_est=round(win_prob, 4),
            seconds_before_close=EVAL_OFFSET,
        ))

    log.info(
        "Replay done: %d trades | skipped: %d no price, %d no move, %d no edge",
        len(trades), skipped_no_price, skipped_no_move, skipped_no_edge,
    )
    return trades, br


def _estimate_win_prob(pct_move: float, seconds_remaining: float) -> float:
    """
    Estimate win probability from the BTC move size and time remaining.
    Uses the same sigmoid approach as the live strategy.

    Larger moves with less time remaining → higher confidence the
    direction will persist to close.
    """
    magnitude = 1.0 / (1.0 + math.exp(-8000 * (pct_move - 0.0004)))

    time_frac = 1.0 - (seconds_remaining / 300.0)
    time_sig = 0.50 + 0.45 / (1.0 + math.exp(-8 * (time_frac - 0.6)))

    raw = magnitude * 0.35 + time_sig * 0.40 + 0.25
    return max(0.01, min(0.99, raw))


def _estimate_entry_price(win_prob: float) -> float:
    """
    Estimate entry price based on win probability.
    Higher confidence → market has moved further → higher entry price.
    Typical range: 0.50-0.62 for these markets.
    """
    base = 0.50
    premium = (win_prob - 0.50) * 0.6
    return round(min(0.65, max(0.50, base + premium)), 2)


# ── Output ────────────────────────────────────────────────────

def print_results(
    trades: list[BacktestTrade],
    bankroll: Bankroll,
    hours_back: int,
    total_rounds: int,
):
    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    total_pnl = sum(t.pnl for t in trades)
    total_wagered = sum(t.bet_dollars for t in trades)

    print()
    print("=" * 72)
    print("  FULLY REAL HISTORICAL BACKTEST")
    print("  BTC prices: Binance  |  Outcomes: Polymarket  |  Fees: real")
    print("=" * 72)
    print(f"  Period:               Last {hours_back} hours")
    print(f"  Resolved rounds:      {total_rounds}")
    print(f"  Trades taken:         {len(trades)}")
    print(f"  Trades skipped:       {total_rounds - len(trades)}")
    print()

    if not trades:
        print("  No trades met the strategy criteria.")
        print("=" * 72)
        return

    print(f"  Wins:                 {len(wins)}")
    print(f"  Losses:               {len(losses)}")
    wr = len(wins) / len(trades) * 100
    print(f"  Win rate:             {wr:.1f}%")
    print()
    print(f"  Total wagered:        ${total_wagered:.2f}")
    print(f"  Total PnL:            ${total_pnl:+.2f}")
    print(f"  ROI:                  {total_pnl/bankroll.starting_balance*100:+.1f}%")
    print()
    print(f"  Starting bankroll:    ${bankroll.starting_balance:.2f}")
    print(f"  Final bankroll:       ${bankroll.current_balance:.2f}")
    print(f"  Peak bankroll:        ${bankroll.peak_balance:.2f}")
    print(f"  Max drawdown:         {bankroll.drawdown*100:.1f}%")

    avg_edge = sum(t.edge for t in trades) / len(trades)
    avg_kelly = sum(t.kelly_fraction for t in trades) / len(trades)
    avg_bet = total_wagered / len(trades)
    avg_pnl = total_pnl / len(trades)
    avg_move = sum(t.btc_move_pct for t in trades) / len(trades)
    avg_wp = sum(t.win_prob_est for t in trades) / len(trades)

    gross_win = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print()
    print("  " + "-" * 54)
    print(f"  Avg BTC move at eval: {avg_move:.4f}%")
    print(f"  Avg win probability:  {avg_wp*100:.1f}%")
    print(f"  Avg edge (fee-adj):   {avg_edge*100:.2f}%")
    print(f"  Avg Kelly fraction:   {avg_kelly*100:.2f}%")
    print(f"  Avg bet size:         ${avg_bet:.2f}")
    print(f"  Avg PnL per trade:    ${avg_pnl:.4f}")
    print(f"  Profit factor:        {pf:.2f}")

    up_outcomes = sum(1 for t in trades if t.outcome == "up")
    dn_outcomes = len(trades) - up_outcomes
    print()
    print("  " + "-" * 54)
    print(f"  Direction accuracy:")
    correct = sum(1 for t in trades if t.direction == t.outcome)
    print(f"    Bot correct:        {correct}/{len(trades)} ({correct/len(trades)*100:.1f}%)")
    print(f"    Market up:          {up_outcomes} ({up_outcomes/len(trades)*100:.0f}%)")
    print(f"    Market down:        {dn_outcomes} ({dn_outcomes/len(trades)*100:.0f}%)")

    max_ws = max_ls = cw = cl = 0
    for t in trades:
        if t.won:
            cw += 1; cl = 0; max_ws = max(max_ws, cw)
        else:
            cl += 1; cw = 0; max_ls = max(max_ls, cl)
    print()
    print(f"  Max win streak:       {max_ws}")
    print(f"  Max loss streak:      {max_ls}")

    print()
    print("  " + "-" * 54)
    print("  TRADE LOG (last 30):")
    hdr = (f"    {'Time':>14}  {'Result':>5}  {'Pick':>4}  {'Real':>4}  "
           f"{'BTC Move':>9}  {'P(w)':>5}  {'Bet':>8}  {'PnL':>9}  {'Balance':>9}")
    print(hdr)
    print("    " + "-" * 84)

    bal = bankroll.starting_balance
    bals = []
    for t in trades:
        bal += t.pnl
        bals.append(bal)

    for i, t in enumerate(trades[-30:]):
        idx = len(trades) - 30 + i if len(trades) > 30 else i
        res = " WIN" if t.won else "LOSS"
        ts_str = time.strftime("%m/%d %H:%M", time.gmtime(t.timestamp))
        move_str = f"{t.btc_move_pct:+.4f}%"
        print(
            f"    {ts_str}  {res:>5}  {t.direction:>4}  {t.outcome:>4}  "
            f"{move_str:>9}  {t.win_prob_est:.2f}  "
            f"${t.bet_dollars:>7.2f}  ${t.pnl:>+8.4f}  ${bals[idx]:>8.2f}"
        )

    print()
    print("=" * 72)
    print("  WHAT IS REAL:")
    print("    - BTC prices at round start & evaluation: Binance 1m klines")
    print("    - Direction decision: real BTC price comparison (not random)")
    print("    - Win/loss outcome: actual Polymarket on-chain resolution")
    print("    - Fees: Polymarket crypto fee formula")
    print("  WHAT IS ESTIMATED:")
    print("    - Entry price (no historical order book available)")
    print("=" * 72)
    print()


def save_trade_log(trades: list[BacktestTrade], filepath: Path):
    data = [asdict(t) for t in trades]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2))
    log.info("Trade log saved to %s (%d trades)", filepath, len(data))


def main():
    parser = argparse.ArgumentParser(
        description="Fully real backtest: Binance prices + Polymarket outcomes"
    )
    parser.add_argument("--hours", type=int, default=24,
                        help="Hours of history (default: 24)")
    parser.add_argument("--bankroll", type=float, default=250.0,
                        help="Starting bankroll USD (default: 250)")
    parser.add_argument("--kelly", type=float, default=0.25,
                        help="Kelly multiplier (default: 0.25)")
    parser.add_argument("--min-edge", type=float, default=0.03,
                        help="Min edge to trade (default: 0.03)")
    parser.add_argument("--confidence-floor", type=float, default=0.55,
                        help="Min win probability (default: 0.55)")
    parser.add_argument("--max-bet", type=float, default=50.0,
                        help="Max bet per trade in USD — reflects real "
                             "Polymarket liquidity (default: 50)")
    parser.add_argument("--save", action="store_true",
                        help="Save bankroll.json and trade_log.json")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  FULLY REAL BACKTEST — Binance prices + Polymarket outcomes")
    print("=" * 72)
    print(f"  Period: last {args.hours} hours  |  Bankroll: ${args.bankroll:.2f}")
    print(f"  Kelly: {args.kelly:.0%}  |  Min edge: {args.min_edge*100:.1f}%  |  Max bet: ${args.max_bet:.0f}")
    print("=" * 72)
    print()

    # Step 1: fetch BTC price history from Binance
    now = int(time.time())
    start = now - (args.hours * 3600)
    btc_prices = fetch_btc_price_history(start - 300, now)

    if len(btc_prices) < 10:
        print("  ERROR: Could not fetch BTC price history from Binance.")
        print("  (May be geo-blocked. Try with a VPN or different network.)")
        return

    # Step 2: fetch resolved rounds from Polymarket
    rounds = fetch_resolved_rounds(args.hours)
    if not rounds:
        print("  No resolved rounds found.")
        return

    up_ct = sum(1 for r in rounds if r.outcome == "up")
    dn_ct = len(rounds) - up_ct
    log.info("Real outcomes: %d up (%.0f%%) / %d down (%.0f%%)",
             up_ct, up_ct / len(rounds) * 100, dn_ct, dn_ct / len(rounds) * 100)

    # Step 3: replay strategy with real data
    trades, bankroll = replay_strategy(
        rounds=rounds,
        btc_prices=btc_prices,
        starting_bankroll=args.bankroll,
        kelly_mult=args.kelly,
        min_edge=args.min_edge,
        confidence_floor=args.confidence_floor,
        max_bet=args.max_bet,
    )

    print_results(trades, bankroll, args.hours, len(rounds))

    if args.save and trades:
        save_bankroll(bankroll)
        log.info("Bankroll saved to %s", BANKROLL_FILE)

        log_file = Path(__file__).resolve().parent.parent / "data" / "trade_log.json"
        save_trade_log(trades, log_file)

        print(f"  Saved: {BANKROLL_FILE}")
        print(f"  Saved: {log_file}")
        print()


if __name__ == "__main__":
    main()
