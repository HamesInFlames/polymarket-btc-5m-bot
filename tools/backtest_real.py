"""
Historical Backtest with FULLY REAL Data (v2)
=============================================
Everything in this backtest is real:

  1. BTC PRICES — fetched from Binance 1-minute klines (historical).
  2. OUTCOMES — fetched from Polymarket's Gamma API.
  3. STRATEGY DECISIONS — compares real BTC price at evaluation
     to real opening price, exactly like the live bot.
  4. FEES & SIZING — real Polymarket fee formula, ruin-calibrated Kelly.

v2 additions:
  - Feature vector collection per trade for model training.
  - Logistic model weight optimiser (gradient-free Nelder-Mead).
  - Risk-of-ruin estimate on the backtest equity curve.
  - Sharpe / Sortino / Calmar ratios.
  - Drawdown analysis.
  - --train flag to optimise and save model weights.

Usage:
    python tools/backtest_real.py
    python tools/backtest_real.py --hours 48 --bankroll 250 --save
    python tools/backtest_real.py --hours 168 --train
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
from src.kelly import (
    kelly_criterion, Bankroll, save_bankroll, BANKROLL_FILE,
    risk_of_ruin, expected_growth_rate, optimal_kelly_for_ruin,
)
from src.market_reader import ROUND_DURATION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest")

GAMMA_DELAY = 0.12
BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"
EVAL_OFFSET = 120


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
    ruin_prob: float = 0.0
    quality_grade: str = ""
    features: dict = None


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


# ── Feature extraction (mirrors live strategy) ───────────────

def _sigmoid(x: float, midpoint: float = 0.5, steepness: float = 10.0) -> float:
    z = steepness * (x - midpoint)
    z = max(-20, min(20, z))
    return 1.0 / (1.0 + math.exp(-z))


def _extract_backtest_features(
    pct_move: float,
    seconds_remaining: float,
    btc_prices: dict[int, float],
    round_start_ts: int,
    eval_ts: int,
    opening_price: float,
    direction: str,
) -> dict:
    """
    Build the same feature vector as the live strategy, adapted for
    backtest (no real-time order book or multi-source oracle).
    """
    magnitude_scaled = _sigmoid(pct_move, midpoint=0.0004, steepness=8000)

    time_fraction = 1.0 - (seconds_remaining / 300.0)
    time_pressure = _sigmoid(time_fraction, midpoint=0.6, steepness=8)

    # Momentum consistency from 1-minute bars
    bar_prices = []
    for ts in range(round_start_ts, eval_ts + 60, 60):
        p = btc_prices.get(ts - (ts % 60))
        if p is not None:
            bar_prices.append(p)

    if len(bar_prices) >= 3:
        deltas = [bar_prices[i] - bar_prices[i-1] for i in range(1, len(bar_prices))]
        if direction == "up":
            consistent = sum(1 for d in deltas if d > 0)
        else:
            consistent = sum(1 for d in deltas if d < 0)
        momentum_consistency = 0.3 + 0.7 * (consistent / len(deltas))
    else:
        momentum_consistency = 0.5

    # Momentum acceleration
    if len(bar_prices) >= 4:
        deltas = [bar_prices[i] - bar_prices[i-1] for i in range(1, len(bar_prices))]
        mid = len(deltas) // 2
        early_mag = sum(abs(d) for d in deltas[:mid]) / max(1, mid)
        late_mag = sum(abs(d) for d in deltas[mid:]) / max(1, len(deltas) - mid)
        accel_ratio = late_mag / early_mag if early_mag > 0 else 1.0
        if direction == "up":
            late_dir = sum(1 for d in deltas[mid:] if d > 0) / max(1, len(deltas) - mid)
        else:
            late_dir = sum(1 for d in deltas[mid:] if d < 0) / max(1, len(deltas) - mid)
        momentum_accel = 0.5 * min(1.0, accel_ratio / 2.0) + 0.5 * late_dir
    else:
        momentum_accel = 0.5

    oracle_agreement = 0.7
    chainlink_agrees = 0.7

    # Volatility regime from bars before round start
    pre_prices = []
    for ts in range(round_start_ts - 600, round_start_ts, 60):
        p = btc_prices.get(ts - (ts % 60))
        if p is not None:
            pre_prices.append(p)

    if len(pre_prices) >= 3:
        returns = [(pre_prices[i] - pre_prices[i-1]) / pre_prices[i-1]
                    for i in range(1, len(pre_prices)) if pre_prices[i-1] > 0]
        if returns:
            mean_r = sum(returns) / len(returns)
            var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            vol = max(0.00005, var ** 0.5)
        else:
            vol = 0.0003
    else:
        vol = 0.0003
    vol_norm = min(1.0, vol / 0.001)

    # Higher-timeframe trend from bars before round start
    if len(pre_prices) >= 4:
        q1_end = len(pre_prices) // 2
        avg_early = sum(pre_prices[:q1_end]) / q1_end
        avg_late = sum(pre_prices[q1_end:]) / (len(pre_prices) - q1_end)
        trend_pct = (avg_late - avg_early) / avg_early if avg_early > 0 else 0
        if direction == "up":
            trend_alignment = _sigmoid(trend_pct, midpoint=0.0, steepness=5000)
        else:
            trend_alignment = _sigmoid(-trend_pct, midpoint=0.0, steepness=5000)
    else:
        trend_alignment = 0.5

    book_imbalance = 0.5
    spread_quality = 0.6

    return {
        "magnitude_scaled": magnitude_scaled,
        "time_pressure": time_pressure,
        "momentum_consistency": momentum_consistency,
        "oracle_agreement": oracle_agreement,
        "chainlink_agrees": chainlink_agrees,
        "volatility_regime": vol_norm,
        "trend_alignment": trend_alignment,
        "book_imbalance": book_imbalance,
        "spread_quality": spread_quality,
        "momentum_acceleration": momentum_accel,
    }


def _logistic_predict(features: dict, weights: dict) -> float:
    """Apply logistic model with given weights."""
    z = weights.get("intercept", -1.8)
    for feat_name, feat_val in features.items():
        w = weights.get(feat_name, 0.0)
        z += w * feat_val
    z = max(-15, min(15, z))
    return max(0.01, min(0.99, 1.0 / (1.0 + math.exp(-z))))


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
    ruin_target: float = 0.05,
    ruin_level: float = 0.10,
    weights: dict = None,
) -> tuple[list[BacktestTrade], Bankroll]:
    """
    Replay the strategy with REAL data.
    Now uses the logistic model for win-prob estimation and
    ruin-calibrated Kelly for sizing.
    """
    if weights is None:
        try:
            from src.strategy import _load_model_weights
            weights = _load_model_weights()
        except Exception:
            from src.strategy import _DEFAULT_WEIGHTS
            weights = dict(_DEFAULT_WEIGHTS)

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

        features = _extract_backtest_features(
            pct_move=pct_move,
            seconds_remaining=EVAL_OFFSET,
            btc_prices=btc_prices,
            round_start_ts=rnd.start_ts,
            eval_ts=eval_ts,
            opening_price=opening_price,
            direction=direction,
        )

        win_prob = _logistic_predict(features, weights)

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
            ruin_target=ruin_target,
            ruin_level=ruin_level,
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
            ruin_prob=round(rec.ruin_prob, 4),
            quality_grade=rec.quality_grade,
            features=features,
        ))

    log.info(
        "Replay done: %d trades | skipped: %d no price, %d no move, %d no edge",
        len(trades), skipped_no_price, skipped_no_move, skipped_no_edge,
    )
    return trades, br


def _estimate_entry_price(win_prob: float) -> float:
    base = 0.50
    premium = (win_prob - 0.50) * 0.6
    return round(min(0.65, max(0.50, base + premium)), 2)


# ── Model weight optimiser ───────────────────────────────────

def train_model_weights(
    rounds: list[ResolvedRound],
    btc_prices: dict[int, float],
    starting_bankroll: float,
    kelly_mult: float,
    min_edge: float,
    confidence_floor: float,
    max_bet: float,
    ruin_target: float = 0.05,
    ruin_level: float = 0.10,
) -> dict:
    """
    Optimise model weights to maximise the backtest objective:
    log(final_bankroll) - penalty * max_drawdown.

    Uses Nelder-Mead (no gradients needed) with a small number of
    parameters. Falls back to random search if scipy is not available.
    """
    from src.strategy import _DEFAULT_WEIGHTS

    param_names = [k for k in _DEFAULT_WEIGHTS if k != "intercept"]
    initial = [_DEFAULT_WEIGHTS[k] for k in param_names]
    best_intercept = _DEFAULT_WEIGHTS["intercept"]

    def objective(params):
        weights = {"intercept": best_intercept}
        for name, val in zip(param_names, params):
            weights[name] = val

        trades, br = replay_strategy(
            rounds=rounds,
            btc_prices=btc_prices,
            starting_bankroll=starting_bankroll,
            kelly_mult=kelly_mult,
            min_edge=min_edge,
            confidence_floor=confidence_floor,
            max_bet=max_bet,
            ruin_target=ruin_target,
            ruin_level=ruin_level,
            weights=weights,
        )

        if not trades or br.current_balance <= 0:
            return 1000.0

        log_return = math.log(br.current_balance / starting_bankroll)
        dd_penalty = br.max_drawdown * 2.0
        # Penalise low trade count (want at least some trades for statistical significance)
        trade_penalty = max(0, 10 - len(trades)) * 0.1

        return -(log_return - dd_penalty - trade_penalty)

    try:
        from scipy.optimize import minimize as scipy_minimize
        log.info("Optimising weights with Nelder-Mead (scipy)...")

        result = scipy_minimize(
            objective, initial, method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 0.05, "fatol": 0.01},
        )
        best_params = result.x
        log.info("Optimisation done: fun=%.4f, iterations=%d", result.fun, result.nit)

    except ImportError:
        log.info("scipy not available — using random search (100 iterations)...")
        import random
        best_score = objective(initial)
        best_params = list(initial)

        for i in range(100):
            candidate = [v + random.gauss(0, 0.3) for v in best_params]
            score = objective(candidate)
            if score < best_score:
                best_score = score
                best_params = candidate
                log.info("  iter %d: new best score=%.4f", i, -score)

    optimised = {"intercept": best_intercept}
    for name, val in zip(param_names, best_params):
        optimised[name] = round(val, 4)

    return optimised


# ── Analytics ─────────────────────────────────────────────────

def compute_analytics(trades: list[BacktestTrade], bankroll: Bankroll) -> dict:
    """Compute comprehensive backtest analytics."""
    if not trades:
        return {}

    pnls = [t.pnl for t in trades]
    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]

    total_pnl = sum(pnls)
    total_wagered = sum(t.bet_dollars for t in trades)

    # Equity curve
    equity = [bankroll.starting_balance]
    for t in trades:
        equity.append(equity[-1] + t.pnl)

    # Max drawdown from equity curve
    peak = equity[0]
    max_dd = 0.0
    max_dd_duration = 0
    current_dd_start = 0

    for i, e in enumerate(equity):
        if e > peak:
            peak = e
            current_dd_start = i
        dd = (peak - e) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = i - current_dd_start

    # Sharpe ratio (per-trade, then annualised assuming 288 trades/day for 5m intervals)
    mean_pnl = sum(pnls) / len(pnls)
    var_pnl = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
    std_pnl = var_pnl ** 0.5
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
    sharpe_annual = sharpe * (288 ** 0.5)

    # Sortino ratio (only downside deviation)
    downside = [p for p in pnls if p < 0]
    if downside:
        downside_var = sum(p ** 2 for p in downside) / len(pnls)
        downside_std = downside_var ** 0.5
        sortino = mean_pnl / downside_std if downside_std > 0 else 0.0
    else:
        sortino = float("inf") if mean_pnl > 0 else 0.0

    # Calmar ratio
    calmar = (total_pnl / bankroll.starting_balance) / max_dd if max_dd > 0 else 0.0

    # Profit factor
    gross_win = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Win/loss streaks
    max_ws = max_ls = cw = cl = 0
    for t in trades:
        if t.won:
            cw += 1; cl = 0; max_ws = max(max_ws, cw)
        else:
            cl += 1; cw = 0; max_ls = max(max_ls, cl)

    # Average win/loss
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

    # Risk of ruin estimate from observed stats
    if trades:
        avg_edge_val = sum(t.edge for t in trades) / len(trades)
        avg_kelly_val = sum(t.kelly_fraction for t in trades) / len(trades)
        avg_price = sum(t.entry_price for t in trades) / len(trades)
        obs_win_rate = len(wins) / len(trades)
        ruin_est = risk_of_ruin(obs_win_rate, avg_price, avg_kelly_val, 0.1)
    else:
        avg_edge_val = avg_kelly_val = ruin_est = 0

    # Grade distribution
    grades = {"A": 0, "B": 0, "C": 0, "D": 0}
    for t in trades:
        g = t.quality_grade if t.quality_grade in grades else "D"
        grades[g] += 1

    # Grade-level win rates
    grade_wr = {}
    for g in ["A", "B", "C", "D"]:
        g_trades = [t for t in trades if t.quality_grade == g]
        if g_trades:
            grade_wr[g] = sum(1 for t in g_trades if t.won) / len(g_trades)

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades),
        "total_pnl": total_pnl,
        "total_wagered": total_wagered,
        "roi": total_pnl / bankroll.starting_balance,
        "starting_bankroll": bankroll.starting_balance,
        "final_bankroll": bankroll.current_balance,
        "peak_bankroll": bankroll.peak_balance,
        "max_drawdown": max_dd,
        "max_dd_duration_trades": max_dd_duration,
        "sharpe_per_trade": sharpe,
        "sharpe_annualised": sharpe_annual,
        "sortino": sortino,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "expectancy": mean_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_edge": avg_edge_val,
        "avg_kelly": avg_kelly_val,
        "risk_of_ruin_est": ruin_est,
        "max_win_streak": max_ws,
        "max_loss_streak": max_ls,
        "grade_distribution": grades,
        "grade_win_rates": grade_wr,
    }


# ── Output ────────────────────────────────────────────────────

def print_results(
    trades: list[BacktestTrade],
    bankroll: Bankroll,
    hours_back: int,
    total_rounds: int,
    analytics: dict,
):
    print()
    print("=" * 72)
    print("  FULLY REAL HISTORICAL BACKTEST (v2)")
    print("  BTC prices: Binance  |  Outcomes: Polymarket  |  Fees: real")
    print("  Model: Logistic regression  |  Sizing: Ruin-calibrated Kelly")
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

    a = analytics
    print(f"  Wins:                 {a['wins']}")
    print(f"  Losses:               {a['losses']}")
    print(f"  Win rate:             {a['win_rate']*100:.1f}%")
    print()
    print(f"  Total wagered:        ${a['total_wagered']:.2f}")
    print(f"  Total PnL:            ${a['total_pnl']:+.2f}")
    print(f"  ROI:                  {a['roi']*100:+.1f}%")
    print()
    print(f"  Starting bankroll:    ${a['starting_bankroll']:.2f}")
    print(f"  Final bankroll:       ${a['final_bankroll']:.2f}")
    print(f"  Peak bankroll:        ${a['peak_bankroll']:.2f}")
    print(f"  Max drawdown:         {a['max_drawdown']*100:.1f}% ({a['max_dd_duration_trades']} trades)")

    print()
    print("  " + "-" * 54)
    print("  RISK-ADJUSTED RETURNS:")
    print(f"    Sharpe (per-trade): {a['sharpe_per_trade']:.3f}")
    print(f"    Sharpe (annual):    {a['sharpe_annualised']:.2f}")
    print(f"    Sortino:            {a['sortino']:.3f}")
    print(f"    Calmar:             {a['calmar']:.2f}")
    print(f"    Profit factor:      {a['profit_factor']:.2f}")

    print()
    print("  " + "-" * 54)
    print("  TRADE STATISTICS:")
    print(f"    Expectancy:         ${a['expectancy']:.4f}/trade")
    print(f"    Avg win:            ${a['avg_win']:.4f}")
    print(f"    Avg loss:           ${a['avg_loss']:.4f}")
    print(f"    Avg edge (fee-adj): {a['avg_edge']*100:.2f}%")
    print(f"    Avg Kelly fraction: {a['avg_kelly']*100:.2f}%")
    print(f"    Risk of ruin est:   {a['risk_of_ruin_est']*100:.2f}%")

    print()
    print("  " + "-" * 54)
    print("  STREAKS:")
    print(f"    Max win streak:     {a['max_win_streak']}")
    print(f"    Max loss streak:    {a['max_loss_streak']}")

    grades = a.get("grade_distribution", {})
    grade_wr = a.get("grade_win_rates", {})
    if any(v > 0 for v in grades.values()):
        print()
        print("  " + "-" * 54)
        print("  BET QUALITY GRADES:")
        for g in ["A", "B", "C", "D"]:
            count = grades.get(g, 0)
            wr = grade_wr.get(g, 0)
            if count > 0:
                print(f"    Grade {g}: {count:>4} trades  ({wr*100:.1f}% win rate)")

    up_outcomes = sum(1 for t in trades if t.outcome == "up")
    dn_outcomes = len(trades) - up_outcomes
    print()
    print("  " + "-" * 54)
    print(f"  Direction accuracy:")
    correct = sum(1 for t in trades if t.direction == t.outcome)
    print(f"    Bot correct:        {correct}/{len(trades)} ({correct/len(trades)*100:.1f}%)")
    print(f"    Market up:          {up_outcomes} ({up_outcomes/len(trades)*100:.0f}%)")
    print(f"    Market down:        {dn_outcomes} ({dn_outcomes/len(trades)*100:.0f}%)")

    print()
    print("  " + "-" * 54)
    print("  TRADE LOG (last 30):")
    hdr = (f"    {'Time':>14}  {'Res':>4} {'Grd':>3}  {'Pick':>4}  {'Real':>4}  "
           f"{'BTC Move':>9}  {'P(w)':>5}  {'Bet':>8}  {'PnL':>9}  {'Balance':>9}")
    print(hdr)
    print("    " + "-" * 88)

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
            f"    {ts_str}  {res:>4} {t.quality_grade:>3}  {t.direction:>4}  {t.outcome:>4}  "
            f"{move_str:>9}  {t.win_prob_est:.2f}  "
            f"${t.bet_dollars:>7.2f}  ${t.pnl:>+8.4f}  ${bals[idx]:>8.2f}"
        )

    print()
    print("=" * 72)
    print("  WHAT IS REAL:")
    print("    - BTC prices at round start & evaluation: Binance 1m klines")
    print("    - Direction decision: real BTC price comparison")
    print("    - Win/loss outcome: actual Polymarket on-chain resolution")
    print("    - Fees: Polymarket crypto fee formula")
    print("    - Sizing: Ruin-calibrated Kelly criterion")
    print("  WHAT IS ESTIMATED:")
    print("    - Entry price (no historical order book)")
    print("    - Oracle agreement / book imbalance features")
    print("=" * 72)
    print()


def save_trade_log(trades: list[BacktestTrade], filepath: Path):
    data = []
    for t in trades:
        d = asdict(t)
        if d.get("features"):
            d["features"] = {k: round(v, 4) for k, v in d["features"].items()}
        data.append(d)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2))
    log.info("Trade log saved to %s (%d trades)", filepath, len(data))


def main():
    parser = argparse.ArgumentParser(
        description="Fully real backtest v2: Binance prices + Polymarket outcomes"
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
                        help="Max bet per trade in USD (default: 50)")
    parser.add_argument("--ruin-target", type=float, default=0.05,
                        help="Max acceptable risk of ruin (default: 0.05)")
    parser.add_argument("--ruin-level", type=float, default=0.10,
                        help="Fraction of bankroll that = ruin (default: 0.10)")
    parser.add_argument("--save", action="store_true",
                        help="Save bankroll.json and trade_log.json")
    parser.add_argument("--train", action="store_true",
                        help="Train & save optimal model weights")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  FULLY REAL BACKTEST v2 — Binance prices + Polymarket outcomes")
    print("=" * 72)
    print(f"  Period: last {args.hours} hours  |  Bankroll: ${args.bankroll:.2f}")
    print(f"  Kelly: {args.kelly:.0%}  |  Min edge: {args.min_edge*100:.1f}%  |  Max bet: ${args.max_bet:.0f}")
    print(f"  Ruin target: {args.ruin_target:.0%}  |  Ruin level: {args.ruin_level:.0%}")
    if args.train:
        print(f"  MODE: TRAINING — will optimise and save model weights")
    print("=" * 72)
    print()

    now = int(time.time())
    start = now - (args.hours * 3600)
    btc_prices = fetch_btc_price_history(start - 300, now)

    if len(btc_prices) < 10:
        print("  ERROR: Could not fetch BTC price history from Binance.")
        print("  (May be geo-blocked. Try with a VPN or different network.)")
        return

    rounds = fetch_resolved_rounds(args.hours)
    if not rounds:
        print("  No resolved rounds found.")
        return

    up_ct = sum(1 for r in rounds if r.outcome == "up")
    dn_ct = len(rounds) - up_ct
    log.info("Real outcomes: %d up (%.0f%%) / %d down (%.0f%%)",
             up_ct, up_ct / len(rounds) * 100, dn_ct, dn_ct / len(rounds) * 100)

    # Optionally train model weights
    if args.train:
        print("  Training model weights (this may take a few minutes)...")
        optimised = train_model_weights(
            rounds=rounds,
            btc_prices=btc_prices,
            starting_bankroll=args.bankroll,
            kelly_mult=args.kelly,
            min_edge=args.min_edge,
            confidence_floor=args.confidence_floor,
            max_bet=args.max_bet,
            ruin_target=args.ruin_target,
            ruin_level=args.ruin_level,
        )
        from src.strategy import save_model_weights
        save_model_weights(optimised)
        print(f"  Optimised weights saved to data/model_weights.json")
        print(f"  Weights: {json.dumps(optimised, indent=2)}")
        print()

    trades, bankroll = replay_strategy(
        rounds=rounds,
        btc_prices=btc_prices,
        starting_bankroll=args.bankroll,
        kelly_mult=args.kelly,
        min_edge=args.min_edge,
        confidence_floor=args.confidence_floor,
        max_bet=args.max_bet,
        ruin_target=args.ruin_target,
        ruin_level=args.ruin_level,
    )

    analytics = compute_analytics(trades, bankroll)
    print_results(trades, bankroll, args.hours, len(rounds), analytics)

    if args.save and trades:
        save_bankroll(bankroll)
        log.info("Bankroll saved to %s", BANKROLL_FILE)

        log_file = Path(__file__).resolve().parent.parent / "data" / "trade_log.json"
        save_trade_log(trades, log_file)

        analytics_file = Path(__file__).resolve().parent.parent / "data" / "backtest_analytics.json"
        analytics_file.parent.mkdir(parents=True, exist_ok=True)
        analytics_file.write_text(json.dumps(analytics, indent=2, default=str))
        log.info("Analytics saved to %s", analytics_file)

        print(f"  Saved: {BANKROLL_FILE}")
        print(f"  Saved: {log_file}")
        print(f"  Saved: {analytics_file}")
        print()


if __name__ == "__main__":
    main()
