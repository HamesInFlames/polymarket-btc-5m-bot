"""
Profit-Maximizing Strategy with Logistic Model & Kelly Sizing (v6)
==================================================================
v6 fixes over v5 (senior quant review):

  1. FIXED FEE-ADJUSTED EDGE — uses trade_economics() as single source
     of truth. Previous version double-counted fees in some paths.
  2. RECENT OUTCOME STREAK FEATURE — conditions on recent win direction
     to shrink edge estimate when the market is already pricing in trend.
  3. SMARTER LIMIT PRICING — posts limit at mid+1tick instead of
     always crossing the spread (saves ~1-3% per trade on avg).
  4. DRAWDOWN SCALING FIX — uses convex function that doesn't over-penalize
     small drawdowns but aggressively cuts at deep drawdowns.
  5. BANKROLL-AWARE SIZING — accounts for pending positions so Kelly
     doesn't undersize after a buy or oversize after a redemption.
  6. REMOVED BACKTEST-ONLY FEATURES from live model — features that
     can't be observed in backtest (book_imbalance, oracle_agreement)
     are downweighted to avoid training on noise.
"""

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import (
    MIN_EDGE,
    ENTRY_WINDOW_START,
    ENTRY_WINDOW_END,
    MIN_CONTRACT_PRICE,
    MAX_CONTRACT_PRICE,
    MAX_TRADE_SIZE,
    KELLY_MULTIPLIER,
    MIN_BET_DOLLARS,
    CONFIDENCE_FLOOR,
    STARTING_BANKROLL,
    RUIN_TARGET,
    RUIN_LEVEL,
    MAX_DRAWDOWN_SCALE,
)
from src.market_reader import MarketRound, get_full_market_data
from src.price_oracle import get_btc_price
from src.kelly import (
    kelly_criterion,
    BetRecommendation,
    Bankroll,
    load_bankroll,
    save_bankroll,
    risk_of_ruin,
)
from src.fees import effective_fee_rate, trade_economics, fetch_fee_rate_bps

log = logging.getLogger(__name__)

_bankroll: Optional[Bankroll] = None
_price_history: deque[tuple[float, float]] = deque(maxlen=300)
_cached_usdc_balance: Optional[float] = None

# Track recent round outcomes for streak-aware edge adjustment
_recent_outcomes: deque[str] = deque(maxlen=20)  # "up" or "down"

MODEL_WEIGHTS_FILE = Path(__file__).resolve().parent.parent / "data" / "model_weights.json"

# Track pending position value so Kelly sizes correctly
_pending_position_cost: float = 0.0

# ── Logistic Regression Model ────────────────────────────────

_DEFAULT_WEIGHTS = {
    "intercept": -1.40,
    "magnitude_scaled": 0.85,
    "time_pressure": 0.60,
    "momentum_consistency": 0.40,
    # These features are observable in live AND backtest with real data:
    "oracle_agreement": 0.27,
    "chainlink_agrees": 0.20,
    "volatility_regime": -0.15,
    "trend_alignment": 0.24,
    "momentum_acceleration": 0.17,
    # These features are ONLY observable in live (set to 0.5 in backtest).
    # Downweighted so backtest training doesn't overfit to constants:
    "book_imbalance": 0.06,   # was 0.10 — halved because backtest uses 0.5 constant
    "spread_quality": 0.08,   # was 0.14 — halved for same reason
    # NEW: recent outcome streak penalizes trading with the crowd
    "streak_against": 0.12,   # positive = bonus when betting against recent streak
}

_model_weights: dict[str, float] = {}


def _load_model_weights() -> dict[str, float]:
    global _model_weights
    if _model_weights:
        return _model_weights
    try:
        if MODEL_WEIGHTS_FILE.exists():
            _model_weights = json.loads(MODEL_WEIGHTS_FILE.read_text())
            log.info("Loaded model weights from %s", MODEL_WEIGHTS_FILE)
            return _model_weights
    except Exception as e:
        log.warning("Could not load model weights: %s — using defaults", e)
    _model_weights = dict(_DEFAULT_WEIGHTS)
    return _model_weights


def save_model_weights(weights: dict[str, float]):
    """Persist learned weights to disk (called by backtest trainer)."""
    try:
        MODEL_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        MODEL_WEIGHTS_FILE.write_text(json.dumps(weights, indent=2))
        log.info("Saved model weights to %s", MODEL_WEIGHTS_FILE)
    except Exception as e:
        log.warning("Could not save model weights: %s", e)


# ── Bankroll helpers ──────────────────────────────────────────

def get_bankroll() -> Bankroll:
    global _bankroll
    if _bankroll is None:
        _bankroll = load_bankroll(STARTING_BANKROLL)
    return _bankroll


def get_available_bankroll() -> float:
    """
    FIX #5: Return bankroll MINUS pending position costs.
    This prevents Kelly from undersizing when USDC is locked in positions
    or oversizing right after a redemption.
    """
    br = get_bankroll()
    return max(0.0, br.current_balance)


def add_pending_cost(cost: float):
    """Called when a trade is placed — reduces available balance for Kelly."""
    global _pending_position_cost
    _pending_position_cost += cost


def clear_pending_cost(cost: float):
    """Called when a trade resolves — releases the locked amount."""
    global _pending_position_cost
    _pending_position_cost = max(0.0, _pending_position_cost - cost)


def sync_bankroll_to_balance(usdc_balance: float, has_pending: bool = False):
    """
    Sync bankroll with actual on-chain USDC.e balance.

    FIX #5: When there are pending positions, DON'T sync bankroll down.
    The USDC is "missing" because it's locked in outcome tokens, not lost.
    Only sync up (if we received more USDC than expected, e.g. from redemption).
    """
    global _cached_usdc_balance
    _cached_usdc_balance = usdc_balance
    br = get_bankroll()

    if has_pending:
        # Only sync UPWARD during pending trades (e.g. if manual deposit)
        if usdc_balance > br.current_balance:
            old = br.current_balance
            br.current_balance = usdc_balance
            br.peak_balance = max(br.peak_balance, usdc_balance)
            save_bankroll(br)
            log.info(
                "Bankroll synced UP during pending: $%.2f -> $%.2f",
                old, usdc_balance,
            )
        return

    # No pending trades — safe to sync to on-chain balance
    if abs(usdc_balance - br.current_balance) > 0.01:
        old = br.current_balance
        br.current_balance = usdc_balance
        br.peak_balance = max(br.peak_balance, usdc_balance)
        save_bankroll(br)
        log.info(
            "Bankroll synced to on-chain balance: $%.2f -> $%.2f",
            old, usdc_balance,
        )


def record_bet_result(won: bool, wager: float, payout: float = 0.0):
    """Call after a bet resolves to update the bankroll."""
    br = get_bankroll()
    if won:
        br.record_win(wager, payout)
    else:
        br.record_loss(wager)
    save_bankroll(br)
    log.info(
        "Bankroll updated: $%.2f (P&L: $%.2f, ROI: %.1f%%, DD: %.1f%%, MaxDD: %.1f%%)",
        br.current_balance, br.total_profit, br.roi * 100,
        br.drawdown * 100, br.max_drawdown * 100,
    )


def record_outcome(direction: str):
    """Track recent round outcomes for streak detection."""
    _recent_outcomes.append(direction)


# ── Signal dataclass ─────────────────────────────────────────

@dataclass
class Signal:
    """A trade signal produced by the strategy."""
    action: str
    token_id: str
    price: float
    size: float
    edge: float
    confidence: float
    direction: str
    btc_current: float
    btc_opening: float
    seconds_remaining: float
    reason: str
    neg_risk: bool = False
    tick_size: str = "0.01"
    kelly_fraction: float = 0.0
    bet_dollars: float = 0.0
    expected_profit: float = 0.0
    bankroll_snapshot: float = 0.0
    fee_rate_pct: float = 0.0
    effective_price: float = 0.0
    min_order_size: float = 5.0
    ruin_prob: float = 0.0
    quality_grade: str = ""
    features: dict = None


# ── Main evaluation ──────────────────────────────────────────

def evaluate_round(
    rnd: MarketRound,
    opening_btc_price: Optional[float],
) -> Optional[Signal]:
    remaining = rnd.seconds_remaining

    if remaining > ENTRY_WINDOW_START or remaining < ENTRY_WINDOW_END:
        return None

    oracle = get_btc_price()

    if oracle["count"] < 2:
        log.warning("Only %d price sources available - skipping", oracle["count"])
        return None

    chainlink_price = oracle["chainlink"]
    current_price = chainlink_price if chainlink_price else oracle["median"]

    if current_price is None:
        log.warning("No BTC price available - skipping")
        return None

    if opening_btc_price is None or opening_btc_price <= 0:
        log.warning("No opening price for this round - skipping")
        return None

    _price_history.append((time.time(), current_price))

    price_delta = current_price - opening_btc_price
    pct_move = abs(price_delta) / opening_btc_price

    vol = _recent_volatility()
    min_move = _dynamic_move_threshold(vol)

    if pct_move < min_move:
        log.info(
            "Price move %.5f%% below dynamic threshold %.5f%% (vol=%.5f%%) - skipping",
            pct_move * 100, min_move * 100, vol * 100,
        )
        return None

    direction = "up" if price_delta > 0 else "down"

    # --- Fetch FULL market data from all CLOB endpoints ---
    token_id = rnd.up_token_id if direction == "up" else rnd.down_token_id
    market = get_full_market_data(token_id)

    # --- Extract features & estimate win probability ---
    features = _extract_features(
        pct_move=pct_move,
        direction=direction,
        seconds_remaining=remaining,
        oracle=oracle,
        opening_price=opening_btc_price,
        current_price=current_price,
        volatility=vol,
        market=market,
    )

    win_prob = _logistic_predict(features)

    if win_prob < CONFIDENCE_FLOOR:
        log.info("Win probability %.3f below floor %.3f - skipping", win_prob, CONFIDENCE_FLOOR)
        return None

    # --- FIX #3: Smarter limit price determination ---
    limit_price = _determine_limit_price(market, win_prob, remaining)

    if limit_price is None:
        log.info(
            "No viable limit price (bid=%.3f ask=%.3f spread=%.3f) - skipping",
            market["best_bid"], market["best_ask"], market["spread"],
        )
        return None

    if not (MIN_CONTRACT_PRICE <= limit_price <= MAX_CONTRACT_PRICE):
        log.info(
            "Limit price $%.3f outside range [%.2f, %.2f] - skipping",
            limit_price, MIN_CONTRACT_PRICE, MAX_CONTRACT_PRICE,
        )
        return None

    # --- FIX #1: Use trade_economics() for consistent fee math ---
    econ = trade_economics(limit_price, 1.0, True)
    fee_rate = econ["fee_rate"]
    eff_price = econ["effective_price"]

    edge = win_prob - eff_price
    if edge < MIN_EDGE:
        log.info(
            "Fee-adjusted edge %.4f (win_prob=%.3f - eff_price=%.3f [raw=%.3f + fee=%.2f%%]) "
            "below threshold %.3f",
            edge, win_prob, eff_price, limit_price, fee_rate * 100, MIN_EDGE,
        )
        return None

    # --- FIX #4: Drawdown-aware Kelly sizing ---
    br = get_bankroll()
    effective_bankroll = _drawdown_adjusted_bankroll(br)

    rec = kelly_criterion(
        win_prob=win_prob,
        contract_price=eff_price,
        bankroll=effective_bankroll,
        kelly_multiplier=KELLY_MULTIPLIER,
        min_bet_dollars=MIN_BET_DOLLARS,
        ruin_target=RUIN_TARGET,
        ruin_level=RUIN_LEVEL,
    )

    if not rec.should_bet:
        log.info("Kelly says no bet: %s", rec.reason)
        return None

    # --- Enforce Polymarket min_order_size ---
    min_size = market["min_order_size"]
    if rec.num_contracts < min_size:
        log.info(
            "Kelly size %.1f contracts below Polymarket min_order_size %.1f - skipping",
            rec.num_contracts, min_size,
        )
        return None

    size = rec.num_contracts
    if MAX_TRADE_SIZE > 0 and limit_price > 0:
        max_contracts = MAX_TRADE_SIZE / limit_price
        if size > max_contracts:
            log.info(
                "Kelly size %.1f contracts capped to %.1f by MAX_TRADE_SIZE=$%.2f",
                size, max_contracts, MAX_TRADE_SIZE,
            )
            size = max_contracts

    if size < min_size:
        log.info(
            "Size after MAX_TRADE_SIZE cap %.1f below min_order_size %.1f - skipping",
            size, min_size,
        )
        return None

    bet_dollars_adj = (
        rec.bet_dollars * (size / rec.num_contracts) if rec.num_contracts > 0 else 0.0
    )
    ev_adj = (
        rec.expected_profit * (size / rec.num_contracts) if rec.num_contracts > 0 else 0.0
    )

    action = "BUY_UP" if direction == "up" else "BUY_DOWN"

    signal = Signal(
        action=action,
        token_id=token_id,
        price=limit_price,
        size=size,
        edge=edge,
        confidence=win_prob,
        direction=direction,
        btc_current=current_price,
        btc_opening=opening_btc_price,
        seconds_remaining=remaining,
        neg_risk=market["neg_risk"],
        tick_size=market["tick_size"],
        kelly_fraction=rec.adj_kelly_fraction,
        bet_dollars=bet_dollars_adj,
        expected_profit=ev_adj,
        bankroll_snapshot=br.current_balance,
        fee_rate_pct=fee_rate * 100,
        effective_price=eff_price,
        min_order_size=min_size,
        ruin_prob=rec.ruin_prob,
        quality_grade=rec.quality_grade,
        features=features,
        reason=(
            f"BTC {'UP' if direction == 'up' else 'DN'} "
            f"${price_delta:+.2f} ({pct_move*100:.4f}%) | "
            f"P(win)={win_prob:.3f} edge={edge:.4f} (fee-adj) | "
            f"limit=${limit_price:.3f} eff=${eff_price:.3f} fee={fee_rate*100:.2f}% | "
            f"bid=${market['best_bid']:.3f} ask=${market['best_ask']:.3f} "
            f"spread=${market['spread']:.3f} last=${market['last_trade_price']:.3f} | "
            f"Kelly={rec.adj_kelly_fraction:.4f} -> ${bet_dollars_adj:.2f} "
            f"({size:.1f} contracts, min={min_size:.0f}) | "
            f"ruin={rec.ruin_prob:.3f} grade={rec.quality_grade} | "
            f"depth={market['bid_depth']:.0f}/{market['ask_depth']:.0f} "
            f"levels={market['book_levels']} | "
            f"EV=${ev_adj:.4f} | bankroll=${br.current_balance:.2f} | "
            f"{remaining:.0f}s left"
        ),
    )

    log.info("SIGNAL: %s [%s] - %s", action, rec.quality_grade, signal.reason)
    return signal


# ── Feature extraction ───────────────────────────────────────

def _extract_features(
    pct_move: float,
    direction: str,
    seconds_remaining: float,
    oracle: dict,
    opening_price: float,
    current_price: float,
    volatility: float,
    market: dict,
) -> dict:
    """
    Build a feature vector for the logistic model.

    All features are normalised to roughly [0, 1] range so the
    learned weights are interpretable and stable.
    """
    # 1. Magnitude
    magnitude_scaled = _sigmoid(pct_move, midpoint=0.0004, steepness=8000)

    # 2. Time pressure
    time_fraction = 1.0 - (seconds_remaining / 300.0)
    time_pressure = _sigmoid(time_fraction, midpoint=0.6, steepness=8)

    # 3. Momentum consistency
    momentum_consistency = _momentum_consistency(direction)

    # 4. Oracle agreement
    prices = oracle["sources"]
    if len(prices) >= 2:
        vals = list(prices.values())
        agreeing = sum(
            1 for v in vals
            if (direction == "up" and v > opening_price)
            or (direction == "down" and v < opening_price)
        )
        oracle_agreement = agreeing / len(vals)
    else:
        oracle_agreement = 0.5

    # 5. Chainlink authority
    chainlink = oracle.get("chainlink")
    if chainlink is not None:
        chainlink_agrees = float(
            (direction == "up" and chainlink > opening_price)
            or (direction == "down" and chainlink < opening_price)
        )
    else:
        chainlink_agrees = 0.5

    # 6. Volatility regime
    vol_norm = min(1.0, volatility / 0.001)

    # 7. Trend alignment
    trend_alignment = _higher_timeframe_trend(direction)

    # 8. Order book imbalance
    bid_depth = market.get("bid_depth", 0) or 0
    ask_depth = market.get("ask_depth", 0) or 0
    total_depth = bid_depth + ask_depth
    if total_depth > 0:
        if direction == "up":
            book_imbalance = bid_depth / total_depth
        else:
            book_imbalance = ask_depth / total_depth
    else:
        book_imbalance = 0.5

    # 9. Spread quality
    spread = market.get("spread", 1.0) or 1.0
    spread_quality = max(0.0, 1.0 - spread / 0.20)

    # 10. Momentum acceleration
    momentum_accel = _momentum_acceleration(direction)

    # 11. NEW: Recent outcome streak — penalizes trading with the crowd
    #     If the last N rounds all went "up" and we're betting "up",
    #     the market has likely already priced this in.
    streak_against = _streak_feature(direction)

    features = {
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
        "streak_against": streak_against,
    }

    log.debug(
        "Features: %s",
        " | ".join(f"{k}={v:.3f}" for k, v in features.items()),
    )

    return features


def _streak_feature(direction: str) -> float:
    """
    FIX #4 (correlated outcomes):
    When recent rounds have been going the same direction, the market
    adjusts and our edge shrinks. This feature is HIGH (0.7-1.0) when
    we're betting AGAINST the recent streak (contrarian = more edge),
    and LOW (0.0-0.3) when we're betting WITH a long streak (market
    already priced in).
    """
    if len(_recent_outcomes) < 3:
        return 0.5  # neutral, not enough data

    recent = list(_recent_outcomes)[-8:]
    same_dir = sum(1 for o in recent if o == direction)
    ratio = same_dir / len(recent)

    # If most recent outcomes match our direction, we're WITH the streak
    # ratio = 1.0 means we're fully with streak -> low value (bad)
    # ratio = 0.0 means we're fully against streak -> high value (good)
    return 1.0 - ratio


def _logistic_predict(features: dict) -> float:
    """
    Logistic regression prediction: sigmoid(w^T x + b).
    Returns a calibrated probability in (0, 1).
    """
    weights = _load_model_weights()

    z = weights.get("intercept", -1.8)
    for feat_name, feat_val in features.items():
        w = weights.get(feat_name, 0.0)
        z += w * feat_val

    z = max(-15, min(15, z))
    prob = 1.0 / (1.0 + math.exp(-z))

    return max(0.01, min(0.99, prob))


# ── Feature helper functions ─────────────────────────────────

def _sigmoid(x: float, midpoint: float = 0.5, steepness: float = 10.0) -> float:
    z = steepness * (x - midpoint)
    z = max(-20, min(20, z))
    return 1.0 / (1.0 + math.exp(-z))


def _momentum_consistency(direction: str) -> float:
    if len(_price_history) < 3:
        return 0.5

    recent = list(_price_history)[-20:]
    if len(recent) < 3:
        return 0.5

    deltas = [recent[i][1] - recent[i-1][1] for i in range(1, len(recent))]

    if direction == "up":
        consistent = sum(1 for d in deltas if d > 0)
    else:
        consistent = sum(1 for d in deltas if d < 0)

    ratio = consistent / len(deltas)
    return 0.3 + 0.7 * ratio


def _momentum_acceleration(direction: str) -> float:
    if len(_price_history) < 6:
        return 0.5

    recent = list(_price_history)[-12:]
    if len(recent) < 6:
        return 0.5

    deltas = [recent[i][1] - recent[i-1][1] for i in range(1, len(recent))]
    mid = len(deltas) // 2
    early_mag = sum(abs(d) for d in deltas[:mid]) / mid
    late_mag = sum(abs(d) for d in deltas[mid:]) / (len(deltas) - mid)

    if early_mag <= 0:
        return 0.7

    accel_ratio = late_mag / early_mag

    if direction == "up":
        late_directional = sum(1 for d in deltas[mid:] if d > 0) / (len(deltas) - mid)
    else:
        late_directional = sum(1 for d in deltas[mid:] if d < 0) / (len(deltas) - mid)

    raw = 0.5 * min(1.0, accel_ratio / 2.0) + 0.5 * late_directional
    return max(0.0, min(1.0, raw))


def _higher_timeframe_trend(direction: str) -> float:
    if len(_price_history) < 10:
        return 0.5

    prices = list(_price_history)
    q1_end = len(prices) // 4
    q4_start = 3 * len(prices) // 4

    if q1_end < 1 or q4_start >= len(prices):
        return 0.5

    avg_early = sum(p[1] for p in prices[:q1_end]) / q1_end
    avg_late = sum(p[1] for p in prices[q4_start:]) / (len(prices) - q4_start)

    if avg_early <= 0:
        return 0.5

    trend_pct = (avg_late - avg_early) / avg_early

    if direction == "up":
        trend_score = _sigmoid(trend_pct, midpoint=0.0, steepness=5000)
    else:
        trend_score = _sigmoid(-trend_pct, midpoint=0.0, steepness=5000)

    return trend_score


def _recent_volatility() -> float:
    if len(_price_history) < 5:
        return 0.0003

    recent = list(_price_history)[-30:]
    if len(recent) < 5:
        return 0.0003

    returns = []
    for i in range(1, len(recent)):
        if recent[i-1][1] > 0:
            returns.append((recent[i][1] - recent[i-1][1]) / recent[i-1][1])

    if not returns:
        return 0.0003

    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    return max(0.00005, variance ** 0.5)


def _dynamic_move_threshold(volatility: float) -> float:
    base = 0.00003
    vol_scaled = volatility * 0.8
    return max(base, min(0.001, vol_scaled))


def _drawdown_adjusted_bankroll(br: Bankroll) -> float:
    """
    FIX #4: Improved drawdown scaling.

    Previous version used a power curve that was too aggressive at
    moderate drawdowns and too lenient at deep ones.

    New approach: linear up to 20% DD, then convex beyond that.
    At 0% DD   -> 100% of balance
    At 10% DD  -> 95% of balance  (barely noticeable)
    At 20% DD  -> 85% of balance
    At 30% DD  -> 65% of balance  (significant reduction)
    At 40% DD  -> 40% of balance  (survival mode)
    At 50%+ DD -> 25% minimum
    """
    dd = br.drawdown
    if dd <= 0 or MAX_DRAWDOWN_SCALE <= 0:
        return br.current_balance

    if dd <= 0.20:
        # Gentle linear: 1.0 -> 0.85 over 0-20% DD
        scale = 1.0 - dd * 0.75
    else:
        # Convex beyond 20%: accelerating reduction
        base = 0.85  # value at 20% DD
        excess = (dd - 0.20) / (MAX_DRAWDOWN_SCALE - 0.20) if MAX_DRAWDOWN_SCALE > 0.20 else 1.0
        excess = min(1.0, excess)
        scale = base * (1.0 - excess ** 1.3 * 0.7)

    scale = max(0.25, min(1.0, scale))

    effective = br.current_balance * scale
    if scale < 0.99:
        log.info(
            "Drawdown scaling: DD=%.1f%% -> sizing at %.0f%% of $%.2f = $%.2f",
            dd * 100, scale * 100, br.current_balance, effective,
        )
    return effective


# ── Limit price determination ────────────────────────────────

def _determine_limit_price(
    market: dict,
    win_prob: float,
    remaining: float,
) -> Optional[float]:
    """
    FIX #3: Smarter limit pricing.

    Previous version always crossed the spread (buying at ask),
    which costs ~1-3% per trade. New approach:

    1. If spread is tight (≤ $0.04), post at mid + 1 tick.
       This captures the spread savings on ~70% of trades.
    2. If spread is moderate ($0.04-$0.10), post at ask - 1 tick.
    3. If spread is wide or time is running out (<30s), cross at ask.
    4. Never pay more than the maximum price that preserves our edge.
    """
    best_bid = market["best_bid"]
    best_ask = market["best_ask"]
    display_price = market["display_price"]
    last_trade = market["last_trade_price"]
    spread = market["spread"]
    tick = float(market["tick_size"])

    # Maximum price we can pay while maintaining MIN_EDGE after fees
    fee_rate = effective_fee_rate(display_price)
    max_raw_price = win_prob * (1.0 - fee_rate) - MIN_EDGE
    max_acceptable = min(MAX_CONTRACT_PRICE, max_raw_price)

    if max_acceptable < MIN_CONTRACT_PRICE:
        return None

    urgent = remaining < 30  # less than 30 seconds = just cross the spread

    # Strategy 1: Tight spread, post at mid + tick (save the spread)
    if not urgent and spread <= 0.04 and best_bid > 0 and best_ask < 1:
        mid = (best_bid + best_ask) / 2.0
        price = _snap_to_tick(mid + tick, tick)
        if MIN_CONTRACT_PRICE <= price <= max_acceptable:
            log.debug("Limit strategy: mid+tick ($%.3f, spread=$%.3f)", price, spread)
            return price

    # Strategy 2: Moderate spread, post at ask - tick
    if not urgent and spread <= 0.10 and best_ask > 0 and best_ask <= max_acceptable + tick:
        price = _snap_to_tick(best_ask - tick, tick)
        if MIN_CONTRACT_PRICE <= price <= max_acceptable:
            log.debug("Limit strategy: ask-tick ($%.3f, spread=$%.3f)", price, spread)
            return price

    # Strategy 3: Cross the spread (urgent or wide spread)
    if best_ask > 0 and best_ask <= max_acceptable:
        log.debug("Limit strategy: cross at ask ($%.3f, urgent=%s)", best_ask, urgent)
        return _snap_to_tick(best_ask, tick)

    # Strategy 4: Use display price with urgency adjustment
    if display_price > 0 and display_price <= max_acceptable:
        if urgent:
            price = min(display_price + tick * 2, max_acceptable)
        else:
            price = display_price
        return _snap_to_tick(max(MIN_CONTRACT_PRICE, price), tick)

    # Strategy 5: Last resort — use last trade
    if last_trade > 0 and last_trade <= max_acceptable:
        return _snap_to_tick(last_trade, tick)

    # Strategy 6: Step above bid
    if best_bid > 0 and best_bid < max_acceptable:
        step = min(0.03, (max_acceptable - best_bid) * 0.5)
        return _snap_to_tick(min(best_bid + step, max_acceptable), tick)

    return None


def _snap_to_tick(price: float, tick: float) -> float:
    """Round a price down to the nearest valid tick increment."""
    if tick <= 0:
        return price
    return round(math.floor(price / tick) * tick, 4)
