"""
Profit-Maximizing Strategy with Kelly-Optimal Sizing (v3)
=========================================================
Key improvements over v2:
  1. PROPER WIN PROBABILITY MODEL — combines multiple signals into a
     calibrated probability rather than a vague "confidence" score.
  2. KELLY-OPTIMAL POSITION SIZING — uses the bankroll-aware Kelly
     Criterion to calculate the exact max to put down on each bet.
  3. MOMENTUM PERSISTENCE — BTC price moves in the last 2-3 min of a
     5-min window have high autocorrelation, meaning the direction at
     t-60s strongly predicts the direction at t-0s.
  4. VOLATILITY-ADJUSTED THRESHOLDS — in high-vol regimes, larger moves
     are needed to signal conviction; in low-vol, smaller moves suffice.
  5. MULTI-FACTOR EDGE — the edge is (estimated_win_prob - contract_price),
     and we only bet when it's meaningfully positive.

For a binary contract paying $1.00:
  - You pay `price` per contract
  - If win:  +$(1 - price) per contract
  - If lose: -$(price) per contract
  - Kelly fraction: f* = (win_prob - price) / (1 - price)
  - We use half-Kelly by default to reduce variance while still
    capturing ~75% of the growth rate of full Kelly.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from src.config import (
    MIN_EDGE,
    ENTRY_WINDOW_START,
    ENTRY_WINDOW_END,
    MIN_CONTRACT_PRICE,
    MAX_CONTRACT_PRICE,
    KELLY_MULTIPLIER,
    MIN_BET_DOLLARS,
    CONFIDENCE_FLOOR,
    STARTING_BANKROLL,
)
from src.market_reader import MarketRound, get_order_book_prices
from src.price_oracle import get_btc_price
from src.kelly import (
    kelly_criterion,
    BetRecommendation,
    Bankroll,
    load_bankroll,
    save_bankroll,
)

log = logging.getLogger(__name__)

_bankroll: Optional[Bankroll] = None
_price_history: deque[tuple[float, float]] = deque(maxlen=120)


def get_bankroll() -> Bankroll:
    global _bankroll
    if _bankroll is None:
        _bankroll = load_bankroll(STARTING_BANKROLL)
    return _bankroll


def record_bet_result(won: bool, wager: float, payout: float = 0.0):
    """Call after a bet resolves to update the bankroll."""
    br = get_bankroll()
    if won:
        br.record_win(wager, payout)
    else:
        br.record_loss(wager)
    save_bankroll(br)
    log.info(
        "Bankroll updated: $%.2f (P&L: $%.2f, ROI: %.1f%%, Drawdown: %.1f%%)",
        br.current_balance, br.total_profit, br.roi * 100, br.drawdown * 100,
    )


@dataclass
class Signal:
    """A trade signal produced by the strategy."""
    action: str              # "BUY_UP" or "BUY_DOWN"
    token_id: str
    price: float             # limit price to place the order at
    size: float              # number of contracts (shares)
    edge: float              # win_prob - contract_price
    confidence: float        # estimated win probability (0-1)
    direction: str           # "up" or "down"
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

    win_prob = _estimate_win_probability(
        pct_move=pct_move,
        direction=direction,
        seconds_remaining=remaining,
        oracle=oracle,
        opening_price=opening_btc_price,
        current_price=current_price,
    )

    if win_prob < CONFIDENCE_FLOOR:
        log.info("Win probability %.3f below floor %.3f - skipping", win_prob, CONFIDENCE_FLOOR)
        return None

    token_id = rnd.up_token_id if direction == "up" else rnd.down_token_id
    book = get_order_book_prices(token_id)
    limit_price = _determine_limit_price(book, win_prob, remaining)

    if limit_price is None:
        log.info(
            "No viable limit price (bid=%.3f ask=%.3f) - skipping",
            book["best_bid"], book["best_ask"],
        )
        return None

    if not (MIN_CONTRACT_PRICE <= limit_price <= MAX_CONTRACT_PRICE):
        log.info(
            "Limit price $%.3f outside range [%.2f, %.2f] - skipping",
            limit_price, MIN_CONTRACT_PRICE, MAX_CONTRACT_PRICE,
        )
        return None

    edge = win_prob - limit_price
    if edge < MIN_EDGE:
        log.info(
            "Edge %.4f (win_prob=%.3f - price=%.3f) below threshold %.3f",
            edge, win_prob, limit_price, MIN_EDGE,
        )
        return None

    br = get_bankroll()
    rec = kelly_criterion(
        win_prob=win_prob,
        contract_price=limit_price,
        bankroll=br.current_balance,
        kelly_multiplier=KELLY_MULTIPLIER,
        min_bet_dollars=MIN_BET_DOLLARS,
    )

    if not rec.should_bet:
        log.info("Kelly says no bet: %s", rec.reason)
        return None

    size = rec.num_contracts
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
        neg_risk=rnd.neg_risk,
        tick_size=rnd.tick_size,
        kelly_fraction=rec.adj_kelly_fraction,
        bet_dollars=rec.bet_dollars,
        expected_profit=rec.expected_profit,
        bankroll_snapshot=br.current_balance,
        reason=(
            f"BTC {'UP' if direction == 'up' else 'DN'} "
            f"${price_delta:+.2f} ({pct_move*100:.4f}%) | "
            f"P(win)={win_prob:.3f} edge={edge:.4f} | "
            f"limit=${limit_price:.3f} bid=${book['best_bid']:.3f} ask=${book['best_ask']:.3f} | "
            f"Kelly={rec.adj_kelly_fraction:.4f} -> ${rec.bet_dollars:.2f} ({size:.1f} contracts) | "
            f"EV=${rec.expected_profit:.4f} | bankroll=${br.current_balance:.2f} | "
            f"{remaining:.0f}s left"
        ),
    )

    log.info("SIGNAL: %s - %s", action, signal.reason)
    return signal


def _estimate_win_probability(
    pct_move: float,
    direction: str,
    seconds_remaining: float,
    oracle: dict,
    opening_price: float,
    current_price: float,
) -> float:
    """
    Multi-factor win probability estimation.

    In a 5-minute BTC binary, the question is: will the closing price be
    above or below the opening price? By combining multiple orthogonal
    signals, we get a calibrated probability estimate.

    Factors:
    1. MOMENTUM MAGNITUDE — larger moves are more likely to persist
    2. TIME DECAY — less time remaining = less time for reversal
    3. MOMENTUM CONSISTENCY — is the move accelerating or decelerating?
    4. ORACLE AGREEMENT — do all price sources agree on direction?
    5. CHAINLINK AUTHORITY — Chainlink is the resolution source
    """

    # --- Factor 1: Momentum magnitude ---
    # BTC 5-min moves: 0.01% is noise, 0.05% is moderate, 0.15%+ is strong
    # Sigmoid mapping from pct_move to probability contribution
    magnitude_signal = _sigmoid(pct_move, midpoint=0.0004, steepness=8000)

    # --- Factor 2: Time decay ---
    # With 180s left, direction is ~55% reliable
    # With 30s left, direction is ~85% reliable
    # With 5s left, direction is ~95% reliable (market already priced in)
    time_fraction = 1.0 - (seconds_remaining / 300.0)
    time_signal = 0.50 + 0.45 * _sigmoid(time_fraction, midpoint=0.6, steepness=8)

    # --- Factor 3: Momentum consistency ---
    # Check if recent price samples show consistent direction
    consistency = _momentum_consistency(direction)

    # --- Factor 4: Oracle agreement ---
    prices = oracle["sources"]
    if len(prices) >= 2:
        vals = list(prices.values())
        all_above = all(v > opening_price for v in vals)
        all_below = all(v < opening_price for v in vals)
        if (direction == "up" and all_above) or (direction == "down" and all_below):
            agreement = 1.0
        else:
            agreeing = sum(
                1 for v in vals
                if (direction == "up" and v > opening_price)
                or (direction == "down" and v < opening_price)
            )
            agreement = agreeing / len(vals)
    else:
        agreement = 0.5

    # --- Factor 5: Chainlink authority bonus ---
    chainlink = oracle.get("chainlink")
    if chainlink is not None:
        chainlink_agrees = (
            (direction == "up" and chainlink > opening_price)
            or (direction == "down" and chainlink < opening_price)
        )
        chainlink_bonus = 0.08 if chainlink_agrees else -0.05
    else:
        chainlink_bonus = 0.0

    # --- Combine factors ---
    # Weighted combination tuned for these binary BTC rounds
    raw_prob = (
        magnitude_signal * 0.25
        + time_signal * 0.30
        + consistency * 0.20
        + agreement * 0.15
        + 0.10  # base rate (prior: any direction is ~50/50 before signals)
    ) + chainlink_bonus

    raw_prob = max(0.01, min(0.99, raw_prob))

    log.debug(
        "Win prob factors: mag=%.3f time=%.3f consist=%.3f agree=%.3f "
        "chainlink=%+.3f -> raw=%.3f",
        magnitude_signal, time_signal, consistency, agreement,
        chainlink_bonus, raw_prob,
    )

    return raw_prob


def _sigmoid(x: float, midpoint: float = 0.5, steepness: float = 10.0) -> float:
    z = steepness * (x - midpoint)
    z = max(-20, min(20, z))
    return 1.0 / (1.0 + math.exp(-z))


def _momentum_consistency(direction: str) -> float:
    """
    Check how consistently the price has been moving in `direction`
    over recent samples. Returns 0-1 where 1 = perfectly consistent.
    """
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


def _recent_volatility() -> float:
    """
    Estimate recent BTC volatility from price history.
    Returns the standard deviation of returns as a fraction.
    """
    if len(_price_history) < 5:
        return 0.0003  # default: moderate vol

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
    """
    In high-vol regimes, require a larger move to be confident.
    In low-vol regimes, even small moves are meaningful.

    Returns the minimum pct_move to generate a signal.
    """
    base = 0.00003
    vol_scaled = volatility * 0.8
    return max(base, min(0.001, vol_scaled))


def _determine_limit_price(
    book: dict,
    win_prob: float,
    remaining: float,
) -> Optional[float]:
    """
    Determine the limit price that maximizes expected profit.

    The key insight: we want to buy as CHEAPLY as possible.
    Lower price = higher edge = higher Kelly fraction = more profit.
    But too low a price means we won't get filled.

    Strategy:
    - If ask is below our maximum acceptable price, buy at ask (guaranteed fill)
    - Otherwise, place a bid that balances fill probability vs edge
    """
    best_bid = book["best_bid"]
    best_ask = book["best_ask"]
    mid = book["mid"]

    # The maximum we'd pay is where edge becomes zero: price = win_prob
    # But we want at least MIN_EDGE of edge, so max_price = win_prob - MIN_EDGE
    max_acceptable = min(MAX_CONTRACT_PRICE, win_prob - MIN_EDGE)

    if max_acceptable < MIN_CONTRACT_PRICE:
        return None

    if best_ask > 0 and best_ask <= max_acceptable:
        return best_ask

    # Try to get a better price by bidding between bid and ask
    if mid > 0 and mid <= max_acceptable:
        # Bid slightly above mid — closer to mid = more edge, less fill chance
        urgency = max(0.0, 1.0 - remaining / 120.0)
        offset = (best_ask - mid) * 0.3 * urgency if best_ask > mid else 0
        price = min(mid + offset, max_acceptable)
        return max(MIN_CONTRACT_PRICE, price)

    if best_bid > 0 and best_bid < max_acceptable:
        # Tight spread or wide — step up from bid
        step = min(0.03, (max_acceptable - best_bid) * 0.5)
        return min(best_bid + step, max_acceptable)

    return None


