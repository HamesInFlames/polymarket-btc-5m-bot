"""
Oracle-Verified Directional Strategy (v2)
------------------------------------------
Key insight: By the last ~30 seconds of a 5-minute BTC round, the CLOB
already prices the likely winner at $0.95-$1.00, leaving no edge.
The profitable window is EARLIER — 2-3 minutes before close — when:
  - The direction signal is emerging but the market is still at ~$0.50-$0.75
  - We can place GTC limit orders that sit on the book
  - If the direction holds, the contract settles at $1.00

Signal logic:
  1. Capture the opening BTC price at the round start.
  2. Starting 180s before close, check current BTC price vs opening.
  3. If BTC has moved meaningfully in one direction:
     a. Calculate confidence based on move size + time decay + oracle agreement
     b. Fetch the order book for the likely-winning contract
     c. Place a GTC limit order at a price that gives us enough edge
  4. If BTC is flat or uncertain, skip the round.
  5. The order stays on the book — if it fills, expected profit is
     (1.0 - fill_price). If it doesn't fill, no loss.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.config import (
    MIN_EDGE,
    ENTRY_WINDOW_START,
    ENTRY_WINDOW_END,
    MIN_CONTRACT_PRICE,
    MAX_CONTRACT_PRICE,
    MAX_TRADE_SIZE,
)
from src.market_reader import MarketRound, get_order_book_prices
from src.price_oracle import get_btc_price

log = logging.getLogger(__name__)


@dataclass
class Signal:
    """A trade signal produced by the strategy."""
    action: str              # "BUY_UP" or "BUY_DOWN"
    token_id: str
    price: float             # limit price to place the order at
    size: float              # number of contracts (shares)
    edge: float              # estimated edge (0-1)
    confidence: float        # how confident the direction call is (0-1)
    direction: str           # "up" or "down"
    btc_current: float
    btc_opening: float
    seconds_remaining: float
    reason: str
    neg_risk: bool = False
    tick_size: str = "0.01"


def evaluate_round(
    rnd: MarketRound,
    opening_btc_price: Optional[float],
) -> Optional[Signal]:
    """
    Evaluate whether to trade a given round.
    Returns a Signal if conditions are met, otherwise None.
    """
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

    price_delta = current_price - opening_btc_price
    pct_move = abs(price_delta) / opening_btc_price

    if pct_move < 0.00005:
        log.info("Price nearly flat (%.4f%%) - too uncertain, skipping", pct_move * 100)
        return None

    direction = "up" if price_delta > 0 else "down"
    confidence = _calculate_confidence(pct_move, remaining, oracle)

    if direction == "up":
        token_id = rnd.up_token_id
    else:
        token_id = rnd.down_token_id

    book = get_order_book_prices(token_id)

    limit_price = _determine_limit_price(book, confidence, remaining)

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

    raw_edge = 1.0 - limit_price
    adjusted_edge = raw_edge * confidence

    if adjusted_edge < MIN_EDGE:
        log.info(
            "Edge %.3f (raw %.3f x conf %.3f) below threshold %.3f",
            adjusted_edge, raw_edge, confidence, MIN_EDGE,
        )
        return None

    size = _calculate_position_size(adjusted_edge, limit_price)

    action = "BUY_UP" if direction == "up" else "BUY_DOWN"

    signal = Signal(
        action=action,
        token_id=token_id,
        price=limit_price,
        size=size,
        edge=adjusted_edge,
        confidence=confidence,
        direction=direction,
        btc_current=current_price,
        btc_opening=opening_btc_price,
        seconds_remaining=remaining,
        neg_risk=rnd.neg_risk,
        tick_size=rnd.tick_size,
        reason=(
            f"BTC {'UP' if direction == 'up' else 'DN'} "
            f"${price_delta:+.2f} ({pct_move*100:.4f}%) | "
            f"limit=${limit_price:.3f} bid=${book['best_bid']:.3f} ask=${book['best_ask']:.3f} | "
            f"edge={adjusted_edge:.3f} conf={confidence:.2f} | "
            f"{remaining:.0f}s left"
        ),
    )

    log.info("SIGNAL: %s - %s", action, signal.reason)
    return signal


def _determine_limit_price(book: dict, confidence: float, remaining: float) -> Optional[float]:
    """
    Determine the limit price for our buy order.

    Strategy:
    - If the ask is reasonable (< MAX_CONTRACT_PRICE), we can buy at or near the ask.
    - If the ask is too high, we place a bid slightly above the current best bid
      and hope to get filled as the market moves.
    - The limit price is a function of our confidence and the current book.
    """
    best_bid = book["best_bid"]
    best_ask = book["best_ask"]
    mid = book["mid"]

    if best_ask <= MAX_CONTRACT_PRICE:
        return best_ask

    if mid > 0 and mid <= MAX_CONTRACT_PRICE:
        premium = min(0.05, (1.0 - confidence) * 0.10)
        return min(mid + premium, MAX_CONTRACT_PRICE)

    if best_bid > 0 and best_bid < MAX_CONTRACT_PRICE:
        step = max(0.01, (MAX_CONTRACT_PRICE - best_bid) * confidence * 0.5)
        return min(best_bid + step, MAX_CONTRACT_PRICE)

    return None


def _calculate_confidence(
    pct_move: float,
    seconds_remaining: float,
    oracle: dict,
) -> float:
    """
    Confidence score from 0 to 1 based on:
      - Magnitude of the BTC price move (bigger = more confident direction holds)
      - Time remaining (less time = direction more locked in)
      - Number of oracle sources agreeing on direction
      - Whether Chainlink specifically is available
    """
    move_conf = min(1.0, pct_move / 0.0008)

    time_conf = max(0.0, min(1.0, 1.0 - (seconds_remaining / 300.0)))

    prices = list(oracle["sources"].values())
    if len(prices) >= 2:
        spread_pct = (max(prices) - min(prices)) / max(prices)
        agreement_conf = max(0.0, 1.0 - spread_pct * 200)
    else:
        agreement_conf = 0.4

    source_conf = min(1.0, oracle["count"] / 3.0)

    chainlink_bonus = 0.10 if oracle["chainlink"] is not None else 0.0

    raw = (
        move_conf * 0.35
        + time_conf * 0.25
        + agreement_conf * 0.15
        + source_conf * 0.15
        + chainlink_bonus
    )

    return max(0.0, min(1.0, raw))


def _calculate_position_size(edge: float, price: float) -> float:
    """
    Kelly-inspired position sizing capped at MAX_TRADE_SIZE.

    Full Kelly: f* = (p * b - q) / b
      where p = win probability, b = odds, q = 1 - p

    We use quarter-Kelly for safety and convert to share count.
    """
    if price >= 1.0 or price <= 0:
        return 0.0

    odds = (1.0 / price) - 1.0
    win_prob = min(0.95, 0.5 + edge)
    kelly = (win_prob * odds - (1.0 - win_prob)) / odds if odds > 0 else 0.0
    kelly = max(0.0, kelly)

    quarter_kelly = kelly * 0.25
    dollars = MAX_TRADE_SIZE * min(1.0, quarter_kelly * 4)
    dollars = max(0.50, min(dollars, MAX_TRADE_SIZE))

    shares = dollars / price
    return round(shares, 2)
