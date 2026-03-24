"""
Profit-Maximizing Strategy with Kelly-Optimal Sizing (v4)
=========================================================
Key improvements over v3:
  1. REALISTIC POLYMARKET FEES — crypto markets charge taker fees that eat
     into edge. Fee formula: fee = C * p * 0.25 * (p*(1-p))^2
     At p=0.50 the effective rate is 1.56%, dropping to near-zero at extremes.
  2. FULL MARKET DATA — uses all CLOB endpoints: /book, /midpoint, /spread,
     /last-trade-price, /fee-rate, /tick-size for accurate pricing.
  3. FEE-ADJUSTED KELLY — the Kelly fraction now accounts for the fee on entry,
     so the "effective price" (price + fee) is used for sizing.
  4. MIN ORDER SIZE enforcement from the order book.
  5. DISPLAY PRICE logic matching Polymarket (midpoint if spread <= $0.10,
     last_trade_price if wider).

For a binary contract paying $1.00 with crypto fees:
  - You pay `price` per contract
  - Fee per contract: price * effective_fee_rate(price)
  - Effective cost: price / (1 - effective_fee_rate)  [fee is taken in shares]
  - If win:  $1.00 per contract received
  - If lose: lose the effective cost
  - Kelly: f* = (win_prob - effective_price) / (1 - effective_price)

Source: https://docs.polymarket.com/trading/fees
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
    MAX_TRADE_SIZE,
    KELLY_MULTIPLIER,
    MIN_BET_DOLLARS,
    CONFIDENCE_FLOOR,
    STARTING_BANKROLL,
)
from src.market_reader import MarketRound, get_full_market_data
from src.price_oracle import get_btc_price
from src.kelly import (
    kelly_criterion,
    BetRecommendation,
    Bankroll,
    load_bankroll,
    save_bankroll,
)
from src.fees import effective_fee_rate, calculate_crypto_fee, fetch_fee_rate_bps

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
    edge: float              # win_prob - effective_price (fee-adjusted)
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
    fee_rate_pct: float = 0.0
    effective_price: float = 0.0
    min_order_size: float = 5.0


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

    # --- Fetch FULL market data from all CLOB endpoints ---
    token_id = rnd.up_token_id if direction == "up" else rnd.down_token_id
    market = get_full_market_data(token_id)

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

    # --- Calculate fee-adjusted edge ---
    # Try dynamic fee from CLOB endpoint first; fall back to hardcoded formula
    live_bps = market.get("fee_rate_bps", 0)
    if live_bps and live_bps > 0:
        fee_rate = (live_bps / 10_000.0) * (limit_price * (1 - limit_price))
    else:
        clob_bps = fetch_fee_rate_bps(token_id)
        if clob_bps is not None and clob_bps > 0:
            fee_rate = (clob_bps / 10_000.0) * (limit_price * (1 - limit_price))
        else:
            fee_rate = effective_fee_rate(limit_price)
    eff_price = limit_price / (1.0 - fee_rate) if fee_rate < 1.0 else limit_price

    edge = win_prob - eff_price
    if edge < MIN_EDGE:
        log.info(
            "Fee-adjusted edge %.4f (win_prob=%.3f - eff_price=%.3f [raw=%.3f + fee=%.2f%%]) "
            "below threshold %.3f",
            edge, win_prob, eff_price, limit_price, fee_rate * 100, MIN_EDGE,
        )
        return None

    # --- Kelly sizing with fee-adjusted effective price ---
    br = get_bankroll()
    rec = kelly_criterion(
        win_prob=win_prob,
        contract_price=eff_price,
        bankroll=br.current_balance,
        kelly_multiplier=KELLY_MULTIPLIER,
        min_bet_dollars=MIN_BET_DOLLARS,
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
        reason=(
            f"BTC {'UP' if direction == 'up' else 'DN'} "
            f"${price_delta:+.2f} ({pct_move*100:.4f}%) | "
            f"P(win)={win_prob:.3f} edge={edge:.4f} (fee-adj) | "
            f"limit=${limit_price:.3f} eff=${eff_price:.3f} fee={fee_rate*100:.2f}% | "
            f"bid=${market['best_bid']:.3f} ask=${market['best_ask']:.3f} "
            f"spread=${market['spread']:.3f} last=${market['last_trade_price']:.3f} | "
            f"Kelly={rec.adj_kelly_fraction:.4f} -> ${bet_dollars_adj:.2f} "
            f"({size:.1f} contracts, min={min_size:.0f}) | "
            f"depth={market['bid_depth']:.0f}/{market['ask_depth']:.0f} "
            f"levels={market['book_levels']} | "
            f"EV=${ev_adj:.4f} | bankroll=${br.current_balance:.2f} | "
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

    Factors:
    1. MOMENTUM MAGNITUDE -- larger moves are more likely to persist
    2. TIME DECAY -- less time remaining = less time for reversal
    3. MOMENTUM CONSISTENCY -- is the move accelerating or decelerating?
    4. ORACLE AGREEMENT -- do all price sources agree on direction?
    5. CHAINLINK AUTHORITY -- Chainlink is the resolution source
    """

    magnitude_signal = _sigmoid(pct_move, midpoint=0.0004, steepness=8000)

    time_fraction = 1.0 - (seconds_remaining / 300.0)
    time_signal = 0.50 + 0.45 * _sigmoid(time_fraction, midpoint=0.6, steepness=8)

    consistency = _momentum_consistency(direction)

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

    chainlink = oracle.get("chainlink")
    if chainlink is not None:
        chainlink_agrees = (
            (direction == "up" and chainlink > opening_price)
            or (direction == "down" and chainlink < opening_price)
        )
        chainlink_bonus = 0.08 if chainlink_agrees else -0.05
    else:
        chainlink_bonus = 0.0

    raw_prob = (
        magnitude_signal * 0.25
        + time_signal * 0.30
        + consistency * 0.20
        + agreement * 0.15
        + 0.10
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
    """
    In high-vol regimes, require a larger move to be confident.
    In low-vol regimes, even small moves are meaningful.
    """
    base = 0.00003
    vol_scaled = volatility * 0.8
    return max(base, min(0.001, vol_scaled))


def _determine_limit_price(
    market: dict,
    win_prob: float,
    remaining: float,
) -> Optional[float]:
    """
    Determine the limit price that maximizes expected profit after fees.

    Uses realistic Polymarket market data: best_bid, best_ask, display_price,
    last_trade_price, and spread to determine the best entry.

    Your order price must conform to the market's tick size.
    """
    best_bid = market["best_bid"]
    best_ask = market["best_ask"]
    display_price = market["display_price"]
    last_trade = market["last_trade_price"]
    spread = market["spread"]

    fee_rate = effective_fee_rate(display_price)
    max_raw_price = win_prob * (1.0 - fee_rate) - MIN_EDGE
    max_acceptable = min(MAX_CONTRACT_PRICE, max_raw_price)

    if max_acceptable < MIN_CONTRACT_PRICE:
        return None

    tick = float(market["tick_size"])

    if best_ask > 0 and best_ask <= max_acceptable:
        return _snap_to_tick(best_ask, tick)

    if spread <= 0.10 and display_price > 0 and display_price <= max_acceptable:
        urgency = max(0.0, 1.0 - remaining / 120.0)
        offset = (best_ask - display_price) * 0.3 * urgency if best_ask > display_price else 0
        price = min(display_price + offset, max_acceptable)
        return _snap_to_tick(max(MIN_CONTRACT_PRICE, price), tick)

    if last_trade > 0 and last_trade <= max_acceptable:
        return _snap_to_tick(last_trade, tick)

    if best_bid > 0 and best_bid < max_acceptable:
        step = min(0.03, (max_acceptable - best_bid) * 0.5)
        return _snap_to_tick(min(best_bid + step, max_acceptable), tick)

    return None


def _snap_to_tick(price: float, tick: float) -> float:
    """Round a price down to the nearest valid tick increment."""
    if tick <= 0:
        return price
    return round(math.floor(price / tick) * tick, 4)
