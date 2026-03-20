"""
Polymarket Market Reader
------------------------
Discovers active BTC 5-minute up/down markets via the Gamma API,
fetches FULL order-book data from the CLOB, and extracts timing metadata.

Uses all available CLOB endpoints for realistic market data:
  - GET /book          → bids, asks, last_trade_price, min_order_size, tick_size
  - GET /price?side=X  → best bid/ask prices (official)
  - GET /spread        → official spread
  - GET /midpoint      → midpoint (displayed price if spread <= $0.10)
  - GET /last-trade-price → last trade price + side
  - GET /fee-rate      → taker fee rate in basis points
  - GET /tick-size     → minimum price increment

Source: https://docs.polymarket.com/trading/orderbook
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

from src.config import GAMMA_HOST, CLOB_HOST

log = logging.getLogger(__name__)

ROUND_DURATION = 300  # 5 minutes in seconds

_fee_rate_cache: dict[str, tuple[float, int]] = {}  # token_id -> (fee_bps, fetch_time)
FEE_CACHE_TTL = 300


@dataclass
class MarketRound:
    """One 5-minute BTC up/down round."""
    event_slug: str
    condition_id: str
    question: str
    up_token_id: str
    down_token_id: str
    start_timestamp: int
    end_timestamp: int
    neg_risk: bool = False
    tick_size: str = "0.01"
    up_price: float = 0.0
    down_price: float = 0.0
    description: str = ""
    fees_enabled: bool = True  # crypto markets always have fees

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.end_timestamp - time.time())

    @property
    def seconds_elapsed(self) -> float:
        return time.time() - self.start_timestamp


def discover_active_btc_5m_markets() -> list[MarketRound]:
    """
    Find the current and upcoming BTC 5-minute rounds by constructing
    their slugs from the current time. Checks the current round
    plus the next 3 upcoming rounds.
    """
    rounds: list[MarketRound] = []
    now = int(time.time())
    current_base = now - (now % ROUND_DURATION)

    for offset in range(-ROUND_DURATION, ROUND_DURATION * 4, ROUND_DURATION):
        ts = current_base + offset
        rnd = _fetch_round(ts)
        if rnd is not None and rnd.seconds_remaining > 0:
            rounds.append(rnd)

    rounds.sort(key=lambda r: r.end_timestamp)
    return rounds


def _fetch_round(start_ts: int) -> Optional[MarketRound]:
    """Fetch a single round by its start timestamp."""
    slug = f"btc-updown-5m-{start_ts}"

    try:
        resp = requests.get(
            f"{GAMMA_HOST}/events/slug/{slug}",
            timeout=8,
        )
        if resp.status_code != 200:
            return None

        event = resp.json()
        markets = event.get("markets", [])
        if not markets:
            return None

        market = markets[0]
        clob_ids = market.get("clobTokenIds")

        if isinstance(clob_ids, str):
            import json
            clob_ids = json.loads(clob_ids)

        if not clob_ids or len(clob_ids) < 2:
            return None

        outcomes = market.get("outcomes")
        if isinstance(outcomes, str):
            import json
            outcomes = json.loads(outcomes)
        if not outcomes:
            outcomes = ["Up", "Down"]

        up_idx = 0
        down_idx = 1
        for i, o in enumerate(outcomes):
            lower = str(o).lower()
            if "up" in lower:
                up_idx = i
            elif "down" in lower:
                down_idx = i

        outcome_prices = market.get("outcomePrices")
        if isinstance(outcome_prices, str):
            import json
            outcome_prices = json.loads(outcome_prices)

        up_price = 0.0
        down_price = 0.0
        if outcome_prices:
            try:
                up_price = float(outcome_prices[up_idx])
                down_price = float(outcome_prices[down_idx])
            except (ValueError, IndexError):
                pass

        end_ts = start_ts + ROUND_DURATION

        end_date = market.get("endDate", "")
        if end_date:
            try:
                dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                end_ts = int(dt.timestamp())
            except ValueError:
                pass

        tick_size = market.get("minimumTickSize") or "0.01"
        fees_enabled = market.get("feesEnabled", True)

        return MarketRound(
            event_slug=slug,
            condition_id=market.get("conditionId", ""),
            question=market.get("question", ""),
            up_token_id=clob_ids[up_idx],
            down_token_id=clob_ids[down_idx],
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            neg_risk=market.get("negRisk", False),
            tick_size=tick_size,
            up_price=up_price,
            down_price=down_price,
            description=market.get("description", ""),
            fees_enabled=fees_enabled,
        )

    except Exception as e:
        log.debug("Failed to fetch round %s: %s", slug, e)
        return None


def get_full_market_data(token_id: str) -> dict:
    """
    Fetch comprehensive market data from ALL relevant CLOB endpoints.

    Returns:
        {
            "best_bid": float,       # from /book or /price?side=SELL
            "best_ask": float,       # from /book or /price?side=BUY
            "mid": float,            # from /midpoint (or computed)
            "spread": float,         # from /spread (or computed)
            "last_trade_price": float,  # from /book or /last-trade-price
            "last_trade_side": str,  # "BUY" or "SELL"
            "min_order_size": float, # from /book
            "tick_size": str,        # from /book
            "neg_risk": bool,        # from /book
            "fee_rate_bps": int,     # from /fee-rate
            "bid_depth": float,      # total bid liquidity in contracts
            "ask_depth": float,      # total ask liquidity in contracts
            "book_levels": int,      # total number of price levels
            "display_price": float,  # what Polymarket shows (mid or last_trade)
        }
    """
    result = {
        "best_bid": 0.0,
        "best_ask": 1.0,
        "mid": 0.5,
        "spread": 1.0,
        "last_trade_price": 0.5,
        "last_trade_side": "",
        "min_order_size": 5.0,
        "tick_size": "0.01",
        "neg_risk": False,
        "fee_rate_bps": 0,
        "bid_depth": 0.0,
        "ask_depth": 0.0,
        "book_levels": 0,
        "display_price": 0.5,
    }

    # --- 1. Full order book (primary data source) ---
    book = _fetch_order_book(token_id)
    if book:
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if bids:
            result["best_bid"] = float(bids[0]["price"])
            result["bid_depth"] = sum(float(b["size"]) for b in bids)
        if asks:
            result["best_ask"] = float(asks[0]["price"])
            result["ask_depth"] = sum(float(a["size"]) for a in asks)

        result["book_levels"] = len(bids) + len(asks)
        result["min_order_size"] = float(book.get("min_order_size", 5))
        result["tick_size"] = book.get("tick_size", "0.01")
        result["neg_risk"] = book.get("neg_risk", False)

        ltp = book.get("last_trade_price", "0.5")
        result["last_trade_price"] = float(ltp) if ltp else 0.5

    # --- 2. Official midpoint (what Polymarket displays as the "price") ---
    midpoint = _fetch_midpoint(token_id)
    if midpoint and 0.001 < midpoint < 0.999:
        result["mid"] = midpoint
    elif result["best_bid"] > 0 and result["best_ask"] < 1:
        result["mid"] = (result["best_bid"] + result["best_ask"]) / 2.0

    # --- 3. Official spread ---
    spread = _fetch_spread(token_id)
    if spread is not None:
        result["spread"] = spread
    else:
        result["spread"] = result["best_ask"] - result["best_bid"]

    # --- 4. Last trade price + side (separate endpoint, more detail) ---
    ltp_data = _fetch_last_trade_price(token_id)
    if ltp_data:
        result["last_trade_price"] = ltp_data["price"]
        result["last_trade_side"] = ltp_data["side"]

    # --- 5. Fee rate for this token ---
    result["fee_rate_bps"] = _fetch_fee_rate(token_id)

    # --- 6. Display price (what Polymarket shows users) ---
    # Per docs: if spread > $0.10, display last_trade_price instead of midpoint
    if result["spread"] > 0.10:
        result["display_price"] = result["last_trade_price"]
        log.info(
            "Wide spread $%.3f > $0.10 for %s — using last trade $%.3f as display price",
            result["spread"], token_id[:16], result["display_price"],
        )
    else:
        result["display_price"] = result["mid"]

    log.debug(
        "Market data for %s: bid=$%.3f ask=$%.3f mid=$%.3f spread=$%.3f "
        "last=$%.3f fee=%dbps depth=%.0f/%.0f min_size=%.1f",
        token_id[:16],
        result["best_bid"], result["best_ask"], result["mid"],
        result["spread"], result["last_trade_price"],
        result["fee_rate_bps"],
        result["bid_depth"], result["ask_depth"],
        result["min_order_size"],
    )

    return result


def get_order_book_prices(token_id: str) -> dict:
    """
    Backward-compatible wrapper that returns the same dict shape
    as before, but now powered by get_full_market_data().
    """
    data = get_full_market_data(token_id)
    return {
        "best_bid": data["best_bid"],
        "best_ask": data["best_ask"],
        "mid": data["mid"],
        "spread": data["spread"],
    }


# ── Individual CLOB endpoint fetchers ─────────────────────────

def _fetch_order_book(token_id: str) -> Optional[dict]:
    """GET /book?token_id=X"""
    try:
        resp = requests.get(
            f"{CLOB_HOST}/book",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("Order book fetch failed for %s: %s", token_id[:16], e)
        return None


def _fetch_midpoint(token_id: str) -> Optional[float]:
    """GET /midpoint?token_id=X — returns the average of best bid and best ask."""
    try:
        resp = requests.get(
            f"{CLOB_HOST}/midpoint",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        mid = float(data.get("mid", 0))
        return mid if mid > 0 else None
    except Exception as e:
        log.debug("Midpoint fetch failed for %s: %s", token_id[:16], e)
        return None


def _fetch_spread(token_id: str) -> Optional[float]:
    """GET /spread?token_id=X — returns the difference between best ask and best bid."""
    try:
        resp = requests.get(
            f"{CLOB_HOST}/spread",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("spread", 0))
    except Exception as e:
        log.debug("Spread fetch failed for %s: %s", token_id[:16], e)
        return None


def _fetch_last_trade_price(token_id: str) -> Optional[dict]:
    """GET /last-trade-price?token_id=X — returns {price, side}."""
    try:
        resp = requests.get(
            f"{CLOB_HOST}/last-trade-price",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        price = float(data.get("price", 0.5))
        side = data.get("side", "")
        return {"price": price, "side": side}
    except Exception as e:
        log.debug("Last trade price fetch failed for %s: %s", token_id[:16], e)
        return None


def _fetch_market_price(token_id: str, side: str) -> Optional[float]:
    """GET /price?token_id=X&side=BUY|SELL — returns the best price for that side."""
    try:
        resp = requests.get(
            f"{CLOB_HOST}/price",
            params={"token_id": token_id, "side": side},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("price", 0))
    except Exception as e:
        log.debug("Price fetch (%s) failed for %s: %s", side, token_id[:16], e)
        return None


def _fetch_fee_rate(token_id: str) -> int:
    """
    GET /fee-rate?token_id=X — returns fee rate in basis points.
    Cached for FEE_CACHE_TTL seconds since fee rates rarely change.
    """
    now = int(time.time())
    cached = _fee_rate_cache.get(token_id)
    if cached and (now - cached[1]) < FEE_CACHE_TTL:
        return cached[0]

    try:
        resp = requests.get(
            f"{CLOB_HOST}/fee-rate",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        bps = int(data.get("fee_rate_bps", 0))
        _fee_rate_cache[token_id] = (bps, now)
        log.debug("Fee rate for %s: %d bps", token_id[:16], bps)
        return bps
    except Exception as e:
        log.debug("Fee rate fetch failed for %s: %s", token_id[:16], e)
        return 0


def _fetch_tick_size(token_id: str) -> Optional[str]:
    """GET /tick-size?token_id=X — returns minimum price increment."""
    try:
        resp = requests.get(
            f"{CLOB_HOST}/tick-size",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        ts = data.get("minimum_tick_size")
        return str(ts) if ts else None
    except Exception as e:
        log.debug("Tick size fetch failed for %s: %s", token_id[:16], e)
        return None


def get_midpoint_price(token_id: str) -> float:
    """Quick helper — just the midpoint."""
    mid = _fetch_midpoint(token_id)
    return mid if mid else 0.5
