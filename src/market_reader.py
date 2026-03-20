"""
Polymarket Market Reader
------------------------
Discovers active BTC 5-minute up/down markets via the Gamma API,
fetches order-book data from the CLOB, and extracts timing metadata.

Market slugs follow the pattern: btc-updown-5m-{start_unix_timestamp}
where the timestamp is the start of each 5-minute window, aligned to 300s.
Each round ends 300 seconds after its start timestamp.
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

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.end_timestamp - time.time())

    @property
    def seconds_elapsed(self) -> float:
        """Seconds since this round started. Negative if round hasn't started yet."""
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
        )

    except Exception as e:
        log.debug("Failed to fetch round %s: %s", slug, e)
        return None


def get_order_book_prices(token_id: str) -> dict:
    """
    Fetch the current best bid/ask from the CLOB for a token.
    Uses both the /book endpoint and the /midpoint endpoint for accuracy.
    Returns {"best_bid": float, "best_ask": float, "mid": float, "spread": float}
    """
    best_bid = 0.0
    best_ask = 1.0

    try:
        resp = requests.get(
            f"{CLOB_HOST}/book",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        book = resp.json()

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if bids:
            best_bid = float(bids[0]["price"])
        if asks:
            best_ask = float(asks[0]["price"])
    except Exception as e:
        log.warning("Order book fetch failed for %s: %s", token_id[:16], e)

    mid_price = 0.5
    try:
        resp = requests.get(
            f"{CLOB_HOST}/midpoint",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        mp = float(data.get("mid", 0))
        if 0.01 < mp < 0.99:
            mid_price = mp
    except Exception:
        pass

    spread = best_ask - best_bid
    if spread > 0.50:
        log.warning(
            "WIDE SPREAD for %s: bid=$%.3f ask=$%.3f spread=$%.3f — using midpoint $%.3f",
            token_id[:16], best_bid, best_ask, spread, mid_price,
        )
        best_bid = max(best_bid, mid_price - 0.02)
        best_ask = min(best_ask, mid_price + 0.02)
        spread = best_ask - best_bid

    mid = (best_bid + best_ask) / 2.0

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
    }


def get_midpoint_price(token_id: str) -> float:
    """Quick helper — just the midpoint."""
    try:
        resp = requests.get(
            f"{CLOB_HOST}/midpoint",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("mid", 0.5))
    except Exception as e:
        log.warning("Midpoint fetch failed: %s", e)
        return 0.5
