"""
Polymarket Data API client — user positions, trades, and analytics.

Endpoint: https://data-api.polymarket.com
Authentication: public (no API key required)

Used for reconciling on-chain balances/positions with internal tracking.
Reference: https://docs.polymarket.com/api-reference/introduction
"""

import logging
from typing import Optional

from src.config import PRIVATE_KEY
from src.http_client import resilient_get

log = logging.getLogger(__name__)

DATA_API_HOST = "https://data-api.polymarket.com"


def _get_wallet_address() -> Optional[str]:
    if not PRIVATE_KEY:
        return None
    try:
        from eth_account import Account
        return Account.from_key(PRIVATE_KEY).address
    except Exception:
        return None


def get_positions(address: Optional[str] = None) -> list[dict]:
    """
    Fetch current open positions for a wallet address.
    Returns list of position dicts with token_id, size, avg_price, etc.
    """
    addr = address or _get_wallet_address()
    if not addr:
        log.warning("No wallet address available for position query")
        return []

    try:
        resp = resilient_get(
            f"{DATA_API_HOST}/positions",
            params={"address": addr},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            positions = data if isinstance(data, list) else data.get("positions", [])
            log.info("Fetched %d positions for %s", len(positions), addr[:10])
            return positions
        log.warning("Positions API returned %d", resp.status_code)
        return []
    except Exception as e:
        log.warning("Failed to fetch positions: %s", e)
        return []


def get_trades(address: Optional[str] = None, limit: int = 50) -> list[dict]:
    """Fetch recent trades for a wallet address."""
    addr = address or _get_wallet_address()
    if not addr:
        return []

    try:
        resp = resilient_get(
            f"{DATA_API_HOST}/trades",
            params={"address": addr, "limit": limit},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else data.get("trades", [])
        return []
    except Exception as e:
        log.warning("Failed to fetch trades: %s", e)
        return []


def get_position_value(address: Optional[str] = None) -> Optional[float]:
    """Fetch total position value in USDC for a wallet."""
    addr = address or _get_wallet_address()
    if not addr:
        return None

    try:
        resp = resilient_get(
            f"{DATA_API_HOST}/value",
            params={"address": addr},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get("value", 0))
        return None
    except Exception as e:
        log.warning("Failed to fetch position value: %s", e)
        return None


def reconcile_bankroll() -> dict:
    """
    Compare on-chain position data with internal tracking.
    Returns a summary dict for logging/dashboard.
    """
    addr = _get_wallet_address()
    if not addr:
        return {"error": "no wallet address"}

    positions = get_positions(addr)
    value = get_position_value(addr)

    open_positions = [p for p in positions if float(p.get("size", 0)) > 0]

    return {
        "address": addr,
        "total_positions": len(positions),
        "open_positions": len(open_positions),
        "estimated_value_usd": value,
        "positions": [
            {
                "token_id": p.get("asset", "")[:16],
                "size": float(p.get("size", 0)),
                "avg_price": float(p.get("avgPrice", 0)),
                "current_value": float(p.get("currentValue", 0)),
            }
            for p in open_positions[:20]
        ],
    }
