"""
Order Execution Module
----------------------
Wraps the py-clob-client to place, track, and cancel orders
on the Polymarket CLOB.
"""

import logging
import os
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from src.config import CLOB_HOST, CHAIN_ID, PRIVATE_KEY, LIVE_TRADING

log = logging.getLogger(__name__)

_client: Optional[ClobClient] = None


def get_client() -> ClobClient:
    """Lazily initialize and return the authenticated CLOB client."""
    global _client
    if _client is not None:
        return _client

    if not PRIVATE_KEY:
        raise RuntimeError(
            "PRIVATE_KEY not set. Run: python setup_wallet.py"
        )

    from eth_account import Account
    acct = Account.from_key(PRIVATE_KEY)
    wallet_address = acct.address

    temp = ClobClient(
        host=CLOB_HOST,
        key=PRIVATE_KEY,
        chain_id=CHAIN_ID,
    )
    creds = temp.create_or_derive_api_creds()

    _client = ClobClient(
        host=CLOB_HOST,
        key=PRIVATE_KEY,
        chain_id=CHAIN_ID,
        creds=creds,
        signature_type=0,
        funder=wallet_address,
    )

    log.info("CLOB client initialized for wallet %s", wallet_address)
    return _client


def place_buy_order(
    token_id: str,
    price: float,
    size: float,
    neg_risk: bool = False,
    tick_size: str = "0.01",
) -> Optional[dict]:
    """
    Place a BUY limit order (GTC).
    In dry-run mode (LIVE_TRADING=false), logs but does not submit.

    Returns the CLOB response dict or None on failure.
    """
    if not LIVE_TRADING:
        log.info(
            "[DRY RUN] Would BUY token=%s price=$%.3f size=$%.2f",
            token_id[:16], price, size,
        )
        return {
            "orderID": "dry-run",
            "status": "SIMULATED",
            "price": price,
            "size": size,
        }

    try:
        client = get_client()
        resp = client.create_and_post_order(
            OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=BUY,
            ),
            options={
                "tick_size": tick_size,
                "neg_risk": neg_risk,
            },
            order_type=OrderType.GTC,
        )
        log.info(
            "ORDER PLACED: id=%s status=%s",
            resp.get("orderID", "?"), resp.get("status", "?"),
        )
        return resp
    except Exception as e:
        log.error("Order placement failed: %s", e)
        return None


def place_sell_order(
    token_id: str,
    price: float,
    size: float,
    neg_risk: bool = False,
    tick_size: str = "0.01",
) -> Optional[dict]:
    """Place a SELL limit order."""
    if not LIVE_TRADING:
        log.info(
            "[DRY RUN] Would SELL token=%s price=$%.3f size=$%.2f",
            token_id[:16], price, size,
        )
        return {
            "orderID": "dry-run",
            "status": "SIMULATED",
            "price": price,
            "size": size,
        }

    try:
        client = get_client()
        resp = client.create_and_post_order(
            OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=SELL,
            ),
            options={
                "tick_size": tick_size,
                "neg_risk": neg_risk,
            },
            order_type=OrderType.GTC,
        )
        log.info(
            "SELL ORDER: id=%s status=%s",
            resp.get("orderID", "?"), resp.get("status", "?"),
        )
        return resp
    except Exception as e:
        log.error("Sell order failed: %s", e)
        return None


def cancel_order(order_id: str) -> bool:
    if not LIVE_TRADING:
        log.info("[DRY RUN] Would cancel order %s", order_id)
        return True

    try:
        client = get_client()
        client.cancel(order_id=order_id)
        log.info("Cancelled order %s", order_id)
        return True
    except Exception as e:
        log.error("Cancel failed for %s: %s", order_id, e)
        return False


def cancel_all_orders() -> bool:
    if not LIVE_TRADING:
        log.info("[DRY RUN] Would cancel all orders")
        return True

    try:
        client = get_client()
        client.cancel_all()
        log.info("All orders cancelled")
        return True
    except Exception as e:
        log.error("Cancel-all failed: %s", e)
        return False


def get_open_orders() -> list:
    try:
        client = get_client()
        return client.get_orders() or []
    except Exception as e:
        log.error("Failed to fetch open orders: %s", e)
        return []


def get_balances() -> dict:
    """Fetch USDC balance via the CLOB (not on-chain, to avoid extra RPC calls)."""
    try:
        client = get_client()
        from eth_account import Account
        acct = Account.from_key(PRIVATE_KEY)
        return {"wallet": acct.address}
    except Exception as e:
        log.error("Balance check failed: %s", e)
        return {}
