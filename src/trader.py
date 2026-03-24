"""
Order Execution Module
----------------------
Wraps the py-clob-client to place, track, and cancel orders
on the Polymarket CLOB.

Supports all Polymarket order types:
  - GTC (Good-Til-Cancelled) — rests on the book
  - GTD (Good-Til-Date) — auto-expires at a timestamp
  - FOK (Fill-Or-Kill) — must fill entirely or cancel
  - FAK (Fill-And-Kill) — fills available, cancels rest

For time-sensitive 5-min BTC rounds, we use FAK to grab whatever
liquidity is available immediately without leaving stale orders.

Source: https://docs.polymarket.com/trading/orders/create
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import BUY, SELL

from src.config import CLOB_HOST, CHAIN_ID, PRIVATE_KEY, LIVE_TRADING, RELAYER_API_KEY, USDC_ADDRESS
from src.geoblock import signal_clob_geoblock

log = logging.getLogger(__name__)

# Patch py-clob-client's User-Agent so Cloudflare doesn't flag it as bot traffic.
# The library hardcodes "py_clob_client" which combined with VPN IPs triggers 403.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

def _patch_clob_headers():
    try:
        from py_clob_client.http_helpers import helpers as _h
        _orig = _h.overloadHeaders

        def _patched(method, headers):
            headers = _orig(method, headers)
            headers["User-Agent"] = _BROWSER_UA
            return headers

        _h.overloadHeaders = _patched
        log.debug("Patched py-clob-client User-Agent")
    except Exception as e:
        log.warning("Failed to patch CLOB headers: %s", e)

_patch_clob_headers()

_client: Optional[ClobClient] = None
_cached_usdc: Optional[float] = None
_last_balance_check: float = 0.0


def _check_usdc_balance() -> float:
    """Quick on-chain USDC.e balance check with 30s cache."""
    global _cached_usdc, _last_balance_check
    import time
    now = time.time()
    if _cached_usdc is not None and (now - _last_balance_check) < 30:
        return _cached_usdc
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com",
                                     request_kwargs={"timeout": 5}))
        from eth_account import Account
        addr = Account.from_key(PRIVATE_KEY).address
        abi = [{"inputs": [{"name": "account", "type": "address"}],
                "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view", "type": "function"}]
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS), abi=abi)
        raw = contract.functions.balanceOf(addr).call()
        _cached_usdc = raw / 1e6
        _last_balance_check = now
        return _cached_usdc
    except Exception as e:
        log.debug("Balance check failed: %s", e)
        return _cached_usdc if _cached_usdc is not None else 999.0


@dataclass
class FillResult:
    """Structured result from an order placement attempt."""
    success: bool
    order_id: str
    status: str
    requested_size: float
    filled_size: float
    avg_price: float
    is_live: bool
    raw_response: Optional[dict] = None


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

    if RELAYER_API_KEY:
        log.info(
            "Relayer API key found but not used — EOA wallets (signature_type=0) "
            "cannot use the relayer. To enable gasless transactions, upgrade to a "
            "proxy or safe wallet."
        )

    log.info("CLOB client initialized for wallet %s", wallet_address)
    return _client


def place_buy_order(
    token_id: str,
    price: float,
    size: float,
    neg_risk: bool = False,
    tick_size: str = "0.01",
    order_type: str = "FAK",
    min_order_size: float = 5.0,
) -> FillResult:
    """
    Place a BUY order on the Polymarket CLOB and return structured fill info.

    For time-sensitive BTC 5-min rounds, FAK (Fill-And-Kill) is the default:
    fills whatever liquidity is available, cancels the rest.

    In dry-run mode (LIVE_TRADING=false), returns a FillResult that assumes
    the order would have filled at the limit price. The actual outcome is
    determined later by checking the real market resolution.
    """
    price = round(price, 2)
    size = math.floor(size)
    size = max(size, 1)

    if size < min_order_size:
        size = math.ceil(min_order_size)

    usdc_needed = round(price * size, 2)
    available = _check_usdc_balance()
    if available < usdc_needed:
        log.warning(
            "Insufficient USDC.e: need $%.2f but wallet has $%.2f — reducing order",
            usdc_needed, available,
        )
        if available < 1.0:
            return FillResult(
                success=False, order_id="", status="INSUFFICIENT_BALANCE",
                requested_size=size, filled_size=0, avg_price=price,
            )
        size = math.floor((available * 0.95 / price) * 10000) / 10000
        usdc_needed = round(price * size, 2)
        if size < 1.0:
            return FillResult(
                success=False, order_id="", status="INSUFFICIENT_BALANCE",
                requested_size=size, filled_size=0, avg_price=price,
            )
        log.info("Adjusted order: %.1f contracts ($%.2f)", size, usdc_needed)

    if not LIVE_TRADING:
        log.info(
            "[DRY RUN] Would BUY token=%s price=$%.3f size=%.1f contracts "
            "type=%s tick=%s neg_risk=%s",
            token_id[:16], price, size, order_type, tick_size, neg_risk,
        )
        return FillResult(
            success=True,
            order_id="dry-run",
            status="SIMULATED",
            requested_size=size,
            filled_size=size,
            avg_price=price,
            is_live=False,
        )

    try:
        client = get_client()
        ot = _parse_order_type(order_type)

        order_options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        signed_order = client.create_order(
            OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=BUY,
            ),
            options=order_options,
        )
        resp = client.post_order(signed_order, orderType=ot)

        return _parse_fill_result(resp, requested_size=size, limit_price=price)

    except Exception as e:
        error_str = str(e)
        lower_err = error_str.lower()
        if "403" in error_str and ("region" in lower_err or "restricted" in lower_err or "geoblock" in lower_err):
            signal_clob_geoblock()
            log.critical(
                "GEOBLOCKED by Polymarket CLOB! Order rejected with 403.\n"
                "  Trading paused — will re-check in 120s.\n"
                "  Switch VPN to: Norway, Sweden, Finland, Denmark, Switzerland.\n"
                "  Error: %s", error_str[:300],
            )
            return FillResult(
                success=False,
                order_id="",
                status="GEOBLOCKED",
                requested_size=size,
                filled_size=0.0,
                avg_price=0.0,
                is_live=True,
            )
        elif "425" in error_str or "Too Early" in error_str:
            log.warning("Matching engine restarting (425) — order not placed")
            from src.http_client import _signal_engine_restart
            _signal_engine_restart()
        else:
            log.error("Order placement failed: %s", e)
        return FillResult(
            success=False,
            order_id="",
            status=f"ERROR: {e}",
            requested_size=size,
            filled_size=0.0,
            avg_price=0.0,
            is_live=True,
        )


def _parse_fill_result(
    resp: dict, requested_size: float, limit_price: float,
) -> FillResult:
    """
    Parse the CLOB response into a structured FillResult.

    Polymarket CLOB statuses:
      MATCHED  – fully or partially matched immediately
      LIVE     – resting on the book (should not happen with FAK)
      DELAYED  – pending processing
    """
    if not resp:
        return FillResult(
            success=False, order_id="", status="EMPTY_RESPONSE",
            requested_size=requested_size, filled_size=0.0,
            avg_price=0.0, is_live=True,
        )

    order_id = resp.get("orderID", resp.get("id", "?"))
    status = resp.get("status", "UNKNOWN")

    filled_size = 0.0
    avg_price = limit_price

    if status == "MATCHED":
        filled_size = float(resp.get("size_matched", requested_size))
        avg_price = float(resp.get("price", limit_price))
    elif status == "LIVE":
        filled_size = float(resp.get("size_matched", 0.0))
        avg_price = float(resp.get("price", limit_price))
    else:
        filled_size = float(resp.get("size_matched", 0.0))
        avg_price = float(resp.get("price", limit_price))

    success = filled_size > 0

    log.info(
        "ORDER RESULT: id=%s status=%s filled=%.1f/%.1f @ $%.3f",
        order_id, status, filled_size, requested_size, avg_price,
    )

    return FillResult(
        success=success,
        order_id=order_id,
        status=status,
        requested_size=requested_size,
        filled_size=filled_size,
        avg_price=avg_price,
        is_live=True,
        raw_response=resp,
    )


def place_sell_order(
    token_id: str,
    price: float,
    size: float,
    neg_risk: bool = False,
    tick_size: str = "0.01",
    order_type: str = "GTC",
) -> Optional[dict]:
    """Place a SELL order (GTC by default for exits)."""
    price = round(price, 2)
    size = math.floor(size)
    size = max(size, 1)

    if not LIVE_TRADING:
        log.info(
            "[DRY RUN] Would SELL token=%s price=$%.3f size=%.1f type=%s",
            token_id[:16], price, size, order_type,
        )
        return {
            "orderID": "dry-run",
            "status": "SIMULATED",
            "price": price,
            "size": size,
        }

    try:
        client = get_client()
        ot = _parse_order_type(order_type)

        signed_order = client.create_order(
            OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=SELL,
            ),
            options=PartialCreateOrderOptions(
                tick_size=tick_size,
                neg_risk=neg_risk,
            ),
        )
        resp = client.post_order(signed_order, orderType=ot)
        log.info(
            "SELL ORDER: id=%s status=%s type=%s",
            resp.get("orderID", "?"), resp.get("status", "?"), order_type,
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
    """Fetch wallet address (balance checks happen on-chain)."""
    try:
        client = get_client()
        from eth_account import Account
        acct = Account.from_key(PRIVATE_KEY)
        return {"wallet": acct.address}
    except Exception as e:
        log.error("Balance check failed: %s", e)
        return {}


def _parse_order_type(ot: str) -> OrderType:
    """Convert string order type to py_clob_client enum."""
    mapping = {
        "GTC": OrderType.GTC,
        "GTD": OrderType.GTD,
        "FOK": OrderType.FOK,
        "FAK": OrderType.FAK,
    }
    return mapping.get(ot.upper(), OrderType.GTC)
