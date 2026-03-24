"""
Geoblock checker — verifies the bot's outbound IP is not blocked by BOTH:
  1. Polymarket website (/api/geoblock)
  2. CLOB trading API (attempts API key derivation — returns 403 if blocked)

The CLOB has stricter geoblocking than the website. The website may allow
certain regions (e.g., Quebec, Canada) while the CLOB blocks all of Canada
at the trading level.

Reference: https://docs.polymarket.com (geographic restrictions)
"""

import logging
from typing import Optional

import requests

log = logging.getLogger(__name__)

GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
CLOB_HOST = "https://clob.polymarket.com"
IP_INFO_URL = "https://ipinfo.io/json"


def check_geoblock(timeout: int = 10) -> dict:
    """
    Query both Polymarket's website geoblock AND the CLOB trading endpoint.

    Returns a dict with:
        - "allowed": bool   (True = trading permitted from this IP)
        - "ip": str         (detected public IP)
        - "region": str     (detected region/country)
        - "website_allowed": bool  (website geoblock result)
        - "clob_allowed": bool     (CLOB trading geoblock result)
        - "raw": dict              (full API response for debugging)
    """
    result = {
        "allowed": True,
        "ip": "unknown",
        "region": "unknown",
        "website_allowed": True,
        "clob_allowed": True,
        "raw": {},
    }

    ip_info = _get_ip_info(timeout)
    if ip_info:
        result["ip"] = ip_info.get("ip", "unknown")
        city = ip_info.get("city", "")
        region = ip_info.get("region", "")
        country = ip_info.get("country", "")
        result["region"] = f"{city}, {region}, {country}".strip(", ")

    # Check 1: Website geoblock
    try:
        resp = requests.get(GEOBLOCK_URL, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            result["raw"]["website"] = data
            blocked = data.get("blocked", False)
            if isinstance(blocked, str):
                blocked = blocked.lower() in ("true", "1", "yes")
            result["website_allowed"] = not blocked
        elif resp.status_code == 403:
            result["website_allowed"] = False
            result["raw"]["website"] = {"status": 403}
    except requests.exceptions.RequestException as e:
        log.warning("Website geoblock check failed: %s", e)
        result["raw"]["website"] = {"error": str(e)}

    # Check 2: CLOB trading endpoint — try to derive API creds
    # If the CLOB blocks our region, it returns 403 on auth endpoints
    clob_blocked = _check_clob_geoblock(timeout)
    result["clob_allowed"] = not clob_blocked
    result["raw"]["clob_blocked"] = clob_blocked

    # The bot only needs the CLOB to be unblocked — the website geoblock
    # is for the polymarket.com frontend which the bot doesn't use.
    result["allowed"] = result["clob_allowed"]
    return result


def _check_clob_geoblock(timeout: int = 10) -> bool:
    """
    Test if the CLOB trading API blocks our IP on authenticated endpoints.
    Uses real API credentials (derived from the private key) to send
    an authenticated request — the geoblock only fires on authenticated
    trading requests.
    Returns True if blocked, False if allowed.
    """
    try:
        from src.config import PRIVATE_KEY, CHAIN_ID
        if not PRIVATE_KEY:
            log.debug("No PRIVATE_KEY — skipping CLOB geoblock probe")
            return False

        from py_clob_client.client import ClobClient

        client = ClobClient(
            host=CLOB_HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
        )
        creds = client.create_or_derive_api_creds()

        from eth_account import Account
        wallet = Account.from_key(PRIVATE_KEY).address

        auth_client = ClobClient(
            host=CLOB_HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            creds=creds,
            signature_type=0,
            funder=wallet,
        )

        # Fetch open orders — this is an authenticated GET that triggers
        # the geoblock on the trading API path
        try:
            auth_client.get_orders()
            return False  # If it succeeds, we're not blocked
        except Exception as e:
            error_str = str(e)
            if "403" in error_str and ("region" in error_str.lower() or "restricted" in error_str.lower() or "geoblock" in error_str.lower()):
                log.warning("CLOB trading API geoblock detected: %s", error_str[:200])
                return True
            # Other errors (network, auth) don't mean geoblock
            log.debug("CLOB probe got non-geoblock error: %s", error_str[:100])
            return False

    except Exception as e:
        log.debug("CLOB geoblock probe failed: %s", e)
        return False


def _get_ip_info(timeout: int = 5) -> Optional[dict]:
    """Fetch public IP and geo info from ipinfo.io (free, no key needed)."""
    try:
        resp = requests.get(IP_INFO_URL, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def assert_not_geoblocked() -> dict:
    """
    Run both geoblock checks and raise RuntimeError if blocked.
    Returns the check result on success.
    """
    result = check_geoblock()

    if result["allowed"]:
        website_status = "OK" if result["website_allowed"] else "BLOCKED (irrelevant for bot)"
        log.info(
            "Geoblock check PASSED — IP: %s | Region: %s | Website: %s | CLOB: %s",
            result["ip"], result["region"], website_status,
            "OK" if result["clob_allowed"] else "BLOCKED",
        )
        if not result["website_allowed"]:
            log.info(
                "Website geoblock shows BLOCKED but this only affects polymarket.com "
                "frontend — the bot uses CLOB/Gamma APIs which are separate."
            )
        return result

    msg = (
        f"GEOBLOCKED by Polymarket CLOB!\n"
        f"  Detected IP:     {result['ip']}\n"
        f"  Detected region: {result['region']}\n"
        f"  Website:         {'ALLOWED' if result['website_allowed'] else 'BLOCKED'}\n"
        f"  CLOB trading:    {'ALLOWED' if result['clob_allowed'] else 'BLOCKED'}\n"
        f"\n"
        f"  The CLOB trading API blocks order placement from this IP.\n"
        f"  NOTE: The CLOB geoblock can only be fully detected when placing\n"
        f"  an actual order — if the probe above shows OK but orders fail\n"
        f"  with 403, the CLOB is blocking your region.\n"
        f"\n"
        f"  To fix: connect VPN to a non-restricted country.\n"
        f"  Try: Portugal, Poland, Singapore, Japan, South Korea, Brazil.\n"
        f"\n"
        f"  Known BLOCKED: Canada, US, UK, Australia, France, Germany (website).\n"
        f"\n"
        f"  Raw: {result['raw']}"
    )
    raise RuntimeError(msg)
