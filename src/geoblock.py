"""
Geoblock checker — verifies the bot's outbound IP against Polymarket's
geoblock API AND probes the actual CLOB trading endpoint.

IMPORTANT: The website geoblock API (polymarket.com/api/geoblock) and
the CLOB trading server (clob.polymarket.com) enforce geoblocking
INDEPENDENTLY. An IP can pass the website check but still get 403'd
by the CLOB. This module checks BOTH.

Sources:
  - https://docs.polymarket.com/developers/CLOB/geoblock
  - https://github.com/Polymarket/py-clob-client/issues/143
"""

import logging
import time
import threading
from typing import Optional

import requests

log = logging.getLogger(__name__)

GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
CLOB_HOST = "https://clob.polymarket.com"
IP_INFO_URL = "https://ipinfo.io/json"

BLOCKED_COUNTRIES = {
    "AU", "BE", "BY", "BI", "CF", "CD", "CU", "DE", "ET", "FR",
    "GB", "IR", "IQ", "IT", "KP", "LB", "LY", "MM", "NI", "NL",
    "RU", "SO", "SS", "SD", "SY", "UM", "US", "VE", "YE", "ZW",
}

CLOSE_ONLY_COUNTRIES = {"PL", "SG", "TH", "TW"}

BLOCKED_REGIONS = {
    "CA": {"ON"},
    "UA": {"43", "14", "09"},
}

# Countries confirmed working for CLOB trading by real users
# (GitHub issues #101, #111, #143). Nordic countries are most reliable.
SAFE_COUNTRIES = [
    "Norway (NO)", "Sweden (SE)", "Finland (FI)",
    "Denmark (DK)", "Switzerland (CH)", "Portugal (PT)",
    "Spain (ES)", "Austria (AT)", "Czech Republic (CZ)",
    "Romania (RO)", "Hungary (HU)", "Ireland (IE)",
    "Japan (JP)", "South Korea (KR)", "Mexico (MX)",
    "Argentina (AR)", "Colombia (CO)",
]


# ── Circuit breaker: shared state for CLOB-level geoblock ────

_geoblock_lock = threading.Lock()
_clob_blocked: bool = False
_clob_blocked_since: float = 0.0
_clob_block_count: int = 0
CLOB_GEOBLOCK_RECHECK_SECONDS = 120


def signal_clob_geoblock():
    """Called by trader.py when an order gets a 403 geoblock from the CLOB."""
    global _clob_blocked, _clob_blocked_since, _clob_block_count
    with _geoblock_lock:
        if not _clob_blocked:
            log.critical(
                "CLOB GEOBLOCKED — pausing all order placement. "
                "Will re-check every %ds. Switch VPN to: %s",
                CLOB_GEOBLOCK_RECHECK_SECONDS,
                ", ".join(SAFE_COUNTRIES[:5]),
            )
        _clob_blocked = True
        _clob_blocked_since = time.time()
        _clob_block_count += 1


def clear_clob_geoblock():
    """Called when a CLOB probe succeeds after a previous block."""
    global _clob_blocked, _clob_block_count
    with _geoblock_lock:
        if _clob_blocked:
            log.info("CLOB geoblock cleared — resuming trading")
        _clob_blocked = False
        _clob_block_count = 0


def is_clob_geoblocked() -> bool:
    """Check if the CLOB circuit breaker is active."""
    with _geoblock_lock:
        if not _clob_blocked:
            return False
        elapsed = time.time() - _clob_blocked_since
        if elapsed >= CLOB_GEOBLOCK_RECHECK_SECONDS:
            return False  # allow a re-probe
        return True


def clob_geoblock_status() -> dict:
    with _geoblock_lock:
        return {
            "blocked": _clob_blocked,
            "since": _clob_blocked_since,
            "count": _clob_block_count,
            "seconds_until_recheck": max(
                0, CLOB_GEOBLOCK_RECHECK_SECONDS - (time.time() - _clob_blocked_since)
            ) if _clob_blocked else 0,
        }


def probe_clob_trading(timeout: int = 10) -> bool:
    """
    Probe the CLOB server to detect trading-level geoblock.
    Uses a lightweight endpoint that still goes through
    Cloudflare / geoblock enforcement.
    Returns True if CLOB is reachable (not geoblocked).
    """
    try:
        resp = requests.get(f"{CLOB_HOST}/time", timeout=timeout)
        if resp.status_code == 403:
            body = ""
            try:
                body = resp.text[:500]
            except Exception:
                pass
            if "restricted" in body.lower() or "geoblock" in body.lower() or "blocked" in body.lower():
                return False
            return False
        return resp.status_code == 200
    except requests.exceptions.RequestException as e:
        log.warning("CLOB probe failed (network error): %s", e)
        return True  # don't block on network errors, let real orders determine


def check_geoblock(timeout: int = 10) -> dict:
    """
    Two-layer geoblock check:
      1. Polymarket website API (polymarket.com/api/geoblock)
      2. Direct CLOB server probe (clob.polymarket.com)

    Both must pass for trading to be allowed.
    """
    result = {
        "allowed": True,
        "ip": "unknown",
        "region": "unknown",
        "country_code": "",
        "region_code": "",
        "block_reason": "",
        "clob_reachable": True,
        "raw": {},
    }

    ip_info = _get_ip_info(timeout)
    if ip_info:
        result["ip"] = ip_info.get("ip", "unknown")
        city = ip_info.get("city", "")
        region = ip_info.get("region", "")
        country = ip_info.get("country", "")
        result["region"] = f"{city}, {region}, {country}".strip(", ")
        result["country_code"] = country
        result["region_code"] = ip_info.get("region", "")

    # Layer 1: Website geoblock API
    try:
        resp = requests.get(GEOBLOCK_URL, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            result["raw"] = data
            if data.get("blocked", False):
                result["allowed"] = False
                cc = data.get("country", "")
                rg = data.get("region", "")
                result["country_code"] = cc
                result["region_code"] = rg
                result["block_reason"] = _explain_block(cc, rg)
                return result
        elif resp.status_code == 403:
            result["allowed"] = False
            result["block_reason"] = "Geoblock API returned 403 Forbidden"
            return result
    except requests.exceptions.RequestException as e:
        log.warning("Geoblock API request failed: %s — checking local list", e)

    # Layer 1b: Cross-reference with known blocked list
    cc = result["country_code"]
    rc = result["region_code"]

    if cc in BLOCKED_COUNTRIES:
        result["allowed"] = False
        result["block_reason"] = _explain_block(cc, rc)
        return result
    elif cc in CLOSE_ONLY_COUNTRIES:
        result["allowed"] = False
        result["block_reason"] = f"{cc} is close-only — cannot open new positions"
        return result
    elif cc in BLOCKED_REGIONS:
        if rc in BLOCKED_REGIONS[cc]:
            result["allowed"] = False
            result["block_reason"] = f"{cc}/{rc} is a blocked region"
            return result

    # Layer 2: Probe the actual CLOB trading server
    clob_ok = probe_clob_trading(timeout)
    result["clob_reachable"] = clob_ok
    if not clob_ok:
        result["allowed"] = False
        result["block_reason"] = (
            f"Website geoblock passed ({cc}) but CLOB server returned 403. "
            f"The CLOB enforces stricter geoblocking than the website API. "
            f"This is a known issue (github.com/Polymarket/py-clob-client/issues/143)."
        )
        signal_clob_geoblock()

    return result


def _explain_block(country_code: str, region_code: str) -> str:
    if country_code in BLOCKED_COUNTRIES:
        return f"{country_code} is on Polymarket's blocked country list"
    if country_code in CLOSE_ONLY_COUNTRIES:
        return f"{country_code} is close-only (cannot open new positions)"
    if country_code in BLOCKED_REGIONS:
        if region_code in BLOCKED_REGIONS[country_code]:
            return f"{country_code}/{region_code} is a blocked region"
    return f"Blocked by Polymarket geoblock API ({country_code}/{region_code})"


def _get_ip_info(timeout: int = 5) -> Optional[dict]:
    try:
        resp = requests.get(IP_INFO_URL, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def assert_not_geoblocked() -> dict:
    """
    Run the full two-layer geoblock check and raise RuntimeError if blocked.
    Returns the check result on success.
    """
    result = check_geoblock()

    if result["allowed"]:
        clob_note = "CLOB OK" if result["clob_reachable"] else "CLOB untested"
        log.info(
            "Geoblock check PASSED — IP: %s | Region: %s | Country: %s | %s",
            result["ip"], result["region"], result["country_code"], clob_note,
        )
        return result

    safe_list = ", ".join(SAFE_COUNTRIES[:6])
    msg = (
        f"GEOBLOCKED by Polymarket!\n"
        f"  Detected IP:     {result['ip']}\n"
        f"  Detected region: {result['region']}\n"
        f"  Country code:    {result['country_code']}\n"
        f"  CLOB reachable:  {result['clob_reachable']}\n"
        f"  Reason:          {result['block_reason']}\n"
        f"\n"
        f"  Switch VPN to a CONFIRMED working country:\n"
        f"    {safe_list}\n"
        f"    (Nordic countries are most reliable for CLOB access)\n"
        f"\n"
        f"  BLOCKED countries (from Polymarket docs):\n"
        f"    US, UK, DE, FR, IT, NL, AU, BE, RU + OFAC-sanctioned\n"
        f"  CLOSE-ONLY (can't open new trades):\n"
        f"    PL, SG, TH, TW\n"
        f"  BLOCKED regions:\n"
        f"    Canada/Ontario, Ukraine/Crimea+Donetsk+Luhansk\n"
        f"\n"
        f"  NOTE: Even countries NOT on the blocked list can be rejected\n"
        f"  by the CLOB server's Cloudflare rules. Nordic VPNs work best.\n"
        f"\n"
        f"  Raw: {result['raw']}"
    )
    raise RuntimeError(msg)
