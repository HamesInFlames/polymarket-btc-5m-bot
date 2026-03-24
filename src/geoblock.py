"""
Geoblock checker — calls Polymarket's /api/geoblock endpoint to verify
the bot's outbound IP is not in a restricted region before trading.

Reference: https://docs.polymarket.com (geographic restrictions)
Endpoint:  GET https://polymarket.com/api/geoblock
"""

import logging
from typing import Optional

import requests

log = logging.getLogger(__name__)

GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
IP_INFO_URL = "https://ipinfo.io/json"


def check_geoblock(timeout: int = 10) -> dict:
    """
    Query Polymarket's geoblock endpoint and return the raw response.

    Returns a dict with at minimum:
        - "allowed": bool   (True = trading permitted from this IP)
        - "ip": str         (detected public IP, if available)
        - "region": str     (detected region/country, if available)
        - "raw": dict       (full API response for debugging)
    """
    result = {"allowed": True, "ip": "unknown", "region": "unknown", "raw": {}}

    ip_info = _get_ip_info(timeout)
    if ip_info:
        result["ip"] = ip_info.get("ip", "unknown")
        city = ip_info.get("city", "")
        region = ip_info.get("region", "")
        country = ip_info.get("country", "")
        result["region"] = f"{city}, {region}, {country}".strip(", ")

    try:
        resp = requests.get(GEOBLOCK_URL, timeout=timeout)

        if resp.status_code == 200:
            data = resp.json()
            result["raw"] = data
            blocked = data.get("blocked", False)
            if isinstance(blocked, str):
                blocked = blocked.lower() in ("true", "1", "yes")
            result["allowed"] = not blocked
        elif resp.status_code == 403:
            result["allowed"] = False
            result["raw"] = {"status": 403, "reason": "Forbidden"}
        else:
            log.warning(
                "Geoblock check returned HTTP %d — assuming allowed (cannot confirm)",
                resp.status_code,
            )
            result["raw"] = {"status": resp.status_code}

    except requests.exceptions.RequestException as e:
        log.warning("Geoblock check failed (network error): %s — assuming allowed", e)
        result["raw"] = {"error": str(e)}

    return result


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
    Run the geoblock check and raise RuntimeError if blocked.
    Returns the check result on success.
    """
    result = check_geoblock()

    if result["allowed"]:
        log.info(
            "Geoblock check PASSED — IP: %s | Region: %s",
            result["ip"], result["region"],
        )
        return result

    msg = (
        f"GEOBLOCKED by Polymarket!\n"
        f"  Detected IP:     {result['ip']}\n"
        f"  Detected region: {result['region']}\n"
        f"\n"
        f"  Polymarket blocks trading from this location.\n"
        f"  If you're using a VPN, make sure it's connected to a\n"
        f"  non-blocked server (e.g., Montreal/Quebec, NOT Ontario).\n"
        f"\n"
        f"  NordVPN fix: open NordVPN -> click the map or server list\n"
        f"  -> pick 'Montreal' or 'Quebec' specifically, NOT 'Canada'\n"
        f"  (auto-select often picks Ontario servers).\n"
        f"\n"
        f"  Raw response: {result['raw']}"
    )
    raise RuntimeError(msg)
