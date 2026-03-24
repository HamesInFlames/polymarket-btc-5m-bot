"""
Geoblock checker — verifies the bot's outbound IP against Polymarket's
official blocked country/region list before trading.

Source: https://docs.polymarket.com/api-reference/geoblock
"""

import logging
from typing import Optional

import requests

log = logging.getLogger(__name__)

GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
IP_INFO_URL = "https://ipinfo.io/json"

# Official Polymarket blocked countries (from docs as of March 2026)
BLOCKED_COUNTRIES = {
    "AU", "BE", "BY", "BI", "CF", "CD", "CU", "DE", "ET", "FR",
    "GB", "IR", "IQ", "IT", "KP", "LB", "LY", "MM", "NI", "NL",
    "RU", "SO", "SS", "SD", "SY", "UM", "US", "VE", "YE", "ZW",
}

# Close-only countries (can close positions but NOT open new ones)
CLOSE_ONLY_COUNTRIES = {"PL", "SG", "TH", "TW"}

# Blocked regions within otherwise allowed countries
BLOCKED_REGIONS = {
    "CA": {"ON"},           # Canada: Ontario
    "UA": {"43", "14", "09"},  # Ukraine: Crimea, Donetsk, Luhansk
}

# Countries confirmed NOT blocked (good VPN targets)
SAFE_COUNTRIES = [
    "Portugal (PT)", "Spain (ES)", "Switzerland (CH)", "Austria (AT)",
    "Czech Republic (CZ)", "Romania (RO)", "Hungary (HU)",
    "Denmark (DK)", "Sweden (SE)", "Norway (NO)", "Finland (FI)",
    "Japan (JP)", "South Korea (KR)", "Brazil (BR)", "Mexico (MX)",
    "India (IN)", "Argentina (AR)", "Colombia (CO)", "Ireland (IE)",
]


def check_geoblock(timeout: int = 10) -> dict:
    """
    Check Polymarket's geoblock endpoint AND cross-reference with
    the official blocked country list.

    Returns a dict with:
        - "allowed": bool
        - "ip": str
        - "region": str
        - "country_code": str
        - "region_code": str
        - "block_reason": str  (why blocked, if applicable)
        - "raw": dict
    """
    result = {
        "allowed": True,
        "ip": "unknown",
        "region": "unknown",
        "country_code": "",
        "region_code": "",
        "block_reason": "",
        "raw": {},
    }

    # Step 1: Get IP info
    ip_info = _get_ip_info(timeout)
    if ip_info:
        result["ip"] = ip_info.get("ip", "unknown")
        city = ip_info.get("city", "")
        region = ip_info.get("region", "")
        country = ip_info.get("country", "")
        result["region"] = f"{city}, {region}, {country}".strip(", ")
        result["country_code"] = country
        result["region_code"] = ip_info.get("region", "")

    # Step 2: Check the official geoblock API
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

    # Step 3: Cross-reference with known blocked list (backup)
    cc = result["country_code"]
    rc = result["region_code"]

    if cc in BLOCKED_COUNTRIES:
        result["allowed"] = False
        result["block_reason"] = _explain_block(cc, rc)
    elif cc in CLOSE_ONLY_COUNTRIES:
        result["allowed"] = False
        result["block_reason"] = f"{cc} is close-only — cannot open new positions"
    elif cc in BLOCKED_REGIONS:
        blocked_regions = BLOCKED_REGIONS[cc]
        if rc in blocked_regions:
            result["allowed"] = False
            result["block_reason"] = f"{cc}/{rc} is a blocked region"

    return result


def _explain_block(country_code: str, region_code: str) -> str:
    """Generate a human-readable explanation for why this location is blocked."""
    if country_code in BLOCKED_COUNTRIES:
        return f"{country_code} is on Polymarket's blocked country list"
    if country_code in CLOSE_ONLY_COUNTRIES:
        return f"{country_code} is close-only (cannot open new positions)"
    if country_code in BLOCKED_REGIONS:
        if region_code in BLOCKED_REGIONS[country_code]:
            return f"{country_code}/{region_code} is a blocked region"
    return f"Blocked by Polymarket geoblock API ({country_code}/{region_code})"


def _get_ip_info(timeout: int = 5) -> Optional[dict]:
    """Fetch public IP and geo info from ipinfo.io."""
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
            "Geoblock check PASSED — IP: %s | Region: %s | Country: %s",
            result["ip"], result["region"], result["country_code"],
        )
        return result

    safe_list = ", ".join(SAFE_COUNTRIES[:8])
    msg = (
        f"GEOBLOCKED by Polymarket!\n"
        f"  Detected IP:     {result['ip']}\n"
        f"  Detected region: {result['region']}\n"
        f"  Country code:    {result['country_code']}\n"
        f"  Reason:          {result['block_reason']}\n"
        f"\n"
        f"  Your VPN is connected to a BLOCKED country.\n"
        f"  Switch to one of these countries in NordVPN:\n"
        f"    {safe_list}\n"
        f"\n"
        f"  BLOCKED countries (from Polymarket docs):\n"
        f"    US, UK, DE, FR, IT, NL, AU, BE, RU + OFAC-sanctioned\n"
        f"  CLOSE-ONLY (can't open new trades):\n"
        f"    PL, SG, TH, TW\n"
        f"  BLOCKED regions:\n"
        f"    Canada/Ontario, Ukraine/Crimea+Donetsk+Luhansk\n"
        f"\n"
        f"  Raw: {result['raw']}"
    )
    raise RuntimeError(msg)
