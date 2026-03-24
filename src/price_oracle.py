"""
Multi-source BTC/USD price oracle.

Queries up to four independent sources and returns the median to reduce
the impact of any single stale or manipulated feed:
  1. Chainlink BTC/USD aggregator on Polygon (on-chain, authoritative)
  2. Binance spot ticker
  3. Coinbase spot price
  4. CoinGecko simple price

The Chainlink feed is the same source Polymarket uses for resolution,
so it gets priority weighting in the edge calculation.
"""

import json
import logging
import statistics
import time
from typing import Optional

import requests
from web3 import Web3

from src.config import (
    POLYGON_RPC_URL,
    CHAINLINK_BTC_USD,
    BINANCE_BTC_URL,
    COINGECKO_BTC_URL,
    COINBASE_BTC_URL,
)

log = logging.getLogger(__name__)

AGGREGATOR_V3_ABI = json.loads("""[
    {"inputs":[],"name":"latestRoundData","outputs":[
        {"name":"roundId","type":"uint80"},
        {"name":"answer","type":"int256"},
        {"name":"startedAt","type":"uint256"},
        {"name":"updatedAt","type":"uint256"},
        {"name":"answeredInRound","type":"uint80"}
    ],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"decimals","outputs":[
        {"name":"","type":"uint8"}
    ],"stateMutability":"view","type":"function"}
]""")

_FALLBACK_RPCS = [
    POLYGON_RPC_URL,
    "https://polygon-bor-rpc.publicnode.com",
]

_w3: Optional[Web3] = None
_w3_index: int = 0
_chainlink_contract = None
_chainlink_decimals: Optional[int] = None

_last_coingecko_call: float = 0.0
_coingecko_cooldown: float = 15.0


def _get_w3() -> Web3:
    global _w3, _w3_index
    if _w3 is None:
        _w3 = Web3(Web3.HTTPProvider(_FALLBACK_RPCS[_w3_index]))
    return _w3


def _rotate_rpc():
    """Switch to the next fallback RPC on repeated failures."""
    global _w3, _w3_index, _chainlink_contract, _chainlink_decimals
    _w3_index = (_w3_index + 1) % len(_FALLBACK_RPCS)
    _w3 = None
    _chainlink_contract = None
    _chainlink_decimals = None
    log.info("Rotating to fallback RPC: %s", _FALLBACK_RPCS[_w3_index][:50])


_rpc_fail_count: int = 0


def _get_chainlink():
    global _chainlink_contract, _chainlink_decimals, _rpc_fail_count
    if _chainlink_contract is None or _chainlink_decimals is None:
        _chainlink_contract = None
        _chainlink_decimals = None
        w3 = _get_w3()
        addr = Web3.to_checksum_address(CHAINLINK_BTC_USD)
        _chainlink_contract = w3.eth.contract(address=addr, abi=AGGREGATOR_V3_ABI)
        _chainlink_decimals = _chainlink_contract.functions.decimals().call()
        _rpc_fail_count = 0
    return _chainlink_contract, _chainlink_decimals


def get_chainlink_btc_price() -> Optional[float]:
    """Fetch BTC/USD from Chainlink on-chain aggregator (Polygon)."""
    global _rpc_fail_count
    try:
        contract, decimals = _get_chainlink()
        data = contract.functions.latestRoundData().call()
        answer = data[1]
        updated_at = data[3]
        age = time.time() - updated_at
        if age > 300:
            log.warning("Chainlink price is %.0fs stale", age)
        _rpc_fail_count = 0
        return float(answer) / (10 ** decimals)
    except Exception as e:
        _rpc_fail_count += 1
        if _rpc_fail_count >= 3:
            _rotate_rpc()
        log.warning("Chainlink fetch failed: %s", e)
        return None


_binance_failures: int = 0

def get_binance_btc_price() -> Optional[float]:
    global _binance_failures
    if _binance_failures > 5:
        return None
    try:
        r = requests.get(BINANCE_BTC_URL, timeout=5)
        r.raise_for_status()
        _binance_failures = 0
        return float(r.json()["price"])
    except Exception as e:
        _binance_failures += 1
        if _binance_failures <= 3:
            log.warning("Binance fetch failed: %s", e)
        elif _binance_failures == 4:
            log.warning("Binance repeatedly failing — disabling (likely geo-blocked)")
        return None


def get_coinbase_btc_price() -> Optional[float]:
    try:
        r = requests.get(COINBASE_BTC_URL, timeout=5)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])
    except Exception as e:
        log.warning("Coinbase fetch failed: %s", e)
        return None


def get_coingecko_btc_price() -> Optional[float]:
    global _last_coingecko_call
    if time.time() - _last_coingecko_call < _coingecko_cooldown:
        return None
    try:
        _last_coingecko_call = time.time()
        r = requests.get(COINGECKO_BTC_URL, timeout=5)
        r.raise_for_status()
        return float(r.json()["bitcoin"]["usd"])
    except Exception as e:
        log.debug("CoinGecko fetch failed: %s", e)
        return None


def get_btc_price() -> dict:
    """
    Returns a dict with:
        chainlink : float | None  — the authoritative resolution price
        median    : float | None  — median of all successful sources
        sources   : dict          — individual source prices
        count     : int           — how many sources responded
    """
    prices = {}

    chainlink = get_chainlink_btc_price()
    if chainlink:
        prices["chainlink"] = chainlink

    binance = get_binance_btc_price()
    if binance:
        prices["binance"] = binance

    coinbase = get_coinbase_btc_price()
    if coinbase:
        prices["coinbase"] = coinbase

    coingecko = get_coingecko_btc_price()
    if coingecko:
        prices["coingecko"] = coingecko

    values = list(prices.values())
    median = statistics.median(values) if values else None

    return {
        "chainlink": chainlink,
        "median": median,
        "sources": prices,
        "count": len(values),
    }
