"""
Polymarket Fee Calculator
=========================
Crypto markets on Polymarket charge taker fees.
Formula: fee = C * p * feeRate * (p * (1 - p))^exponent

For crypto markets:
  feeRate  = 0.25
  exponent = 2
  Max effective rate = 1.56% at price $0.50

Source: https://docs.polymarket.com/trading/fees
"""

import logging
from typing import Optional

import requests

from src.config import CLOB_HOST

log = logging.getLogger(__name__)

CRYPTO_FEE_RATE = 0.25
CRYPTO_EXPONENT = 2


def calculate_crypto_fee(num_contracts: float, price: float) -> float:
    """
    Calculate the taker fee for a crypto market trade.

    fee = C * p * feeRate * (p * (1 - p))^exponent
    """
    if price <= 0 or price >= 1 or num_contracts <= 0:
        return 0.0

    fee = num_contracts * price * CRYPTO_FEE_RATE * (price * (1 - price)) ** CRYPTO_EXPONENT
    return round(fee, 4)


def effective_fee_rate(price: float) -> float:
    """
    Return the effective fee rate at a given price.
    Peaks at 1.56% for price = 0.50.
    """
    if price <= 0 or price >= 1:
        return 0.0
    return CRYPTO_FEE_RATE * (price * (1 - price)) ** CRYPTO_EXPONENT


def net_profit_per_contract(price: float, win: bool) -> float:
    """
    Calculate the net profit per contract after fees.
    For a BUY at `price`:
      - Win:  payout $1.00 - price - fee
      - Lose: -price (fee already paid at entry)
    """
    fee_per_contract = price * effective_fee_rate(price)
    if win:
        return 1.0 - price - fee_per_contract
    else:
        return -(price + fee_per_contract)


def fetch_fee_rate_bps(token_id: str) -> Optional[int]:
    """Fetch the fee rate in basis points from the CLOB for a specific token."""
    try:
        resp = requests.get(
            f"{CLOB_HOST}/fee-rate",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return int(data.get("fee_rate_bps", 0))
    except Exception as e:
        log.warning("Fee rate fetch failed: %s", e)
        return None
