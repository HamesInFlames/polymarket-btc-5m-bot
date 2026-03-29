"""
Polymarket Fee Calculator (v2 — FIXED)
=======================================
Crypto markets on Polymarket charge taker fees.
Formula: fee = C * p * feeRate * (p * (1 - p))^exponent

For crypto markets:
  feeRate  = 0.25
  exponent = 2
  Max effective rate = 1.56% at price $0.50

v2 fixes:
  - Separated GROSS cost (what leaves your wallet) from NET contracts
    (what you receive after fees). Previous version conflated these,
    causing PnL calculation errors that compounded over hundreds of trades.
  - Added trade_economics() helper that returns all the numbers you need
    in one place so no module can get the math wrong independently.

Source: https://docs.polymarket.com/trading/fees
"""

import logging
from typing import Optional

from src.config import CLOB_HOST
from src.http_client import resilient_get

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
    return round(fee, 6)


def effective_fee_rate(price: float) -> float:
    """
    Return the effective fee rate at a given price.
    This is the fraction of (contracts * price) taken as fee.
    Peaks at 1.56% for price = 0.50.
    """
    if price <= 0 or price >= 1:
        return 0.0
    return CRYPTO_FEE_RATE * (price * (1 - price)) ** CRYPTO_EXPONENT


def trade_economics(price: float, num_contracts: float, won: bool) -> dict:
    """
    SINGLE SOURCE OF TRUTH for trade economics.

    Every module that needs to compute costs, payouts, or PnL should
    call this function instead of doing its own fee math.

    Returns:
        {
            "gross_cost":         float,  # what left the wallet (contracts * price)
            "fee_amount":         float,  # total fee in dollars
            "fee_rate":           float,  # effective fee rate (0-0.0156)
            "net_contracts":      float,  # contracts received after fee deduction
            "effective_price":    float,  # true cost per contract = gross / net
            "payout":             float,  # $1.00 per net contract if won, else 0
            "pnl":                float,  # payout - gross_cost
            "pnl_per_contract":   float,  # pnl / num_contracts (for logging)
        }
    """
    if price <= 0 or price >= 1 or num_contracts <= 0:
        return {
            "gross_cost": 0.0, "fee_amount": 0.0, "fee_rate": 0.0,
            "net_contracts": 0.0, "effective_price": 0.0,
            "payout": 0.0, "pnl": 0.0, "pnl_per_contract": 0.0,
        }

    fee_rate = effective_fee_rate(price)
    gross_cost = num_contracts * price
    fee_amount = gross_cost * fee_rate
    # You pay gross_cost from wallet. You receive (1 - fee_rate) fraction
    # of the contracts. Each winning contract pays $1.00.
    net_contracts = num_contracts * (1.0 - fee_rate)
    effective_price = gross_cost / net_contracts if net_contracts > 0 else price

    if won:
        payout = net_contracts * 1.0
        pnl = payout - gross_cost
    else:
        payout = 0.0
        pnl = -gross_cost

    return {
        "gross_cost": round(gross_cost, 6),
        "fee_amount": round(fee_amount, 6),
        "fee_rate": fee_rate,
        "net_contracts": round(net_contracts, 6),
        "effective_price": round(effective_price, 6),
        "payout": round(payout, 6),
        "pnl": round(pnl, 6),
        "pnl_per_contract": round(pnl / num_contracts, 6) if num_contracts > 0 else 0.0,
    }


def net_profit_per_contract(price: float, win: bool) -> float:
    """
    Calculate the net profit per contract after fees.
    For a BUY at `price`:
      - Win:  payout $1.00 - price - fee
      - Lose: -price (fee already paid at entry)

    DEPRECATED: Use trade_economics() instead for accurate multi-contract math.
    """
    fee_per_contract = price * effective_fee_rate(price)
    if win:
        return 1.0 - price - fee_per_contract
    else:
        return -(price + fee_per_contract)


def fetch_fee_rate_bps(token_id: str) -> Optional[int]:
    """Fetch the fee rate in basis points from the CLOB for a specific token."""
    try:
        resp = resilient_get(
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
