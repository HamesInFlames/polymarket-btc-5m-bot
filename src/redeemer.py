"""
Auto-Redeemer — converts winning outcome tokens back to USDC after resolution.

Uses direct on-chain calls to the CTF contract's redeemPositions().
For EOA wallets this is a simple web3 transaction costing ~0.001 POL gas.

Reference: https://docs.polymarket.com/trading/ctf/redeem
"""

import logging
import time
from typing import Optional

from web3 import Web3
from eth_account import Account

from src.config import (
    PRIVATE_KEY,
    POLYGON_RPC_URL,
    CTF_ADDRESS,
    NEG_RISK_CTF_EXCHANGE,
    USDC_ADDRESS,
    CHAIN_ID,
)

log = logging.getLogger(__name__)

HASH_ZERO = b"\x00" * 32

CTF_REDEEM_ABI = [
    {
        "name": "redeemPositions",
        "type": "function",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "outputs": [],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]

NEG_RISK_REDEEM_ABI = [
    {
        "name": "redeemPositions",
        "type": "function",
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "amounts", "type": "uint256[]"},
        ],
        "outputs": [],
    },
]

_w3: Optional[Web3] = None
_account = None


_REDEEMER_RPCS = [
    "https://polygon-bor-rpc.publicnode.com",
    POLYGON_RPC_URL,
    "https://polygon.meowrpc.com",
]


def _get_web3():
    global _w3, _account
    if _w3 is not None:
        return _w3, _account

    for rpc in _REDEEMER_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
            w3.eth.block_number
            _w3 = w3
            break
        except Exception:
            continue

    if _w3 is None:
        raise RuntimeError("Cannot connect to any Polygon RPC")

    _account = Account.from_key(PRIVATE_KEY)
    log.info("Redeemer web3 connected, wallet %s", _account.address)
    return _w3, _account


def redeem_winning_position(
    condition_id: str,
    neg_risk: bool = False,
) -> Optional[str]:
    """
    Redeem winning outcome tokens for a resolved market.

    Args:
        condition_id: The market condition ID (hex string with 0x prefix)
        neg_risk: Whether this is a neg-risk market

    Returns:
        Transaction hash on success, None on failure.
    """
    if not PRIVATE_KEY:
        log.warning("No PRIVATE_KEY — cannot redeem on-chain")
        return None

    try:
        w3, acct = _get_web3()
        address = acct.address

        if not condition_id.startswith("0x"):
            condition_id = "0x" + condition_id
        cond_bytes = bytes.fromhex(condition_id[2:].zfill(64))

        if neg_risk:
            return _redeem_neg_risk(w3, acct, address, cond_bytes)
        else:
            return _redeem_standard(w3, acct, address, cond_bytes)

    except Exception as e:
        log.error("Redemption failed for %s: %s", condition_id[:16], e)
        return None


def _redeem_standard(w3, acct, address, condition_id_bytes) -> Optional[str]:
    """Standard CTF redemption: redeemPositions(collateral, parent, conditionId, [1,2])"""
    ctf = w3.eth.contract(
        address=Web3.to_checksum_address(CTF_ADDRESS),
        abi=CTF_REDEEM_ABI,
    )

    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", w3.eth.gas_price)

    txn = ctf.functions.redeemPositions(
        Web3.to_checksum_address(USDC_ADDRESS),
        HASH_ZERO,
        condition_id_bytes,
        [1, 2],
    ).build_transaction({
        "from": address,
        "nonce": w3.eth.get_transaction_count(address),
        "gas": 300_000,
        "maxFeePerGas": base_fee * 2,
        "maxPriorityFeePerGas": w3.to_wei(30, "gwei"),
        "chainId": CHAIN_ID,
    })

    signed = acct.sign_transaction(txn)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    hex_hash = tx_hash.hex()

    log.info("Redeem tx sent: %s (waiting for confirmation...)", hex_hash)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

    if receipt["status"] == 1:
        log.info("Redeem SUCCESS: %s (gas used: %d)", hex_hash, receipt["gasUsed"])
        return hex_hash
    else:
        log.error("Redeem REVERTED: %s", hex_hash)
        return None


def _redeem_neg_risk(w3, acct, address, condition_id_bytes) -> Optional[str]:
    """Neg-risk redemption via the NegRiskCTFExchange contract."""
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(NEG_RISK_CTF_EXCHANGE),
        abi=NEG_RISK_REDEEM_ABI,
    )

    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", w3.eth.gas_price)

    txn = contract.functions.redeemPositions(
        condition_id_bytes,
        [0, 0],
    ).build_transaction({
        "from": address,
        "nonce": w3.eth.get_transaction_count(address),
        "gas": 300_000,
        "maxFeePerGas": base_fee * 2,
        "maxPriorityFeePerGas": w3.to_wei(30, "gwei"),
        "chainId": CHAIN_ID,
    })

    signed = acct.sign_transaction(txn)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    hex_hash = tx_hash.hex()

    log.info("Neg-risk redeem tx sent: %s", hex_hash)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

    if receipt["status"] == 1:
        log.info("Neg-risk redeem SUCCESS: %s (gas: %d)", hex_hash, receipt["gasUsed"])
        return hex_hash
    else:
        log.error("Neg-risk redeem REVERTED: %s", hex_hash)
        return None


def check_pol_balance() -> float:
    """Check POL (gas token) balance — needed for on-chain redemptions."""
    try:
        w3, acct = _get_web3()
        bal_wei = w3.eth.get_balance(acct.address)
        return float(w3.from_wei(bal_wei, "ether"))
    except Exception:
        return 0.0


ERC20_BALANCE_ABI = [
    {
        "name": "balanceOf",
        "type": "function",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "name": "decimals",
        "type": "function",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
    },
]


def fetch_wallet_balances(address: str | None = None) -> dict:
    """
    Fetch USDC.e and POL balances for a wallet address on Polygon.

    Returns dict with keys: address, usdc, pol.
    """
    from src.config import WALLET_ADDRESS
    addr = address or WALLET_ADDRESS
    if not addr:
        return {"address": "", "usdc": 0.0, "pol": 0.0}

    try:
        w3, _ = _get_web3()
        checksum = Web3.to_checksum_address(addr)

        pol_wei = w3.eth.get_balance(checksum)
        pol = float(w3.from_wei(pol_wei, "ether"))

        usdc_contract = w3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS),
            abi=ERC20_BALANCE_ABI,
        )
        usdc_raw = usdc_contract.functions.balanceOf(checksum).call()
        usdc = usdc_raw / 1e6  # USDC.e has 6 decimals

        log.info("Wallet %s — USDC: $%.4f  POL: %.4f", addr[:10], usdc, pol)
        return {"address": addr, "usdc": usdc, "pol": pol}

    except Exception as e:
        log.warning("Failed to fetch wallet balances for %s: %s", addr[:10], e)
        return {"address": addr, "usdc": 0.0, "pol": 0.0}
