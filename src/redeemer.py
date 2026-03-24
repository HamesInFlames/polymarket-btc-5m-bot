"""
Auto-Redeemer — converts winning outcome tokens back to USDC after resolution.

Uses direct on-chain calls to the CTF contract's redeemPositions().
For EOA wallets this is a simple web3 transaction costing ~0.001 POL gas.

Non-blocking: redeems run in a background thread so the main loop stays free
to discover and trade the next round. A callback fires on completion to
trigger an immediate wallet balance refresh.

Reference: https://docs.polymarket.com/trading/ctf/redeem
"""

import logging
import threading
import time
from typing import Callable, Optional

import requests

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
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
_w3_lock = threading.Lock()

_on_redeem_complete: Optional[Callable[[str, bool], None]] = None


def set_redeem_callback(callback: Callable[[str, bool], None]):
    """Register a callback(condition_id, success) invoked when a background redeem finishes."""
    global _on_redeem_complete
    _on_redeem_complete = callback


_REDEEMER_RPCS = [
    "https://polygon-bor-rpc.publicnode.com",
    POLYGON_RPC_URL,
    "https://polygon.meowrpc.com",
]


def _reset_web3():
    """Force reconnection on next call (e.g. after RPC failure)."""
    global _w3, _account
    with _w3_lock:
        _w3 = None
        _account = None


def _get_web3():
    global _w3, _account
    with _w3_lock:
        if _w3 is not None:
            try:
                _w3.eth.block_number
                return _w3, _account
            except Exception:
                log.warning("Stale RPC connection — reconnecting")
                _w3 = None
                _account = None

        for rpc in _REDEEMER_RPCS:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
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
    NON-BLOCKING: sends the tx and waits for receipt in a background thread.
    Returns the tx hash immediately after broadcast (before confirmation).
    The on_redeem_complete callback fires when the receipt arrives.
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
            tx_hash = _send_neg_risk_tx(w3, acct, address, cond_bytes)
        else:
            tx_hash = _send_standard_tx(w3, acct, address, cond_bytes)

        if tx_hash is None:
            return None

        hex_hash = tx_hash.hex()
        log.info("Redeem tx broadcast: %s — waiting for receipt in background", hex_hash)

        t = threading.Thread(
            target=_wait_for_receipt,
            args=(w3, tx_hash, condition_id, hex_hash),
            daemon=True,
            name=f"redeem-{condition_id[:12]}",
        )
        t.start()

        return hex_hash

    except Exception as e:
        log.error("Redemption failed for %s: %s", condition_id[:16], e)
        _reset_web3()
        return None


def redeem_winning_position_blocking(
    condition_id: str,
    neg_risk: bool = False,
) -> Optional[str]:
    """Blocking version for standalone scripts (redeem_winnings.py)."""
    if not PRIVATE_KEY:
        return None
    try:
        w3, acct = _get_web3()
        address = acct.address
        if not condition_id.startswith("0x"):
            condition_id = "0x" + condition_id
        cond_bytes = bytes.fromhex(condition_id[2:].zfill(64))
        if neg_risk:
            tx_hash = _send_neg_risk_tx(w3, acct, address, cond_bytes)
        else:
            tx_hash = _send_standard_tx(w3, acct, address, cond_bytes)
        if tx_hash is None:
            return None
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        hex_hash = tx_hash.hex()
        if receipt["status"] == 1:
            log.info("Redeem SUCCESS: %s (gas: %d)", hex_hash, receipt["gasUsed"])
            return hex_hash
        log.error("Redeem REVERTED: %s", hex_hash)
        return None
    except Exception as e:
        log.error("Blocking redeem failed for %s: %s", condition_id[:16], e)
        _reset_web3()
        return None


def _wait_for_receipt(w3, tx_hash, condition_id: str, hex_hash: str):
    """Background thread: wait for tx confirmation, then fire callback."""
    success = False
    try:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)
        if receipt["status"] == 1:
            log.info("Redeem CONFIRMED: %s (gas: %d)", hex_hash, receipt["gasUsed"])
            success = True
        else:
            log.error("Redeem REVERTED: %s", hex_hash)
    except Exception as e:
        log.error("Redeem receipt wait failed for %s: %s", hex_hash, e)
        _reset_web3()

    if _on_redeem_complete:
        try:
            _on_redeem_complete(condition_id, success)
        except Exception as e:
            log.error("Redeem callback error: %s", e)


def _send_standard_tx(w3, acct, address, condition_id_bytes):
    """Build, sign, and broadcast a standard CTF redemption tx. Returns tx_hash or None."""
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
    return w3.eth.send_raw_transaction(signed.raw_transaction)


def _send_neg_risk_tx(w3, acct, address, condition_id_bytes):
    """Build, sign, and broadcast a neg-risk redemption tx. Returns tx_hash or None."""
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
    return w3.eth.send_raw_transaction(signed.raw_transaction)


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


GAMMA_POSITIONS_URL = "https://gamma-api.polymarket.com/positions"
DATA_API_POSITIONS_URL = "https://data-api.polymarket.com/positions"


def _fetch_open_positions(wallet_lower: str) -> list:
    positions = []
    try:
        resp = requests.get(
            GAMMA_POSITIONS_URL,
            params={"user": wallet_lower},
            timeout=15,
        )
        if resp.ok:
            positions = resp.json()
            if isinstance(positions, list) and positions:
                return positions
    except Exception as e:
        log.debug("Gamma positions fetch: %s", e)

    try:
        resp = requests.get(
            DATA_API_POSITIONS_URL,
            params={"user": wallet_lower},
            timeout=15,
        )
        if resp.ok:
            data = resp.json()
            if isinstance(data, list):
                return data
    except Exception as e:
        log.debug("Data API positions fetch: %s", e)

    return []


def sweep_unredeemed_positions() -> int:
    """
    Query Polymarket APIs for open positions and redeem each unique condition.
    Safe to call on a timer: no-ops when nothing is redeemable or tx reverts.
    Returns the number of successful on-chain redemptions.
    """
    if not PRIVATE_KEY:
        return 0

    try:
        _, acct = _get_web3()
        addr = acct.address.lower()
    except Exception as e:
        log.warning("Redeem sweep: cannot connect web3: %s", e)
        return 0

    positions = _fetch_open_positions(addr)
    if not positions:
        log.debug("Redeem sweep: no positions from API")
        return 0

    checked: set[str] = set()
    redeemed = 0

    for pos in positions:
        cid = pos.get("conditionId", pos.get("condition_id", ""))
        if not cid or cid in checked:
            continue

        size = float(pos.get("size", pos.get("amount", 0)) or 0)
        if size <= 0:
            continue

        checked.add(cid)
        neg_risk = bool(pos.get("negRisk", pos.get("neg_risk", False)))
        outcome = pos.get("outcome", pos.get("title", "?"))

        log.info(
            "Redeem sweep: trying %s | %s | cid=%s...",
            outcome, f"{size:.2f} shares", cid[:16],
        )
        try:
            txh = redeem_winning_position_blocking(cid, neg_risk=neg_risk)
            if txh:
                redeemed += 1
            time.sleep(2)
        except Exception as e:
            err = str(e).lower()
            if "revert" in err or "execution reverted" in err:
                log.debug("Redeem sweep: not redeemable for %s", cid[:16])
            else:
                log.warning("Redeem sweep failed for %s: %s", cid[:16], e)
            time.sleep(1)

    if redeemed:
        log.info("Redeem sweep finished: %d successful redemption(s)", redeemed)
    return redeemed
