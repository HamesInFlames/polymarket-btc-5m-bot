"""
Redeem all winning outcome tokens from resolved Polymarket BTC 5-min markets.
Scans recent markets, checks for token balances, and redeems any found.
"""

import os
import sys
import time
import requests
from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

PRIVATE_KEY = os.getenv("PRIVATE_KEY", "").strip()
WALLET = Account.from_key(PRIVATE_KEY).address if PRIVATE_KEY else ""

from web3.middleware import ExtraDataToPOAMiddleware

RPC = "https://polygon-bor-rpc.publicnode.com"
w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 15}))
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
CTF_ADDRESS = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
NEG_RISK_CTF = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")
GAMMA_API = "https://gamma-api.polymarket.com"

CTF_ABI = [
    {"name": "balanceOf", "type": "function",
     "inputs": [{"name": "account", "type": "address"}, {"name": "id", "type": "uint256"}],
     "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view"},
    {"name": "redeemPositions", "type": "function",
     "inputs": [
         {"name": "collateralToken", "type": "address"},
         {"name": "parentCollectionId", "type": "bytes32"},
         {"name": "conditionId", "type": "bytes32"},
         {"name": "indexSets", "type": "uint256[]"},
     ], "outputs": []},
]

ERC20_ABI = [
    {"name": "balanceOf", "type": "function",
     "inputs": [{"name": "account", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view"},
]


def send_tx(account, tx):
    tx["nonce"] = w3.eth.get_transaction_count(account.address)
    tx["gas"] = w3.eth.estimate_gas(tx)
    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", w3.eth.gas_price)
    tx["maxFeePerGas"] = base_fee * 2
    tx["maxPriorityFeePerGas"] = w3.to_wei(30, "gwei")
    tx.pop("gasPrice", None)
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  TX: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    return receipt


def get_recent_btc_markets():
    """Fetch recent BTC 5-min markets from Gamma API."""
    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={"slug_contains": "btc-updown-5m", "closed": "true", "limit": 50},
            timeout=15,
        )
        if resp.ok:
            return resp.json()
    except Exception:
        pass

    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={"tag": "btc-5-minute", "closed": "true", "limit": 100},
            timeout=15,
        )
        if resp.ok:
            return resp.json()
    except Exception:
        pass

    return []


def find_token_ids_from_clob():
    """Query the CLOB for recent BTC markets and their token IDs."""
    tokens = []
    try:
        resp = requests.get(
            "https://clob.polymarket.com/markets",
            params={"next_cursor": "MA=="},
            timeout=15,
        )
        if not resp.ok:
            return tokens
        data = resp.json()
        for market in data.get("data", data) if isinstance(data, dict) else data:
            q = (market.get("question", "") + market.get("description", "")).lower()
            if "bitcoin" in q and ("5 minute" in q or "5-minute" in q):
                for token in market.get("tokens", []):
                    tid = token.get("token_id", "")
                    if tid:
                        tokens.append({
                            "token_id": tid,
                            "condition_id": market.get("condition_id", ""),
                            "question": market.get("question", "")[:60],
                            "outcome": token.get("outcome", ""),
                        })
    except Exception as e:
        print(f"  CLOB query failed: {e}")
    return tokens


def main():
    if not PRIVATE_KEY:
        print("ERROR: No PRIVATE_KEY in .env")
        sys.exit(1)

    account = Account.from_key(PRIVATE_KEY)
    addr = account.address
    print(f"Wallet: {addr}")

    ctf = w3.eth.contract(address=CTF_ADDRESS, abi=CTF_ABI)
    usdc = w3.eth.contract(address=USDC_E, abi=ERC20_ABI)

    usdc_before = usdc.functions.balanceOf(addr).call() / 1e6
    print(f"USDC.e before: ${usdc_before:.6f}")
    print()

    # Strategy: query the Polymarket data API for positions
    print("Checking Polymarket positions API...")
    positions = []
    try:
        resp = requests.get(
            f"{GAMMA_API}/positions",
            params={"user": addr.lower()},
            timeout=15,
        )
        if resp.ok:
            positions = resp.json()
            print(f"  Found {len(positions)} position(s)")
    except Exception as e:
        print(f"  Positions API failed: {e}")

    if not positions:
        try:
            resp = requests.get(
                f"https://data-api.polymarket.com/positions",
                params={"user": addr.lower()},
                timeout=15,
            )
            if resp.ok:
                positions = resp.json()
                print(f"  Found {len(positions)} position(s) from data API")
        except Exception as e:
            print(f"  Data API failed: {e}")

    # Check each position for redeemable tokens
    redeemed = 0
    checked = set()

    for pos in positions:
        cid = pos.get("conditionId", pos.get("condition_id", ""))
        if not cid or cid in checked:
            continue
        checked.add(cid)

        asset = pos.get("asset", pos.get("token_id", ""))
        size = float(pos.get("size", pos.get("amount", 0)))

        if size <= 0 and asset:
            bal = ctf.functions.balanceOf(addr, int(asset)).call()
            if bal > 0:
                size = bal / 1e6

        if size <= 0:
            continue

        outcome = pos.get("outcome", pos.get("title", "?"))
        print(f"\n  Position: {outcome} | size={size:.2f} | cid={cid[:16]}...")

        if not cid.startswith("0x"):
            cid = "0x" + cid
        cond_bytes = bytes.fromhex(cid[2:].zfill(64))

        print(f"  Attempting redemption...")
        try:
            tx = ctf.functions.redeemPositions(
                USDC_E,
                b"\x00" * 32,
                cond_bytes,
                [1, 2],
            ).build_transaction({
                "from": addr,
                "chainId": 137,
            })
            receipt = send_tx(account, tx)
            if receipt["status"] == 1:
                print(f"  REDEEMED OK (gas: {receipt['gasUsed']})")
                redeemed += 1
            else:
                print(f"  REVERTED")
            time.sleep(3)
        except Exception as e:
            err = str(e)
            if "execution reverted" in err.lower():
                print(f"  No redeemable tokens (already redeemed or lost)")
            else:
                print(f"  Failed: {e}")

    # Also try brute-force: check token balances for known recent markets
    print("\n\nScanning CLOB for recent BTC 5-min markets...")
    clob_tokens = find_token_ids_from_clob()
    print(f"  Found {len(clob_tokens)} tokens to check")

    for tok in clob_tokens:
        tid = tok["token_id"]
        cid = tok["condition_id"]
        if cid in checked:
            continue

        try:
            bal = ctf.functions.balanceOf(addr, int(tid)).call()
        except Exception:
            continue

        if bal == 0:
            continue

        checked.add(cid)
        human_bal = bal / 1e6
        print(f"\n  Found {human_bal:.2f} tokens: {tok['question']} ({tok['outcome']})")
        print(f"  condition_id: {cid[:16]}...")

        if not cid.startswith("0x"):
            cid = "0x" + cid
        cond_bytes = bytes.fromhex(cid[2:].zfill(64))

        try:
            tx = ctf.functions.redeemPositions(
                USDC_E,
                b"\x00" * 32,
                cond_bytes,
                [1, 2],
            ).build_transaction({
                "from": addr,
                "chainId": 137,
            })
            receipt = send_tx(account, tx)
            if receipt["status"] == 1:
                print(f"  REDEEMED OK (gas: {receipt['gasUsed']})")
                redeemed += 1
            else:
                print(f"  REVERTED")
            time.sleep(3)
        except Exception as e:
            err = str(e)
            if "execution reverted" in err.lower():
                print(f"  Not redeemable (already redeemed or lost)")
            else:
                print(f"  Failed: {e}")

    print(f"\n\nRedeemed {redeemed} position(s)")
    usdc_after = usdc.functions.balanceOf(addr).call() / 1e6
    gained = usdc_after - usdc_before
    print(f"USDC.e before: ${usdc_before:.6f}")
    print(f"USDC.e after:  ${usdc_after:.6f}")
    print(f"Gained:        ${gained:.6f}")


if __name__ == "__main__":
    main()
