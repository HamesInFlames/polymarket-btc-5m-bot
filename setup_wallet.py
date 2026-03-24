"""
Wallet Setup Helper for Polymarket Bot
---------------------------------------
Generates a new Polygon wallet, sets CLOB API credentials,
and runs the on-chain approval transactions needed for trading.

Usage:
    python setup_wallet.py              # Generate new wallet
    python setup_wallet.py --approve    # Run on-chain approvals for existing wallet
"""

import os
import sys
import json
import argparse

from dotenv import load_dotenv, set_key
from eth_account import Account
from web3 import Web3

USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
CTF_ADDRESS = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
CTF_EXCHANGE = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
NEG_RISK_CTF_EXCHANGE = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")
NEG_RISK_ADAPTER = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")

ERC20_APPROVE_ABI = json.loads(
    '[{"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],'
    '"name":"approve","outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"}]'
)

ERC1155_APPROVAL_ABI = json.loads(
    '[{"inputs":[{"name":"operator","type":"address"},{"name":"approved","type":"bool"}],'
    '"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"}]'
)

MAX_UINT256 = 2**256 - 1

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def generate_wallet():
    acct = Account.create()
    print("\n===  NEW POLYGON WALLET GENERATED  ===")
    print(f"  Address:     {acct.address}")
    print(f"  Private Key: {acct.key.hex()}")
    print("=" * 42)
    print("\nIMPORTANT — Before trading you must fund this wallet:")
    print("  1. Send USDC.e (PoS) to the address above on Polygon network")
    print("  2. Send a small amount of POL (ex-MATIC) for gas fees (~0.5 POL is plenty)")
    print("  3. Run:  python setup_wallet.py --approve")
    print()

    if not os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r" if os.path.exists(ENV_PATH) else "w") as f:
            pass

    if os.path.exists(os.path.join(os.path.dirname(ENV_PATH), ".env.example")) and not os.path.exists(ENV_PATH):
        import shutil
        shutil.copy(
            os.path.join(os.path.dirname(ENV_PATH), ".env.example"),
            ENV_PATH,
        )

    if not os.path.exists(ENV_PATH):
        open(ENV_PATH, "w").close()

    set_key(ENV_PATH, "PRIVATE_KEY", acct.key.hex())
    print(f"Private key saved to {ENV_PATH}")
    return acct


def _send_tx(w3: Web3, account, tx):
    tx["nonce"] = w3.eth.get_transaction_count(account.address)
    tx["gas"] = w3.eth.estimate_gas(tx)
    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", w3.eth.gas_price)
    tx["maxFeePerGas"] = base_fee * 2
    tx["maxPriorityFeePerGas"] = w3.to_wei(30, "gwei")
    tx.pop("gasPrice", None)
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    return receipt


def run_approvals():
    load_dotenv(ENV_PATH)
    pk = os.getenv("PRIVATE_KEY", "").strip()
    if not pk:
        print("ERROR: PRIVATE_KEY not set in .env — run  python setup_wallet.py  first")
        sys.exit(1)

    import time
    rpc_urls = [
        "https://polygon-bor-rpc.publicnode.com",
        os.getenv("POLYGON_RPC_URL", ""),
        "https://polygon.meowrpc.com",
    ]
    rpc_urls = [u for u in rpc_urls if u]

    from web3.middleware import ExtraDataToPOAMiddleware

    w3 = None
    for rpc in rpc_urls:
        for attempt in range(3):
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 15}))
                w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                w3.eth.block_number
                print(f"Connected to RPC: {rpc[:50]}...")
                break
            except Exception:
                time.sleep(2)
                w3 = None
        if w3 is not None:
            break

    if w3 is None:
        print("ERROR: Cannot connect to any Polygon RPC")
        sys.exit(1)

    account = Account.from_key(pk)
    print(f"\nWallet: {account.address}")

    bal_pol = w3.eth.get_balance(account.address)
    usdc = w3.eth.contract(address=USDC_ADDRESS, abi=ERC20_APPROVE_ABI)

    print(f"POL balance:  {w3.from_wei(bal_pol, 'ether'):.4f}")
    if bal_pol == 0:
        print("ERROR: No POL for gas. Send at least 0.5 POL to this address first.")
        sys.exit(1)

    approvals = [
        ("USDC -> CTF Exchange", usdc, CTF_EXCHANGE),
        ("USDC -> NegRisk CTF Exchange", usdc, NEG_RISK_CTF_EXCHANGE),
    ]

    ctf = w3.eth.contract(address=CTF_ADDRESS, abi=ERC1155_APPROVAL_ABI)
    erc1155_approvals = [
        ("CTF -> CTF Exchange", ctf, CTF_EXCHANGE),
        ("CTF -> NegRisk CTF Exchange", ctf, NEG_RISK_CTF_EXCHANGE),
        ("CTF -> NegRisk Adapter", ctf, NEG_RISK_ADAPTER),
    ]

    all_txs = [(l, c, s, "approve") for l, c, s in approvals] + \
               [(l, c, o, "setApprovalForAll") for l, c, o in erc1155_approvals]

    for label, contract, target, method in all_txs:
        print(f"  Approving {label} ... ", end="", flush=True)
        for attempt in range(3):
            try:
                if method == "approve":
                    tx = contract.functions.approve(target, MAX_UINT256).build_transaction(
                        {"from": account.address, "chainId": 137}
                    )
                else:
                    tx = contract.functions.setApprovalForAll(target, True).build_transaction(
                        {"from": account.address, "chainId": 137}
                    )
                receipt = _send_tx(w3, account, tx)
                status = "OK" if receipt["status"] == 1 else "FAILED"
                print(f"{status}  tx: {receipt['transactionHash'].hex()}")
                time.sleep(3)
                break
            except Exception as e:
                if "nonce" in str(e).lower() and attempt < 2:
                    print(f"nonce conflict, retrying...", end=" ", flush=True)
                    time.sleep(5)
                else:
                    print(f"FAILED: {e}")
                    break

    print("\nAll approvals complete. You can now run the bot.")


def derive_api_creds():
    load_dotenv(ENV_PATH)
    pk = os.getenv("PRIVATE_KEY", "").strip()
    if not pk:
        print("ERROR: PRIVATE_KEY not set.")
        sys.exit(1)

    from py_clob_client.client import ClobClient

    client = ClobClient(
        host="https://clob.polymarket.com",
        key=pk,
        chain_id=137,
    )
    creds = client.create_or_derive_api_creds()
    print("\nCLOB API Credentials (auto-derived, no need to save):")
    print(f"  API Key:        {creds.api_key}")
    print(f"  API Secret:     {creds.api_secret}")
    print(f"  API Passphrase: {creds.api_passphrase}")
    print("\nThese are deterministically derived from your private key —")
    print("the bot re-derives them on every startup.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket Bot Wallet Setup")
    parser.add_argument("--approve", action="store_true", help="Run on-chain approvals")
    parser.add_argument("--creds", action="store_true", help="Derive and display CLOB API creds")
    args = parser.parse_args()

    if args.approve:
        run_approvals()
    elif args.creds:
        derive_api_creds()
    else:
        generate_wallet()
