"""
Swap native USDC -> USDC.e (PoS) on Polygon via Uniswap V3 SwapRouter.
Polymarket CLOB requires USDC.e, not native USDC.
"""

import os
import sys
import time

from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

NATIVE_USDC = Web3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")
USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
SWAP_ROUTER = Web3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")

ERC20_ABI = [
    {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf",
     "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
     "name": "approve", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "decimals",
     "outputs": [{"name": "", "type": "uint8"}], "stateMutability": "view", "type": "function"},
]

# Uniswap V3 SwapRouter exactInputSingle
SWAP_ROUTER_ABI = [
    {
        "inputs": [{
            "components": [
                {"name": "tokenIn", "type": "address"},
                {"name": "tokenOut", "type": "address"},
                {"name": "fee", "type": "uint24"},
                {"name": "recipient", "type": "address"},
                {"name": "deadline", "type": "uint256"},
                {"name": "amountIn", "type": "uint256"},
                {"name": "amountOutMinimum", "type": "uint256"},
                {"name": "sqrtPriceLimitX96", "type": "uint160"},
            ],
            "name": "params",
            "type": "tuple",
        }],
        "name": "exactInputSingle",
        "outputs": [{"name": "amountOut", "type": "uint256"}],
        "stateMutability": "payable",
        "type": "function",
    }
]


def send_tx(w3, account, tx):
    tx["nonce"] = w3.eth.get_transaction_count(account.address)
    tx["gas"] = w3.eth.estimate_gas(tx)
    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", w3.eth.gas_price)
    tx["maxFeePerGas"] = base_fee * 2
    tx["maxPriorityFeePerGas"] = w3.to_wei(30, "gwei")
    tx.pop("gasPrice", None)
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  TX sent: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    return receipt


def main():
    pk = os.getenv("PRIVATE_KEY", "").strip()
    if not pk:
        print("ERROR: No PRIVATE_KEY in .env")
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com"))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    account = Account.from_key(pk)
    addr = account.address
    print(f"Wallet: {addr}")

    native_usdc = w3.eth.contract(address=NATIVE_USDC, abi=ERC20_ABI)
    usdc_e = w3.eth.contract(address=USDC_E, abi=ERC20_ABI)

    balance = native_usdc.functions.balanceOf(addr).call()
    decimals = native_usdc.functions.decimals().call()
    human_bal = balance / (10 ** decimals)

    print(f"Native USDC balance: ${human_bal:.6f}")
    if balance == 0:
        print("No native USDC to swap.")
        sys.exit(0)

    usdc_e_before = usdc_e.functions.balanceOf(addr).call() / 1e6
    print(f"USDC.e balance before: ${usdc_e_before:.6f}")

    # Step 1: Approve SwapRouter to spend native USDC
    print("\n1. Approving SwapRouter to spend native USDC...", flush=True)
    tx = native_usdc.functions.approve(SWAP_ROUTER, balance).build_transaction(
        {"from": addr, "chainId": 137}
    )
    receipt = send_tx(w3, account, tx)
    if receipt["status"] != 1:
        print("  Approve FAILED!")
        sys.exit(1)
    print("  Approved OK")
    time.sleep(3)

    # Step 2: Swap via Uniswap V3 (100 bps fee tier = stablecoin pool)
    print(f"\n2. Swapping ${human_bal:.2f} native USDC -> USDC.e ...", flush=True)
    router = w3.eth.contract(address=SWAP_ROUTER, abi=SWAP_ROUTER_ABI)

    # Accept up to 1% slippage on stablecoin swap
    min_out = int(balance * 0.99)

    swap_params = (
        NATIVE_USDC,       # tokenIn
        USDC_E,            # tokenOut
        100,               # fee (0.01% pool for stablecoins)
        addr,              # recipient
        int(time.time()) + 600,  # deadline
        balance,           # amountIn
        min_out,           # amountOutMinimum
        0,                 # sqrtPriceLimitX96 (0 = no limit)
    )

    tx = router.functions.exactInputSingle(swap_params).build_transaction(
        {"from": addr, "chainId": 137, "value": 0}
    )
    receipt = send_tx(w3, account, tx)
    if receipt["status"] != 1:
        print("  Swap FAILED! Check on Polygonscan.")
        sys.exit(1)

    time.sleep(3)
    usdc_e_after = usdc_e.functions.balanceOf(addr).call() / 1e6
    gained = usdc_e_after - usdc_e_before
    print(f"  Swap OK! Received ${gained:.4f} USDC.e")
    print(f"\nUSDC.e balance: ${usdc_e_after:.6f}")
    print("Ready to trade on Polymarket!")


if __name__ == "__main__":
    main()
