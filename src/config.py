"""
Central configuration — loads from .env and provides typed defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# ── Wallet / RPC ──────────────────────────────────────────────
PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
POLYGON_RPC_URL: str = os.getenv("POLYGON_RPC_URL", "https://polygon.drpc.org")
CHAIN_ID: int = 137
CLOB_HOST: str = "https://clob.polymarket.com"
GAMMA_HOST: str = "https://gamma-api.polymarket.com"

# ── Wallet ────────────────────────────────────────────────────
WALLET_ADDRESS: str = os.getenv("WALLET_ADDRESS", "0x96418edEc291543e0e7Cb2Ef90aEEb050B2E05d1")

# ── Contract addresses (Polygon) ─────────────────────────────
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# Chainlink BTC/USD aggregator on Polygon
CHAINLINK_BTC_USD = "0xc907E116054Ad103354f2D350FD2514433D57F6f"

# ── Relayer / Redemption ─────────────────────────────────────
RELAYER_API_KEY: str = os.getenv("RELAYER_API_KEY", "")
RELAYER_API_KEY_ADDRESS: str = os.getenv("RELAYER_API_KEY_ADDRESS", "")

# ── Trading parameters ────────────────────────────────────────
MIN_EDGE: float = _float("MIN_EDGE", 0.05)
ENTRY_WINDOW_START: int = _int("ENTRY_WINDOW_START", 180)
ENTRY_WINDOW_END: int = _int("ENTRY_WINDOW_END", 5)
MIN_CONTRACT_PRICE: float = _float("MIN_CONTRACT_PRICE", 0.50)
MAX_CONTRACT_PRICE: float = _float("MAX_CONTRACT_PRICE", 0.95)
MAX_TRADE_SIZE: float = _float("MAX_TRADE_SIZE", 10.0)
LIVE_TRADING: bool = _bool("LIVE_TRADING", False)

MAX_DAILY_LOSS: float = _float("MAX_DAILY_LOSS", 20.0)
MAX_CONSECUTIVE_LOSSES: int = _int("MAX_CONSECUTIVE_LOSSES", 5)

# ── Kelly / Bankroll parameters ───────────────────────────────
STARTING_BANKROLL: float = _float("STARTING_BANKROLL", 33.69)
KELLY_MULTIPLIER: float = _float("KELLY_MULTIPLIER", 0.50)
MIN_BET_DOLLARS: float = _float("MIN_BET_DOLLARS", 2.50)
VOLATILITY_LOOKBACK: int = _int("VOLATILITY_LOOKBACK", 20)
CONFIDENCE_FLOOR: float = _float("CONFIDENCE_FLOOR", 0.55)

# ── External price APIs (fallbacks) ──────────────────────────
BINANCE_BTC_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
COINGECKO_BTC_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
COINBASE_BTC_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
