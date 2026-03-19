# Polymarket BTC 5-Minute Oracle-Verified Trading Bot

> **DISCLAIMER:** This software is for **educational and research purposes only**.
> It is **not financial advice**. Algorithmic trading carries **substantial risk of loss**.
> Over 90% of back-tested strategies fail in live markets. Always consult a qualified
> financial advisor before risking real capital.

## What It Does

This bot trades Polymarket's **"Bitcoin Up or Down — 5 Minutes"** binary prediction markets.
It uses an **oracle-verified directional strategy**: in the final seconds of each 5-minute round,
it compares the real BTC price (from Chainlink on-chain oracle + Binance/Coinbase/CoinGecko) to the
round's opening price. If BTC has moved decisively in one direction and the CLOB contract is still
underpriced, the bot buys the likely-winning outcome for a profit when the market resolves at $1.00.

### Strategy Edge

| Factor | How It Helps |
|--------|-------------|
| **Chainlink oracle** | Same price source Polymarket uses for resolution — reduces settlement risk |
| **Multi-source verification** | Cross-checks 4 independent BTC price feeds to avoid stale/bad data |
| **Smart entry (3 min to 5s before close)** | Enters early enough that contracts are still ~$0.50-$0.70, not $1.00 |
| **Dynamic confidence scoring** | Only trades when price move + time decay + oracle agreement are strong |
| **Quarter-Kelly position sizing** | Mathematically sized bets — aggressive enough to compound, conservative enough to survive losing streaks |

### Risk Controls

- Daily loss limit (default: $20) — bot pauses automatically
- Consecutive loss limit (default: 5) — bot pauses for cooldown
- Minimum edge threshold (default: 5%) — only trades with positive expected value
- GTC limit orders at smart price levels — sits on the book and fills when the market moves to us
- Dry-run mode by default — no real money until you explicitly enable it

---

## Quick Start (Step by Step)

### 1. Install Python

You need **Python 3.9 or higher**. Check with:

```
python --version
```

If you don't have it, download from https://www.python.org/downloads/  
**Important:** During install, check the box that says "Add Python to PATH".

### 2. Clone or Download This Project

Put the `polymarket_bot` folder somewhere on your computer (e.g. `C:\Users\YourName\polymarket_bot`).

### 3. Install Dependencies

Open a terminal (PowerShell or Command Prompt) in the project folder and run:

```
cd C:\Users\JamesPc\polymarket_bot
pip install -r requirements.txt
```

### 4. Generate a Wallet

```
python setup_wallet.py
```

This creates a brand-new Polygon wallet and saves the private key to `.env`.
It will print your wallet address — you'll need to fund it next.

### 5. Fund Your Wallet

Send the following to your wallet address on the **Polygon network**:

| Token | Amount | Purpose |
|-------|--------|---------|
| **USDC.e** (PoS bridged USDC) | $10–$50 to start | Trading capital |
| **POL** (formerly MATIC) | 0.5–1.0 POL | Gas fees for approvals |

You can buy these on any exchange (Coinbase, Binance, etc.) and withdraw to Polygon,
or use a bridge like https://portal.polygon.technology/bridge.

**USDC.e contract on Polygon:** `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`

### 6. Run On-Chain Approvals

This is a one-time step that gives the Polymarket exchange contracts permission to
use your USDC and trade outcome tokens:

```
python setup_wallet.py --approve
```

It will send 5 small approval transactions (costs a fraction of a cent in POL gas).

### 7. Start the Bot (Dry Run)

```
python bot.py
```

By default the bot runs in **dry-run mode** — it finds markets, evaluates signals,
and logs what it *would* trade, but doesn't place real orders. Watch the logs to see
it working.

### 8. Go Live

When you're satisfied with the dry-run behavior, edit `.env` and change:

```
LIVE_TRADING=true
```

Then restart: `python bot.py`

---

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIVATE_KEY` | (generated) | Polygon wallet private key |
| `POLYGON_RPC_URL` | `https://polygon-rpc.com` | Polygon JSON-RPC endpoint |
| `MAX_TRADE_SIZE` | `2.0` | Max USDC per trade |
| `MIN_EDGE` | `0.05` | Minimum edge (5%) to trigger a trade |
| `MAX_DAILY_LOSS` | `20.0` | Bot pauses after this much daily loss |
| `MAX_CONSECUTIVE_LOSSES` | `5` | Bot pauses after N losses in a row |
| `ENTRY_WINDOW_START` | `25` | Start looking for trades this many seconds before close |
| `ENTRY_WINDOW_END` | `3` | Stop entering this many seconds before close |
| `MIN_CONTRACT_PRICE` | `0.50` | Only buy contracts priced above this |
| `MAX_CONTRACT_PRICE` | `0.95` | Only buy contracts priced below this |
| `LIVE_TRADING` | `false` | Set `true` to place real orders |

---

## Project Structure

```
polymarket_bot/
├── bot.py                  # Main entry point — run this
├── setup_wallet.py         # Wallet generation + on-chain approvals
├── requirements.txt        # Python dependencies
├── .env                    # Your config (created by setup_wallet.py)
├── .env.example            # Template config with all options
└── src/
    ├── config.py           # Loads .env and provides typed constants
    ├── price_oracle.py     # Multi-source BTC price (Chainlink + CEX APIs)
    ├── market_reader.py    # Discovers active Polymarket 5-min BTC rounds
    ├── strategy.py         # Oracle-verified directional signal logic
    ├── risk_manager.py     # PnL tracking, loss limits, pause logic
    └── trader.py           # CLOB order placement via py-clob-client
```

---

## How the Strategy Works

```
Every ~1-2 seconds:
  |
  +-- Discover active 5-min BTC rounds (Gamma API)
  |
  +-- For rounds that just started (0-60s elapsed):
  |     +-- Capture opening BTC price from Chainlink oracle
  |
  +-- For rounds 180s -> 5s from close:
  |     +-- Fetch current BTC price from 4 sources
  |     +-- Compare to opening price -> determine direction (Up/Down)
  |     +-- Calculate confidence score (move size + time + oracle agreement)
  |     +-- Fetch CLOB order book for the likely-winning contract
  |     +-- Determine smart limit price (at or below ask, above bid)
  |     +-- Calculate edge = (1.0 - limit_price) x confidence
  |     +-- If edge > MIN_EDGE and price in [0.50, 0.95]:
  |     |     +-- Size position (quarter-Kelly)
  |     |     +-- Place GTC limit BUY order
  |     +-- Otherwise: skip this round
  |
  +-- Risk manager checks:
        +-- Daily loss limit
        +-- Consecutive loss limit
        +-- Pause or shutdown if triggered
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `PRIVATE_KEY not set` | Run `python setup_wallet.py` first |
| `insufficient balance` | Fund your wallet with more USDC.e on Polygon |
| `insufficient allowance` | Run `python setup_wallet.py --approve` |
| `No active BTC 5-min markets` | Markets may be paused; check polymarket.com |
| `Chainlink price is Xs stale` | Normal during low-activity periods; bot uses CEX fallbacks |
| `Risk check blocked` | Bot auto-paused; restart to reset, or wait for day rollover |

---

## Caveats

- **Hypothetical results are not indicative of future performance.** Dry-run PnL estimates use
  the strategy's edge calculation, not actual market outcomes.
- **Execution risk:** FOK orders may fail to fill if liquidity is thin or prices move.
- **Oracle risk:** If Chainlink reports a stale or incorrect price, the bot may trade the wrong direction.
- **Regulatory risk:** Prediction markets face varying legal status by jurisdiction.
  Ensure you comply with your local laws.
- **This is not financial advice.** The author accepts no liability for trading losses.
