# AGENTS.md — project conventions

## Project
- Name: Polymarket BTC 5-Minute Oracle-Verified Trading Bot
- Developer: James (Kim Consultant)
- Stack: Python 3.13 (Docker image), py-clob-client, web3, eth-account, requests, python-dotenv, FastAPI, uvicorn, websockets
- Repo: github.com/HamesInFlames/polymarket-btc-5m-bot
- Deploy: Railway, built from the Dockerfile (`railway.toml`), health check at `/api/health`

## Commands
- Dev: `python server.py` (dashboard server, also the Railway start command)
- Bot: `python bot.py` (owner runs this, see Do NOT)
- Build: none
- Test: none (`test_live.py` is a live connectivity script, not a test suite)
- Lint: none

## Code conventions
- Bot logic lives in `src/` (strategy, trader, risk, fees, price oracle, market reader)
- Entry points at the root: `bot.py`, `server.py`
- Helper scripts at the root: `setup_wallet.py`, `redeem_winnings.py`, `swap_usdc.py`
- Dashboard page: `templates/dashboard.html`
- Backtest and calculator scripts: `tools/`
- Settings come from environment variables (see `.env.example`)

## Workflow
- Read tasks/todo.md before starting work
- Update tasks/status.md after each session
- Log decisions in tasks/decisions.md

## Do NOT
- Change existing styling, copy or behaviour the task did not ask about
- Run the bot, place trades or run setup_wallet.py without the owner asking
- Install packages without documenting in tasks/decisions.md
- Modify deployment config without approval
- Delete .env files

## Lessons
<!-- added by /kc-workflow:reflect -->
