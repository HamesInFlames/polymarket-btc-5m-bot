"""
Shared bot state — thread-safe store that the trading loop writes to
and the web dashboard reads from.
"""

import logging
import threading
import time
from dataclasses import dataclass, field


@dataclass
class TradeEntry:
    timestamp: float
    direction: str
    action: str
    price: float
    size: float
    edge: float
    confidence: float
    pnl: float
    won: bool
    btc_price: float
    condition_id: str
    reason: str


class BotState:
    _lock = threading.Lock()

    def __init__(self):
        self.started_at: float = 0.0
        self.mode: str = "DRY RUN"
        self.running: bool = False
        self.cycle: int = 0
        self.trade_count: int = 0
        self.max_trades: int = 0

        self.btc_price: float = 0.0
        self.btc_sources: dict = {}
        self.btc_updated_at: float = 0.0

        self.active_rounds: list[dict] = []

        self.trades: list[TradeEntry] = []

        self.total_pnl: float = 0.0
        self.daily_pnl: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.win_rate: float = 0.0
        self.consecutive_losses: int = 0
        self.paused: bool = False
        self.pause_reason: str = ""

        self.last_error: str = ""
        self.last_error_at: float = 0.0
        self.log_lines: list[dict] = []

    def update_bot_status(self, running: bool, cycle: int, trade_count: int,
                          max_trades: int, mode: str):
        with self._lock:
            if not self.started_at and running:
                self.started_at = time.time()
            self.running = running
            self.cycle = cycle
            self.trade_count = trade_count
            self.max_trades = max_trades
            self.mode = mode

    def update_btc_price(self, oracle: dict):
        with self._lock:
            median = oracle.get("median")
            if median:
                self.btc_price = median
            self.btc_sources = oracle.get("sources", {})
            self.btc_updated_at = time.time()

    def update_rounds(self, rounds: list):
        with self._lock:
            self.active_rounds = [
                {
                    "slug": r.event_slug,
                    "condition_id": r.condition_id[:16],
                    "question": r.question,
                    "seconds_remaining": r.seconds_remaining,
                    "end_timestamp": r.end_timestamp,
                    "up_price": r.up_price,
                    "down_price": r.down_price,
                }
                for r in rounds
            ]

    def clear_rounds(self):
        with self._lock:
            self.active_rounds = []

    def add_trade(self, entry: TradeEntry):
        with self._lock:
            self.trades.append(entry)
            if len(self.trades) > 500:
                self.trades = self.trades[-500:]

    def update_risk_stats(self, stats: dict):
        with self._lock:
            self.total_pnl = stats.get("total_pnl", 0.0)
            self.daily_pnl = stats.get("daily_pnl", 0.0)
            self.wins = stats.get("wins", 0)
            self.losses = stats.get("losses", 0)
            self.win_rate = stats.get("win_rate", 0.0)
            self.consecutive_losses = stats.get("consecutive_losses", 0)
            self.paused = stats.get("paused", False)
            self.pause_reason = stats.get("pause_reason", "")

    def set_error(self, error: str):
        with self._lock:
            self.last_error = error
            self.last_error_at = time.time()

    def add_log(self, level: str, message: str):
        with self._lock:
            self.log_lines.append({
                "ts": time.time(),
                "level": level,
                "msg": message,
            })
            if len(self.log_lines) > 200:
                self.log_lines = self.log_lines[-200:]

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            uptime = now - self.started_at if self.started_at else 0
            return {
                "bot": {
                    "running": self.running,
                    "mode": self.mode,
                    "cycle": self.cycle,
                    "trade_count": self.trade_count,
                    "max_trades": self.max_trades,
                    "uptime_seconds": uptime,
                    "started_at": self.started_at,
                },
                "btc": {
                    "price": self.btc_price,
                    "sources": self.btc_sources,
                    "updated_at": self.btc_updated_at,
                    "age_seconds": now - self.btc_updated_at if self.btc_updated_at else -1,
                },
                "rounds": self.active_rounds,
                "stats": {
                    "total_pnl": self.total_pnl,
                    "daily_pnl": self.daily_pnl,
                    "wins": self.wins,
                    "losses": self.losses,
                    "win_rate": self.win_rate,
                    "consecutive_losses": self.consecutive_losses,
                    "paused": self.paused,
                    "pause_reason": self.pause_reason,
                },
                "trades": [
                    {
                        "timestamp": t.timestamp,
                        "direction": t.direction,
                        "action": t.action,
                        "price": t.price,
                        "size": t.size,
                        "edge": t.edge,
                        "confidence": t.confidence,
                        "pnl": t.pnl,
                        "won": t.won,
                        "btc_price": t.btc_price,
                        "condition_id": t.condition_id,
                        "reason": t.reason,
                    }
                    for t in self.trades[-50:]
                ],
                "logs": self.log_lines[-80:],
                "error": {
                    "message": self.last_error,
                    "at": self.last_error_at,
                },
            }


state = BotState()


class DashboardLogHandler(logging.Handler):
    """Logging handler that mirrors log records into BotState for the dashboard."""

    _LEVEL_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARN",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "ERROR",
    }

    _NOISY_LOGGERS = frozenset({"urllib3", "web3", "httpx", "uvicorn", "asyncio"})

    def __init__(self, bot_state: BotState):
        super().__init__(level=logging.INFO)
        self._state = bot_state

    def emit(self, record: logging.LogRecord):
        if any(record.name.startswith(n) for n in self._NOISY_LOGGERS):
            return
        level = self._LEVEL_MAP.get(record.levelno, "INFO")
        try:
            msg = self.format(record) if self.formatter else record.getMessage()
            self._state.add_log(level, msg)
            if record.levelno >= logging.ERROR:
                self._state.set_error(msg)
        except Exception:
            pass


def install_log_handler():
    """Call once at startup to pipe all log output into the dashboard state."""
    handler = DashboardLogHandler(state)
    handler.setFormatter(logging.Formatter("%(name)-16s  %(message)s"))
    logging.getLogger().addHandler(handler)
