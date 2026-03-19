"""
Risk Manager
------------
Tracks P&L, enforces daily loss limits, consecutive-loss pauses,
and provides an emergency shutdown flag.
"""

import logging
import time
from dataclasses import dataclass, field

from src.config import MAX_DAILY_LOSS, MAX_CONSECUTIVE_LOSSES

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    timestamp: float
    direction: str
    token_id: str
    price: float
    size: float
    edge: float
    pnl: float = 0.0
    resolved: bool = False
    won: bool = False


@dataclass
class RiskManager:
    max_daily_loss: float = MAX_DAILY_LOSS
    max_consecutive_losses: int = MAX_CONSECUTIVE_LOSSES
    _trades: list = field(default_factory=list)
    _paused: bool = False
    _pause_reason: str = ""
    _day_start: float = field(default_factory=time.time)

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def pause_reason(self) -> str:
        return self._pause_reason

    def record_trade(self, trade: TradeRecord):
        self._trades.append(trade)
        log.info(
            "Trade recorded: %s %s @ $%.3f, size=$%.2f",
            trade.direction, trade.token_id[:12], trade.price, trade.size,
        )

    def record_result(self, trade_index: int, won: bool, pnl: float):
        if trade_index < len(self._trades):
            t = self._trades[trade_index]
            t.resolved = True
            t.won = won
            t.pnl = pnl
            log.info("Result: %s PnL=$%.4f", "WIN" if won else "LOSS", pnl)
        self._check_limits()

    def record_pnl(self, won: bool, pnl: float, direction: str = ""):
        rec = TradeRecord(
            timestamp=time.time(),
            direction=direction,
            token_id="",
            price=0,
            size=abs(pnl),
            edge=0,
            pnl=pnl,
            resolved=True,
            won=won,
        )
        self._trades.append(rec)
        log.info("Result: %s PnL=$%.4f", "WIN" if won else "LOSS", pnl)
        self._check_limits()

    def pre_trade_check(self) -> tuple[bool, str]:
        """Returns (allowed, reason). Call before placing any trade."""
        if self._paused:
            return False, f"Bot paused: {self._pause_reason}"

        self._maybe_reset_day()
        self._check_limits()

        if self._paused:
            return False, f"Bot paused: {self._pause_reason}"

        return True, "OK"

    def force_resume(self):
        self._paused = False
        self._pause_reason = ""
        log.info("Risk manager manually resumed")

    def daily_pnl(self) -> float:
        return sum(
            t.pnl for t in self._trades
            if t.resolved and t.timestamp >= self._day_start
        )

    def total_pnl(self) -> float:
        return sum(t.pnl for t in self._trades if t.resolved)

    def win_rate(self) -> float:
        resolved = [t for t in self._trades if t.resolved]
        if not resolved:
            return 0.0
        return sum(1 for t in resolved if t.won) / len(resolved)

    def total_trades(self) -> int:
        return len(self._trades)

    def consecutive_losses(self) -> int:
        count = 0
        for t in reversed(self._trades):
            if not t.resolved:
                continue
            if t.won:
                break
            count += 1
        return count

    def stats_summary(self) -> dict:
        resolved = [t for t in self._trades if t.resolved]
        wins = [t for t in resolved if t.won]
        losses = [t for t in resolved if not t.won]
        return {
            "total_trades": len(self._trades),
            "resolved": len(resolved),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": self.win_rate(),
            "daily_pnl": self.daily_pnl(),
            "total_pnl": self.total_pnl(),
            "consecutive_losses": self.consecutive_losses(),
            "paused": self._paused,
            "pause_reason": self._pause_reason,
        }

    def _check_limits(self):
        daily = self.daily_pnl()
        if daily < -self.max_daily_loss:
            self._paused = True
            self._pause_reason = (
                f"Daily loss limit hit: ${daily:.2f} "
                f"(max -${self.max_daily_loss:.2f})"
            )
            log.warning("RISK: %s", self._pause_reason)
            return

        consec = self.consecutive_losses()
        if consec >= self.max_consecutive_losses:
            self._paused = True
            self._pause_reason = (
                f"{consec} consecutive losses (max {self.max_consecutive_losses})"
            )
            log.warning("RISK: %s", self._pause_reason)
            return

    def _maybe_reset_day(self):
        now = time.time()
        if now - self._day_start > 86400:
            log.info(
                "Day reset — yesterday PnL: $%.4f, trades: %d",
                self.daily_pnl(), len([
                    t for t in self._trades
                    if t.timestamp >= self._day_start
                ]),
            )
            self._day_start = now
            if self._paused and "Daily loss" in self._pause_reason:
                self._paused = False
                self._pause_reason = ""
                log.info("Daily loss pause lifted on day reset")
