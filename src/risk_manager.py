"""
Risk Manager (v4) — Enforced Limits
-------------------------------------
Tracks P&L and trade statistics AND enforces daily loss limits and
consecutive loss limits. When a limit is hit, trading is paused until
conditions improve (new day for daily loss, a win for consecutive losses).

Kelly Criterion remains the primary sizing governor; this module acts
as a circuit breaker for catastrophic streaks.
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
    bet_dollars: float = 0.0
    kelly_fraction: float = 0.0
    win_prob: float = 0.0


@dataclass
class RiskManager:
    _trades: list = field(default_factory=list)
    _day_start: float = field(default_factory=time.time)
    _paused: bool = False
    _pause_reason: str = ""

    def record_trade(self, trade: TradeRecord):
        self._trades.append(trade)
        log.info(
            "Trade recorded: %s %s @ $%.3f, size=$%.2f, bet=$%.2f, kelly=%.4f",
            trade.direction, trade.token_id[:12], trade.price, trade.size,
            trade.bet_dollars, trade.kelly_fraction,
        )

    def record_result(self, trade_index: int, won: bool, pnl: float):
        if trade_index < len(self._trades):
            t = self._trades[trade_index]
            t.resolved = True
            t.won = won
            t.pnl = pnl
            log.info("Result: %s PnL=$%.4f", "WIN" if won else "LOSS", pnl)
        self._check_limits()

    def record_pnl(self, won: bool, pnl: float, direction: str = "",
                   bet_dollars: float = 0.0, kelly_fraction: float = 0.0,
                   win_prob: float = 0.0):
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
            bet_dollars=bet_dollars,
            kelly_fraction=kelly_fraction,
            win_prob=win_prob,
        )
        self._trades.append(rec)
        log.info("Result: %s PnL=$%.4f", "WIN" if won else "LOSS", pnl)
        self._check_limits()

    def pre_trade_check(self) -> tuple[bool, str]:
        """
        Returns (allowed, reason). If limits are breached, returns False
        with a human-readable explanation. Called before every trade attempt.
        """
        self._maybe_reset_day()
        self._check_limits()

        if self._paused:
            log.warning("Trading PAUSED: %s", self._pause_reason)
            return False, self._pause_reason

        return True, "OK"

    def _check_limits(self):
        """Evaluate risk limits and set/clear pause state."""
        daily = self.daily_pnl()
        consec = self.consecutive_losses()

        if MAX_DAILY_LOSS > 0 and daily <= -MAX_DAILY_LOSS:
            if not self._paused or "daily" not in self._pause_reason:
                log.warning(
                    "RISK LIMIT: Daily loss $%.2f exceeds max $%.2f — PAUSING",
                    abs(daily), MAX_DAILY_LOSS,
                )
            self._paused = True
            self._pause_reason = (
                f"Daily loss ${abs(daily):.2f} >= limit ${MAX_DAILY_LOSS:.2f}"
            )
            return

        if MAX_CONSECUTIVE_LOSSES > 0 and consec >= MAX_CONSECUTIVE_LOSSES:
            if not self._paused or "consecutive" not in self._pause_reason:
                log.warning(
                    "RISK LIMIT: %d consecutive losses >= max %d — PAUSING",
                    consec, MAX_CONSECUTIVE_LOSSES,
                )
            self._paused = True
            self._pause_reason = (
                f"{consec} consecutive losses >= limit {MAX_CONSECUTIVE_LOSSES}"
            )
            return

        if self._paused:
            log.info("Risk limits clear — RESUMING trading")
            self._paused = False
            self._pause_reason = ""

    @property
    def is_paused(self) -> bool:
        return self._paused

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

    def avg_edge(self) -> float:
        resolved = [t for t in self._trades if t.resolved and t.edge > 0]
        if not resolved:
            return 0.0
        return sum(t.edge for t in resolved) / len(resolved)

    def avg_kelly(self) -> float:
        resolved = [t for t in self._trades if t.resolved and t.kelly_fraction > 0]
        if not resolved:
            return 0.0
        return sum(t.kelly_fraction for t in resolved) / len(resolved)

    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self._trades if t.resolved and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self._trades if t.resolved and t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

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
            "profit_factor": self.profit_factor(),
            "avg_edge": self.avg_edge(),
            "avg_kelly": self.avg_kelly(),
        }

    def _maybe_reset_day(self):
        now = time.time()
        if now - self._day_start > 86400:
            log.info(
                "Day reset -- yesterday PnL: $%.4f, trades: %d",
                self.daily_pnl(), len([
                    t for t in self._trades
                    if t.timestamp >= self._day_start
                ]),
            )
            self._day_start = now
            if self._paused and "daily" in self._pause_reason.lower():
                log.info("New day — clearing daily loss pause")
                self._paused = False
                self._pause_reason = ""
