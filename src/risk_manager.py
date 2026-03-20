"""
Risk Manager (v3) — Stats-Only, No Restrictions
-------------------------------------------------
Tracks P&L and trade statistics. No pausing, no limits, no blocking.
Kelly Criterion is the only risk governor — every profit reinvests.
"""

import logging
import time
from dataclasses import dataclass, field

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

    def pre_trade_check(self) -> tuple[bool, str]:
        """Always allows trading — no restrictions."""
        self._maybe_reset_day()
        return True, "OK"

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
            "paused": False,
            "pause_reason": "",
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
