"""
Kelly Criterion Engine for Binary Polymarket Bets (v2)
======================================================
For a binary contract paying $1.00 on win:
  - Cost per contract: p  (the market price you pay)
  - Profit if win:     1 - p
  - Loss if lose:      p
  - Odds (net-to-1):   b = (1 - p) / p

Full Kelly fraction:  f* = (win_prob - price) / (1 - price)

NEW in v2:
  - Ruin-calibrated Kelly: auto-finds the largest fractional Kelly
    multiplier that keeps risk-of-ruin below a user-defined target.
  - Fee-aware Kelly: feeds the effective (post-fee) price into the
    Kelly formula so the edge and sizing are consistent.
  - Bankroll drawdown tracking with max-drawdown field.
  - Bet quality grade (A/B/C/D) based on edge, Kelly, and growth rate.
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

BANKROLL_FILE = Path(__file__).resolve().parent.parent / "data" / "bankroll.json"


@dataclass
class BetRecommendation:
    """Output of the Kelly engine for a single opportunity."""
    should_bet: bool
    kelly_fraction: float
    adj_kelly_fraction: float
    bet_dollars: float
    num_contracts: float
    expected_value: float
    expected_profit: float
    growth_rate: float
    win_prob: float
    contract_price: float
    edge: float
    bankroll: float
    reason: str
    ruin_prob: float = 0.0
    quality_grade: str = ""


@dataclass
class Bankroll:
    """Persistent bankroll tracker with drawdown analytics."""
    starting_balance: float
    current_balance: float
    peak_balance: float
    total_wagered: float = 0.0
    total_won: float = 0.0
    total_lost: float = 0.0
    num_bets: int = 0
    num_wins: int = 0
    max_drawdown: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def total_profit(self) -> float:
        return self.current_balance - self.starting_balance

    @property
    def roi(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return self.total_profit / self.starting_balance

    @property
    def drawdown(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance

    @property
    def win_rate(self) -> float:
        if self.num_bets == 0:
            return 0.0
        return self.num_wins / self.num_bets

    @property
    def avg_bet(self) -> float:
        if self.num_bets == 0:
            return 0.0
        return self.total_wagered / self.num_bets

    def record_win(self, wager: float, payout: float):
        profit = payout - wager
        self.current_balance += profit
        self.total_wagered += wager
        self.total_won += profit
        self.num_bets += 1
        self.num_wins += 1
        self.peak_balance = max(self.peak_balance, self.current_balance)
        self.max_drawdown = max(self.max_drawdown, self.drawdown)
        self.last_updated = time.time()

    def record_loss(self, wager: float):
        self.current_balance -= wager
        self.total_wagered += wager
        self.total_lost += wager
        self.num_bets += 1
        self.max_drawdown = max(self.max_drawdown, self.drawdown)
        self.last_updated = time.time()

    def to_dict(self) -> dict:
        return {
            "starting_balance": self.starting_balance,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "total_wagered": self.total_wagered,
            "total_won": self.total_won,
            "total_lost": self.total_lost,
            "num_bets": self.num_bets,
            "num_wins": self.num_wins,
            "max_drawdown": self.max_drawdown,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Bankroll":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_bankroll(starting_balance: float) -> Bankroll:
    try:
        if BANKROLL_FILE.exists():
            data = json.loads(BANKROLL_FILE.read_text())
            br = Bankroll.from_dict(data)
            log.info(
                "Loaded bankroll: $%.2f (started $%.2f, peak $%.2f, max_dd %.1f%%)",
                br.current_balance, br.starting_balance, br.peak_balance,
                br.max_drawdown * 100,
            )
            return br
    except Exception as e:
        log.warning("Could not load bankroll file: %s", e)

    br = Bankroll(
        starting_balance=starting_balance,
        current_balance=starting_balance,
        peak_balance=starting_balance,
    )
    save_bankroll(br)
    return br


def save_bankroll(br: Bankroll):
    try:
        BANKROLL_FILE.parent.mkdir(parents=True, exist_ok=True)
        BANKROLL_FILE.write_text(json.dumps(br.to_dict(), indent=2))
    except Exception as e:
        log.warning("Could not save bankroll: %s", e)


def kelly_criterion(
    win_prob: float,
    contract_price: float,
    bankroll: float,
    kelly_multiplier: float = 1.0,
    min_bet_dollars: float = 0.50,
    ruin_target: float = 0.05,
    ruin_level: float = 0.1,
) -> BetRecommendation:
    """
    Calculate the optimal bet for a binary Polymarket contract.

    Uses the full Kelly formula with optional ruin-calibrated cap:
    if the requested kelly_multiplier would produce a risk-of-ruin
    above ruin_target, the multiplier is reduced automatically.

    Args:
        win_prob:           Estimated probability of winning (0-1)
        contract_price:     Market price of the contract (0-1), should be
                            fee-adjusted (effective price) for consistency
        bankroll:           Current bankroll in dollars
        kelly_multiplier:   Maximum fraction of Kelly to use
        min_bet_dollars:    Minimum bet to bother placing
        ruin_target:        Maximum acceptable risk of ruin (0-1)
        ruin_level:         What fraction of bankroll counts as "ruin"

    Returns:
        BetRecommendation with all sizing details
    """
    no_bet = BetRecommendation(
        should_bet=False, kelly_fraction=0, adj_kelly_fraction=0,
        bet_dollars=0, num_contracts=0, expected_value=0,
        expected_profit=0, growth_rate=0, win_prob=win_prob,
        contract_price=contract_price, edge=0, bankroll=bankroll, reason="",
    )

    if contract_price <= 0.01 or contract_price >= 0.99:
        no_bet.reason = f"Price {contract_price:.3f} out of tradeable range"
        return no_bet

    if bankroll < min_bet_dollars:
        no_bet.reason = f"Bankroll ${bankroll:.2f} below minimum bet"
        return no_bet

    edge = win_prob - contract_price
    if edge <= 0:
        no_bet.reason = f"Negative edge: win_prob={win_prob:.3f} <= price={contract_price:.3f}"
        no_bet.edge = edge
        return no_bet

    kelly_f = edge / (1.0 - contract_price)

    # Ruin-calibrate: find largest multiplier <= kelly_multiplier
    # such that risk_of_ruin stays below ruin_target
    adj_mult = _calibrate_multiplier(
        win_prob, contract_price, kelly_f,
        max_mult=kelly_multiplier,
        ruin_target=ruin_target,
        ruin_level=ruin_level,
    )
    adj_f = kelly_f * adj_mult

    bet_dollars = bankroll * adj_f
    bet_dollars = max(0.0, bet_dollars)

    if bet_dollars < min_bet_dollars:
        no_bet.reason = (
            f"Bet ${bet_dollars:.2f} below minimum ${min_bet_dollars:.2f} "
            f"(kelly_f={kelly_f:.4f}, adj={adj_f:.4f}, mult={adj_mult:.3f})"
        )
        no_bet.edge = edge
        no_bet.kelly_fraction = kelly_f
        return no_bet

    num_contracts = bet_dollars / contract_price

    ev_per_dollar = edge / contract_price
    ev_absolute = edge * num_contracts

    g = expected_growth_rate(win_prob, contract_price, adj_f)

    ruin_p = risk_of_ruin(win_prob, contract_price, adj_f, ruin_level)
    grade = _bet_quality_grade(edge, g, ruin_p)

    return BetRecommendation(
        should_bet=True,
        kelly_fraction=kelly_f,
        adj_kelly_fraction=adj_f,
        bet_dollars=round(bet_dollars, 2),
        num_contracts=round(num_contracts, 2),
        expected_value=ev_per_dollar,
        expected_profit=round(ev_absolute, 4),
        growth_rate=g,
        win_prob=win_prob,
        contract_price=contract_price,
        edge=edge,
        bankroll=bankroll,
        ruin_prob=ruin_p,
        quality_grade=grade,
        reason=(
            f"Kelly={kelly_f:.4f} x{adj_mult:.3f} -> {adj_f:.4f} of ${bankroll:.2f} = "
            f"${bet_dollars:.2f} ({num_contracts:.1f} contracts) "
            f"EV=${ev_absolute:.4f} edge={edge:.4f} "
            f"ruin={ruin_p:.3f} grade={grade}"
        ),
    )


def _calibrate_multiplier(
    win_prob: float,
    price: float,
    kelly_f: float,
    max_mult: float,
    ruin_target: float,
    ruin_level: float,
) -> float:
    """
    Binary search for the largest multiplier in [0.01, max_mult] such that
    the resulting risk_of_ruin <= ruin_target.

    If even max_mult is safe, return max_mult unchanged.
    If no multiplier is safe, return the smallest tested value.
    """
    if kelly_f <= 0 or max_mult <= 0:
        return max_mult

    ruin_at_max = risk_of_ruin(win_prob, price, kelly_f * max_mult, ruin_level)
    if ruin_at_max <= ruin_target:
        return max_mult

    lo, hi = 0.01, max_mult
    for _ in range(20):
        mid = (lo + hi) / 2
        r = risk_of_ruin(win_prob, price, kelly_f * mid, ruin_level)
        if r <= ruin_target:
            lo = mid
        else:
            hi = mid

    return lo


def _bet_quality_grade(edge: float, growth_rate: float, ruin_prob: float) -> str:
    """
    A simple quality grade:
      A = strong edge, positive growth, low ruin
      B = decent edge, positive growth
      C = marginal edge
      D = barely tradeable
    """
    if edge >= 0.08 and growth_rate > 0.001 and ruin_prob < 0.03:
        return "A"
    if edge >= 0.05 and growth_rate > 0.0005 and ruin_prob < 0.10:
        return "B"
    if edge >= 0.03 and growth_rate > 0:
        return "C"
    return "D"


def expected_growth_rate(win_prob: float, price: float, fraction: float) -> float:
    """
    Compute the expected log-growth rate for a given bet fraction.
    Kelly maximizes this function.
    """
    if fraction <= 0 or price <= 0 or price >= 1:
        return 0.0
    if win_prob <= 0 or win_prob >= 1:
        return 0.0

    odds = (1.0 - price) / price
    win_term = win_prob * math.log(1 + fraction * odds)
    lose_term = (1 - win_prob) * math.log(max(1e-10, 1 - fraction))
    return win_term + lose_term


def risk_of_ruin(
    win_prob: float,
    price: float,
    bet_fraction: float,
    ruin_level: float = 0.1,
) -> float:
    """
    Probability of bankroll dropping to `ruin_level` fraction of current
    bankroll before doubling.

    Uses the gambler's ruin approximation for biased random walks on
    a multiplicative bankroll process:

      After a win  the bankroll is multiplied by (1 + f*b)
      After a loss the bankroll is multiplied by (1 - f)

    where f = bet_fraction and b = (1-price)/price (net odds).

    We convert to an additive random walk in log-space and apply
    the classical formula P(ruin before target) for a walk with
    drift mu and step sigma.
    """
    if bet_fraction <= 0 or win_prob <= 0 or price <= 0 or price >= 1:
        return 1.0

    b = (1.0 - price) / price
    win_mult = 1.0 + bet_fraction * b
    lose_mult = 1.0 - bet_fraction

    if lose_mult <= 0:
        return 1.0

    log_win = math.log(win_mult)
    log_lose = math.log(lose_mult)

    mu = win_prob * log_win + (1 - win_prob) * log_lose
    sigma2 = win_prob * log_win**2 + (1 - win_prob) * log_lose**2 - mu**2

    if mu <= 0:
        return 1.0
    if sigma2 <= 0:
        return 0.0

    log_ruin = math.log(ruin_level)
    log_target = math.log(2.0)

    # Classical formula: P(hit ruin before target) in additive walk
    # with drift mu, variance sigma^2 per step
    # P = (exp(-2*mu*T/sigma^2) - 1) / (exp(-2*mu*R/sigma^2) - 1)
    # where R = log(ruin_level), T = log(target)
    # Simplified via the Wald approximation
    lam = 2.0 * mu / sigma2

    try:
        exp_target = math.exp(-lam * log_target)
        exp_ruin = math.exp(-lam * log_ruin)
        p_ruin = (1.0 - exp_target) / (exp_ruin - exp_target)
        return max(0.0, min(1.0, p_ruin))
    except (OverflowError, ZeroDivisionError):
        return 0.01 if mu > 0 else 0.5


def optimal_kelly_for_ruin(
    win_prob: float,
    price: float,
    max_ruin: float = 0.05,
    ruin_level: float = 0.1,
) -> float:
    """
    Find the largest fractional Kelly multiplier (0 to 1) such that
    risk_of_ruin stays below max_ruin.

    Useful for one-off calibration or display purposes.
    """
    kelly_f = (win_prob - price) / (1.0 - price)
    if kelly_f <= 0:
        return 0.0

    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        r = risk_of_ruin(win_prob, price, kelly_f * mid, ruin_level)
        if r <= max_ruin:
            lo = mid
        else:
            hi = mid
    return lo
