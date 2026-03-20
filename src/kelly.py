"""
Kelly Criterion Engine for Binary Polymarket Bets
==================================================
For a binary contract paying $1.00 on win:
  - Cost per contract: p  (the market price you pay)
  - Profit if win:     1 - p
  - Loss if lose:      p
  - Odds (net-to-1):   b = (1 - p) / p

Full Kelly fraction:  f* = (win_prob - price) / (1 - price)

This is the GROWTH-RATE OPTIMAL bet size as a fraction of bankroll.
We apply fractional Kelly (configurable) to reduce variance.
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
    kelly_fraction: float       # raw Kelly fraction of bankroll
    adj_kelly_fraction: float   # after fractional Kelly multiplier
    bet_dollars: float          # actual dollars to wager
    num_contracts: float        # bet_dollars / contract_price
    expected_value: float       # EV per dollar risked
    expected_profit: float      # EV in absolute dollars
    growth_rate: float          # expected log-growth of bankroll
    win_prob: float
    contract_price: float
    edge: float                 # win_prob - contract_price
    bankroll: float
    reason: str


@dataclass
class Bankroll:
    """Persistent bankroll tracker."""
    starting_balance: float
    current_balance: float
    peak_balance: float
    total_wagered: float = 0.0
    total_won: float = 0.0
    total_lost: float = 0.0
    num_bets: int = 0
    num_wins: int = 0
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

    def record_win(self, wager: float, payout: float):
        profit = payout - wager
        self.current_balance += profit
        self.total_wagered += wager
        self.total_won += profit
        self.num_bets += 1
        self.num_wins += 1
        self.peak_balance = max(self.peak_balance, self.current_balance)
        self.last_updated = time.time()

    def record_loss(self, wager: float):
        self.current_balance -= wager
        self.total_wagered += wager
        self.total_lost += wager
        self.num_bets += 1
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
                "Loaded bankroll: $%.2f (started $%.2f, peak $%.2f)",
                br.current_balance, br.starting_balance, br.peak_balance,
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
) -> BetRecommendation:
    """
    Calculate the optimal bet for a binary Polymarket contract.
    UNRESTRICTED — Kelly decides the fraction, entire bankroll compounds.

    Args:
        win_prob:           Estimated probability of winning (0-1)
        contract_price:     Market price of the contract (0-1)
        bankroll:           Current bankroll in dollars (all profits reinvested)
        kelly_multiplier:   Fraction of Kelly to use (1.0 = full Kelly)
        min_bet_dollars:    Minimum bet to bother placing

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

    # f* = (win_prob - price) / (1 - price)  — the growth-rate optimal fraction
    kelly_f = edge / (1.0 - contract_price)

    adj_f = kelly_f * kelly_multiplier

    # No caps — Kelly fraction is the ONLY governor
    bet_dollars = bankroll * adj_f
    bet_dollars = max(0.0, bet_dollars)

    if bet_dollars < min_bet_dollars:
        no_bet.reason = (
            f"Bet ${bet_dollars:.2f} below minimum ${min_bet_dollars:.2f} "
            f"(kelly_f={kelly_f:.4f}, adj={adj_f:.4f})"
        )
        no_bet.edge = edge
        no_bet.kelly_fraction = kelly_f
        return no_bet

    num_contracts = bet_dollars / contract_price

    ev_per_dollar = edge / contract_price
    ev_absolute = edge * num_contracts

    if win_prob > 0 and win_prob < 1:
        g = (win_prob * math.log(1 + adj_f * (1 - contract_price) / contract_price)
             + (1 - win_prob) * math.log(max(1e-10, 1 - adj_f)))
    else:
        g = 0.0

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
        reason=(
            f"Kelly={kelly_f:.4f} x{kelly_multiplier} -> {adj_f:.4f} of ${bankroll:.2f} = "
            f"${bet_dollars:.2f} ({num_contracts:.1f} contracts) "
            f"EV=${ev_absolute:.4f} edge={edge:.4f}"
        ),
    )


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


def risk_of_ruin(win_prob: float, price: float, bet_fraction: float, ruin_level: float = 0.1) -> float:
    """
    Approximate probability of bankroll dropping to `ruin_level` fraction
    of current bankroll before doubling.

    Uses the gambler's ruin approximation for biased random walks.
    """
    if win_prob <= 0.5 or bet_fraction <= 0:
        return 1.0

    odds = (1.0 - price) / price
    q = 1 - win_prob
    ratio = q / (win_prob * odds) if (win_prob * odds) > 0 else 1.0

    if abs(ratio - 1.0) < 1e-6:
        return 0.5

    n_steps_to_ruin = -math.log(ruin_level) / bet_fraction
    n_steps_to_double = math.log(2) / bet_fraction

    try:
        numerator = 1 - ratio ** n_steps_to_double
        denominator = ratio ** (-n_steps_to_ruin) - ratio ** n_steps_to_double
        return max(0.0, min(1.0, numerator / denominator)) if denominator != 0 else 0.5
    except (OverflowError, ZeroDivisionError):
        return 0.01 if win_prob > 0.6 else 0.5
