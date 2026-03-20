"""
Profit Target Calculator
========================
Answers: "How much do I need to start with to hit $X in Y days?"

Simulates the Kelly-optimal compounding with realistic constraints:
  - Bet size capped at MAX_BET_FRACTION of bankroll AND absolute MAX_TRADE_SIZE
  - Accounts for the transition from exponential growth (small bankroll)
    to linear growth (when bets hit the dollar cap)
  - Shows multiple scenarios for different win rates and trade frequencies
  - Includes Monte Carlo simulation for realistic variance

Usage:
    python tools/profit_calculator.py
    python tools/profit_calculator.py --target 100000 --days 30
    python tools/profit_calculator.py --target 100000 --days 30 --start 5000
"""

import argparse
import math
import random
import statistics
import sys


def kelly_fraction(win_prob: float, contract_price: float, multiplier: float = 0.5) -> float:
    """Half-Kelly fraction for a binary bet."""
    edge = win_prob - contract_price
    if edge <= 0 or contract_price >= 1.0:
        return 0.0
    return multiplier * edge / (1.0 - contract_price)


def simulate_path(
    start: float,
    win_prob: float,
    avg_price: float,
    trades_per_day: int,
    days: int,
    kelly_mult: float = 0.5,
    max_bet_frac: float = 0.10,
    max_bet_abs: float = 50.0,
    seed: int | None = None,
) -> list[float]:
    """
    Monte Carlo: simulate one path of bankroll growth.
    Returns daily bankroll snapshots.
    """
    rng = random.Random(seed)
    bankroll = start
    daily_snapshots = [bankroll]

    kf = kelly_fraction(win_prob, avg_price, kelly_mult)
    if kf <= 0:
        return [start] * (days + 1)

    for day in range(days):
        for _ in range(trades_per_day):
            bet_frac = min(kf, max_bet_frac)
            bet = bankroll * bet_frac
            bet = min(bet, max_bet_abs)
            bet = max(0.01, bet)

            if bet > bankroll * 0.95:
                break

            contracts = bet / avg_price
            if rng.random() < win_prob:
                profit = contracts * (1.0 - avg_price)
                bankroll += profit
            else:
                bankroll -= bet

            if bankroll < 0.50:
                bankroll = 0.0
                break

        daily_snapshots.append(bankroll)
        if bankroll <= 0:
            daily_snapshots.extend([0.0] * (days - day - 1))
            break

    return daily_snapshots


def theoretical_growth(
    start: float,
    win_prob: float,
    avg_price: float,
    trades_per_day: int,
    days: int,
    kelly_mult: float = 0.5,
    max_bet_frac: float = 0.10,
    max_bet_abs: float = 50.0,
) -> list[float]:
    """
    Deterministic expected-value path (no randomness).
    Useful for understanding the theoretical trajectory.
    """
    kf = kelly_fraction(win_prob, avg_price, kelly_mult)
    if kf <= 0:
        return [start] * (days + 1)

    bankroll = start
    daily = [bankroll]

    for day in range(days):
        for _ in range(trades_per_day):
            bet_frac = min(kf, max_bet_frac)
            bet = bankroll * bet_frac
            bet = min(bet, max_bet_abs)

            contracts = bet / avg_price
            ev = win_prob * contracts * (1.0 - avg_price) - (1.0 - win_prob) * bet
            bankroll += ev

        daily.append(bankroll)

    return daily


def find_starting_bankroll(
    target: float,
    days: int,
    win_prob: float,
    avg_price: float,
    trades_per_day: int,
    kelly_mult: float = 0.5,
    max_bet_frac: float = 0.10,
    max_bet_abs: float = 50.0,
) -> float:
    """Binary search for the starting bankroll needed to hit target."""
    lo, hi = 1.0, target * 2

    for _ in range(100):
        mid = (lo + hi) / 2
        path = theoretical_growth(
            mid, win_prob, avg_price, trades_per_day, days,
            kelly_mult, max_bet_frac, max_bet_abs,
        )
        if path[-1] >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.01:
            break

    return hi


def monte_carlo_analysis(
    start: float,
    target: float,
    win_prob: float,
    avg_price: float,
    trades_per_day: int,
    days: int,
    kelly_mult: float = 0.5,
    max_bet_frac: float = 0.10,
    max_bet_abs: float = 50.0,
    n_sims: int = 5000,
) -> dict:
    """Run Monte Carlo and return statistics."""
    finals = []
    hit_target = 0
    went_bust = 0

    for i in range(n_sims):
        path = simulate_path(
            start, win_prob, avg_price, trades_per_day, days,
            kelly_mult, max_bet_frac, max_bet_abs, seed=i,
        )
        final = path[-1]
        finals.append(final)
        if final >= target:
            hit_target += 1
        if final <= 0.50:
            went_bust += 1

    finals.sort()
    return {
        "median": finals[len(finals) // 2],
        "mean": statistics.mean(finals),
        "p10": finals[int(len(finals) * 0.10)],
        "p25": finals[int(len(finals) * 0.25)],
        "p75": finals[int(len(finals) * 0.75)],
        "p90": finals[int(len(finals) * 0.90)],
        "min": finals[0],
        "max": finals[-1],
        "hit_target_pct": hit_target / n_sims * 100,
        "bust_pct": went_bust / n_sims * 100,
    }


SCENARIOS = [
    {
        "name": "Conservative",
        "win_prob": 0.57,
        "avg_price": 0.52,
        "trades_per_day": 15,
        "desc": "57% win rate, $0.52 avg price, 15 trades/day",
    },
    {
        "name": "Moderate",
        "win_prob": 0.60,
        "avg_price": 0.53,
        "trades_per_day": 25,
        "desc": "60% win rate, $0.53 avg price, 25 trades/day",
    },
    {
        "name": "Aggressive",
        "win_prob": 0.63,
        "avg_price": 0.52,
        "trades_per_day": 35,
        "desc": "63% win rate, $0.52 avg price, 35 trades/day",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Profit Target Calculator")
    parser.add_argument("--target", type=float, default=100_000, help="Dollar target (default: 100000)")
    parser.add_argument("--days", type=int, default=30, help="Time horizon in days (default: 30)")
    parser.add_argument("--start", type=float, default=0, help="If set, analyze this specific starting amount")
    parser.add_argument("--max-bet", type=float, default=50, help="Max bet in dollars (default: 50)")
    parser.add_argument("--kelly", type=float, default=0.5, help="Kelly multiplier (default: 0.5 = half-Kelly)")
    parser.add_argument("--max-frac", type=float, default=0.10, help="Max bet as fraction of bankroll (default: 0.10)")
    parser.add_argument("--sims", type=int, default=5000, help="Monte Carlo simulations (default: 5000)")
    args = parser.parse_args()

    target = args.target
    days = args.days

    print()
    print("=" * 72)
    print(f"  PROFIT TARGET CALCULATOR — ${target:,.0f} in {days} days")
    print("=" * 72)
    print(f"  Kelly multiplier:   {args.kelly:.0%}")
    print(f"  Max bet fraction:   {args.max_frac:.0%} of bankroll")
    print(f"  Max bet cap:        ${args.max_bet:.0f}")
    print(f"  Monte Carlo sims:   {args.sims:,}")
    print("=" * 72)

    # ── Phase 1: How much do you need to start? ────────────────
    print()
    print("  REQUIRED STARTING CAPITAL (theoretical, no variance)")
    print("  " + "-" * 56)

    for sc in SCENARIOS:
        needed = find_starting_bankroll(
            target, days, sc["win_prob"], sc["avg_price"],
            sc["trades_per_day"], args.kelly, args.max_frac, args.max_bet,
        )

        kf = kelly_fraction(sc["win_prob"], sc["avg_price"], args.kelly)
        edge = sc["win_prob"] - sc["avg_price"]
        ev_per_bet = edge / sc["avg_price"]

        cap_bankroll = args.max_bet / min(kf, args.max_frac) if min(kf, args.max_frac) > 0 else float("inf")

        print()
        print(f"  [{sc['name'].upper()}] {sc['desc']}")
        print(f"    Edge per trade:     {edge*100:.1f}%")
        print(f"    Kelly fraction:     {kf*100:.2f}%")
        print(f"    EV per $1 risked:   ${ev_per_bet:.4f}")
        print(f"    Bet cap kicks in at: ${cap_bankroll:,.0f} bankroll")

        ev_per_trade_at_cap = args.max_bet * ev_per_bet
        daily_ev_at_cap = ev_per_trade_at_cap * sc["trades_per_day"]

        print(f"    EV per trade (at cap): ${ev_per_trade_at_cap:.2f}")
        print(f"    Daily EV (at cap):  ${daily_ev_at_cap:.2f}")
        print(f"    -----------------------------------------")
        print(f"    >>> START WITH:     ${needed:,.2f}")

        days_linear = (target - cap_bankroll) / daily_ev_at_cap if daily_ev_at_cap > 0 and target > cap_bankroll else 0
        if days_linear > days:
            print(f"    WARNING: With ${args.max_bet} bet cap, linear phase alone needs ~{days_linear:.0f} days")
            print(f"      -> Raise MAX_TRADE_SIZE to ${target / (days * sc['trades_per_day'] * ev_per_bet):.0f} to make it feasible")

    # ── Phase 2: Monte Carlo on a specific starting amount ─────
    if args.start > 0:
        analyze_amount = args.start
    else:
        moderate = SCENARIOS[1]
        analyze_amount = find_starting_bankroll(
            target, days, moderate["win_prob"], moderate["avg_price"],
            moderate["trades_per_day"], args.kelly, args.max_frac, args.max_bet,
        )

    print()
    print("=" * 72)
    print(f"  MONTE CARLO: Starting with ${analyze_amount:,.2f} -> target ${target:,.0f}")
    print("=" * 72)

    for sc in SCENARIOS:
        mc = monte_carlo_analysis(
            analyze_amount, target, sc["win_prob"], sc["avg_price"],
            sc["trades_per_day"], days, args.kelly, args.max_frac,
            args.max_bet, args.sims,
        )
        print()
        print(f"  [{sc['name'].upper()}]")
        print(f"    Median final:       ${mc['median']:,.2f}")
        print(f"    Mean final:         ${mc['mean']:,.2f}")
        print(f"    10th percentile:    ${mc['p10']:,.2f}")
        print(f"    25th percentile:    ${mc['p25']:,.2f}")
        print(f"    75th percentile:    ${mc['p75']:,.2f}")
        print(f"    90th percentile:    ${mc['p90']:,.2f}")
        print(f"    Best case:          ${mc['max']:,.2f}")
        print(f"    Worst case:         ${mc['min']:,.2f}")
        print(f"    Hit ${target:,.0f}:    {mc['hit_target_pct']:.1f}%")
        print(f"    Went bust:          {mc['bust_pct']:.1f}%")

    # ── Phase 3: What MAX_TRADE_SIZE do you actually need? ─────
    print()
    print("=" * 72)
    print(f"  RECOMMENDED MAX_TRADE_SIZE for ${target:,.0f} in {days} days")
    print("=" * 72)

    for start_amt in [500, 1000, 2500, 5000, 10000]:
        sc = SCENARIOS[1]  # moderate
        for max_b in [50, 100, 250, 500, 1000, 2000, 5000]:
            path = theoretical_growth(
                start_amt, sc["win_prob"], sc["avg_price"],
                sc["trades_per_day"], days, args.kelly, args.max_frac, max_b,
            )
            if path[-1] >= target:
                print(f"  Start ${start_amt:>6,} -> MAX_TRADE_SIZE=${max_b:>5,} -> ${path[-1]:>12,.2f} [OK]")
                break
        else:
            path = theoretical_growth(
                start_amt, sc["win_prob"], sc["avg_price"],
                sc["trades_per_day"], days, args.kelly, args.max_frac, 5000,
            )
            print(f"  Start ${start_amt:>6,} → even $5,000 bets only reach ${path[-1]:>12,.2f}")

    print()
    print("=" * 72)
    print("  IMPORTANT CAVEATS")
    print("=" * 72)
    print("  * These are MATHEMATICAL PROJECTIONS, not guarantees")
    print("  * Real win rates depend on market conditions and liquidity")
    print("  * Polymarket order books may not have enough depth for large bets")
    print("  * Variance is real -- the Monte Carlo shows the range of outcomes")
    print("  * Start small, verify the win rate is real, THEN scale up")
    print("  * A 57% win rate on 52c contracts is already an excellent edge")
    print("  * If actual win rate drops below ~53%, the strategy loses money")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
