"""
Calculate REALISTIC profit accounting for odds on alternate spreads.

When a team is up 20, betting them -7 won't be at -110.
It might be -300 to -500 (depending on book).

We need to find situations where our win rate exceeds implied break-even.
"""

import json
from collections import defaultdict


def load_data():
    with open('data/comprehensive_validation.json', 'r') as f:
        return json.load(f)


def odds_to_implied_prob(american_odds):
    """Convert American odds to implied probability."""
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def prob_to_american_odds(prob):
    """Convert probability to American odds."""
    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    else:
        return 100 * (1 - prob) / prob


def estimate_alternate_spread_odds(current_spread, bet_spread):
    """
    Estimate what odds you'd get betting an alternate spread.

    If market has team -20 and you bet -7, you're buying 13 points.
    Rule of thumb: ~15-20 cents per point in NBA

    More accurate: Each point is worth roughly -15 to -20 in odds movement
    """
    points_bought = current_spread - bet_spread  # e.g., 20 - 7 = 13 points

    # Approximate odds adjustment
    # Starting from -110 (fair odds), each point toward the favorite costs ~15-20
    # This is a rough estimate - actual varies by book and game state

    if points_bought <= 0:
        return -110  # No adjustment needed

    # Conservative estimate: -20 per point
    odds_adjustment = points_bought * 20
    estimated_odds = -110 - odds_adjustment

    # Cap at realistic limits (most books don't go beyond -1000)
    return max(estimated_odds, -800)


def calculate_expected_value(win_prob, american_odds, bet_size=100):
    """Calculate expected value of a bet."""
    if american_odds < 0:
        win_amount = bet_size * (100 / abs(american_odds))
    else:
        win_amount = bet_size * (american_odds / 100)

    ev = (win_prob * win_amount) - ((1 - win_prob) * bet_size)
    return ev


def analyze_realistic_profit(data):
    """
    Calculate realistic profit for different strategies.
    """
    print("="*70)
    print("REALISTIC PROFIT ANALYSIS")
    print("Accounting for alternate spread odds")
    print("="*70)

    # Test different scenarios
    scenarios = [
        # (name, min_lead, min_mom, bet_spread, estimated_current_spread)
        ("Lead 15-17, bet -5", 15, 10, 5, 16),
        ("Lead 15-17, bet -7", 15, 10, 7, 16),
        ("Lead 18-20, bet -5", 18, 10, 5, 19),
        ("Lead 18-20, bet -7", 18, 10, 7, 19),
        ("Lead 20-24, bet -5", 20, 10, 5, 22),
        ("Lead 20-24, bet -7", 20, 10, 7, 22),
        ("Lead 20-24, bet -10", 20, 10, 10, 22),
    ]

    print(f"\n{'Scenario':<25} {'Win%':<8} {'Est.Odds':<10} {'Break-even':<12} {'EV/$100':<10} {'Edge'}")
    print("-"*80)

    for name, min_lead, min_mom, bet_spread, est_market in scenarios:
        # Filter data
        if "15-17" in name:
            filtered = [r for r in data
                       if 15 <= r['actual_lead'] <= 17
                       and r['actual_mom'] >= min_mom
                       and r['ml_won']]
        elif "18-20" in name:
            filtered = [r for r in data
                       if 18 <= r['actual_lead'] <= 20
                       and r['actual_mom'] >= min_mom
                       and r['ml_won']]
        else:  # 20-24
            filtered = [r for r in data
                       if 20 <= r['actual_lead'] <= 24
                       and r['actual_mom'] >= min_mom
                       and r['ml_won']]

        if len(filtered) < 20:
            continue

        # Calculate win rate
        wins = sum(1 for r in filtered if r['final_margin'] >= bet_spread)
        win_pct = wins / len(filtered)

        # Estimate odds
        est_odds = estimate_alternate_spread_odds(est_market, bet_spread)
        break_even = odds_to_implied_prob(est_odds)

        # Calculate EV
        ev = calculate_expected_value(win_pct, est_odds)
        edge = win_pct - break_even

        edge_marker = "✓ PROFIT" if edge > 0.05 else ("~ break-even" if edge > -0.02 else "✗ LOSS")

        print(f"{name:<25} {win_pct:.1%}{'':.<3} {est_odds:<10.0f} {break_even:.1%}{'':.<7} ${ev:+.2f}{'':.<5} {edge_marker}")


def find_true_edge_opportunities(data):
    """
    Find opportunities where win rate EXCEEDS the implied odds.
    """
    print("\n" + "="*70)
    print("TRUE EDGE OPPORTUNITIES")
    print("Where win% significantly exceeds break-even probability")
    print("="*70)

    # The key insight: we need situations where
    # 1. Our win rate is high
    # 2. The market doesn't price it correctly

    # This happens when momentum is HIGH but market only sees lead
    # Market prices based on lead, but momentum predicts EXTENSION

    print("\n--- MOMENTUM EDGE STRATEGY ---")
    print("High momentum = market underestimates win probability")
    print()

    for lead_range, (low, high) in [("15-19", (15, 19)), ("20-24", (20, 24))]:
        print(f"\n{lead_range} point leads:")

        # Low momentum vs high momentum
        for mom_thresh in [12, 14, 16]:
            high_mom = [r for r in data
                       if low <= r['actual_lead'] <= high
                       and r['actual_mom'] >= mom_thresh
                       and r['ml_won']]

            if len(high_mom) < 15:
                continue

            # At different bet spreads
            for bet_spread in [5, 7, 10]:
                wins = sum(1 for r in high_mom if r['final_margin'] >= bet_spread)
                win_pct = wins / len(high_mom)

                # Estimate odds (market sees lead, we see momentum)
                avg_lead = sum(r['actual_lead'] for r in high_mom) / len(high_mom)
                est_odds = estimate_alternate_spread_odds(avg_lead, bet_spread)
                break_even = odds_to_implied_prob(est_odds)

                edge = win_pct - break_even
                ev = calculate_expected_value(win_pct, est_odds)

                if edge > 0.10:  # 10%+ edge
                    print(f"  Mom>={mom_thresh}, bet -{bet_spread}: {win_pct:.1%} win "
                          f"(need {break_even:.1%}), EV=${ev:+.1f}/bet ← EDGE")


def analyze_moneyline_edge(data):
    """
    What about MONEYLINE bets instead of spreads?
    If team is up 15 with momentum, what are ML odds and do we have edge?
    """
    print("\n" + "="*70)
    print("MONEYLINE EDGE ANALYSIS")
    print("="*70)

    print("""
When a team is up 15, live ML might be around -800 to -1500.
Our win rate is ~96-100%, but is that enough to overcome the odds?
""")

    test_cases = [
        ("Lead 12-14, Mom>=10", 12, 14, 10, -400),
        ("Lead 15-17, Mom>=10", 15, 17, 10, -600),
        ("Lead 15-17, Mom>=12", 15, 17, 12, -600),
        ("Lead 18-20, Mom>=10", 18, 20, 10, -1000),
        ("Lead 18-20, Mom>=12", 18, 20, 12, -1000),
        ("Lead 20+, Mom>=10", 20, 50, 10, -1500),
        ("Lead 20+, Mom>=12", 20, 50, 12, -1500),
    ]

    print(f"\n{'Condition':<25} {'Win%':<8} {'Est.ML Odds':<12} {'Break-even':<12} {'EV/$100':<10}")
    print("-"*75)

    for name, low_lead, high_lead, min_mom, est_ml_odds in test_cases:
        filtered = [r for r in data
                   if low_lead <= r['actual_lead'] <= high_lead
                   and r['actual_mom'] >= min_mom]

        if len(filtered) < 20:
            continue

        wins = sum(1 for r in filtered if r['ml_won'])
        win_pct = wins / len(filtered)
        break_even = odds_to_implied_prob(est_ml_odds)
        ev = calculate_expected_value(win_pct, est_ml_odds)

        marker = "✓ EDGE" if win_pct > break_even else ""
        print(f"{name:<25} {win_pct:.1%}{'':.<3} {est_ml_odds:<12} {break_even:.1%}{'':.<7} ${ev:+.2f} {marker}")


def find_sweet_spot(data):
    """
    Find the SWEET SPOT where:
    1. Win rate is high enough
    2. But lead isn't so large that odds are prohibitive
    """
    print("\n" + "="*70)
    print("SWEET SPOT ANALYSIS")
    print("Not too small (risky) not too big (bad odds)")
    print("="*70)

    # The sweet spot is likely:
    # - Lead 12-17 (high enough to win, low enough for decent odds)
    # - High momentum (indicates lead will hold or extend)
    # - Mid-game (enough time for momentum to matter)

    print("\nSearching for lead ranges with best risk/reward ratio...\n")

    results = []

    for lead_low in range(10, 22, 2):
        lead_high = lead_low + 4

        for min_mom in [10, 12, 14]:
            for bet_spread in [5, 7]:
                filtered = [r for r in data
                           if lead_low <= r['actual_lead'] <= lead_high
                           and r['actual_mom'] >= min_mom
                           and r['ml_won']]

                if len(filtered) < 20:
                    continue

                wins = sum(1 for r in filtered if r['final_margin'] >= bet_spread)
                win_pct = wins / len(filtered)

                avg_lead = (lead_low + lead_high) / 2
                est_odds = estimate_alternate_spread_odds(avg_lead, bet_spread)
                break_even = odds_to_implied_prob(est_odds)
                ev = calculate_expected_value(win_pct, est_odds)

                results.append({
                    'lead_range': f"{lead_low}-{lead_high}",
                    'min_mom': min_mom,
                    'bet_spread': bet_spread,
                    'n': len(filtered),
                    'win_pct': win_pct,
                    'est_odds': est_odds,
                    'break_even': break_even,
                    'ev': ev,
                    'edge': win_pct - break_even
                })

    # Sort by EV
    results.sort(key=lambda x: -x['ev'])

    print(f"{'Lead':<10} {'Mom':<6} {'Bet':<6} {'N':<5} {'Win%':<8} {'Odds':<8} {'BE%':<8} {'EV':<8} {'Edge'}")
    print("-"*70)

    for r in results[:15]:
        marker = "← BEST" if r['ev'] > 5 else ""
        print(f"{r['lead_range']:<10} >={r['min_mom']:<4} -{r['bet_spread']:<4} {r['n']:<5} "
              f"{r['win_pct']:.1%}{'':.<3} {r['est_odds']:.0f}{'':.<4} {r['break_even']:.1%}{'':.<3} "
              f"${r['ev']:+.1f}{'':.<3} {r['edge']:+.1%} {marker}")


def main():
    print("Loading data...")
    data = load_data()

    # Dedupe
    seen = set()
    unique = []
    for r in data:
        key = (r['date'], r['home_team'], r['away_team'], r['actual_lead'], r['actual_mom'])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"Unique signals: {len(unique)}")

    analyze_realistic_profit(unique)
    find_true_edge_opportunities(unique)
    analyze_moneyline_edge(unique)
    find_sweet_spot(unique)


if __name__ == "__main__":
    main()
