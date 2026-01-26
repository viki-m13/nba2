"""
Try different spread strategies:
1. Reduced spread (bet half the current lead)
2. Fixed spread (always bet -7.5 or -10.5 when conditions met)
3. Very strict thresholds
"""

import json
from collections import defaultdict


def load_data():
    """Load the comprehensive validation data."""
    with open('data/comprehensive_validation.json', 'r') as f:
        return json.load(f)


def analyze_reduced_spreads(data):
    """What if we bet HALF the current lead as spread?"""
    print("="*70)
    print("STRATEGY 1: BET HALF THE CURRENT LEAD")
    print("If up 16, bet -8 instead of -16")
    print("="*70)

    for reduction in [0.3, 0.4, 0.5, 0.6, 0.7]:
        wins = 0
        total = 0

        for r in data:
            if r['window'] != 'Halftime':
                continue
            if r['min_lead'] < 15 or r['min_mom'] < 10:
                continue

            actual_lead = r['actual_lead']
            final_margin = r['final_margin']
            ml_won = r['ml_won']

            if not ml_won:
                continue  # Only count games where team won

            reduced_spread = actual_lead * reduction
            covered = final_margin >= reduced_spread

            total += 1
            if covered:
                wins += 1

        if total > 0:
            print(f"Bet {reduction:.0%} of lead: {wins}/{total} = {wins/total:.1%}")


def analyze_fixed_spreads(data):
    """What if we bet a FIXED spread when conditions are met?"""
    print("\n" + "="*70)
    print("STRATEGY 2: BET FIXED SPREAD WHEN LEADING BIG")
    print("="*70)

    # Filter to halftime with lead >= 15, mom >= 10
    filtered = [r for r in data
                if r['window'] == 'Halftime'
                and r['min_lead'] >= 15
                and r['min_mom'] >= 10]

    print(f"\nCondition: Halftime, Lead>=15, Mom>=10 ({len(filtered)} trades)")

    for fixed_spread in [5, 6, 7, 8, 10, 12]:
        wins = 0
        for r in filtered:
            final_margin = r['final_margin']
            ml_won = r['ml_won']

            # Team must win by at least fixed_spread
            if ml_won and final_margin >= fixed_spread:
                wins += 1

        if len(filtered) > 0:
            print(f"Bet -{fixed_spread}: {wins}/{len(filtered)} = {wins/len(filtered):.1%}")


def analyze_very_strict(data):
    """What if we use VERY strict thresholds?"""
    print("\n" + "="*70)
    print("STRATEGY 3: VERY STRICT THRESHOLDS")
    print("="*70)

    configs = [
        ('Halftime', 20, 12),
        ('Halftime', 20, 14),
        ('Halftime', 18, 14),
        ('Early Q3', 20, 12),
        ('Early Q3', 20, 14),
        ('Late Q2', 20, 12),
    ]

    for window, min_lead, min_mom in configs:
        filtered = [r for r in data
                    if r['window'] == window
                    and r['min_lead'] >= min_lead
                    and r['min_mom'] >= min_mom]

        if not filtered:
            continue

        # Full spread coverage
        full_wins = sum(1 for r in filtered if r['spread_covered'])

        # Half spread coverage
        half_wins = sum(1 for r in filtered
                       if r['ml_won'] and r['final_margin'] >= r['actual_lead'] * 0.5)

        # Fixed -7 spread
        fixed7_wins = sum(1 for r in filtered
                         if r['ml_won'] and r['final_margin'] >= 7)

        print(f"\n{window} Lead>={min_lead} Mom>={min_mom} ({len(filtered)} trades):")
        print(f"  Full spread (-lead): {full_wins}/{len(filtered)} = {full_wins/len(filtered):.1%}")
        print(f"  Half spread (-lead/2): {half_wins}/{len(filtered)} = {half_wins/len(filtered):.1%}")
        print(f"  Fixed -7: {fixed7_wins}/{len(filtered)} = {fixed7_wins/len(filtered):.1%}")


def analyze_lead_ranges(data):
    """Analyze by actual lead at signal time."""
    print("\n" + "="*70)
    print("STRATEGY 4: ANALYZE BY ACTUAL LEAD SIZE")
    print("="*70)

    # Group by lead ranges
    ranges = [(15, 17), (18, 20), (21, 24), (25, 30), (31, 40)]

    for low, high in ranges:
        filtered = [r for r in data
                    if low <= r['actual_lead'] <= high
                    and r['min_mom'] >= 10]

        if len(filtered) < 5:
            continue

        full_wins = sum(1 for r in filtered if r['spread_covered'])
        ml_wins = sum(1 for r in filtered if r['ml_won'])
        half_wins = sum(1 for r in filtered
                       if r['ml_won'] and r['final_margin'] >= r['actual_lead'] * 0.5)

        print(f"\nLead {low}-{high} pts, Mom>=10 ({len(filtered)} trades):")
        print(f"  Moneyline: {ml_wins}/{len(filtered)} = {ml_wins/len(filtered):.1%}")
        print(f"  Full spread: {full_wins}/{len(filtered)} = {full_wins/len(filtered):.1%}")
        print(f"  Half spread: {half_wins}/{len(filtered)} = {half_wins/len(filtered):.1%}")


def find_any_good_signal(data):
    """Brute force search for ANY signal with 75%+ spread WR."""
    print("\n" + "="*70)
    print("BRUTE FORCE: FIND ANY 75%+ SPREAD SIGNAL")
    print("="*70)

    best = []

    for window in ['Halftime', 'Early Q3', 'Late Q2', 'Mid Q3', 'Q2-Q3']:
        for min_lead in range(15, 30):
            for min_mom in range(8, 20):
                filtered = [r for r in data
                            if r['window'] == window
                            and r['actual_lead'] >= min_lead
                            and r['actual_mom'] >= min_mom]

                if len(filtered) < 10:
                    continue

                # Try full spread
                full_wins = sum(1 for r in filtered if r['spread_covered'])
                full_wr = full_wins / len(filtered)

                if full_wr >= 0.70:
                    best.append({
                        'window': window,
                        'min_lead': min_lead,
                        'min_mom': min_mom,
                        'trades': len(filtered),
                        'wins': full_wins,
                        'wr': full_wr,
                        'type': 'full'
                    })

                # Try half spread
                half_wins = sum(1 for r in filtered
                               if r['ml_won'] and r['final_margin'] >= r['actual_lead'] * 0.5)
                half_wr = half_wins / len(filtered)

                if half_wr >= 0.80:
                    best.append({
                        'window': window,
                        'min_lead': min_lead,
                        'min_mom': min_mom,
                        'trades': len(filtered),
                        'wins': half_wins,
                        'wr': half_wr,
                        'type': 'half'
                    })

    best.sort(key=lambda x: (-x['wr'], -x['trades']))

    if best:
        print(f"\n{'Type':<6} {'Window':<12} {'Lead':<6} {'Mom':<5} {'W/L':<8} {'WR':<8}")
        print("-"*50)
        for b in best[:20]:
            print(f"{b['type']:<6} {b['window']:<12} >={b['min_lead']:<4} >={b['min_mom']:<3} "
                  f"{b['wins']}/{b['trades']:<5} {b['wr']:.1%}")
    else:
        print("No signals found with 70%+ full or 80%+ half spread coverage.")


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} records")

    analyze_reduced_spreads(data)
    analyze_fixed_spreads(data)
    analyze_very_strict(data)
    analyze_lead_ranges(data)
    find_any_good_signal(data)


if __name__ == "__main__":
    main()
