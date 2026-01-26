"""
Find EDGE over market - not just accuracy.

The market prices based on current lead. We need to find:
1. When does final margin EXCEED current lead? (market underprices)
2. When does final margin FALL SHORT of current lead? (market overprices)

Edge = situations where outcome differs from market expectation
"""

import json
from collections import defaultdict
import statistics


def load_data():
    with open('data/comprehensive_validation.json', 'r') as f:
        return json.load(f)


def analyze_margin_vs_lead(data):
    """
    Key question: Given current lead X, what is the DISTRIBUTION of final margins?
    If market prices at -X, we need to know when final margin > X or < X.
    """
    print("="*70)
    print("MARGIN ANALYSIS: Final Margin vs Lead at Signal")
    print("="*70)

    # Group by lead ranges
    lead_buckets = defaultdict(list)

    for r in data:
        lead = r['actual_lead']
        final = r['final_margin']
        ml_won = r['ml_won']

        # Only count when team won (otherwise margin is negative for them)
        if ml_won:
            margin_diff = final - lead  # positive = extended lead, negative = lead shrunk

            if 10 <= lead <= 14:
                lead_buckets['10-14'].append(margin_diff)
            elif 15 <= lead <= 19:
                lead_buckets['15-19'].append(margin_diff)
            elif 20 <= lead <= 24:
                lead_buckets['20-24'].append(margin_diff)
            elif 25 <= lead <= 29:
                lead_buckets['25-29'].append(margin_diff)
            elif lead >= 30:
                lead_buckets['30+'].append(margin_diff)

    print("\nWhen team WINS, how does final margin compare to lead at signal?")
    print("Positive = they extended lead, Negative = lead shrunk\n")

    print(f"{'Lead Range':<12} {'Samples':<10} {'Avg Change':<12} {'Median':<10} {'Std Dev':<10}")
    print("-"*55)

    for bucket in ['10-14', '15-19', '20-24', '25-29', '30+']:
        if bucket in lead_buckets and len(lead_buckets[bucket]) >= 10:
            diffs = lead_buckets[bucket]
            avg = statistics.mean(diffs)
            med = statistics.median(diffs)
            std = statistics.stdev(diffs) if len(diffs) > 1 else 0

            print(f"{bucket:<12} {len(diffs):<10} {avg:+.1f}{'':.<6} {med:+.1f}{'':.<5} {std:.1f}")

    return lead_buckets


def find_lead_extension_patterns(data):
    """
    Find patterns where teams EXTEND their lead (final > lead at signal).
    This is where we have edge - market prices at current lead, but team wins by MORE.
    """
    print("\n" + "="*70)
    print("EDGE FINDER: When do teams EXTEND their lead?")
    print("Market prices ~current lead. Edge = final margin > current lead")
    print("="*70)

    # Test different conditions
    conditions = []

    for window in ['Halftime', 'Early Q3', 'Late Q2', 'Q2-Q3']:
        for min_lead in [12, 15, 18, 20]:
            for min_mom in [10, 12, 14, 16]:
                filtered = [r for r in data
                           if r['window'] == window
                           and r['actual_lead'] >= min_lead
                           and r['actual_mom'] >= min_mom
                           and r['ml_won']]  # Only when they won

                if len(filtered) < 15:
                    continue

                # Count how many EXTENDED their lead
                extended = sum(1 for r in filtered if r['final_margin'] > r['actual_lead'])
                maintained = sum(1 for r in filtered if r['final_margin'] >= r['actual_lead'])

                # Average margin change
                margin_changes = [r['final_margin'] - r['actual_lead'] for r in filtered]
                avg_change = statistics.mean(margin_changes)

                # This is key: if avg_change > 0, teams tend to EXTEND lead
                # That means market (pricing at -lead) is UNDERPRICING

                conditions.append({
                    'window': window,
                    'min_lead': min_lead,
                    'min_mom': min_mom,
                    'samples': len(filtered),
                    'extended': extended,
                    'extend_rate': extended / len(filtered),
                    'maintained_rate': maintained / len(filtered),
                    'avg_change': avg_change,
                })

    # Sort by extend rate
    conditions.sort(key=lambda x: (-x['avg_change'], -x['samples']))

    print(f"\n{'Window':<12} {'Lead':<6} {'Mom':<5} {'N':<5} {'Extend%':<9} {'Avg Δ':<8}")
    print("-"*50)

    for c in conditions[:20]:
        if c['avg_change'] > 0:
            marker = "← EDGE"
        else:
            marker = ""
        print(f"{c['window']:<12} >={c['min_lead']:<4} >={c['min_mom']:<3} {c['samples']:<5} "
              f"{c['extend_rate']:.1%}{'':.<4} {c['avg_change']:+.1f} {marker}")


def find_momentum_edge(data):
    """
    High momentum might indicate team will EXTEND lead.
    This is actual edge - market sees lead, but not momentum impact.
    """
    print("\n" + "="*70)
    print("MOMENTUM EDGE: Does high momentum predict lead EXTENSION?")
    print("="*70)

    # Compare low vs high momentum at same lead
    for lead_range, (low, high) in [('15-20', (15, 20)), ('20-25', (20, 25)), ('25-30', (25, 30))]:
        print(f"\n{lead_range} point leads:")

        for mom_threshold in [8, 10, 12, 14, 16]:
            low_mom = [r for r in data
                      if low <= r['actual_lead'] < high
                      and r['actual_mom'] < mom_threshold
                      and r['ml_won']]

            high_mom = [r for r in data
                       if low <= r['actual_lead'] < high
                       and r['actual_mom'] >= mom_threshold
                       and r['ml_won']]

            if len(low_mom) < 10 or len(high_mom) < 10:
                continue

            low_avg = statistics.mean([r['final_margin'] - r['actual_lead'] for r in low_mom])
            high_avg = statistics.mean([r['final_margin'] - r['actual_lead'] for r in high_mom])

            edge = high_avg - low_avg

            marker = "← MOMENTUM EDGE" if edge > 2 else ""
            print(f"  Mom<{mom_threshold}: {low_avg:+.1f} change ({len(low_mom)} games)")
            print(f"  Mom>={mom_threshold}: {high_avg:+.1f} change ({len(high_mom)} games) {marker}")
            print()


def find_time_edge(data):
    """
    At what time does market misprice the most?
    """
    print("\n" + "="*70)
    print("TIME EDGE: When is market most wrong?")
    print("="*70)

    # Filter to high-confidence situations (lead >= 15, mom >= 10)
    filtered = [r for r in data
               if r['actual_lead'] >= 15
               and r['actual_mom'] >= 10
               and r['ml_won']]

    # Group by time window
    windows = {
        'Q2 (30-36min)': (30, 36),
        'Late Q2 (24-30)': (24, 30),
        'Halftime (18-24)': (18, 24),
        'Early Q3 (15-21)': (15, 21),
        'Mid Q3 (12-18)': (12, 18),
    }

    print(f"\n{'Time Window':<20} {'N':<6} {'Avg Margin Δ':<15} {'Extended%':<12}")
    print("-"*55)

    for name, (min_t, max_t) in windows.items():
        window_data = [r for r in filtered
                      if min_t <= r['mins_remaining'] <= max_t]

        if len(window_data) < 20:
            continue

        changes = [r['final_margin'] - r['actual_lead'] for r in window_data]
        avg_change = statistics.mean(changes)
        extended = sum(1 for c in changes if c > 0) / len(changes)

        marker = "← BEST EDGE" if avg_change > -2 else ""
        print(f"{name:<20} {len(window_data):<6} {avg_change:+.1f}{'':.<10} {extended:.1%}{'':.<7} {marker}")


def find_blowout_extension(data):
    """
    In blowouts, does the leading team keep scoring or let up?
    Key insight: Garbage time USUALLY favors losing team.
    But in SOME situations, winning team keeps pressing.
    """
    print("\n" + "="*70)
    print("BLOWOUT ANALYSIS: When do big leads EXTEND vs SHRINK?")
    print("="*70)

    # Big leads only (20+)
    blowouts = [r for r in data
               if r['actual_lead'] >= 20
               and r['ml_won']]

    print(f"\nTotal blowouts (20+ lead when won): {len(blowouts)}")

    changes = [r['final_margin'] - r['actual_lead'] for r in blowouts]
    extended = [r for r in blowouts if r['final_margin'] > r['actual_lead']]
    shrunk = [r for r in blowouts if r['final_margin'] < r['actual_lead']]

    print(f"Lead extended: {len(extended)} ({len(extended)/len(blowouts):.1%})")
    print(f"Lead shrunk: {len(shrunk)} ({len(shrunk)/len(blowouts):.1%})")
    print(f"Average change: {statistics.mean(changes):+.1f} points")

    # Find what predicts extension
    print("\nWhat predicts lead EXTENSION in blowouts?")

    for mom_thresh in [10, 12, 14, 16]:
        high_mom = [r for r in blowouts if r['actual_mom'] >= mom_thresh]
        if len(high_mom) < 10:
            continue

        ext_rate = sum(1 for r in high_mom if r['final_margin'] > r['actual_lead']) / len(high_mom)
        avg_ch = statistics.mean([r['final_margin'] - r['actual_lead'] for r in high_mom])

        print(f"  Mom >= {mom_thresh}: {ext_rate:.1%} extend, {avg_ch:+.1f} avg change ({len(high_mom)} games)")


def calculate_actual_edge(data):
    """
    Calculate actual betting edge assuming market prices at current lead.
    """
    print("\n" + "="*70)
    print("ACTUAL BETTING EDGE CALCULATION")
    print("="*70)

    print("""
Assumptions:
- Market live spread ≈ current lead (e.g., up 15 → market has -15)
- We can bet alternate spreads at adjusted odds
- Standard -110 juice on spreads

Key question: If we bet at a DIFFERENT spread than market, do we profit?
""")

    # Test: What if we always bet -7 when lead >= 15?
    # We need team to win by 8+ to cover -7

    test_cases = [
        ("Bet -5 when Lead>=15, Mom>=10", 15, 10, 5),
        ("Bet -7 when Lead>=15, Mom>=10", 15, 10, 7),
        ("Bet -10 when Lead>=15, Mom>=10", 15, 10, 10),
        ("Bet -5 when Lead>=20, Mom>=12", 20, 12, 5),
        ("Bet -7 when Lead>=20, Mom>=12", 20, 12, 7),
        ("Bet -10 when Lead>=20, Mom>=12", 20, 12, 10),
    ]

    print(f"{'Strategy':<35} {'W/L':<10} {'Win%':<8} {'Implied Edge':<12}")
    print("-"*70)

    for name, min_lead, min_mom, bet_spread in test_cases:
        filtered = [r for r in data
                   if r['actual_lead'] >= min_lead
                   and r['actual_mom'] >= min_mom
                   and r['ml_won']]  # Only when team won

        if len(filtered) < 20:
            continue

        # Win if final margin >= bet_spread
        wins = sum(1 for r in filtered if r['final_margin'] >= bet_spread)
        total = len(filtered)
        win_pct = wins / total

        # At -110 odds, we need 52.4% to break even
        # Edge = actual win% - 52.4%
        edge = win_pct - 0.524

        edge_str = f"{edge:+.1%}" if edge > 0 else f"{edge:.1%}"
        marker = " ← PROFITABLE" if edge > 0.05 else ""

        print(f"{name:<35} {wins}/{total:<6} {win_pct:.1%}{'':.<3} {edge_str}{marker}")


def find_best_edge_signals(data):
    """
    Find the signals with actual positive edge.
    """
    print("\n" + "="*70)
    print("BEST EDGE SIGNALS (>10% edge over break-even)")
    print("="*70)

    best_signals = []

    for window in ['Halftime', 'Early Q3', 'Late Q2', 'Q2-Q3', 'Mid Q3']:
        for min_lead in range(12, 28, 2):
            for min_mom in range(8, 18, 2):
                for bet_spread in [5, 6, 7, 8, 10]:
                    filtered = [r for r in data
                               if r['window'] == window
                               and r['actual_lead'] >= min_lead
                               and r['actual_mom'] >= min_mom
                               and r['ml_won']]

                    if len(filtered) < 20:
                        continue

                    wins = sum(1 for r in filtered if r['final_margin'] >= bet_spread)
                    win_pct = wins / len(filtered)
                    edge = win_pct - 0.524  # break-even at -110

                    if edge >= 0.10:  # 10%+ edge
                        best_signals.append({
                            'window': window,
                            'min_lead': min_lead,
                            'min_mom': min_mom,
                            'bet_spread': bet_spread,
                            'wins': wins,
                            'total': len(filtered),
                            'win_pct': win_pct,
                            'edge': edge,
                        })

    best_signals.sort(key=lambda x: (-x['edge'], -x['total']))

    print(f"\n{'Window':<12} {'Lead':<6} {'Mom':<5} {'Bet':<6} {'W/L':<10} {'Win%':<8} {'Edge':<8}")
    print("-"*65)

    for s in best_signals[:25]:
        print(f"{s['window']:<12} >={s['min_lead']:<4} >={s['min_mom']:<3} -{s['bet_spread']:<4} "
              f"{s['wins']}/{s['total']:<6} {s['win_pct']:.1%}{'':.<3} {s['edge']:+.1%}")


def main():
    print("Loading 300-game dataset...")
    data = load_data()

    # Deduplicate - keep only unique game signals
    seen = set()
    unique_data = []
    for r in data:
        key = (r['date'], r['home_team'], r['away_team'], r['window'])
        if key not in seen:
            seen.add(key)
            unique_data.append(r)

    print(f"Unique game-window combinations: {len(unique_data)}")

    analyze_margin_vs_lead(unique_data)
    find_lead_extension_patterns(unique_data)
    find_momentum_edge(unique_data)
    find_time_edge(unique_data)
    find_blowout_extension(unique_data)
    calculate_actual_edge(unique_data)
    find_best_edge_signals(data)  # Use full data for more granularity


if __name__ == "__main__":
    main()
