"""
Find signals that actually work for SPREAD COVERAGE.

The problem: Betting that a team maintains their exact lead is ~50%.
Solution: Find patterns where spread coverage is reliable.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path


def load_results():
    """Load previously collected results."""
    try:
        with open('cache/spread_validation_results.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None


def analyze_by_lead_threshold(results):
    """Find minimum lead required for reliable spread coverage."""
    print("="*70)
    print("SPREAD COVERAGE BY MINIMUM LEAD")
    print("="*70)

    for min_lead in [5, 7, 10, 12, 15, 18, 20, 22, 25]:
        filtered = [r for r in results if r['lead_at_signal'] >= min_lead]
        if not filtered:
            continue

        wins = sum(1 for r in filtered if r['spread_covered'])
        total = len(filtered)
        wr = wins / total if total > 0 else 0

        ml_wins = sum(1 for r in filtered if r['moneyline_won'])
        ml_wr = ml_wins / total

        print(f"Lead >= {min_lead:>2}: {wins:>3}/{total:<3} = {wr:.1%} spread | {ml_wr:.1%} ML")


def analyze_by_momentum_threshold(results):
    """Find minimum momentum required."""
    print("\n" + "="*70)
    print("SPREAD COVERAGE BY MINIMUM MOMENTUM")
    print("="*70)

    for min_mom in [3, 5, 7, 10, 12, 15]:
        filtered = [r for r in results if r['momentum'] >= min_mom]
        if not filtered:
            continue

        wins = sum(1 for r in filtered if r['spread_covered'])
        total = len(filtered)
        wr = wins / total if total > 0 else 0

        ml_wins = sum(1 for r in filtered if r['moneyline_won'])
        ml_wr = ml_wins / total

        print(f"Mom >= {min_mom:>2}: {wins:>3}/{total:<3} = {wr:.1%} spread | {ml_wr:.1%} ML")


def analyze_combined(results):
    """Find best Lead + Momentum combination for spread."""
    print("\n" + "="*70)
    print("SPREAD COVERAGE BY LEAD + MOMENTUM COMBINATION")
    print("="*70)

    best = []

    for min_lead in [10, 12, 15, 18, 20, 22, 25]:
        for min_mom in [5, 7, 10, 12, 15]:
            filtered = [r for r in results
                        if r['lead_at_signal'] >= min_lead
                        and r['momentum'] >= min_mom]

            if len(filtered) < 3:
                continue

            wins = sum(1 for r in filtered if r['spread_covered'])
            total = len(filtered)
            wr = wins / total

            if wr >= 0.70:
                best.append({
                    'lead': min_lead,
                    'mom': min_mom,
                    'wins': wins,
                    'total': total,
                    'wr': wr,
                })

    # Sort by win rate then sample size
    best.sort(key=lambda x: (-x['wr'], -x['total']))

    print(f"\n{'Lead':<6} {'Mom':<6} {'Wins':<6} {'Total':<7} {'Spread WR':<10}")
    print("-"*40)

    for b in best[:20]:
        print(f">={b['lead']:<4} >={b['mom']:<4} {b['wins']:<6} {b['total']:<7} {b['wr']:.1%}")


def analyze_by_time_window(results):
    """Find best time windows for spread coverage."""
    print("\n" + "="*70)
    print("SPREAD COVERAGE BY TIME WINDOW")
    print("="*70)

    windows = [
        (30, 48, 'Q2'),
        (24, 30, 'Late Q2'),
        (18, 24, 'Halftime'),
        (12, 18, 'Q3'),
        (8, 12, 'Early Q4'),
        (2, 8, 'Final'),
    ]

    for min_m, max_m, name in windows:
        filtered = [r for r in results
                    if min_m <= r['mins_remaining'] <= max_m]

        if not filtered:
            continue

        wins = sum(1 for r in filtered if r['spread_covered'])
        total = len(filtered)
        wr = wins / total if total > 0 else 0

        print(f"{name:<12} ({min_m}-{max_m}min): {wins:>3}/{total:<3} = {wr:.1%}")


def analyze_time_and_lead(results):
    """Find best time + lead combinations."""
    print("\n" + "="*70)
    print("SPREAD COVERAGE BY TIME WINDOW + LEAD")
    print("="*70)

    windows = [
        (24, 48, 'Q2/Early Q3'),
        (18, 24, 'Halftime'),
        (12, 18, 'Q3'),
        (8, 12, 'Early Q4'),
        (2, 8, 'Final'),
    ]

    best_combos = []

    for min_m, max_m, name in windows:
        for min_lead in [10, 12, 15, 18, 20]:
            for min_mom in [5, 7, 10, 12]:
                filtered = [r for r in results
                            if min_m <= r['mins_remaining'] <= max_m
                            and r['lead_at_signal'] >= min_lead
                            and r['momentum'] >= min_mom]

                if len(filtered) < 3:
                    continue

                wins = sum(1 for r in filtered if r['spread_covered'])
                total = len(filtered)
                wr = wins / total

                if wr >= 0.75:
                    best_combos.append({
                        'window': name,
                        'min_mins': min_m,
                        'max_mins': max_m,
                        'lead': min_lead,
                        'mom': min_mom,
                        'wins': wins,
                        'total': total,
                        'wr': wr,
                    })

    best_combos.sort(key=lambda x: (-x['wr'], -x['total']))

    print(f"\n{'Window':<15} {'Lead':<6} {'Mom':<5} {'Wins':<5} {'Total':<6} {'WR':<8}")
    print("-"*55)

    for b in best_combos[:15]:
        print(f"{b['window']:<15} >={b['lead']:<4} >={b['mom']:<3} {b['wins']:<5} {b['total']:<6} {b['wr']:.1%}")


def find_best_spread_strategy(results):
    """Find the optimal strategy for spread coverage."""
    print("\n" + "="*70)
    print("OPTIMAL SPREAD COVERAGE STRATEGY")
    print("="*70)

    # Test various configurations
    configs = []

    for min_lead in range(15, 28):
        for min_mom in range(8, 18):
            filtered = [r for r in results
                        if r['lead_at_signal'] >= min_lead
                        and r['momentum'] >= min_mom]

            if len(filtered) < 5:
                continue

            wins = sum(1 for r in filtered if r['spread_covered'])
            total = len(filtered)
            wr = wins / total

            configs.append({
                'lead': min_lead,
                'mom': min_mom,
                'wins': wins,
                'total': total,
                'wr': wr,
            })

    # Find configs with 80%+ win rate
    good_configs = [c for c in configs if c['wr'] >= 0.80]
    good_configs.sort(key=lambda x: (-x['total'], -x['wr']))

    print("\n80%+ SPREAD WIN RATE CONFIGURATIONS:")
    print(f"\n{'Lead':<6} {'Mom':<6} {'Wins':<6} {'Total':<7} {'WR':<8}")
    print("-"*35)

    for c in good_configs[:10]:
        print(f">={c['lead']:<4} >={c['mom']:<4} {c['wins']:<6} {c['total']:<7} {c['wr']:.1%}")

    # Find best balance of win rate and volume
    print("\n\nBEST CONFIGURATIONS (balancing WR and volume):")
    for c in configs:
        c['score'] = c['wr'] * 100 + c['total'] * 0.5

    configs.sort(key=lambda x: -x['score'])

    print(f"\n{'Lead':<6} {'Mom':<6} {'Wins':<6} {'Total':<7} {'WR':<8} {'Score':<8}")
    print("-"*45)

    for c in configs[:10]:
        print(f">={c['lead']:<4} >={c['mom']:<4} {c['wins']:<6} {c['total']:<7} {c['wr']:.1%}    {c['score']:.1f}")


def main():
    data = load_results()
    if not data:
        print("No results found. Run validate_spread_coverage.py first.")
        return

    results = data['results']
    print(f"Analyzing {len(results)} game signals...")

    analyze_by_lead_threshold(results)
    analyze_by_momentum_threshold(results)
    analyze_combined(results)
    analyze_by_time_window(results)
    analyze_time_and_lead(results)
    find_best_spread_strategy(results)

    # Show the losses that would have been avoided with stricter thresholds
    print("\n" + "="*70)
    print("ANALYSIS: WHAT THRESHOLDS AVOID YOUR LOSSES?")
    print("="*70)

    print("\nYour loss cases:")
    print("1. MIA Early Q4: Lead=10, Mom=6 → Lost spread")
    print("2. NOP Mid Q3 Alt: Lead=17, Mom=9 → Lost spread")

    print("\nWith Lead>=18, Mom>=10:")
    filtered = [r for r in results
                if r['lead_at_signal'] >= 18 and r['momentum'] >= 10]
    wins = sum(1 for r in filtered if r['spread_covered'])
    print(f"  {wins}/{len(filtered)} = {wins/len(filtered)*100:.1f}% spread coverage")

    print("\nWith Lead>=20, Mom>=10:")
    filtered = [r for r in results
                if r['lead_at_signal'] >= 20 and r['momentum'] >= 10]
    if filtered:
        wins = sum(1 for r in filtered if r['spread_covered'])
        print(f"  {wins}/{len(filtered)} = {wins/len(filtered)*100:.1f}% spread coverage")
    else:
        print("  No samples")


if __name__ == "__main__":
    main()
