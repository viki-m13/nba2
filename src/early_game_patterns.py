"""
Early Game Pattern Discovery

Goal: Find patterns BEFORE the final 8 minutes that predict
game outcome with 90%+ accuracy.

Time windows to explore:
- Halftime (24 min remaining)
- Q3 (12-24 min remaining)
- Early Q4 (8-12 min remaining)
- Mid-game (15-30 min remaining)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


def generate_games(n_games: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate games with realistic dynamics."""
    np.random.seed(seed)
    all_data = []

    for g in range(n_games):
        gid = f"G{g:05d}"

        # Team quality difference (hidden driver of outcome)
        home_quality = np.random.normal(0, 10)

        hs, as_ = 0, 0
        h_hist, a_hist = [], []

        # Quarter scoring
        q_h = [0, 0, 0, 0]
        q_a = [0, 0, 0, 0]

        states = []

        for secs in range(0, 48*60+1, 30):
            q = min(secs // (12*60) + 1, 4)
            q_idx = q - 1
            mins_left = max((48*60 - secs) / 60, 0)
            mins_elapsed = secs / 60

            # Scoring with quality influence + momentum
            h_mom = np.mean(h_hist[-8:]) - 0.5 if len(h_hist) >= 8 else 0
            a_mom = np.mean(a_hist[-8:]) - 0.5 if len(a_hist) >= 8 else 0

            h_rate = 0.55 * (1 + home_quality/100 + h_mom*0.1)
            a_rate = 0.55 * (1 - home_quality/100 + a_mom*0.1)

            hp = np.random.poisson(max(h_rate, 0.1))
            ap = np.random.poisson(max(a_rate, 0.1))

            hs += hp
            as_ += ap
            q_h[q_idx] += hp
            q_a[q_idx] += ap
            h_hist.append(hp)
            a_hist.append(ap)

            # Features
            diff = hs - as_
            mom_2 = sum(h_hist[-4:]) - sum(a_hist[-4:]) if len(h_hist) >= 4 else 0
            mom_5 = sum(h_hist[-10:]) - sum(a_hist[-10:]) if len(h_hist) >= 10 else 0
            mom_10 = sum(h_hist[-20:]) - sum(a_hist[-20:]) if len(h_hist) >= 20 else 0

            # Quarter differentials
            q1_diff = q_h[0] - q_a[0] if q >= 1 else 0
            q2_diff = q_h[1] - q_a[1] if q >= 2 else 0
            q3_diff = q_h[2] - q_a[2] if q >= 3 else 0
            curr_q_diff = q_h[q_idx] - q_a[q_idx]

            # Efficiency (scoring rate)
            h_eff = sum(h_hist[-20:]) / 20 if len(h_hist) >= 20 else 0.5
            a_eff = sum(a_hist[-20:]) / 20 if len(a_hist) >= 20 else 0.5

            # Run detection (consecutive scoring advantage)
            run = 0
            for i in range(min(len(h_hist), 10)):
                idx = -(i+1)
                if h_hist[idx] > a_hist[idx]:
                    run += 1
                elif a_hist[idx] > h_hist[idx]:
                    run -= 1

            states.append({
                'gid': gid,
                'secs': secs,
                'quarter': q,
                'mins_left': mins_left,
                'mins_elapsed': mins_elapsed,
                'hs': hs, 'as': as_,
                'diff': diff,
                'abs_diff': abs(diff),
                'mom_2': mom_2,
                'mom_5': mom_5,
                'mom_10': mom_10,
                'q1_diff': q1_diff,
                'q2_diff': q2_diff,
                'q3_diff': q3_diff,
                'curr_q_diff': curr_q_diff,
                'h_eff': h_eff,
                'a_eff': a_eff,
                'eff_diff': h_eff - a_eff,
                'run': run,
            })

        # Determine winner
        final = states[-1]
        winner = 'home' if final['hs'] > final['as'] else 'away'

        for s in states:
            s['winner'] = winner
            s['home_wins'] = 1 if winner == 'home' else 0

        all_data.extend(states)

    return pd.DataFrame(all_data)


def test_pattern(df: pd.DataFrame,
                 min_diff: int,
                 min_mom: int,
                 min_mins: float,
                 max_mins: float,
                 extra_conditions: dict = None) -> Dict:
    """Test a specific pattern."""

    mask = (
        (df['mins_left'] >= min_mins) &
        (df['mins_left'] <= max_mins)
    )

    # Home version
    home_mask = mask & (df['diff'] >= min_diff) & (df['mom_5'] >= min_mom)
    if extra_conditions:
        for col, val in extra_conditions.items():
            if val > 0:
                home_mask = home_mask & (df[col] >= val)
            else:
                home_mask = home_mask & (df[col] <= val)

    # Away version
    away_mask = mask & (df['diff'] <= -min_diff) & (df['mom_5'] <= -min_mom)
    if extra_conditions:
        for col, val in extra_conditions.items():
            if val > 0:
                away_mask = away_mask & (df[col] <= -val)
            else:
                away_mask = away_mask & (df[col] >= -val)

    home_subset = df[home_mask]
    away_subset = df[away_mask]

    # Count unique games
    home_games = home_subset.groupby('gid')['home_wins'].first()
    away_games = away_subset.groupby('gid')['home_wins'].first()

    total_games = len(home_games) + len(away_games)
    if total_games < 30:
        return None

    home_wins = home_games.sum()
    away_wins = len(away_games) - away_games.sum()

    wr = (home_wins + away_wins) / total_games
    n_total_games = df['gid'].nunique()

    return {
        'trades': total_games,
        'wins': home_wins + away_wins,
        'wr': wr,
        'coverage': total_games / n_total_games,
    }


def search_time_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Search patterns across different time windows."""
    results = []

    # Time windows to test
    windows = [
        (8, 12, "Early Q4"),
        (12, 18, "Late Q3"),
        (18, 24, "Early Q3 / Halftime"),
        (24, 30, "Late Q2"),
        (30, 36, "Mid Q2"),
        (10, 15, "Q3-Q4 transition"),
        (15, 20, "Mid Q3"),
        (20, 28, "Around halftime"),
    ]

    # Parameter ranges
    leads = [5, 7, 10, 12, 15, 18, 20]
    momentums = [3, 5, 7, 10, 12, 15]

    for min_mins, max_mins, desc in windows:
        for lead in leads:
            for mom in momentums:
                res = test_pattern(df, lead, mom, min_mins, max_mins)
                if res and res['trades'] >= 30:
                    results.append({
                        'window': desc,
                        'min_mins': min_mins,
                        'max_mins': max_mins,
                        'lead': lead,
                        'mom': mom,
                        **res
                    })

    return pd.DataFrame(results)


def search_compound_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Search for compound patterns (multiple conditions)."""
    results = []

    # Patterns to test:
    # 1. Lead + Momentum + Quarter dominance
    # 2. Lead + Momentum + Efficiency edge
    # 3. Lead + Multi-timeframe momentum agreement
    # 4. Halftime lead + Q3 momentum

    windows = [
        (10, 15, "Q3-Q4"),
        (15, 20, "Mid Q3"),
        (20, 26, "Halftime+"),
    ]

    for min_mins, max_mins, desc in windows:
        for lead in [7, 10, 12, 15]:
            for mom in [5, 7, 10]:
                # Pattern 1: Lead + Mom + Current quarter dominance
                for q_dom in [5, 7, 10]:
                    res = test_pattern(df, lead, mom, min_mins, max_mins,
                                      {'curr_q_diff': q_dom})
                    if res and res['wr'] >= 0.85:
                        results.append({
                            'pattern': f'lead{lead}_mom{mom}_qdom{q_dom}',
                            'window': desc,
                            'min_mins': min_mins,
                            **res
                        })

                # Pattern 2: Lead + Mom + Efficiency
                for eff in [0.2, 0.3, 0.4]:
                    res = test_pattern(df, lead, mom, min_mins, max_mins,
                                      {'eff_diff': eff})
                    if res and res['wr'] >= 0.85:
                        results.append({
                            'pattern': f'lead{lead}_mom{mom}_eff{eff}',
                            'window': desc,
                            'min_mins': min_mins,
                            **res
                        })

                # Pattern 3: Lead + Both short and long momentum
                for mom_long in [7, 10, 12]:
                    res = test_pattern(df, lead, mom, min_mins, max_mins,
                                      {'mom_10': mom_long})
                    if res and res['wr'] >= 0.85:
                        results.append({
                            'pattern': f'lead{lead}_mom{mom}_mom10_{mom_long}',
                            'window': desc,
                            'min_mins': min_mins,
                            **res
                        })

    return pd.DataFrame(results)


def search_halftime_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Special focus on halftime patterns."""
    results = []

    # At halftime (24 mins left), what predicts outcome?
    halftime = df[(df['mins_left'] >= 23) & (df['mins_left'] <= 25)]

    for lead in [5, 7, 10, 12, 15, 18, 20]:
        for q1_dom in [0, 3, 5, 7]:
            for q2_dom in [0, 3, 5, 7]:
                mask_h = (
                    (halftime['diff'] >= lead) &
                    (halftime['q1_diff'] >= q1_dom) &
                    (halftime['q2_diff'] >= q2_dom)
                )
                mask_a = (
                    (halftime['diff'] <= -lead) &
                    (halftime['q1_diff'] <= -q1_dom) &
                    (halftime['q2_diff'] <= -q2_dom)
                )

                h_games = halftime[mask_h].groupby('gid')['home_wins'].first()
                a_games = halftime[mask_a].groupby('gid')['home_wins'].first()

                total = len(h_games) + len(a_games)
                if total < 30:
                    continue

                wins = h_games.sum() + (len(a_games) - a_games.sum())
                wr = wins / total

                if wr >= 0.80:
                    results.append({
                        'pattern': f'HT_lead{lead}_q1dom{q1_dom}_q2dom{q2_dom}',
                        'lead': lead,
                        'trades': total,
                        'wr': wr,
                        'coverage': total / df['gid'].nunique(),
                    })

    return pd.DataFrame(results)


def main():
    print("="*70)
    print("EARLY GAME PATTERN DISCOVERY")
    print("Finding 90%+ accurate patterns BEFORE final 8 minutes")
    print("="*70)

    print("\nGenerating 5000 games...")
    df = generate_games(5000)
    n_games = df['gid'].nunique()
    print(f"Generated {len(df):,} data points from {n_games:,} games")

    # Search different time windows
    print("\n" + "-"*60)
    print("SEARCHING TIME WINDOWS")
    print("-"*60)

    window_results = search_time_windows(df)

    # Filter for 90%+ win rate
    wr90 = window_results[window_results['wr'] >= 0.90]
    print(f"\n90%+ Win Rate Patterns: {len(wr90)}")

    if len(wr90) > 0:
        # Sort by how early in game (higher min_mins = earlier)
        wr90_sorted = wr90.sort_values(['min_mins', 'coverage'], ascending=[False, False])
        print("\nEarliest patterns with 90%+ WR:")
        print(wr90_sorted.head(20).to_string(index=False))

    # 85%+ with better coverage
    wr85 = window_results[(window_results['wr'] >= 0.85) & (window_results['min_mins'] >= 12)]
    print(f"\n85%+ Win Rate (before Q4): {len(wr85)}")
    if len(wr85) > 0:
        print(wr85.sort_values('wr', ascending=False).head(15).to_string(index=False))

    # Compound patterns
    print("\n" + "-"*60)
    print("COMPOUND PATTERNS (multiple conditions)")
    print("-"*60)

    compound_results = search_compound_patterns(df)
    if len(compound_results) > 0:
        wr90_compound = compound_results[compound_results['wr'] >= 0.90]
        print(f"\n90%+ WR Compound Patterns: {len(wr90_compound)}")
        if len(wr90_compound) > 0:
            print(wr90_compound.sort_values(['min_mins', 'wr'],
                  ascending=[False, False]).head(15).to_string(index=False))

    # Halftime patterns
    print("\n" + "-"*60)
    print("HALFTIME PATTERNS (24 min remaining)")
    print("-"*60)

    ht_results = search_halftime_patterns(df)
    if len(ht_results) > 0:
        wr90_ht = ht_results[ht_results['wr'] >= 0.90]
        print(f"\n90%+ WR Halftime Patterns: {len(wr90_ht)}")
        if len(wr90_ht) > 0:
            print(wr90_ht.sort_values('wr', ascending=False).head(10).to_string(index=False))

        print(f"\n85%+ WR Halftime Patterns:")
        wr85_ht = ht_results[ht_results['wr'] >= 0.85].sort_values('wr', ascending=False)
        print(wr85_ht.head(10).to_string(index=False))

    # Best early patterns
    all_early = window_results[window_results['min_mins'] >= 15]  # 15+ min remaining
    if len(all_early) > 0:
        best_early = all_early.sort_values('wr', ascending=False)

        print(f"\n{'='*70}")
        print("BEST PATTERNS WITH 15+ MINUTES REMAINING")
        print("="*70)
        print(best_early.head(20).to_string(index=False))

        # The very best
        if len(best_early) > 0:
            best = best_early.iloc[0]
            print(f"""
{'='*70}
BEST EARLY GAME PATTERN
{'='*70}
Time Window: {best['window']} ({best['min_mins']:.0f}-{best['max_mins']:.0f} min remaining)

ENTRY RULE:
  Lead >= {int(best['lead'])} points
  Momentum (5min) >= {int(best['mom'])} points

RESULTS:
  Win Rate: {best['wr']:.1%}
  Trades: {int(best['trades'])}
  Coverage: {best['coverage']:.1%}
""")

    # Save results
    window_results.to_csv('/home/user/nba2/output/early_patterns.csv', index=False)
    print(f"\nResults saved to output/early_patterns.csv")

    return window_results


if __name__ == "__main__":
    results = main()
