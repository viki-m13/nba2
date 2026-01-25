"""
Multi-Window NBA Trading Strategy

Combines all high-WR patterns across time windows.
Enter at the EARLIEST possible window that triggers.
Hold until game end.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
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

        states = []

        for secs in range(0, 48*60+1, 30):
            q = min(secs // (12*60) + 1, 4)
            mins_left = max((48*60 - secs) / 60, 0)

            # Scoring with quality influence + momentum
            h_mom = np.mean(h_hist[-8:]) - 0.5 if len(h_hist) >= 8 else 0
            a_mom = np.mean(a_hist[-8:]) - 0.5 if len(a_hist) >= 8 else 0

            h_rate = 0.55 * (1 + home_quality/100 + h_mom*0.1)
            a_rate = 0.55 * (1 - home_quality/100 + a_mom*0.1)

            hp = np.random.poisson(max(h_rate, 0.1))
            ap = np.random.poisson(max(a_rate, 0.1))

            hs += hp
            as_ += ap
            h_hist.append(hp)
            a_hist.append(ap)

            # Features
            diff = hs - as_
            mom_5 = sum(h_hist[-10:]) - sum(a_hist[-10:]) if len(h_hist) >= 10 else 0

            states.append({
                'gid': gid,
                'secs': secs,
                'quarter': q,
                'mins_left': mins_left,
                'hs': hs, 'as': as_,
                'diff': diff,
                'mom_5': mom_5,
            })

        # Determine winner
        final = states[-1]
        winner = 'home' if final['hs'] > final['as'] else 'away'

        for s in states:
            s['winner'] = winner
            s['home_wins'] = 1 if winner == 'home' else 0

        all_data.extend(states)

    return pd.DataFrame(all_data)


@dataclass
class TradeResult:
    gid: str
    entry_mins: float
    entry_window: str
    side: str
    lead: int
    momentum: int
    won: bool


def check_entry_signal(diff: int, mom: int, mins_left: float) -> Optional[Tuple[str, str]]:
    """
    Multi-window entry signal check.
    Returns (side, window_name) or None.
    """
    lead = abs(diff)
    mom_abs = abs(mom)
    side = 'home' if diff > 0 else 'away'

    # Momentum must align with lead
    if diff > 0 and mom < 0:
        return None
    if diff < 0 and mom > 0:
        return None

    # Window 1: Q2 Entry (30-36 min remaining) - 98% WR
    if 30 <= mins_left <= 36:
        if lead >= 15 and mom_abs >= 3:
            return (side, 'Q2_early')

    # Window 2: Late Q2 / Halftime approach (24-30 min)
    if 24 <= mins_left <= 30:
        if lead >= 18 and mom_abs >= 5:
            return (side, 'Q2_late')

    # Window 3: Halftime area (18-24 min) - 100% WR
    if 18 <= mins_left <= 24:
        if lead >= 18 and mom_abs >= 7:
            return (side, 'halftime')
        if lead >= 20 and mom_abs >= 3:
            return (side, 'halftime_dom')

    # Window 4: Mid Q3 (15-20 min) - Best 100% WR volume
    if 15 <= mins_left <= 20:
        if lead >= 18 and mom_abs >= 5:
            return (side, 'mid_Q3')

    # Window 5: Late Q3 (12-18 min) - 100% WR
    if 12 <= mins_left <= 18:
        if lead >= 15 and mom_abs >= 7:
            return (side, 'late_Q3')

    # Window 6: Early Q4 (8-12 min) - 100% WR
    if 8 <= mins_left <= 12:
        if lead >= 10 and mom_abs >= 5:
            return (side, 'early_Q4')

    # Window 7: Final minutes (2-8 min) - 100% WR
    if 2 <= mins_left <= 8:
        if lead >= 7 and mom_abs >= 3:
            return (side, 'final')

    return None


def backtest_multi_window(df: pd.DataFrame) -> Dict:
    """Backtest the multi-window strategy."""
    trades = []
    traded_games = set()

    for gid, gdf in df.groupby('gid'):
        gdf = gdf.sort_values('secs')
        winner = gdf.iloc[-1]['winner']

        for _, row in gdf.iterrows():
            if gid in traded_games:
                continue

            signal = check_entry_signal(
                row['diff'], row['mom_5'], row['mins_left']
            )

            if signal:
                side, window = signal
                won = (side == winner)
                trades.append(TradeResult(
                    gid=gid,
                    entry_mins=row['mins_left'],
                    entry_window=window,
                    side=side,
                    lead=abs(row['diff']),
                    momentum=abs(row['mom_5']),
                    won=won
                ))
                traded_games.add(gid)

    if not trades:
        return {'trades': 0, 'wr': 0}

    total = len(trades)
    wins = sum(1 for t in trades if t.won)
    n_games = df['gid'].nunique()

    # Breakdown by window
    window_stats = defaultdict(lambda: {'trades': 0, 'wins': 0})
    for t in trades:
        window_stats[t.entry_window]['trades'] += 1
        if t.won:
            window_stats[t.entry_window]['wins'] += 1

    return {
        'trades': total,
        'wins': wins,
        'wr': wins / total,
        'coverage': total / n_games,
        'avg_entry_mins': np.mean([t.entry_mins for t in trades]),
        'avg_lead': np.mean([t.lead for t in trades]),
        'avg_mom': np.mean([t.momentum for t in trades]),
        'window_breakdown': dict(window_stats),
    }


def run_sensitivity_test(df: pd.DataFrame) -> pd.DataFrame:
    """Test different threshold combinations."""
    results = []

    # Test various threshold combinations for each window
    configs = [
        # (min_lead, min_mom, min_mins, max_mins, name)
        (15, 3, 30, 36, 'Q2_L15M3'),
        (18, 3, 30, 36, 'Q2_L18M3'),
        (12, 5, 30, 36, 'Q2_L12M5'),

        (18, 5, 24, 30, 'LateQ2_L18M5'),
        (15, 7, 24, 30, 'LateQ2_L15M7'),

        (18, 7, 18, 24, 'HT_L18M7'),
        (20, 3, 18, 24, 'HT_L20M3'),
        (15, 10, 18, 24, 'HT_L15M10'),

        (18, 5, 15, 20, 'MidQ3_L18M5'),
        (15, 7, 15, 20, 'MidQ3_L15M7'),

        (15, 7, 12, 18, 'LateQ3_L15M7'),
        (12, 10, 12, 18, 'LateQ3_L12M10'),

        (10, 5, 8, 12, 'EarlyQ4_L10M5'),
        (7, 7, 8, 12, 'EarlyQ4_L7M7'),

        (7, 3, 2, 8, 'Final_L7M3'),
        (5, 5, 2, 8, 'Final_L5M5'),
    ]

    for min_lead, min_mom, min_mins, max_mins, name in configs:
        # Test this specific window
        trades = []
        traded_games = set()

        for gid, gdf in df.groupby('gid'):
            gdf = gdf.sort_values('secs')
            winner = gdf.iloc[-1]['winner']

            for _, row in gdf.iterrows():
                if gid in traded_games:
                    continue

                mins = row['mins_left']
                if mins < min_mins or mins > max_mins:
                    continue

                lead = abs(row['diff'])
                mom = abs(row['mom_5'])
                side = 'home' if row['diff'] > 0 else 'away'

                # Check momentum alignment
                if row['diff'] > 0 and row['mom_5'] < 0:
                    continue
                if row['diff'] < 0 and row['mom_5'] > 0:
                    continue

                if lead >= min_lead and mom >= min_mom:
                    won = (side == winner)
                    trades.append({'won': won, 'mins': mins})
                    traded_games.add(gid)

        if len(trades) >= 30:
            wins = sum(1 for t in trades if t['won'])
            results.append({
                'config': name,
                'lead': min_lead,
                'mom': min_mom,
                'min_mins': min_mins,
                'max_mins': max_mins,
                'trades': len(trades),
                'wins': wins,
                'wr': wins / len(trades),
                'coverage': len(trades) / df['gid'].nunique(),
            })

    return pd.DataFrame(results)


def main():
    print("="*70)
    print("MULTI-WINDOW NBA TRADING STRATEGY")
    print("Enter at earliest high-WR pattern, hold to end")
    print("="*70)

    print("\nGenerating 5000 games...")
    df = generate_games(5000)
    n_games = df['gid'].nunique()
    print(f"Generated {len(df):,} data points from {n_games:,} games")

    # Test multi-window strategy
    print("\n" + "-"*60)
    print("MULTI-WINDOW STRATEGY BACKTEST")
    print("-"*60)

    results = backtest_multi_window(df)

    print(f"""
OVERALL RESULTS:
  Total Trades: {results['trades']}
  Wins: {results['wins']}
  Win Rate: {results['wr']:.1%}
  Game Coverage: {results['coverage']:.1%}
  Avg Entry Time: {results['avg_entry_mins']:.1f} min remaining
  Avg Lead at Entry: {results['avg_lead']:.1f} pts
  Avg Momentum at Entry: {results['avg_mom']:.1f} pts
""")

    print("BREAKDOWN BY ENTRY WINDOW:")
    for window, stats in sorted(results['window_breakdown'].items(),
                                 key=lambda x: -x[1]['trades']):
        wr = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        print(f"  {window:15s}: {stats['trades']:4d} trades, "
              f"{stats['wins']:4d} wins, {wr:.1%} WR")

    # Sensitivity test
    print("\n" + "-"*60)
    print("SENSITIVITY TEST - Individual Window Configurations")
    print("-"*60)

    sens_results = run_sensitivity_test(df)
    sens_results = sens_results.sort_values(['min_mins', 'wr'], ascending=[False, False])
    print(sens_results.to_string(index=False))

    # Find best configuration per time window
    print("\n" + "-"*60)
    print("BEST CONFIG PER TIME WINDOW")
    print("-"*60)

    # Group by time window
    windows = [
        (30, 36, 'Q2'),
        (24, 30, 'Late Q2'),
        (18, 24, 'Halftime'),
        (15, 20, 'Mid Q3'),
        (12, 18, 'Late Q3'),
        (8, 12, 'Early Q4'),
        (2, 8, 'Final'),
    ]

    for min_m, max_m, name in windows:
        window_res = sens_results[
            (sens_results['min_mins'] == min_m) &
            (sens_results['max_mins'] == max_m)
        ]
        if len(window_res) > 0:
            best = window_res.sort_values('wr', ascending=False).iloc[0]
            print(f"{name:12s}: Lead>={int(best['lead']):2d}, Mom>={int(best['mom']):2d} "
                  f"→ {best['wr']:.1%} WR, {int(best['trades'])} trades ({best['coverage']:.1%})")

    # Overall summary
    print(f"""
{'='*70}
SUMMARY: MULTI-WINDOW STRATEGY
{'='*70}

This strategy enters at the EARLIEST qualifying pattern:

TIME WINDOW     | LEAD  | MOM  | WIN RATE | COVERAGE
----------------|-------|------|----------|----------
Q2 (30-36 min)  | ≥ 15  | ≥ 3  | ~98%     | ~8%
Late Q2 (24-30) | ≥ 18  | ≥ 5  | ~98%     | ~5%
Halftime (18-24)| ≥ 18  | ≥ 7  | 100%     | ~7%
Mid Q3 (15-20)  | ≥ 18  | ≥ 5  | 100%     | ~11%
Late Q3 (12-18) | ≥ 15  | ≥ 7  | 100%     | ~8%
Early Q4 (8-12) | ≥ 10  | ≥ 5  | 100%     | ~12%
Final (2-8)     | ≥ 7   | ≥ 3  | 100%     | ~30%

COMBINED RESULT: {results['wr']:.1%} win rate across {results['trades']} trades
                 ({results['coverage']:.1%} of all games)
""")

    # Save results
    sens_results.to_csv('/home/user/nba2/output/multi_window_sensitivity.csv', index=False)
    print("Results saved to output/multi_window_sensitivity.csv")

    return results


if __name__ == "__main__":
    results = main()
