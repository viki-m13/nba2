"""
Predictive Pattern Discovery for NBA Game Outcomes

Goal: Find in-game patterns that predict final winner with 90%+ accuracy
Approach: Hold position until game end, like holding a stock

Key insight: We need to find NON-OBVIOUS patterns that predict outcomes
- Not just "team up 20 wins" (obvious, no edge)
- Find momentum/efficiency patterns that predict comebacks or holds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


def generate_games_with_outcomes(n_games: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate games with realistic outcomes and patterns."""
    np.random.seed(seed)
    all_data = []

    for g in range(n_games):
        game_id = f"G{g:05d}"

        # Team strengths (affects final outcome)
        home_strength = np.random.normal(0, 10)

        # Game will be decided by strength + randomness
        # But in-game patterns should correlate with outcome

        home_score, away_score = 0, 0
        h_history, a_history = [], []

        # Track quarters
        q_home_pts = [0, 0, 0, 0]
        q_away_pts = [0, 0, 0, 0]

        states = []

        for secs in range(0, 48*60 + 1, 30):  # Every 30 seconds
            quarter = min(secs // (12*60) + 1, 4)
            q_idx = quarter - 1
            mins_remaining = max((48*60 - secs) / 60, 0)
            mins_elapsed = secs / 60

            # Scoring influenced by team strength
            # Also add "momentum" - hot/cold streaks
            recent_home_rate = np.mean(h_history[-10:]) if len(h_history) >= 10 else 0.5
            recent_away_rate = np.mean(a_history[-10:]) if len(a_history) >= 10 else 0.5

            # Momentum persistence
            h_momentum = 0.1 * (recent_home_rate - 0.5)
            a_momentum = 0.1 * (recent_away_rate - 0.5)

            # Base rates with strength and momentum
            h_rate = 0.55 * (1 + home_strength/100 + h_momentum)
            a_rate = 0.55 * (1 - home_strength/100 + a_momentum)

            h_pts = np.random.poisson(max(h_rate, 0.1))
            a_pts = np.random.poisson(max(a_rate, 0.1))

            home_score += h_pts
            away_score += a_pts
            q_home_pts[q_idx] += h_pts
            q_away_pts[q_idx] += a_pts

            h_history.append(h_pts)
            a_history.append(a_pts)

            # Calculate features
            score_diff = home_score - away_score

            # Momentum features
            mom_1min = sum(h_history[-2:]) - sum(a_history[-2:]) if len(h_history) >= 2 else 0
            mom_2min = sum(h_history[-4:]) - sum(a_history[-4:]) if len(h_history) >= 4 else 0
            mom_5min = sum(h_history[-10:]) - sum(a_history[-10:]) if len(h_history) >= 10 else 0

            # Quarter performance
            q_diff = q_home_pts[q_idx] - q_away_pts[q_idx] if secs > 0 else 0

            # Efficiency proxy (scoring rate)
            h_eff = sum(h_history[-10:]) / 10 if len(h_history) >= 10 else 0.5
            a_eff = sum(a_history[-10:]) / 10 if len(a_history) >= 10 else 0.5

            states.append({
                'game_id': game_id,
                'secs': secs,
                'quarter': quarter,
                'mins_remaining': mins_remaining,
                'mins_elapsed': mins_elapsed,
                'home_score': home_score,
                'away_score': away_score,
                'score_diff': score_diff,
                'abs_diff': abs(score_diff),
                'mom_1min': mom_1min,
                'mom_2min': mom_2min,
                'mom_5min': mom_5min,
                'q_diff': q_diff,  # Current quarter differential
                'h_eff': h_eff,
                'a_eff': a_eff,
                'eff_diff': h_eff - a_eff,
                'home_strength': home_strength,  # Hidden, for validation
            })

        # Determine final winner
        final_home = states[-1]['home_score']
        final_away = states[-1]['away_score']
        winner = 'home' if final_home > final_away else 'away'

        # Add winner to all states
        for s in states:
            s['winner'] = winner
            s['home_wins'] = 1 if winner == 'home' else 0

        all_data.extend(states)

    return pd.DataFrame(all_data)


def analyze_pattern_winrates(df: pd.DataFrame) -> Dict:
    """
    Analyze win rates for various pattern combinations.
    Find patterns that predict winner with 90%+ accuracy.
    """
    results = {}

    # Pattern 1: Leading team with positive momentum
    # "Team is ahead AND has momentum"
    for lead in [3, 5, 7, 10, 12, 15]:
        for mom in [3, 5, 7, 10]:
            for mins_left in [12, 10, 8, 6, 5, 4, 3]:
                mask = (
                    (df['score_diff'] >= lead) &
                    (df['mom_2min'] >= mom) &
                    (df['mins_remaining'] <= mins_left) &
                    (df['mins_remaining'] >= mins_left - 2)
                )
                subset = df[mask]
                if len(subset) > 20:
                    wr = subset['home_wins'].mean()
                    results[f'home_lead{lead}_mom{mom}_min{mins_left}'] = {
                        'n': len(subset),
                        'wr': wr,
                        'side': 'home'
                    }

                # Away version
                mask_away = (
                    (df['score_diff'] <= -lead) &
                    (df['mom_2min'] <= -mom) &
                    (df['mins_remaining'] <= mins_left) &
                    (df['mins_remaining'] >= mins_left - 2)
                )
                subset_away = df[mask_away]
                if len(subset_away) > 20:
                    wr_away = 1 - subset_away['home_wins'].mean()
                    results[f'away_lead{lead}_mom{mom}_min{mins_left}'] = {
                        'n': len(subset_away),
                        'wr': wr_away,
                        'side': 'away'
                    }

    # Pattern 2: Comeback pattern - trailing but hot
    for deficit in [5, 7, 10]:
        for mom in [6, 8, 10, 12]:
            for mins_left in [10, 8, 6, 5]:
                # Home trailing but has momentum
                mask = (
                    (df['score_diff'] >= -deficit) &
                    (df['score_diff'] <= -3) &
                    (df['mom_2min'] >= mom) &
                    (df['mins_remaining'] <= mins_left) &
                    (df['mins_remaining'] >= mins_left - 2)
                )
                subset = df[mask]
                if len(subset) > 20:
                    wr = subset['home_wins'].mean()
                    results[f'home_comeback_def{deficit}_mom{mom}_min{mins_left}'] = {
                        'n': len(subset),
                        'wr': wr,
                        'side': 'home'
                    }

    # Pattern 3: Quarter dominance
    # "Team dominated this quarter"
    for q in [3, 4]:
        for q_margin in [6, 8, 10, 12]:
            for lead in [-5, 0, 5]:
                mask = (
                    (df['quarter'] == q) &
                    (df['q_diff'] >= q_margin) &
                    (df['score_diff'] >= lead) &
                    (df['mins_remaining'] >= 2) &
                    (df['mins_remaining'] <= 6)
                )
                subset = df[mask]
                if len(subset) > 20:
                    wr = subset['home_wins'].mean()
                    results[f'home_q{q}dom{q_margin}_lead{lead}'] = {
                        'n': len(subset),
                        'wr': wr,
                        'side': 'home'
                    }

    # Pattern 4: Efficiency edge
    for eff_edge in [0.3, 0.4, 0.5]:
        for mins_left in [8, 6, 5, 4]:
            mask = (
                (df['eff_diff'] >= eff_edge) &
                (df['mins_remaining'] <= mins_left) &
                (df['mins_remaining'] >= mins_left - 2) &
                (df['abs_diff'] <= 10)  # Close game
            )
            subset = df[mask]
            if len(subset) > 20:
                wr = subset['home_wins'].mean()
                results[f'home_eff{eff_edge}_min{mins_left}'] = {
                    'n': len(subset),
                    'wr': wr,
                    'side': 'home'
                }

    return results


def find_high_winrate_patterns(results: Dict, min_wr: float = 0.85, min_n: int = 50) -> List:
    """Find patterns with win rate >= threshold."""
    high_wr = []
    for pattern, data in results.items():
        if data['wr'] >= min_wr and data['n'] >= min_n:
            high_wr.append({
                'pattern': pattern,
                'win_rate': data['wr'],
                'n': data['n'],
                'side': data['side']
            })
    return sorted(high_wr, key=lambda x: (-x['win_rate'], -x['n']))


def backtest_pattern(df: pd.DataFrame,
                     min_lead: int,
                     min_mom: int,
                     max_mins: float,
                     min_mins: float = 2) -> Dict:
    """
    Backtest a specific pattern: leading team with momentum.
    Hold until game end.
    """
    trades = []
    traded_games = set()

    for game_id, gdf in df.groupby('game_id'):
        gdf = gdf.sort_values('secs')
        winner = gdf.iloc[-1]['winner']

        for _, row in gdf.iterrows():
            if game_id in traded_games:
                continue

            mins = row['mins_remaining']
            if mins > max_mins or mins < min_mins:
                continue

            diff = row['score_diff']
            mom = row['mom_2min']

            # Check for signal
            side = None
            if diff >= min_lead and mom >= min_mom:
                side = 'home'
            elif diff <= -min_lead and mom <= -min_mom:
                side = 'away'

            if side:
                won = (side == winner)
                trades.append({
                    'game_id': game_id,
                    'side': side,
                    'won': won,
                    'mins_remaining': mins,
                    'lead': abs(diff),
                    'momentum': abs(mom),
                })
                traded_games.add(game_id)

    if not trades:
        return {'trades': 0, 'wr': 0, 'coverage': 0}

    wins = sum(t['won'] for t in trades)
    total = len(trades)
    n_games = df['game_id'].nunique()

    return {
        'trades': total,
        'wins': wins,
        'wr': wins / total,
        'coverage': total / n_games,
        'avg_lead': np.mean([t['lead'] for t in trades]),
        'avg_mom': np.mean([t['momentum'] for t in trades]),
        'avg_mins': np.mean([t['mins_remaining'] for t in trades]),
    }


def grid_search_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Search for high win rate patterns."""
    results = []

    # Lead + Momentum + Time combinations
    for lead in [3, 5, 7, 10, 12]:
        for mom in [3, 5, 7, 10]:
            for max_mins in [12, 10, 8, 6, 5, 4]:
                res = backtest_pattern(df, lead, mom, max_mins)
                if res['trades'] >= 30:
                    results.append({
                        'lead': lead,
                        'mom': mom,
                        'max_mins': max_mins,
                        **res
                    })

    return pd.DataFrame(results)


def main():
    print("="*70)
    print("PREDICTIVE PATTERN DISCOVERY")
    print("Finding patterns that predict game outcome with 90%+ accuracy")
    print("="*70)

    print("\nGenerating 5000 games with realistic outcomes...")
    df = generate_games_with_outcomes(n_games=5000)
    n_games = df['game_id'].nunique()
    print(f"Generated {len(df):,} data points from {n_games:,} games")

    # Analyze patterns
    print("\n" + "-"*60)
    print("ANALYZING PATTERN WIN RATES")
    print("-"*60)

    pattern_results = analyze_pattern_winrates(df)
    high_wr_patterns = find_high_winrate_patterns(pattern_results, min_wr=0.85, min_n=50)

    print(f"\nFound {len(high_wr_patterns)} patterns with 85%+ win rate and 50+ samples")

    if high_wr_patterns:
        print("\nTop 20 patterns:")
        for i, p in enumerate(high_wr_patterns[:20]):
            print(f"  {i+1}. {p['pattern']}: {p['win_rate']:.1%} ({p['n']} samples)")

    # Grid search
    print("\n" + "-"*60)
    print("GRID SEARCH: Lead + Momentum + Time")
    print("-"*60)

    grid_results = grid_search_patterns(df)

    # Filter for 90%+ win rate
    wr90 = grid_results[grid_results['wr'] >= 0.90].sort_values('coverage', ascending=False)
    print(f"\n90%+ Win Rate Patterns: {len(wr90)}")
    if len(wr90) > 0:
        print(wr90.head(15).to_string(index=False))

    # Filter for 85%+ with good coverage
    wr85_cov = grid_results[(grid_results['wr'] >= 0.85) & (grid_results['coverage'] >= 0.3)]
    wr85_cov = wr85_cov.sort_values('wr', ascending=False)
    print(f"\n85%+ Win Rate with 30%+ Coverage: {len(wr85_cov)}")
    if len(wr85_cov) > 0:
        print(wr85_cov.head(15).to_string(index=False))

    # Best overall
    if len(grid_results) > 0:
        # Score: WR * coverage
        grid_results['score'] = grid_results['wr'] * grid_results['coverage']
        best = grid_results.sort_values('score', ascending=False).iloc[0]

        print(f"\n{'='*70}")
        print("BEST PATTERN FOR HOLD-TO-END TRADING")
        print("="*70)
        print(f"""
ENTRY RULE:
  Leading team (home or away) must have:
  - Lead >= {int(best['lead'])} points
  - Momentum (2min) >= {int(best['mom'])} points in their favor
  - Time remaining <= {best['max_mins']:.0f} minutes

ACTION:
  - Bet on the leading team with momentum
  - HOLD UNTIL GAME END

RESULTS:
  - Win Rate: {best['wr']:.1%}
  - Total Trades: {int(best['trades'])}
  - Game Coverage: {best['coverage']:.1%}
  - Avg Lead at Entry: {best['avg_lead']:.1f} pts
  - Avg Momentum at Entry: {best['avg_mom']:.1f} pts
  - Avg Time Remaining: {best['avg_mins']:.1f} min
""")

        # Also show the 90%+ option
        if len(wr90) > 0:
            best90 = wr90.iloc[0]
            print(f"""
ALTERNATIVE: 90%+ WIN RATE PATTERN
  - Lead >= {int(best90['lead'])} pts
  - Momentum >= {int(best90['mom'])} pts
  - Time <= {best90['max_mins']:.0f} min
  - Win Rate: {best90['wr']:.1%}
  - Coverage: {best90['coverage']:.1%}
""")

    # Save results
    grid_results.to_csv('/home/user/nba2/output/pattern_search_results.csv', index=False)
    print(f"\nResults saved to output/pattern_search_results.csv")

    return grid_results


if __name__ == "__main__":
    results = main()
