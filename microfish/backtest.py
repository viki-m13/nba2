"""
Microfish Backtesting Framework
=================================
Walk-forward validation for the pure rules-based strategy engine.
No ML, no AI — evaluates deterministic rules against historical data.

Anti-overfitting measures:
1. Walk-forward validation (expanding window, never look ahead)
2. Purge gap between calibration window and test window
3. All features lagged by FEATURE_LAG_DAYS
4. Cross-season consistency checks
5. Statistical significance tests on win rates
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    RESULTS_DIR, BACKTEST_SEASONS, MIN_AMERICAN_ODDS,
    WALK_FORWARD_TRAIN_DAYS, WALK_FORWARD_STEP_DAYS, PURGE_DAYS
)
from data_pipeline import (
    load_all_seasons, build_features, get_plus200_opportunities,
    american_to_decimal, american_to_implied_prob
)


def run_full_backtest(rules: dict = None) -> dict:
    """
    Run a complete backtest of the strategy rules.
    If no rules provided, loads from strategy_rules.json.
    """
    from strategy_engine import StrategyEngine

    print("=" * 70)
    print("MICROFISH MLB +200 STRATEGY BACKTEST")
    print("=" * 70)
    print(f"Strategy: Pure Rules Engine (no AI/ML)")
    print(f"Seasons: {BACKTEST_SEASONS}")
    print(f"Min odds: +{MIN_AMERICAN_ODDS}")
    print()

    # Load rules
    if rules is None:
        from microfish_dev import load_strategy_rules
        rules = load_strategy_rules()
        if not rules:
            return {}

    engine = StrategyEngine(rules)
    print("Strategy loaded:")
    print(engine.summary())
    print()

    # Load data
    print("\nSTEP 1: Loading historical data...")
    raw_data = load_all_seasons()
    if raw_data.empty:
        print("ERROR: No data loaded. Set THE_ODDS_API_KEY or add cached data.")
        return {}
    print(f"  Total games: {len(raw_data)}")

    # Build features
    print("\nSTEP 2: Building features (all lagged, no leakage)...")
    features = build_features(raw_data)
    print(f"  Feature rows: {len(features)}")

    # Extract opportunities
    print("\nSTEP 3: Extracting +200 underdog opportunities...")
    opportunities = get_plus200_opportunities(features)
    print(f"  Total +200 opportunities: {len(opportunities)}")

    if opportunities.empty:
        print("ERROR: No +200 opportunities found")
        return {}

    # Walk-forward backtest
    print("\nSTEP 4: Walk-forward validation...")
    results = run_backtest_with_rules(opportunities, engine)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print_results(results)

    # Overfitting checks
    print("\nOVERFITTING CHECKS:")
    overfit = check_overfitting(results)
    results['overfitting_checks'] = overfit

    # Save results
    save_results(results)

    return results


def run_backtest_with_rules(opportunities_df: pd.DataFrame,
                             engine) -> dict:
    """
    Walk-forward backtest with the rules engine.

    Walk-forward approach:
    - We don't "train" (no ML), but we use the walk-forward structure
      to ensure temporal discipline
    - For each test window, only evaluate games in that window
    - Features already enforce lag, so no leakage
    """
    opps = opportunities_df.sort_values('date').copy()

    if opps.empty:
        return {'total_bets': 0, 'accuracy': 0, 'roi': 0}

    min_date = pd.Timestamp(opps['date'].min())
    max_date = pd.Timestamp(opps['date'].max())

    # Skip the first WALK_FORWARD_TRAIN_DAYS as warm-up for features
    test_start = min_date + pd.Timedelta(days=WALK_FORWARD_TRAIN_DAYS + PURGE_DAYS)

    all_bets = []
    all_evaluations = []
    step = 0

    current_start = test_start
    while current_start < max_date:
        current_end = current_start + pd.Timedelta(days=WALK_FORWARD_STEP_DAYS)

        test_mask = (opps['date'] >= current_start) & (opps['date'] < current_end)
        test_df = opps[test_mask]

        if not test_df.empty:
            evaluated = engine.evaluate_batch(test_df)
            for _, row in evaluated.iterrows():
                row_dict = row.to_dict()
                row_dict['walk_forward_step'] = step
                all_evaluations.append(row_dict)
                if row_dict.get('recommendation') == 'BET':
                    all_bets.append(row_dict)

        current_start = current_end
        step += 1

    bets_df = pd.DataFrame(all_bets)
    all_evals_df = pd.DataFrame(all_evaluations)

    return calculate_results(bets_df, all_evals_df)


def calculate_results(bets_df: pd.DataFrame,
                       all_evals_df: pd.DataFrame) -> dict:
    """Calculate comprehensive backtest statistics."""
    if bets_df.empty:
        return {
            'total_bets': 0, 'wins': 0, 'losses': 0,
            'accuracy': 0, 'roi': 0, 'avg_odds': 0,
            'daily_frequency': 0, 'season_results': {},
            'total_opportunities': len(all_evals_df),
        }

    total_bets = len(bets_df)
    wins = int(bets_df['bet_won'].sum()) if 'bet_won' in bets_df.columns else 0
    losses = total_bets - wins
    accuracy = wins / total_bets if total_bets > 0 else 0

    # ROI
    total_wagered = total_bets
    total_payout = 0
    if 'bet_won' in bets_df.columns and 'bet_odds' in bets_df.columns:
        for _, bet in bets_df.iterrows():
            if bet['bet_won'] == 1:
                total_payout += american_to_decimal(bet['bet_odds'])
    roi = ((total_payout - total_wagered) / total_wagered * 100
           if total_wagered > 0 else 0)

    # Frequency
    unique_dates = 0
    date_range = 0
    daily_frequency = 0
    avg_bets_per_day = 0
    if 'date' in bets_df.columns:
        unique_dates = bets_df['date'].nunique()
        date_range = (bets_df['date'].max() - bets_df['date'].min()).days + 1
        daily_frequency = unique_dates / date_range if date_range > 0 else 0
        avg_bets_per_day = total_bets / date_range if date_range > 0 else 0

    # Season breakdown
    season_results = {}
    if 'season' in bets_df.columns and 'bet_won' in bets_df.columns:
        for season in sorted(bets_df['season'].unique()):
            s = bets_df[bets_df['season'] == season]
            s_wins = int(s['bet_won'].sum())
            season_results[int(season)] = {
                'bets': len(s),
                'wins': s_wins,
                'accuracy': s_wins / len(s) if len(s) > 0 else 0,
            }

    avg_odds = bets_df['bet_odds'].mean() if 'bet_odds' in bets_df.columns else 0

    # Store bets for detailed analysis
    bets_list = []
    if not bets_df.empty:
        for _, b in bets_df.iterrows():
            bets_list.append({
                'date': str(b.get('date', '')),
                'team': b.get('bet_team', ''),
                'odds': float(b.get('bet_odds', 0)),
                'won': int(b.get('bet_won', 0)),
                'weight': float(b.get('total_weight', 0)),
                'rules': b.get('rules_triggered', ''),
            })

    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'accuracy': accuracy,
        'roi': roi,
        'avg_odds': float(avg_odds),
        'unique_bet_dates': unique_dates,
        'date_range_days': date_range,
        'daily_frequency': daily_frequency,
        'avg_bets_per_day': avg_bets_per_day,
        'season_results': season_results,
        'total_opportunities': len(all_evals_df),
        'selectivity': total_bets / len(all_evals_df) if len(all_evals_df) > 0 else 0,
        'bets_detail': bets_list,
    }


def check_overfitting(results: dict) -> dict:
    """Run overfitting checks."""
    checks = {}

    # Season consistency
    season_results = results.get('season_results', {})
    if len(season_results) >= 2:
        accs = [s['accuracy'] for s in season_results.values() if s['bets'] >= 3]
        if accs:
            checks['season_acc_mean'] = float(np.mean(accs))
            checks['season_acc_std'] = float(np.std(accs))
            checks['season_consistent'] = np.std(accs) < 0.20
            print(f"  Season accuracy: mean={np.mean(accs):.3f}, "
                  f"std={np.std(accs):.3f} "
                  f"({'PASS' if checks['season_consistent'] else 'WARN'})")

    # Sample size
    total_bets = results.get('total_bets', 0)
    accuracy = results.get('accuracy', 0)
    checks['sufficient_sample'] = total_bets >= 30
    print(f"  Sample size: {total_bets} bets "
          f"({'PASS' if checks['sufficient_sample'] else 'WARN: <30'})")

    # Frequency check
    freq = results.get('daily_frequency', 0)
    checks['adequate_frequency'] = freq >= 0.30
    print(f"  Daily frequency: {freq:.1%} "
          f"({'PASS' if checks['adequate_frequency'] else 'WARN: <30%'})")

    # Statistical significance (binomial test against breakeven)
    if total_bets > 0:
        wins = results.get('wins', 0)
        # At +200, breakeven is 33.3%
        from scipy.stats import binomtest
        try:
            result = binomtest(wins, total_bets, 0.333, alternative='greater')
            p_value = result.pvalue
            checks['stat_sig_p_value'] = float(p_value)
            checks['stat_significant'] = p_value < 0.05
            print(f"  Statistical significance (vs 33.3%): p={p_value:.4f} "
                  f"({'PASS' if checks['stat_significant'] else 'NOT SIG'})")
        except Exception:
            checks['stat_sig_p_value'] = None

    return checks


def print_results(results: dict):
    """Print formatted backtest results."""
    print(f"\n  Total bets placed: {results['total_bets']}")
    print(f"  Wins: {results.get('wins', 0)}")
    print(f"  Losses: {results.get('losses', 0)}")
    print(f"  ACCURACY: {results['accuracy']:.1%}")
    print(f"  ROI: {results['roi']:.1f}%")
    print(f"  Average odds: +{results['avg_odds']:.0f}")
    print(f"  Unique betting days: {results.get('unique_bet_dates', 0)}")
    print(f"  Date range: {results.get('date_range_days', 0)} days")
    print(f"  Daily frequency: {results.get('daily_frequency', 0):.1%}")
    print(f"  Avg bets per day: {results.get('avg_bets_per_day', 0):.2f}")
    print(f"  Selectivity: {results.get('selectivity', 0):.3%}")
    print(f"  Total opportunities analyzed: {results.get('total_opportunities', 0)}")

    if 'season_results' in results:
        print(f"\n  BY SEASON:")
        for season, sr in sorted(results.get('season_results', {}).items()):
            print(f"    {season}: {sr['bets']} bets, {sr['wins']} wins, "
                  f"{sr['accuracy']:.1%} accuracy")


def save_results(results: dict):
    """Save backtest results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Clean for JSON serialization
    clean = {}
    for k, v in results.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, dict):
            clean[k] = {
                str(k2): (float(v2) if isinstance(v2, (np.floating,)) else
                          int(v2) if isinstance(v2, (np.integer,)) else v2)
                for k2, v2 in v.items()
            }
        else:
            clean[k] = v

    summary_file = os.path.join(RESULTS_DIR, 'backtest_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"\n  Results saved to {summary_file}")
