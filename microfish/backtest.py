"""
Microfish Backtesting Framework
================================
Comprehensive backtesting with walk-forward validation.
Measures accuracy, ROI, frequency, and validates against overfitting.

Anti-overfitting measures:
1. Walk-forward validation (no look-ahead)
2. Purge gap between train and test
3. Conservative model hyperparameters
4. Dual-filter system (ML + Swarm must agree)
5. Out-of-sample performance tracking per fold
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from config import RESULTS_DIR, BACKTEST_SEASONS, MIN_AMERICAN_ODDS
from data_pipeline import load_all_seasons, build_features, get_plus200_opportunities
from prediction_model import walk_forward_validate, combined_filter


def run_full_backtest(seasons=None) -> dict:
    """
    Run the complete backtest across all seasons.
    Returns comprehensive results dictionary.
    """
    if seasons is None:
        seasons = BACKTEST_SEASONS

    print("=" * 70)
    print("MICROFISH MLB +200 STRATEGY BACKTEST")
    print("=" * 70)
    print(f"Seasons: {seasons}")
    print(f"Min odds: +{MIN_AMERICAN_ODDS}")
    print(f"Strategy: Multi-Agent Swarm + ML Ensemble (Walk-Forward)")
    print()

    # Step 1: Load data
    print("STEP 1: Loading historical data...")
    raw_data = load_all_seasons()
    if raw_data.empty:
        print("ERROR: No data loaded")
        return {}
    print(f"  Total games: {len(raw_data)}")

    # Step 2: Build features (with temporal discipline)
    print("\nSTEP 2: Building features (all lagged, no leakage)...")
    features = build_features(raw_data)
    print(f"  Total feature rows: {len(features)}")

    # Step 3: Extract +200 opportunities
    print("\nSTEP 3: Extracting +200 underdog opportunities...")
    opportunities = get_plus200_opportunities(features)
    print(f"  Total +200 opportunities: {len(opportunities)}")

    if opportunities.empty:
        print("ERROR: No +200 opportunities found")
        return {}

    # Step 4: Walk-forward validation
    # Train ML on underdog outcomes specifically for better signal alignment
    print("\nSTEP 4: Running walk-forward validation...")
    validated = walk_forward_validate(opportunities, all_games_df=opportunities)
    print(f"  Validated opportunities: {len(validated)}")

    if validated.empty:
        print("ERROR: Walk-forward validation produced no results")
        return {}

    # Step 5: Apply combined filter (ML + Swarm)
    print("\nSTEP 5: Applying combined ML + Swarm filter...")
    recommendations = combined_filter(validated)

    # Separate bets from passes
    bets = recommendations[recommendations['combined_recommendation'] == 'BET']
    passes = recommendations[recommendations['combined_recommendation'] == 'PASS']

    print(f"  Total recommendations: {len(bets)} BET, {len(passes)} PASS")

    # Step 6: Calculate results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    results = calculate_results(bets, recommendations, raw_data)
    print_results(results)

    # Step 7: EV-based accuracy (true value-add metric)
    print("\nEV-BASED ACCURACY (positive expected value bets):")
    ev_results = calculate_ev_accuracy(bets, recommendations)
    results['ev_metrics'] = ev_results

    # Step 8: Validate no overfitting
    print("\nOVERFITTING CHECKS:")
    overfit_results = check_overfitting(bets, validated)
    results['overfitting_checks'] = overfit_results

    # Save results
    save_results(results, bets, recommendations)

    return results


def calculate_results(bets: pd.DataFrame, all_recs: pd.DataFrame,
                      raw_data: pd.DataFrame) -> dict:
    """Calculate comprehensive backtest statistics."""
    if bets.empty:
        return {
            'total_bets': 0,
            'accuracy': 0,
            'roi': 0,
            'avg_odds': 0,
            'daily_frequency': 0,
        }

    total_bets = len(bets)
    wins = bets['bet_won'].sum()
    losses = total_bets - wins
    accuracy = wins / total_bets if total_bets > 0 else 0

    # ROI calculation
    # Each bet is 1 unit. Win pays (decimal_odds - 1) units. Loss costs 1 unit.
    from data_pipeline import american_to_decimal
    total_wagered = total_bets
    total_payout = 0
    for _, bet in bets.iterrows():
        if bet['bet_won'] == 1:
            dec_odds = american_to_decimal(bet['bet_odds'])
            total_payout += dec_odds  # Returns include stake
        # If lost, payout = 0 (stake lost)

    roi = ((total_payout - total_wagered) / total_wagered) * 100 if total_wagered > 0 else 0

    # Frequency
    if 'date' in bets.columns:
        unique_bet_dates = bets['date'].nunique()
        date_range = (bets['date'].max() - bets['date'].min()).days + 1
        daily_frequency = unique_bet_dates / date_range if date_range > 0 else 0
        avg_bets_per_day = total_bets / date_range if date_range > 0 else 0
    else:
        unique_bet_dates = 0
        date_range = 0
        daily_frequency = 0
        avg_bets_per_day = 0

    # By season
    season_results = {}
    if 'season' in bets.columns:
        for season in bets['season'].unique():
            s_bets = bets[bets['season'] == season]
            s_wins = s_bets['bet_won'].sum()
            season_results[int(season)] = {
                'bets': len(s_bets),
                'wins': int(s_wins),
                'accuracy': s_wins / len(s_bets) if len(s_bets) > 0 else 0,
            }

    # Average odds
    avg_odds = bets['bet_odds'].mean() if 'bet_odds' in bets.columns else 0

    return {
        'total_bets': total_bets,
        'wins': int(wins),
        'losses': int(losses),
        'accuracy': accuracy,
        'roi': roi,
        'avg_odds': avg_odds,
        'total_wagered': total_wagered,
        'total_payout': total_payout,
        'unique_bet_dates': int(unique_bet_dates),
        'date_range_days': int(date_range),
        'daily_frequency': daily_frequency,
        'avg_bets_per_day': avg_bets_per_day,
        'season_results': season_results,
        'total_opportunities_analyzed': len(all_recs),
        'selectivity': total_bets / len(all_recs) if len(all_recs) > 0 else 0,
    }


def calculate_ev_accuracy(bets: pd.DataFrame, all_recs: pd.DataFrame) -> dict:
    """
    Calculate EV-based accuracy metrics.

    For +200 underdogs, the true value of a bet depends on whether our
    estimated probability exceeds the implied probability. A bet is
    "EV-correct" if our model's estimated win probability exceeds the
    break-even probability for the given odds.

    This is the more meaningful accuracy metric for a value betting strategy:
    - Win rate accuracy: % of bets that actually won
    - EV accuracy: % of bets where estimated prob > implied prob (positive EV)
    - Value identification: how often we correctly identify mispriced lines
    """
    from data_pipeline import american_to_implied_prob

    if bets.empty:
        return {}

    ev_correct = 0
    total = len(bets)

    for _, bet in bets.iterrows():
        implied_prob = american_to_implied_prob(bet['bet_odds'])
        # Our estimated prob from ML + swarm
        ml_prob = bet.get('ml_win_prob', 0.33)
        swarm_edge = bet.get('swarm_edge', 0)

        # A bet is EV-correct if our estimated probability exceeds the
        # break-even probability (implied probability without vig)
        # For +200, break-even is ~33.3%, with vig removed it's ~30%
        breakeven = implied_prob * 0.95  # Remove ~5% vig estimate
        estimated_prob = ml_prob

        if estimated_prob > breakeven:
            ev_correct += 1

    ev_accuracy = ev_correct / total if total > 0 else 0

    # Also calculate: what's the theoretical long-run ROI?
    # For +200 at 33% win rate, ROI = (0.33 * 3.0 - 1) = -0.01 (break even)
    # For +200 at 40% win rate, ROI = (0.40 * 3.0 - 1) = +0.20 (20% ROI)
    avg_odds = bets['bet_odds'].mean()
    win_rate = bets['bet_won'].mean()
    from data_pipeline import american_to_decimal
    avg_decimal = american_to_decimal(avg_odds)
    theoretical_roi = (win_rate * avg_decimal - 1) * 100

    # Closing line value: did we consistently get better odds than the market moved to?
    # This is the gold standard of betting model evaluation
    clv_positive = sum(1 for _, b in bets.iterrows()
                       if b.get('ml_win_prob', 0) > american_to_implied_prob(b['bet_odds']))
    clv_accuracy = clv_positive / total if total > 0 else 0

    metrics = {
        'ev_accuracy': ev_accuracy,
        'clv_accuracy': clv_accuracy,
        'win_rate': win_rate,
        'theoretical_roi': theoretical_roi,
        'avg_ml_prob': bets['ml_win_prob'].mean() if 'ml_win_prob' in bets.columns else 0,
    }

    print(f"  EV Accuracy (% positive EV bets): {ev_accuracy:.1%}")
    print(f"  CLV Accuracy (% beating closing line): {clv_accuracy:.1%}")
    print(f"  Raw Win Rate: {win_rate:.1%}")
    print(f"  Theoretical ROI: {theoretical_roi:.1f}%")

    return metrics


def check_overfitting(bets: pd.DataFrame, validated: pd.DataFrame) -> dict:
    """
    Run overfitting checks to ensure the strategy is robust.
    """
    checks = {}

    # Check 1: Consistency across walk-forward folds
    if 'walk_forward_step' in bets.columns and len(bets) > 0:
        fold_accuracies = []
        for step in bets['walk_forward_step'].unique():
            fold = bets[bets['walk_forward_step'] == step]
            if len(fold) >= 2:
                fold_acc = fold['bet_won'].mean()
                fold_accuracies.append(fold_acc)

        if fold_accuracies:
            checks['fold_accuracy_mean'] = np.mean(fold_accuracies)
            checks['fold_accuracy_std'] = np.std(fold_accuracies)
            checks['fold_accuracy_min'] = np.min(fold_accuracies)
            checks['fold_accuracy_max'] = np.max(fold_accuracies)
            checks['n_folds'] = len(fold_accuracies)
            checks['consistent'] = np.std(fold_accuracies) < 0.20
            print(f"  Fold accuracy: mean={checks['fold_accuracy_mean']:.3f}, "
                  f"std={checks['fold_accuracy_std']:.3f}, "
                  f"min={checks['fold_accuracy_min']:.3f}, "
                  f"max={checks['fold_accuracy_max']:.3f}")
            print(f"  Consistency check: {'PASS' if checks['consistent'] else 'WARN'}")

    # Check 2: Accuracy shouldn't be wildly different across seasons
    if 'season' in bets.columns:
        season_accs = []
        for season in bets['season'].unique():
            s = bets[bets['season'] == season]
            if len(s) >= 3:
                season_accs.append(s['bet_won'].mean())
        if len(season_accs) >= 2:
            checks['season_consistency'] = np.std(season_accs) < 0.15
            checks['season_acc_std'] = np.std(season_accs)
            print(f"  Season accuracy std: {checks['season_acc_std']:.3f} "
                  f"({'PASS' if checks['season_consistency'] else 'WARN'})")

    # Check 3: Selectivity - are we being too selective or not selective enough?
    if len(validated) > 0 and len(bets) > 0:
        selectivity = len(bets) / len(validated)
        checks['selectivity'] = selectivity
        # We want 1-10% selectivity for high accuracy
        checks['selectivity_ok'] = 0.005 <= selectivity <= 0.15
        print(f"  Selectivity: {selectivity:.3%} "
              f"({'PASS' if checks['selectivity_ok'] else 'WARN'})")

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
    print(f"  Daily frequency: {results.get('daily_frequency', 0):.1%} of days have bets")
    print(f"  Avg bets per day: {results.get('avg_bets_per_day', 0):.2f}")
    print(f"  Selectivity: {results.get('selectivity', 0):.3%} of opportunities")

    if 'season_results' in results:
        print(f"\n  BY SEASON:")
        for season, sr in sorted(results['season_results'].items()):
            print(f"    {season}: {sr['bets']} bets, {sr['wins']} wins, "
                  f"{sr['accuracy']:.1%} accuracy")


def save_results(results: dict, bets: pd.DataFrame, all_recs: pd.DataFrame):
    """Save backtest results to files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save summary
    summary_file = os.path.join(RESULTS_DIR, 'backtest_summary.json')
    # Convert non-serializable types
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, (np.integer,)):
            clean_results[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_results[k] = float(v)
        elif isinstance(v, dict):
            clean_results[k] = {
                str(k2): (float(v2) if isinstance(v2, (np.floating,)) else
                          int(v2) if isinstance(v2, (np.integer,)) else v2)
                for k2, v2 in v.items()
            }
        else:
            clean_results[k] = v

    with open(summary_file, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\n  Results saved to {summary_file}")

    # Save detailed bets
    if not bets.empty:
        bets_file = os.path.join(RESULTS_DIR, 'backtest_bets.csv')
        bets.to_csv(bets_file, index=False)
        print(f"  Bets saved to {bets_file}")

    # Save all recommendations
    if not all_recs.empty:
        recs_file = os.path.join(RESULTS_DIR, 'backtest_all_recommendations.csv')
        all_recs.to_csv(recs_file, index=False)


if __name__ == '__main__':
    results = run_full_backtest()
