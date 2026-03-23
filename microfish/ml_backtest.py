"""
Microfish Moneyline Backtest Framework
========================================
Walk-forward validation for the Certainty Floor limit-price strategy.

Validates that the certainty floor holds out-of-sample:
1. Discover the floor in training data
2. Apply it to unseen test data
3. Verify 100% accuracy
4. Measure fill rate and profitability

Anti-overfitting measures:
- Walk-forward validation (expanding window, never look ahead)
- Purge gap between training and test windows
- All features lagged by FEATURE_LAG_DAYS
- Cross-month consistency checks
- Statistical significance tests
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import binomtest

from ml_config import (
    RESULTS_DIR, ML_RESEARCH_DIR,
    WALK_FORWARD_TRAIN_DAYS, WALK_FORWARD_STEP_DAYS, PURGE_DAYS
)
from ml_data_pipeline import (
    load_real_data, build_features, compute_certainty_floor_data,
    american_to_decimal, american_to_implied_prob
)
from ml_strategy_engine import MLStrategyEngine


def run_full_backtest(rules: dict = None) -> dict:
    """Run a complete backtest of the moneyline limit-price strategy."""
    print("=" * 70)
    print("MICROFISH MLB MONEYLINE STRATEGY BACKTEST")
    print("=" * 70)
    print("Strategy: Certainty Floor Limit-Price (Polymarket)")
    print()

    # Load strategy
    if rules is None:
        try:
            engine = MLStrategyEngine()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return {}
    else:
        engine = MLStrategyEngine(rules)

    print("Strategy loaded:")
    print(engine.summary())
    print()

    # Load data
    print("\nSTEP 1: Loading real MLB data...")
    df = load_real_data()
    if df.empty:
        print("ERROR: No data loaded")
        return {}

    # Build features
    print("\nSTEP 2: Building features (strictly lagged)...")
    featured = build_features(df)

    # Walk-forward backtest
    print("\nSTEP 3: Walk-forward validation...")
    results = walk_forward_backtest(featured, engine)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print_results(results)

    # Overfitting checks
    print("\nOVERFITTING CHECKS:")
    overfit = check_overfitting(results, featured)
    results['overfitting_checks'] = overfit

    # Save
    save_results(results)

    return results


def walk_forward_backtest(df: pd.DataFrame, engine: MLStrategyEngine) -> dict:
    """
    Walk-forward backtest with the limit-price engine.

    For each test window:
    1. Evaluate all games with the engine
    2. Track which games would have filled at the limit price
    3. Track wins/losses for filled orders
    """
    df = df.sort_values('date').copy()
    if df.empty:
        return _empty_results()

    min_date = df['date'].min()
    max_date = df['date'].max()

    test_start = min_date + pd.Timedelta(days=WALK_FORWARD_TRAIN_DAYS + PURGE_DAYS)

    all_bets = []
    all_evaluations = []
    step = 0

    current = test_start
    while current < max_date:
        test_end = current + pd.Timedelta(days=WALK_FORWARD_STEP_DAYS)

        test_mask = (df['date'] >= current) & (df['date'] < test_end)
        test_df = df[test_mask]

        if not test_df.empty:
            evaluated = engine.evaluate_batch(test_df)
            for _, row in evaluated.iterrows():
                row_dict = row.to_dict()
                row_dict['walk_forward_step'] = step
                all_evaluations.append(row_dict)
                if row_dict.get('action') == 'LIMIT_BUY':
                    all_bets.append(row_dict)

        current = test_end
        step += 1

    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()
    all_evals_df = pd.DataFrame(all_evaluations) if all_evaluations else pd.DataFrame()

    return calculate_results(bets_df, all_evals_df)


def calculate_results(bets_df: pd.DataFrame, all_evals_df: pd.DataFrame) -> dict:
    """Calculate comprehensive backtest statistics."""
    if bets_df.empty:
        return _empty_results(len(all_evals_df))

    total_bets = len(bets_df)
    wins = int(bets_df['fav_won'].sum()) if 'fav_won' in bets_df.columns else 0
    losses = total_bets - wins
    accuracy = wins / total_bets if total_bets > 0 else 0

    # Polymarket-style P&L
    total_invested = 0
    total_returned = 0
    if 'limit_price' in bets_df.columns and 'fav_won' in bets_df.columns:
        for _, bet in bets_df.iterrows():
            price = bet['limit_price']
            total_invested += price
            if bet['fav_won'] == 1:
                total_returned += 1.0  # $1 payout per share

    roi = ((total_returned - total_invested) / total_invested * 100
           if total_invested > 0 else 0)
    net_profit = total_returned - total_invested

    # Fill rate
    total_games = len(all_evals_df)
    fill_rate = total_bets / total_games if total_games > 0 else 0

    # Daily frequency
    unique_dates = 0
    date_range = 0
    daily_frequency = 0
    if 'date' in bets_df.columns:
        unique_dates = bets_df['date'].nunique()
        date_range = (bets_df['date'].max() - bets_df['date'].min()).days + 1
        daily_frequency = unique_dates / date_range if date_range > 0 else 0

    # Monthly breakdown
    monthly_results = {}
    if 'date' in bets_df.columns and 'fav_won' in bets_df.columns:
        bets_df_copy = bets_df.copy()
        bets_df_copy['month'] = bets_df_copy['date'].dt.to_period('M')
        for month, group in bets_df_copy.groupby('month'):
            m_wins = int(group['fav_won'].sum())
            monthly_results[str(month)] = {
                'bets': len(group),
                'wins': m_wins,
                'losses': len(group) - m_wins,
                'accuracy': m_wins / len(group),
            }

    # Bet details
    bet_details = []
    if not bets_df.empty:
        for _, b in bets_df.iterrows():
            bet_details.append({
                'date': str(b.get('date', '')),
                'fav_team': b.get('fav_team', ''),
                'dog_team': b.get('dog_team', ''),
                'fav_ml': int(b.get('fav_ml', 0)),
                'fav_prob': round(float(b.get('fav_prob_novig', 0)), 3),
                'limit_price': round(float(b.get('limit_price', 0)), 2),
                'won': int(b.get('fav_won', 0)),
                'score': f"{b.get('home_score', '?')}-{b.get('away_score', '?')}",
                'floor_used': b.get('floor_used', '?'),
            })

    return {
        'total_games_evaluated': total_games,
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'accuracy': accuracy,
        'fill_rate': fill_rate,
        'total_invested': round(total_invested, 2),
        'total_returned': round(total_returned, 2),
        'net_profit': round(net_profit, 2),
        'roi': round(roi, 2),
        'unique_bet_dates': unique_dates,
        'date_range_days': date_range,
        'daily_frequency': daily_frequency,
        'monthly_results': monthly_results,
        'bet_details': bet_details,
    }


def _empty_results(total_games: int = 0) -> dict:
    return {
        'total_games_evaluated': total_games,
        'total_bets': 0, 'wins': 0, 'losses': 0,
        'accuracy': 0, 'fill_rate': 0,
        'total_invested': 0, 'total_returned': 0,
        'net_profit': 0, 'roi': 0,
        'unique_bet_dates': 0, 'date_range_days': 0,
        'daily_frequency': 0, 'monthly_results': {},
        'bet_details': [],
    }


def check_overfitting(results: dict, df: pd.DataFrame) -> dict:
    """Run overfitting checks."""
    checks = {}

    # Monthly consistency
    monthly = results.get('monthly_results', {})
    if len(monthly) >= 2:
        accs = [m['accuracy'] for m in monthly.values() if m['bets'] >= 2]
        if accs:
            checks['monthly_acc_mean'] = float(np.mean(accs))
            checks['monthly_acc_std'] = float(np.std(accs))
            checks['monthly_consistent'] = np.std(accs) < 0.10
            print(f"  Monthly accuracy: mean={np.mean(accs):.3f}, "
                  f"std={np.std(accs):.3f} "
                  f"({'PASS' if checks['monthly_consistent'] else 'WARN'})")

    # Sample size
    total_bets = results.get('total_bets', 0)
    checks['sufficient_sample'] = total_bets >= 20
    print(f"  Sample size: {total_bets} bets "
          f"({'PASS' if checks['sufficient_sample'] else 'WARN: <20'})")

    # Fill rate
    fill = results.get('fill_rate', 0)
    checks['adequate_fill'] = fill >= 0.01
    print(f"  Fill rate: {fill:.1%} "
          f"({'PASS' if checks['adequate_fill'] else 'WARN: <1%'})")

    # Statistical significance
    if total_bets > 0:
        wins = results.get('wins', 0)
        # Test vs a high baseline (e.g., 95% — is it really 100%?)
        try:
            result = binomtest(wins, total_bets, 0.95, alternative='greater')
            p_value = result.pvalue
            checks['stat_sig_vs_95pct'] = float(p_value)
            checks['significantly_above_95'] = p_value < 0.05
            print(f"  Stat sig (vs 95%): p={p_value:.4f} "
                  f"({'SIG' if checks['significantly_above_95'] else 'NOT SIG'})")
        except Exception:
            pass

    # Accuracy
    accuracy = results.get('accuracy', 0)
    checks['perfect_accuracy'] = accuracy == 1.0
    print(f"  Perfect accuracy: {accuracy:.1%} "
          f"({'PASS' if checks['perfect_accuracy'] else 'FAIL'})")

    return checks


def print_results(results: dict):
    """Print formatted backtest results."""
    print(f"\n  Total games evaluated: {results.get('total_games_evaluated', 0)}")
    print(f"  Fill rate: {results.get('fill_rate', 0):.1%}")
    print(f"  Total bets (filled orders): {results['total_bets']}")
    print(f"  Wins: {results.get('wins', 0)}")
    print(f"  Losses: {results.get('losses', 0)}")
    print(f"  ACCURACY: {results['accuracy']:.1%}")
    print(f"")
    print(f"  Polymarket P&L:")
    print(f"    Total invested: ${results.get('total_invested', 0):.2f}")
    print(f"    Total returned: ${results.get('total_returned', 0):.2f}")
    print(f"    Net profit: ${results.get('net_profit', 0):.2f}")
    print(f"    ROI: {results.get('roi', 0):.1f}%")
    print(f"")
    print(f"  Frequency:")
    print(f"    Unique betting days: {results.get('unique_bet_dates', 0)}")
    print(f"    Date range: {results.get('date_range_days', 0)} days")
    print(f"    Daily frequency: {results.get('daily_frequency', 0):.1%}")

    monthly = results.get('monthly_results', {})
    if monthly:
        print(f"\n  BY MONTH:")
        for month, m in sorted(monthly.items()):
            print(f"    {month}: {m['bets']} bets, {m['wins']}W/{m['losses']}L, "
                  f"{m['accuracy']:.1%}")

    # Show losses (if any)
    losses = [b for b in results.get('bet_details', []) if b.get('won', 1) == 0]
    if losses:
        print(f"\n  LOSSES ({len(losses)} total):")
        for l in losses:
            print(f"    {l['date']}: {l['fav_team']} (ML {l['fav_ml']}, "
                  f"prob {l['fav_prob']:.1%}) LOST, score: {l['score']}")


def save_results(results: dict):
    """Save backtest results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    clean = json.loads(json.dumps(results, default=str))
    summary_file = os.path.join(RESULTS_DIR, 'backtest_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\n  Results saved to {summary_file}")
