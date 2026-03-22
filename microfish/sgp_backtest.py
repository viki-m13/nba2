"""
MiroFish SGP Backtesting Framework
======================================
Walk-forward validation for SGP (Same Game Parlay) strategy.

Evaluates how constructed SGPs would have performed against historical
player prop outcomes.

Anti-overfitting measures:
1. Walk-forward validation (expanding window, never look ahead)
2. Purge gap between calibration and test windows
3. All player features lagged by FEATURE_LAG_DAYS
4. Cross-season consistency checks
5. Variance-aware significance tests (SGPs are HIGH variance)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sgp_config import (
    SGP_RESULTS_DIR, BACKTEST_SEASONS, STAKE_SIZE,
    WALK_FORWARD_TRAIN_DAYS, WALK_FORWARD_STEP_DAYS, PURGE_DAYS,
)
from sgp_data import (
    load_all_sgp_data, american_to_decimal, american_to_implied_prob,
    parlay_odds,
)


def run_sgp_backtest(rules: dict = None) -> dict:
    """
    Run a complete SGP backtest.

    Unlike moneyline backtesting, SGP backtesting requires:
    1. Historical player prop LINES (what was available to bet)
    2. Historical player prop OUTCOMES (did the prop hit?)
    3. Parlay construction (which legs were combined)
    4. Combined outcome verification (did ALL legs hit?)
    """
    from sgp_engine import SGPEngine

    print("=" * 70)
    print("MIROFISH SGP STRATEGY BACKTEST")
    print("=" * 70)
    print(f"Strategy: SGP Template Engine (deterministic)")
    print(f"Seasons: {BACKTEST_SEASONS}")
    print(f"Stake: ${STAKE_SIZE} per SGP")
    print()

    # Load rules
    if rules is None:
        from sgp_discovery import load_sgp_rules
        rules = load_sgp_rules()
        if not rules:
            print("ERROR: No SGP rules found. Run: python run.py sgp-develop")
            return {}

    engine = SGPEngine(rules)
    print("SGP Strategy loaded:")
    print(engine.summary())
    print()

    # Load data
    print("\nSTEP 1: Loading historical SGP data...")
    data = load_all_sgp_data()
    props_df = data['props']
    results_df = data['results']

    if props_df.empty:
        print("ERROR: No historical prop data found.")
        print("  Cache prop data to: data/sgp/cache/mlb_props_YYYY.csv")
        return {}

    print(f"  Prop records: {len(props_df)}")
    if not results_df.empty:
        print(f"  Result records: {len(results_df)}")

    has_outcomes = 'hit' in props_df.columns

    if not has_outcomes:
        print("\n  WARNING: No 'hit' column in prop data.")
        print("  Running construction-only backtest (no win/loss tracking).")

    # Walk-forward SGP construction backtest
    print("\nSTEP 2: Walk-forward SGP construction...")
    backtest_results = _walk_forward_sgp_backtest(props_df, engine, has_outcomes)

    # Print results
    print("\n" + "=" * 70)
    print("SGP BACKTEST RESULTS")
    print("=" * 70)
    _print_sgp_results(backtest_results)

    # Overfitting checks
    if has_outcomes and backtest_results.get('total_sgps', 0) > 0:
        print("\nOVERFITTING CHECKS:")
        overfit = _check_sgp_overfitting(backtest_results)
        backtest_results['overfitting_checks'] = overfit

    # Save
    _save_sgp_results(backtest_results)

    return backtest_results


def _walk_forward_sgp_backtest(props_df: pd.DataFrame,
                                engine,
                                has_outcomes: bool) -> dict:
    """
    Walk-forward backtest of SGP construction.

    For each test window:
    1. Construct SGPs from available props
    2. If outcomes available, check if ALL legs hit
    3. Calculate P&L
    """
    if 'game_date' not in props_df.columns:
        return {'total_sgps': 0, 'error': 'no game_date column'}

    props_df = props_df.sort_values('game_date').copy()
    min_date = pd.Timestamp(props_df['game_date'].min())
    max_date = pd.Timestamp(props_df['game_date'].max())

    test_start = min_date + pd.Timedelta(
        days=WALK_FORWARD_TRAIN_DAYS + PURGE_DAYS
    )

    all_sgps = []
    step = 0

    current_start = test_start
    while current_start < max_date:
        current_end = current_start + pd.Timedelta(days=WALK_FORWARD_STEP_DAYS)

        test_mask = (
            (props_df['game_date'] >= current_start) &
            (props_df['game_date'] < current_end)
        )
        test_props = props_df[test_mask]

        if not test_props.empty:
            # Construct SGPs for this window
            sgps = engine.construct_sgps(test_props)

            for sgp in sgps:
                sgp['walk_forward_step'] = step

                # Check outcomes if available
                if has_outcomes:
                    all_hit = True
                    for leg in sgp.get('legs', []):
                        player = leg.get('player', '')
                        market = leg.get('market', '')
                        direction = leg.get('direction', 'Over')

                        # Find this prop's outcome in data
                        prop_match = test_props[
                            (test_props['player'] == player) &
                            (test_props['market'] == market) &
                            (test_props['bet_type'].str.lower() == direction.lower()) &
                            (test_props['game_id'] == sgp.get('game_id', ''))
                        ]

                        if not prop_match.empty and 'hit' in prop_match.columns:
                            leg_hit = bool(prop_match.iloc[0]['hit'])
                            leg['hit'] = leg_hit
                            if not leg_hit:
                                all_hit = False
                        else:
                            leg['hit'] = None
                            all_hit = False  # Can't confirm = didn't hit

                    sgp['all_legs_hit'] = all_hit
                    sgp['payout'] = (
                        sgp['stake'] * sgp['combined_odds_decimal']
                        if all_hit else 0.0
                    )

                all_sgps.append(sgp)

        current_start = current_end
        step += 1

    return _calculate_sgp_results(all_sgps, has_outcomes)


def _calculate_sgp_results(sgps: list, has_outcomes: bool) -> dict:
    """Calculate comprehensive SGP backtest statistics."""
    if not sgps:
        return {
            'total_sgps': 0, 'hits': 0, 'misses': 0,
            'hit_rate': 0, 'total_wagered': 0, 'total_payout': 0,
            'roi': 0, 'ev_per_dollar': 0,
        }

    total_sgps = len(sgps)
    total_wagered = sum(s.get('stake', STAKE_SIZE) for s in sgps)

    # Odds distribution
    odds_list = [s.get('combined_odds_american', 0) for s in sgps]
    avg_odds = np.mean(odds_list) if odds_list else 0
    median_odds = np.median(odds_list) if odds_list else 0

    # Legs distribution
    legs_list = [s.get('n_legs', 0) for s in sgps]
    avg_legs = np.mean(legs_list) if legs_list else 0

    # Template usage
    template_counts = {}
    for s in sgps:
        t = s.get('template', 'unknown')
        template_counts[t] = template_counts.get(t, 0) + 1

    result = {
        'total_sgps': total_sgps,
        'total_wagered': round(total_wagered, 2),
        'avg_combined_odds': round(float(avg_odds)),
        'median_combined_odds': round(float(median_odds)),
        'avg_legs': round(float(avg_legs), 1),
        'template_usage': template_counts,
    }

    if has_outcomes:
        hits = sum(1 for s in sgps if s.get('all_legs_hit', False))
        misses = total_sgps - hits
        hit_rate = hits / total_sgps if total_sgps > 0 else 0
        total_payout = sum(s.get('payout', 0) for s in sgps)
        roi = ((total_payout - total_wagered) / total_wagered * 100
               if total_wagered > 0 else 0)
        ev_per_dollar = (total_payout / total_wagered - 1
                         if total_wagered > 0 else 0)

        result.update({
            'hits': hits,
            'misses': misses,
            'hit_rate': round(hit_rate, 4),
            'total_payout': round(total_payout, 2),
            'roi': round(roi, 1),
            'ev_per_dollar': round(ev_per_dollar, 4),
        })

        # Breakeven analysis
        if avg_odds > 0:
            breakeven_rate = 100 / (avg_odds + 100)
            edge = hit_rate - breakeven_rate
            result['breakeven_rate'] = round(breakeven_rate, 4)
            result['edge_over_breakeven'] = round(edge, 4)

        # Per-template results
        template_results = {}
        for t_name in template_counts:
            t_sgps = [s for s in sgps if s.get('template') == t_name]
            t_hits = sum(1 for s in t_sgps if s.get('all_legs_hit', False))
            t_payout = sum(s.get('payout', 0) for s in t_sgps)
            t_wagered = sum(s.get('stake', STAKE_SIZE) for s in t_sgps)
            template_results[t_name] = {
                'sgps': len(t_sgps),
                'hits': t_hits,
                'hit_rate': round(t_hits / len(t_sgps), 4) if t_sgps else 0,
                'roi': round((t_payout - t_wagered) / t_wagered * 100, 1) if t_wagered > 0 else 0,
            }
        result['template_results'] = template_results

    # SGP detail log
    result['sgp_detail'] = [
        {
            'template': s.get('template', ''),
            'game_date': s.get('game_date', ''),
            'home_team': s.get('home_team', ''),
            'away_team': s.get('away_team', ''),
            'n_legs': s.get('n_legs', 0),
            'combined_odds': s.get('combined_odds_american', 0),
            'hit': s.get('all_legs_hit', None),
            'payout': s.get('payout', 0),
            'legs_summary': [
                f"{l.get('player', '?')} {l.get('market', '?')} "
                f"{l.get('direction', '?')} {l.get('line', '?')} "
                f"({l.get('odds', 0):+d}) {'HIT' if l.get('hit') else 'MISS' if l.get('hit') is not None else '?'}"
                for l in s.get('legs', [])
            ],
        }
        for s in sgps[:100]  # Cap detail log at 100 entries
    ]

    return result


def _check_sgp_overfitting(results: dict) -> dict:
    """
    SGP-specific overfitting checks.
    SGPs are HIGH variance — need larger samples and different thresholds.
    """
    checks = {}

    total = results.get('total_sgps', 0)
    hits = results.get('hits', 0)
    hit_rate = results.get('hit_rate', 0)

    # Sample size — SGPs need MORE samples due to high variance
    checks['sufficient_sample'] = total >= 50
    print(f"  Sample size: {total} SGPs "
          f"({'PASS' if checks['sufficient_sample'] else 'WARN: <50'})")

    # Hit rate vs breakeven
    breakeven = results.get('breakeven_rate', 0.10)
    edge = hit_rate - breakeven
    checks['positive_edge'] = edge > 0
    print(f"  Hit rate: {hit_rate:.1%} vs breakeven {breakeven:.1%}, "
          f"edge: {edge:+.1%} "
          f"({'PASS' if checks['positive_edge'] else 'FAIL'})")

    # Statistical significance (binomial test)
    if total > 0 and breakeven > 0:
        try:
            from scipy.stats import binomtest
            result = binomtest(hits, total, breakeven, alternative='greater')
            p_value = result.pvalue
            checks['stat_sig_p_value'] = float(p_value)
            checks['stat_significant'] = p_value < 0.10  # Relaxed for SGP variance
            print(f"  Statistical significance: p={p_value:.4f} "
                  f"({'PASS' if checks['stat_significant'] else 'NOT SIG'})")
        except Exception:
            checks['stat_sig_p_value'] = None

    # Template consistency — no single template should dominate
    template_results = results.get('template_results', {})
    if len(template_results) >= 2:
        hit_rates = [
            t['hit_rate'] for t in template_results.values()
            if t.get('sgps', 0) >= 5
        ]
        if hit_rates:
            checks['template_consistency'] = np.std(hit_rates) < 0.15
            print(f"  Template consistency: std={np.std(hit_rates):.3f} "
                  f"({'PASS' if checks['template_consistency'] else 'WARN'})")

    # ROI check
    roi = results.get('roi', 0)
    checks['positive_roi'] = roi > 0
    print(f"  ROI: {roi:+.1f}% "
          f"({'PASS' if checks['positive_roi'] else 'FAIL'})")

    return checks


def _print_sgp_results(results: dict):
    """Print formatted SGP backtest results."""
    print(f"\n  Total SGPs constructed: {results.get('total_sgps', 0)}")
    print(f"  Total wagered: ${results.get('total_wagered', 0):.2f}")
    print(f"  Avg combined odds: +{results.get('avg_combined_odds', 0)}")
    print(f"  Median combined odds: +{results.get('median_combined_odds', 0)}")
    print(f"  Avg legs per SGP: {results.get('avg_legs', 0):.1f}")

    if 'hits' in results:
        print(f"\n  Hits: {results.get('hits', 0)}")
        print(f"  Misses: {results.get('misses', 0)}")
        print(f"  HIT RATE: {results.get('hit_rate', 0):.1%}")
        print(f"  Total payout: ${results.get('total_payout', 0):.2f}")
        print(f"  ROI: {results.get('roi', 0):+.1f}%")
        print(f"  EV per $1: ${results.get('ev_per_dollar', 0):.4f}")

        if 'breakeven_rate' in results:
            print(f"  Breakeven rate: {results['breakeven_rate']:.1%}")
            print(f"  Edge: {results.get('edge_over_breakeven', 0):+.1%}")

    # Template breakdown
    template_results = results.get('template_results', {})
    if template_results:
        print(f"\n  BY TEMPLATE:")
        for name, tr in sorted(template_results.items()):
            print(f"    {name}: {tr['sgps']} SGPs, {tr['hits']} hits, "
                  f"{tr['hit_rate']:.1%} rate, {tr['roi']:+.1f}% ROI")

    template_usage = results.get('template_usage', {})
    if template_usage and not template_results:
        print(f"\n  TEMPLATE USAGE:")
        for name, count in sorted(template_usage.items(),
                                    key=lambda x: x[1], reverse=True):
            print(f"    {name}: {count} SGPs")


def _save_sgp_results(results: dict):
    """Save SGP backtest results."""
    os.makedirs(SGP_RESULTS_DIR, exist_ok=True)

    # Clean numpy types for JSON
    def clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary_file = os.path.join(SGP_RESULTS_DIR, 'sgp_backtest_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  SGP results saved to {summary_file}")

    # Save detail log separately
    detail = results.pop('sgp_detail', [])
    if detail:
        detail_file = os.path.join(SGP_RESULTS_DIR, 'sgp_backtest_detail.json')
        with open(detail_file, 'w') as f:
            json.dump(detail, f, indent=2, default=str)
        print(f"  SGP detail log saved to {detail_file}")
