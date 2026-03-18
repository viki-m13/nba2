#!/usr/bin/env python3
"""
NBA Experiment — Run backtest, optimization, and validation.

Usage:
    python experiments/nba/run_backtest.py                    # backtest only
    python experiments/nba/run_backtest.py --optimize 50      # optimize for 50 iterations
    python experiments/nba/run_backtest.py --validate         # full validation suite
    python experiments/nba/run_backtest.py --objective roi    # roi/accuracy/balanced
"""

import sys
import os
import argparse
import json

# Ensure experiments root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_loader import load_nba_boxscores, load_nba_odds
from shared.validation import full_validation_report
from shared.optimizer import run_optimization
from nba.strategy import run_backtest, NBA_ROI_CONFIG, NBA_PARAM_RANGES


def main():
    parser = argparse.ArgumentParser(description='NBA ROI Experiment')
    parser.add_argument('--optimize', type=int, default=0,
                        help='Number of optimization iterations (0 = backtest only)')
    parser.add_argument('--validate', action='store_true',
                        help='Run full validation suite')
    parser.add_argument('--objective', default='roi',
                        choices=['roi', 'accuracy', 'balanced', 'max_roi', 'parlay_roi',
                                 'plus_money_accuracy'],
                        help='Optimization objective')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config JSON')
    args = parser.parse_args()

    print("=" * 70)
    print("NBA ROI EXPERIMENT")
    print("=" * 70)

    # Load data
    print("\nLoading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    print(f"  Boxscores: {len(box_scores)} games")
    print(f"  Odds: {len(odds_data)} records")

    # Filter out All-Star games and other non-standard games
    box_scores = [g for g in box_scores if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    print(f"  After filtering: {len(box_scores)} regular games")

    dates = sorted(set(g['date'] for g in box_scores))
    print(f"  Date range: {dates[0]} to {dates[-1]} ({len(dates)} dates)")

    # Load config
    config = dict(NBA_ROI_CONFIG)
    if args.config:
        with open(args.config) as f:
            loaded = json.load(f)
            config.update(loaded.get('config', loaded))
        print(f"\n  Loaded config from {args.config}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

    if args.optimize > 0:
        # Run optimizer
        best_config, results = run_optimization(
            backtest_fn=run_backtest,
            box_scores=box_scores,
            odds_data=odds_data,
            base_config=config,
            param_ranges=NBA_PARAM_RANGES,
            iterations=args.optimize,
            objective=args.objective,
            verbose=args.verbose,
            output_dir=output_dir,
        )
        config = best_config
    else:
        # Standard backtest
        print("\nRunning walk-forward backtest...")
        results = run_backtest(box_scores, odds_data, config, verbose=args.verbose)
        _print_full_results(results)

    # Validation
    if args.validate:
        print("\n\nRunning validation suite...")
        full_validation_report(
            results,
            backtest_fn=run_backtest,
            config=config,
            box_scores=box_scores,
            odds_data=odds_data,
        )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'backtest_results.json')
    save_results = {k: v for k, v in results.items() if k != 'picks'}
    save_results['config'] = config
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save picks detail
    picks_path = os.path.join(output_dir, 'picks_detail.json')
    picks_out = []
    for p in results.get('picks', []):
        pick = {k: v for k, v in p.items()
                if k in ['date', 'player', 'stat', 'line', 'odds', 'bet_type',
                         'hit', 'actual', 'pnl', 'wager', 'edge', 'ev',
                         'combined_score', 'hit_rate']}
        if 'legs' in p:
            pick['legs'] = p['legs']
        picks_out.append(pick)
    with open(picks_path, 'w') as f:
        json.dump(picks_out, f, indent=2)
    print(f"Picks saved to {picks_path}")


def _print_full_results(results):
    """Print detailed backtest results."""
    acc = results.get('accuracy', results.get('individual_accuracy', 0))

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Total Picks:     {results['total_picks']}")
    print(f"  Total Wins:      {results['total_wins']}")
    print(f"  Accuracy:        {acc*100:.1f}%")
    print(f"  ROI:             {results['total_roi']*100:.1f}%")
    print(f"  P&L:             ${results['total_pnl']:+,.0f}")
    print(f"  Wagered:         ${results.get('total_wagered', 0):,.0f}")
    print(f"  Max Drawdown:    ${results.get('max_drawdown', 0):,.0f}")
    print(f"  Max Loss Streak: {results.get('max_loss_streak', 0)}")
    print(f"  Max Win Streak:  {results.get('max_win_streak', 0)}")
    print(f"  Active Days:     {results.get('active_days', 0)}")
    print(f"  Avg Daily P&L:   ${results.get('avg_daily_pnl', 0):+,.0f}")

    if results.get('singles_total', 0) > 0:
        print(f"\n  SINGLES:")
        print(f"    Total: {results['singles_total']}, Wins: {results['singles_wins']}")
        print(f"    Accuracy: {results['singles_accuracy']*100:.1f}%")
        print(f"    P&L: ${results['singles_pnl']:+,.0f}")

    if results.get('parlay_total', 0) > 0:
        print(f"\n  PARLAYS:")
        print(f"    Total: {results['parlay_total']}, Wins: {results['parlay_wins']}")
        print(f"    Accuracy: {results['parlay_accuracy']*100:.1f}%")
        print(f"    P&L: ${results['parlay_pnl']:+,.0f}")
        if results.get('parlay_leg_total', 0) > 0:
            print(f"    Leg Accuracy: {results['parlay_leg_accuracy']*100:.1f}% "
                  f"({results['parlay_leg_wins']}/{results['parlay_leg_total']})")
        # Breakdown by leg count
        parlay_by_legs = results.get('parlay_by_legs', {})
        if parlay_by_legs:
            for n_legs in sorted(parlay_by_legs.keys()):
                info = parlay_by_legs[n_legs]
                acc = info['wins'] / info['total'] * 100 if info['total'] > 0 else 0
                print(f"    {n_legs}-leg: {info['total']} total, {info['wins']} wins "
                      f"({acc:.0f}%), ${info['pnl']:+,.0f}")

    # Top picks by edge
    picks = results.get('picks', [])
    if picks:
        singles = [p for p in picks if p.get('bet_type') == 'single']
        if singles:
            print(f"\n  TOP SINGLES BY EDGE:")
            top = sorted(singles, key=lambda p: p.get('edge', 0), reverse=True)[:5]
            for p in top:
                hit_str = 'HIT' if p.get('hit') else 'MISS'
                print(f"    {p.get('date','')} {p['player']} {p['stat']} o{p['line']} "
                      f"({p['odds']:+d}) edge={p.get('edge',0):.3f} "
                      f"actual={p.get('actual','')} {hit_str} ${p.get('pnl',0):+d}")

    # Daily P&L
    daily = results.get('daily', [])
    if daily:
        print(f"\n  DAILY P&L:")
        for d in daily:
            print(f"    {d['date']}: {d['n_picks']} picks, "
                  f"{d['wins']} wins, ${d['pnl']:+d}")


if __name__ == '__main__':
    main()
