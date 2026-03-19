#!/usr/bin/env python3
"""
MLB V8 TEPE — Backtest Runner with Structural Sweep
====================================================
Runs walk-forward backtest for the Temporal Edge Persistence Engine.

Usage:
    python experiments/mlb/run_backtest_v8_tepe.py                  # default config
    python experiments/mlb/run_backtest_v8_tepe.py --sweep           # structural sweep
    python experiments/mlb/run_backtest_v8_tepe.py --verbose         # detailed output
"""

import sys
import os
import argparse
import json
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_loader import load_mlb_boxscores, load_mlb_odds
from mlb.strategy_v8_tepe import run_backtest, TEPE_CONFIG


def print_results(results, config=None):
    """Print detailed backtest results."""
    acc = results.get('accuracy', 0)

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS — MLB V8 TEPE")
    print("=" * 70)
    print(f"  Total Picks:     {results['total_picks']}")
    print(f"  Total Wins:      {results['total_wins']}")
    print(f"  Accuracy:        {acc * 100:.1f}%")
    print(f"  ROI:             {results['total_roi'] * 100:.1f}%")
    print(f"  P&L:             ${results['total_pnl']:+,.0f}")
    print(f"  Wagered:         ${results.get('total_wagered', 0):,.0f}")
    print(f"  Max Drawdown:    ${results.get('max_drawdown', 0):,.0f}")
    print(f"  Max Win Streak:  {results.get('max_win_streak', 0)}")
    print(f"  Max Loss Streak: {results.get('max_loss_streak', 0)}")
    print(f"  Active Days:     {results.get('active_days', 0)} / {results.get('total_days', 0)}"
          f" ({results.get('day_coverage', 0) * 100:.0f}%)")
    print(f"  Avg Daily P&L:   ${results.get('avg_daily_pnl', 0):+,.0f}")

    by_legs = results.get('by_legs', {})
    for n in sorted(by_legs.keys()):
        s = by_legs[n]
        print(f"\n  {n}-LEG PARLAYS:")
        print(f"    Total: {s['total']}, Wins: {s['wins']}")
        print(f"    Parlay Accuracy: {s['accuracy'] * 100:.1f}%")
        print(f"    Leg Accuracy:    {s['leg_accuracy'] * 100:.1f}%"
              f" ({s['legs_hits']}/{s['legs_total']})")
        print(f"    ROI: {s['roi'] * 100:.1f}%")
        print(f"    P&L: ${s['pnl']:+,.0f}")

    if config:
        print(f"\n  Config: K={config.get('K_FOLDS')}, "
              f"FS={config.get('FOLD_SIZE')}, "
              f"ME={config.get('MIN_EDGE')}, "
              f"Stats={config.get('STATS_ALLOWED')}, "
              f"Odds=[{config.get('LEG_MIN_ODDS')},{config.get('LEG_MAX_ODDS')}], "
              f"Elite={config.get('ELITE_TOP_N')}")

    daily = results.get('daily', [])
    if daily:
        print(f"\n  DAILY P&L (last 20):")
        for d in daily[-20:]:
            print(f"    {d['date']}: {d['n_picks']} picks, "
                  f"{d['wins']} wins, ${d['pnl']:+d}")


def run_sweep(box_scores, odds_data, base_config, verbose=False):
    """
    Structural parameter sweep.

    NOTE: These are STRUCTURAL parameters (fold count, fold size, elite pool),
    NOT statistical thresholds. The search space is intentionally small
    (~100 combinations) to prevent overfitting.
    """
    sweep_params = {
        'K_FOLDS': [3, 4],
        'FOLD_SIZE': [5, 7],
        'MIN_EDGE': [0.005, 0.01, 0.02, 0.03, 0.05],
        'STATS_ALLOWED': [['h', 'tb'], ['h']],
        'LEG_MAX_ODDS': [-110, -150, -200],
        'ELITE_TOP_N': [6, 8, 10, 15],
    }

    # Generate all combinations
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    combos = list(itertools.product(*values))
    print(f"\nStructural sweep: {len(combos)} configurations")
    print("=" * 70)

    results_list = []

    for i, combo in enumerate(combos):
        config = dict(base_config)
        for k, v in zip(keys, combo):
            config[k] = v

        try:
            results = run_backtest(box_scores, odds_data, config, verbose=False)
        except Exception as e:
            continue

        if results['total_picks'] == 0:
            continue

        summary = {
            'config': {k: config[k] for k in keys},
            'picks': results['total_picks'],
            'accuracy': results['accuracy'],
            'roi': results['total_roi'],
            'pnl': results['total_pnl'],
            'active_days': results['active_days'],
            'day_coverage': results.get('day_coverage', 0),
        }

        # Add per-leg-size accuracy
        for n, s in results.get('by_legs', {}).items():
            summary[f'{n}L_accuracy'] = s['accuracy']
            summary[f'{n}L_leg_accuracy'] = s['leg_accuracy']
            summary[f'{n}L_count'] = s['total']
            summary[f'{n}L_roi'] = s['roi']

        results_list.append(summary)

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(combos)}] ...")

    print(f"\nCompleted {len(results_list)} valid configurations")

    if not results_list:
        print("No valid results found!")
        return None, None

    # Sort by different objectives
    print("\n" + "=" * 70)
    print("TOP CONFIGS BY ACCURACY (min 10 picks, min 10 active days)")
    print("=" * 70)
    filtered = [r for r in results_list
                if r['picks'] >= 10 and r['active_days'] >= 10]
    if filtered:
        by_acc = sorted(filtered, key=lambda r: r['accuracy'], reverse=True)
        for r in by_acc[:10]:
            cfg = r['config']
            leg_info = ""
            for n in [2, 3]:
                if f'{n}L_accuracy' in r:
                    leg_info += f" | {n}L: {r[f'{n}L_accuracy']*100:.0f}% acc ({r[f'{n}L_count']}p)"
                    leg_info += f" legs={r[f'{n}L_leg_accuracy']*100:.0f}%"
            print(f"  Acc={r['accuracy']*100:.1f}% ROI={r['roi']*100:.1f}% "
                  f"P&L=${r['pnl']:+,.0f} Picks={r['picks']} "
                  f"Days={r['active_days']} ({r['day_coverage']*100:.0f}%)"
                  f"{leg_info}")
            print(f"    K={cfg['K_FOLDS']} FS={cfg['FOLD_SIZE']} "
                  f"ME={cfg['MIN_EDGE']} Stats={cfg['STATS_ALLOWED']} "
                  f"MaxOdds={cfg['LEG_MAX_ODDS']} Elite={cfg['ELITE_TOP_N']}")

    print("\n" + "=" * 70)
    print("TOP CONFIGS BY ROI (min 10 picks, min 10 active days)")
    print("=" * 70)
    if filtered:
        by_roi = sorted(filtered, key=lambda r: r['roi'], reverse=True)
        for r in by_roi[:10]:
            cfg = r['config']
            print(f"  ROI={r['roi']*100:.1f}% Acc={r['accuracy']*100:.1f}% "
                  f"P&L=${r['pnl']:+,.0f} Picks={r['picks']} "
                  f"Days={r['active_days']} ({r['day_coverage']*100:.0f}%)")
            print(f"    K={cfg['K_FOLDS']} FS={cfg['FOLD_SIZE']} "
                  f"ME={cfg['MIN_EDGE']} Stats={cfg['STATS_ALLOWED']} "
                  f"MaxOdds={cfg['LEG_MAX_ODDS']} Elite={cfg['ELITE_TOP_N']}")

    print("\n" + "=" * 70)
    print("TOP CONFIGS BY BALANCED SCORE (accuracy × sqrt(picks) × (1+roi))")
    print("=" * 70)
    if filtered:
        for r in filtered:
            r['balanced'] = (r['accuracy'] *
                             math.sqrt(r['picks']) *
                             max(0.1, 1 + r['roi']))
        by_bal = sorted(filtered, key=lambda r: r['balanced'], reverse=True)
        for r in by_bal[:10]:
            cfg = r['config']
            leg_info = ""
            for n in [2, 3]:
                if f'{n}L_accuracy' in r:
                    leg_info += f" | {n}L: {r[f'{n}L_accuracy']*100:.0f}% ({r[f'{n}L_count']})"
            print(f"  Bal={r['balanced']:.2f} Acc={r['accuracy']*100:.1f}% "
                  f"ROI={r['roi']*100:.1f}% P&L=${r['pnl']:+,.0f} "
                  f"Picks={r['picks']} Days={r['active_days']}"
                  f"{leg_info}")
            print(f"    K={cfg['K_FOLDS']} FS={cfg['FOLD_SIZE']} "
                  f"ME={cfg['MIN_EDGE']} Stats={cfg['STATS_ALLOWED']} "
                  f"MaxOdds={cfg['LEG_MAX_ODDS']} Elite={cfg['ELITE_TOP_N']}")

    # Return best balanced config
    if filtered:
        best = by_bal[0]
        best_config = dict(base_config)
        for k, v in best['config'].items():
            best_config[k] = v
        return best_config, results_list
    return None, results_list


import math


def main():
    parser = argparse.ArgumentParser(description='MLB V8 TEPE Backtest')
    parser.add_argument('--sweep', action='store_true',
                        help='Run structural parameter sweep')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config JSON')
    args = parser.parse_args()

    print("=" * 70)
    print("MLB V8 TEPE — Temporal Edge Persistence Engine")
    print("=" * 70)

    # Load data
    print("\nLoading MLB data...")
    box_scores = load_mlb_boxscores()
    odds_data = load_mlb_odds()
    print(f"  Boxscores: {len(box_scores)} games")
    print(f"  Odds: {len(odds_data)} records")

    dates = sorted(set(g['date'] for g in box_scores))
    print(f"  Date range: {dates[0]} to {dates[-1]} ({len(dates)} dates)")

    # Load config
    config = dict(TEPE_CONFIG)
    if args.config:
        with open(args.config) as f:
            loaded = json.load(f)
            config.update(loaded.get('config', loaded))
        print(f"  Loaded config from {args.config}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    if args.sweep:
        best_config, all_results = run_sweep(
            box_scores, odds_data, config, verbose=args.verbose
        )

        if best_config:
            print("\n\n" + "=" * 70)
            print("BEST CONFIG — FULL RESULTS")
            print("=" * 70)
            results = run_backtest(box_scores, odds_data, best_config,
                                   verbose=args.verbose)
            print_results(results, best_config)

            # Save best config
            config_path = os.path.join(output_dir, 'tepe_best_config.json')
            with open(config_path, 'w') as f:
                json.dump({'config': best_config}, f, indent=2)
            print(f"\nBest config saved to {config_path}")

        # Save sweep results
        sweep_path = os.path.join(output_dir, 'tepe_sweep_results.json')
        with open(sweep_path, 'w') as f:
            json.dump(all_results or [], f, indent=2, default=str)
        print(f"Sweep results saved to {sweep_path}")

    else:
        print(f"\nConfig: K={config['K_FOLDS']}, FS={config['FOLD_SIZE']}, "
              f"ME={config['MIN_EDGE']}, Stats={config['STATS_ALLOWED']}")
        print("\nRunning walk-forward backtest...")
        results = run_backtest(box_scores, odds_data, config,
                               verbose=args.verbose)
        print_results(results, config)

        # Save results
        results_path = os.path.join(output_dir, 'tepe_backtest_results.json')
        save_results = {k: v for k, v in results.items() if k != 'picks'}
        save_results['config'] = config
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")

        # Save picks detail
        picks_path = os.path.join(output_dir, 'tepe_picks_detail.json')
        picks_out = []
        for p in results.get('picks', []):
            pick = {k: v for k, v in p.items()
                    if k in ['date', 'bet_type', 'n_legs', 'hit', 'pnl',
                             'wager', 'parlay_american', 'parlay_decimal',
                             'combined_rate', 'min_min_edge', 'avg_min_edge']}
            if 'legs' in p:
                pick['legs'] = p['legs']
            picks_out.append(pick)
        with open(picks_path, 'w') as f:
            json.dump(picks_out, f, indent=2, default=str)
        print(f"Picks saved to {picks_path}")


if __name__ == '__main__':
    main()
