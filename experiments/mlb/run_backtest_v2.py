#!/usr/bin/env python3
"""
MLB HECE v2 — Plus-Money Accuracy Strategy

Uses existing MLB signal stack (GFT, BEQ, ESI, PBF, ABVC, CGSM, VPD, OAL)
with plus-money-only filter and high accuracy gates.

Usage:
    python experiments/mlb/run_backtest_v2.py                      # backtest
    python experiments/mlb/run_backtest_v2.py --optimize 500       # optimize
"""

import sys, os, argparse, json, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_loader import load_mlb_boxscores, load_mlb_odds
from shared.validation import full_validation_report
from shared.optimizer import run_optimization, OBJECTIVE_FNS
from mlb.strategy import run_backtest, MLB_ROI_CONFIG, MLB_PARAM_RANGES


def compute_mlb_hece_score(results):
    """MLB HECE objective: maximize accuracy on plus-money with volume."""
    n_picks = results['total_picks']
    if n_picks < 8:
        return -1

    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    roi = results.get('total_roi', 0)
    pnl = results.get('total_pnl', 0)
    max_dd = results.get('max_drawdown', 0)

    if acc < 0.60:
        return max(0.001, acc * 0.1)

    if acc >= 0.97:   acc_mult = 150.0
    elif acc >= 0.95: acc_mult = 80.0
    elif acc >= 0.93: acc_mult = 50.0
    elif acc >= 0.91: acc_mult = 35.0
    elif acc >= 0.90: acc_mult = 25.0
    elif acc >= 0.85: acc_mult = 10.0
    elif acc >= 0.80: acc_mult = 5.0
    elif acc >= 0.75: acc_mult = 3.0
    elif acc >= 0.70: acc_mult = 2.0
    else:             acc_mult = 0.5

    if n_picks < 12:     vol = n_picks / 12
    elif n_picks <= 30:  vol = 1.0 + (n_picks - 12) * 0.05
    elif n_picks <= 60:  vol = 1.9 + (n_picks - 30) * 0.04
    else:                vol = 3.1 + (n_picks - 60) * 0.02

    if roi >= 1.0:     roi_mult = 5.0
    elif roi >= 0.50:  roi_mult = 3.5
    elif roi >= 0.30:  roi_mult = 2.5
    elif roi >= 0.15:  roi_mult = 2.0
    elif roi >= 0:     roi_mult = 1.0
    else:              roi_mult = 0.2

    dd_penalty = max(0.4, 1 - max_dd / max(1, abs(pnl) + 1) * 0.5) if pnl > 0 else 0.3
    active_days = results.get('active_days', 1)
    day_mult = min(2.5, 1.0 + active_days * 0.05)
    pnl_mult = 1 + math.log10(max(1, abs(pnl))) * 0.3 if pnl > 0 else 0.5

    return acc * acc_mult * vol * roi_mult * dd_penalty * day_mult * pnl_mult


OBJECTIVE_FNS['mlb_hece'] = compute_mlb_hece_score

# MLB plus-money config — modify base config for plus-money accuracy
MLB_PM_CONFIG = dict(MLB_ROI_CONFIG)
MLB_PM_CONFIG.update({
    'MIN_ODDS': 100,         # Plus-money only
    'MAX_ODDS': 400,
    'GATE_MIN_HIT_RATE': 0.70,
    'MIN_EV': 0.10,
    'BEQ_MIN_EDGE': 0.10,
    'SINGLE_MIN_SCORE': 0.40,
})

# Adjusted param ranges for plus-money optimization
MLB_PM_PARAM_RANGES = dict(MLB_PARAM_RANGES)
MLB_PM_PARAM_RANGES.update({
    'MIN_ODDS': (100, 160),
    'MAX_ODDS': (200, 600),
    'GATE_MIN_HIT_RATE': (0.50, 0.90),
    'MIN_EV': (0.05, 0.25),
    'BEQ_MIN_EDGE': (0.03, 0.20),
    'SINGLE_MIN_SCORE': (0.25, 0.65),
    'KELLY_FRACTION': (0.15, 0.50),
})


def main():
    parser = argparse.ArgumentParser(description='MLB HECE v2')
    parser.add_argument('--optimize', type=int, default=0)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--objective', default='mlb_hece',
                        choices=list(OBJECTIVE_FNS.keys()))
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("MLB PLUS-MONEY ACCURACY STRATEGY v2")
    print("=" * 70)

    print("\nLoading MLB data...")
    box_scores = load_mlb_boxscores()
    odds_data = load_mlb_odds()
    dates = sorted(set(g['date'] for g in box_scores))
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")
    print(f"  Date range: {dates[0]} to {dates[-1]} ({len(dates)} dates)")

    config = dict(MLB_PM_CONFIG)
    if args.config:
        with open(args.config) as f:
            loaded = json.load(f)
            config.update(loaded.get('config', loaded))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_v2')

    if args.optimize > 0:
        best_config, results = run_optimization(
            backtest_fn=run_backtest,
            box_scores=box_scores, odds_data=odds_data,
            base_config=config, param_ranges=MLB_PM_PARAM_RANGES,
            iterations=args.optimize, objective=args.objective,
            verbose=args.verbose, output_dir=output_dir,
        )
        config = best_config
    else:
        print("\nRunning walk-forward backtest...")
        results = run_backtest(box_scores, odds_data, config, verbose=args.verbose)

    _print_results(results)

    if args.validate:
        full_validation_report(results, backtest_fn=run_backtest,
                               config=config, box_scores=box_scores,
                               odds_data=odds_data)

    os.makedirs(output_dir, exist_ok=True)
    save = {k: v for k, v in results.items() if k != 'picks'}
    save['config'] = config
    with open(os.path.join(output_dir, 'backtest_results.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)

    picks_out = [{k: v for k, v in p.items()
                  if k in ['date', 'player', 'stat', 'line', 'odds', 'bet_type',
                           'hit', 'actual', 'pnl', 'wager', 'edge', 'ev',
                           'combined_score', 'hit_rate', 'legs']}
                 for p in results.get('picks', [])]
    with open(os.path.join(output_dir, 'picks_detail.json'), 'w') as f:
        json.dump(picks_out, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def _print_results(results):
    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    print("\n" + "=" * 70)
    print("MLB HECE RESULTS")
    print("=" * 70)
    print(f"  Picks: {results['total_picks']}  Wins: {results['total_wins']}  "
          f"Acc: {acc*100:.1f}%")
    print(f"  ROI: {results['total_roi']*100:.1f}%  P&L: ${results['total_pnl']:+,.0f}")
    print(f"  Max DD: ${results.get('max_drawdown',0):,.0f}  "
          f"Active Days: {results.get('active_days',0)}")

    picks = results.get('picks', [])
    singles = [p for p in picks if p.get('bet_type') == 'single']
    if singles:
        print(f"\n  TOP SINGLES:")
        for p in sorted(singles, key=lambda x: x.get('edge', 0), reverse=True)[:8]:
            h = 'HIT' if p.get('hit') else 'MISS'
            print(f"    {p.get('date','')} {p['player']} {p['stat']} o{p['line']} "
                  f"({p['odds']:+d}) edge={p.get('edge',0):.3f} "
                  f"actual={p.get('actual','')} {h} ${p.get('pnl',0):+d}")


if __name__ == '__main__':
    main()
