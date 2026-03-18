#!/usr/bin/env python3
"""
NBA HECE v3 — Hybrid Ensemble Certainty Engine

Usage:
    python experiments/nba/run_backtest_v3.py                      # backtest only
    python experiments/nba/run_backtest_v3.py --optimize 500       # optimize
    python experiments/nba/run_backtest_v3.py --validate           # validation
    python experiments/nba/run_backtest_v3.py --objective hece     # HECE objective
"""

import sys, os, argparse, json, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_loader import load_nba_boxscores, load_nba_odds
from shared.validation import full_validation_report
from shared.optimizer import run_optimization, OBJECTIVE_FNS
from nba.strategy_v3 import run_backtest, HECE_CONFIG, HECE_PARAM_RANGES


def compute_hece_score(results):
    """
    HECE objective: maximize ROI while enforcing high accuracy on plus-money.
    Massive reward for 90%+ accuracy. Volume-aware to prevent tiny samples.
    """
    n_picks = results['total_picks']
    if n_picks < 8:
        return -1

    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    roi = results.get('total_roi', 0)
    pnl = results.get('total_pnl', 0)
    max_dd = results.get('max_drawdown', 0)

    if acc < 0.60:
        return max(0.001, acc * 0.1)

    # Accuracy tiers — exponential reward at 90%+
    if acc >= 0.97:   acc_mult = 150.0
    elif acc >= 0.95: acc_mult = 80.0
    elif acc >= 0.93: acc_mult = 50.0
    elif acc >= 0.91: acc_mult = 35.0
    elif acc >= 0.90: acc_mult = 25.0
    elif acc >= 0.87: acc_mult = 15.0
    elif acc >= 0.85: acc_mult = 10.0
    elif acc >= 0.80: acc_mult = 5.0
    elif acc >= 0.75: acc_mult = 3.0
    elif acc >= 0.70: acc_mult = 2.0
    else:             acc_mult = 0.5

    # Volume tiers
    if n_picks < 12:     vol = n_picks / 12
    elif n_picks <= 30:  vol = 1.0 + (n_picks - 12) * 0.05
    elif n_picks <= 60:  vol = 1.9 + (n_picks - 30) * 0.04
    elif n_picks <= 100: vol = 3.1 + (n_picks - 60) * 0.03
    else:                vol = 4.3 + (n_picks - 100) * 0.01

    # ROI bonus
    if roi >= 1.0:     roi_mult = 5.0
    elif roi >= 0.50:  roi_mult = 3.5
    elif roi >= 0.30:  roi_mult = 2.5
    elif roi >= 0.15:  roi_mult = 2.0
    elif roi >= 0:     roi_mult = 1.0
    else:              roi_mult = 0.2

    # Drawdown penalty
    dd_penalty = max(0.4, 1 - max_dd / max(1, abs(pnl) + 1) * 0.5) if pnl > 0 else 0.3

    # Active days spread
    active_days = results.get('active_days', 1)
    day_mult = min(2.5, 1.0 + active_days * 0.05)

    # P&L magnitude
    pnl_mult = 1 + math.log10(max(1, abs(pnl))) * 0.3 if pnl > 0 else 0.5

    return acc * acc_mult * vol * roi_mult * dd_penalty * day_mult * pnl_mult


OBJECTIVE_FNS['hece'] = compute_hece_score


def main():
    parser = argparse.ArgumentParser(description='NBA HECE v3 Experiment')
    parser.add_argument('--optimize', type=int, default=0)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--objective', default='hece',
                        choices=list(OBJECTIVE_FNS.keys()))
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("NBA HYBRID ENSEMBLE CERTAINTY ENGINE (HECE) v3")
    print("=" * 70)

    print("\nLoading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    dates = sorted(set(g['date'] for g in box_scores))
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")
    print(f"  Date range: {dates[0]} to {dates[-1]} ({len(dates)} dates)")

    config = dict(HECE_CONFIG)
    if args.config:
        with open(args.config) as f:
            loaded = json.load(f)
            config.update(loaded.get('config', loaded))
        print(f"  Loaded config from {args.config}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_v3')

    if args.optimize > 0:
        best_config, results = run_optimization(
            backtest_fn=run_backtest,
            box_scores=box_scores, odds_data=odds_data,
            base_config=config, param_ranges=HECE_PARAM_RANGES,
            iterations=args.optimize, objective=args.objective,
            verbose=args.verbose, output_dir=output_dir,
        )
        config = best_config
    else:
        print("\nRunning walk-forward backtest...")
        results = run_backtest(box_scores, odds_data, config, verbose=args.verbose)

    _print_results(results)

    if args.validate:
        print("\n\nRunning validation suite...")
        full_validation_report(results, backtest_fn=run_backtest,
                               config=config, box_scores=box_scores,
                               odds_data=odds_data)

    os.makedirs(output_dir, exist_ok=True)
    save = {k: v for k, v in results.items() if k != 'picks'}
    save['config'] = config
    with open(os.path.join(output_dir, 'backtest_results.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)

    picks_out = []
    for p in results.get('picks', []):
        pick = {k: v for k, v in p.items()
                if k in ['date', 'player', 'stat', 'line', 'direction', 'odds',
                         'bet_type', 'hit', 'actual', 'pnl', 'wager', 'edge',
                         'ev', 'combined_score', 'hit_rate', 'bayesian_prob',
                         'true_prob', 'market_prob', 'boost_total']}
        if 'legs' in p: pick['legs'] = p['legs']
        picks_out.append(pick)
    with open(os.path.join(output_dir, 'picks_detail.json'), 'w') as f:
        json.dump(picks_out, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def _print_results(results):
    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    print("\n" + "=" * 70)
    print("HECE v3 BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Picks: {results['total_picks']}  Wins: {results['total_wins']}  "
          f"Acc: {acc*100:.1f}%")
    print(f"  ROI: {results['total_roi']*100:.1f}%  P&L: ${results['total_pnl']:+,.0f}  "
          f"Wagered: ${results.get('total_wagered', 0):,.0f}")
    print(f"  Max DD: ${results.get('max_drawdown', 0):,.0f}  "
          f"Win Streak: {results.get('max_win_streak', 0)}  "
          f"Loss Streak: {results.get('max_loss_streak', 0)}")
    print(f"  Active Days: {results.get('active_days', 0)}  "
          f"Avg Daily P&L: ${results.get('avg_daily_pnl', 0):+,.0f}")

    if results.get('singles_total', 0):
        print(f"\n  SINGLES: {results['singles_total']} ({results['singles_accuracy']*100:.1f}%) "
              f"${results['singles_pnl']:+,.0f}")
    if results.get('parlay_total', 0):
        print(f"  PARLAYS: {results['parlay_total']} ({results['parlay_accuracy']*100:.1f}%) "
              f"${results['parlay_pnl']:+,.0f}")
        for n, info in sorted(results.get('parlay_by_legs', {}).items()):
            a = info['wins'] / info['total'] * 100 if info['total'] else 0
            print(f"    {n}-leg: {info['total']}x ({a:.0f}%) ${info['pnl']:+,.0f}")

    picks = results.get('picks', [])
    singles = [p for p in picks if p.get('bet_type') == 'single']
    if singles:
        print(f"\n  TOP SINGLES:")
        for p in sorted(singles, key=lambda x: x.get('edge', 0), reverse=True)[:8]:
            h = 'HIT' if p.get('hit') else 'MISS'
            print(f"    {p.get('date','')} {p['player']} {p['stat']} o{p['line']} "
                  f"({p['odds']:+d}) score={p.get('combined_score',0):.3f} "
                  f"edge={p.get('edge',0):.3f} actual={p.get('actual','')} "
                  f"{h} ${p.get('pnl',0):+d}")

    daily = results.get('daily', [])
    if daily:
        print(f"\n  DAILY:")
        for d in daily:
            print(f"    {d['date']}: {d['n_picks']}p {d['wins']}w ${d['pnl']:+d}")


if __name__ == '__main__':
    main()
