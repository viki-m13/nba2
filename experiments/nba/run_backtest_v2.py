#!/usr/bin/env python3
"""
NBA Experiment v2 — Convergence Certainty Engine (CCE)

Usage:
    python experiments/nba/run_backtest_v2.py                      # backtest only
    python experiments/nba/run_backtest_v2.py --optimize 200       # optimize
    python experiments/nba/run_backtest_v2.py --validate           # validation suite
    python experiments/nba/run_backtest_v2.py --objective cce      # CCE objective
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_loader import load_nba_boxscores, load_nba_odds
from shared.validation import full_validation_report
from shared.optimizer import run_optimization, OBJECTIVE_FNS
from nba.strategy_v2 import run_backtest, CCE_CONFIG, CCE_PARAM_RANGES


def compute_cce_score(results):
    """
    CCE-specific objective: maximize ROI while enforcing 90%+ accuracy
    on positive-odds bets. Heavily penalizes accuracy below 90%.
    Volume matters but accuracy is king.
    """
    n_picks = results['total_picks']
    if n_picks < 8:
        return -1

    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    roi = results.get('total_roi', 0)
    pnl = results.get('total_pnl', 0)
    max_dd = results.get('max_drawdown', 0)

    # HARD FLOOR: accuracy must be >= 70% to score at all
    if acc < 0.70:
        return max(0.001, acc * 0.1)

    # Accuracy is KING — exponential tiers targeting 90%+
    if acc >= 0.97:
        acc_mult = 100.0
    elif acc >= 0.95:
        acc_mult = 60.0
    elif acc >= 0.93:
        acc_mult = 40.0
    elif acc >= 0.91:
        acc_mult = 25.0
    elif acc >= 0.90:
        acc_mult = 20.0
    elif acc >= 0.87:
        acc_mult = 12.0
    elif acc >= 0.85:
        acc_mult = 8.0
    elif acc >= 0.80:
        acc_mult = 4.0
    elif acc >= 0.75:
        acc_mult = 2.0
    else:
        acc_mult = 0.5

    # Volume: need enough picks for statistical significance
    if n_picks < 12:
        vol = n_picks / 12
    elif n_picks <= 30:
        vol = 1.0 + (n_picks - 12) * 0.04
    elif n_picks <= 60:
        vol = 1.72 + (n_picks - 30) * 0.03
    elif n_picks <= 100:
        vol = 2.62 + (n_picks - 60) * 0.02
    else:
        vol = 3.42 + (n_picks - 100) * 0.01

    # ROI bonus (at 90% acc on plus-money, ROI should be very high)
    if roi >= 1.0:
        roi_mult = 4.0
    elif roi >= 0.50:
        roi_mult = 3.0
    elif roi >= 0.30:
        roi_mult = 2.5
    elif roi >= 0.15:
        roi_mult = 2.0
    elif roi >= 0:
        roi_mult = 1.0
    else:
        roi_mult = 0.2

    # Drawdown penalty
    if pnl > 0:
        dd_penalty = max(0.5, 1 - max_dd / max(1, pnl) * 0.5)
    else:
        dd_penalty = 0.3

    # Active days spread
    active_days = results.get('active_days', 1)
    day_mult = min(2.0, 1.0 + active_days * 0.04)

    # P&L magnitude bonus
    pnl_mult = 1.0
    if pnl > 0:
        import math
        pnl_mult = 1 + math.log10(max(1, pnl)) * 0.3

    score = acc * acc_mult * vol * roi_mult * dd_penalty * day_mult * pnl_mult
    return score


# Register the CCE objective
OBJECTIVE_FNS['cce'] = compute_cce_score


def main():
    parser = argparse.ArgumentParser(description='NBA CCE Experiment v2')
    parser.add_argument('--optimize', type=int, default=0,
                        help='Number of optimization iterations')
    parser.add_argument('--validate', action='store_true',
                        help='Run full validation suite')
    parser.add_argument('--objective', default='cce',
                        choices=list(OBJECTIVE_FNS.keys()),
                        help='Optimization objective')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config JSON')
    args = parser.parse_args()

    print("=" * 70)
    print("NBA CONVERGENCE CERTAINTY ENGINE (CCE) v2")
    print("=" * 70)

    # Load data
    print("\nLoading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    print(f"  Boxscores: {len(box_scores)} games")
    print(f"  Odds: {len(odds_data)} records")

    # Filter out All-Star games
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    print(f"  After filtering: {len(box_scores)} regular games")

    dates = sorted(set(g['date'] for g in box_scores))
    print(f"  Date range: {dates[0]} to {dates[-1]} ({len(dates)} dates)")

    # Load config
    config = dict(CCE_CONFIG)
    if args.config:
        with open(args.config) as f:
            loaded = json.load(f)
            config.update(loaded.get('config', loaded))
        print(f"\n  Loaded config from {args.config}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_v2')

    if args.optimize > 0:
        best_config, results = run_optimization(
            backtest_fn=run_backtest,
            box_scores=box_scores,
            odds_data=odds_data,
            base_config=config,
            param_ranges=CCE_PARAM_RANGES,
            iterations=args.optimize,
            objective=args.objective,
            verbose=args.verbose,
            output_dir=output_dir,
        )
        config = best_config
    else:
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
                if k in ['date', 'player', 'stat', 'line', 'direction', 'odds',
                         'bet_type', 'hit', 'actual', 'pnl', 'wager', 'edge',
                         'ev', 'our_prob', 'min_window_hr', 'bootstrap_lower',
                         'overall_hr', 'h1_hr', 'h2_hr', 'std_distance',
                         'market_prob']}
        if 'legs' in p:
            pick['legs'] = p['legs']
        picks_out.append(pick)
    with open(picks_path, 'w') as f:
        json.dump(picks_out, f, indent=2)
    print(f"Picks saved to {picks_path}")


def _print_full_results(results):
    acc = results.get('accuracy', results.get('individual_accuracy', 0))

    print("\n" + "=" * 70)
    print("CCE BACKTEST RESULTS")
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

    # Direction breakdown
    over_total = results.get('over_total', 0)
    under_total = results.get('under_total', 0)
    over_wins = results.get('over_wins', 0)
    under_wins = results.get('under_wins', 0)
    if over_total > 0 or under_total > 0:
        print(f"\n  DIRECTION BREAKDOWN:")
        if over_total > 0:
            print(f"    Overs:  {over_total} picks, {over_wins} wins "
                  f"({over_wins/over_total*100:.1f}%)")
        if under_total > 0:
            print(f"    Unders: {under_total} picks, {under_wins} wins "
                  f"({under_wins/under_total*100:.1f}%)")

    if results.get('singles_total', 0) > 0:
        print(f"\n  SINGLES:")
        print(f"    Total: {results['singles_total']}, "
              f"Wins: {results['singles_wins']}")
        print(f"    Accuracy: {results['singles_accuracy']*100:.1f}%")
        print(f"    P&L: ${results['singles_pnl']:+,.0f}")

    if results.get('parlay_total', 0) > 0:
        print(f"\n  PARLAYS:")
        print(f"    Total: {results['parlay_total']}, "
              f"Wins: {results['parlay_wins']}")
        print(f"    Accuracy: {results['parlay_accuracy']*100:.1f}%")
        print(f"    P&L: ${results['parlay_pnl']:+,.0f}")
        parlay_by_legs = results.get('parlay_by_legs', {})
        for n_legs in sorted(parlay_by_legs.keys()):
            info = parlay_by_legs[n_legs]
            leg_acc = info['wins'] / info['total'] * 100 if info['total'] > 0 else 0
            print(f"    {n_legs}-leg: {info['total']} total, {info['wins']} wins "
                  f"({leg_acc:.0f}%), ${info['pnl']:+,.0f}")

    # Top picks by edge
    picks = results.get('picks', [])
    if picks:
        singles = [p for p in picks if p.get('bet_type') == 'single']
        if singles:
            print(f"\n  TOP SINGLES BY EDGE:")
            top = sorted(singles, key=lambda p: p.get('edge', 0), reverse=True)[:8]
            for p in top:
                hit_str = 'HIT' if p.get('hit') else 'MISS'
                dir_str = p.get('direction', 'over')[0].upper()
                print(f"    {p.get('date','')} {p['player']} "
                      f"{p['stat']} {dir_str}{p['line']} "
                      f"({p['odds']:+d}) prob={p.get('our_prob',0):.3f} "
                      f"edge={p.get('edge',0):.3f} "
                      f"actual={p.get('actual','')} {hit_str} "
                      f"${p.get('pnl',0):+d}")

    # Daily P&L
    daily = results.get('daily', [])
    if daily:
        print(f"\n  DAILY P&L:")
        for d in daily:
            print(f"    {d['date']}: {d['n_picks']} picks, "
                  f"{d['wins']} wins, ${d['pnl']:+d}")


if __name__ == '__main__':
    main()
