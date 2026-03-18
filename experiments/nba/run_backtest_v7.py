"""
NBA V7 Backtest Runner — Safe-Leg Parlay Amplification Engine
==============================================================
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_loader import load_nba_boxscores, load_nba_odds
from nba.strategy_v7_parlays import (
    run_backtest, NBA_V7_CONFIG, NBA_V7_CONFIG_RELAXED, NBA_V7_CONFIG_STRICT,
)

def run_all():
    print("Loading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    configs = [
        ("V7 Default (safe-leg parlays)", NBA_V7_CONFIG),
        ("V7 Relaxed (more legs)", NBA_V7_CONFIG_RELAXED),
        ("V7 Strict (highest accuracy)", NBA_V7_CONFIG_STRICT),

        # Wider odds range for more legs
        ("V7 Wider odds (-500 to -100)", {
            **NBA_V7_CONFIG,
            'LEG_MIN_ODDS': -500,
            'LEG_MAX_ODDS': -100,
            'LEG_MIN_HR_20': 0.80,
            'LEG_MIN_HR_10': 0.75,
            'LEG_MIN_HR_5': 0.75,
            'BEQ_MIN_TRUE_PROB': 0.78,
            'PARLAY_MIN_COMBINED_ODDS': -150,
            'MAX_PARLAYS_PER_DAY': 8,
        }),

        # Favoring 2-leg parlays (highest parlay accuracy)
        ("V7 2-leg only", {
            **NBA_V7_CONFIG,
            'PARLAY_MIN_LEGS': 2,
            'PARLAY_MAX_LEGS': 2,
            'LEG_MIN_HR_20': 0.85,
            'BEQ_MIN_TRUE_PROB': 0.85,
            'PARLAY_MIN_COMBINED_ODDS': -200,
            'MAX_PARLAYS_PER_DAY': 5,
        }),

        # 3-leg plus money target
        ("V7 3-leg +100 target", {
            **NBA_V7_CONFIG,
            'PARLAY_MIN_LEGS': 3,
            'PARLAY_MAX_LEGS': 3,
            'LEG_MIN_HR_20': 0.80,
            'LEG_MIN_HR_10': 0.75,
            'LEG_MIN_HR_5': 0.75,
            'BEQ_MIN_TRUE_PROB': 0.78,
            'PARLAY_MIN_COMBINED_ODDS': 100,
            'MAX_PARLAYS_PER_DAY': 5,
        }),

        # Volume: more leg candidates
        ("V7 Volume (relaxed gates)", {
            **NBA_V7_CONFIG,
            'LEG_MIN_HR_20': 0.75,
            'LEG_MIN_HR_10': 0.70,
            'LEG_MIN_HR_5': 0.70,
            'LEG_MIN_CLEARANCE': -1.0,
            'LEG_MIN_WINDOWS_PASSING': 2,
            'BEQ_MIN_TRUE_PROB': 0.72,
            'PARLAY_MIN_COMBINED_ODDS': -200,
            'PARLAY_MIN_EV': 0.02,
            'MAX_PARLAYS_PER_DAY': 10,
            'MIN_GAMES': 10,
        }),

        # Conservative: only very safe legs
        ("V7 Ultra-safe (HR>=0.90, prob>=0.88)", {
            **NBA_V7_CONFIG,
            'LEG_MIN_HR_20': 0.90,
            'LEG_MIN_HR_10': 0.85,
            'LEG_MIN_HR_5': 0.85,
            'LEG_MIN_CLEARANCE': 0.5,
            'BEQ_MIN_TRUE_PROB': 0.88,
            'PARLAY_MIN_COMBINED_ODDS': -300,
            'PARLAY_MAX_LEGS': 3,
        }),

        # Singles only mode
        ("V7 Singles only (no parlays)", {
            **NBA_V7_CONFIG,
            'PARLAY_MIN_LEGS': 99,  # Effectively disable parlays
            'SINGLES_ENABLED': True,
            'SINGLE_MIN_HR_20': 0.85,
            'SINGLE_MIN_TRUE_PROB': 0.82,
            'SINGLE_MIN_ODDS': -500,
            'SINGLE_MAX_ODDS': -110,
            'MAX_SINGLES_PER_DAY': 8,
        }),

        # Parlay-only (no singles)
        ("V7 Parlays only", {
            **NBA_V7_CONFIG,
            'SINGLES_ENABLED': False,
            'LEG_MIN_HR_20': 0.80,
            'BEQ_MIN_TRUE_PROB': 0.78,
            'PARLAY_MIN_COMBINED_ODDS': -200,
            'MAX_PARLAYS_PER_DAY': 6,
        }),
    ]

    print(f"\n{'Config':<45} {'Total':>6} {'Wins':>6} {'Acc%':>7} {'Days':>6} {'Day%':>7} "
          f"{'P&L':>8} {'ROI%':>7} {'PLeg':>6} {'PLAcc':>7} {'Prlys':>6} {'PAcc%':>7} {'PM':>4} {'PMW':>4}")
    print("-" * 155)

    all_results = {}
    best_score = 0
    best_name = ""
    best_results = None

    for name, cfg in configs:
        r = run_backtest(box_scores, odds_data, cfg, verbose=False)
        all_results[name] = r
        tp = r['total_picks']
        if tp > 0:
            acc = r['accuracy'] * 100
            day_pct = r['day_coverage'] * 100
            roi = r['total_roi'] * 100
            pl_acc = r['parlay_leg_accuracy'] * 100
            p_acc = r['parlay_accuracy'] * 100
            print(f"{name:<45} {tp:>6} {r['total_wins']:>6} {acc:>7.1f} "
                  f"{r['active_days']:>6} {day_pct:>7.1f} ${r['total_pnl']:>+7d} {roi:>7.1f} "
                  f"{r['parlay_leg_total']:>6} {pl_acc:>7.1f} "
                  f"{r['parlay_total']:>6} {p_acc:>7.1f} "
                  f"{r['plus_money_parlays']:>4} {r['plus_money_wins']:>4}")

            # Score: parlays with 90% accuracy and good volume
            score = (
                min(100, acc) * 0.3 +
                min(100, p_acc) * 0.25 +
                day_pct * 0.25 +
                min(100, max(0, roi)) * 0.20
            )
            if score > best_score:
                best_score = score
                best_name = name
                best_results = r
        else:
            print(f"{name:<45} {'0':>6} {'—':>6} {'—':>7} {'0':>6} {'0.0':>7} "
                  f"{'$0':>8} {'—':>7} {'0':>6} {'—':>7} {'0':>6} {'—':>7} {'0':>4} {'0':>4}")

    # Detailed output of best
    if best_results:
        print(f"\n{'='*70}")
        print(f"BEST: {best_name}")
        print(f"{'='*70}")
        r = best_results
        print(f"  Total: {r['total_picks']} picks, {r['total_wins']} wins, "
              f"{r['accuracy']*100:.1f}% accuracy")
        print(f"  Days: {r['active_days']}/{r['total_days']} ({r['day_coverage']*100:.1f}%)")
        print(f"  P&L: ${r['total_pnl']:+d}, ROI: {r['total_roi']*100:.1f}%")
        print(f"  Singles: {r['singles_total']} picks, {r['singles_wins']} wins, "
              f"{r['singles_accuracy']*100:.1f}% acc, ${r['singles_pnl']:+d}")
        print(f"  Parlays: {r['parlay_total']} bets, {r['parlay_wins']} wins, "
              f"{r['parlay_accuracy']*100:.1f}% acc, ${r['parlay_pnl']:+d}")
        print(f"  Parlay Legs: {r['parlay_leg_total']} total, {r['parlay_leg_wins']} wins, "
              f"{r['parlay_leg_accuracy']*100:.1f}% acc")
        print(f"  Plus-Money Parlays: {r['plus_money_parlays']}, {r['plus_money_wins']} wins, "
              f"{r['plus_money_accuracy']*100:.1f}% acc")
        if r.get('parlay_by_legs'):
            print(f"  By leg count:")
            for n, stats in sorted(r['parlay_by_legs'].items()):
                acc_n = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"    {n}-leg: {stats['total']} parlays, {stats['wins']} wins, "
                      f"{acc_n:.1f}% acc, ${stats['pnl']:+d}")

    # Run verbose on best
    if best_name:
        print(f"\n{'='*70}")
        print(f"VERBOSE RUN: {best_name}")
        print(f"{'='*70}")
        r = run_backtest(box_scores, odds_data,
                        [cfg for n, cfg in configs if n == best_name][0],
                        verbose=True)

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'output_v7')
    os.makedirs(output_dir, exist_ok=True)
    if best_results:
        save = {k: v for k, v in best_results.items() if k != 'picks'}
        save['picks_summary'] = []
        for p in best_results['picks']:
            entry = {
                'date': p['date'], 'bet_type': p['bet_type'],
                'hit': p['hit'], 'pnl': p['pnl'], 'wager': p['wager'],
            }
            if p['bet_type'] == 'parlay':
                entry['n_legs'] = p['n_legs']
                entry['parlay_american'] = p['parlay_american']
                entry['legs'] = p['legs']
            else:
                entry.update({
                    'player': p.get('player'), 'stat': p.get('stat'),
                    'line': p.get('line'), 'odds': p.get('odds'),
                    'actual': p.get('actual'),
                })
            save['picks_summary'].append(entry)

        with open(os.path.join(output_dir, 'v7_best_results.json'), 'w') as f:
            json.dump(save, f, indent=2)
        print(f"\n  Results saved to output_v7/")


if __name__ == '__main__':
    run_all()
