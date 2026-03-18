"""
NBA V6 Backtest Runner — Dual-Polarity Regime Engine (DPRE)
============================================================
Tests multiple configurations of the V6 strategy which bets both overs AND unders.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_loader import load_nba_boxscores, load_nba_odds
from nba.strategy_v6 import run_backtest, NBA_V6_CONFIG, NBA_V6_CONFIG_PLUS200

def run_all():
    print("Loading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    # Filter All-Star games
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    configs = [
        # --- Baseline configs ---
        ("V6 Default (+100, dual-polarity)", NBA_V6_CONFIG),

        ("V6 Relaxed floor/ceiling (clr=-0.5)", {
            **NBA_V6_CONFIG,
            'OVER_MIN_CLEARANCE': -0.5, 'UNDER_MIN_CLEARANCE': -0.5,
            'OVER_MIN_HR_20': 0.65, 'OVER_MIN_HR_10': 0.60, 'OVER_MIN_HR_5': 0.70,
            'UNDER_MIN_HR_20': 0.65, 'UNDER_MIN_HR_10': 0.60, 'UNDER_MIN_HR_5': 0.70,
            'CWU_REQUIRE_ALL': False,
            'ACR_MAX_DAILY': 10,
        }),

        ("V6 Tight floor/ceiling (clr=1.0)", {
            **NBA_V6_CONFIG,
            'OVER_MIN_CLEARANCE': 1.0, 'UNDER_MIN_CLEARANCE': 1.0,
            'ACR_MAX_DAILY': 10,
        }),

        ("V6 Under-only", {
            **NBA_V6_CONFIG,
            'OVER_MIN_HR_20': 9.0,  # Effectively disable overs
        }),

        ("V6 Over-only (compare to V5)", {
            **NBA_V6_CONFIG,
            'UNDER_MIN_HR_20': 9.0,  # Effectively disable unders
        }),

        # --- Wider odds ---
        ("V6 Wider odds (+100 to +500)", {
            **NBA_V6_CONFIG,
            'MAX_ODDS': 500,
            'OVER_MIN_HR_20': 0.70, 'OVER_MIN_HR_5': 0.75,
            'UNDER_MIN_HR_20': 0.70, 'UNDER_MIN_HR_5': 0.75,
            'ACR_MAX_DAILY': 10,
        }),

        # --- +200 minimum ---
        ("V6 +200 min odds", NBA_V6_CONFIG_PLUS200),

        ("V6 +200 relaxed (clr=-0.5)", {
            **NBA_V6_CONFIG_PLUS200,
            'OVER_MIN_CLEARANCE': -0.5, 'UNDER_MIN_CLEARANCE': -0.5,
            'OVER_MIN_HR_20': 0.60, 'OVER_MIN_HR_10': 0.55, 'OVER_MIN_HR_5': 0.60,
            'UNDER_MIN_HR_20': 0.60, 'UNDER_MIN_HR_10': 0.55, 'UNDER_MIN_HR_5': 0.60,
            'CWU_REQUIRE_ALL': False,
            'BEQ_MIN_EDGE': 0.03,
            'MIN_EV': 0.05,
            'ACR_MAX_DAILY': 12,
            'ACR_MIN_CONFIDENCE': 0.20,
        }),

        # --- Volume-focused ---
        ("V6 Volume push (relax all)", {
            **NBA_V6_CONFIG,
            'OVER_MIN_CLEARANCE': -1.0, 'UNDER_MIN_CLEARANCE': -1.0,
            'OVER_MIN_HR_20': 0.55, 'OVER_MIN_HR_10': 0.50, 'OVER_MIN_HR_5': 0.55,
            'UNDER_MIN_HR_20': 0.55, 'UNDER_MIN_HR_10': 0.50, 'UNDER_MIN_HR_5': 0.55,
            'CWU_REQUIRE_ALL': False,
            'CWU_MIN_CONVERGENCE': 0.30,
            'BEQ_MIN_EDGE': 0.02,
            'MIN_EV': 0.03,
            'ESI_MIN_STABILITY': 0.08,
            'ACR_MAX_DAILY': 15,
            'ACR_MIN_CONFIDENCE': 0.15,
            'MIN_GAMES': 10,
        }),

        # --- High-accuracy focused ---
        ("V6 Strict accuracy (HR>=0.85)", {
            **NBA_V6_CONFIG,
            'OVER_MIN_HR_20': 0.85, 'OVER_MIN_HR_10': 0.80, 'OVER_MIN_HR_5': 0.90,
            'UNDER_MIN_HR_20': 0.85, 'UNDER_MIN_HR_10': 0.80, 'UNDER_MIN_HR_5': 0.90,
            'OVER_MIN_CLEARANCE': 1.0, 'UNDER_MIN_CLEARANCE': 1.0,
            'BEQ_MIN_EDGE': 0.08,
            'MIN_EV': 0.10,
            'ACR_MAX_DAILY': 10,
        }),

        # --- Regime-heavy ---
        ("V6 Strong regime boost", {
            **NBA_V6_CONFIG,
            'RCF_PACE_WEIGHT': 0.15,
            'RCF_DEFENSE_WEIGHT': 0.18,
            'RCF_REST_WEIGHT': 0.10,
            'RCF_STABILITY_WEIGHT': 0.12,
            'OVER_MIN_CLEARANCE': -0.5, 'UNDER_MIN_CLEARANCE': -0.5,
            'OVER_MIN_HR_20': 0.65, 'UNDER_MIN_HR_20': 0.65,
            'ACR_MIN_CONFIDENCE': 0.35,
            'ACR_MAX_DAILY': 8,
        }),

        # --- No CWU requirement (just hit rate + BEQ) ---
        ("V6 No CWU, just HR+BEQ", {
            **NBA_V6_CONFIG,
            'CWU_REQUIRE_ALL': False,
            'CWU_MIN_CONVERGENCE': 0.0,
            'OVER_MIN_CLEARANCE': -2.0, 'UNDER_MIN_CLEARANCE': -2.0,
            'OVER_MIN_HR_20': 0.70, 'OVER_MIN_HR_10': 0.65, 'OVER_MIN_HR_5': 0.75,
            'UNDER_MIN_HR_20': 0.70, 'UNDER_MIN_HR_10': 0.65, 'UNDER_MIN_HR_5': 0.75,
            'BEQ_MIN_EDGE': 0.05,
            'MIN_EV': 0.06,
            'ACR_MAX_DAILY': 10,
            'ACR_MIN_CONFIDENCE': 0.25,
        }),
    ]

    print(f"\n{'Config':<45} {'Picks':>6} {'Wins':>6} {'Acc%':>7} {'Days':>6} {'Day%':>7} "
          f"{'P&L':>8} {'ROI%':>7} {'Over':>6} {'OAcc%':>7} {'Under':>6} {'UAcc%':>7}")
    print("-" * 145)

    best_results = None
    best_name = ""
    best_score = 0

    for name, cfg in configs:
        r = run_backtest(box_scores, odds_data, cfg, verbose=False)
        tp = r['total_picks']
        if tp > 0:
            acc = r['accuracy'] * 100
            day_pct = r['day_coverage'] * 100
            roi = r['total_roi'] * 100
            print(f"{name:<45} {tp:>6} {r['total_wins']:>6} {acc:>7.1f} "
                  f"{r['active_days']:>6} {day_pct:>7.1f} ${r['total_pnl']:>+7d} {roi:>7.1f} "
                  f"{r['over_picks']:>6} {r['over_accuracy']*100:>7.1f} "
                  f"{r['under_picks']:>6} {r['under_accuracy']*100:>7.1f}")

            # Score: balance accuracy, volume, and profitability
            score = acc * 0.4 + day_pct * 0.3 + min(100, max(0, roi)) * 0.3
            if score > best_score:
                best_score = score
                best_results = r
                best_name = name
        else:
            print(f"{name:<45} {'0':>6} {'—':>6} {'—':>7} {'0':>6} {'0.0':>7} "
                  f"{'$0':>8} {'—':>7} {'0':>6} {'—':>7} {'0':>6} {'—':>7}")

    # Show detailed results for the best config
    if best_results:
        print(f"\n{'='*60}")
        print(f"BEST CONFIG: {best_name}")
        print(f"{'='*60}")
        print(f"  Picks: {best_results['total_picks']}, Wins: {best_results['total_wins']}")
        print(f"  Accuracy: {best_results['accuracy']*100:.1f}%")
        print(f"  Active Days: {best_results['active_days']}/{best_results['total_days']} "
              f"({best_results['day_coverage']*100:.1f}%)")
        print(f"  P&L: ${best_results['total_pnl']:+d}, ROI: {best_results['total_roi']*100:.1f}%")
        print(f"  Over: {best_results['over_picks']} picks, {best_results['over_accuracy']*100:.1f}% acc")
        print(f"  Under: {best_results['under_picks']} picks, {best_results['under_accuracy']*100:.1f}% acc")
        print(f"  Max Drawdown: ${best_results['max_drawdown']}")

        # Show individual picks
        print(f"\n  {'Date':<12} {'Player':<25} {'Stat':>5} {'Dir':>5} {'Line':>6} {'Odds':>6} "
              f"{'Actual':>6} {'Result':>6} {'P&L':>7}")
        print(f"  {'-'*100}")
        for p in sorted(best_results['picks'], key=lambda x: x['date']):
            dir_str = 'OVER' if p['direction'] == 'over' else 'UNDR'
            status = 'WIN' if p['hit'] else 'LOSS'
            print(f"  {p['date']:<12} {p['player']:<25} {p['stat']:>5} {dir_str:>5} "
                  f"{p['line']:>6.1f} +{p['odds']:>5} {p['actual']:>6} {status:>6} ${p['pnl']:>+6d}")

    # Run verbose version of best config
    print(f"\n{'='*60}")
    print(f"VERBOSE: V6 Default")
    print(f"{'='*60}")
    r = run_backtest(box_scores, odds_data, NBA_V6_CONFIG, verbose=True)
    print(f"\n  Summary: {r['total_picks']} picks, {r['total_wins']} wins, "
          f"{r['accuracy']*100:.1f}% acc, {r['active_days']} days")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'output_v6')
    os.makedirs(output_dir, exist_ok=True)

    if best_results:
        save_results = {k: v for k, v in best_results.items() if k != 'picks'}
        save_results['picks_summary'] = [{
            'date': p['date'], 'player': p['player'], 'stat': p['stat'],
            'direction': p['direction'], 'line': p['line'], 'odds': p['odds'],
            'actual': p['actual'], 'hit': p['hit'], 'pnl': p['pnl'],
        } for p in best_results['picks']]

        with open(os.path.join(output_dir, 'v6_best_results.json'), 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\n  Results saved to output_v6/v6_best_results.json")


if __name__ == '__main__':
    run_all()
