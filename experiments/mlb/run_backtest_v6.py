"""
MLB V6 Backtest Runner — Poisson Dual-Polarity Engine (PDPE)
=============================================================
Tests multiple configurations of the V6 strategy for MLB.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_loader import load_mlb_boxscores, load_mlb_odds
from mlb.strategy_v6 import run_backtest, MLB_V6_CONFIG, MLB_V6_CONFIG_PLUS200

def run_all():
    print("Loading MLB data...")
    box_scores = load_mlb_boxscores()
    odds_data = load_mlb_odds()
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    configs = [
        ("V6 Default (+100, dual-polarity)", MLB_V6_CONFIG),

        ("V6 Relaxed (clr=-0.5, HR>=0.55)", {
            **MLB_V6_CONFIG,
            'OVER_MIN_CLEARANCE': -0.5, 'UNDER_MIN_CLEARANCE': -0.5,
            'OVER_MIN_HR_15': 0.55, 'OVER_MIN_HR_10': 0.50, 'OVER_MIN_HR_5': 0.55,
            'UNDER_MIN_HR_15': 0.55, 'UNDER_MIN_HR_10': 0.50, 'UNDER_MIN_HR_5': 0.55,
            'CWU_REQUIRE_ALL': False,
            'BEQ_MIN_EDGE': 0.02,
            'MIN_EV': 0.03,
            'ACR_MAX_DAILY': 10,
            'ACR_MIN_CONFIDENCE': 0.15,
            'MIN_GAMES': 8,
        }),

        ("V6 Under-only", {
            **MLB_V6_CONFIG,
            'OVER_MIN_HR_15': 9.0,
            'ACR_MAX_DAILY': 8,
        }),

        ("V6 Over-only", {
            **MLB_V6_CONFIG,
            'UNDER_MIN_HR_15': 9.0,
            'ACR_MAX_DAILY': 8,
        }),

        ("V6 TB-friendly (relaxed)", {
            **MLB_V6_CONFIG,
            'OVER_MIN_CLEARANCE': -1.0, 'UNDER_MIN_CLEARANCE': -1.0,
            'OVER_MIN_HR_15': 0.55, 'UNDER_MIN_HR_15': 0.55,
            'CWU_REQUIRE_ALL': False,
            'BEQ_MIN_EDGE': 0.02,
            'MIN_EV': 0.03,
            'ACR_MAX_DAILY': 10,
            'ACR_MIN_CONFIDENCE': 0.15,
            'MIN_GAMES': 8,
            'POISSON_MAX_DISCREPANCY': 0.30,
        }),

        ("V6 No Poisson validation", {
            **MLB_V6_CONFIG,
            'POISSON_ENABLED': False,
            'OVER_MIN_CLEARANCE': -0.5, 'UNDER_MIN_CLEARANCE': -0.5,
            'OVER_MIN_HR_15': 0.60, 'UNDER_MIN_HR_15': 0.60,
            'CWU_REQUIRE_ALL': False,
            'ACR_MAX_DAILY': 8,
            'ACR_MIN_CONFIDENCE': 0.20,
        }),

        ("V6 +200 min odds", MLB_V6_CONFIG_PLUS200),

        ("V6 +200 relaxed", {
            **MLB_V6_CONFIG_PLUS200,
            'OVER_MIN_CLEARANCE': -0.5, 'UNDER_MIN_CLEARANCE': -0.5,
            'OVER_MIN_HR_15': 0.55, 'UNDER_MIN_HR_15': 0.55,
            'CWU_REQUIRE_ALL': False,
            'BEQ_MIN_EDGE': 0.03,
            'MIN_EV': 0.05,
            'ACR_MAX_DAILY': 10,
            'ACR_MIN_CONFIDENCE': 0.15,
        }),

        ("V6 Volume push", {
            **MLB_V6_CONFIG,
            'OVER_MIN_CLEARANCE': -2.0, 'UNDER_MIN_CLEARANCE': -2.0,
            'OVER_MIN_HR_15': 0.50, 'OVER_MIN_HR_10': 0.45, 'OVER_MIN_HR_5': 0.50,
            'UNDER_MIN_HR_15': 0.50, 'UNDER_MIN_HR_10': 0.45, 'UNDER_MIN_HR_5': 0.50,
            'CWU_REQUIRE_ALL': False,
            'CWU_MIN_CONVERGENCE': 0.20,
            'BEQ_MIN_EDGE': 0.01,
            'MIN_EV': 0.02,
            'ACR_MAX_DAILY': 15,
            'ACR_MIN_CONFIDENCE': 0.10,
            'MIN_GAMES': 8,
            'POISSON_ENABLED': False,
            'ABVG_MIN_AVG': 2.0,
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

            score = acc * 0.4 + day_pct * 0.3 + min(100, max(0, roi)) * 0.3
            if score > best_score:
                best_score = score
                best_results = r
                best_name = name
        else:
            print(f"{name:<45} {'0':>6} {'—':>6} {'—':>7} {'0':>6} {'0.0':>7} "
                  f"{'$0':>8} {'—':>7} {'0':>6} {'—':>7} {'0':>6} {'—':>7}")

    if best_results:
        print(f"\n{'='*60}")
        print(f"BEST: {best_name}")
        print(f"  Picks: {best_results['total_picks']}, Wins: {best_results['total_wins']}, "
              f"Acc: {best_results['accuracy']*100:.1f}%")
        print(f"  Days: {best_results['active_days']}/{best_results['total_days']} "
              f"({best_results['day_coverage']*100:.1f}%)")
        print(f"  P&L: ${best_results['total_pnl']:+d}, ROI: {best_results['total_roi']*100:.1f}%")

        # Show picks
        for p in sorted(best_results['picks'], key=lambda x: x['date']):
            dir_str = 'O' if p['direction'] == 'over' else 'U'
            status = 'W' if p['hit'] else 'L'
            print(f"    {p['date']} {p['player']:<22} {p['stat']:>3} {dir_str}{p['line']:>5.1f} "
                  f"+{p['odds']:>4} → {p['actual']:>3} [{status}] ${p['pnl']:>+5d}")

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'output_v6')
    os.makedirs(output_dir, exist_ok=True)
    if best_results:
        save = {k: v for k, v in best_results.items() if k != 'picks'}
        save['picks_summary'] = [{
            'date': p['date'], 'player': p['player'], 'stat': p['stat'],
            'direction': p['direction'], 'line': p['line'], 'odds': p['odds'],
            'actual': p['actual'], 'hit': p['hit'], 'pnl': p['pnl'],
        } for p in best_results['picks']]
        with open(os.path.join(output_dir, 'v6_best_results.json'), 'w') as f:
            json.dump(save, f, indent=2)


if __name__ == '__main__':
    run_all()
