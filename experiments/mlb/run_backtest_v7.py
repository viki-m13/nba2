"""MLB V7 Backtest Runner"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_loader import load_mlb_boxscores, load_mlb_odds
from mlb.strategy_v7_parlays import run_backtest, MLB_V7_CONFIG, MLB_V7_CONFIG_RELAXED

def run_all():
    print("Loading MLB data...")
    box_scores = load_mlb_boxscores()
    odds_data = load_mlb_odds()
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    configs = [
        ("V7 Default", MLB_V7_CONFIG),
        ("V7 Relaxed", MLB_V7_CONFIG_RELAXED),
        ("V7 2-leg only", {**MLB_V7_CONFIG, 'PARLAY_MAX_LEGS': 2, 'PARLAY_MIN_COMBINED_ODDS': -300}),
        ("V7 Parlays only", {**MLB_V7_CONFIG, 'SINGLES_ENABLED': False}),
        ("V7 Volume push", {
            **MLB_V7_CONFIG,
            'LEG_MIN_HR_15': 0.60, 'LEG_MIN_HR_10': 0.55, 'LEG_MIN_HR_5': 0.55,
            'LEG_MIN_CLEARANCE': -2.0, 'LEG_MIN_WINDOWS_PASSING': 1,
            'BEQ_MIN_TRUE_PROB': 0.65, 'PARLAY_MIN_COMBINED_ODDS': -400,
            'PARLAY_MIN_EV': 0.01, 'MAX_PARLAYS_PER_DAY': 10, 'MIN_GAMES': 8,
        }),
        ("V7 Ultra-safe", {
            **MLB_V7_CONFIG,
            'LEG_MIN_HR_15': 0.85, 'LEG_MIN_HR_10': 0.80, 'LEG_MIN_HR_5': 0.80,
            'BEQ_MIN_TRUE_PROB': 0.82, 'LEG_MIN_CLEARANCE': 0.5,
        }),
    ]

    print(f"\n{'Config':<40} {'Tot':>5} {'Win':>5} {'Acc%':>7} {'Days':>5} {'Day%':>7} "
          f"{'P&L':>8} {'ROI%':>7} {'Prly':>5} {'PAcc':>7} {'PLeg':>5} {'PLAcc':>7}")
    print("-" * 130)

    for name, cfg in configs:
        r = run_backtest(box_scores, odds_data, cfg, verbose=False)
        tp = r['total_picks']
        if tp > 0:
            print(f"{name:<40} {tp:>5} {r['total_wins']:>5} {r['accuracy']*100:>7.1f} "
                  f"{r['active_days']:>5} {r['day_coverage']*100:>7.1f} ${r['total_pnl']:>+7d} "
                  f"{r['total_roi']*100:>7.1f} {r['parlay_total']:>5} "
                  f"{r['parlay_accuracy']*100:>7.1f} {r['parlay_leg_total']:>5} "
                  f"{r['parlay_leg_accuracy']*100:>7.1f}")
        else:
            print(f"{name:<40} {'0':>5} {'—':>5} {'—':>7} {'0':>5} {'0.0':>7} "
                  f"{'$0':>8} {'—':>7} {'0':>5} {'—':>7} {'0':>5} {'—':>7}")

    # Verbose best
    print(f"\n{'='*60}\nVERBOSE: V7 Relaxed\n{'='*60}")
    r = run_backtest(box_scores, odds_data, MLB_V7_CONFIG_RELAXED, verbose=True)
    print(f"\n  Summary: {r['total_picks']} picks, {r['total_wins']} wins, "
          f"{r['accuracy']*100:.1f}%, ${r['total_pnl']:+d}")

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'output_v7')
    os.makedirs(output_dir, exist_ok=True)

if __name__ == '__main__':
    run_all()
