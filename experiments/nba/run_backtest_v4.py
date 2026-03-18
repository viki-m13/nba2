#!/usr/bin/env python3
"""
Run V4 AFCE backtest with random search optimization.
"""
import sys, os, json, copy, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_loader import load_nba_boxscores, load_nba_odds
from nba.strategy_v4 import NBA_V4_CONFIG, NBA_V4_PARAM_RANGES, run_backtest

def main():
    print("Loading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores if g.get('home','') not in ['STARS','WORLD'] and g.get('away','') not in ['STARS','WORLD']]
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    # First run with default config
    print("\n" + "=" * 60)
    print("V4 AFCE — Default Config")
    print("=" * 60)
    results = run_backtest(box_scores, odds_data, NBA_V4_CONFIG, verbose=True)
    print_results(results)

    # Random search optimization
    print("\n" + "=" * 60)
    print("RANDOM SEARCH OPTIMIZATION (500 iterations)")
    print("=" * 60)

    best_score = compute_score(results)
    best_config = copy.deepcopy(NBA_V4_CONFIG)
    best_results = results

    random.seed(42)
    N_ITER = 500

    for i in range(N_ITER):
        cfg = copy.deepcopy(NBA_V4_CONFIG)

        # Randomly perturb parameters
        for param, (lo, hi) in NBA_V4_PARAM_RANGES.items():
            if isinstance(lo, int) and isinstance(hi, int):
                cfg[param] = random.randint(lo, hi)
            else:
                cfg[param] = round(random.uniform(lo, hi), 4)

        r = run_backtest(box_scores, odds_data, cfg, verbose=False)
        score = compute_score(r)

        if score > best_score and r['total_picks'] >= 5:
            best_score = score
            best_config = cfg
            best_results = r
            print(f"  [{i+1}/{N_ITER}] NEW BEST: {r['total_picks']} picks, "
                  f"{r['total_wins']} wins, "
                  f"{r['accuracy']*100:.1f}% acc, "
                  f"{r['day_coverage']*100:.1f}% days, "
                  f"${r['total_pnl']:+d} P&L, "
                  f"{r['total_roi']*100:.1f}% ROI "
                  f"[score={score:.1f}]")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N_ITER}] checked... best score={best_score:.1f}")

    if best_results and best_results['total_picks'] > 0:
        print("\n" + "=" * 60)
        print("BEST V4 CONFIG RESULTS")
        print("=" * 60)
        print_results(best_results)

        print("\nBest config:")
        for k, v in sorted(best_config.items()):
            if k in NBA_V4_PARAM_RANGES:
                default = NBA_V4_CONFIG.get(k)
                marker = " *" if v != default else ""
                print(f"  {k}: {v}{marker}")

        # Save best config
        out_dir = os.path.join(os.path.dirname(__file__), 'output_v4')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'v4_config.json'), 'w') as f:
            json.dump({
                'config': best_config,
                'results': {k: v for k, v in best_results.items() if k != 'picks'},
                'score': best_score,
                'strategy': 'AFCE_v4',
                'description': 'Adaptive Floor Convergence Engine — optimized for 90%+ accuracy with high volume on plus-money',
            }, f, indent=2)
        print(f"\nSaved to {out_dir}/v4_config.json")

        # Run verbose with best config
        print("\n" + "=" * 60)
        print("BEST CONFIG — DETAILED PICKS")
        print("=" * 60)
        run_backtest(box_scores, odds_data, best_config, verbose=True)
    else:
        print("\nNo qualifying configurations found. The strategy may need redesign.")


def compute_score(results):
    """Score function rewarding accuracy >= 90% AND day coverage >= 50%."""
    acc = results.get('accuracy', 0)
    coverage = results.get('day_coverage', 0)
    picks = results.get('total_picks', 0)
    roi = results.get('total_roi', 0)

    if picks < 5:
        return 0

    # Accuracy reward: exponential bonus at 90%+
    if acc >= 0.90:
        acc_score = 100 + (acc - 0.90) * 500
    elif acc >= 0.80:
        acc_score = 50 + (acc - 0.80) * 500
    elif acc >= 0.70:
        acc_score = 20 + (acc - 0.70) * 300
    else:
        acc_score = max(0, acc * 20)

    # Coverage reward
    if coverage >= 0.50:
        cov_score = 50 + (coverage - 0.50) * 100
    elif coverage >= 0.30:
        cov_score = 20 + (coverage - 0.30) * 150
    elif coverage >= 0.10:
        cov_score = (coverage - 0.10) * 100
    else:
        cov_score = 0

    # Volume bonus
    vol_score = min(30, picks * 0.5)

    # ROI bonus
    roi_score = max(0, min(50, roi * 50))

    return acc_score + cov_score + vol_score + roi_score


def print_results(results):
    print(f"  Picks: {results['total_picks']}")
    print(f"  Wins: {results['total_wins']}")
    print(f"  Accuracy: {results['accuracy']*100:.1f}%")
    print(f"  Active days: {results['active_days']}/{results['total_days']} ({results['day_coverage']*100:.1f}%)")
    print(f"  Total P&L: ${results['total_pnl']:+d}")
    print(f"  Total wagered: ${results['total_wagered']}")
    print(f"  ROI: {results['total_roi']*100:.1f}%")
    print(f"  Max drawdown: ${results['max_drawdown']}")


if __name__ == '__main__':
    main()
