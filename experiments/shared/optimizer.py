"""
Self-contained AutoResearch-style optimizer for experiments.
Monotonic ratchet with temporal consistency regularization.
Does NOT import from src/ — completely independent.

Usage:
    from shared.optimizer import run_optimization
    best_config, results = run_optimization(
        backtest_fn=strategy.run_backtest,
        box_scores=boxes, odds_data=odds,
        base_config=DEFAULT_CONFIG,
        param_ranges=PARAM_RANGES,
        iterations=100,
        objective='roi',  # or 'accuracy' or 'balanced' or 'max_roi'
    )
"""

import random
import time
import json
import os
import math
from copy import deepcopy


def compute_roi_score(results):
    """
    ROI-focused objective function.
    Prioritizes ROI while maintaining reasonable accuracy.
    """
    if results['total_picks'] < 10:
        return -1

    roi = results.get('total_roi', 0)
    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    n_picks = results['total_picks']
    max_dd = results.get('max_drawdown', 0)
    total_wager = results.get('total_wagered', 1)

    # Volume: prefer 15-100 picks
    if n_picks < 15:
        vol = n_picks / 15
    elif n_picks <= 100:
        vol = 1.0
    else:
        vol = 1.0 + (n_picks - 100) * 0.002

    # Drawdown penalty
    dd_ratio = max_dd / max(1, total_wager)
    dd_penalty = max(0.3, 1 - dd_ratio * 2)

    # ROI tiers (primary objective)
    if roi >= 1.0:
        roi_mult = 4.0
    elif roi >= 0.50:
        roi_mult = 3.0
    elif roi >= 0.30:
        roi_mult = 2.5
    elif roi >= 0.15:
        roi_mult = 2.0
    elif roi >= 0.05:
        roi_mult = 1.5
    elif roi >= 0:
        roi_mult = 1.0
    else:
        roi_mult = 0.3

    # Accuracy floor (don't sacrifice too much accuracy)
    if acc >= 0.90:
        acc_mult = 2.0
    elif acc >= 0.80:
        acc_mult = 1.5
    elif acc >= 0.70:
        acc_mult = 1.0
    elif acc >= 0.60:
        acc_mult = 0.7
    else:
        acc_mult = 0.3

    score = (1 + max(0, roi)) * acc * vol * dd_penalty * roi_mult * acc_mult
    return score


def compute_max_roi_score(results):
    """
    Extreme ROI objective — targets 1000%+ ROI.
    Heavily rewards ROI, accepts lower accuracy for plus-money payouts.
    """
    if results['total_picks'] < 5:
        return -1

    roi = results.get('total_roi', 0)
    pnl = results.get('total_pnl', 0)
    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    n_picks = results['total_picks']
    max_dd = results.get('max_drawdown', 0)
    total_wager = results.get('total_wagered', 1)

    if roi <= 0:
        return max(0.001, (1 + roi) * 0.1)

    # Volume: need at least some picks for statistical significance
    if n_picks < 8:
        vol = n_picks / 8
    elif n_picks <= 50:
        vol = 1.0
    else:
        vol = 1.0 + (n_picks - 50) * 0.003

    # Drawdown relative to P&L (acceptable if drawdown < total profit)
    if pnl > 0:
        dd_ratio = max_dd / max(1, pnl)
        dd_penalty = max(0.5, 1 - dd_ratio * 0.5)
    else:
        dd_penalty = 0.5

    # ROI is the KING metric — exponential scaling
    if roi >= 10.0:
        roi_mult = 50.0
    elif roi >= 5.0:
        roi_mult = 30.0
    elif roi >= 3.0:
        roi_mult = 20.0
    elif roi >= 2.0:
        roi_mult = 12.0
    elif roi >= 1.0:
        roi_mult = 8.0
    elif roi >= 0.50:
        roi_mult = 4.0
    elif roi >= 0.20:
        roi_mult = 2.0
    else:
        roi_mult = 1.0

    # Accuracy: for plus-money, even 40% can be profitable
    if acc >= 0.70:
        acc_mult = 2.0
    elif acc >= 0.50:
        acc_mult = 1.5
    elif acc >= 0.35:
        acc_mult = 1.0
    else:
        acc_mult = 0.5

    # P&L magnitude bonus
    pnl_mult = 1 + math.log10(max(1, pnl)) * 0.3

    score = roi * roi_mult * vol * dd_penalty * acc_mult * pnl_mult
    return score


def compute_accuracy_score(results):
    """Accuracy-focused objective (maintains >90% accuracy)."""
    if results['total_picks'] < 10:
        return -1

    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    roi = results.get('total_roi', 0)
    n_picks = results['total_picks']

    vol = min(1.0, n_picks / 40)
    if n_picks >= 80:
        vol *= 1.3

    if acc >= 0.95:
        acc_bonus = 8.0
    elif acc >= 0.93:
        acc_bonus = 6.0
    elif acc >= 0.90:
        acc_bonus = 4.0
    elif acc >= 0.85:
        acc_bonus = 2.0
    elif acc >= 0.80:
        acc_bonus = 1.0
    else:
        acc_bonus = 0.3

    roi_mult = 1.5 if roi >= 0.50 else (1.2 if roi >= 0.20 else 1.0)

    return acc * acc_bonus * vol * roi_mult * (1 + max(0, roi))


def compute_balanced_score(results):
    """Balanced objective: accuracy * ROI * volume."""
    if results['total_picks'] < 10:
        return -1

    roi = results.get('total_roi', 0)
    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    n_picks = results['total_picks']

    vol = min(1.0, n_picks / 30)
    if n_picks >= 60:
        vol *= 1.2

    roi_component = max(0.1, 1 + roi)
    acc_component = acc

    if acc >= 0.90:
        acc_bonus = 3.0
    elif acc >= 0.85:
        acc_bonus = 2.0
    elif acc >= 0.80:
        acc_bonus = 1.5
    else:
        acc_bonus = 0.5

    if roi >= 0.50:
        roi_bonus = 2.5
    elif roi >= 0.20:
        roi_bonus = 2.0
    elif roi >= 0.05:
        roi_bonus = 1.5
    elif roi >= 0:
        roi_bonus = 1.0
    else:
        roi_bonus = 0.3

    return acc_component * roi_component * vol * acc_bonus * roi_bonus


OBJECTIVE_FNS = {
    'roi': compute_roi_score,
    'accuracy': compute_accuracy_score,
    'balanced': compute_balanced_score,
    'max_roi': compute_max_roi_score,
}


def temporal_consistency_penalty(results, max_gap=0.20):
    """
    Penalize configs where first-half vs second-half performance
    differs significantly. Prevents time-period overfitting.
    Relaxed for max_roi since smaller sample sizes are expected.
    """
    picks = results.get('picks', [])
    if len(picks) < 8:
        return 1.0

    sorted_picks = sorted(picks, key=lambda p: p.get('date', ''))
    mid = len(sorted_picks) // 2
    h1 = sorted_picks[:mid]
    h2 = sorted_picks[mid:]

    h1_acc = sum(1 for p in h1 if p.get('hit')) / len(h1) if h1 else 0
    h2_acc = sum(1 for p in h2 if p.get('hit')) / len(h2) if h2 else 0
    gap = abs(h1_acc - h2_acc)

    if gap > max_gap:
        return max(0.3, 1 - (gap - max_gap) * 2)
    return 1.0


def run_optimization(backtest_fn, box_scores, odds_data, base_config, param_ranges,
                     iterations=100, objective='roi', verbose=True, output_dir=None):
    """
    AutoResearch-style monotonic ratchet optimizer.

    Args:
        backtest_fn: callable(box_scores, odds_data, config) -> results dict
        box_scores: list of game box scores
        odds_data: list of odds records
        base_config: starting config dict
        param_ranges: dict of param_name -> (min, max) tuples
        iterations: number of optimization iterations
        objective: 'roi', 'accuracy', 'balanced', or 'max_roi'
        verbose: print progress
        output_dir: directory to save results (optional)

    Returns:
        (best_config, final_results) tuple
    """
    score_fn = OBJECTIVE_FNS.get(objective, compute_roi_score)

    print("=" * 70)
    print(f"OPTIMIZER — Objective: {objective.upper()}")
    print("=" * 70)

    # Baseline
    config = dict(base_config)
    print("\nRunning baseline walk-forward backtest...")
    baseline = backtest_fn(box_scores, odds_data, config)
    raw_score = score_fn(baseline)
    penalty = temporal_consistency_penalty(baseline)
    best_score = raw_score * penalty

    _print_results("BASELINE", baseline, best_score)

    best_config = dict(config)
    improvements = 0
    log = []

    for i in range(1, iterations + 1):
        # Multi-parameter mutation (1-3 params)
        n_params = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        params = random.sample(list(param_ranges.keys()), min(n_params, len(param_ranges)))

        test_config = dict(best_config)
        changed = []
        for param in params:
            low, high = param_ranges[param]
            current = test_config.get(param, base_config.get(param))

            if isinstance(current, int) and isinstance(low, int) and isinstance(high, int):
                new_val = random.randint(int(low), int(high))
            else:
                current_f = float(current) if current is not None else (low + high) / 2
                delta = (high - low) * 0.25
                new_val = current_f + random.uniform(-delta, delta)
                new_val = max(low, min(high, round(new_val, 4)))

            test_config[param] = new_val
            changed.append((param, current, new_val))

        start = time.time()
        try:
            results = backtest_fn(box_scores, odds_data, test_config)
        except Exception as e:
            if verbose:
                print(f"[{i}/{iterations}] ERROR: {e}")
            continue
        elapsed = time.time() - start

        raw_score = score_fn(results)
        penalty = temporal_consistency_penalty(results)
        score = raw_score * penalty

        status = 'KEEP' if score > best_score else 'DISCARD'
        acc = results.get('accuracy', results.get('individual_accuracy', 0))

        if verbose and (status == 'KEEP' or i % 25 == 0):
            changes_str = ', '.join(f"{p}: {o} -> {n}" for p, o, n in changed[:2])
            print(f"\n[{i}/{iterations}] {changes_str}")
            print(f"  Score: {score:.4f} {'>' if score > best_score else '<='} {best_score:.4f} -> {status}")
            print(f"  Acc: {acc*100:.1f}%, ROI: {results['total_roi']*100:.1f}%, "
                  f"P&L: ${results['total_pnl']:+,.0f}, Picks: {results['total_picks']}, Time: {elapsed:.1f}s")

        log.append({
            'iteration': i, 'params': [(p, o, n) for p, o, n in changed],
            'score': score, 'accuracy': acc, 'roi': results['total_roi'],
            'pnl': results['total_pnl'],
            'picks': results['total_picks'], 'status': status,
        })

        if status == 'KEEP':
            for p, _, v in changed:
                best_config[p] = v
            best_score = score
            improvements += 1
            if verbose:
                print(f"  ** IMPROVEMENT #{improvements} **")

    # Final evaluation
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    final = backtest_fn(box_scores, odds_data, best_config)
    _print_results("FINAL RESULT", final, best_score)
    print(f"  Total improvements: {improvements}/{iterations}")

    # Save if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, f'{objective}_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'config': best_config,
                'score': best_score,
                'results_summary': {
                    'accuracy': final.get('accuracy', final.get('individual_accuracy', 0)),
                    'roi': final['total_roi'],
                    'total_picks': final['total_picks'],
                    'total_pnl': final['total_pnl'],
                },
                'improvements': improvements,
                'iterations': iterations,
            }, f, indent=2)
        print(f"\n  Config saved to {config_path}")

        log_path = os.path.join(output_dir, f'{objective}_optimization_log.json')
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2, default=str)

    return best_config, final


def _print_results(label, results, score=None):
    """Print formatted backtest results."""
    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    print(f"\n  --- {label} ---")
    print(f"  Total picks: {results['total_picks']}")
    print(f"  Wins: {results['total_wins']}")
    print(f"  Accuracy: {acc*100:.1f}%")
    print(f"  ROI: {results['total_roi']*100:.1f}%")
    print(f"  P&L: ${results['total_pnl']:+,.0f}")
    print(f"  Wagered: ${results.get('total_wagered', 0):,.0f}")
    print(f"  Max Drawdown: ${results.get('max_drawdown', 0):,.0f}")
    if results.get('singles_total', 0) > 0:
        print(f"  Singles: {results['singles_total']} ({results['singles_accuracy']*100:.1f}% acc, ${results['singles_pnl']:+,.0f})")
    if results.get('parlay_total', 0) > 0:
        print(f"  Parlays: {results['parlay_total']} ({results['parlay_accuracy']*100:.1f}% acc, ${results['parlay_pnl']:+,.0f})")
    if score is not None:
        print(f"  Score: {score:.4f}")
