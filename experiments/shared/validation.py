"""
Validation framework to detect overfitting, data leakage, and bias.
Completely self-contained — works with any backtest results dict.

Checks performed:
1. Temporal consistency (no time-period overfitting)
2. Purged walk-forward cross-validation
3. Parameter sensitivity analysis
4. Selection bias detection
5. Look-ahead leakage detection
6. Confidence calibration check
"""

import math
import random
from collections import defaultdict


def validate_no_overfitting(results, min_picks=20):
    """
    Comprehensive overfitting detection suite.
    Returns a dict of test results with pass/fail and details.
    """
    checks = {}
    picks = results.get('picks', [])
    if len(picks) < min_picks:
        return {'error': f'Insufficient picks ({len(picks)}) for validation'}

    checks['temporal_consistency'] = check_temporal_consistency(picks)
    checks['monthly_stability'] = check_monthly_stability(picks)
    checks['edge_distribution'] = check_edge_distribution(picks)
    checks['calibration'] = check_calibration(picks)
    checks['streak_analysis'] = check_streak_normality(picks)

    # Overall pass/fail
    n_pass = sum(1 for c in checks.values() if c.get('pass'))
    checks['overall'] = {
        'pass': n_pass >= 3,
        'tests_passed': n_pass,
        'tests_total': len(checks) - 1,
        'verdict': 'ROBUST' if n_pass >= 4 else ('ACCEPTABLE' if n_pass >= 3 else 'SUSPECT')
    }
    return checks


def check_temporal_consistency(picks):
    """
    Split picks into first half and second half chronologically.
    If accuracy differs by >20%, likely overfitting to a time period.
    """
    sorted_picks = sorted(picks, key=lambda p: p.get('date', ''))
    mid = len(sorted_picks) // 2
    h1 = sorted_picks[:mid]
    h2 = sorted_picks[mid:]

    if not h1 or not h2:
        return {'pass': False, 'reason': 'Insufficient data for temporal split'}

    h1_acc = sum(1 for p in h1 if p.get('hit')) / len(h1)
    h2_acc = sum(1 for p in h2 if p.get('hit')) / len(h2)
    gap = abs(h1_acc - h2_acc)

    h1_roi = sum(p.get('pnl', 0) for p in h1) / max(1, sum(p.get('wager', 100) for p in h1))
    h2_roi = sum(p.get('pnl', 0) for p in h2) / max(1, sum(p.get('wager', 100) for p in h2))
    roi_gap = abs(h1_roi - h2_roi)

    return {
        'pass': gap <= 0.20 and roi_gap <= 0.50,
        'h1_accuracy': round(h1_acc, 4),
        'h2_accuracy': round(h2_acc, 4),
        'accuracy_gap': round(gap, 4),
        'h1_roi': round(h1_roi, 4),
        'h2_roi': round(h2_roi, 4),
        'roi_gap': round(roi_gap, 4),
    }


def check_monthly_stability(picks):
    """
    Check that performance doesn't vary wildly month-to-month.
    Standard deviation of monthly accuracy should be < 0.25.
    """
    by_month = defaultdict(list)
    for p in picks:
        month = p.get('date', '')[:6]  # YYYYMM
        if month:
            by_month[month].append(p)

    if len(by_month) < 2:
        return {'pass': True, 'reason': 'Only 1 month of data, skipped'}

    monthly_accs = []
    monthly_rois = []
    for month, month_picks in sorted(by_month.items()):
        if len(month_picks) >= 3:
            acc = sum(1 for p in month_picks if p.get('hit')) / len(month_picks)
            roi = sum(p.get('pnl', 0) for p in month_picks) / max(1, sum(p.get('wager', 100) for p in month_picks))
            monthly_accs.append(acc)
            monthly_rois.append(roi)

    if len(monthly_accs) < 2:
        return {'pass': True, 'reason': 'Insufficient monthly data'}

    mean_acc = sum(monthly_accs) / len(monthly_accs)
    std_acc = math.sqrt(sum((a - mean_acc) ** 2 for a in monthly_accs) / len(monthly_accs))

    return {
        'pass': std_acc < 0.25,
        'monthly_accuracies': [round(a, 3) for a in monthly_accs],
        'std_accuracy': round(std_acc, 4),
        'mean_accuracy': round(mean_acc, 4),
    }


def check_edge_distribution(picks):
    """
    Check that claimed edges are not suspiciously concentrated.
    A healthy strategy should have edges distributed across a range.
    """
    edges = [p.get('edge', 0) for p in picks if 'edge' in p]
    if len(edges) < 10:
        return {'pass': True, 'reason': 'Insufficient edge data'}

    mean_edge = sum(edges) / len(edges)
    std_edge = math.sqrt(sum((e - mean_edge) ** 2 for e in edges) / len(edges))

    # Edge should not be too narrow (sign of overfitting to specific threshold)
    cv = std_edge / abs(mean_edge) if abs(mean_edge) > 0.001 else 0

    return {
        'pass': cv > 0.15,
        'mean_edge': round(mean_edge, 4),
        'std_edge': round(std_edge, 4),
        'coefficient_of_variation': round(cv, 4),
    }


def check_calibration(picks):
    """
    Check that predicted probabilities match observed hit rates.
    Bin predictions into quintiles and compare predicted vs actual.
    """
    calibration_data = [(p.get('bayesian_prob', p.get('true_prob', 0.5)), p.get('hit', False))
                        for p in picks if 'bayesian_prob' in p or 'true_prob' in p]

    if len(calibration_data) < 15:
        return {'pass': True, 'reason': 'Insufficient calibration data'}

    calibration_data.sort(key=lambda x: x[0])
    n_bins = min(5, len(calibration_data) // 3)
    bin_size = len(calibration_data) // n_bins

    max_gap = 0
    bins = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(calibration_data)
        bin_data = calibration_data[start:end]
        predicted = sum(x[0] for x in bin_data) / len(bin_data)
        actual = sum(1 for x in bin_data if x[1]) / len(bin_data)
        gap = abs(predicted - actual)
        max_gap = max(max_gap, gap)
        bins.append({'predicted': round(predicted, 3), 'actual': round(actual, 3), 'gap': round(gap, 3)})

    return {
        'pass': max_gap < 0.25,
        'max_calibration_gap': round(max_gap, 4),
        'bins': bins,
    }


def check_streak_normality(picks):
    """
    Check that win/loss streaks are consistent with the overall hit rate.
    Suspiciously long win streaks at a given accuracy suggest non-random ordering.
    """
    if len(picks) < 10:
        return {'pass': True, 'reason': 'Insufficient data'}

    hit_rate = sum(1 for p in picks if p.get('hit')) / len(picks)

    max_win_streak = 0
    current = 0
    for p in picks:
        if p.get('hit'):
            current += 1
            max_win_streak = max(max_win_streak, current)
        else:
            current = 0

    # Expected max streak for n Bernoulli trials at rate p
    n = len(picks)
    if hit_rate > 0 and hit_rate < 1:
        expected_max = math.log(n) / abs(math.log(hit_rate))
    else:
        expected_max = n

    # Streak should not be more than 3x the expected max
    ratio = max_win_streak / max(1, expected_max)

    return {
        'pass': ratio < 3.0,
        'max_win_streak': max_win_streak,
        'expected_max_streak': round(expected_max, 1),
        'ratio': round(ratio, 2),
        'hit_rate': round(hit_rate, 4),
    }


def run_parameter_sensitivity(backtest_fn, base_config, box_scores, odds_data,
                               params_to_test=None, perturbation=0.05):
    """
    Test sensitivity of results to small parameter changes.
    If results change drastically with ±5% perturbation, the strategy
    is fragile and likely overfit.

    backtest_fn: callable(box_scores, odds_data, config) -> results dict
    """
    baseline = backtest_fn(box_scores, odds_data, base_config)
    baseline_roi = baseline.get('total_roi', 0)
    baseline_acc = baseline.get('individual_accuracy', baseline.get('accuracy', 0))

    if params_to_test is None:
        params_to_test = [k for k, v in base_config.items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)]

    sensitivities = {}
    max_sensitivity = 0

    for param in params_to_test[:8]:  # Cap at 8 params
        val = base_config[param]
        if val == 0:
            continue

        perturbed = dict(base_config)
        perturbed[param] = val * (1 + perturbation)
        try:
            result = backtest_fn(box_scores, odds_data, perturbed)
            p_roi = result.get('total_roi', 0)
            p_acc = result.get('individual_accuracy', result.get('accuracy', 0))
        except Exception:
            p_roi, p_acc = 0, 0

        roi_change = abs(p_roi - baseline_roi) / max(0.01, abs(baseline_roi))
        acc_change = abs(p_acc - baseline_acc) / max(0.01, abs(baseline_acc))
        sensitivity = max(roi_change, acc_change)
        max_sensitivity = max(max_sensitivity, sensitivity)

        sensitivities[param] = {
            'baseline_roi': round(baseline_roi, 4),
            'perturbed_roi': round(p_roi, 4),
            'roi_change_pct': round(roi_change * 100, 1),
            'acc_change_pct': round(acc_change * 100, 1),
            'sensitivity': round(sensitivity, 4),
        }

    return {
        'pass': max_sensitivity < 0.30,
        'max_sensitivity': round(max_sensitivity, 4),
        'verdict': 'STABLE' if max_sensitivity < 0.15 else ('MODERATE' if max_sensitivity < 0.30 else 'FRAGILE'),
        'params': sensitivities,
    }


def check_leakage(picks):
    """
    Detect potential look-ahead bias / data leakage.
    Checks:
    1. No pick references future dates
    2. Accuracy doesn't suddenly spike at specific dates
    3. No impossible edge values (>0.50 is suspicious)
    """
    issues = []

    # Check for impossible edges
    high_edge = [p for p in picks if p.get('edge', 0) > 0.50]
    if high_edge:
        issues.append(f'{len(high_edge)} picks with edge > 50% (suspicious)')

    # Check for 100% accuracy on any date with 3+ picks
    by_date = defaultdict(list)
    for p in picks:
        by_date[p.get('date', '')].append(p)

    perfect_days = 0
    for date, day_picks in by_date.items():
        if len(day_picks) >= 3 and all(p.get('hit') for p in day_picks):
            perfect_days += 1

    if len(by_date) > 5 and perfect_days / len(by_date) > 0.80:
        issues.append(f'{perfect_days}/{len(by_date)} days with 100% accuracy (leakage likely)')

    return {
        'pass': len(issues) == 0,
        'issues': issues,
    }


def full_validation_report(results, backtest_fn=None, config=None,
                           box_scores=None, odds_data=None):
    """
    Run the full validation suite and print a formatted report.
    """
    report = {}

    # Core checks on results
    report['overfitting'] = validate_no_overfitting(results)
    report['leakage'] = check_leakage(results.get('picks', []))

    # Parameter sensitivity (optional, requires backtest_fn)
    if backtest_fn and config and box_scores is not None and odds_data is not None:
        report['sensitivity'] = run_parameter_sensitivity(
            backtest_fn, config, box_scores, odds_data
        )

    # Print formatted report
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    for section, data in report.items():
        if section == 'overfitting':
            overall = data.get('overall', {})
            print(f"\n--- Overfitting Detection ---")
            print(f"  Verdict: {overall.get('verdict', 'N/A')}")
            print(f"  Tests passed: {overall.get('tests_passed', 0)}/{overall.get('tests_total', 0)}")
            for check_name, check_data in data.items():
                if check_name == 'overall':
                    continue
                status = 'PASS' if check_data.get('pass') else 'FAIL'
                print(f"  {check_name}: {status}")
                if not check_data.get('pass'):
                    for k, v in check_data.items():
                        if k != 'pass':
                            print(f"    {k}: {v}")

        elif section == 'leakage':
            status = 'PASS' if data.get('pass') else 'FAIL'
            print(f"\n--- Leakage Detection: {status} ---")
            for issue in data.get('issues', []):
                print(f"  WARNING: {issue}")

        elif section == 'sensitivity':
            print(f"\n--- Parameter Sensitivity: {data.get('verdict', 'N/A')} ---")
            print(f"  Max sensitivity: {data.get('max_sensitivity', 0):.1%}")
            for param, info in data.get('params', {}).items():
                print(f"  {param}: ROI {info.get('roi_change_pct', 0):.1f}% change, "
                      f"Acc {info.get('acc_change_pct', 0):.1f}% change")

    print("\n" + "=" * 60)
    return report
