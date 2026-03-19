#!/usr/bin/env python3
"""
MLB V8 CSPE Backtest Runner — Poisson-Bayesian Optimization
============================================================
Searches for the optimal MLB V8 configuration targeting 90%+ accuracy.
"""

import sys, os, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_loader import load_mlb_boxscores, load_mlb_odds
from mlb.strategy_v8_cspe import run_backtest, MLB_V8_CONFIG


def print_results(name, r):
    for key, stats in sorted(r.get('by_legs', {}).items()):
        label = f"{name} [{key}L]"
        acc = stats['accuracy'] * 100
        roi = stats['roi'] * 100
        lacc = stats['leg_accuracy'] * 100
        print(f"{label:<55} | {stats['total']:>4} | {stats['wins']:>3} | {acc:>6.1f}% | "
              f"${stats['pnl']:>+7d} | {roi:>+7.1f}% | {stats['legs_total']:>4} | {lacc:>6.1f}%")
    total_roi = r['total_roi'] * 100
    total_acc = r['accuracy'] * 100
    legs_total = sum(s['legs_total'] for s in r['by_legs'].values())
    legs_hits = sum(s['legs_hits'] for s in r['by_legs'].values())
    leg_acc = legs_hits / legs_total * 100 if legs_total > 0 else 0
    print(f"  >> TOTAL: {r['total_picks']} picks, {r['total_wins']}W, "
          f"{total_acc:.1f}% acc, {leg_acc:.1f}% leg acc, "
          f"${r['total_pnl']:+d}, {total_roi:+.1f}% ROI, {r['active_days']} days\n")


def temporal_validation(picks):
    if not picks:
        return
    all_dates = sorted(set(p['date'] for p in picks))
    mid = len(all_dates) // 2
    h1_dates = set(all_dates[:mid])
    h2_dates = set(all_dates[mid:])
    h1 = [p for p in picks if p['date'] in h1_dates]
    h2 = [p for p in picks if p['date'] in h2_dates]
    for label, half in [("H1", h1), ("H2", h2)]:
        if half:
            h_acc = sum(1 for p in half if p['hit']) / len(half) * 100
            h_pnl = sum(p['pnl'] for p in half)
            h_wag = sum(p['wager'] for p in half)
            h_roi = h_pnl / h_wag * 100 if h_wag > 0 else 0
            legs = [l for p in half for l in p.get('legs', [])]
            l_acc = sum(1 for l in legs if l.get('hit')) / len(legs) * 100 if legs else 0
            print(f"  {label}: {len(half)} picks, {h_acc:.1f}% acc, {l_acc:.1f}% leg acc, "
                  f"${h_pnl:+d}, {h_roi:+.1f}% ROI")
    if h1 and h2:
        h1_acc = sum(1 for p in h1 if p['hit']) / len(h1) * 100
        h2_acc = sum(1 for p in h2 if p['hit']) / len(h2) * 100
        gap = abs(h1_acc - h2_acc)
        print(f"  Gap: {gap:.1f}pp ({'STABLE' if gap <= 15 else 'UNSTABLE'})")


def monthly_breakdown(picks):
    if not picks:
        return
    by_month = defaultdict(list)
    for p in picks:
        by_month[p['date'][:6]].append(p)
    for month in sorted(by_month.keys()):
        mp = by_month[month]
        acc = sum(1 for p in mp if p['hit']) / len(mp) * 100
        pnl = sum(p['pnl'] for p in mp)
        legs = [l for p in mp for l in p.get('legs', [])]
        l_acc = sum(1 for l in legs if l.get('hit')) / len(legs) * 100 if legs else 0
        print(f"  {month}: {len(mp)} picks, {acc:.1f}% acc, {l_acc:.1f}% leg acc, ${pnl:+d}")


def run_all():
    print("Loading MLB data...")
    box_scores = load_mlb_boxscores()
    odds_data = load_mlb_odds()
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records\n")

    print(f"{'Config':<55} | {'#':>4} | {'W':>3} | {'Acc':>7} | "
          f"{'P&L':>8} | {'ROI':>8} | {'Legs':>4} | {'LAcc':>7}")
    print("-" * 120)

    configs = []

    # ========= BASELINE =========
    configs.append(("BASELINE", {**MLB_V8_CONFIG}))

    # ========= VARY POISSON MIN PROB =========
    for prob in [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96]:
        configs.append((f"PBF>={prob:.2f}", {**MLB_V8_CONFIG, 'PBF_MIN_PROB': prob}))

    # ========= VARY HARD_MIN_HR =========
    for hr in [0.80, 0.85, 0.90, 0.95, 1.0]:
        configs.append((f"HR>={hr:.2f}", {**MLB_V8_CONFIG, 'HARD_MIN_HR': hr}))

    # ========= VARY PERFECT HR WINDOW =========
    for pw in [5, 6, 8, 10, 12]:
        configs.append((f"PERF_W={pw}", {**MLB_V8_CONFIG, 'PERFECT_HR_WINDOW': pw}))

    # ========= NO PERFECT HR =========
    configs.append(("NO_PERF_HR", {**MLB_V8_CONFIG, 'PERFECT_HR_REQUIRED': False}))

    # ========= VARY MIN_STREAK =========
    for ms in [1, 2, 3, 4, 5]:
        configs.append((f"STREAK>={ms}", {**MLB_V8_CONFIG, 'MIN_STREAK': ms}))

    # ========= VARY CONSISTENCY =========
    for cmin in [0.70, 0.80, 0.90, 1.0]:
        configs.append((f"CONSIST>={cmin:.2f}", {
            **MLB_V8_CONFIG, 'CONSISTENCY_MIN_HR': cmin}))

    # ========= 2L ONLY vs 2L+3L =========
    configs.append(("2L_ONLY", {**MLB_V8_CONFIG, 'PARLAY_LEGS': [2]}))
    configs.append(("3L_ONLY", {**MLB_V8_CONFIG, 'PARLAY_LEGS': [3]}))

    # ========= ODDS RANGE =========
    configs.append(("ODDS_-350_-120", {**MLB_V8_CONFIG, 'LEG_MIN_ODDS': -350, 'LEG_MAX_ODDS': -120}))
    configs.append(("ODDS_-300_-150", {**MLB_V8_CONFIG, 'LEG_MIN_ODDS': -300, 'LEG_MAX_ODDS': -150}))
    configs.append(("ODDS_-260_-150", {**MLB_V8_CONFIG, 'LEG_MIN_ODDS': -260, 'LEG_MAX_ODDS': -150}))
    configs.append(("ODDS_-500_-110", {**MLB_V8_CONFIG, 'LEG_MIN_ODDS': -500, 'LEG_MAX_ODDS': -110}))

    # ========= MIN_GAMES =========
    for mg in [8, 10, 12, 15, 20]:
        configs.append((f"MINGAMES={mg}", {**MLB_V8_CONFIG, 'MIN_GAMES': mg}))

    # ========= PBF_DECAY =========
    for d in [0.88, 0.90, 0.92, 0.94, 0.96]:
        configs.append((f"DECAY={d:.2f}", {**MLB_V8_CONFIG, 'PBF_DECAY': d}))

    # ========= COMBINED OPTIMIZED =========
    configs.append(("OPT_A", {
        **MLB_V8_CONFIG,
        'PBF_MIN_PROB': 0.92, 'HARD_MIN_HR': 0.90,
        'PERFECT_HR_WINDOW': 6, 'MIN_STREAK': 3,
        'CONSISTENCY_MIN_HR': 0.90,
        'PARLAY_LEGS': [2],
    }))
    configs.append(("OPT_B", {
        **MLB_V8_CONFIG,
        'PBF_MIN_PROB': 0.94, 'HARD_MIN_HR': 0.90,
        'PERFECT_HR_WINDOW': 5, 'MIN_STREAK': 2,
        'CONSISTENCY_MIN_HR': 0.80,
        'PARLAY_LEGS': [2],
    }))
    configs.append(("OPT_C", {
        **MLB_V8_CONFIG,
        'PBF_MIN_PROB': 0.90, 'HARD_MIN_HR': 0.85,
        'PERFECT_HR_WINDOW': 8, 'MIN_STREAK': 2,
        'CONSISTENCY_MIN_HR': 0.80,
        'LEG_MIN_ODDS': -350, 'LEG_MAX_ODDS': -120,
        'PARLAY_LEGS': [2],
    }))
    configs.append(("OPT_D", {
        **MLB_V8_CONFIG,
        'PBF_MIN_PROB': 0.92, 'HARD_MIN_HR': 0.90,
        'PERFECT_HR_WINDOW': 6, 'MIN_STREAK': 3,
        'CONSISTENCY_MIN_HR': 0.90,
        'LEG_MIN_ODDS': -350, 'LEG_MAX_ODDS': -130,
        'PARLAY_LEGS': [2], 'ELITE_TOP_N': 4,
    }))
    configs.append(("OPT_E", {
        **MLB_V8_CONFIG,
        'PBF_MIN_PROB': 0.95, 'HARD_MIN_HR': 0.95,
        'PERFECT_HR_WINDOW': 5, 'MIN_STREAK': 3,
        'CONSISTENCY_MIN_HR': 0.90,
        'PARLAY_LEGS': [2], 'ELITE_TOP_N': 4,
        'MIN_GAMES': 10,
    }))
    configs.append(("OPT_F_relax", {
        **MLB_V8_CONFIG,
        'PBF_MIN_PROB': 0.88, 'HARD_MIN_HR': 0.80,
        'PERFECT_HR_REQUIRED': False, 'MIN_STREAK': 1,
        'CONSISTENCY_MIN_HR': 0.70,
        'PARLAY_LEGS': [2], 'ELITE_TOP_N': 8,
        'LEG_MIN_ODDS': -500, 'LEG_MAX_ODDS': -110,
    }))
    configs.append(("OPT_G_ultra", {
        **MLB_V8_CONFIG,
        'PBF_MIN_PROB': 0.96, 'HARD_MIN_HR': 1.0,
        'PERFECT_HR_WINDOW': 10, 'MIN_STREAK': 5,
        'CONSISTENCY_MIN_HR': 1.0,
        'PARLAY_LEGS': [2],
    }))
    configs.append(("OPT_H_balanced", {
        **MLB_V8_CONFIG,
        'PBF_MIN_PROB': 0.92, 'HARD_MIN_HR': 0.90,
        'PERFECT_HR_WINDOW': 5, 'MIN_STREAK': 2,
        'CONSISTENCY_MIN_HR': 0.80,
        'LEG_MIN_ODDS': -500, 'LEG_MAX_ODDS': -120,
        'PARLAY_LEGS': [2, 3], 'ELITE_TOP_N': 6,
        'MIN_GAMES': 10,
    }))

    best_score = 0
    best_name = ""
    best_cfg = None

    for name, cfg in configs:
        r = run_backtest(box_scores, odds_data, cfg, verbose=False)
        if r['total_picks'] > 0:
            print_results(name, r)
            # Score: accuracy * volume weight (prefer 90%+ with 10+ picks)
            vol = min(1.0, r['total_picks'] / 10)
            score = r['accuracy'] * 100 * vol if r['total_picks'] >= 3 else 0
            if score > best_score:
                best_score = score
                best_name = name
                best_cfg = cfg

    if best_cfg:
        print(f"\n{'='*70}")
        print(f"BEST CONFIG: {best_name} (Score: {best_score:.1f})")
        print(f"{'='*70}")
        r = run_backtest(box_scores, odds_data, best_cfg, verbose=True)

        print(f"\n--- Temporal Validation ---")
        temporal_validation(r['picks'])
        print(f"\n--- Monthly Breakdown ---")
        monthly_breakdown(r['picks'])

        output_dir = os.path.join(os.path.dirname(__file__), 'output_v8')
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'v8_cspe_results.json'), 'w') as f:
            json.dump({
                'config_name': best_name,
                'config': best_cfg,
                'total_picks': r['total_picks'],
                'total_wins': r['total_wins'],
                'accuracy': r['accuracy'],
                'total_pnl': r['total_pnl'],
                'total_roi': r['total_roi'],
                'active_days': r['active_days'],
                'by_legs': r['by_legs'],
            }, f, indent=2)
        with open(os.path.join(output_dir, 'v8_cspe_picks.json'), 'w') as f:
            json.dump(r['picks'], f, indent=2)
        print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    run_all()
