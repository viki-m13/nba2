#!/usr/bin/env python3
"""
NBA V8 CSPE Backtest Runner — Dual-Tier Convergence Score Parlay Engine
=======================================================================
Tests Core + Amplifier tiers and validates against overfitting.
"""

import sys, os, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_loader import load_nba_boxscores, load_nba_odds
from nba.strategy_v8_cspe import run_backtest, NBA_V8_CONFIG


def print_results(name, r):
    for key, stats in sorted(r.get('by_legs', {}).items()):
        label = f"{name} [{key}]"
        acc = stats['accuracy'] * 100
        roi = stats['roi'] * 100
        lacc = stats['leg_accuracy'] * 100
        print(f"{label:<50} | {stats['total']:>5} | {stats['wins']:>4} | {acc:>6.1f} | "
              f"${stats['pnl']:>+7d} | {roi:>+8.1f} | {stats['leg_total']:>5} | {lacc:>6.1f}")

    # Tier summary
    for tier, stats in r.get('by_tier', {}).items():
        roi = stats['roi'] * 100 if stats['wagered'] > 0 else 0
        acc = stats['accuracy'] * 100
        print(f"  >> {tier.upper()} TIER: {stats['total']} picks, {stats['wins']}W, "
              f"{acc:.1f}% acc, ${stats['pnl']:+d}, {roi:+.1f}% ROI")

    total_roi = r['total_roi'] * 100
    print(f"  >> COMBINED: {r['total_picks']} picks, {r['total_wins']}W, "
          f"${r['total_pnl']:+d}, {total_roi:+.1f}% ROI, {r['active_days']} active days\n")


def run_all():
    print("Loading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records\n")

    print(f"{'Config':<50} | {'Picks':>5} | {'Hit':>4} | {'Acc%':>6} | "
          f"{'P&L':>8} | {'ROI%':>8} | {'Legs':>5} | {'LAcc%':>6}")
    print("-" * 120)

    configs = []

    # Baseline dual-tier
    configs.append(("baseline_dual_tier", {**NBA_V8_CONFIG}))

    # Vary amplifier edge threshold
    for edge_min in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        configs.append((f"amp_edge>={edge_min:.2f}", {
            **NBA_V8_CONFIG,
            'AMP_MIN_EDGE': edge_min,
        }))

    # Vary amplifier hit rate threshold
    for hr_min in [0.45, 0.50, 0.55, 0.60, 0.65]:
        configs.append((f"amp_hr>={hr_min:.2f}", {
            **NBA_V8_CONFIG,
            'AMP_MIN_HR': hr_min,
        }))

    # Vary amplifier odds range
    configs.append(("amp_odds_wide", {
        **NBA_V8_CONFIG,
        'AMP_LEG_MAX_ODDS': 600,
    }))
    configs.append(("amp_plus_only", {
        **NBA_V8_CONFIG,
        'AMP_LEG_MIN_ODDS': -180,
        'AMP_LEG_MAX_ODDS': 500,
    }))
    configs.append(("amp_moderate", {
        **NBA_V8_CONFIG,
        'AMP_LEG_MIN_ODDS': -300,
        'AMP_LEG_MAX_ODDS': 200,
    }))

    # Vary amplifier parlay sizes
    configs.append(("amp_3L_only", {
        **NBA_V8_CONFIG,
        'AMP_PARLAY_LEGS': [3],
    }))
    configs.append(("amp_4L_only", {
        **NBA_V8_CONFIG,
        'AMP_PARLAY_LEGS': [4],
    }))
    configs.append(("amp_3_4L", {
        **NBA_V8_CONFIG,
        'AMP_PARLAY_LEGS': [3, 4],
    }))

    # Stats variations
    configs.append(("pts_only", {
        **NBA_V8_CONFIG,
        'STATS_ALLOWED': ['pts'],
    }))
    configs.append(("pts+ast", {
        **NBA_V8_CONFIG,
        'STATS_ALLOWED': ['pts', 'ast'],
    }))

    # More amplifier parlays per day
    configs.append(("amp_3parlays", {
        **NBA_V8_CONFIG,
        'AMP_MAX_PARLAYS': 3,
    }))
    configs.append(("amp_4parlays", {
        **NBA_V8_CONFIG,
        'AMP_MAX_PARLAYS': 4,
    }))

    # Amplifier-only (no core tier)
    configs.append(("amp_only", {
        **NBA_V8_CONFIG,
        'CORE_PARLAY_LEGS': [],  # Disable core
    }))

    # Aggressive amplifier
    configs.append(("aggressive_amp", {
        **NBA_V8_CONFIG,
        'AMP_MIN_HR': 0.50,
        'AMP_MIN_EDGE': 0.06,
        'AMP_MIN_BCF_LB': 0.42,
        'AMP_PARLAY_LEGS': [3, 4],
        'AMP_MAX_PARLAYS': 3,
        'AMP_LEG_MAX_ODDS': 500,
    }))

    # Very aggressive amplifier
    configs.append(("very_aggressive_amp", {
        **NBA_V8_CONFIG,
        'AMP_MIN_HR': 0.45,
        'AMP_MIN_EDGE': 0.05,
        'AMP_MIN_BCF_LB': 0.38,
        'AMP_PARLAY_LEGS': [3, 4, 5],
        'AMP_MAX_PARLAYS': 4,
        'AMP_LEG_MAX_ODDS': 600,
    }))

    # === HYBRID FOCUS CONFIGS ===
    # Hybrid only (no standalone core/amp)
    configs.append(("hybrid_only", {
        **NBA_V8_CONFIG,
        'CORE_PARLAY_LEGS': [],
        'AMP_PARLAY_LEGS': [],
        'HYBRID_ENABLED': True,
        'HYBRID_AMP_COUNT': [3, 4],
        'HYBRID_MAX_PARLAYS': 2,
    }))

    # Hybrid with wider amp odds
    configs.append(("hybrid_wide_odds", {
        **NBA_V8_CONFIG,
        'CORE_PARLAY_LEGS': [],
        'AMP_PARLAY_LEGS': [],
        'AMP_LEG_MAX_ODDS': 600,
        'HYBRID_ENABLED': True,
        'HYBRID_AMP_COUNT': [3, 4],
        'HYBRID_MAX_PARLAYS': 3,
    }))

    # Hybrid with plus-money amps
    configs.append(("hybrid_plus_amps", {
        **NBA_V8_CONFIG,
        'CORE_PARLAY_LEGS': [],
        'AMP_PARLAY_LEGS': [],
        'AMP_LEG_MIN_ODDS': -180,
        'AMP_LEG_MAX_ODDS': 500,
        'HYBRID_ENABLED': True,
        'HYBRID_AMP_COUNT': [3, 4],
        'HYBRID_MAX_PARLAYS': 3,
    }))

    # Hybrid with more amps per parlay
    configs.append(("hybrid_5amps", {
        **NBA_V8_CONFIG,
        'CORE_PARLAY_LEGS': [],
        'AMP_PARLAY_LEGS': [],
        'HYBRID_ENABLED': True,
        'HYBRID_AMP_COUNT': [4, 5],
        'HYBRID_MAX_PARLAYS': 2,
        'HYBRID_WAGER_6LEG': 20,
        'HYBRID_WAGER_7LEG': 15,
    }))

    # Hybrid with aggressive amp filtering
    configs.append(("hybrid_aggressive", {
        **NBA_V8_CONFIG,
        'CORE_PARLAY_LEGS': [],
        'AMP_PARLAY_LEGS': [],
        'AMP_MIN_HR': 0.50,
        'AMP_MIN_EDGE': 0.05,
        'AMP_MIN_BCF_LB': 0.40,
        'AMP_LEG_MAX_ODDS': 600,
        'HYBRID_ENABLED': True,
        'HYBRID_AMP_COUNT': [3, 4, 5],
        'HYBRID_MAX_PARLAYS': 3,
        'HYBRID_WAGER_5LEG': 30,
        'HYBRID_WAGER_6LEG': 20,
        'HYBRID_WAGER_7LEG': 15,
    }))

    # Full portfolio: core + amp + hybrid
    configs.append(("full_portfolio", {
        **NBA_V8_CONFIG,
        'AMP_MIN_HR': 0.50,
        'AMP_MIN_EDGE': 0.06,
        'AMP_MIN_BCF_LB': 0.42,
        'AMP_PARLAY_LEGS': [3, 4],
        'AMP_MAX_PARLAYS': 2,
        'AMP_LEG_MAX_ODDS': 500,
        'HYBRID_ENABLED': True,
        'HYBRID_AMP_COUNT': [3, 4],
        'HYBRID_MAX_PARLAYS': 2,
    }))

    best_roi = -999
    best_name = ""
    best_cfg = None

    for name, cfg in configs:
        r = run_backtest(box_scores, odds_data, cfg, verbose=False)
        if r['total_picks'] > 0:
            print_results(name, r)
            if r['total_roi'] > best_roi and r['total_picks'] >= 5:
                best_roi = r['total_roi']
                best_name = name
                best_cfg = cfg

    # Deep dive on best
    if best_cfg:
        print(f"\n{'='*70}")
        print(f"BEST CONFIG: {best_name} (ROI: {best_roi*100:+.1f}%)")
        print(f"{'='*70}")
        r = run_backtest(box_scores, odds_data, best_cfg, verbose=True)

        # Temporal validation
        if r['picks']:
            all_dates = sorted(set(p['date'] for p in r['picks']))
            mid = len(all_dates) // 2
            h1_dates = set(all_dates[:mid])
            h2_dates = set(all_dates[mid:])
            h1 = [p for p in r['picks'] if p['date'] in h1_dates]
            h2 = [p for p in r['picks'] if p['date'] in h2_dates]
            for label, half in [("H1", h1), ("H2", h2)]:
                if half:
                    h_acc = sum(1 for p in half if p['hit']) / len(half) * 100
                    h_pnl = sum(p['pnl'] for p in half)
                    h_wag = sum(p['wager'] for p in half)
                    h_roi = h_pnl / h_wag * 100 if h_wag > 0 else 0
                    print(f"  {label}: {len(half)} picks, {h_acc:.1f}% acc, "
                          f"${h_pnl:+d}, {h_roi:+.1f}% ROI")


if __name__ == '__main__':
    run_all()
