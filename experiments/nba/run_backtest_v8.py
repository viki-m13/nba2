"""
NBA V8 Backtest Runner — Multi-Gate Certainty Cascade
======================================================
Tests multiple parlay configurations and validates against overfitting.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_loader import load_nba_boxscores, load_nba_odds
from nba.strategy_v8_mgcc import run_backtest, NBA_V8_CONFIG


def run_all():
    print("Loading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records\n")

    configs = [
        # --- By stat category ---
        ("All stats, 2+5+8 legs", {
            **NBA_V8_CONFIG,
            'PARLAY_LEGS': [2, 5, 8],
            'STATS_ALLOWED': ['pts', 'reb', 'ast', 'pra'],
        }),
        ("PTS only, 2+5+8 legs", {
            **NBA_V8_CONFIG,
            'PARLAY_LEGS': [2, 5, 8],
            'STATS_ALLOWED': ['pts'],
        }),
        ("PTS+AST, 2+5+8 legs", {
            **NBA_V8_CONFIG,
            'PARLAY_LEGS': [2, 5, 8],
            'STATS_ALLOWED': ['pts', 'ast'],
        }),

        # --- Single parlay sizes ---
        ("All stats, 2L only", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [2],
            'STATS_ALLOWED': ['pts', 'reb', 'ast', 'pra'],
        }),
        ("All stats, 5L only", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [5],
            'STATS_ALLOWED': ['pts', 'reb', 'ast', 'pra'],
        }),
        ("All stats, 8L only", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [8],
            'STATS_ALLOWED': ['pts', 'reb', 'ast', 'pra'],
        }),
        ("All stats, 10L only", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [10],
            'STATS_ALLOWED': ['pts', 'reb', 'ast', 'pra'],
        }),
        ("PTS, 5L only", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [5],
            'STATS_ALLOWED': ['pts'],
        }),
        ("PTS, 8L only", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [8],
            'STATS_ALLOWED': ['pts'],
        }),
        ("PTS, 10L only", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [10],
            'STATS_ALLOWED': ['pts'],
        }),
        ("PTS+AST, 8L only", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [8],
            'STATS_ALLOWED': ['pts', 'ast'],
        }),
        ("PTS+AST, 10L only", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [10],
            'STATS_ALLOWED': ['pts', 'ast'],
        }),

        # --- Different top-N pools ---
        ("All stats, 8L, top5", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [8], 'CRS_TOP_N': 5,
            'STATS_ALLOWED': ['pts', 'reb', 'ast', 'pra'],
        }),
        ("All stats, 8L, top15", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [8], 'CRS_TOP_N': 15,
            'STATS_ALLOWED': ['pts', 'reb', 'ast', 'pra'],
        }),
        ("PTS, 5L, top5", {
            **NBA_V8_CONFIG, 'PARLAY_LEGS': [5], 'CRS_TOP_N': 5,
            'STATS_ALLOWED': ['pts'],
        }),
    ]

    print(f"{'Config':<35} | {'Prlys':>5} | {'Hit':>4} | {'Acc%':>6} | {'Days':>4} | {'Day%':>5} | "
          f"{'P&L':>7} | {'ROI%':>6} | {'Legs':>5} | {'LAcc%':>6}")
    print("-" * 110)

    all_results = {}
    for name, cfg in configs:
        r = run_backtest(box_scores, odds_data, cfg, verbose=False)
        all_results[name] = r

        for n_legs, stats in sorted(r.get('by_legs', {}).items()):
            label = f"{name} ({n_legs}L)"
            acc = stats['accuracy'] * 100
            roi = stats['roi'] * 100
            lacc = stats['leg_accuracy'] * 100
            days = len(set(p['date'] for p in r['picks'] if p.get('n_legs') == n_legs))
            total_days = r['total_days']
            day_pct = days / total_days * 100 if total_days else 0
            print(f"{label:<35} | {stats['total']:>5} | {stats['wins']:>4} | {acc:>6.1f} | "
                  f"{days:>4} | {day_pct:>5.1f} | ${stats['pnl']:>+6d} | {roi:>6.1f} | "
                  f"{stats['leg_total']:>5} | {lacc:>6.1f}")

    # ============================================================
    # TEMPORAL VALIDATION — Split into H1 and H2
    # ============================================================
    print(f"\n{'='*70}")
    print("TEMPORAL VALIDATION (H1 vs H2)")
    print(f"{'='*70}")

    best_cfg = {
        **NBA_V8_CONFIG, 'PARLAY_LEGS': [8],
        'STATS_ALLOWED': ['pts', 'ast'],
    }
    r_full = run_backtest(box_scores, odds_data, best_cfg, verbose=False)

    if r_full['picks']:
        all_dates = sorted(set(p['date'] for p in r_full['picks']))
        mid = len(all_dates) // 2
        h1_dates = set(all_dates[:mid])
        h2_dates = set(all_dates[mid:])

        h1 = [p for p in r_full['picks'] if p['date'] in h1_dates]
        h2 = [p for p in r_full['picks'] if p['date'] in h2_dates]

        h1_acc = sum(1 for p in h1 if p['hit']) / len(h1) * 100 if h1 else 0
        h2_acc = sum(1 for p in h2 if p['hit']) / len(h2) * 100 if h2 else 0
        h1_legs = sum(len(p.get('legs', [])) for p in h1)
        h2_legs = sum(len(p.get('legs', [])) for p in h2)
        h1_lhits = sum(sum(1 for l in p.get('legs', []) if l.get('hit')) for p in h1)
        h2_lhits = sum(sum(1 for l in p.get('legs', []) if l.get('hit')) for p in h2)

        print(f"  H1: {len(h1)} parlays, {h1_acc:.1f}% parlay acc, "
              f"{h1_lhits}/{h1_legs} legs ({h1_lhits/h1_legs*100:.1f}% leg acc)" if h1_legs else "  H1: no picks")
        print(f"  H2: {len(h2)} parlays, {h2_acc:.1f}% parlay acc, "
              f"{h2_lhits}/{h2_legs} legs ({h2_lhits/h2_legs*100:.1f}% leg acc)" if h2_legs else "  H2: no picks")
        diff = abs(h1_acc - h2_acc)
        print(f"  Gap: {diff:.1f}pp {'✓ STABLE' if diff < 20 else '✗ UNSTABLE'}")

    # ============================================================
    # MONTHLY BREAKDOWN
    # ============================================================
    print(f"\n{'='*70}")
    print("MONTHLY BREAKDOWN (PTS+AST, 8L)")
    print(f"{'='*70}")

    if r_full['picks']:
        by_month = defaultdict(list)
        for p in r_full['picks']:
            month = p['date'][:6]
            by_month[month].append(p)

        print(f"  {'Month':<10} {'Parlays':>7} {'Hit':>5} {'Acc%':>6} {'Legs':>5} {'LAcc%':>6} {'P&L':>8}")
        print(f"  {'-'*55}")
        for month in sorted(by_month.keys()):
            picks = by_month[month]
            hits = sum(1 for p in picks if p['hit'])
            acc = hits / len(picks) * 100 if picks else 0
            legs = sum(len(p.get('legs', [])) for p in picks)
            lhits = sum(sum(1 for l in p.get('legs', []) if l.get('hit')) for p in picks)
            lacc = lhits / legs * 100 if legs else 0
            pnl = sum(p['pnl'] for p in picks)
            print(f"  {month:<10} {len(picks):>7} {hits:>5} {acc:>6.1f} {legs:>5} {lacc:>6.1f} ${pnl:>+7d}")

    # ============================================================
    # VERBOSE: Show individual parlays
    # ============================================================
    print(f"\n{'='*70}")
    print("ALL PARLAYS: PTS+AST 8L")
    print(f"{'='*70}")
    r = run_backtest(box_scores, odds_data, best_cfg, verbose=True)

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'output_v8')
    os.makedirs(output_dir, exist_ok=True)

    save = {k: v for k, v in r.items() if k not in ('picks',)}
    save['picks_detail'] = [{
        'date': p['date'], 'n_legs': p['n_legs'],
        'parlay_american': p['parlay_american'],
        'hit': p['hit'], 'pnl': p['pnl'], 'wager': p['wager'],
        'legs': p.get('legs', []),
    } for p in r['picks']]

    with open(os.path.join(output_dir, 'v8_results.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nResults saved to output_v8/v8_results.json")


if __name__ == '__main__':
    run_all()
