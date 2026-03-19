#!/usr/bin/env python3
"""
V8 MGCC — Generate Picks & History (Separate from original & positive-odds)
============================================================================
Generates all historical backtest picks and today's signals for V8 MGCC
multi-leg parlays. Saves to webapp/v8-mgcc/data/.

Usage:
    python experiments/scripts/generate_v8_picks.py
"""

import sys
import os
import json
from datetime import datetime

EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXPERIMENTS_DIR)

from shared.data_loader import load_nba_boxscores, load_nba_odds, load_mlb_boxscores, load_mlb_odds

WEBAPP_DATA = os.path.join(EXPERIMENTS_DIR, '..', 'webapp', 'v8-mgcc', 'data')


def _convert_pick_to_signal(pick, engine_label):
    """Convert a backtest pick to webapp signal format."""
    legs = []
    for leg in pick.get('legs', []):
        legs.append({
            'player': leg.get('player', ''),
            'team': '',
            'line': leg.get('line', 0),
            'odds': leg.get('odds', 0),
            'stat': leg.get('stat', ''),
            'statLabel': leg.get('stat', '').upper(),
            'cascadeScore': leg.get('lower_bound', 0),
            'hit': leg.get('hit'),
            'actual': leg.get('actual'),
            'gates': leg.get('gates', 0),
            'clearance': leg.get('clearance', 0),
            'hr_20': leg.get('hr_20', 0),
        })
    return {
        'date': pick.get('date', ''),
        'betType': 'parlay',
        'n_legs': pick.get('n_legs', len(legs)),
        'legs': legs,
        'odds': pick.get('parlay_american', 0),
        'hit': pick.get('hit'),
        'pnl': pick.get('pnl', 0),
        'wager': pick.get('wager', 100),
        'bet': f"V8 {pick.get('n_legs', 2)}-Leg MGCC [{engine_label}]",
        'engine': f'v8_mgcc_{engine_label}',
        'source': 'backtest',
        'combined_prob': pick.get('combined_prob', 0),
        'parlay_ev': pick.get('parlay_ev', 0),
    }


def generate_nba_v8():
    """Generate NBA V8 MGCC picks."""
    print("\n=== NBA V8 MGCC PARLAYS ===")

    from nba.strategy_v8_mgcc import run_backtest, NBA_V8_CONFIG

    print("  Loading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    # Run champion + experimental configs
    configs = {
        'pts_2L': {
            **NBA_V8_CONFIG,
            'PARLAY_LEGS': [2],
            'STATS_ALLOWED': ['pts'],
            'CRS_TOP_N': 10,
        },
        'pts_5L': {
            **NBA_V8_CONFIG,
            'PARLAY_LEGS': [5],
            'STATS_ALLOWED': ['pts'],
            'CRS_TOP_N': 10,
        },
        'pts_8L': {
            **NBA_V8_CONFIG,
            'PARLAY_LEGS': [8],
            'STATS_ALLOWED': ['pts'],
            'CRS_TOP_N': 10,
        },
        'pts_10L': {
            **NBA_V8_CONFIG,
            'PARLAY_LEGS': [10],
            'STATS_ALLOWED': ['pts'],
            'CRS_TOP_N': 10,
        },
        'ptsast_8L': {
            **NBA_V8_CONFIG,
            'PARLAY_LEGS': [8],
            'STATS_ALLOWED': ['pts', 'ast'],
            'CRS_TOP_N': 10,
        },
        'all_2L': {
            **NBA_V8_CONFIG,
            'PARLAY_LEGS': [2],
            'STATS_ALLOWED': ['pts', 'reb', 'ast', 'pra'],
            'CRS_TOP_N': 10,
        },
    }

    all_signals = []
    all_by_legs = {}

    for label, config in configs.items():
        print(f"\n  Running V8 [{label}]...")
        results = run_backtest(box_scores, odds_data, config, verbose=False)

        for n_legs, stats in results.get('by_legs', {}).items():
            if stats['total'] > 0:
                print(f"    {n_legs}L: {stats['total']} parlays, {stats['wins']} wins, "
                      f"{stats['accuracy']*100:.1f}% acc, {stats['leg_accuracy']*100:.1f}% leg acc, "
                      f"${stats['pnl']:+d}")
                key = f"{label}_{n_legs}L"
                all_by_legs[key] = stats

        for pick in results.get('picks', []):
            all_signals.append(_convert_pick_to_signal(pick, f'nba_{label}'))

    # Merge with existing live signals
    os.makedirs(WEBAPP_DATA, exist_ok=True)
    signals_path = os.path.join(WEBAPP_DATA, 'nba_v8_signals.json')

    existing = []
    try:
        with open(signals_path) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    live_signals = [s for s in existing if s.get('source') == 'live']
    live_dates = set(s['date'] for s in live_signals)
    merged = [s for s in all_signals if s['date'] not in live_dates]
    merged.extend(live_signals)

    with open(signals_path, 'w') as f:
        json.dump(merged, f, indent=2)

    # Compute aggregate stats
    resolved = [s for s in merged if s.get('hit') is not None]
    wins = sum(1 for s in resolved if s['hit'])
    total_pnl = sum(s.get('pnl', 0) for s in resolved)
    total_wager = sum(s.get('wager', 100) for s in resolved)

    # Compute leg-level stats
    all_legs_list = []
    for s in resolved:
        all_legs_list.extend(s.get('legs', []))
    leg_hits = sum(1 for l in all_legs_list if l.get('hit'))

    stats = {
        'parlays': {
            'total': len(resolved),
            'wins': wins,
            'accuracy': wins / len(resolved) if resolved else 0,
            'pnl': total_pnl,
            'roi': total_pnl / total_wager if total_wager > 0 else 0,
            'leg_total': len(all_legs_list),
            'leg_hits': leg_hits,
            'leg_accuracy': leg_hits / len(all_legs_list) if all_legs_list else 0,
        },
        'by_legs': all_by_legs,
        'model': 'NBA V8 MGCC (Multi-Gate Certainty Cascade)',
        'innovation': '7-Gate Hierarchical Evidence Stacking with Confidence-Ranked Selection',
        'generated': datetime.now().isoformat(),
    }
    with open(os.path.join(WEBAPP_DATA, 'nba_v8_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Recommendations (today's picks — empty until live seeder runs)
    today = datetime.now().strftime('%Y%m%d')
    today_live = [s for s in live_signals if s['date'] == today]
    recs = {
        'generated': datetime.now().isoformat(),
        'date': today,
        'engine': 'V8 MGCC (Multi-Gate Certainty Cascade)',
        'picks': today_live,
    }
    with open(os.path.join(WEBAPP_DATA, 'nba_v8_recommendations.json'), 'w') as f:
        json.dump(recs, f, indent=2)

    print(f"\n  Saved {len(merged)} NBA V8 signals to {WEBAPP_DATA}/")
    return merged


def generate_mlb_v8():
    """Generate MLB V8 MGCC picks."""
    print("\n=== MLB V8 MGCC PARLAYS ===")

    from mlb.strategy_v8_mgcc import run_backtest, MLB_V8_CONFIG

    print("  Loading MLB data...")
    try:
        box_scores = load_mlb_boxscores()
        odds_data = load_mlb_odds()
    except FileNotFoundError:
        print("  MLB data not found, skipping")
        # Create empty files
        os.makedirs(WEBAPP_DATA, exist_ok=True)
        for fname in ['mlb_v8_signals.json', 'mlb_v8_stats.json', 'mlb_v8_recommendations.json']:
            path = os.path.join(WEBAPP_DATA, fname)
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump([] if 'signals' in fname else {}, f)
        return []

    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    configs = {
        'all_2L': {**MLB_V8_CONFIG, 'PARLAY_LEGS': [2]},
        'all_3L': {**MLB_V8_CONFIG, 'PARLAY_LEGS': [3]},
    }

    all_signals = []
    all_by_legs = {}

    for label, config in configs.items():
        print(f"\n  Running MLB V8 [{label}]...")
        results = run_backtest(box_scores, odds_data, config, verbose=False)

        for n_legs, stats in results.get('by_legs', {}).items():
            if stats['total'] > 0:
                print(f"    {n_legs}L: {stats['total']} parlays, {stats['wins']} wins, "
                      f"{stats['accuracy']*100:.1f}% acc, ${stats['pnl']:+d}")
                all_by_legs[f"{label}_{n_legs}L"] = stats

        for pick in results.get('picks', []):
            all_signals.append(_convert_pick_to_signal(pick, f'mlb_{label}'))

    os.makedirs(WEBAPP_DATA, exist_ok=True)

    with open(os.path.join(WEBAPP_DATA, 'mlb_v8_signals.json'), 'w') as f:
        json.dump(all_signals, f, indent=2)

    resolved = [s for s in all_signals if s.get('hit') is not None]
    wins = sum(1 for s in resolved if s['hit'])
    total_pnl = sum(s.get('pnl', 0) for s in resolved)
    total_wager = sum(s.get('wager', 100) for s in resolved)
    all_legs_list = []
    for s in resolved:
        all_legs_list.extend(s.get('legs', []))
    leg_hits = sum(1 for l in all_legs_list if l.get('hit'))

    stats = {
        'parlays': {
            'total': len(resolved),
            'wins': wins,
            'accuracy': wins / len(resolved) if resolved else 0,
            'pnl': total_pnl,
            'roi': total_pnl / total_wager if total_wager > 0 else 0,
            'leg_total': len(all_legs_list),
            'leg_hits': leg_hits,
            'leg_accuracy': leg_hits / len(all_legs_list) if all_legs_list else 0,
        },
        'by_legs': all_by_legs,
        'model': 'MLB V8 MGCC',
        'generated': datetime.now().isoformat(),
    }
    with open(os.path.join(WEBAPP_DATA, 'mlb_v8_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    today = datetime.now().strftime('%Y%m%d')
    recs = {
        'generated': datetime.now().isoformat(),
        'date': today,
        'engine': 'MLB V8 MGCC',
        'picks': [],
    }
    with open(os.path.join(WEBAPP_DATA, 'mlb_v8_recommendations.json'), 'w') as f:
        json.dump(recs, f, indent=2)

    print(f"\n  Saved {len(all_signals)} MLB V8 signals to {WEBAPP_DATA}/")
    return all_signals


def main():
    print("=" * 60)
    print("V8 MGCC — Pick Generator (Separate Pipeline)")
    print("=" * 60)

    generate_nba_v8()
    generate_mlb_v8()

    print("\n" + "=" * 60)
    print("DONE — V8 MGCC data saved to webapp/v8-mgcc/data/")
    print("=" * 60)


if __name__ == '__main__':
    main()
