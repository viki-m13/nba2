#!/usr/bin/env python3
"""
V8 CSPE — Generate Picks & History (Separate from original & positive-odds)
============================================================================
Generates all historical backtest picks and today's signals for V8 CSPE
dual-tier parlays. Saves to webapp/v8-mgcc/data/.

Uses the Convergence Score Parlay Engine (CSPE) with:
- Core tier: 2L heavy-favorite parlays for consistency
- Amplifier tier: 4-5L edge-ranked parlays for ROI
- Hybrid tier: Anchor-Amplifier fusion parlays

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
            'cascadeScore': leg.get('score', leg.get('bcf_lb', 0)),
            'hit': leg.get('hit'),
            'actual': leg.get('actual'),
            'gates': 0,  # CSPE uses continuous scoring, not gates
            'clearance': leg.get('clearance', 0),
            'hr_20': leg.get('hr_20', 0),
            'edge': leg.get('edge', 0),
            'tier': leg.get('tier', ''),
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
        'bet': f"V8 {pick.get('tier', 'amp').upper()} {pick.get('n_legs', 2)}-Leg [{engine_label}]",
        'engine': f'v8_cspe_{engine_label}',
        'source': 'backtest',
        'combined_prob': pick.get('combined_prob', 0),
        'parlay_ev': pick.get('parlay_ev', 0),
        'tier': pick.get('tier', ''),
    }


def generate_nba_v8():
    """Generate NBA V8 CSPE picks."""
    print("\n=== NBA V8 CSPE PARLAYS ===")

    from nba.strategy_v8_cspe import run_backtest, NBA_V8_CONFIG

    print("  Loading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    # Run the main config
    print(f"\n  Running V8 CSPE...")
    results = run_backtest(box_scores, odds_data, NBA_V8_CONFIG, verbose=False)

    all_signals = []
    all_by_legs = {}

    for key, stats in results.get('by_legs', {}).items():
        if stats['total'] > 0:
            print(f"    {key}: {stats['total']} parlays, {stats['wins']} wins, "
                  f"{stats['accuracy']*100:.1f}% acc, "
                  f"${stats['pnl']:+d}, {stats['roi']*100:+.1f}% ROI")
            all_by_legs[key] = stats

    for tier, stats in results.get('by_tier', {}).items():
        if stats['total'] > 0:
            print(f"  >> {tier.upper()}: {stats['total']}p, {stats['wins']}W, "
                  f"${stats['pnl']:+d}, {stats['roi']*100:+.1f}% ROI")

    print(f"  >> TOTAL: {results['total_picks']}p, ${results['total_pnl']:+d}, "
          f"{results['total_roi']*100:+.1f}% ROI, {results['active_days']} active days")

    for pick in results.get('picks', []):
        all_signals.append(_convert_pick_to_signal(pick, 'nba'))

    # Save
    os.makedirs(WEBAPP_DATA, exist_ok=True)
    signals_path = os.path.join(WEBAPP_DATA, 'nba_v8_signals.json')

    # Merge with existing live signals
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

    # Stats
    resolved = [s for s in merged if s.get('hit') is not None]
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
        'by_tier': results.get('by_tier', {}),
        'model': 'NBA V8 CSPE (Convergence Score Parlay Engine)',
        'innovation': 'Edge-centric dual-tier scoring with anchor-amplifier fusion',
        'generated': datetime.now().isoformat(),
    }
    with open(os.path.join(WEBAPP_DATA, 'nba_v8_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Recommendations
    today = datetime.now().strftime('%Y%m%d')
    today_live = [s for s in live_signals if s['date'] == today]
    recs = {
        'generated': datetime.now().isoformat(),
        'date': today,
        'engine': 'V8 CSPE (Convergence Score Parlay Engine)',
        'picks': today_live,
    }
    with open(os.path.join(WEBAPP_DATA, 'nba_v8_recommendations.json'), 'w') as f:
        json.dump(recs, f, indent=2)

    print(f"\n  Saved {len(merged)} NBA V8 signals to {WEBAPP_DATA}/")
    return merged


def generate_mlb_v8():
    """Generate MLB V8 CSPE picks using the proven V3 Poisson-Bayesian engine.

    The MLB V8 model uses the V3 engine's 16-signal gate cascade with
    Poisson-Bayesian Fusion for discrete MLB stats. This is run via Node.js
    since the engine is implemented in JavaScript.
    """
    print("\n=== MLB V8 CSPE (Poisson-Bayesian) ===")

    import subprocess

    script_path = os.path.join(EXPERIMENTS_DIR, 'scripts', 'generate_mlb_v8_from_engine.js')
    if not os.path.exists(script_path):
        print("  MLB V8 generator script not found, skipping")
        return []

    try:
        result = subprocess.run(
            ['node', script_path],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.join(EXPERIMENTS_DIR, '..'),
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Failed to run MLB V8 generator: {e}")
        return []

    # Load the generated signals
    signals_path = os.path.join(WEBAPP_DATA, 'mlb_v8_signals.json')
    try:
        with open(signals_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def main():
    print("=" * 60)
    print("V8 CSPE — Convergence Score Parlay Engine")
    print("=" * 60)

    generate_nba_v8()
    generate_mlb_v8()

    print("\n" + "=" * 60)
    print("DONE — V8 CSPE data saved to webapp/v8-mgcc/data/")
    print("=" * 60)


if __name__ == '__main__':
    main()
