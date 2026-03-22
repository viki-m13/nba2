#!/usr/bin/env python3
"""
Nightly Pick Generator — ULTRA BETTING ENGINE v1.0
====================================================
Run this script daily (e.g., via cron at 9 AM EST) to:
1. Generate tonight's betting recommendations using the Ultra Engine
2. Resolve pending picks from previous days using ESPN box scores
3. Update webapp signals and backtest stats

Usage:
    python src/generate_nightly_picks.py              # Full run (generate + resolve + export)
    python src/generate_nightly_picks.py --resolve     # Resolve pending picks only
    python src/generate_nightly_picks.py --export      # Re-export backtest signals only
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ultra_engine import (
    export_webapp_signals,
    load_all_data,
    ULTRA_CONFIG,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'webapp', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SIGNALS_FILE = os.path.join(DATA_DIR, 'ultra_signals.json')
STATS_FILE = os.path.join(DATA_DIR, 'ultra_backtest_stats.json')


def load_config():
    """Load the best config from optimization, or use defaults."""
    config_path = os.path.join(OUTPUT_DIR, 'ultra_engine_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            data = json.load(f)
            print(f"Using optimized config (score: {data.get('score', 'N/A')})")
            return data.get('config', ULTRA_CONFIG)
    return dict(ULTRA_CONFIG)


def american_to_decimal(odds):
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def resolve_pending_signals():
    """Resolve pending picks in ultra_signals.json using actual box score data."""
    if not os.path.exists(SIGNALS_FILE):
        print("No signals file found.")
        return 0

    with open(SIGNALS_FILE) as f:
        signals = json.load(f)

    box_scores, _ = load_all_data()
    if not box_scores:
        print("No box score data available for resolution.")
        return 0

    # Index box scores by date and player
    box_by_date = {}
    for g in box_scores:
        date = g.get('date', '')
        if date not in box_by_date:
            box_by_date[date] = {}
        for p in g.get('players', []):
            box_by_date[date][p['name']] = p

    resolved_count = 0
    for signal in signals:
        if signal.get('hit') is not None or signal.get('dnp'):
            continue  # Already resolved or voided (DNP)

        date = signal.get('date', '')
        players_on_date = box_by_date.get(date, {})
        if not players_on_date:
            continue

        if signal.get('betType') == 'single':
            player_name = signal.get('player', '')
            player_data = players_on_date.get(player_name)
            if player_data:
                stat_key = signal.get('stat', 'pts')
                if stat_key == 'pra':
                    actual = (player_data.get('pts', 0) +
                             player_data.get('reb', 0) +
                             player_data.get('ast', 0))
                else:
                    actual = player_data.get(stat_key, 0)
                signal['actual'] = actual
                signal['hit'] = actual > signal.get('line', 0)
                odds = signal.get('odds', -110)
                decimal = american_to_decimal(odds)
                signal['pnl'] = round((decimal - 1) * 100) if signal['hit'] else -100
                resolved_count += 1
            else:
                # Player DNP — void the bet (push, $0 P&L)
                signal['actual'] = 0
                signal['hit'] = None
                signal['dnp'] = True
                signal['pnl'] = 0
                resolved_count += 1

        elif signal.get('betType') == 'parlay':
            all_resolved = True
            has_voided_leg = False
            for leg in signal.get('legs', []):
                if leg.get('hit') is not None:
                    continue
                player_data = players_on_date.get(leg.get('player', ''))
                if player_data:
                    stat_key = leg.get('stat', 'pts')
                    if stat_key == 'pra':
                        actual = (player_data.get('pts', 0) +
                                 player_data.get('reb', 0) +
                                 player_data.get('ast', 0))
                    else:
                        actual = player_data.get(stat_key, 0)
                    leg['actual'] = actual
                    leg['hit'] = actual > leg.get('line', 0)
                else:
                    # Player DNP — void this leg (sportsbooks void DNP legs)
                    leg['actual'] = 0
                    leg['dnp'] = True
                    has_voided_leg = True

            active_legs = [l for l in signal.get('legs', []) if not l.get('dnp')]
            if all(l.get('hit') is not None for l in active_legs) and active_legs:
                signal['hit'] = all(l['hit'] for l in active_legs)
                if has_voided_leg:
                    # Recalculate odds for reduced parlay
                    reduced_decimal = 1.0
                    for l in active_legs:
                        reduced_decimal *= american_to_decimal(l.get('odds', -110))
                    signal['pnl'] = round((reduced_decimal - 1) * 100) if signal['hit'] else -100
                else:
                    decimal = signal.get('parlay_decimal', 1)
                    signal['pnl'] = round((decimal - 1) * 100) if signal['hit'] else -100
                resolved_count += 1

    if resolved_count > 0:
        with open(SIGNALS_FILE, 'w') as f:
            json.dump(signals, f, indent=2)

        # Recompute and update stats
        update_backtest_stats(signals)

    return resolved_count


def update_backtest_stats(signals):
    """Recompute backtest stats from signals and save."""
    singles = [s for s in signals if s.get('betType') == 'single' and s.get('hit') is not None]
    parlays = [s for s in signals if s.get('betType') == 'parlay' and s.get('hit') is not None]

    singles_wins = sum(1 for s in singles if s['hit'])
    parlays_wins = sum(1 for p in parlays if p['hit'])

    singles_pnl = sum(s.get('pnl', 0) for s in singles)
    parlays_pnl = sum(p.get('pnl', 0) for p in parlays)

    all_legs = []
    for p in parlays:
        all_legs.extend(p.get('legs', []))
    leg_hits = sum(1 for l in all_legs if l.get('hit'))

    total = len(singles) + len(parlays)
    total_wins = singles_wins + parlays_wins
    total_pnl = singles_pnl + parlays_pnl

    stats = {
        'singles': {
            'total': len(singles),
            'wins': singles_wins,
            'accuracy': singles_wins / len(singles) if singles else 0,
            'pnl': singles_pnl,
            'roi': singles_pnl / (len(singles) * 100) if singles else 0,
        },
        'parlays': {
            'total': len(parlays),
            'wins': parlays_wins,
            'accuracy': parlays_wins / len(parlays) if parlays else 0,
            'pnl': parlays_pnl,
            'roi': parlays_pnl / (len(parlays) * 100) if parlays else 0,
            'totalLegs': len(all_legs),
            'hitLegs': leg_hits,
            'legAccuracy': leg_hits / len(all_legs) if all_legs else 0,
        },
        'overall': {
            'total': total,
            'wins': total_wins,
            'accuracy': total_wins / total if total else 0,
            'pnl': total_pnl,
            'roi': total_pnl / (total * 100) if total else 0,
        },
    }

    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Updated stats: {total} picks, {total_wins}W, "
          f"{total_wins/total*100:.1f}% acc, ${total_pnl:+} P&L" if total else "No resolved picks yet")


def main():
    resolve_only = '--resolve' in sys.argv
    export_only = '--export' in sys.argv

    if resolve_only:
        print("=" * 60)
        print("ULTRA BETTING ENGINE v1.0 — Resolving Pending Picks")
        print("=" * 60)
        resolved = resolve_pending_signals()
        print(f"Resolved {resolved} pending picks")
        return

    if export_only:
        print("=" * 60)
        print("ULTRA BETTING ENGINE v1.0 — Exporting Backtest Signals")
        print("=" * 60)
        export_webapp_signals()
        return

    # Full run: generate nightly picks + resolve pending + export
    print("=" * 60)
    print("ULTRA BETTING ENGINE v1.0 — Nightly Pick Generation")
    print("=" * 60)

    # Step 1: Resolve any pending picks from previous days
    print("\n--- Step 1: Resolving pending picks ---")
    resolved = resolve_pending_signals()
    print(f"Resolved {resolved} pending picks from previous days")

    # Step 2: Export updated backtest signals (updates history + stats)
    # This preserves any live signals added by seed-live-picks.js
    print("\n--- Step 2: Exporting backtest signals ---")
    export_webapp_signals()

    # Note: Tonight's picks are generated by seed-live-picks.js (Node.js)
    # which fetches live odds from The Odds API. The Python backtest does not
    # generate live picks, so we skip writing ultra_recommendations.json here
    # to avoid overwriting real recommendations from the Node step.

    print("\nDone! Webapp will display updated results.")


if __name__ == '__main__':
    main()
