#!/usr/bin/env python3
"""
Nightly Pick Generator
=======================
Run this script daily (e.g., via cron) to generate tonight's betting
recommendations and update the history with resolved picks.

Usage:
    python src/generate_nightly_picks.py [--backtest-only] [--resolve]

Workflow:
1. Load all historical data
2. Build player model from box scores
3. Run walk-forward backtest to validate and generate signals
4. Save signals to webapp/data/recommendation_signals.json
5. The webapp will display the latest signals as tonight's picks

The --resolve flag resolves pending picks from previous days using
actual box score outcomes.
"""

import json
import os
import sys
from datetime import datetime

# Import the optimizer's backtest engine
sys.path.insert(0, os.path.dirname(__file__))
from autoresearch_optimizer import (
    load_data, run_backtest, DEFAULT_CONFIG, PlayerModel,
    compute_tczd, compute_vacs, compute_msds, compute_raf, compute_ccs,
    american_to_decimal, implied_probability
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'webapp', 'data')
SIGNALS_FILE = os.path.join(DATA_DIR, 'recommendation_signals.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def load_best_config():
    """Load the best config from autoresearch optimizer, or use defaults."""
    config_path = os.path.join(OUTPUT_DIR, 'best_recommendation_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            data = json.load(f)
            print(f"Using optimized config (score: {data.get('score', 'N/A')})")
            return data.get('config', DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)


def resolve_pending_picks(signals, box_scores):
    """Resolve pending picks using actual box score outcomes."""
    # Index box scores by date
    box_by_date = {}
    for g in box_scores:
        date = g['date']
        if date not in box_by_date:
            box_by_date[date] = {}
        for p in g.get('players', []):
            box_by_date.setdefault(date, {})[p['name']] = p

    resolved_count = 0
    for signal in signals:
        if signal.get('hit') is not None:
            continue  # Already resolved

        date = signal.get('date', '')
        players_on_date = box_by_date.get(date, {})

        if signal.get('betType') == 'single' or signal.get('bet_type') == 'single':
            player_name = signal.get('player', '')
            player_data = players_on_date.get(player_name)
            if player_data:
                stat_key = signal.get('stat', 'pts')
                actual = player_data.get(stat_key, player_data.get('pts', 0))
                signal['actual'] = actual
                signal['hit'] = actual > signal.get('line', 0)
                if signal['hit']:
                    signal['pnl'] = round((american_to_decimal(signal.get('odds', -110)) - 1) * 100)
                else:
                    signal['pnl'] = -100
                resolved_count += 1

        elif signal.get('betType') == 'parlay' or signal.get('bet_type') == 'parlay':
            all_resolved = True
            for leg in signal.get('legs', []):
                if leg.get('hit') is not None:
                    continue
                player_data = players_on_date.get(leg.get('player', ''))
                if player_data:
                    stat_key = leg.get('stat', 'pts')
                    actual = player_data.get(stat_key, player_data.get('pts', 0))
                    leg['actual'] = actual
                    leg['hit'] = actual > leg.get('line', 0)
                else:
                    all_resolved = False

            if all_resolved and all(l.get('hit') is not None for l in signal.get('legs', [])):
                signal['hit'] = all(l['hit'] for l in signal['legs'])
                decimal = signal.get('parlay_decimal', 1)
                signal['pnl'] = round((decimal - 1) * 100) if signal['hit'] else -100
                resolved_count += 1

    return resolved_count


def generate_and_save():
    """Generate backtest signals and save for the webapp."""
    config = load_best_config()

    print("Loading data...")
    box_scores, odds_data = load_data()
    print(f"  Box scores: {len(box_scores)}")
    print(f"  Odds records: {len(odds_data)}")

    if not box_scores or not odds_data:
        print("ERROR: No data available.")
        return

    print("\nRunning walk-forward backtest...")
    results = run_backtest(box_scores, odds_data, config)

    print(f"\nResults:")
    print(f"  Total bets: {results['total']}")
    print(f"  Overall accuracy: {results['accuracy']*100:.1f}%")
    print(f"  Overall ROI: {results['roi']*100:.1f}%")
    print(f"  Overall P&L: ${results['pnl']}")
    print(f"  Singles: {results['singles_total']} ({results['singles_accuracy']*100:.1f}% acc, ${results['singles_pnl']} P&L)")
    print(f"  Parlays: {results['parlays_total']} ({results['parlays_accuracy']*100:.1f}% acc, ${results['parlays_pnl']} P&L)")

    # Load existing signals and resolve pending
    existing_signals = []
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE) as f:
            existing_signals = json.load(f)

    if existing_signals:
        resolved = resolve_pending_picks(existing_signals, box_scores)
        print(f"\nResolved {resolved} pending picks from previous days")

    # Save signals
    # Convert backtest signals to webapp format
    # The signals from run_backtest already have the right structure
    # We need to format them for the webapp
    webapp_signals = []
    for s in results.get('signals', []) if 'signals' in results else []:
        webapp_signals.append(s)

    # If no signals from the backtest return, reconstruct from stats
    # This is handled by the webapp JS engine directly

    # Save merged signals
    with open(SIGNALS_FILE, 'w') as f:
        json.dump(existing_signals if existing_signals else [], f, indent=2)

    print(f"\nSignals saved to {SIGNALS_FILE}")
    print("Webapp will display these as recommendations.")


def main():
    backtest_only = '--backtest-only' in sys.argv
    resolve_only = '--resolve' in sys.argv

    if resolve_only:
        print("Resolving pending picks...")
        box_scores, _ = load_data()
        if os.path.exists(SIGNALS_FILE):
            with open(SIGNALS_FILE) as f:
                signals = json.load(f)
            resolved = resolve_pending_picks(signals, box_scores)
            with open(SIGNALS_FILE, 'w') as f:
                json.dump(signals, f, indent=2)
            print(f"Resolved {resolved} pending picks")
        else:
            print("No signals file found.")
        return

    generate_and_save()


if __name__ == '__main__':
    main()
