#!/usr/bin/env python3
"""
Positive Odds Models — Generate Picks & History
================================================
Generates all historical backtest picks and today's signals for both
NBA and MLB positive-odds models. Saves in webapp-compatible format.

All contained within experiments/ — does NOT touch main webapp or cron.

Usage:
    python experiments/scripts/generate_picks.py           # Generate all
    python experiments/scripts/generate_picks.py --nba     # NBA only
    python experiments/scripts/generate_picks.py --mlb     # MLB only
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Add experiments root to path
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXPERIMENTS_DIR)

from shared.data_loader import load_nba_boxscores, load_nba_odds, load_mlb_boxscores, load_mlb_odds

WEBAPP_DATA = os.path.join(EXPERIMENTS_DIR, '..', 'webapp', 'positive-odds', 'data')


def generate_nba_picks():
    """Generate NBA positive-odds picks using champion config."""
    print("\n=== NBA POSITIVE ODDS MODEL ===")

    # Import V1 strategy (the champion)
    from nba.strategy import run_backtest

    # Load champion config
    config_path = os.path.join(EXPERIMENTS_DIR, 'nba', 'output_v3', 'plus_money_accuracy_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)['config']
        print(f"  Loaded optimized config from {config_path}")
    else:
        # Fallback to default champion
        champion_path = os.path.join(EXPERIMENTS_DIR, 'nba', 'output_v3', 'champion_config.json')
        with open(champion_path) as f:
            config = json.load(f)['config']
        print(f"  Loaded champion config from {champion_path}")

    # Load data
    print("  Loading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    # Run backtest
    print("  Running walk-forward backtest...")
    results = run_backtest(box_scores, odds_data, config, verbose=False)

    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    print(f"  Results: {results['total_picks']} picks, {results['total_wins']} wins, "
          f"{acc*100:.1f}% accuracy, {results['total_roi']*100:.1f}% ROI")

    # Convert to webapp signal format (tag as backtest source)
    backtest_signals = []
    for pick in results.get('picks', []):
        if pick.get('bet_type') == 'single':
            signal = {
                'date': pick.get('date', ''),
                'betType': 'single',
                'player': pick.get('player', ''),
                'team': '',
                'opponent': '',
                'line': pick.get('line', 0),
                'odds': pick.get('odds', 0),
                'stat': pick.get('stat', ''),
                'statLabel': pick.get('stat', '').upper(),
                'cascadeScore': pick.get('combined_score', 0),
                'hitRate': pick.get('hit_rate', 0),
                'edge': pick.get('edge', 0),
                'ev': pick.get('ev', 0),
                'actual': pick.get('actual'),
                'hit': pick.get('hit'),
                'pnl': pick.get('pnl', 0),
                'wager': pick.get('wager', 100),
                'bet': f"{pick.get('player', '')} O{pick.get('line', '')} {pick.get('stat', '').upper()}",
                'engine': 'positive_odds_nba',
                'betSubType': 'single',
                'source': 'backtest',
            }
            backtest_signals.append(signal)
        elif pick.get('bet_type') in ('parlay', 'parlay_plb'):
            legs = []
            for leg in pick.get('legs', []):
                legs.append({
                    'player': leg.get('player', ''),
                    'team': '',
                    'line': leg.get('line', 0),
                    'odds': leg.get('odds', 0),
                    'stat': leg.get('stat', ''),
                    'statLabel': leg.get('stat', '').upper(),
                    'cascadeScore': leg.get('combined_score', 0),
                    'hit': leg.get('hit'),
                    'actual': leg.get('actual'),
                    'edge': leg.get('edge', 0),
                })
            signal = {
                'date': pick.get('date', ''),
                'betType': 'parlay',
                'n_legs': pick.get('n_legs', len(legs)),
                'legs': legs,
                'odds': int(pick.get('parlay_decimal', 1) * 100 - 100) if pick.get('parlay_decimal', 1) >= 2 else int(-100 / (pick.get('parlay_decimal', 2) - 1)),
                'hit': pick.get('hit'),
                'pnl': pick.get('pnl', 0),
                'wager': pick.get('wager', 100),
                'bet': f"{pick.get('n_legs', 2)}-Leg Parlay",
                'engine': 'positive_odds_nba',
                'source': 'backtest',
            }
            backtest_signals.append(signal)

    # Merge with existing live signals (preserve live picks, replace backtest)
    os.makedirs(WEBAPP_DATA, exist_ok=True)
    existing_signals = []
    signals_path = os.path.join(WEBAPP_DATA, 'nba_signals.json')
    try:
        with open(signals_path) as f:
            existing_signals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Keep live signals, remove old backtest signals
    live_signals = [s for s in existing_signals if s.get('source') == 'live']
    live_dates = set(s['date'] for s in live_signals)

    # Don't include backtest signals for dates that have live picks
    merged = [s for s in backtest_signals if s['date'] not in live_dates]
    merged.extend(live_signals)

    # Compute stats from all signals (both live and backtest)
    all_signals = merged
    singles = [s for s in all_signals if s.get('betType') == 'single' and s.get('hit') is not None]
    parlays = [s for s in all_signals if s.get('betType') == 'parlay' and s.get('hit') is not None]
    s_wins = sum(1 for s in singles if s['hit'])
    p_wins = sum(1 for p in parlays if p['hit'])
    total = len(singles) + len(parlays)
    total_wins = s_wins + p_wins
    total_pnl = sum(s.get('pnl', 0) for s in singles) + sum(p.get('pnl', 0) for p in parlays)
    total_wagered = sum(s.get('wager', 100) for s in singles) + sum(p.get('wager', 100) for p in parlays)

    stats = {
        'singles': {
            'total': len(singles),
            'wins': s_wins,
            'accuracy': s_wins / len(singles) if singles else 0,
            'pnl': sum(s.get('pnl', 0) for s in singles),
        },
        'parlays': {
            'total': len(parlays),
            'wins': p_wins,
            'accuracy': p_wins / len(parlays) if parlays else 0,
            'pnl': sum(p.get('pnl', 0) for p in parlays),
        },
        'overall': {
            'total': total,
            'wins': total_wins,
            'accuracy': total_wins / total if total > 0 else 0,
            'pnl': total_pnl,
            'roi': total_pnl / total_wagered if total_wagered > 0 else 0,
        },
        'model': 'NBA Positive Odds v1 (HECE Champion)',
        'config_version': 'optimized',
        'min_odds': '+100',
        'generated': datetime.now().isoformat(),
    }

    # Save signals and stats (recommendations left for seed_live_picks.py)
    with open(signals_path, 'w') as f:
        json.dump(merged, f, indent=2)
    with open(os.path.join(WEBAPP_DATA, 'nba_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved {len(merged)} signals ({len(live_signals)} live + {len(merged) - len(live_signals)} backtest) to {WEBAPP_DATA}/")
    return results


def generate_mlb_picks():
    """Generate MLB positive-odds picks using champion config."""
    print("\n=== MLB POSITIVE ODDS MODEL ===")

    from mlb.strategy import run_backtest

    # Load champion config
    config_path = os.path.join(EXPERIMENTS_DIR, 'mlb', 'output_v2', 'mlb_hece_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)['config']
        print(f"  Loaded optimized config from {config_path}")
    else:
        from mlb.run_backtest_v2 import MLB_PM_CONFIG
        config = dict(MLB_PM_CONFIG)
        print("  Using default MLB plus-money config")

    # Load data
    print("  Loading MLB data...")
    try:
        box_scores = load_mlb_boxscores()
        odds_data = load_mlb_odds()
        print(f"  {len(box_scores)} games, {len(odds_data)} odds records")
    except FileNotFoundError:
        print("  MLB data not found, skipping")
        return None

    # Run backtest
    print("  Running walk-forward backtest...")
    results = run_backtest(box_scores, odds_data, config, verbose=False)

    acc = results.get('accuracy', results.get('individual_accuracy', 0))
    print(f"  Results: {results['total_picks']} picks, {results['total_wins']} wins, "
          f"{acc*100:.1f}% accuracy, {results['total_roi']*100:.1f}% ROI")

    # Convert to webapp signal format (tag as backtest source)
    backtest_signals = []
    for pick in results.get('picks', []):
        if pick.get('bet_type') == 'single':
            signal = {
                'date': pick.get('date', ''),
                'betType': 'single',
                'player': pick.get('player', ''),
                'team': '',
                'opponent': '',
                'line': pick.get('line', 0),
                'odds': pick.get('odds', 0),
                'stat': pick.get('stat', ''),
                'statLabel': pick.get('stat', '').upper(),
                'cascadeScore': pick.get('combined_score', 0),
                'hitRate': pick.get('hit_rate', 0),
                'edge': pick.get('edge', 0),
                'ev': pick.get('ev', 0),
                'actual': pick.get('actual'),
                'hit': pick.get('hit'),
                'pnl': pick.get('pnl', 0),
                'wager': pick.get('wager', 100),
                'bet': f"{pick.get('player', '')} O{pick.get('line', '')} {pick.get('stat', '').upper()}",
                'engine': 'positive_odds_mlb',
                'betSubType': 'single',
                'source': 'backtest',
            }
            backtest_signals.append(signal)

    # Merge with existing live signals (preserve live picks, replace backtest)
    os.makedirs(WEBAPP_DATA, exist_ok=True)
    existing_signals = []
    signals_path = os.path.join(WEBAPP_DATA, 'mlb_signals.json')
    try:
        with open(signals_path) as f:
            existing_signals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    live_signals = [s for s in existing_signals if s.get('source') == 'live']
    live_dates = set(s['date'] for s in live_signals)
    merged = [s for s in backtest_signals if s['date'] not in live_dates]
    merged.extend(live_signals)

    # Compute stats from all signals
    all_signals = merged
    singles = [s for s in all_signals if s.get('betType') == 'single' and s.get('hit') is not None]
    parlays = [s for s in all_signals if s.get('betType') == 'parlay' and s.get('hit') is not None]
    s_wins = sum(1 for s in singles if s['hit'])
    p_wins = sum(1 for p in parlays if p['hit'])
    total = len(singles) + len(parlays)
    total_wins = s_wins + p_wins
    total_pnl = sum(s.get('pnl', 0) for s in singles) + sum(p.get('pnl', 0) for p in parlays)
    total_wagered = sum(s.get('wager', 100) for s in singles) + sum(p.get('wager', 100) for p in parlays)

    stats = {
        'singles': {
            'total': len(singles),
            'wins': s_wins,
            'accuracy': s_wins / len(singles) if singles else 0,
            'pnl': sum(s.get('pnl', 0) for s in singles),
        },
        'overall': {
            'total': total,
            'wins': total_wins,
            'accuracy': total_wins / total if total > 0 else 0,
            'pnl': total_pnl,
            'roi': total_pnl / total_wagered if total_wagered > 0 else 0,
        },
        'model': 'MLB Positive Odds v1 (HECE Champion)',
        'config_version': 'optimized',
        'min_odds': '+100',
        'generated': datetime.now().isoformat(),
    }

    # Save signals and stats (recommendations left for seed_live_picks_mlb.py)
    with open(signals_path, 'w') as f:
        json.dump(merged, f, indent=2)
    with open(os.path.join(WEBAPP_DATA, 'mlb_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved {len(merged)} signals ({len(live_signals)} live + {len(merged) - len(live_signals)} backtest) to {WEBAPP_DATA}/")
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate positive odds picks')
    parser.add_argument('--nba', action='store_true', help='NBA only')
    parser.add_argument('--mlb', action='store_true', help='MLB only')
    args = parser.parse_args()

    print("=" * 60)
    print("POSITIVE ODDS MODELS — Pick Generator")
    print("=" * 60)

    do_all = not args.nba and not args.mlb

    if do_all or args.nba:
        generate_nba_picks()

    if do_all or args.mlb:
        generate_mlb_picks()

    print("\n" + "=" * 60)
    print("DONE — Backtest history saved to webapp/positive-odds/data/")
    print("=" * 60)


if __name__ == '__main__':
    main()
