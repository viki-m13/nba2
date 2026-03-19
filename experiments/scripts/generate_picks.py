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

    # Save signals and stats
    with open(signals_path, 'w') as f:
        json.dump(merged, f, indent=2)
    with open(os.path.join(WEBAPP_DATA, 'nba_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Write recommendations with today's date
    # If live picks exist for today they take priority; otherwise show empty
    today = datetime.now().strftime('%Y%m%d')
    today_live = [s for s in live_signals if s['date'] == today]
    if today_live:
        # Live picks already exist for today — use them
        recs_picks = today_live
    else:
        # No live picks yet — seed_live_picks.py will override when it runs
        recs_picks = []
    recommendations = {
        'generated': datetime.now().isoformat(),
        'date': today,
        'engine': 'NBA Positive Odds HECE v1',
        'picks': recs_picks,
    }
    with open(os.path.join(WEBAPP_DATA, 'nba_recommendations.json'), 'w') as f:
        json.dump(recommendations, f, indent=2)

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

    # Save signals and stats
    with open(signals_path, 'w') as f:
        json.dump(merged, f, indent=2)
    with open(os.path.join(WEBAPP_DATA, 'mlb_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Write recommendations with today's date
    today = datetime.now().strftime('%Y%m%d')
    today_live = [s for s in live_signals if s['date'] == today]
    if today_live:
        recs_picks = today_live
    else:
        recs_picks = []
    recommendations = {
        'generated': datetime.now().isoformat(),
        'date': today,
        'engine': 'MLB Positive Odds HECE v1',
        'picks': recs_picks,
    }
    with open(os.path.join(WEBAPP_DATA, 'mlb_recommendations.json'), 'w') as f:
        json.dump(recommendations, f, indent=2)

    print(f"  Saved {len(merged)} signals ({len(live_signals)} live + {len(merged) - len(live_signals)} backtest) to {WEBAPP_DATA}/")
    return results


def generate_nba_v8_parlays():
    """Generate NBA V8 MGCC parlay picks (Multi-Gate Certainty Cascade)."""
    print("\n=== NBA V8 MGCC PARLAYS ===")

    from nba.strategy_v8_mgcc import run_backtest, NBA_V8_CONFIG

    print("  Loading NBA data...")
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    # Run configs: Champion PTS 2L + multi-leg experiments
    configs_to_run = {
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

    all_backtest_signals = []
    for label, config in configs_to_run.items():
        print(f"\n  Running V8 MGCC [{label}]...")
        results = run_backtest(box_scores, odds_data, config, verbose=False)

        for n_legs, stats in results.get('by_legs', {}).items():
            if stats['total'] > 0:
                print(f"    {n_legs}L: {stats['total']} parlays, {stats['wins']} wins, "
                      f"{stats['accuracy']*100:.1f}% acc, {stats['leg_accuracy']*100:.1f}% leg acc, "
                      f"${stats['pnl']:+d} P&L, {stats['roi']*100:.1f}% ROI")

        # Convert to webapp format
        for pick in results.get('picks', []):
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
                    'edge': 0,
                    'gates': leg.get('gates', 0),
                    'clearance': leg.get('clearance', 0),
                })
            signal = {
                'date': pick.get('date', ''),
                'betType': 'parlay',
                'n_legs': pick.get('n_legs', len(legs)),
                'legs': legs,
                'odds': pick.get('parlay_american', 0),
                'hit': pick.get('hit'),
                'pnl': pick.get('pnl', 0),
                'wager': pick.get('wager', 100),
                'bet': f"V8 {pick.get('n_legs', 2)}-Leg MGCC Parlay [{label}]",
                'engine': f'v8_mgcc_nba_{label}',
                'source': 'backtest',
                'combined_prob': pick.get('combined_prob', 0),
                'parlay_ev': pick.get('parlay_ev', 0),
            }
            all_backtest_signals.append(signal)

    # Save V8 signals separately
    os.makedirs(WEBAPP_DATA, exist_ok=True)
    signals_path = os.path.join(WEBAPP_DATA, 'nba_v8_mgcc_signals.json')

    # Merge with existing live signals
    existing = []
    try:
        with open(signals_path) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    live_signals = [s for s in existing if s.get('source') == 'live']
    live_dates = set(s['date'] for s in live_signals)
    merged = [s for s in all_backtest_signals if s['date'] not in live_dates]
    merged.extend(live_signals)

    with open(signals_path, 'w') as f:
        json.dump(merged, f, indent=2)

    # Stats
    resolved = [s for s in merged if s.get('hit') is not None]
    wins = sum(1 for s in resolved if s['hit'])
    total_pnl = sum(s.get('pnl', 0) for s in resolved)
    total_wager = sum(s.get('wager', 100) for s in resolved)
    stats = {
        'parlays': {
            'total': len(resolved),
            'wins': wins,
            'accuracy': wins / len(resolved) if resolved else 0,
            'pnl': total_pnl,
            'roi': total_pnl / total_wager if total_wager > 0 else 0,
        },
        'model': 'NBA V8 MGCC (Multi-Gate Certainty Cascade)',
        'innovation': 'Hierarchical Evidence Stacking for multi-leg parlays',
        'generated': datetime.now().isoformat(),
    }
    with open(os.path.join(WEBAPP_DATA, 'nba_v8_mgcc_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved {len(merged)} V8 MGCC signals to {signals_path}")
    return merged


def generate_mlb_v8_parlays():
    """Generate MLB V8 MGCC parlay picks."""
    print("\n=== MLB V8 MGCC PARLAYS ===")

    from mlb.strategy_v8_mgcc import run_backtest, MLB_V8_CONFIG

    print("  Loading MLB data...")
    try:
        box_scores = load_mlb_boxscores()
        odds_data = load_mlb_odds()
    except FileNotFoundError:
        print("  MLB data not found, skipping")
        return []

    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    configs_to_run = {
        'all_2L': {**MLB_V8_CONFIG, 'PARLAY_LEGS': [2]},
        'all_3L': {**MLB_V8_CONFIG, 'PARLAY_LEGS': [3]},
        'relaxed_2L': {**MLB_V8_CONFIG, 'PARLAY_LEGS': [2],
                       'BCF_MIN_LB': 0.80, 'MWU_MIN_HR': 0.60, 'SC_MIN_STREAK': 1},
    }

    all_backtest_signals = []
    for label, config in configs_to_run.items():
        print(f"\n  Running MLB V8 MGCC [{label}]...")
        results = run_backtest(box_scores, odds_data, config, verbose=False)

        for n_legs, stats in results.get('by_legs', {}).items():
            if stats['total'] > 0:
                print(f"    {n_legs}L: {stats['total']} parlays, {stats['wins']} wins, "
                      f"{stats['accuracy']*100:.1f}% acc, {stats['leg_accuracy']*100:.1f}% leg acc, "
                      f"${stats['pnl']:+d} P&L")

        for pick in results.get('picks', []):
            legs = [{'player': l.get('player', ''), 'team': '',
                     'line': l.get('line', 0), 'odds': l.get('odds', 0),
                     'stat': l.get('stat', ''), 'statLabel': l.get('stat', '').upper(),
                     'cascadeScore': l.get('lower_bound', 0),
                     'hit': l.get('hit'), 'actual': l.get('actual'),
                     'gates': l.get('gates', 0), 'clearance': l.get('clearance', 0)}
                    for l in pick.get('legs', [])]
            all_backtest_signals.append({
                'date': pick.get('date', ''), 'betType': 'parlay',
                'n_legs': pick.get('n_legs', len(legs)), 'legs': legs,
                'odds': pick.get('parlay_american', 0),
                'hit': pick.get('hit'), 'pnl': pick.get('pnl', 0),
                'wager': pick.get('wager', 100),
                'bet': f"V8 {pick.get('n_legs', 2)}-Leg MLB MGCC [{label}]",
                'engine': f'v8_mgcc_mlb_{label}', 'source': 'backtest',
            })

    os.makedirs(WEBAPP_DATA, exist_ok=True)
    signals_path = os.path.join(WEBAPP_DATA, 'mlb_v8_mgcc_signals.json')
    with open(signals_path, 'w') as f:
        json.dump(all_backtest_signals, f, indent=2)
    print(f"\n  Saved {len(all_backtest_signals)} MLB V8 signals to {signals_path}")
    return all_backtest_signals


def main():
    parser = argparse.ArgumentParser(description='Generate positive odds picks')
    parser.add_argument('--nba', action='store_true', help='NBA only')
    parser.add_argument('--mlb', action='store_true', help='MLB only')
    parser.add_argument('--v8', action='store_true', help='V8 MGCC parlays only')
    args = parser.parse_args()

    print("=" * 60)
    print("POSITIVE ODDS MODELS — Pick Generator")
    print("=" * 60)

    do_all = not args.nba and not args.mlb and not args.v8

    if do_all or args.nba:
        generate_nba_picks()

    if do_all or args.mlb:
        generate_mlb_picks()

    if do_all or args.v8:
        generate_nba_v8_parlays()
        generate_mlb_v8_parlays()

    print("\n" + "=" * 60)
    print("DONE — Backtest history saved to webapp/positive-odds/data/")
    print("=" * 60)


if __name__ == '__main__':
    main()
