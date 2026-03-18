#!/usr/bin/env python3
"""
Positive Odds — NBA Live Picks & Result Resolution
====================================================
Mirrors the original scripts/seed-live-picks.js but uses the experiment
strategy (HECE positive-odds model). Does NOT touch main webapp data.

Flow:
  1. Resolve pending picks from previous days via ESPN box scores
  2. Build player model from all historical box scores
  3. Fetch live odds from The Odds API
  4. Generate tonight's picks using experiment strategy compute_signal
  5. Save signals, stats, recommendations to webapp/positive-odds/data/
"""

import sys
import os
import json
import math
import subprocess
import time
from datetime import datetime, timedelta
from collections import defaultdict

# Add experiments root to path
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

from nba.strategy import (
    NBAPlayerModel, compute_signal, compute_plus_money_signal,
    build_parlays, build_plus_money_parlays, _update_model, _safe_int,
)
from shared.odds_math import pnl_for_bet, decimal_to_american, parlay_decimal_odds
from shared.data_loader import load_nba_boxscores

WEBAPP_DATA = os.path.join(PROJECT_ROOT, 'webapp', 'positive-odds', 'data')
SIGNALS_FILE = os.path.join(WEBAPP_DATA, 'nba_signals.json')
STATS_FILE = os.path.join(WEBAPP_DATA, 'nba_stats.json')
RECS_FILE = os.path.join(WEBAPP_DATA, 'nba_recommendations.json')

ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8'
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'

TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GS', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NO',
    'New York Knicks': 'NY', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SA',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTAH',
    'Washington Wizards': 'WSH',
}


def team_abbr(name):
    return TEAM_MAP.get(name, name)


def curl_fetch(url):
    """Fetch JSON via curl (bypasses DNS issues in sandboxed environments)."""
    try:
        result = subprocess.run(
            ['curl', '-s', url],
            capture_output=True, text=True, timeout=30
        )
        return json.loads(result.stdout) if result.stdout else None
    except Exception:
        return None


def get_date_str(days_ago=0):
    d = datetime.now() - timedelta(days=days_ago)
    return d.strftime('%Y%m%d')


# =============================================================================
# Resolve pending signals against ESPN box scores
# =============================================================================
def resolve_results():
    """Resolve pending picks using ESPN box score data."""
    try:
        with open(SIGNALS_FILE) as f:
            signals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    pending = [s for s in signals if s.get('hit') is None]
    if not pending:
        print('No pending signals to resolve.')
        return

    pending_dates = sorted(set(s['date'] for s in pending))
    print(f'Resolving {len(pending)} pending signals from {len(pending_dates)} dates...')

    resolved = 0

    for date in pending_dates:
        try:
            scoreboard_url = f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}'
            scoreboard_data = curl_fetch(scoreboard_url)
            if not scoreboard_data or not scoreboard_data.get('events'):
                continue

            player_stats = {}

            for event in scoreboard_data['events']:
                status = event.get('status', {}).get('type', {})
                if not status.get('completed'):
                    continue

                try:
                    box_url = f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event["id"]}'
                    box_data = curl_fetch(box_url)
                    if not box_data:
                        continue

                    players = box_data.get('boxscore', {}).get('players', [])
                    for team in players:
                        for stat_group in team.get('statistics', []):
                            headers = [l.lower() for l in stat_group.get('labels', [])]
                            pts_idx = headers.index('pts') if 'pts' in headers else -1
                            reb_idx = headers.index('reb') if 'reb' in headers else -1
                            ast_idx = headers.index('ast') if 'ast' in headers else -1

                            for athlete in stat_group.get('athletes', []):
                                name = athlete.get('athlete', {}).get('displayName')
                                if not name:
                                    continue
                                stats = athlete.get('stats', [])
                                pts = int(stats[pts_idx]) if pts_idx >= 0 and pts_idx < len(stats) else 0
                                reb = int(stats[reb_idx]) if reb_idx >= 0 and reb_idx < len(stats) else 0
                                ast = int(stats[ast_idx]) if ast_idx >= 0 and ast_idx < len(stats) else 0
                                player_stats[name] = {'pts': pts, 'reb': reb, 'ast': ast, 'pra': pts + reb + ast}
                except Exception:
                    pass
                time.sleep(0.1)

            # Resolve signals for this date
            for signal in signals:
                if signal['date'] != date or signal.get('hit') is not None:
                    continue

                if signal.get('betType') == 'single':
                    ps = player_stats.get(signal.get('player'))
                    if not ps:
                        continue
                    stat_key = signal.get('stat', 'pts')
                    actual = ps.get(stat_key, 0)
                    signal['actual'] = actual
                    signal['hit'] = actual > signal['line']
                    odds = signal.get('odds', 100)
                    decimal = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
                    signal['pnl'] = round((decimal - 1) * signal.get('wager', 100)) if signal['hit'] else -signal.get('wager', 100)
                    resolved += 1

                elif signal.get('betType') == 'parlay' and signal.get('legs'):
                    all_hit = True
                    any_resolved = False
                    for leg in signal['legs']:
                        if leg.get('hit') is not None:
                            if not leg['hit']:
                                all_hit = False
                            continue
                        ps = player_stats.get(leg.get('player'))
                        if not ps:
                            all_hit = False
                            continue
                        stat_key = leg.get('stat', 'pts')
                        actual = ps.get(stat_key, 0)
                        leg['actual'] = actual
                        leg['hit'] = actual > leg['line']
                        if not leg['hit']:
                            all_hit = False
                        any_resolved = True

                    if any_resolved and all(l.get('hit') is not None for l in signal['legs']):
                        signal['hit'] = all_hit
                        parlay_decimal = 1
                        for l in signal['legs']:
                            lo = l.get('odds', 100)
                            ld = (lo / 100) + 1 if lo > 0 else (100 / abs(lo)) + 1
                            parlay_decimal *= ld
                        wager = signal.get('wager', 100)
                        signal['pnl'] = round((parlay_decimal - 1) * wager) if all_hit else -wager
                        resolved += 1
        except Exception as e:
            print(f'  Error resolving date {date}: {e}')
        time.sleep(0.2)

    if resolved > 0:
        with open(SIGNALS_FILE, 'w') as f:
            json.dump(signals, f, indent=2)
        print(f'Resolved {resolved} signals.')
        update_stats(signals)
    else:
        print('No signals could be resolved (games may not be complete yet).')


def update_stats(signals):
    """Recompute stats from resolved signals."""
    singles = [s for s in signals if s.get('betType') == 'single' and s.get('hit') is not None]
    parlays = [s for s in signals if s.get('betType') == 'parlay' and s.get('hit') is not None]

    s_wins = sum(1 for s in singles if s['hit'])
    p_wins = sum(1 for p in parlays if p['hit'])
    s_pnl = sum(s.get('pnl', 0) for s in singles)
    p_pnl = sum(p.get('pnl', 0) for p in parlays)
    s_wagered = sum(s.get('wager', 100) for s in singles)
    p_wagered = sum(p.get('wager', 100) for p in parlays)
    total = len(singles) + len(parlays)
    total_wins = s_wins + p_wins
    total_pnl = s_pnl + p_pnl
    total_wagered = s_wagered + p_wagered

    stats = {
        'singles': {
            'total': len(singles), 'wins': s_wins,
            'accuracy': s_wins / len(singles) if singles else 0,
            'pnl': s_pnl,
        },
        'parlays': {
            'total': len(parlays), 'wins': p_wins,
            'accuracy': p_wins / len(parlays) if parlays else 0,
            'pnl': p_pnl,
        },
        'overall': {
            'total': total, 'wins': total_wins,
            'accuracy': total_wins / total if total > 0 else 0,
            'pnl': total_pnl,
            'roi': total_pnl / total_wagered if total_wagered > 0 else 0,
        },
        'model': 'NBA Positive Odds v1 (HECE Champion)',
        'config_version': 'optimized',
        'min_odds': '+100',
        'generated': datetime.now().isoformat(),
    }
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    print('Updated backtest stats.')


# =============================================================================
# Build player model from all box scores
# =============================================================================
def build_player_model():
    """Build NBAPlayerModel from all historical box scores."""
    box_scores = load_nba_boxscores()
    # Filter All-Star games
    box_scores = [g for g in box_scores
                  if g.get('home', '') not in ['STARS', 'WORLD']
                  and g.get('away', '') not in ['STARS', 'WORLD']]

    model = NBAPlayerModel()
    sorted_games = sorted(box_scores, key=lambda g: g['date'])

    for game in sorted_games:
        _update_model(model, game, game['date'])

    print(f'Player model built: {len(model.profiles)} players from {len(sorted_games)} games')
    return model


# =============================================================================
# Fetch live odds from The Odds API
# =============================================================================
def fetch_live_odds():
    """Fetch tonight's games and FanDuel player props from The Odds API."""
    print('\nFetching tonight\'s games from Odds API...')

    events_url = f'{ODDS_API_BASE}/sports/basketball_nba/events?apiKey={ODDS_API_KEY}'
    events = curl_fetch(events_url)
    if not events or isinstance(events, dict) and events.get('error_code'):
        print('Could not fetch events.')
        return None

    if not events:
        print('No games scheduled tonight.')
        return None

    print(f'Found {len(events)} games tonight')

    live_odds = {}
    for event in events:
        try:
            markets = 'player_points,player_rebounds,player_assists,player_points_rebounds_assists,player_points_alternate,player_rebounds_alternate,player_assists_alternate,player_points_rebounds_assists_alternate'
            props_url = f'{ODDS_API_BASE}/sports/basketball_nba/events/{event["id"]}/odds?apiKey={ODDS_API_KEY}&regions=us&markets={markets}&oddsFormat=american&bookmakers=fanduel'
            props_data = curl_fetch(props_url)
            if not props_data:
                continue

            home_abbr = team_abbr(event.get('home_team', ''))
            away_abbr = team_abbr(event.get('away_team', ''))
            game_key = f'{away_abbr}@{home_abbr}'

            bookmakers = props_data.get('bookmakers', [])
            fd = next((b for b in bookmakers if b['key'] == 'fanduel'), None)
            if not fd:
                continue

            game_props = {
                'playerProps': {},  # pts
                'rebProps': {},
                'astProps': {},
                'praProps': {},
            }
            market_map = {
                'player_points': 'playerProps',
                'player_points_alternate': 'playerProps',
                'player_rebounds': 'rebProps',
                'player_rebounds_alternate': 'rebProps',
                'player_assists': 'astProps',
                'player_assists_alternate': 'astProps',
                'player_points_rebounds_assists': 'praProps',
                'player_points_rebounds_assists_alternate': 'praProps',
            }

            for mkt in fd.get('markets', []):
                prop_type = market_map.get(mkt['key'])
                if not prop_type:
                    continue
                for outcome in mkt.get('outcomes', []):
                    player = outcome.get('description')
                    threshold = outcome.get('point')
                    if not player or threshold is None:
                        continue
                    if player not in game_props[prop_type]:
                        game_props[prop_type][player] = {}
                    thr_str = str(threshold)
                    if thr_str not in game_props[prop_type][player]:
                        game_props[prop_type][player][thr_str] = {}
                    if outcome.get('name') == 'Over':
                        game_props[prop_type][player][thr_str]['overOdds'] = outcome['price']

            live_odds[game_key] = {
                'gameKey': game_key,
                'homeTeam': home_abbr,
                'awayTeam': away_abbr,
                **game_props,
            }

            pts_count = len(game_props['playerProps'])
            reb_count = len(game_props['rebProps'])
            ast_count = len(game_props['astProps'])
            pra_count = len(game_props['praProps'])
            print(f'  {game_key}: {pts_count} PTS, {reb_count} REB, {ast_count} AST, {pra_count} PRA props')

        except Exception as e:
            print(f'  Error fetching props for {event.get("id")}: {e}')

        time.sleep(0.2)

    return live_odds


# =============================================================================
# Generate tonight's picks using experiment strategy
# =============================================================================
def generate_tonight_picks(model, live_odds, config):
    """Generate picks using the experiment HECE strategy on live odds."""
    all_signals = []

    for game_key, odds_data in live_odds.items():
        home_team = odds_data.get('homeTeam', '')
        away_team = odds_data.get('awayTeam', '')

        # Get all players with props and check if they're in our model
        all_players = set()
        for prop_type in ['playerProps', 'rebProps', 'astProps', 'praProps']:
            all_players.update(odds_data.get(prop_type, {}).keys())

        for player_name in all_players:
            player_team = model.get_team(player_name)
            if not player_team:
                continue

            ha = 'home' if player_team == home_team else 'away'
            opponent = away_team if player_team == home_team else home_team

            # Standard props: pts, reb, ast
            for prop_key, stat_key in [
                ('playerProps', 'pts'),
                ('rebProps', 'reb'),
                ('astProps', 'ast'),
            ]:
                props = odds_data.get(prop_key, {}).get(player_name, {})
                for thr_str, data in props.items():
                    try:
                        line = float(thr_str)
                    except (ValueError, TypeError):
                        continue
                    odds_val = data.get('overOdds')
                    if odds_val is None:
                        continue
                    if odds_val < config.get('MIN_ODDS', -250) or odds_val > config.get('MAX_ODDS', 500):
                        continue

                    sig = compute_signal(
                        model, player_name, stat_key, line, odds_val, config,
                        home_away=ha, date=get_date_str(0), opponent=opponent
                    )
                    if sig:
                        sig['date'] = get_date_str(0)
                        sig['game_key'] = game_key
                        all_signals.append(sig)

            # PRA props
            pra_props = odds_data.get('praProps', {}).get(player_name, {})
            for thr_str, data in pra_props.items():
                try:
                    line = float(thr_str)
                except (ValueError, TypeError):
                    continue
                odds_val = data.get('overOdds')
                if odds_val is None:
                    continue
                if odds_val < config.get('MIN_ODDS', -250) or odds_val > config.get('MAX_ODDS', 500):
                    continue

                sig = compute_signal(
                    model, player_name, 'pra', line, odds_val, config,
                    home_away=ha, date=get_date_str(0), opponent=opponent
                )
                if sig:
                    sig['date'] = get_date_str(0)
                    sig['game_key'] = game_key
                    all_signals.append(sig)

    if not all_signals:
        return [], []

    # Sort by EV and deduplicate by player
    all_signals.sort(key=lambda s: s['ev'], reverse=True)
    seen = set()
    unique = []
    for s in all_signals:
        if s['player'] not in seen:
            unique.append(s)
            seen.add(s['player'])

    # Singles
    singles = [s for s in unique if s['combined_score'] >= config['SINGLE_MIN_SCORE']]
    single_picks = []
    for s in singles[:config['MAX_DAILY_BETS']]:
        wager = max(50, min(400, round(config['UNIT_SIZE'] * (1 + s['kelly'] * 2))))
        single_picks.append({**s, 'bet_type': 'single', 'wager': wager,
                            'hit': None, 'actual': None})

    # Standard parlays
    parlay_elig = [s for s in unique if s['combined_score'] >= config['PARLAY_LEG_MIN_SCORE']]
    parlay_picks = []
    parlays = build_parlays(parlay_elig, model, config)
    max_parlays = config.get('PARLAY_MAX_PER_DAY', 3)
    for p in parlays[:max_parlays]:
        wager_key = f"PARLAY_WAGER_{p['n_legs']}LEG"
        wager = config.get(wager_key, 100)
        parlay_picks.append({
            'bet_type': 'parlay', 'n_legs': p['n_legs'],
            'legs': [{'player': l['player'], 'stat': l['stat'],
                      'line': l['line'], 'odds': l['odds'],
                      'hit': None, 'actual': None,
                      'combined_score': l.get('combined_score', 0),
                      'edge': l['edge']} for l in p['legs']],
            'hit': None, 'pnl': None, 'wager': wager,
            'date': get_date_str(0), 'parlay_decimal': p['parlay_decimal'],
        })

    # Pipeline B: Plus-money parlays
    if config.get('PLB_ENABLED', True):
        plb_signals = []
        for game_key, odds_data in live_odds.items():
            home_team = odds_data.get('homeTeam', '')
            away_team = odds_data.get('awayTeam', '')

            all_players_plb = set()
            for prop_type in ['playerProps', 'rebProps', 'astProps', 'praProps']:
                all_players_plb.update(odds_data.get(prop_type, {}).keys())

            for player_name in all_players_plb:
                player_team = model.get_team(player_name)
                if not player_team:
                    continue
                ha = 'home' if player_team == home_team else 'away'
                opponent = away_team if player_team == home_team else home_team

                for prop_key, stat_key in [
                    ('playerProps', 'pts'), ('rebProps', 'reb'),
                    ('astProps', 'ast'),
                ]:
                    props = odds_data.get(prop_key, {}).get(player_name, {})
                    for thr_str, data in props.items():
                        try:
                            line = float(thr_str)
                        except (ValueError, TypeError):
                            continue
                        odds_val = data.get('overOdds')
                        if odds_val is None:
                            continue
                        if odds_val < config.get('PLB_MIN_ODDS', 100):
                            continue
                        if odds_val > config.get('PLB_MAX_ODDS', 500):
                            continue
                        sig = compute_plus_money_signal(
                            model, player_name, stat_key, line, odds_val, config,
                            home_away=ha, date=get_date_str(0), opponent=opponent
                        )
                        if sig:
                            sig['date'] = get_date_str(0)
                            sig['game_key'] = game_key
                            plb_signals.append(sig)

                # PRA for Pipeline B
                pra_props = odds_data.get('praProps', {}).get(player_name, {})
                for thr_str, data in pra_props.items():
                    try:
                        line = float(thr_str)
                    except (ValueError, TypeError):
                        continue
                    odds_val = data.get('overOdds')
                    if odds_val is None:
                        continue
                    if odds_val < config.get('PLB_MIN_ODDS', 100):
                        continue
                    if odds_val > config.get('PLB_MAX_ODDS', 500):
                        continue
                    sig = compute_plus_money_signal(
                        model, player_name, 'pra', line, odds_val, config,
                        home_away=ha, date=get_date_str(0), opponent=opponent
                    )
                    if sig:
                        sig['date'] = get_date_str(0)
                        sig['game_key'] = game_key
                        plb_signals.append(sig)

        if plb_signals:
            plb_by_player = {}
            for s in plb_signals:
                if s['player'] not in plb_by_player or s['ev'] > plb_by_player[s['player']]['ev']:
                    plb_by_player[s['player']] = s
            plb_unique = list(plb_by_player.values())

            plb_parlays = build_plus_money_parlays(plb_unique, model, config)
            for p in plb_parlays:
                wager_key = f"PLB_WAGER_{p['n_legs']}LEG"
                wager = config.get(wager_key, 50)
                parlay_picks.append({
                    'bet_type': 'parlay_plb', 'n_legs': p['n_legs'],
                    'legs': [{'player': l['player'], 'stat': l['stat'],
                              'line': l['line'], 'odds': l['odds'],
                              'hit': None, 'actual': None,
                              'combined_score': l.get('combined_score', 0),
                              'edge': l['edge']} for l in p['legs']],
                    'hit': None, 'pnl': None, 'wager': wager,
                    'date': get_date_str(0), 'parlay_decimal': p['parlay_decimal'],
                    'pipeline': 'B',
                })

    return single_picks, parlay_picks


def format_signal(pick):
    """Convert strategy pick to webapp signal format."""
    if pick['bet_type'] == 'single':
        return {
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
            'pnl': pick.get('pnl'),
            'wager': pick.get('wager', 100),
            'bet': f"{pick.get('player', '')} O{pick.get('line', '')} {pick.get('stat', '').upper()}",
            'engine': 'positive_odds_nba',
            'betSubType': 'single',
            'source': 'live',
        }
    else:
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
        parlay_decimal = pick.get('parlay_decimal', 1)
        if parlay_decimal >= 2:
            odds = int(parlay_decimal * 100 - 100)
        else:
            odds = int(-100 / (parlay_decimal - 1)) if parlay_decimal > 1 else 100

        return {
            'date': pick.get('date', ''),
            'betType': 'parlay',
            'n_legs': pick.get('n_legs', len(legs)),
            'legs': legs,
            'odds': odds,
            'hit': pick.get('hit'),
            'pnl': pick.get('pnl'),
            'wager': pick.get('wager', 100),
            'bet': f"{pick.get('n_legs', 2)}-Leg Parlay",
            'engine': 'positive_odds_nba',
            'source': 'live',
        }


def main():
    print('\n' + '=' * 60)
    print('POSITIVE ODDS — NBA Live Picks')
    print('=' * 60)

    os.makedirs(WEBAPP_DATA, exist_ok=True)

    # Step 0: Resolve pending picks
    resolve_results()

    # Load config
    config_path = os.path.join(EXPERIMENTS_DIR, 'nba', 'output_v3', 'plus_money_accuracy_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)['config']
        print(f'Loaded optimized config from {config_path}')
    else:
        champion_path = os.path.join(EXPERIMENTS_DIR, 'nba', 'output_v3', 'champion_config.json')
        with open(champion_path) as f:
            config = json.load(f)['config']
        print(f'Loaded champion config from {champion_path}')

    # Build player model
    print('\nBuilding player model from box scores...')
    model = build_player_model()

    # Fetch live odds
    live_odds = fetch_live_odds()
    today = get_date_str(0)

    if not live_odds:
        print('No live odds available.')
        with open(RECS_FILE, 'w') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'date': today,
                'engine': 'NBA Positive Odds HECE v1',
                'picks': [],
            }, f, indent=2)
        return

    # Generate picks
    print('\nRunning HECE signal analysis...')
    single_picks, parlay_picks = generate_tonight_picks(model, live_odds, config)

    all_picks = single_picks + parlay_picks
    if not all_picks:
        print('No bets meet quality thresholds tonight.')
        with open(RECS_FILE, 'w') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'date': today,
                'engine': 'NBA Positive Odds HECE v1',
                'picks': [],
            }, f, indent=2)
        return

    # Format signals
    new_signals = [format_signal(p) for p in all_picks]

    # Load existing signals and merge
    try:
        with open(SIGNALS_FILE) as f:
            existing_signals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_signals = []

    # Remove today's existing live signals (avoid duplicates)
    existing_signals = [s for s in existing_signals if not (s.get('date') == today and s.get('source') == 'live')]
    existing_signals.extend(new_signals)

    # Remove backtest signals for dates that have live picks
    live_dates = set(s['date'] for s in existing_signals if s.get('source') == 'live')
    existing_signals = [s for s in existing_signals if s.get('source') == 'live' or s['date'] not in live_dates]

    with open(SIGNALS_FILE, 'w') as f:
        json.dump(existing_signals, f, indent=2)

    # Save recommendations
    recommendations = {
        'generated': datetime.now().isoformat(),
        'date': today,
        'engine': 'NBA Positive Odds HECE v1',
        'picks': new_signals,
    }
    with open(RECS_FILE, 'w') as f:
        json.dump(recommendations, f, indent=2)

    # Update stats
    update_stats(existing_signals)

    # Print summary
    print('\n' + '=' * 60)
    print('TONIGHT\'S POSITIVE ODDS PICKS:')
    print('=' * 60)

    for p in single_picks:
        print(f"  SINGLE: {p['player']} O{p['line']} {p['stat'].upper()} | Odds: +{p['odds']} | Score: {p['combined_score']*100:.0f}%")
    for p in parlay_picks:
        legs_str = ' + '.join(f"{l['player']} O{l['line']} {l['stat'].upper()}" for l in p['legs'])
        print(f"  PARLAY: {legs_str}")

    print(f'\nSingles: {len(single_picks)}')
    print(f'Parlays: {len(parlay_picks)}')
    print(f'\nSaved to {SIGNALS_FILE}')

    # Check API quota
    try:
        result = subprocess.run(
            ['curl', '-s', '-I', f'{ODDS_API_BASE}/sports/basketball_nba/events?apiKey={ODDS_API_KEY}'],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split('\n'):
            if 'x-requests-remaining' in line.lower():
                print(f'\nOdds API: {line.strip()}')
    except Exception:
        pass


if __name__ == '__main__':
    main()
