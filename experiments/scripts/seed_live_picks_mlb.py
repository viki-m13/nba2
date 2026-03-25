#!/usr/bin/env python3
"""
Positive Odds — MLB Live Picks & Result Resolution
====================================================
Mirrors the original mlb/scripts/seed-live-picks-mlb.js but uses the
experiment strategy (HECE positive-odds model). Does NOT touch main webapp data.

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
import subprocess
import time
from datetime import datetime, timedelta

# Add experiments root to path
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

from mlb.strategy import (
    MLBPlayerModel, compute_signal, build_parlays,
    _update_model, _safe_int,
)
from shared.odds_math import pnl_for_bet
from shared.data_loader import load_mlb_boxscores

WEBAPP_DATA = os.path.join(PROJECT_ROOT, 'webapp', 'positive-odds', 'data')
SIGNALS_FILE = os.path.join(WEBAPP_DATA, 'mlb_signals.json')
STATS_FILE = os.path.join(WEBAPP_DATA, 'mlb_stats.json')
RECS_FILE = os.path.join(WEBAPP_DATA, 'mlb_recommendations.json')

ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8'
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'

TEAM_MAP = {
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB', 'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH',
}


def team_abbr(name):
    return TEAM_MAP.get(name, name)


def curl_fetch(url):
    """Fetch JSON via curl."""
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
    """Resolve pending picks using ESPN MLB box score data."""
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
            scoreboard_url = f'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date}'
            scoreboard_data = curl_fetch(scoreboard_url)
            if not scoreboard_data or not scoreboard_data.get('events'):
                continue

            player_stats = {}

            for event in scoreboard_data['events']:
                status = event.get('status', {}).get('type', {})
                if not status.get('completed'):
                    continue

                try:
                    box_url = f'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={event["id"]}'
                    box_data = curl_fetch(box_url)
                    if not box_data:
                        continue

                    rosters = box_data.get('rosters', box_data.get('boxscore', {}).get('players', []))
                    for team in rosters:
                        for category in team.get('statistics', []):
                            if category.get('type') != 'batting' and category.get('name') != 'batting':
                                continue
                            labels = [l.lower() for l in category.get('labels', [])]
                            for athlete in category.get('athletes', []):
                                name = athlete.get('athlete', {}).get('displayName')
                                if not name:
                                    continue
                                stats = athlete.get('stats', [])
                                stat_obj = {}
                                for i, label in enumerate(labels):
                                    if i < len(stats):
                                        stat_obj[label] = stats[i]

                                h = int(stat_obj.get('h', 0) or 0)
                                doubles = int(stat_obj.get('2b', 0) or 0)
                                triples = int(stat_obj.get('3b', 0) or 0)
                                hr = int(stat_obj.get('hr', 0) or 0)
                                rbi = int(stat_obj.get('rbi', 0) or 0)
                                r = int(stat_obj.get('r', 0) or 0)
                                tb = h + doubles + triples * 2 + hr * 3

                                player_stats[name] = {'h': h, 'tb': tb, 'rbi': rbi, 'r': r, 'hr': hr}
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
                    stat_key = signal.get('stat', 'h')
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
                        stat_key = leg.get('stat', 'h')
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
        'model': 'MLB Positive Odds v1 (HECE Champion)',
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
    """Build MLBPlayerModel from all historical box scores."""
    try:
        box_scores = load_mlb_boxscores()
    except FileNotFoundError:
        print('MLB box scores not found.')
        return None

    model = MLBPlayerModel()
    sorted_games = sorted(box_scores, key=lambda g: g['date'])

    for game in sorted_games:
        _update_model(model, game, game['date'])

    print(f'Player model built: {len(model.profiles)} players from {len(sorted_games)} games')
    return model


# =============================================================================
# Fetch live odds from The Odds API
# =============================================================================
def fetch_live_odds():
    """Fetch tonight's games and FanDuel batter props from The Odds API."""
    print('\nFetching tonight\'s games from Odds API...')

    events = []
    # Regular season
    reg_url = f'{ODDS_API_BASE}/sports/baseball_mlb/events?apiKey={ODDS_API_KEY}'
    reg_events = curl_fetch(reg_url)
    if reg_events and not isinstance(reg_events, dict):
        events.extend(reg_events)

    # Preseason/spring training
    pre_url = f'{ODDS_API_BASE}/sports/baseball_mlb_preseason/events?apiKey={ODDS_API_KEY}'
    pre_events = curl_fetch(pre_url)
    if pre_events and not isinstance(pre_events, dict):
        for e in pre_events:
            e['_sportKey'] = 'baseball_mlb_preseason'
        events.extend(pre_events)

    if not events:
        print('No games scheduled tonight.')
        return None

    print(f'Found {len(events)} games tonight')

    live_odds = {}
    for event in events:
        try:
            markets = 'batter_hits_alternate,batter_total_bases_alternate,batter_rbis_alternate,batter_runs_scored_alternate'
            sport_key = event.get('_sportKey', 'baseball_mlb')
            props_url = f'{ODDS_API_BASE}/sports/{sport_key}/events/{event["id"]}/odds?apiKey={ODDS_API_KEY}&regions=us&markets={markets}&oddsFormat=american&bookmakers=fanduel'
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
                'hitsProps': {},
                'tbProps': {},
                'rbiProps': {},
                'runsProps': {},
            }
            market_map = {
                'batter_hits_alternate': 'hitsProps',
                'batter_total_bases_alternate': 'tbProps',
                'batter_rbis_alternate': 'rbiProps',
                'batter_runs_scored_alternate': 'runsProps',
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

            h_count = len(game_props['hitsProps'])
            tb_count = len(game_props['tbProps'])
            rbi_count = len(game_props['rbiProps'])
            r_count = len(game_props['runsProps'])
            print(f'  {game_key}: {h_count} H, {tb_count} TB, {rbi_count} RBI, {r_count} R props')

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

        # Get all players with props
        all_players = set()
        for prop_type in ['hitsProps', 'tbProps', 'rbiProps', 'runsProps']:
            all_players.update(odds_data.get(prop_type, {}).keys())

        for player_name in all_players:
            player_team = model.get_team(player_name)
            if not player_team:
                continue

            ha = 'home' if player_team == home_team else 'away'
            opponent = away_team if player_team == home_team else home_team

            for prop_key, stat_key in [
                ('hitsProps', 'h'),
                ('tbProps', 'tb'),
                ('rbiProps', 'rbi'),
                ('runsProps', 'r'),
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
                    if odds_val < config.get('MIN_ODDS', 100) or odds_val > config.get('MAX_ODDS', 500):
                        continue

                    sig = compute_signal(
                        model, player_name, stat_key, line, odds_val, config,
                        home_away=ha, opponent=opponent
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
        wager = max(50, min(300, round(config['UNIT_SIZE'] * (1 + s['kelly']))))
        single_picks.append({**s, 'bet_type': 'single', 'wager': wager,
                            'hit': None, 'actual': None})

    # Parlays
    parlay_elig = [s for s in unique if s['combined_score'] >= config['PARLAY_LEG_MIN_SCORE']]
    parlay_picks = []
    parlays = build_parlays(parlay_elig, model, config)
    for p in parlays[:1]:
        wager = config['UNIT_SIZE']
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
            'engine': 'positive_odds_mlb',
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
            'engine': 'positive_odds_mlb',
            'source': 'live',
        }


def _write_recs_if_empty(recs_file, today, engine):
    """Only write empty recs if no prior picks exist for today."""
    try:
        prev = json.loads(open(recs_file).read())
        if prev.get('date') == today and prev.get('picks'):
            print(f'Keeping {len(prev["picks"])} picks from earlier runs.')
            return
    except Exception:
        pass
    with open(recs_file, 'w') as f:
        json.dump({'generated': datetime.now().isoformat(), 'date': today,
                   'engine': engine, 'picks': []}, f, indent=2)


def main():
    print('\n' + '=' * 60)
    print('POSITIVE ODDS — MLB Live Picks')
    print('=' * 60)

    os.makedirs(WEBAPP_DATA, exist_ok=True)

    # Step 0: Resolve pending picks
    resolve_results()

    # Load config
    config_path = os.path.join(EXPERIMENTS_DIR, 'mlb', 'output_v2', 'mlb_hece_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)['config']
        print(f'Loaded optimized config from {config_path}')
    else:
        print('MLB config not found, skipping.')
        return

    # Build player model
    print('\nBuilding player model from box scores...')
    model = build_player_model()
    if not model:
        return

    # Fetch live odds
    live_odds = fetch_live_odds()
    today = get_date_str(0)

    if not live_odds:
        print('No live odds available.')
        _write_recs_if_empty(RECS_FILE, today, 'MLB Positive Odds HECE v1')
        return

    # Generate picks
    print('\nRunning HECE signal analysis...')
    single_picks, parlay_picks = generate_tonight_picks(model, live_odds, config)

    all_picks = single_picks + parlay_picks
    if not all_picks:
        print('No bets meet quality thresholds tonight.')
        _write_recs_if_empty(RECS_FILE, today, 'MLB Positive Odds HECE v1')
        return

    # Format signals
    new_signals = [format_signal(p) for p in all_picks]
    for sig in new_signals:
        sig['generated_at'] = datetime.now().isoformat()

    # Load existing signals and merge (accumulate, don't overwrite)
    try:
        with open(SIGNALS_FILE) as f:
            existing_signals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_signals = []

    # Accumulate: keep prior live picks, add only unique new ones
    today_existing_live = [s for s in existing_signals if s.get('date') == today and s.get('source') == 'live']
    existing_keys = set()
    for s in today_existing_live:
        if s.get('betType') == 'single':
            existing_keys.add(f"{s.get('player')}|{s.get('line')}|{s.get('stat', 'pts')}|single")
        elif s.get('betType') == 'parlay' and s.get('legs'):
            leg_key = '~'.join(sorted(f"{l.get('player')}|{l.get('line')}|{l.get('stat', 'pts')}" for l in s['legs']))
            existing_keys.add(f"parlay|{leg_key}")

    unique_new = []
    for s in new_signals:
        if s.get('betType') == 'single':
            key = f"{s.get('player')}|{s.get('line')}|{s.get('stat', 'pts')}|single"
        elif s.get('betType') == 'parlay' and s.get('legs'):
            leg_key = '~'.join(sorted(f"{l.get('player')}|{l.get('line')}|{l.get('stat', 'pts')}" for l in s['legs']))
            key = f"parlay|{leg_key}"
        else:
            key = None
        if key and key not in existing_keys:
            unique_new.append(s)

    existing_signals.extend(unique_new)

    # Remove backtest signals for dates that have live picks
    live_dates = set(s['date'] for s in existing_signals if s.get('source') == 'live')
    existing_signals = [s for s in existing_signals if s.get('source') == 'live' or s['date'] not in live_dates]

    with open(SIGNALS_FILE, 'w') as f:
        json.dump(existing_signals, f, indent=2)

    # Save recommendations — accumulate across runs
    prev_picks = []
    try:
        prev_recs = json.loads(open(RECS_FILE).read())
        if prev_recs.get('date') == today:
            prev_picks = prev_recs.get('picks', [])
    except Exception:
        pass

    prev_keys = set()
    for s in prev_picks:
        if s.get('betType') == 'single':
            prev_keys.add(f"{s.get('player')}|{s.get('line')}|{s.get('stat', 'pts')}|single")
        elif s.get('betType') == 'parlay' and s.get('legs'):
            leg_key = '~'.join(sorted(f"{l.get('player')}|{l.get('line')}|{l.get('stat', 'pts')}" for l in s['legs']))
            prev_keys.add(f"parlay|{leg_key}")
    merged_picks = list(prev_picks)
    for s in new_signals:
        if s.get('betType') == 'single':
            key = f"{s.get('player')}|{s.get('line')}|{s.get('stat', 'pts')}|single"
        elif s.get('betType') == 'parlay' and s.get('legs'):
            leg_key = '~'.join(sorted(f"{l.get('player')}|{l.get('line')}|{l.get('stat', 'pts')}" for l in s['legs']))
            key = f"parlay|{leg_key}"
        else:
            key = None
        if key and key not in prev_keys:
            merged_picks.append(s)

    recommendations = {
        'generated': datetime.now().isoformat(),
        'date': today,
        'engine': 'MLB Positive Odds HECE v1',
        'picks': merged_picks,
    }
    with open(RECS_FILE, 'w') as f:
        json.dump(recommendations, f, indent=2)

    # Update stats
    update_stats(existing_signals)

    # Print summary
    print('\n' + '=' * 60)
    print('TONIGHT\'S MLB POSITIVE ODDS PICKS:')
    print('=' * 60)

    for p in single_picks:
        print(f"  SINGLE: {p['player']} O{p['line']} {p['stat'].upper()} | Odds: +{p['odds']} | Score: {p['combined_score']*100:.0f}%")
    for p in parlay_picks:
        legs_str = ' + '.join(f"{l['player']} O{l['line']} {l['stat'].upper()}" for l in p['legs'])
        print(f"  PARLAY: {legs_str}")

    print(f'\nSingles: {len(single_picks)}')
    print(f'Parlays: {len(parlay_picks)}')
    print(f'\nSaved to {SIGNALS_FILE}')


if __name__ == '__main__':
    main()
