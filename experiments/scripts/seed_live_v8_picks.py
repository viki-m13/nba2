#!/usr/bin/env python3
"""
V8 CSPE — NBA & MLB Live Picks & Result Resolution
====================================================
Mirrors the Positive Odds seed_live_picks.py flow but uses V8 CSPE strategy.

Flow:
  1. Resolve pending picks from previous days via ESPN box scores
  2. Build player model from all historical box scores
  3. Fetch live odds from The Odds API
  4. Generate tonight's picks using V8 CSPE generate_picks_for_date
  5. Save signals, stats, recommendations to webapp/v8-mgcc/data/
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

from nba.strategy import NBAPlayerModel, _update_model
from nba.strategy_v8_cspe import generate_picks_for_date, NBA_V8_CONFIG, compute_convergence_score
from shared.data_loader import load_nba_boxscores

WEBAPP_DATA = os.path.join(PROJECT_ROOT, 'webapp', 'v8-mgcc', 'data')
SIGNALS_FILE = os.path.join(WEBAPP_DATA, 'nba_v8_signals.json')
STATS_FILE = os.path.join(WEBAPP_DATA, 'nba_v8_stats.json')
RECS_FILE = os.path.join(WEBAPP_DATA, 'nba_v8_recommendations.json')

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

                if signal.get('betType') == 'parlay' and signal.get('legs'):
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

                elif signal.get('betType') == 'single':
                    ps = player_stats.get(signal.get('player'))
                    if not ps:
                        continue
                    stat_key = signal.get('stat', 'pts')
                    actual = ps.get(stat_key, 0)
                    signal['actual'] = actual
                    signal['hit'] = actual > signal['line']
                    odds = signal.get('odds', -110)
                    decimal = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
                    signal['pnl'] = round((decimal - 1) * signal.get('wager', 100)) if signal['hit'] else -signal.get('wager', 100)
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
    resolved = [s for s in signals if s.get('hit') is not None]
    wins = sum(1 for s in resolved if s['hit'])
    total_pnl = sum(s.get('pnl', 0) for s in resolved)
    total_wager = sum(s.get('wager', 100) for s in resolved)
    all_legs = []
    for s in resolved:
        all_legs.extend(s.get('legs', []))
    leg_hits = sum(1 for l in all_legs if l.get('hit'))

    stats = {
        'total': len(resolved),
        'wins': wins,
        'accuracy': wins / len(resolved) if resolved else 0,
        'pnl': total_pnl,
        'roi': total_pnl / total_wager if total_wager > 0 else 0,
        'leg_total': len(all_legs),
        'leg_hits': leg_hits,
        'leg_accuracy': leg_hits / len(all_legs) if all_legs else 0,
        'model': 'V8 CSPE (Convergence Score Parlay Engine)',
        'generated': datetime.now().isoformat(),
    }
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Updated stats: {len(resolved)} picks, {wins}W, '
          f'{stats["accuracy"]*100:.1f}% acc, ${total_pnl:+d} P&L')


# =============================================================================
# Build player model from all box scores
# =============================================================================
def build_player_model():
    box_scores = load_nba_boxscores()
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
    print('\nFetching tonight\'s games from Odds API...')

    events_url = f'{ODDS_API_BASE}/sports/basketball_nba/events?apiKey={ODDS_API_KEY}'
    events = curl_fetch(events_url)
    if not events or (isinstance(events, dict) and events.get('error_code')):
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
                'playerProps': {},
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

            live_odds[game_key] = game_props

            pts_count = len(game_props['playerProps'])
            reb_count = len(game_props['rebProps'])
            ast_count = len(game_props['astProps'])
            pra_count = len(game_props['praProps'])
            print(f'  {game_key}: {pts_count} PTS, {reb_count} REB, {ast_count} AST, {pra_count} PRA props')

        except Exception as e:
            print(f'  Error fetching props for {event.get("id")}: {e}')

        time.sleep(0.2)

    return live_odds


def format_signal(pick):
    """Convert V8 CSPE recommendation to webapp signal format."""
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
            'hit': None,
            'actual': None,
            'edge': leg.get('edge', 0),
            'hr_20': leg.get('hit_rate_20g', 0),
            'clearance': leg.get('clearance', 0),
            'streak': leg.get('streak', 0),
        })
    return {
        'date': get_date_str(0),
        'betType': 'parlay',
        'n_legs': pick.get('n_legs', len(legs)),
        'legs': legs,
        'odds': pick.get('parlay_american', 0),
        'hit': None,
        'pnl': None,
        'wager': pick.get('wager', 100),
        'bet': f"V8 CSPE {pick.get('n_legs', 2)}-Leg Parlay",
        'engine': 'v8_cspe_nba',
        'source': 'live',
        'combined_prob': pick.get('combined_prob', 0),
        'parlay_ev': pick.get('expected_value', 0),
        'generated_at': datetime.now().isoformat(),
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


def _dedup_key(s):
    """Generate dedup key for a signal."""
    if s.get('betType') == 'parlay' and s.get('legs'):
        leg_key = '~'.join(sorted(f"{l.get('player')}|{l.get('line')}|{l.get('stat', 'pts')}" for l in s['legs']))
        return f"parlay|{leg_key}"
    elif s.get('betType') == 'single':
        return f"{s.get('player')}|{s.get('line')}|{s.get('stat', 'pts')}|single"
    return None


def main():
    print('\n' + '=' * 60)
    print('V8 CSPE — NBA Live Picks')
    print('=' * 60)

    os.makedirs(WEBAPP_DATA, exist_ok=True)

    # Step 0: Resolve pending picks
    resolve_results()

    # Build player model
    print('\nBuilding player model from box scores...')
    model = build_player_model()

    # Fetch live odds
    live_odds = fetch_live_odds()
    today = get_date_str(0)

    if not live_odds:
        print('No live odds available.')
        _write_recs_if_empty(RECS_FILE, today, 'V8 CSPE (Convergence Score Parlay Engine)')
        return

    # Generate picks using V8 CSPE strategy
    print('\nRunning V8 CSPE signal analysis...')
    recs = generate_picks_for_date(model, live_odds, NBA_V8_CONFIG)

    if not recs:
        print('No bets meet V8 CSPE quality thresholds tonight.')
        _write_recs_if_empty(RECS_FILE, today, 'V8 CSPE (Convergence Score Parlay Engine)')
        return

    # Format signals
    new_signals = [format_signal(r) for r in recs]

    # Load existing signals and merge (accumulate, don't overwrite)
    try:
        with open(SIGNALS_FILE) as f:
            existing_signals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_signals = []

    # Accumulate: keep prior live picks, add only unique new ones
    today_existing_live = [s for s in existing_signals if s.get('date') == today and s.get('source') == 'live']
    existing_keys = set(_dedup_key(s) for s in today_existing_live if _dedup_key(s))

    unique_new = [s for s in new_signals if _dedup_key(s) and _dedup_key(s) not in existing_keys]
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

    prev_keys = set(_dedup_key(s) for s in prev_picks if _dedup_key(s))
    merged_picks = list(prev_picks) + [s for s in new_signals if _dedup_key(s) and _dedup_key(s) not in prev_keys]

    recommendations = {
        'generated': datetime.now().isoformat(),
        'date': today,
        'engine': 'V8 CSPE (Convergence Score Parlay Engine)',
        'picks': merged_picks,
    }
    with open(RECS_FILE, 'w') as f:
        json.dump(recommendations, f, indent=2)

    # Update stats
    update_stats(existing_signals)

    # Print summary
    print('\n' + '=' * 60)
    print('TONIGHT\'S V8 CSPE PICKS:')
    print('=' * 60)

    for r in recs:
        legs_str = ' + '.join(f"{l['player']} O{l['line']} {l['stat'].upper()}" for l in r['legs'])
        print(f"  PARLAY ({r['n_legs']}L): {legs_str} | Odds: {r['parlay_american']:+d}")

    print(f'\nParlays: {len(recs)}')
    print(f'Saved to {SIGNALS_FILE}')


if __name__ == '__main__':
    main()
