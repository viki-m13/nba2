"""
Standalone data loader for experiments.
Reads historical odds and boxscores from the webapp/data directories.
Does NOT import anything from src/ — completely self-contained.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_nba_boxscores():
    """Load ALL NBA player boxscores from webapp/data (merge all seasons)."""
    data_dir = os.path.join(BASE_DIR, 'webapp', 'data')
    all_games = []
    seen = set()

    for fname in ['player_boxscores.json', 'player_boxscores_2026.json']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                games = json.load(f)
            for g in games:
                key = (g.get('date', ''), g.get('home', ''), g.get('away', ''))
                if key not in seen:
                    all_games.append(g)
                    seen.add(key)

    if not all_games:
        raise FileNotFoundError("NBA boxscores not found")
    return all_games


def load_nba_odds():
    """Load ALL NBA historical odds from webapp/data (merge all seasons)."""
    data_dir = os.path.join(BASE_DIR, 'webapp', 'data')
    all_odds = []
    seen = set()

    for fname in ['historical_odds.json', 'historical_odds_2026.json']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                odds = json.load(f)
            for o in odds:
                key = (o.get('date', ''), o.get('gameKey', ''))
                if key not in seen:
                    all_odds.append(o)
                    seen.add(key)

    # Also merge rebound/assist props into main odds data
    reb_ast_path = os.path.join(data_dir, 'historical_reb_ast_props.json')
    if os.path.exists(reb_ast_path):
        with open(reb_ast_path, 'r') as f:
            reb_ast = json.load(f)
        # Index existing odds by (date, gameKey)
        odds_idx = {}
        for o in all_odds:
            k = (o.get('date', ''), o.get('gameKey', ''))
            odds_idx[k] = o
        # Merge reb/ast props
        for rp in reb_ast:
            k = (rp.get('date', ''), rp.get('gameKey', ''))
            if k in odds_idx:
                target = odds_idx[k]
                if 'rebProps' in rp and 'rebProps' not in target:
                    target['rebProps'] = rp['rebProps']
                if 'astProps' in rp and 'astProps' not in target:
                    target['astProps'] = rp['astProps']

    if not all_odds:
        raise FileNotFoundError("NBA odds not found")
    return all_odds


def load_mlb_boxscores():
    """Load MLB player boxscores."""
    # Try expanded first, fall back to original
    for fname in ['mlb_player_boxscores_expanded.json', 'mlb_player_boxscores.json']:
        path = os.path.join(BASE_DIR, 'mlb', 'webapp', 'data', fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError("MLB boxscores not found")


def load_mlb_odds():
    """Load MLB historical odds."""
    for fname in ['mlb_historical_odds_expanded.json', 'mlb_historical_odds.json']:
        path = os.path.join(BASE_DIR, 'mlb', 'webapp', 'data', fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError("MLB odds not found")


def index_by_date(records, date_field='date'):
    """Group records by date into a dict."""
    by_date = {}
    for r in records:
        d = r.get(date_field, '')
        if d not in by_date:
            by_date[d] = []
        by_date[d].append(r)
    return by_date


def index_odds_by_date_game(odds_data):
    """Index odds by (date, gameKey) for fast lookup."""
    idx = {}
    for od in odds_data:
        date = od.get('date', '')
        key = od.get('gameKey', f"{od.get('awayTeam', '')}@{od.get('homeTeam', '')}")
        if date not in idx:
            idx[date] = {}
        idx[date][key] = od
    return idx
