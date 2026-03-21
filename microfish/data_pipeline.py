"""
Microfish Data Pipeline
=======================
Fetches real MLB historical odds from The Odds API.
Builds features with strict temporal discipline — NO leakage, NO forward bias.
All features use ONLY past data relative to game date.

Data sources:
1. The Odds API (historical + live odds)
2. Cached CSV files from prior API fetches

NO synthetic data generation. If no data is available, the pipeline fails
clearly rather than fabricating fake data.
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import (
    THE_ODDS_API_KEY, DATA_DIR, CACHE_DIR,
    FEATURE_LAG_DAYS, ROLLING_WINDOWS, MIN_AMERICAN_ODDS
)


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return (american / 100) + 1
    else:
        return (100 / abs(american)) + 1


def american_to_implied_prob(american: float) -> float:
    """Convert American odds to implied probability."""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def fetch_historical_odds(season: int) -> pd.DataFrame:
    """
    Fetch real historical MLB odds from The Odds API.
    Falls back to cached CSV if API unavailable.
    Never generates synthetic data.
    """
    # Check for cached API response first
    api_cache = os.path.join(CACHE_DIR, f'odds_mlb_{season}.json')
    if os.path.exists(api_cache):
        print(f"  Loading cached API odds for {season}")
        with open(api_cache, 'r') as f:
            data = json.load(f)
        return _parse_odds_api_response(data, season)

    # Check for cached CSV
    csv_cache = os.path.join(CACHE_DIR, f'mlb_historical_{season}.csv')
    if os.path.exists(csv_cache):
        print(f"  Loading cached CSV for {season}")
        df = pd.read_csv(csv_cache, parse_dates=['date'])
        return df

    # Try live API fetch
    if not THE_ODDS_API_KEY:
        print(f"  WARNING: No Odds API key and no cached data for {season}")
        return pd.DataFrame()

    print(f"  Fetching from Odds API for {season}...")
    all_events = []
    start_month = 4
    end_month = 10

    for month in range(start_month, end_month + 1):
        date_str = f"{season}-{month:02d}-01T00:00:00Z"
        url = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
        params = {
            'apiKey': THE_ODDS_API_KEY,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american',
            'date': date_str,
        }

        for attempt in range(4):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    result = resp.json()
                    events = result.get('data', result) if isinstance(result, dict) else result
                    if isinstance(events, list):
                        all_events.extend(events)
                    break
                elif resp.status_code == 422:
                    break  # Invalid date, skip
                else:
                    print(f"  API returned {resp.status_code} for {date_str}")
            except requests.exceptions.RequestException as e:
                if attempt < 3:
                    time.sleep(2 ** (attempt + 1))
                    continue
                print(f"  Error fetching {date_str}: {e}")
        time.sleep(1)  # Rate limit

    if all_events:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(api_cache, 'w') as f:
            json.dump(all_events, f)
        return _parse_odds_api_response(all_events, season)

    print(f"  No data available for {season}")
    return pd.DataFrame()


def _parse_odds_api_response(events: list, season: int) -> pd.DataFrame:
    """Parse The Odds API response into a clean DataFrame."""
    rows = []
    for event in events:
        game_date = event.get('commence_time', '')
        if isinstance(game_date, str) and 'T' in game_date:
            game_date = game_date[:10]

        home = event.get('home_team', '')
        away = event.get('away_team', '')
        game_id = event.get('id', f'{season}_{len(rows):04d}')

        bookmakers = event.get('bookmakers', [])
        # Use consensus (average across books) for most accurate odds
        all_home_odds = []
        all_away_odds = []
        for bk in bookmakers:
            for market in bk.get('markets', []):
                if market.get('key') == 'h2h':
                    outcomes = {o['name']: o['price'] for o in market.get('outcomes', [])}
                    if home in outcomes and away in outcomes:
                        all_home_odds.append(outcomes[home])
                        all_away_odds.append(outcomes[away])

        if all_home_odds and all_away_odds:
            rows.append({
                'date': game_date,
                'game_id': game_id,
                'home_team': home,
                'away_team': away,
                'home_odds': round(np.median(all_home_odds)),
                'away_odds': round(np.median(all_away_odds)),
                'bookmaker': 'consensus',
                'season': season,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def load_season_data(season: int) -> pd.DataFrame:
    """Load a full season of data."""
    print(f"Loading {season} season data...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = fetch_historical_odds(season)
    if df.empty:
        print(f"  WARNING: No data for {season}")
    else:
        print(f"  {len(df)} game records loaded")
    return df


def load_all_seasons() -> pd.DataFrame:
    """Load all configured seasons."""
    from config import BACKTEST_SEASONS
    dfs = []
    for season in BACKTEST_SEASONS:
        df = load_season_data(season)
        if not df.empty:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def add_game_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add game outcomes to the dataset.
    If 'home_wins' column already exists, use it.
    Otherwise, outcomes must be added from an external source.
    """
    if 'home_wins' in df.columns:
        return df
    # Without outcomes, we cannot backtest
    print("  WARNING: No outcome data (home_wins column). Cannot backtest without results.")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features using ONLY past data. Every feature is lagged by at
    least FEATURE_LAG_DAYS to prevent any forward information leakage.
    """
    df = df.sort_values('date').copy()

    # Pre-compute team game logs from historical results
    team_logs = {}
    for _, row in df.iterrows():
        if 'home_wins' not in row or pd.isna(row.get('home_wins')):
            continue
        for side in ['home', 'away']:
            team = row[f'{side}_team']
            if team not in team_logs:
                team_logs[team] = []
            is_home = (side == 'home')
            won = (row['home_wins'] == 1) == is_home
            team_logs[team].append({
                'date': row['date'],
                'won': won,
                'is_home': is_home,
                'odds': row[f'{side}_odds'],
                'opp_odds': row[f'{"away" if is_home else "home"}_odds'],
            })

    for team in team_logs:
        team_logs[team].sort(key=lambda x: x['date'])

    all_features = []
    for _, row in df.iterrows():
        game_date = row['date']
        cutoff = game_date - timedelta(days=FEATURE_LAG_DAYS)

        features = {
            'date': game_date,
            'game_id': row.get('game_id', ''),
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_odds': row['home_odds'],
            'away_odds': row['away_odds'],
            'season': row['season'],
        }
        if 'home_wins' in row:
            features['home_wins'] = row['home_wins']

        # Pitcher features if available
        for col in ['home_pitcher_rank', 'away_pitcher_rank',
                     'home_rest_days', 'away_rest_days']:
            if col in row and not pd.isna(row[col]):
                features[col] = row[col]

        for side in ['home', 'away']:
            team = row[f'{side}_team']
            logs = [g for g in team_logs.get(team, []) if g['date'] < cutoff]

            for window in ROLLING_WINDOWS:
                recent = logs[-window:] if len(logs) >= window else logs
                if len(recent) >= 3:
                    wins = sum(1 for g in recent if g['won'])
                    features[f'{side}_win_pct_{window}d'] = wins / len(recent)

                    home_games = [g for g in recent if g['is_home']]
                    features[f'{side}_home_win_pct_{window}d'] = (
                        sum(1 for g in home_games if g['won']) /
                        max(1, len(home_games))
                    )

                    features[f'{side}_avg_odds_{window}d'] = np.mean(
                        [g['odds'] for g in recent]
                    )

                    dog_games = [g for g in recent if g['odds'] > 0]
                    features[f'{side}_dog_win_pct_{window}d'] = (
                        sum(1 for g in dog_games if g['won']) /
                        max(1, len(dog_games)) if dog_games else 0.0
                    )

                    streak = 0
                    for g in reversed(recent):
                        if g['won']:
                            streak += 1
                        else:
                            break
                    features[f'{side}_streak_{window}d'] = streak

                    # Losing streak
                    l_streak = 0
                    for g in reversed(recent):
                        if not g['won']:
                            l_streak += 1
                        else:
                            break
                    features[f'{side}_loss_streak_{window}d'] = l_streak
                else:
                    features[f'{side}_win_pct_{window}d'] = 0.5
                    features[f'{side}_home_win_pct_{window}d'] = 0.5
                    features[f'{side}_avg_odds_{window}d'] = 0
                    features[f'{side}_dog_win_pct_{window}d'] = 0.0
                    features[f'{side}_streak_{window}d'] = 0
                    features[f'{side}_loss_streak_{window}d'] = 0

        # Odds-derived features (known at game time, not leakage)
        features['home_implied_prob'] = american_to_implied_prob(row['home_odds'])
        features['away_implied_prob'] = american_to_implied_prob(row['away_odds'])
        features['odds_diff'] = row['away_odds'] - row['home_odds']
        features['implied_prob_diff'] = (
            features['home_implied_prob'] - features['away_implied_prob']
        )

        all_features.append(features)

    return pd.DataFrame(all_features)


def get_plus200_opportunities(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all +200 underdog betting opportunities.
    Each row = one potential bet (team at +200 or higher).
    """
    rows = []

    for _, game in feature_df.iterrows():
        for side in ['home', 'away']:
            opp_side = 'away' if side == 'home' else 'home'
            odds = game[f'{side}_odds']

            if odds >= MIN_AMERICAN_ODDS:
                row = game.to_dict()
                row['bet_side'] = side
                row['bet_team'] = game[f'{side}_team']
                row['opp_team'] = game[f'{opp_side}_team']
                row['bet_odds'] = odds
                row['bet_implied_prob'] = american_to_implied_prob(odds)

                if 'home_wins' in game and not pd.isna(game['home_wins']):
                    row['bet_won'] = int(
                        (game['home_wins'] == 1) == (side == 'home')
                    )

                # Pitcher features for the bet
                row['dog_pitcher_rank'] = game.get(
                    f'{side}_pitcher_rank',
                    game.get(f'{side.replace("home","away").replace("away","home")}_pitcher_rank', 3)
                )
                row['fav_pitcher_rank'] = game.get(
                    f'{opp_side}_pitcher_rank', 3
                )

                # Prefix dog/fav features for strategy engine
                for window in ROLLING_WINDOWS:
                    row[f'dog_win_pct_{window}d'] = game.get(
                        f'{side}_win_pct_{window}d', 0.5
                    )
                    row[f'fav_win_pct_{window}d'] = game.get(
                        f'{opp_side}_win_pct_{window}d', 0.5
                    )
                    row[f'dog_dog_win_pct_{window}d'] = game.get(
                        f'{side}_dog_win_pct_{window}d', 0.0
                    )
                    row[f'dog_streak_{window}d'] = game.get(
                        f'{side}_streak_{window}d', 0
                    )
                    row[f'fav_loss_streak_{window}d'] = game.get(
                        f'{opp_side}_loss_streak_{window}d', 0
                    )
                    row[f'dog_avg_odds_{window}d'] = game.get(
                        f'{side}_avg_odds_{window}d', 0
                    )
                    row[f'dog_home_win_pct_{window}d'] = game.get(
                        f'{side}_home_win_pct_{window}d', 0.5
                    )

                rows.append(row)

    return pd.DataFrame(rows)
