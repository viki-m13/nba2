"""
MiroFish SGP Data Pipeline
============================
Fetches MLB player props, game odds, and box score outcomes.
Builds correlation matrices and feature sets for SGP discovery.

All features use ONLY past data relative to game date.
NO synthetic data. NO forward leakage.

Data sources:
1. The Odds API — player props + game odds (live & historical)
2. Cached CSV/JSON files from prior fetches
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from sgp_config import (
    THE_ODDS_API_KEY, SGP_DATA_DIR, SGP_CACHE_DIR,
    PROP_MARKETS, GAME_MARKETS, FEATURE_LAG_DAYS,
    ROLLING_WINDOWS, BACKTEST_SEASONS
)


def american_to_decimal(american: float) -> float:
    if american > 0:
        return (american / 100) + 1
    elif american < 0:
        return (100 / abs(american)) + 1
    return 1.0


def american_to_implied_prob(american: float) -> float:
    if american > 0:
        return 100 / (american + 100)
    elif american < 0:
        return abs(american) / (abs(american) + 100)
    return 0.5


def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    elif decimal_odds > 1.0:
        return round(-100 / (decimal_odds - 1))
    return 0


def parlay_odds(legs: list) -> float:
    """Calculate combined parlay decimal odds from list of decimal odds."""
    combined = 1.0
    for leg in legs:
        combined *= leg
    return combined


def parlay_american(legs_american: list) -> float:
    """Calculate combined parlay odds from list of American odds."""
    decimals = [american_to_decimal(a) for a in legs_american]
    combined = parlay_odds(decimals)
    return decimal_to_american(combined)


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_player_props(game_date: str, sport: str = 'baseball_mlb') -> list:
    """
    Fetch player props from The Odds API for a specific date.
    Returns raw API response data.
    """
    cache_file = os.path.join(SGP_CACHE_DIR, f'props_{game_date}.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)

    if not THE_ODDS_API_KEY:
        return []

    all_props = []
    for market in PROP_MARKETS:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {
            'apiKey': THE_ODDS_API_KEY,
            'regions': 'us',
            'markets': market,
            'oddsFormat': 'american',
        }
        for attempt in range(4):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    for event in data:
                        event['_market_type'] = market
                        event['_fetch_date'] = game_date
                    all_props.extend(data)
                    break
                elif resp.status_code == 422:
                    break
            except requests.exceptions.RequestException:
                if attempt < 3:
                    time.sleep(2 ** (attempt + 1))
        time.sleep(0.5)  # Rate limit

    if all_props:
        os.makedirs(SGP_CACHE_DIR, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(all_props, f)

    return all_props


def fetch_game_odds(game_date: str, sport: str = 'baseball_mlb') -> list:
    """Fetch game-level odds (moneyline, totals, spreads)."""
    cache_file = os.path.join(SGP_CACHE_DIR, f'games_{game_date}.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)

    if not THE_ODDS_API_KEY:
        return []

    all_games = []
    for market in GAME_MARKETS:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {
            'apiKey': THE_ODDS_API_KEY,
            'regions': 'us',
            'markets': market,
            'oddsFormat': 'american',
        }
        for attempt in range(4):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    for event in data:
                        event['_market_type'] = market
                    all_games.extend(data)
                    break
            except requests.exceptions.RequestException:
                if attempt < 3:
                    time.sleep(2 ** (attempt + 1))
        time.sleep(0.5)

    if all_games:
        os.makedirs(SGP_CACHE_DIR, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(all_games, f)

    return all_games


# ============================================================================
# DATA PARSING & STRUCTURING
# ============================================================================

def parse_props_to_dataframe(raw_props: list) -> pd.DataFrame:
    """Parse raw props API data into structured DataFrame."""
    rows = []
    for event in raw_props:
        game_id = event.get('id', '')
        home = event.get('home_team', '')
        away = event.get('away_team', '')
        commence = event.get('commence_time', '')
        game_date = commence[:10] if commence else ''
        market_type = event.get('_market_type', '')

        for bookmaker in event.get('bookmakers', []):
            book_name = bookmaker.get('key', '')
            for market in bookmaker.get('markets', []):
                market_key = market.get('key', market_type)
                for outcome in market.get('outcomes', []):
                    player = outcome.get('description', '')
                    name = outcome.get('name', '')  # Over/Under
                    price = outcome.get('price', 0)
                    point = outcome.get('point', None)

                    rows.append({
                        'game_id': game_id,
                        'game_date': game_date,
                        'home_team': home,
                        'away_team': away,
                        'bookmaker': book_name,
                        'market': market_key,
                        'player': player,
                        'bet_type': name,  # Over/Under
                        'line': point,
                        'odds': price,
                        'implied_prob': american_to_implied_prob(price),
                    })

    return pd.DataFrame(rows)


def parse_games_to_dataframe(raw_games: list) -> pd.DataFrame:
    """Parse raw game odds into structured DataFrame."""
    rows = []
    for event in raw_games:
        game_id = event.get('id', '')
        home = event.get('home_team', '')
        away = event.get('away_team', '')
        commence = event.get('commence_time', '')
        game_date = commence[:10] if commence else ''
        market_type = event.get('_market_type', '')

        for bookmaker in event.get('bookmakers', []):
            book_name = bookmaker.get('key', '')
            for market in bookmaker.get('markets', []):
                market_key = market.get('key', market_type)
                for outcome in market.get('outcomes', []):
                    name = outcome.get('name', '')
                    price = outcome.get('price', 0)
                    point = outcome.get('point', None)

                    rows.append({
                        'game_id': game_id,
                        'game_date': game_date,
                        'home_team': home,
                        'away_team': away,
                        'bookmaker': book_name,
                        'market': market_key,
                        'outcome': name,
                        'line': point,
                        'odds': price,
                        'implied_prob': american_to_implied_prob(price),
                    })

    return pd.DataFrame(rows)


# ============================================================================
# HISTORICAL DATA LOADING (for backtesting)
# ============================================================================

def load_historical_props(season: int) -> pd.DataFrame:
    """Load historical player prop results for backtesting."""
    csv_file = os.path.join(SGP_CACHE_DIR, f'mlb_props_{season}.csv')
    if os.path.exists(csv_file):
        print(f"  Loading cached props data for {season}")
        df = pd.read_csv(csv_file)
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        return df

    json_file = os.path.join(SGP_CACHE_DIR, f'mlb_props_{season}.json')
    if os.path.exists(json_file):
        print(f"  Loading cached JSON props for {season}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        return parse_props_to_dataframe(data)

    print(f"  No props data for {season}. Cache files needed.")
    return pd.DataFrame()


def load_historical_results(season: int) -> pd.DataFrame:
    """Load historical game/player results for outcome verification."""
    csv_file = os.path.join(SGP_CACHE_DIR, f'mlb_results_{season}.csv')
    if os.path.exists(csv_file):
        print(f"  Loading cached results for {season}")
        return pd.read_csv(csv_file, parse_dates=['game_date'])

    # Try the existing moneyline cache as fallback for game outcomes
    ml_cache = os.path.join(SGP_CACHE_DIR, '..', 'cache',
                            f'mlb_historical_{season}.csv')
    if os.path.exists(ml_cache):
        print(f"  Loading moneyline cache as fallback for {season}")
        return pd.read_csv(ml_cache, parse_dates=['date'])

    return pd.DataFrame()


def load_all_sgp_data() -> dict:
    """Load all available SGP data across seasons."""
    data = {'props': [], 'results': []}

    for season in BACKTEST_SEASONS:
        props = load_historical_props(season)
        if not props.empty:
            data['props'].append(props)

        results = load_historical_results(season)
        if not results.empty:
            data['results'].append(results)

    if data['props']:
        data['props'] = pd.concat(data['props'], ignore_index=True)
    else:
        data['props'] = pd.DataFrame()

    if data['results']:
        data['results'] = pd.concat(data['results'], ignore_index=True)
    else:
        data['results'] = pd.DataFrame()

    return data


# ============================================================================
# FEATURE ENGINEERING FOR SGP
# ============================================================================

def build_sgp_features(props_df: pd.DataFrame,
                       results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build SGP-specific features from historical prop and result data.
    All features use ONLY past data (lagged by FEATURE_LAG_DAYS).

    Features include:
    - Player prop hit rates (rolling windows)
    - Prop correlation matrices (which props hit together)
    - Line movement signals
    - Team scoring environment features
    - Pitcher matchup features
    """
    if props_df.empty:
        return pd.DataFrame()

    features_list = []

    # Group by game
    if 'game_date' not in props_df.columns:
        return pd.DataFrame()

    props_df = props_df.sort_values('game_date').copy()

    # Build player history lookup (all past data only)
    player_history = defaultdict(list)

    # Track prop outcomes if we have results
    has_outcomes = 'hit' in props_df.columns

    for game_date in sorted(props_df['game_date'].unique()):
        cutoff = game_date - timedelta(days=FEATURE_LAG_DAYS) if isinstance(
            game_date, (datetime, pd.Timestamp)
        ) else game_date

        game_props = props_df[props_df['game_date'] == game_date]

        for game_id in game_props['game_id'].unique():
            game = game_props[game_props['game_id'] == game_id]
            if game.empty:
                continue

            home_team = game.iloc[0].get('home_team', '')
            away_team = game.iloc[0].get('away_team', '')

            game_features = {
                'game_id': game_id,
                'game_date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'n_props_available': len(game),
                'n_unique_players': game['player'].nunique()
                    if 'player' in game.columns else 0,
                'n_markets': game['market'].nunique()
                    if 'market' in game.columns else 0,
            }

            # Consensus odds analysis per prop
            if 'odds' in game.columns:
                over_props = game[game['bet_type'] == 'Over'] if 'bet_type' in game.columns else game
                if not over_props.empty:
                    game_features['avg_over_odds'] = over_props['odds'].mean()
                    game_features['avg_over_implied'] = over_props['implied_prob'].mean()
                    game_features['n_plus_odds_props'] = (over_props['odds'] > 0).sum()
                    game_features['n_heavy_fav_props'] = (over_props['odds'] < -200).sum()

                    # Juice analysis — how much vig the book is charging
                    game_features['avg_vig'] = over_props['implied_prob'].mean()

            # Player-level rolling features (using history before cutoff)
            player_features = {}
            for _, prop in game.iterrows():
                player = prop.get('player', '')
                market = prop.get('market', '')
                if not player:
                    continue

                key = f"{player}_{market}"
                history = [h for h in player_history.get(key, [])
                          if h['date'] < cutoff] if isinstance(cutoff, (datetime, pd.Timestamp)) else []

                for window in ROLLING_WINDOWS:
                    recent = history[-window:] if len(history) >= window else history
                    if len(recent) >= 3:
                        hit_rate = sum(1 for h in recent if h.get('hit', False)) / len(recent)
                        player_features[f'{key}_hit_rate_{window}d'] = hit_rate

            if player_features:
                game_features['avg_player_hit_rate_7d'] = np.mean([
                    v for k, v in player_features.items() if '7d' in k
                ]) if any('7d' in k for k in player_features) else 0.5
                game_features['avg_player_hit_rate_14d'] = np.mean([
                    v for k, v in player_features.items() if '14d' in k
                ]) if any('14d' in k for k in player_features) else 0.5
                game_features['min_player_hit_rate_7d'] = min([
                    v for k, v in player_features.items() if '7d' in k
                ], default=0.5)

            # Store outcome if available
            if has_outcomes:
                game_features['sgp_hit'] = game.get('sgp_hit', 0)

            features_list.append(game_features)

            # Update player histories for future games
            if has_outcomes:
                for _, prop in game.iterrows():
                    player = prop.get('player', '')
                    market = prop.get('market', '')
                    if player:
                        key = f"{player}_{market}"
                        player_history[key].append({
                            'date': game_date,
                            'hit': prop.get('hit', False),
                            'odds': prop.get('odds', 0),
                            'line': prop.get('line', 0),
                        })

    return pd.DataFrame(features_list)


def compute_correlation_matrix(props_df: pd.DataFrame,
                                min_co_occurrences: int = 10) -> dict:
    """
    Compute pairwise correlation between prop outcomes within games.
    This is the core of SGP discovery — finding which legs hit TOGETHER.

    Returns dict of {(prop_a, prop_b): correlation_stats}
    """
    if props_df.empty or 'hit' not in props_df.columns:
        return {}

    correlations = {}
    games = props_df.groupby('game_id')

    for game_id, game in games:
        props_in_game = game[['player', 'market', 'bet_type', 'hit']].drop_duplicates()

        prop_list = list(props_in_game.itertuples())
        for i in range(len(prop_list)):
            for j in range(i + 1, len(prop_list)):
                a = prop_list[i]
                b = prop_list[j]
                key = (
                    f"{a.player}_{a.market}_{a.bet_type}",
                    f"{b.player}_{b.market}_{b.bet_type}"
                )
                if key not in correlations:
                    correlations[key] = {'both_hit': 0, 'a_hit': 0, 'b_hit': 0,
                                         'neither': 0, 'total': 0}
                correlations[key]['total'] += 1
                a_hit = bool(a.hit)
                b_hit = bool(b.hit)
                if a_hit and b_hit:
                    correlations[key]['both_hit'] += 1
                if a_hit:
                    correlations[key]['a_hit'] += 1
                if b_hit:
                    correlations[key]['b_hit'] += 1
                if not a_hit and not b_hit:
                    correlations[key]['neither'] += 1

    # Calculate conditional probabilities and lift
    significant = {}
    for key, stats in correlations.items():
        if stats['total'] < min_co_occurrences:
            continue

        p_a = stats['a_hit'] / stats['total']
        p_b = stats['b_hit'] / stats['total']
        p_both = stats['both_hit'] / stats['total']
        p_independent = p_a * p_b

        # Lift = P(A&B) / P(A)*P(B) — how much more likely they hit together
        lift = p_both / p_independent if p_independent > 0 else 0

        significant[key] = {
            'p_a': round(p_a, 4),
            'p_b': round(p_b, 4),
            'p_both': round(p_both, 4),
            'p_independent': round(p_independent, 4),
            'lift': round(lift, 4),
            'co_occurrences': stats['total'],
            'both_hit_count': stats['both_hit'],
        }

    return significant


def format_sgp_data_summary(props_df: pd.DataFrame,
                             games_df: pd.DataFrame = None,
                             correlations: dict = None) -> str:
    """Create comprehensive data summary for MiroFish agents."""
    lines = []
    lines.append("=== MLB SGP (SAME GAME PARLAY) DATA SUMMARY ===")

    if not props_df.empty:
        lines.append(f"Total prop records: {len(props_df)}")
        if 'game_id' in props_df.columns:
            lines.append(f"Unique games: {props_df['game_id'].nunique()}")
        if 'player' in props_df.columns:
            lines.append(f"Unique players: {props_df['player'].nunique()}")
        if 'market' in props_df.columns:
            lines.append(f"Markets: {props_df['market'].unique().tolist()}")
        lines.append("")

        # Prop hit rates by market
        if 'hit' in props_df.columns and 'market' in props_df.columns:
            lines.append("PROP HIT RATES BY MARKET:")
            for market in sorted(props_df['market'].unique()):
                subset = props_df[props_df['market'] == market]
                if len(subset) >= 10:
                    hit_rate = subset['hit'].mean()
                    lines.append(f"  {market}: {subset['hit'].sum()}/{len(subset)} = {hit_rate:.1%}")
            lines.append("")

        # Odds distribution
        if 'odds' in props_df.columns:
            lines.append("ODDS DISTRIBUTION:")
            for bracket, label in [
                ((-500, -200), "Heavy Fav (-500 to -200)"),
                ((-200, -110), "Moderate Fav (-200 to -110)"),
                ((-110, 110), "Near Even (-110 to +110)"),
                ((110, 200), "Slight Dog (+110 to +200)"),
                ((200, 500), "Moderate Dog (+200 to +500)"),
            ]:
                subset = props_df[(props_df['odds'] >= bracket[0]) & (props_df['odds'] < bracket[1])]
                if len(subset) >= 5 and 'hit' in props_df.columns:
                    hit_rate = subset['hit'].mean()
                    lines.append(f"  {label}: {subset['hit'].sum()}/{len(subset)} = {hit_rate:.1%}")
            lines.append("")

    # Correlation summary
    if correlations:
        lines.append("TOP CORRELATED PROP PAIRS (lift > 1.2):")
        sorted_corrs = sorted(correlations.items(),
                              key=lambda x: x[1]['lift'], reverse=True)
        for (prop_a, prop_b), stats in sorted_corrs[:30]:
            if stats['lift'] > 1.2 and stats['co_occurrences'] >= 15:
                lines.append(
                    f"  {prop_a} + {prop_b}: "
                    f"lift={stats['lift']:.2f}, "
                    f"P(both)={stats['p_both']:.1%} vs "
                    f"P(indep)={stats['p_independent']:.1%}, "
                    f"n={stats['co_occurrences']}"
                )
        lines.append("")

        lines.append("ANTI-CORRELATED PAIRS (lift < 0.8, AVOID):")
        for (prop_a, prop_b), stats in sorted_corrs[-15:]:
            if stats['lift'] < 0.8 and stats['co_occurrences'] >= 15:
                lines.append(
                    f"  {prop_a} + {prop_b}: "
                    f"lift={stats['lift']:.2f}, "
                    f"n={stats['co_occurrences']}"
                )
        lines.append("")

    # SGP construction analysis
    if 'hit' in props_df.columns and 'game_id' in props_df.columns:
        lines.append("SGP FEASIBILITY ANALYSIS:")
        # What % of games have enough high-confidence props for a 3-leg SGP?
        games = props_df.groupby('game_id')
        feasible_3leg = 0
        feasible_4leg = 0
        total_games = 0
        for gid, game in games:
            total_games += 1
            high_conf = game[game['implied_prob'] >= 0.65]
            if len(high_conf) >= 3:
                feasible_3leg += 1
            if len(high_conf) >= 4:
                feasible_4leg += 1
        if total_games > 0:
            lines.append(f"  Games with 3+ high-conf legs: {feasible_3leg}/{total_games} = {feasible_3leg/total_games:.1%}")
            lines.append(f"  Games with 4+ high-conf legs: {feasible_4leg}/{total_games} = {feasible_4leg/total_games:.1%}")

    return '\n'.join(lines)
