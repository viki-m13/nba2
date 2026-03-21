"""
Microfish Daily Picks Generator
=================================
Generates today's MLB +200 underdog bet recommendations
using the pure rules-based strategy engine.

No AI at runtime — uses deterministic rules from strategy_rules.json.
"""

import json
import os
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import (
    THE_ODDS_API_KEY, MIN_AMERICAN_ODDS, RESULTS_DIR, CACHE_DIR
)
from data_pipeline import (
    american_to_implied_prob, american_to_decimal, build_features,
    get_plus200_opportunities
)
from strategy_engine import StrategyEngine


def fetch_todays_odds() -> pd.DataFrame:
    """Fetch today's MLB odds from The Odds API."""
    if not THE_ODDS_API_KEY:
        print("ERROR: THE_ODDS_API_KEY required for live picks")
        return pd.DataFrame()

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        'apiKey': THE_ODDS_API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
    }

    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                events = resp.json()
                return _parse_live_odds(events)
            else:
                print(f"API returned {resp.status_code}")
        except requests.exceptions.RequestException as e:
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
                continue
            print(f"Error fetching odds: {e}")

    return pd.DataFrame()


def _parse_live_odds(events: list) -> pd.DataFrame:
    """Parse live odds API response."""
    rows = []

    for event in events:
        game_date = event.get('commence_time', '')
        if isinstance(game_date, str) and 'T' in game_date:
            game_date = game_date[:10]

        home = event.get('home_team', '')
        away = event.get('away_team', '')
        game_id = event.get('id', '')

        all_home = []
        all_away = []
        for bk in event.get('bookmakers', []):
            for market in bk.get('markets', []):
                if market.get('key') == 'h2h':
                    outcomes = {o['name']: o['price']
                                for o in market.get('outcomes', [])}
                    if home in outcomes and away in outcomes:
                        all_home.append(outcomes[home])
                        all_away.append(outcomes[away])

        if all_home and all_away:
            rows.append({
                'date': pd.Timestamp(game_date),
                'game_id': game_id,
                'home_team': home,
                'away_team': away,
                'home_odds': round(np.median(all_home)),
                'away_odds': round(np.median(all_away)),
                'bookmaker': 'consensus',
                'season': int(game_date[:4]) if game_date else 2026,
            })

    return pd.DataFrame(rows)


def generate_daily_picks():
    """Generate today's picks using the rules engine."""
    print("=" * 70)
    print("MICROFISH DAILY PICKS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Strategy: Pure Rules Engine (no AI/ML)")
    print("=" * 70)

    # Load strategy
    try:
        engine = StrategyEngine()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    print(f"\n{engine.summary()}\n")

    # Fetch today's odds
    print("\nFetching today's MLB odds...")
    today_odds = fetch_todays_odds()

    if today_odds.empty:
        print("No games found for today.")
        return

    print(f"Found {len(today_odds)} games")

    # Load recent historical data for feature computation
    print("Loading recent game history for features...")
    from data_pipeline import load_season_data
    current_year = datetime.now().year
    recent = load_season_data(current_year)

    if not recent.empty:
        combined = pd.concat([recent, today_odds], ignore_index=True)
    else:
        combined = today_odds

    # Build features
    features = build_features(combined)

    # Get +200 opportunities from TODAY only
    today_str = datetime.now().strftime('%Y-%m-%d')
    today_features = features[
        features['date'] == pd.Timestamp(today_str)
    ]

    if today_features.empty:
        print("No games with features found for today.")
        return

    opportunities = get_plus200_opportunities(today_features)
    print(f"\n+200 underdog opportunities today: {len(opportunities)}")

    if opportunities.empty:
        print("No +200 underdogs today.")
        return

    # Evaluate with strategy engine
    evaluated = engine.evaluate_batch(opportunities)
    bets = evaluated[evaluated['recommendation'] == 'BET']

    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS: {len(bets)} BET, {len(evaluated) - len(bets)} PASS")
    print(f"{'='*70}")

    if not bets.empty:
        for _, bet in bets.iterrows():
            print(f"\n  BET: {bet['bet_team']} (+{bet['bet_odds']:.0f})")
            print(f"    vs {bet['opp_team']}")
            print(f"    Implied prob: {bet['bet_implied_prob']:.1%}")
            print(f"    Rules triggered: {bet.get('rules_triggered', 'N/A')}")
            print(f"    Total weight: {bet.get('total_weight', 0)}")
    else:
        print("\n  No bets recommended today.")

    # Save picks
    os.makedirs(RESULTS_DIR, exist_ok=True)
    picks_file = os.path.join(RESULTS_DIR, f'picks_{today_str}.json')
    picks_data = {
        'date': today_str,
        'total_opportunities': len(opportunities),
        'bets': len(bets),
        'picks': [],
    }
    if not bets.empty:
        for _, b in bets.iterrows():
            picks_data['picks'].append({
                'team': b.get('bet_team', ''),
                'opponent': b.get('opp_team', ''),
                'odds': float(b.get('bet_odds', 0)),
                'implied_prob': float(b.get('bet_implied_prob', 0)),
                'rules': b.get('rules_triggered', ''),
                'weight': float(b.get('total_weight', 0)),
            })

    with open(picks_file, 'w') as f:
        json.dump(picks_data, f, indent=2)
    print(f"\n  Picks saved to {picks_file}")
