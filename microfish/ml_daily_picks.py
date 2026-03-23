"""
Microfish Moneyline Daily Picks Generator
============================================
Generates today's Polymarket limit orders using the Certainty Floor strategy.

No AI at runtime — uses deterministic rules from ml_strategy_rules.json.

Output: For each MLB game today, determines whether to place a
LIMIT BUY order on Polymarket and at what price.
"""

import json
import os
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ml_config import (
    THE_ODDS_API_KEY, RESULTS_DIR, CACHE_DIR, POLYMARKET_FEE_RATE
)
from ml_data_pipeline import (
    american_to_implied_prob, american_to_decimal, build_features
)
from ml_strategy_engine import MLStrategyEngine


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
    """Parse live odds API response into game-level data."""
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
            home_ml = round(np.median(all_home))
            away_ml = round(np.median(all_away))

            # Determine favorite
            if home_ml < away_ml:
                fav_side = 'home'
            elif away_ml < home_ml:
                fav_side = 'away'
            else:
                fav_side = 'home'

            fav_ml = home_ml if fav_side == 'home' else away_ml
            dog_ml = away_ml if fav_side == 'home' else home_ml
            fav_team = home if fav_side == 'home' else away
            dog_team = away if fav_side == 'home' else home

            fav_implied = american_to_implied_prob(fav_ml)
            dog_implied = american_to_implied_prob(dog_ml)
            total_implied = fav_implied + dog_implied
            fav_prob_novig = fav_implied / total_implied

            rows.append({
                'date': pd.Timestamp(game_date),
                'game_id': game_id,
                'home_team': home,
                'away_team': away,
                'home_ml': home_ml,
                'away_ml': away_ml,
                'favorite': fav_side,
                'fav_team': fav_team,
                'dog_team': dog_team,
                'fav_ml': fav_ml,
                'dog_ml': dog_ml,
                'fav_implied_prob': fav_implied,
                'dog_implied_prob': dog_implied,
                'fav_prob_novig': fav_prob_novig,
                'season': int(game_date[:4]) if game_date else 2026,
            })

    return pd.DataFrame(rows)


def generate_daily_picks():
    """Generate today's Polymarket limit orders."""
    print("=" * 70)
    print("MICROFISH MONEYLINE DAILY PICKS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Strategy: Certainty Floor Limit-Price (Polymarket)")
    print("=" * 70)

    # Load strategy
    try:
        engine = MLStrategyEngine()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return []

    print(f"\n{engine.summary()}\n")

    # Fetch today's odds
    print("\nFetching today's MLB odds...")
    today_odds = fetch_todays_odds()

    if today_odds.empty:
        print("No games found for today.")
        return []

    print(f"Found {len(today_odds)} games")

    # Evaluate each game
    evaluated = engine.evaluate_batch(today_odds)

    limit_buys = evaluated[evaluated['action'] == 'LIMIT_BUY']
    skips = evaluated[evaluated['action'] == 'SKIP']

    print(f"\n{'='*70}")
    print(f"POLYMARKET LIMIT ORDERS: {len(limit_buys)} LIMIT_BUY, {len(skips)} SKIP")
    print(f"{'='*70}")

    orders = []
    if not limit_buys.empty:
        for _, bet in limit_buys.iterrows():
            price = bet.get('limit_price', 0)
            profit = 1.0 - price
            profit_after_fee = profit - POLYMARKET_FEE_RATE

            print(f"\n  LIMIT BUY: {bet['fav_team']}")
            print(f"    vs {bet['dog_team']}")
            print(f"    Current market: ~{bet.get('fav_prob_novig', 0):.0%}")
            print(f"    Sportsbook ML: {bet.get('fav_ml', '?')}")
            print(f"    Limit price: ${price:.2f} ({price:.0%})")
            print(f"    Profit if win: ${profit_after_fee:.2f}/share (after fees)")
            print(f"    Floor used: {bet.get('floor_used', '?')}")

            orders.append({
                'team': bet.get('fav_team', ''),
                'opponent': bet.get('dog_team', ''),
                'fav_ml': int(bet.get('fav_ml', 0)),
                'current_prob': round(float(bet.get('fav_prob_novig', 0)), 3),
                'limit_price': round(price, 2),
                'profit_per_share': round(profit_after_fee, 4),
                'floor_used': bet.get('floor_used', ''),
            })
    else:
        print("\n  No limit orders today — no games reach the Certainty Floor.")
        print(f"  Floor required: {engine.base_floor:.0%}")
        if not today_odds.empty:
            max_prob = today_odds['fav_prob_novig'].max()
            print(f"  Highest implied prob today: {max_prob:.1%}")

    # Also show games that are CLOSE to the floor
    if not skips.empty and 'current_prob' in skips.columns:
        near_floor = skips[skips['current_prob'] >= engine.base_floor - 0.10]
        if not near_floor.empty:
            print(f"\n  NEAR MISSES (within 10pp of floor):")
            for _, g in near_floor.iterrows():
                print(f"    {g.get('fav_team', '?')} vs {g.get('dog_team', '?')}: "
                      f"{g.get('current_prob', 0):.0%} (need {engine.base_floor:.0%})")

    # Save picks
    os.makedirs(RESULTS_DIR, exist_ok=True)
    today_str = datetime.now().strftime('%Y-%m-%d')
    picks_file = os.path.join(RESULTS_DIR, f'picks_{today_str}.json')
    picks_data = {
        'date': today_str,
        'strategy': 'Certainty Floor Limit-Price',
        'floor_pct': engine.base_floor * 100,
        'total_games': len(today_odds),
        'limit_orders': len(orders),
        'orders': orders,
    }

    with open(picks_file, 'w') as f:
        json.dump(picks_data, f, indent=2)
    print(f"\n  Picks saved to {picks_file}")

    return orders
