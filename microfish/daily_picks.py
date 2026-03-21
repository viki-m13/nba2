"""
Microfish Daily Picks Generator
================================
Generates daily MLB +200 bet recommendations using the swarm model.
This is the live/production system that uses Claude API for real analysis.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import (
    THE_ODDS_API_KEY, ANTHROPIC_API_KEY, DATA_DIR, RESULTS_DIR,
    MIN_AMERICAN_ODDS, BACKTEST_SEASONS
)
from data_pipeline import (
    build_features, get_plus200_opportunities, american_to_implied_prob,
    load_all_seasons
)
from prediction_model import MicrofishEnsemble, prepare_ml_features, FEATURE_COLUMNS
from swarm_model import swarm_analyze, format_game_context


def fetch_todays_odds() -> pd.DataFrame:
    """Fetch today's MLB odds from The Odds API."""
    if not THE_ODDS_API_KEY:
        print("No Odds API key configured. Using recent historical data for demo.")
        return _demo_today_odds()

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        'apiKey': THE_ODDS_API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            events = resp.json()
            return _parse_todays_odds(events)
        else:
            print(f"API error: {resp.status_code}")
            return _demo_today_odds()
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return _demo_today_odds()


def _parse_todays_odds(events: list) -> pd.DataFrame:
    """Parse today's odds from The Odds API."""
    rows = []
    for event in events:
        game_date = event.get('commence_time', '')[:10]
        home = event.get('home_team', '')
        away = event.get('away_team', '')

        # Get consensus odds (average across bookmakers)
        home_odds_list = []
        away_odds_list = []

        for bk in event.get('bookmakers', []):
            for market in bk.get('markets', []):
                if market.get('key') == 'h2h':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == home:
                            home_odds_list.append(outcome.get('price', 0))
                        elif outcome.get('name') == away:
                            away_odds_list.append(outcome.get('price', 0))

        if home_odds_list and away_odds_list:
            rows.append({
                'date': pd.Timestamp(game_date),
                'home_team': home,
                'away_team': away,
                'home_odds': round(np.median(home_odds_list)),
                'away_odds': round(np.median(away_odds_list)),
                'home_wins': -1,  # Unknown (future game)
                'season': datetime.now().year,
                'bookmaker': 'consensus',
            })

    return pd.DataFrame(rows)


def _demo_today_odds() -> pd.DataFrame:
    """Generate demo odds for today based on recent historical patterns."""
    today = pd.Timestamp(datetime.now().date())
    np.random.seed(int(today.timestamp()) % 2**31)

    from data_pipeline import MLB_TEAMS, TEAM_STRENGTH

    current_year = datetime.now().year
    strengths = TEAM_STRENGTH.get(current_year, TEAM_STRENGTH[2025])
    teams = list(strengths.keys())

    rows = []
    available = list(range(len(teams)))
    np.random.shuffle(available)
    n_games = min(15, len(available) // 2)

    for i in range(n_games):
        home = teams[available[2 * i]]
        away = teams[available[2 * i + 1]]
        home_str = strengths.get(home, 0.5)
        away_str = strengths.get(away, 0.5)
        home_prob = (home_str + 0.04) / (home_str + 0.04 + away_str)

        from data_pipeline import _prob_to_american
        vig = np.random.uniform(0.03, 0.05)
        noise = np.random.normal(0, 0.02)
        home_odds = round(_prob_to_american(home_prob + vig / 2 + noise))
        away_odds = round(_prob_to_american(1 - home_prob + vig / 2 - noise))

        rows.append({
            'date': today,
            'home_team': home,
            'away_team': away,
            'home_odds': home_odds,
            'away_odds': away_odds,
            'home_wins': -1,
            'season': current_year,
            'bookmaker': 'consensus',
        })

    return pd.DataFrame(rows)


def generate_daily_picks(use_claude_api: bool = True) -> dict:
    """
    Generate today's picks using the full microfish pipeline:
    1. Fetch today's odds
    2. Load historical data for feature computation
    3. Build features using only past data
    4. Run ML model for initial screening
    5. Run swarm analysis (Claude API) on candidates
    6. Output final recommendations
    """
    print("=" * 70)
    print(f"MICROFISH DAILY PICKS - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 70)

    # Load historical data for feature computation
    print("\nLoading historical data for feature computation...")
    historical = load_all_seasons()
    if historical.empty:
        print("ERROR: Could not load historical data")
        return {}

    # Fetch today's games
    print("Fetching today's odds...")
    today_odds = fetch_todays_odds()
    if today_odds.empty:
        print("No games found for today")
        return {'picks': [], 'date': str(datetime.now().date())}

    print(f"Found {len(today_odds)} games today")

    # Combine with historical for feature building
    combined = pd.concat([historical, today_odds], ignore_index=True)
    features = build_features(combined, as_of_date=pd.Timestamp(datetime.now().date()))

    # Get today's +200 opportunities
    today_features = features[features['date'] == features['date'].max()]
    opportunities = get_plus200_opportunities(today_features)

    if opportunities.empty:
        print("No +200 underdog opportunities today")
        return {'picks': [], 'date': str(datetime.now().date())}

    print(f"Found {len(opportunities)} +200 opportunities")

    # Train model on historical data
    print("Training ML model on historical data...")
    hist_features = build_features(historical)
    hist_opps = get_plus200_opportunities(hist_features)

    if not hist_opps.empty:
        X_train, y_train = prepare_ml_features(hist_opps)
        model = MicrofishEnsemble()
        model.fit(X_train, y_train)

        # Screen today's opportunities with ML
        X_today, _ = prepare_ml_features(opportunities)
        ml_probs = model.predict_proba(X_today)
        opportunities['ml_win_prob'] = ml_probs
    else:
        opportunities['ml_win_prob'] = 0.33

    # Run swarm analysis on high-ML-prob candidates
    picks = []
    for _, opp in opportunities.iterrows():
        game = opp.to_dict()

        # ML pre-screen: only send to swarm if ML sees potential
        if game.get('ml_win_prob', 0) < 0.35:
            continue

        print(f"\n  Analyzing: {game['bet_team']} at +{game['bet_odds']}...")

        # Run swarm
        swarm_result = swarm_analyze(game, use_api=use_claude_api and bool(ANTHROPIC_API_KEY))

        if swarm_result['recommendation'] == 'BET':
            pick = {
                'date': str(game.get('date', '')),
                'game': f"{game.get('away_team', '')} @ {game.get('home_team', '')}",
                'bet_team': game.get('bet_team', ''),
                'bet_side': game.get('bet_side', ''),
                'odds': game.get('bet_odds', 0),
                'ml_win_prob': float(game.get('ml_win_prob', 0)),
                'swarm_confidence': swarm_result['confidence'],
                'swarm_edge': swarm_result['edge'],
                'agents_agreeing': swarm_result['agents_agreeing'],
            }
            picks.append(pick)
            print(f"    >>> BET: {pick['bet_team']} +{pick['odds']} "
                  f"(ML: {pick['ml_win_prob']:.1%}, Swarm: {pick['swarm_confidence']:.2f})")
        else:
            print(f"    PASS (Swarm confidence: {swarm_result['confidence']:.2f})")

    result = {
        'date': str(datetime.now().date()),
        'total_games': len(today_odds),
        'plus200_opportunities': len(opportunities),
        'picks': picks,
        'generated_at': datetime.now().isoformat(),
    }

    # Save picks
    os.makedirs(RESULTS_DIR, exist_ok=True)
    picks_file = os.path.join(RESULTS_DIR, f"picks_{datetime.now().strftime('%Y%m%d')}.json")
    with open(picks_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nPicks saved to {picks_file}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {len(picks)} picks for today")
    for p in picks:
        print(f"  {p['bet_team']} +{p['odds']} "
              f"({p['game']}) - Confidence: {p['swarm_confidence']:.2f}")
    print(f"{'=' * 70}")

    return result


if __name__ == '__main__':
    generate_daily_picks(use_claude_api=True)
