"""
Comprehensive spread validation on 150+ real NBA games.
Find signals with HIGH spread coverage accuracy.
"""

import requests
import time
import json
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def fetch_espn_scoreboard(date_str):
    """Fetch scoreboard for a date."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    try:
        resp = requests.get(url, timeout=30)
        return resp.json()
    except:
        return None


def fetch_espn_game_pbp(game_id):
    """Fetch play-by-play."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    try:
        resp = requests.get(url, timeout=30)
        return resp.json()
    except:
        return None


def get_games_for_date(date_str):
    """Get completed games."""
    data = fetch_espn_scoreboard(date_str)
    if not data:
        return []

    games = []
    for event in data.get('events', []):
        game_id = event.get('id')
        status = event.get('status', {}).get('type', {}).get('completed', False)

        if game_id and status:
            competition = event.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])

            home_score = away_score = 0
            home_team = away_team = ''

            for comp in competitors:
                score = int(comp.get('score', 0))
                team = comp.get('team', {}).get('abbreviation', '')
                if comp.get('homeAway') == 'home':
                    home_score = score
                    home_team = team
                else:
                    away_score = score
                    away_team = team

            if home_score > 0 and away_score > 0:
                games.append({
                    'id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'date': date_str,
                })

    return games


def parse_clock(period, clock_str):
    """Convert to minutes remaining."""
    if period > 4:
        return 0
    try:
        parts = str(clock_str).split(':')
        mins = int(parts[0])
        secs = int(parts[1]) if len(parts) > 1 else 0
        period_time = mins + secs / 60
        remaining_periods = 4 - period
        return period_time + (remaining_periods * 12)
    except:
        return None


def extract_plays(game_data):
    """Extract plays with scores."""
    plays_raw = game_data.get('plays', [])
    plays = []
    for play in plays_raw:
        period = play.get('period', {}).get('number', 0)
        clock = play.get('clock', {}).get('displayValue', '')
        home_score = play.get('homeScore', 0)
        away_score = play.get('awayScore', 0)
        if home_score > 0 or away_score > 0:
            plays.append({
                'period': period,
                'clock': clock,
                'home_score': home_score,
                'away_score': away_score,
            })
    return plays


def build_states(plays):
    """Build game states with momentum."""
    states = []
    score_history = []

    for play in plays:
        period = play['period']
        if period > 4:
            continue

        mins_remaining = parse_clock(period, play['clock'])
        if mins_remaining is None:
            continue

        home_score = play['home_score']
        away_score = play['away_score']
        score_history.append((mins_remaining, home_score, away_score))

        # 5-min momentum
        home_5min = away_5min = 0
        for past_mins, past_home, past_away in reversed(score_history[:-1]):
            if past_mins - mins_remaining >= 5:
                home_5min = home_score - past_home
                away_5min = away_score - past_away
                break

        states.append({
            'mins_remaining': mins_remaining,
            'home_score': home_score,
            'away_score': away_score,
            'home_5min': home_5min,
            'away_5min': away_5min,
        })

    return states


def test_signal(state, min_lead, min_mom, min_mins, max_mins):
    """Test if a specific signal triggers."""
    score_diff = state['home_score'] - state['away_score']
    lead = abs(score_diff)
    momentum = state['home_5min'] - state['away_5min']
    mom = abs(momentum)
    mins = state['mins_remaining']

    if score_diff == 0:
        return None

    side = 'home' if score_diff > 0 else 'away'

    # Momentum must align
    if score_diff > 0 and momentum <= 0:
        return None
    if score_diff < 0 and momentum >= 0:
        return None

    # Check thresholds
    if min_mins <= mins <= max_mins:
        if lead >= min_lead and mom >= min_mom:
            return {'side': side, 'lead': lead, 'mom': mom, 'mins': mins}

    return None


def backtest_signal(games_data, min_lead, min_mom, min_mins, max_mins):
    """Backtest a specific signal configuration."""
    results = []

    for game in games_data:
        states = game['states']
        final_home = game['final_home']
        final_away = game['final_away']
        final_margin = final_home - final_away

        # Find first signal
        for state in states:
            sig = test_signal(state, min_lead, min_mom, min_mins, max_mins)
            if sig:
                # Check spread coverage
                if sig['side'] == 'home':
                    spread_covered = final_margin >= sig['lead']
                    ml_won = final_margin > 0
                else:
                    spread_covered = (-final_margin) >= sig['lead']
                    ml_won = final_margin < 0

                results.append({
                    'spread_covered': spread_covered,
                    'ml_won': ml_won,
                    'lead': sig['lead'],
                    'mom': sig['mom'],
                    'mins': sig['mins'],
                    'side': sig['side'],
                    'final_margin': abs(final_margin),
                    'game': game,
                })
                break  # One trade per game

    return results


def main():
    print("="*70)
    print("COMPREHENSIVE SPREAD VALIDATION - 150+ GAMES")
    print("="*70)

    # Generate many dates
    dates = []

    # 2023-24 season (Oct 2023 - Apr 2024)
    for month in [10, 11, 12]:
        for day in range(1, 29, 2):
            dates.append(f"2023{month:02d}{day:02d}")
    for month in [1, 2, 3, 4]:
        for day in range(1, 29, 2):
            dates.append(f"2024{month:02d}{day:02d}")

    # 2022-23 season
    for month in [11, 12]:
        for day in range(1, 29, 3):
            dates.append(f"2022{month:02d}{day:02d}")
    for month in [1, 2, 3]:
        for day in range(1, 29, 3):
            dates.append(f"2023{month:02d}{day:02d}")

    print(f"Checking {len(dates)} dates...")

    # Fetch all games
    all_games = []
    for i, date_str in enumerate(dates):
        if i % 20 == 0:
            print(f"Fetching date {i+1}/{len(dates)}...")
        games = get_games_for_date(date_str)
        all_games.extend(games)
        time.sleep(0.15)

    print(f"\nTotal games found: {len(all_games)}")

    # Process games - get play-by-play
    print("\nFetching play-by-play data...")
    games_data = []

    for i, game in enumerate(all_games[:300]):  # Process up to 300 games
        if i % 50 == 0:
            print(f"Processing game {i+1}/{min(len(all_games), 300)}...")

        pbp = fetch_espn_game_pbp(game['id'])
        time.sleep(0.2)

        if not pbp:
            continue

        plays = extract_plays(pbp)
        if len(plays) < 20:
            continue

        states = build_states(plays)
        if len(states) < 20:
            continue

        games_data.append({
            'id': game['id'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'final_home': game['home_score'],
            'final_away': game['away_score'],
            'states': states,
            'date': game['date'],
        })

    print(f"\nGames with valid play-by-play: {len(games_data)}")

    # Test many signal configurations
    print("\n" + "="*70)
    print("TESTING SIGNAL CONFIGURATIONS FOR SPREAD COVERAGE")
    print("="*70)

    configs = []

    # Time windows
    windows = [
        (18, 24, 'Halftime'),
        (15, 21, 'Early Q3'),
        (12, 18, 'Mid Q3'),
        (20, 26, 'Late Q2'),
        (16, 22, 'Q2-Q3'),
    ]

    for min_mins, max_mins, window_name in windows:
        for min_lead in [12, 14, 15, 16, 18, 20]:
            for min_mom in [8, 10, 12, 14]:
                results = backtest_signal(games_data, min_lead, min_mom, min_mins, max_mins)

                if len(results) >= 5:  # Need at least 5 samples
                    spread_wins = sum(1 for r in results if r['spread_covered'])
                    ml_wins = sum(1 for r in results if r['ml_won'])
                    total = len(results)

                    configs.append({
                        'window': window_name,
                        'min_mins': min_mins,
                        'max_mins': max_mins,
                        'min_lead': min_lead,
                        'min_mom': min_mom,
                        'total': total,
                        'spread_wins': spread_wins,
                        'spread_wr': spread_wins / total,
                        'ml_wins': ml_wins,
                        'ml_wr': ml_wins / total,
                        'results': results,
                    })

    # Sort by spread win rate
    configs.sort(key=lambda x: (-x['spread_wr'], -x['total']))

    print(f"\n{'Window':<12} {'Lead':<6} {'Mom':<5} {'Trades':<7} {'Spread WR':<10} {'ML WR':<8}")
    print("-"*60)

    for c in configs[:25]:
        marker = " ✓" if c['spread_wr'] >= 0.80 else ""
        print(f"{c['window']:<12} >={c['min_lead']:<4} >={c['min_mom']:<3} {c['total']:<7} "
              f"{c['spread_wr']:.1%}{marker:<4}     {c['ml_wr']:.1%}")

    # Find best configs with 80%+ spread WR
    print("\n" + "="*70)
    print("SIGNALS WITH 80%+ SPREAD WIN RATE")
    print("="*70)

    good_configs = [c for c in configs if c['spread_wr'] >= 0.80 and c['total'] >= 5]

    if good_configs:
        print(f"\n{'Window':<12} {'Time':<10} {'Lead':<6} {'Mom':<5} {'W/L':<8} {'Spread WR':<10}")
        print("-"*55)

        for c in good_configs:
            print(f"{c['window']:<12} {c['min_mins']}-{c['max_mins']}min  >={c['min_lead']:<4} >={c['min_mom']:<3} "
                  f"{c['spread_wins']}/{c['total']:<5} {c['spread_wr']:.1%}")
    else:
        print("\nNo signals found with 80%+ spread win rate.")
        print("Best available:")
        for c in configs[:5]:
            print(f"  {c['window']}: Lead>={c['min_lead']}, Mom>={c['min_mom']} → {c['spread_wr']:.1%} ({c['total']} trades)")

    # Save all results
    save_results = []
    for c in configs:
        for r in c['results']:
            save_results.append({
                'signal': f"{c['window']}_L{c['min_lead']}_M{c['min_mom']}",
                'window': c['window'],
                'min_lead': c['min_lead'],
                'min_mom': c['min_mom'],
                'actual_lead': r['lead'],
                'actual_mom': r['mom'],
                'mins_remaining': r['mins'],
                'side': r['side'],
                'spread_covered': r['spread_covered'],
                'ml_won': r['ml_won'],
                'final_margin': r['final_margin'],
                'home_team': r['game']['home_team'],
                'away_team': r['game']['away_team'],
                'final_home': r['game']['final_home'],
                'final_away': r['game']['final_away'],
                'date': r['game']['date'],
            })

    with open('data/comprehensive_validation.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to data/comprehensive_validation.json ({len(save_results)} records)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total games analyzed: {len(games_data)}")
    print(f"Signal configurations tested: {len(configs)}")

    if good_configs:
        best = good_configs[0]
        print(f"\nBEST SIGNAL FOR SPREAD BETTING:")
        print(f"  Window: {best['window']} ({best['min_mins']}-{best['max_mins']} min)")
        print(f"  Lead: >= {best['min_lead']} points")
        print(f"  Momentum: >= {best['min_mom']} points")
        print(f"  Spread Win Rate: {best['spread_wr']:.1%} ({best['spread_wins']}/{best['total']})")
        print(f"  Moneyline Win Rate: {best['ml_wr']:.1%}")


if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    main()
