"""
Fetch real NBA game data from multiple sources.
Try basketball-reference, then ESPN, then nba_api as fallback.
"""

import requests
import time
import json
import re
from pathlib import Path
from datetime import datetime, timedelta

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def fetch_espn_game_pbp(game_id):
    """Fetch play-by-play from ESPN API."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"

    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        return data
    except Exception as e:
        print(f"ESPN error for {game_id}: {e}")
        return None


def fetch_espn_scoreboard(date_str):
    """Fetch scoreboard for a date from ESPN. Format: YYYYMMDD"""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"

    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        return data
    except Exception as e:
        print(f"ESPN scoreboard error: {e}")
        return None


def get_espn_games_for_date(date_str):
    """Get list of game IDs for a date."""
    data = fetch_espn_scoreboard(date_str)
    if not data:
        return []

    games = []
    for event in data.get('events', []):
        game_id = event.get('id')
        status = event.get('status', {}).get('type', {}).get('completed', False)
        if game_id and status:
            games.append({
                'id': game_id,
                'name': event.get('name', ''),
                'date': date_str,
            })

    return games


def extract_espn_game_data(game_data):
    """Extract scores and plays from ESPN game data."""
    if not game_data:
        return None

    boxscore = game_data.get('boxscore', {})
    teams = boxscore.get('teams', [])

    if len(teams) < 2:
        return None

    # Get team info
    away_team = teams[0].get('team', {}).get('abbreviation', 'AWAY')
    home_team = teams[1].get('team', {}).get('abbreviation', 'HOME')

    # Get final scores from boxscore
    away_score = 0
    home_score = 0

    for team in teams:
        stats = team.get('statistics', [])
        for stat in stats:
            if stat.get('name') == 'points':
                if team.get('homeAway') == 'away':
                    away_score = int(stat.get('displayValue', 0))
                else:
                    home_score = int(stat.get('displayValue', 0))

    # Try to get play-by-play
    plays_data = game_data.get('plays', [])

    plays = []
    for play in plays_data:
        period = play.get('period', {}).get('number', 0)
        clock = play.get('clock', {}).get('displayValue', '')

        # Get score at this point
        away_pts = play.get('awayScore', 0)
        home_pts = play.get('homeScore', 0)

        plays.append({
            'period': period,
            'clock': clock,
            'home_score': home_pts,
            'away_score': away_pts,
        })

    return {
        'home_team': home_team,
        'away_team': away_team,
        'final_home': home_score,
        'final_away': away_score,
        'plays': plays,
    }


def parse_clock_to_mins_remaining(period, clock_str):
    """Convert period and clock to minutes remaining."""
    if period > 4:
        return 0  # Overtime

    try:
        # Clock format: "5:30" or "0:45"
        parts = clock_str.split(':')
        mins = int(parts[0])
        secs = int(parts[1]) if len(parts) > 1 else 0

        period_time = mins + secs / 60
        remaining_periods = 4 - period
        total_mins = period_time + (remaining_periods * 12)

        return total_mins
    except:
        return None


def process_espn_game(game_data):
    """Process ESPN game data into states for backtesting."""
    extracted = extract_espn_game_data(game_data)
    if not extracted:
        return None

    plays = extracted['plays']
    if len(plays) < 20:
        return None

    states = []
    score_history = []

    for play in plays:
        period = play['period']
        clock = play['clock']
        home_score = play['home_score']
        away_score = play['away_score']

        if period > 4:
            continue

        mins_remaining = parse_clock_to_mins_remaining(period, clock)
        if mins_remaining is None:
            continue

        score_history.append((mins_remaining, home_score, away_score))

        # Calculate 5-minute momentum
        home_5min = 0
        away_5min = 0

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

    return {
        'home_team': extracted['home_team'],
        'away_team': extracted['away_team'],
        'final_home': extracted['final_home'],
        'final_away': extracted['final_away'],
        'states': states,
    }


def get_signal(home_score, away_score, home_pts_5min, away_pts_5min, mins_remaining):
    """Check for entry signal - ORIGINAL thresholds for testing."""
    score_diff = home_score - away_score
    lead = abs(score_diff)
    momentum = home_pts_5min - away_pts_5min
    mom = abs(momentum)

    if score_diff == 0:
        return None

    side = 'home' if score_diff > 0 else 'away'

    if score_diff > 0 and momentum <= 0:
        return None
    if score_diff < 0 and momentum >= 0:
        return None

    # ALL ORIGINAL SIGNALS
    if 30 <= mins_remaining <= 36:
        if lead >= 18 and mom >= 3:
            return (side, 'Q2_selective', lead, mom)
        if lead >= 15 and mom >= 3:
            return (side, 'Q2_early', lead, mom)

    if 24 <= mins_remaining <= 30:
        if lead >= 18 and mom >= 5:
            return (side, 'late_Q2', lead, mom)
        if lead >= 15 and mom >= 7:
            return (side, 'late_Q2_alt', lead, mom)

    if 18 <= mins_remaining <= 24:
        if lead >= 20 and mom >= 3:
            return (side, 'halftime_dominant', lead, mom)
        if lead >= 18 and mom >= 7:
            return (side, 'halftime', lead, mom)
        if lead >= 15 and mom >= 10:
            return (side, 'halftime_momentum', lead, mom)

    if 15 <= mins_remaining <= 20:
        if lead >= 18 and mom >= 5:
            return (side, 'mid_Q3', lead, mom)
        if lead >= 15 and mom >= 7:
            return (side, 'mid_Q3_alt', lead, mom)

    if 12 <= mins_remaining <= 18:
        if lead >= 15 and mom >= 7:
            return (side, 'late_Q3', lead, mom)
        if lead >= 12 and mom >= 10:
            return (side, 'late_Q3_momentum', lead, mom)

    if 8 <= mins_remaining <= 12:
        if lead >= 10 and mom >= 5:
            return (side, 'early_Q4', lead, mom)
        if lead >= 7 and mom >= 7:
            return (side, 'early_Q4_alt', lead, mom)

    if 2 <= mins_remaining <= 8:
        if lead >= 7 and mom >= 3:
            return (side, 'final', lead, mom)
        if lead >= 5 and mom >= 5:
            return (side, 'final_alt', lead, mom)

    return None


def backtest_game(game_processed):
    """Backtest a single processed game."""
    if not game_processed:
        return None

    states = game_processed['states']
    final_home = game_processed['final_home']
    final_away = game_processed['final_away']

    if final_home == final_away or final_home == 0:
        return None

    winner = 'home' if final_home > final_away else 'away'

    for state in states:
        signal = get_signal(
            state['home_score'],
            state['away_score'],
            state['home_5min'],
            state['away_5min'],
            state['mins_remaining']
        )

        if signal:
            side, signal_name, lead, mom = signal
            won = (side == winner)

            return {
                'signal': signal_name,
                'side': side,
                'lead': lead,
                'momentum': mom,
                'mins_remaining': state['mins_remaining'],
                'won': won,
                'final_home': final_home,
                'final_away': final_away,
                'home_team': game_processed['home_team'],
                'away_team': game_processed['away_team'],
            }

    return None  # No signal triggered


def main():
    print("="*70)
    print("REAL NBA GAME VALIDATION - ESPN DATA")
    print("="*70)

    # Generate dates for last few seasons
    # 2023-24 season: Oct 2023 - Apr 2024
    # 2022-23 season: Oct 2022 - Apr 2023

    dates = []

    # Sample dates from 2023-24 season
    for month in [10, 11, 12]:  # Oct-Dec 2023
        for day in [5, 10, 15, 20, 25]:
            dates.append(f"2023{month:02d}{day:02d}")

    for month in [1, 2, 3]:  # Jan-Mar 2024
        for day in [5, 10, 15, 20, 25]:
            dates.append(f"2024{month:02d}{day:02d}")

    # Sample dates from 2022-23 season
    for month in [11, 12]:  # Nov-Dec 2022
        for day in [5, 15, 25]:
            dates.append(f"2022{month:02d}{day:02d}")

    for month in [1, 2, 3]:  # Jan-Mar 2023
        for day in [5, 15, 25]:
            dates.append(f"2023{month:02d}{day:02d}")

    print(f"Checking {len(dates)} dates...")

    all_games = []
    for date_str in dates:
        games = get_espn_games_for_date(date_str)
        all_games.extend(games)
        if games:
            print(f"{date_str}: {len(games)} games")
        time.sleep(0.3)

    print(f"\nTotal games found: {len(all_games)}")

    if len(all_games) == 0:
        print("No games found!")
        return

    # Process games and backtest
    results = []
    signal_stats = {}

    for i, game_info in enumerate(all_games[:200]):  # Limit to 200 games
        if i % 20 == 0:
            print(f"Processing game {i+1}/{min(len(all_games), 200)}...")

        game_data = fetch_espn_game_pbp(game_info['id'])
        time.sleep(0.5)

        if not game_data:
            continue

        processed = process_espn_game(game_data)
        if not processed:
            continue

        result = backtest_game(processed)
        if result:
            results.append(result)

            sig = result['signal']
            if sig not in signal_stats:
                signal_stats[sig] = {'wins': 0, 'losses': 0}

            if result['won']:
                signal_stats[sig]['wins'] += 1
            else:
                signal_stats[sig]['losses'] += 1

    # Print results
    print("\n" + "="*70)
    print("RESULTS BY SIGNAL (REAL NBA GAMES)")
    print("="*70)

    total_wins = 0
    total_trades = 0

    print(f"\n{'Signal':<20} {'Wins':<6} {'Losses':<8} {'Total':<7} {'Win Rate':<10}")
    print("-"*55)

    for signal_name in sorted(signal_stats.keys()):
        stats = signal_stats[signal_name]
        wins = stats['wins']
        losses = stats['losses']
        total = wins + losses
        wr = wins / total if total > 0 else 0

        total_wins += wins
        total_trades += total

        marker = " ***FAIL" if wr < 0.85 else ""
        print(f"{signal_name:<20} {wins:<6} {losses:<8} {total:<7} {wr:.1%}{marker}")

    print("-"*55)
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    print(f"{'TOTAL':<20} {total_wins:<6} {total_trades - total_wins:<8} {total_trades:<7} {overall_wr:.1%}")

    # Show all losses
    print("\n" + "="*70)
    print("ALL LOSSES")
    print("="*70)

    losses = [r for r in results if not r['won']]
    for loss in losses:
        final_diff = abs(loss['final_home'] - loss['final_away'])
        loser = loss['home_team'] if loss['side'] == 'home' else loss['away_team']
        winner_team = loss['away_team'] if loss['side'] == 'home' else loss['home_team']
        print(f"{loss['signal']:<18} L={loss['lead']:>2} M={loss['momentum']:>2} "
              f"@{loss['mins_remaining']:>4.1f}min | "
              f"Bet {loser}, lost to {winner_team} ({loss['final_home']}-{loss['final_away']})")

    # Save results
    import pickle
    with open('cache/real_game_results.pkl', 'wb') as f:
        pickle.dump({'results': results, 'signal_stats': signal_stats}, f)

    print(f"\nResults saved to cache/real_game_results.pkl")


if __name__ == "__main__":
    main()
