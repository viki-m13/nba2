"""
Fetch real NBA game data from ESPN - fixed version.
"""

import requests
import time
import json
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def fetch_espn_game_pbp(game_id):
    """Fetch play-by-play from ESPN API."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"

    try:
        resp = requests.get(url, timeout=30)
        return resp.json()
    except Exception as e:
        return None


def fetch_espn_scoreboard(date_str):
    """Fetch scoreboard for a date from ESPN."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"

    try:
        resp = requests.get(url, timeout=30)
        return resp.json()
    except Exception as e:
        return None


def get_espn_games_for_date(date_str):
    """Get list of completed games for a date."""
    data = fetch_espn_scoreboard(date_str)
    if not data:
        return []

    games = []
    for event in data.get('events', []):
        game_id = event.get('id')
        status = event.get('status', {}).get('type', {}).get('completed', False)

        if game_id and status:
            # Get scores from competition
            competition = event.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])

            home_score = 0
            away_score = 0
            home_team = ''
            away_team = ''

            for comp in competitors:
                score = int(comp.get('score', 0))
                team = comp.get('team', {}).get('abbreviation', '')
                if comp.get('homeAway') == 'home':
                    home_score = score
                    home_team = team
                else:
                    away_score = score
                    away_team = team

            games.append({
                'id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'date': date_str,
            })

    return games


def parse_clock_to_mins_remaining(period, clock_str):
    """Convert period and clock to minutes remaining."""
    if period > 4:
        return 0

    try:
        parts = str(clock_str).split(':')
        mins = int(parts[0])
        secs = int(parts[1]) if len(parts) > 1 else 0

        period_time = mins + secs / 60
        remaining_periods = 4 - period
        total_mins = period_time + (remaining_periods * 12)

        return total_mins
    except:
        return None


def extract_scoring_plays(game_data):
    """Extract scoring progression from ESPN game data."""
    plays_raw = game_data.get('plays', [])

    if not plays_raw:
        return []

    plays = []
    for play in plays_raw:
        period = play.get('period', {}).get('number', 0)
        clock = play.get('clock', {}).get('displayValue', '')
        home_score = play.get('homeScore', 0)
        away_score = play.get('awayScore', 0)

        # Only keep plays with valid scores
        if home_score > 0 or away_score > 0:
            plays.append({
                'period': period,
                'clock': clock,
                'home_score': home_score,
                'away_score': away_score,
            })

    return plays


def build_game_states(plays, sample_interval=1.0):
    """Build game states from plays, sampling every minute."""
    if not plays:
        return []

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

        # Record this score
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

    return states


def get_signal(home_score, away_score, home_pts_5min, away_pts_5min, mins_remaining):
    """Check for entry signal."""
    score_diff = home_score - away_score
    lead = abs(score_diff)
    momentum = home_pts_5min - away_pts_5min
    mom = abs(momentum)

    if score_diff == 0 or lead == 0:
        return None

    side = 'home' if score_diff > 0 else 'away'

    # Momentum must align
    if score_diff > 0 and momentum <= 0:
        return None
    if score_diff < 0 and momentum >= 0:
        return None

    # Check all signals
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


def backtest_game(game_info, states):
    """Backtest a single game."""
    if not states:
        return None

    final_home = game_info['home_score']
    final_away = game_info['away_score']

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
                'home_team': game_info['home_team'],
                'away_team': game_info['away_team'],
            }

    return None


def main():
    print("="*70)
    print("REAL NBA GAME VALIDATION - ESPN DATA v2")
    print("="*70)

    # Sample dates
    dates = []
    for month in [11, 12]:
        for day in [5, 10, 15, 20, 25, 28]:
            dates.append(f"2023{month:02d}{day:02d}")
    for month in [1, 2, 3]:
        for day in [5, 10, 15, 20, 25]:
            dates.append(f"2024{month:02d}{day:02d}")

    print(f"Checking {len(dates)} dates...")

    all_games = []
    for date_str in dates:
        games = get_espn_games_for_date(date_str)
        all_games.extend(games)
        if games:
            print(f"{date_str}: {len(games)} games")
        time.sleep(0.2)

    print(f"\nTotal completed games: {len(all_games)}")

    # Debug: show sample game scores
    if all_games:
        print("\nSample games:")
        for g in all_games[:5]:
            print(f"  {g['away_team']} {g['away_score']} @ {g['home_team']} {g['home_score']}")

    # Process games
    results = []
    signal_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
    games_with_data = 0
    games_with_signal = 0

    for i, game_info in enumerate(all_games[:150]):
        if i % 30 == 0:
            print(f"\nProcessing game {i+1}/{min(len(all_games), 150)}...")

        game_data = fetch_espn_game_pbp(game_info['id'])
        time.sleep(0.3)

        if not game_data:
            continue

        plays = extract_scoring_plays(game_data)
        if not plays:
            continue

        games_with_data += 1

        # Debug first game
        if i == 0:
            print(f"\nFirst game plays sample ({len(plays)} total):")
            for p in plays[:5]:
                print(f"  P{p['period']} {p['clock']}: {p['home_score']}-{p['away_score']}")

        states = build_game_states(plays)
        if not states:
            continue

        # Debug first game states
        if i == 0:
            print(f"\nFirst game states sample ({len(states)} total):")
            for s in states[:5]:
                print(f"  {s['mins_remaining']:.1f}min: {s['home_score']}-{s['away_score']} "
                      f"(5min mom: +{s['home_5min']}/{s['away_5min']})")

        result = backtest_game(game_info, states)
        if result:
            games_with_signal += 1
            results.append(result)

            sig = result['signal']
            if result['won']:
                signal_stats[sig]['wins'] += 1
            else:
                signal_stats[sig]['losses'] += 1

    # Results
    print("\n" + "="*70)
    print(f"PROCESSING SUMMARY")
    print("="*70)
    print(f"Games with play data: {games_with_data}")
    print(f"Games with signals: {games_with_signal}")

    print("\n" + "="*70)
    print("RESULTS BY SIGNAL (REAL NBA GAMES)")
    print("="*70)

    total_wins = 0
    total_trades = 0

    print(f"\n{'Signal':<20} {'Wins':<6} {'Losses':<8} {'Total':<7} {'Win Rate':<10}")
    print("-"*60)

    for sig in sorted(signal_stats.keys()):
        stats = signal_stats[sig]
        wins = stats['wins']
        losses = stats['losses']
        total = wins + losses
        wr = wins / total if total > 0 else 0

        total_wins += wins
        total_trades += total

        marker = " *** RISKY" if wr < 0.90 else ""
        print(f"{sig:<20} {wins:<6} {losses:<8} {total:<7} {wr:.1%}{marker}")

    if total_trades > 0:
        print("-"*60)
        overall_wr = total_wins / total_trades
        print(f"{'TOTAL':<20} {total_wins:<6} {total_trades - total_wins:<8} {total_trades:<7} {overall_wr:.1%}")

    # Show losses
    losses = [r for r in results if not r['won']]
    if losses:
        print("\n" + "="*70)
        print("ALL LOSSES (for analysis)")
        print("="*70)

        for loss in losses:
            loser = loss['home_team'] if loss['side'] == 'home' else loss['away_team']
            winner_team = loss['away_team'] if loss['side'] == 'home' else loss['home_team']
            print(f"{loss['signal']:<18} L={loss['lead']:>2} M={loss['momentum']:>2} "
                  f"@{loss['mins_remaining']:>5.1f}min | "
                  f"Bet {loser}, lost to {winner_team} "
                  f"(Final: {loss['final_away']}-{loss['final_home']})")


if __name__ == "__main__":
    main()
