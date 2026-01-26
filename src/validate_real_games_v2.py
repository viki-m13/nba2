"""
Validate strategy signals on REAL NBA games.
Uses proper headers and rate limiting to avoid API issues.
"""

import time
import pickle
import os
import requests
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Headers to mimic browser request
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Connection': 'keep-alive',
    'Referer': 'https://stats.nba.com/',
}


def get_games_for_season(season="2020-21", max_games=50):
    """Get completed regular season games."""
    cache_file = CACHE_DIR / f"games_{season}_v2.pkl"

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Fetching games for {season}...")
    url = "https://stats.nba.com/stats/leaguegamefinder"
    params = {
        "LeagueID": "00",
        "Season": season,
        "SeasonType": "Regular Season",
    }

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=60)
        data = resp.json()
        rows = data['resultSets'][0]['rowSet']
        headers = data['resultSets'][0]['headers']
        game_id_idx = headers.index('GAME_ID')

        game_ids = list(set(row[game_id_idx] for row in rows))[:max_games]

        with open(cache_file, 'wb') as f:
            pickle.dump(game_ids, f)

        return game_ids
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_play_by_play(game_id, retries=3):
    """Get play-by-play data for a game."""
    cache_file = CACHE_DIR / f"pbp_{game_id}_v2.pkl"

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    url = "https://stats.nba.com/stats/playbyplayv2"
    params = {
        "GameID": game_id,
        "StartPeriod": 0,
        "EndPeriod": 10,
    }

    for attempt in range(retries):
        try:
            time.sleep(1.0)  # Rate limit
            resp = requests.get(url, headers=HEADERS, params=params, timeout=60)
            data = resp.json()

            rows = data['resultSets'][0]['rowSet']
            headers_list = data['resultSets'][0]['headers']

            # Convert to list of dicts
            plays = []
            for row in rows:
                play = {h: row[i] for i, h in enumerate(headers_list)}
                plays.append(play)

            with open(cache_file, 'wb') as f:
                pickle.dump(plays, f)

            return plays
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {game_id}: {e}")
            time.sleep(2 ** attempt)

    return None


def parse_time_to_mins_remaining(period, pctimestring):
    """Convert period and time string to minutes remaining in regulation."""
    if period > 4:
        return 0

    try:
        parts = str(pctimestring).split(':')
        mins = int(parts[0])
        secs = int(parts[1]) if len(parts) > 1 else 0

        period_time = mins + secs / 60
        remaining_periods = 4 - period
        total_mins = period_time + (remaining_periods * 12)

        return total_mins
    except:
        return None


def extract_game_states(plays):
    """Extract game states with scores and calculate momentum."""
    states = []

    home_score = 0
    away_score = 0
    score_history = []

    for play in plays:
        # Update scores
        if play.get('SCOREHOME') and play.get('SCOREAWAY'):
            try:
                home_score = int(play['SCOREHOME'])
                away_score = int(play['SCOREAWAY'])
            except:
                continue

        period = play.get('PERIOD', 0)
        pctimestring = play.get('PCTIMESTRING', '')

        if period > 4:
            continue

        mins_remaining = parse_time_to_mins_remaining(period, pctimestring)
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

    return states


def get_signal(home_score, away_score, home_pts_5min, away_pts_5min, mins_remaining):
    """Check for entry signal."""
    score_diff = home_score - away_score
    lead = abs(score_diff)
    momentum = home_pts_5min - away_pts_5min
    mom = abs(momentum)

    if score_diff == 0:
        return None

    side = 'home' if score_diff > 0 else 'away'

    # Momentum must align with lead
    if score_diff > 0 and momentum <= 0:
        return None
    if score_diff < 0 and momentum >= 0:
        return None

    # Q2 EARLY: 30-36 min remaining
    if 30 <= mins_remaining <= 36:
        if lead >= 18 and mom >= 3:
            return (side, 'Q2_selective', lead, mom)
        if lead >= 15 and mom >= 3:
            return (side, 'Q2_early', lead, mom)

    # LATE Q2: 24-30 min remaining
    if 24 <= mins_remaining <= 30:
        if lead >= 18 and mom >= 5:
            return (side, 'late_Q2', lead, mom)
        if lead >= 15 and mom >= 7:
            return (side, 'late_Q2_alt', lead, mom)

    # HALFTIME: 18-24 min remaining
    if 18 <= mins_remaining <= 24:
        if lead >= 20 and mom >= 3:
            return (side, 'halftime_dominant', lead, mom)
        if lead >= 18 and mom >= 7:
            return (side, 'halftime', lead, mom)
        if lead >= 15 and mom >= 10:
            return (side, 'halftime_momentum', lead, mom)

    # MID Q3: 15-20 min remaining
    if 15 <= mins_remaining <= 20:
        if lead >= 18 and mom >= 5:
            return (side, 'mid_Q3', lead, mom)
        if lead >= 15 and mom >= 7:
            return (side, 'mid_Q3_alt', lead, mom)

    # LATE Q3: 12-18 min remaining
    if 12 <= mins_remaining <= 18:
        if lead >= 15 and mom >= 7:
            return (side, 'late_Q3', lead, mom)
        if lead >= 12 and mom >= 10:
            return (side, 'late_Q3_momentum', lead, mom)

    # EARLY Q4: 8-12 min remaining
    if 8 <= mins_remaining <= 12:
        if lead >= 10 and mom >= 5:
            return (side, 'early_Q4', lead, mom)
        if lead >= 7 and mom >= 7:
            return (side, 'early_Q4_alt', lead, mom)

    # FINAL: 2-8 min remaining
    if 2 <= mins_remaining <= 8:
        if lead >= 7 and mom >= 3:
            return (side, 'final', lead, mom)
        if lead >= 5 and mom >= 5:
            return (side, 'final_alt', lead, mom)

    return None


def backtest_real_games(game_ids, verbose=True):
    """Backtest strategy on real NBA games."""
    results = []
    signal_stats = {}

    for i, game_id in enumerate(game_ids):
        if verbose:
            print(f"Processing game {i+1}/{len(game_ids)}: {game_id}")

        plays = get_play_by_play(game_id)
        if plays is None or len(plays) == 0:
            continue

        states = extract_game_states(plays)
        if len(states) < 10:
            continue

        # Get final score
        final_state = states[-1]
        final_home = final_state['home_score']
        final_away = final_state['away_score']

        if final_home == final_away:
            continue  # Skip ties (shouldn't happen but just in case)

        winner = 'home' if final_home > final_away else 'away'

        # Check for first signal
        traded = False
        for state in states:
            if traded:
                break

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

                results.append({
                    'game_id': game_id,
                    'signal': signal_name,
                    'side': side,
                    'lead': lead,
                    'momentum': mom,
                    'mins_remaining': state['mins_remaining'],
                    'won': won,
                    'final_home': final_home,
                    'final_away': final_away,
                })

                if signal_name not in signal_stats:
                    signal_stats[signal_name] = {'wins': 0, 'losses': 0}

                if won:
                    signal_stats[signal_name]['wins'] += 1
                else:
                    signal_stats[signal_name]['losses'] += 1

                traded = True

    return results, signal_stats


def main():
    print("="*70)
    print("REAL NBA GAME VALIDATION")
    print("="*70)

    # Get games from older seasons (more stable API)
    all_game_ids = []
    for season in ["2020-21", "2019-20", "2018-19"]:
        try:
            game_ids = get_games_for_season(season, max_games=100)
            all_game_ids.extend(game_ids)
            print(f"Got {len(game_ids)} games from {season}")
            time.sleep(1)
        except Exception as e:
            print(f"Error getting {season}: {e}")

    print(f"\nTotal games to analyze: {len(all_game_ids)}")

    if len(all_game_ids) == 0:
        print("No games to analyze!")
        return

    # Run backtest
    print("\nRunning backtest on real games...")
    results, signal_stats = backtest_real_games(all_game_ids)

    # Print results
    print("\n" + "="*70)
    print("RESULTS BY SIGNAL")
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

        marker = " ***" if wr < 0.90 else ""
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
        print(f"{loss['signal']:<20} L={loss['lead']:>2} M={loss['momentum']:>2} "
              f"@{loss['mins_remaining']:>4.1f}min | "
              f"Final: {loss['final_home']}-{loss['final_away']} (lost by {final_diff})")

    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Games analyzed: {len(all_game_ids)}")
    print(f"Games with trades: {len(results)}")
    print(f"Coverage: {len(results) / len(all_game_ids) * 100:.1f}%")
    print(f"Win rate: {overall_wr:.1%}")
    print(f"Total losses: {len(losses)}")


if __name__ == "__main__":
    main()
