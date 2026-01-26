"""
Validate strategy signals on REAL NBA games using nba_api.
"""

import time
import pickle
import os
from pathlib import Path

# Try to import nba_api
try:
    from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2
    from nba_api.stats.static import teams
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False
    print("nba_api not installed. Run: pip install nba_api")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_games_for_season(season="2023-24", max_games=100):
    """Get completed regular season games."""
    cache_file = CACHE_DIR / f"games_{season}.pkl"

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Fetching games for {season}...")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00"
    )
    time.sleep(0.6)

    games_df = finder.get_data_frames()[0]

    # Get unique game IDs (each game appears twice, once per team)
    game_ids = games_df['GAME_ID'].unique()[:max_games]

    with open(cache_file, 'wb') as f:
        pickle.dump(list(game_ids), f)

    return list(game_ids)


def get_play_by_play(game_id):
    """Get play-by-play data for a game."""
    cache_file = CACHE_DIR / f"pbp_{game_id}.pkl"

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    try:
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        time.sleep(0.6)
        df = pbp.get_data_frames()[0]

        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)

        return df
    except Exception as e:
        print(f"Error fetching {game_id}: {e}")
        return None


def parse_time_to_mins_remaining(period, pctimestring):
    """Convert period and time string to minutes remaining in regulation."""
    if period > 4:
        return 0  # Overtime

    try:
        parts = pctimestring.split(':')
        mins = int(parts[0])
        secs = int(parts[1]) if len(parts) > 1 else 0

        # Time left in current period
        period_time = mins + secs / 60

        # Add remaining full periods
        remaining_periods = 4 - period
        total_mins = period_time + (remaining_periods * 12)

        return total_mins
    except:
        return None


def extract_game_states(pbp_df):
    """Extract game states with scores and calculate momentum."""
    states = []

    home_score = 0
    away_score = 0
    score_history = []  # List of (mins_remaining, home_pts, away_pts)

    for _, row in pbp_df.iterrows():
        # Update scores if available
        if row['SCOREHOME'] and row['SCOREAWAY']:
            try:
                home_score = int(row['SCOREHOME'])
                away_score = int(row['SCOREAWAY'])
            except:
                continue

        period = row['PERIOD']
        pctimestring = row['PCTIMESTRING']

        if period > 4:
            continue  # Skip overtime

        mins_remaining = parse_time_to_mins_remaining(period, pctimestring)
        if mins_remaining is None:
            continue

        # Record score at this time
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
        if verbose and i % 10 == 0:
            print(f"Processing game {i+1}/{len(game_ids)}...")

        pbp = get_play_by_play(game_id)
        if pbp is None or len(pbp) == 0:
            continue

        states = extract_game_states(pbp)
        if len(states) < 10:
            continue

        # Get final score
        final_state = states[-1]
        final_home = final_state['home_score']
        final_away = final_state['away_score']
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
    if not HAS_NBA_API:
        print("Cannot run without nba_api. Install with: pip install nba_api")
        return

    print("="*70)
    print("REAL NBA GAME VALIDATION")
    print("="*70)

    # Get games from multiple seasons
    all_game_ids = []
    for season in ["2023-24", "2022-23", "2021-22"]:
        try:
            game_ids = get_games_for_season(season, max_games=150)
            all_game_ids.extend(game_ids)
            print(f"Got {len(game_ids)} games from {season}")
        except Exception as e:
            print(f"Error getting {season}: {e}")

    print(f"\nTotal games to analyze: {len(all_game_ids)}")

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

        print(f"{signal_name:<20} {wins:<6} {losses:<8} {total:<7} {wr:.1%}")

    print("-"*55)
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    print(f"{'TOTAL':<20} {total_wins:<6} {total_trades - total_wins:<8} {total_trades:<7} {overall_wr:.1%}")

    # Show losses
    print("\n" + "="*70)
    print("LOSSES (for analysis)")
    print("="*70)

    losses = [r for r in results if not r['won']]
    for loss in losses[:20]:  # Show first 20
        print(f"Game {loss['game_id']}: {loss['signal']} signal, "
              f"Lead={loss['lead']}, Mom={loss['momentum']}, "
              f"Mins={loss['mins_remaining']:.1f}, "
              f"Final: {loss['final_home']}-{loss['final_away']}")

    if len(losses) > 20:
        print(f"... and {len(losses) - 20} more losses")

    # Coverage
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Games analyzed: {len(all_game_ids)}")
    print(f"Trades made: {total_trades}")
    print(f"Coverage: {total_trades / len(all_game_ids) * 100:.1f}%")
    print(f"Win rate: {overall_wr:.1%}")
    print(f"Losses: {total_trades - total_wins}")


if __name__ == "__main__":
    main()
