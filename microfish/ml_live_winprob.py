"""
ESPN Live Win Probability Downloader
======================================
Downloads play-by-play win probability data from ESPN for every MLB game
in our dataset. This gives us the LIVE in-game probability at every at-bat,
which is what Polymarket prices track.

This is the missing piece: instead of pre-game sportsbook odds, we now have
the actual in-game win probability curve for every game.
"""

import os
import json
import time
import requests
import sys
from datetime import datetime


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'moneyline')
CACHE_FILE = os.path.join(DATA_DIR, 'espn_live_winprob_cache.json')
SUMMARY_FILE = os.path.join(DATA_DIR, 'live_winprob_summary.json')


def load_game_ids():
    """Get all game event IDs from our boxscore data."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Expanded boxscores have 2025 regular season
    box_file = os.path.join(base, 'mlb', 'webapp', 'data',
                            'mlb_player_boxscores_expanded.json')
    if not os.path.exists(box_file):
        box_file = os.path.join(base, 'mlb', 'webapp', 'data',
                                'mlb_player_boxscores.json')

    with open(box_file, 'r') as f:
        games = json.load(f)

    # Filter to 2025 regular season (April-September)
    regular = []
    for g in games:
        date = g.get('date', '')
        if date.startswith('2025') and int(date[4:6]) >= 4:
            regular.append(g)

    return regular


def download_win_probability(event_id: str, retries: int = 3) -> dict:
    """Download win probability data for a single game from ESPN."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={event_id}"

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                wp_data = data.get('winprobability', [])
                plays = data.get('plays', [])

                # Build play lookup for context
                play_lookup = {}
                for p in plays:
                    play_lookup[p.get('id', '')] = {
                        'inning': p.get('period', {}).get('number', 0)
                            if isinstance(p.get('period'), dict)
                            else p.get('period', 0),
                        'homeScore': p.get('homeScore', 0),
                        'awayScore': p.get('awayScore', 0),
                        'outs': p.get('outs', 0),
                        'text': p.get('text', '')[:100],
                    }

                # Build win probability timeline
                timeline = []
                for wp in wp_data:
                    pid = wp.get('playId', '')
                    hwp = wp.get('homeWinPercentage', 0.5)
                    play_ctx = play_lookup.get(pid, {})

                    timeline.append({
                        'homeWP': round(hwp, 4),
                        'awayWP': round(1 - hwp, 4),
                        'inning': play_ctx.get('inning', 0),
                        'homeScore': play_ctx.get('homeScore', 0),
                        'awayScore': play_ctx.get('awayScore', 0),
                    })

                return {
                    'event_id': event_id,
                    'n_plays': len(timeline),
                    'timeline': timeline,
                }

            elif resp.status_code == 404:
                return {'event_id': event_id, 'n_plays': 0, 'timeline': [], 'error': '404'}
            else:
                if attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return {'event_id': event_id, 'n_plays': 0, 'timeline': [], 'error': str(e)}

    return {'event_id': event_id, 'n_plays': 0, 'timeline': [], 'error': 'max_retries'}


def download_all(batch_size: int = 50, delay: float = 0.3):
    """
    Download win probability data for all games.
    Caches results incrementally to avoid re-downloading on restart.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        print(f"  Loaded cache: {len(cache)} games already downloaded")

    # Get game list
    games = load_game_ids()
    print(f"  Total 2025 regular season games: {len(games)}")

    # Filter to games not yet cached
    to_download = []
    for g in games:
        eid = g.get('event_id', '')
        if eid and eid not in cache:
            to_download.append(g)

    print(f"  Remaining to download: {len(to_download)}")

    if not to_download:
        print("  All games already cached!")
        return cache

    downloaded = 0
    errors = 0
    start_time = time.time()

    for i, game in enumerate(to_download):
        eid = game['event_id']

        result = download_win_probability(eid)
        cache[eid] = {
            'date': game.get('date', ''),
            'home': game.get('home', ''),
            'away': game.get('away', ''),
            'home_score': game.get('home_score', 0),
            'away_score': game.get('away_score', 0),
            **result,
        }

        if result.get('error'):
            errors += 1
        else:
            downloaded += 1

        # Progress
        if (i + 1) % 25 == 0 or i == len(to_download) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(to_download) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(to_download)}] "
                  f"OK={downloaded} Err={errors} "
                  f"Rate={rate:.1f}/s ETA={remaining:.0f}s")

        # Save cache every batch_size games
        if (i + 1) % batch_size == 0:
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f)

        # Rate limit
        time.sleep(delay)

    # Final save
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    print(f"\n  Download complete: {downloaded} OK, {errors} errors")
    print(f"  Total cached: {len(cache)} games")
    print(f"  Saved to {CACHE_FILE}")

    return cache


def compute_live_summary(cache: dict) -> dict:
    """
    Compute summary statistics from live win probability data.
    For each threshold (70-99%), determine:
    - How many games had either team reach that live WP
    - Did the team at that WP always win?
    """
    thresholds = list(range(60, 100))
    results = {}

    valid_games = {k: v for k, v in cache.items()
                   if v.get('n_plays', 0) > 0 and not v.get('error')}

    print(f"\n  Analyzing {len(valid_games)} games with win probability data")

    for threshold in thresholds:
        pct = threshold / 100.0
        games_reaching = 0
        correct = 0
        incorrect = 0
        details = []

        for eid, game in valid_games.items():
            timeline = game.get('timeline', [])
            if not timeline:
                continue

            home_score = game.get('home_score', 0)
            away_score = game.get('away_score', 0)
            if home_score == away_score:
                continue

            home_won = home_score > away_score

            # Did home ever reach this threshold?
            home_max_wp = max(t['homeWP'] for t in timeline)
            away_max_wp = max(t['awayWP'] for t in timeline)

            # Check if home side reached threshold
            if home_max_wp >= pct:
                games_reaching += 1
                if home_won:
                    correct += 1
                else:
                    incorrect += 1
                    # Find when it first crossed
                    first_cross_inn = '?'
                    for t in timeline:
                        if t['homeWP'] >= pct:
                            first_cross_inn = t.get('inning', '?')
                            break
                    details.append({
                        'date': game.get('date', ''),
                        'game': f"{game.get('away', '')}@{game.get('home', '')}",
                        'side': 'home',
                        'team': game.get('home', ''),
                        'max_wp': round(home_max_wp, 3),
                        'final': f"{away_score}-{home_score}",
                        'first_cross_inning': first_cross_inn,
                    })

            # Check if away side reached threshold
            if away_max_wp >= pct:
                games_reaching += 1
                if not home_won:
                    correct += 1
                else:
                    incorrect += 1
                    first_cross_inn = '?'
                    for t in timeline:
                        if t['awayWP'] >= pct:
                            first_cross_inn = t.get('inning', '?')
                            break
                    details.append({
                        'date': game.get('date', ''),
                        'game': f"{game.get('away', '')}@{game.get('home', '')}",
                        'side': 'away',
                        'team': game.get('away', ''),
                        'max_wp': round(away_max_wp, 3),
                        'final': f"{away_score}-{home_score}",
                        'first_cross_inning': first_cross_inn,
                    })

        total = correct + incorrect
        win_rate = correct / total if total > 0 else 0

        results[threshold] = {
            'threshold_pct': threshold,
            'games_reaching': games_reaching,
            'correct': correct,
            'incorrect': incorrect,
            'total_instances': total,
            'win_rate': round(win_rate, 6),
            'fill_rate': round(games_reaching / (len(valid_games) * 2), 4)
                if valid_games else 0,
            'comebacks': details[:20],  # Cap stored examples
        }

    return results


def print_live_analysis(results: dict, total_games: int):
    """Print the live win probability analysis."""
    print("\n" + "=" * 75)
    print("LIVE WIN PROBABILITY CERTAINTY FLOOR ANALYSIS")
    print("Data: ESPN play-by-play win probability, 2025 MLB season")
    print("=" * 75)
    print(f"\n{'Threshold':>10} {'Instances':>10} {'Correct':>8} {'Comebacks':>10} "
          f"{'Win Rate':>10} {'Fill/Game':>10}")
    print("-" * 65)

    floor_found = None
    for threshold in range(60, 100):
        r = results.get(threshold, {})
        total = r.get('total_instances', 0)
        if total == 0:
            continue
        correct = r.get('correct', 0)
        incorrect = r.get('incorrect', 0)
        wr = r.get('win_rate', 0)
        fill = r.get('games_reaching', 0) / total_games if total_games > 0 else 0

        marker = ""
        if wr == 1.0 and total >= 1 and floor_found is None:
            floor_found = threshold
            marker = " <-- LIVE FLOOR"

        print(f"   >= {threshold}%  {total:>10} {correct:>8} {incorrect:>10} "
              f"{wr:>9.1%} {fill:>9.1%}{marker}")

    if floor_found:
        r = results[floor_found]
        print(f"\nLIVE CERTAINTY FLOOR: {floor_found}%")
        print(f"  Instances: {r['total_instances']}")
        print(f"  Win rate: 100.0%")
        print(f"  Limit price: ${floor_found/100:.2f}")
        print(f"  Profit per share: ${1-floor_found/100:.2f}")

    # Show comebacks
    print("\nNOTABLE COMEBACKS (team reached high WP but lost):")
    all_comebacks = []
    for threshold in range(60, 100):
        r = results.get(threshold, {})
        for cb in r.get('comebacks', []):
            if cb not in all_comebacks:
                all_comebacks.append(cb)

    # Sort by max_wp descending
    all_comebacks.sort(key=lambda x: x.get('max_wp', 0), reverse=True)
    seen = set()
    for cb in all_comebacks[:30]:
        key = (cb.get('date', ''), cb.get('game', ''), cb.get('side', ''))
        if key in seen:
            continue
        seen.add(key)
        print(f"  {cb['date']}: {cb['game']} — {cb['team']} reached "
              f"{cb['max_wp']:.0%} (inning {cb['first_cross_inning']}) "
              f"but LOST {cb['final']}")

    return floor_found


def run_full_download_and_analysis():
    """Complete pipeline: download + analyze."""
    print("=" * 75)
    print("ESPN LIVE WIN PROBABILITY DATA PIPELINE")
    print("=" * 75)

    print("\nPHASE 1: Downloading win probability data from ESPN...")
    cache = download_all()

    valid = {k: v for k, v in cache.items()
             if v.get('n_plays', 0) > 0 and not v.get('error')}
    print(f"\n  Valid games with WP data: {len(valid)}")

    print("\nPHASE 2: Analyzing live certainty floors...")
    results = compute_live_summary(cache)

    floor = print_live_analysis(results, len(valid))

    # Save summary
    os.makedirs(DATA_DIR, exist_ok=True)
    summary = {
        'generated': datetime.now().isoformat(),
        'total_games_downloaded': len(cache),
        'valid_games': len(valid),
        'live_certainty_floor': floor,
        'threshold_results': {
            str(k): {kk: vv for kk, vv in v.items() if kk != 'comebacks'}
            for k, v in results.items()
        },
        'comebacks': {
            str(k): v.get('comebacks', [])
            for k, v in results.items()
            if v.get('incorrect', 0) > 0
        },
    }

    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to {SUMMARY_FILE}")

    return results, floor


if __name__ == '__main__':
    run_full_download_and_analysis()
