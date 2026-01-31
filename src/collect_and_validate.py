"""
COLLECT AND VALIDATE: Large-Scale NBA Game Data Collection + Strategy Validation
==================================================================================

Collects play-by-play data from ESPN API for 2+ NBA seasons, then validates
the momentum-aligned ML strategy with proper deduplication (one bet per game).

Two phases:
1. COLLECT: Fetch games and PBP, save raw game states to disk
2. VALIDATE: Run strategy conditions against saved games, produce honest results

Saves intermediate data so we never need to re-fetch from ESPN.
"""

import json
import math
import os
import sys
import time
import statistics
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

try:
    import requests
except ImportError:
    import urllib.request
    import urllib.error

    class _FakeResponse:
        def __init__(self, data):
            self._data = data
        def json(self):
            return json.loads(self._data)

    class requests:
        @staticmethod
        def get(url, timeout=30):
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return _FakeResponse(resp.read())


BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = BASE_DIR / 'cache' / 'games_pbp'
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / 'all_games_states.json'


# =============================================================================
# PHASE 1: DATA COLLECTION
# =============================================================================

def fetch_espn_scoreboard(date_str: str) -> Optional[dict]:
    """Fetch scoreboard for a date from ESPN API."""
    try:
        resp = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}",
            timeout=30
        )
        return resp.json()
    except Exception as e:
        return None


def fetch_espn_pbp(game_id: str) -> Optional[dict]:
    """Fetch play-by-play summary from ESPN API."""
    cache_file = CACHE_DIR / f"{game_id}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    try:
        resp = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}",
            timeout=30
        )
        data = resp.json()
        # Cache it
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        return data
    except Exception:
        return None


def parse_clock(period: int, clock_str: str) -> Optional[float]:
    """Convert period + clock to total minutes remaining in regulation."""
    if period > 4:
        return 0.0
    try:
        parts = str(clock_str).split(':')
        mins = int(parts[0])
        secs = int(parts[1]) if len(parts) > 1 else 0
        period_time = mins + secs / 60.0
        remaining_periods = 4 - period
        return period_time + (remaining_periods * 12)
    except:
        return None


def extract_game_states(pbp_data: dict) -> List[dict]:
    """Extract game states from play-by-play data."""
    plays_raw = pbp_data.get('plays', [])
    if not plays_raw:
        return []

    score_history = []
    states = []

    for play in plays_raw:
        period = play.get('period', {}).get('number', 0)
        if period > 4:
            continue

        clock = play.get('clock', {}).get('displayValue', '')
        home_score = play.get('homeScore', 0)
        away_score = play.get('awayScore', 0)

        if home_score == 0 and away_score == 0:
            continue

        mins_remaining = parse_clock(period, clock)
        if mins_remaining is None:
            continue

        score_history.append((mins_remaining, home_score, away_score))

        # 5-minute momentum calculation
        home_5min = 0
        away_5min = 0
        for past_mins, past_home, past_away in reversed(score_history[:-1]):
            if past_mins - mins_remaining >= 5:
                home_5min = home_score - past_home
                away_5min = away_score - past_away
                break

        states.append({
            'mins_remaining': round(mins_remaining, 2),
            'home_score': home_score,
            'away_score': away_score,
            'home_5min': home_5min,
            'away_5min': away_5min,
        })

    return states


def generate_date_range(start: str, end: str) -> List[str]:
    """Generate all dates between start and end (YYYYMMDD format)."""
    start_dt = datetime.strptime(start, '%Y%m%d')
    end_dt = datetime.strptime(end, '%Y%m%d')
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    return dates


def collect_games(seasons: List[Tuple[str, str]], delay: float = 0.3) -> List[dict]:
    """
    Collect games from ESPN for the given date ranges.

    Args:
        seasons: List of (start_date, end_date) tuples in YYYYMMDD format
        delay: Seconds between API calls

    Returns: List of game dicts with states
    """
    # Check for existing data
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        print(f"Found existing data: {len(existing)} games")

        existing_ids = {g['id'] for g in existing}
        print(f"  Date range: {existing[0]['date']} to {existing[-1]['date']}")
    else:
        existing = []
        existing_ids = set()

    # Generate all dates
    all_dates = []
    for start, end in seasons:
        all_dates.extend(generate_date_range(start, end))
    all_dates = sorted(set(all_dates))

    print(f"\nChecking {len(all_dates)} dates across {len(seasons)} seasons...")

    # Fetch scoreboards and collect game IDs
    all_game_refs = []
    scoreboard_cache = BASE_DIR / 'cache' / 'scoreboards'
    scoreboard_cache.mkdir(parents=True, exist_ok=True)

    for i, date_str in enumerate(all_dates):
        if i % 30 == 0:
            print(f"  Fetching scoreboards: {i}/{len(all_dates)} dates...")

        cache_file = scoreboard_cache / f"{date_str}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
        else:
            data = fetch_espn_scoreboard(date_str)
            if data:
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
            time.sleep(delay * 0.5)  # Scoreboards are lightweight

        if not data:
            continue

        for event in data.get('events', []):
            game_id = event.get('id')
            status = event.get('status', {}).get('type', {}).get('completed', False)

            if not game_id or not status:
                continue

            if game_id in existing_ids:
                continue  # Already have this game

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
                all_game_refs.append({
                    'id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'date': date_str,
                })

    print(f"\n  New games to fetch PBP for: {len(all_game_refs)}")

    # Fetch play-by-play for new games
    new_games = []
    failed = 0
    for i, game_ref in enumerate(all_game_refs):
        if i % 50 == 0 and i > 0:
            print(f"  Fetching PBP: {i}/{len(all_game_refs)} (got {len(new_games)} valid, {failed} failed)...")

        pbp = fetch_espn_pbp(game_ref['id'])
        time.sleep(delay)

        if not pbp:
            failed += 1
            continue

        states = extract_game_states(pbp)
        if len(states) < 20:
            failed += 1
            continue

        new_games.append({
            'id': game_ref['id'],
            'home_team': game_ref['home_team'],
            'away_team': game_ref['away_team'],
            'final_home': game_ref['home_score'],
            'final_away': game_ref['away_score'],
            'date': game_ref['date'],
            'states': states,
        })

    print(f"\n  New games with valid PBP: {len(new_games)} (failed: {failed})")

    # Merge with existing
    all_games = existing + new_games
    all_games.sort(key=lambda g: g['date'])

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_games, f)
    print(f"\n  Total games saved: {len(all_games)}")
    print(f"  Saved to: {OUTPUT_FILE}")

    return all_games


# =============================================================================
# PHASE 2: STRATEGY VALIDATION
# =============================================================================

# Market model (calibrated, sigma=2.6)
def normal_cdf(x: float) -> float:
    if x > 6: return 0.9999
    if x < -6: return 0.0001
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def market_win_prob(lead: int, mins_remaining: float) -> float:
    if lead <= 0 or mins_remaining <= 0: return 0.5
    z = lead / (2.6 * math.sqrt(max(mins_remaining, 0.5)))
    return max(0.51, min(0.998, normal_cdf(z)))


def live_ml_odds(lead: int, mins_remaining: float, vig: float = 0.045) -> float:
    prob = market_win_prob(lead, mins_remaining)
    mp = min(0.995, prob + vig * prob)
    if mp >= 0.5:
        return -(mp / (1 - mp)) * 100
    return ((1 - mp) / mp) * 100


def ml_payout(odds: float) -> float:
    if odds < 0:
        return 100 / abs(odds)
    return odds / 100


def find_first_entry(
    states: List[dict],
    final_home: int, final_away: int,
    lead_min: int, lead_max: int,
    mom_min: int,
    time_min: float, time_max: float,
) -> Optional[dict]:
    """
    Find the FIRST moment in a game where strategy conditions are met.
    Returns entry details + outcome, or None if conditions never met.
    """
    for state in states:
        mins = state['mins_remaining']
        if not (time_min <= mins <= time_max):
            continue

        home_score = state['home_score']
        away_score = state['away_score']
        score_diff = home_score - away_score
        lead = abs(score_diff)

        if not (lead_min <= lead <= lead_max):
            continue

        momentum = state['home_5min'] - state['away_5min']
        mom = abs(momentum)

        if mom < mom_min:
            continue

        # Momentum must align with lead
        if score_diff > 0 and momentum <= 0:
            continue
        if score_diff < 0 and momentum >= 0:
            continue

        # All conditions met - this is our entry point
        side = 'home' if score_diff > 0 else 'away'

        if side == 'home':
            final_margin = final_home - final_away
        else:
            final_margin = final_away - final_home

        ml_won = final_margin > 0
        spread_covered = final_margin > lead  # Beat the live spread
        lead_extension = final_margin - lead

        return {
            'lead': lead,
            'momentum': mom,
            'mins_remaining': mins,
            'side': side,
            'home_score_at_entry': home_score,
            'away_score_at_entry': away_score,
            'final_home': final_home,
            'final_away': final_away,
            'final_margin': final_margin,
            'ml_won': ml_won,
            'spread_covered': spread_covered,
            'lead_extension': lead_extension,
        }

    return None


def validate_strategy(
    games: List[dict],
    name: str,
    lead_min: int, lead_max: int,
    mom_min: int,
    time_min: float, time_max: float,
) -> Optional[dict]:
    """Validate a strategy across all games with ONE bet per game."""
    entries = []

    for game in games:
        entry = find_first_entry(
            game['states'],
            game['final_home'], game['final_away'],
            lead_min, lead_max, mom_min, time_min, time_max
        )
        if entry:
            entry['date'] = game['date']
            entry['home_team'] = game['home_team']
            entry['away_team'] = game['away_team']
            entry['game_id'] = game['id']
            entries.append(entry)

    n = len(entries)
    if n < 3:
        return None

    # ML metrics
    ml_wins = sum(1 for e in entries if e['ml_won'])
    ml_losses = n - ml_wins
    ml_wr = ml_wins / n

    ml_pnl = 0.0
    ml_bets = []
    for e in entries:
        odds = live_ml_odds(e['lead'], e['mins_remaining'])
        pay = ml_payout(odds)
        result = pay if e['ml_won'] else -1.0
        ml_pnl += result
        ml_bets.append(result)

    ml_roi = ml_pnl / n * 100

    # Average market prob and edge
    avg_mkt = sum(market_win_prob(e['lead'], e['mins_remaining']) for e in entries) / n
    ml_edge = (ml_wr - avg_mkt) * 100

    # Average odds
    avg_odds = statistics.mean([live_ml_odds(e['lead'], e['mins_remaining']) for e in entries])

    # Spread metrics
    sp_covers = sum(1 for e in entries if e['spread_covered'])
    sp_pushes = sum(1 for e in entries if e['final_margin'] == e['lead'])
    sp_losses = n - sp_covers - sp_pushes
    sp_cover_rate = sp_covers / n * 100
    sp_active = n - sp_pushes
    sp_pnl = sp_covers * (100/110) - sp_losses * 1.0
    sp_roi = sp_pnl / sp_active * 100 if sp_active > 0 else 0

    # Lead extension
    extensions = [e['lead_extension'] for e in entries]
    avg_ext = statistics.mean(extensions)
    med_ext = statistics.median(extensions)
    pct_grew = sum(1 for x in extensions if x > 0) / n * 100

    # Drawdown
    cum = 0
    peak = 0
    max_dd = 0
    for b in ml_bets:
        cum += b
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd

    # Monthly consistency
    by_month = defaultdict(lambda: {'pnl': 0.0, 'n': 0, 'wins': 0})
    for e, b in zip(entries, ml_bets):
        month = e['date'][:6]
        by_month[month]['pnl'] += b
        by_month[month]['n'] += 1
        by_month[month]['wins'] += 1 if e['ml_won'] else 0

    prof_months = sum(1 for m in by_month.values() if m['pnl'] > 0)
    total_months = len(by_month)

    # Loss streak
    max_loss_streak = 0
    current = 0
    for e in entries:
        if not e['ml_won']:
            current += 1
            max_loss_streak = max(max_loss_streak, current)
        else:
            current = 0

    # By season
    by_season = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0.0})
    for e, b in zip(entries, ml_bets):
        year = int(e['date'][:4])
        month = int(e['date'][4:6])
        season = f"{year}-{year+1}" if month >= 10 else f"{year-1}-{year}"
        by_season[season]['n'] += 1
        by_season[season]['wins'] += 1 if e['ml_won'] else 0
        by_season[season]['pnl'] += b

    return {
        'name': name,
        'n_games': n,
        'ml_wins': ml_wins,
        'ml_losses': ml_losses,
        'ml_wr': ml_wr * 100,
        'ml_pnl': ml_pnl,
        'ml_roi': ml_roi,
        'ml_edge': ml_edge,
        'avg_odds': avg_odds,
        'avg_mkt_prob': avg_mkt * 100,
        'sp_cover_rate': sp_cover_rate,
        'sp_pnl': sp_pnl,
        'sp_roi': sp_roi,
        'avg_ext': avg_ext,
        'med_ext': med_ext,
        'pct_grew': pct_grew,
        'max_dd': max_dd,
        'max_loss_streak': max_loss_streak,
        'prof_months': prof_months,
        'total_months': total_months,
        'by_month': dict(by_month),
        'by_season': dict(by_season),
        'entries': entries,
    }


def print_full_report(result: dict):
    """Print detailed strategy report."""
    r = result
    print(f"\n{'='*75}")
    print(f"  {r['name']}")
    print(f"{'='*75}")
    print(f"  Games: {r['n_games']} (ONE bet per game)")

    dates = [e['date'] for e in r['entries']]
    print(f"  Range: {min(dates)} to {max(dates)}")

    print(f"\n  --- MONEYLINE ---")
    print(f"  Win Rate:     {r['ml_wr']:.1f}% ({r['ml_wins']}W / {r['ml_losses']}L)")
    print(f"  Avg Mkt Prob: {r['avg_mkt_prob']:.1f}%")
    print(f"  Edge:         {r['ml_edge']:+.1f}%")
    print(f"  Avg Odds:     {r['avg_odds']:.0f}")
    print(f"  Total P&L:    {r['ml_pnl']:+.2f}u")
    print(f"  ROI:          {r['ml_roi']:+.2f}%")
    print(f"  Max Drawdown: {r['max_dd']:.2f}u")
    print(f"  Max Loss Run: {r['max_loss_streak']}")

    print(f"\n  --- LIVE SPREAD ---")
    print(f"  Cover Rate:   {r['sp_cover_rate']:.1f}%")
    sp_profitable = "YES" if r['sp_cover_rate'] > 52.4 else "NO"
    print(f"  Profitable:   {sp_profitable} (need >52.4%)")
    print(f"  P&L:          {r['sp_pnl']:+.2f}u | ROI: {r['sp_roi']:+.2f}%")

    print(f"\n  --- LEAD EXTENSION ---")
    print(f"  Avg: {r['avg_ext']:+.1f} | Med: {r['med_ext']:+.1f} | Grew: {r['pct_grew']:.0f}%")

    print(f"\n  --- BY SEASON ---")
    for season in sorted(r['by_season'].keys()):
        s = r['by_season'][season]
        wr = s['wins'] / s['n'] * 100 if s['n'] > 0 else 0
        roi = s['pnl'] / s['n'] * 100 if s['n'] > 0 else 0
        print(f"    {season}: {s['n']:>3} games, WR={wr:>5.1f}%, P&L={s['pnl']:>+6.2f}u, ROI={roi:>+5.1f}%")

    print(f"\n  --- MONTHLY ({r['prof_months']}/{r['total_months']} profitable) ---")
    for month in sorted(r['by_month'].keys()):
        m = r['by_month'][month]
        wr = m['wins'] / m['n'] * 100 if m['n'] > 0 else 0
        print(f"    {month}: {m['n']:>3} games, WR={wr:>5.1f}%, P&L={m['pnl']:>+6.2f}u")

    # Losses
    losses = [e for e in r['entries'] if not e['ml_won']]
    if losses:
        print(f"\n  --- ML LOSSES ({len(losses)}) ---")
        for e in losses:
            mkt = market_win_prob(e['lead'], e['mins_remaining']) * 100
            print(f"    {e['date']}: {e['home_team']} vs {e['away_team']} | "
                  f"L={e['lead']}, M={e['momentum']}, T={e['mins_remaining']:.0f}min | "
                  f"Final={e['final_home']}-{e['final_away']} (margin={e['final_margin']}) | "
                  f"Mkt={mkt:.0f}%")
    else:
        print(f"\n  --- NO ML LOSSES ---")


def run_full_validation(games: List[dict]):
    """Run comprehensive validation with grid search and detailed reports."""
    print("\n" + "=" * 80)
    print(f"STRATEGY VALIDATION ON {len(games)} GAMES")
    print("=" * 80)

    # Date range
    dates = sorted(set(g['date'] for g in games))
    print(f"Date range: {dates[0]} to {dates[-1]} ({len(dates)} unique dates)")

    # Grid search
    print("\n" + "-" * 80)
    print("GRID SEARCH (one bet per game, all conditions)")
    print("-" * 80)

    all_results = []

    for lead_min in [8, 10, 12]:
        for lead_max in [12, 14, 16, 19, 24]:
            if lead_max < lead_min: continue
            for mom_min in [8, 10, 12, 14, 16]:
                for time_min in [12, 15, 18]:
                    for time_max in [18, 21, 24]:
                        if time_max <= time_min: continue

                        label = f"L{lead_min}-{lead_max} M{mom_min}+ T{time_min}-{time_max}"
                        result = validate_strategy(
                            games, label,
                            lead_min, lead_max, mom_min, time_min, time_max
                        )
                        if result and result['n_games'] >= 5:
                            all_results.append(result)

    # Sort by ROI
    all_results.sort(key=lambda x: x['ml_roi'], reverse=True)

    print(f"\n  TOP 25 ML STRATEGIES (min 5 games):")
    print(f"  {'Conditions':<28} {'G':>4} {'WR%':>5} {'Edge':>5} {'ROI%':>6} {'P&L':>6} {'L':>2} {'M+/M':>4} {'SpCov':>5}")
    print(f"  {'-'*28} {'-'*4} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*2} {'-'*4} {'-'*5}")
    for r in all_results[:25]:
        print(f"  {r['name']:<28} {r['n_games']:>4} {r['ml_wr']:>4.0f}% {r['ml_edge']:>+4.1f}% "
              f"{r['ml_roi']:>+5.1f}% {r['ml_pnl']:>+5.1f}u {r['ml_losses']:>2} "
              f"{r['prof_months']}/{r['total_months']} {r['sp_cover_rate']:>4.0f}%")

    # With min 10 games
    r10 = [r for r in all_results if r['n_games'] >= 10]
    if r10:
        r10.sort(key=lambda x: x['ml_roi'], reverse=True)
        print(f"\n  TOP 20 ML STRATEGIES (min 10 games):")
        print(f"  {'Conditions':<28} {'G':>4} {'WR%':>5} {'Edge':>5} {'ROI%':>6} {'P&L':>6} {'L':>2} {'M+/M':>4} {'SpCov':>5}")
        print(f"  {'-'*28} {'-'*4} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*2} {'-'*4} {'-'*5}")
        for r in r10[:20]:
            print(f"  {r['name']:<28} {r['n_games']:>4} {r['ml_wr']:>4.0f}% {r['ml_edge']:>+4.1f}% "
                  f"{r['ml_roi']:>+5.1f}% {r['ml_pnl']:>+5.1f}u {r['ml_losses']:>2} "
                  f"{r['prof_months']}/{r['total_months']} {r['sp_cover_rate']:>4.0f}%")

    # With min 20 games
    r20 = [r for r in all_results if r['n_games'] >= 20]
    if r20:
        r20.sort(key=lambda x: x['ml_roi'], reverse=True)
        print(f"\n  TOP 15 ML STRATEGIES (min 20 games):")
        print(f"  {'Conditions':<28} {'G':>4} {'WR%':>5} {'Edge':>5} {'ROI%':>6} {'P&L':>6} {'L':>2} {'M+/M':>4} {'SpCov':>5}")
        print(f"  {'-'*28} {'-'*4} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*2} {'-'*4} {'-'*5}")
        for r in r20[:15]:
            print(f"  {r['name']:<28} {r['n_games']:>4} {r['ml_wr']:>4.0f}% {r['ml_edge']:>+4.1f}% "
                  f"{r['ml_roi']:>+5.1f}% {r['ml_pnl']:>+5.1f}u {r['ml_losses']:>2} "
                  f"{r['prof_months']}/{r['total_months']} {r['sp_cover_rate']:>4.0f}%")

    # Spread strategies
    spread_results = [r for r in all_results if r['sp_cover_rate'] > 52.4]
    if spread_results:
        spread_results.sort(key=lambda x: x['sp_roi'], reverse=True)
        print(f"\n  TOP 15 LIVE SPREAD STRATEGIES (>52.4% cover):")
        print(f"  {'Conditions':<28} {'G':>4} {'Cover%':>6} {'SpROI':>6} {'SpPnL':>6} {'AvgExt':>6} {'%Grew':>5}")
        print(f"  {'-'*28} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
        for r in spread_results[:15]:
            print(f"  {r['name']:<28} {r['n_games']:>4} {r['sp_cover_rate']:>5.1f}% "
                  f"{r['sp_roi']:>+5.1f}% {r['sp_pnl']:>+5.1f}u {r['avg_ext']:>+5.1f} {r['pct_grew']:>4.0f}%")

    # Named strategy detailed reports
    print("\n\n" + "#" * 80)
    print("DETAILED REPORTS FOR KEY STRATEGIES")
    print("#" * 80)

    named_strategies = [
        ("PRIME: L10-14, M12+, 18-24min", 10, 14, 12, 18, 24),
        ("STANDARD: L10-16, M12+, 18-24min", 10, 16, 12, 18, 24),
        ("BROAD: L10-19, M12+, 18-24min", 10, 19, 12, 18, 24),
        ("RELAXED MOM: L10-14, M10+, 18-24min", 10, 14, 10, 18, 24),
        ("HIGH MOM: L10-14, M14+, 12-24min", 10, 14, 14, 12, 24),
        ("LOWER LEAD: L8-12, M12+, 18-24min", 8, 12, 12, 18, 24),
        ("VERY HIGH MOM: L10-19, M16+, 18-24min", 10, 19, 16, 18, 24),
    ]

    for name, lmin, lmax, mmin, tmin, tmax in named_strategies:
        result = validate_strategy(games, name, lmin, lmax, mmin, tmin, tmax)
        if result and result['n_games'] >= 3:
            print_full_report(result)
        else:
            print(f"\n  {name}: insufficient data (< 3 games)")

    # Cross-validation: split by season
    seasons_in_data = set()
    for g in games:
        year = int(g['date'][:4])
        month = int(g['date'][4:6])
        season = f"{year}-{year+1}" if month >= 10 else f"{year-1}-{year}"
        seasons_in_data.add(season)

    if len(seasons_in_data) > 1:
        print("\n\n" + "=" * 80)
        print("CROSS-SEASON VALIDATION")
        print("=" * 80)

        for name, lmin, lmax, mmin, tmin, tmax in named_strategies[:4]:
            print(f"\n  {name}:")
            for season in sorted(seasons_in_data):
                season_games = []
                for g in games:
                    year = int(g['date'][:4])
                    month = int(g['date'][4:6])
                    g_season = f"{year}-{year+1}" if month >= 10 else f"{year-1}-{year}"
                    if g_season == season:
                        season_games.append(g)

                result = validate_strategy(
                    season_games, f"{name} [{season}]",
                    lmin, lmax, mmin, tmin, tmax
                )
                if result and result['n_games'] >= 1:
                    wr = result['ml_wr']
                    roi = result['ml_roi']
                    n = result['n_games']
                    losses = result['ml_losses']
                    sp_cov = result['sp_cover_rate']
                    print(f"    {season}: {n:>3} games, WR={wr:>5.1f}%, ROI={roi:>+5.1f}%, "
                          f"Losses={losses}, SpreadCov={sp_cov:.0f}%")
                else:
                    print(f"    {season}: no qualifying games")

    # Vig sensitivity for top strategy
    print("\n\n" + "=" * 80)
    print("VIG SENSITIVITY")
    print("=" * 80)

    for name, lmin, lmax, mmin, tmin, tmax in named_strategies[:3]:
        entries = []
        for game in games:
            entry = find_first_entry(
                game['states'], game['final_home'], game['final_away'],
                lmin, lmax, mmin, tmin, tmax
            )
            if entry:
                entries.append(entry)

        n = len(entries)
        if n < 5: continue

        print(f"\n  {name} ({n} games):")
        print(f"  {'Vig':>6} {'AvgOdds':>8} {'P&L':>7} {'ROI':>7}")
        print(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*7}")

        for vig in [0.02, 0.03, 0.045, 0.06, 0.08, 0.10, 0.12]:
            pnl = 0
            odds_list = []
            for e in entries:
                odds = live_ml_odds(e['lead'], e['mins_remaining'], vig=vig)
                pay = ml_payout(odds)
                pnl += pay if e['ml_won'] else -1.0
                odds_list.append(odds)
            roi = pnl / n * 100
            avg = statistics.mean(odds_list)
            marker = " <-- std" if vig == 0.045 else ""
            print(f"  {vig:>5.1%} {avg:>7.0f} {pnl:>+6.2f}u {roi:>+5.1f}%{marker}")

    # Game-by-game for prime strategy
    print("\n\n" + "=" * 80)
    print("GAME-BY-GAME: PRIME STRATEGY (L10-14, M12+, 18-24min)")
    print("=" * 80)

    entries = []
    for game in games:
        entry = find_first_entry(
            game['states'], game['final_home'], game['final_away'],
            10, 14, 12, 18, 24
        )
        if entry:
            entry['date'] = game['date']
            entry['home_team'] = game['home_team']
            entry['away_team'] = game['away_team']
            entries.append(entry)

    entries.sort(key=lambda e: e['date'])

    if entries:
        print(f"\n  {'Date':<10} {'Matchup':<14} {'L':>3} {'M':>3} {'T':>5} {'Odds':>6} "
              f"{'Final':>10} {'Marg':>5} {'Ext':>4} {'ML':>3} {'Sp':>3}")
        print(f"  {'-'*10} {'-'*14} {'-'*3} {'-'*3} {'-'*5} {'-'*6} "
              f"{'-'*10} {'-'*5} {'-'*4} {'-'*3} {'-'*3}")

        ml_pnl_total = 0
        sp_pnl_total = 0
        for e in entries:
            odds = live_ml_odds(e['lead'], e['mins_remaining'])
            pay = ml_payout(odds)
            ml_r = pay if e['ml_won'] else -1.0
            sp_r = 100/110 if e['spread_covered'] else (-1.0 if e['final_margin'] != e['lead'] else 0)
            ml_pnl_total += ml_r
            sp_pnl_total += sp_r

            matchup = f"{e['home_team']}v{e['away_team']}"
            final = f"{e['final_home']}-{e['final_away']}"
            ml_mark = "W" if e['ml_won'] else "L"
            sp_mark = "W" if e['spread_covered'] else ("P" if e['final_margin'] == e['lead'] else "L")

            print(f"  {e['date']:<10} {matchup:<14} {e['lead']:>3} {e['momentum']:>3} "
                  f"{e['mins_remaining']:>5.1f} {odds:>6.0f} {final:>10} {e['final_margin']:>5} "
                  f"{e['lead_extension']:>+4} {ml_mark:>3} {sp_mark:>3}")

        n = len(entries)
        print(f"\n  TOTALS ({n} games):")
        print(f"  ML P&L:  {ml_pnl_total:>+.2f}u (ROI: {ml_pnl_total/n*100:+.1f}%)")
        print(f"  Sp P&L:  {sp_pnl_total:>+.2f}u (ROI: {sp_pnl_total/n*100:+.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("NBA STRATEGY: LARGE-SCALE DATA COLLECTION + VALIDATION")
    print("=" * 80)

    # Define seasons to collect
    # 2023-24: Oct 24, 2023 to Apr 14, 2024
    # 2022-23: Oct 18, 2022 to Apr 9, 2023
    # 2021-22: Oct 19, 2021 to Apr 10, 2022
    seasons = [
        ('20231024', '20240414'),  # 2023-24
        ('20221018', '20230409'),  # 2022-23
        ('20211019', '20220410'),  # 2021-22
    ]

    print(f"\nCollecting data from {len(seasons)} seasons...")
    print("(Using ESPN API with caching - subsequent runs will be fast)")

    games = collect_games(seasons, delay=0.25)

    print(f"\n\nTotal games loaded: {len(games)}")

    # Run validation
    run_full_validation(games)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
