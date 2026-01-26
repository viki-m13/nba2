"""
Validate strategy for SPREAD COVERAGE, not just winner prediction.

The app bets: "Leading team will maintain/extend their lead"
- If team is up 10, bet them -10 spread
- They must WIN BY MORE THAN their current lead to cover

This is MUCH harder than predicting the winner!
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
    except:
        return None


def fetch_espn_scoreboard(date_str):
    """Fetch scoreboard for a date."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    try:
        resp = requests.get(url, timeout=30)
        return resp.json()
    except:
        return None


def get_espn_games_for_date(date_str):
    """Get completed games for a date."""
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
    """Extract scoring plays."""
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


def get_signal(home_score, away_score, home_5min, away_5min, mins_remaining):
    """Check for signal - returns signal info if triggered."""
    score_diff = home_score - away_score
    lead = abs(score_diff)
    momentum = home_5min - away_5min
    mom = abs(momentum)

    if score_diff == 0:
        return None

    side = 'home' if score_diff > 0 else 'away'

    # Momentum must align
    if score_diff > 0 and momentum <= 0:
        return None
    if score_diff < 0 and momentum >= 0:
        return None

    # All original signals
    if 30 <= mins_remaining <= 36:
        if lead >= 18 and mom >= 3:
            return {'side': side, 'signal': 'Q2_selective', 'lead': lead, 'mom': mom}
        if lead >= 15 and mom >= 3:
            return {'side': side, 'signal': 'Q2_early', 'lead': lead, 'mom': mom}

    if 24 <= mins_remaining <= 30:
        if lead >= 18 and mom >= 5:
            return {'side': side, 'signal': 'late_Q2', 'lead': lead, 'mom': mom}
        if lead >= 15 and mom >= 7:
            return {'side': side, 'signal': 'late_Q2_alt', 'lead': lead, 'mom': mom}

    if 18 <= mins_remaining <= 24:
        if lead >= 20 and mom >= 3:
            return {'side': side, 'signal': 'halftime_dominant', 'lead': lead, 'mom': mom}
        if lead >= 18 and mom >= 7:
            return {'side': side, 'signal': 'halftime', 'lead': lead, 'mom': mom}
        if lead >= 15 and mom >= 10:
            return {'side': side, 'signal': 'halftime_momentum', 'lead': lead, 'mom': mom}

    if 15 <= mins_remaining <= 20:
        if lead >= 18 and mom >= 5:
            return {'side': side, 'signal': 'mid_Q3', 'lead': lead, 'mom': mom}
        if lead >= 15 and mom >= 7:
            return {'side': side, 'signal': 'mid_Q3_alt', 'lead': lead, 'mom': mom}

    if 12 <= mins_remaining <= 18:
        if lead >= 15 and mom >= 7:
            return {'side': side, 'signal': 'late_Q3', 'lead': lead, 'mom': mom}
        if lead >= 12 and mom >= 10:
            return {'side': side, 'signal': 'late_Q3_momentum', 'lead': lead, 'mom': mom}

    if 8 <= mins_remaining <= 12:
        if lead >= 10 and mom >= 5:
            return {'side': side, 'signal': 'early_Q4', 'lead': lead, 'mom': mom}
        if lead >= 7 and mom >= 7:
            return {'side': side, 'signal': 'early_Q4_alt', 'lead': lead, 'mom': mom}

    if 2 <= mins_remaining <= 8:
        if lead >= 7 and mom >= 3:
            return {'side': side, 'signal': 'final', 'lead': lead, 'mom': mom}
        if lead >= 5 and mom >= 5:
            return {'side': side, 'signal': 'final_alt', 'lead': lead, 'mom': mom}

    return None


def backtest_spread(game_info, states):
    """
    Backtest for SPREAD coverage.

    If signal triggers at lead=X, we bet the leading team -X spread.
    To WIN: final margin must be >= lead at signal time.
    """
    if not states:
        return None

    final_home = game_info['home_score']
    final_away = game_info['away_score']
    final_margin = final_home - final_away  # positive = home won

    if final_home == 0 or final_away == 0:
        return None

    for state in states:
        sig = get_signal(
            state['home_score'], state['away_score'],
            state['home_5min'], state['away_5min'],
            state['mins_remaining']
        )

        if sig:
            lead_at_signal = sig['lead']
            side = sig['side']

            # Check spread coverage
            if side == 'home':
                # Bet home -lead_at_signal
                # Win if final_margin >= lead_at_signal
                spread_covered = final_margin >= lead_at_signal
            else:
                # Bet away -lead_at_signal
                # Win if -final_margin >= lead_at_signal (away wins by at least lead)
                spread_covered = (-final_margin) >= lead_at_signal

            return {
                'signal': sig['signal'],
                'side': side,
                'lead_at_signal': lead_at_signal,
                'momentum': sig['mom'],
                'mins_remaining': state['mins_remaining'],
                'final_margin': abs(final_margin),
                'winner': 'home' if final_margin > 0 else 'away',
                'spread_covered': spread_covered,
                'moneyline_won': (side == 'home' and final_margin > 0) or (side == 'away' and final_margin < 0),
                'home_team': game_info['home_team'],
                'away_team': game_info['away_team'],
                'final_home': final_home,
                'final_away': final_away,
            }

    return None


def main():
    print("="*70)
    print("SPREAD COVERAGE VALIDATION")
    print("Does leading team MAINTAIN their lead through game end?")
    print("="*70)

    # Collect dates
    dates = []
    for month in [11, 12]:
        for day in [5, 10, 15, 20, 25, 28]:
            dates.append(f"2023{month:02d}{day:02d}")
    for month in [1, 2, 3, 4]:
        for day in [5, 10, 15, 20, 25]:
            dates.append(f"2024{month:02d}{day:02d}")

    print(f"Fetching games from {len(dates)} dates...")

    all_games = []
    for date_str in dates:
        games = get_espn_games_for_date(date_str)
        all_games.extend(games)
        time.sleep(0.2)

    print(f"Found {len(all_games)} completed games")

    # Process games
    results = []
    signal_stats = defaultdict(lambda: {
        'spread_wins': 0, 'spread_losses': 0,
        'ml_wins': 0, 'ml_losses': 0
    })

    games_processed = 0
    for i, game_info in enumerate(all_games[:200]):
        if i % 40 == 0:
            print(f"Processing {i+1}/{min(len(all_games), 200)}...")

        game_data = fetch_espn_game_pbp(game_info['id'])
        time.sleep(0.3)

        if not game_data:
            continue

        plays = extract_plays(game_data)
        if not plays:
            continue

        states = build_states(plays)
        if not states:
            continue

        games_processed += 1
        result = backtest_spread(game_info, states)

        if result:
            results.append(result)
            sig = result['signal']

            if result['spread_covered']:
                signal_stats[sig]['spread_wins'] += 1
            else:
                signal_stats[sig]['spread_losses'] += 1

            if result['moneyline_won']:
                signal_stats[sig]['ml_wins'] += 1
            else:
                signal_stats[sig]['ml_losses'] += 1

    # Print results
    print("\n" + "="*70)
    print("RESULTS: SPREAD COVERAGE vs MONEYLINE")
    print("="*70)

    print(f"\n{'Signal':<20} {'Spread W':<10} {'Spread L':<10} {'Spread%':<10} {'ML%':<10}")
    print("-"*65)

    total_spread_w = total_spread_l = 0
    total_ml_w = total_ml_l = 0

    for sig in sorted(signal_stats.keys()):
        s = signal_stats[sig]
        sw, sl = s['spread_wins'], s['spread_losses']
        mw, ml = s['ml_wins'], s['ml_losses']

        total_spread_w += sw
        total_spread_l += sl
        total_ml_w += mw
        total_ml_l += ml

        st = sw + sl
        mt = mw + ml
        spread_wr = sw / st if st > 0 else 0
        ml_wr = mw / mt if mt > 0 else 0

        marker = " ***" if spread_wr < 0.60 else ""
        print(f"{sig:<20} {sw:<10} {sl:<10} {spread_wr:.1%}{marker:<6} {ml_wr:.1%}")

    print("-"*65)
    st = total_spread_w + total_spread_l
    mt = total_ml_w + total_ml_l
    print(f"{'TOTAL':<20} {total_spread_w:<10} {total_spread_l:<10} "
          f"{total_spread_w/st:.1%}      {total_ml_w/mt:.1%}")

    # Show spread losses
    print("\n" + "="*70)
    print("SPREAD LOSSES (bet team didn't maintain lead)")
    print("="*70)

    losses = [r for r in results if not r['spread_covered']]
    for loss in losses[:30]:
        team = loss['home_team'] if loss['side'] == 'home' else loss['away_team']
        opp = loss['away_team'] if loss['side'] == 'home' else loss['home_team']
        won_game = "WON" if loss['moneyline_won'] else "LOST"
        print(f"{loss['signal']:<18} L={loss['lead_at_signal']:>2} M={loss['momentum']:>2} "
              f"@{loss['mins_remaining']:>5.1f}m | "
              f"{team} {won_game} game, final margin={loss['final_margin']} "
              f"(needed {loss['lead_at_signal']})")

    # Save data for future use
    save_data = {
        'results': results,
        'signal_stats': dict(signal_stats),
        'games_processed': games_processed,
    }

    import pickle
    with open('cache/spread_validation_results.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    # Also save as JSON for easy viewing
    json_results = []
    for r in results:
        json_results.append({
            'signal': r['signal'],
            'team': r['home_team'] if r['side'] == 'home' else r['away_team'],
            'opponent': r['away_team'] if r['side'] == 'home' else r['home_team'],
            'lead_at_signal': r['lead_at_signal'],
            'momentum': r['momentum'],
            'mins_remaining': round(r['mins_remaining'], 1),
            'final_score': f"{r['final_away']}-{r['final_home']}" if r['side'] == 'away' else f"{r['final_home']}-{r['final_away']}",
            'final_margin': r['final_margin'],
            'spread_covered': r['spread_covered'],
            'moneyline_won': r['moneyline_won'],
        })

    with open('data/historical_games.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nData saved to:")
    print(f"  - cache/spread_validation_results.pkl")
    print(f"  - data/historical_games.json")


if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    main()
