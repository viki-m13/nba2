"""
NBA LIVE BETTING: TRADER-STYLE STRATEGY ANALYSIS
==================================================

Think like a proprietary trader, not a typical sports bettor.

KEY INSIGHT FROM PRIOR VALIDATION:
  Momentum-aligned favorites UNDERPERFORM market expectations by ~7%.
  Market prices them at ~87% win prob, actual is ~80%.

  A trader's response: FADE THE MISPRICING.

STRATEGIES EXPLORED:
  1. UNDERDOG ML (fade the overpriced favorite)
  2. UNDERDOG SPREAD (mean reversion - leads compress ~4pts)
  3. OVER/UNDER (scoring pace after momentum surges)
  4. TRAILING TEAM SCORING BURSTS (comeback predictor)
  5. LEAD COLLAPSE DETECTION (when big leads evaporate)
  6. COMBINED SIGNALS (multi-factor approach)

Data: 2,310 real NBA games, 2 full seasons, ESPN PBP
"""

import json
import math
import os
import statistics
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple


BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = BASE_DIR / 'cache'


# =============================================================================
# DATA LOADING (from validated_strategy.py)
# =============================================================================

def parse_clock(period: int, clock_str: str) -> Optional[float]:
    if period > 4:
        return None
    try:
        parts = str(clock_str).split(':')
        mins = int(parts[0])
        secs = int(parts[1]) if len(parts) > 1 else 0
        period_time = mins + secs / 60.0
        remaining_periods = 4 - period
        return round(period_time + (remaining_periods * 12), 2)
    except:
        return None


def build_game_from_pbp(pbp_data: dict) -> Optional[dict]:
    """Build game with FULL state history (scoring AND non-scoring for pace)."""
    plays_raw = pbp_data.get('plays', [])
    if not plays_raw:
        return None

    header = pbp_data.get('header', {})
    competitions = header.get('competitions', [{}])
    if not competitions:
        return None

    comp = competitions[0]
    competitors = comp.get('competitors', [])

    home_team = away_team = ''
    final_home = final_away = 0
    game_date = comp.get('date', '')[:10].replace('-', '')

    for c in competitors:
        team_abbr = c.get('team', {}).get('abbreviation', '')
        score = int(c.get('score', 0))
        if c.get('homeAway') == 'home':
            home_team = team_abbr
            final_home = score
        else:
            away_team = team_abbr
            final_away = score

    if not home_team or not away_team or final_home == 0 or final_away == 0:
        return None

    # Build states on SCORING plays only
    prev_home = 0
    prev_away = 0
    score_history = []
    states = []

    for play in plays_raw:
        period = play.get('period', {}).get('number', 0)
        if period > 4:
            continue

        clock = play.get('clock', {}).get('displayValue', '')
        home_score = play.get('homeScore', 0)
        away_score = play.get('awayScore', 0)

        mins_remaining = parse_clock(period, clock)
        if mins_remaining is None:
            continue

        if home_score == prev_home and away_score == prev_away:
            continue
        if home_score == 0 and away_score == 0:
            continue

        prev_home = home_score
        prev_away = away_score
        score_history.append((mins_remaining, home_score, away_score))

        # 5-minute momentum
        home_5min = 0
        away_5min = 0
        for past_mins, past_home, past_away in reversed(score_history[:-1]):
            time_diff = past_mins - mins_remaining
            if time_diff >= 5.0:
                home_5min = home_score - past_home
                away_5min = away_score - past_away
                break

        # 3-minute momentum (shorter window for burst detection)
        home_3min = 0
        away_3min = 0
        for past_mins, past_home, past_away in reversed(score_history[:-1]):
            time_diff = past_mins - mins_remaining
            if time_diff >= 3.0:
                home_3min = home_score - past_home
                away_3min = away_score - past_away
                break

        # 10-minute scoring (for total/pace analysis)
        home_10min = 0
        away_10min = 0
        for past_mins, past_home, past_away in reversed(score_history[:-1]):
            time_diff = past_mins - mins_remaining
            if time_diff >= 10.0:
                home_10min = home_score - past_home
                away_10min = away_score - past_away
                break

        states.append({
            'mins_remaining': mins_remaining,
            'home_score': home_score,
            'away_score': away_score,
            'home_5min': home_5min,
            'away_5min': away_5min,
            'home_3min': home_3min,
            'away_3min': away_3min,
            'home_10min': home_10min,
            'away_10min': away_10min,
        })

    if len(states) < 20:
        return None

    game_id = pbp_data.get('header', {}).get('id', '')

    # Compute total regulation points
    total_points = final_home + final_away

    return {
        'id': game_id,
        'date': game_date,
        'home_team': home_team,
        'away_team': away_team,
        'final_home': final_home,
        'final_away': final_away,
        'total_points': total_points,
        'n_states': len(states),
        'states': states,
    }


def load_all_games() -> List[dict]:
    pbp_dir = CACHE_DIR / 'games_pbp'
    if not pbp_dir.exists():
        print("ERROR: No cached PBP data.")
        return []

    files = sorted(pbp_dir.glob('*.json'))
    print(f"Loading {len(files)} cached PBP files...")

    games = []
    for i, f in enumerate(files):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(files)}...")
        try:
            with open(f) as fh:
                pbp = json.load(fh)
            game = build_game_from_pbp(pbp)
            if game and game['date']:
                games.append(game)
        except:
            continue

    seen = set()
    unique = []
    for g in games:
        if g['id'] not in seen:
            seen.add(g['id'])
            unique.append(g)
    unique.sort(key=lambda g: g['date'])
    print(f"  Loaded {len(unique)} games ({unique[0]['date']} to {unique[-1]['date']})")
    return unique


# =============================================================================
# MARKET MODEL
# =============================================================================

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


def fav_ml_odds(prob: float, vig: float = 0.045) -> float:
    """ML odds for the favorite (negative American)."""
    mp = min(0.995, prob + vig * prob)
    return -(mp / (1 - mp)) * 100


def dog_ml_odds(prob: float, vig: float = 0.045) -> float:
    """
    ML odds for the underdog (positive American).

    Vig makes odds WORSE for the bettor (lower payout).
    Each side's implied prob is inflated by (1 + vig), creating overround.
    """
    dog_true = 1 - prob
    # Vig inflates implied prob (reduces payout for bettor)
    dog_implied = dog_true * (1 + vig)
    dog_implied = max(0.01, min(0.99, dog_implied))
    # American positive odds
    return ((1 - dog_implied) / dog_implied) * 100


def ml_payout(odds: float) -> float:
    """Payout per unit risked."""
    if odds < 0:
        return 100 / abs(odds)
    else:
        return odds / 100


# =============================================================================
# STRATEGY 1: FADE THE FAVORITE (UNDERDOG ML)
# =============================================================================

def find_underdog_ml_entry(
    game: dict,
    lead_min: int, lead_max: int,
    mom_min: int,
    time_min: float, time_max: float,
) -> Optional[dict]:
    """
    CONTRARIAN: When favorite has big lead + aligned momentum,
    bet the UNDERDOG moneyline.

    Thesis: Market overprices momentum. Actual WR for favorite is ~80%
    when market implies ~87%. Underdog wins 20% at +500-700 odds.
    """
    states = game['states']
    final_home = game['final_home']
    final_away = game['final_away']

    for state in states:
        mins = state['mins_remaining']
        if not (time_min <= mins <= time_max):
            continue

        h = state['home_score']
        a = state['away_score']
        diff = h - a
        lead = abs(diff)

        if not (lead_min <= lead <= lead_max):
            continue

        momentum = state['home_5min'] - state['away_5min']
        mom = abs(momentum)
        if mom < mom_min:
            continue

        # Momentum must be ALIGNED with lead (this is the condition we're fading)
        if diff > 0 and momentum <= 0:
            continue
        if diff < 0 and momentum >= 0:
            continue

        # We bet the UNDERDOG (trailing team)
        fav_side = 'home' if diff > 0 else 'away'
        dog_side = 'away' if diff > 0 else 'home'

        # Did the underdog win?
        if dog_side == 'home':
            dog_won = final_home > final_away
            dog_final_margin = final_home - final_away
        else:
            dog_won = final_away > final_home
            dog_final_margin = final_away - final_home

        prob = market_win_prob(lead, mins)
        dog_odds = dog_ml_odds(prob)
        fav_odds_val = fav_ml_odds(prob)

        return {
            'date': game['date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'game_id': game['id'],
            'fav_side': fav_side,
            'dog_side': dog_side,
            'lead': lead,
            'momentum': mom,
            'mins_remaining': mins,
            'entry_home': h,
            'entry_away': a,
            'final_home': final_home,
            'final_away': final_away,
            'dog_won': dog_won,
            'dog_final_margin': dog_final_margin,
            'market_prob': prob,
            'dog_odds': dog_odds,
            'fav_odds': fav_odds_val,
        }

    return None


# =============================================================================
# STRATEGY 2: UNDERDOG SPREAD (MEAN REVERSION)
# =============================================================================

def find_underdog_spread_entry(
    game: dict,
    lead_min: int, lead_max: int,
    mom_min: int,
    time_min: float, time_max: float,
) -> Optional[dict]:
    """
    MEAN REVERSION: Take the underdog on the live spread.

    Thesis: Leads compress by ~4pts on average. If team leads by 12
    and you take the underdog +12, you win when the lead shrinks
    (which happens ~70% of the time).
    """
    states = game['states']
    final_home = game['final_home']
    final_away = game['final_away']

    for state in states:
        mins = state['mins_remaining']
        if not (time_min <= mins <= time_max):
            continue

        h = state['home_score']
        a = state['away_score']
        diff = h - a
        lead = abs(diff)

        if not (lead_min <= lead <= lead_max):
            continue

        momentum = state['home_5min'] - state['away_5min']
        mom = abs(momentum)
        if mom < mom_min:
            continue

        # Momentum aligned with lead (the condition we fade)
        if diff > 0 and momentum <= 0:
            continue
        if diff < 0 and momentum >= 0:
            continue

        # Underdog gets +lead on the spread
        # Underdog covers if final margin (from fav perspective) < lead at entry
        fav_side = 'home' if diff > 0 else 'away'
        if fav_side == 'home':
            fav_final_margin = final_home - final_away
        else:
            fav_final_margin = final_away - final_home

        # Underdog covers spread if favorite's final margin < lead at entry
        # (i.e., the lead shrank)
        dog_covers = fav_final_margin < lead
        push = fav_final_margin == lead
        lead_change = fav_final_margin - lead  # negative = lead shrank

        return {
            'date': game['date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'game_id': game['id'],
            'fav_side': fav_side,
            'lead': lead,
            'momentum': mom,
            'mins_remaining': mins,
            'final_home': final_home,
            'final_away': final_away,
            'fav_final_margin': fav_final_margin,
            'dog_covers': dog_covers,
            'push': push,
            'lead_change': lead_change,
        }

    return None


# =============================================================================
# STRATEGY 3: OVER/UNDER ON REMAINING POINTS
# =============================================================================

def find_total_entry(
    game: dict,
    lead_min: int, lead_max: int,
    mom_min: int,
    time_min: float, time_max: float,
) -> Optional[dict]:
    """
    TOTAL SCORING: When a team has a huge momentum surge, what happens
    to the scoring pace for the rest of the game?

    Thesis: After big momentum surges, the trailing team often changes
    strategy (press, foul, faster pace) leading to MORE total points.
    Or: blowouts lead to garbage time with MORE scoring.
    Or: leading team coasts, resulting in FEWER total points.
    """
    states = game['states']
    final_home = game['final_home']
    final_away = game['final_away']
    total_final = final_home + final_away

    for state in states:
        mins = state['mins_remaining']
        if not (time_min <= mins <= time_max):
            continue

        h = state['home_score']
        a = state['away_score']
        diff = h - a
        lead = abs(diff)

        if not (lead_min <= lead <= lead_max):
            continue

        momentum = state['home_5min'] - state['away_5min']
        mom = abs(momentum)
        if mom < mom_min:
            continue

        # Momentum aligned
        if diff > 0 and momentum <= 0:
            continue
        if diff < 0 and momentum >= 0:
            continue

        current_total = h + a
        remaining_points = total_final - current_total

        # Combined scoring rate in last 10 mins
        combined_10min = state['home_10min'] + state['away_10min']

        # Project remaining points based on current pace
        # Pace = points per minute over last 10 mins
        pace_10min = combined_10min / 10.0 if combined_10min > 0 else 4.0
        projected_remaining = pace_10min * mins

        # Also compute season average pace for comparison
        # NBA average: ~220 total points / 48 min = ~4.58 pts/min
        avg_pace = 4.5
        avg_projected = avg_pace * mins

        return {
            'date': game['date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'game_id': game['id'],
            'lead': lead,
            'momentum': mom,
            'mins_remaining': mins,
            'current_total': current_total,
            'final_total': total_final,
            'remaining_points': remaining_points,
            'pace_10min': pace_10min,
            'projected_remaining': projected_remaining,
            'avg_projected': avg_projected,
            'over_avg': remaining_points > avg_projected,
            'combined_10min': combined_10min,
            'final_home': final_home,
            'final_away': final_away,
        }

    return None


# =============================================================================
# STRATEGY 4: TRAILING TEAM BURST DETECTION
# =============================================================================

def find_trailing_burst_entry(
    game: dict,
    lead_min: int, lead_max: int,
    trailing_3min_min: int,
    time_min: float, time_max: float,
) -> Optional[dict]:
    """
    COMEBACK DETECTION: Bet on trailing team when they're showing
    signs of a run (high 3-min scoring) while still behind.

    Thesis: A trailing team that's scoring heavily in a short window
    is more likely to continue the run. The market is slow to adjust
    live spreads during rapid scoring bursts.
    """
    states = game['states']
    final_home = game['final_home']
    final_away = game['final_away']

    for state in states:
        mins = state['mins_remaining']
        if not (time_min <= mins <= time_max):
            continue

        h = state['home_score']
        a = state['away_score']
        diff = h - a
        lead = abs(diff)

        if not (lead_min <= lead <= lead_max):
            continue

        # Who's trailing?
        if diff > 0:
            trailing_side = 'away'
            trailing_3min = state['away_3min']
            leading_3min = state['home_3min']
        else:
            trailing_side = 'home'
            trailing_3min = state['home_3min']
            leading_3min = state['away_3min']

        # Trailing team must be on a burst
        if trailing_3min < trailing_3min_min:
            continue

        # Momentum must be AGAINST the lead (trailing team scoring more)
        # This is the OPPOSITE of our old strategy
        momentum_5 = state['home_5min'] - state['away_5min']
        if diff > 0 and momentum_5 > 0:
            # Leading team still has 5-min momentum - not a real comeback
            continue
        if diff < 0 and momentum_5 < 0:
            continue

        # Trailing team is on a run despite being behind
        if trailing_side == 'home':
            trailing_won = final_home > final_away
            trailing_final_margin = final_home - final_away
        else:
            trailing_won = final_away > final_home
            trailing_final_margin = final_away - final_home

        fav_final_margin = -trailing_final_margin
        dog_covers_spread = fav_final_margin < lead

        prob = market_win_prob(lead, mins)
        d_odds = dog_ml_odds(prob)

        return {
            'date': game['date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'game_id': game['id'],
            'trailing_side': trailing_side,
            'lead': lead,
            'trailing_3min': trailing_3min,
            'leading_3min': leading_3min,
            'mins_remaining': mins,
            'final_home': final_home,
            'final_away': final_away,
            'trailing_won': trailing_won,
            'trailing_final_margin': trailing_final_margin,
            'dog_covers_spread': dog_covers_spread,
            'dog_odds': d_odds,
            'market_prob': prob,
        }

    return None


# =============================================================================
# STRATEGY 5: LEAD COLLAPSE AFTER PEAK
# =============================================================================

def find_lead_collapse_entry(
    game: dict,
    peak_lead_min: int,
    current_lead_max: int,
    lead_drop_min: int,
    time_min: float, time_max: float,
) -> Optional[dict]:
    """
    LEAD COLLAPSE: When a team had a big lead that has already started
    shrinking, bet on the trailing team to continue the comeback.

    Thesis: A lead that's already collapsing (was 18, now 10) signals
    momentum shift. The market's live spread might not fully reflect
    the psychological shift.
    """
    states = game['states']
    final_home = game['final_home']
    final_away = game['final_away']

    max_lead_by_side = {'home': 0, 'away': 0}

    for state in states:
        h = state['home_score']
        a = state['away_score']
        diff = h - a
        if diff > 0:
            max_lead_by_side['home'] = max(max_lead_by_side['home'], diff)
        elif diff < 0:
            max_lead_by_side['away'] = max(max_lead_by_side['away'], abs(diff))

        mins = state['mins_remaining']
        if not (time_min <= mins <= time_max):
            continue

        lead = abs(diff)
        if lead > current_lead_max:
            continue

        # Which side had the big lead?
        if diff > 0:
            leading_side = 'home'
            peak = max_lead_by_side['home']
        elif diff < 0:
            leading_side = 'away'
            peak = max_lead_by_side['away']
        else:
            continue

        if peak < peak_lead_min:
            continue

        lead_drop = peak - lead
        if lead_drop < lead_drop_min:
            continue

        trailing_side = 'away' if leading_side == 'home' else 'home'

        if trailing_side == 'home':
            trailing_won = final_home > final_away
            trailing_margin = final_home - final_away
        else:
            trailing_won = final_away > final_home
            trailing_margin = final_away - final_home

        # Current spread is the current lead
        dog_covers = abs(final_home - final_away) < lead if (final_home > final_away) == (leading_side == 'home') else True

        prob = market_win_prob(lead, mins)
        d_odds = dog_ml_odds(prob)

        return {
            'date': game['date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'game_id': game['id'],
            'leading_side': leading_side,
            'trailing_side': trailing_side,
            'peak_lead': peak,
            'current_lead': lead,
            'lead_drop': lead_drop,
            'mins_remaining': mins,
            'final_home': final_home,
            'final_away': final_away,
            'trailing_won': trailing_won,
            'trailing_margin': trailing_margin,
            'dog_covers': dog_covers,
            'dog_odds': d_odds,
            'market_prob': prob,
        }

    return None


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def eval_underdog_ml(entries: List[dict], name: str) -> Optional[dict]:
    """Evaluate underdog ML strategy."""
    n = len(entries)
    if n < 5:
        return None

    wins = sum(1 for e in entries if e['dog_won'])
    losses = n - wins
    wr = wins / n

    pnl = 0.0
    bets = []
    for e in entries:
        pay = ml_payout(e['dog_odds'])
        bet_result = pay if e['dog_won'] else -1.0
        pnl += bet_result
        bets.append(bet_result)

    roi = pnl / n * 100
    avg_odds = statistics.mean([e['dog_odds'] for e in entries])
    avg_mkt = statistics.mean([e['market_prob'] for e in entries])

    # Expected value per bet
    avg_payout = statistics.mean([ml_payout(e['dog_odds']) for e in entries])
    ev_per_bet = wr * avg_payout - (1 - wr) * 1.0

    # By season
    by_season = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0.0})
    for e, b in zip(entries, bets):
        year = int(e['date'][:4])
        m = int(e['date'][4:6])
        season = f"{year}-{year+1}" if m >= 10 else f"{year-1}-{year}"
        by_season[season]['n'] += 1
        by_season[season]['wins'] += 1 if e['dog_won'] else 0
        by_season[season]['pnl'] += b

    # Max drawdown
    cum = peak = max_dd = 0.0
    for b in bets:
        cum += b
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd

    return {
        'name': name, 'n': n, 'wins': wins, 'losses': losses,
        'wr': wr * 100, 'pnl': pnl, 'roi': roi,
        'avg_odds': avg_odds, 'avg_mkt': avg_mkt * 100,
        'ev_per_bet': ev_per_bet, 'max_dd': max_dd,
        'by_season': dict(by_season), 'entries': entries,
    }


def eval_underdog_spread(entries: List[dict], name: str) -> Optional[dict]:
    """Evaluate underdog spread (take the points) strategy."""
    n = len(entries)
    if n < 5:
        return None

    covers = sum(1 for e in entries if e['dog_covers'])
    pushes = sum(1 for e in entries if e['push'])
    losses = n - covers - pushes
    cover_rate = covers / n

    # Standard -110 pricing
    active = n - pushes
    pnl = covers * (100 / 110) - losses * 1.0
    roi = pnl / active * 100 if active > 0 else 0

    avg_lead = statistics.mean([e['lead'] for e in entries])
    avg_change = statistics.mean([e['lead_change'] for e in entries])

    # By season
    by_season = defaultdict(lambda: {'n': 0, 'covers': 0, 'pnl': 0.0})
    for e in entries:
        year = int(e['date'][:4])
        m = int(e['date'][4:6])
        season = f"{year}-{year+1}" if m >= 10 else f"{year-1}-{year}"
        by_season[season]['n'] += 1
        by_season[season]['covers'] += 1 if e['dog_covers'] else 0
        sp = (100/110) if e['dog_covers'] else (0 if e['push'] else -1.0)
        by_season[season]['pnl'] += sp

    return {
        'name': name, 'n': n, 'covers': covers, 'pushes': pushes, 'losses': losses,
        'cover_rate': cover_rate * 100, 'pnl': pnl, 'roi': roi,
        'avg_lead': avg_lead, 'avg_change': avg_change,
        'by_season': dict(by_season), 'entries': entries,
    }


def eval_total(entries: List[dict], name: str) -> Optional[dict]:
    """Evaluate over/under total scoring patterns."""
    n = len(entries)
    if n < 10:
        return None

    remaining = [e['remaining_points'] for e in entries]
    projected = [e['avg_projected'] for e in entries]
    paces = [e['pace_10min'] for e in entries]

    avg_remaining = statistics.mean(remaining)
    avg_projected = statistics.mean(projected)
    avg_pace = statistics.mean(paces)

    # How often does actual remaining exceed projected?
    over_count = sum(1 for r, p in zip(remaining, projected) if r > p)
    over_rate = over_count / n * 100

    # If we bet OVER at the projected line
    # Standard over/under is at -110
    over_pnl = over_count * (100/110) - (n - over_count) * 1.0
    over_roi = over_pnl / n * 100

    # If we bet UNDER
    under_count = n - over_count
    under_pnl = under_count * (100/110) - over_count * 1.0
    under_roi = under_pnl / n * 100

    # Deviation from average
    deviations = [r - p for r, p in zip(remaining, projected)]
    avg_dev = statistics.mean(deviations)

    return {
        'name': name, 'n': n,
        'avg_remaining': avg_remaining,
        'avg_projected': avg_projected,
        'avg_pace': avg_pace,
        'over_rate': over_rate,
        'over_pnl': over_pnl, 'over_roi': over_roi,
        'under_pnl': under_pnl, 'under_roi': under_roi,
        'avg_deviation': avg_dev,
        'entries': entries,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 78)
    print("NBA LIVE BETTING: TRADER-STYLE STRATEGY ANALYSIS")
    print("Think like a prop trader, not a typical sports bettor")
    print("=" * 78)

    games = load_all_games()
    if len(games) < 100:
        print("Not enough games")
        return

    # Season split for cross-validation
    season_games = defaultdict(list)
    for g in games:
        year = int(g['date'][:4])
        month = int(g['date'][4:6])
        season = f"{year}-{year+1}" if month >= 10 else f"{year-1}-{year}"
        season_games[season].append(g)

    print(f"\n  Seasons: {', '.join(f'{s} ({len(gs)}G)' for s, gs in sorted(season_games.items()))}")

    # =========================================================================
    # STRATEGY 1: FADE THE FAVORITE (UNDERDOG ML)
    # =========================================================================
    print("\n\n" + "#" * 78)
    print("STRATEGY 1: FADE THE FAVORITE (Underdog ML)")
    print("Thesis: Market overprices momentum-aligned favorites by ~7%")
    print("#" * 78)

    dog_configs = [
        ("Dog ML: L10-14 M12+ T18-24", 10, 14, 12, 18, 24),
        ("Dog ML: L10-16 M12+ T18-24", 10, 16, 12, 18, 24),
        ("Dog ML: L10-19 M12+ T18-24", 10, 19, 12, 18, 24),
        ("Dog ML: L8-12 M12+ T18-24", 8, 12, 12, 18, 24),
        ("Dog ML: L10-14 M10+ T18-24", 10, 14, 10, 18, 24),
        ("Dog ML: L10-14 M14+ T18-24", 10, 14, 14, 18, 24),
        ("Dog ML: L10-14 M12+ T15-24", 10, 14, 12, 15, 24),
        ("Dog ML: L10-14 M12+ T12-24", 10, 14, 12, 12, 24),
        ("Dog ML: L8-14 M12+ T18-24", 8, 14, 12, 18, 24),
        ("Dog ML: L8-14 M10+ T15-24", 8, 14, 10, 15, 24),
        ("Dog ML: L6-10 M10+ T18-24", 6, 10, 10, 18, 24),
        ("Dog ML: L6-12 M10+ T18-24", 6, 12, 10, 18, 24),
        ("Dog ML: L8-12 M10+ T15-24", 8, 12, 10, 15, 24),
        ("Dog ML: L10-14 M16+ T18-24", 10, 14, 16, 18, 24),
        ("Dog ML: L12-16 M14+ T15-21", 12, 16, 14, 15, 21),
    ]

    print(f"\n  {'Strategy':<35} {'G':>4} {'WR%':>5} {'AvgOdds':>8} {'P&L':>7} {'ROI%':>6} {'EV/bet':>7} {'MaxDD':>6}")
    print(f"  {'-'*35} {'-'*4} {'-'*5} {'-'*8} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")

    dog_results = []
    for name, lmin, lmax, mmin, tmin, tmax in dog_configs:
        entries = []
        for game in games:
            e = find_underdog_ml_entry(game, lmin, lmax, mmin, tmin, tmax)
            if e:
                entries.append(e)
        r = eval_underdog_ml(entries, name)
        if r:
            dog_results.append(r)
            print(f"  {r['name']:<35} {r['n']:>4} {r['wr']:>4.1f}% {r['avg_odds']:>+7.0f} "
                  f"{r['pnl']:>+6.1f}u {r['roi']:>+5.1f}% {r['ev_per_bet']:>+6.3f} {r['max_dd']:>5.1f}")

    # Detail best underdog ML
    if dog_results:
        best_dog = max(dog_results, key=lambda x: x['roi'])
        print(f"\n  BEST UNDERDOG ML: {best_dog['name']}")
        print(f"  {best_dog['n']}G, WR={best_dog['wr']:.1f}%, ROI={best_dog['roi']:+.1f}%, P&L={best_dog['pnl']:+.1f}u")
        print(f"  Avg underdog odds: {best_dog['avg_odds']:+.0f}")
        print(f"\n  Cross-season:")
        for s in sorted(best_dog['by_season'].keys()):
            d = best_dog['by_season'][s]
            wr = d['wins']/d['n']*100 if d['n'] > 0 else 0
            roi = d['pnl']/d['n']*100 if d['n'] > 0 else 0
            print(f"    {s}: {d['n']:>3}G, WR={wr:>5.1f}%, P&L={d['pnl']:>+6.1f}u, ROI={roi:>+5.1f}%")

    # =========================================================================
    # STRATEGY 2: UNDERDOG SPREAD (TAKE THE POINTS)
    # =========================================================================
    print("\n\n" + "#" * 78)
    print("STRATEGY 2: UNDERDOG SPREAD (Take the Points)")
    print("Thesis: Leads compress ~4pts on avg. Take underdog +lead at -110")
    print("#" * 78)

    spread_configs = [
        ("Dog Spread: L10-14 M12+ T18-24", 10, 14, 12, 18, 24),
        ("Dog Spread: L10-16 M12+ T18-24", 10, 16, 12, 18, 24),
        ("Dog Spread: L10-19 M12+ T18-24", 10, 19, 12, 18, 24),
        ("Dog Spread: L8-12 M12+ T18-24", 8, 12, 12, 18, 24),
        ("Dog Spread: L10-14 M10+ T18-24", 10, 14, 10, 18, 24),
        ("Dog Spread: L10-14 M14+ T18-24", 10, 14, 14, 18, 24),
        ("Dog Spread: L10-14 M12+ T15-24", 10, 14, 12, 15, 24),
        ("Dog Spread: L10-14 M12+ T12-24", 10, 14, 12, 12, 24),
        ("Dog Spread: L8-14 M12+ T18-24", 8, 14, 12, 18, 24),
        ("Dog Spread: L8-14 M10+ T15-24", 8, 14, 10, 15, 24),
        ("Dog Spread: L6-10 M10+ T18-24", 6, 10, 10, 18, 24),
        ("Dog Spread: L6-12 M10+ T18-24", 6, 12, 10, 18, 24),
        ("Dog Spread: L14-19 M12+ T18-24", 14, 19, 12, 18, 24),
        ("Dog Spread: L10-14 M16+ T18-24", 10, 14, 16, 18, 24),
        ("Dog Spread: L12-16 M14+ T15-21", 12, 16, 14, 15, 21),
        # Also try WITHOUT momentum requirement (just big leads)
        ("Dog Spread: L10-14 M0+ T18-24", 10, 14, 0, 18, 24),
        ("Dog Spread: L14-19 M0+ T18-24", 14, 19, 0, 18, 24),
        ("Dog Spread: L10-19 M0+ T18-24", 10, 19, 0, 18, 24),
    ]

    print(f"\n  {'Strategy':<35} {'G':>4} {'Cover%':>6} {'P&L':>7} {'ROI%':>6} {'AvgLead':>7} {'AvgChg':>7}")
    print(f"  {'-'*35} {'-'*4} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*7}")

    spread_results = []
    for name, lmin, lmax, mmin, tmin, tmax in spread_configs:
        entries = []
        for game in games:
            e = find_underdog_spread_entry(game, lmin, lmax, mmin, tmin, tmax)
            if e:
                entries.append(e)
        r = eval_underdog_spread(entries, name)
        if r:
            spread_results.append(r)
            print(f"  {r['name']:<35} {r['n']:>4} {r['cover_rate']:>5.1f}% "
                  f"{r['pnl']:>+6.1f}u {r['roi']:>+5.1f}% {r['avg_lead']:>6.1f} {r['avg_change']:>+6.1f}")

    # Detail best spread
    if spread_results:
        best_sp = max(spread_results, key=lambda x: x['roi'])
        print(f"\n  BEST UNDERDOG SPREAD: {best_sp['name']}")
        print(f"  {best_sp['n']}G, Cover={best_sp['cover_rate']:.1f}%, ROI={best_sp['roi']:+.1f}%, P&L={best_sp['pnl']:+.1f}u")
        print(f"\n  Cross-season:")
        for s in sorted(best_sp['by_season'].keys()):
            d = best_sp['by_season'][s]
            cr = d['covers']/d['n']*100 if d['n'] > 0 else 0
            roi = d['pnl']/d['n']*100 if d['n'] > 0 else 0
            print(f"    {s}: {d['n']:>3}G, Cover={cr:>5.1f}%, P&L={d['pnl']:>+6.1f}u, ROI={roi:>+5.1f}%")

    # =========================================================================
    # STRATEGY 3: OVER/UNDER REMAINING POINTS
    # =========================================================================
    print("\n\n" + "#" * 78)
    print("STRATEGY 3: OVER/UNDER REMAINING POINTS")
    print("Thesis: Momentum surges affect remaining-game scoring pace")
    print("#" * 78)

    total_configs = [
        ("Total: L10-14 M12+ T18-24", 10, 14, 12, 18, 24),
        ("Total: L10-19 M12+ T18-24", 10, 19, 12, 18, 24),
        ("Total: L10-14 M10+ T18-24", 10, 14, 10, 18, 24),
        ("Total: L6-10 M10+ T18-24", 6, 10, 10, 18, 24),
        ("Total: L10-14 M14+ T18-24", 10, 14, 14, 18, 24),
        ("Total: L10-19 M10+ T15-24", 10, 19, 10, 15, 24),
        ("Total: L14-19 M12+ T18-24", 14, 19, 12, 18, 24),
        ("Total: L8-14 M12+ T15-24", 8, 14, 12, 15, 24),
        # High momentum only
        ("Total: L10-19 M16+ T18-24", 10, 19, 16, 18, 24),
        ("Total: L10-14 M16+ T15-24", 10, 14, 16, 15, 24),
    ]

    print(f"\n  {'Strategy':<35} {'G':>4} {'AvgRem':>6} {'AvgProj':>7} {'Dev':>5} {'Over%':>5} {'O_ROI':>6} {'U_ROI':>6} {'Pace':>5}")
    print(f"  {'-'*35} {'-'*4} {'-'*6} {'-'*7} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*5}")

    for name, lmin, lmax, mmin, tmin, tmax in total_configs:
        entries = []
        for game in games:
            e = find_total_entry(game, lmin, lmax, mmin, tmin, tmax)
            if e:
                entries.append(e)
        r = eval_total(entries, name)
        if r:
            print(f"  {r['name']:<35} {r['n']:>4} {r['avg_remaining']:>5.1f} {r['avg_projected']:>6.1f} "
                  f"{r['avg_deviation']:>+4.1f} {r['over_rate']:>4.0f}% {r['over_roi']:>+5.1f}% "
                  f"{r['under_roi']:>+5.1f}% {r['avg_pace']:>4.1f}")

    # =========================================================================
    # STRATEGY 4: TRAILING TEAM BURST
    # =========================================================================
    print("\n\n" + "#" * 78)
    print("STRATEGY 4: TRAILING TEAM SCORING BURST")
    print("Thesis: Trailing team on a 3-min run signals a live comeback")
    print("#" * 78)

    burst_configs = [
        ("Burst: L8-14 Trail3m8+ T18-24", 8, 14, 8, 18, 24),
        ("Burst: L8-14 Trail3m10+ T18-24", 8, 14, 10, 18, 24),
        ("Burst: L8-14 Trail3m12+ T18-24", 8, 14, 12, 18, 24),
        ("Burst: L10-16 Trail3m8+ T18-24", 10, 16, 8, 18, 24),
        ("Burst: L10-16 Trail3m10+ T18-24", 10, 16, 10, 18, 24),
        ("Burst: L10-16 Trail3m12+ T18-24", 10, 16, 12, 18, 24),
        ("Burst: L6-12 Trail3m10+ T15-24", 6, 12, 10, 15, 24),
        ("Burst: L6-12 Trail3m12+ T15-24", 6, 12, 12, 15, 24),
        ("Burst: L8-14 Trail3m10+ T12-24", 8, 14, 10, 12, 24),
        ("Burst: L10-14 Trail3m10+ T15-21", 10, 14, 10, 15, 21),
    ]

    print(f"\n  {'Strategy':<35} {'G':>4} {'ML_WR':>5} {'ML_ROI':>6} {'SpCov':>5} {'SP_ROI':>6} {'AvgOdds':>8}")
    print(f"  {'-'*35} {'-'*4} {'-'*5} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")

    for name, lmin, lmax, burst_min, tmin, tmax in burst_configs:
        entries = []
        for game in games:
            e = find_trailing_burst_entry(game, lmin, lmax, burst_min, tmin, tmax)
            if e:
                entries.append(e)

        n = len(entries)
        if n < 5:
            continue

        ml_wins = sum(1 for e in entries if e['trailing_won'])
        ml_wr = ml_wins / n * 100
        ml_pnl = sum(ml_payout(e['dog_odds']) if e['trailing_won'] else -1.0 for e in entries)
        ml_roi = ml_pnl / n * 100

        sp_covers = sum(1 for e in entries if e['dog_covers_spread'])
        sp_rate = sp_covers / n * 100
        sp_pnl = sp_covers * (100/110) - (n - sp_covers) * 1.0
        sp_roi = sp_pnl / n * 100

        avg_odds = statistics.mean([e['dog_odds'] for e in entries])

        print(f"  {name:<35} {n:>4} {ml_wr:>4.1f}% {ml_roi:>+5.1f}% "
              f"{sp_rate:>4.1f}% {sp_roi:>+5.1f}% {avg_odds:>+7.0f}")

    # =========================================================================
    # STRATEGY 5: LEAD COLLAPSE
    # =========================================================================
    print("\n\n" + "#" * 78)
    print("STRATEGY 5: LEAD COLLAPSE DETECTION")
    print("Thesis: Lead that peaked high but has dropped signals momentum shift")
    print("#" * 78)

    collapse_configs = [
        ("Collapse: Peak15+ Now<10 Drop6+ T18-24", 15, 10, 6, 18, 24),
        ("Collapse: Peak15+ Now<10 Drop8+ T18-24", 15, 10, 8, 18, 24),
        ("Collapse: Peak18+ Now<12 Drop8+ T18-24", 18, 12, 8, 18, 24),
        ("Collapse: Peak15+ Now<10 Drop6+ T12-24", 15, 10, 6, 12, 24),
        ("Collapse: Peak12+ Now<8 Drop5+ T18-24", 12, 8, 5, 18, 24),
        ("Collapse: Peak20+ Now<12 Drop10+ T15-24", 20, 12, 10, 15, 24),
        ("Collapse: Peak15+ Now<8 Drop8+ T15-24", 15, 8, 8, 15, 24),
        ("Collapse: Peak18+ Now<10 Drop10+ T12-24", 18, 10, 10, 12, 24),
    ]

    print(f"\n  {'Strategy':<42} {'G':>4} {'ML_WR':>5} {'ML_ROI':>6} {'SpCov':>5} {'SP_ROI':>6} {'AvgOdds':>8}")
    print(f"  {'-'*42} {'-'*4} {'-'*5} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")

    for name, peak_min, cur_max, drop_min, tmin, tmax in collapse_configs:
        entries = []
        for game in games:
            e = find_lead_collapse_entry(game, peak_min, cur_max, drop_min, tmin, tmax)
            if e:
                entries.append(e)

        n = len(entries)
        if n < 5:
            continue

        ml_wins = sum(1 for e in entries if e['trailing_won'])
        ml_wr = ml_wins / n * 100
        ml_pnl = sum(ml_payout(e['dog_odds']) if e['trailing_won'] else -1.0 for e in entries)
        ml_roi = ml_pnl / n * 100

        sp_covers = sum(1 for e in entries if e['dog_covers'])
        sp_rate = sp_covers / n * 100
        sp_pnl = sp_covers * (100/110) - (n - sp_covers) * 1.0
        sp_roi = sp_pnl / n * 100

        avg_odds = statistics.mean([e['dog_odds'] for e in entries])

        print(f"  {name:<42} {n:>4} {ml_wr:>4.1f}% {ml_roi:>+5.1f}% "
              f"{sp_rate:>4.1f}% {sp_roi:>+5.1f}% {avg_odds:>+7.0f}")

    # =========================================================================
    # MEGA GRID SEARCH: UNDERDOG SPREAD (the most promising angle)
    # =========================================================================
    print("\n\n" + "#" * 78)
    print("MEGA GRID SEARCH: UNDERDOG SPREAD")
    print("(The most promising angle - leads compress ~4pts on average)")
    print("#" * 78)

    grid_results = []
    for lmin in [4, 6, 8, 10, 12, 14]:
        for lmax in [8, 10, 12, 14, 16, 19, 24]:
            if lmax <= lmin: continue
            for mmin in [0, 6, 8, 10, 12, 14, 16]:
                for tmin in [6, 12, 15, 18, 21]:
                    for tmax in [12, 15, 18, 21, 24, 30]:
                        if tmax <= tmin: continue
                        entries = []
                        for game in games:
                            e = find_underdog_spread_entry(game, lmin, lmax, mmin, tmin, tmax)
                            if e:
                                entries.append(e)
                        n = len(entries)
                        if n < 15:
                            continue
                        covers = sum(1 for e in entries if e['dog_covers'])
                        pushes = sum(1 for e in entries if e['push'])
                        active = n - pushes
                        if active < 10: continue
                        pnl = covers * (100/110) - (n - covers - pushes) * 1.0
                        roi = pnl / active * 100
                        cr = covers / n * 100
                        grid_results.append({
                            'label': f"L{lmin}-{lmax} M{mmin}+ T{tmin}-{tmax}",
                            'n': n, 'covers': covers, 'pushes': pushes,
                            'cover_rate': cr, 'pnl': pnl, 'roi': roi,
                        })

    grid_results.sort(key=lambda x: x['roi'], reverse=True)

    for min_n, count, label in [(15, 20, "15+"), (25, 15, "25+"), (40, 15, "40+"), (60, 10, "60+")]:
        filtered = [r for r in grid_results if r['n'] >= min_n]
        if not filtered: continue
        filtered.sort(key=lambda x: x['roi'], reverse=True)
        print(f"\n  TOP {count} UNDERDOG SPREAD (min {label} games):")
        print(f"  {'Conditions':<28} {'G':>4} {'Cover%':>6} {'P&L':>7} {'ROI%':>6}")
        print(f"  {'-'*28} {'-'*4} {'-'*6} {'-'*7} {'-'*6}")
        for r in filtered[:count]:
            print(f"  {r['label']:<28} {r['n']:>4} {r['cover_rate']:>5.1f}% {r['pnl']:>+6.1f}u {r['roi']:>+5.1f}%")

    # =========================================================================
    # MEGA GRID: UNDERDOG ML
    # =========================================================================
    print("\n\n" + "#" * 78)
    print("MEGA GRID SEARCH: UNDERDOG ML")
    print("#" * 78)

    ml_grid = []
    for lmin in [4, 6, 8, 10, 12]:
        for lmax in [8, 10, 12, 14, 16, 19]:
            if lmax <= lmin: continue
            for mmin in [0, 6, 8, 10, 12, 14, 16]:
                for tmin in [6, 12, 15, 18]:
                    for tmax in [12, 15, 18, 21, 24]:
                        if tmax <= tmin: continue
                        entries = []
                        for game in games:
                            e = find_underdog_ml_entry(game, lmin, lmax, mmin, tmin, tmax)
                            if e:
                                entries.append(e)
                        r = eval_underdog_ml(entries, f"L{lmin}-{lmax} M{mmin}+ T{tmin}-{tmax}")
                        if r and r['n'] >= 15:
                            ml_grid.append(r)

    ml_grid.sort(key=lambda x: x['roi'], reverse=True)

    for min_n, count, label in [(15, 20, "15+"), (25, 15, "25+"), (40, 10, "40+")]:
        filtered = [r for r in ml_grid if r['n'] >= min_n]
        if not filtered: continue
        filtered.sort(key=lambda x: x['roi'], reverse=True)
        print(f"\n  TOP {count} UNDERDOG ML (min {label} games):")
        print(f"  {'Conditions':<28} {'G':>4} {'WR%':>5} {'AvgOdds':>8} {'P&L':>7} {'ROI%':>6}")
        print(f"  {'-'*28} {'-'*4} {'-'*5} {'-'*8} {'-'*7} {'-'*6}")
        for r in filtered[:count]:
            print(f"  {r['name']:<28} {r['n']:>4} {r['wr']:>4.1f}% {r['avg_odds']:>+7.0f} "
                  f"{r['pnl']:>+6.1f}u {r['roi']:>+5.1f}%")

    # =========================================================================
    # CROSS-SEASON VALIDATION OF TOP RESULTS
    # =========================================================================
    print("\n\n" + "#" * 78)
    print("CROSS-SEASON VALIDATION OF TOP STRATEGIES")
    print("#" * 78)

    # Top underdog spread configs to validate
    if grid_results:
        top_spread = grid_results[:5]
        print("\n  TOP UNDERDOG SPREAD - BY SEASON:")
        for r in top_spread:
            label = r['label']
            # Parse config from label
            parts = label.replace('L', '').replace('M', ' ').replace('T', ' ').replace('+', '').replace('-', ' ').split()
            if len(parts) >= 5:
                lmin, lmax, mmin, tmin, tmax = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            else:
                continue

            print(f"\n  {label} (overall: {r['n']}G, {r['cover_rate']:.1f}% cover, {r['roi']:+.1f}% ROI):")
            for season in sorted(season_games.keys()):
                entries = []
                for game in season_games[season]:
                    e = find_underdog_spread_entry(game, lmin, lmax, mmin, tmin, tmax)
                    if e:
                        entries.append(e)
                n = len(entries)
                if n < 3:
                    print(f"    {season}: <3 games")
                    continue
                covers = sum(1 for e in entries if e['dog_covers'])
                pushes = sum(1 for e in entries if e['push'])
                active = n - pushes
                pnl = covers * (100/110) - (n - covers - pushes) * 1.0
                cr = covers / n * 100
                roi = pnl / active * 100 if active > 0 else 0
                print(f"    {season}: {n:>3}G, Cover={cr:>5.1f}%, P&L={pnl:>+6.1f}u, ROI={roi:>+5.1f}%")

    # Top underdog ML configs
    if ml_grid:
        top_ml = ml_grid[:5]
        print("\n  TOP UNDERDOG ML - BY SEASON:")
        for r in top_ml:
            label = r['name']
            parts = label.replace('L', '').replace('M', ' ').replace('T', ' ').replace('+', '').replace('-', ' ').split()
            if len(parts) >= 5:
                lmin, lmax, mmin, tmin, tmax = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            else:
                continue

            print(f"\n  {label} (overall: {r['n']}G, WR={r['wr']:.1f}%, {r['roi']:+.1f}% ROI):")
            for season in sorted(season_games.keys()):
                entries = []
                for game in season_games[season]:
                    e = find_underdog_ml_entry(game, lmin, lmax, mmin, tmin, tmax)
                    if e:
                        entries.append(e)
                sr = eval_underdog_ml(entries, f"{label} [{season}]")
                if sr:
                    print(f"    {season}: {sr['n']:>3}G, WR={sr['wr']:>5.1f}%, P&L={sr['pnl']:>+6.1f}u, ROI={sr['roi']:>+5.1f}%")
                else:
                    print(f"    {season}: <5 games")

    # =========================================================================
    # GAME-BY-GAME FOR BEST STRATEGIES
    # =========================================================================
    if dog_results:
        best = max(dog_results, key=lambda x: x['roi'])
        if best['roi'] > 0:
            print(f"\n\n{'='*78}")
            print(f"GAME-BY-GAME: {best['name']} (ROI={best['roi']:+.1f}%)")
            print(f"{'='*78}")
            print(f"\n  {'Date':<10} {'Matchup':<14} {'Dog':<5} {'L':>3} {'M':>3} {'T':>5} {'Odds':>7} {'Final':>10} {'Won':>4} {'P&L':>6}")
            print(f"  {'-'*10} {'-'*14} {'-'*5} {'-'*3} {'-'*3} {'-'*5} {'-'*7} {'-'*10} {'-'*4} {'-'*6}")
            cum = 0.0
            for e in best['entries']:
                pay = ml_payout(e['dog_odds'])
                bet_r = pay if e['dog_won'] else -1.0
                cum += bet_r
                matchup = f"{e['home_team']}v{e['away_team']}"
                final = f"{e['final_home']}-{e['final_away']}"
                mark = "W" if e['dog_won'] else "L"
                print(f"  {e['date']:<10} {matchup:<14} {e['dog_side']:<5} {e['lead']:>3} {e['momentum']:>3} "
                      f"{e['mins_remaining']:>5.1f} {e['dog_odds']:>+6.0f} {final:>10} {mark:>4} {bet_r:>+5.2f}")
            print(f"\n  CUMULATIVE: {cum:>+.2f}u over {best['n']} bets = {best['roi']:+.1f}% ROI")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n\n" + "=" * 78)
    print("FINAL TRADER ASSESSMENT")
    print("=" * 78)

    print(f"""
  DATA: {len(games)} real NBA games, {len(set(g['date'][:6] for g in games))} months

  STRATEGIES TESTED:
  1. Underdog ML (fade overpriced favorite)
  2. Underdog Spread (mean reversion on leads)
  3. Over/Under remaining points (scoring pace)
  4. Trailing team burst detection (comeback momentum)
  5. Lead collapse detection (momentum shift)
  + Comprehensive grid search across all parameters
    """)

    # Summarize what worked
    any_profitable = False
    if dog_results:
        profitable_dogs = [r for r in dog_results if r['roi'] > 0]
        if profitable_dogs:
            print("  PROFITABLE UNDERDOG ML STRATEGIES:")
            for r in sorted(profitable_dogs, key=lambda x: x['roi'], reverse=True):
                print(f"    {r['name']}: {r['n']}G, WR={r['wr']:.1f}%, ROI={r['roi']:+.1f}%")
            any_profitable = True

    if spread_results:
        profitable_sp = [r for r in spread_results if r['roi'] > 0]
        if profitable_sp:
            print("  PROFITABLE UNDERDOG SPREAD STRATEGIES:")
            for r in sorted(profitable_sp, key=lambda x: x['roi'], reverse=True):
                print(f"    {r['name']}: {r['n']}G, Cover={r['cover_rate']:.1f}%, ROI={r['roi']:+.1f}%")
            any_profitable = True

    if grid_results:
        profitable_grid = [r for r in grid_results if r['roi'] > 0 and r['n'] >= 25]
        if profitable_grid:
            print(f"\n  PROFITABLE SPREAD GRID RESULTS (25+ games):")
            for r in profitable_grid[:10]:
                print(f"    {r['label']}: {r['n']}G, Cover={r['cover_rate']:.1f}%, ROI={r['roi']:+.1f}%")
            any_profitable = True

    if ml_grid:
        profitable_ml_grid = [r for r in ml_grid if r['roi'] > 0 and r['n'] >= 25]
        if profitable_ml_grid:
            print(f"\n  PROFITABLE ML GRID RESULTS (25+ games):")
            for r in profitable_ml_grid[:10]:
                print(f"    {r['name']}: {r['n']}G, WR={r['wr']:.1f}%, ROI={r['roi']:+.1f}%")
            any_profitable = True

    if not any_profitable:
        print("  NO PROFITABLE STRATEGIES FOUND WITH 25+ GAME SAMPLES.")

    print(f"\n{'='*78}")


if __name__ == '__main__':
    main()
