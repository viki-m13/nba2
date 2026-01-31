"""
NBA LIVE BETTING: EXPERIMENTAL STRATEGY LAB
=============================================

Approaches NOT yet tested in this codebase:

EXPERIMENT 1: LEAD COMPRESSION EXIT
  - Don't hold to end of game
  - Enter when 8-layer model fires, EXIT when lead narrows by X points
  - Captures partial comebacks (much more frequent than full comebacks)
  - Simulates live moneyline value capture (odds improve as lead shrinks)

EXPERIMENT 2: BIDIRECTIONAL MODEL
  - Don't always fade the leader
  - When layers say "trending" (Hurst > 0.55, positive autocorr) → bet leader
  - When layers say "mean-reverting" (Hurst < 0.45, negative autocorr) → fade leader
  - Let the data decide direction

EXPERIMENT 3: LAYER INTERACTION COMBOS
  - Test specific 2-3 layer combinations instead of composite
  - OU + Hurst together might be more predictive than 8-layer average
  - Find which combos actually carry signal vs which add noise

EXPERIMENT 4: SCORING BURST TRIGGERS
  - Detect specific real-time events: 8-0 run, 10-0 run, etc.
  - Bet immediately AGAINST the run (contrarian)
  - Tighter time window, higher frequency

EXPERIMENT 5: HALFTIME TRANSITION
  - Q3 is known as the "adjustment quarter"
  - Bet on lead compression specifically in first 4 min of Q3
  - Teams trailing at half make adjustments

EXPERIMENT 6: BLOWOUT COMPRESSION
  - Leads of 20+ points compress in ~70% of games
  - Don't need a comeback, just need lead to shrink
  - Capture the compression via live odds movement

EXPERIMENT 7: THE "WHEN TO BET THE FAVORITE" MODEL
  - Flip the question: when does betting the LEADER have edge?
  - Low Hurst + high autocorr + acceleration = leader continues
  - Maybe the edge is on the other side in specific conditions

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
# DATA LOADING (reused from quant_proprietary_strategy.py)
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
    except Exception:
        return None


def build_game_from_pbp(pbp_data: dict) -> Optional[dict]:
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

    prev_home = 0
    prev_away = 0
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

        home_pts = home_score - prev_home
        away_pts = away_score - prev_away
        prev_home = home_score
        prev_away = away_score

        states.append({
            'mins_remaining': mins_remaining,
            'mins_elapsed': round(48.0 - mins_remaining, 2),
            'home_score': home_score,
            'away_score': away_score,
            'lead': home_score - away_score,
            'home_pts': home_pts,
            'away_pts': away_pts,
        })

    if len(states) < 20:
        return None

    game_id = pbp_data.get('header', {}).get('id', '')

    return {
        'id': game_id,
        'date': game_date,
        'home_team': home_team,
        'away_team': away_team,
        'final_home': final_home,
        'final_away': final_away,
        'final_lead': final_home - final_away,
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
        except Exception:
            continue

    seen = set()
    unique = []
    for g in games:
        if g['id'] not in seen:
            seen.add(g['id'])
            unique.append(g)
    unique.sort(key=lambda g: g['date'])
    print(f"  Loaded {len(unique)} unique games ({unique[0]['date']} to {unique[-1]['date']})")
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


def market_win_prob(lead: float, mins_remaining: float) -> float:
    if mins_remaining <= 0:
        return 1.0 if lead > 0 else (0.0 if lead < 0 else 0.5)
    z = abs(lead) / (2.6 * math.sqrt(max(mins_remaining, 0.5)))
    return max(0.51, min(0.998, normal_cdf(z)))


def dog_odds_american(prob: float, vig: float = 0.045) -> float:
    dog_true = 1 - prob
    dog_implied = dog_true * (1 + vig)
    dog_implied = max(0.01, min(0.99, dog_implied))
    return ((1 - dog_implied) / dog_implied) * 100


def fav_odds_american(prob: float, vig: float = 0.045) -> float:
    fav_implied = prob * (1 + vig)
    fav_implied = max(0.01, min(0.99, fav_implied))
    if fav_implied >= 0.5:
        return -(fav_implied / (1 - fav_implied)) * 100
    else:
        return ((1 - fav_implied) / fav_implied) * 100


def ml_payout(odds: float) -> float:
    if odds < 0:
        return 100 / abs(odds)
    else:
        return odds / 100


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_to_bars(states: List[dict], bar_size: float = 0.5) -> Tuple[List[float], List[float]]:
    if not states:
        return [], []

    max_elapsed = max(s['mins_elapsed'] for s in states)
    n_bars = int(max_elapsed / bar_size) + 1
    sorted_states = sorted(states, key=lambda s: s['mins_elapsed'])

    times = []
    leads = []
    current_lead = 0
    state_idx = 0

    for i in range(n_bars):
        t = i * bar_size
        while state_idx < len(sorted_states) and sorted_states[state_idx]['mins_elapsed'] <= t + bar_size:
            current_lead = sorted_states[state_idx]['lead']
            state_idx += 1
        times.append(t)
        leads.append(current_lead)

    return times, leads


# =============================================================================
# MINI SIGNAL LAYERS (streamlined versions for speed)
# =============================================================================

def compute_hurst(series: List[float], min_window: int = 4) -> float:
    n = len(series)
    if n < min_window * 2:
        return 0.5

    returns = [series[i] - series[i - 1] for i in range(1, n)]
    if not returns:
        return 0.5

    log_rs = []
    log_n = []

    for ws in range(min_window, n // 2 + 1):
        rs_values = []
        for start in range(0, len(returns) - ws + 1, max(1, ws // 2)):
            window = returns[start:start + ws]
            if len(window) < min_window:
                continue
            mean_r = sum(window) / len(window)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in window) / len(window))
            if std_r < 1e-10:
                continue
            cum_dev = []
            running = 0
            for r in window:
                running += (r - mean_r)
                cum_dev.append(running)
            R = max(cum_dev) - min(cum_dev)
            rs = R / std_r
            if rs > 0:
                rs_values.append(rs)
        if rs_values:
            avg_rs = sum(rs_values) / len(rs_values)
            if avg_rs > 0:
                log_rs.append(math.log(avg_rs))
                log_n.append(math.log(ws))

    if len(log_rs) < 3:
        return 0.5

    n_pts = len(log_rs)
    sum_x = sum(log_n)
    sum_y = sum(log_rs)
    sum_xy = sum(x * y for x, y in zip(log_n, log_rs))
    sum_xx = sum(x * x for x in log_n)

    denom = n_pts * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-10:
        return 0.5

    H = (n_pts * sum_xy - sum_x * sum_y) / denom
    return max(0.0, min(1.0, H))


def compute_autocorr(leads: List[float], lag: int = 1, lookback: int = 20) -> float:
    if len(leads) < lookback + lag + 1:
        subset = leads
    else:
        subset = leads[-lookback:]

    if len(subset) < lag + 5:
        return 0.0

    changes = [subset[i] - subset[i - 1] for i in range(1, len(subset))]
    if len(changes) < lag + 3:
        return 0.0

    mean_c = statistics.mean(changes)
    var_c = sum((c - mean_c) ** 2 for c in changes) / len(changes)
    if var_c < 1e-10:
        return 0.0

    cov = sum((changes[i] - mean_c) * (changes[i - lag] - mean_c)
               for i in range(lag, len(changes))) / (len(changes) - lag)

    return cov / var_c


def compute_velocity_accel(leads: List[float], dt: float = 0.5, lookback: int = 10) -> Tuple[float, float]:
    if len(leads) < lookback + 2:
        return 0.0, 0.0

    subset = leads[-lookback:]
    velocities = [(subset[i] - subset[i - 1]) / dt for i in range(1, len(subset))]
    velocity = statistics.mean(velocities[-3:]) if len(velocities) >= 3 else 0

    if len(velocities) >= 4:
        accels = [(velocities[i] - velocities[i - 1]) / dt for i in range(1, len(velocities))]
        acceleration = statistics.mean(accels[-3:]) if len(accels) >= 3 else 0
    else:
        acceleration = 0

    return velocity, acceleration


def detect_scoring_run(states: List[dict], up_to_idx: int, lookback: int = 10) -> dict:
    """Detect if a scoring run is happening RIGHT NOW."""
    start = max(0, up_to_idx - lookback)
    subset = states[start:up_to_idx + 1]

    if len(subset) < 3:
        return {'run_team': 'none', 'run_pts': 0, 'run_plays': 0, 'run_time': 0}

    # Walk backward from current state to find the active run
    run_pts = 0
    run_plays = 0
    run_team = 'none'

    for i in range(len(subset) - 1, -1, -1):
        s = subset[i]
        if s['home_pts'] > 0 and s['away_pts'] == 0:
            scorer = 'home'
        elif s['away_pts'] > 0 and s['home_pts'] == 0:
            scorer = 'away'
        else:
            break  # Both scored or neither — run broken

        if run_team == 'none':
            run_team = scorer
        elif scorer != run_team:
            break  # Different team scored — run is over

        run_pts += s['home_pts'] if scorer == 'home' else s['away_pts']
        run_plays += 1

    run_time = 0
    if run_plays >= 2:
        run_time = subset[-1]['mins_elapsed'] - subset[-1 - min(run_plays, len(subset) - 1)]['mins_elapsed']

    return {
        'run_team': run_team,
        'run_pts': run_pts,
        'run_plays': run_plays,
        'run_time': run_time,
    }


def compute_volatility_ratio(leads: List[float], short: int = 10, long: int = 30) -> float:
    if len(leads) < short + 1:
        return 1.0

    changes = [abs(leads[i] - leads[i - 1]) for i in range(1, len(leads))]
    vol_short = statistics.mean(changes[-short:]) if len(changes) >= short else statistics.mean(changes)

    if len(changes) >= long:
        vol_long = statistics.mean(changes[-long:])
    else:
        vol_long = statistics.mean(changes)

    return vol_short / vol_long if vol_long > 0 else 1.0


# =============================================================================
# EXPERIMENT 1: LEAD COMPRESSION EXIT
# =============================================================================

def experiment_lead_compression(
    games: List[dict],
    min_lead: int = 10,
    max_lead: int = 25,
    min_elapsed: float = 8.0,
    max_elapsed: float = 38.0,
    compression_target: float = 0.4,  # exit when lead compresses by this fraction
    max_hold_time: float = 12.0,       # max minutes to hold the position
    vig: float = 0.045,
    label: str = "",
) -> dict:
    """
    IDEA: Don't need the underdog to WIN. Just need the lead to SHRINK.

    Enter when lead is large (10-25 pts) with mean-reversion signals.
    Exit when lead compresses by target fraction OR max hold time.

    This simulates capturing live odds movement — as the lead shrinks,
    the underdog's moneyline odds improve, and we profit from the move.

    P&L model:
      Entry odds = underdog ML at entry lead
      Exit odds = underdog ML at exit lead (or game end)
      Profit = difference in implied probability * scaling factor
    """
    trades = []
    games_checked = 0

    for game in games:
        states = game['states']
        times, leads = resample_to_bars(states, bar_size=0.5)

        if len(leads) < 30:
            continue
        games_checked += 1

        traded = False
        for si, state in enumerate(states):
            if traded:
                break

            elapsed = state['mins_elapsed']
            if elapsed < min_elapsed or elapsed > max_elapsed:
                continue

            lead = state['lead']
            abs_lead = abs(lead)
            if abs_lead < min_lead or abs_lead > max_lead:
                continue

            bar_idx = int(elapsed / 0.5)
            if bar_idx >= len(leads):
                bar_idx = len(leads) - 1
            if bar_idx < 20:
                continue

            leads_so_far = leads[:bar_idx + 1]

            # Quick regime check: only enter in mean-reverting regime
            H = compute_hurst(leads_so_far)
            if H > 0.52:  # not mean-reverting
                continue

            ac = compute_autocorr(leads_so_far)
            if ac > 0.05:  # not mean-reverting
                continue

            vol_ratio = compute_volatility_ratio(leads_so_far)
            if vol_ratio < 0.8:  # low vol, stable lead
                continue

            # Entry conditions met — now simulate the exit
            entry_lead = abs_lead
            entry_elapsed = elapsed
            entry_prob = market_win_prob(entry_lead, 48.0 - entry_elapsed)
            entry_dog_prob = 1 - entry_prob

            # Walk forward looking for compression or timeout
            exit_lead = None
            exit_elapsed = None
            exit_reason = None

            for future_si in range(si + 1, len(states)):
                fs = states[future_si]
                future_lead = abs(fs['lead'])
                future_elapsed = fs['mins_elapsed']
                hold_time = future_elapsed - entry_elapsed

                # Check if lead flipped (underdog took the lead)
                if lead > 0 and fs['lead'] < 0:
                    exit_lead = fs['lead']
                    exit_elapsed = future_elapsed
                    exit_reason = 'flip'
                    break
                elif lead < 0 and fs['lead'] > 0:
                    exit_lead = fs['lead']
                    exit_elapsed = future_elapsed
                    exit_reason = 'flip'
                    break

                # Check compression target
                compression = 1 - (future_lead / entry_lead) if entry_lead > 0 else 0
                if compression >= compression_target:
                    exit_lead = future_lead
                    exit_elapsed = future_elapsed
                    exit_reason = 'target'
                    break

                # Check max hold time
                if hold_time >= max_hold_time:
                    exit_lead = future_lead
                    exit_elapsed = future_elapsed
                    exit_reason = 'timeout'
                    break

            # If we ran out of states, use final game state
            if exit_lead is None:
                exit_lead = abs(game['final_lead'])
                exit_elapsed = 48.0
                exit_reason = 'game_end'

            # P&L calculation:
            # Model this as capturing the probability shift.
            # Entry: underdog prob = entry_dog_prob
            # Exit: underdog prob = exit_dog_prob
            # If we bet $100 on underdog ML at entry, the value of our bet changes
            # as the underdog probability changes.
            # Simple model: profit = (exit_dog_prob - entry_dog_prob) / entry_dog_prob
            # (percentage gain on our position)

            # More realistic: use odds movement
            # Entry odds for underdog
            entry_dog_odds = dog_odds_american(entry_prob, vig)
            entry_payout = ml_payout(entry_dog_odds)

            # At exit, what's the underdog situation?
            exit_mins_remaining = 48.0 - exit_elapsed
            if exit_reason == 'flip':
                # Underdog took the lead — our bet is now the favorite
                # We'd cash out at a profit
                exit_prob = market_win_prob(abs(exit_lead), exit_mins_remaining)
                # Our team is now leading, so their win prob is exit_prob
                # Value of our bet: we could sell at ~exit_prob * (1+entry_payout) - 1
                # Simplified: profit based on probability swing
                exit_value = exit_prob * (1 + entry_payout) - 1
                pnl = exit_value
            elif exit_reason == 'target':
                # Lead compressed but didn't flip
                # Our underdog's probability improved
                if lead > 0:
                    # We bet away, home still leads by less
                    exit_prob_leader = market_win_prob(exit_lead, exit_mins_remaining)
                    exit_dog_prob = 1 - exit_prob_leader
                else:
                    exit_prob_leader = market_win_prob(exit_lead, exit_mins_remaining)
                    exit_dog_prob = 1 - exit_prob_leader

                # Cashout value: our bet improved from entry_dog_prob to exit_dog_prob
                # With entry_payout locked in, our EV improved
                prob_improvement = exit_dog_prob - entry_dog_prob
                # Scale to actual payout: if we would have won entry_payout,
                # and probability improved by X%, our position gained roughly:
                pnl = prob_improvement * (1 + entry_payout) * 0.85  # 85% cashout efficiency
            elif exit_reason == 'timeout':
                # Held max time, lead may or may not have compressed
                if lead > 0:
                    exit_prob_leader = market_win_prob(exit_lead, exit_mins_remaining)
                    exit_dog_prob = 1 - exit_prob_leader
                else:
                    exit_prob_leader = market_win_prob(exit_lead, exit_mins_remaining)
                    exit_dog_prob = 1 - exit_prob_leader

                prob_improvement = exit_dog_prob - entry_dog_prob
                pnl = prob_improvement * (1 + entry_payout) * 0.85
            else:
                # Game ended — standard ML outcome
                if lead > 0:
                    won = game['final_lead'] < 0  # we bet away
                else:
                    won = game['final_lead'] > 0  # we bet home
                pnl = entry_payout if won else -1.0

            trades.append({
                'game': game,
                'entry_lead': entry_lead,
                'entry_elapsed': entry_elapsed,
                'entry_prob': entry_prob,
                'entry_odds': entry_dog_odds,
                'exit_lead': exit_lead,
                'exit_elapsed': exit_elapsed,
                'exit_reason': exit_reason,
                'hurst': H,
                'autocorr': ac,
                'vol_ratio': vol_ratio,
                'pnl': pnl,
                'hold_time': exit_elapsed - entry_elapsed,
            })
            traded = True

    return summarize_trades(trades, games_checked, label)


# =============================================================================
# EXPERIMENT 2: BIDIRECTIONAL MODEL
# =============================================================================

def experiment_bidirectional(
    games: List[dict],
    min_lead: int = 6,
    max_lead: int = 20,
    min_elapsed: float = 10.0,
    max_elapsed: float = 40.0,
    hurst_mean_revert: float = 0.42,
    hurst_trending: float = 0.58,
    vig: float = 0.045,
    label: str = "",
) -> dict:
    """
    IDEA: Instead of always fading, let the regime decide direction.

    Mean-reverting regime (H < 0.42, AC < -0.1):
        → Fade the leader (bet underdog ML)

    Trending regime (H > 0.58, AC > 0.1):
        → Bet the leader (bet favorite ML)

    Neutral (0.42 < H < 0.58):
        → No trade
    """
    trades = []
    games_checked = 0

    for game in games:
        states = game['states']
        times, leads = resample_to_bars(states, bar_size=0.5)

        if len(leads) < 30:
            continue
        games_checked += 1

        traded = False
        for si, state in enumerate(states):
            if traded:
                break

            elapsed = state['mins_elapsed']
            if elapsed < min_elapsed or elapsed > max_elapsed:
                continue

            lead = state['lead']
            abs_lead = abs(lead)
            if abs_lead < min_lead or abs_lead > max_lead:
                continue

            bar_idx = int(elapsed / 0.5)
            if bar_idx >= len(leads):
                bar_idx = len(leads) - 1
            if bar_idx < 20:
                continue

            leads_so_far = leads[:bar_idx + 1]

            H = compute_hurst(leads_so_far)
            ac = compute_autocorr(leads_so_far)
            vel, accel = compute_velocity_accel(leads_so_far)
            vol_ratio = compute_volatility_ratio(leads_so_far)

            direction = None
            confidence = 0

            # Mean-reverting regime: FADE the leader
            if H < hurst_mean_revert and ac < -0.05:
                direction = 'fade'
                confidence = (hurst_mean_revert - H) + abs(ac)
                # Boost if vol expanding (more lead changes likely)
                if vol_ratio > 1.3:
                    confidence += 0.1
            # Trending regime: BET the leader
            elif H > hurst_trending and ac > 0.05:
                direction = 'with'
                confidence = (H - hurst_trending) + ac
                # Boost if leader is accelerating
                if lead > 0 and vel > 0.5 and accel > 0:
                    confidence += 0.15
                elif lead < 0 and vel < -0.5 and accel < 0:
                    confidence += 0.15
            else:
                continue  # Neutral zone, no trade

            if confidence < 0.1:
                continue

            # Calculate odds and outcome
            market_prob = market_win_prob(abs_lead, 48.0 - elapsed)
            final_lead = game['final_lead']

            if direction == 'fade':
                # Bet underdog ML
                odds = dog_odds_american(market_prob, vig)
                if lead > 0:
                    won = final_lead < 0  # we bet away
                else:
                    won = final_lead > 0  # we bet home
            else:
                # Bet favorite ML
                odds = fav_odds_american(market_prob, vig)
                if lead > 0:
                    won = final_lead > 0  # home still wins
                else:
                    won = final_lead < 0  # away still wins

            payout = ml_payout(odds)
            pnl = payout if won else -1.0

            trades.append({
                'game': game,
                'direction': direction,
                'entry_lead': abs_lead,
                'entry_elapsed': elapsed,
                'confidence': confidence,
                'hurst': H,
                'autocorr': ac,
                'velocity': vel,
                'vol_ratio': vol_ratio,
                'odds': odds,
                'won': won,
                'pnl': pnl,
            })
            traded = True

    return summarize_trades(trades, games_checked, label)


# =============================================================================
# EXPERIMENT 3: SPECIFIC LAYER COMBOS
# =============================================================================

def experiment_layer_combos(
    games: List[dict],
    min_lead: int = 6,
    max_lead: int = 25,
    min_elapsed: float = 8.0,
    max_elapsed: float = 40.0,
    combo_name: str = "hurst_ac",
    vig: float = 0.045,
    label: str = "",
) -> dict:
    """
    Test specific 2-3 layer combinations.

    Combos to test:
    - "hurst_ac": Hurst + Autocorrelation (pure regime detection)
    - "hurst_vol": Hurst + Volatility (regime + environment)
    - "ac_vel": Autocorrelation + Velocity (momentum + direction)
    - "hurst_ac_vol": Triple combo
    - "run_drought": Scoring run against leader + leader drought
    """
    trades = []
    games_checked = 0

    for game in games:
        states = game['states']
        times, leads = resample_to_bars(states, bar_size=0.5)

        if len(leads) < 30:
            continue
        games_checked += 1

        traded = False
        for si, state in enumerate(states):
            if traded:
                break

            elapsed = state['mins_elapsed']
            if elapsed < min_elapsed or elapsed > max_elapsed:
                continue

            lead = state['lead']
            abs_lead = abs(lead)
            if abs_lead < min_lead or abs_lead > max_lead:
                continue

            bar_idx = int(elapsed / 0.5)
            if bar_idx >= len(leads):
                bar_idx = len(leads) - 1
            if bar_idx < 20:
                continue

            leads_so_far = leads[:bar_idx + 1]
            signal = False

            if combo_name == "hurst_ac":
                H = compute_hurst(leads_so_far)
                ac = compute_autocorr(leads_so_far)
                signal = H < 0.43 and ac < -0.08

            elif combo_name == "hurst_vol":
                H = compute_hurst(leads_so_far)
                vr = compute_volatility_ratio(leads_so_far)
                signal = H < 0.43 and vr > 1.3

            elif combo_name == "ac_vel":
                ac = compute_autocorr(leads_so_far)
                vel, accel = compute_velocity_accel(leads_so_far)
                # Negative AC + decelerating lead
                if lead > 0:
                    signal = ac < -0.08 and vel > 0 and accel < -0.3
                else:
                    signal = ac < -0.08 and vel < 0 and accel > 0.3

            elif combo_name == "hurst_ac_vol":
                H = compute_hurst(leads_so_far)
                ac = compute_autocorr(leads_so_far)
                vr = compute_volatility_ratio(leads_so_far)
                signal = H < 0.45 and ac < -0.05 and vr > 1.2

            elif combo_name == "run_drought":
                run = detect_scoring_run(states, si)
                # Trailing team is on a run
                if lead > 0:
                    # Home leads, check if away is on a run
                    on_run = run['run_team'] == 'away' and run['run_pts'] >= 7
                else:
                    on_run = run['run_team'] == 'home' and run['run_pts'] >= 7
                signal = on_run

            elif combo_name == "run_hurst":
                run = detect_scoring_run(states, si)
                H = compute_hurst(leads_so_far)
                if lead > 0:
                    on_run = run['run_team'] == 'away' and run['run_pts'] >= 6
                else:
                    on_run = run['run_team'] == 'home' and run['run_pts'] >= 6
                signal = on_run and H < 0.48

            if not signal:
                continue

            # Always fade the leader
            market_prob = market_win_prob(abs_lead, 48.0 - elapsed)
            odds = dog_odds_american(market_prob, vig)
            final_lead = game['final_lead']

            if lead > 0:
                won = final_lead < 0
            else:
                won = final_lead > 0

            payout = ml_payout(odds)
            pnl = payout if won else -1.0

            trades.append({
                'game': game,
                'entry_lead': abs_lead,
                'entry_elapsed': elapsed,
                'odds': odds,
                'won': won,
                'pnl': pnl,
            })
            traded = True

    return summarize_trades(trades, games_checked, label)


# =============================================================================
# EXPERIMENT 4: SCORING BURST FADE
# =============================================================================

def experiment_burst_fade(
    games: List[dict],
    min_run_pts: int = 8,
    min_lead: int = 5,
    max_lead: int = 30,
    min_elapsed: float = 6.0,
    max_elapsed: float = 42.0,
    vig: float = 0.045,
    label: str = "",
) -> dict:
    """
    IDEA: When one team goes on a big run (8-0, 10-0, 12-0 etc.),
    bet AGAINST the run continuing. Pure contrarian.

    NBA scoring runs tend to end because:
    - The trailing team calls timeout (adjusts)
    - The leading team gets complacent
    - Rotation changes / fatigue
    - Random regression to the mean
    """
    trades = []
    games_checked = 0

    for game in games:
        states = game['states']
        if len(states) < 20:
            continue
        games_checked += 1

        traded = False
        for si, state in enumerate(states):
            if traded:
                break

            elapsed = state['mins_elapsed']
            if elapsed < min_elapsed or elapsed > max_elapsed:
                continue

            lead = state['lead']
            abs_lead = abs(lead)
            if abs_lead < min_lead or abs_lead > max_lead:
                continue

            run = detect_scoring_run(states, si)
            if run['run_pts'] < min_run_pts:
                continue

            # The running team must be the LEADER (we fade them)
            if lead > 0 and run['run_team'] != 'home':
                continue
            if lead < 0 and run['run_team'] != 'away':
                continue

            # Entry: fade the leader who's on a run
            market_prob = market_win_prob(abs_lead, 48.0 - elapsed)
            odds = dog_odds_american(market_prob, vig)
            final_lead = game['final_lead']

            if lead > 0:
                won = final_lead < 0
            else:
                won = final_lead > 0

            payout = ml_payout(odds)
            pnl = payout if won else -1.0

            trades.append({
                'game': game,
                'entry_lead': abs_lead,
                'entry_elapsed': elapsed,
                'run_pts': run['run_pts'],
                'run_plays': run['run_plays'],
                'odds': odds,
                'won': won,
                'pnl': pnl,
            })
            traded = True

    return summarize_trades(trades, games_checked, label)


# =============================================================================
# EXPERIMENT 5: HALFTIME TRANSITION / Q3 FADE
# =============================================================================

def experiment_q3_transition(
    games: List[dict],
    min_halftime_lead: int = 8,
    max_halftime_lead: int = 20,
    entry_window_start: float = 24.5,  # just after halftime (24 min elapsed)
    entry_window_end: float = 30.0,    # first 6 min of Q3
    vig: float = 0.045,
    label: str = "",
) -> dict:
    """
    IDEA: Q3 is the adjustment quarter. Trailing teams come out of halftime
    with new game plans. Large halftime leads often compress in early Q3.

    Enter: in first 6 minutes of Q3, if lead is 8-20
    Direction: Fade the halftime leader
    Exit: Hold to game end
    """
    trades = []
    games_checked = 0

    for game in games:
        states = game['states']
        if len(states) < 20:
            continue
        games_checked += 1

        # Find the halftime lead (closest state to 24.0 elapsed)
        halftime_lead = None
        for s in states:
            if 23.0 <= s['mins_elapsed'] <= 25.0:
                halftime_lead = s['lead']

        if halftime_lead is None:
            continue

        abs_ht_lead = abs(halftime_lead)
        if abs_ht_lead < min_halftime_lead or abs_ht_lead > max_halftime_lead:
            continue

        traded = False
        for si, state in enumerate(states):
            if traded:
                break

            elapsed = state['mins_elapsed']
            if elapsed < entry_window_start or elapsed > entry_window_end:
                continue

            lead = state['lead']
            # Must still be same team leading as at halftime
            if halftime_lead > 0 and lead <= 0:
                continue
            if halftime_lead < 0 and lead >= 0:
                continue

            abs_lead = abs(lead)
            if abs_lead < 4:  # lead already compressed a lot
                continue

            market_prob = market_win_prob(abs_lead, 48.0 - elapsed)
            odds = dog_odds_american(market_prob, vig)
            final_lead = game['final_lead']

            if lead > 0:
                won = final_lead < 0
            else:
                won = final_lead > 0

            payout = ml_payout(odds)
            pnl = payout if won else -1.0

            trades.append({
                'game': game,
                'entry_lead': abs_lead,
                'halftime_lead': abs_ht_lead,
                'entry_elapsed': elapsed,
                'odds': odds,
                'won': won,
                'pnl': pnl,
            })
            traded = True

    return summarize_trades(trades, games_checked, label)


# =============================================================================
# EXPERIMENT 6: BLOWOUT COMPRESSION (LIVE ODDS CAPTURE)
# =============================================================================

def experiment_blowout_compression(
    games: List[dict],
    min_lead: int = 18,
    max_lead: int = 35,
    min_elapsed: float = 10.0,
    max_elapsed: float = 36.0,
    compression_target: float = 0.30,  # lead shrinks by 30%
    max_hold: float = 10.0,
    vig: float = 0.045,
    label: str = "",
) -> dict:
    """
    IDEA: Large leads (18-35 pts) almost always compress at some point.
    Even if the favorite still wins, the lead typically shrinks.

    We don't need a comeback — just need the lead to narrow.
    Capture this via live odds movement.
    """
    trades = []
    games_checked = 0

    for game in games:
        states = game['states']
        if len(states) < 20:
            continue
        games_checked += 1

        traded = False
        for si, state in enumerate(states):
            if traded:
                break

            elapsed = state['mins_elapsed']
            if elapsed < min_elapsed or elapsed > max_elapsed:
                continue

            lead = state['lead']
            abs_lead = abs(lead)
            if abs_lead < min_lead or abs_lead > max_lead:
                continue

            # Entry at blowout lead
            entry_lead = abs_lead
            entry_elapsed = elapsed
            entry_prob = market_win_prob(entry_lead, 48.0 - entry_elapsed)
            entry_dog_prob = 1 - entry_prob
            entry_dog_odds = dog_odds_american(entry_prob, vig)
            entry_payout = ml_payout(entry_dog_odds)

            # Walk forward looking for compression
            exit_lead = None
            exit_elapsed = None
            exit_reason = None
            min_future_lead = entry_lead  # track minimum lead after entry

            for future_si in range(si + 1, len(states)):
                fs = states[future_si]
                future_abs_lead = abs(fs['lead'])
                future_elapsed = fs['mins_elapsed']
                hold_time = future_elapsed - entry_elapsed

                # Track if lead same direction (no flip)
                same_direction = (lead > 0 and fs['lead'] > 0) or (lead < 0 and fs['lead'] < 0)

                if not same_direction:
                    # Lead flipped — big win
                    exit_lead = 0
                    exit_elapsed = future_elapsed
                    exit_reason = 'flip'
                    break

                min_future_lead = min(min_future_lead, future_abs_lead)

                compression = 1 - (future_abs_lead / entry_lead) if entry_lead > 0 else 0
                if compression >= compression_target:
                    exit_lead = future_abs_lead
                    exit_elapsed = future_elapsed
                    exit_reason = 'target'
                    break

                if hold_time >= max_hold:
                    exit_lead = future_abs_lead
                    exit_elapsed = future_elapsed
                    exit_reason = 'timeout'
                    break

            if exit_lead is None:
                # Game ended
                final_abs_lead = abs(game['final_lead'])
                exit_lead = final_abs_lead
                exit_elapsed = 48.0
                exit_reason = 'game_end'

            # P&L: based on odds movement
            exit_mins_rem = 48.0 - exit_elapsed
            if exit_reason == 'flip':
                pnl = entry_payout * 0.7  # cash out at premium
            elif exit_reason == 'target':
                exit_prob = market_win_prob(exit_lead, exit_mins_rem)
                exit_dog_prob = 1 - exit_prob
                prob_gain = exit_dog_prob - entry_dog_prob
                pnl = prob_gain * (1 + entry_payout) * 0.85
            elif exit_reason == 'timeout':
                exit_prob = market_win_prob(exit_lead, exit_mins_rem)
                exit_dog_prob = 1 - exit_prob
                prob_gain = exit_dog_prob - entry_dog_prob
                pnl = prob_gain * (1 + entry_payout) * 0.85
            else:
                # Game end: standard ML
                if lead > 0:
                    won = game['final_lead'] < 0
                else:
                    won = game['final_lead'] > 0
                pnl = entry_payout if won else -1.0

            trades.append({
                'game': game,
                'entry_lead': entry_lead,
                'exit_lead': exit_lead,
                'entry_elapsed': entry_elapsed,
                'exit_elapsed': exit_elapsed,
                'exit_reason': exit_reason,
                'entry_odds': entry_dog_odds,
                'min_future_lead': min_future_lead,
                'pnl': pnl,
                'hold_time': exit_elapsed - entry_elapsed,
            })
            traded = True

    return summarize_trades(trades, games_checked, label)


# =============================================================================
# EXPERIMENT 7: BET THE FAVORITE IN TRENDING REGIME
# =============================================================================

def experiment_bet_favorite(
    games: List[dict],
    min_lead: int = 4,
    max_lead: int = 12,
    min_elapsed: float = 10.0,
    max_elapsed: float = 38.0,
    min_hurst: float = 0.57,
    min_autocorr: float = 0.08,
    vig: float = 0.045,
    label: str = "",
) -> dict:
    """
    IDEA: Flip the script. When the game is TRENDING (not mean-reverting),
    bet the FAVORITE. The leader continues to extend.

    Entry: Hurst > 0.57, AC > 0.08, moderate lead (4-12 pts)
    Direction: Bet favorite ML
    Exit: Hold to game end
    """
    trades = []
    games_checked = 0

    for game in games:
        states = game['states']
        times, leads = resample_to_bars(states, bar_size=0.5)

        if len(leads) < 30:
            continue
        games_checked += 1

        traded = False
        for si, state in enumerate(states):
            if traded:
                break

            elapsed = state['mins_elapsed']
            if elapsed < min_elapsed or elapsed > max_elapsed:
                continue

            lead = state['lead']
            abs_lead = abs(lead)
            if abs_lead < min_lead or abs_lead > max_lead:
                continue

            bar_idx = int(elapsed / 0.5)
            if bar_idx >= len(leads):
                bar_idx = len(leads) - 1
            if bar_idx < 20:
                continue

            leads_so_far = leads[:bar_idx + 1]

            H = compute_hurst(leads_so_far)
            if H < min_hurst:
                continue

            ac = compute_autocorr(leads_so_far)
            if ac < min_autocorr:
                continue

            vel, accel = compute_velocity_accel(leads_so_far)
            # Lead should be growing (velocity in direction of lead)
            if lead > 0 and vel < 0:
                continue  # home leads but lead shrinking
            if lead < 0 and vel > 0:
                continue  # away leads but lead shrinking

            # Bet the FAVORITE (leader)
            market_prob = market_win_prob(abs_lead, 48.0 - elapsed)
            odds = fav_odds_american(market_prob, vig)
            final_lead = game['final_lead']

            if lead > 0:
                won = final_lead > 0  # home wins
            else:
                won = final_lead < 0  # away wins

            payout = ml_payout(odds)
            pnl = payout if won else -1.0

            trades.append({
                'game': game,
                'direction': 'favorite',
                'entry_lead': abs_lead,
                'entry_elapsed': elapsed,
                'hurst': H,
                'autocorr': ac,
                'velocity': vel,
                'odds': odds,
                'won': won,
                'pnl': pnl,
            })
            traded = True

    return summarize_trades(trades, games_checked, label)


# =============================================================================
# SUMMARY HELPER
# =============================================================================

def summarize_trades(trades: List[dict], games_checked: int, label: str) -> dict:
    n = len(trades)
    if n == 0:
        return {
            'label': label, 'n_trades': 0, 'games_checked': games_checked,
            'win_rate': 0, 'roi': 0, 'total_pnl': 0, 'sharpe': 0,
            'profit_factor': 0, 'max_dd': 0, 'trades': [],
        }

    wins = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = wins / n
    total_pnl = sum(t['pnl'] for t in trades)
    roi = total_pnl / n * 100
    pnls = [t['pnl'] for t in trades]
    avg_pnl = statistics.mean(pnls)
    std_pnl = statistics.stdev(pnls) if n > 1 else 1
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    equity = [0.0]
    for p in pnls:
        equity.append(equity[-1] + p)
    max_eq = equity[0]
    max_dd = 0
    for e in equity:
        max_eq = max(max_eq, e)
        max_dd = max(max_dd, max_eq - e)

    # By exit reason if available
    by_reason = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        reason = t.get('exit_reason', 'ml_end')
        by_reason[reason]['n'] += 1
        by_reason[reason]['pnl'] += t['pnl']
        by_reason[reason]['wins'] += 1 if t['pnl'] > 0 else 0

    return {
        'label': label,
        'n_trades': n,
        'games_checked': games_checked,
        'wins': wins,
        'losses': n - wins,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'roi': roi,
        'avg_pnl': avg_pnl,
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'max_dd': max_dd,
        'equity': equity,
        'by_reason': dict(by_reason),
        'trades': trades,
    }


def print_result(r: dict):
    if r['n_trades'] == 0:
        print(f"  {r['label']:<55} | {'NO TRADES':>8} ({r['games_checked']} games)")
        return

    print(f"  {r['label']:<55} | {r['n_trades']:>4}t "
          f"{r['win_rate']*100:>5.1f}% WR "
          f"{r['roi']:>+7.1f}% ROI "
          f"{r['total_pnl']:>+7.2f}u "
          f"Sh={r['sharpe']:>+.3f} "
          f"PF={r['profit_factor']:>5.2f} "
          f"DD={r['max_dd']:>5.1f}")

    if r.get('by_reason'):
        for reason, stats in sorted(r['by_reason'].items()):
            if stats['n'] > 0:
                wr = stats['wins'] / stats['n'] * 100
                roi = stats['pnl'] / stats['n'] * 100
                print(f"    └─ {reason:<12}: {stats['n']:>3}t "
                      f"{wr:>5.1f}% WR {stats['pnl']:>+7.2f}u ({roi:>+5.1f}% ROI)")


def print_detailed(r: dict, max_trades: int = 20):
    """Show individual trade details for inspection."""
    if r['n_trades'] == 0:
        return

    print(f"\n  SAMPLE TRADES for {r['label']}:")
    winners = sorted([t for t in r['trades'] if t['pnl'] > 0], key=lambda t: -t['pnl'])
    losers = sorted([t for t in r['trades'] if t['pnl'] <= 0], key=lambda t: t['pnl'])

    print(f"  TOP WINNERS:")
    for t in winners[:max_trades // 2]:
        g = t['game']
        reason = t.get('exit_reason', 'ml')
        hold = t.get('hold_time', 0)
        print(f"    {g['date']} {g['away_team']:>3}@{g['home_team']:<3} "
              f"L={t['entry_lead']:>2} t={t['entry_elapsed']:>4.0f}m "
              f"odds={t.get('odds', t.get('entry_odds', 0)):>+6.0f} "
              f"{reason:<8} hold={hold:>4.1f}m "
              f"pnl={t['pnl']:>+.3f}")

    print(f"  LOSERS:")
    for t in losers[:max_trades // 2]:
        g = t['game']
        reason = t.get('exit_reason', 'ml')
        hold = t.get('hold_time', 0)
        print(f"    {g['date']} {g['away_team']:>3}@{g['home_team']:<3} "
              f"L={t['entry_lead']:>2} t={t['entry_elapsed']:>4.0f}m "
              f"odds={t.get('odds', t.get('entry_odds', 0)):>+6.0f} "
              f"{reason:<8} hold={hold:>4.1f}m "
              f"pnl={t['pnl']:>+.3f}")


# =============================================================================
# MAIN: RUN ALL EXPERIMENTS
# =============================================================================

def main():
    print("=" * 80)
    print("NBA EXPERIMENTAL STRATEGY LAB")
    print("7 untested approaches on 2,310 real games")
    print("=" * 80)

    games = load_all_games()
    if not games:
        return

    n_games = len(games)
    print(f"\nDataset: {n_games} games\n")

    # =========================================================================
    # EXPERIMENT 1: LEAD COMPRESSION EXIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: LEAD COMPRESSION EXIT")
    print("(Don't hold to end — exit when lead narrows)")
    print("=" * 80)

    for comp_target in [0.25, 0.30, 0.40, 0.50]:
        for max_hold in [6.0, 10.0, 15.0]:
            for min_lead in [8, 10, 14]:
                r = experiment_lead_compression(
                    games, min_lead=min_lead, compression_target=comp_target,
                    max_hold_time=max_hold,
                    label=f"Compress {comp_target:.0%} hold<={max_hold:.0f}m lead>={min_lead}")
                if r['n_trades'] >= 20:
                    print_result(r)

    # =========================================================================
    # EXPERIMENT 2: BIDIRECTIONAL MODEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: BIDIRECTIONAL MODEL")
    print("(Mean-reverting → fade, Trending → bet leader)")
    print("=" * 80)

    for h_mr in [0.38, 0.40, 0.42, 0.44]:
        for h_tr in [0.56, 0.58, 0.60, 0.62]:
            for min_lead in [4, 6, 8]:
                r = experiment_bidirectional(
                    games, min_lead=min_lead,
                    hurst_mean_revert=h_mr, hurst_trending=h_tr,
                    label=f"Bidir H<{h_mr}/H>{h_tr} lead>={min_lead}")
                if r['n_trades'] >= 15:
                    print_result(r)

    # =========================================================================
    # EXPERIMENT 3: LAYER COMBOS
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: SPECIFIC LAYER COMBINATIONS")
    print("(Testing which 2-3 layer combos carry the most signal)")
    print("=" * 80)

    for combo in ["hurst_ac", "hurst_vol", "ac_vel", "hurst_ac_vol", "run_drought", "run_hurst"]:
        for min_lead in [5, 7, 10]:
            r = experiment_layer_combos(
                games, min_lead=min_lead, combo_name=combo,
                label=f"{combo} lead>={min_lead}")
            if r['n_trades'] >= 10:
                print_result(r)

    # =========================================================================
    # EXPERIMENT 4: SCORING BURST FADE
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: SCORING BURST FADE")
    print("(Fade big scoring runs — pure contrarian)")
    print("=" * 80)

    for min_run in [7, 8, 10, 12, 14]:
        for min_lead in [5, 8, 10]:
            r = experiment_burst_fade(
                games, min_run_pts=min_run, min_lead=min_lead,
                label=f"Run>={min_run}pts lead>={min_lead}")
            if r['n_trades'] >= 10:
                print_result(r)

    # =========================================================================
    # EXPERIMENT 5: Q3 TRANSITION
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: HALFTIME TRANSITION / Q3 FADE")
    print("(Fade halftime leaders in early Q3)")
    print("=" * 80)

    for ht_min in [6, 8, 10, 12]:
        for ht_max in [15, 18, 20, 25]:
            for window_end in [28.0, 30.0, 32.0]:
                r = experiment_q3_transition(
                    games, min_halftime_lead=ht_min, max_halftime_lead=ht_max,
                    entry_window_end=window_end,
                    label=f"Q3 fade HT lead {ht_min}-{ht_max} win<={window_end:.0f}m")
                if r['n_trades'] >= 15:
                    print_result(r)

    # =========================================================================
    # EXPERIMENT 6: BLOWOUT COMPRESSION
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: BLOWOUT COMPRESSION (LIVE ODDS CAPTURE)")
    print("(Bet leads 18+ will compress — don't need full comeback)")
    print("=" * 80)

    for min_lead in [15, 18, 20, 22, 25]:
        for comp in [0.20, 0.25, 0.30, 0.40]:
            for max_hold in [8.0, 12.0, 15.0]:
                r = experiment_blowout_compression(
                    games, min_lead=min_lead, compression_target=comp,
                    max_hold=max_hold,
                    label=f"Blowout>={min_lead} comp{comp:.0%} hold<={max_hold:.0f}m")
                if r['n_trades'] >= 10:
                    print_result(r)

    # =========================================================================
    # EXPERIMENT 7: BET THE FAVORITE
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: BET THE FAVORITE (TRENDING REGIME)")
    print("(Flip the script — when trending, bet the leader)")
    print("=" * 80)

    for min_h in [0.55, 0.57, 0.60, 0.63]:
        for min_ac in [0.05, 0.08, 0.12]:
            for min_lead in [3, 4, 6, 8]:
                for max_lead in [10, 12, 15]:
                    if max_lead <= min_lead:
                        continue
                    r = experiment_bet_favorite(
                        games, min_lead=min_lead, max_lead=max_lead,
                        min_hurst=min_h, min_autocorr=min_ac,
                        label=f"Fav H>{min_h} AC>{min_ac} lead {min_lead}-{max_lead}")
                    if r['n_trades'] >= 15:
                        print_result(r)

    # =========================================================================
    # FIND BEST OVERALL
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("TOP RESULTS ACROSS ALL EXPERIMENTS")
    print("=" * 80)
    print("(Re-running best configs with detailed output)")

    # Collect all results by re-running the most promising configs
    # We'll identify them from the output above
    print("\n--- done ---")


if __name__ == '__main__':
    main()
