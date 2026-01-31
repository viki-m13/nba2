#!/usr/bin/env python3
"""
NBA Breakout Detection + ML Pattern Recognition Strategy
=========================================================

Uses stock-trading breakout concepts + machine learning trained on 2,310 real
NBA games with full play-by-play data.

Breakout Detection:
- Support/Resistance levels on the lead series
- Range compression → expansion (Bollinger squeeze)
- Momentum breakouts with volume confirmation
- Channel breakouts (Donchian-style)
- Moving average crossovers on lead series

ML Pattern Recognition:
- 60+ features extracted from real PBP data
- XGBoost with walk-forward validation (train on past seasons, test on next)
- No look-ahead bias - strict temporal separation
- Calibrated probability outputs for Kelly sizing
"""

import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA LOADING - Parse all 2,310 real games
# ============================================================================

PBP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'games_pbp')
VALIDATED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'validated_games.json')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')


def clock_to_seconds(clock_str, period):
    """Convert game clock + period to total seconds remaining in game."""
    try:
        parts = clock_str.split(':')
        mins = int(parts[0])
        secs = int(parts[1]) if len(parts) > 1 else 0
        period_secs = mins * 60 + secs
        # 4 quarters, 12 min each = 2880 seconds total
        # OT = 5 min periods
        if period <= 4:
            remaining_periods = (4 - period) * 720
            return remaining_periods + period_secs
        else:
            # Overtime
            return period_secs
    except:
        return None


def seconds_to_game_minutes(total_secs_remaining):
    """Convert seconds remaining to game minutes elapsed (0-48)."""
    return (2880 - total_secs_remaining) / 60.0


def load_game_pbp(game_id):
    """Load and parse a single game's play-by-play into a lead series."""
    filepath = os.path.join(PBP_DIR, f'{game_id}.json')
    if not os.path.exists(filepath):
        return None

    with open(filepath) as f:
        data = json.load(f)

    plays = data.get('plays', [])
    if not plays:
        return None

    # Extract header info
    header = data.get('header', {})
    competitions = header.get('competitions', [{}])
    comp = competitions[0] if competitions else {}
    competitors = comp.get('competitors', [])

    home_team = away_team = None
    final_home = final_away = 0
    for c in competitors:
        team = c.get('team', {})
        abbr = team.get('abbreviation', '?')
        score = int(c.get('score', 0))
        if c.get('homeAway') == 'home':
            home_team = abbr
            final_home = score
        else:
            away_team = abbr
            final_away = score

    # Build lead series from plays
    states = []
    for p in plays:
        home_score = p.get('homeScore', 0)
        away_score = p.get('awayScore', 0)
        period = p.get('period', {}).get('number', 0)
        clock = p.get('clock', {}).get('displayValue', '0:00')
        is_scoring = p.get('scoringPlay', False)

        secs_remaining = clock_to_seconds(clock, period)
        if secs_remaining is None:
            continue

        lead = home_score - away_score  # positive = home leads
        game_mins = seconds_to_game_minutes(secs_remaining)

        states.append({
            'secs_remaining': secs_remaining,
            'game_mins': game_mins,
            'home_score': home_score,
            'away_score': away_score,
            'lead': lead,
            'period': period,
            'is_scoring': is_scoring,
            'total_points': home_score + away_score,
        })

    if not states:
        return None

    # Win probability data
    wp_data = data.get('winprobability', [])
    wp_by_play = {}
    for wp in wp_data:
        wp_by_play[wp.get('playId', '')] = wp.get('homeWinPercentage', 0.5)

    # Attach win probabilities to states
    for i, p in enumerate(plays):
        play_id = p.get('id', '')
        if i < len(states):
            states[i]['home_wp'] = wp_by_play.get(play_id, 0.5)

    return {
        'game_id': game_id,
        'home_team': home_team,
        'away_team': away_team,
        'final_home': final_home,
        'final_away': final_away,
        'final_margin': final_home - final_away,
        'states': states,
    }


def load_all_games():
    """Load all 2,310 games with metadata."""
    with open(VALIDATED_PATH) as f:
        game_list = json.load(f)

    games = []
    for g in game_list:
        game_id = g['id']
        pbp = load_game_pbp(game_id)
        if pbp:
            pbp['date'] = g['date']
            pbp['season'] = get_season(g['date'])
            games.append(pbp)

    games.sort(key=lambda x: x['date'])
    return games


def get_season(date_str):
    """Get NBA season from date string (YYYYMMDD)."""
    year = int(date_str[:4])
    month = int(date_str[4:6])
    if month >= 10:
        return f"{year}-{year+1}"
    else:
        return f"{year-1}-{year}"


# ============================================================================
# SECTION 2: BREAKOUT DETECTION ENGINE
# ============================================================================

class BreakoutDetector:
    """
    Stock-trading breakout detection applied to NBA lead series.

    Concepts:
    - Support/Resistance: Where the lead has bounced before
    - Bollinger Squeeze: Low volatility → high volatility breakout
    - Donchian Channel: N-play high/low breakout
    - Moving Average Crossover: Short vs long lead momentum
    - Volume Confirmation: Scoring pace validates breakout
    """

    @staticmethod
    def find_support_resistance(lead_series, window=20):
        """Find support/resistance levels from local extrema in lead series."""
        if len(lead_series) < window:
            return [], []

        arr = np.array(lead_series, dtype=float)

        # Find local maxima (resistance) and minima (support)
        order = max(3, window // 4)
        if len(arr) < 2 * order + 1:
            return [], []

        local_max_idx = argrelextrema(arr, np.greater_equal, order=order)[0]
        local_min_idx = argrelextrema(arr, np.less_equal, order=order)[0]

        resistance = arr[local_max_idx].tolist() if len(local_max_idx) > 0 else []
        support = arr[local_min_idx].tolist() if len(local_min_idx) > 0 else []

        return support, resistance

    @staticmethod
    def bollinger_squeeze(lead_series, window=20, squeeze_threshold=0.4):
        """
        Detect Bollinger Band squeeze - low volatility precedes breakout.

        Returns: (is_squeeze, bandwidth, direction)
        - direction: +1 if breaking upward (home extending), -1 if breaking down
        """
        if len(lead_series) < window + 5:
            return False, 0, 0

        arr = np.array(lead_series[-window:], dtype=float)
        ma = np.mean(arr)
        std = np.std(arr)

        if std < 0.01:
            return True, 0, 0

        bandwidth = (2 * std) / abs(ma) if abs(ma) > 0.5 else 2 * std

        # Check recent direction after squeeze
        recent = lead_series[-5:]
        direction = 1 if recent[-1] > ma else -1

        # Compare current bandwidth to historical
        if len(lead_series) >= window * 2:
            historical_std = np.std(lead_series[-(window*2):-window])
            squeeze_ratio = std / max(historical_std, 0.01)
            is_squeeze = squeeze_ratio < squeeze_threshold
        else:
            is_squeeze = bandwidth < squeeze_threshold

        return is_squeeze, bandwidth, direction

    @staticmethod
    def donchian_breakout(lead_series, channel_window=15):
        """
        Donchian Channel breakout - lead breaking N-play high or low.

        Returns: (breakout_type, channel_high, channel_low, current)
        - breakout_type: 'high' (home breaking out), 'low' (away breaking out), None
        """
        if len(lead_series) < channel_window + 1:
            return None, 0, 0, 0

        channel = lead_series[-(channel_window+1):-1]
        current = lead_series[-1]

        channel_high = max(channel)
        channel_low = min(channel)

        if current > channel_high:
            return 'high', channel_high, channel_low, current
        elif current < channel_low:
            return 'low', channel_high, channel_low, current
        else:
            return None, channel_high, channel_low, current

    @staticmethod
    def ma_crossover(lead_series, short_window=5, long_window=15):
        """
        Moving average crossover on lead series.

        Returns: (crossover_type, short_ma, long_ma, spread)
        - 'golden_cross': short MA crosses above long (home momentum)
        - 'death_cross': short MA crosses below long (away momentum)
        """
        if len(lead_series) < long_window + 2:
            return None, 0, 0, 0

        short_ma_now = np.mean(lead_series[-short_window:])
        long_ma_now = np.mean(lead_series[-long_window:])
        short_ma_prev = np.mean(lead_series[-(short_window+1):-1])
        long_ma_prev = np.mean(lead_series[-(long_window+1):-1])

        spread = short_ma_now - long_ma_now

        if short_ma_prev <= long_ma_prev and short_ma_now > long_ma_now:
            return 'golden_cross', short_ma_now, long_ma_now, spread
        elif short_ma_prev >= long_ma_prev and short_ma_now < long_ma_now:
            return 'death_cross', short_ma_now, long_ma_now, spread
        else:
            return None, short_ma_now, long_ma_now, spread

    @staticmethod
    def momentum_breakout(lead_series, lookback=10, threshold=1.5):
        """
        Momentum breakout - lead change velocity exceeding threshold.

        Inspired by rate-of-change (ROC) indicator in stock trading.
        Returns: (is_breakout, roc, direction)
        """
        if len(lead_series) < lookback + 1:
            return False, 0, 0

        current = lead_series[-1]
        past = lead_series[-(lookback+1)]
        roc = current - past

        # Calculate typical ROC magnitude over the series
        rocs = [lead_series[i] - lead_series[i-lookback]
                for i in range(lookback, len(lead_series))]
        if not rocs:
            return False, roc, np.sign(roc)

        avg_abs_roc = np.mean(np.abs(rocs))

        is_breakout = abs(roc) > max(threshold * avg_abs_roc, 3)
        return is_breakout, roc, np.sign(roc)

    @staticmethod
    def volume_confirmation(scoring_times, current_idx, lookback=10):
        """
        Volume confirmation - is scoring pace accelerating?

        In stocks, breakouts with volume are more reliable.
        In NBA, 'volume' = scoring frequency/pace.
        Returns: (pace_ratio, is_confirmed)
        """
        if current_idx < lookback * 2:
            return 1.0, False

        # Count scoring plays in recent vs prior window
        recent_scoring = sum(scoring_times[current_idx-lookback:current_idx])
        prior_scoring = sum(scoring_times[current_idx-lookback*2:current_idx-lookback])

        if prior_scoring == 0:
            return 2.0 if recent_scoring > 0 else 1.0, recent_scoring > 3

        pace_ratio = recent_scoring / prior_scoring
        return pace_ratio, pace_ratio > 1.3

    @staticmethod
    def range_compression_score(lead_series, window=20):
        """
        Measure how compressed the lead range is - tighter ranges lead to
        bigger breakouts (like coiled spring).

        Returns: compression_score (0-1, higher = more compressed)
        """
        if len(lead_series) < window:
            return 0

        recent = lead_series[-window:]
        lead_range = max(recent) - min(recent)

        if len(lead_series) >= window * 2:
            prior = lead_series[-(window*2):-window]
            prior_range = max(prior) - min(prior)
            if prior_range > 0:
                return 1.0 - min(lead_range / prior_range, 1.0)

        # Absolute compression: NBA leads typically range 5-15 points
        return max(0, 1.0 - lead_range / 15.0)

    @staticmethod
    def detect_all_breakouts(lead_series, scoring_flags, idx):
        """Run all breakout detections and return feature dict."""
        features = {}

        # Support/Resistance
        support, resistance = BreakoutDetector.find_support_resistance(lead_series)
        current_lead = lead_series[-1] if lead_series else 0
        features['near_resistance'] = min([abs(current_lead - r) for r in resistance]) if resistance else 99
        features['near_support'] = min([abs(current_lead - s) for s in support]) if support else 99
        features['above_resistance'] = int(any(current_lead > r for r in resistance)) if resistance else 0
        features['below_support'] = int(any(current_lead < s for s in support)) if support else 0
        features['n_support_levels'] = len(support)
        features['n_resistance_levels'] = len(resistance)

        # Bollinger Squeeze
        is_squeeze, bw, direction = BreakoutDetector.bollinger_squeeze(lead_series)
        features['bb_squeeze'] = int(is_squeeze)
        features['bb_bandwidth'] = bw
        features['bb_direction'] = direction

        # Donchian Channel
        breakout_type, ch_hi, ch_lo, curr = BreakoutDetector.donchian_breakout(lead_series)
        features['donchian_breakout_high'] = int(breakout_type == 'high')
        features['donchian_breakout_low'] = int(breakout_type == 'low')
        features['donchian_channel_width'] = ch_hi - ch_lo
        features['donchian_position'] = (curr - ch_lo) / max(ch_hi - ch_lo, 0.01)

        # MA Crossover
        cross_type, short_ma, long_ma, spread = BreakoutDetector.ma_crossover(lead_series)
        features['ma_golden_cross'] = int(cross_type == 'golden_cross')
        features['ma_death_cross'] = int(cross_type == 'death_cross')
        features['ma_spread'] = spread
        features['ma_short'] = short_ma
        features['ma_long'] = long_ma

        # Momentum Breakout
        is_breakout, roc, direction = BreakoutDetector.momentum_breakout(lead_series)
        features['momentum_breakout'] = int(is_breakout)
        features['momentum_roc'] = roc
        features['momentum_direction'] = direction

        # Volume Confirmation
        pace_ratio, is_confirmed = BreakoutDetector.volume_confirmation(scoring_flags, idx)
        features['volume_pace_ratio'] = pace_ratio
        features['volume_confirmed'] = int(is_confirmed)

        # Range Compression
        features['range_compression'] = BreakoutDetector.range_compression_score(lead_series)

        return features


# ============================================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================================

def extract_features_at_state(game, state_idx):
    """
    Extract 60+ features from a game at a specific state (play index).

    This is the heart of the ML system - every feature is computed from data
    available BEFORE the prediction point (no look-ahead).
    """
    states = game['states']
    if state_idx < 20:  # Need minimum history
        return None

    current = states[state_idx]
    lead = current['lead']
    game_mins = current['game_mins']
    secs_remaining = current['secs_remaining']
    period = current['period']

    # Skip overtime and very early game
    if period > 4 or game_mins < 6:
        return None

    features = {}

    # === BASIC GAME STATE ===
    features['lead'] = lead
    features['abs_lead'] = abs(lead)
    features['home_leading'] = int(lead > 0)
    features['game_mins'] = game_mins
    features['secs_remaining'] = secs_remaining
    features['period'] = period
    features['total_points'] = current['total_points']
    features['home_score'] = current['home_score']
    features['away_score'] = current['away_score']

    # Scoring pace
    if game_mins > 1:
        features['pace'] = current['total_points'] / game_mins
        features['home_pace'] = current['home_score'] / game_mins
        features['away_pace'] = current['away_score'] / game_mins
    else:
        features['pace'] = 0
        features['home_pace'] = 0
        features['away_pace'] = 0

    # === LEAD SERIES FEATURES ===
    lead_series = [s['lead'] for s in states[:state_idx+1]]
    abs_lead_series = [abs(s['lead']) for s in states[:state_idx+1]]

    # Lead statistics
    features['lead_mean'] = np.mean(lead_series)
    features['lead_std'] = np.std(lead_series)
    features['lead_max'] = max(lead_series)
    features['lead_min'] = min(lead_series)
    features['lead_range'] = max(lead_series) - min(lead_series)
    features['lead_skew'] = float(stats.skew(lead_series)) if len(lead_series) > 2 else 0

    # Recent lead changes (velocity/acceleration)
    if len(lead_series) >= 10:
        recent_10 = lead_series[-10:]
        features['lead_velocity_10'] = recent_10[-1] - recent_10[0]
        features['lead_std_10'] = np.std(recent_10)

        # Acceleration: change in velocity
        if len(lead_series) >= 20:
            prev_10 = lead_series[-20:-10]
            vel_now = recent_10[-1] - recent_10[0]
            vel_prev = prev_10[-1] - prev_10[0]
            features['lead_acceleration'] = vel_now - vel_prev
        else:
            features['lead_acceleration'] = 0
    else:
        features['lead_velocity_10'] = 0
        features['lead_std_10'] = 0
        features['lead_acceleration'] = 0

    if len(lead_series) >= 5:
        recent_5 = lead_series[-5:]
        features['lead_velocity_5'] = recent_5[-1] - recent_5[0]
        features['lead_std_5'] = np.std(recent_5)
    else:
        features['lead_velocity_5'] = 0
        features['lead_std_5'] = 0

    # === MOMENTUM FEATURES ===
    # 5-minute momentum: score change for each team in last ~5 game minutes
    lookback_secs = 300  # 5 minutes of game time
    target_secs = secs_remaining + lookback_secs
    mom_start_idx = None
    for i in range(state_idx, -1, -1):
        if states[i]['secs_remaining'] >= target_secs:
            mom_start_idx = i
            break

    if mom_start_idx is not None:
        home_mom = current['home_score'] - states[mom_start_idx]['home_score']
        away_mom = current['away_score'] - states[mom_start_idx]['away_score']
        features['home_5min_momentum'] = home_mom
        features['away_5min_momentum'] = away_mom
        features['net_5min_momentum'] = home_mom - away_mom
        features['momentum_diff'] = abs(home_mom - away_mom)
    else:
        features['home_5min_momentum'] = 0
        features['away_5min_momentum'] = 0
        features['net_5min_momentum'] = 0
        features['momentum_diff'] = 0

    # 3-minute momentum
    lookback_secs_3 = 180
    target_secs_3 = secs_remaining + lookback_secs_3
    mom_start_idx_3 = None
    for i in range(state_idx, -1, -1):
        if states[i]['secs_remaining'] >= target_secs_3:
            mom_start_idx_3 = i
            break

    if mom_start_idx_3 is not None:
        features['net_3min_momentum'] = (
            (current['home_score'] - states[mom_start_idx_3]['home_score']) -
            (current['away_score'] - states[mom_start_idx_3]['away_score'])
        )
    else:
        features['net_3min_momentum'] = 0

    # === SCORING RUN DETECTION ===
    # Find current scoring run length
    scoring_plays = [(i, s) for i, s in enumerate(states[:state_idx+1]) if s['is_scoring']]
    run_length = 0
    run_team = 0  # +1=home, -1=away
    if len(scoring_plays) >= 2:
        # Track consecutive scores by same "side"
        prev_lead = scoring_plays[-1][1]['lead']
        for i in range(len(scoring_plays)-2, -1, -1):
            curr_lead = scoring_plays[i][1]['lead']
            next_lead = scoring_plays[i+1][1]['lead']
            # Determine which team scored
            diff = next_lead - curr_lead
            if i == len(scoring_plays) - 2:
                run_team = np.sign(diff) if diff != 0 else 0
            if np.sign(diff) == run_team and run_team != 0:
                run_length += 1
            else:
                break

    features['scoring_run_length'] = run_length
    features['scoring_run_team'] = run_team  # +1 home, -1 away
    features['leader_on_run'] = int(
        (run_team > 0 and lead > 0) or (run_team < 0 and lead < 0)
    )
    features['trailer_on_run'] = int(
        (run_team > 0 and lead < 0) or (run_team < 0 and lead > 0)
    )

    # === LEAD CHANGE / CROSSOVER FEATURES ===
    lead_changes = 0
    ties = 0
    for i in range(1, len(lead_series)):
        if lead_series[i] * lead_series[i-1] < 0:  # Sign change
            lead_changes += 1
        if lead_series[i] == 0:
            ties += 1

    features['lead_changes'] = lead_changes
    features['ties'] = ties
    features['lead_changes_per_min'] = lead_changes / max(game_mins, 1)

    # === TIME-DEPENDENT FEATURES ===
    # How long has the leader been leading?
    lead_duration = 0
    current_sign = np.sign(lead)
    for i in range(len(lead_series)-1, -1, -1):
        if np.sign(lead_series[i]) == current_sign:
            lead_duration += 1
        else:
            break
    features['lead_duration'] = lead_duration
    features['lead_duration_pct'] = lead_duration / max(len(lead_series), 1)

    # === WIN PROBABILITY FEATURES ===
    wp = current.get('home_wp', 0.5)
    features['home_wp'] = wp
    features['wp_implied_lead'] = wp - 0.5  # Normalized
    features['wp_vs_lead_divergence'] = (wp - 0.5) * 20 - lead  # WP says vs actual lead

    # WP momentum
    if state_idx >= 10:
        wp_10_ago = states[state_idx-10].get('home_wp', 0.5)
        features['wp_momentum'] = wp - wp_10_ago
    else:
        features['wp_momentum'] = 0

    # === QUARTER-SPECIFIC FEATURES ===
    features['is_q1'] = int(period == 1)
    features['is_q2'] = int(period == 2)
    features['is_q3'] = int(period == 3)
    features['is_q4'] = int(period == 4)

    # Minutes remaining in current quarter
    if period <= 4:
        quarter_secs = secs_remaining - (4 - period) * 720
        features['quarter_mins_remaining'] = quarter_secs / 60.0
    else:
        features['quarter_mins_remaining'] = secs_remaining / 60.0

    # === HURST EXPONENT (mean reversion detection) ===
    if len(lead_series) >= 20:
        try:
            features['hurst'] = _compute_hurst(lead_series[-50:] if len(lead_series) >= 50 else lead_series)
        except:
            features['hurst'] = 0.5
    else:
        features['hurst'] = 0.5

    # === AUTOCORRELATION ===
    if len(lead_series) >= 15:
        diffs = np.diff(lead_series[-30:] if len(lead_series) >= 30 else lead_series)
        if len(diffs) > 5 and np.std(diffs) > 0:
            features['autocorr_1'] = np.corrcoef(diffs[:-1], diffs[1:])[0, 1]
        else:
            features['autocorr_1'] = 0
    else:
        features['autocorr_1'] = 0

    # === BREAKOUT FEATURES ===
    scoring_flags = [s['is_scoring'] for s in states[:state_idx+1]]
    breakout_features = BreakoutDetector.detect_all_breakouts(
        lead_series, scoring_flags, state_idx
    )
    features.update(breakout_features)

    # === PATTERN FEATURES ===
    # Lead at key game milestones (if we've passed them)
    for milestone in [6, 12, 18, 24, 30, 36]:  # game minutes
        if game_mins >= milestone:
            # Find state closest to this milestone
            for s in states[:state_idx+1]:
                if s['game_mins'] >= milestone:
                    features[f'lead_at_{milestone}min'] = s['lead']
                    break
            else:
                features[f'lead_at_{milestone}min'] = 0
        else:
            features[f'lead_at_{milestone}min'] = 0

    # Quarter-end leads
    for q in range(1, period):
        q_end_states = [s for s in states[:state_idx+1] if s['period'] == q]
        if q_end_states:
            features[f'lead_end_q{q}'] = q_end_states[-1]['lead']
        else:
            features[f'lead_end_q{q}'] = 0

    return features


def _compute_hurst(series):
    """Compute Hurst exponent using R/S analysis."""
    if len(series) < 10:
        return 0.5
    arr = np.array(series, dtype=float)
    diffs = np.diff(arr)
    if np.std(diffs) < 0.001:
        return 0.5

    n = len(diffs)
    max_k = min(n // 2, 50)
    if max_k < 4:
        return 0.5

    rs_values = []
    ns = []
    for k in range(4, max_k + 1):
        rs_list = []
        for start in range(0, n - k + 1, k):
            chunk = diffs[start:start+k]
            if len(chunk) < 4:
                continue
            mean_chunk = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_chunk)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
            ns.append(k)

    if len(ns) < 3:
        return 0.5

    log_n = np.log(ns)
    log_rs = np.log(rs_values)
    slope, _, _, _, _ = stats.linregress(log_n, log_rs)
    return np.clip(slope, 0, 1)


# ============================================================================
# SECTION 4: DATASET BUILDING
# ============================================================================

def build_dataset(games, sample_windows=None):
    """
    Build ML dataset from all games.

    For each game, we sample states at key decision points and extract features.
    Target: will the trailing team cover the spread / win outright?

    sample_windows: list of (min_game_mins, max_game_mins) to sample from.
    If None, uses standard NBA decision windows.
    """
    if sample_windows is None:
        sample_windows = [
            (6, 12),    # Q1 mid-late
            (12, 18),   # Q2 early-mid
            (18, 24),   # Q2 late / Halftime
            (24, 30),   # Q3 early-mid
            (30, 36),   # Q3 late
            (36, 42),   # Q4 early-mid
            (42, 46),   # Q4 late
        ]

    rows = []
    for game in games:
        states = game['states']
        final_margin = game['final_margin']  # positive = home won

        for win_start, win_end in sample_windows:
            # Find states in this window
            candidates = []
            for i, s in enumerate(states):
                if win_start <= s['game_mins'] <= win_end:
                    candidates.append(i)

            if not candidates:
                continue

            # Sample 1-2 states per window to avoid overweighting long games
            if len(candidates) > 2:
                # Pick states where lead is interesting (not 0)
                interesting = [i for i in candidates if abs(states[i]['lead']) >= 3]
                if interesting:
                    sampled = [interesting[len(interesting)//2]]
                else:
                    sampled = [candidates[len(candidates)//2]]
            else:
                sampled = candidates[:1]

            for idx in sampled:
                features = extract_features_at_state(game, idx)
                if features is None:
                    continue

                current_lead = states[idx]['lead']
                if abs(current_lead) < 2:
                    continue  # Skip near-tie games, no actionable signal

                # === TARGET VARIABLES ===
                # From perspective of the TRAILING team:
                # If home leads (lead > 0), trailing team is away
                # "trailing team covers" = final margin is closer than current lead

                if current_lead > 0:
                    # Home leads. Trailing (away) covers if they close the gap.
                    trailing_covers = final_margin < current_lead  # margin got smaller
                    trailing_wins = final_margin < 0  # away actually won
                    margin_change = current_lead - final_margin  # positive = trailing gained
                elif current_lead < 0:
                    # Away leads. Trailing (home) covers if they close the gap.
                    trailing_covers = final_margin > current_lead  # margin got less negative
                    trailing_wins = final_margin > 0  # home actually won
                    margin_change = final_margin - current_lead  # positive = trailing gained
                else:
                    continue

                features['target_covers'] = int(trailing_covers)
                features['target_wins'] = int(trailing_wins)
                features['target_margin_change'] = margin_change
                features['game_id'] = game['game_id']
                features['date'] = game['date']
                features['season'] = game['season']

                rows.append(features)

    return pd.DataFrame(rows)


# ============================================================================
# SECTION 5: ML MODEL TRAINING & WALK-FORWARD VALIDATION
# ============================================================================

def train_ml_model(df, target='target_covers'):
    """
    Train XGBoost with walk-forward validation.

    Split by season - train on earlier seasons, test on later ones.
    This ensures NO look-ahead bias.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, brier_score_loss,
                                  classification_report)

    try:
        from xgboost import XGBClassifier
        use_xgb = True
    except ImportError:
        use_xgb = False

    seasons = sorted(df['season'].unique())
    print(f"\nSeasons available: {seasons}")
    print(f"Total samples: {len(df)}")
    print(f"Target distribution: {df[target].mean():.3f} positive rate")

    # Feature columns - everything except targets and metadata
    meta_cols = ['target_covers', 'target_wins', 'target_margin_change',
                 'game_id', 'date', 'season']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Fill NaN
    df[feature_cols] = df[feature_cols].fillna(0)

    # Walk-forward: train on seasons 0..k, test on season k+1
    all_predictions = []
    all_actuals = []
    all_probas = []
    all_meta = []
    fold_results = []

    for test_idx in range(1, len(seasons)):
        train_seasons = seasons[:test_idx]
        test_season = seasons[test_idx]

        train_mask = df['season'].isin(train_seasons)
        test_mask = df['season'] == test_season

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target].values

        if len(X_train) < 50 or len(X_test) < 20:
            continue

        print(f"\n--- Fold: Train on {train_seasons}, Test on {test_season} ---")
        print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

        if use_xgb:
            model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=1.0,
                reg_lambda=3.0,
                random_state=42,
                eval_metric='logloss',
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42,
            )

        model.fit(X_train, y_train)

        # Calibrate probabilities
        try:
            cal_model = CalibratedClassifierCV(model, cv=3, method='isotonic')
            cal_model.fit(X_train, y_train)
            probas = cal_model.predict_proba(X_test)[:, 1]
        except:
            probas = model.predict_proba(X_test)[:, 1]

        preds = (probas >= 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probas)
        brier = brier_score_loss(y_test, probas)

        print(f"    Accuracy: {acc:.3f}")
        print(f"    AUC-ROC:  {auc:.3f}")
        print(f"    Brier:    {brier:.3f}")

        all_predictions.extend(preds)
        all_actuals.extend(y_test)
        all_probas.extend(probas)
        all_meta.extend(df.loc[test_mask, ['game_id', 'date', 'season',
                                            'abs_lead', 'game_mins']].to_dict('records'))

        fold_results.append({
            'train_seasons': train_seasons,
            'test_season': test_season,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': acc,
            'auc': auc,
            'brier': brier,
        })

    # Train final model on all data for deployment
    X_all = df[feature_cols].values
    y_all = df[target].values

    if use_xgb:
        final_model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=1.0,
            reg_lambda=3.0,
            random_state=42,
            eval_metric='logloss',
        )
    else:
        final_model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )

    final_model.fit(X_all, y_all)

    # Feature importance
    if hasattr(final_model, 'feature_importances_'):
        importance = dict(zip(feature_cols, final_model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:25]
    else:
        top_features = []

    return {
        'model': final_model,
        'feature_cols': feature_cols,
        'fold_results': fold_results,
        'all_predictions': all_predictions,
        'all_actuals': all_actuals,
        'all_probas': all_probas,
        'all_meta': all_meta,
        'top_features': top_features,
    }


# ============================================================================
# SECTION 6: STRATEGY - HIGH CONFIDENCE SIGNAL SELECTION
# ============================================================================

def evaluate_strategy(results, target='target_covers'):
    """
    Evaluate trading strategy using only HIGH CONFIDENCE signals.

    The key insight: we don't bet on every game. We only bet when
    the ML model + breakout detection gives us extreme confidence.
    """
    probas = np.array(results['all_probas'])
    actuals = np.array(results['all_actuals'])
    meta = results['all_meta']

    print("\n" + "="*70)
    print("STRATEGY EVALUATION - HIGH CONFIDENCE SIGNALS ONLY")
    print("="*70)

    # Test different confidence thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    best_roi = -999
    best_threshold = 0
    strategy_results = []

    for thresh in thresholds:
        # Signals where model says trailing team covers with high confidence
        mask = probas >= thresh
        n_signals = mask.sum()

        if n_signals < 5:
            continue

        win_rate = actuals[mask].mean()
        n_wins = actuals[mask].sum()

        # ROI calculation
        # Betting on trailing team to cover spread:
        # Win: +0.91 (standard -110 odds)
        # Lose: -1.00
        roi = (win_rate * 0.91 - (1 - win_rate) * 1.0) * 100

        # For moneyline (trailing team to win outright):
        # Average underdog odds roughly +200 (pays 2:1)
        # This varies but let's use conservative +150 avg

        # Sharpe-like metric
        returns = np.where(actuals[mask] == 1, 0.91, -1.0)
        sharpe = np.mean(returns) / max(np.std(returns), 0.01) * np.sqrt(len(returns))

        # Kelly criterion
        p = win_rate
        q = 1 - p
        b = 0.91  # odds ratio
        kelly = (p * b - q) / b if p * b > q else 0

        result = {
            'threshold': thresh,
            'n_signals': int(n_signals),
            'n_wins': int(n_wins),
            'win_rate': win_rate,
            'roi_pct': roi,
            'sharpe': sharpe,
            'kelly': kelly,
            'coverage': n_signals / len(probas),
        }
        strategy_results.append(result)

        if roi > best_roi and n_signals >= 20:
            best_roi = roi
            best_threshold = thresh

        status = "***BEST***" if roi == best_roi and n_signals >= 20 else ""
        print(f"\nThreshold >= {thresh:.0%}:")
        print(f"  Signals: {n_signals:,} ({n_signals/len(probas):.1%} of games)")
        print(f"  Win Rate: {win_rate:.1%} ({n_wins}/{n_signals})")
        print(f"  ROI: {roi:+.1f}%")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Kelly: {kelly:.1%}")
        print(f"  {status}")

    return strategy_results, best_threshold


def analyze_breakout_edge(df):
    """Analyze which breakout signals have the most edge."""
    print("\n" + "="*70)
    print("BREAKOUT SIGNAL ANALYSIS")
    print("="*70)

    breakout_signals = {
        'Bollinger Squeeze': 'bb_squeeze',
        'Donchian High Breakout': 'donchian_breakout_high',
        'Donchian Low Breakout': 'donchian_breakout_low',
        'Golden Cross (MA)': 'ma_golden_cross',
        'Death Cross (MA)': 'ma_death_cross',
        'Momentum Breakout': 'momentum_breakout',
        'Volume Confirmed': 'volume_confirmed',
        'Above Resistance': 'above_resistance',
        'Below Support': 'below_support',
        'Trailer on Run': 'trailer_on_run',
    }

    results = []
    base_rate = df['target_covers'].mean()
    print(f"\nBase rate (trailing team covers): {base_rate:.1%}")

    for name, col in breakout_signals.items():
        if col not in df.columns:
            continue

        signal_mask = df[col] == 1
        n_signals = signal_mask.sum()
        if n_signals < 10:
            continue

        signal_wr = df.loc[signal_mask, 'target_covers'].mean()
        edge = signal_wr - base_rate
        roi = (signal_wr * 0.91 - (1 - signal_wr)) * 100

        # Also check combined with abs_lead >= 8
        combined_mask = signal_mask & (df['abs_lead'] >= 8)
        n_combined = combined_mask.sum()
        if n_combined >= 5:
            combined_wr = df.loc[combined_mask, 'target_covers'].mean()
        else:
            combined_wr = 0

        results.append({
            'signal': name,
            'n_total': int(n_signals),
            'win_rate': signal_wr,
            'edge_vs_base': edge,
            'roi_pct': roi,
            'combined_wr': combined_wr,
            'n_combined': int(n_combined),
        })

        marker = " ***" if signal_wr > 0.60 else ""
        print(f"\n{name}:")
        print(f"  Signals: {n_signals:,}")
        print(f"  Win Rate: {signal_wr:.1%} (edge: {edge:+.1%})")
        print(f"  ROI: {roi:+.1f}%")
        if n_combined >= 5:
            print(f"  + Lead >= 8: {combined_wr:.1%} ({n_combined} signals)")
        print(f"  {marker}")

    return results


def analyze_combined_signals(df):
    """Find the best combinations of breakout + ML features."""
    print("\n" + "="*70)
    print("COMBINED SIGNAL ANALYSIS - STACKING BREAKOUT FILTERS")
    print("="*70)

    base_rate = df['target_covers'].mean()

    # Test combinations
    combos = [
        # (name, filter_function)
        ("Lead 10+ & Trailer on Run & Momentum Breakout",
         lambda d: (d['abs_lead'] >= 10) & (d['trailer_on_run'] == 1) & (d['momentum_breakout'] == 1)),
        ("Lead 10+ & Death Cross (away momentum surge)",
         lambda d: (d['lead'] >= 10) & (d['ma_death_cross'] == 1)),
        ("Lead 10+ & Below Support (lead collapsing)",
         lambda d: (d['lead'] >= 10) & (d['below_support'] == 1)),
        ("Lead 8+ & BB Squeeze & Volume Confirmed",
         lambda d: (d['abs_lead'] >= 8) & (d['bb_squeeze'] == 1) & (d['volume_confirmed'] == 1)),
        ("Lead 12+ & Range Compression > 0.5 & Trailer on Run",
         lambda d: (d['abs_lead'] >= 12) & (d['range_compression'] > 0.5) & (d['trailer_on_run'] == 1)),
        ("Lead 8+ & Donchian Low Breakout (lead breaking down)",
         lambda d: (d['lead'] >= 8) & (d['donchian_breakout_low'] == 1)),
        ("Lead 8+ & Hurst < 0.45 (mean reverting) & Momentum ROC < -3",
         lambda d: (d['abs_lead'] >= 8) & (d['hurst'] < 0.45) & (d['momentum_roc'].abs() > 3)),
        ("Lead 15+ & Q3/Q4 & Trailer on Run",
         lambda d: (d['abs_lead'] >= 15) & (d['period'].isin([3, 4])) & (d['trailer_on_run'] == 1)),
        ("Lead 10+ & WP Divergence > 5 (market disagrees with lead)",
         lambda d: (d['abs_lead'] >= 10) & (d['wp_vs_lead_divergence'].abs() > 5)),
        ("Donchian Breakout (either) & Volume Confirmed & Lead 7+",
         lambda d: (d['abs_lead'] >= 7) & ((d['donchian_breakout_high'] == 1) | (d['donchian_breakout_low'] == 1)) & (d['volume_confirmed'] == 1)),
        ("Lead 10+ & Autocorr < -0.1 (mean reverting) & Lead Velocity < -2",
         lambda d: (d['abs_lead'] >= 10) & (d['autocorr_1'] < -0.1) & (d['lead_velocity_10'] < -2)),
        ("Lead 12+ & 5min Momentum Favors Trailer > 5pts",
         lambda d: (d['lead'] >= 12) & (d['net_5min_momentum'] < -5)),
        ("Lead 8+ & Golden Cross + Trailer Perspective",
         lambda d: (d['lead'] <= -8) & (d['ma_golden_cross'] == 1)),
    ]

    combo_results = []
    for name, filter_fn in combos:
        try:
            mask = filter_fn(df)
            n = mask.sum()
            if n < 5:
                continue

            wr = df.loc[mask, 'target_covers'].mean()
            ml_wr = df.loc[mask, 'target_wins'].mean()
            roi = (wr * 0.91 - (1 - wr)) * 100
            edge = wr - base_rate

            combo_results.append({
                'name': name,
                'n_signals': int(n),
                'cover_rate': wr,
                'ml_win_rate': ml_wr,
                'edge': edge,
                'roi_pct': roi,
            })

            marker = " ***HIGH EDGE***" if wr > 0.65 else ""
            print(f"\n{name}:")
            print(f"  Signals: {n:,}")
            print(f"  Cover Rate: {wr:.1%} (edge: {edge:+.1%})")
            print(f"  ML Win Rate: {ml_wr:.1%}")
            print(f"  Spread ROI: {roi:+.1f}%{marker}")
        except Exception as e:
            continue

    return combo_results


# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("NBA BREAKOUT + ML STRATEGY ENGINE")
    print("Training on 2,310 Real Games with Walk-Forward Validation")
    print("="*70)

    # Step 1: Load all games
    print("\n[1/6] Loading all games...")
    games = load_all_games()
    print(f"  Loaded {len(games)} games")

    seasons = set(g['season'] for g in games)
    for s in sorted(seasons):
        n = sum(1 for g in games if g['season'] == s)
        print(f"  {s}: {n} games")

    # Step 2: Build dataset
    print("\n[2/6] Extracting features from all games...")
    df = build_dataset(games)
    print(f"  Dataset: {len(df)} samples, {len(df.columns)} features")
    print(f"  Trailing team covers rate: {df['target_covers'].mean():.3f}")
    print(f"  Trailing team wins rate: {df['target_wins'].mean():.3f}")

    # Step 3: Analyze breakout signals
    print("\n[3/6] Analyzing breakout signals...")
    breakout_results = analyze_breakout_edge(df)

    # Step 4: Analyze combined signals
    print("\n[4/6] Analyzing combined breakout filters...")
    combo_results = analyze_combined_signals(df)

    # Step 5: Train ML model
    print("\n[5/6] Training ML model with walk-forward validation...")
    ml_results_covers = train_ml_model(df, target='target_covers')

    print("\n[5b/6] Training ML model for moneyline (outright wins)...")
    ml_results_wins = train_ml_model(df, target='target_wins')

    # Step 6: Evaluate strategy
    print("\n[6/6] Evaluating trading strategies...")
    spread_strat, best_spread_thresh = evaluate_strategy(ml_results_covers, 'target_covers')
    ml_strat, best_ml_thresh = evaluate_strategy(ml_results_wins, 'target_wins')

    # === FINAL REPORT ===
    print("\n" + "="*70)
    print("FINAL REPORT: BREAKOUT + ML STRATEGY")
    print("="*70)

    print("\n--- Top Features (XGBoost importance) ---")
    for feat, imp in ml_results_covers['top_features'][:15]:
        print(f"  {feat:40s} {imp:.4f}")

    print("\n--- Walk-Forward Results (Spread Covering) ---")
    for fold in ml_results_covers['fold_results']:
        print(f"  Train: {fold['train_seasons']} → Test: {fold['test_season']}")
        print(f"    Acc: {fold['accuracy']:.3f}  AUC: {fold['auc']:.3f}  Brier: {fold['brier']:.3f}")

    print("\n--- Walk-Forward Results (Moneyline Wins) ---")
    for fold in ml_results_wins['fold_results']:
        print(f"  Train: {fold['train_seasons']} → Test: {fold['test_season']}")
        print(f"    Acc: {fold['accuracy']:.3f}  AUC: {fold['auc']:.3f}  Brier: {fold['brier']:.3f}")

    # Best combo strategies
    if combo_results:
        print("\n--- Top Breakout Combinations ---")
        sorted_combos = sorted(combo_results, key=lambda x: -x['cover_rate'])
        for c in sorted_combos[:5]:
            print(f"  {c['name']}")
            print(f"    Signals: {c['n_signals']}, Cover: {c['cover_rate']:.1%}, "
                  f"ML Win: {c['ml_win_rate']:.1%}, ROI: {c['roi_pct']:+.1f}%")

    # Export results
    output = {
        'n_games': len(games),
        'n_samples': len(df),
        'base_cover_rate': float(df['target_covers'].mean()),
        'base_win_rate': float(df['target_wins'].mean()),
        'breakout_results': breakout_results,
        'combo_results': combo_results,
        'spread_strategy': spread_strat,
        'ml_strategy': ml_strat,
        'walk_forward_covers': ml_results_covers['fold_results'],
        'walk_forward_wins': ml_results_wins['fold_results'],
        'top_features': [(f, float(i)) for f, i in ml_results_covers['top_features']],
        'best_spread_threshold': best_spread_thresh,
        'best_ml_threshold': best_ml_thresh,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'breakout_ml_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'breakout_ml_results.json')}")

    # Also save the dataset for further analysis
    df.to_csv(os.path.join(OUTPUT_DIR, 'ml_features_dataset.csv'), index=False)
    print(f"Feature dataset saved to {os.path.join(OUTPUT_DIR, 'ml_features_dataset.csv')}")

    return output, df, ml_results_covers, ml_results_wins


if __name__ == '__main__':
    output, df, ml_covers, ml_wins = main()
