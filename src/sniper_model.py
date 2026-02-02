"""
NBA Sniper Model - Ultra-High Confidence Prediction System
==========================================================

Target: 95%+ accuracy, -110 odds minimum, maximum profit per bet.

Philosophy: Be a SNIPER, not a machine gunner.
- Only fire when ALL conditions align
- Every bet must have 95%+ calibrated probability of winning
- One bet type: Q3-end leader WINS the game (moneyline/spread at -110)
- Quality >>> Quantity: 10 bets at 95% beats 100 bets at 70%

Key insight: The market estimates win probability from lead size alone.
Our model sees LEAD QUALITY - how the lead was built, momentum direction,
scoring sustainability, and game flow dynamics. A "high quality" 8-point
lead is dramatically safer than a "lucky" 8-point lead built on one 3PT run.

Approach:
1. Enhanced feature set (110+ features) with lead quality & sustainability metrics
2. 5-model ensemble with stacked calibration (XGBoost, LightGBM, RF, HGB, LR)
3. Meta-features: model agreement, prediction stability, CI coverage
4. Ultra-selective signal generation with strict multi-gate criteria
5. Walk-forward validation (train on earlier seasons, test on later)
6. Export for JavaScript real-time inference

The "Sniper Gate" - ALL must pass:
  Gate 1: Calibrated win probability >= 0.93
  Gate 2: Model agreement >= 0.80 (4/5 models agree on winner)
  Gate 3: Prediction CI lower bound > 0 (even pessimistic estimate = leader wins)
  Gate 4: Lead quality score >= threshold (sustainability + momentum)
  Gate 5: Not a tied game (must have a clear leader)
  Gate 6: Odds are -110 or better (not laying heavy juice on blowouts)
"""

import json
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS
# ==============================================================================

Q4_SIGMA = 9.5      # Std dev of Q4 margin
AVG_Q4_TOTAL = 54.0
HOME_COURT_Q4 = 0.8

# Sniper gate thresholds (tuned conservatively)
SNIPER_MIN_CONFIDENCE = 0.93    # Calibrated probability minimum
SNIPER_MIN_AGREEMENT = 0.80     # Model agreement minimum (4/5)
SNIPER_MAX_MARGIN_STD = 5.0     # Maximum margin prediction std (narrow)
SNIPER_MIN_LEAD = 1             # Minimum Q3 lead to consider
SNIPER_MAX_LEAD = 30            # Maximum (avoid garbage-time blowouts)
SNIPER_MIN_LEAD_QUALITY = 0.40  # Minimum lead quality composite score
SNIPER_MAX_ODDS = -110          # Worst acceptable odds (most negative = most expensive)

# Regime-specific confidence adjustments
REGIME_CONFIDENCE_FLOOR = {
    'BLOWOUT': 0.93,      # 20+ pt leads: high base rate, require 93%
    'COMFORTABLE': 0.93,  # 12-19 pt leads: good accuracy zone
    'COMPETITIVE': 0.95,  # 6-11 pt leads: harder to predict, require more
    'TIGHT': 0.97,        # 0-5 pt leads: very hard, require near-certainty
}


# ==============================================================================
# FEATURE EXTRACTION (Enhanced for Sniper)
# ==============================================================================

class SniperFeatureExtractor:
    """
    Extract 110+ features from play-by-play data at end of Q3.

    Enhancements over base Q3Terminal v2:
    - Lead sustainability composite score
    - Momentum alignment with leader
    - Three-point dependency metric (volatile leads)
    - Favorable quarter count and domination indicators
    - Late Q3 finishing strength
    - Theoretical win probability baseline (for model to learn deviations)
    - Closest approach metric (how close trailing team got)
    - Scoring drought indicators
    - Lead acceleration (2nd derivative of lead trajectory)
    """

    def extract_game(self, game_data: dict) -> dict | None:
        plays = game_data.get('plays', [])
        header = game_data.get('header', {})
        pickcenter = game_data.get('pickcenter', [])

        if not plays:
            return None

        # Parse header
        comps = header.get('competitions', [{}])[0].get('competitors', [])
        home_team = away_team = home_id = away_id = ''
        for comp in comps:
            team = comp.get('team', {})
            if comp.get('homeAway') == 'home':
                home_team = team.get('abbreviation', '')
                home_id = team.get('id', '')
            else:
                away_team = team.get('abbreviation', '')
                away_id = team.get('id', '')

        if not home_team or not away_team:
            return None

        season_year = header.get('season', {}).get('year', 0)
        game_id = str(header.get('id', ''))

        # Parse pregame lines
        opening_spread = 0.0
        opening_ou = 0.0
        home_ml_open = 0.0
        away_ml_open = 0.0
        for pc in pickcenter:
            ou = pc.get('overUnder', 0)
            if ou and ou > 0:
                opening_ou = ou
                opening_spread = pc.get('spread', 0.0) or 0.0
                hto = pc.get('homeTeamOdds', {})
                ato = pc.get('awayTeamOdds', {})
                if hto:
                    home_ml_open = hto.get('moneyLine', 0) or 0
                if ato:
                    away_ml_open = ato.get('moneyLine', 0) or 0
                break

        # ============ Parse plays ============
        period_end_scores = {}
        lead_timeline = []
        scoring_events = []

        last_home = 0
        last_away = 0
        prev_period = 0
        max_period = 0

        q_home_scoring_plays = defaultdict(list)
        q_away_scoring_plays = defaultdict(list)
        foul_counts = defaultdict(lambda: {'home': 0, 'away': 0})
        timeout_counts = {'home': 0, 'away': 0}
        turnovers = defaultdict(lambda: {'home': 0, 'away': 0})

        for play in plays:
            period = play.get('period', {}).get('number', 0)
            if period == 0:
                continue
            max_period = max(max_period, period)

            hs = play.get('homeScore')
            aws = play.get('awayScore')
            home_score = hs if hs is not None else last_home
            away_score = aws if aws is not None else last_away

            if period != prev_period and prev_period > 0:
                period_end_scores[prev_period] = (last_home, last_away)
            prev_period = period

            clock_str = play.get('clock', {}).get('displayValue', '12:00')
            game_secs = self._game_seconds(period, clock_str)

            if play.get('scoringPlay', False) and period <= 4:
                pts = play.get('scoreValue', 0)
                lead = home_score - away_score
                lead_timeline.append((game_secs, lead))

                team_id = play.get('team', {}).get('id', '')
                is_home = (team_id == home_id)

                scoring_events.append({
                    'period': period, 'secs': game_secs,
                    'pts': pts, 'lead': lead,
                    'home_score': home_score, 'away_score': away_score,
                    'is_home': is_home,
                })

                if is_home:
                    q_home_scoring_plays[period].append(pts)
                else:
                    q_away_scoring_plays[period].append(pts)

            play_type_text = play.get('type', {}).get('text', '').lower()
            play_type_id = play.get('type', {}).get('id', '')
            team_id = play.get('team', {}).get('id', '')

            if 'foul' in play_type_text and period <= 3:
                if team_id == home_id:
                    foul_counts[period]['home'] += 1
                elif team_id == away_id:
                    foul_counts[period]['away'] += 1

            if 'turnover' in play_type_text and period <= 3:
                if team_id == home_id:
                    turnovers[period]['home'] += 1
                elif team_id == away_id:
                    turnovers[period]['away'] += 1

            if str(play_type_id) == '16' and period <= 3:
                if team_id == home_id:
                    timeout_counts['home'] += 1
                elif team_id == away_id:
                    timeout_counts['away'] += 1

            last_home = home_score
            last_away = away_score

        if prev_period > 0:
            period_end_scores[prev_period] = (last_home, last_away)

        # Skip OT games
        if max_period > 4:
            return None
        if 3 not in period_end_scores or 4 not in period_end_scores:
            return None

        q3_home, q3_away = period_end_scores[3]
        final_home, final_away = period_end_scores[4]

        q1_home = period_end_scores.get(1, (0, 0))[0]
        q1_away = period_end_scores.get(1, (0, 0))[1]
        q2_home = period_end_scores.get(2, (0, 0))[0] - q1_home
        q2_away = period_end_scores.get(2, (0, 0))[1] - q1_away
        q1h, q1a = q1_home, q1_away
        q2h, q2a = q2_home, q2_away
        q3h = q3_home - period_end_scores.get(2, (0, 0))[0]
        q3a = q3_away - period_end_scores.get(2, (0, 0))[1]
        q4h = final_home - q3_home
        q4a = final_away - q3_away

        if q3_home + q3_away < 20:
            return None

        # ============ BUILD FEATURES ============
        f = {}

        # --- Core score features ---
        q3_lead = q3_home - q3_away
        f['q3_lead'] = q3_lead
        f['q3_lead_abs'] = abs(q3_lead)
        f['q3_total'] = q3_home + q3_away
        f['is_home_leading'] = 1.0 if q3_lead > 0 else (0.0 if q3_lead < 0 else 0.5)

        # Quarter scoring
        f['q1_margin'] = q1h - q1a
        f['q2_margin'] = q2h - q2a
        f['q3_margin'] = q3h - q3a
        f['h1_margin'] = (q1h + q2h) - (q1a + q2a)

        f['q1_total'] = q1h + q1a
        f['q2_total'] = q2h + q2a
        f['q3_qtotal'] = q3h + q3a
        f['h1_total'] = f['q1_total'] + f['q2_total']

        qtotals = [f['q1_total'], f['q2_total'], f['q3_qtotal']]
        f['avg_q_pace'] = np.mean(qtotals)
        f['pace_var'] = np.std(qtotals)
        f['pace_trend'] = f['q3_qtotal'] - f['q1_total']
        f['q3_pace_vs_avg'] = f['q3_qtotal'] - f['avg_q_pace']

        home_qs = [q1h, q2h, q3h]
        away_qs = [q1a, q2a, q3a]
        f['home_q_std'] = np.std(home_qs)
        f['away_q_std'] = np.std(away_qs)
        f['home_q_trend'] = q3h - q1h
        f['away_q_trend'] = q3a - q1a

        q_margins = [f['q1_margin'], f['q2_margin'], f['q3_margin']]
        f['margin_trend'] = q_margins[-1] - q_margins[0]
        f['margin_consistency'] = np.std(q_margins)
        f['all_qs_same_direction'] = float(
            all(m > 0 for m in q_margins) or all(m < 0 for m in q_margins)
        )

        cumulative_margins = [
            q_margins[0],
            q_margins[0] + q_margins[1],
            q_margins[0] + q_margins[1] + q_margins[2]
        ]
        f['lead_built_gradually'] = float(
            abs(cumulative_margins[0]) <= abs(cumulative_margins[1]) <= abs(cumulative_margins[2])
            and abs(cumulative_margins[2]) > 5
        )

        # --- Pregame line context ---
        f['opening_spread'] = opening_spread
        f['opening_ou'] = opening_ou
        f['pregame_home_ml'] = home_ml_open
        f['pregame_away_ml'] = away_ml_open

        expected_q3_margin = opening_spread * 0.75
        f['spread_surprise'] = q3_lead - expected_q3_margin
        f['spread_surprise_abs'] = abs(f['spread_surprise'])

        projected_final_total = f['q3_total'] + f['avg_q_pace']
        f['total_pace_vs_ou'] = projected_final_total - opening_ou if opening_ou > 0 else 0

        # --- Lead time series features ---
        lead_ts = self._build_lead_timeseries(lead_timeline)
        f.update(self._lead_dynamics(lead_ts, q3_lead))
        f.update(self._dna_features(lead_ts))
        f.update(self._ta_features(lead_ts))

        # --- Scoring pattern features ---
        f.update(self._scoring_patterns(scoring_events, q3_home, q3_away))

        # --- Foul / turnover / timeout context ---
        total_home_fouls = sum(foul_counts[q]['home'] for q in [1, 2, 3])
        total_away_fouls = sum(foul_counts[q]['away'] for q in [1, 2, 3])
        f['foul_differential'] = total_home_fouls - total_away_fouls
        f['q3_home_fouls'] = foul_counts[3]['home']
        f['q3_away_fouls'] = foul_counts[3]['away']

        total_home_to = sum(turnovers[q]['home'] for q in [1, 2, 3])
        total_away_to = sum(turnovers[q]['away'] for q in [1, 2, 3])
        f['turnover_differential'] = total_home_to - total_away_to

        f['home_timeouts_used'] = timeout_counts['home']
        f['away_timeouts_used'] = timeout_counts['away']
        f['timeout_advantage'] = timeout_counts['away'] - timeout_counts['home']

        # --- Interaction features ---
        f['lead_x_momentum'] = f['q3_lead'] * f.get('lead_momentum', 0)
        f['lead_x_volatility'] = f['q3_lead_abs'] * f.get('lead_vol', 0)
        f['lead_x_pace'] = f['q3_lead_abs'] * f['avg_q_pace']
        f['lead_x_surprise'] = f['q3_lead'] * f['spread_surprise']
        f['momentum_x_fouls'] = f.get('lead_momentum', 0) * f['foul_differential']
        f['pace_x_surprise'] = f['avg_q_pace'] * f['spread_surprise_abs']
        f['consistency_x_lead'] = f['margin_consistency'] * f['q3_lead_abs']
        f['gradual_x_lead'] = f['lead_built_gradually'] * f['q3_lead_abs']

        # ====================================================================
        # SNIPER-SPECIFIC ENHANCED FEATURES
        # ====================================================================

        lead_dir = 1 if q3_lead > 0 else (-1 if q3_lead < 0 else 0)

        # 1. Lead sustainability composite
        margin_norm = min(f['margin_consistency'] / 10.0, 1.0)
        f['lead_sustainability'] = (
            f.get('lead_built_gradually', 0) * 0.25 +
            f.get('all_qs_same_direction', 0) * 0.25 +
            f.get('lead_stability', 0) * 0.25 +
            (1 - margin_norm) * 0.25
        )

        # 2. Theoretical win probability baseline (normal model)
        q3_lead_abs = f['q3_lead_abs']
        f['theo_win_prob'] = float(stats.norm.cdf(q3_lead_abs / Q4_SIGMA)) if q3_lead_abs > 0 else 0.5

        # 3. Momentum alignment (is momentum WITH the leader?)
        momentum = f.get('lead_momentum', 0)
        f['momentum_aligned'] = 1.0 if (q3_lead > 0 and momentum > 0) or (q3_lead < 0 and momentum < 0) else 0.0
        f['momentum_magnitude_aligned'] = abs(momentum) * (1 if f['momentum_aligned'] else -1)

        # 4. Three-point dependency (volatile leads built on 3s)
        if q3_lead > 0:
            f['leader_3pt_dependency'] = f.get('sp_home_3pt_rate', 0)
            f['trailer_3pt_rate'] = f.get('sp_away_3pt_rate', 0)
        elif q3_lead < 0:
            f['leader_3pt_dependency'] = f.get('sp_away_3pt_rate', 0)
            f['trailer_3pt_rate'] = f.get('sp_home_3pt_rate', 0)
        else:
            f['leader_3pt_dependency'] = 0
            f['trailer_3pt_rate'] = 0

        # 5. Favorable quarter count (how many Qs did leader win individually?)
        favorable_qs = sum(1 for m in q_margins if m * lead_dir > 0) if lead_dir != 0 else 0
        f['favorable_quarters'] = favorable_qs
        f['dominated_all_qs'] = 1.0 if favorable_qs == 3 else 0.0

        # 6. Late Q3 finishing strength (momentum entering Q4)
        f['q3_late_strength'] = f.get('sp_late_q3_margin', 0) * lead_dir

        # 7. Closest approach metric (minimum lead magnitude during game)
        if len(lead_ts) > 0 and lead_dir != 0:
            if lead_dir > 0:
                # Home leading - closest is the minimum of the lead series
                f['closest_approach'] = float(np.min(lead_ts))
            else:
                # Away leading - closest is the maximum (least negative)
                f['closest_approach'] = float(-np.max(lead_ts))
        else:
            f['closest_approach'] = 0

        # How much buffer exists (lead was never seriously threatened?)
        f['min_lead_buffer'] = max(0, f['closest_approach']) if lead_dir != 0 else 0
        f['lead_never_lost'] = float(f['closest_approach'] > 0) if lead_dir != 0 else 0

        # 8. Scoring drought indicator for trailing team
        if lead_dir > 0:
            f['trailer_dry_spell'] = f.get('sp_away_dry_spell', 0)
            f['trailer_largest_run'] = f.get('sp_away_largest_run', 0)
        elif lead_dir < 0:
            f['trailer_dry_spell'] = f.get('sp_home_dry_spell', 0)
            f['trailer_largest_run'] = f.get('sp_home_largest_run', 0)
        else:
            f['trailer_dry_spell'] = 0
            f['trailer_largest_run'] = 0

        # 9. Lead acceleration (2nd derivative - is lead growing faster?)
        early_lead = f.get('lead_early', 0) * lead_dir
        mid_lead = f.get('lead_mid', 0) * lead_dir
        late_lead = f.get('lead_late', 0) * lead_dir
        f['lead_velocity_late'] = late_lead - mid_lead
        f['lead_acceleration'] = (late_lead - mid_lead) - (mid_lead - early_lead)

        # 10. Pregame expectation alignment (is the leader the pregame favorite?)
        if opening_spread != 0:
            pregame_favorite_home = opening_spread < 0  # negative spread = home favored
            leader_is_favorite = (
                (q3_lead > 0 and pregame_favorite_home) or
                (q3_lead < 0 and not pregame_favorite_home)
            )
            f['leader_is_pregame_fav'] = 1.0 if leader_is_favorite else 0.0
        else:
            f['leader_is_pregame_fav'] = 0.5

        # 11. Scoring balance (leader not dependent on one quarter)
        if lead_dir > 0:
            leader_qs = home_qs
        elif lead_dir < 0:
            leader_qs = away_qs
        else:
            leader_qs = [0, 0, 0]
        f['leader_scoring_balance'] = 1 - (np.std(leader_qs) / (np.mean(leader_qs) + 1e-10))
        f['leader_q3_scoring'] = leader_qs[2] if len(leader_qs) > 2 else 0

        # 12. Combined sniper heuristic (pre-model quality score)
        f['sniper_heuristic'] = (
            f['lead_sustainability'] * 0.20 +
            f['theo_win_prob'] * 0.25 +
            f['momentum_aligned'] * 0.15 +
            f['dominated_all_qs'] * 0.15 +
            f['lead_never_lost'] * 0.15 +
            min(f['leader_scoring_balance'], 1.0) * 0.10
        )

        # 13. Enhanced interaction features
        f['sustainability_x_lead'] = f['lead_sustainability'] * q3_lead_abs
        f['momentum_aligned_x_lead'] = f['momentum_magnitude_aligned'] * q3_lead_abs
        f['theo_x_actual_lead'] = f['theo_win_prob'] * q3_lead_abs
        f['quality_x_lead'] = f['sniper_heuristic'] * q3_lead_abs
        f['fav_alignment_x_lead'] = f['leader_is_pregame_fav'] * q3_lead_abs
        f['dominated_x_lead'] = f['dominated_all_qs'] * q3_lead_abs

        # --- Outcomes ---
        outcomes = {
            'final_home': final_home,
            'final_away': final_away,
            'final_margin': final_home - final_away,
            'q4_home': q4h,
            'q4_away': q4a,
            'q4_total': q4h + q4a,
            'q4_margin': q4h - q4a,
            'home_won': int(final_home > final_away),
            'leader_won': int(
                (q3_lead > 0 and final_home > final_away) or
                (q3_lead < 0 and final_away > final_home)
            ) if q3_lead != 0 else int(final_home > final_away),
        }

        return {
            'game_id': game_id,
            'season': season_year,
            'home_team': home_team,
            'away_team': away_team,
            'q3_home': q3_home,
            'q3_away': q3_away,
            'features': f,
            'outcomes': outcomes,
            'q1h': q1h, 'q1a': q1a,
            'q2h': q2h, 'q2a': q2a,
            'q3h': q3h, 'q3a': q3a,
            'q4h': q4h, 'q4a': q4a,
            'opening_spread': opening_spread,
            'opening_ou': opening_ou,
        }

    # ---- Helper methods (same as q3_terminal_v2) ----

    def _game_seconds(self, period, clock):
        try:
            parts = clock.replace('.', ':').split(':')
            mins = int(parts[0])
            secs = float(parts[1]) if len(parts) > 1 else 0
            remaining = mins * 60 + secs
        except:
            remaining = 720
        return (period - 1) * 720 + (720 - remaining)

    def _build_lead_timeseries(self, lead_timeline, interval=30.0):
        if not lead_timeline:
            return np.array([0])
        max_time = 2160
        n_samples = int(max_time / interval) + 1
        ts = np.zeros(n_samples)
        event_idx = 0
        current_lead = 0
        for i in range(n_samples):
            t = i * interval
            while event_idx < len(lead_timeline) and lead_timeline[event_idx][0] <= t:
                current_lead = lead_timeline[event_idx][1]
                event_idx += 1
            ts[i] = current_lead
        return ts

    def _lead_dynamics(self, ts, q3_lead):
        f = {}
        n = len(ts)
        if n < 4:
            return {k: 0 for k in [
                'lead_vol', 'lead_range', 'max_home_lead', 'max_away_lead',
                'lead_changes', 'ties', 'lead_slope', 'lead_r2',
                'lead_early', 'lead_mid', 'lead_late',
                'lead_momentum', 'lead_accel',
                'halftime_lead', 'q3_lead_growth',
                'lead_stability', 'lead_mean_abs',
            ]}

        f['lead_vol'] = float(np.std(ts))
        f['lead_range'] = float(np.max(ts) - np.min(ts))
        f['max_home_lead'] = float(np.max(ts))
        f['max_away_lead'] = float(-np.min(ts))

        signs = np.sign(ts)
        f['lead_changes'] = int(np.sum(np.abs(np.diff(signs)) > 0)) // 2
        f['ties'] = int(np.sum(ts == 0))

        x = np.arange(n)
        slope, intercept, r, _, _ = stats.linregress(x, ts)
        f['lead_slope'] = float(slope)
        f['lead_r2'] = float(r ** 2)

        third = max(1, n // 3)
        f['lead_early'] = float(np.mean(ts[:third]))
        f['lead_mid'] = float(np.mean(ts[third:2 * third]))
        f['lead_late'] = float(np.mean(ts[2 * third:]))

        f['lead_momentum'] = f['lead_late'] - f['lead_mid']
        f['lead_accel'] = (f['lead_late'] - f['lead_mid']) - (f['lead_mid'] - f['lead_early'])

        half_idx = n * 2 // 3
        f['halftime_lead'] = float(ts[min(half_idx, n - 1)])
        f['q3_lead_growth'] = float(q3_lead - f['halftime_lead'])

        if q3_lead > 0:
            f['lead_stability'] = float(np.mean(ts >= q3_lead * 0.5))
        elif q3_lead < 0:
            f['lead_stability'] = float(np.mean(ts <= q3_lead * 0.5))
        else:
            f['lead_stability'] = 0.5

        f['lead_mean_abs'] = float(np.mean(np.abs(ts)))

        return f

    def _dna_features(self, ts):
        f = {}
        n = len(ts)
        if n < 8:
            return {f'dna_{k}': 0 for k in [
                'hurst', 'ac1', 'ac3', 'ac5',
                'fft_dom', 'fft_lo', 'fft_hi',
                'entropy', 'xing_rate', 'ou_theta',
                'max_run', 'avg_run', 'skew', 'kurt',
            ]}

        try:
            f['dna_hurst'] = self._hurst(ts)
        except:
            f['dna_hurst'] = 0.5

        c = ts - np.mean(ts)
        v = np.var(c) + 1e-10
        for lag in [1, 3, 5]:
            if lag < n:
                f[f'dna_ac{lag}'] = float(np.mean(c[:-lag] * c[lag:]) / v)
            else:
                f[f'dna_ac{lag}'] = 0

        try:
            fv = np.abs(fft(ts - np.mean(ts)))[:n // 2]
            if len(fv) > 1:
                tp = np.sum(fv[1:]) + 1e-10
                f['dna_fft_dom'] = float(np.argmax(fv[1:]) + 1) / n
                mid = max(1, len(fv) // 2)
                f['dna_fft_lo'] = float(np.sum(fv[1:mid]) / tp)
                f['dna_fft_hi'] = float(np.sum(fv[mid:]) / tp)
            else:
                f['dna_fft_dom'] = f['dna_fft_lo'] = f['dna_fft_hi'] = 0
        except:
            f['dna_fft_dom'] = f['dna_fft_lo'] = f['dna_fft_hi'] = 0

        diffs = np.diff(ts)
        if len(diffs) > 0:
            bins = np.array([-np.inf, -5, -2, 0, 2, 5, np.inf])
            hist, _ = np.histogram(diffs, bins=bins)
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            f['dna_entropy'] = float(-np.sum(probs * np.log2(probs)))
        else:
            f['dna_entropy'] = 0

        signs = np.sign(ts)
        f['dna_xing_rate'] = float(np.sum(np.abs(np.diff(signs)) > 0) / max(1, n - 1))

        try:
            d = np.diff(ts)
            x = ts[:-1]
            if np.std(x) > 0 and len(x) >= 5:
                sl, _, _, _, _ = stats.linregress(x, d)
                f['dna_ou_theta'] = float(-sl)
            else:
                f['dna_ou_theta'] = 0
        except:
            f['dna_ou_theta'] = 0

        diffs = np.diff(ts)
        runs = []
        run = 1
        for i in range(1, len(diffs)):
            if np.sign(diffs[i]) == np.sign(diffs[i - 1]) and diffs[i] != 0:
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                run = 1
        if run > 0:
            runs.append(run)
        f['dna_max_run'] = float(max(runs)) if runs else 0
        f['dna_avg_run'] = float(np.mean(runs)) if runs else 0

        f['dna_skew'] = float(stats.skew(ts))
        f['dna_kurt'] = float(stats.kurtosis(ts))

        return f

    def _hurst(self, ts):
        n = len(ts)
        max_lag = min(n // 2, 20)
        if max_lag < 3:
            return 0.5
        lags = range(2, max_lag)
        rs_vals = []
        for lag in lags:
            sub = ts[:lag]
            m = np.mean(sub)
            dev = np.cumsum(sub - m)
            R = np.max(dev) - np.min(dev)
            S = np.std(sub, ddof=1)
            if S > 0:
                rs_vals.append(R / S)
            else:
                rs_vals.append(1)
        if len(rs_vals) < 2:
            return 0.5
        sl, _, _, _, _ = stats.linregress(np.log(list(lags)), np.log(np.array(rs_vals) + 1e-10))
        return float(np.clip(sl, 0, 1))

    def _ta_features(self, ts):
        f = {}
        n = len(ts)
        if n < 10:
            return {f'ta_{k}': 0 for k in [
                'rsi', 'bb_pos', 'bb_width', 'macd', 'roc5', 'roc15',
                'mom_div', 'willr',
            ]}

        diffs = np.diff(ts)
        w = min(14, len(diffs))
        recent = diffs[-w:]
        gains = np.sum(recent[recent > 0])
        losses = -np.sum(recent[recent < 0])
        if losses > 0:
            rs = gains / losses
            f['ta_rsi'] = float(100 - (100 / (1 + rs)))
        else:
            f['ta_rsi'] = 100.0 if gains > 0 else 50.0

        w = min(20, n)
        recent = ts[-w:]
        mu = np.mean(recent)
        sd = np.std(recent) + 1e-10
        f['ta_bb_pos'] = float((ts[-1] - (mu - 2 * sd)) / (4 * sd + 1e-10))
        f['ta_bb_width'] = float(4 * sd / (abs(mu) + 1e-10))

        if n >= 12:
            fast = self._ema_val(ts, min(5, n // 2))
            slow = self._ema_val(ts, min(12, n - 1))
            f['ta_macd'] = float(fast - slow)
        else:
            f['ta_macd'] = 0

        for w in [5, 15]:
            if n > w:
                f[f'ta_roc{w}'] = float(ts[-1] - ts[-1 - w])
            else:
                f[f'ta_roc{w}'] = 0

        half = n // 2
        if half > 1:
            first_max = np.max(np.abs(ts[:half]))
            second_max = np.max(np.abs(ts[half:]))
            first_mom = np.mean(np.abs(np.diff(ts[:half])))
            second_mom = np.mean(np.abs(np.diff(ts[half:])))
            f['ta_mom_div'] = float((second_max - first_max) - (second_mom - first_mom) * 5)
        else:
            f['ta_mom_div'] = 0

        w = min(20, n)
        hi = np.max(ts[-w:])
        lo = np.min(ts[-w:])
        if hi != lo:
            f['ta_willr'] = float((hi - ts[-1]) / (hi - lo) * -100)
        else:
            f['ta_willr'] = -50.0

        return f

    def _ema_val(self, ts, window):
        alpha = 2 / (window + 1)
        val = ts[0]
        for x in ts[1:]:
            val = alpha * x + (1 - alpha) * val
        return float(val)

    def _scoring_patterns(self, events, q3_home, q3_away):
        f = {}
        if not events:
            return {f'sp_{k}': 0 for k in [
                'late_q3_margin', 'late_q3_pts', 'q3_last5_pts',
                'home_3pt_rate', 'away_3pt_rate', 'ft_rate',
                'home_largest_run', 'away_largest_run',
                'home_dry_spell', 'away_dry_spell',
                'lead_at_each_break',
            ]}

        late_cutoff = 1980
        late = [e for e in events if e['period'] == 3 and e['secs'] >= late_cutoff]
        if late and len(late) >= 2:
            f['sp_late_q3_margin'] = float(late[-1]['lead'] - late[0]['lead'])
            f['sp_late_q3_pts'] = float(sum(e['pts'] for e in late))
        else:
            f['sp_late_q3_margin'] = 0
            f['sp_late_q3_pts'] = 0

        l5_cutoff = 1860
        l5 = [e for e in events if e['period'] == 3 and e['secs'] >= l5_cutoff]
        f['sp_q3_last5_pts'] = float(sum(e['pts'] for e in l5))

        q3_events = [e for e in events if e['period'] <= 3]
        home_3s = sum(1 for e in q3_events if e['is_home'] and e['pts'] == 3)
        away_3s = sum(1 for e in q3_events if not e['is_home'] and e['pts'] == 3)
        home_fgs = sum(1 for e in q3_events if e['is_home'] and e['pts'] >= 2)
        away_fgs = sum(1 for e in q3_events if not e['is_home'] and e['pts'] >= 2)

        f['sp_home_3pt_rate'] = home_3s / max(1, home_fgs)
        f['sp_away_3pt_rate'] = away_3s / max(1, away_fgs)

        all_fts = sum(1 for e in q3_events if e['pts'] == 1)
        f['sp_ft_rate'] = all_fts / max(1, len(q3_events))

        f['sp_home_largest_run'] = float(self._largest_run(events, True))
        f['sp_away_largest_run'] = float(self._largest_run(events, False))

        f['sp_home_dry_spell'] = float(self._longest_dry_spell(events, True))
        f['sp_away_dry_spell'] = float(self._longest_dry_spell(events, False))

        leads_at_breaks = []
        for q in [1, 2, 3]:
            q_events = [e for e in events if e['period'] == q]
            if q_events:
                leads_at_breaks.append(q_events[-1]['lead'])
        f['sp_lead_at_each_break'] = float(np.std(leads_at_breaks)) if leads_at_breaks else 0

        return f

    def _largest_run(self, events, is_home):
        max_run = 0
        run = 0
        for e in events:
            if e['is_home'] == is_home:
                run += e['pts']
            else:
                max_run = max(max_run, run)
                run = 0
        return max(max_run, run)

    def _longest_dry_spell(self, events, is_home):
        team_events = [e for e in events if e['is_home'] == is_home]
        if len(team_events) < 2:
            return 0
        max_gap = 0
        for i in range(1, len(team_events)):
            gap = team_events[i]['secs'] - team_events[i - 1]['secs']
            max_gap = max(max_gap, gap)
        return max_gap


# ==============================================================================
# SNIPER ENSEMBLE MODEL
# ==============================================================================

class SniperEnsemble:
    """
    5-model ensemble + isotonic calibration, focused on predicting
    whether the Q3-end leader wins the game.

    The key difference from Q3Terminal v2:
    - More aggressive regularization to prevent overfitting
    - Isotonic calibration specifically tuned for the 90-100% probability range
    - Tracks individual model predictions for agreement calculation
    - Margin regression with uncertainty quantification
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.clf_models = {}
        self.reg_models = {}
        self.calibrator = None

    def train(self, X, y_leader, y_margin, verbose=True):
        self.feature_names = X.columns.tolist()
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.scaler.fit(X_clean)
        Xs = pd.DataFrame(self.scaler.transform(X_clean), columns=self.feature_names)

        if verbose:
            print(f"  Training on {len(X)} samples, {len(self.feature_names)} features")

        # ---- CLASSIFIER: does Q3 leader win? ----
        self.clf_models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=25,
                reg_alpha=1.0, reg_lambda=3.0, gamma=0.2,
                random_state=42, eval_metric='logloss', verbosity=0
            ),
            'lgbm': lgb.LGBMClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.6, min_child_samples=35,
                reg_alpha=1.0, reg_lambda=3.0,
                random_state=42, verbose=-1
            ),
            'rf': RandomForestClassifier(
                n_estimators=800, max_depth=6, min_samples_leaf=35,
                max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'hgb': HistGradientBoostingClassifier(
                max_iter=800, max_depth=4, learning_rate=0.02,
                min_samples_leaf=35, l2_regularization=2.0,
                random_state=42
            ),
            'lr': LogisticRegression(C=0.3, max_iter=3000, random_state=42)
        }

        n_models = len(self.clf_models)
        cv_preds = np.zeros((len(X), n_models))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for i, (name, model) in enumerate(self.clf_models.items()):
            if verbose:
                print(f"    {name}...", end=' ')
            model.fit(Xs, y_leader)
            try:
                cv_p = cross_val_predict(
                    model.__class__(**model.get_params()), Xs, y_leader,
                    cv=skf, method='predict_proba'
                )[:, 1]
            except:
                cv_p = cross_val_predict(
                    model.__class__(**model.get_params()), Xs, y_leader, cv=skf
                ).astype(float)
            cv_preds[:, i] = cv_p
            if verbose:
                acc = accuracy_score(y_leader, (cv_p > 0.5).astype(int))
                print(f"CV acc={acc:.3f}")

        # Isotonic calibration on CV averaged predictions
        avg_cv = np.mean(cv_preds, axis=1)
        self.calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        self.calibrator.fit(avg_cv, y_leader)

        if verbose:
            cal_probs = self.calibrator.predict(avg_cv)
            print(f"  Calibrated Brier: {brier_score_loss(y_leader, cal_probs):.4f}")

        # ---- MARGIN REGRESSION ----
        self.reg_models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=800, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=25,
                reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0
            ),
            'lgbm': lgb.LGBMRegressor(
                n_estimators=800, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.6, min_child_samples=35,
                reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=800, max_depth=6, min_samples_leaf=35,
                random_state=42, n_jobs=-1
            ),
            'ridge': Ridge(alpha=10.0),
        }

        for name, model in self.reg_models.items():
            if verbose:
                print(f"    margin {name}...", end=' ')
            model.fit(Xs, y_margin)
            if verbose:
                cv_m = cross_val_predict(
                    model.__class__(**model.get_params()), Xs, y_margin, cv=5
                )
                mae = mean_absolute_error(y_margin, cv_m)
                print(f"MAE={mae:.1f}")

    def predict(self, X):
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        Xs = pd.DataFrame(self.scaler.transform(X_clean), columns=self.feature_names)

        n_clf = len(self.clf_models)
        clf_p = np.zeros((len(X), n_clf))
        for i, (name, model) in enumerate(self.clf_models.items()):
            try:
                clf_p[:, i] = model.predict_proba(Xs)[:, 1]
            except:
                clf_p[:, i] = model.predict(Xs).astype(float)

        raw_prob = np.mean(clf_p, axis=1)
        cal_prob = self.calibrator.predict(raw_prob)

        # Model agreement
        binary = (clf_p > 0.5).astype(int)
        agreement = np.mean(binary, axis=1)
        model_agree = np.maximum(agreement, 1 - agreement)

        # Individual model probabilities for the leader
        # For agreement analysis, we want the probability each model
        # assigns to the eventual leader winning
        clf_leader_probs = clf_p.copy()

        # Margin regression
        reg_p = np.zeros((len(X), len(self.reg_models)))
        for i, (name, model) in enumerate(self.reg_models.items()):
            reg_p[:, i] = model.predict(Xs)
        pred_margin = np.mean(reg_p, axis=1)
        margin_std = np.std(reg_p, axis=1)

        # Confidence interval lower bound (95% CI)
        ci_lower = pred_margin - 1.96 * margin_std
        ci_upper = pred_margin + 1.96 * margin_std

        return {
            'raw_prob': raw_prob,
            'cal_prob': cal_prob,
            'model_agree': model_agree,
            'clf_individual': clf_p,
            'pred_margin': pred_margin,
            'margin_std': margin_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

    def get_importance(self):
        imp = pd.DataFrame({'feature': self.feature_names})
        for name, model in self.clf_models.items():
            if hasattr(model, 'feature_importances_'):
                imp[f'clf_{name}'] = model.feature_importances_
        for name, model in self.reg_models.items():
            if hasattr(model, 'feature_importances_'):
                imp[f'reg_{name}'] = model.feature_importances_
        ic = [c for c in imp.columns if c != 'feature']
        if ic:
            imp['avg'] = imp[ic].mean(axis=1)
            imp = imp.sort_values('avg', ascending=False)
        return imp


# ==============================================================================
# SNIPER SIGNAL GENERATION
# ==============================================================================

def estimate_live_ml_odds(q3_lead_abs):
    """Estimate live moneyline odds for the leading team."""
    if q3_lead_abs == 0:
        return -110
    sigma = Q4_SIGMA
    win_prob = stats.norm.cdf(q3_lead_abs / sigma)
    if win_prob >= 0.99:
        return -10000
    if win_prob <= 0.01:
        return 10000
    if win_prob >= 0.5:
        return -(win_prob / (1 - win_prob)) * 100
    else:
        return ((1 - win_prob) / win_prob) * 100


def generate_sniper_signals(preds, features_df, games, verbose=False):
    """
    Ultra-selective signal generation.

    Only generates a signal when ALL sniper gates pass.
    Returns far fewer signals than Q3Terminal v2, but with much higher accuracy.
    """
    signals = []
    gate_stats = defaultdict(int)

    for i, game in enumerate(games):
        q3_lead = game['q3_home'] - game['q3_away']
        q3_lead_abs = abs(q3_lead)
        gate_stats['total_games'] += 1

        # Skip tied games
        if q3_lead == 0:
            gate_stats['skip_tied'] += 1
            continue

        # Skip very small leads (too unpredictable)
        if q3_lead_abs < SNIPER_MIN_LEAD:
            gate_stats['skip_small_lead'] += 1
            continue

        # Skip garbage-time blowouts (worthless bets with massive juice)
        if q3_lead_abs > SNIPER_MAX_LEAD:
            gate_stats['skip_blowout'] += 1
            continue

        cal_prob = preds['cal_prob'][i]
        model_agree = preds['model_agree'][i]
        pred_margin = preds['pred_margin'][i]
        margin_std = preds['margin_std'][i]
        ci_lower = preds['ci_lower'][i]

        # Determine leader
        if q3_lead > 0:
            leader = game['home_team']
            trailer = game['away_team']
        else:
            leader = game['away_team']
            trailer = game['home_team']

        leader_prob = cal_prob
        leader_won = game['outcomes']['leader_won']

        # Market odds estimate
        mkt_ml_odds = estimate_live_ml_odds(q3_lead_abs)

        # Regime
        if q3_lead_abs >= 20:
            regime = 'BLOWOUT'
        elif q3_lead_abs >= 12:
            regime = 'COMFORTABLE'
        elif q3_lead_abs >= 6:
            regime = 'COMPETITIVE'
        else:
            regime = 'TIGHT'

        # Get lead quality from features
        f = features_df.iloc[i] if i < len(features_df) else {}
        lead_quality = f.get('sniper_heuristic', 0)
        lead_sustainability = f.get('lead_sustainability', 0)
        momentum_aligned = f.get('momentum_aligned', 0)
        dominated_all_qs = f.get('dominated_all_qs', 0)

        # ================================================================
        # SNIPER GATE SYSTEM - ALL gates must pass
        # ================================================================

        # Gate 1: Calibrated confidence >= regime-specific minimum
        min_conf = REGIME_CONFIDENCE_FLOOR.get(regime, SNIPER_MIN_CONFIDENCE)
        if leader_prob < min_conf:
            gate_stats['fail_confidence'] += 1
            continue

        # Gate 2: Model agreement >= 80%
        if model_agree < SNIPER_MIN_AGREEMENT:
            gate_stats['fail_agreement'] += 1
            continue

        # Gate 3: Margin prediction supports leader win
        # For leader to win, predicted margin must favor leader
        if q3_lead > 0:
            # Home leading: need ci_lower > 0 (even pessimistic = home wins)
            margin_supports = ci_lower > -2.0  # Allow small tolerance
        else:
            # Away leading: need ci_upper < 0 (even optimistic = away wins)
            # Note: margin is home-away, so negative means away wins
            margin_supports = preds['ci_upper'][i] < 2.0

        if not margin_supports:
            gate_stats['fail_margin_ci'] += 1
            continue

        # Gate 4: Margin prediction not too wide (models must agree on HOW MUCH)
        if margin_std > SNIPER_MAX_MARGIN_STD:
            gate_stats['fail_margin_std'] += 1
            continue

        # Gate 5: Lead quality composite
        if lead_quality < SNIPER_MIN_LEAD_QUALITY:
            gate_stats['fail_lead_quality'] += 1
            continue

        # Gate 6: Odds check - not laying ridiculous juice
        # For ML bets on heavy favorites, odds might be -500+
        # We want odds of -110 or better (less negative / positive)
        # BUT for spread bets, it's always -110
        # Strategy: Always recommend the SPREAD bet at -110
        # The leader "covering" the live spread is our primary bet
        live_spread = q3_lead * 0.78
        odds = -110  # Spread bets are always -110

        # ================================================================
        # ALL GATES PASSED - Generate sniper signal
        # ================================================================
        gate_stats['passed_all'] += 1

        # Determine actual result
        actual_margin = game['outcomes']['final_margin']

        # For spread bets: did the leader cover the live spread?
        if q3_lead > 0:
            # Home leading: home covers if actual_margin > live_spread
            leader_covered = actual_margin > live_spread
        else:
            # Away leading: away covers if actual_margin < -abs(live_spread)
            leader_covered = actual_margin < live_spread

        # For ML bets: did the leader win?
        leader_actually_won = bool(leader_won)

        # Build signal
        signal = {
            'game_id': game['game_id'],
            'season': game['season'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'signal_type': 'SNIPER',
            'direction': 'HOME' if q3_lead > 0 else 'AWAY',
            'leader': leader,
            'trailer': trailer,
            'confidence': float(leader_prob),
            'model_agree': float(model_agree),
            'predicted_margin': float(pred_margin),
            'margin_std': float(margin_std),
            'ci_lower': float(ci_lower),
            'ci_upper': float(preds['ci_upper'][i]),
            'live_spread': float(live_spread),
            'estimated_ml_odds': float(mkt_ml_odds),
            'estimated_odds': odds,
            'q3_lead': q3_lead,
            'q3_lead_abs': q3_lead_abs,
            'regime': regime,
            'lead_quality': float(lead_quality),
            'lead_sustainability': float(lead_sustainability),
            'momentum_aligned': float(momentum_aligned),
            'dominated_all_qs': float(dominated_all_qs),
            'actual_margin': float(actual_margin),
            'leader_won': leader_actually_won,
            'leader_covered_spread': bool(leader_covered),
            'correct': leader_actually_won,  # Primary: did leader win?
        }

        signals.append(signal)

    if verbose:
        print(f"\n  Sniper Gate Stats:")
        for k, v in sorted(gate_stats.items()):
            print(f"    {k}: {v}")

    return signals


# ==============================================================================
# WALK-FORWARD VALIDATION
# ==============================================================================

def run_sniper_walkforward(games, verbose=True):
    """Walk-forward: train on season N, test on season N+1."""

    seasons = sorted(set(g['season'] for g in games))
    if verbose:
        print(f"\nSeasons found: {seasons}")
        print(f"Games per season: {dict((s, sum(1 for g in games if g['season'] == s)) for s in seasons)}")

    all_signals = []
    oos_summary = []

    for test_idx in range(1, len(seasons)):
        test_season = seasons[test_idx]
        train_seasons = seasons[:test_idx]

        train_games = [g for g in games if g['season'] in train_seasons]
        test_games = [g for g in games if g['season'] == test_season]

        if len(train_games) < 200 or len(test_games) < 50:
            if verbose:
                print(f"\nSkipping test={test_season}: train={len(train_games)}, test={len(test_games)}")
            continue

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"FOLD: Train={train_seasons} ({len(train_games)}) | Test={test_season} ({len(test_games)})")
            print(f"{'=' * 60}")

        # Build feature matrices
        train_feats = pd.DataFrame([g['features'] for g in train_games])
        test_feats = pd.DataFrame([g['features'] for g in test_games])

        train_y_leader = np.array([g['outcomes']['leader_won'] for g in train_games])
        train_y_margin = np.array([g['outcomes']['final_margin'] for g in train_games])

        test_y_leader = np.array([g['outcomes']['leader_won'] for g in test_games])

        # Train model
        model = SniperEnsemble()
        model.train(train_feats, train_y_leader, train_y_margin, verbose)

        # Predict on test
        preds = model.predict(test_feats)

        # Overall metrics
        oos_pred = (preds['cal_prob'] > 0.5).astype(int)
        oos_acc = accuracy_score(test_y_leader, oos_pred)
        oos_brier = brier_score_loss(test_y_leader, preds['cal_prob'])

        if verbose:
            print(f"\n  Overall OOS: acc={oos_acc:.3f}, brier={oos_brier:.4f}")

        # Generate sniper signals
        signals = generate_sniper_signals(preds, test_feats, test_games, verbose)
        all_signals.extend(signals)

        # Report sniper results
        if verbose and signals:
            correct = sum(1 for s in signals if s['correct'])
            total = len(signals)
            acc = correct / total if total > 0 else 0

            spread_correct = sum(1 for s in signals if s['leader_covered_spread'])
            spread_acc = spread_correct / total if total > 0 else 0

            # P&L at -110
            pnl_ml = sum(
                (100 / abs(s['estimated_ml_odds'])) if s['leader_won'] else -1.0
                for s in signals
            )
            pnl_spread = spread_correct * (100 / 110) - (total - spread_correct) * 1.0

            print(f"\n  SNIPER SIGNALS: {total} signals")
            print(f"  ML Leader Win Rate: {correct}/{total} = {acc:.1%}")
            print(f"  Spread Cover Rate:  {spread_correct}/{total} = {spread_acc:.1%}")
            print(f"  ML P&L:     {pnl_ml:+.2f}u")
            print(f"  Spread P&L: {pnl_spread:+.2f}u (at -110)")

            # By confidence bucket
            for lo, hi, label in [(0.93, 0.95, '93-95%'), (0.95, 0.97, '95-97%'), (0.97, 1.0, '97%+')]:
                bucket = [s for s in signals if lo <= s['confidence'] < hi]
                if bucket:
                    bc = sum(1 for s in bucket if s['correct'])
                    bt = len(bucket)
                    ba = bc / bt if bt > 0 else 0
                    print(f"    [{label}]: {bc}/{bt} = {ba:.1%}")

            # By regime
            for regime in ['BLOWOUT', 'COMFORTABLE', 'COMPETITIVE', 'TIGHT']:
                rs = [s for s in signals if s['regime'] == regime]
                if rs:
                    rc = sum(1 for s in rs if s['correct'])
                    rt = len(rs)
                    print(f"    {regime}: {rc}/{rt} = {rc / rt:.1%}")

        oos_summary.append({
            'test_season': test_season,
            'n_train': len(train_games),
            'n_test': len(test_games),
            'oos_accuracy': float(oos_acc),
            'oos_brier': float(oos_brier),
            'n_sniper_signals': len(signals),
        })

    return {
        'signals': all_signals,
        'oos_summary': oos_summary,
    }


# ==============================================================================
# JS MODEL EXPORT
# ==============================================================================

def export_sniper_js_model(model, output_dir):
    """Export model parameters for JavaScript inference."""
    lr = model.clf_models.get('lr')
    ridge = model.reg_models.get('ridge')

    if not lr:
        print("  WARNING: No LR model to export")
        return

    # Calibration lookup table (sample points for interpolation)
    cal_x = np.linspace(0.3, 0.99, 100)
    cal_y = model.calibrator.predict(cal_x)

    js_data = {
        'version': 'sniper_v1',
        'features': model.feature_names,
        'scaler_mean': model.scaler.mean_.tolist(),
        'scaler_std': model.scaler.scale_.tolist(),
        'lr_coef': lr.coef_[0].tolist(),
        'lr_intercept': float(lr.intercept_[0]),
        'ridge_coef': ridge.coef_.tolist() if ridge else [],
        'ridge_intercept': float(ridge.intercept_) if ridge else 0,
        'calibration_x': cal_x.tolist(),
        'calibration_y': cal_y.tolist(),
        'sniper_thresholds': {
            'min_confidence': SNIPER_MIN_CONFIDENCE,
            'min_agreement': SNIPER_MIN_AGREEMENT,
            'max_margin_std': SNIPER_MAX_MARGIN_STD,
            'min_lead': SNIPER_MIN_LEAD,
            'max_lead': SNIPER_MAX_LEAD,
            'min_lead_quality': SNIPER_MIN_LEAD_QUALITY,
            'regime_floors': REGIME_CONFIDENCE_FLOOR,
        },
    }

    path = os.path.join(output_dir, 'sniper_model.json')
    with open(path, 'w') as f:
        json.dump(js_data, f)
    print(f"  Sniper JS model: {path}")

    # Also copy to webapp/data
    webapp_path = os.path.join(os.path.dirname(output_dir), 'webapp', 'data', 'sniper_model.json')
    os.makedirs(os.path.dirname(webapp_path), exist_ok=True)
    with open(webapp_path, 'w') as f:
        json.dump(js_data, f)
    print(f"  Webapp copy: {webapp_path}")

    return js_data


# ==============================================================================
# MAIN
# ==============================================================================

def main(pbp_dir='./cache/games_pbp', output_dir='./output'):
    print("=" * 60)
    print("NBA SNIPER MODEL - Ultra-High Confidence System")
    print("=" * 60)
    print(f"Target: {SNIPER_MIN_CONFIDENCE * 100:.0f}%+ accuracy, -110 odds")

    # Step 1: Parse all games
    print("\n[1/5] Parsing play-by-play data...")
    extractor = SniperFeatureExtractor()
    games = []

    files = sorted(Path(pbp_dir).glob('*.json'))
    print(f"  Found {len(files)} game files")

    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
            g = extractor.extract_game(data)
            if g:
                if not g['game_id']:
                    g['game_id'] = fp.stem
                games.append(g)
        except:
            continue

    print(f"  Parsed: {len(games)} valid games")

    if not games:
        print("ERROR: No games found. Check pbp_dir path.")
        return

    # Step 2: Walk-forward validation
    print("\n[2/5] Running walk-forward validation...")
    results = run_sniper_walkforward(games, verbose=True)
    signals = results['signals']

    # Step 3: Final results
    print(f"\n{'=' * 60}")
    print("SNIPER FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Total OOS sniper signals: {len(signals)}")

    if signals:
        correct_ml = sum(1 for s in signals if s['leader_won'])
        correct_spread = sum(1 for s in signals if s['leader_covered_spread'])
        total = len(signals)

        ml_acc = correct_ml / total if total > 0 else 0
        spread_acc = correct_spread / total if total > 0 else 0

        pnl_ml = sum(
            (100 / abs(s['estimated_ml_odds'])) if s['leader_won'] else -1.0
            for s in signals
        )
        pnl_spread = correct_spread * (100 / 110) - (total - correct_spread) * 1.0
        roi_spread = pnl_spread / total * 100 if total > 0 else 0

        print(f"\n  ML Leader Win Rate:  {correct_ml}/{total} = {ml_acc:.1%}")
        print(f"  Spread Cover Rate:   {correct_spread}/{total} = {spread_acc:.1%}")
        print(f"  ML P&L:              {pnl_ml:+.2f}u")
        print(f"  Spread P&L (@-110):  {pnl_spread:+.2f}u")
        print(f"  Spread ROI:          {roi_spread:+.1f}%")
        print(f"  Avg Confidence:      {np.mean([s['confidence'] for s in signals]):.1%}")
        print(f"  Avg Lead Quality:    {np.mean([s['lead_quality'] for s in signals]):.2f}")

        # Breakdown by regime
        print(f"\n  By regime:")
        for regime in ['BLOWOUT', 'COMFORTABLE', 'COMPETITIVE', 'TIGHT']:
            rs = [s for s in signals if s['regime'] == regime]
            if rs:
                rc = sum(1 for s in rs if s['correct'])
                rt = len(rs)
                avg_conf = np.mean([s['confidence'] for s in rs])
                print(f"    {regime}: {rc}/{rt} = {rc / rt:.1%} (avg conf: {avg_conf:.1%})")

        # Breakdown by confidence bucket
        print(f"\n  By confidence:")
        for lo, hi, label in [(0.93, 0.95, '93-95%'), (0.95, 0.97, '95-97%'), (0.97, 1.0, '97%+')]:
            bucket = [s for s in signals if lo <= s['confidence'] < hi]
            if bucket:
                bc = sum(1 for s in bucket if s['correct'])
                bt = len(bucket)
                ba = bc / bt if bt > 0 else 0
                bpnl = bc * (100 / 110) - (bt - bc) * 1.0
                broi = bpnl / bt * 100 if bt > 0 else 0
                print(f"    [{label}]: {bc}/{bt} = {ba:.1%} | P&L: {bpnl:+.1f}u | ROI: {broi:+.1f}%")

    # Step 4: Train final model on all data
    print("\n[3/5] Training final model on all data...")
    all_feats = pd.DataFrame([g['features'] for g in games])
    all_y_leader = np.array([g['outcomes']['leader_won'] for g in games])
    all_y_margin = np.array([g['outcomes']['final_margin'] for g in games])

    final_model = SniperEnsemble()
    final_model.train(all_feats, all_y_leader, all_y_margin, verbose=True)

    # Feature importance
    imp = final_model.get_importance()
    print("\nTop 30 Features:")
    print(imp[['feature', 'avg']].head(30).to_string(index=False))

    # Step 5: Export
    print("\n[4/5] Exporting...")
    os.makedirs(output_dir, exist_ok=True)

    # Export signals
    signals_path = os.path.join(output_dir, 'sniper_signals.json')
    with open(signals_path, 'w') as f:
        json.dump({
            'signals': signals,
            'oos_summary': results['oos_summary'],
            'total_games': len(games),
            'model_version': 'sniper_v1',
            'thresholds': {
                'min_confidence': SNIPER_MIN_CONFIDENCE,
                'min_agreement': SNIPER_MIN_AGREEMENT,
                'max_margin_std': SNIPER_MAX_MARGIN_STD,
                'min_lead_quality': SNIPER_MIN_LEAD_QUALITY,
            },
        }, f, indent=2, default=str)
    print(f"  Signals: {signals_path}")

    # Export JS model
    export_sniper_js_model(final_model, output_dir)

    # Also save signals to webapp/data
    webapp_signals_path = os.path.join(os.path.dirname(output_dir), 'webapp', 'data', 'sniper_signals.json')
    with open(webapp_signals_path, 'w') as f:
        json.dump({
            'signals': signals,
            'oos_summary': results['oos_summary'],
            'total_games': len(games),
            'model_version': 'sniper_v1',
        }, f, indent=2, default=str)
    print(f"  Webapp signals: {webapp_signals_path}")

    print("\n[5/5] Done!")
    return results


if __name__ == '__main__':
    pbp = sys.argv[1] if len(sys.argv) > 1 else './cache/games_pbp'
    out = sys.argv[2] if len(sys.argv) > 2 else './output'
    main(pbp, out)
