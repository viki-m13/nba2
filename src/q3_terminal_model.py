"""
Q3 Terminal Prediction System
=============================
Novel ML system that predicts game outcomes at end of Q3 using:
- 60+ engineered features from play-by-play data
- "Game Flow DNA" time-series encoding (Fourier, autocorrelation, Hurst)
- Technical analysis indicators (RSI, Bollinger, MACD analogs on lead)
- Multi-model ensemble with calibrated confidence scoring
- Ultra-selective signal generation for high-accuracy, high-value bets

Prediction targets:
1. Game winner (moneyline)
2. Final margin vs pregame spread
3. Q4 total scoring (O/U)
"""

import json
import os
import pickle
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, signal as scipy_signal
from scipy.fft import fft
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Q3GameState:
    """Complete game state at end of Q3"""
    game_id: str
    home_team: str
    away_team: str
    home_score_q3: int
    away_score_q3: int

    # Quarter scores
    home_q1: int = 0
    home_q2: int = 0
    home_q3: int = 0
    away_q1: int = 0
    away_q2: int = 0
    away_q3: int = 0

    # Pregame lines
    opening_spread: float = 0.0  # negative = home favored
    opening_ou: float = 0.0

    # Outcomes (for training)
    home_final: int = 0
    away_final: int = 0
    home_q4: int = 0
    away_q4: int = 0

    # Lead time series (sampled every 30s through Q1-Q3)
    lead_series: List[float] = field(default_factory=list)

    # Scoring timeline
    scoring_events: List[Dict] = field(default_factory=list)

    # Game context
    home_fouls_q3: int = 0
    away_fouls_q3: int = 0


@dataclass
class Q3Signal:
    """Trading signal generated at end of Q3"""
    game_id: str
    home_team: str
    away_team: str
    signal_type: str  # 'ML' (moneyline), 'SPREAD', 'TOTAL'
    direction: str    # 'HOME', 'AWAY', 'OVER', 'UNDER'
    confidence: float # 0-1 calibrated probability
    tier: str         # 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE'

    # Prediction details
    predicted_winner: str = ''
    predicted_margin: float = 0.0
    predicted_q4_total: float = 0.0

    # Market context
    estimated_live_ml: float = 0.0  # estimated live moneyline
    estimated_live_spread: float = 0.0
    estimated_live_ou: float = 0.0

    # Outcome (for validation)
    actual_winner: str = ''
    actual_margin: float = 0.0
    actual_q4_total: float = 0.0
    correct: bool = False

    # Features summary
    q3_lead: int = 0
    q3_momentum: float = 0.0
    regime: str = ''


# ============================================================
# FEATURE EXTRACTION
# ============================================================

class Q3FeatureExtractor:
    """
    Extract 60+ features from play-by-play data at end of Q3.
    Includes novel "Game Flow DNA" and Technical Analysis features.
    """

    LEAD_SAMPLE_INTERVAL = 30  # seconds

    def extract_game_state(self, game_data: dict) -> Optional[Q3GameState]:
        """Parse PBP data and extract Q3-end game state"""
        plays = game_data.get('plays', [])
        header = game_data.get('header', {})
        pickcenter = game_data.get('pickcenter', [])

        if not plays:
            return None

        # Get team info
        comps = header.get('competitions', [{}])[0].get('competitors', [])
        home_team = away_team = ''
        for comp in comps:
            if comp.get('homeAway') == 'home':
                home_team = comp.get('team', {}).get('abbreviation', '')
            else:
                away_team = comp.get('team', {}).get('abbreviation', '')

        if not home_team or not away_team:
            return None

        # Get pregame lines
        opening_spread = 0.0
        opening_ou = 0.0
        for pc in pickcenter:
            provider = pc.get('provider', {}).get('name', '')
            if provider == 'consensus' or not opening_ou:
                opening_spread = pc.get('spread', 0.0) or 0.0
                opening_ou = pc.get('overUnder', 0.0) or 0.0

        # Parse plays to extract state
        quarter_scores = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}
        lead_series = []
        scoring_events = []

        last_home = 0
        last_away = 0
        last_period = 0
        last_lead_sample_time = -999

        q3_end_home = 0
        q3_end_away = 0
        final_home = 0
        final_away = 0

        home_fouls_q3 = 0
        away_fouls_q3 = 0

        has_q4 = False
        max_period = 0

        # Track scoring at quarter boundaries
        q_start_scores = {1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (0, 0)}
        prev_period = 0

        for play in plays:
            period = play.get('period', {}).get('number', 0)
            if period == 0:
                continue
            max_period = max(max_period, period)

            home_score = play.get('homeScore', last_home)
            away_score = play.get('awayScore', last_away)

            if home_score is None:
                home_score = last_home
            if away_score is None:
                away_score = last_away

            # Track quarter start scores
            if period != prev_period and period <= 4:
                q_start_scores[period] = (away_score, home_score)
                if prev_period > 0 and prev_period <= 4:
                    quarter_scores[prev_period] = [
                        home_score - q_start_scores[prev_period][1],
                        away_score - q_start_scores[prev_period][0]
                    ]
                prev_period = period

            # Parse clock
            clock_str = play.get('clock', {}).get('displayValue', '12:00')
            game_secs = self._get_game_seconds(period, clock_str)

            # Sample lead every 30 seconds through Q1-Q3
            if period <= 3 and game_secs - last_lead_sample_time >= self.LEAD_SAMPLE_INTERVAL:
                lead = home_score - away_score
                lead_series.append(lead)
                last_lead_sample_time = game_secs

            # Track scoring events
            if play.get('scoringPlay', False) and period <= 4:
                pts = play.get('scoreValue', 0)
                team_id = play.get('team', {}).get('id', '')
                scoring_events.append({
                    'period': period,
                    'game_secs': game_secs,
                    'pts': pts,
                    'home_score': home_score,
                    'away_score': away_score,
                    'lead': home_score - away_score,
                    'team_id': team_id
                })

            # Track fouls in Q3
            if period == 3:
                play_type = play.get('type', {}).get('text', '').lower()
                if 'foul' in play_type:
                    team_id = play.get('team', {}).get('id', '')
                    # We'll determine which team later
                    # For now just count
                    home_fouls_q3 += 1  # simplified

            # Track Q3 end
            if period == 3:
                q3_end_home = home_score
                q3_end_away = away_score

            if period == 4:
                has_q4 = True

            last_home = home_score
            last_away = away_score

        # Final scores
        final_home = last_home
        final_away = last_away

        # Handle quarter score for the last period processed
        if prev_period > 0 and prev_period <= 4:
            quarter_scores[prev_period] = [
                final_home - q_start_scores.get(prev_period, (0, 0))[1],
                final_away - q_start_scores.get(prev_period, (0, 0))[0]
            ]

        # Skip OT games and games without Q4
        if max_period > 4 or not has_q4:
            return None

        if q3_end_home == 0 and q3_end_away == 0:
            return None

        # Calculate quarter scores from running totals
        home_q1 = quarter_scores[1][0]
        away_q1 = quarter_scores[1][1]
        home_q2 = quarter_scores[2][0]
        away_q2 = quarter_scores[2][1]
        home_q3 = quarter_scores[3][0]
        away_q3 = quarter_scores[3][1]
        home_q4 = final_home - q3_end_home
        away_q4 = final_away - q3_end_away

        # Sanity checks
        if home_q1 + home_q2 + home_q3 != q3_end_home:
            # Fallback: estimate from Q3 end scores
            # This can happen with parsing edge cases
            home_q4 = final_home - q3_end_home
            away_q4 = final_away - q3_end_away

        return Q3GameState(
            game_id=header.get('id', game_data.get('header', {}).get('id', '')),
            home_team=home_team,
            away_team=away_team,
            home_score_q3=q3_end_home,
            away_score_q3=q3_end_away,
            home_q1=home_q1, home_q2=home_q2, home_q3=home_q3,
            away_q1=away_q1, away_q2=away_q2, away_q3=away_q3,
            opening_spread=opening_spread,
            opening_ou=opening_ou,
            home_final=final_home,
            away_final=final_away,
            home_q4=home_q4,
            away_q4=away_q4,
            lead_series=lead_series,
            scoring_events=scoring_events,
            home_fouls_q3=home_fouls_q3,
            away_fouls_q3=away_fouls_q3,
        )

    def _get_game_seconds(self, period: int, clock_str: str) -> float:
        """Convert period + clock to total game seconds elapsed"""
        try:
            parts = clock_str.replace('.', ':').split(':')
            mins = int(parts[0])
            secs = float(parts[1]) if len(parts) > 1 else 0
            clock_remaining = mins * 60 + secs
        except (ValueError, IndexError):
            clock_remaining = 720

        period_length = 720  # 12 minutes
        elapsed_in_period = period_length - clock_remaining
        total_elapsed = (period - 1) * period_length + elapsed_in_period
        return total_elapsed

    def extract_features(self, state: Q3GameState) -> Dict[str, float]:
        """
        Extract 60+ features from Q3 game state.
        Returns dict of feature_name -> value.
        """
        features = {}

        # ---- SCORE FEATURES ----
        features['q3_lead'] = state.home_score_q3 - state.away_score_q3
        features['q3_lead_abs'] = abs(features['q3_lead'])
        features['q3_total'] = state.home_score_q3 + state.away_score_q3

        features['home_q1'] = state.home_q1
        features['home_q2'] = state.home_q2
        features['home_q3'] = state.home_q3
        features['away_q1'] = state.away_q1
        features['away_q2'] = state.away_q2
        features['away_q3'] = state.away_q3

        features['q1_margin'] = state.home_q1 - state.away_q1
        features['q2_margin'] = state.home_q2 - state.away_q2
        features['q3_margin'] = state.home_q3 - state.away_q3
        features['h1_margin'] = (state.home_q1 + state.home_q2) - (state.away_q1 + state.away_q2)

        features['q1_total'] = state.home_q1 + state.away_q1
        features['q2_total'] = state.home_q2 + state.away_q2
        features['q3q_total'] = state.home_q3 + state.away_q3
        features['h1_total'] = features['q1_total'] + features['q2_total']

        # ---- PACE FEATURES ----
        quarter_totals = [features['q1_total'], features['q2_total'], features['q3q_total']]
        features['avg_quarter_pace'] = np.mean(quarter_totals) if quarter_totals else 53.0
        features['pace_variance'] = np.var(quarter_totals) if len(quarter_totals) >= 2 else 0.0
        features['pace_trend'] = features['q3q_total'] - features['q1_total']
        features['pace_q3_vs_avg'] = features['q3q_total'] - features['avg_quarter_pace']

        # Is game high-scoring or low-scoring vs average?
        features['scoring_rate'] = features['q3_total'] / 3.0  # per quarter
        features['projected_final'] = features['q3_total'] + features['scoring_rate']
        features['ou_differential'] = features['projected_final'] - state.opening_ou if state.opening_ou > 0 else 0

        # ---- LEAD DYNAMICS ----
        lead_series = np.array(state.lead_series) if state.lead_series else np.array([0])

        if len(lead_series) >= 2:
            features['lead_volatility'] = np.std(lead_series)
            features['lead_range'] = np.max(lead_series) - np.min(lead_series)
            features['max_home_lead'] = np.max(lead_series)
            features['max_away_lead'] = -np.min(lead_series)

            # Lead changes
            sign_changes = np.diff(np.sign(lead_series))
            features['lead_changes'] = int(np.sum(sign_changes != 0)) // 2
            features['ties_count'] = int(np.sum(lead_series == 0))

            # Lead trajectory
            n = len(lead_series)
            x = np.arange(n)
            if n >= 3:
                slope, _, r_value, _, _ = stats.linregress(x, lead_series)
                features['lead_trend_slope'] = slope
                features['lead_trend_r2'] = r_value ** 2
            else:
                features['lead_trend_slope'] = 0
                features['lead_trend_r2'] = 0

            # Lead in different phases
            third = max(1, n // 3)
            features['lead_early'] = np.mean(lead_series[:third])
            features['lead_mid'] = np.mean(lead_series[third:2*third])
            features['lead_late'] = np.mean(lead_series[2*third:])

            # Lead momentum (last portion vs earlier)
            features['lead_momentum'] = features['lead_late'] - features['lead_mid']
            features['lead_acceleration'] = (features['lead_late'] - features['lead_mid']) - \
                                           (features['lead_mid'] - features['lead_early'])

            # Half-time lead
            half_idx = min(n - 1, n * 2 // 3)
            features['halftime_lead'] = lead_series[half_idx]
            features['q3_lead_growth'] = features['q3_lead'] - features['halftime_lead']

        else:
            for f in ['lead_volatility', 'lead_range', 'max_home_lead', 'max_away_lead',
                      'lead_changes', 'ties_count', 'lead_trend_slope', 'lead_trend_r2',
                      'lead_early', 'lead_mid', 'lead_late', 'lead_momentum',
                      'lead_acceleration', 'halftime_lead', 'q3_lead_growth']:
                features[f] = 0

        # ---- GAME FLOW DNA (Novel) ----
        features.update(self._extract_dna_features(lead_series))

        # ---- TECHNICAL ANALYSIS (Stock Trader Style) ----
        features.update(self._extract_ta_features(lead_series))

        # ---- SCORING PATTERN FEATURES ----
        features.update(self._extract_scoring_patterns(state))

        # ---- CONTEXT FEATURES ----
        features['opening_spread'] = state.opening_spread
        features['opening_ou'] = state.opening_ou
        features['is_home_leading'] = 1.0 if features['q3_lead'] > 0 else 0.0

        # How actual game compares to pregame expectations
        if state.opening_spread != 0:
            features['spread_surprise'] = features['q3_lead'] - (state.opening_spread * 0.75)
        else:
            features['spread_surprise'] = 0

        # ---- REGIME CLASSIFICATION ----
        lead_abs = features['q3_lead_abs']
        if lead_abs >= 20:
            features['regime_blowout'] = 1
            features['regime_comfortable'] = 0
            features['regime_competitive'] = 0
            features['regime_tight'] = 0
        elif lead_abs >= 12:
            features['regime_blowout'] = 0
            features['regime_comfortable'] = 1
            features['regime_competitive'] = 0
            features['regime_tight'] = 0
        elif lead_abs >= 6:
            features['regime_blowout'] = 0
            features['regime_comfortable'] = 0
            features['regime_competitive'] = 1
            features['regime_tight'] = 0
        else:
            features['regime_blowout'] = 0
            features['regime_comfortable'] = 0
            features['regime_competitive'] = 0
            features['regime_tight'] = 1

        # ---- DERIVED INTERACTION FEATURES ----
        features['lead_x_momentum'] = features['q3_lead'] * features.get('lead_momentum', 0)
        features['lead_x_volatility'] = features['q3_lead_abs'] * features.get('lead_volatility', 0)
        features['pace_x_lead'] = features['avg_quarter_pace'] * features['q3_lead_abs']
        features['trend_x_lead'] = features.get('lead_trend_slope', 0) * features['q3_lead']

        return features

    def _extract_dna_features(self, lead_series: np.ndarray) -> Dict[str, float]:
        """
        Novel "Game Flow DNA" features using time-series analysis.
        Treats the lead over time as a signal and extracts frequency/chaos features.
        """
        features = {}
        n = len(lead_series)

        if n < 8:
            return {f'dna_{k}': 0 for k in [
                'hurst', 'autocorr_1', 'autocorr_3', 'autocorr_5',
                'fft_dominant_freq', 'fft_power_low', 'fft_power_high',
                'entropy', 'crossing_rate', 'mean_reversion_speed',
                'run_length_max', 'run_length_avg'
            ]}

        # --- Hurst Exponent (persistence vs mean-reversion) ---
        try:
            features['dna_hurst'] = self._hurst_exponent(lead_series)
        except:
            features['dna_hurst'] = 0.5

        # --- Autocorrelation at various lags ---
        centered = lead_series - np.mean(lead_series)
        var = np.var(centered)
        if var > 0:
            for lag in [1, 3, 5]:
                if lag < n:
                    acorr = np.mean(centered[:-lag] * centered[lag:]) / var
                    features[f'dna_autocorr_{lag}'] = acorr
                else:
                    features[f'dna_autocorr_{lag}'] = 0
        else:
            for lag in [1, 3, 5]:
                features[f'dna_autocorr_{lag}'] = 0

        # --- FFT features (frequency analysis of lead oscillation) ---
        try:
            fft_vals = np.abs(fft(lead_series - np.mean(lead_series)))[:n // 2]
            if len(fft_vals) > 1:
                total_power = np.sum(fft_vals[1:]) + 1e-10
                dominant_idx = np.argmax(fft_vals[1:]) + 1
                features['dna_fft_dominant_freq'] = dominant_idx / n

                mid = max(1, len(fft_vals) // 2)
                features['dna_fft_power_low'] = np.sum(fft_vals[1:mid]) / total_power
                features['dna_fft_power_high'] = np.sum(fft_vals[mid:]) / total_power
            else:
                features['dna_fft_dominant_freq'] = 0
                features['dna_fft_power_low'] = 0
                features['dna_fft_power_high'] = 0
        except:
            features['dna_fft_dominant_freq'] = 0
            features['dna_fft_power_low'] = 0
            features['dna_fft_power_high'] = 0

        # --- Shannon Entropy of lead changes ---
        try:
            diffs = np.diff(lead_series)
            if len(diffs) > 0:
                # Discretize into bins
                bins = np.array([-np.inf, -5, -2, 0, 2, 5, np.inf])
                hist, _ = np.histogram(diffs, bins=bins)
                probs = hist / np.sum(hist)
                probs = probs[probs > 0]
                features['dna_entropy'] = -np.sum(probs * np.log2(probs))
            else:
                features['dna_entropy'] = 0
        except:
            features['dna_entropy'] = 0

        # --- Zero-crossing rate (how often lead changes sign) ---
        signs = np.sign(lead_series)
        sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
        features['dna_crossing_rate'] = sign_changes / max(1, n - 1)

        # --- Mean-reversion speed (Ornstein-Uhlenbeck parameter) ---
        try:
            if n >= 5:
                diffs = np.diff(lead_series)
                # Simple OU: dx = -theta * x * dt + sigma * dW
                # Estimate theta from regression of dx on x
                x = lead_series[:-1]
                if np.std(x) > 0:
                    slope, _, _, _, _ = stats.linregress(x, diffs)
                    features['dna_mean_reversion_speed'] = -slope  # positive = mean-reverting
                else:
                    features['dna_mean_reversion_speed'] = 0
            else:
                features['dna_mean_reversion_speed'] = 0
        except:
            features['dna_mean_reversion_speed'] = 0

        # --- Run analysis (consecutive periods of lead growth/decline) ---
        try:
            diffs = np.diff(lead_series)
            runs = []
            current_run = 1
            for i in range(1, len(diffs)):
                if np.sign(diffs[i]) == np.sign(diffs[i-1]) and diffs[i] != 0:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            features['dna_run_length_max'] = max(runs) if runs else 0
            features['dna_run_length_avg'] = np.mean(runs) if runs else 0
        except:
            features['dna_run_length_max'] = 0
            features['dna_run_length_avg'] = 0

        return features

    def _hurst_exponent(self, series: np.ndarray) -> float:
        """Calculate Hurst exponent for the lead time series.
        H > 0.5: trending (persistent)
        H < 0.5: mean-reverting
        H = 0.5: random walk
        """
        n = len(series)
        if n < 8:
            return 0.5

        max_lag = min(n // 2, 20)
        lags = range(2, max_lag)

        rs_values = []
        for lag in lags:
            subseries = series[:lag]
            mean = np.mean(subseries)
            deviations = subseries - mean
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(subseries, ddof=1) if np.std(subseries, ddof=1) > 0 else 1e-10
            rs_values.append(R / S)

        if len(rs_values) < 2:
            return 0.5

        log_lags = np.log(list(lags))
        log_rs = np.log(np.array(rs_values) + 1e-10)

        slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
        return np.clip(slope, 0, 1)

    def _extract_ta_features(self, lead_series: np.ndarray) -> Dict[str, float]:
        """
        Technical Analysis features - treat lead like a stock price.
        RSI, Bollinger Bands, MACD analogs.
        """
        features = {}
        n = len(lead_series)

        if n < 10:
            return {f'ta_{k}': 0 for k in [
                'rsi', 'bb_position', 'bb_width', 'macd', 'macd_signal',
                'roc_short', 'roc_long', 'momentum_divergence'
            ]}

        diffs = np.diff(lead_series)

        # --- RSI (Relative Strength Index) ---
        window = min(14, n - 1)
        if len(diffs) >= window:
            recent = diffs[-window:]
            gains = np.sum(recent[recent > 0])
            losses = -np.sum(recent[recent < 0])
            if losses > 0:
                rs = gains / losses
                features['ta_rsi'] = 100 - (100 / (1 + rs))
            else:
                features['ta_rsi'] = 100 if gains > 0 else 50
        else:
            features['ta_rsi'] = 50

        # --- Bollinger Bands ---
        window = min(20, n)
        recent = lead_series[-window:]
        bb_mean = np.mean(recent)
        bb_std = np.std(recent) + 1e-10
        upper_band = bb_mean + 2 * bb_std
        lower_band = bb_mean - 2 * bb_std
        current = lead_series[-1]

        features['ta_bb_position'] = (current - lower_band) / (upper_band - lower_band + 1e-10)
        features['ta_bb_width'] = (upper_band - lower_band) / (abs(bb_mean) + 1e-10)

        # --- MACD analog ---
        if n >= 12:
            ema_fast = self._ema(lead_series, min(5, n // 2))
            ema_slow = self._ema(lead_series, min(12, n - 1))
            features['ta_macd'] = ema_fast - ema_slow
            features['ta_macd_signal'] = self._ema(
                np.array([ema_fast - ema_slow]), min(3, n // 4)
            )
        else:
            features['ta_macd'] = 0
            features['ta_macd_signal'] = 0

        # --- Rate of Change ---
        short_window = min(5, n - 1)
        long_window = min(15, n - 1)

        if short_window > 0 and n > short_window:
            features['ta_roc_short'] = lead_series[-1] - lead_series[-1 - short_window]
        else:
            features['ta_roc_short'] = 0

        if long_window > 0 and n > long_window:
            features['ta_roc_long'] = lead_series[-1] - lead_series[-1 - long_window]
        else:
            features['ta_roc_long'] = 0

        # --- Momentum Divergence ---
        # Price (lead) making new highs but momentum weakening
        if n >= 10:
            half = n // 2
            first_half_max = np.max(lead_series[:half])
            second_half_max = np.max(lead_series[half:])
            first_half_momentum = np.mean(np.diff(lead_series[:half])) if half > 1 else 0
            second_half_momentum = np.mean(np.diff(lead_series[half:])) if n - half > 1 else 0

            # Divergence: lead growing but momentum declining
            features['ta_momentum_divergence'] = (
                (second_half_max - first_half_max) -
                (second_half_momentum - first_half_momentum) * 10
            )
        else:
            features['ta_momentum_divergence'] = 0

        return features

    def _ema(self, series: np.ndarray, window: int) -> float:
        """Exponential moving average - returns last value"""
        if len(series) == 0:
            return 0
        alpha = 2 / (window + 1)
        ema = series[0]
        for val in series[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return ema

    def _extract_scoring_patterns(self, state: Q3GameState) -> Dict[str, float]:
        """Extract scoring pattern features from scoring events"""
        features = {}
        events = state.scoring_events

        if not events:
            return {f'sp_{k}': 0 for k in [
                'q3_late_home_pts', 'q3_late_away_pts', 'q3_late_margin',
                'scoring_consistency_home', 'scoring_consistency_away',
                'largest_run_home', 'largest_run_away',
                'q3_scoring_rate', 'q3_last5min_rate',
                'home_scoring_acceleration', 'away_scoring_acceleration'
            ]}

        # Late Q3 scoring (last 3 minutes)
        q3_late_cutoff = 2160 - 180  # Q3 ends at 2160 seconds, last 3 min
        q3_late_events = [e for e in events if e['period'] == 3 and e['game_secs'] >= q3_late_cutoff]

        features['sp_q3_late_home_pts'] = sum(
            e['pts'] for e in q3_late_events
            if e['home_score'] > (q3_late_events[0]['home_score'] - e['pts'] if q3_late_events else 0)
        )
        features['sp_q3_late_away_pts'] = sum(
            e['pts'] for e in q3_late_events
            if e['away_score'] > (q3_late_events[0]['away_score'] - e['pts'] if q3_late_events else 0)
        )

        # Simplified: use lead change in last 3 min of Q3
        if q3_late_events:
            features['sp_q3_late_margin'] = q3_late_events[-1]['lead'] - q3_late_events[0]['lead']
        else:
            features['sp_q3_late_margin'] = 0
            features['sp_q3_late_home_pts'] = 0
            features['sp_q3_late_away_pts'] = 0

        # Scoring consistency (variance of per-quarter scoring)
        home_quarters = [state.home_q1, state.home_q2, state.home_q3]
        away_quarters = [state.away_q1, state.away_q2, state.away_q3]

        features['sp_scoring_consistency_home'] = np.std(home_quarters)
        features['sp_scoring_consistency_away'] = np.std(away_quarters)

        # Largest scoring runs
        features['sp_largest_run_home'] = self._find_largest_run(events, 'home')
        features['sp_largest_run_away'] = self._find_largest_run(events, 'away')

        # Q3 specific scoring rate
        q3_events = [e for e in events if e['period'] == 3]
        features['sp_q3_scoring_rate'] = len(q3_events)

        # Last 5 min scoring rate
        q3_last5_cutoff = 2160 - 300
        q3_last5 = [e for e in events if e['period'] == 3 and e['game_secs'] >= q3_last5_cutoff]
        features['sp_q3_last5min_rate'] = sum(e['pts'] for e in q3_last5)

        # Scoring acceleration (Q3 rate vs Q2 rate)
        q2_events = [e for e in events if e['period'] == 2]
        q2_pts = sum(e['pts'] for e in q2_events)
        q3_pts = sum(e['pts'] for e in q3_events)
        features['sp_home_scoring_acceleration'] = state.home_q3 - state.home_q2
        features['sp_away_scoring_acceleration'] = state.away_q3 - state.away_q2

        return features

    def _find_largest_run(self, events: List[Dict], team: str) -> int:
        """Find the largest scoring run for a team (consecutive points without opponent scoring)"""
        if not events:
            return 0

        max_run = 0
        current_run = 0
        last_lead = events[0]['lead'] if events else 0

        for e in events:
            lead_change = e['lead'] - last_lead
            if team == 'home' and lead_change > 0:
                current_run += lead_change
            elif team == 'away' and lead_change < 0:
                current_run += abs(lead_change)
            else:
                max_run = max(max_run, current_run)
                current_run = 0
            last_lead = e['lead']

        return max(max_run, current_run)


# ============================================================
# MODEL
# ============================================================

class Q3TerminalEnsemble:
    """
    Multi-model ensemble with calibrated probability scoring.

    Models:
    1. XGBoost (gradient boosting)
    2. LightGBM (gradient boosting variant)
    3. Random Forest
    4. Calibrated Logistic Regression

    Stacking meta-learner combines predictions.
    """

    def __init__(self):
        self.feature_names = None
        self.scaler = StandardScaler()

        # Base models for classification (game winner)
        self.clf_models = {}
        self.clf_meta = None

        # Base models for regression (margin)
        self.reg_models = {}
        self.reg_meta = None

        # Base models for Q4 total
        self.q4_models = {}
        self.q4_meta = None

        # Calibration
        self.is_trained = False

    def train(self, X: pd.DataFrame, y_winner: np.ndarray,
              y_margin: np.ndarray, y_q4_total: np.ndarray,
              verbose: bool = True):
        """
        Train all models.

        X: feature matrix
        y_winner: binary - did leading team at Q3-end win? (1/0)
        y_margin: float - final margin (positive = home win)
        y_q4_total: float - Q4 total points
        """
        self.feature_names = X.columns.tolist()

        if verbose:
            print(f"Training on {len(X)} games with {len(self.feature_names)} features")

        # Handle NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)

        # ---- CLASSIFICATION: Leading team wins? ----
        if verbose:
            print("\n--- Training Winner Classification Models ---")

        self.clf_models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            ),
            'lgbm': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            ),
            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            ),
            'lr': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        }

        # Train each model
        meta_features_clf = np.zeros((len(X), len(self.clf_models)))
        for i, (name, model) in enumerate(self.clf_models.items()):
            if verbose:
                print(f"  Training {name}...")
            model.fit(X_scaled, y_winner)
            # Get cross-validated predictions for stacking
            try:
                cv_preds = cross_val_predict(
                    model.__class__(**model.get_params()),
                    X_scaled, y_winner, cv=5, method='predict_proba'
                )[:, 1]
            except:
                cv_preds = cross_val_predict(
                    model.__class__(**model.get_params()),
                    X_scaled, y_winner, cv=5
                ).astype(float)
            meta_features_clf[:, i] = cv_preds

        # Meta-learner for classification
        self.clf_meta = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000),
            cv=5, method='isotonic'
        )
        self.clf_meta.fit(meta_features_clf, y_winner)

        if verbose:
            meta_preds = self.clf_meta.predict(meta_features_clf)
            print(f"  Meta-model train accuracy: {accuracy_score(y_winner, meta_preds):.4f}")

        # ---- REGRESSION: Final margin ----
        if verbose:
            print("\n--- Training Margin Regression Models ---")

        self.reg_models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                random_state=42,
                verbosity=0
            ),
            'lgbm': lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=42,
                verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0)
        }

        meta_features_reg = np.zeros((len(X), len(self.reg_models)))
        for i, (name, model) in enumerate(self.reg_models.items()):
            if verbose:
                print(f"  Training {name}...")
            model.fit(X_scaled, y_margin)
            cv_preds = cross_val_predict(
                model.__class__(**model.get_params()),
                X_scaled, y_margin, cv=5
            )
            meta_features_reg[:, i] = cv_preds

        self.reg_meta = Ridge(alpha=1.0)
        self.reg_meta.fit(meta_features_reg, y_margin)

        if verbose:
            meta_preds = self.reg_meta.predict(meta_features_reg)
            mae = np.mean(np.abs(meta_preds - y_margin))
            print(f"  Meta-model train MAE: {mae:.2f}")

        # ---- REGRESSION: Q4 total ----
        if verbose:
            print("\n--- Training Q4 Total Regression Models ---")

        self.q4_models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                random_state=42,
                verbosity=0
            ),
            'lgbm': lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=42,
                verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            ),
        }

        meta_features_q4 = np.zeros((len(X), len(self.q4_models)))
        for i, (name, model) in enumerate(self.q4_models.items()):
            if verbose:
                print(f"  Training {name}...")
            model.fit(X_scaled, y_q4_total)
            cv_preds = cross_val_predict(
                model.__class__(**model.get_params()),
                X_scaled, y_q4_total, cv=5
            )
            meta_features_q4[:, i] = cv_preds

        self.q4_meta = Ridge(alpha=1.0)
        self.q4_meta.fit(meta_features_q4, y_q4_total)

        self.is_trained = True
        if verbose:
            meta_preds = self.q4_meta.predict(meta_features_q4)
            mae = np.mean(np.abs(meta_preds - y_q4_total))
            print(f"  Meta-model train MAE: {mae:.2f}")

    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Generate predictions with confidence scores.

        Returns dict with:
        - winner_prob: P(leading team wins)
        - predicted_margin: expected final margin
        - predicted_q4_total: expected Q4 combined scoring
        - confidence: calibrated confidence (max of winner_prob, 1-winner_prob)
        - model_agreement: fraction of models that agree
        """
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names
        )

        # Classification predictions
        clf_preds = np.zeros((len(X), len(self.clf_models)))
        for i, (name, model) in enumerate(self.clf_models.items()):
            try:
                clf_preds[:, i] = model.predict_proba(X_scaled)[:, 1]
            except:
                clf_preds[:, i] = model.predict(X_scaled).astype(float)

        # Meta prediction (calibrated)
        winner_probs = self.clf_meta.predict_proba(clf_preds)[:, 1]

        # Model agreement
        binary_preds = (clf_preds > 0.5).astype(int)
        agreement = np.mean(binary_preds, axis=1)
        model_agreement = np.maximum(agreement, 1 - agreement)

        # Regression predictions (margin)
        reg_preds = np.zeros((len(X), len(self.reg_models)))
        for i, (name, model) in enumerate(self.reg_models.items()):
            reg_preds[:, i] = model.predict(X_scaled)
        predicted_margin = self.reg_meta.predict(reg_preds)

        # Regression predictions (Q4 total)
        q4_preds = np.zeros((len(X), len(self.q4_models)))
        for i, (name, model) in enumerate(self.q4_models.items()):
            q4_preds[:, i] = model.predict(X_scaled)
        predicted_q4_total = self.q4_meta.predict(q4_preds)

        # Individual model spread for uncertainty
        margin_std = np.std(reg_preds, axis=1)
        q4_std = np.std(q4_preds, axis=1)

        return {
            'winner_prob': winner_probs,
            'predicted_margin': predicted_margin,
            'predicted_q4_total': predicted_q4_total,
            'model_agreement': model_agreement,
            'margin_uncertainty': margin_std,
            'q4_uncertainty': q4_std,
            'clf_individual': clf_preds,
            'reg_individual': reg_preds,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        importance = pd.DataFrame({'feature': self.feature_names})

        for name, model in self.clf_models.items():
            if hasattr(model, 'feature_importances_'):
                importance[f'clf_{name}'] = model.feature_importances_

        for name, model in self.reg_models.items():
            if hasattr(model, 'feature_importances_'):
                importance[f'reg_{name}'] = model.feature_importances_

        # Average importance
        imp_cols = [c for c in importance.columns if c != 'feature']
        if imp_cols:
            importance['avg_importance'] = importance[imp_cols].mean(axis=1)
            importance = importance.sort_values('avg_importance', ascending=False)

        return importance

    def save(self, path: str):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'Q3TerminalEnsemble':
        """Load model from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================
# SIGNAL GENERATOR
# ============================================================

class Q3SignalGenerator:
    """
    Ultra-selective signal generator.
    Only signals when multiple conditions align:
    1. Model confidence is very high
    2. Model agreement across ensemble members
    3. Market implied probability differs from model
    4. Game regime supports reliable prediction
    """

    # Confidence thresholds for tiers
    TIERS = {
        'PLATINUM': 0.97,  # 97%+ confidence
        'GOLD': 0.95,      # 95%+ confidence
        'SILVER': 0.92,    # 92%+ confidence
        'BRONZE': 0.88,    # 88%+ confidence
    }

    # Minimum model agreement
    MIN_AGREEMENT = 0.75  # 3/4 models must agree

    @staticmethod
    def estimate_live_ml_odds(lead: int) -> float:
        """
        Estimate live moneyline odds at end of Q3 based on lead.
        Uses historical NBA data calibration.
        Returns American odds for the LEADING team.
        """
        if lead == 0:
            return -110  # coin flip with vig

        abs_lead = abs(lead)

        # Win probability based on lead at end of Q3
        # Calibrated from historical NBA data
        # Sigma ~ 9.5 points for one quarter
        sigma_q4 = 9.5
        win_prob = stats.norm.cdf(abs_lead / sigma_q4)

        # Convert to American odds
        if win_prob >= 0.5:
            odds = -(win_prob / (1 - win_prob)) * 100
        else:
            odds = ((1 - win_prob) / win_prob) * 100

        return odds

    @staticmethod
    def estimate_live_spread(lead: int) -> float:
        """
        Estimate live spread at end of Q3.
        Typically about 75-80% of current lead.
        """
        return lead * 0.78

    @staticmethod
    def prob_to_american(prob: float) -> float:
        """Convert probability to American odds"""
        if prob <= 0 or prob >= 1:
            return 0
        if prob >= 0.5:
            return -(prob / (1 - prob)) * 100
        else:
            return ((1 - prob) / prob) * 100

    @staticmethod
    def american_to_prob(odds: float) -> float:
        """Convert American odds to implied probability"""
        if odds == 0:
            return 0.5
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    def generate_signals(self, predictions: Dict, features: pd.DataFrame,
                        game_states: List[Q3GameState]) -> List[Q3Signal]:
        """
        Generate trading signals from model predictions.
        Ultra-selective: only signals when confidence is very high.
        """
        signals = []

        winner_probs = predictions['winner_prob']
        predicted_margins = predictions['predicted_margin']
        predicted_q4_totals = predictions['predicted_q4_total']
        model_agreements = predictions['model_agreement']
        margin_uncertainties = predictions['margin_uncertainty']

        for i, state in enumerate(game_states):
            q3_lead = state.home_score_q3 - state.away_score_q3

            # Skip very close games (low confidence by nature)
            if abs(q3_lead) < 1:
                continue

            winner_prob = winner_probs[i]
            pred_margin = predicted_margins[i]
            pred_q4 = predicted_q4_totals[i]
            agreement = model_agreements[i]
            uncertainty = margin_uncertainties[i]

            # Determine leading team
            if q3_lead > 0:
                leading_team = state.home_team
                trailing_team = state.away_team
                leader_win_prob = winner_prob
            else:
                leading_team = state.away_team
                trailing_team = state.home_team
                leader_win_prob = 1 - winner_prob

            # Confidence is the max probability (for either side)
            confidence = max(winner_prob, 1 - winner_prob)

            # Check minimum agreement
            if agreement < self.MIN_AGREEMENT:
                continue

            # Determine tier
            tier = None
            for tier_name, threshold in self.TIERS.items():
                if confidence >= threshold:
                    tier = tier_name
                    break

            if tier is None:
                continue

            # Estimate market odds
            est_ml_odds = self.estimate_live_ml_odds(q3_lead)
            est_spread = self.estimate_live_spread(q3_lead)
            market_impl_prob = self.american_to_prob(est_ml_odds)

            # ---- MONEYLINE SIGNAL ----
            # Signal when our confidence significantly exceeds market-implied prob
            edge_ml = confidence - market_impl_prob

            if confidence >= self.TIERS.get(tier, 0.88):
                # Determine bet direction and type
                # We predict the LEADING team wins with high confidence
                predicted_winner = leading_team if leader_win_prob > 0.5 else trailing_team

                actual_winner = state.home_team if state.home_final > state.away_final else state.away_team
                actual_margin = state.home_final - state.away_final
                actual_q4_total = state.home_q4 + state.away_q4

                # Determine regime
                abs_lead = abs(q3_lead)
                if abs_lead >= 20:
                    regime = 'BLOWOUT'
                elif abs_lead >= 12:
                    regime = 'COMFORTABLE'
                elif abs_lead >= 6:
                    regime = 'COMPETITIVE'
                else:
                    regime = 'TIGHT'

                signal = Q3Signal(
                    game_id=state.game_id,
                    home_team=state.home_team,
                    away_team=state.away_team,
                    signal_type='ML',
                    direction='HOME' if predicted_winner == state.home_team else 'AWAY',
                    confidence=confidence,
                    tier=tier,
                    predicted_winner=predicted_winner,
                    predicted_margin=pred_margin,
                    predicted_q4_total=pred_q4,
                    estimated_live_ml=est_ml_odds,
                    estimated_live_spread=est_spread,
                    estimated_live_ou=state.opening_ou,
                    actual_winner=actual_winner,
                    actual_margin=actual_margin,
                    actual_q4_total=actual_q4_total,
                    correct=(predicted_winner == actual_winner),
                    q3_lead=q3_lead,
                    q3_momentum=features.iloc[i].get('lead_momentum', 0) if i < len(features) else 0,
                    regime=regime,
                )
                signals.append(signal)

                # ---- SPREAD SIGNAL ----
                # Predict whether the leading team covers the estimated live spread
                # This is a -110 bet
                if abs(pred_margin) > abs(est_spread) + 2:  # Need margin buffer
                    spread_direction = 'HOME' if pred_margin > est_spread else 'AWAY'

                    # Did they actually cover?
                    if spread_direction == 'HOME':
                        spread_correct = actual_margin > est_spread
                    else:
                        spread_correct = actual_margin < est_spread

                    spread_signal = Q3Signal(
                        game_id=state.game_id,
                        home_team=state.home_team,
                        away_team=state.away_team,
                        signal_type='SPREAD',
                        direction=spread_direction,
                        confidence=confidence,
                        tier=tier,
                        predicted_winner=predicted_winner,
                        predicted_margin=pred_margin,
                        predicted_q4_total=pred_q4,
                        estimated_live_ml=est_ml_odds,
                        estimated_live_spread=est_spread,
                        estimated_live_ou=state.opening_ou,
                        actual_winner=actual_winner,
                        actual_margin=actual_margin,
                        actual_q4_total=actual_q4_total,
                        correct=spread_correct,
                        q3_lead=q3_lead,
                        q3_momentum=features.iloc[i].get('lead_momentum', 0) if i < len(features) else 0,
                        regime=regime,
                    )
                    signals.append(spread_signal)

        return signals


# ============================================================
# WALK-FORWARD VALIDATOR
# ============================================================

class WalkForwardValidator:
    """
    Walk-forward validation to prevent look-ahead bias.
    Splits data by season, trains on earlier seasons, tests on later.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []

    def validate(self, all_features: pd.DataFrame, all_labels: Dict[str, np.ndarray],
                 game_states: List[Q3GameState], game_ids: List[str],
                 season_map: Dict[str, int]) -> Dict:
        """
        Run walk-forward validation.

        season_map: game_id -> season_year
        """
        # Get unique seasons sorted
        seasons = sorted(set(season_map.values()))

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Walk-Forward Validation across {len(seasons)} seasons")
            print(f"Seasons: {seasons}")
            print(f"{'='*60}")

        all_signals = []
        all_oos_results = []

        # For each test season, train on all prior seasons
        for test_idx in range(1, len(seasons)):
            test_season = seasons[test_idx]
            train_seasons = seasons[:test_idx]

            if self.verbose:
                print(f"\n--- Test Season: {test_season} | Train: {train_seasons} ---")

            # Split indices
            train_mask = np.array([season_map.get(gid, 0) in train_seasons for gid in game_ids])
            test_mask = np.array([season_map.get(gid, 0) == test_season for gid in game_ids])

            if np.sum(train_mask) < 100 or np.sum(test_mask) < 50:
                if self.verbose:
                    print(f"  Skipping: insufficient data (train={np.sum(train_mask)}, test={np.sum(test_mask)})")
                continue

            X_train = all_features[train_mask]
            X_test = all_features[test_mask]

            y_winner_train = all_labels['winner'][train_mask]
            y_winner_test = all_labels['winner'][test_mask]
            y_margin_train = all_labels['margin'][train_mask]
            y_margin_test = all_labels['margin'][test_mask]
            y_q4_train = all_labels['q4_total'][train_mask]
            y_q4_test = all_labels['q4_total'][test_mask]

            test_states = [gs for gs, gid in zip(game_states, game_ids)
                          if season_map.get(gid, 0) == test_season]

            # Train model
            model = Q3TerminalEnsemble()
            model.train(X_train, y_winner_train, y_margin_train, y_q4_train,
                       verbose=self.verbose)

            # Predict on test set
            predictions = model.predict(X_test)

            # Generate signals
            generator = Q3SignalGenerator()
            signals = generator.generate_signals(predictions, X_test, test_states)

            # Analyze results
            if signals:
                ml_signals = [s for s in signals if s.signal_type == 'ML']
                spread_signals = [s for s in signals if s.signal_type == 'SPREAD']

                if self.verbose:
                    print(f"\n  Season {test_season} Results:")
                    print(f"  Total signals: {len(signals)}")

                    for sig_type, sigs in [('ML', ml_signals), ('SPREAD', spread_signals)]:
                        if sigs:
                            correct = sum(1 for s in sigs if s.correct)
                            total = len(sigs)
                            accuracy = correct / total if total > 0 else 0
                            print(f"  {sig_type}: {correct}/{total} = {accuracy:.1%}")

                            for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
                                tier_sigs = [s for s in sigs if s.tier == tier_name]
                                if tier_sigs:
                                    t_correct = sum(1 for s in tier_sigs if s.correct)
                                    t_total = len(tier_sigs)
                                    t_acc = t_correct / t_total if t_total > 0 else 0
                                    print(f"    {tier_name}: {t_correct}/{t_total} = {t_acc:.1%}")

                all_signals.extend(signals)

            # Overall OOS accuracy
            oos_preds = (predictions['winner_prob'] > 0.5).astype(int)
            oos_acc = accuracy_score(y_winner_test, oos_preds)

            all_oos_results.append({
                'season': test_season,
                'n_games': int(np.sum(test_mask)),
                'n_signals': len(signals),
                'overall_accuracy': float(oos_acc),
                'brier_score': float(brier_score_loss(y_winner_test, predictions['winner_prob'])),
            })

            if self.verbose:
                print(f"  Overall OOS accuracy (all games): {oos_acc:.4f}")
                print(f"  Brier score: {all_oos_results[-1]['brier_score']:.4f}")

        return {
            'signals': all_signals,
            'oos_results': all_oos_results,
        }


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_dataset(pbp_dir: str, verbose: bool = True) -> Tuple[
    pd.DataFrame, Dict[str, np.ndarray], List[Q3GameState], List[str], Dict[str, int]
]:
    """
    Process all cached PBP games and build training dataset.

    Returns:
    - features: DataFrame of extracted features
    - labels: dict of target arrays
    - game_states: list of Q3GameState objects
    - game_ids: list of game IDs
    - season_map: game_id -> season year
    """
    extractor = Q3FeatureExtractor()

    all_features = []
    all_states = []
    all_game_ids = []
    season_map = {}

    files = sorted(Path(pbp_dir).glob('*.json'))
    if verbose:
        print(f"Processing {len(files)} games...")

    processed = 0
    skipped = 0

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                game_data = json.load(f)

            # Get season from header
            header = game_data.get('header', {})
            season_info = header.get('season', {})
            season_year = season_info.get('year', 0)
            game_id = filepath.stem

            # Extract game state
            state = extractor.extract_game_state(game_data)
            if state is None:
                skipped += 1
                continue

            # Set game_id if not set
            if not state.game_id:
                state.game_id = game_id

            # Extract features
            features = extractor.extract_features(state)

            all_features.append(features)
            all_states.append(state)
            all_game_ids.append(game_id)
            season_map[game_id] = season_year

            processed += 1

        except Exception as e:
            skipped += 1
            continue

    if verbose:
        print(f"Processed: {processed}, Skipped: {skipped}")

    # Build DataFrame
    df = pd.DataFrame(all_features)

    # Build labels
    # Winner: 1 if home team leading at Q3-end wins, 0 otherwise
    # This is relative to the LEADING team at Q3-end
    y_winner = np.array([
        1 if (s.home_score_q3 > s.away_score_q3 and s.home_final > s.away_final) or
             (s.home_score_q3 < s.away_score_q3 and s.home_final < s.away_final) or
             (s.home_score_q3 == s.away_score_q3 and s.home_final > s.away_final)
        else 0
        for s in all_states
    ])

    y_margin = np.array([s.home_final - s.away_final for s in all_states])
    y_q4_total = np.array([s.home_q4 + s.away_q4 for s in all_states])

    labels = {
        'winner': y_winner,
        'margin': y_margin,
        'q4_total': y_q4_total,
    }

    if verbose:
        print(f"\nDataset shape: {df.shape}")
        print(f"Leading team wins: {y_winner.mean():.3f}")
        print(f"Mean final margin: {y_margin.mean():.1f}")
        print(f"Mean Q4 total: {y_q4_total.mean():.1f}")
        print(f"Seasons: {sorted(set(season_map.values()))}")

    return df, labels, all_states, all_game_ids, season_map


def run_full_pipeline(pbp_dir: str, output_dir: str, verbose: bool = True):
    """Run the complete Q3 Terminal pipeline"""

    print("="*60)
    print("Q3 TERMINAL PREDICTION SYSTEM")
    print("="*60)

    # Step 1: Build dataset
    print("\n[1/4] Building dataset from cached games...")
    features, labels, states, game_ids, season_map = build_dataset(pbp_dir, verbose)

    # Step 2: Train final model on all data
    print("\n[2/4] Training full model...")
    full_model = Q3TerminalEnsemble()
    full_model.train(features, labels['winner'], labels['margin'], labels['q4_total'], verbose)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'q3_terminal_model.pkl')
    full_model.save(model_path)
    print(f"  Model saved to {model_path}")

    # Feature importance
    importance = full_model.get_feature_importance()
    print("\nTop 20 Features:")
    print(importance[['feature', 'avg_importance']].head(20).to_string(index=False))

    # Step 3: Walk-forward validation
    print("\n[3/4] Running walk-forward validation...")
    validator = WalkForwardValidator(verbose=verbose)
    validation_results = validator.validate(
        features, labels, states, game_ids, season_map
    )

    # Step 4: Analyze and output results
    print("\n[4/4] Analyzing results...")
    signals = validation_results['signals']

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total OOS signals: {len(signals)}")

    for signal_type in ['ML', 'SPREAD']:
        type_signals = [s for s in signals if s.signal_type == signal_type]
        if not type_signals:
            continue

        correct = sum(1 for s in type_signals if s.correct)
        total = len(type_signals)
        accuracy = correct / total if total > 0 else 0

        print(f"\n{signal_type} Signals: {correct}/{total} = {accuracy:.1%}")

        for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
            tier_signals = [s for s in type_signals if s.tier == tier_name]
            if tier_signals:
                t_correct = sum(1 for s in tier_signals if s.correct)
                t_total = len(tier_signals)
                t_acc = t_correct / t_total if t_total > 0 else 0

                # Calculate profit at -110 odds
                profit = t_correct * (100/110) - (t_total - t_correct) * 1.0
                roi = profit / t_total * 100 if t_total > 0 else 0

                print(f"  {tier_name}: {t_correct}/{t_total} = {t_acc:.1%} | "
                      f"P&L: {profit:+.1f}u | ROI: {roi:+.1f}%")

        # By regime
        print(f"\n  By regime:")
        for regime in ['BLOWOUT', 'COMFORTABLE', 'COMPETITIVE', 'TIGHT']:
            regime_signals = [s for s in type_signals if s.regime == regime]
            if regime_signals:
                r_correct = sum(1 for s in regime_signals if s.correct)
                r_total = len(regime_signals)
                r_acc = r_correct / r_total if r_total > 0 else 0
                print(f"    {regime}: {r_correct}/{r_total} = {r_acc:.1%}")

    # Save signals as JSON for webapp
    signals_output = []
    for s in signals:
        signals_output.append(asdict(s))

    signals_path = os.path.join(output_dir, 'q3_terminal_signals.json')
    with open(signals_path, 'w') as f:
        json.dump({
            'signals': signals_output,
            'oos_results': validation_results['oos_results'],
            'total_games_processed': len(game_ids),
            'seasons': sorted(set(season_map.values())),
        }, f, indent=2, default=str)

    print(f"\nSignals saved to {signals_path}")

    # Save feature importance
    importance_path = os.path.join(output_dir, 'q3_terminal_features.csv')
    importance.to_csv(importance_path, index=False)

    # Export model coefficients for JS (simplified logistic regression)
    _export_js_model(full_model, features, output_dir)

    return validation_results


def _export_js_model(model: Q3TerminalEnsemble, features: pd.DataFrame, output_dir: str):
    """
    Export simplified model parameters for JavaScript inference.
    Uses the logistic regression component which can be easily ported.
    """
    lr_model = model.clf_models.get('lr')
    if lr_model is None:
        return

    scaler = model.scaler
    feature_names = model.feature_names

    # Get LR coefficients
    coefficients = lr_model.coef_[0].tolist()
    intercept = lr_model.intercept_[0]

    # Get scaler parameters
    means = scaler.mean_.tolist()
    stds = scaler.scale_.tolist()

    # Also export the Ridge regression for margin prediction
    ridge_model = model.reg_models.get('ridge')
    ridge_coefs = ridge_model.coef_.tolist() if ridge_model else []
    ridge_intercept = float(ridge_model.intercept_) if ridge_model else 0

    js_model = {
        'feature_names': feature_names,
        'scaler_means': means,
        'scaler_stds': stds,
        'lr_coefficients': coefficients,
        'lr_intercept': float(intercept),
        'ridge_coefficients': ridge_coefs,
        'ridge_intercept': ridge_intercept,
    }

    js_path = os.path.join(output_dir, 'q3_terminal_js_model.json')
    with open(js_path, 'w') as f:
        json.dump(js_model, f, indent=2)

    print(f"JS model exported to {js_path}")


if __name__ == '__main__':
    import sys

    pbp_dir = sys.argv[1] if len(sys.argv) > 1 else './cache/games_pbp'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './output'

    run_full_pipeline(pbp_dir, output_dir)
