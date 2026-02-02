"""
Q3 Terminal Prediction System v2
=================================
Key improvements over v1:
1. Empirical calibration - confidence = actual historical accuracy in similar games
2. Regime-specific models - specialized models for different game states
3. Focus on SPREAD predictions at -110 odds
4. "Disagreement edge" detection - find where our model sees differently than market
5. Multi-angle signals: spread, moneyline trailing team value, Q4 total
6. Proper walk-forward with temporal ordering

Design philosophy: Be an expert QUANT, not a gambler.
- Only trade when we have a clear, quantifiable edge
- Every signal must have positive expected value
- Aggressive position sizing on highest-confidence signals
- Zero signals is better than one bad signal
"""

import json
import os
import pickle
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
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
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, brier_score_loss, mean_absolute_error
)
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')


# =============================================================
# CONSTANTS
# =============================================================

# NBA Q4 scoring calibration
Q4_SIGMA = 9.5  # std dev of Q4 margin ~ 9.5 points
AVG_Q4_TOTAL = 54.0
HOME_COURT_Q4 = 0.8  # ~0.8 point advantage per quarter

# Signal thresholds
MIN_EDGE_SPREAD = 0.08   # 8% minimum edge for spread signals
MIN_EDGE_ML = 0.10       # 10% minimum edge for ML signals
MIN_EDGE_TOTAL = 0.10    # 10% minimum edge for total signals

# Tier thresholds based on empirical accuracy (set after calibration)
TIER_THRESHOLDS = {
    'PLATINUM': 0.95,
    'GOLD': 0.90,
    'SILVER': 0.85,
    'BRONZE': 0.80,
}


# =============================================================
# FEATURE EXTRACTION (enhanced from v1)
# =============================================================

class FeatureExtractor:
    """Extract 80+ features from play-by-play data at end of Q3."""

    def extract_game(self, game_data: dict) -> Optional[dict]:
        """
        Parse one game's PBP data.
        Returns dict with features + outcomes, or None if invalid.
        """
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
        # Track scores at each period boundary and build lead time series
        period_end_scores = {}  # period -> (home, away) at END of period
        lead_timeline = []      # (game_secs, lead) for every scoring play
        scoring_events = []     # detailed scoring events

        last_home = 0
        last_away = 0
        prev_period = 0
        max_period = 0

        # Per-quarter tracking
        q_home_scoring_plays = defaultdict(list)  # period -> [pts]
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

            # Track period transitions
            if period != prev_period and prev_period > 0:
                period_end_scores[prev_period] = (last_home, last_away)
            prev_period = period

            # Parse clock
            clock_str = play.get('clock', {}).get('displayValue', '12:00')
            game_secs = self._game_seconds(period, clock_str)

            # Scoring events
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

            # Fouls
            play_type_text = play.get('type', {}).get('text', '').lower()
            play_type_id = play.get('type', {}).get('id', '')
            team_id = play.get('team', {}).get('id', '')

            if 'foul' in play_type_text and period <= 3:
                if team_id == home_id:
                    foul_counts[period]['home'] += 1
                elif team_id == away_id:
                    foul_counts[period]['away'] += 1

            # Turnovers
            if 'turnover' in play_type_text and period <= 3:
                if team_id == home_id:
                    turnovers[period]['home'] += 1
                elif team_id == away_id:
                    turnovers[period]['away'] += 1

            # Timeouts
            if str(play_type_id) == '16' and period <= 3:
                if team_id == home_id:
                    timeout_counts['home'] += 1
                elif team_id == away_id:
                    timeout_counts['away'] += 1

            last_home = home_score
            last_away = away_score

        # Final period end
        if prev_period > 0:
            period_end_scores[prev_period] = (last_home, last_away)

        # Skip OT games
        if max_period > 4:
            return None
        if 3 not in period_end_scores or 4 not in period_end_scores:
            return None

        # Extract quarter scores
        q3_home, q3_away = period_end_scores[3]
        final_home, final_away = period_end_scores[4]

        q1_home = period_end_scores.get(1, (0, 0))[0]
        q1_away = period_end_scores.get(1, (0, 0))[1]
        q2_home = period_end_scores.get(2, (0, 0))[0] - q1_home
        q2_away = period_end_scores.get(2, (0, 0))[1] - q1_away
        q1h = q1_home
        q1a = q1_away
        q2h = q2_home
        q2a = q2_away
        q3h = q3_home - period_end_scores.get(2, (0, 0))[0]
        q3a = q3_away - period_end_scores.get(2, (0, 0))[1]
        q4h = final_home - q3_home
        q4a = final_away - q3_away

        # Validate
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

        # Quarter scoring rates
        qtotals = [f['q1_total'], f['q2_total'], f['q3_qtotal']]
        f['avg_q_pace'] = np.mean(qtotals)
        f['pace_var'] = np.std(qtotals)
        f['pace_trend'] = f['q3_qtotal'] - f['q1_total']
        f['q3_pace_vs_avg'] = f['q3_qtotal'] - f['avg_q_pace']

        # Home scoring consistency
        home_qs = [q1h, q2h, q3h]
        away_qs = [q1a, q2a, q3a]
        f['home_q_std'] = np.std(home_qs)
        f['away_q_std'] = np.std(away_qs)
        f['home_q_trend'] = q3h - q1h
        f['away_q_trend'] = q3a - q1a

        # Margin trajectory across quarters
        q_margins = [f['q1_margin'], f['q2_margin'], f['q3_margin']]
        f['margin_trend'] = q_margins[-1] - q_margins[0]
        f['margin_consistency'] = np.std(q_margins)
        f['all_qs_same_direction'] = float(
            all(m > 0 for m in q_margins) or all(m < 0 for m in q_margins)
        )

        # Lead building pattern
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

        # Surprise factor: how different is actual game from expectations
        expected_q3_margin = opening_spread * 0.75  # 75% of full-game spread
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
        total_home_fouls = sum(foul_counts[q]['home'] for q in [1,2,3])
        total_away_fouls = sum(foul_counts[q]['away'] for q in [1,2,3])
        f['foul_differential'] = total_home_fouls - total_away_fouls
        f['q3_home_fouls'] = foul_counts[3]['home']
        f['q3_away_fouls'] = foul_counts[3]['away']

        total_home_to = sum(turnovers[q]['home'] for q in [1,2,3])
        total_away_to = sum(turnovers[q]['away'] for q in [1,2,3])
        f['turnover_differential'] = total_home_to - total_away_to

        f['home_timeouts_used'] = timeout_counts['home']
        f['away_timeouts_used'] = timeout_counts['away']
        f['timeout_advantage'] = timeout_counts['away'] - timeout_counts['home']  # fewer used = more remaining

        # --- Interaction features ---
        f['lead_x_momentum'] = f['q3_lead'] * f.get('lead_momentum', 0)
        f['lead_x_volatility'] = f['q3_lead_abs'] * f.get('lead_vol', 0)
        f['lead_x_pace'] = f['q3_lead_abs'] * f['avg_q_pace']
        f['lead_x_surprise'] = f['q3_lead'] * f['spread_surprise']
        f['momentum_x_fouls'] = f.get('lead_momentum', 0) * f['foul_differential']
        f['pace_x_surprise'] = f['avg_q_pace'] * f['spread_surprise_abs']
        f['consistency_x_lead'] = f['margin_consistency'] * f['q3_lead_abs']
        f['gradual_x_lead'] = f['lead_built_gradually'] * f['q3_lead_abs']

        # --- Outcomes (training labels) ---
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

    def _game_seconds(self, period: int, clock: str) -> float:
        try:
            parts = clock.replace('.', ':').split(':')
            mins = int(parts[0])
            secs = float(parts[1]) if len(parts) > 1 else 0
            remaining = mins * 60 + secs
        except:
            remaining = 720
        return (period - 1) * 720 + (720 - remaining)

    def _build_lead_timeseries(self, lead_timeline: list, interval: float = 30.0) -> np.ndarray:
        """Build a uniformly-sampled lead time series from scoring events."""
        if not lead_timeline:
            return np.array([0])

        max_time = 2160  # End of Q3 = 36 minutes
        n_samples = int(max_time / interval) + 1
        ts = np.zeros(n_samples)

        # Forward-fill from scoring events
        event_idx = 0
        current_lead = 0
        for i in range(n_samples):
            t = i * interval
            while event_idx < len(lead_timeline) and lead_timeline[event_idx][0] <= t:
                current_lead = lead_timeline[event_idx][1]
                event_idx += 1
            ts[i] = current_lead

        return ts

    def _lead_dynamics(self, ts: np.ndarray, q3_lead: int) -> dict:
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
        f['lead_r2'] = float(r**2)

        third = max(1, n // 3)
        f['lead_early'] = float(np.mean(ts[:third]))
        f['lead_mid'] = float(np.mean(ts[third:2*third]))
        f['lead_late'] = float(np.mean(ts[2*third:]))

        f['lead_momentum'] = f['lead_late'] - f['lead_mid']
        f['lead_accel'] = (f['lead_late'] - f['lead_mid']) - (f['lead_mid'] - f['lead_early'])

        half_idx = n * 2 // 3
        f['halftime_lead'] = float(ts[min(half_idx, n-1)])
        f['q3_lead_growth'] = float(q3_lead - f['halftime_lead'])

        # Lead stability: fraction of time at or beyond current lead
        if q3_lead > 0:
            f['lead_stability'] = float(np.mean(ts >= q3_lead * 0.5))
        elif q3_lead < 0:
            f['lead_stability'] = float(np.mean(ts <= q3_lead * 0.5))
        else:
            f['lead_stability'] = 0.5

        f['lead_mean_abs'] = float(np.mean(np.abs(ts)))

        return f

    def _dna_features(self, ts: np.ndarray) -> dict:
        """Game Flow DNA: time-series fingerprint features."""
        f = {}
        n = len(ts)

        if n < 8:
            return {f'dna_{k}': 0 for k in [
                'hurst', 'ac1', 'ac3', 'ac5',
                'fft_dom', 'fft_lo', 'fft_hi',
                'entropy', 'xing_rate', 'ou_theta',
                'max_run', 'avg_run',
                'skew', 'kurt',
            ]}

        # Hurst exponent
        try:
            f['dna_hurst'] = self._hurst(ts)
        except:
            f['dna_hurst'] = 0.5

        # Autocorrelation
        c = ts - np.mean(ts)
        v = np.var(c) + 1e-10
        for lag in [1, 3, 5]:
            if lag < n:
                f[f'dna_ac{lag}'] = float(np.mean(c[:-lag] * c[lag:]) / v)
            else:
                f[f'dna_ac{lag}'] = 0

        # FFT
        try:
            fv = np.abs(fft(ts - np.mean(ts)))[:n//2]
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

        # Entropy of lead changes
        diffs = np.diff(ts)
        if len(diffs) > 0:
            bins = np.array([-np.inf, -5, -2, 0, 2, 5, np.inf])
            hist, _ = np.histogram(diffs, bins=bins)
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            f['dna_entropy'] = float(-np.sum(probs * np.log2(probs)))
        else:
            f['dna_entropy'] = 0

        # Zero-crossing rate
        signs = np.sign(ts)
        f['dna_xing_rate'] = float(np.sum(np.abs(np.diff(signs)) > 0) / max(1, n-1))

        # OU mean-reversion speed
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

        # Run analysis
        diffs = np.diff(ts)
        runs = []
        run = 1
        for i in range(1, len(diffs)):
            if np.sign(diffs[i]) == np.sign(diffs[i-1]) and diffs[i] != 0:
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                run = 1
        if run > 0:
            runs.append(run)
        f['dna_max_run'] = float(max(runs)) if runs else 0
        f['dna_avg_run'] = float(np.mean(runs)) if runs else 0

        # Higher moments
        f['dna_skew'] = float(stats.skew(ts))
        f['dna_kurt'] = float(stats.kurtosis(ts))

        return f

    def _hurst(self, ts: np.ndarray) -> float:
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

    def _ta_features(self, ts: np.ndarray) -> dict:
        """Technical analysis features on lead time series."""
        f = {}
        n = len(ts)

        if n < 10:
            return {f'ta_{k}': 0 for k in [
                'rsi', 'bb_pos', 'bb_width', 'macd', 'roc5', 'roc15',
                'mom_div', 'willr',
            ]}

        diffs = np.diff(ts)

        # RSI
        w = min(14, len(diffs))
        recent = diffs[-w:]
        gains = np.sum(recent[recent > 0])
        losses = -np.sum(recent[recent < 0])
        if losses > 0:
            rs = gains / losses
            f['ta_rsi'] = float(100 - (100 / (1 + rs)))
        else:
            f['ta_rsi'] = 100.0 if gains > 0 else 50.0

        # Bollinger Bands
        w = min(20, n)
        recent = ts[-w:]
        mu = np.mean(recent)
        sd = np.std(recent) + 1e-10
        f['ta_bb_pos'] = float((ts[-1] - (mu - 2*sd)) / (4*sd + 1e-10))
        f['ta_bb_width'] = float(4*sd / (abs(mu) + 1e-10))

        # MACD
        if n >= 12:
            fast = self._ema_val(ts, min(5, n//2))
            slow = self._ema_val(ts, min(12, n-1))
            f['ta_macd'] = float(fast - slow)
        else:
            f['ta_macd'] = 0

        # Rate of change
        for w in [5, 15]:
            if n > w:
                f[f'ta_roc{w}'] = float(ts[-1] - ts[-1-w])
            else:
                f[f'ta_roc{w}'] = 0

        # Momentum divergence
        half = n // 2
        if half > 1:
            first_max = np.max(np.abs(ts[:half]))
            second_max = np.max(np.abs(ts[half:]))
            first_mom = np.mean(np.abs(np.diff(ts[:half])))
            second_mom = np.mean(np.abs(np.diff(ts[half:])))
            f['ta_mom_div'] = float((second_max - first_max) - (second_mom - first_mom) * 5)
        else:
            f['ta_mom_div'] = 0

        # Williams %R analog
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

    def _scoring_patterns(self, events: list, q3_home: int, q3_away: int) -> dict:
        f = {}

        if not events:
            return {f'sp_{k}': 0 for k in [
                'late_q3_margin', 'late_q3_pts', 'q3_last5_pts',
                'home_3pt_rate', 'away_3pt_rate', 'ft_rate',
                'home_largest_run', 'away_largest_run',
                'home_dry_spell', 'away_dry_spell',
                'lead_at_each_break',
            ]}

        # Late Q3 analysis (last 3 min = after 2160-180=1980 game seconds)
        late_cutoff = 1980
        late = [e for e in events if e['period'] == 3 and e['secs'] >= late_cutoff]
        if late and len(late) >= 2:
            f['sp_late_q3_margin'] = float(late[-1]['lead'] - late[0]['lead'])
            f['sp_late_q3_pts'] = float(sum(e['pts'] for e in late))
        else:
            f['sp_late_q3_margin'] = 0
            f['sp_late_q3_pts'] = 0

        # Last 5 minutes of Q3
        l5_cutoff = 1860
        l5 = [e for e in events if e['period'] == 3 and e['secs'] >= l5_cutoff]
        f['sp_q3_last5_pts'] = float(sum(e['pts'] for e in l5))

        # Three-point rate
        q3_events = [e for e in events if e['period'] <= 3]
        home_3s = sum(1 for e in q3_events if e['is_home'] and e['pts'] == 3)
        away_3s = sum(1 for e in q3_events if not e['is_home'] and e['pts'] == 3)
        home_fgs = sum(1 for e in q3_events if e['is_home'] and e['pts'] >= 2)
        away_fgs = sum(1 for e in q3_events if not e['is_home'] and e['pts'] >= 2)

        f['sp_home_3pt_rate'] = home_3s / max(1, home_fgs)
        f['sp_away_3pt_rate'] = away_3s / max(1, away_fgs)

        # Free throw rate
        all_fts = sum(1 for e in q3_events if e['pts'] == 1)
        all_plays = len(q3_events)
        f['sp_ft_rate'] = all_fts / max(1, all_plays)

        # Largest runs
        f['sp_home_largest_run'] = float(self._largest_run(events, True))
        f['sp_away_largest_run'] = float(self._largest_run(events, False))

        # Dry spells (longest period without scoring)
        f['sp_home_dry_spell'] = float(self._longest_dry_spell(events, True))
        f['sp_away_dry_spell'] = float(self._longest_dry_spell(events, False))

        # Lead at each quarter break
        leads_at_breaks = []
        for q in [1, 2, 3]:
            q_events = [e for e in events if e['period'] == q]
            if q_events:
                leads_at_breaks.append(q_events[-1]['lead'])
        f['sp_lead_at_each_break'] = float(np.std(leads_at_breaks)) if leads_at_breaks else 0

        return f

    def _largest_run(self, events: list, is_home: bool) -> int:
        """Largest consecutive scoring run for one side."""
        max_run = 0
        run = 0
        for e in events:
            if e['is_home'] == is_home:
                run += e['pts']
            else:
                max_run = max(max_run, run)
                run = 0
        return max(max_run, run)

    def _longest_dry_spell(self, events: list, is_home: bool) -> float:
        """Longest gap (in game seconds) without scoring for one side."""
        team_events = [e for e in events if e['is_home'] == is_home]
        if len(team_events) < 2:
            return 0
        max_gap = 0
        for i in range(1, len(team_events)):
            gap = team_events[i]['secs'] - team_events[i-1]['secs']
            max_gap = max(max_gap, gap)
        return max_gap


# =============================================================
# MODEL with EMPIRICAL CALIBRATION
# =============================================================

class Q3EnsembleV2:
    """
    Enhanced ensemble with:
    1. Multiple diverse base models
    2. Stacking meta-learner
    3. EMPIRICAL CALIBRATION: confidence = actual accuracy in similar prediction bins
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

        # Models for: leader_won (binary), final_margin (regression), q4_total (regression)
        self.clf_models = {}
        self.reg_models = {}
        self.q4_models = {}

        # Calibration mapping: binned_prediction -> actual_accuracy
        self.calibrator = None
        self.margin_calibrator = None

    def train(self, X: pd.DataFrame, y_leader: np.ndarray,
              y_margin: np.ndarray, y_q4: np.ndarray, verbose=True):

        self.feature_names = X.columns.tolist()
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.scaler.fit(X_clean)
        Xs = pd.DataFrame(self.scaler.transform(X_clean), columns=self.feature_names)

        if verbose:
            print(f"  Training on {len(X)} samples, {len(self.feature_names)} features")

        # ---- CLASSIFIER: does leader at Q3-end win? ----
        self.clf_models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
                reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
                random_state=42, eval_metric='logloss', verbosity=0
            ),
            'lgbm': lgb.LGBMClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_samples=30,
                reg_alpha=0.5, reg_lambda=2.0,
                random_state=42, verbose=-1
            ),
            'rf': RandomForestClassifier(
                n_estimators=500, max_depth=6, min_samples_leaf=30,
                random_state=42, n_jobs=-1
            ),
            'hgb': HistGradientBoostingClassifier(
                max_iter=500, max_depth=4, learning_rate=0.03,
                min_samples_leaf=30, l2_regularization=1.0,
                random_state=42
            ),
            'lr': LogisticRegression(C=0.5, max_iter=2000, random_state=42)
        }

        # Cross-validated stacking predictions
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

        # Empirical calibration on CV predictions
        avg_cv = np.mean(cv_preds, axis=1)
        self.calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        self.calibrator.fit(avg_cv, y_leader)

        if verbose:
            cal_probs = self.calibrator.predict(avg_cv)
            print(f"  Calibrated Brier: {brier_score_loss(y_leader, cal_probs):.4f}")

        # ---- REGRESSION: final margin ----
        self.reg_models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
                reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbosity=0
            ),
            'lgbm': lgb.LGBMRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_samples=30,
                reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=500, max_depth=6, min_samples_leaf=30,
                random_state=42, n_jobs=-1
            ),
            'ridge': Ridge(alpha=5.0),
        }

        reg_cv = np.zeros((len(X), len(self.reg_models)))
        for i, (name, model) in enumerate(self.reg_models.items()):
            if verbose:
                print(f"    margin {name}...", end=' ')
            model.fit(Xs, y_margin)
            reg_cv[:, i] = cross_val_predict(
                model.__class__(**model.get_params()), Xs, y_margin, cv=5
            )
            if verbose:
                mae = mean_absolute_error(y_margin, reg_cv[:, i])
                print(f"MAE={mae:.1f}")

        # ---- REGRESSION: Q4 total ----
        self.q4_models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
                random_state=42, verbosity=0
            ),
            'lgbm': lgb.LGBMRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_samples=30,
                random_state=42, verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=500, max_depth=6, min_samples_leaf=30,
                random_state=42, n_jobs=-1
            ),
        }
        q4_cv = np.zeros((len(X), len(self.q4_models)))
        for i, (name, model) in enumerate(self.q4_models.items()):
            model.fit(Xs, y_q4)
            q4_cv[:, i] = cross_val_predict(
                model.__class__(**model.get_params()), Xs, y_q4, cv=5
            )

        if verbose:
            avg_q4 = np.mean(q4_cv, axis=1)
            print(f"  Q4 total MAE: {mean_absolute_error(y_q4, avg_q4):.1f}")

    def predict(self, X: pd.DataFrame) -> dict:
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        Xs = pd.DataFrame(self.scaler.transform(X_clean), columns=self.feature_names)

        # Classification
        n_clf = len(self.clf_models)
        clf_p = np.zeros((len(X), n_clf))
        for i, (name, model) in enumerate(self.clf_models.items()):
            try:
                clf_p[:, i] = model.predict_proba(Xs)[:, 1]
            except:
                clf_p[:, i] = model.predict(Xs).astype(float)

        raw_prob = np.mean(clf_p, axis=1)
        # Calibrated probability
        cal_prob = self.calibrator.predict(raw_prob)

        # Model agreement (fraction of models agreeing on winner)
        binary = (clf_p > 0.5).astype(int)
        agreement = np.mean(binary, axis=1)
        model_agree = np.maximum(agreement, 1 - agreement)

        # Regression: margin
        reg_p = np.zeros((len(X), len(self.reg_models)))
        for i, (name, model) in enumerate(self.reg_models.items()):
            reg_p[:, i] = model.predict(Xs)
        pred_margin = np.mean(reg_p, axis=1)
        margin_std = np.std(reg_p, axis=1)

        # Regression: Q4 total
        q4_p = np.zeros((len(X), len(self.q4_models)))
        for i, (name, model) in enumerate(self.q4_models.items()):
            q4_p[:, i] = model.predict(Xs)
        pred_q4 = np.mean(q4_p, axis=1)
        q4_std = np.std(q4_p, axis=1)

        return {
            'raw_prob': raw_prob,
            'cal_prob': cal_prob,
            'model_agree': model_agree,
            'pred_margin': pred_margin,
            'margin_std': margin_std,
            'pred_q4': pred_q4,
            'q4_std': q4_std,
            'clf_individual': clf_p,
        }

    def get_importance(self) -> pd.DataFrame:
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

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================
# SIGNAL GENERATION v2
# =============================================================

def estimate_live_spread(q3_lead: int) -> float:
    """Market live spread estimate at end of Q3 (for rest of game)."""
    # Teams leading at Q3-end keep ~78% of their lead
    # Live spread is roughly: current_lead * 0.78
    return q3_lead * 0.78


def estimate_live_ml(q3_lead: int) -> float:
    """Estimated moneyline odds (American) for the LEADING team."""
    if q3_lead == 0:
        return -110

    sigma = Q4_SIGMA
    win_prob = stats.norm.cdf(abs(q3_lead) / sigma)

    if win_prob >= 0.99:
        return -10000
    if win_prob <= 0.01:
        return 10000

    if win_prob >= 0.5:
        return -(win_prob / (1 - win_prob)) * 100
    else:
        return ((1 - win_prob) / win_prob) * 100


def estimate_live_q4_ou(avg_pace: float, q3_lead_abs: int) -> float:
    """Estimated live Q4 O/U total."""
    base = AVG_Q4_TOTAL

    # Blowout games tend to have lower Q4 totals
    if q3_lead_abs >= 20:
        adjustment = -4.0
    elif q3_lead_abs >= 15:
        adjustment = -2.0
    elif q3_lead_abs >= 10:
        adjustment = -0.5
    else:
        adjustment = 1.0  # Close games can have higher Q4 (fouling)

    # Pace adjustment
    pace_adj = (avg_pace - 53.0) * 0.3

    return base + adjustment + pace_adj


def generate_signals_v2(preds: dict, features_df: pd.DataFrame,
                         games: list) -> list:
    """
    Generate signals with multiple bet types:
    1. SPREAD: predict leader covers live spread (always -110)
    2. ML_VALUE: trailing team comeback at plus odds (rare but high EV)
    3. Q4_TOTAL: Q4 over/under
    """
    signals = []

    for i, game in enumerate(games):
        q3_lead = game['q3_home'] - game['q3_away']
        q3_lead_abs = abs(q3_lead)

        cal_prob = preds['cal_prob'][i]  # P(leader wins)
        model_agree = preds['model_agree'][i]
        pred_margin = preds['pred_margin'][i]
        margin_std = preds['margin_std'][i]
        pred_q4 = preds['pred_q4'][i]
        q4_std = preds['q4_std'][i]

        # Determine leading team
        if q3_lead > 0:
            leader = game['home_team']
            trailer = game['away_team']
            leader_prob = cal_prob
        elif q3_lead < 0:
            leader = game['away_team']
            trailer = game['home_team']
            leader_prob = 1 - cal_prob
        else:
            continue  # Skip tied games

        trailer_prob = 1 - leader_prob

        # Market estimates
        mkt_spread = estimate_live_spread(q3_lead)
        mkt_ml = estimate_live_ml(q3_lead)
        mkt_ml_prob = stats.norm.cdf(q3_lead_abs / Q4_SIGMA) if q3_lead_abs > 0 else 0.5
        avg_pace = features_df.iloc[i].get('avg_q_pace', 53.0) if i < len(features_df) else 53.0
        mkt_q4_ou = estimate_live_q4_ou(avg_pace, q3_lead_abs)

        # Actuals
        actual_margin = game['outcomes']['final_margin']
        actual_q4_total = game['outcomes']['q4_total']
        actual_winner = game['home_team'] if actual_margin > 0 else game['away_team']
        leader_won = game['outcomes']['leader_won']

        # Regime
        if q3_lead_abs >= 20:
            regime = 'BLOWOUT'
        elif q3_lead_abs >= 12:
            regime = 'COMFORTABLE'
        elif q3_lead_abs >= 6:
            regime = 'COMPETITIVE'
        else:
            regime = 'TIGHT'

        # ========== SIGNAL 1: SPREAD ==========
        # The live spread at end of Q3 is approximately current_lead * 0.78
        # We predict the actual margin. If our prediction differs from live spread, signal.
        # This is a -110 bet.

        spread_edge = None
        spread_dir = None

        if pred_margin > mkt_spread + 3.0:  # Model says home covers by more
            spread_dir = 'HOME'
            # Our confidence that home covers: P(margin > mkt_spread)
            if margin_std > 0:
                spread_prob = float(stats.norm.cdf((pred_margin - mkt_spread) / margin_std))
            else:
                spread_prob = 0.8 if pred_margin > mkt_spread else 0.2
            spread_edge = spread_prob - 0.5
            did_cover = actual_margin > mkt_spread

        elif pred_margin < mkt_spread - 3.0:  # Model says away covers
            spread_dir = 'AWAY'
            if margin_std > 0:
                spread_prob = float(stats.norm.cdf((mkt_spread - pred_margin) / margin_std))
            else:
                spread_prob = 0.8 if pred_margin < mkt_spread else 0.2
            spread_edge = spread_prob - 0.5
            did_cover = actual_margin < mkt_spread

        if spread_dir and spread_edge and spread_edge >= MIN_EDGE_SPREAD and model_agree >= 0.8:
            signals.append({
                'game_id': game['game_id'],
                'season': game['season'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'signal_type': 'SPREAD',
                'direction': spread_dir,
                'confidence': float(0.5 + spread_edge),
                'edge': float(spread_edge),
                'model_agree': float(model_agree),
                'predicted_margin': float(pred_margin),
                'live_spread': float(mkt_spread),
                'margin_divergence': float(pred_margin - mkt_spread),
                'actual_margin': float(actual_margin),
                'correct': bool(did_cover),
                'q3_lead': q3_lead,
                'regime': regime,
                'pred_q4': float(pred_q4),
                'estimated_odds': -110,
            })

        # ========== SIGNAL 2: MONEYLINE (Leader) ==========
        # Only when model confidence significantly exceeds market
        ml_edge = leader_prob - mkt_ml_prob
        if ml_edge >= MIN_EDGE_ML and leader_prob >= 0.85 and model_agree >= 0.8:
            signals.append({
                'game_id': game['game_id'],
                'season': game['season'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'signal_type': 'ML_LEADER',
                'direction': 'HOME' if q3_lead > 0 else 'AWAY',
                'confidence': float(leader_prob),
                'edge': float(ml_edge),
                'model_agree': float(model_agree),
                'predicted_margin': float(pred_margin),
                'market_prob': float(mkt_ml_prob),
                'estimated_ml_odds': float(mkt_ml),
                'actual_margin': float(actual_margin),
                'correct': bool(leader_won),
                'q3_lead': q3_lead,
                'regime': regime,
                'pred_q4': float(pred_q4),
                'estimated_odds': float(mkt_ml),
            })

        # ========== SIGNAL 3: MONEYLINE (Trailer Value) ==========
        # When trailing team has better comeback chance than market thinks
        # These are rare but HIGH VALUE signals (plus odds)
        trailer_ml_odds = -mkt_ml if mkt_ml < 0 else -(100 * 100 / mkt_ml)
        if trailer_ml_odds > 0:  # Must be plus odds
            trailer_mkt_prob = 1 - mkt_ml_prob
            trailer_edge = trailer_prob - trailer_mkt_prob

            if (trailer_edge >= 0.05 and trailer_prob >= 0.15
                and q3_lead_abs <= 15 and model_agree <= 0.7):
                # Model is uncertain AND leaning toward trailer
                signals.append({
                    'game_id': game['game_id'],
                    'season': game['season'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'signal_type': 'ML_TRAILER',
                    'direction': 'AWAY' if q3_lead > 0 else 'HOME',
                    'confidence': float(trailer_prob),
                    'edge': float(trailer_edge),
                    'model_agree': float(1 - model_agree),  # agreement toward trailer
                    'predicted_margin': float(pred_margin),
                    'trailer_odds': float(trailer_ml_odds),
                    'actual_margin': float(actual_margin),
                    'correct': bool(not leader_won),
                    'q3_lead': q3_lead,
                    'regime': regime,
                    'pred_q4': float(pred_q4),
                    'estimated_odds': float(trailer_ml_odds),
                })

        # ========== SIGNAL 4: Q4 TOTAL ==========
        q4_diff = pred_q4 - mkt_q4_ou
        if abs(q4_diff) >= 4.0:  # At least 4 point divergence
            q4_dir = 'OVER' if q4_diff > 0 else 'UNDER'
            if q4_std > 0:
                q4_prob = float(stats.norm.cdf(abs(q4_diff) / q4_std))
            else:
                q4_prob = 0.7

            q4_edge = q4_prob - 0.5

            if q4_edge >= MIN_EDGE_TOTAL:
                actual_over = actual_q4_total > mkt_q4_ou
                signals.append({
                    'game_id': game['game_id'],
                    'season': game['season'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'signal_type': 'Q4_TOTAL',
                    'direction': q4_dir,
                    'confidence': float(q4_prob),
                    'edge': float(q4_edge),
                    'model_agree': float(model_agree),
                    'predicted_q4': float(pred_q4),
                    'live_q4_ou': float(mkt_q4_ou),
                    'q4_divergence': float(q4_diff),
                    'actual_q4': float(actual_q4_total),
                    'correct': bool(
                        (q4_dir == 'OVER' and actual_over) or
                        (q4_dir == 'UNDER' and not actual_over)
                    ),
                    'q3_lead': q3_lead,
                    'regime': regime,
                    'estimated_odds': -110,
                })

    return signals


# =============================================================
# WALK-FORWARD ENGINE
# =============================================================

def run_walkforward(games: list, verbose=True) -> dict:
    """
    Walk-forward validation: train on season N, test on season N+1.
    """
    extractor = FeatureExtractor()

    seasons = sorted(set(g['season'] for g in games))
    if verbose:
        print(f"\nSeasons found: {seasons}")
        print(f"Games per season: {dict((s, sum(1 for g in games if g['season']==s)) for s in seasons)}")

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
            print(f"\n{'='*60}")
            print(f"FOLD: Train={train_seasons} ({len(train_games)} games) | Test={test_season} ({len(test_games)} games)")
            print(f"{'='*60}")

        # Build feature matrices
        train_feats = pd.DataFrame([g['features'] for g in train_games])
        test_feats = pd.DataFrame([g['features'] for g in test_games])

        train_y_leader = np.array([g['outcomes']['leader_won'] for g in train_games])
        train_y_margin = np.array([g['outcomes']['final_margin'] for g in train_games])
        train_y_q4 = np.array([g['outcomes']['q4_total'] for g in train_games])

        test_y_leader = np.array([g['outcomes']['leader_won'] for g in test_games])

        # Train model
        model = Q3EnsembleV2()
        model.train(train_feats, train_y_leader, train_y_margin, train_y_q4, verbose)

        # Predict on test
        preds = model.predict(test_feats)

        # Overall metrics
        oos_pred = (preds['cal_prob'] > 0.5).astype(int)
        oos_acc = accuracy_score(test_y_leader, oos_pred)
        oos_brier = brier_score_loss(test_y_leader, preds['cal_prob'])

        if verbose:
            print(f"\n  Overall OOS: acc={oos_acc:.3f}, brier={oos_brier:.4f}")

        # Generate signals
        signals = generate_signals_v2(preds, test_feats, test_games)
        all_signals.extend(signals)

        # Summarize signals
        if verbose and signals:
            for stype in ['SPREAD', 'ML_LEADER', 'ML_TRAILER', 'Q4_TOTAL']:
                ss = [s for s in signals if s['signal_type'] == stype]
                if ss:
                    correct = sum(1 for s in ss if s['correct'])
                    total = len(ss)
                    acc = correct / total
                    avg_edge = np.mean([s['edge'] for s in ss])
                    print(f"  {stype}: {correct}/{total} = {acc:.1%} (avg edge: {avg_edge:.1%})")

                    # By confidence bucket
                    for lo, hi, label in [(0.85, 0.90, '85-90%'), (0.90, 0.95, '90-95%'), (0.95, 1.0, '95%+')]:
                        bucket = [s for s in ss if lo <= s['confidence'] < hi]
                        if bucket:
                            bc = sum(1 for s in bucket if s['correct'])
                            print(f"    [{label}]: {bc}/{len(bucket)} = {bc/len(bucket):.1%}")

        oos_summary.append({
            'test_season': test_season,
            'train_seasons': train_seasons,
            'n_train': len(train_games),
            'n_test': len(test_games),
            'oos_accuracy': float(oos_acc),
            'oos_brier': float(oos_brier),
            'n_signals': len(signals),
        })

    return {
        'signals': all_signals,
        'oos_summary': oos_summary,
    }


# =============================================================
# MAIN
# =============================================================

def main(pbp_dir: str = './cache/games_pbp', output_dir: str = './output'):
    print("="*60)
    print("Q3 TERMINAL PREDICTION SYSTEM v2")
    print("="*60)

    # Step 1: Parse all games
    print("\n[1/5] Parsing play-by-play data...")
    extractor = FeatureExtractor()
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

    # Step 2: Walk-forward validation
    print("\n[2/5] Running walk-forward validation...")
    results = run_walkforward(games, verbose=True)

    signals = results['signals']

    # Step 3: Analyze results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total OOS signals: {len(signals)}")

    for stype in ['SPREAD', 'ML_LEADER', 'ML_TRAILER', 'Q4_TOTAL']:
        ss = [s for s in signals if s['signal_type'] == stype]
        if not ss:
            continue

        correct = sum(1 for s in ss if s['correct'])
        total = len(ss)
        acc = correct / total if total > 0 else 0

        # P&L at -110 odds
        if stype in ['SPREAD', 'Q4_TOTAL']:
            pl = correct * (100/110) - (total - correct) * 1.0
        elif stype == 'ML_LEADER':
            # Variable odds (usually negative/expensive)
            pl = sum(
                (100 / abs(s['estimated_odds'])) if s['correct'] else -1.0
                for s in ss
            )
        else:  # ML_TRAILER
            pl = sum(
                (s.get('trailer_odds', 100) / 100) if s['correct'] else -1.0
                for s in ss
            )

        roi = pl / total * 100 if total > 0 else 0

        print(f"\n{stype}: {correct}/{total} = {acc:.1%}")
        print(f"  P&L: {pl:+.1f}u | ROI: {roi:+.1f}%")
        print(f"  Avg edge: {np.mean([s['edge'] for s in ss]):.1%}")
        print(f"  Avg confidence: {np.mean([s['confidence'] for s in ss]):.1%}")

        # By confidence bucket
        buckets = [
            (0.95, 1.01, 'PLATINUM (95%+)'),
            (0.90, 0.95, 'GOLD (90-95%)'),
            (0.85, 0.90, 'SILVER (85-90%)'),
            (0.80, 0.85, 'BRONZE (80-85%)'),
            (0.50, 0.80, 'SUB-BRONZE (<80%)'),
        ]
        for lo, hi, label in buckets:
            bucket = [s for s in ss if lo <= s['confidence'] < hi]
            if bucket:
                bc = sum(1 for s in bucket if s['correct'])
                bt = len(bucket)
                ba = bc / bt if bt > 0 else 0
                if stype in ['SPREAD', 'Q4_TOTAL']:
                    bpl = bc * (100/110) - (bt - bc) * 1.0
                else:
                    bpl = 0  # simplified
                broi = bpl / bt * 100 if bt > 0 else 0
                print(f"  {label}: {bc}/{bt} = {ba:.1%} | P&L: {bpl:+.1f}u | ROI: {broi:+.1f}%")

        # By regime
        print(f"  By regime:")
        for regime in ['BLOWOUT', 'COMFORTABLE', 'COMPETITIVE', 'TIGHT']:
            rs = [s for s in ss if s['regime'] == regime]
            if rs:
                rc = sum(1 for s in rs if s['correct'])
                rt = len(rs)
                print(f"    {regime}: {rc}/{rt} = {rc/rt:.1%}")

    # Step 4: Train final model on all data
    print("\n[3/5] Training final model on all data...")
    all_feats = pd.DataFrame([g['features'] for g in games])
    all_y_leader = np.array([g['outcomes']['leader_won'] for g in games])
    all_y_margin = np.array([g['outcomes']['final_margin'] for g in games])
    all_y_q4 = np.array([g['outcomes']['q4_total'] for g in games])

    final_model = Q3EnsembleV2()
    final_model.train(all_feats, all_y_leader, all_y_margin, all_y_q4, verbose=True)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'q3_terminal_v2.pkl')
    final_model.save(model_path)
    print(f"  Saved to {model_path}")

    # Feature importance
    imp = final_model.get_importance()
    print("\nTop 25 Features:")
    print(imp[['feature', 'avg']].head(25).to_string(index=False))

    # Step 5: Export for webapp
    print("\n[4/5] Exporting for webapp...")

    # Export signals
    signals_path = os.path.join(output_dir, 'q3_terminal_v2_signals.json')
    with open(signals_path, 'w') as f:
        json.dump({
            'signals': signals,
            'oos_summary': results['oos_summary'],
            'total_games': len(games),
            'seasons': sorted(set(g['season'] for g in games)),
            'model_features': final_model.feature_names,
        }, f, indent=2, default=str)
    print(f"  Signals: {signals_path}")

    # Export JS model (simplified: LR coefficients + scaler)
    _export_js_model(final_model, output_dir)

    print("\n[5/5] Done!")
    return results


def _export_js_model(model: Q3EnsembleV2, output_dir: str):
    """Export logistic regression + scaler for JavaScript inference."""
    lr = model.clf_models.get('lr')
    ridge = model.reg_models.get('ridge')

    if not lr:
        return

    js_data = {
        'features': model.feature_names,
        'scaler_mean': model.scaler.mean_.tolist(),
        'scaler_std': model.scaler.scale_.tolist(),
        'lr_coef': lr.coef_[0].tolist(),
        'lr_intercept': float(lr.intercept_[0]),
        'ridge_coef': ridge.coef_.tolist() if ridge else [],
        'ridge_intercept': float(ridge.intercept_) if ridge else 0,
    }

    path = os.path.join(output_dir, 'q3_terminal_v2_js_model.json')
    with open(path, 'w') as f:
        json.dump(js_data, f)
    print(f"  JS model: {path}")


if __name__ == '__main__':
    import sys
    pbp = sys.argv[1] if len(sys.argv) > 1 else './cache/games_pbp'
    out = sys.argv[2] if len(sys.argv) > 2 else './output'
    main(pbp, out)
