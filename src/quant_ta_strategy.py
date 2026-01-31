"""
NBA LIVE BETTING: QUANTITATIVE TECHNICAL ANALYSIS STRATEGY
============================================================

Approach: Treat the in-game lead margin as a PRICE SERIES and apply
real technical analysis indicators used by prop trading desks.

The "price" = home_score - away_score (home lead).
The "time axis" = minutes elapsed (0 to 48).
Scoring plays are irregular -> resample to regular 30-second bars.

INDICATORS APPLIED:
  1. EMA (fast/slow) -> trend direction & crossovers
  2. RSI -> overbought/oversold lead extensions
  3. Bollinger Bands -> volatility regime & mean reversion
  4. MACD -> momentum shifts & divergences
  5. Rate of Change (ROC) -> lead acceleration/deceleration
  6. Scoring Pace (Volume) -> intensity of action
  7. VWAP analog -> volume-weighted average lead
  8. Support/Resistance -> previous lead levels that held

SIGNAL TYPES:
  - RSI_FADE: RSI > 75 (overbought lead) -> fade the leader
  - BB_REVERT: Price outside upper/lower band -> mean reversion
  - MACD_CROSS: MACD line crosses signal line -> momentum shift
  - EMA_CROSS: Fast EMA crosses slow EMA -> trend reversal
  - DIVERGENCE: Lead making new highs but RSI/MACD declining
  - SQUEEZE_BREAK: Bollinger squeeze then breakout -> directional
  - PACE_SPIKE: Scoring pace surge -> trailing team comeback indicator

RISK MANAGEMENT (Quant Style):
  - Kelly criterion position sizing
  - Max exposure per game
  - Stop-loss on lead movement against position
  - Track Sharpe ratio, max drawdown, win rate, expectancy

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
# DATA LOADING (reuse proven loader)
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
    """Build game with scoring play states."""
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

        prev_home = home_score
        prev_away = away_score

        states.append({
            'mins_remaining': mins_remaining,
            'mins_elapsed': round(48.0 - mins_remaining, 2),
            'home_score': home_score,
            'away_score': away_score,
            'lead': home_score - away_score,  # positive = home leads
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
    prob = normal_cdf(z)
    return max(0.51, min(0.998, prob))


def implied_odds(prob: float, vig: float = 0.045) -> float:
    """American odds for the favorite side (negative)."""
    mp = min(0.995, prob + vig * prob)
    return -(mp / (1 - mp)) * 100


def dog_odds(prob: float, vig: float = 0.045) -> float:
    """American odds for the underdog side (positive)."""
    dog_true = 1 - prob
    dog_implied = dog_true * (1 + vig)
    dog_implied = max(0.01, min(0.99, dog_implied))
    return ((1 - dog_implied) / dog_implied) * 100


def ml_payout(odds: float) -> float:
    """Payout per unit risked."""
    if odds < 0:
        return 100 / abs(odds)
    else:
        return odds / 100


# =============================================================================
# TECHNICAL ANALYSIS ENGINE
# =============================================================================

class LeadSeries:
    """
    Converts irregular scoring-play data into a regular time series
    and computes all TA indicators. This is the core "chart" object.
    """

    def __init__(self, states: List[dict], bar_size: float = 0.5):
        """
        Args:
            states: list of scoring play dicts with mins_elapsed, lead
            bar_size: bar interval in minutes (0.5 = 30-second bars)
        """
        self.bar_size = bar_size
        self.times = []       # minutes elapsed
        self.leads = []       # lead value at each bar (forward-filled)
        self.volumes = []     # scoring events per bar (proxy for volume)
        self.raw_states = states

        self._build_bars()
        self._compute_indicators()

    def _build_bars(self):
        """Resample irregular scoring events into regular bars (like candlesticks)."""
        if not self.raw_states:
            return

        max_elapsed = max(s['mins_elapsed'] for s in self.raw_states)
        n_bars = int(max_elapsed / self.bar_size) + 1

        # Sort states by elapsed time
        sorted_states = sorted(self.raw_states, key=lambda s: s['mins_elapsed'])

        current_lead = 0
        state_idx = 0

        for i in range(n_bars):
            t = i * self.bar_size
            self.times.append(t)

            # Forward-fill: use latest lead value at or before this time
            events_in_bar = 0
            while state_idx < len(sorted_states) and sorted_states[state_idx]['mins_elapsed'] <= t + self.bar_size:
                current_lead = sorted_states[state_idx]['lead']
                state_idx += 1
                events_in_bar += 1

            self.leads.append(current_lead)
            self.volumes.append(events_in_bar)

    def _compute_indicators(self):
        """Compute all TA indicators on the lead series."""
        n = len(self.leads)
        if n < 10:
            self.ema_fast = self.leads[:]
            self.ema_slow = self.leads[:]
            self.rsi = [50.0] * n
            self.bb_upper = [0.0] * n
            self.bb_lower = [0.0] * n
            self.bb_mid = [0.0] * n
            self.bb_width = [0.0] * n
            self.macd_line = [0.0] * n
            self.macd_signal = [0.0] * n
            self.macd_hist = [0.0] * n
            self.roc = [0.0] * n
            self.pace = [0.0] * n
            self.vwap = [0.0] * n
            return

        # --- EMA (fast=5 bars=2.5min, slow=14 bars=7min) ---
        self.ema_fast = self._ema(self.leads, 5)
        self.ema_slow = self._ema(self.leads, 14)

        # --- RSI (period=10 bars=5min) ---
        self.rsi = self._rsi(self.leads, 10)

        # --- Bollinger Bands (period=14, std=2.0) ---
        bb_period = 14
        self.bb_mid = self._sma(self.leads, bb_period)
        self.bb_upper = [0.0] * n
        self.bb_lower = [0.0] * n
        self.bb_width = [0.0] * n
        for i in range(n):
            if i < bb_period - 1:
                self.bb_upper[i] = self.bb_mid[i] + 4
                self.bb_lower[i] = self.bb_mid[i] - 4
                self.bb_width[i] = 8.0
            else:
                window = self.leads[i - bb_period + 1:i + 1]
                std = statistics.stdev(window) if len(window) > 1 else 1.0
                std = max(std, 0.5)  # floor
                self.bb_upper[i] = self.bb_mid[i] + 2.0 * std
                self.bb_lower[i] = self.bb_mid[i] - 2.0 * std
                self.bb_width[i] = 4.0 * std

        # --- MACD (fast=5, slow=14, signal=5) ---
        self.macd_line = [self.ema_fast[i] - self.ema_slow[i] for i in range(n)]
        self.macd_signal = self._ema(self.macd_line, 5)
        self.macd_hist = [self.macd_line[i] - self.macd_signal[i] for i in range(n)]

        # --- Rate of Change (5-bar lookback = 2.5 min) ---
        roc_period = 5
        self.roc = [0.0] * n
        for i in range(roc_period, n):
            self.roc[i] = self.leads[i] - self.leads[i - roc_period]

        # --- Scoring Pace (rolling 10-bar sum of volume = 5 min window) ---
        self.pace = self._sma(self.volumes, 10)

        # --- VWAP analog: volume-weighted average lead ---
        self.vwap = [0.0] * n
        cum_vl = 0.0
        cum_v = 0.0
        for i in range(n):
            v = max(self.volumes[i], 0.1)  # min weight
            cum_vl += self.leads[i] * v
            cum_v += v
            self.vwap[i] = cum_vl / cum_v

    @staticmethod
    def _ema(data: List[float], period: int) -> List[float]:
        result = [0.0] * len(data)
        if not data:
            return result
        k = 2.0 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = data[i] * k + result[i - 1] * (1 - k)
        return result

    @staticmethod
    def _sma(data: List[float], period: int) -> List[float]:
        result = [0.0] * len(data)
        if not data:
            return result
        running_sum = 0.0
        for i in range(len(data)):
            running_sum += data[i]
            if i >= period:
                running_sum -= data[i - period]
            window_size = min(i + 1, period)
            result[i] = running_sum / window_size
        return result

    @staticmethod
    def _rsi(data: List[float], period: int) -> List[float]:
        n = len(data)
        result = [50.0] * n
        if n < period + 1:
            return result

        gains = [0.0] * n
        losses = [0.0] * n
        for i in range(1, n):
            change = data[i] - data[i - 1]
            gains[i] = max(change, 0)
            losses[i] = abs(min(change, 0))

        avg_gain = sum(gains[1:period + 1]) / period
        avg_loss = sum(losses[1:period + 1]) / period

        for i in range(period, n):
            if i > period:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                result[i] = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))

        return result

    def get_state_at_bar(self, bar_idx: int) -> dict:
        """Get all indicator values at a specific bar."""
        return {
            'time': self.times[bar_idx],
            'mins_remaining': 48.0 - self.times[bar_idx],
            'lead': self.leads[bar_idx],
            'ema_fast': self.ema_fast[bar_idx],
            'ema_slow': self.ema_slow[bar_idx],
            'rsi': self.rsi[bar_idx],
            'bb_upper': self.bb_upper[bar_idx],
            'bb_lower': self.bb_lower[bar_idx],
            'bb_mid': self.bb_mid[bar_idx],
            'bb_width': self.bb_width[bar_idx],
            'macd_line': self.macd_line[bar_idx],
            'macd_signal': self.macd_signal[bar_idx],
            'macd_hist': self.macd_hist[bar_idx],
            'roc': self.roc[bar_idx],
            'pace': self.pace[bar_idx],
            'vwap': self.vwap[bar_idx],
            'volume': self.volumes[bar_idx],
        }


# =============================================================================
# SIGNAL DETECTION ENGINE
# =============================================================================

class Signal:
    """A trading signal detected by the TA engine."""
    def __init__(self, signal_type: str, direction: str, bar_idx: int,
                 state: dict, strength: float, reason: str):
        self.signal_type = signal_type    # RSI_FADE, BB_REVERT, MACD_CROSS, etc.
        self.direction = direction         # 'bet_home' or 'bet_away'
        self.bar_idx = bar_idx
        self.state = state
        self.strength = strength           # 0-1 confidence
        self.reason = reason

    def __repr__(self):
        return f"Signal({self.signal_type}, {self.direction}, t={self.state['time']:.1f}, lead={self.state['lead']}, str={self.strength:.2f})"


def detect_signals(series: LeadSeries, min_time: float = 10.0, max_time: float = 40.0) -> List[Signal]:
    """
    Scan the lead series for all TA signals within the tradeable window.

    Args:
        series: LeadSeries with computed indicators
        min_time: earliest minutes_elapsed to consider (skip early game noise)
        max_time: latest minutes_elapsed (avoid end-of-game where spreads collapse)
    """
    signals = []
    n = len(series.leads)

    for i in range(2, n):  # need at least 2 bars of history
        t = series.times[i]
        if t < min_time or t > max_time:
            continue

        state = series.get_state_at_bar(i)
        prev_state = series.get_state_at_bar(i - 1)
        lead = state['lead']

        # Skip tiny leads (no edge in close games)
        if abs(lead) < 4:
            continue

        # ---------------------------------------------------------------
        # SIGNAL 1: RSI FADE (overbought/oversold lead)
        # ---------------------------------------------------------------
        # RSI > 75: home lead overextended -> bet away (fade)
        # RSI < 25: away lead overextended -> bet home (fade)
        if state['rsi'] > 75 and lead > 5:
            strength = min(1.0, (state['rsi'] - 75) / 25)
            signals.append(Signal(
                'RSI_FADE', 'bet_away', i, state, strength,
                f"RSI={state['rsi']:.0f} overbought, home lead {lead} overextended"
            ))
        elif state['rsi'] < 25 and lead < -5:
            strength = min(1.0, (25 - state['rsi']) / 25)
            signals.append(Signal(
                'RSI_FADE', 'bet_home', i, state, strength,
                f"RSI={state['rsi']:.0f} oversold, away lead {abs(lead)} overextended"
            ))

        # ---------------------------------------------------------------
        # SIGNAL 2: BOLLINGER BAND REVERSION
        # ---------------------------------------------------------------
        # Lead above upper band -> snap back expected -> fade leader
        # Lead below lower band -> snap back expected -> fade leader
        if lead > state['bb_upper'] and state['bb_width'] > 3:
            excess = (lead - state['bb_upper']) / max(state['bb_width'], 1)
            strength = min(1.0, excess)
            signals.append(Signal(
                'BB_REVERT', 'bet_away', i, state, strength,
                f"Lead {lead} above BB upper {state['bb_upper']:.1f}, width={state['bb_width']:.1f}"
            ))
        elif lead < state['bb_lower'] and state['bb_width'] > 3:
            excess = (state['bb_lower'] - lead) / max(state['bb_width'], 1)
            strength = min(1.0, excess)
            signals.append(Signal(
                'BB_REVERT', 'bet_home', i, state, strength,
                f"Lead {lead} below BB lower {state['bb_lower']:.1f}, width={state['bb_width']:.1f}"
            ))

        # ---------------------------------------------------------------
        # SIGNAL 3: MACD CROSSOVER (momentum shift)
        # ---------------------------------------------------------------
        # MACD line crosses below signal while lead is positive -> fading momentum
        if (prev_state['macd_hist'] > 0 and state['macd_hist'] <= 0
                and lead > 6):
            strength = min(1.0, abs(prev_state['macd_hist']) / 3)
            signals.append(Signal(
                'MACD_CROSS', 'bet_away', i, state, strength,
                f"MACD bearish cross, home lead={lead} but momentum fading"
            ))
        elif (prev_state['macd_hist'] < 0 and state['macd_hist'] >= 0
              and lead < -6):
            strength = min(1.0, abs(prev_state['macd_hist']) / 3)
            signals.append(Signal(
                'MACD_CROSS', 'bet_home', i, state, strength,
                f"MACD bullish cross, away lead={abs(lead)} but momentum fading"
            ))

        # ---------------------------------------------------------------
        # SIGNAL 4: EMA CROSSOVER (trend reversal)
        # ---------------------------------------------------------------
        # Fast EMA crosses below slow EMA while lead is positive -> trend reversal
        if (prev_state['ema_fast'] > prev_state['ema_slow']
                and state['ema_fast'] <= state['ema_slow']
                and lead > 4):
            gap = abs(prev_state['ema_fast'] - prev_state['ema_slow'])
            strength = min(1.0, gap / 3)
            signals.append(Signal(
                'EMA_CROSS', 'bet_away', i, state, strength,
                f"EMA death cross, home lead={lead} but trend reversing"
            ))
        elif (prev_state['ema_fast'] < prev_state['ema_slow']
              and state['ema_fast'] >= state['ema_slow']
              and lead < -4):
            gap = abs(prev_state['ema_fast'] - prev_state['ema_slow'])
            strength = min(1.0, gap / 3)
            signals.append(Signal(
                'EMA_CROSS', 'bet_home', i, state, strength,
                f"EMA golden cross, away lead={abs(lead)} but trend reversing"
            ))

        # ---------------------------------------------------------------
        # SIGNAL 5: LEAD DIVERGENCE (new high lead but weakening indicators)
        # ---------------------------------------------------------------
        # Lead at recent high but RSI declining -> bearish divergence
        if i >= 10:
            recent_leads = series.leads[i - 10:i + 1]
            recent_rsis = series.rsi[i - 10:i + 1]
            max_lead_idx = recent_leads.index(max(recent_leads))
            min_lead_idx = recent_leads.index(min(recent_leads))

            # Bearish divergence: lead near high but RSI lower than at prev high
            if (lead > 6 and max_lead_idx >= 7 and lead >= max(recent_leads) * 0.95
                    and recent_rsis[-1] < recent_rsis[max_lead_idx] - 10):
                strength = min(1.0, (recent_rsis[max_lead_idx] - recent_rsis[-1]) / 30)
                signals.append(Signal(
                    'DIVERGENCE', 'bet_away', i, state, strength,
                    f"Bearish divergence: lead={lead} near high but RSI declining"
                ))
            # Bullish divergence: lead near low but RSI higher than at prev low
            elif (lead < -6 and min_lead_idx >= 7 and lead <= min(recent_leads) * 0.95
                  and recent_rsis[-1] > recent_rsis[min_lead_idx] + 10):
                strength = min(1.0, (recent_rsis[-1] - recent_rsis[min_lead_idx]) / 30)
                signals.append(Signal(
                    'DIVERGENCE', 'bet_home', i, state, strength,
                    f"Bullish divergence: lead={lead} near low but RSI rising"
                ))

        # ---------------------------------------------------------------
        # SIGNAL 6: PACE SPIKE (scoring burst by trailing team)
        # ---------------------------------------------------------------
        # High scoring pace + ROC moving against the leader -> comeback
        if state['pace'] > 1.5 and state['roc'] < -3 and lead > 5:
            strength = min(1.0, abs(state['roc']) / 6)
            signals.append(Signal(
                'PACE_SPIKE', 'bet_away', i, state, strength,
                f"Scoring burst: pace={state['pace']:.1f}, ROC={state['roc']:.1f}, lead shrinking"
            ))
        elif state['pace'] > 1.5 and state['roc'] > 3 and lead < -5:
            strength = min(1.0, state['roc'] / 6)
            signals.append(Signal(
                'PACE_SPIKE', 'bet_home', i, state, strength,
                f"Scoring burst: pace={state['pace']:.1f}, ROC={state['roc']:.1f}, deficit shrinking"
            ))

        # ---------------------------------------------------------------
        # SIGNAL 7: VWAP REJECTION
        # ---------------------------------------------------------------
        # Lead crossed below VWAP while leader still has points advantage
        if (lead > 4 and state['lead'] < state['vwap'] - 1
                and prev_state['lead'] >= prev_state['vwap']):
            strength = min(1.0, (state['vwap'] - lead) / 4)
            signals.append(Signal(
                'VWAP_REJECT', 'bet_away', i, state, strength,
                f"Lead {lead} dropped below VWAP {state['vwap']:.1f}"
            ))
        elif (lead < -4 and state['lead'] > state['vwap'] + 1
              and prev_state['lead'] <= prev_state['vwap']):
            strength = min(1.0, (lead - state['vwap']) / 4)
            signals.append(Signal(
                'VWAP_REJECT', 'bet_home', i, state, strength,
                f"Lead {lead} rose above VWAP {state['vwap']:.1f}"
            ))

    return signals


# =============================================================================
# TRADE EXECUTION & RISK MANAGEMENT
# =============================================================================

class Trade:
    """A single trade with entry, exit, and P&L tracking."""
    def __init__(self, signal: Signal, game: dict, entry_odds: float,
                 position_size: float = 1.0):
        self.signal = signal
        self.game = game
        self.entry_bar = signal.bar_idx
        self.entry_time = signal.state['time']
        self.entry_lead = signal.state['lead']
        self.entry_odds = entry_odds
        self.direction = signal.direction
        self.position_size = position_size

        # Resolve outcome
        final_lead = game['final_lead']
        if self.direction == 'bet_home':
            self.won = final_lead > 0
        else:
            self.won = final_lead < 0

        # P&L
        payout = ml_payout(entry_odds)
        if self.won:
            self.pnl = payout * position_size
        else:
            self.pnl = -1.0 * position_size

    def __repr__(self):
        w = "W" if self.won else "L"
        return (f"Trade({self.signal.signal_type} {self.direction} "
                f"t={self.entry_time:.0f} lead={self.entry_lead} "
                f"odds={self.entry_odds:+.0f} {w} pnl={self.pnl:+.2f})")


def execute_trades_for_game(
    game: dict,
    series: LeadSeries,
    signals: List[Signal],
    min_signals: int = 2,
    min_avg_strength: float = 0.3,
    vig: float = 0.045,
    max_trades_per_game: int = 1,
) -> List[Trade]:
    """
    Execute trades based on confluent signals (multiple TA indicators agreeing).

    Like a real quant desk: don't trade on a single indicator. Require CONFLUENCE.

    Args:
        min_signals: minimum number of signals in same direction within a window
        min_avg_strength: minimum average signal strength
        max_trades_per_game: limit exposure per game
    """
    if not signals:
        return []

    trades = []

    # Group signals by time window (within 3 bars = 1.5 min)
    WINDOW = 3  # bars

    # Find clusters of agreeing signals
    i = 0
    while i < len(signals):
        cluster_home = []
        cluster_away = []
        j = i

        # Collect signals within window
        while j < len(signals) and signals[j].bar_idx <= signals[i].bar_idx + WINDOW:
            if signals[j].direction == 'bet_home':
                cluster_home.append(signals[j])
            else:
                cluster_away.append(signals[j])
            j += 1

        # Check for confluence
        for direction, cluster in [('bet_home', cluster_home), ('bet_away', cluster_away)]:
            if len(cluster) < min_signals:
                continue

            # Require different signal types (not just the same indicator firing multiple times)
            unique_types = set(s.signal_type for s in cluster)
            if len(unique_types) < min(2, min_signals):
                continue

            avg_strength = sum(s.strength for s in cluster) / len(cluster)
            if avg_strength < min_avg_strength:
                continue

            # Use the strongest signal as the anchor
            best_signal = max(cluster, key=lambda s: s.strength)
            lead = best_signal.state['lead']
            mins_rem = best_signal.state['mins_remaining']

            # Calculate market odds for the underdog side
            prob = market_win_prob(abs(lead), mins_rem)

            if direction == 'bet_away' and lead > 0:
                # Betting away when home leads -> away is underdog
                odds = dog_odds(prob, vig)
            elif direction == 'bet_home' and lead < 0:
                # Betting home when away leads -> home is underdog
                odds = dog_odds(prob, vig)
            elif direction == 'bet_home' and lead > 0:
                # Betting home when home leads -> home is favorite
                odds = implied_odds(prob, vig)
            else:
                # Betting away when away leads -> away is favorite
                odds = implied_odds(prob, vig)

            # Kelly criterion for position sizing
            if odds > 0:
                payout = odds / 100
            else:
                payout = 100 / abs(odds)

            # Estimate edge from signal strength
            # Conservative: signal strength maps to a small estimated edge
            est_win_prob = 0.5 + avg_strength * 0.15  # at most 65% estimated
            kelly_f = (est_win_prob * payout - (1 - est_win_prob)) / payout
            kelly_f = max(0, min(0.1, kelly_f))  # cap at 10% of bankroll

            # Minimum kelly to trade
            if kelly_f < 0.01:
                continue

            position_size = 1.0  # flat betting for clean P&L analysis

            trade = Trade(
                signal=best_signal,
                game=game,
                entry_odds=odds,
                position_size=position_size,
            )
            trade.n_signals = len(cluster)
            trade.unique_types = unique_types
            trade.avg_strength = avg_strength
            trade.kelly = kelly_f
            trade.cluster_signals = [s.signal_type for s in cluster]
            trades.append(trade)

            if len(trades) >= max_trades_per_game:
                return trades

        i = j if j > i else i + 1

    return trades


# =============================================================================
# FULL BACKTEST ENGINE
# =============================================================================

def backtest(
    games: List[dict],
    min_signals: int = 2,
    min_avg_strength: float = 0.3,
    min_time: float = 10.0,
    max_time: float = 40.0,
    vig: float = 0.045,
    label: str = "BACKTEST",
) -> dict:
    """Run full backtest across all games."""
    all_trades = []
    games_with_signals = 0
    games_with_trades = 0

    for game in games:
        series = LeadSeries(game['states'], bar_size=0.5)
        signals = detect_signals(series, min_time=min_time, max_time=max_time)

        if signals:
            games_with_signals += 1

        trades = execute_trades_for_game(
            game, series, signals,
            min_signals=min_signals,
            min_avg_strength=min_avg_strength,
            vig=vig,
        )

        if trades:
            games_with_trades += 1
            all_trades.extend(trades)

    # Compute stats
    n_trades = len(all_trades)
    if n_trades == 0:
        return {'label': label, 'n_trades': 0}

    wins = sum(1 for t in all_trades if t.won)
    losses = n_trades - wins
    win_rate = wins / n_trades

    total_pnl = sum(t.pnl for t in all_trades)
    roi = total_pnl / n_trades * 100

    # Equity curve
    equity = [0.0]
    for t in all_trades:
        equity.append(equity[-1] + t.pnl)

    max_equity = equity[0]
    max_drawdown = 0
    for e in equity:
        max_equity = max(max_equity, e)
        dd = max_equity - e
        max_drawdown = max(max_drawdown, dd)

    # Sharpe approximation (per-trade)
    pnls = [t.pnl for t in all_trades]
    avg_pnl = statistics.mean(pnls)
    if len(pnls) > 1:
        std_pnl = statistics.stdev(pnls)
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
    else:
        sharpe = 0

    # Profit factor
    gross_profit = sum(t.pnl for t in all_trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in all_trades if t.pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win/loss streaks
    max_win_streak = 0
    max_loss_streak = 0
    cur_streak = 0
    for t in all_trades:
        if t.won:
            cur_streak = cur_streak + 1 if cur_streak > 0 else 1
            max_win_streak = max(max_win_streak, cur_streak)
        else:
            cur_streak = cur_streak - 1 if cur_streak < 0 else -1
            max_loss_streak = max(max_loss_streak, abs(cur_streak))

    # Average odds
    avg_odds = statistics.mean(t.entry_odds for t in all_trades)

    # Signal type distribution
    signal_dist = defaultdict(int)
    for t in all_trades:
        signal_dist[t.signal.signal_type] += 1

    # Per-signal-type performance
    signal_perf = {}
    for stype in signal_dist:
        st_trades = [t for t in all_trades if t.signal.signal_type == stype]
        st_wins = sum(1 for t in st_trades if t.won)
        st_pnl = sum(t.pnl for t in st_trades)
        signal_perf[stype] = {
            'trades': len(st_trades),
            'win_rate': st_wins / len(st_trades) if st_trades else 0,
            'pnl': st_pnl,
            'roi': st_pnl / len(st_trades) * 100 if st_trades else 0,
        }

    # Bet direction stats
    home_trades = [t for t in all_trades if t.direction == 'bet_home']
    away_trades = [t for t in all_trades if t.direction == 'bet_away']

    # Underdog vs favorite stats
    dog_trades = [t for t in all_trades if t.entry_odds > 0]
    fav_trades = [t for t in all_trades if t.entry_odds < 0]

    return {
        'label': label,
        'n_games': len(games),
        'games_with_signals': games_with_signals,
        'games_with_trades': games_with_trades,
        'n_trades': n_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'roi': roi,
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'avg_odds': avg_odds,
        'avg_pnl': avg_pnl,
        'equity_curve': equity,
        'signal_dist': dict(signal_dist),
        'signal_perf': signal_perf,
        'home_bets': len(home_trades),
        'away_bets': len(away_trades),
        'dog_bets': len(dog_trades),
        'fav_bets': len(fav_trades),
        'dog_wr': sum(1 for t in dog_trades if t.won) / len(dog_trades) if dog_trades else 0,
        'fav_wr': sum(1 for t in fav_trades if t.won) / len(fav_trades) if fav_trades else 0,
        'dog_pnl': sum(t.pnl for t in dog_trades),
        'fav_pnl': sum(t.pnl for t in fav_trades),
        'all_trades': all_trades,
    }


def print_results(r: dict):
    """Print formatted backtest results."""
    if r['n_trades'] == 0:
        print(f"\n{r['label']}: NO TRADES")
        return

    print(f"""
{'='*70}
{r['label']}
{'='*70}
DATASET: {r['n_games']} games | {r['games_with_signals']} with signals | {r['games_with_trades']} with trades

PERFORMANCE:
  Trades: {r['n_trades']}  ({r['wins']}W / {r['losses']}L)
  Win Rate: {r['win_rate']*100:.1f}%
  Total P&L: {r['total_pnl']:+.2f} units
  ROI: {r['roi']:+.1f}%
  Avg P&L/trade: {r['avg_pnl']:+.3f} units

RISK METRICS:
  Sharpe Ratio: {r['sharpe']:.3f}
  Profit Factor: {r['profit_factor']:.2f}
  Max Drawdown: {r['max_drawdown']:.2f} units
  Max Win Streak: {r['max_win_streak']}
  Max Loss Streak: {r['max_loss_streak']}

MARKET:
  Avg Entry Odds: {r['avg_odds']:+.0f}
  Underdog bets: {r['dog_bets']} ({r['dog_wr']*100:.1f}% WR, {r['dog_pnl']:+.2f}u)
  Favorite bets: {r['fav_bets']} ({r['fav_wr']*100:.1f}% WR, {r['fav_pnl']:+.2f}u)

SIGNAL BREAKDOWN:""")

    for stype, perf in sorted(r['signal_perf'].items(), key=lambda x: -x[1]['trades']):
        print(f"  {stype:15s}: {perf['trades']:4d} trades | {perf['win_rate']*100:5.1f}% WR | {perf['pnl']:+7.2f}u | {perf['roi']:+6.1f}% ROI")

    # Equity curve summary
    eq = r['equity_curve']
    peak = max(eq)
    valley = min(eq)
    print(f"\nEQUITY CURVE:")
    print(f"  Peak: {peak:+.2f}u | Valley: {valley:+.2f}u | Final: {eq[-1]:+.2f}u")

    # Simple ASCII equity curve
    if len(eq) > 10:
        n_cols = 60
        step = max(1, len(eq) // n_cols)
        sampled = [eq[i] for i in range(0, len(eq), step)]
        mn, mx = min(sampled), max(sampled)
        rng = mx - mn if mx != mn else 1
        rows = 10
        grid = [[' '] * len(sampled) for _ in range(rows)]
        for ci, v in enumerate(sampled):
            ri = int((v - mn) / rng * (rows - 1))
            ri = rows - 1 - ri  # flip
            grid[ri][ci] = '*'
        for row in grid:
            print(f"  |{''.join(row)}|")
        print(f"  +{'-' * len(sampled)}+")


# =============================================================================
# CROSS-SEASON VALIDATION
# =============================================================================

def split_seasons(games: List[dict]) -> dict:
    """Split games into seasons based on date."""
    seasons = defaultdict(list)
    for g in games:
        date = g['date']
        if not date or len(date) < 6:
            continue
        year = int(date[:4])
        month = int(date[4:6])
        # NBA season: Oct-Jun, season labeled by start year
        if month >= 10:
            season = f"{year}-{year+1}"
        else:
            season = f"{year-1}-{year}"
        seasons[season].append(g)
    return dict(seasons)


# =============================================================================
# MULTI-CONFIGURATION SWEEP
# =============================================================================

def run_config_sweep(games: List[dict]):
    """
    Sweep multiple configurations like a quant research pipeline.
    Test different signal confluence requirements and time windows.
    """
    print("\n" + "=" * 70)
    print("CONFIGURATION SWEEP (Quant Parameter Grid)")
    print("=" * 70)

    configs = [
        # (min_signals, min_strength, min_time, max_time, label)
        (2, 0.2, 10, 40, "2 signals, loose, full window"),
        (2, 0.4, 10, 40, "2 signals, moderate, full window"),
        (2, 0.5, 10, 40, "2 signals, strict, full window"),
        (3, 0.3, 10, 40, "3 signals, moderate, full window"),
        (3, 0.4, 10, 40, "3 signals, strict, full window"),
        (2, 0.3, 12, 36, "2 signals, Q2-Q3 only"),
        (2, 0.3, 20, 40, "2 signals, second half only"),
        (2, 0.3, 10, 30, "2 signals, mid-game (avoid Q4)"),
        (3, 0.3, 15, 38, "3 signals, tight window"),
        (2, 0.2, 10, 42, "2 signals, loose, extended"),
    ]

    results = []
    for min_sig, min_str, min_t, max_t, label in configs:
        r = backtest(
            games,
            min_signals=min_sig,
            min_avg_strength=min_str,
            min_time=min_t,
            max_time=max_t,
            label=label,
        )
        results.append(r)

    # Summary table
    print(f"\n{'Config':<40} {'Trades':>6} {'WR':>6} {'ROI':>7} {'PnL':>8} {'Sharpe':>7} {'PF':>6} {'DD':>6}")
    print("-" * 90)
    for r in results:
        if r['n_trades'] == 0:
            print(f"{r['label']:<40} {'--':>6}")
            continue
        print(f"{r['label']:<40} {r['n_trades']:>6} {r['win_rate']*100:>5.1f}% {r['roi']:>+6.1f}% {r['total_pnl']:>+7.1f}u {r['sharpe']:>7.3f} {r['profit_factor']:>5.2f} {r['max_drawdown']:>5.1f}")

    return results


# =============================================================================
# MAIN: FULL ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("NBA QUANT TA STRATEGY - STOCK TRADER APPROACH")
    print("Treating lead margin as price series with full technical analysis")
    print("=" * 70)

    games = load_all_games()
    if not games:
        print("No games loaded.")
        return

    print(f"\nTotal games: {len(games)}")

    # -------------------------------------------------------------------------
    # 1. FULL DATASET BACKTEST (primary configuration)
    # -------------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# PHASE 1: PRIMARY BACKTEST (2-signal confluence, full window)")
    print("#" * 70)

    primary = backtest(
        games,
        min_signals=2,
        min_avg_strength=0.3,
        min_time=10.0,
        max_time=40.0,
        label="PRIMARY: 2+ confluent signals"
    )
    print_results(primary)

    # -------------------------------------------------------------------------
    # 2. STRICT CONFLUENCE (3+ signals)
    # -------------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# PHASE 2: STRICT CONFLUENCE (3+ different TA signals agreeing)")
    print("#" * 70)

    strict = backtest(
        games,
        min_signals=3,
        min_avg_strength=0.35,
        min_time=10.0,
        max_time=40.0,
        label="STRICT: 3+ confluent signals"
    )
    print_results(strict)

    # -------------------------------------------------------------------------
    # 3. CROSS-SEASON VALIDATION
    # -------------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# PHASE 3: CROSS-SEASON VALIDATION")
    print("#" * 70)

    seasons = split_seasons(games)
    for season_name, season_games in sorted(seasons.items()):
        if len(season_games) < 100:
            continue
        r = backtest(
            season_games,
            min_signals=2,
            min_avg_strength=0.3,
            min_time=10.0,
            max_time=40.0,
            label=f"SEASON {season_name} (2-signal)"
        )
        print_results(r)

    # -------------------------------------------------------------------------
    # 4. CONFIGURATION SWEEP
    # -------------------------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("# PHASE 4: PARAMETER SWEEP")
    print("#" * 70)

    sweep_results = run_config_sweep(games)

    # -------------------------------------------------------------------------
    # 5. DEEP DIVE: Best configuration analysis
    # -------------------------------------------------------------------------
    profitable = [r for r in sweep_results if r['n_trades'] >= 30 and r['roi'] > 0]

    if profitable:
        best = max(profitable, key=lambda r: r['roi'])
        print("\n\n" + "#" * 70)
        print("# PHASE 5: BEST CONFIGURATION DEEP DIVE")
        print("#" * 70)
        print_results(best)

        # Show sample trades
        print("\nSAMPLE TRADES (first 20):")
        for t in best['all_trades'][:20]:
            w = "WIN " if t.won else "LOSS"
            print(f"  {t.game['date']} {t.game['away_team']}@{t.game['home_team']} | "
                  f"{t.signal.signal_type:15s} {t.direction:8s} | "
                  f"lead={t.entry_lead:+3d} t={t.entry_time:4.1f} | "
                  f"odds={t.entry_odds:+6.0f} | {w} pnl={t.pnl:+.2f} | "
                  f"signals={t.cluster_signals}")

        # Monthly breakdown
        print("\nMONTHLY BREAKDOWN:")
        monthly = defaultdict(list)
        for t in best['all_trades']:
            month = t.game['date'][:6]
            monthly[month].append(t)

        for month in sorted(monthly.keys()):
            trades = monthly[month]
            w = sum(1 for t in trades if t.won)
            pnl = sum(t.pnl for t in trades)
            print(f"  {month}: {len(trades):3d} trades | {w}W/{len(trades)-w}L | {w/len(trades)*100:5.1f}% | {pnl:+7.2f}u")
    else:
        print("\n\n" + "#" * 70)
        print("# NO PROFITABLE CONFIG WITH 30+ TRADES FOUND")
        print("#" * 70)
        print("\nAll configurations with 30+ trades:")
        for r in sorted(sweep_results, key=lambda r: r.get('roi', -999), reverse=True):
            if r['n_trades'] >= 10:
                print(f"  {r['label']}: {r['n_trades']} trades, {r['win_rate']*100:.1f}% WR, {r['roi']:+.1f}% ROI")

    # -------------------------------------------------------------------------
    # 6. HONEST VERDICT
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    all_results = [primary, strict] + [
        backtest(season_games, min_signals=2, min_avg_strength=0.3,
                 min_time=10.0, max_time=40.0, label=f"s{sn}")
        for sn, season_games in sorted(seasons.items()) if len(season_games) >= 100
    ]

    profitable_count = sum(1 for r in all_results if r['n_trades'] >= 20 and r['roi'] > 0)
    total_with_trades = sum(1 for r in all_results if r['n_trades'] >= 20)

    if profitable_count == 0:
        print("""
RESULT: NO PROFITABLE EDGE FOUND via technical analysis.

The lead margin behaves differently from stock prices:
  - NBA scoring is discrete (2s and 3s), not continuous
  - Leads have natural mean reversion ALREADY PRICED by the market
  - TA indicators on a 48-minute series have very few data points
  - The market's efficiency absorbs these patterns

Technical analysis on lead margins does NOT produce an exploitable edge
when realistic vig (4.5%) is applied. The market is too efficient.
""")
    elif profitable_count < total_with_trades // 2:
        print(f"""
RESULT: MIXED - {profitable_count}/{total_with_trades} configs profitable.

Some TA patterns show edge but inconsistently across seasons.
This could be noise. Requires larger dataset to confirm.
""")
    else:
        print(f"""
RESULT: POTENTIAL EDGE - {profitable_count}/{total_with_trades} configs profitable.

TA-based signals show consistent edge across seasons.
Key: Signal confluence (multiple indicators agreeing) is critical.
Next step: Paper trade for one season to validate out-of-sample.
""")


if __name__ == '__main__':
    main()
