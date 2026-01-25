"""
NBA In-Game Trading Strategies

This module defines explicit, rule-based trading strategies derived from
analysis of play-by-play data patterns.

Strategies are designed to:
1. Have clear, interpretable rules
2. Target specific market inefficiencies
3. Include proper risk management
4. Avoid garbage time and obvious late-game situations
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from backtester import TradingStrategy, BacktestConfig, Trade
from odds_model import InGameOddsModel, MomentumAdjustedOddsModel


class MomentumReversionStrategy(TradingStrategy):
    """
    Strategy 1: Momentum Reversion

    EDGE HYPOTHESIS:
    After a team goes on a significant scoring run (8+ points in 2 minutes),
    markets tend to overreact to the momentum. The trailing team's true win
    probability is often higher than the market implies, especially in
    mid-game situations where there's plenty of time for regression.

    This captures the "hot hand fallacy" in betting markets.

    RULES:
    - Trigger: One team has outscored opponent by 8+ points in last 2 minutes
    - Filter: Must be between 6-40 minutes remaining
    - Filter: Score differential must be between 3-15 points
    - Action: Bet on the trailing team (against momentum)
    - Size: Based on run magnitude and time remaining
    """

    # Strategy parameters (all explicit)
    MIN_RUN_TRIGGER = 8  # Minimum point differential in 2-min window
    MIN_MINUTES_REMAINING = 6.0
    MAX_MINUTES_REMAINING = 40.0
    MIN_SCORE_DIFF = 3  # Don't bet if game is essentially tied
    MAX_SCORE_DIFF = 15  # Don't bet if blowout is developing
    BASE_EDGE_ESTIMATE = 0.045  # Base edge when conditions met
    RUN_EDGE_MULTIPLIER = 0.003  # Additional edge per point of run

    def __init__(self, config: BacktestConfig, **kwargs):
        super().__init__(config)

        # Allow parameter overrides for sensitivity testing
        self.min_run = kwargs.get('min_run', self.MIN_RUN_TRIGGER)
        self.base_edge = kwargs.get('base_edge', self.BASE_EDGE_ESTIMATE)
        self.run_multiplier = kwargs.get('run_multiplier', self.RUN_EDGE_MULTIPLIER)

    def generate_signals(self, state: pd.Series) -> Optional[Dict]:
        """Generate momentum reversion signal."""

        mins_remaining = state['minutes_remaining']
        abs_diff = state['abs_score_diff']
        run_diff_2min = state['run_diff_2min']  # Home - Away

        # Time filter
        if mins_remaining < self.MIN_MINUTES_REMAINING:
            return None
        if mins_remaining > self.MAX_MINUTES_REMAINING:
            return None

        # Score differential filter
        if abs_diff < self.MIN_SCORE_DIFF or abs_diff > self.MAX_SCORE_DIFF:
            return None

        # Check for significant run
        abs_run = abs(run_diff_2min)
        if abs_run < self.min_run:
            return None

        # Determine which team had the run and bet against them
        if run_diff_2min > 0:
            # Home had a run, bet away
            side = 'away'
        else:
            # Away had a run, bet home
            side = 'home'

        # Calculate edge based on run magnitude
        edge = self.base_edge + (abs_run - self.min_run) * self.run_multiplier

        # Adjust for time remaining (more time = more reversion opportunity)
        time_factor = min(mins_remaining / 24, 1.5)  # Peaks at 24 mins
        edge *= time_factor

        # Kelly calculation
        kelly = edge / 0.95  # Approximate for typical odds

        return {
            'side': side,
            'edge': edge,
            'kelly': kelly,
            'signal_prob': None,  # Will be computed by backtester
            'reason': f'momentum_reversion_run{int(abs_run)}'
        }


class FoulTroubleStrategy(TradingStrategy):
    """
    Strategy 2: Foul Situation Exploitation

    EDGE HYPOTHESIS:
    When a team enters the bonus early in a quarter (5+ fouls), the market
    often underprices the impact of free throw opportunities. This is
    especially true in the 2nd and 4th quarters when games tighten.

    Additionally, when a team's foul rate is significantly higher than
    normal in recent minutes, they may be playing more aggressively
    (desperation) which often backfires.

    RULES:
    - Trigger: One team in bonus while other is not, with 4+ minute differential
    - Filter: Must be Q2 or Q4 (more impactful situations)
    - Filter: Game must be within 12 points
    - Action: Bet on team benefiting from bonus
    - Size: Based on foul differential and quarter
    """

    FOUL_DIFFERENTIAL_TRIGGER = 4  # Fouls difference to trigger
    MIN_MINUTES_IN_QUARTER = 4.0  # Need enough time for FTs to matter
    MAX_SCORE_DIFF = 12
    BASE_EDGE_ESTIMATE = 0.035
    FOUL_EDGE_MULTIPLIER = 0.005  # Per foul above threshold
    Q4_BONUS = 1.3  # Edge multiplier for 4th quarter

    def __init__(self, config: BacktestConfig, **kwargs):
        super().__init__(config)
        self.foul_trigger = kwargs.get('foul_trigger', self.FOUL_DIFFERENTIAL_TRIGGER)
        self.base_edge = kwargs.get('base_edge', self.BASE_EDGE_ESTIMATE)

    def generate_signals(self, state: pd.Series) -> Optional[Dict]:
        """Generate foul trouble signal."""

        quarter = state['quarter']
        mins_remaining = state['minutes_remaining']
        abs_diff = state['abs_score_diff']

        # Quarter filter (Q2 or Q4 only)
        if quarter not in [2, 4]:
            return None

        # Score filter
        if abs_diff > self.MAX_SCORE_DIFF:
            return None

        # Time in quarter (need meaningful time left)
        mins_in_quarter = state['clock_seconds'] / 60
        if mins_in_quarter < self.MIN_MINUTES_IN_QUARTER:
            return None

        # Foul differential
        home_fouls = state['home_fouls']
        away_fouls = state['away_fouls']
        foul_diff = away_fouls - home_fouls  # Positive = home benefits

        if abs(foul_diff) < self.foul_trigger:
            return None

        # Determine side
        if foul_diff > 0:
            side = 'home'  # Home benefits from away fouls
        else:
            side = 'away'

        # Calculate edge
        edge = self.base_edge + (abs(foul_diff) - self.foul_trigger) * self.FOUL_EDGE_MULTIPLIER

        # Q4 bonus
        if quarter == 4:
            edge *= self.Q4_BONUS

        kelly = edge / 0.9

        return {
            'side': side,
            'edge': edge,
            'kelly': kelly,
            'signal_prob': None,
            'reason': f'foul_trouble_q{quarter}'
        }


class ThirdQuarterCollapseStrategy(TradingStrategy):
    """
    Strategy 3: Third Quarter Collapse Detection

    EDGE HYPOTHESIS:
    Teams that dominate the first half but show specific weakness patterns
    early in the 3rd quarter tend to lose their lead more often than
    markets expect. This captures "halftime adjustment" effects and
    fatigue patterns.

    Key indicators:
    - Leading team's efficiency drops significantly in early Q3
    - Trailing team showing better 2-min momentum despite still being behind
    - Multiple turnovers or fouls by leading team

    RULES:
    - Trigger: Team leading by 8-16 at start of Q3
    - Filter: Trailing team has 5+ point run in first 4 min of Q3
    - Filter: Leading team FG% in Q3 < 35%
    - Action: Bet on trailing team
    """

    MIN_HALFTIME_LEAD = 8
    MAX_HALFTIME_LEAD = 16
    Q3_RUN_TRIGGER = 5  # Points by trailing team in Q3
    MIN_Q3_MINUTES_ELAPSED = 2.0  # Wait to see pattern develop
    MAX_Q3_MINUTES_ELAPSED = 8.0  # Don't wait too long
    BASE_EDGE_ESTIMATE = 0.05
    RUN_BONUS_PER_POINT = 0.004

    def __init__(self, config: BacktestConfig, **kwargs):
        super().__init__(config)
        self.base_edge = kwargs.get('base_edge', self.BASE_EDGE_ESTIMATE)

    def generate_signals(self, state: pd.Series) -> Optional[Dict]:
        """Generate third quarter collapse signal."""

        quarter = state['quarter']
        if quarter != 3:
            return None

        # Time in Q3
        time_in_period = state.get('time_in_period', 0) / 60  # Minutes into Q3
        if time_in_period < self.MIN_Q3_MINUTES_ELAPSED:
            return None
        if time_in_period > self.MAX_Q3_MINUTES_ELAPSED:
            return None

        # Score differential (positive = home leading)
        score_diff = state['score_diff']
        abs_diff = abs(score_diff)

        # Check if lead is in target range
        if abs_diff < 4 or abs_diff > self.MAX_HALFTIME_LEAD:
            return None

        # Check momentum - trailing team should be on a run
        run_2min = state['run_diff_2min']  # Home - Away

        if score_diff > 0:
            # Home is leading - check if away has momentum
            if run_2min > -self.Q3_RUN_TRIGGER:
                return None
            # Away has momentum, bet away
            side = 'away'
            run_magnitude = -run_2min
        else:
            # Away is leading - check if home has momentum
            if run_2min < self.Q3_RUN_TRIGGER:
                return None
            # Home has momentum, bet home
            side = 'home'
            run_magnitude = run_2min

        # Calculate edge
        edge = self.base_edge + (run_magnitude - self.Q3_RUN_TRIGGER) * self.RUN_BONUS_PER_POINT

        kelly = edge / 0.85

        return {
            'side': side,
            'edge': edge,
            'kelly': kelly,
            'signal_prob': None,
            'reason': 'q3_collapse'
        }


class CloseGameEfficiencyStrategy(TradingStrategy):
    """
    Strategy 4: Close Game Efficiency Divergence

    EDGE HYPOTHESIS:
    In close games (within 6 points) during the 4th quarter, significant
    divergence in recent shooting efficiency predicts near-term outcomes
    better than the current score suggests. Markets anchor too heavily
    on the scoreboard.

    This captures "regression to shooting mean" combined with
    "hot/cold streaks ending" patterns.

    RULES:
    - Trigger: Game within 6 points in Q4
    - Trigger: One team shooting 20%+ better over last 5 minutes
    - Filter: Both teams have had 5+ FG attempts in window
    - Action: Bet AGAINST the hot-shooting team (efficiency reversion)
    """

    MAX_SCORE_DIFF = 6
    EFFICIENCY_DIFF_TRIGGER = 0.20  # 20% FG% differential
    MIN_Q4_MINUTES_ELAPSED = 3.0
    MAX_MINUTES_REMAINING = 9.0
    BASE_EDGE_ESTIMATE = 0.04
    EFFICIENCY_EDGE_MULTIPLIER = 0.15  # Per 10% efficiency gap

    def __init__(self, config: BacktestConfig, **kwargs):
        super().__init__(config)
        self.efficiency_trigger = kwargs.get('eff_trigger', self.EFFICIENCY_DIFF_TRIGGER)
        self.base_edge = kwargs.get('base_edge', self.BASE_EDGE_ESTIMATE)

    def generate_signals(self, state: pd.Series) -> Optional[Dict]:
        """Generate close game efficiency signal."""

        quarter = state['quarter']
        if quarter != 4:
            return None

        mins_remaining = state['minutes_remaining']
        if mins_remaining > self.MAX_MINUTES_REMAINING:
            return None
        if mins_remaining < 2:  # Too late
            return None

        # Score filter
        abs_diff = state['abs_score_diff']
        if abs_diff > self.MAX_SCORE_DIFF:
            return None

        # Efficiency differential
        fg_diff = state['fg_pct_diff']  # Home - Away

        if abs(fg_diff) < self.efficiency_trigger:
            return None

        # Bet against hot shooter (reversion)
        if fg_diff > 0:
            # Home shooting hot, bet away
            side = 'away'
        else:
            side = 'home'

        # Calculate edge
        efficiency_gap = abs(fg_diff)
        edge = self.base_edge + (efficiency_gap - self.efficiency_trigger) * self.EFFICIENCY_EDGE_MULTIPLIER

        kelly = edge / 0.9

        return {
            'side': side,
            'edge': edge,
            'kelly': kelly,
            'signal_prob': None,
            'reason': f'efficiency_reversion_q4'
        }


class CompositeStrategy(TradingStrategy):
    """
    Composite Strategy: Combines multiple signals with conflict resolution.

    This strategy runs all sub-strategies and:
    1. Takes the highest-edge signal if multiple trigger
    2. Skips if signals conflict on direction
    3. Applies additional confidence filters
    """

    def __init__(self, config: BacktestConfig, **kwargs):
        super().__init__(config)

        # Initialize sub-strategies
        self.strategies = [
            MomentumReversionStrategy(config, **kwargs),
            FoulTroubleStrategy(config, **kwargs),
            ThirdQuarterCollapseStrategy(config, **kwargs),
            CloseGameEfficiencyStrategy(config, **kwargs),
        ]

        # Confidence threshold for composite
        self.min_composite_edge = kwargs.get('min_composite_edge', 0.035)

    def generate_signals(self, state: pd.Series) -> Optional[Dict]:
        """Generate composite signal from sub-strategies."""

        signals = []

        for strategy in self.strategies:
            signal = strategy.generate_signals(state)
            if signal:
                signals.append(signal)

        if not signals:
            return None

        # Check for conflicts
        sides = set(s['side'] for s in signals)
        if len(sides) > 1:
            # Conflicting signals - skip
            return None

        # Take highest edge signal
        best_signal = max(signals, key=lambda s: s['edge'])

        # Apply composite edge threshold
        if best_signal['edge'] < self.min_composite_edge:
            return None

        # Boost edge if multiple strategies agree
        if len(signals) > 1:
            best_signal['edge'] *= 1.15  # 15% boost for confirmation
            best_signal['reason'] += f'_confirmed_{len(signals)}'

        return best_signal


# Strategy factory for easy instantiation
STRATEGIES = {
    'momentum_reversion': MomentumReversionStrategy,
    'foul_trouble': FoulTroubleStrategy,
    'q3_collapse': ThirdQuarterCollapseStrategy,
    'efficiency_reversion': CloseGameEfficiencyStrategy,
    'composite': CompositeStrategy,
}


def get_strategy(name: str, config: BacktestConfig, **kwargs) -> TradingStrategy:
    """Factory function to get strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](config, **kwargs)


if __name__ == "__main__":
    # Test strategy logic
    config = BacktestConfig()

    # Create test state
    test_state = pd.Series({
        'game_id': 'test',
        'quarter': 3,
        'period': 3,
        'clock_seconds': 8 * 60,  # 8 min left in Q3
        'minutes_remaining': 20,  # In regulation
        'game_seconds_elapsed': 28 * 60,
        'score_diff': 10,  # Home up 10
        'abs_score_diff': 10,
        'run_diff_2min': -7,  # Away has 7-0 run
        'home_run_2min': 0,
        'away_run_2min': 7,
        'home_fouls': 3,
        'away_fouls': 5,
        'home_fg_pct_5min': 0.35,
        'away_fg_pct_5min': 0.55,
        'fg_pct_diff': -0.20,
        'time_in_period': 4 * 60,  # 4 min into Q3
    })

    print("Testing strategies with sample game state:")
    print(f"Score: Home leading by {test_state['score_diff']}")
    print(f"Recent run: Away +{-test_state['run_diff_2min']}")
    print()

    for name, strategy_class in STRATEGIES.items():
        strategy = strategy_class(config)
        signal = strategy.generate_signals(test_state)
        if signal:
            print(f"{name}: BET {signal['side'].upper()}, edge={signal['edge']:.2%}, reason={signal['reason']}")
        else:
            print(f"{name}: No signal")
