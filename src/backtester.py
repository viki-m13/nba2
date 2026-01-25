"""
Backtesting Engine for NBA In-Game Trading Strategy

This module provides realistic backtesting with:
1. Transaction costs / vig modeling
2. Position sizing and exposure limits
3. Walk-forward validation
4. Performance metrics and risk analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings

from odds_model import (
    InGameOddsModel,
    MomentumAdjustedOddsModel,
    OddsEstimate,
    expected_value_per_bet,
    american_odds_to_prob,
    prob_to_american_odds
)


@dataclass
class Trade:
    """Represents a single trade/bet."""
    game_id: str
    timestamp: float  # Game seconds elapsed
    side: str  # 'home' or 'away'
    stake: float
    odds: float  # American odds at entry
    signal_prob: float  # Our model's probability
    edge: float  # Estimated edge at entry

    # Outcome (filled after game ends)
    result: Optional[str] = None  # 'win', 'loss', 'push'
    pnl: Optional[float] = None
    actual_winner: Optional[str] = None

    # Additional context
    score_diff: int = 0
    minutes_remaining: float = 0
    momentum_2min: float = 0
    reason: str = ""


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Execution assumptions
    vig_pct: float = 4.5  # Market vigorish
    slippage_prob: float = 0.02  # Additional slippage in fast markets

    # Position limits
    max_stake_per_trade: float = 1.0  # Max units per trade
    max_trades_per_game: int = 3  # Limit overtrading
    max_daily_exposure: float = 10.0  # Max total stake per day
    max_concurrent_positions: int = 5  # Across all games

    # Signal thresholds
    min_edge_threshold: float = 0.03  # Minimum edge to trade
    min_kelly_fraction: float = 0.01  # Minimum Kelly to trade

    # Timing constraints
    min_minutes_remaining: float = 2.0  # Don't trade in final 2 min
    max_minutes_remaining: float = 45.0  # Don't trade too early

    # Garbage time filter
    blowout_threshold: int = 20  # Ignore games with lead > 20
    blowout_time_threshold: float = 6.0  # Only filter blowouts in last 6 min

    # Walk-forward settings
    train_seasons: int = 2  # Number of seasons for training
    test_seasons: int = 1  # Number of seasons for testing


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    trades: List[Trade]
    daily_pnl: pd.DataFrame
    season_summary: pd.DataFrame

    # Aggregate metrics
    total_pnl: float = 0
    total_trades: int = 0
    win_rate: float = 0
    avg_edge: float = 0
    avg_pnl_per_trade: float = 0
    sharpe_ratio: float = 0
    max_drawdown: float = 0
    profit_factor: float = 0

    # By season
    pnl_by_season: Dict[str, float] = field(default_factory=dict)
    trades_by_season: Dict[str, int] = field(default_factory=dict)


class TradingStrategy:
    """
    Base class for trading strategies.

    Subclasses implement the `generate_signals` method.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.odds_model = InGameOddsModel(vig_pct=config.vig_pct)

    def generate_signals(self, game_state: pd.Series) -> Optional[Dict]:
        """
        Generate trading signal for a game state.

        Args:
            game_state: Series with game state features

        Returns:
            Dict with signal info or None for no trade
            Required keys: 'side', 'edge', 'kelly', 'reason'
        """
        raise NotImplementedError

    def should_filter(self, game_state: pd.Series) -> Tuple[bool, str]:
        """
        Check if game state should be filtered (no trading).

        Args:
            game_state: Series with game state features

        Returns:
            Tuple of (should_filter, reason)
        """
        mins_remaining = game_state['minutes_remaining']
        abs_diff = game_state['abs_score_diff']

        # Time filters
        if mins_remaining < self.config.min_minutes_remaining:
            return True, "too_late"

        if mins_remaining > self.config.max_minutes_remaining:
            return True, "too_early"

        # Garbage time filter
        if (mins_remaining < self.config.blowout_time_threshold and
            abs_diff > self.config.blowout_threshold):
            return True, "blowout"

        return False, ""


class Backtester:
    """
    Runs realistic backtests of trading strategies.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.odds_model = InGameOddsModel(vig_pct=config.vig_pct)

    def run_backtest(self,
                    strategy: TradingStrategy,
                    game_states_df: pd.DataFrame,
                    game_outcomes: Dict[str, str]) -> BacktestResults:
        """
        Run backtest on historical data.

        Args:
            strategy: Trading strategy to test
            game_states_df: DataFrame with all game states
            game_outcomes: Dict mapping game_id -> 'home' or 'away' (winner)

        Returns:
            BacktestResults with all metrics
        """
        trades = []
        games_traded = set()
        trades_by_game = {}

        # Group by game
        for game_id, game_df in game_states_df.groupby('game_id'):
            if game_id not in game_outcomes:
                continue

            winner = game_outcomes[game_id]
            game_trades = []

            # Process each time step in the game
            for _, state in game_df.iterrows():
                # Check if we've hit trade limit for this game
                if len(game_trades) >= self.config.max_trades_per_game:
                    continue

                # Check filters
                should_filter, filter_reason = strategy.should_filter(state)
                if should_filter:
                    continue

                # Generate signal
                signal = strategy.generate_signals(state)

                if signal is None:
                    continue

                # Check edge threshold
                if signal['edge'] < self.config.min_edge_threshold:
                    continue

                # Get market odds
                odds_est = self.odds_model.estimate_odds(
                    state['score_diff'],
                    state['minutes_remaining']
                )

                if signal['side'] == 'home':
                    market_odds = odds_est.home_market_odds
                    signal_prob = signal.get('signal_prob', odds_est.home_win_prob + signal['edge'])
                else:
                    market_odds = odds_est.away_market_odds
                    signal_prob = signal.get('signal_prob', odds_est.away_win_prob + signal['edge'])

                # Size the trade (using Kelly or fixed)
                kelly = signal.get('kelly', 0.1)
                stake = min(
                    kelly * 0.25,  # Quarter Kelly for safety
                    self.config.max_stake_per_trade
                )

                if stake < 0.01:
                    continue

                # Create trade
                trade = Trade(
                    game_id=game_id,
                    timestamp=state['game_seconds_elapsed'],
                    side=signal['side'],
                    stake=stake,
                    odds=market_odds,
                    signal_prob=signal_prob,
                    edge=signal['edge'],
                    score_diff=int(state['score_diff']),
                    minutes_remaining=state['minutes_remaining'],
                    momentum_2min=state.get('momentum_2min', 0),
                    reason=signal.get('reason', '')
                )

                game_trades.append(trade)

            # Settle trades for this game
            for trade in game_trades:
                trade.actual_winner = winner
                if trade.side == winner:
                    trade.result = 'win'
                    if trade.odds > 0:
                        trade.pnl = trade.stake * (trade.odds / 100)
                    else:
                        trade.pnl = trade.stake * (100 / (-trade.odds))
                else:
                    trade.result = 'loss'
                    trade.pnl = -trade.stake

            trades.extend(game_trades)

        # Compute results
        return self._compute_results(trades, game_states_df)

    def _compute_results(self, trades: List[Trade],
                        game_states_df: pd.DataFrame) -> BacktestResults:
        """Compute backtest metrics from trades."""

        if not trades:
            return BacktestResults(
                trades=[],
                daily_pnl=pd.DataFrame(),
                season_summary=pd.DataFrame()
            )

        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame([{
            'game_id': t.game_id,
            'timestamp': t.timestamp,
            'side': t.side,
            'stake': t.stake,
            'odds': t.odds,
            'edge': t.edge,
            'result': t.result,
            'pnl': t.pnl,
            'minutes_remaining': t.minutes_remaining,
            'score_diff': t.score_diff,
            'reason': t.reason
        } for t in trades])

        # Get season from game_id (format: 00XXYYYYY where XX is season)
        def get_season_from_game_id(gid):
            try:
                # Game IDs like 0022300001 -> season 23 -> 2023-24
                season_code = str(gid)[2:4]
                return f"20{season_code}-{int(season_code)+1:02d}"
            except:
                return "unknown"

        trades_df['season'] = trades_df['game_id'].apply(get_season_from_game_id)

        # Aggregate metrics
        total_pnl = trades_df['pnl'].sum()
        total_trades = len(trades_df)
        win_rate = (trades_df['result'] == 'win').mean()
        avg_edge = trades_df['edge'].mean()
        avg_pnl_per_trade = trades_df['pnl'].mean()

        # Profit factor
        wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losses = -trades_df[trades_df['pnl'] < 0]['pnl'].sum()
        profit_factor = wins / losses if losses > 0 else float('inf')

        # Sharpe (assuming daily aggregation proxy)
        pnl_std = trades_df['pnl'].std()
        sharpe = (avg_pnl_per_trade / pnl_std * np.sqrt(252)) if pnl_std > 0 else 0

        # Drawdown
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max()

        # By season
        season_stats = trades_df.groupby('season').agg({
            'pnl': ['sum', 'count', 'mean'],
            'edge': 'mean',
            'result': lambda x: (x == 'win').mean()
        }).round(4)

        pnl_by_season = trades_df.groupby('season')['pnl'].sum().to_dict()
        trades_by_season = trades_df.groupby('season')['pnl'].count().to_dict()

        return BacktestResults(
            trades=trades,
            daily_pnl=trades_df,
            season_summary=season_stats,
            total_pnl=total_pnl,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_edge=avg_edge,
            avg_pnl_per_trade=avg_pnl_per_trade,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            pnl_by_season=pnl_by_season,
            trades_by_season=trades_by_season
        )


def run_walk_forward_backtest(strategy_class,
                              strategy_params: Dict,
                              all_seasons_data: Dict[str, pd.DataFrame],
                              game_outcomes: Dict[str, str],
                              config: BacktestConfig) -> Dict:
    """
    Run walk-forward backtest across multiple seasons.

    Args:
        strategy_class: Strategy class to instantiate
        strategy_params: Parameters for strategy
        all_seasons_data: Dict mapping season -> game states DataFrame
        game_outcomes: Dict mapping game_id -> winner
        config: Backtest configuration

    Returns:
        Dict with walk-forward results
    """
    seasons = sorted(all_seasons_data.keys())

    if len(seasons) < config.train_seasons + config.test_seasons:
        raise ValueError(f"Need at least {config.train_seasons + config.test_seasons} seasons")

    results_by_test_season = {}

    # Walk forward: train on N seasons, test on next season
    for i in range(config.train_seasons, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[i - config.train_seasons:i]

        # Train data (for parameter tuning if needed)
        train_data = pd.concat([all_seasons_data[s] for s in train_seasons])

        # Test data
        test_data = all_seasons_data[test_season]

        # Create strategy with params
        strategy = strategy_class(config, **strategy_params)

        # Run backtest on test season
        backtester = Backtester(config)
        results = backtester.run_backtest(strategy, test_data, game_outcomes)

        results_by_test_season[test_season] = {
            'results': results,
            'train_seasons': train_seasons,
            'metrics': {
                'total_pnl': results.total_pnl,
                'total_trades': results.total_trades,
                'win_rate': results.win_rate,
                'avg_edge': results.avg_edge,
                'sharpe': results.sharpe_ratio,
                'max_dd': results.max_drawdown
            }
        }

    return results_by_test_season


def print_backtest_summary(results: BacktestResults, title: str = "Backtest Results"):
    """Print formatted backtest summary."""

    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    print(f"\nOverall Performance:")
    print(f"  Total P&L:           {results.total_pnl:+.2f} units")
    print(f"  Total Trades:        {results.total_trades}")
    print(f"  Win Rate:            {results.win_rate:.1%}")
    print(f"  Avg Edge:            {results.avg_edge:.2%}")
    print(f"  Avg P&L/Trade:       {results.avg_pnl_per_trade:+.4f} units")
    print(f"  Profit Factor:       {results.profit_factor:.2f}")
    print(f"  Sharpe Ratio:        {results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:        {results.max_drawdown:.2f} units")

    if results.pnl_by_season:
        print(f"\nPerformance by Season:")
        for season in sorted(results.pnl_by_season.keys()):
            pnl = results.pnl_by_season[season]
            trades = results.trades_by_season[season]
            print(f"  {season}: P&L={pnl:+.2f}, Trades={trades}")

    if len(results.trades) > 0:
        # Analyze by reason
        trades_df = results.daily_pnl
        if 'reason' in trades_df.columns:
            print(f"\nPerformance by Signal Type:")
            by_reason = trades_df.groupby('reason').agg({
                'pnl': ['sum', 'count', 'mean'],
            }).round(4)
            print(by_reason)


if __name__ == "__main__":
    # Quick test
    config = BacktestConfig()
    print("Backtest config loaded successfully")
    print(f"Vig: {config.vig_pct}%")
    print(f"Min edge: {config.min_edge_threshold:.1%}")
