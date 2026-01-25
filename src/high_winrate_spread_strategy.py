"""
High Win Rate Spread Trading Strategy

Optimized for 80-90%+ win rate by:
1. Using tight take-profits (capture small moves)
2. Wider stop-losses (give trades room)
3. Highly selective entry conditions
4. Multiple confirmation signals

The trade-off: smaller wins per trade, but much higher consistency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    LONG_HOME = "long_home"
    LONG_AWAY = "long_away"
    FLAT = "flat"


@dataclass
class HighWRConfig:
    """Configuration optimized for high win rate."""
    # Entry - be very selective
    min_spread_deviation: float = 3.0  # Wait for bigger mispricing
    min_momentum_magnitude: float = 6  # Need strong momentum signal
    max_minutes_remaining: float = 40.0
    min_minutes_remaining: float = 5.0

    # Exit - small profits, wide stops
    take_profit_points: float = 0.75  # Take small profits quickly
    stop_loss_points: float = 4.0  # Wide stop to avoid getting stopped
    time_stop_seconds: float = 180  # 3 min max hold

    # Additional filters
    require_momentum_confirmation: bool = True
    min_volatility_for_entry: float = 0.8


@dataclass
class Position:
    game_id: str
    entry_time: float
    entry_spread: float
    side: Side
    entry_fair: float


@dataclass
class Trade:
    game_id: str
    entry_time: float
    exit_time: float
    entry_spread: float
    exit_spread: float
    side: Side
    pnl: float
    exit_reason: str
    hold_time: float


class HighWinRateSpreadStrategy:
    """
    Strategy optimized for maximum win rate (80%+).

    Key insight: In spread trading, you can achieve very high win rates
    by taking small profits quickly while using wider stops. This works
    because spread movements are mean-reverting in the short term.

    Trade-off: Average win is small, but you win very often.
    """

    HOME_COURT_ADVANTAGE = 3.0
    MOMENTUM_OVERREACTION_FACTOR = 0.5

    def __init__(self, config: HighWRConfig = None):
        self.config = config or HighWRConfig()
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.last_entry_time: Dict[str, float] = {}
        self.trade_count: Dict[str, int] = {}

    def get_fair_spread(self, score_diff: float, mins_remaining: float,
                        momentum_2min: float, momentum_5min: float) -> float:
        """Calculate model's fair spread."""
        fair = -score_diff
        fair -= self.HOME_COURT_ADVANTAGE * (mins_remaining / 48)
        # Momentum reversion expectation
        fair += momentum_2min * 0.3 * self.MOMENTUM_OVERREACTION_FACTOR
        return fair

    def get_market_spread(self, score_diff: float, mins_remaining: float,
                          momentum_2min: float) -> float:
        """Estimate current market spread (with overreaction)."""
        market = -score_diff
        market -= self.HOME_COURT_ADVANTAGE * (mins_remaining / 48)
        market -= momentum_2min * 0.25  # Market moves WITH momentum
        return market

    def check_entry_conditions(self, state: pd.Series) -> Optional[Dict]:
        """Check for high-confidence entry."""
        game_id = state['game_id']
        game_secs = state['game_seconds_elapsed']
        mins_rem = state['minutes_remaining']
        score_diff = state['score_diff']
        mom_2 = state.get('momentum_2min', state.get('run_diff_2min', 0))
        mom_5 = state.get('momentum_5min', state.get('run_diff_5min', 0))

        # Position check
        if game_id in self.positions:
            return None

        # Time filters
        if mins_rem < self.config.min_minutes_remaining:
            return None
        if mins_rem > self.config.max_minutes_remaining:
            return None

        # Blowout filter
        if abs(score_diff) > 22:
            return None

        # Cooldown
        if game_secs - self.last_entry_time.get(game_id, 0) < 45:
            return None

        # Trade limit per game
        if self.trade_count.get(game_id, 0) >= 15:
            return None

        # Momentum confirmation
        if self.config.require_momentum_confirmation:
            if abs(mom_2) < self.config.min_momentum_magnitude:
                return None
            # Momentum should be in same direction over 2 and 5 min
            if np.sign(mom_2) != np.sign(mom_5) and abs(mom_5) > 3:
                return None

        # Calculate spreads
        fair = self.get_fair_spread(score_diff, mins_rem, mom_2, mom_5)
        market = self.get_market_spread(score_diff, mins_rem, mom_2)
        deviation = market - fair

        # Check for entry
        if abs(deviation) < self.config.min_spread_deviation:
            return None

        # Determine side
        if deviation < -self.config.min_spread_deviation:
            side = Side.LONG_AWAY
        else:
            side = Side.LONG_HOME

        # Store position
        self.positions[game_id] = Position(
            game_id=game_id,
            entry_time=game_secs,
            entry_spread=market,
            side=side,
            entry_fair=fair
        )

        return {'action': 'enter', 'side': side, 'deviation': deviation}

    def check_exit_conditions(self, state: pd.Series) -> Optional[Dict]:
        """Check for exit."""
        game_id = state['game_id']
        pos = self.positions.get(game_id)
        if pos is None:
            return None

        game_secs = state['game_seconds_elapsed']
        mins_rem = state['minutes_remaining']
        score_diff = state['score_diff']
        mom_2 = state.get('momentum_2min', state.get('run_diff_2min', 0))

        current_spread = self.get_market_spread(score_diff, mins_rem, mom_2)

        # P&L calculation
        if pos.side == Side.LONG_AWAY:
            spread_change = current_spread - pos.entry_spread
        else:
            spread_change = pos.entry_spread - current_spread

        exit_reason = None

        # Take profit (tight)
        if spread_change >= self.config.take_profit_points:
            exit_reason = "take_profit"

        # Stop loss (wide)
        elif spread_change <= -self.config.stop_loss_points:
            exit_reason = "stop_loss"

        # Time stop
        elif game_secs - pos.entry_time >= self.config.time_stop_seconds:
            # Exit at current P&L
            exit_reason = "time_stop"

        # Game ending
        elif mins_rem < 2:
            exit_reason = "game_end"

        if exit_reason is None:
            return None

        # Record trade
        trade = Trade(
            game_id=game_id,
            entry_time=pos.entry_time,
            exit_time=game_secs,
            entry_spread=pos.entry_spread,
            exit_spread=current_spread,
            side=pos.side,
            pnl=spread_change,
            exit_reason=exit_reason,
            hold_time=game_secs - pos.entry_time
        )
        self.trades.append(trade)

        del self.positions[game_id]
        self.last_entry_time[game_id] = game_secs
        self.trade_count[game_id] = self.trade_count.get(game_id, 0) + 1

        return {'action': 'exit', 'reason': exit_reason, 'pnl': spread_change}

    def process(self, state: pd.Series) -> Optional[Dict]:
        """Process a game state."""
        game_id = state['game_id']
        if game_id in self.positions:
            return self.check_exit_conditions(state)
        else:
            return self.check_entry_conditions(state)


def generate_test_data(n_games: int = 3000, seed: int = 123) -> pd.DataFrame:
    """Generate test game data."""
    np.random.seed(seed)
    all_states = []

    for g in range(n_games):
        game_id = f"G{g:05d}"
        home_score, away_score = 0, 0
        recent_h, recent_a = [], []

        for secs in range(60, 48*60+1, 15):
            period = min(secs // (12*60) + 1, 4)
            mins_rem = max((48*60 - secs) / 60, 0)

            # Score evolution
            h_pts = np.random.poisson(0.55)
            a_pts = np.random.poisson(0.55)
            home_score += h_pts
            away_score += a_pts

            recent_h.append(h_pts)
            recent_a.append(a_pts)
            if len(recent_h) > 20:
                recent_h.pop(0)
                recent_a.pop(0)

            mom_2 = sum(recent_h[-8:]) - sum(recent_a[-8:])
            mom_5 = sum(recent_h) - sum(recent_a)

            all_states.append({
                'game_id': game_id,
                'game_seconds_elapsed': secs,
                'minutes_remaining': mins_rem,
                'score_diff': home_score - away_score,
                'momentum_2min': mom_2,
                'momentum_5min': mom_5,
            })

    return pd.DataFrame(all_states)


def backtest_high_wr_strategy(config: HighWRConfig = None,
                               n_games: int = 3000) -> Dict:
    """Run backtest with high win rate config."""
    if config is None:
        config = HighWRConfig()

    print("Generating test data...")
    df = generate_test_data(n_games=n_games)
    print(f"Generated {len(df):,} states from {n_games:,} games")

    strategy = HighWinRateSpreadStrategy(config)

    print("Running backtest...")
    for game_id, gdf in df.groupby('game_id'):
        for _, state in gdf.iterrows():
            strategy.process(state)

    trades = strategy.trades
    if not trades:
        return {'error': 'No trades'}

    # Calculate metrics
    total = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    win_rate = wins / total

    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / total

    winners = [t.pnl for t in trades if t.pnl > 0]
    losers = [t.pnl for t in trades if t.pnl <= 0]
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0

    tp_exits = sum(1 for t in trades if t.exit_reason == 'take_profit')
    sl_exits = sum(1 for t in trades if t.exit_reason == 'stop_loss')
    ts_exits = sum(1 for t in trades if t.exit_reason == 'time_stop')

    unique_games = len(set(t.game_id for t in trades))
    tpg = total / unique_games if unique_games > 0 else 0

    avg_hold = np.mean([t.hold_time for t in trades])

    gross_win = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    return {
        'total_trades': total,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': pf,
        'trades_per_game': tpg,
        'avg_hold_time': avg_hold,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'ts_exits': ts_exits,
        'games_with_trades': unique_games,
    }


def optimize_for_high_winrate():
    """Find parameters that maximize win rate while staying profitable."""
    print("="*70)
    print("OPTIMIZING FOR HIGH WIN RATE")
    print("="*70)

    results = []

    # The key: smaller TP relative to SL = higher win rate
    for tp in [0.5, 0.75, 1.0, 1.25]:
        for sl in [3.0, 3.5, 4.0, 5.0, 6.0]:
            for min_dev in [2.5, 3.0, 3.5, 4.0]:
                for min_mom in [5, 6, 7, 8]:
                    if tp >= sl * 0.4:  # TP should be much smaller than SL
                        continue

                    config = HighWRConfig(
                        take_profit_points=tp,
                        stop_loss_points=sl,
                        min_spread_deviation=min_dev,
                        min_momentum_magnitude=min_mom
                    )

                    res = backtest_high_wr_strategy(config, n_games=2000)

                    if res.get('total_trades', 0) < 500:
                        continue

                    results.append({
                        'tp': tp,
                        'sl': sl,
                        'min_dev': min_dev,
                        'min_mom': min_mom,
                        **res
                    })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('win_rate', ascending=False)

    print("\nTop 15 by Win Rate:")
    print(results_df.head(15)[['tp', 'sl', 'min_dev', 'min_mom',
                                'total_trades', 'win_rate', 'profit_factor',
                                'trades_per_game']].to_string(index=False))

    # Also show best profitable ones
    profitable = results_df[results_df['profit_factor'] > 1.0]
    print("\n\nTop 10 Profitable by Win Rate:")
    print(profitable.head(10)[['tp', 'sl', 'min_dev', 'min_mom',
                                'total_trades', 'win_rate', 'profit_factor',
                                'total_pnl']].to_string(index=False))

    return results_df


def main():
    print("="*70)
    print("HIGH WIN RATE SPREAD TRADING STRATEGY")
    print("="*70)

    # Run with aggressive high-WR settings
    config = HighWRConfig(
        take_profit_points=0.5,  # Very tight TP
        stop_loss_points=5.0,    # Wide SL
        min_spread_deviation=3.5,
        min_momentum_magnitude=7,
        time_stop_seconds=120,
    )

    print(f"\nConfiguration:")
    print(f"  Take Profit: {config.take_profit_points} points")
    print(f"  Stop Loss: {config.stop_loss_points} points")
    print(f"  Min Deviation: {config.min_spread_deviation} points")
    print(f"  Min Momentum: {config.min_momentum_magnitude}")

    result = backtest_high_wr_strategy(config, n_games=5000)

    print(f"\n{'='*50}")
    print("RESULTS")
    print('='*50)
    print(f"Total Trades:     {result['total_trades']:,}")
    print(f"WIN RATE:         {result['win_rate']:.1%}")
    print(f"Trades per Game:  {result['trades_per_game']:.1f}")
    print(f"Profit Factor:    {result['profit_factor']:.2f}")
    print(f"Total P&L:        {result['total_pnl']:+.1f} points")
    print(f"Avg Hold Time:    {result['avg_hold_time']:.0f} sec")
    print(f"\nExit Breakdown:")
    print(f"  Take Profit: {result['tp_exits']} ({result['tp_exits']/result['total_trades']*100:.0f}%)")
    print(f"  Stop Loss:   {result['sl_exits']} ({result['sl_exits']/result['total_trades']*100:.0f}%)")
    print(f"  Time Stop:   {result['ts_exits']} ({result['ts_exits']/result['total_trades']*100:.0f}%)")

    # Run optimization
    print("\n")
    opt_results = optimize_for_high_winrate()

    # Show the absolute best
    best = opt_results.iloc[0]
    print(f"\n{'='*70}")
    print("BEST HIGH WIN RATE CONFIGURATION")
    print('='*70)
    print(f"""
Parameters:
  Take Profit:       {best['tp']} points
  Stop Loss:         {best['sl']} points
  Min Deviation:     {best['min_dev']} points
  Min Momentum:      {best['min_mom']} points

Results:
  WIN RATE:          {best['win_rate']:.1%}
  Total Trades:      {int(best['total_trades']):,}
  Trades per Game:   {best['trades_per_game']:.1f}
  Profit Factor:     {best['profit_factor']:.2f}
  Total P&L:         {best['total_pnl']:+.1f} points
""")


if __name__ == "__main__":
    main()
