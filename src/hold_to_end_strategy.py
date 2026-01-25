"""
Hold-to-End Spread Strategy

Enter when spread is mispriced (momentum overreaction), but hold until
game end instead of taking profit/stop loss.

The hypothesis: if we only enter when spreads have significantly
overreacted, the final score should favor our position more often than not.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    LONG_HOME = "long_home"  # Bet home covers
    LONG_AWAY = "long_away"  # Bet away covers


@dataclass
class HoldToEndConfig:
    """Config for hold-to-end strategy."""
    # Entry conditions only (no exit rules needed)
    min_spread_deviation: float = 3.0
    min_momentum_magnitude: float = 6
    min_minutes_remaining: float = 8.0  # Enter earlier since we're holding
    max_minutes_remaining: float = 40.0
    max_score_diff: float = 20.0  # Skip blowouts

    # Only enter once per game
    one_trade_per_game: bool = True


@dataclass
class HoldToEndTrade:
    """A completed hold-to-end trade."""
    game_id: str
    entry_time: float
    entry_spread: float  # Our entry spread
    entry_fair_spread: float
    side: Side
    final_margin: float  # Actual final home - away
    spread_result: float  # Points we won/lost by
    won: bool


class HoldToEndStrategy:
    """
    Strategy that enters on mispricing but holds to game end.

    Entry: When market spread deviates from fair by threshold
    Exit: Game end - compare final margin to entry spread
    """

    HOME_COURT_ADVANTAGE = 3.0
    MOMENTUM_OVERREACTION = 0.5

    def __init__(self, config: HoldToEndConfig = None):
        self.config = config or HoldToEndConfig()
        self.pending_trades: Dict[str, dict] = {}  # Trades waiting for game end
        self.completed_trades: List[HoldToEndTrade] = []
        self.traded_games: set = set()

    def get_fair_spread(self, score_diff: float, mins_remaining: float,
                        momentum_2min: float, momentum_5min: float) -> float:
        """Calculate fair spread."""
        fair = -score_diff
        fair -= self.HOME_COURT_ADVANTAGE * (mins_remaining / 48)
        fair += momentum_2min * 0.3 * self.MOMENTUM_OVERREACTION
        return fair

    def get_market_spread(self, score_diff: float, mins_remaining: float,
                          momentum_2min: float) -> float:
        """Estimate market spread."""
        market = -score_diff
        market -= self.HOME_COURT_ADVANTAGE * (mins_remaining / 48)
        market -= momentum_2min * 0.25
        return market

    def process_state(self, state: pd.Series) -> Optional[str]:
        """Process a game state. Returns action taken."""
        game_id = state['game_id']
        mins_rem = state['minutes_remaining']

        # Check for game end
        if mins_rem <= 0 or state['game_seconds_elapsed'] >= 48*60 - 15:
            return self._settle_game(game_id, state)

        # Already have a trade for this game?
        if game_id in self.pending_trades:
            return None

        if self.config.one_trade_per_game and game_id in self.traded_games:
            return None

        # Check entry
        return self._check_entry(state)

    def _check_entry(self, state: pd.Series) -> Optional[str]:
        """Check for entry signal."""
        game_id = state['game_id']
        mins_rem = state['minutes_remaining']
        score_diff = state['score_diff']
        mom_2 = state.get('momentum_2min', state.get('run_diff_2min', 0))
        mom_5 = state.get('momentum_5min', state.get('run_diff_5min', 0))

        # Filters
        if mins_rem < self.config.min_minutes_remaining:
            return None
        if mins_rem > self.config.max_minutes_remaining:
            return None
        if abs(score_diff) > self.config.max_score_diff:
            return None
        if abs(mom_2) < self.config.min_momentum_magnitude:
            return None

        # Calculate spreads
        fair = self.get_fair_spread(score_diff, mins_rem, mom_2, mom_5)
        market = self.get_market_spread(score_diff, mins_rem, mom_2)
        deviation = market - fair

        if abs(deviation) < self.config.min_spread_deviation:
            return None

        # Determine side
        if deviation < 0:
            side = Side.LONG_AWAY
        else:
            side = Side.LONG_HOME

        # Record pending trade
        self.pending_trades[game_id] = {
            'entry_time': state['game_seconds_elapsed'],
            'entry_spread': market,
            'entry_fair': fair,
            'side': side,
            'deviation': deviation,
        }
        self.traded_games.add(game_id)

        return f"ENTER {side.value}"

    def _settle_game(self, game_id: str, final_state: pd.Series) -> Optional[str]:
        """Settle a trade at game end."""
        if game_id not in self.pending_trades:
            return None

        trade_info = self.pending_trades.pop(game_id)

        # Final margin (home - away)
        final_margin = final_state['score_diff']

        # Did we win?
        entry_spread = trade_info['entry_spread']
        side = trade_info['side']

        if side == Side.LONG_AWAY:
            # We bet away covers. Entry spread was (e.g.) Home -10
            # Away covers if final margin < 10 (home wins by less than spread)
            # In spread terms: we entered at -10, final is -final_margin
            # We win if final_margin < -entry_spread (since entry_spread is negative for home favorite)
            # Actually: entry_spread = -10 means home -10
            # Final spread at game end = -final_margin
            # We went LONG AWAY, meaning we bought at +10 (or sold home at -10)
            # We win if home wins by LESS than 10, or away wins
            # Math: we win if final_margin < -entry_spread
            # If entry_spread = -10, we win if final_margin < 10
            spread_result = -entry_spread - final_margin  # How much we won by
            won = spread_result > 0
        else:
            # LONG_HOME: we bet home covers
            # Entry spread was (e.g.) Home -3 (entry_spread = -3)
            # We win if home wins by MORE than 3
            # Math: we win if final_margin > -entry_spread
            spread_result = final_margin - (-entry_spread)  # How much home beat spread by
            won = spread_result > 0

        trade = HoldToEndTrade(
            game_id=game_id,
            entry_time=trade_info['entry_time'],
            entry_spread=entry_spread,
            entry_fair_spread=trade_info['entry_fair'],
            side=side,
            final_margin=final_margin,
            spread_result=spread_result,
            won=won
        )
        self.completed_trades.append(trade)

        return f"SETTLED: {'WIN' if won else 'LOSS'} by {spread_result:.1f}"


def generate_games_with_outcomes(n_games: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate game data including final outcomes."""
    np.random.seed(seed)
    all_states = []

    for g in range(n_games):
        game_id = f"G{g:05d}"

        # Team skill differential (affects final outcome)
        home_skill = np.random.normal(0, 8)  # Home team strength

        home_score, away_score = 0, 0
        recent_h, recent_a = [], []

        for secs in range(60, 48*60 + 1, 15):
            mins_rem = max((48*60 - secs) / 60, 0)

            # Scoring with skill influence
            base_rate = 0.55
            h_pts = np.random.poisson(base_rate * (1 + home_skill/100))
            a_pts = np.random.poisson(base_rate * (1 - home_skill/100))

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
                'home_score': home_score,
                'away_score': away_score,
                'momentum_2min': mom_2,
                'momentum_5min': mom_5,
            })

    return pd.DataFrame(all_states)


def run_hold_to_end_backtest(config: HoldToEndConfig = None,
                              n_games: int = 5000) -> Dict:
    """Run backtest of hold-to-end strategy."""
    if config is None:
        config = HoldToEndConfig()

    print(f"Generating {n_games} games...")
    df = generate_games_with_outcomes(n_games=n_games)

    strategy = HoldToEndStrategy(config)

    print("Running backtest...")
    for game_id, gdf in df.groupby('game_id'):
        gdf = gdf.sort_values('game_seconds_elapsed')
        for _, state in gdf.iterrows():
            strategy.process_state(state)

        # Force settle at last state
        if game_id in strategy.pending_trades:
            strategy._settle_game(game_id, gdf.iloc[-1])

    trades = strategy.completed_trades

    if not trades:
        return {'error': 'No trades'}

    # Calculate metrics
    total = len(trades)
    wins = sum(1 for t in trades if t.won)
    losses = total - wins
    win_rate = wins / total

    # P&L in spread points (assuming 1 unit per trade)
    # Standard spread bet: risk 1.1 to win 1 (with -110 odds)
    # Simplified: +1 for win, -1 for loss
    total_pnl = sum(1 if t.won else -1 for t in trades)

    # Average margin of victory/defeat
    win_margins = [t.spread_result for t in trades if t.won]
    loss_margins = [t.spread_result for t in trades if not t.won]

    avg_win_margin = np.mean(win_margins) if win_margins else 0
    avg_loss_margin = np.mean(loss_margins) if loss_margins else 0

    # Games with trades
    unique_games = len(set(t.game_id for t in trades))

    return {
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win_margin': avg_win_margin,
        'avg_loss_margin': avg_loss_margin,
        'games_traded': unique_games,
        'trade_pct': unique_games / n_games,
        'trades': trades,
    }


def optimize_hold_to_end():
    """Find best parameters for hold-to-end strategy."""
    print("="*60)
    print("OPTIMIZING HOLD-TO-END STRATEGY")
    print("="*60)

    results = []

    for min_dev in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for min_mom in [4, 5, 6, 7, 8, 10]:
            for min_mins in [6, 8, 10, 12, 15]:
                config = HoldToEndConfig(
                    min_spread_deviation=min_dev,
                    min_momentum_magnitude=min_mom,
                    min_minutes_remaining=min_mins,
                )

                res = run_hold_to_end_backtest(config, n_games=3000)

                if res.get('total_trades', 0) < 100:
                    continue

                results.append({
                    'min_dev': min_dev,
                    'min_mom': min_mom,
                    'min_mins': min_mins,
                    'trades': res['total_trades'],
                    'win_rate': res['win_rate'],
                    'pnl': res['total_pnl'],
                    'trade_pct': res['trade_pct'],
                })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('win_rate', ascending=False)

    print("\nTop 15 by Win Rate:")
    print(results_df.head(15).to_string(index=False))

    print("\n\nTop 10 by P&L:")
    print(results_df.sort_values('pnl', ascending=False).head(10).to_string(index=False))

    return results_df


def main():
    print("="*70)
    print("HOLD-TO-END SPREAD STRATEGY")
    print("(Enter on mispricing, hold until game ends)")
    print("="*70)

    # Default config
    config = HoldToEndConfig(
        min_spread_deviation=3.0,
        min_momentum_magnitude=6,
        min_minutes_remaining=8.0,
    )

    print(f"\nConfiguration:")
    print(f"  Min Spread Deviation: {config.min_spread_deviation}")
    print(f"  Min Momentum: {config.min_momentum_magnitude}")
    print(f"  Min Minutes Remaining: {config.min_minutes_remaining}")
    print(f"  One Trade Per Game: {config.one_trade_per_game}")

    result = run_hold_to_end_backtest(config, n_games=5000)

    print(f"\n{'='*50}")
    print("RESULTS (Hold to Game End)")
    print('='*50)
    print(f"Total Trades:      {result['total_trades']:,}")
    print(f"Wins:              {result['wins']:,}")
    print(f"Losses:            {result['losses']:,}")
    print(f"WIN RATE:          {result['win_rate']:.1%}")
    print(f"Total P&L:         {result['total_pnl']:+} units")
    print(f"Games Traded:      {result['games_traded']:,} / 5000 ({result['trade_pct']:.1%})")
    print(f"Avg Win Margin:    {result['avg_win_margin']:+.1f} points")
    print(f"Avg Loss Margin:   {result['avg_loss_margin']:+.1f} points")

    # Run optimization
    print("\n")
    opt_df = optimize_hold_to_end()

    # Best config
    best = opt_df.iloc[0]
    print(f"\n{'='*70}")
    print("BEST HOLD-TO-END CONFIGURATION")
    print('='*70)
    print(f"""
Parameters:
  Min Spread Deviation:  {best['min_dev']} points
  Min Momentum:          {best['min_mom']} points
  Min Minutes Remaining: {best['min_mins']} minutes

Results:
  WIN RATE:        {best['win_rate']:.1%}
  Total Trades:    {int(best['trades']):,}
  Total P&L:       {best['pnl']:+.0f} units
  Games Traded:    {best['trade_pct']:.1%} of games
""")


if __name__ == "__main__":
    main()
