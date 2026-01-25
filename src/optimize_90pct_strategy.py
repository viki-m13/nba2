"""
Optimize for 90%+ Win Rate Strategy

Based on research findings:
- spread_dev_3.0+ gives 90%+ win rate
- Tighter TP, wider SL increases win rate
- Trade frequency can be increased with multiple signals

Goal: Find strategy with 90%+ win rate AND trades every game
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


def generate_game(game_id: str, seed: int = None) -> pd.DataFrame:
    """Generate single game with realistic dynamics."""
    if seed:
        np.random.seed(seed)

    states = []
    home_edge = np.random.normal(2, 6)
    volatility = np.random.choice([0.8, 1.0, 1.2], p=[0.25, 0.5, 0.25])

    home_score, away_score = 0, 0
    h_hist, a_hist = deque(maxlen=40), deque(maxlen=40)
    last_spread = -home_edge / 2

    for secs in range(0, 48*60+1, 15):
        mins_rem = max((48*60 - secs) / 60, 0)

        # Scoring
        h_pts = np.random.poisson(0.55 * volatility * (1 + home_edge/100))
        a_pts = np.random.poisson(0.55 * volatility * (1 - home_edge/100))
        home_score += h_pts
        away_score += a_pts
        h_hist.append(h_pts)
        a_hist.append(a_pts)

        # Momentum
        mom_2 = sum(list(h_hist)[-8:]) - sum(list(a_hist)[-8:])
        mom_5 = sum(list(h_hist)[-20:]) - sum(list(a_hist)[-20:])

        score_diff = home_score - away_score

        # Spreads
        hca_remaining = 3.0 * (mins_rem / 48)
        fair_spread = -score_diff - hca_remaining
        overreaction = mom_2 * 0.25 * np.random.uniform(0.8, 1.2)
        market_spread = fair_spread - overreaction + np.random.normal(0, 0.2)

        spread_change = market_spread - last_spread
        last_spread = market_spread

        states.append({
            'game_id': game_id,
            'secs': secs,
            'mins_rem': mins_rem,
            'score_diff': score_diff,
            'mom_2': mom_2,
            'mom_5': mom_5,
            'market_spread': market_spread,
            'fair_spread': fair_spread,
            'deviation': market_spread - fair_spread,
            'spread_change': spread_change,
        })

    return pd.DataFrame(states)


def generate_dataset(n_games: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate dataset."""
    np.random.seed(seed)
    return pd.concat([generate_game(f"G{i:05d}", seed+i) for i in range(n_games)], ignore_index=True)


@dataclass
class Trade:
    game_id: str
    entry_secs: float
    exit_secs: float
    side: str
    pnl: float
    reason: str


class Strategy:
    """High win rate spread trading strategy."""

    def __init__(self,
                 deviation_threshold: float = 2.5,
                 momentum_threshold: float = 5,
                 take_profit: float = 0.4,
                 stop_loss: float = 5.0,
                 time_stop: float = 90,
                 min_mins_remaining: float = 4,
                 max_mins_remaining: float = 44,
                 cooldown: float = 20,
                 use_momentum_signal: bool = True,
                 use_deviation_signal: bool = True):

        self.deviation_threshold = deviation_threshold
        self.momentum_threshold = momentum_threshold
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.time_stop = time_stop
        self.min_mins = min_mins_remaining
        self.max_mins = max_mins_remaining
        self.cooldown = cooldown
        self.use_momentum = use_momentum_signal
        self.use_deviation = use_deviation_signal

        self.positions = {}
        self.trades = []
        self.last_entry = {}
        self.trade_count = {}

    def check_signal(self, state: pd.Series) -> Optional[Tuple[str, str]]:
        """Check for entry signal. Returns (side, signal_type) or None."""
        signals = []

        # Deviation signal
        if self.use_deviation:
            dev = state['deviation']
            if abs(dev) >= self.deviation_threshold:
                side = 'away' if dev < 0 else 'home'
                signals.append((side, 'deviation'))

        # Momentum signal
        if self.use_momentum:
            mom = state['mom_2']
            if abs(mom) >= self.momentum_threshold:
                side = 'away' if mom > 0 else 'home'
                signals.append((side, 'momentum'))

        # If multiple signals agree, stronger conviction
        if len(signals) >= 1:
            return signals[0]

        return None

    def process(self, state: pd.Series) -> Optional[str]:
        """Process state, return action taken."""
        gid = state['game_id']
        secs = state['secs']
        mins = state['mins_rem']
        spread = state['market_spread']

        # Check exit for existing position
        if gid in self.positions:
            pos = self.positions[gid]
            entry_spread, entry_secs, side = pos

            if side == 'away':
                pnl = spread - entry_spread
            else:
                pnl = entry_spread - spread

            reason = None
            if pnl >= self.take_profit:
                reason = 'tp'
            elif pnl <= -self.stop_loss:
                reason = 'sl'
            elif secs - entry_secs >= self.time_stop:
                reason = 'time'
            elif mins < 1:
                reason = 'end'

            if reason:
                self.trades.append(Trade(gid, entry_secs, secs, side, pnl, reason))
                del self.positions[gid]
                return f"EXIT_{reason}"

            return None

        # Check for new entry
        if mins < self.min_mins or mins > self.max_mins:
            return None

        if abs(state['score_diff']) > 20:
            return None

        if secs - self.last_entry.get(gid, -999) < self.cooldown:
            return None

        signal = self.check_signal(state)
        if signal is None:
            return None

        side, sig_type = signal
        self.positions[gid] = (spread, secs, side)
        self.last_entry[gid] = secs
        self.trade_count[gid] = self.trade_count.get(gid, 0) + 1

        return f"ENTER_{side}_{sig_type}"


def run_backtest(df: pd.DataFrame, **params) -> Dict:
    """Run backtest with given parameters."""
    strategy = Strategy(**params)

    for gid, gdf in df.groupby('game_id'):
        for _, state in gdf.sort_values('secs').iterrows():
            strategy.process(state)

        # Close any open position at game end
        if gid in strategy.positions:
            final = gdf.iloc[-1]
            pos = strategy.positions[gid]
            entry_spread, entry_secs, side = pos
            if side == 'away':
                pnl = final['market_spread'] - entry_spread
            else:
                pnl = entry_spread - final['market_spread']
            strategy.trades.append(Trade(gid, entry_secs, final['secs'], side, pnl, 'end'))
            del strategy.positions[gid]

    trades = strategy.trades
    if not trades:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0}

    total = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    pnls = [t.pnl for t in trades]

    games_with_trades = len(set(t.game_id for t in trades))
    n_games = df['game_id'].nunique()

    return {
        'trades': total,
        'wins': wins,
        'win_rate': wins / total,
        'pnl': sum(pnls),
        'avg_pnl': np.mean(pnls),
        'sharpe': np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0,
        'games_traded': games_with_trades,
        'game_coverage': games_with_trades / n_games,
        'trades_per_game': total / games_with_trades if games_with_trades > 0 else 0,
        'tp_pct': sum(1 for t in trades if t.reason == 'tp') / total,
        'sl_pct': sum(1 for t in trades if t.reason == 'sl') / total,
    }


def grid_search(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive grid search."""
    results = []

    # Parameter ranges
    dev_thresholds = [2.0, 2.5, 3.0, 3.5]
    mom_thresholds = [4, 5, 6, 7]
    take_profits = [0.3, 0.4, 0.5, 0.6, 0.75]
    stop_losses = [3.0, 4.0, 5.0, 6.0]
    time_stops = [60, 90, 120]
    cooldowns = [15, 20, 30]

    total = len(dev_thresholds) * len(mom_thresholds) * len(take_profits) * len(stop_losses) * len(time_stops) * len(cooldowns)
    print(f"Testing {total} combinations...")

    count = 0
    for dev in dev_thresholds:
        for mom in mom_thresholds:
            for tp in take_profits:
                for sl in stop_losses:
                    if tp >= sl * 0.5:  # TP should be much smaller than SL
                        continue
                    for ts in time_stops:
                        for cd in cooldowns:
                            count += 1
                            if count % 100 == 0:
                                print(f"  {count}/{total}...")

                            res = run_backtest(
                                df,
                                deviation_threshold=dev,
                                momentum_threshold=mom,
                                take_profit=tp,
                                stop_loss=sl,
                                time_stop=ts,
                                cooldown=cd,
                            )

                            if res['trades'] < 100:
                                continue

                            results.append({
                                'dev_thresh': dev,
                                'mom_thresh': mom,
                                'tp': tp,
                                'sl': sl,
                                'time_stop': ts,
                                'cooldown': cd,
                                **res
                            })

    return pd.DataFrame(results)


def main():
    print("="*70)
    print("OPTIMIZING FOR 90%+ WIN RATE + HIGH TRADE FREQUENCY")
    print("="*70)

    # Generate data
    print("\nGenerating 5000 games...")
    df = generate_dataset(n_games=5000)
    n_games = df['game_id'].nunique()
    print(f"Generated {len(df):,} states from {n_games} games")

    # Quick test with promising parameters
    print("\n" + "-"*60)
    print("QUICK TESTS - Promising Parameter Sets")
    print("-"*60)

    test_configs = [
        # (dev, mom, tp, sl, time, cooldown, description)
        (2.5, 5, 0.4, 5.0, 90, 20, "Balanced"),
        (2.0, 4, 0.35, 5.0, 60, 15, "Aggressive entry"),
        (3.0, 6, 0.5, 4.0, 120, 30, "Selective"),
        (2.0, 5, 0.3, 6.0, 60, 15, "Ultra tight TP"),
        (2.5, 4, 0.4, 5.0, 90, 15, "Lower cooldown"),
    ]

    for dev, mom, tp, sl, ts, cd, desc in test_configs:
        res = run_backtest(df, deviation_threshold=dev, momentum_threshold=mom,
                          take_profit=tp, stop_loss=sl, time_stop=ts, cooldown=cd)
        print(f"{desc}: WR={res['win_rate']:.1%}, Trades={res['trades']}, "
              f"Coverage={res['game_coverage']:.0%}, TPG={res['trades_per_game']:.1f}")

    # Full grid search
    print("\n" + "-"*60)
    print("FULL GRID SEARCH")
    print("-"*60)

    results = grid_search(df)

    # Analyze results
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)

    # Filter for 90%+ win rate
    high_wr = results[results['win_rate'] >= 0.90]
    print(f"\nStrategies with 90%+ win rate: {len(high_wr)}")

    if len(high_wr) > 0:
        # Sort by game coverage (trades almost every game)
        high_wr_sorted = high_wr.sort_values('game_coverage', ascending=False)

        print("\nTop 10 by Game Coverage (with 90%+ WR):")
        print(high_wr_sorted.head(10)[['dev_thresh', 'mom_thresh', 'tp', 'sl',
                                        'win_rate', 'trades', 'game_coverage',
                                        'trades_per_game', 'pnl']].to_string(index=False))

        # Best balanced
        best = high_wr_sorted.iloc[0]

        print(f"\n{'='*70}")
        print("BEST STRATEGY (90%+ WR with highest coverage)")
        print("="*70)
        print(f"""
ENTRY RULES:
  - Spread Deviation >= {best['dev_thresh']} points from fair value
  - OR Momentum >= {best['mom_thresh']} points in 2 minutes
  - Time: 4-44 minutes remaining
  - Score diff <= 20 points
  - Cooldown: {best['cooldown']} seconds between trades

EXIT RULES:
  - Take Profit: {best['tp']} points
  - Stop Loss: {best['sl']} points
  - Time Stop: {best['time_stop']} seconds

RESULTS:
  - Win Rate: {best['win_rate']:.1%}
  - Total Trades: {int(best['trades']):,}
  - Games Traded: {best['game_coverage']:.0%} of all games
  - Trades per Game: {best['trades_per_game']:.1f}
  - Total P&L: {best['pnl']:+.0f} points
  - Take Profit %: {best['tp_pct']:.0%}
  - Stop Loss %: {best['sl_pct']:.0%}
""")

    # Also show highest win rate strategies
    print("\n" + "-"*60)
    print("TOP 10 BY WIN RATE (regardless of coverage)")
    print("-"*60)
    top_wr = results.sort_values('win_rate', ascending=False).head(10)
    print(top_wr[['dev_thresh', 'mom_thresh', 'tp', 'sl',
                   'win_rate', 'trades', 'game_coverage', 'pnl']].to_string(index=False))

    # Highest coverage strategies
    print("\n" + "-"*60)
    print("TOP 10 BY GAME COVERAGE (with WR >= 85%)")
    print("-"*60)
    high_coverage = results[results['win_rate'] >= 0.85].sort_values('game_coverage', ascending=False).head(10)
    if len(high_coverage) > 0:
        print(high_coverage[['dev_thresh', 'mom_thresh', 'tp', 'sl',
                             'win_rate', 'trades', 'game_coverage', 'trades_per_game']].to_string(index=False))

    # Save results
    results.to_csv('/home/user/nba2/output/optimization_results.csv', index=False)
    print(f"\nSaved {len(results)} results to output/optimization_results.csv")

    return results


if __name__ == "__main__":
    results = main()
