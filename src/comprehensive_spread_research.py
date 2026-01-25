"""
Comprehensive Spread Trading Research

Think like a professional quant trader:
1. Market microstructure - how does the spread actually move?
2. Multiple signal types - not just momentum
3. Advanced exit strategies - trailing stops, scaling, time decay
4. Regime filters - volatility, game phase, score context
5. Pattern recognition - specific sequences
6. Ensemble approaches - combine multiple edges
7. Risk management - drawdown control, position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA GENERATION - More realistic spread dynamics
# =============================================================================

def generate_realistic_game(game_id: str, seed: int = None) -> pd.DataFrame:
    """
    Generate a single game with realistic spread dynamics.

    Models:
    - Scoring as Poisson process with momentum clustering
    - Spread overreaction and reversion
    - Different volatility regimes
    - Clutch time behavior changes
    """
    if seed:
        np.random.seed(seed)

    states = []

    # Game characteristics
    home_edge = np.random.normal(2, 6)  # Home team quality edge
    game_volatility = np.random.choice([0.7, 1.0, 1.3], p=[0.2, 0.5, 0.3])
    pace_factor = np.random.uniform(0.9, 1.1)

    home_score, away_score = 0, 0
    home_scoring_history = deque(maxlen=40)  # Last 10 min of scoring
    away_scoring_history = deque(maxlen=40)

    # Track "true" spread vs market spread
    last_market_spread = -home_edge / 2  # Initial spread

    for secs in range(0, 48*60 + 1, 15):
        period = min(secs // (12*60) + 1, 4)
        clock_secs = 12*60 - (secs % (12*60))
        mins_remaining = max((48*60 - secs) / 60, 0)
        mins_elapsed = secs / 60

        # Scoring rates vary by game phase
        if period == 1:
            phase_mult = 0.95  # Slower start
        elif period == 4 and mins_remaining < 3:
            phase_mult = 1.15  # Clutch time intensity
        else:
            phase_mult = 1.0

        # Base scoring rate
        base_rate = 0.55 * pace_factor * game_volatility * phase_mult

        # Momentum effect - hot hand clustering
        recent_home = sum(home_scoring_history) if home_scoring_history else 0
        recent_away = sum(away_scoring_history) if away_scoring_history else 0
        momentum = (recent_home - recent_away) / max(len(home_scoring_history), 1)

        # Scoring with skill + momentum + randomness
        home_rate = base_rate * (1 + home_edge/100 + momentum * 0.02)
        away_rate = base_rate * (1 - home_edge/100 - momentum * 0.02)

        h_pts = np.random.poisson(max(home_rate, 0.1))
        a_pts = np.random.poisson(max(away_rate, 0.1))

        home_score += h_pts
        away_score += a_pts

        home_scoring_history.append(h_pts)
        away_scoring_history.append(a_pts)

        # Calculate various momentum metrics
        mom_1min = sum(list(home_scoring_history)[-4:]) - sum(list(away_scoring_history)[-4:])
        mom_2min = sum(list(home_scoring_history)[-8:]) - sum(list(away_scoring_history)[-8:])
        mom_3min = sum(list(home_scoring_history)[-12:]) - sum(list(away_scoring_history)[-12:])
        mom_5min = sum(list(home_scoring_history)[-20:]) - sum(list(away_scoring_history)[-20:])

        score_diff = home_score - away_score

        # Market spread dynamics
        # True fair spread based on score and remaining HCA
        remaining_hca = 3.0 * (mins_remaining / 48)
        true_fair_spread = -score_diff - remaining_hca

        # Market overreacts to recent momentum
        overreaction = mom_2min * 0.25 * np.random.uniform(0.8, 1.2)
        market_spread = true_fair_spread - overreaction

        # Add some noise/bid-ask simulation
        spread_noise = np.random.normal(0, 0.3)
        market_spread += spread_noise

        # Spread change from last tick
        spread_change = market_spread - last_market_spread
        last_market_spread = market_spread

        # Volatility of recent spread changes

        states.append({
            'game_id': game_id,
            'seconds': secs,
            'period': period,
            'clock_secs': clock_secs,
            'mins_remaining': mins_remaining,
            'mins_elapsed': mins_elapsed,
            'home_score': home_score,
            'away_score': away_score,
            'score_diff': score_diff,
            'abs_diff': abs(score_diff),
            'total_score': home_score + away_score,
            'mom_1min': mom_1min,
            'mom_2min': mom_2min,
            'mom_3min': mom_3min,
            'mom_5min': mom_5min,
            'market_spread': market_spread,
            'fair_spread': true_fair_spread,
            'spread_deviation': market_spread - true_fair_spread,
            'spread_change': spread_change,
            'game_volatility': game_volatility,
            'pace_factor': pace_factor,
        })

    return pd.DataFrame(states)


def generate_dataset(n_games: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate full dataset."""
    np.random.seed(seed)
    all_games = []
    for i in range(n_games):
        game = generate_realistic_game(f"G{i:05d}", seed=seed+i)
        all_games.append(game)
    return pd.concat(all_games, ignore_index=True)


# =============================================================================
# SIGNAL GENERATORS - Multiple entry signal types
# =============================================================================

class SignalType(Enum):
    MOMENTUM_REVERSION = "momentum_reversion"
    SPREAD_DEVIATION = "spread_deviation"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    MEAN_REVERSION_SPREAD = "mean_reversion_spread"
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"
    QUARTER_TRANSITION = "quarter_transition"


def signal_momentum_reversion(state: pd.Series, threshold: float = 6) -> Optional[Tuple[str, float]]:
    """Classic momentum reversion - bet against big runs."""
    mom = state['mom_2min']
    if abs(mom) >= threshold:
        side = 'away' if mom > 0 else 'home'
        strength = min(abs(mom) / threshold, 2.0)
        return (side, strength)
    return None


def signal_spread_deviation(state: pd.Series, threshold: float = 2.5) -> Optional[Tuple[str, float]]:
    """Trade when market spread deviates from fair."""
    dev = state['spread_deviation']
    if abs(dev) >= threshold:
        side = 'away' if dev < 0 else 'home'
        strength = min(abs(dev) / threshold, 2.0)
        return (side, strength)
    return None


def signal_momentum_exhaustion(state: pd.Series,
                                short_thresh: float = 5,
                                long_thresh: float = 2) -> Optional[Tuple[str, float]]:
    """
    Momentum exhaustion - short-term momentum high but medium-term fading.
    This catches the end of runs.
    """
    mom_1 = state['mom_1min']
    mom_3 = state['mom_3min']

    # Strong short-term but weaker medium-term = exhaustion
    if abs(mom_1) >= short_thresh:
        if abs(mom_3) < abs(mom_1) * 0.7:  # Medium term not keeping up
            side = 'away' if mom_1 > 0 else 'home'
            strength = abs(mom_1) / short_thresh
            return (side, strength)
    return None


def signal_spread_mean_reversion(states_history: List[pd.Series],
                                  lookback: int = 8,
                                  threshold: float = 1.5) -> Optional[Tuple[str, float]]:
    """
    Mean reversion on spread itself - if spread moved too much recently,
    expect pullback.
    """
    if len(states_history) < lookback:
        return None

    recent_changes = [s['spread_change'] for s in states_history[-lookback:]]
    cumulative_move = sum(recent_changes)

    if abs(cumulative_move) >= threshold:
        side = 'home' if cumulative_move > 0 else 'away'  # Bet against the move
        strength = min(abs(cumulative_move) / threshold, 2.0)
        return (side, strength)
    return None


def signal_quarter_start(state: pd.Series) -> Optional[Tuple[str, float]]:
    """
    Trade at quarter starts based on end-of-quarter momentum.
    Teams often come out flat after hot streaks.
    """
    clock = state['clock_secs']
    period = state['period']

    # First 90 seconds of Q2, Q3, Q4
    if period > 1 and clock >= 10.5 * 60:
        mom = state['mom_3min']
        if abs(mom) >= 6:
            side = 'away' if mom > 0 else 'home'
            return (side, 1.5)
    return None


# =============================================================================
# EXIT STRATEGIES - Multiple exit approaches
# =============================================================================

@dataclass
class Position:
    game_id: str
    entry_time: float
    entry_spread: float
    side: str
    size: float = 1.0
    entry_signal: str = ""
    max_profit: float = 0
    max_loss: float = 0


@dataclass
class ExitStrategy:
    name: str
    take_profit: float = 1.0
    stop_loss: float = 3.0
    time_stop: float = 180
    trailing_stop: Optional[float] = None  # Activate after X profit
    trailing_distance: float = 0.5
    break_even_trigger: float = 0.0  # Move stop to break even after X profit
    scale_out_levels: List[Tuple[float, float]] = field(default_factory=list)  # (profit_level, pct_to_close)


# Pre-defined exit strategies to test
EXIT_STRATEGIES = {
    'tight_tp': ExitStrategy(
        name='tight_tp',
        take_profit=0.5,
        stop_loss=4.0,
        time_stop=120,
    ),
    'medium_tp': ExitStrategy(
        name='medium_tp',
        take_profit=1.0,
        stop_loss=3.0,
        time_stop=180,
    ),
    'wide_tp': ExitStrategy(
        name='wide_tp',
        take_profit=1.5,
        stop_loss=2.5,
        time_stop=240,
    ),
    'trailing': ExitStrategy(
        name='trailing',
        take_profit=2.0,  # Higher target
        stop_loss=3.0,
        time_stop=300,
        trailing_stop=0.75,  # Activate after 0.75 profit
        trailing_distance=0.4,
    ),
    'break_even': ExitStrategy(
        name='break_even',
        take_profit=1.5,
        stop_loss=3.0,
        time_stop=240,
        break_even_trigger=0.5,  # Move stop to BE after 0.5 profit
    ),
    'scale_out': ExitStrategy(
        name='scale_out',
        take_profit=2.0,
        stop_loss=3.0,
        time_stop=300,
        scale_out_levels=[(0.5, 0.33), (1.0, 0.33)],  # Take 1/3 at 0.5, 1/3 at 1.0
    ),
    'ultra_tight': ExitStrategy(
        name='ultra_tight',
        take_profit=0.3,
        stop_loss=5.0,
        time_stop=90,
    ),
}


def check_exit(position: Position,
               current_spread: float,
               current_time: float,
               strategy: ExitStrategy) -> Optional[Tuple[str, float, float]]:
    """
    Check if position should be exited.

    Returns: (exit_reason, pnl, size_to_close) or None
    """
    # Calculate current P&L
    if position.side == 'away':
        pnl = current_spread - position.entry_spread
    else:
        pnl = position.entry_spread - current_spread

    # Update max profit/loss tracking
    position.max_profit = max(position.max_profit, pnl)
    position.max_loss = min(position.max_loss, pnl)

    hold_time = current_time - position.entry_time

    # Check exits in order of priority

    # 1. Take profit
    if pnl >= strategy.take_profit:
        return ('take_profit', pnl, position.size)

    # 2. Stop loss
    stop_level = -strategy.stop_loss

    # Adjust for break-even stop
    if strategy.break_even_trigger > 0 and position.max_profit >= strategy.break_even_trigger:
        stop_level = 0.05  # Slight profit to cover costs

    # Adjust for trailing stop
    if strategy.trailing_stop and position.max_profit >= strategy.trailing_stop:
        trailing_level = position.max_profit - strategy.trailing_distance
        stop_level = max(stop_level, trailing_level)

    if pnl <= stop_level:
        return ('stop_loss', pnl, position.size)

    # 3. Time stop
    if hold_time >= strategy.time_stop:
        return ('time_stop', pnl, position.size)

    # 4. Scale out
    for level, pct in strategy.scale_out_levels:
        if pnl >= level and position.size > pct:
            # This is simplified - in reality would track partial closes
            pass

    return None


# =============================================================================
# FILTERS - When NOT to trade
# =============================================================================

def filter_blowout(state: pd.Series, threshold: float = 18) -> bool:
    """Don't trade blowouts."""
    return state['abs_diff'] > threshold


def filter_late_game(state: pd.Series, min_mins: float = 3) -> bool:
    """Don't trade too late."""
    return state['mins_remaining'] < min_mins


def filter_early_game(state: pd.Series, max_mins: float = 44) -> bool:
    """Don't trade too early."""
    return state['mins_remaining'] > max_mins


def filter_low_volatility(state: pd.Series, threshold: float = 0.8) -> bool:
    """Skip low volatility games."""
    return state['game_volatility'] < threshold


def filter_halftime(state: pd.Series) -> bool:
    """Don't trade during halftime."""
    return state['period'] == 2 and state['clock_secs'] < 30


# =============================================================================
# COMPREHENSIVE BACKTEST ENGINE
# =============================================================================

@dataclass
class TradeResult:
    game_id: str
    entry_time: float
    exit_time: float
    entry_spread: float
    exit_spread: float
    side: str
    pnl: float
    exit_reason: str
    signal_type: str
    hold_time: float


@dataclass
class BacktestResult:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    avg_hold_time: float
    trades_per_game: float
    exit_breakdown: Dict[str, int]
    trades: List[TradeResult]


def run_backtest(
    df: pd.DataFrame,
    signal_func: Callable,
    exit_strategy: ExitStrategy,
    filters: List[Callable] = None,
    min_signal_strength: float = 1.0,
    max_trades_per_game: int = 10,
    cooldown_seconds: float = 30,
) -> BacktestResult:
    """
    Run comprehensive backtest with given signal and exit strategy.
    """
    if filters is None:
        filters = [filter_blowout, filter_late_game, filter_early_game]

    trades = []
    positions: Dict[str, Position] = {}
    last_trade_time: Dict[str, float] = {}
    trade_count: Dict[str, int] = {}
    state_history: Dict[str, List] = {}

    for game_id, game_df in df.groupby('game_id'):
        game_df = game_df.sort_values('seconds')
        state_history[game_id] = []

        for _, state in game_df.iterrows():
            state_history[game_id].append(state)
            secs = state['seconds']

            # Check existing position
            if game_id in positions:
                pos = positions[game_id]
                exit_result = check_exit(pos, state['market_spread'], secs, exit_strategy)

                if exit_result:
                    reason, pnl, size = exit_result
                    trades.append(TradeResult(
                        game_id=game_id,
                        entry_time=pos.entry_time,
                        exit_time=secs,
                        entry_spread=pos.entry_spread,
                        exit_spread=state['market_spread'],
                        side=pos.side,
                        pnl=pnl * size,
                        exit_reason=reason,
                        signal_type=pos.entry_signal,
                        hold_time=secs - pos.entry_time,
                    ))
                    del positions[game_id]
                continue

            # Check filters
            if any(f(state) for f in filters):
                continue

            # Cooldown
            if secs - last_trade_time.get(game_id, -999) < cooldown_seconds:
                continue

            # Trade limit
            if trade_count.get(game_id, 0) >= max_trades_per_game:
                continue

            # Check for signal
            try:
                signal = signal_func(state, state_history[game_id])
            except TypeError:
                signal = signal_func(state)

            if signal is None:
                continue

            side, strength = signal
            if strength < min_signal_strength:
                continue

            # Enter position
            positions[game_id] = Position(
                game_id=game_id,
                entry_time=secs,
                entry_spread=state['market_spread'],
                side=side,
                entry_signal=signal_func.__name__,
            )
            last_trade_time[game_id] = secs
            trade_count[game_id] = trade_count.get(game_id, 0) + 1

    # Force close remaining positions at game end
    for game_id, pos in positions.items():
        game_df = df[df['game_id'] == game_id]
        final_state = game_df.iloc[-1]
        pnl = (final_state['market_spread'] - pos.entry_spread) if pos.side == 'away' else (pos.entry_spread - final_state['market_spread'])
        trades.append(TradeResult(
            game_id=game_id,
            entry_time=pos.entry_time,
            exit_time=final_state['seconds'],
            entry_spread=pos.entry_spread,
            exit_spread=final_state['market_spread'],
            side=pos.side,
            pnl=pnl,
            exit_reason='game_end',
            signal_type=pos.entry_signal,
            hold_time=final_state['seconds'] - pos.entry_time,
        ))

    # Calculate metrics
    if not trades:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, [])

    total = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    win_rate = wins / total

    pnls = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    avg_pnl = np.mean(pnls)

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0

    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    avg_hold = np.mean([t.hold_time for t in trades])
    unique_games = len(set(t.game_id for t in trades))
    tpg = total / unique_games if unique_games > 0 else 0

    exit_breakdown = {}
    for t in trades:
        exit_breakdown[t.exit_reason] = exit_breakdown.get(t.exit_reason, 0) + 1

    return BacktestResult(
        total_trades=total,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        profit_factor=profit_factor,
        sharpe=sharpe,
        max_drawdown=max_dd,
        avg_hold_time=avg_hold,
        trades_per_game=tpg,
        exit_breakdown=exit_breakdown,
        trades=trades,
    )


# =============================================================================
# COMPREHENSIVE RESEARCH - Test many combinations
# =============================================================================

def research_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Test all signal types with various parameters."""
    print("\n" + "="*70)
    print("SIGNAL RESEARCH")
    print("="*70)

    results = []
    exit_strat = EXIT_STRATEGIES['tight_tp']  # Hold exit constant

    # Momentum reversion with different thresholds
    for thresh in [4, 5, 6, 7, 8, 10]:
        def sig(state, thresh=thresh):
            return signal_momentum_reversion(state, thresh)
        result = run_backtest(df, sig, exit_strat)
        results.append({
            'signal': f'mom_rev_{thresh}',
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'pnl': result.total_pnl,
            'sharpe': result.sharpe,
            'pf': result.profit_factor,
        })
        print(f"mom_rev_{thresh}: {result.total_trades} trades, WR={result.win_rate:.1%}, PnL={result.total_pnl:+.0f}")

    # Spread deviation with different thresholds
    for thresh in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        def sig(state, thresh=thresh):
            return signal_spread_deviation(state, thresh)
        result = run_backtest(df, sig, exit_strat)
        results.append({
            'signal': f'spread_dev_{thresh}',
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'pnl': result.total_pnl,
            'sharpe': result.sharpe,
            'pf': result.profit_factor,
        })
        print(f"spread_dev_{thresh}: {result.total_trades} trades, WR={result.win_rate:.1%}, PnL={result.total_pnl:+.0f}")

    # Momentum exhaustion
    for short in [4, 5, 6]:
        def sig(state, short=short):
            return signal_momentum_exhaustion(state, short, 2)
        result = run_backtest(df, sig, exit_strat)
        results.append({
            'signal': f'mom_exhaust_{short}',
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'pnl': result.total_pnl,
            'sharpe': result.sharpe,
            'pf': result.profit_factor,
        })
        print(f"mom_exhaust_{short}: {result.total_trades} trades, WR={result.win_rate:.1%}, PnL={result.total_pnl:+.0f}")

    return pd.DataFrame(results)


def research_exits(df: pd.DataFrame, best_signal_func) -> pd.DataFrame:
    """Test all exit strategies with the best signal."""
    print("\n" + "="*70)
    print("EXIT STRATEGY RESEARCH")
    print("="*70)

    results = []

    for name, strat in EXIT_STRATEGIES.items():
        result = run_backtest(df, best_signal_func, strat)
        results.append({
            'exit': name,
            'tp': strat.take_profit,
            'sl': strat.stop_loss,
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'pnl': result.total_pnl,
            'sharpe': result.sharpe,
            'pf': result.profit_factor,
            'max_dd': result.max_drawdown,
        })
        print(f"{name}: WR={result.win_rate:.1%}, PnL={result.total_pnl:+.0f}, Sharpe={result.sharpe:.2f}")

    return pd.DataFrame(results)


def research_tp_sl_grid(df: pd.DataFrame, signal_func) -> pd.DataFrame:
    """Exhaustive grid search over TP/SL combinations."""
    print("\n" + "="*70)
    print("TP/SL GRID SEARCH")
    print("="*70)

    results = []

    tp_range = [0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5]
    sl_range = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    time_range = [60, 90, 120, 180]

    total_combos = len(tp_range) * len(sl_range) * len(time_range)
    print(f"Testing {total_combos} combinations...")

    combo_count = 0
    for tp in tp_range:
        for sl in sl_range:
            if tp >= sl * 0.6:  # Skip unreasonable combos
                continue
            for ts in time_range:
                combo_count += 1
                strat = ExitStrategy(
                    name=f'tp{tp}_sl{sl}_ts{ts}',
                    take_profit=tp,
                    stop_loss=sl,
                    time_stop=ts,
                )
                result = run_backtest(df, signal_func, strat)

                if result.total_trades < 100:
                    continue

                results.append({
                    'tp': tp,
                    'sl': sl,
                    'time_stop': ts,
                    'ratio': sl/tp,
                    'trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'pnl': result.total_pnl,
                    'sharpe': result.sharpe,
                    'pf': result.profit_factor,
                    'max_dd': result.max_drawdown,
                    'avg_hold': result.avg_hold_time,
                })

    print(f"Tested {combo_count} combinations, {len(results)} valid results")
    return pd.DataFrame(results)


def research_combined_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Test combinations of signals."""
    print("\n" + "="*70)
    print("COMBINED SIGNAL RESEARCH")
    print("="*70)

    def combined_signal_v1(state, history=None):
        """Require both momentum and spread deviation."""
        mom = signal_momentum_reversion(state, threshold=5)
        dev = signal_spread_deviation(state, threshold=2.0)

        if mom and dev:
            if mom[0] == dev[0]:  # Same direction
                return (mom[0], mom[1] + dev[1])
        return None

    def combined_signal_v2(state, history=None):
        """Momentum + exhaustion confirmation."""
        mom = signal_momentum_reversion(state, threshold=6)
        exh = signal_momentum_exhaustion(state, 5, 2)

        if mom and exh:
            if mom[0] == exh[0]:
                return (mom[0], mom[1] * 1.5)
        return None

    def combined_signal_v3(state, history=None):
        """Strong spread deviation only."""
        dev = signal_spread_deviation(state, threshold=3.0)
        if dev and dev[1] >= 1.2:
            return dev
        return None

    results = []
    strat = EXIT_STRATEGIES['tight_tp']

    for name, sig in [('combined_v1', combined_signal_v1),
                       ('combined_v2', combined_signal_v2),
                       ('combined_v3', combined_signal_v3)]:
        result = run_backtest(df, sig, strat)
        results.append({
            'signal': name,
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'pnl': result.total_pnl,
            'sharpe': result.sharpe,
            'pf': result.profit_factor,
        })
        print(f"{name}: {result.total_trades} trades, WR={result.win_rate:.1%}, PnL={result.total_pnl:+.0f}")

    return pd.DataFrame(results)


def main():
    print("="*70)
    print("COMPREHENSIVE SPREAD TRADING RESEARCH")
    print("="*70)

    # Generate data
    print("\nGenerating 5000 games with realistic spread dynamics...")
    df = generate_dataset(n_games=5000)
    print(f"Generated {len(df):,} data points")

    # Research signals
    signal_results = research_signals(df)

    # Find best signal
    best_signal_row = signal_results.sort_values('sharpe', ascending=False).iloc[0]
    print(f"\nBest signal by Sharpe: {best_signal_row['signal']}")

    # Use spread deviation as our best signal
    def best_signal(state, history=None):
        return signal_spread_deviation(state, threshold=2.5)

    # Research exit strategies
    exit_results = research_exits(df, best_signal)

    # Grid search TP/SL
    grid_results = research_tp_sl_grid(df, best_signal)

    # Sort and show best
    print("\n" + "="*70)
    print("TOP 10 TP/SL COMBINATIONS BY WIN RATE")
    print("="*70)
    top_wr = grid_results.sort_values('win_rate', ascending=False).head(10)
    print(top_wr.to_string(index=False))

    print("\n" + "="*70)
    print("TOP 10 TP/SL COMBINATIONS BY SHARPE")
    print("="*70)
    top_sharpe = grid_results.sort_values('sharpe', ascending=False).head(10)
    print(top_sharpe.to_string(index=False))

    print("\n" + "="*70)
    print("TOP 10 TP/SL COMBINATIONS BY P&L")
    print("="*70)
    top_pnl = grid_results.sort_values('pnl', ascending=False).head(10)
    print(top_pnl.to_string(index=False))

    # Test combined signals
    combined_results = research_combined_signals(df)

    # Final best strategy
    best = grid_results.sort_values('win_rate', ascending=False).iloc[0]
    print("\n" + "="*70)
    print("BEST OVERALL STRATEGY")
    print("="*70)
    print(f"""
Entry Signal: Spread Deviation >= 2.5 points from fair value
Exit Strategy:
  - Take Profit: {best['tp']} points
  - Stop Loss: {best['sl']} points
  - Time Stop: {best['time_stop']} seconds

Results:
  - Win Rate: {best['win_rate']:.1%}
  - Total Trades: {int(best['trades']):,}
  - Total P&L: {best['pnl']:+.0f} points
  - Sharpe Ratio: {best['sharpe']:.2f}
  - Profit Factor: {best['pf']:.2f}
  - Max Drawdown: {best['max_dd']:.1f} points
  - Avg Hold Time: {best['avg_hold']:.0f} seconds
  - SL/TP Ratio: {best['ratio']:.1f}x
""")

    # Save results
    grid_results.to_csv('/home/user/nba2/output/grid_search_results.csv', index=False)
    print("\nGrid search results saved to output/grid_search_results.csv")

    return grid_results


if __name__ == "__main__":
    results = main()
