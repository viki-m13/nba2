"""
Run Spread Trading Backtest

This script runs the spread trading strategy on simulated NBA game data
and outputs comprehensive results.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from spread_trading_strategy import (
    SpreadTradingConfig,
    SpreadTradingStrategy,
    run_spread_backtest,
    print_spread_results,
    SpreadBacktestResults
)


def generate_game_states_for_spread_trading(n_games: int = 1000,
                                             seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic game state data optimized for spread trading analysis.

    Includes realistic momentum patterns and spread movements.
    """
    np.random.seed(seed)

    all_states = []

    for game_num in range(n_games):
        game_id = f"00230{game_num:05d}"
        states = _simulate_game_with_spreads(game_id)
        all_states.extend(states)

    return pd.DataFrame(all_states)


def _simulate_game_with_spreads(game_id: str) -> List[Dict]:
    """
    Simulate a single game with realistic spread movements.

    Key patterns we model:
    1. Scoring runs cause spread overreaction
    2. Spreads tend to revert after overreaction
    3. Higher volatility in closer games
    """
    states = []

    # Game parameters
    home_skill = np.random.normal(0, 4)  # Team strength
    game_volatility = np.random.uniform(0.8, 1.5)  # How swingy is this game

    home_score = 0
    away_score = 0

    # Track recent scoring for momentum calculation
    recent_home_scores = []
    recent_away_scores = []

    # Simulate every 15 seconds for granular spread movements
    for game_seconds in range(60, 48*60 + 1, 15):
        period = min((game_seconds // (12*60)) + 1, 4)
        clock_seconds = 12 * 60 - (game_seconds - (period - 1) * 12 * 60)
        minutes_remaining = max((48 * 60 - game_seconds) / 60, 0)

        # Scoring events (probabilistic)
        # Higher chance of scoring during runs
        home_momentum = sum(recent_home_scores[-8:]) - sum(recent_away_scores[-8:]) if recent_home_scores else 0

        # Base scoring rate ~2.2 pts/min per team = 0.55 pts per 15 sec
        base_rate = 0.55 * game_volatility

        # Momentum affects scoring (teams on runs tend to continue briefly)
        momentum_boost = 0.05 * abs(home_momentum)

        home_pts = np.random.poisson(base_rate * (1 + home_skill/50 + (0.02 if home_momentum > 0 else 0)))
        away_pts = np.random.poisson(base_rate * (1 - home_skill/50 + (0.02 if home_momentum < 0 else 0)))

        home_score += home_pts
        away_score += away_pts

        # Track for momentum
        recent_home_scores.append(home_pts)
        recent_away_scores.append(away_pts)
        if len(recent_home_scores) > 20:  # Keep last 5 min (20 x 15sec)
            recent_home_scores.pop(0)
            recent_away_scores.pop(0)

        # Calculate momentum (2 min = 8 samples, 5 min = 20 samples)
        home_2min = sum(recent_home_scores[-8:])
        away_2min = sum(recent_away_scores[-8:])
        home_5min = sum(recent_home_scores)
        away_5min = sum(recent_away_scores)

        momentum_2min = home_2min - away_2min
        momentum_5min = home_5min - away_5min

        score_diff = home_score - away_score

        state = {
            'game_id': game_id,
            'period': period,
            'quarter': period,
            'clock_seconds': max(clock_seconds, 0),
            'game_seconds_elapsed': game_seconds,
            'game_seconds_remaining': max(48*60 - game_seconds, 0),
            'minutes_remaining': minutes_remaining,
            'home_score': home_score,
            'away_score': away_score,
            'score_diff': score_diff,
            'abs_score_diff': abs(score_diff),
            'momentum_2min': momentum_2min,
            'momentum_5min': momentum_5min,
            'run_diff_2min': momentum_2min,
            'run_diff_5min': momentum_5min,
            'home_run_2min': max(home_2min - away_2min, 0),
            'away_run_2min': max(away_2min - home_2min, 0),
            'game_volatility': game_volatility,
        }
        states.append(state)

    return states


def run_parameter_optimization(game_states: pd.DataFrame) -> Dict:
    """
    Find optimal parameters for spread trading strategy.
    """
    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION")
    print("="*60)

    best_result = None
    best_config = None
    best_score = -float('inf')

    # Grid search over key parameters
    param_grid = {
        'min_spread_deviation': [1.5, 2.0, 2.5, 3.0],
        'take_profit_points': [1.0, 1.5, 2.0],
        'stop_loss_points': [2.0, 2.5, 3.0, 3.5],
        'time_stop_seconds': [180, 300, 420],
    }

    results = []

    for min_dev in param_grid['min_spread_deviation']:
        for tp in param_grid['take_profit_points']:
            for sl in param_grid['stop_loss_points']:
                for ts in param_grid['time_stop_seconds']:
                    # Skip invalid combos (TP should be less than SL)
                    if tp >= sl:
                        continue

                    config = SpreadTradingConfig(
                        min_spread_deviation=min_dev,
                        take_profit_points=tp,
                        stop_loss_points=sl,
                        time_stop_seconds=ts
                    )

                    result = run_spread_backtest(game_states, config)

                    if result.total_trades < 100:
                        continue

                    # Score: prioritize win rate and profit factor
                    score = result.win_rate * result.profit_factor * np.log1p(result.total_trades)

                    results.append({
                        'min_dev': min_dev,
                        'tp': tp,
                        'sl': sl,
                        'ts': ts,
                        'trades': result.total_trades,
                        'win_rate': result.win_rate,
                        'pf': result.profit_factor,
                        'pnl': result.total_pnl,
                        'score': score
                    })

                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_config = config

    # Print top results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    print("\nTop 10 Parameter Combinations:")
    print(results_df.head(10).to_string(index=False))

    return {
        'best_config': best_config,
        'best_result': best_result,
        'all_results': results_df
    }


def run_full_analysis():
    """Run complete spread trading analysis."""

    print("="*70)
    print("NBA IN-GAME SPREAD TRADING STRATEGY BACKTEST")
    print("="*70)

    # Generate data
    print("\nGenerating game data (5000 games)...")
    game_states = generate_game_states_for_spread_trading(n_games=5000, seed=42)
    print(f"Generated {len(game_states):,} game states")

    # Run with default config first
    print("\n" + "-"*60)
    print("DEFAULT CONFIGURATION RESULTS")
    print("-"*60)

    default_config = SpreadTradingConfig(
        min_spread_deviation=2.0,
        take_profit_points=1.5,
        stop_loss_points=2.5,
        time_stop_seconds=300
    )

    default_result = run_spread_backtest(game_states, default_config)
    print_spread_results(default_result)

    # Parameter optimization
    opt_results = run_parameter_optimization(game_states)

    # Run with optimized config
    print("\n" + "-"*60)
    print("OPTIMIZED CONFIGURATION RESULTS")
    print("-"*60)

    if opt_results['best_config']:
        opt_config = opt_results['best_config']
        print(f"\nBest Parameters:")
        print(f"  Min Spread Deviation: {opt_config.min_spread_deviation}")
        print(f"  Take Profit: {opt_config.take_profit_points}")
        print(f"  Stop Loss: {opt_config.stop_loss_points}")
        print(f"  Time Stop: {opt_config.time_stop_seconds}")

        print_spread_results(opt_results['best_result'])

    # Season stability check
    print("\n" + "-"*60)
    print("STABILITY ANALYSIS (by 1000-game blocks)")
    print("-"*60)

    unique_games = game_states['game_id'].unique()
    n_blocks = 5
    block_size = len(unique_games) // n_blocks

    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        block_games = unique_games[start_idx:end_idx]
        block_states = game_states[game_states['game_id'].isin(block_games)]

        result = run_spread_backtest(block_states, default_config)
        status = "✓" if result.win_rate > 0.7 else "○"
        print(f"  Block {i+1}: {status} Win Rate={result.win_rate:.1%}, Trades={result.total_trades}, P&L={result.total_pnl:+.1f}")

    # Final summary
    print("\n" + "="*70)
    print("FINAL STRATEGY SUMMARY")
    print("="*70)

    best = opt_results['best_result'] if opt_results['best_result'] else default_result

    print(f"""
SPREAD TRADING STRATEGY RULEBOOK
================================

Entry Rules:
- Enter when market spread deviates {opt_config.min_spread_deviation if opt_results['best_config'] else 2.0}+ points from fair value
- Fair value = score_diff + remaining_HCA - momentum_reversion_adjustment
- If market too favorable to home → LONG AWAY (expect spread to increase)
- If market too favorable to away → LONG HOME (expect spread to decrease)

Exit Rules:
- TAKE PROFIT: When spread moves {opt_config.take_profit_points if opt_results['best_config'] else 1.5} points in your favor
- STOP LOSS: When spread moves {opt_config.stop_loss_points if opt_results['best_config'] else 2.5} points against you
- TIME STOP: Exit after {(opt_config.time_stop_seconds if opt_results['best_config'] else 300)/60:.0f} minutes regardless

Filters:
- No trades in final 4 minutes
- No trades when score diff > 25 points
- Max 10 trades per game
- 60 second cooldown between trades

RESULTS:
- Win Rate: {best.win_rate:.1%}
- Trades per Game: {best.trades_per_game:.1f}
- Profit Factor: {best.profit_factor:.2f}
- Total P&L: {best.total_pnl:+.1f} spread points

Exit Breakdown:
- Take Profit exits: {best.take_profit_count/best.total_trades*100:.0f}%
- Stop Loss exits: {best.stop_loss_count/best.total_trades*100:.0f}%
- Time Stop exits: {best.time_stop_count/best.total_trades*100:.0f}%
""")

    # Save results
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "spread_trading_results.json", 'w') as f:
        json.dump({
            'total_trades': best.total_trades,
            'win_rate': best.win_rate,
            'profit_factor': best.profit_factor,
            'total_pnl': best.total_pnl,
            'trades_per_game': best.trades_per_game,
            'take_profit_pct': best.take_profit_count / best.total_trades,
            'stop_loss_pct': best.stop_loss_count / best.total_trades,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/spread_trading_results.json")

    return best


if __name__ == "__main__":
    results = run_full_analysis()
