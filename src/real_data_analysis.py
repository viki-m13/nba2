"""
Real PBP Data Analysis Script

This script attempts to pull actual NBA play-by-play data using nba_api
and runs the strategy analysis on real historical games.

Note: Due to rate limits and API constraints, this may take significant time.
Use the cached data once collected for faster iteration.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from data_collector import CachedNBADataCollector, get_game_seconds_elapsed, parse_game_clock
from feature_engineering import PBPFeatureExtractor, states_to_dataframe, compute_derived_features
from backtester import Backtester, BacktestConfig, print_backtest_summary
from strategy import STRATEGIES, CompositeStrategy


def process_pbp_to_features(pbp_df: pd.DataFrame,
                            game_info: Dict,
                            sample_interval: float = 30.0) -> pd.DataFrame:
    """
    Process raw PBP data into feature DataFrame.

    Args:
        pbp_df: Raw play-by-play DataFrame
        game_info: Game metadata
        sample_interval: Sampling interval in seconds

    Returns:
        DataFrame with features for each time sample
    """
    if pbp_df.empty:
        return pd.DataFrame()

    # Get team IDs from game info or PBP
    team_id = game_info.get('TEAM_ID', 0)
    matchup = game_info.get('MATCHUP', '')

    # Determine home/away from matchup (format: "TOR vs. BOS" or "TOR @ BOS")
    is_home = 'vs.' in matchup

    # Get unique team IDs from PBP
    team_ids = pbp_df['PLAYER1_TEAM_ID'].dropna().unique()
    team_ids = [int(t) for t in team_ids if pd.notna(t) and t > 0]

    if len(team_ids) < 2:
        return pd.DataFrame()

    # Assign home/away based on position (first listed is usually home in 'vs' format)
    if is_home:
        home_team_id = team_id
        away_team_id = [t for t in team_ids if t != team_id][0] if team_id in team_ids else team_ids[1]
    else:
        away_team_id = team_id
        home_team_id = [t for t in team_ids if t != team_id][0] if team_id in team_ids else team_ids[0]

    # Extract game states
    extractor = PBPFeatureExtractor()
    states = extractor.extract_game_states(pbp_df, home_team_id, away_team_id, sample_interval)

    if not states:
        return pd.DataFrame()

    # Convert to DataFrame and add derived features
    df = states_to_dataframe(states)
    df = compute_derived_features(df)

    return df


def determine_game_winner(pbp_df: pd.DataFrame, game_info: Dict) -> Optional[str]:
    """
    Determine the winner of a game from PBP data.

    Returns:
        'home' or 'away', or None if cannot determine
    """
    if pbp_df.empty:
        return None

    # Get final score from last SCORE column entry
    scores = pbp_df['SCORE'].dropna()
    if scores.empty:
        return None

    final_score_str = scores.iloc[-1]
    try:
        parts = str(final_score_str).split(' - ')
        if len(parts) == 2:
            away_score = int(parts[0])
            home_score = int(parts[1])

            if home_score > away_score:
                return 'home'
            elif away_score > home_score:
                return 'away'
            else:
                return None  # Tie (shouldn't happen in NBA)
    except:
        pass

    return None


def collect_and_analyze_real_data(seasons: List[str],
                                   max_games_per_season: int = 100,
                                   config: BacktestConfig = None):
    """
    Collect real NBA data and run strategy analysis.

    Args:
        seasons: List of seasons to analyze
        max_games_per_season: Max games to collect per season
        config: Backtest configuration
    """
    if config is None:
        config = BacktestConfig()

    print("="*70)
    print("NBA IN-GAME TRADING - REAL DATA ANALYSIS")
    print("="*70)

    collector = CachedNBADataCollector(delay_seconds=0.7)

    all_states = []
    all_outcomes = {}

    for season in seasons:
        print(f"\nProcessing {season}...")

        # Collect games
        season_data = collector.collect_season_pbp(
            season,
            season_type="Regular Season",
            max_games=max_games_per_season
        )

        print(f"  Collected {len(season_data)} games")

        # Process each game
        for game_data in season_data:
            game_id = game_data['game_id']
            pbp_df = game_data['pbp']
            game_info = game_data['game_info']

            # Extract features
            features_df = process_pbp_to_features(pbp_df, game_info)

            if features_df.empty:
                continue

            features_df['season'] = season
            all_states.append(features_df)

            # Determine winner
            winner = determine_game_winner(pbp_df, game_info)
            if winner:
                all_outcomes[game_id] = winner

    if not all_states:
        print("No data collected!")
        return None

    combined_states = pd.concat(all_states, ignore_index=True)
    print(f"\nTotal: {len(combined_states)} game states, {len(all_outcomes)} games with outcomes")

    # Run backtests
    results = {}
    backtester = Backtester(config)

    print("\n" + "-"*70)
    print("STRATEGY BACKTESTS ON REAL DATA")
    print("-"*70)

    for strategy_name, strategy_class in STRATEGIES.items():
        print(f"\n>>> {strategy_name}")

        strategy = strategy_class(config)
        result = backtester.run_backtest(strategy, combined_states, all_outcomes)
        results[strategy_name] = result

        if result.total_trades > 0:
            print(f"    Trades: {result.total_trades}")
            print(f"    Win Rate: {result.win_rate:.1%}")
            print(f"    P&L: {result.total_pnl:+.2f}")
            print(f"    Sharpe: {result.sharpe_ratio:.2f}")

    return {
        'results': results,
        'states': combined_states,
        'outcomes': all_outcomes
    }


def save_processed_data(data: Dict, output_path: str):
    """Save processed data for future use."""
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved processed data to {output_path}")


def load_processed_data(input_path: str) -> Optional[Dict]:
    """Load previously processed data."""
    if not os.path.exists(input_path):
        return None
    with open(input_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NBA Real Data Analysis')
    parser.add_argument('--seasons', nargs='+', default=['2023-24'],
                        help='Seasons to analyze')
    parser.add_argument('--max-games', type=int, default=50,
                        help='Max games per season')
    parser.add_argument('--load-cache', type=str, default=None,
                        help='Load from cached processed data')
    parser.add_argument('--save-cache', type=str, default=None,
                        help='Save processed data to cache')

    args = parser.parse_args()

    config = BacktestConfig()

    if args.load_cache:
        print(f"Loading from cache: {args.load_cache}")
        data = load_processed_data(args.load_cache)
        if data:
            # Re-run analysis on loaded data
            combined_states = data['states']
            all_outcomes = data['outcomes']

            backtester = Backtester(config)
            for strategy_name, strategy_class in STRATEGIES.items():
                strategy = strategy_class(config)
                result = backtester.run_backtest(strategy, combined_states, all_outcomes)
                print(f"{strategy_name}: {result.total_trades} trades, P&L={result.total_pnl:+.2f}")
        else:
            print("Cache not found, collecting fresh data...")
            data = collect_and_analyze_real_data(args.seasons, args.max_games, config)
    else:
        data = collect_and_analyze_real_data(args.seasons, args.max_games, config)

    if args.save_cache and data:
        save_processed_data(data, args.save_cache)
