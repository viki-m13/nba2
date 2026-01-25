"""
NBA Play-by-Play Data Collector with Caching

This module handles data collection from nba_api with persistent caching
to avoid repeatedly hitting endpoints when pulling multiple seasons.
"""

import os
import json
import time
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

# nba_api imports
from nba_api.stats.endpoints import (
    playbyplayv2,
    leaguegamefinder,
    boxscoretraditionalv2,
    boxscoresummaryv2
)
from nba_api.stats.static import teams


class CachedNBADataCollector:
    """
    Collects NBA play-by-play data with persistent caching.

    Cache structure:
    - data/cache/games_{season}.pkl: Game metadata for each season
    - data/cache/pbp_{game_id}.pkl: Play-by-play for each game
    - data/cache/boxscore_{game_id}.pkl: Box score for each game
    """

    def __init__(self, cache_dir: str = "data/cache", delay_seconds: float = 0.6):
        """
        Initialize the collector.

        Args:
            cache_dir: Directory for caching data
            delay_seconds: Delay between API calls to avoid rate limiting
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_seconds = delay_seconds

        # Get all NBA teams
        self.nba_teams = {team['id']: team for team in teams.get_teams()}

    def _cache_path(self, prefix: str, key: str) -> Path:
        """Generate cache file path."""
        return self.cache_dir / f"{prefix}_{key}.pkl"

    def _load_from_cache(self, prefix: str, key: str) -> Optional[Any]:
        """Load data from cache if exists."""
        path = self._cache_path(prefix, key)
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache read error for {path}: {e}")
        return None

    def _save_to_cache(self, prefix: str, key: str, data: Any):
        """Save data to cache."""
        path = self._cache_path(prefix, key)
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Cache write error for {path}: {e}")

    def _api_call_with_retry(self, func, max_retries: int = 3, **kwargs):
        """Execute API call with retry logic."""
        for attempt in range(max_retries):
            try:
                time.sleep(self.delay_seconds)
                return func(**kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff
                    print(f"API error (attempt {attempt + 1}): {e}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def get_season_games(self, season: str, season_type: str = "Regular Season") -> pd.DataFrame:
        """
        Get all games for a season.

        Args:
            season: Season string (e.g., "2022-23")
            season_type: "Regular Season" or "Playoffs"

        Returns:
            DataFrame with game metadata
        """
        cache_key = f"{season}_{season_type.replace(' ', '_')}"
        cached = self._load_from_cache("games", cache_key)

        if cached is not None:
            return cached

        print(f"Fetching games for {season} {season_type}...")

        all_games = []

        # Get games for each team and deduplicate
        for team_id in tqdm(list(self.nba_teams.keys()), desc="Teams"):
            try:
                finder = self._api_call_with_retry(
                    leaguegamefinder.LeagueGameFinder,
                    team_id_nullable=team_id,
                    season_nullable=season,
                    season_type_nullable=season_type
                )
                games_df = finder.get_data_frames()[0]
                all_games.append(games_df)
            except Exception as e:
                print(f"Error fetching games for team {team_id}: {e}")
                continue

        if not all_games:
            return pd.DataFrame()

        games_df = pd.concat(all_games, ignore_index=True)

        # Deduplicate (each game appears twice, once per team)
        games_df = games_df.drop_duplicates(subset=['GAME_ID'])
        games_df = games_df.sort_values('GAME_DATE').reset_index(drop=True)

        self._save_to_cache("games", cache_key, games_df)
        return games_df

    def get_play_by_play(self, game_id: str) -> pd.DataFrame:
        """
        Get play-by-play data for a specific game.

        Args:
            game_id: NBA game ID

        Returns:
            DataFrame with play-by-play events
        """
        cached = self._load_from_cache("pbp", game_id)

        if cached is not None:
            return cached

        try:
            pbp = self._api_call_with_retry(
                playbyplayv2.PlayByPlayV2,
                game_id=game_id
            )
            pbp_df = pbp.get_data_frames()[0]
            self._save_to_cache("pbp", game_id, pbp_df)
            return pbp_df
        except Exception as e:
            print(f"Error fetching PBP for game {game_id}: {e}")
            return pd.DataFrame()

    def get_box_score(self, game_id: str) -> Dict[str, pd.DataFrame]:
        """
        Get box score for a specific game.

        Args:
            game_id: NBA game ID

        Returns:
            Dict with player and team stats DataFrames
        """
        cached = self._load_from_cache("boxscore", game_id)

        if cached is not None:
            return cached

        try:
            box = self._api_call_with_retry(
                boxscoretraditionalv2.BoxScoreTraditionalV2,
                game_id=game_id
            )
            dfs = box.get_data_frames()
            result = {
                'players': dfs[0] if len(dfs) > 0 else pd.DataFrame(),
                'teams': dfs[1] if len(dfs) > 1 else pd.DataFrame()
            }
            self._save_to_cache("boxscore", game_id, result)
            return result
        except Exception as e:
            print(f"Error fetching box score for game {game_id}: {e}")
            return {'players': pd.DataFrame(), 'teams': pd.DataFrame()}

    def get_game_summary(self, game_id: str) -> Dict[str, Any]:
        """
        Get game summary including final scores and team info.

        Args:
            game_id: NBA game ID

        Returns:
            Dict with game summary information
        """
        cached = self._load_from_cache("summary", game_id)

        if cached is not None:
            return cached

        try:
            summary = self._api_call_with_retry(
                boxscoresummaryv2.BoxScoreSummaryV2,
                game_id=game_id
            )
            dfs = summary.get_data_frames()
            result = {
                'game_summary': dfs[0] if len(dfs) > 0 else pd.DataFrame(),
                'line_score': dfs[5] if len(dfs) > 5 else pd.DataFrame()
            }
            self._save_to_cache("summary", game_id, result)
            return result
        except Exception as e:
            print(f"Error fetching summary for game {game_id}: {e}")
            return {'game_summary': pd.DataFrame(), 'line_score': pd.DataFrame()}

    def collect_season_pbp(self, season: str,
                          season_type: str = "Regular Season",
                          max_games: Optional[int] = None) -> List[Dict]:
        """
        Collect all play-by-play data for a season.

        Args:
            season: Season string (e.g., "2022-23")
            season_type: "Regular Season" or "Playoffs"
            max_games: Maximum number of games to collect (for testing)

        Returns:
            List of dicts with game metadata and PBP data
        """
        games_df = self.get_season_games(season, season_type)

        if games_df.empty:
            return []

        game_ids = games_df['GAME_ID'].unique()

        if max_games:
            game_ids = game_ids[:max_games]

        results = []

        for game_id in tqdm(game_ids, desc=f"Collecting PBP for {season}"):
            pbp_df = self.get_play_by_play(game_id)

            if pbp_df.empty:
                continue

            game_info = games_df[games_df['GAME_ID'] == game_id].iloc[0].to_dict()
            summary = self.get_game_summary(game_id)

            results.append({
                'game_id': game_id,
                'game_info': game_info,
                'pbp': pbp_df,
                'summary': summary
            })

        return results

    def collect_multiple_seasons(self, seasons: List[str],
                                 season_type: str = "Regular Season",
                                 max_games_per_season: Optional[int] = None) -> List[Dict]:
        """
        Collect PBP data for multiple seasons.

        Args:
            seasons: List of season strings
            season_type: "Regular Season" or "Playoffs"
            max_games_per_season: Max games per season (for testing)

        Returns:
            Combined list of all game data
        """
        all_data = []

        for season in seasons:
            print(f"\n{'='*50}")
            print(f"Collecting {season} {season_type}")
            print(f"{'='*50}")

            season_data = self.collect_season_pbp(
                season, season_type, max_games_per_season
            )
            all_data.extend(season_data)

            print(f"Collected {len(season_data)} games for {season}")

        return all_data


def parse_game_clock(pctimestring: str) -> float:
    """
    Parse game clock string to seconds remaining in period.

    Args:
        pctimestring: Clock string like "11:45" or "0:30"

    Returns:
        Seconds remaining in period
    """
    if pd.isna(pctimestring) or not pctimestring:
        return 0.0

    try:
        parts = str(pctimestring).split(':')
        minutes = int(parts[0])
        seconds = int(parts[1]) if len(parts) > 1 else 0
        return minutes * 60 + seconds
    except:
        return 0.0


def get_game_seconds_elapsed(period: int, pctimestring: str) -> float:
    """
    Calculate total seconds elapsed in game.

    Args:
        period: Current period (1-4 for regulation, 5+ for OT)
        pctimestring: Clock string

    Returns:
        Total seconds elapsed in game
    """
    period_length = 12 * 60  # 12 minutes per period
    ot_length = 5 * 60  # 5 minutes per OT

    seconds_in_period = parse_game_clock(pctimestring)

    if period <= 4:
        # Regulation
        completed_periods = (period - 1) * period_length
        current_period_elapsed = period_length - seconds_in_period
    else:
        # Overtime
        completed_periods = 4 * period_length + (period - 5) * ot_length
        current_period_elapsed = ot_length - seconds_in_period

    return completed_periods + current_period_elapsed


def get_game_seconds_remaining(period: int, pctimestring: str) -> float:
    """
    Calculate seconds remaining in regulation.

    Args:
        period: Current period
        pctimestring: Clock string

    Returns:
        Seconds remaining (can be negative in OT)
    """
    total_regulation = 4 * 12 * 60  # 48 minutes
    elapsed = get_game_seconds_elapsed(period, pctimestring)
    return total_regulation - elapsed


if __name__ == "__main__":
    # Test the collector
    collector = CachedNBADataCollector()

    # Test with a single season, limited games
    data = collector.collect_season_pbp("2023-24", max_games=5)
    print(f"\nCollected {len(data)} games")

    if data:
        sample_pbp = data[0]['pbp']
        print(f"\nSample PBP shape: {sample_pbp.shape}")
        print(f"Columns: {sample_pbp.columns.tolist()}")
