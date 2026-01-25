"""
Feature Engineering for NBA In-Game Trading Strategy

This module processes play-by-play data to create features that are
available at decision time (no look-ahead bias).

Key Feature Categories:
1. Game State: Score, time, period
2. Momentum: Scoring runs, possession efficiency trends
3. Foul Situation: Team fouls, bonus status
4. Lineup/Fatigue Proxies: Timeout patterns, substitution rates
5. Shot Quality: Recent shot selection patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from data_collector import (
    parse_game_clock,
    get_game_seconds_elapsed,
    get_game_seconds_remaining
)


@dataclass
class GameState:
    """Represents the state of a game at a specific moment."""
    game_id: str
    period: int
    clock_seconds: float  # Seconds remaining in period
    game_seconds_elapsed: float
    game_seconds_remaining: float  # Remaining in regulation

    home_score: int
    away_score: int
    score_diff: int  # Home - Away (positive = home leading)

    # Momentum features (computed over rolling windows)
    home_run_last_2min: int  # Points scored by home in last 2 mins
    away_run_last_2min: int
    home_run_last_5min: int
    away_run_last_5min: int

    # Possession-level efficiency (last N possessions)
    home_ppp_last_10: float  # Points per possession, last 10
    away_ppp_last_10: float

    # Foul situation
    home_team_fouls: int  # Current period fouls
    away_team_fouls: int
    home_in_bonus: bool
    away_in_bonus: bool

    # Timeout situation
    home_timeouts_remaining: int
    away_timeouts_remaining: int

    # Recent event patterns
    consecutive_home_scores: int  # Consecutive scoring plays by home
    consecutive_away_scores: int
    home_fg_pct_last_5min: float  # FG% in last 5 minutes
    away_fg_pct_last_5min: float

    # Lead changes and ties
    lead_changes_this_half: int
    times_tied_this_half: int

    # Shot clock proxy (last possession outcome)
    last_possession_home: bool
    last_play_type: str  # "made_fg", "missed_fg", "turnover", "free_throw", etc.


class PBPFeatureExtractor:
    """
    Extracts trading-relevant features from play-by-play data.

    All features are computed using only information available at the
    decision timestamp (no look-ahead bias).
    """

    # Event type mappings from nba_api
    MADE_SHOT_TYPES = [1, 2]  # Made field goal, made free throw
    MISSED_SHOT_TYPES = [2]  # Missed FG
    FREE_THROW_TYPE = 3
    REBOUND_TYPE = 4
    TURNOVER_TYPE = 5
    FOUL_TYPE = 6
    SUBSTITUTION_TYPE = 8
    TIMEOUT_TYPE = 9
    JUMP_BALL_TYPE = 10
    PERIOD_START_TYPE = 12
    PERIOD_END_TYPE = 13

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def extract_game_states(self, pbp_df: pd.DataFrame,
                           home_team_id: int,
                           away_team_id: int,
                           sample_interval_seconds: float = 30.0) -> List[GameState]:
        """
        Extract game states at regular intervals throughout the game.

        Args:
            pbp_df: Play-by-play DataFrame for a single game
            home_team_id: Home team ID
            away_team_id: Away team ID
            sample_interval_seconds: How often to sample game state

        Returns:
            List of GameState objects at each sample point
        """
        if pbp_df.empty:
            return []

        # Sort by event order
        pbp_df = pbp_df.sort_values(['PERIOD', 'EVENTNUM']).reset_index(drop=True)

        # Pre-compute event timestamps
        pbp_df['GAME_SECONDS'] = pbp_df.apply(
            lambda r: get_game_seconds_elapsed(r['PERIOD'], r['PCTIMESTRING']),
            axis=1
        )

        # Identify scoring events
        pbp_df['IS_HOME_SCORE'] = (
            (pbp_df['EVENTMSGTYPE'] == 1) &  # Made field goal
            (pbp_df['PLAYER1_TEAM_ID'] == home_team_id)
        ) | (
            (pbp_df['EVENTMSGTYPE'] == 3) &  # Free throw
            (pbp_df['PLAYER1_TEAM_ID'] == home_team_id) &
            (pbp_df['EVENTMSGACTIONTYPE'].isin([10, 12, 15]))  # Made FT types
        )

        pbp_df['IS_AWAY_SCORE'] = (
            (pbp_df['EVENTMSGTYPE'] == 1) &
            (pbp_df['PLAYER1_TEAM_ID'] == away_team_id)
        ) | (
            (pbp_df['EVENTMSGTYPE'] == 3) &
            (pbp_df['PLAYER1_TEAM_ID'] == away_team_id) &
            (pbp_df['EVENTMSGACTIONTYPE'].isin([10, 12, 15]))
        )

        # Extract points from score columns
        pbp_df = self._parse_scores(pbp_df, home_team_id, away_team_id)

        # Generate sample times
        game_id = pbp_df['GAME_ID'].iloc[0] if 'GAME_ID' in pbp_df.columns else "unknown"
        max_seconds = pbp_df['GAME_SECONDS'].max()

        # Sample every interval_seconds, but only at valid play times
        sample_times = np.arange(60, max_seconds, sample_interval_seconds)

        states = []
        for sample_time in sample_times:
            state = self._compute_state_at_time(
                pbp_df, sample_time, game_id, home_team_id, away_team_id
            )
            if state is not None:
                states.append(state)

        return states

    def _parse_scores(self, pbp_df: pd.DataFrame,
                     home_team_id: int,
                     away_team_id: int) -> pd.DataFrame:
        """Parse score columns to get home/away scores at each event."""

        def parse_score(score_str):
            if pd.isna(score_str):
                return None, None
            try:
                parts = str(score_str).split(' - ')
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
            except:
                pass
            return None, None

        # SCORE column format is "AWAY - HOME"
        scores = pbp_df['SCORE'].apply(lambda x: pd.Series(parse_score(x)))
        pbp_df['AWAY_SCORE'] = scores[0]
        pbp_df['HOME_SCORE'] = scores[1]

        # Forward fill scores
        pbp_df['AWAY_SCORE'] = pbp_df['AWAY_SCORE'].fillna(method='ffill').fillna(0).astype(int)
        pbp_df['HOME_SCORE'] = pbp_df['HOME_SCORE'].fillna(method='ffill').fillna(0).astype(int)

        return pbp_df

    def _compute_state_at_time(self, pbp_df: pd.DataFrame,
                               target_time: float,
                               game_id: str,
                               home_team_id: int,
                               away_team_id: int) -> Optional[GameState]:
        """
        Compute game state at a specific time using only past information.

        Args:
            pbp_df: Full game PBP with computed columns
            target_time: Game seconds at which to compute state
            game_id: Game identifier
            home_team_id: Home team ID
            away_team_id: Away team ID

        Returns:
            GameState object or None if insufficient data
        """
        # Get all events up to target time
        past_events = pbp_df[pbp_df['GAME_SECONDS'] <= target_time]

        if past_events.empty:
            return None

        last_event = past_events.iloc[-1]

        # Basic game state
        period = int(last_event['PERIOD'])
        clock_str = last_event['PCTIMESTRING']
        clock_seconds = parse_game_clock(clock_str)

        home_score = int(last_event['HOME_SCORE'])
        away_score = int(last_event['AWAY_SCORE'])
        score_diff = home_score - away_score

        # Compute momentum features - runs in last 2 and 5 minutes
        two_min_ago = target_time - 120
        five_min_ago = target_time - 300

        events_last_2min = past_events[past_events['GAME_SECONDS'] > two_min_ago]
        events_last_5min = past_events[past_events['GAME_SECONDS'] > five_min_ago]

        home_run_2min, away_run_2min = self._compute_scoring_run(
            events_last_2min, home_team_id, away_team_id
        )
        home_run_5min, away_run_5min = self._compute_scoring_run(
            events_last_5min, home_team_id, away_team_id
        )

        # Points per possession (approximate using scoring plays / possessions)
        home_ppp_10, away_ppp_10 = self._compute_recent_ppp(
            past_events, home_team_id, away_team_id, n_possessions=10
        )

        # Foul situation (current period only)
        current_period_events = past_events[past_events['PERIOD'] == period]
        home_fouls, away_fouls = self._count_team_fouls(
            current_period_events, home_team_id, away_team_id
        )

        # Bonus status (5 fouls in period or 2 in last 2 min of period)
        home_in_bonus = away_fouls >= 5
        away_in_bonus = home_fouls >= 5

        # Timeout situation (approximate - start with 7)
        home_timeouts = self._count_timeouts_remaining(past_events, home_team_id)
        away_timeouts = self._count_timeouts_remaining(past_events, away_team_id)

        # Consecutive scoring
        consec_home, consec_away = self._count_consecutive_scores(
            past_events, home_team_id, away_team_id
        )

        # Recent FG%
        home_fg_pct = self._compute_recent_fg_pct(events_last_5min, home_team_id)
        away_fg_pct = self._compute_recent_fg_pct(events_last_5min, away_team_id)

        # Lead changes and ties this half
        half = 1 if period <= 2 else 2
        half_start = 0 if half == 1 else 24 * 60
        half_events = past_events[past_events['GAME_SECONDS'] >= half_start]
        lead_changes, times_tied = self._count_lead_changes_and_ties(half_events)

        # Last possession info
        last_poss_home, last_play_type = self._get_last_possession_info(
            past_events, home_team_id, away_team_id
        )

        return GameState(
            game_id=game_id,
            period=period,
            clock_seconds=clock_seconds,
            game_seconds_elapsed=target_time,
            game_seconds_remaining=48*60 - target_time,
            home_score=home_score,
            away_score=away_score,
            score_diff=score_diff,
            home_run_last_2min=home_run_2min,
            away_run_last_2min=away_run_2min,
            home_run_last_5min=home_run_5min,
            away_run_last_5min=away_run_5min,
            home_ppp_last_10=home_ppp_10,
            away_ppp_last_10=away_ppp_10,
            home_team_fouls=home_fouls,
            away_team_fouls=away_fouls,
            home_in_bonus=home_in_bonus,
            away_in_bonus=away_in_bonus,
            home_timeouts_remaining=home_timeouts,
            away_timeouts_remaining=away_timeouts,
            consecutive_home_scores=consec_home,
            consecutive_away_scores=consec_away,
            home_fg_pct_last_5min=home_fg_pct,
            away_fg_pct_last_5min=away_fg_pct,
            lead_changes_this_half=lead_changes,
            times_tied_this_half=times_tied,
            last_possession_home=last_poss_home,
            last_play_type=last_play_type
        )

    def _compute_scoring_run(self, events: pd.DataFrame,
                            home_team_id: int,
                            away_team_id: int) -> Tuple[int, int]:
        """Compute points scored by each team in given events."""
        if events.empty:
            return 0, 0

        home_pts = 0
        away_pts = 0

        for _, event in events.iterrows():
            if event['EVENTMSGTYPE'] == 1:  # Made field goal
                # Determine points from description or action type
                pts = 2
                desc = str(event.get('HOMEDESCRIPTION', '')) + str(event.get('VISITORDESCRIPTION', ''))
                if '3PT' in desc.upper():
                    pts = 3

                if event['PLAYER1_TEAM_ID'] == home_team_id:
                    home_pts += pts
                elif event['PLAYER1_TEAM_ID'] == away_team_id:
                    away_pts += pts

            elif event['EVENTMSGTYPE'] == 3:  # Free throw
                # Check if made (action types 10, 12, 15 are made FTs)
                if event['EVENTMSGACTIONTYPE'] in [10, 12, 15]:
                    if event['PLAYER1_TEAM_ID'] == home_team_id:
                        home_pts += 1
                    elif event['PLAYER1_TEAM_ID'] == away_team_id:
                        away_pts += 1

        return home_pts, away_pts

    def _compute_recent_ppp(self, events: pd.DataFrame,
                           home_team_id: int,
                           away_team_id: int,
                           n_possessions: int = 10) -> Tuple[float, float]:
        """
        Estimate points per possession for last N possessions.

        A possession ends on: made FG, defensive rebound after miss,
        turnover, or made final FT.
        """
        # Simplified: count scoring events and possession-ending events
        home_pts = 0
        away_pts = 0
        home_poss = 0
        away_poss = 0

        possession_ending_types = [1, 4, 5]  # Made FG, rebound, turnover

        for _, event in events.tail(100).iterrows():  # Look at recent events
            evt_type = event['EVENTMSGTYPE']
            team_id = event.get('PLAYER1_TEAM_ID')

            if evt_type == 1:  # Made FG
                pts = 3 if '3PT' in str(event.get('HOMEDESCRIPTION', '')) + str(event.get('VISITORDESCRIPTION', '')) else 2
                if team_id == home_team_id:
                    home_pts += pts
                    home_poss += 1
                elif team_id == away_team_id:
                    away_pts += pts
                    away_poss += 1

            elif evt_type == 5:  # Turnover
                if team_id == home_team_id:
                    home_poss += 1
                elif team_id == away_team_id:
                    away_poss += 1

        home_ppp = home_pts / max(home_poss, 1)
        away_ppp = away_pts / max(away_poss, 1)

        return round(home_ppp, 2), round(away_ppp, 2)

    def _count_team_fouls(self, events: pd.DataFrame,
                         home_team_id: int,
                         away_team_id: int) -> Tuple[int, int]:
        """Count team fouls in given events."""
        foul_events = events[events['EVENTMSGTYPE'] == 6]

        home_fouls = len(foul_events[foul_events['PLAYER1_TEAM_ID'] == home_team_id])
        away_fouls = len(foul_events[foul_events['PLAYER1_TEAM_ID'] == away_team_id])

        return home_fouls, away_fouls

    def _count_timeouts_remaining(self, events: pd.DataFrame, team_id: int) -> int:
        """Estimate timeouts remaining for a team."""
        # NBA teams get 7 timeouts per game (changed in 2017-18)
        timeout_events = events[
            (events['EVENTMSGTYPE'] == 9) &
            (events['PLAYER1_TEAM_ID'] == team_id)
        ]
        return max(7 - len(timeout_events), 0)

    def _count_consecutive_scores(self, events: pd.DataFrame,
                                  home_team_id: int,
                                  away_team_id: int) -> Tuple[int, int]:
        """Count consecutive scoring plays for each team."""
        scoring_events = events[events['EVENTMSGTYPE'].isin([1, 3])]

        if scoring_events.empty:
            return 0, 0

        # Get last few scoring events
        recent_scores = scoring_events.tail(10)

        consec_home = 0
        consec_away = 0

        # Count from most recent going back
        for _, event in recent_scores.iloc[::-1].iterrows():
            if event['PLAYER1_TEAM_ID'] == home_team_id:
                if consec_away > 0:
                    break
                consec_home += 1
            elif event['PLAYER1_TEAM_ID'] == away_team_id:
                if consec_home > 0:
                    break
                consec_away += 1

        return consec_home, consec_away

    def _compute_recent_fg_pct(self, events: pd.DataFrame, team_id: int) -> float:
        """Compute FG% for a team in given events."""
        # Field goal attempts: type 1 (made) or type 2 (missed)
        fg_events = events[
            (events['EVENTMSGTYPE'].isin([1, 2])) &
            (events['PLAYER1_TEAM_ID'] == team_id)
        ]

        if len(fg_events) == 0:
            return 0.5  # Default to league average

        made = len(fg_events[fg_events['EVENTMSGTYPE'] == 1])
        return round(made / len(fg_events), 3)

    def _count_lead_changes_and_ties(self, events: pd.DataFrame) -> Tuple[int, int]:
        """Count lead changes and ties in given events."""
        if events.empty:
            return 0, 0

        scores = events[['HOME_SCORE', 'AWAY_SCORE']].dropna()
        if len(scores) < 2:
            return 0, 0

        scores['DIFF'] = scores['HOME_SCORE'] - scores['AWAY_SCORE']
        scores['LEAD_SIGN'] = np.sign(scores['DIFF'])

        lead_changes = (scores['LEAD_SIGN'].diff() != 0).sum()
        times_tied = (scores['DIFF'] == 0).sum()

        return int(lead_changes), int(times_tied)

    def _get_last_possession_info(self, events: pd.DataFrame,
                                  home_team_id: int,
                                  away_team_id: int) -> Tuple[bool, str]:
        """Get information about the last possession."""
        # Look at last few events to determine possession and outcome
        recent = events.tail(5)

        last_poss_home = True  # Default
        last_play_type = "unknown"

        for _, event in recent.iloc[::-1].iterrows():
            evt_type = event['EVENTMSGTYPE']
            team_id = event.get('PLAYER1_TEAM_ID')

            if evt_type == 1:  # Made FG
                last_poss_home = (team_id == home_team_id)
                last_play_type = "made_fg"
                break
            elif evt_type == 2:  # Missed FG
                last_poss_home = (team_id == home_team_id)
                last_play_type = "missed_fg"
                break
            elif evt_type == 3:  # Free throw
                last_poss_home = (team_id == home_team_id)
                last_play_type = "free_throw"
                break
            elif evt_type == 5:  # Turnover
                last_poss_home = (team_id == home_team_id)
                last_play_type = "turnover"
                break

        return last_poss_home, last_play_type


def states_to_dataframe(states: List[GameState]) -> pd.DataFrame:
    """Convert list of GameState objects to DataFrame."""
    if not states:
        return pd.DataFrame()

    records = []
    for state in states:
        records.append({
            'game_id': state.game_id,
            'period': state.period,
            'clock_seconds': state.clock_seconds,
            'game_seconds_elapsed': state.game_seconds_elapsed,
            'game_seconds_remaining': state.game_seconds_remaining,
            'home_score': state.home_score,
            'away_score': state.away_score,
            'score_diff': state.score_diff,
            'home_run_2min': state.home_run_last_2min,
            'away_run_2min': state.away_run_last_2min,
            'home_run_5min': state.home_run_last_5min,
            'away_run_5min': state.away_run_last_5min,
            'run_diff_2min': state.home_run_last_2min - state.away_run_last_2min,
            'run_diff_5min': state.home_run_last_5min - state.away_run_last_5min,
            'home_ppp_10': state.home_ppp_last_10,
            'away_ppp_10': state.away_ppp_last_10,
            'ppp_diff': state.home_ppp_last_10 - state.away_ppp_last_10,
            'home_fouls': state.home_team_fouls,
            'away_fouls': state.away_team_fouls,
            'home_in_bonus': int(state.home_in_bonus),
            'away_in_bonus': int(state.away_in_bonus),
            'home_timeouts': state.home_timeouts_remaining,
            'away_timeouts': state.away_timeouts_remaining,
            'consec_home_scores': state.consecutive_home_scores,
            'consec_away_scores': state.consecutive_away_scores,
            'home_fg_pct_5min': state.home_fg_pct_last_5min,
            'away_fg_pct_5min': state.away_fg_pct_last_5min,
            'lead_changes': state.lead_changes_this_half,
            'times_tied': state.times_tied_this_half,
            'last_poss_home': int(state.last_possession_home),
            'last_play_type': state.last_play_type
        })

    return pd.DataFrame(records)


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features useful for trading signals.

    Args:
        df: DataFrame with basic game states

    Returns:
        DataFrame with additional derived features
    """
    df = df.copy()

    # Time-based features
    df['quarter'] = df['period'].clip(upper=4)
    df['is_overtime'] = (df['period'] > 4).astype(int)
    df['minutes_remaining'] = df['game_seconds_remaining'] / 60
    df['minutes_elapsed'] = df['game_seconds_elapsed'] / 60

    # Period-specific time
    df['time_in_period'] = (12 * 60) - df['clock_seconds']
    df['period_pct_complete'] = df['time_in_period'] / (12 * 60)

    # Score context
    df['total_score'] = df['home_score'] + df['away_score']
    df['abs_score_diff'] = df['score_diff'].abs()

    # Momentum indicators
    df['momentum_2min'] = df['run_diff_2min']  # Positive = home momentum
    df['momentum_5min'] = df['run_diff_5min']

    # Run severity (how big is the recent run?)
    df['max_run_2min'] = df[['home_run_2min', 'away_run_2min']].max(axis=1)
    df['run_differential_2min'] = df['home_run_2min'] - df['away_run_2min']

    # Trailing team on a run (potential comeback signal)
    df['trailing_team_momentum'] = np.where(
        df['score_diff'] > 0,  # Home leading
        -df['momentum_2min'],  # Negative if away has momentum
        df['momentum_2min']    # Positive if home (trailing) has momentum
    )

    # Leading team on a run (potential blowout signal)
    df['leading_team_momentum'] = np.where(
        df['score_diff'] > 0,  # Home leading
        df['momentum_2min'],   # Positive if home (leading) has momentum
        -df['momentum_2min']   # Negative if away (leading) has momentum
    )

    # Efficiency gap
    df['efficiency_gap'] = df['home_ppp_10'] - df['away_ppp_10']

    # FG% differential
    df['fg_pct_diff'] = df['home_fg_pct_5min'] - df['away_fg_pct_5min']

    # Foul differential (useful for predicting bonus FTs)
    df['foul_diff'] = df['away_fouls'] - df['home_fouls']  # Positive = home benefits

    # Close game indicator
    df['is_close_game'] = (df['abs_score_diff'] <= 10).astype(int)
    df['is_very_close'] = (df['abs_score_diff'] <= 5).astype(int)

    # Blowout indicator (for filtering garbage time)
    df['is_blowout'] = (df['abs_score_diff'] > 20).astype(int)

    # Game phase
    df['game_phase'] = pd.cut(
        df['minutes_elapsed'],
        bins=[0, 12, 24, 36, 48, 100],
        labels=['Q1', 'Q2', 'Q3', 'Q4', 'OT']
    )

    # Late game indicators
    df['is_clutch_time'] = (
        (df['minutes_remaining'] <= 5) &
        (df['abs_score_diff'] <= 5)
    ).astype(int)

    df['is_late_game'] = (df['minutes_remaining'] <= 6).astype(int)

    # Streak indicators
    df['home_on_streak'] = (df['consec_home_scores'] >= 3).astype(int)
    df['away_on_streak'] = (df['consec_away_scores'] >= 3).astype(int)

    return df


if __name__ == "__main__":
    # Test feature extraction
    from data_collector import CachedNBADataCollector

    collector = CachedNBADataCollector()
    extractor = PBPFeatureExtractor()

    # Get a sample game
    data = collector.collect_season_pbp("2023-24", max_games=1)

    if data:
        game = data[0]
        pbp = game['pbp']
        game_info = game['game_info']

        # Get team IDs
        home_team_id = game_info.get('TEAM_ID', 0)

        # For away team, we need to look at the line score
        print(f"Game: {game['game_id']}")
        print(f"PBP shape: {pbp.shape}")

        # Extract states (using placeholder team IDs for test)
        states = extractor.extract_game_states(pbp, home_team_id, 0)
        df = states_to_dataframe(states)
        df = compute_derived_features(df)

        print(f"\nExtracted {len(df)} game states")
        print(f"Columns: {df.columns.tolist()}")
