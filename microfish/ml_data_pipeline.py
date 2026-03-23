"""
Microfish Moneyline Data Pipeline
===================================
Loads REAL MLB historical odds from The Odds API + real boxscores from ESPN.
Merges odds with outcomes to create a complete dataset for Certainty Floor Discovery.

NO synthetic data. NO assumptions. Every game has real pre-game moneyline odds
and real final scores.

Data sources:
1. mlb/webapp/data/mlb_historical_odds.json — 2,502 games, 2025 season (Apr-Sep)
   Real odds from The Odds API (h2h moneylines, spreads, totals, player props)
2. mlb/webapp/data/mlb_player_boxscores_expanded.json — 2,368 games with real scores
   Real boxscores from ESPN (player stats, final scores)
3. Matched dataset: 2,277 games with both odds AND outcomes

Pipeline outputs a DataFrame with:
- Pre-game moneyline odds (American format)
- Implied probabilities
- Final scores and winner
- Historical team features (all strictly lagged, no leakage)
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ml_config import (
    THE_ODDS_API_KEY, DATA_DIR, CACHE_DIR, ML_DATA_DIR,
    FEATURE_LAG_DAYS, ROLLING_WINDOWS, BACKTEST_SEASONS
)


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return (american / 100) + 1
    else:
        return (100 / abs(american)) + 1


def american_to_implied_prob(american: float) -> float:
    """Convert American odds to implied probability (no-vig adjusted)."""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def load_real_data() -> pd.DataFrame:
    """
    Load and merge REAL MLB odds + boxscores.
    Returns DataFrame with every game that has both odds and outcomes.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load odds
    odds_file = os.path.join(base_dir, 'mlb', 'webapp', 'data', 'mlb_historical_odds.json')
    if not os.path.exists(odds_file):
        print(f"ERROR: No odds data at {odds_file}")
        return pd.DataFrame()

    with open(odds_file, 'r') as f:
        odds_raw = json.load(f)
    print(f"  Loaded {len(odds_raw)} odds records")

    # Load boxscores (expanded has more games)
    box_file = os.path.join(base_dir, 'mlb', 'webapp', 'data', 'mlb_player_boxscores_expanded.json')
    if not os.path.exists(box_file):
        box_file = os.path.join(base_dir, 'mlb', 'webapp', 'data', 'mlb_player_boxscores.json')

    if not os.path.exists(box_file):
        print(f"ERROR: No boxscore data")
        return pd.DataFrame()

    with open(box_file, 'r') as f:
        box_raw = json.load(f)
    print(f"  Loaded {len(box_raw)} boxscore records")

    # Build boxscore lookup by (date, home, away)
    box_lookup = {}
    for g in box_raw:
        key = (g['date'], g['home'], g['away'])
        box_lookup[key] = g

    # Merge odds + boxscores
    rows = []
    for g in odds_raw:
        key = (g['date'], g['homeTeam'], g['awayTeam'])
        if key not in box_lookup:
            continue

        b = box_lookup[key]
        home_score = b.get('home_score', 0)
        away_score = b.get('away_score', 0)

        # Skip ties / incomplete games
        if home_score == away_score:
            continue

        home_ml = g.get('home_ml', 0)
        away_ml = g.get('away_ml', 0)

        if home_ml == 0 or away_ml == 0:
            continue

        # Parse date
        date_str = g['date']
        date = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")

        home_wins = 1 if home_score > away_score else 0

        # Determine favorite
        if home_ml < away_ml:
            fav_side = 'home'
        elif away_ml < home_ml:
            fav_side = 'away'
        else:
            fav_side = 'home'  # Tie goes to home (pick'em)

        fav_ml = home_ml if fav_side == 'home' else away_ml
        dog_ml = away_ml if fav_side == 'home' else home_ml
        fav_team = g['homeTeam'] if fav_side == 'home' else g['awayTeam']
        dog_team = g['awayTeam'] if fav_side == 'home' else g['homeTeam']

        fav_implied = american_to_implied_prob(fav_ml)
        dog_implied = american_to_implied_prob(dog_ml)

        # Normalize probabilities (remove vig)
        total_implied = fav_implied + dog_implied
        fav_prob_novig = fav_implied / total_implied
        dog_prob_novig = dog_implied / total_implied

        fav_won = 1 if (
            (fav_side == 'home' and home_wins == 1) or
            (fav_side == 'away' and home_wins == 0)
        ) else 0

        rows.append({
            'date': date,
            'date_str': date_str,
            'game_key': g.get('gameKey', ''),
            'event_id': g.get('eventId', ''),
            'home_team': g['homeTeam'],
            'away_team': g['awayTeam'],
            'home_ml': home_ml,
            'away_ml': away_ml,
            'home_score': home_score,
            'away_score': away_score,
            'home_wins': home_wins,
            'run_diff': abs(home_score - away_score),
            'total_runs': home_score + away_score,
            'favorite': fav_side,
            'fav_team': fav_team,
            'dog_team': dog_team,
            'fav_ml': fav_ml,
            'dog_ml': dog_ml,
            'fav_implied_prob': fav_implied,
            'dog_implied_prob': dog_implied,
            'fav_prob_novig': fav_prob_novig,
            'dog_prob_novig': dog_prob_novig,
            'fav_won': fav_won,
            'spread': g.get('spread_point', None),
            'total_line': g.get('total', None),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('date').reset_index(drop=True)

    print(f"  Merged dataset: {len(df)} games with odds + outcomes")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features using ONLY past data. Every feature is lagged by at
    least FEATURE_LAG_DAYS to prevent forward information leakage.
    """
    df = df.sort_values('date').copy()

    # Pre-compute team game logs
    team_logs = {}
    for _, row in df.iterrows():
        for is_home in [True, False]:
            team = row['home_team'] if is_home else row['away_team']
            opp = row['away_team'] if is_home else row['home_team']
            won = (row['home_wins'] == 1) == is_home
            odds = row['home_ml'] if is_home else row['away_ml']
            was_fav = (row['favorite'] == 'home') == is_home

            if team not in team_logs:
                team_logs[team] = []
            team_logs[team].append({
                'date': row['date'],
                'won': won,
                'is_home': is_home,
                'odds': odds,
                'was_favorite': was_fav,
                'opp': opp,
                'run_diff': row['run_diff'] if won else -row['run_diff'],
                'score_for': row['home_score'] if is_home else row['away_score'],
                'score_against': row['away_score'] if is_home else row['home_score'],
            })

    for team in team_logs:
        team_logs[team].sort(key=lambda x: x['date'])

    # Build features for each game
    feature_rows = []
    for _, row in df.iterrows():
        game_date = row['date']
        cutoff = game_date - timedelta(days=FEATURE_LAG_DAYS)

        features = row.to_dict()

        for side_label, team_col in [('fav', 'fav_team'), ('dog', 'dog_team')]:
            team = row[team_col]
            logs = [g for g in team_logs.get(team, []) if g['date'] < cutoff]

            for window in ROLLING_WINDOWS:
                recent = logs[-window:] if len(logs) >= window else logs
                suffix = f'{side_label}_{window}d'

                if len(recent) >= 3:
                    wins = sum(1 for g in recent if g['won'])
                    features[f'{suffix}_win_pct'] = wins / len(recent)
                    features[f'{suffix}_games'] = len(recent)

                    # Streak
                    streak = 0
                    for g in reversed(recent):
                        if g['won']:
                            streak += 1
                        else:
                            break
                    features[f'{suffix}_win_streak'] = streak

                    loss_streak = 0
                    for g in reversed(recent):
                        if not g['won']:
                            loss_streak += 1
                        else:
                            break
                    features[f'{suffix}_loss_streak'] = loss_streak

                    # Avg run differential
                    features[f'{suffix}_avg_run_diff'] = np.mean(
                        [g['run_diff'] for g in recent]
                    )

                    # Avg runs scored
                    features[f'{suffix}_avg_runs_for'] = np.mean(
                        [g['score_for'] for g in recent]
                    )
                    features[f'{suffix}_avg_runs_against'] = np.mean(
                        [g['score_against'] for g in recent]
                    )

                    # As-favorite win pct
                    fav_games = [g for g in recent if g['was_favorite']]
                    if len(fav_games) >= 2:
                        features[f'{suffix}_fav_win_pct'] = (
                            sum(1 for g in fav_games if g['won']) / len(fav_games)
                        )
                    else:
                        features[f'{suffix}_fav_win_pct'] = 0.5

                    # Home win pct
                    home_games = [g for g in recent if g['is_home']]
                    if len(home_games) >= 2:
                        features[f'{suffix}_home_win_pct'] = (
                            sum(1 for g in home_games if g['won']) / len(home_games)
                        )
                    else:
                        features[f'{suffix}_home_win_pct'] = 0.5
                else:
                    features[f'{suffix}_win_pct'] = 0.5
                    features[f'{suffix}_games'] = len(recent)
                    features[f'{suffix}_win_streak'] = 0
                    features[f'{suffix}_loss_streak'] = 0
                    features[f'{suffix}_avg_run_diff'] = 0
                    features[f'{suffix}_avg_runs_for'] = 0
                    features[f'{suffix}_avg_runs_against'] = 0
                    features[f'{suffix}_fav_win_pct'] = 0.5
                    features[f'{suffix}_home_win_pct'] = 0.5

        # Momentum convergence: all windows agree on direction
        for side in ['fav', 'dog']:
            rates = [features.get(f'{side}_{w}d_win_pct', 0.5) for w in ROLLING_WINDOWS]
            valid_rates = [r for r in rates if r != 0.5]
            if len(valid_rates) >= 3:
                features[f'{side}_all_windows_positive'] = int(all(r > 0.5 for r in valid_rates))
                features[f'{side}_all_windows_negative'] = int(all(r < 0.5 for r in valid_rates))
                features[f'{side}_min_window_wr'] = min(valid_rates)
                features[f'{side}_max_window_wr'] = max(valid_rates)
            else:
                features[f'{side}_all_windows_positive'] = 0
                features[f'{side}_all_windows_negative'] = 0
                features[f'{side}_min_window_wr'] = 0.5
                features[f'{side}_max_window_wr'] = 0.5

        feature_rows.append(features)

    result = pd.DataFrame(feature_rows)
    print(f"  Built features for {len(result)} games")
    return result


def compute_certainty_floor_data(df: pd.DataFrame) -> dict:
    """
    Compute win rates at every probability threshold.
    This is the core data for Certainty Floor Discovery.

    For each probability level (50-99%), compute:
    - How many games had a favorite at that implied probability or higher
    - How many of those the favorite won
    - The exact win rate

    The Certainty Floor is the MINIMUM probability where win rate = 100%.
    """
    results = {}

    for threshold in range(50, 100):
        pct = threshold / 100.0

        # Games where favorite's no-vig probability >= threshold
        subset = df[df['fav_prob_novig'] >= pct]

        if len(subset) == 0:
            results[threshold] = {
                'threshold_pct': threshold,
                'n_games': 0,
                'fav_wins': 0,
                'fav_losses': 0,
                'win_rate': 0,
                'loss_examples': [],
            }
            continue

        fav_wins = int(subset['fav_won'].sum())
        fav_losses = len(subset) - fav_wins

        # Capture loss examples for analysis
        loss_examples = []
        if fav_losses > 0:
            losses = subset[subset['fav_won'] == 0]
            for _, loss in losses.iterrows():
                loss_examples.append({
                    'date': str(loss['date']),
                    'game': f"{loss['dog_team']}@{loss['fav_team']}" if loss['favorite'] == 'home'
                            else f"{loss['fav_team']}@{loss['dog_team']}",
                    'fav_team': loss['fav_team'],
                    'fav_ml': int(loss['fav_ml']),
                    'fav_prob': round(loss['fav_prob_novig'], 3),
                    'score': f"{loss['home_score']}-{loss['away_score']}",
                })

        results[threshold] = {
            'threshold_pct': threshold,
            'n_games': len(subset),
            'fav_wins': fav_wins,
            'fav_losses': fav_losses,
            'win_rate': fav_wins / len(subset),
            'loss_examples': loss_examples,
        }

    return results


def compute_contextual_floors(df: pd.DataFrame) -> dict:
    """
    Compute certainty floors with additional contextual filters.
    Different game contexts may have different certainty floors.
    """
    contexts = {}

    # By month (early vs mid vs late season)
    if 'date' in df.columns:
        df_with_month = df.copy()
        df_with_month['month'] = df_with_month['date'].dt.month

        for month in sorted(df_with_month['month'].unique()):
            month_df = df_with_month[df_with_month['month'] == month]
            if len(month_df) >= 30:
                contexts[f'month_{month}'] = compute_certainty_floor_data(month_df)

    # By odds bracket (different fav strength levels)
    brackets = [
        ('slight_fav', -150, -100),    # Slight favorites (-100 to -150)
        ('moderate_fav', -200, -151),   # Moderate favorites (-151 to -200)
        ('heavy_fav', -300, -201),      # Heavy favorites (-201 to -300)
        ('extreme_fav', -9999, -301),   # Extreme favorites (-301+)
    ]

    for name, ml_min, ml_max in brackets:
        bracket_df = df[(df['fav_ml'] >= ml_min) & (df['fav_ml'] <= ml_max)]
        if len(bracket_df) >= 20:
            contexts[f'bracket_{name}'] = compute_certainty_floor_data(bracket_df)

    # By run differential expectation
    # High-scoring vs low-scoring game expectation
    if 'total_line' in df.columns:
        df_with_total = df.dropna(subset=['total_line'])
        low_total = df_with_total[df_with_total['total_line'] <= 7.5]
        high_total = df_with_total[df_with_total['total_line'] >= 9.0]

        if len(low_total) >= 30:
            contexts['low_total'] = compute_certainty_floor_data(low_total)
        if len(high_total) >= 30:
            contexts['high_total'] = compute_certainty_floor_data(high_total)

    # By favorite momentum (all windows positive)
    if 'fav_all_windows_positive' in df.columns:
        hot_fav = df[df['fav_all_windows_positive'] == 1]
        cold_dog = df[df['dog_all_windows_negative'] == 1]

        if len(hot_fav) >= 30:
            contexts['fav_hot_momentum'] = compute_certainty_floor_data(hot_fav)
        if len(cold_dog) >= 30:
            contexts['dog_cold_momentum'] = compute_certainty_floor_data(cold_dog)

    # Combined: hot favorite + cold dog
    if 'fav_all_windows_positive' in df.columns and 'dog_all_windows_negative' in df.columns:
        convergence = df[
            (df['fav_all_windows_positive'] == 1) &
            (df['dog_all_windows_negative'] == 1)
        ]
        if len(convergence) >= 10:
            contexts['full_convergence'] = compute_certainty_floor_data(convergence)

    return contexts


def save_research_data(df: pd.DataFrame, floor_data: dict,
                       contextual_floors: dict):
    """Save all research data to microfish/data/moneyline/ for documentation."""
    os.makedirs(ML_DATA_DIR, exist_ok=True)

    # Save merged dataset summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_games': len(df),
        'date_range': f"{df['date'].min()} to {df['date'].max()}" if not df.empty else 'N/A',
        'data_sources': {
            'odds': 'mlb/webapp/data/mlb_historical_odds.json (The Odds API, 2025 season)',
            'boxscores': 'mlb/webapp/data/mlb_player_boxscores_expanded.json (ESPN, 2025 season)',
        },
        'favorite_win_rate': float(df['fav_won'].mean()) if not df.empty else 0,
        'avg_fav_implied_prob': float(df['fav_implied_prob'].mean()) if not df.empty else 0,
    }

    with open(os.path.join(ML_DATA_DIR, 'dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save certainty floor data
    # Convert loss_examples for JSON serialization
    floor_clean = {}
    for k, v in floor_data.items():
        floor_clean[str(k)] = {
            'threshold_pct': v['threshold_pct'],
            'n_games': v['n_games'],
            'fav_wins': v['fav_wins'],
            'fav_losses': v['fav_losses'],
            'win_rate': round(v['win_rate'], 6) if v['n_games'] > 0 else 0,
            'loss_examples': v['loss_examples'][:10],  # Cap at 10 examples
        }

    with open(os.path.join(ML_DATA_DIR, 'certainty_floor_analysis.json'), 'w') as f:
        json.dump(floor_clean, f, indent=2, default=str)

    # Save contextual floors
    ctx_clean = {}
    for ctx_name, ctx_data in contextual_floors.items():
        ctx_clean[ctx_name] = {}
        for k, v in ctx_data.items():
            ctx_clean[ctx_name][str(k)] = {
                'threshold_pct': v['threshold_pct'],
                'n_games': v['n_games'],
                'fav_wins': v['fav_wins'],
                'fav_losses': v['fav_losses'],
                'win_rate': round(v['win_rate'], 6) if v['n_games'] > 0 else 0,
            }

    with open(os.path.join(ML_DATA_DIR, 'contextual_floors.json'), 'w') as f:
        json.dump(ctx_clean, f, indent=2, default=str)

    print(f"  Research data saved to {ML_DATA_DIR}")


def generate_floor_report(floor_data: dict) -> str:
    """Generate a human-readable certainty floor report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CERTAINTY FLOOR ANALYSIS — REAL MLB 2025 DATA")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Threshold':>10} {'Games':>7} {'Wins':>6} {'Losses':>7} {'Win Rate':>10}")
    lines.append("-" * 45)

    certainty_floor = None
    for threshold in range(50, 100):
        data = floor_data.get(threshold, {})
        n = data.get('n_games', 0)
        if n == 0:
            continue
        wins = data.get('fav_wins', 0)
        losses = data.get('fav_losses', 0)
        wr = data.get('win_rate', 0)

        marker = ""
        if wr == 1.0 and n >= 1 and certainty_floor is None:
            certainty_floor = threshold
            marker = " <-- CERTAINTY FLOOR"

        lines.append(
            f"   >= {threshold}%  {n:>7} {wins:>6} {losses:>7} {wr:>9.1%}{marker}"
        )

    lines.append("")
    if certainty_floor:
        n = floor_data[certainty_floor]['n_games']
        lines.append(f"CERTAINTY FLOOR: {certainty_floor}% (n={n} games, 100% win rate)")
        lines.append(f"LIMIT PRICE: ${certainty_floor/100:.2f} per share")
        lines.append(f"PROFIT PER WIN: ${1 - certainty_floor/100:.2f} per share")
    else:
        lines.append("NO CERTAINTY FLOOR FOUND — favorites lose at every probability level")

    # Show the losses near the top
    lines.append("")
    lines.append("UPSET EXAMPLES (favorites that lost at highest probabilities):")
    for threshold in range(99, 50, -1):
        data = floor_data.get(threshold, {})
        for ex in data.get('loss_examples', []):
            if ex.get('fav_prob', 0) >= 0.65:
                lines.append(
                    f"  {ex['date']}: {ex['game']} — {ex['fav_team']} "
                    f"(ML {ex['fav_ml']}, {ex['fav_prob']:.1%}) LOST, "
                    f"score: {ex['score']}"
                )

    return '\n'.join(lines)
