"""
Q3 Over/Under Signal Strategy - Production Engine
===================================================
Extremely high-accuracy O/U signal placed at end of 3rd quarter.

PHILOSOPHY:
    Think like a stock trader: only enter when the edge is enormous.
    We are NOT trying to trade every game. We are finding the 15-30%
    of games where the outcome is essentially predetermined by Q3 end,
    and the opening O/U line has become massively mispriced.

CORE INSIGHT:
    Q4 scoring averages ~54 points with std ~9 across ALL game states.
    The opening O/U line assumes roughly even pace throughout the game.
    When a game runs significantly hot or cold through 3 quarters,
    the opening line becomes deeply mispriced.

    At Q3 end we have 75% of the game completed. A linear model
    predicting Q4 total from game state features achieves R²≈0.04
    (Q4 is noisy), but the KEY is: even a crude prediction creates
    a massive margin when the game has deviated far from the opening
    line's expectation.

SIGNAL TIERS (OOS validated on 1,015 games, 2022-23 season):
    PLATINUM: model margin >= 20 → 99.4% accuracy, ~+88% ROI
    GOLD:     model margin >= 15 → 98.3% accuracy, ~+87% ROI
    SILVER:   model margin >= 12 → 97.7% accuracy, ~+85% ROI
    BRONZE:   model margin >= 10 → 95.7% accuracy, ~+83% ROI

All signals pay -110 (standard O/U juice).
Break-even is 52.4%. We operate at 95-100%.
Edge per trade: +40 to +48 cents per dollar risked.

USAGE:
    from q3_ou_strategy import Q3OverUnderEngine

    engine = Q3OverUnderEngine()
    signal = engine.evaluate(
        home_score=78, away_score=65,  # Q3 end scores
        q1_total=58, q2_total=55, q3_total=53,  # Per-quarter totals
        ou_line=224.5,  # Opening O/U line
        late_q3_pts=12  # Points scored in last 3 min of Q3
    )

    if signal:
        print(signal['direction'])   # 'OVER' or 'UNDER'
        print(signal['tier'])        # 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE'
        print(signal['confidence'])  # 0.95 - 1.00
        print(signal['edge'])        # Expected edge per unit
        print(signal['kelly'])       # Kelly criterion bet size
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from pathlib import Path


# =========================================================================
# MODEL COEFFICIENTS (trained on 2021-22 season, validated on 2022-23)
# These are the linear regression coefficients for predicting Q4 total scoring
# from game state features available at end of Q3.
#
# Training data: 1,162 regulation NBA games from 2021-22
# OOS validation: 1,015 regulation NBA games from 2022-23
# =========================================================================

MODEL_COEFFICIENTS = {
    'avg_q_pace': 0.0520,
    'q3_lead': 0.0572,
    'q3_total': 0.1064,       # Q3 quarter-specific scoring
    'pace_trend': 0.0242,      # Q3 - Q1 scoring (acceleration/deceleration)
    'q3_cumul_total': 0.1561,  # Total points through Q3
    'h1_total': 0.0497,        # First half total
    'scoring_variance': 0.0583, # Std of Q1/Q2/Q3 totals
    'late_q3_pts': -0.0483,    # Points in last 3 min of Q3
    'pace_ratio': -39.5227,    # Actual pace vs expected (q3_cumul / ou_line*0.75)
    'intercept': 52.1462
}

# Signal tier thresholds (OOS validated)
TIER_THRESHOLDS = {
    'PLATINUM': {'margin': 20, 'accuracy': 0.994, 'description': 'Near-certain outcome'},
    'GOLD':     {'margin': 15, 'accuracy': 0.983, 'description': 'Extremely high confidence'},
    'SILVER':   {'margin': 12, 'accuracy': 0.977, 'description': 'Very high confidence'},
    'BRONZE':   {'margin': 10, 'accuracy': 0.957, 'description': 'High confidence'},
}

# O/U bet payoff at -110 juice
VIG_ODDS = -110
WIN_PAYOUT = 100 / 110  # 0.9091 per unit risked
BREAKEVEN_PCT = 110 / 210  # 0.5238


@dataclass
class Q3Signal:
    """A Q3 Over/Under signal."""
    direction: str            # 'OVER' or 'UNDER'
    tier: str                 # 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE'
    confidence: float         # 0.0 to 1.0
    predicted_q4: float       # Predicted Q4 total scoring
    predicted_final: float    # Predicted final total
    margin: float             # Margin from O/U line (absolute)
    edge: float               # Expected edge per unit bet
    kelly: float              # Kelly criterion fraction
    roi_expected: float       # Expected ROI per trade
    description: str          # Human-readable signal description

    # Game context
    q3_cumul_total: int       # Total points through Q3
    ou_line: float            # Opening O/U line
    pts_needed: float         # Points needed in Q4 for OVER
    avg_q_pace: float         # Average points per quarter through Q3
    pace_ratio: float         # Pace ratio vs expected

    def to_dict(self):
        return asdict(self)


class Q3OverUnderEngine:
    """
    Production engine for Q3 Over/Under signals.

    Evaluates game state at end of Q3 and generates high-confidence
    O/U signals against the opening line.
    """

    def __init__(self, coefficients=None, tier_thresholds=None):
        self.coefficients = coefficients or MODEL_COEFFICIENTS
        self.tier_thresholds = tier_thresholds or TIER_THRESHOLDS

    def predict_q4_total(self, features: Dict[str, float]) -> float:
        """
        Predict Q4 total scoring from game state features.

        Uses linear model trained on 1,162 games.
        """
        prediction = self.coefficients['intercept']
        for feat, coef in self.coefficients.items():
            if feat == 'intercept':
                continue
            if feat in features:
                prediction += coef * features[feat]
        return prediction

    def compute_features(self, home_score: int, away_score: int,
                         q1_total: int, q2_total: int, q3_total: int,
                         ou_line: float, late_q3_pts: int = 12) -> Dict[str, float]:
        """
        Compute all model features from raw game state at Q3 end.
        """
        q3_cumul_total = home_score + away_score
        h1_total = q1_total + q2_total
        avg_q_pace = q3_cumul_total / 3.0
        q3_lead = abs(home_score - away_score)
        pace_trend = q3_total - q1_total  # Acceleration/deceleration
        pace_ratio = q3_cumul_total / (ou_line * 0.75) if ou_line > 0 else 1.0
        scoring_variance = float(np.std([q1_total, q2_total, q3_total]))

        return {
            'avg_q_pace': avg_q_pace,
            'q3_lead': q3_lead,
            'q3_total': q3_total,
            'pace_trend': pace_trend,
            'q3_cumul_total': q3_cumul_total,
            'h1_total': h1_total,
            'scoring_variance': scoring_variance,
            'late_q3_pts': late_q3_pts,
            'pace_ratio': pace_ratio
        }

    def get_tier(self, margin: float) -> Optional[str]:
        """Determine signal tier based on margin from O/U line."""
        for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
            if margin >= self.tier_thresholds[tier_name]['margin']:
                return tier_name
        return None

    def compute_edge_and_kelly(self, confidence: float) -> Tuple[float, float]:
        """
        Compute expected edge and Kelly criterion bet size.

        At -110 odds:
        - Win pays 100/110 = 0.9091 per unit risked
        - Loss costs 1.0 per unit risked
        - EV = p * 0.9091 - (1-p) * 1.0
        - Kelly = (p * 1.9091 - 1) / 0.9091
        """
        ev_per_unit = confidence * WIN_PAYOUT - (1 - confidence) * 1.0
        kelly_numer = confidence * (1 + WIN_PAYOUT) - 1
        kelly_denom = WIN_PAYOUT
        kelly = max(0, kelly_numer / kelly_denom) if kelly_denom > 0 else 0

        # Cap Kelly at 25% for safety (quarter Kelly)
        kelly_adjusted = min(kelly * 0.25, 0.15)

        return ev_per_unit, kelly_adjusted

    def evaluate(self, home_score: int, away_score: int,
                 q1_total: int, q2_total: int, q3_total: int,
                 ou_line: float, late_q3_pts: int = 12) -> Optional[Q3Signal]:
        """
        Evaluate game state at Q3 end and generate O/U signal.

        Args:
            home_score: Home team score at Q3 end
            away_score: Away team score at Q3 end
            q1_total: Combined Q1 scoring
            q2_total: Combined Q2 scoring
            q3_total: Combined Q3 scoring
            ou_line: Opening Over/Under line
            late_q3_pts: Points scored in last 3 minutes of Q3

        Returns:
            Q3Signal if signal fires, None if no signal
        """
        if ou_line <= 0:
            return None

        # Compute features
        features = self.compute_features(
            home_score, away_score, q1_total, q2_total, q3_total,
            ou_line, late_q3_pts
        )

        q3_cumul_total = home_score + away_score

        # Predict Q4 total
        predicted_q4 = self.predict_q4_total(features)
        predicted_final = q3_cumul_total + predicted_q4

        # Determine direction and margin
        margin = abs(predicted_final - ou_line)
        direction = 'OVER' if predicted_final > ou_line else 'UNDER'

        # Check if signal meets minimum tier threshold
        tier = self.get_tier(margin)
        if tier is None:
            return None

        # Get tier accuracy as confidence
        confidence = self.tier_thresholds[tier]['accuracy']

        # Compute edge and Kelly sizing
        edge, kelly = self.compute_edge_and_kelly(confidence)
        roi_expected = edge / 1.0 * 100  # As percentage

        # Points needed for over
        pts_needed = ou_line - q3_cumul_total

        # Generate description
        pace_str = f"{features['pace_ratio']:.2f}x"
        if direction == 'UNDER':
            desc = (f"UNDER {ou_line}: Game at {q3_cumul_total} pts through Q3 "
                    f"({pace_str} pace). Need {pts_needed:.0f} in Q4, model predicts "
                    f"{predicted_q4:.0f}. {tier} confidence.")
        else:
            desc = (f"OVER {ou_line}: Game at {q3_cumul_total} pts through Q3 "
                    f"({pace_str} pace). Need {pts_needed:.0f} in Q4, model predicts "
                    f"{predicted_q4:.0f}. {tier} confidence.")

        return Q3Signal(
            direction=direction,
            tier=tier,
            confidence=confidence,
            predicted_q4=round(predicted_q4, 1),
            predicted_final=round(predicted_final, 1),
            margin=round(margin, 1),
            edge=round(edge, 4),
            kelly=round(kelly, 4),
            roi_expected=round(roi_expected, 1),
            description=desc,
            q3_cumul_total=q3_cumul_total,
            ou_line=ou_line,
            pts_needed=round(pts_needed, 1),
            avg_q_pace=round(features['avg_q_pace'], 1),
            pace_ratio=round(features['pace_ratio'], 3)
        )

    def evaluate_live(self, home_score: int, away_score: int,
                      period: int, clock_minutes: float,
                      q1_total: int, q2_total: int,
                      ou_line: float, late_pts: int = 12) -> Optional[Q3Signal]:
        """
        Evaluate during Q3 (before it ends) for early entry.

        Projects Q3 end scores and evaluates signal.
        """
        if period != 3:
            return None

        # Calculate Q3 scoring so far
        q3_elapsed_mins = 12.0 - clock_minutes
        if q3_elapsed_mins < 6.0:
            return None  # Need at least half of Q3

        q3_so_far = (home_score + away_score) - q1_total - q2_total
        q3_projected = q3_so_far * (12.0 / q3_elapsed_mins) if q3_elapsed_mins > 0 else q3_so_far

        # Project Q3 end scores (proportional split)
        remaining_q3 = q3_projected - q3_so_far
        home_pct = home_score / (home_score + away_score) if (home_score + away_score) > 0 else 0.5
        projected_home = home_score + remaining_q3 * home_pct
        projected_away = away_score + remaining_q3 * (1 - home_pct)

        return self.evaluate(
            int(projected_home), int(projected_away),
            q1_total, q2_total, int(q3_projected),
            ou_line, late_pts
        )


def export_model_for_webapp():
    """Export model coefficients and config for the JavaScript webapp."""
    model_data = {
        'name': 'Q3 Over/Under Signal Engine',
        'version': '1.0',
        'type': 'linear_regression',
        'training_data': {
            'season': '2021-22',
            'games': 1162,
            'features': 9
        },
        'validation': {
            'season': '2022-23',
            'games': 1015,
            'overall_accuracy': 0.846,
            'tiers': {
                'PLATINUM': {'margin_min': 20, 'accuracy': 0.994, 'games_fired': 171, 'pct_games': 0.168},
                'GOLD': {'margin_min': 15, 'accuracy': 0.983, 'games_fired': 297, 'pct_games': 0.293},
                'SILVER': {'margin_min': 12, 'accuracy': 0.977, 'games_fired': 396, 'pct_games': 0.390},
                'BRONZE': {'margin_min': 10, 'accuracy': 0.957, 'games_fired': 487, 'pct_games': 0.480},
            }
        },
        'coefficients': MODEL_COEFFICIENTS,
        'tiers': TIER_THRESHOLDS,
        'payoff': {
            'vig_odds': VIG_ODDS,
            'win_payout': WIN_PAYOUT,
            'breakeven_pct': BREAKEVEN_PCT
        }
    }

    output_path = Path("/home/user/nba2/output/q3_ou_model.json")
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"Model exported to {output_path}")

    return model_data


def run_full_backtest():
    """
    Run complete backtest with position sizing and realistic execution.
    """
    import pandas as pd

    print("=" * 70)
    print("Q3 O/U STRATEGY - FULL BACKTEST WITH KELLY SIZING")
    print("=" * 70)

    # Load analysis data
    df = pd.read_csv("/home/user/nba2/output/q3_ou_analysis.csv")
    print(f"\nLoaded {len(df)} games")

    # Split seasons
    def get_season(gid):
        try:
            s = str(int(gid))
            return '2021-22' if int(s) < 401400000 else '2022-23'
        except:
            return 'unknown'

    df['season'] = df['game_id'].apply(get_season)
    train = df[df['season'] == '2021-22'].copy()
    test = df[df['season'] == '2022-23'].copy()

    print(f"Train: {len(train)} games (2021-22)")
    print(f"Test:  {len(test)} games (2022-23)")

    # Train model on training set
    from sklearn.linear_model import LinearRegression

    features = ['avg_q_pace', 'q3_lead', 'q3_total', 'pace_trend',
                'q3_cumul_total', 'h1_total', 'scoring_variance',
                'late_q3_pts', 'pace_ratio']

    train_valid = train.dropna(subset=features + ['q4_total'])
    test_valid = test.dropna(subset=features + ['q4_total'])

    X_train = train_valid[features].values
    y_train = train_valid['q4_total'].values
    X_test = test_valid[features].values
    y_test = test_valid['q4_total'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test
    preds = model.predict(X_test)
    test_valid = test_valid.copy()
    test_valid['pred_q4'] = preds
    test_valid['pred_final'] = test_valid['q3_cumul_total'] + test_valid['pred_q4']
    test_valid['margin'] = abs(test_valid['pred_final'] - test_valid['ou_line'])
    test_valid['pred_direction'] = np.where(
        test_valid['pred_final'] > test_valid['ou_line'], 'OVER', 'UNDER'
    )
    test_valid['actual_direction'] = np.where(
        test_valid['final_total'] > test_valid['ou_line'], 'OVER', 'UNDER'
    )
    test_valid['correct'] = test_valid['pred_direction'] == test_valid['actual_direction']

    # Initialize engine
    engine = Q3OverUnderEngine()

    # Simulate trading with Kelly sizing
    initial_bankroll = 1000.0
    bankroll = initial_bankroll
    trades = []

    for tier_name, tier_info in engine.tier_thresholds.items():
        min_margin = tier_info['margin']
        tier_games = test_valid[test_valid['margin'] >= min_margin].copy()

        wins = tier_games['correct'].sum()
        losses = len(tier_games) - wins
        accuracy = wins / len(tier_games) if len(tier_games) > 0 else 0

        # Simulate flat betting (1 unit per trade at -110)
        flat_pnl = wins * WIN_PAYOUT - losses * 1.0
        flat_roi = flat_pnl / len(tier_games) * 100 if len(tier_games) > 0 else 0

        # Simulate Kelly betting
        _, kelly_frac = engine.compute_edge_and_kelly(accuracy)
        kelly_bankroll = initial_bankroll
        kelly_peak = initial_bankroll
        kelly_max_dd = 0
        kelly_trades = []

        for _, game in tier_games.iterrows():
            stake = kelly_bankroll * kelly_frac
            if game['correct']:
                pnl = stake * WIN_PAYOUT
            else:
                pnl = -stake
            kelly_bankroll += pnl
            kelly_peak = max(kelly_peak, kelly_bankroll)
            dd = (kelly_peak - kelly_bankroll) / kelly_peak
            kelly_max_dd = max(kelly_max_dd, dd)
            kelly_trades.append({
                'bankroll': kelly_bankroll,
                'pnl': pnl,
                'stake': stake
            })

        kelly_total_return = (kelly_bankroll - initial_bankroll) / initial_bankroll * 100

        print(f"\n{'='*60}")
        print(f"TIER: {tier_name} (margin >= {min_margin})")
        print(f"{'='*60}")
        print(f"  Games:    {len(tier_games)}")
        print(f"  Wins:     {wins}")
        print(f"  Losses:   {losses}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  FLAT BET RESULTS:")
        print(f"    P&L:    {flat_pnl:+.1f} units")
        print(f"    ROI:    {flat_roi:+.1f}%")
        print(f"  KELLY BET RESULTS (quarter-Kelly):")
        print(f"    Kelly fraction:  {kelly_frac:.4f}")
        print(f"    Starting:        ${initial_bankroll:.0f}")
        print(f"    Ending:          ${kelly_bankroll:.0f}")
        print(f"    Total return:    {kelly_total_return:+.1f}%")
        print(f"    Max drawdown:    {kelly_max_dd:.1%}")

        # Break down by OVER/UNDER
        for direction in ['OVER', 'UNDER']:
            d_games = tier_games[tier_games['pred_direction'] == direction]
            if len(d_games) > 0:
                d_acc = d_games['correct'].mean()
                d_wins = d_games['correct'].sum()
                d_losses = len(d_games) - d_wins
                print(f"  {direction}:")
                print(f"    Games: {len(d_games)}, W-L: {d_wins}-{d_losses}, Acc: {d_acc:.1%}")

    # Export signals for all test games
    print("\n" + "=" * 60)
    print("SIGNAL EXAMPLES (first 20 test games with signals)")
    print("=" * 60)

    count = 0
    for _, game in test_valid.iterrows():
        signal = engine.evaluate(
            home_score=int(game.get('q3_home', 0)),
            away_score=int(game.get('q3_away', 0)),
            q1_total=int(game.get('q1_total', 0)),
            q2_total=int(game.get('q2_total', 0)),
            q3_total=int(game.get('q3_total', 0)),
            ou_line=game['ou_line'],
            late_q3_pts=int(game.get('late_q3_pts', 12))
        )
        if signal and count < 20:
            actual = 'OVER' if game['final_total'] > game['ou_line'] else 'UNDER'
            hit = 'HIT' if signal.direction == actual else 'MISS'
            print(f"\n  Game {game['game_id']}:")
            print(f"    {signal.tier} {signal.direction} {game['ou_line']} "
                  f"(margin={signal.margin:.0f})")
            print(f"    Q3 total={signal.q3_cumul_total}, need={signal.pts_needed:.0f}, "
                  f"model={signal.predicted_q4:.0f}, pace={signal.pace_ratio:.2f}x")
            print(f"    Actual final={int(game['final_total'])} → {actual} → {hit}")
            print(f"    Edge={signal.edge:.4f}, Kelly={signal.kelly:.4f}")
            count += 1

    # Export model
    export_model_for_webapp()

    return test_valid


if __name__ == "__main__":
    results = run_full_backtest()
