"""
Q3 Over/Under Signal Strategy - Production Engine
===================================================
Predicts final game total at end of Q3 using a 9-feature linear model,
trained on 2021-22 NBA season (1,162 games).

HONEST ASSESSMENT:
    The model achieves high directional accuracy (~96%) against the
    OPENING O/U line. However, by Q3 end the opening line is stale.
    No sportsbook still offers the opening number at that point.

    When compared against an estimated LIVE O/U line (what a market
    maker would price at Q3 end), the model's accuracy drops to ~52%,
    which is essentially breakeven at -110 juice.

    The core reason: a simple 3-feature market proxy (pace, lead,
    pre-game line) captures nearly all the same information as our
    9-feature model. R² for Q4 scoring is ~0.04 regardless of model
    complexity — Q4 is inherently noisy.

WHAT THE MODEL IS USEFUL FOR:
    - Confirming whether a game is tracking OVER or UNDER the opening line
    - Estimating Q4 scoring within ±9 points (1 std dev)
    - Educational: demonstrating market efficiency in live betting
    - Pre-game analysis: if you took OVER pre-game, the model tells
      you whether the game is trending your way

WHAT THE MODEL CANNOT DO:
    - Beat the live market at Q3 end (the market has already adjusted)
    - Generate profitable signals at standard -110 odds vs live lines
    - Provide edge beyond what a simple pace+lead regression gives

SIGNAL TIERS (accuracy vs OPENING line, OOS 2022-23):
    PLATINUM: model margin >= 20 from opening line → ~99% correct
    GOLD:     model margin >= 15 from opening line → ~98% correct
    SILVER:   model margin >= 12 from opening line → ~97% correct
    BRONZE:   model margin >= 10 from opening line → ~96% correct

    NOTE: These tiers measure agreement with observable game flow,
    NOT edge against live betting markets.

USAGE:
    from q3_ou_strategy import Q3OverUnderEngine

    engine = Q3OverUnderEngine()
    signal = engine.evaluate(
        home_score=78, away_score=65,
        q1_total=58, q2_total=55, q3_total=53,
        ou_line=224.5,
        late_q3_pts=12
    )

    if signal:
        print(signal['direction'])        # 'OVER' or 'UNDER'
        print(signal['tier'])             # 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE'
        print(signal['opening_margin'])   # Margin vs opening line (large, but stale)
        print(signal['live_ou_estimate']) # Estimated live O/U line at Q3 end
        print(signal['live_margin'])      # Margin vs live line (small, honest)
        print(signal['live_edge'])        # Edge vs live market (~0, honest)
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple
from pathlib import Path


# =========================================================================
# MODEL COEFFICIENTS (trained on 2021-22 season, 1,162 regulation games)
# 9-feature linear regression predicting Q4 total scoring.
# =========================================================================

MODEL_COEFFICIENTS = {
    'avg_q_pace': 0.0577540572,
    'q3_lead': 0.0450073879,
    'q3_total': 0.1291405317,
    'pace_trend': -0.0145670630,
    'q3_cumul_total': 0.1732621717,
    'h1_total': 0.0441216400,
    'scoring_variance': 0.0267483526,
    'late_q3_pts': -0.1166156333,
    'pace_ratio': -43.0823638232,
    'intercept': 52.9033412611
}

# =========================================================================
# MARKET PROXY MODEL (trained on same 2021-22 season)
# Simulates what an efficient market maker would estimate for Q4 scoring
# given only: game pace, score differential, and pre-game O/U line.
#
# This 3-feature model achieves RMSE 8.65 on OOS data — nearly identical
# to our 9-feature model's RMSE of 8.64. The market is efficient.
#
# Features: avg_q_pace, q3_lead, ou_line/4 (pre-game quarter pace)
# =========================================================================

MARKET_PROXY_COEFFICIENTS = {
    'avg_q_pace': -0.0166437427,
    'q3_lead': 0.0439679156,
    'ou_q': 0.8167063332,
    'intercept': 8.5718642911
}

# Signal tier thresholds
# Accuracy values from 5-fold CV on training set (not test set — no leakage)
TIER_THRESHOLDS = {
    'PLATINUM': {'margin': 20, 'cv_accuracy': 0.986, 'description': 'Strong opening-line signal'},
    'GOLD':     {'margin': 15, 'cv_accuracy': 0.982, 'description': 'High opening-line signal'},
    'SILVER':   {'margin': 12, 'cv_accuracy': 0.970, 'description': 'Moderate opening-line signal'},
    'BRONZE':   {'margin': 10, 'cv_accuracy': 0.962, 'description': 'Opening-line signal'},
}

# O/U bet payoff at -110 juice
VIG_ODDS = -110
WIN_PAYOUT = 100 / 110  # 0.9091 per unit risked
BREAKEVEN_PCT = 110 / 210  # 0.5238


@dataclass
class Q3Signal:
    """A Q3 Over/Under signal with honest edge assessment."""
    direction: str            # 'OVER' or 'UNDER'
    tier: str                 # 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE'
    predicted_q4: float       # Predicted Q4 total scoring
    predicted_final: float    # Predicted final total
    description: str          # Human-readable signal description

    # Opening line analysis (what the current code reports — real but misleading)
    opening_margin: float     # Margin from opening O/U line
    opening_confidence: float # CV accuracy at this tier vs opening line
    opening_edge: float       # Expected edge vs opening line (NOT achievable)
    opening_kelly: float      # Kelly fraction vs opening line (NOT achievable)

    # Live line analysis (honest assessment of actual tradeable edge)
    live_ou_estimate: float   # Estimated live O/U line at Q3 end
    live_margin: float        # Margin from live line (typically very small)
    live_edge: float          # Expected edge vs live market (~0)

    # Game context
    q3_cumul_total: int       # Total points through Q3
    ou_line: float            # Opening O/U line
    pts_needed: float         # Points needed in Q4 for OVER (opening line)
    avg_q_pace: float         # Average points per quarter through Q3
    pace_ratio: float         # Pace ratio vs expected
    market_q4_estimate: float # Market's Q4 estimate

    def to_dict(self):
        return asdict(self)


class Q3OverUnderEngine:
    """
    Production engine for Q3 Over/Under signals.

    Evaluates game state at end of Q3 and generates O/U signals
    with both opening-line and live-line edge estimates.
    """

    def __init__(self, coefficients=None, market_coefficients=None,
                 tier_thresholds=None):
        self.coefficients = coefficients or MODEL_COEFFICIENTS
        self.market_coefficients = market_coefficients or MARKET_PROXY_COEFFICIENTS
        self.tier_thresholds = tier_thresholds or TIER_THRESHOLDS

    def predict_q4_total(self, features: Dict[str, float]) -> float:
        """Predict Q4 total scoring from game state features (9-feature model)."""
        prediction = self.coefficients['intercept']
        for feat, coef in self.coefficients.items():
            if feat == 'intercept':
                continue
            if feat in features:
                prediction += coef * features[feat]
        return prediction

    def estimate_market_q4(self, avg_q_pace: float, q3_lead: float,
                           ou_line: float) -> float:
        """
        Estimate what the market would price Q4 total scoring at.

        Uses a 3-feature market proxy model trained on the same training
        data. This represents a sophisticated but simple market maker
        who knows the pre-game line, current pace, and score differential.

        In reality, market makers have even more information (injuries,
        team tendencies, betting flow), so this is a generous lower
        bound on market efficiency.
        """
        mc = self.market_coefficients
        ou_q = ou_line / 4.0
        return (mc['intercept']
                + mc['avg_q_pace'] * avg_q_pace
                + mc['q3_lead'] * q3_lead
                + mc['ou_q'] * ou_q)

    def estimate_live_ou(self, q3_cumul_total: int, avg_q_pace: float,
                         q3_lead: float, ou_line: float) -> float:
        """
        Estimate the live O/U line at Q3 end.

        live_ou = points_scored_through_Q3 + market_expected_Q4
        """
        market_q4 = self.estimate_market_q4(avg_q_pace, q3_lead, ou_line)
        return q3_cumul_total + market_q4

    def compute_features(self, home_score: int, away_score: int,
                         q1_total: int, q2_total: int, q3_total: int,
                         ou_line: float, late_q3_pts: int = 12) -> Dict[str, float]:
        """Compute all model features from raw game state at Q3 end."""
        q3_cumul_total = home_score + away_score
        h1_total = q1_total + q2_total
        avg_q_pace = q3_cumul_total / 3.0
        q3_lead = abs(home_score - away_score)
        pace_trend = q3_total - q1_total
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
        """Determine signal tier based on margin from opening O/U line."""
        for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
            if margin >= self.tier_thresholds[tier_name]['margin']:
                return tier_name
        return None

    def compute_edge_and_kelly(self, confidence: float) -> Tuple[float, float]:
        """
        Compute expected edge and Kelly criterion bet size at -110 odds.
        """
        ev_per_unit = confidence * WIN_PAYOUT - (1 - confidence) * 1.0
        kelly_numer = confidence * (1 + WIN_PAYOUT) - 1
        kelly_denom = WIN_PAYOUT
        kelly = max(0, kelly_numer / kelly_denom) if kelly_denom > 0 else 0
        kelly_adjusted = min(kelly * 0.25, 0.15)
        return ev_per_unit, kelly_adjusted

    def evaluate(self, home_score: int, away_score: int,
                 q1_total: int, q2_total: int, q3_total: int,
                 ou_line: float, late_q3_pts: int = 12) -> Optional[Q3Signal]:
        """
        Evaluate game state at Q3 end and generate O/U signal.

        Returns a signal with both opening-line metrics (high accuracy
        but not actionable) and live-line metrics (honest edge estimate).
        """
        if ou_line <= 0:
            return None

        features = self.compute_features(
            home_score, away_score, q1_total, q2_total, q3_total,
            ou_line, late_q3_pts
        )

        q3_cumul_total = home_score + away_score

        # Our model's Q4 prediction
        predicted_q4 = self.predict_q4_total(features)
        predicted_final = q3_cumul_total + predicted_q4

        # Market's Q4 estimate and live line
        market_q4 = self.estimate_market_q4(
            features['avg_q_pace'], features['q3_lead'], ou_line
        )
        live_ou = q3_cumul_total + market_q4

        # Opening line metrics
        opening_margin = abs(predicted_final - ou_line)
        direction = 'OVER' if predicted_final > ou_line else 'UNDER'

        # Live line metrics
        live_margin = abs(predicted_final - live_ou)

        # Check if signal meets minimum tier threshold (vs opening line)
        tier = self.get_tier(opening_margin)
        if tier is None:
            return None

        # Opening line edge (high, but not achievable — line is stale)
        opening_confidence = self.tier_thresholds[tier]['cv_accuracy']
        opening_edge, opening_kelly = self.compute_edge_and_kelly(opening_confidence)

        # Live line edge (honest — typically ~0)
        # With live margin of ~1-2 points, directional accuracy is ~52%
        # We use a conservative estimate based on the live margin
        live_accuracy_estimate = 0.524  # Empirically ~breakeven
        if live_margin >= 3:
            live_accuracy_estimate = 0.54  # Slight edge when model disagrees more
        elif live_margin >= 2:
            live_accuracy_estimate = 0.53
        live_edge = live_accuracy_estimate * WIN_PAYOUT - (1 - live_accuracy_estimate) * 1.0

        pts_needed = ou_line - q3_cumul_total

        pace_str = f"{features['pace_ratio']:.2f}x"
        desc = (f"{direction} {ou_line}: Game at {q3_cumul_total} pts through Q3 "
                f"({pace_str} pace). Need {pts_needed:.0f} in Q4, model predicts "
                f"{predicted_q4:.0f}. {tier} vs opening line. "
                f"Live line est ~{live_ou:.1f} (margin {live_margin:.1f}).")

        return Q3Signal(
            direction=direction,
            tier=tier,
            predicted_q4=round(predicted_q4, 1),
            predicted_final=round(predicted_final, 1),
            description=desc,
            opening_margin=round(opening_margin, 1),
            opening_confidence=opening_confidence,
            opening_edge=round(opening_edge, 4),
            opening_kelly=round(opening_kelly, 4),
            live_ou_estimate=round(live_ou, 1),
            live_margin=round(live_margin, 1),
            live_edge=round(live_edge, 4),
            q3_cumul_total=q3_cumul_total,
            ou_line=ou_line,
            pts_needed=round(pts_needed, 1),
            avg_q_pace=round(features['avg_q_pace'], 1),
            pace_ratio=round(features['pace_ratio'], 3),
            market_q4_estimate=round(market_q4, 1)
        )


def export_model_for_webapp():
    """Export model coefficients and config for the JavaScript webapp."""
    model_data = {
        'name': 'Q3 Over/Under Signal Engine',
        'version': '2.0',
        'type': 'linear_regression',
        'training_data': {
            'season': '2021-22',
            'games': 1162,
            'features': 9
        },
        'validation': {
            'season': '2022-23',
            'games': 1015,
            'note': 'Accuracy figures below are vs OPENING line, not live market',
            'tiers': {
                'PLATINUM': {'margin_min': 20, 'cv_accuracy': 0.986},
                'GOLD': {'margin_min': 15, 'cv_accuracy': 0.982},
                'SILVER': {'margin_min': 12, 'cv_accuracy': 0.970},
                'BRONZE': {'margin_min': 10, 'cv_accuracy': 0.962},
            },
            'live_market_accuracy': 0.524,
            'live_market_note': 'Accuracy vs estimated live line is ~52.4% (breakeven at -110)'
        },
        'coefficients': MODEL_COEFFICIENTS,
        'market_proxy_coefficients': MARKET_PROXY_COEFFICIENTS,
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
    Run complete backtest with BOTH opening-line and live-line analysis.
    Shows the difference between illusory edge (vs stale line) and
    real edge (vs live market).
    """
    import pandas as pd

    print("=" * 70)
    print("Q3 O/U STRATEGY - FULL BACKTEST")
    print("Opening Line vs Live Line Comparison")
    print("=" * 70)

    df = pd.read_csv("/home/user/nba2/output/q3_ou_analysis.csv")
    print(f"\nLoaded {len(df)} games")

    def get_season(gid):
        try:
            s = str(int(gid))
            return '2021-22' if int(s) < 401400000 else '2022-23'
        except:
            return 'unknown'

    df['season'] = df['game_id'].apply(get_season)
    train = df[df['season'] == '2021-22'].copy()
    test = df[df['season'] == '2022-23'].copy()

    print(f"Train: {len(train)} games (2021-22) — model was trained on this")
    print(f"Test:  {len(test)} games (2022-23) — true out-of-sample")

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

    # Our predictions
    preds = model.predict(X_test)
    test_valid = test_valid.copy()
    test_valid['pred_q4'] = preds
    test_valid['pred_final'] = test_valid['q3_cumul_total'] + test_valid['pred_q4']

    # Market proxy predictions
    engine = Q3OverUnderEngine()
    test_valid['market_q4'] = test_valid.apply(
        lambda r: engine.estimate_market_q4(
            r['avg_q_pace'], r['q3_lead'], r['ou_line']
        ), axis=1
    )
    test_valid['live_ou'] = test_valid['q3_cumul_total'] + test_valid['market_q4']

    # Opening line metrics
    test_valid['opening_margin'] = abs(test_valid['pred_final'] - test_valid['ou_line'])
    test_valid['pred_direction'] = np.where(
        test_valid['pred_final'] > test_valid['ou_line'], 'OVER', 'UNDER'
    )
    test_valid['actual_direction'] = np.where(
        test_valid['final_total'] > test_valid['ou_line'], 'OVER', 'UNDER'
    )
    test_valid['opening_correct'] = (
        test_valid['pred_direction'] == test_valid['actual_direction']
    )

    # Live line metrics
    test_valid['live_margin'] = abs(test_valid['pred_final'] - test_valid['live_ou'])
    test_valid['live_direction'] = np.where(
        test_valid['pred_final'] > test_valid['live_ou'], 'OVER', 'UNDER'
    )
    test_valid['actual_direction_live'] = np.where(
        test_valid['final_total'] > test_valid['live_ou'], 'OVER', 'UNDER'
    )
    test_valid['live_correct'] = (
        test_valid['live_direction'] == test_valid['actual_direction_live']
    )

    # Print comparison
    print(f"\n{'='*70}")
    print("ACCURACY: OPENING LINE vs LIVE LINE (OOS 2022-23 only)")
    print(f"{'='*70}")

    overall_opening = test_valid['opening_correct'].mean()
    overall_live = test_valid['live_correct'].mean()
    print(f"\n  Overall accuracy vs OPENING line: {overall_opening:.1%}")
    print(f"  Overall accuracy vs LIVE line:    {overall_live:.1%}")
    print(f"  Breakeven at -110:                {BREAKEVEN_PCT:.1%}")

    print(f"\n  {'Tier':<10} {'Margin':<8} {'Games':<8} {'Opening Acc':<14} {'Live Acc':<14}")
    print(f"  {'-'*54}")

    for tier_name, tier_info in engine.tier_thresholds.items():
        min_margin = tier_info['margin']
        tier_games = test_valid[test_valid['opening_margin'] >= min_margin]
        if len(tier_games) == 0:
            continue
        o_acc = tier_games['opening_correct'].mean()
        l_acc = tier_games['live_correct'].mean()
        print(f"  {tier_name:<10} >= {min_margin:<5} {len(tier_games):<8} "
              f"{o_acc:<14.1%} {l_acc:<14.1%}")

    # Profitability comparison
    print(f"\n{'='*70}")
    print("PROFITABILITY: OPENING LINE vs LIVE LINE (flat 1-unit at -110)")
    print(f"{'='*70}")

    for tier_name, tier_info in engine.tier_thresholds.items():
        min_margin = tier_info['margin']
        tier_games = test_valid[test_valid['opening_margin'] >= min_margin]
        if len(tier_games) == 0:
            continue

        # Opening line P&L
        o_wins = tier_games['opening_correct'].sum()
        o_losses = len(tier_games) - o_wins
        o_pnl = o_wins * WIN_PAYOUT - o_losses
        o_roi = o_pnl / len(tier_games) * 100

        # Live line P&L
        l_wins = tier_games['live_correct'].sum()
        l_losses = len(tier_games) - l_wins
        l_pnl = l_wins * WIN_PAYOUT - l_losses
        l_roi = l_pnl / len(tier_games) * 100

        print(f"\n  {tier_name} (opening margin >= {min_margin}):")
        print(f"    Games: {len(tier_games)}")
        print(f"    vs Opening line: {o_wins}W-{o_losses}L, "
              f"PnL={o_pnl:+.1f}u, ROI={o_roi:+.1f}%")
        print(f"    vs Live line:    {l_wins}W-{l_losses}L, "
              f"PnL={l_pnl:+.1f}u, ROI={l_roi:+.1f}%")

    # Model comparison
    print(f"\n{'='*70}")
    print("MODEL COMPARISON (why there's no live-market edge)")
    print(f"{'='*70}")

    our_rmse = np.sqrt(((preds - y_test) ** 2).mean())
    market_rmse = np.sqrt(((test_valid['market_q4'].values - y_test) ** 2).mean())
    print(f"\n  Our 9-feature model RMSE:    {our_rmse:.2f} points")
    print(f"  Market proxy (3-feat) RMSE:  {market_rmse:.2f} points")
    print(f"  Difference:                  {our_rmse - market_rmse:+.2f} points")
    print(f"\n  Average live margin (model vs market): "
          f"{test_valid['live_margin'].mean():.1f} points")
    print(f"  Median live margin:                    "
          f"{test_valid['live_margin'].median():.1f} points")
    print(f"\n  Conclusion: Our extra features (late_q3_pts, scoring_variance,")
    print(f"  pace_trend, etc.) add essentially zero predictive power over")
    print(f"  what a market maker already knows (pace + lead + pre-game line).")

    # Export model
    export_model_for_webapp()

    return test_valid


if __name__ == "__main__":
    results = run_full_backtest()
