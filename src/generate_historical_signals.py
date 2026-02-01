"""
Generate Historical Q3 O/U Signals from Cached ESPN Game Data
==============================================================
Processes all cached ESPN play-by-play game files, runs the Q3 O/U model,
and exports validated signals as JSON for the webapp.

STRICT RULES:
- Only games with a real ESPN pickcenter O/U line (no fallbacks)
- Only regulation games (no overtime)
- No default assumptions for missing features
- Every signal validated against actual final total
- 2021-22 season results clearly labeled as IN-SAMPLE (model trained on this)
- 2022-23 season results are true OUT-OF-SAMPLE
- Confidence values from 5-fold CV on training set (no test-set leakage)
- Live O/U line estimated via market proxy for honest edge assessment
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PBP_DIR = Path("/home/user/nba2/cache/games_pbp")
OUTPUT_DIR = Path("/home/user/nba2/output")
WEBAPP_DIR = Path("/home/user/nba2/webapp/js")

# Model coefficients (trained on 2021-22 season only)
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
    'intercept': 52.9033412611,
}

# Market proxy model (also trained on 2021-22 only)
# Estimates what a market maker would set the live Q4 total at
MARKET_PROXY_COEFFICIENTS = {
    'avg_q_pace': -0.0166437427,
    'q3_lead': 0.0439679156,
    'ou_q': 0.8167063332,
    'intercept': 8.5718642911,
}

TIER_THRESHOLDS = {
    'PLATINUM': {'margin': 20, 'cv_accuracy': 0.986},
    'GOLD': {'margin': 15, 'cv_accuracy': 0.982},
    'SILVER': {'margin': 12, 'cv_accuracy': 0.970},
    'BRONZE': {'margin': 10, 'cv_accuracy': 0.962},
}

WIN_PAYOUT = 100 / 110  # 0.9091
BREAKEVEN_PCT = 110 / 210  # 0.5238


def predict_q4(features):
    """Predict Q4 total from features using our 9-feature linear model."""
    prediction = MODEL_COEFFICIENTS['intercept']
    for feat, coef in MODEL_COEFFICIENTS.items():
        if feat == 'intercept':
            continue
        if feat in features:
            prediction += coef * features[feat]
    return prediction


def estimate_market_q4(avg_q_pace, q3_lead, ou_line):
    """Estimate what the market would price Q4 scoring at (3-feature proxy)."""
    mc = MARKET_PROXY_COEFFICIENTS
    ou_q = ou_line / 4.0
    return (mc['intercept']
            + mc['avg_q_pace'] * avg_q_pace
            + mc['q3_lead'] * q3_lead
            + mc['ou_q'] * ou_q)


def compute_features(home_score, away_score, q1_total, q2_total, q3_total, ou_line, late_q3_pts):
    """Compute all model features from raw game state."""
    q3_cumul_total = home_score + away_score
    h1_total = q1_total + q2_total
    avg_q_pace = q3_cumul_total / 3.0
    q3_lead = abs(home_score - away_score)
    pace_trend = q3_total - q1_total
    pace_ratio = q3_cumul_total / (ou_line * 0.75) if ou_line > 0 else None
    scoring_variance = float(np.std([q1_total, q2_total, q3_total]))

    if pace_ratio is None:
        return None

    return {
        'avg_q_pace': avg_q_pace,
        'q3_lead': q3_lead,
        'q3_total': q3_total,
        'pace_trend': pace_trend,
        'q3_cumul_total': q3_cumul_total,
        'h1_total': h1_total,
        'scoring_variance': scoring_variance,
        'late_q3_pts': late_q3_pts,
        'pace_ratio': pace_ratio,
    }


def get_tier(margin):
    """Determine signal tier based on opening-line margin."""
    for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
        if margin >= TIER_THRESHOLDS[tier_name]['margin']:
            return tier_name
    return None


def get_season(game_id):
    """Determine season from game ID."""
    try:
        gid = int(game_id)
        return '2021-22' if gid < 401400000 else '2022-23'
    except (ValueError, TypeError):
        return 'unknown'


def extract_and_evaluate(filepath):
    """Extract game data from ESPN JSON and evaluate Q3 O/U signal."""
    with open(filepath) as f:
        data = json.load(f)

    plays = data.get("plays", [])
    if not plays:
        return None, "no_plays"

    game_id = filepath.stem

    # Get O/U line from pickcenter - STRICT: must exist, no fallback
    ou_line = None
    pickcenter = data.get("pickcenter", [])
    for pc in pickcenter:
        if "overUnder" in pc and pc["overUnder"] is not None:
            ou_line = float(pc["overUnder"])
            break

    if ou_line is None or ou_line <= 0:
        return None, "no_ou_line"

    # Get team info and date from header
    header = data.get("header", {})
    competitions = header.get("competitions", [])
    home_team = away_team = ""
    game_date = ""
    if competitions:
        comp = competitions[0]
        game_date = comp.get("date", "")
        for c in comp.get("competitors", []):
            if c.get("homeAway") == "home":
                home_team = c.get("team", {}).get("abbreviation", "")
            else:
                away_team = c.get("team", {}).get("abbreviation", "")

    # Track scores at end of each quarter
    quarter_end_scores = {}
    scoring_timeline = []
    last_away = 0
    last_home = 0
    last_period = 1

    for play in plays:
        period = play.get("period", {}).get("number", 0)
        away_score = play.get("awayScore", last_away)
        home_score = play.get("homeScore", last_home)

        # Parse clock
        clock_str = play.get("clock", {}).get("displayValue", "0:00")
        try:
            parts = clock_str.replace(".", ":").split(":")
            mins = int(parts[0])
            secs = float(parts[1]) if len(parts) > 1 else 0
            clock_secs = mins * 60 + secs
        except (ValueError, IndexError):
            clock_secs = 0

        # Track scoring plays
        if play.get("scoringPlay", False):
            pts = play.get("scoreValue", 0)
            scoring_timeline.append({
                "period": period,
                "clock_secs": clock_secs,
                "away_score": away_score,
                "home_score": home_score,
                "points": pts,
            })

        # Track end of quarter
        if period != last_period:
            quarter_end_scores[last_period] = (last_away, last_home)

        last_away = away_score
        last_home = home_score
        last_period = period

    # Final period end
    quarter_end_scores[last_period] = (last_away, last_home)

    # Must have at least 4 quarters
    if 3 not in quarter_end_scores or 4 not in quarter_end_scores:
        return None, "incomplete_game"

    # Check for overtime - STRICT: only regulation games
    has_ot = max(quarter_end_scores.keys()) > 4
    if has_ot:
        return None, "overtime"

    # Calculate per-quarter scoring
    q_scores = {}
    prev_away, prev_home = 0, 0
    for q in sorted(quarter_end_scores.keys()):
        away_q, home_q = quarter_end_scores[q]
        q_scores[q] = {
            "away": away_q - prev_away,
            "home": home_q - prev_home,
            "total": (away_q - prev_away) + (home_q - prev_home),
        }
        prev_away, prev_home = away_q, home_q

    q1_total = q_scores.get(1, {}).get("total", 0)
    q2_total = q_scores.get(2, {}).get("total", 0)
    q3_total = q_scores.get(3, {}).get("total", 0)
    q4_total = q_scores.get(4, {}).get("total", 0)

    # Q3 end scores
    q3_away, q3_home = quarter_end_scores[3]
    q3_cumul_total = q3_away + q3_home

    # Final scores (regulation only)
    final_away, final_home = quarter_end_scores[4]
    final_total = final_away + final_home

    # Late Q3 scoring (last 3 minutes = 180 seconds)
    late_q3_pts = 0
    q3_scoring = [s for s in scoring_timeline if s["period"] == 3 and s["clock_secs"] <= 180]
    if q3_scoring:
        late_q3_pts = sum(s["points"] for s in q3_scoring)

    # STRICT: validate scoring data quality
    q3_all_scoring = [s for s in scoring_timeline if s["period"] == 3]
    if not q3_all_scoring and q3_total > 0:
        return None, "no_q3_scoring_plays"

    if q1_total <= 0 or q2_total <= 0 or q3_total <= 0:
        return None, "zero_quarter"

    # Compute features
    features = compute_features(q3_home, q3_away, q1_total, q2_total, q3_total, ou_line, late_q3_pts)
    if features is None:
        return None, "feature_computation_failed"

    # Predict Q4 (our model)
    predicted_q4 = predict_q4(features)
    predicted_final = q3_cumul_total + predicted_q4

    # Estimate market Q4 and live line
    market_q4 = estimate_market_q4(features['avg_q_pace'], features['q3_lead'], ou_line)
    live_ou = q3_cumul_total + market_q4

    # Opening line metrics
    opening_margin = abs(predicted_final - ou_line)
    direction = 'OVER' if predicted_final > ou_line else 'UNDER'

    # Live line metrics
    live_margin = abs(predicted_final - live_ou)
    live_direction = 'OVER' if predicted_final > live_ou else 'UNDER'

    # Check tier (based on opening margin)
    tier = get_tier(opening_margin)

    # Determine actual outcomes
    actual_direction_opening = 'OVER' if final_total > ou_line else 'UNDER'
    actual_direction_live = 'OVER' if final_total > live_ou else 'UNDER'
    is_push_opening = final_total == ou_line

    # Season label
    season = get_season(game_id)

    # Build result
    result = {
        'gameId': game_id,
        'date': game_date[:10] if game_date else '',
        'season': season,
        'homeTeam': home_team,
        'awayTeam': away_team,

        # Our prediction
        'direction': direction,
        'tier': tier,
        'predictedQ4': round(predicted_q4, 1),
        'predictedFinal': round(predicted_final, 1),

        # Opening line analysis
        'ouLine': ou_line,
        'openingMargin': round(opening_margin, 1),
        'openingCorrect': direction == actual_direction_opening and not is_push_opening,
        'actualDirectionOpening': actual_direction_opening,
        'isPushOpening': is_push_opening,

        # Live line analysis (honest edge)
        'liveOuEstimate': round(live_ou, 1),
        'liveMargin': round(live_margin, 1),
        'liveDirection': live_direction,
        'liveCorrect': live_direction == actual_direction_live,
        'actualDirectionLive': actual_direction_live,
        'marketQ4Estimate': round(market_q4, 1),

        # Game state
        'q3CumulTotal': q3_cumul_total,
        'ptsNeeded': round(ou_line - q3_cumul_total, 1),
        'q1Total': q1_total,
        'q2Total': q2_total,
        'q3Total': q3_total,
        'q4Total': q4_total,
        'finalTotal': final_total,
        'avgQPace': round(features['avg_q_pace'], 1),
        'paceRatio': round(features['pace_ratio'], 3),
        'q3Lead': features['q3_lead'],
        'lateQ3Pts': late_q3_pts,
        'homeScore': q3_home,
        'awayScore': q3_away,
    }

    # Add edge/kelly only for signals (tier is not None)
    if tier is not None:
        cv_accuracy = TIER_THRESHOLDS[tier]['cv_accuracy']
        opening_ev = cv_accuracy * WIN_PAYOUT - (1 - cv_accuracy) * 1.0
        kelly_numer = cv_accuracy * (1 + WIN_PAYOUT) - 1
        kelly = max(0, kelly_numer / WIN_PAYOUT) if WIN_PAYOUT > 0 else 0
        kelly_adjusted = min(kelly * 0.25, 0.15)

        result['cvAccuracy'] = cv_accuracy
        result['openingEdge'] = round(opening_ev, 4)
        result['openingKelly'] = round(kelly_adjusted, 4)
        result['openingRoi'] = round(opening_ev * 100, 1)

        # Live edge is ~0 (breakeven)
        live_acc = 0.524
        live_ev = live_acc * WIN_PAYOUT - (1 - live_acc) * 1.0
        result['liveEdge'] = round(live_ev, 4)
        result['liveRoi'] = round(live_ev * 100, 1)

        return result, "signal"
    else:
        return result, "no_signal"


def main():
    print("=" * 70)
    print("GENERATING HISTORICAL Q3 O/U SIGNALS")
    print("With Live Line Estimation & Leakage-Free Validation")
    print("=" * 70)

    pbp_files = sorted(PBP_DIR.glob("*.json"))
    print(f"\nFound {len(pbp_files)} game files")

    signals = []
    no_signals = []
    skip_reasons = defaultdict(int)
    all_games = []

    for filepath in pbp_files:
        try:
            result, reason = extract_and_evaluate(filepath)
            if result is None:
                skip_reasons[reason] += 1
            elif reason == "signal":
                signals.append(result)
                all_games.append(result)
            elif reason == "no_signal":
                no_signals.append(result)
                all_games.append(result)
        except Exception as e:
            skip_reasons[f"error:{type(e).__name__}"] += 1

    print(f"\nResults:")
    print(f"  Games with signals:    {len(signals)}")
    print(f"  Games without signals: {len(no_signals)}")
    print(f"  Total valid games:     {len(all_games)}")
    print(f"\nSkipped games:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # ================================================================
    # VALIDATION - Separate in-sample from out-of-sample
    # ================================================================
    print(f"\n{'='*70}")
    print("SIGNAL VALIDATION (Leakage-Free)")
    print(f"{'='*70}")

    # Split signals by season
    insample_signals = [s for s in signals if s['season'] == '2021-22']
    oos_signals = [s for s in signals if s['season'] == '2022-23']

    print(f"\n  WARNING: 2021-22 signals are IN-SAMPLE (model trained on this data).")
    print(f"  Only 2022-23 results reflect true out-of-sample performance.\n")

    for label, sigs, is_oos in [
        ("2021-22 (IN-SAMPLE — do not trust)", insample_signals, False),
        ("2022-23 (OUT-OF-SAMPLE — true performance)", oos_signals, True),
    ]:
        if not sigs:
            continue

        print(f"\n  Season: {label}")
        print(f"  {'─'*60}")

        # Opening line accuracy
        opening_correct = sum(1 for s in sigs if s['openingCorrect'])
        opening_incorrect = sum(1 for s in sigs if not s['openingCorrect'] and not s['isPushOpening'])
        opening_push = sum(1 for s in sigs if s['isPushOpening'])
        opening_acc = opening_correct / len(sigs) if sigs else 0

        # Live line accuracy
        live_correct = sum(1 for s in sigs if s['liveCorrect'])
        live_incorrect = len(sigs) - live_correct
        live_acc = live_correct / len(sigs) if sigs else 0

        print(f"    Signals: {len(sigs)}")
        print(f"    vs OPENING line: {opening_correct}W-{opening_incorrect}L-{opening_push}P = {opening_acc:.1%}")
        print(f"    vs LIVE line:    {live_correct}W-{live_incorrect}L = {live_acc:.1%}")
        print(f"    (Breakeven at -110: {BREAKEVEN_PCT:.1%})")

        # By tier
        for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
            tier_sigs = [s for s in sigs if s['tier'] == tier_name]
            if not tier_sigs:
                continue
            o_correct = sum(1 for s in tier_sigs if s['openingCorrect'])
            o_acc = o_correct / len(tier_sigs)
            l_correct = sum(1 for s in tier_sigs if s['liveCorrect'])
            l_acc = l_correct / len(tier_sigs)
            print(f"\n    {tier_name} (margin >= {TIER_THRESHOLDS[tier_name]['margin']}):")
            print(f"      Games: {len(tier_sigs)}")
            print(f"      vs Opening: {o_acc:.1%}  |  vs Live: {l_acc:.1%}")

    # ================================================================
    # PROFITABILITY
    # ================================================================
    print(f"\n{'='*70}")
    print("PROFITABILITY SIMULATION (flat 1-unit bets at -110)")
    print(f"{'='*70}")

    for label, sigs in [("OOS 2022-23", oos_signals), ("IN-SAMPLE 2021-22", insample_signals)]:
        print(f"\n  --- {label} ---")
        for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
            tier_sigs = [s for s in sigs if s['tier'] == tier_name]
            if not tier_sigs:
                continue

            # vs Opening line
            o_wins = sum(1 for s in tier_sigs if s['openingCorrect'])
            o_losses = sum(1 for s in tier_sigs if not s['openingCorrect'] and not s['isPushOpening'])
            o_pnl = o_wins * WIN_PAYOUT - o_losses
            o_roi = o_pnl / len(tier_sigs) * 100

            # vs Live line
            l_wins = sum(1 for s in tier_sigs if s['liveCorrect'])
            l_losses = len(tier_sigs) - l_wins
            l_pnl = l_wins * WIN_PAYOUT - l_losses
            l_roi = l_pnl / len(tier_sigs) * 100

            print(f"\n  {tier_name}: {len(tier_sigs)} signals")
            print(f"    vs Opening: {o_wins}W-{o_losses}L, PnL={o_pnl:+.1f}u, ROI={o_roi:+.1f}%")
            print(f"    vs Live:    {l_wins}W-{l_losses}L, PnL={l_pnl:+.1f}u, ROI={l_roi:+.1f}%")

    # ================================================================
    # EQUITY CURVE (OOS only, both opening and live)
    # ================================================================

    # Sort OOS signals by date for equity curve
    oos_signals.sort(key=lambda x: x.get('date', ''))

    cumulative_opening_pnl = 0
    cumulative_live_pnl = 0
    equity_curve = []

    for s in oos_signals:
        # Opening line P&L
        if s['openingCorrect']:
            opening_pnl = WIN_PAYOUT
        elif s['isPushOpening']:
            opening_pnl = 0
        else:
            opening_pnl = -1.0
        cumulative_opening_pnl += opening_pnl

        # Live line P&L
        if s['liveCorrect']:
            live_pnl = WIN_PAYOUT
        else:
            live_pnl = -1.0
        cumulative_live_pnl += live_pnl

        s['openingPnl'] = round(opening_pnl, 4)
        s['cumOpeningPnl'] = round(cumulative_opening_pnl, 2)
        s['livePnl'] = round(live_pnl, 4)
        s['cumLivePnl'] = round(cumulative_live_pnl, 2)

        equity_curve.append({
            'date': s['date'],
            'tier': s['tier'],
            'openingCorrect': s['openingCorrect'],
            'liveCorrect': s['liveCorrect'],
            'openingPnl': round(opening_pnl, 4),
            'livePnl': round(live_pnl, 4),
            'cumOpeningPnl': round(cumulative_opening_pnl, 2),
            'cumLivePnl': round(cumulative_live_pnl, 2),
        })

    # Also add cumulative to insample signals for completeness
    insample_signals.sort(key=lambda x: x.get('date', ''))
    cum_is_opening = 0
    cum_is_live = 0
    for s in insample_signals:
        if s['openingCorrect']:
            o_pnl = WIN_PAYOUT
        elif s['isPushOpening']:
            o_pnl = 0
        else:
            o_pnl = -1.0
        l_pnl = WIN_PAYOUT if s['liveCorrect'] else -1.0
        cum_is_opening += o_pnl
        cum_is_live += l_pnl
        s['openingPnl'] = round(o_pnl, 4)
        s['cumOpeningPnl'] = round(cum_is_opening, 2)
        s['livePnl'] = round(l_pnl, 4)
        s['cumLivePnl'] = round(cum_is_live, 2)

    # ================================================================
    # EXPORT
    # ================================================================
    print(f"\n{'='*70}")
    print("EXPORTING FOR WEBAPP")
    print(f"{'='*70}")

    # Combine all signals sorted by date
    all_signals_sorted = sorted(signals, key=lambda x: x.get('date', ''))

    # Summary stats (OOS only for honest reporting)
    oos_correct_opening = sum(1 for s in oos_signals if s['openingCorrect'])
    oos_incorrect_opening = sum(1 for s in oos_signals if not s['openingCorrect'] and not s['isPushOpening'])
    oos_push_opening = sum(1 for s in oos_signals if s['isPushOpening'])
    oos_correct_live = sum(1 for s in oos_signals if s['liveCorrect'])

    webapp_data = {
        'generated': datetime.now().isoformat(),
        'version': '2.0',
        'totalGames': len(all_games),
        'totalSignals': len(signals),
        'oosSignals': len(oos_signals),
        'insampleSignals': len(insample_signals),
        'signals': all_signals_sorted,
        'equityCurve': equity_curve,
        'methodology': {
            'model': '9-feature linear regression predicting Q4 total',
            'training': '2021-22 season (1,162 games)',
            'validation': '2022-23 season (OOS)',
            'marketProxy': '3-feature linear model (pace, lead, pre-game O/U) simulating live line',
            'confidenceSource': '5-fold CV on training set (no test-set leakage)',
            'leakageNotes': [
                '2021-22 results are IN-SAMPLE and should not be trusted for accuracy claims',
                'Tier accuracy values from CV on training set, not from test set',
                'Live line estimated via market proxy to assess realistic edge',
                'Market proxy RMSE (8.65) nearly identical to our model RMSE (8.64)',
            ],
        },
        'summary': {
            'oos': {
                'signals': len(oos_signals),
                'vsOpening': {
                    'correct': oos_correct_opening,
                    'incorrect': oos_incorrect_opening,
                    'pushes': oos_push_opening,
                    'accuracy': round(oos_correct_opening / len(oos_signals), 4) if oos_signals else 0,
                    'pnl': round(oos_correct_opening * WIN_PAYOUT - oos_incorrect_opening, 2),
                    'roi': round((oos_correct_opening * WIN_PAYOUT - oos_incorrect_opening) / len(oos_signals) * 100, 1) if oos_signals else 0,
                    'note': 'vs stale opening line — NOT achievable in live market',
                },
                'vsLive': {
                    'correct': oos_correct_live,
                    'incorrect': len(oos_signals) - oos_correct_live,
                    'accuracy': round(oos_correct_live / len(oos_signals), 4) if oos_signals else 0,
                    'pnl': round(oos_correct_live * WIN_PAYOUT - (len(oos_signals) - oos_correct_live), 2),
                    'roi': round((oos_correct_live * WIN_PAYOUT - (len(oos_signals) - oos_correct_live)) / len(oos_signals) * 100, 1) if oos_signals else 0,
                    'note': 'vs estimated live line — honest edge assessment',
                },
            },
            'byTier': {},
        }
    }

    for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
        tier_oos = [s for s in oos_signals if s['tier'] == tier_name]
        if not tier_oos:
            continue
        o_wins = sum(1 for s in tier_oos if s['openingCorrect'])
        o_losses = sum(1 for s in tier_oos if not s['openingCorrect'] and not s['isPushOpening'])
        o_pushes = sum(1 for s in tier_oos if s['isPushOpening'])
        l_wins = sum(1 for s in tier_oos if s['liveCorrect'])

        webapp_data['summary']['byTier'][tier_name] = {
            'signals': len(tier_oos),
            'vsOpening': {
                'wins': o_wins,
                'losses': o_losses,
                'pushes': o_pushes,
                'accuracy': round(o_wins / len(tier_oos), 4),
                'pnl': round(o_wins * WIN_PAYOUT - o_losses, 2),
                'roi': round((o_wins * WIN_PAYOUT - o_losses) / len(tier_oos) * 100, 1),
            },
            'vsLive': {
                'wins': l_wins,
                'losses': len(tier_oos) - l_wins,
                'accuracy': round(l_wins / len(tier_oos), 4),
                'pnl': round(l_wins * WIN_PAYOUT - (len(tier_oos) - l_wins), 2),
                'roi': round((l_wins * WIN_PAYOUT - (len(tier_oos) - l_wins)) / len(tier_oos) * 100, 1),
            },
        }

    # Save full JSON
    output_path = OUTPUT_DIR / "q3_ou_historical_signals.json"
    with open(output_path, 'w') as f:
        json.dump(webapp_data, f, indent=2)
    print(f"  Full data saved to {output_path}")
    print(f"  Total signals: {len(signals)}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Save webapp JS module
    js_output = WEBAPP_DIR / "historical-data.js"
    js_content = f"// Auto-generated historical Q3 O/U signals from {len(all_games)} ESPN games\n"
    js_content += f"// Generated: {datetime.now().isoformat()}\n"
    js_content += f"// Version 2.0: Includes live line estimates and leakage-free validation\n"
    js_content += f"// OOS signals (2022-23): {len(oos_signals)}\n"
    js_content += f"// In-sample signals (2021-22): {len(insample_signals)} (do not trust accuracy)\n\n"
    js_content += f"window.HistoricalData = {json.dumps(webapp_data, indent=2)};\n"

    with open(js_output, 'w') as f:
        f.write(js_content)
    print(f"  Webapp JS saved to {js_output}")
    print(f"  File size: {js_output.stat().st_size / 1024:.1f} KB")

    return webapp_data


if __name__ == "__main__":
    main()
