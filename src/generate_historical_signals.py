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

# Model coefficients (same as q3_ou_strategy.py and q3-ou-engine.js)
MODEL_COEFFICIENTS = {
    'avg_q_pace': 0.0520,
    'q3_lead': 0.0572,
    'q3_total': 0.1064,
    'pace_trend': 0.0242,
    'q3_cumul_total': 0.1561,
    'h1_total': 0.0497,
    'scoring_variance': 0.0583,
    'late_q3_pts': -0.0483,
    'pace_ratio': -39.5227,
    'intercept': 52.1462,
}

TIER_THRESHOLDS = {
    'PLATINUM': {'margin': 20, 'accuracy': 0.994},
    'GOLD': {'margin': 15, 'accuracy': 0.983},
    'SILVER': {'margin': 12, 'accuracy': 0.977},
    'BRONZE': {'margin': 10, 'accuracy': 0.957},
}

WIN_PAYOUT = 100 / 110  # 0.9091


def predict_q4(features):
    """Predict Q4 total from features using linear model."""
    prediction = MODEL_COEFFICIENTS['intercept']
    for feat, coef in MODEL_COEFFICIENTS.items():
        if feat == 'intercept':
            continue
        if feat in features:
            prediction += coef * features[feat]
    return prediction


def compute_features(home_score, away_score, q1_total, q2_total, q3_total, ou_line, late_q3_pts):
    """Compute all model features from raw game state."""
    q3_cumul_total = home_score + away_score
    h1_total = q1_total + q2_total
    avg_q_pace = q3_cumul_total / 3.0
    q3_lead = abs(home_score - away_score)
    pace_trend = q3_total - q1_total
    pace_ratio = q3_cumul_total / (ou_line * 0.75) if ou_line > 0 else None
    mean_q = (q1_total + q2_total + q3_total) / 3.0
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
    """Determine signal tier."""
    for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
        if margin >= TIER_THRESHOLDS[tier_name]['margin']:
            return tier_name
    return None


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
        except:
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

    # STRICT: late_q3_pts must be computed, not defaulted
    # If we have scoring plays in Q3 but none in last 3 min, late_q3_pts = 0 is correct
    # If we have no scoring plays at all in Q3, that's suspicious
    q3_all_scoring = [s for s in scoring_timeline if s["period"] == 3]
    if not q3_all_scoring and q3_total > 0:
        # Q3 had scoring but no scoring plays tracked - data quality issue
        return None, "no_q3_scoring_plays"

    # Validate quarter totals are reasonable
    if q1_total <= 0 or q2_total <= 0 or q3_total <= 0:
        return None, "zero_quarter"

    # Compute features
    features = compute_features(q3_home, q3_away, q1_total, q2_total, q3_total, ou_line, late_q3_pts)
    if features is None:
        return None, "feature_computation_failed"

    # Predict Q4
    predicted_q4 = predict_q4(features)
    predicted_final = q3_cumul_total + predicted_q4
    margin = abs(predicted_final - ou_line)
    direction = 'OVER' if predicted_final > ou_line else 'UNDER'

    # Check tier
    tier = get_tier(margin)

    # Determine actual outcome
    actual_direction = 'OVER' if final_total > ou_line else 'UNDER'
    is_push = final_total == ou_line

    # Signal result
    if tier is not None:
        confidence = TIER_THRESHOLDS[tier]['accuracy']
        ev = confidence * WIN_PAYOUT - (1 - confidence) * 1.0
        kelly_numer = confidence * (1 + WIN_PAYOUT) - 1
        kelly = max(0, kelly_numer / WIN_PAYOUT) if WIN_PAYOUT > 0 else 0
        kelly_adjusted = min(kelly * 0.25, 0.15)

        correct = direction == actual_direction and not is_push

        signal = {
            'gameId': game_id,
            'date': game_date[:10] if game_date else '',
            'homeTeam': home_team,
            'awayTeam': away_team,
            'direction': direction,
            'tier': tier,
            'confidence': confidence,
            'margin': round(margin, 1),
            'predictedQ4': round(predicted_q4, 1),
            'predictedFinal': round(predicted_final, 1),
            'ouLine': ou_line,
            'q3CumulTotal': q3_cumul_total,
            'ptsNeeded': round(ou_line - q3_cumul_total, 1),
            'q1Total': q1_total,
            'q2Total': q2_total,
            'q3Total': q3_total,
            'q4Total': q4_total,
            'finalTotal': final_total,
            'actualDirection': actual_direction,
            'correct': correct,
            'isPush': is_push,
            'edge': round(ev, 4),
            'kelly': round(kelly_adjusted, 4),
            'roi': round(ev * 100, 1),
            'avgQPace': round(features['avg_q_pace'], 1),
            'paceRatio': round(features['pace_ratio'], 3),
            'q3Lead': features['q3_lead'],
            'lateQ3Pts': late_q3_pts,
            'homeScore': q3_home,
            'awayScore': q3_away,
        }
        return signal, "signal"
    else:
        # No signal (margin too low), but still track for stats
        return {
            'gameId': game_id,
            'date': game_date[:10] if game_date else '',
            'homeTeam': home_team,
            'awayTeam': away_team,
            'direction': direction,
            'tier': None,
            'margin': round(margin, 1),
            'predictedQ4': round(predicted_q4, 1),
            'predictedFinal': round(predicted_final, 1),
            'ouLine': ou_line,
            'q3CumulTotal': q3_cumul_total,
            'finalTotal': final_total,
            'actualDirection': actual_direction,
            'correct': direction == actual_direction and not is_push,
            'isPush': is_push,
        }, "no_signal"


def main():
    print("=" * 70)
    print("GENERATING HISTORICAL Q3 O/U SIGNALS")
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

    # Validate signals
    print(f"\n{'='*70}")
    print("SIGNAL VALIDATION")
    print(f"{'='*70}")

    # Split by season for validation
    def get_season(game_id):
        try:
            gid = int(game_id)
            if gid < 401400000:
                return '2021-22'
            else:
                return '2022-23'
        except:
            return 'unknown'

    for s in signals:
        s['season'] = get_season(s['gameId'])
    for s in all_games:
        s['season'] = get_season(s['gameId'])

    # Overall signal accuracy
    total_correct = sum(1 for s in signals if s['correct'])
    total_incorrect = sum(1 for s in signals if not s['correct'] and not s['isPush'])
    total_push = sum(1 for s in signals if s['isPush'])

    print(f"\nOverall signal accuracy: {total_correct}/{len(signals)} = {total_correct/len(signals):.1%}")
    print(f"  Correct: {total_correct}")
    print(f"  Incorrect: {total_incorrect}")
    print(f"  Push: {total_push}")

    # By tier
    for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
        tier_signals = [s for s in signals if s['tier'] == tier_name]
        if not tier_signals:
            continue
        correct = sum(1 for s in tier_signals if s['correct'])
        incorrect = sum(1 for s in tier_signals if not s['correct'] and not s['isPush'])
        pushes = sum(1 for s in tier_signals if s['isPush'])
        acc = correct / len(tier_signals) if tier_signals else 0

        print(f"\n  {tier_name} (margin >= {TIER_THRESHOLDS[tier_name]['margin']}):")
        print(f"    Games: {len(tier_signals)}")
        print(f"    W-L-P: {correct}-{incorrect}-{pushes}")
        print(f"    Accuracy: {acc:.1%}")

        # By direction
        for direction in ['OVER', 'UNDER']:
            dir_signals = [s for s in tier_signals if s['direction'] == direction]
            if not dir_signals:
                continue
            dir_correct = sum(1 for s in dir_signals if s['correct'])
            dir_acc = dir_correct / len(dir_signals) if dir_signals else 0
            print(f"    {direction}: {len(dir_signals)} games, {dir_correct} correct, {dir_acc:.1%}")

    # By season
    for season in ['2021-22', '2022-23']:
        season_signals = [s for s in signals if s['season'] == season]
        if not season_signals:
            continue
        correct = sum(1 for s in season_signals if s['correct'])
        acc = correct / len(season_signals)
        print(f"\n  Season {season}: {len(season_signals)} signals, {correct} correct, {acc:.1%}")

        for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
            tier_signals = [s for s in season_signals if s['tier'] == tier_name]
            if not tier_signals:
                continue
            t_correct = sum(1 for s in tier_signals if s['correct'])
            t_acc = t_correct / len(tier_signals)
            print(f"    {tier_name}: {len(tier_signals)} signals, {t_correct} correct, {t_acc:.1%}")

    # Profitability simulation
    print(f"\n{'='*70}")
    print("PROFITABILITY SIMULATION (flat 1-unit bets at -110)")
    print(f"{'='*70}")

    for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
        tier_signals = [s for s in signals if s['tier'] == tier_name]
        if not tier_signals:
            continue
        wins = sum(1 for s in tier_signals if s['correct'])
        losses = sum(1 for s in tier_signals if not s['correct'] and not s['isPush'])
        pnl = wins * WIN_PAYOUT - losses * 1.0
        roi = pnl / len(tier_signals) * 100 if tier_signals else 0
        print(f"\n  {tier_name}:")
        print(f"    Trades: {len(tier_signals)}, W-L: {wins}-{losses}")
        print(f"    P&L: {pnl:+.1f} units")
        print(f"    ROI: {roi:+.1f}%")

    # Sort signals by date
    signals.sort(key=lambda x: x.get('date', ''))

    # Export for webapp
    print(f"\n{'='*70}")
    print("EXPORTING FOR WEBAPP")
    print(f"{'='*70}")

    # Calculate cumulative P&L for equity curve
    cumulative_pnl = 0
    bankroll = 1000
    equity_curve = []
    for s in signals:
        if s['correct']:
            pnl = WIN_PAYOUT
            bankroll_pnl = bankroll * s['kelly'] * WIN_PAYOUT
        elif s['isPush']:
            pnl = 0
            bankroll_pnl = 0
        else:
            pnl = -1.0
            bankroll_pnl = -bankroll * s['kelly']
        cumulative_pnl += pnl
        bankroll += bankroll_pnl
        s['cumulativePnl'] = round(cumulative_pnl, 2)
        s['bankroll'] = round(bankroll, 2)
        equity_curve.append({
            'date': s['date'],
            'pnl': round(pnl, 4),
            'cumPnl': round(cumulative_pnl, 2),
            'bankroll': round(bankroll, 2),
            'tier': s['tier'],
            'correct': s['correct'],
        })

    # Summary stats for webapp
    webapp_data = {
        'generated': datetime.now().isoformat(),
        'totalGames': len(all_games),
        'totalSignals': len(signals),
        'signals': signals,
        'equityCurve': equity_curve,
        'summary': {
            'overall': {
                'signals': len(signals),
                'correct': total_correct,
                'incorrect': total_incorrect,
                'pushes': total_push,
                'accuracy': round(total_correct / len(signals), 4) if signals else 0,
                'pnl': round(total_correct * WIN_PAYOUT - total_incorrect, 2),
                'roi': round((total_correct * WIN_PAYOUT - total_incorrect) / len(signals) * 100, 1) if signals else 0,
            },
            'byTier': {},
            'bySeason': {},
        }
    }

    for tier_name in ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']:
        tier_sigs = [s for s in signals if s['tier'] == tier_name]
        if not tier_sigs:
            continue
        wins = sum(1 for s in tier_sigs if s['correct'])
        losses = sum(1 for s in tier_sigs if not s['correct'] and not s['isPush'])
        pushes = sum(1 for s in tier_sigs if s['isPush'])
        pnl = wins * WIN_PAYOUT - losses
        webapp_data['summary']['byTier'][tier_name] = {
            'signals': len(tier_sigs),
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'accuracy': round(wins / len(tier_sigs), 4),
            'pnl': round(pnl, 2),
            'roi': round(pnl / len(tier_sigs) * 100, 1),
        }

    for season in ['2021-22', '2022-23']:
        season_sigs = [s for s in signals if s['season'] == season]
        if not season_sigs:
            continue
        wins = sum(1 for s in season_sigs if s['correct'])
        losses = sum(1 for s in season_sigs if not s['correct'] and not s['isPush'])
        webapp_data['summary']['bySeason'][season] = {
            'signals': len(season_sigs),
            'wins': wins,
            'losses': losses,
            'accuracy': round(wins / len(season_sigs), 4),
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
    js_content += f"// Source: ESPN pickcenter O/U lines (no fallbacks, no assumptions)\n"
    js_content += f"// Games: 2021-22 and 2022-23 NBA seasons (regulation only)\n\n"
    js_content += f"window.HistoricalData = {json.dumps(webapp_data, indent=2)};\n"

    with open(js_output, 'w') as f:
        f.write(js_content)
    print(f"  Webapp JS saved to {js_output}")
    print(f"  File size: {js_output.stat().st_size / 1024:.1f} KB")

    return webapp_data


if __name__ == "__main__":
    main()
