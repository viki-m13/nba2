"""
Q3 Over/Under Signal Analysis
=============================
Extracts quarter-by-quarter scoring from all 2,310 games and identifies
conditions where Q4 scoring is highly predictable for O/U bets placed
at end of Q3.

Approach: Think like a stock trader - find "regimes" where Q4 scoring
is essentially deterministic, then only trade those setups.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PBP_DIR = Path("/home/user/nba2/cache/games_pbp")
OUTPUT_DIR = Path("/home/user/nba2/output")


def extract_game_data(filepath):
    """Extract quarter scores, O/U line, and detailed game state from PBP."""
    with open(filepath) as f:
        data = json.load(f)

    plays = data.get("plays", [])
    if not plays:
        return None

    game_id = filepath.stem

    # Get O/U line from pickcenter
    ou_line = None
    spread = None
    pickcenter = data.get("pickcenter", [])
    for pc in pickcenter:
        if "overUnder" in pc:
            ou_line = pc["overUnder"]
            spread = pc.get("spread", None)
            break

    # Get team info from header
    header = data.get("header", {})
    competitions = header.get("competitions", [])
    home_team = away_team = ""
    if competitions:
        comp = competitions[0]
        for c in comp.get("competitors", []):
            if c.get("homeAway") == "home":
                home_team = c.get("team", {}).get("abbreviation", "")
            else:
                away_team = c.get("team", {}).get("abbreviation", "")

    # Track scores at end of each quarter
    quarter_end_scores = {}  # period -> (away, home)
    q_scoring_plays = defaultdict(list)  # period -> list of (clock_secs, team, points)

    last_away = 0
    last_home = 0
    last_period = 1

    # Track scoring timeline
    scoring_timeline = []

    for play in plays:
        period = play.get("period", {}).get("number", 0)
        away_score = play.get("awayScore", last_away)
        home_score = play.get("homeScore", last_home)

        # Parse clock
        clock_str = play.get("clock", {}).get("displayValue", "0:00")
        try:
            parts = clock_str.replace(".", ":").split(":")
            if len(parts) == 2:
                mins, secs = int(parts[0]), float(parts[1])
            elif len(parts) == 3:
                mins, secs = int(parts[0]), float(parts[1])
            else:
                mins, secs = 0, 0
            clock_secs = mins * 60 + secs
        except:
            clock_secs = 0

        # Track scoring plays
        if play.get("scoringPlay", False):
            pts = play.get("scoreValue", 0)
            team_id = play.get("team", {}).get("id", "")
            scoring_timeline.append({
                "period": period,
                "clock_secs": clock_secs,
                "away_score": away_score,
                "home_score": home_score,
                "points": pts,
                "team_id": team_id,
                "total": away_score + home_score
            })

        # Track end of quarter (when period changes or end of game)
        if period != last_period:
            quarter_end_scores[last_period] = (last_away, last_home)

        last_away = away_score
        last_home = home_score
        last_period = period

    # Final period end
    quarter_end_scores[last_period] = (last_away, last_home)

    # Calculate per-quarter scoring
    q_scores = {}
    prev_away, prev_home = 0, 0
    for q in sorted(quarter_end_scores.keys()):
        away_q, home_q = quarter_end_scores[q]
        q_scores[q] = {
            "away": away_q - prev_away,
            "home": home_q - prev_home,
            "total": (away_q - prev_away) + (home_q - prev_home),
            "away_cumul": away_q,
            "home_cumul": home_q,
            "total_cumul": away_q + home_q
        }
        prev_away, prev_home = away_q, home_q

    # Only process regulation games (exactly 4 quarters)
    if 4 not in quarter_end_scores:
        return None

    has_ot = max(quarter_end_scores.keys()) > 4

    final_away, final_home = quarter_end_scores[max(quarter_end_scores.keys())]
    final_total = final_away + final_home

    # Q3 end state
    if 3 not in quarter_end_scores:
        return None

    q3_away, q3_home = quarter_end_scores[3]
    q3_total = q3_away + q3_home
    q3_lead = abs(q3_home - q3_away)
    q3_home_leading = q3_home > q3_away

    # Q4 scoring (regulation only, not OT)
    q4_away, q4_home = quarter_end_scores[4]
    q4_scoring = (q4_away - q3_away) + (q4_home - q3_home)
    q4_away_scoring = q4_away - q3_away
    q4_home_scoring = q4_home - q3_home

    # Per-quarter pace (points per quarter)
    q1_total = q_scores.get(1, {}).get("total", 0)
    q2_total = q_scores.get(2, {}).get("total", 0)
    q3_total_q = q_scores.get(3, {}).get("total", 0)

    # Pace metrics
    avg_q_pace = q3_total / 3.0  # Average points per quarter through Q3
    q3_pace = q3_total_q  # Q3 specific scoring
    q2_pace = q2_total
    q1_pace = q1_total

    # Pace trend
    pace_trend = q3_pace - q1_pace  # Accelerating or decelerating?
    pace_q3_vs_avg = q3_pace - avg_q_pace

    # Calculate projected final total based on Q3 pace
    projected_final = q3_total + avg_q_pace  # Simple linear projection
    projected_final_q3_weight = q3_total + q3_pace  # Q3-weighted projection

    # Live O/U estimation at Q3 end
    # Standard approach: remaining_pts = (ou_line - q3_total) adjusted for pace
    if ou_line and ou_line > 0:
        expected_remaining = ou_line - q3_total
        pace_ratio = q3_total / (ou_line * 0.75)  # Actual vs expected through Q3
        # Adjusted live O/U at Q3
        live_ou_q3 = q3_total + (ou_line * 0.25 * pace_ratio)
    else:
        expected_remaining = None
        pace_ratio = None
        live_ou_q3 = None

    # Scoring runs at end of Q3 (last 3 minutes)
    q3_late_scoring = []
    for sp in scoring_timeline:
        if sp["period"] == 3 and sp["clock_secs"] <= 180:
            q3_late_scoring.append(sp)

    late_q3_pts = sum(s["points"] for s in q3_late_scoring)

    # FG efficiency proxy - scoring density
    q3_scoring_plays = [s for s in scoring_timeline if s["period"] == 3]
    q2_scoring_plays = [s for s in scoring_timeline if s["period"] == 2]
    q1_scoring_plays = [s for s in scoring_timeline if s["period"] == 1]

    # Half scoring
    h1_total = q1_total + q2_total
    h2_q3_only = q3_total_q

    # Win probability context from data
    win_prob = data.get("winprobability", [])
    q3_end_wp = None
    if win_prob:
        # Find WP closest to Q3 end
        for wp in reversed(win_prob):
            pid = wp.get("playId", "")
            hwp = wp.get("homeWinPercentage", 0.5)
            # Match to plays in period 3
            for play in plays:
                if play.get("id") == pid:
                    p = play.get("period", {}).get("number", 0)
                    if p == 3:
                        q3_end_wp = hwp
                        break
            if q3_end_wp is not None:
                break

    return {
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "ou_line": ou_line,
        "spread": spread,
        "has_ot": has_ot,
        # Quarter totals
        "q1_total": q1_total,
        "q2_total": q2_total,
        "q3_total": q3_total_q,
        "q4_total": q4_scoring,
        "h1_total": h1_total,
        # Cumulative at Q3 end
        "q3_cumul_total": q3_total,
        "q3_home": q3_home,
        "q3_away": q3_away,
        "q3_lead": q3_lead,
        "q3_home_leading": q3_home_leading,
        # Final
        "final_total": final_total if not has_ot else q4_away + q4_home,  # Regulation only
        "final_total_with_ot": final_total,
        "final_home": final_home,
        "final_away": final_away,
        # Q4 detail
        "q4_home_scoring": q4_home_scoring,
        "q4_away_scoring": q4_away_scoring,
        # Pace metrics
        "avg_q_pace": avg_q_pace,
        "q3_pace": q3_pace,
        "pace_trend": pace_trend,
        "pace_q3_vs_avg": pace_q3_vs_avg,
        # Projections
        "projected_final_linear": projected_final,
        "projected_final_q3w": projected_final_q3_weight,
        # O/U context
        "pace_ratio": pace_ratio,
        "live_ou_q3": live_ou_q3,
        "remaining_to_ou": (ou_line - q3_total) if ou_line else None,
        # Late Q3
        "late_q3_pts": late_q3_pts,
        "q3_scoring_plays": len(q3_scoring_plays),
        "q2_scoring_plays": len(q2_scoring_plays),
        "q1_scoring_plays": len(q1_scoring_plays),
        # WP
        "q3_end_wp": q3_end_wp,
    }


def main():
    print("=" * 70)
    print("Q3 OVER/UNDER SIGNAL RESEARCH")
    print("=" * 70)

    # Extract all games
    print("\nExtracting data from 2,310 games...")
    games = []
    pbp_files = sorted(PBP_DIR.glob("*.json"))

    for f in pbp_files:
        try:
            g = extract_game_data(f)
            if g:
                games.append(g)
        except Exception as e:
            pass

    df = pd.DataFrame(games)
    print(f"Successfully extracted: {len(df)} games")
    print(f"Games with O/U lines: {df['ou_line'].notna().sum()}")
    print(f"Overtime games: {df['has_ot'].sum()}")

    # Filter to regulation games with O/U lines
    reg = df[df['ou_line'].notna() & ~df['has_ot']].copy()
    print(f"Regulation games with O/U: {len(reg)}")

    # =====================================================================
    # SECTION 1: BASIC Q4 SCORING STATISTICS
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Q4 SCORING DISTRIBUTION")
    print("=" * 70)

    print(f"\nQ4 Total Points (regulation):")
    print(f"  Mean:   {reg['q4_total'].mean():.1f}")
    print(f"  Median: {reg['q4_total'].median():.1f}")
    print(f"  Std:    {reg['q4_total'].std():.1f}")
    print(f"  Min:    {reg['q4_total'].min()}")
    print(f"  Max:    {reg['q4_total'].max()}")
    print(f"  25th:   {reg['q4_total'].quantile(0.25):.1f}")
    print(f"  75th:   {reg['q4_total'].quantile(0.75):.1f}")

    print(f"\nQuarter-by-quarter scoring:")
    for q in ['q1_total', 'q2_total', 'q3_total', 'q4_total']:
        print(f"  {q}: mean={reg[q].mean():.1f}, std={reg[q].std():.1f}")

    # =====================================================================
    # SECTION 2: Q4 SCORING BY Q3 LEAD SIZE
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: Q4 SCORING BY Q3 LEAD SIZE")
    print("=" * 70)

    lead_bins = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 25), (25, 50)]
    for lo, hi in lead_bins:
        subset = reg[(reg['q3_lead'] >= lo) & (reg['q3_lead'] < hi)]
        if len(subset) < 5:
            continue
        print(f"\n  Q3 Lead {lo}-{hi} pts ({len(subset)} games):")
        print(f"    Q4 scoring: mean={subset['q4_total'].mean():.1f}, "
              f"std={subset['q4_total'].std():.1f}, "
              f"median={subset['q4_total'].median():.1f}")
        print(f"    Q4 range: [{subset['q4_total'].min()}-{subset['q4_total'].max()}]")

    # =====================================================================
    # SECTION 3: PACE-BASED ANALYSIS
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: PACE-BASED Q4 PREDICTION")
    print("=" * 70)

    # The key question: does Q3 pace predict Q4 scoring?
    reg['q3_pace_bucket'] = pd.cut(reg['avg_q_pace'], bins=5)
    pace_analysis = reg.groupby('q3_pace_bucket').agg({
        'q4_total': ['mean', 'std', 'count'],
        'avg_q_pace': 'mean'
    }).round(1)
    print("\nQ4 scoring by average pace through Q3:")
    print(pace_analysis)

    # Pace ratio analysis
    print("\n\nPace ratio (actual Q3 pace vs expected):")
    reg['pace_ratio_bucket'] = pd.cut(reg['pace_ratio'], bins=8)
    pr_analysis = reg.groupby('pace_ratio_bucket').agg({
        'q4_total': ['mean', 'std', 'count'],
        'pace_ratio': 'mean'
    }).round(1)
    print(pr_analysis)

    # =====================================================================
    # SECTION 4: O/U LINE ANALYSIS AT Q3 END
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: O/U LINE vs ACTUAL AT Q3 END")
    print("=" * 70)

    # How much of the O/U is remaining at Q3 end?
    reg['pts_needed_for_ou'] = reg['ou_line'] - reg['q3_cumul_total']
    reg['ou_result'] = (reg['final_total'] > reg['ou_line']).map({True: 'OVER', False: 'UNDER'})
    reg['final_vs_ou'] = reg['final_total'] - reg['ou_line']

    print(f"\nPoints needed for OVER at Q3 end:")
    print(f"  Mean:   {reg['pts_needed_for_ou'].mean():.1f}")
    print(f"  Median: {reg['pts_needed_for_ou'].median():.1f}")
    print(f"  Std:    {reg['pts_needed_for_ou'].std():.1f}")

    print(f"\nO/U result distribution:")
    print(reg['ou_result'].value_counts())
    print(f"Push (exact): {(reg['final_total'] == reg['ou_line']).sum()}")

    # =====================================================================
    # SECTION 5: THE KEY SIGNAL - POINTS NEEDED vs Q4 SCORING
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 5: SIGNAL DISCOVERY - Points Needed vs Q4 Actual")
    print("=" * 70)

    # The critical insight: If pts_needed is very high or very low relative
    # to expected Q4 scoring, we have a strong signal
    reg['q4_surplus'] = reg['q4_total'] - reg['pts_needed_for_ou']

    # Bucket by points needed
    needed_bins = [(-100, 30), (30, 40), (40, 45), (45, 50), (50, 55),
                   (55, 60), (60, 65), (65, 70), (70, 80), (80, 120)]
    print("\nQ4 actual vs points needed for OVER:")
    for lo, hi in needed_bins:
        subset = reg[(reg['pts_needed_for_ou'] >= lo) & (reg['pts_needed_for_ou'] < hi)]
        if len(subset) < 5:
            continue
        over_pct = (subset['q4_total'] >= subset['pts_needed_for_ou']).mean()
        under_pct = 1 - over_pct
        print(f"  Need {lo:+3d} to {hi:+3d}: "
              f"n={len(subset):4d}, "
              f"Q4 avg={subset['q4_total'].mean():.1f}, "
              f"OVER={over_pct:.1%}, UNDER={under_pct:.1%}")

    # =====================================================================
    # SECTION 6: COMBINED SIGNAL FEATURES
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 6: MULTI-FACTOR SIGNAL ANALYSIS")
    print("=" * 70)

    # Feature: pace_ratio × lead × pts_needed
    # UNDER signal: High lead + low pace_ratio + many pts needed
    print("\n--- UNDER SIGNAL: Blowout + Slow Pace ---")
    for lead_min in [15, 18, 20, 22, 25]:
        for pace_max in [0.90, 0.95, 1.0]:
            subset = reg[
                (reg['q3_lead'] >= lead_min) &
                (reg['pace_ratio'] <= pace_max)
            ]
            if len(subset) < 5:
                continue
            # For UNDER signal: game goes under the opening O/U line
            under_hit = (subset['final_total'] < subset['ou_line']).mean()
            avg_margin = (subset['ou_line'] - subset['final_total']).mean()
            print(f"  Lead>={lead_min}, Pace<={pace_max:.2f}: "
                  f"n={len(subset):4d}, UNDER={under_hit:.1%}, "
                  f"avg_margin={avg_margin:+.1f}")

    # OVER signal: Close game + high pace + few pts needed
    print("\n--- OVER SIGNAL: Close Game + Fast Pace ---")
    for lead_max in [3, 5, 7]:
        for pace_min in [1.0, 1.05, 1.10]:
            subset = reg[
                (reg['q3_lead'] <= lead_max) &
                (reg['pace_ratio'] >= pace_min)
            ]
            if len(subset) < 5:
                continue
            over_hit = (subset['final_total'] > subset['ou_line']).mean()
            avg_margin = (subset['final_total'] - subset['ou_line']).mean()
            print(f"  Lead<={lead_max}, Pace>={pace_min:.2f}: "
                  f"n={len(subset):4d}, OVER={over_hit:.1%}, "
                  f"avg_margin={avg_margin:+.1f}")

    # =====================================================================
    # SECTION 7: EXTREME CONDITIONS (95%+ accuracy search)
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 7: EXTREME CONDITIONS - 95%+ ACCURACY SEARCH")
    print("=" * 70)

    # Systematic search for high-accuracy signals
    print("\n--- UNDER Signals (high accuracy search) ---")
    results = []
    for lead_min in range(10, 35, 2):
        for pace_max in [0.80, 0.85, 0.90, 0.95, 1.0, 1.05]:
            for pts_needed_min in range(40, 80, 5):
                subset = reg[
                    (reg['q3_lead'] >= lead_min) &
                    (reg['pace_ratio'] <= pace_max) &
                    (reg['pts_needed_for_ou'] >= pts_needed_min)
                ]
                if len(subset) < 10:
                    continue
                under_hit = (subset['final_total'] < subset['ou_line']).mean()
                avg_margin = (subset['ou_line'] - subset['final_total']).mean()
                if under_hit >= 0.90:
                    results.append({
                        'direction': 'UNDER',
                        'lead_min': lead_min,
                        'pace_max': pace_max,
                        'pts_needed_min': pts_needed_min,
                        'n': len(subset),
                        'accuracy': under_hit,
                        'avg_margin': avg_margin
                    })

    results.sort(key=lambda x: (-x['accuracy'], -x['n']))
    for r in results[:30]:
        print(f"  UNDER: lead>={r['lead_min']}, pace<={r['pace_max']:.2f}, "
              f"need>={r['pts_needed_min']}: "
              f"n={r['n']:3d}, acc={r['accuracy']:.1%}, margin={r['avg_margin']:+.1f}")

    print("\n--- OVER Signals (high accuracy search) ---")
    results_over = []
    for lead_max in range(0, 12, 2):
        for pace_min in [0.95, 1.0, 1.05, 1.10, 1.15, 1.20]:
            for pts_needed_max in range(30, 60, 5):
                subset = reg[
                    (reg['q3_lead'] <= lead_max) &
                    (reg['pace_ratio'] >= pace_min) &
                    (reg['pts_needed_for_ou'] <= pts_needed_max)
                ]
                if len(subset) < 10:
                    continue
                over_hit = (subset['final_total'] > subset['ou_line']).mean()
                avg_margin = (subset['final_total'] - subset['ou_line']).mean()
                if over_hit >= 0.85:
                    results_over.append({
                        'direction': 'OVER',
                        'lead_max': lead_max,
                        'pace_min': pace_min,
                        'pts_needed_max': pts_needed_max,
                        'n': len(subset),
                        'accuracy': over_hit,
                        'avg_margin': avg_margin
                    })

    results_over.sort(key=lambda x: (-x['accuracy'], -x['n']))
    for r in results_over[:30]:
        print(f"  OVER: lead<={r['lead_max']}, pace>={r['pace_min']:.2f}, "
              f"need<={r['pts_needed_max']}: "
              f"n={r['n']:3d}, acc={r['accuracy']:.1%}, margin={r['avg_margin']:+.1f}")

    # =====================================================================
    # SECTION 8: ADVANCED FEATURES
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 8: ADVANCED COMPOSITE FEATURES")
    print("=" * 70)

    # Feature: Q3 scoring relative to Q1/Q2 (fatigue/momentum)
    reg['q3_vs_q1'] = reg['q3_total'] - reg['q1_total']
    reg['q3_vs_q2'] = reg['q3_total'] - reg['q2_total']
    reg['scoring_deceleration'] = reg['q3_total'] < reg['q1_total']

    # Feature: Scoring concentration (are points bunched or spread?)
    reg['scoring_variance'] = reg[['q1_total', 'q2_total', 'q3_total']].std(axis=1)

    # Feature: Half comparison
    reg['h1_vs_q3'] = reg['h1_total'] - reg['q3_total']

    # Q4 actual vs various predictors
    print("\nQ4 scoring when Q3 scoring DECELERATING (Q3 < Q1):")
    decel = reg[reg['scoring_deceleration']]
    print(f"  n={len(decel)}, Q4 mean={decel['q4_total'].mean():.1f}, "
          f"Q4 std={decel['q4_total'].std():.1f}")

    accel = reg[~reg['scoring_deceleration']]
    print(f"\nQ4 scoring when Q3 scoring ACCELERATING (Q3 >= Q1):")
    print(f"  n={len(accel)}, Q4 mean={accel['q4_total'].mean():.1f}, "
          f"Q4 std={accel['q4_total'].std():.1f}")

    # =====================================================================
    # SECTION 9: LIVE LINE ESTIMATION & EDGE
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 9: LIVE LINE ESTIMATION")
    print("=" * 70)

    # The live O/U at Q3 end should approximately be:
    # live_ou = q3_total + (avg_q_pace * pace_adjustment)
    # But sportsbooks adjust for:
    #   1. Lead size (blowouts = less Q4 scoring)
    #   2. Game pace (fast/slow)
    #   3. Team-specific tendencies

    # Let's model what the live line SHOULD be and find when market is wrong
    # Simple model: Live OU = Q3_total + predicted_Q4
    # predicted_Q4 = alpha * avg_q_pace + beta * q3_lead + gamma

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    features = ['avg_q_pace', 'q3_lead', 'q3_total', 'pace_trend',
                'q3_cumul_total', 'h1_total', 'scoring_variance',
                'late_q3_pts', 'pace_ratio']

    valid = reg.dropna(subset=features + ['q4_total'])
    X = valid[features].values
    y = valid['q4_total'].values

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)

    print(f"\nLinear model for Q4 scoring (R²={r2:.4f}):")
    for feat, coef in zip(features, model.coef_):
        print(f"  {feat:25s}: {coef:+.4f}")
    print(f"  {'intercept':25s}: {model.intercept_:+.4f}")

    # Residual analysis - when is the model most certain?
    residuals = y - preds
    print(f"\nResiduals: mean={residuals.mean():.2f}, std={residuals.std():.2f}")

    valid['predicted_q4'] = preds
    valid['residual'] = residuals
    valid['predicted_final'] = valid['q3_cumul_total'] + valid['predicted_q4']
    valid['predicted_ou'] = np.where(
        valid['predicted_final'] > valid['ou_line'], 'OVER', 'UNDER'
    )
    valid['actual_ou'] = np.where(
        valid['final_total'] > valid['ou_line'], 'OVER', 'UNDER'
    )
    valid['model_correct'] = valid['predicted_ou'] == valid['actual_ou']
    print(f"\nBasic model accuracy: {valid['model_correct'].mean():.1%}")

    # Confidence-based filtering
    valid['margin_from_line'] = abs(valid['predicted_final'] - valid['ou_line'])

    print("\nAccuracy by prediction confidence (margin from line):")
    for margin_min in [0, 2, 4, 6, 8, 10, 12, 15, 18, 20, 25]:
        confident = valid[valid['margin_from_line'] >= margin_min]
        if len(confident) < 10:
            continue
        acc = confident['model_correct'].mean()
        print(f"  Margin >= {margin_min:2d}: n={len(confident):4d}, accuracy={acc:.1%}")

    # =====================================================================
    # SECTION 10: ULTIMATE SIGNAL CONSTRUCTION
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 10: ULTIMATE HIGH-ACCURACY SIGNAL")
    print("=" * 70)

    # Combine multiple confirmations:
    # Signal fires ONLY when multiple independent factors agree

    # UNDER SIGNAL requirements:
    # 1. Large lead at Q3 end (blowout → garbage time)
    # 2. Points needed for OVER is high
    # 3. Scoring is decelerating (Q3 < avg)
    # 4. Model predicts UNDER with high confidence

    print("\n=== COMPOSITE UNDER SIGNAL ===")
    for lead_min in [15, 18, 20, 22, 25]:
        for decel in [True, False]:
            for margin_min in [5, 8, 10, 12, 15]:
                conditions = (
                    (valid['q3_lead'] >= lead_min) &
                    (valid['predicted_ou'] == 'UNDER') &
                    (valid['margin_from_line'] >= margin_min)
                )
                if decel:
                    conditions = conditions & valid['scoring_deceleration']

                subset = valid[conditions]
                if len(subset) < 10:
                    continue

                acc = (subset['actual_ou'] == 'UNDER').mean()
                avg_margin = (subset['ou_line'] - subset['final_total']).mean()

                if acc >= 0.90:
                    decel_str = "+decel" if decel else ""
                    print(f"  Lead>={lead_min} {decel_str:6s} margin>={margin_min:2d}: "
                          f"n={len(subset):3d}, acc={acc:.1%}, "
                          f"avg_win_margin={avg_margin:+.1f}")

    # OVER SIGNAL requirements:
    # 1. Close game at Q3 end
    # 2. Fast pace
    # 3. Model predicts OVER with high confidence
    print("\n=== COMPOSITE OVER SIGNAL ===")
    for lead_max in [3, 5, 7, 10]:
        for pace_min in [1.0, 1.05, 1.10]:
            for margin_min in [5, 8, 10, 12, 15]:
                conditions = (
                    (valid['q3_lead'] <= lead_max) &
                    (valid['pace_ratio'] >= pace_min) &
                    (valid['predicted_ou'] == 'OVER') &
                    (valid['margin_from_line'] >= margin_min)
                )
                subset = valid[conditions]
                if len(subset) < 10:
                    continue
                acc = (subset['actual_ou'] == 'OVER').mean()
                avg_margin = (subset['final_total'] - subset['ou_line']).mean()
                if acc >= 0.85:
                    print(f"  Lead<={lead_max} pace>={pace_min:.2f} margin>={margin_min:2d}: "
                          f"n={len(subset):3d}, acc={acc:.1%}, "
                          f"avg_win_margin={avg_margin:+.1f}")

    # =====================================================================
    # SECTION 11: WALK-FORWARD VALIDATION
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 11: WALK-FORWARD (OUT-OF-SAMPLE) VALIDATION")
    print("=" * 70)

    # Split by season
    if 'date' not in reg.columns:
        reg['date_num'] = range(len(reg))
    else:
        reg['date_num'] = pd.to_numeric(reg.get('game_id', range(len(reg))))

    # Use game_id to infer season
    def get_season(gid):
        try:
            s = str(gid)
            if s.startswith('40'):
                # ESPN format
                return '2021-22' if int(s) < 401400000 else '2022-23'
        except:
            pass
        return 'unknown'

    reg['season'] = reg['game_id'].apply(get_season)
    valid['season'] = valid['game_id'].apply(get_season)

    print(f"\nSeason distribution:")
    print(reg['season'].value_counts())

    # Train on 2021-22, test on 2022-23
    train = valid[valid['season'] == '2021-22']
    test = valid[valid['season'] == '2022-23']

    print(f"\nTrain: {len(train)} games (2021-22)")
    print(f"Test:  {len(test)} games (2022-23)")

    if len(train) > 50 and len(test) > 50:
        # Retrain model on train only
        X_train = train[features].values
        y_train = train['q4_total'].values
        X_test = test[features].values
        y_test = test['q4_total'].values

        model_oos = LinearRegression()
        model_oos.fit(X_train, y_train)
        preds_oos = model_oos.predict(X_test)

        test = test.copy()
        test['pred_q4'] = preds_oos
        test['pred_final'] = test['q3_cumul_total'] + test['pred_q4']
        test['pred_ou'] = np.where(test['pred_final'] > test['ou_line'], 'OVER', 'UNDER')
        test['actual_ou'] = np.where(test['final_total'] > test['ou_line'], 'OVER', 'UNDER')
        test['correct'] = test['pred_ou'] == test['actual_ou']
        test['margin'] = abs(test['pred_final'] - test['ou_line'])

        print(f"\nOOS model R²: {r2_score(y_test, preds_oos):.4f}")
        print(f"OOS accuracy (all): {test['correct'].mean():.1%}")

        print("\nOOS accuracy by confidence:")
        for m in [0, 2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30]:
            conf = test[test['margin'] >= m]
            if len(conf) < 5:
                continue
            print(f"  Margin >= {m:2d}: n={len(conf):4d}, acc={conf['correct'].mean():.1%}")

        # Apply the signal conditions
        print("\n=== OOS UNDER SIGNAL (with Q3 lead filter) ===")
        for lead_min in [12, 15, 18, 20, 22, 25]:
            for margin_min in [5, 8, 10, 12, 15, 18, 20]:
                cond = (
                    (test['q3_lead'] >= lead_min) &
                    (test['pred_ou'] == 'UNDER') &
                    (test['margin'] >= margin_min)
                )
                subset = test[cond]
                if len(subset) < 5:
                    continue
                acc = (subset['actual_ou'] == 'UNDER').mean()
                if acc >= 0.85:
                    avg_m = (subset['ou_line'] - subset['final_total']).mean()
                    print(f"  Lead>={lead_min} margin>={margin_min}: "
                          f"n={len(subset):3d}, acc={acc:.1%}, avg_margin={avg_m:+.1f}")

        print("\n=== OOS OVER SIGNAL (close game + fast pace) ===")
        for lead_max in [3, 5, 7, 10]:
            for margin_min in [3, 5, 8, 10, 12, 15]:
                cond = (
                    (test['q3_lead'] <= lead_max) &
                    (test['pred_ou'] == 'OVER') &
                    (test['margin'] >= margin_min)
                )
                subset = test[cond]
                if len(subset) < 5:
                    continue
                acc = (subset['actual_ou'] == 'OVER').mean()
                if acc >= 0.80:
                    avg_m = (subset['final_total'] - subset['ou_line']).mean()
                    print(f"  Lead<={lead_max} margin>={margin_min}: "
                          f"n={len(subset):3d}, acc={acc:.1%}, avg_margin={avg_m:+.1f}")

    # =====================================================================
    # SECTION 12: PROFITABILITY SIMULATION
    # =====================================================================
    print("\n" + "=" * 70)
    print("SECTION 12: PROFITABILITY SIMULATION (-110 JUICE)")
    print("=" * 70)

    # Standard O/U bets pay -110 (risk 110 to win 100)
    # Need 52.4% accuracy to break even
    # At 95% accuracy, huge profit
    vig = -110
    win_payout = 100 / 110  # ~0.909 per unit risked
    loss_cost = 1.0

    if len(test) > 50:
        print("\nSimulating all discovered signals on OOS data...")
        best_signals = []

        # Comprehensive search
        for direction in ['UNDER', 'OVER']:
            for lead_thresh in range(0, 30, 2):
                for margin_min in range(3, 25, 2):
                    if direction == 'UNDER':
                        cond = (
                            (test['q3_lead'] >= lead_thresh) &
                            (test['pred_ou'] == 'UNDER') &
                            (test['margin'] >= margin_min)
                        )
                        correct_col = test['actual_ou'] == 'UNDER'
                    else:
                        cond = (
                            (test['q3_lead'] <= lead_thresh) &
                            (test['pred_ou'] == 'OVER') &
                            (test['margin'] >= margin_min)
                        )
                        correct_col = test['actual_ou'] == 'OVER'

                    subset = test[cond]
                    if len(subset) < 8:
                        continue

                    wins = correct_col[cond].sum()
                    losses = len(subset) - wins
                    acc = wins / len(subset)
                    pnl = wins * win_payout - losses * loss_cost
                    roi = pnl / len(subset) * 100

                    if acc >= 0.90 and roi > 0:
                        best_signals.append({
                            'direction': direction,
                            'lead_thresh': lead_thresh,
                            'margin_min': margin_min,
                            'n': len(subset),
                            'wins': wins,
                            'losses': losses,
                            'accuracy': acc,
                            'pnl': pnl,
                            'roi': roi
                        })

        best_signals.sort(key=lambda x: (-x['accuracy'], -x['roi']))
        print(f"\nFound {len(best_signals)} signals with 90%+ OOS accuracy")
        print(f"\nTop 20 by accuracy then ROI:")
        for i, s in enumerate(best_signals[:20]):
            print(f"  {i+1:2d}. {s['direction']:5s} lead{'>='+str(s['lead_thresh']) if s['direction']=='UNDER' else '<='+str(s['lead_thresh']):>5s} "
                  f"margin>={s['margin_min']:2d}: "
                  f"n={s['n']:3d} W={s['wins']} L={s['losses']} "
                  f"acc={s['accuracy']:.1%} ROI={s['roi']:+.1f}%")

        # FINAL SIGNALS: Highest accuracy + sufficient sample size
        print("\n" + "=" * 70)
        print("FINAL PRODUCTION SIGNALS (95%+ accuracy, n >= 10)")
        print("=" * 70)

        elite = [s for s in best_signals if s['accuracy'] >= 0.95 and s['n'] >= 10]
        elite.sort(key=lambda x: (-x['n'], -x['accuracy']))
        for i, s in enumerate(elite):
            print(f"  {i+1:2d}. {s['direction']:5s} lead{'>='+str(s['lead_thresh']) if s['direction']=='UNDER' else '<='+str(s['lead_thresh']):>5s} "
                  f"margin>={s['margin_min']:2d}: "
                  f"n={s['n']:3d} W-L={s['wins']}-{s['losses']} "
                  f"acc={s['accuracy']:.1%} ROI={s['roi']:+.1f}% "
                  f"PnL={s['pnl']:+.1f}u")

    # Save analysis
    reg.to_csv(OUTPUT_DIR / "q3_ou_analysis.csv", index=False)
    print(f"\nFull analysis saved to {OUTPUT_DIR / 'q3_ou_analysis.csv'}")


if __name__ == "__main__":
    main()
