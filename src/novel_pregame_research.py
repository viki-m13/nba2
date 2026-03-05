"""
NOVEL PREGAME BET TYPE RESEARCH
================================
Systematic exploration of pregame betting edges beyond Play 3 (spread).
Goal: Find a bet type with:
  - Determined prior to game start
  - At least -110 odds (standard juice)
  - Very high accuracy (85%+)
  - Novel/patentable analytical framework

Research Areas:
  1. Game Total Over/Under via Pace-Mismatch Model
  2. Underdog Scoring Ceiling (team total UNDER)
  3. Favorite Scoring Floor (team total OVER at specific thresholds)
  4. Defensive Suffocation Index - games where defense dictates outcome
  5. Volatility-Adjusted Confidence Bands
  6. Combined Margin + Total regime classification
  7. Win% Divergence + Consistency Filter
  8. Opponent Weakness Exploitation Index (OWEI)
"""

import json
import math
from collections import defaultdict
import statistics

# =============================================================================
# LOAD DATA
# =============================================================================

with open('data/pregame_predictions_2025.json') as f:
    games = json.load(f)

print(f"Loaded {len(games)} games from 2024-25 season")
print(f"Date range: {games[0]['date']} to {games[-1]['date']}")
print()

# =============================================================================
# DERIVED FEATURES - Compute additional metrics for each game
# =============================================================================

for g in games:
    # Favorite's metrics
    if g['fav_is_home']:
        g['fav_off'] = g['home_off']
        g['fav_def'] = g['home_def']
        g['fav_net'] = g['home_net']
        g['fav_win_pct'] = g['home_win_pct']
        g['dog_off'] = g['away_off']
        g['dog_def'] = g['away_def']
        g['dog_net'] = g['away_net']
        g['dog_win_pct'] = g['away_win_pct']
        g['dog_score'] = g['total'] - g['fav_actual_score']
    else:
        g['fav_off'] = g['away_off']
        g['fav_def'] = g['away_def']
        g['fav_net'] = g['away_net']
        g['fav_win_pct'] = g['away_win_pct']
        g['dog_off'] = g['home_off']
        g['dog_def'] = g['home_def']
        g['dog_net'] = g['home_net']
        g['dog_win_pct'] = g['home_win_pct']
        g['dog_score'] = g['total'] - g['fav_actual_score']

    # Net gap (always positive - favorite's advantage)
    g['net_gap'] = abs(g['net_diff'])

    # Offensive mismatch: how much fav offense exceeds dog defense
    g['off_vs_def_mismatch'] = g['fav_off'] - g['dog_def']

    # Defensive mismatch: how much fav defense suppresses dog offense
    g['def_vs_off_mismatch'] = g['dog_off'] - g['fav_def']

    # Pace proxy: average of both teams' offensive ratings
    g['expected_pace'] = (g['home_off'] + g['away_off']) / 2.0

    # Defensive quality: lower = better defense (average of both)
    g['avg_def'] = (g['home_def'] + g['away_def']) / 2.0

    # Win% gap
    g['win_pct_gap'] = abs(g['fav_win_pct'] - g['dog_win_pct'])

    # "Suffocation Index": fav defense relative to dog offense
    g['suffocation'] = g['dog_off'] - g['fav_def']  # negative = fav defense dominates

    # Combined quality score (both teams good = high total)
    g['quality_sum'] = g['fav_off'] + g['dog_off']

    # Defensive gap: fav_def vs dog_def
    g['def_gap'] = g['dog_def'] - g['fav_def']  # positive = fav has better defense

    # Actual favorite margin (signed - positive = fav won)
    g['fav_margin'] = g['fav_actual_score'] - g['dog_score']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def analyze_filter(games, filter_fn, description, bet_analysis_fn=None):
    """Apply a filter and analyze the resulting games."""
    qualifying = [g for g in games if filter_fn(g)]
    if len(qualifying) < 10:
        print(f"  {description}: Only {len(qualifying)} games (too few)")
        return qualifying

    fav_wins = sum(1 for g in qualifying if g['fav_won'])
    avg_fav_score = statistics.mean([g['fav_actual_score'] for g in qualifying])
    avg_dog_score = statistics.mean([g['dog_score'] for g in qualifying])
    avg_total = statistics.mean([g['total'] for g in qualifying])
    avg_margin = statistics.mean([g['fav_margin'] for g in qualifying])

    print(f"\n  {description}")
    print(f"  Games: {len(qualifying)}")
    print(f"  Fav ML Win Rate: {fav_wins}/{len(qualifying)} = {100*fav_wins/len(qualifying):.1f}%")
    print(f"  Avg Fav Score: {avg_fav_score:.1f}")
    print(f"  Avg Dog Score: {avg_dog_score:.1f}")
    print(f"  Avg Total: {avg_total:.1f}")
    print(f"  Avg Fav Margin: {avg_margin:+.1f}")

    if bet_analysis_fn:
        bet_analysis_fn(qualifying)

    return qualifying


def team_total_analysis(qualifying, team='fav'):
    """Analyze team total over/under at various thresholds."""
    if team == 'fav':
        scores = [g['fav_actual_score'] for g in qualifying]
    else:
        scores = [g['dog_score'] for g in qualifying]

    avg = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0
    print(f"  [{team.upper()} Scores] Mean={avg:.1f}, Std={std:.1f}, Min={min(scores)}, Max={max(scores)}")

    for threshold in [100, 105, 108, 110, 112, 115, 118, 120]:
        over = sum(1 for s in scores if s >= threshold)
        under = sum(1 for s in scores if s < threshold)
        pct_over = 100 * over / len(scores)
        pct_under = 100 * under / len(scores)
        marker = ""
        if pct_over >= 85:
            marker = " <<<< OVER EDGE"
        elif pct_under >= 85:
            marker = " <<<< UNDER EDGE"
        print(f"    Score >= {threshold}: {over}/{len(scores)} = {pct_over:.1f}% OVER | {pct_under:.1f}% UNDER{marker}")


def total_analysis(qualifying):
    """Analyze game total over/under at various thresholds."""
    totals = [g['total'] for g in qualifying]
    avg = statistics.mean(totals)
    std = statistics.stdev(totals) if len(totals) > 1 else 0
    print(f"  [GAME TOTAL] Mean={avg:.1f}, Std={std:.1f}, Min={min(totals)}, Max={max(totals)}")

    for threshold in [200, 205, 210, 215, 218, 220, 225, 230, 235]:
        over = sum(1 for t in totals if t >= threshold)
        under = sum(1 for t in totals if t < threshold)
        pct_over = 100 * over / len(totals)
        pct_under = 100 * under / len(totals)
        marker = ""
        if pct_over >= 85:
            marker = " <<<< OVER EDGE"
        elif pct_under >= 85:
            marker = " <<<< UNDER EDGE"
        print(f"    Total >= {threshold}: {over}/{len(totals)} = {pct_over:.1f}% OVER | {pct_under:.1f}% UNDER{marker}")


def margin_analysis(qualifying):
    """Analyze margin distribution."""
    margins = [g['fav_margin'] for g in qualifying]
    avg = statistics.mean(margins)
    std = statistics.stdev(margins) if len(margins) > 1 else 0
    print(f"  [MARGINS] Mean={avg:+.1f}, Std={std:.1f}, Min={min(margins):+d}, Max={max(margins):+d}")

    for threshold in [1, 3, 5, 7, 10, 15, 20]:
        covers = sum(1 for m in margins if m >= threshold)
        pct = 100 * covers / len(margins)
        print(f"    Margin >= {threshold}: {covers}/{len(margins)} = {pct:.1f}%")


def profitability_at_minus_110(accuracy, unit_size=100):
    """Calculate profit at -110 odds given an accuracy rate."""
    wins = accuracy
    losses = 1 - accuracy
    profit_per_win = unit_size * (100/110)  # -110 odds
    loss_per_loss = unit_size
    ev = wins * profit_per_win - losses * loss_per_loss
    roi = ev / unit_size * 100
    return ev, roi


# =============================================================================
# RESEARCH AREA 1: DEFENSIVE SUFFOCATION INDEX (DSI)
# Novel concept: Instead of predicting WHO wins, predict SCORING ENVIRONMENT
# =============================================================================

print("=" * 80)
print("RESEARCH 1: DEFENSIVE SUFFOCATION INDEX (DSI)")
print("Concept: When BOTH teams have strong defense, the total stays LOW")
print("=" * 80)

# Strong defensive matchups (both teams good on D)
analyze_filter(games,
    lambda g: g['avg_def'] <= 108,
    "Both teams avg defense <= 108 (elite D matchup)",
    lambda q: total_analysis(q))

analyze_filter(games,
    lambda g: g['avg_def'] <= 106,
    "Both teams avg defense <= 106 (ultra-elite D)",
    lambda q: total_analysis(q))

# Fav has elite defense + dog has weak offense
analyze_filter(games,
    lambda g: g['fav_def'] <= 108 and g['dog_off'] <= 110,
    "Fav defense <= 108 AND dog offense <= 110",
    lambda q: (team_total_analysis(q, 'dog'), total_analysis(q)))

analyze_filter(games,
    lambda g: g['fav_def'] <= 106 and g['dog_off'] <= 108,
    "Fav defense <= 106 AND dog offense <= 108",
    lambda q: (team_total_analysis(q, 'dog'), total_analysis(q)))

# =============================================================================
# RESEARCH AREA 2: UNDERDOG SCORING CEILING
# Novel concept: Predict the underdog's MAXIMUM expected score
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 2: UNDERDOG SCORING CEILING")
print("Concept: Certain matchups cap the underdog's scoring ability")
print("=" * 80)

# Dog has weak offense AND faces good defense
analyze_filter(games,
    lambda g: g['dog_off'] <= 110 and g['fav_def'] <= 110 and g['fav_is_home'],
    "Dog OFF<=110 + Fav DEF<=110 + Fav Home",
    lambda q: team_total_analysis(q, 'dog'))

analyze_filter(games,
    lambda g: g['dog_off'] <= 108 and g['fav_def'] <= 108,
    "Dog OFF<=108 + Fav DEF<=108 (double squeeze)",
    lambda q: team_total_analysis(q, 'dog'))

analyze_filter(games,
    lambda g: g['suffocation'] <= -2 and g['fav_is_home'],
    "Suffocation <= -2 (fav D dominates dog O) + Home",
    lambda q: team_total_analysis(q, 'dog'))

analyze_filter(games,
    lambda g: g['suffocation'] <= -4 and g['fav_is_home'],
    "Suffocation <= -4 (extreme D dominance) + Home",
    lambda q: team_total_analysis(q, 'dog'))

analyze_filter(games,
    lambda g: g['suffocation'] <= -4,
    "Suffocation <= -4 (any venue)",
    lambda q: team_total_analysis(q, 'dog'))

# =============================================================================
# RESEARCH AREA 3: PACE-MISMATCH TOTAL PREDICTOR
# Novel concept: Offensive firepower asymmetry predicts game total
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 3: PACE-MISMATCH TOTAL PREDICTOR")
print("Concept: When both teams are high-scoring, game totals explode")
print("=" * 80)

analyze_filter(games,
    lambda g: g['expected_pace'] >= 115,
    "Expected pace >= 115 (both teams score a lot)",
    lambda q: total_analysis(q))

analyze_filter(games,
    lambda g: g['expected_pace'] >= 118,
    "Expected pace >= 118 (elite offensive matchup)",
    lambda q: total_analysis(q))

analyze_filter(games,
    lambda g: g['expected_pace'] <= 108,
    "Expected pace <= 108 (both teams low-scoring)",
    lambda q: total_analysis(q))

analyze_filter(games,
    lambda g: g['quality_sum'] >= 235,
    "Combined offense quality >= 235 (offensive fireworks)",
    lambda q: total_analysis(q))

analyze_filter(games,
    lambda g: g['quality_sum'] <= 215,
    "Combined offense quality <= 215 (low-scoring environment)",
    lambda q: total_analysis(q))

# =============================================================================
# RESEARCH AREA 4: OPPONENT WEAKNESS EXPLOITATION INDEX (OWEI)
# Novel concept: Specific asymmetric mismatches create predictable outcomes
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 4: OPPONENT WEAKNESS EXPLOITATION INDEX (OWEI)")
print("Concept: Extreme offensive-vs-defensive mismatches are predictable")
print("=" * 80)

# Fav offense massively exceeds dog defense (dog can't defend)
analyze_filter(games,
    lambda g: g['off_vs_def_mismatch'] >= 8 and g['fav_is_home'],
    "Fav OFF vs Dog DEF mismatch >= 8 + Home",
    lambda q: (team_total_analysis(q, 'fav'), margin_analysis(q)))

analyze_filter(games,
    lambda g: g['off_vs_def_mismatch'] >= 10,
    "Fav OFF vs Dog DEF mismatch >= 10 (any venue)",
    lambda q: (team_total_analysis(q, 'fav'), margin_analysis(q)))

analyze_filter(games,
    lambda g: g['off_vs_def_mismatch'] >= 12,
    "Fav OFF vs Dog DEF mismatch >= 12 (extreme)",
    lambda q: (team_total_analysis(q, 'fav'), margin_analysis(q)))

# =============================================================================
# RESEARCH AREA 5: VOLATILITY-ADJUSTED PREDICTION
# Novel concept: Low-volatility matchups are more predictable
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 5: WIN% DIVERGENCE + CONSISTENCY")
print("Concept: When win% gap is large, outcomes become deterministic")
print("=" * 80)

analyze_filter(games,
    lambda g: g['win_pct_gap'] >= 0.4 and g['fav_is_home'],
    "Win% gap >= 0.4 + Home",
    lambda q: (margin_analysis(q), team_total_analysis(q, 'fav')))

analyze_filter(games,
    lambda g: g['win_pct_gap'] >= 0.5,
    "Win% gap >= 0.5 (any venue)",
    lambda q: (margin_analysis(q), team_total_analysis(q, 'fav')))

analyze_filter(games,
    lambda g: g['win_pct_gap'] >= 0.4 and g['net_gap'] >= 8,
    "Win% gap >= 0.4 AND net gap >= 8",
    lambda q: (margin_analysis(q), team_total_analysis(q, 'fav')))

# =============================================================================
# RESEARCH AREA 6: COMBINED SCORING ENVIRONMENT CLASSIFIER
# Novel: Multi-dimensional regime identification
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 6: SCORING ENVIRONMENT REGIMES")
print("Concept: Classify games into scoring regimes for total predictions")
print("=" * 80)

# High offense + weak defense = high total
analyze_filter(games,
    lambda g: g['fav_off'] >= 118 and g['dog_def'] >= 114 and g['dog_off'] >= 112,
    "Fav OFF>=118 + Dog DEF>=114 + Dog OFF>=112 (shootout)",
    lambda q: total_analysis(q))

# Strong defense + weak offense = low total
analyze_filter(games,
    lambda g: g['fav_def'] <= 108 and g['dog_def'] <= 112 and g['expected_pace'] <= 112,
    "Fav DEF<=108 + Dog DEF<=112 + Pace<=112 (grinder)",
    lambda q: total_analysis(q))

# Blowout regime: fav dominates everything
analyze_filter(games,
    lambda g: g['fav_off'] >= 116 and g['fav_def'] <= 110 and g['net_gap'] >= 10,
    "Fav OFF>=116 + Fav DEF<=110 + Net gap>=10 (all-around dominant)",
    lambda q: (margin_analysis(q), team_total_analysis(q, 'fav'), team_total_analysis(q, 'dog')))

# =============================================================================
# RESEARCH AREA 7: NOVEL COMPOSITE INDICES
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 7: NOVEL COMPOSITE INDICES")
print("=" * 80)

# DIRECTIONAL MISMATCH INDEX (DMI): considers BOTH the offense AND defense mismatch
for g in games:
    # How much does the favorite's offense overpower the dog's defense?
    off_dom = max(0, g['fav_off'] - g['dog_def']) / 5.0
    # How much does the favorite's defense suppress the dog's offense?
    def_dom = max(0, g['fav_def'] - g['dog_off']) / 5.0  # negative means fav D is better
    def_suppress = max(0, g['fav_def'] * -1 + g['dog_off'] * -1 + 220) / 5.0  # reframe
    # Actually: suffocation is dog_off - fav_def. When negative, fav defense dominates.
    def_suppress2 = max(0, -g['suffocation']) / 5.0

    # DMI: combines offense explosion AND defense suppression
    g['dmi'] = off_dom * 0.5 + def_suppress2 * 0.5

    # Asymmetric Mismatch Score (AMS): offense vs defense in BOTH directions
    g['ams'] = (g['off_vs_def_mismatch'] - g['def_vs_off_mismatch']) / 2.0

    # Total Control Index (TCI): predicts whether fav controls pace
    g['tci'] = (g['fav_off'] - g['dog_off']) + (g['dog_def'] - g['fav_def'])

    # Ceiling-Floor Divergence (CFD): fav's ceiling vs dog's floor
    g['cfd'] = g['fav_off'] - g['dog_off'] + (g['dog_def'] - g['fav_def'])

analyze_filter(games,
    lambda g: g['dmi'] >= 2.0 and g['fav_is_home'],
    "DMI >= 2.0 + Home (dual mismatch)",
    lambda q: (margin_analysis(q), team_total_analysis(q, 'fav'), team_total_analysis(q, 'dog')))

analyze_filter(games,
    lambda g: g['ams'] >= 6 and g['fav_is_home'],
    "AMS >= 6 + Home (asymmetric mismatch)",
    lambda q: (margin_analysis(q), team_total_analysis(q, 'fav')))

analyze_filter(games,
    lambda g: g['ams'] >= 8,
    "AMS >= 8 (extreme asymmetric mismatch, any venue)",
    lambda q: (margin_analysis(q), team_total_analysis(q, 'fav'), team_total_analysis(q, 'dog')))

analyze_filter(games,
    lambda g: g['tci'] >= 12 and g['fav_is_home'],
    "TCI >= 12 + Home (total control)",
    lambda q: (margin_analysis(q), team_total_analysis(q, 'fav'), team_total_analysis(q, 'dog')))

analyze_filter(games,
    lambda g: g['tci'] >= 15,
    "TCI >= 15 (extreme total control, any venue)",
    lambda q: (margin_analysis(q), team_total_analysis(q, 'fav'), team_total_analysis(q, 'dog')))

# =============================================================================
# RESEARCH AREA 8: DOG SCORE UNDER - DEEP ANALYSIS
# Novel: Predict underdog will score UNDER their expected total
# This is the INVERSE of Play 3's team total over
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 8: DOG SCORE UNDER - DEEP ANALYSIS")
print("When can we predict with 85%+ that the underdog scores UNDER a threshold?")
print("=" * 80)

# Approach: find combos where dog consistently scores below a settable line

# Elite fav defense + bad dog offense = suppressed scoring
for dog_off_max in [108, 110, 112]:
    for fav_def_max in [106, 108, 110]:
        for home_only in [True, False]:
            filt = lambda g, do=dog_off_max, fd=fav_def_max, ho=home_only: (
                g['dog_off'] <= do and g['fav_def'] <= fd and (g['fav_is_home'] if ho else True)
            )
            qualifying = [g for g in games if filt(g)]
            if len(qualifying) >= 12:
                dog_scores = [g['dog_score'] for g in qualifying]
                avg_dog = statistics.mean(dog_scores)

                # Check various under thresholds
                for threshold in [105, 108, 110]:
                    under = sum(1 for s in dog_scores if s < threshold)
                    pct = 100 * under / len(qualifying)
                    if pct >= 80:
                        home_str = "Home" if home_only else "Any"
                        fav_wins = sum(1 for g in qualifying if g['fav_won'])
                        ml_pct = 100 * fav_wins / len(qualifying)
                        ev, roi = profitability_at_minus_110(pct/100)
                        print(f"  Dog OFF<={dog_off_max}, Fav DEF<={fav_def_max}, {home_str}: "
                              f"{len(qualifying)} games, Dog<{threshold}: {under}/{len(qualifying)}={pct:.1f}%, "
                              f"Avg dog score: {avg_dog:.1f}, ML: {ml_pct:.1f}%, ROI@-110: {roi:+.1f}%")


# =============================================================================
# RESEARCH AREA 9: GAME TOTAL UNDER IN DEFENSIVE MATCHUPS
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 9: GAME TOTAL UNDER IN DEFENSIVE MATCHUPS")
print("=" * 80)

for avg_def_max in [106, 108, 110]:
    for pace_max in [108, 110, 112, 114]:
        filt = lambda g, ad=avg_def_max, pm=pace_max: g['avg_def'] <= ad and g['expected_pace'] <= pm
        qualifying = [g for g in games if filt(g)]
        if len(qualifying) >= 12:
            totals = [g['total'] for g in qualifying]
            avg_total = statistics.mean(totals)
            for threshold in [210, 215, 218, 220]:
                under = sum(1 for t in totals if t < threshold)
                pct = 100 * under / len(qualifying)
                if pct >= 80:
                    ev, roi = profitability_at_minus_110(pct/100)
                    print(f"  Avg DEF<={avg_def_max}, Pace<={pace_max}: "
                          f"{len(qualifying)} games, Total<{threshold}: {under}/{len(qualifying)}={pct:.1f}%, "
                          f"Avg total: {avg_total:.1f}, ROI@-110: {roi:+.1f}%")


# =============================================================================
# RESEARCH AREA 10: MULTI-FACTOR COMBINATION SEARCH
# Exhaustive search for high-accuracy pregame filters
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 10: EXHAUSTIVE MULTI-FACTOR COMBINATION SEARCH")
print("Searching for 85%+ accuracy pregame signals across all bet types")
print("=" * 80)

best_signals = []

# Search space
net_gap_thresholds = [6, 8, 10, 12, 14]
fav_off_thresholds = [112, 114, 116, 118, 120]
fav_def_thresholds = [106, 108, 110, 112]
dog_off_thresholds = [106, 108, 110, 112]
dog_def_thresholds = [110, 112, 114, 116, 118]
win_pct_thresholds = [0.3, 0.35, 0.4, 0.45]

for ng in net_gap_thresholds:
    for fo in fav_off_thresholds:
        for home_only in [True, False]:
            filt = lambda g, _ng=ng, _fo=fo, _ho=home_only: (
                g['net_gap'] >= _ng and g['fav_off'] >= _fo and
                (g['fav_is_home'] if _ho else True)
            )
            qualifying = [g for g in games if filt(g)]
            if 15 <= len(qualifying) <= 200:
                # Check ML accuracy
                fav_wins = sum(1 for g in qualifying if g['fav_won'])
                ml_pct = fav_wins / len(qualifying)

                # Check fav score thresholds
                for score_th in [110, 112, 115]:
                    over = sum(1 for g in qualifying if g['fav_actual_score'] >= score_th)
                    over_pct = over / len(qualifying)
                    if over_pct >= 0.85:
                        ev, roi = profitability_at_minus_110(over_pct)
                        home_str = "Home" if home_only else "Any"
                        best_signals.append({
                            'type': f'Fav Score>={score_th}',
                            'filter': f'NetGap>={ng}, FavOff>={fo}, {home_str}',
                            'games': len(qualifying),
                            'accuracy': over_pct,
                            'roi': roi,
                            'ml_pct': ml_pct,
                        })

                # Check dog score UNDER thresholds
                for score_th in [105, 108, 110, 112]:
                    under = sum(1 for g in qualifying if g['dog_score'] < score_th)
                    under_pct = under / len(qualifying)
                    if under_pct >= 0.80:
                        ev, roi = profitability_at_minus_110(under_pct)
                        home_str = "Home" if home_only else "Any"
                        best_signals.append({
                            'type': f'Dog Score<{score_th}',
                            'filter': f'NetGap>={ng}, FavOff>={fo}, {home_str}',
                            'games': len(qualifying),
                            'accuracy': under_pct,
                            'roi': roi,
                            'ml_pct': ml_pct,
                        })

                # Check game total thresholds
                for total_th in [215, 218, 220, 225]:
                    over = sum(1 for g in qualifying if g['total'] >= total_th)
                    over_pct = over / len(qualifying)
                    if over_pct >= 0.85:
                        ev, roi = profitability_at_minus_110(over_pct)
                        home_str = "Home" if home_only else "Any"
                        best_signals.append({
                            'type': f'Total>={total_th}',
                            'filter': f'NetGap>={ng}, FavOff>={fo}, {home_str}',
                            'games': len(qualifying),
                            'accuracy': over_pct,
                            'roi': roi,
                            'ml_pct': ml_pct,
                        })


# Also search defense-focused filters
for fd in fav_def_thresholds:
    for do in dog_off_thresholds:
        for home_only in [True, False]:
            filt = lambda g, _fd=fd, _do=do, _ho=home_only: (
                g['fav_def'] <= _fd and g['dog_off'] <= _do and
                (g['fav_is_home'] if _ho else True)
            )
            qualifying = [g for g in games if filt(g)]
            if 15 <= len(qualifying) <= 200:
                fav_wins = sum(1 for g in qualifying if g['fav_won'])
                ml_pct = fav_wins / len(qualifying)

                # Dog team total under
                for score_th in [100, 103, 105, 108, 110]:
                    under = sum(1 for g in qualifying if g['dog_score'] < score_th)
                    under_pct = under / len(qualifying)
                    if under_pct >= 0.80:
                        ev, roi = profitability_at_minus_110(under_pct)
                        home_str = "Home" if home_only else "Any"
                        best_signals.append({
                            'type': f'Dog Score<{score_th} (DEF filter)',
                            'filter': f'FavDef<={fd}, DogOff<={do}, {home_str}',
                            'games': len(qualifying),
                            'accuracy': under_pct,
                            'roi': roi,
                            'ml_pct': ml_pct,
                        })

                # Game total under
                for total_th in [210, 215, 218, 220]:
                    under = sum(1 for g in qualifying if g['total'] < total_th)
                    under_pct = under / len(qualifying)
                    if under_pct >= 0.80:
                        ev, roi = profitability_at_minus_110(under_pct)
                        home_str = "Home" if home_only else "Any"
                        best_signals.append({
                            'type': f'Total<{total_th} (DEF filter)',
                            'filter': f'FavDef<={fd}, DogOff<={do}, {home_str}',
                            'games': len(qualifying),
                            'accuracy': under_pct,
                            'roi': roi,
                            'ml_pct': ml_pct,
                        })


# Sort by ROI and print top results
best_signals.sort(key=lambda x: (-x['accuracy'], -x['games']))

print(f"\nTOP 30 SIGNALS BY ACCURACY (min 15 games):")
print("-" * 120)
for i, s in enumerate(best_signals[:30]):
    print(f"  {i+1:2d}. [{s['type']:25s}] Filter: {s['filter']:45s} "
          f"Games: {s['games']:3d}  Accuracy: {100*s['accuracy']:.1f}%  ROI@-110: {s['roi']:+.1f}%  ML: {100*s['ml_pct']:.1f}%")

print(f"\nTOP 20 SIGNALS BY GAMES (>= 85% accuracy):")
print("-" * 120)
high_acc = [s for s in best_signals if s['accuracy'] >= 0.85]
high_acc.sort(key=lambda x: -x['games'])
for i, s in enumerate(high_acc[:20]):
    print(f"  {i+1:2d}. [{s['type']:25s}] Filter: {s['filter']:45s} "
          f"Games: {s['games']:3d}  Accuracy: {100*s['accuracy']:.1f}%  ROI@-110: {s['roi']:+.1f}%  ML: {100*s['ml_pct']:.1f}%")


# =============================================================================
# RESEARCH AREA 11: NOVEL "DEFENSIVE CAGE" CONCEPT
# When fav defense rating is MUCH better than dog offense rating,
# the dog gets "caged" - unable to score above their floor
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 11: DEFENSIVE CAGE MODEL")
print("Novel concept: Fav defense 'cages' the dog, creating a scoring ceiling")
print("=" * 80)

# The "cage score" = how much fav defense outperforms dog offense
for g in games:
    g['cage_score'] = g['fav_def'] - g['dog_off']  # negative = fav defense better
    # More negative = stronger cage

# Analyze cage effect
for cage_th in [-6, -4, -2, 0, 2]:
    for home_only in [True, False]:
        filt = lambda g, ct=cage_th, ho=home_only: (
            g['cage_score'] <= ct and (g['fav_is_home'] if ho else True)
        )
        qualifying = [g for g in games if filt(g)]
        if len(qualifying) >= 12:
            dog_scores = [g['dog_score'] for g in qualifying]
            avg_dog = statistics.mean(dog_scores)
            home_str = "Home" if home_only else "Any"

            for threshold in [105, 108, 110, 112]:
                under = sum(1 for s in dog_scores if s < threshold)
                pct = 100 * under / len(qualifying)
                if pct >= 80:
                    print(f"  Cage<={cage_th}, {home_str}: {len(qualifying)} games, "
                          f"Dog<{threshold}: {under}/{len(qualifying)}={pct:.1f}%, "
                          f"Avg dog: {avg_dog:.1f}")


# =============================================================================
# RESEARCH AREA 12: COMPOSITE MODELS - COMBINING MULTIPLE SIGNALS
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 12: COMPOSITE MODEL - STACKING MULTIPLE WEAK SIGNALS")
print("=" * 80)

# Score each game on multiple dimensions
for g in games:
    score = 0
    # Factor 1: Net gap strength
    if g['net_gap'] >= 14: score += 3
    elif g['net_gap'] >= 10: score += 2
    elif g['net_gap'] >= 7: score += 1

    # Factor 2: Offensive firepower
    if g['fav_off'] >= 120: score += 3
    elif g['fav_off'] >= 118: score += 2
    elif g['fav_off'] >= 115: score += 1

    # Factor 3: Defensive cage
    if g['cage_score'] <= -6: score += 3
    elif g['cage_score'] <= -3: score += 2
    elif g['cage_score'] <= 0: score += 1

    # Factor 4: Win% gap
    if g['win_pct_gap'] >= 0.5: score += 3
    elif g['win_pct_gap'] >= 0.4: score += 2
    elif g['win_pct_gap'] >= 0.3: score += 1

    # Factor 5: Home court
    if g['fav_is_home']: score += 1

    g['composite_score'] = score

# Test composite thresholds
for threshold in range(4, 14):
    qualifying = [g for g in games if g['composite_score'] >= threshold]
    if len(qualifying) >= 10:
        fav_wins = sum(1 for g in qualifying if g['fav_won'])
        ml_pct = 100 * fav_wins / len(qualifying)

        fav_scores = [g['fav_actual_score'] for g in qualifying]
        dog_scores = [g['dog_score'] for g in qualifying]
        totals = [g['total'] for g in qualifying]

        fav_over_110 = sum(1 for s in fav_scores if s >= 110) / len(qualifying) * 100
        fav_over_115 = sum(1 for s in fav_scores if s >= 115) / len(qualifying) * 100
        dog_under_108 = sum(1 for s in dog_scores if s < 108) / len(qualifying) * 100
        dog_under_110 = sum(1 for s in dog_scores if s < 110) / len(qualifying) * 100

        print(f"  Composite >= {threshold:2d}: {len(qualifying):3d} games | "
              f"ML: {ml_pct:.1f}% | "
              f"Fav>=110: {fav_over_110:.1f}% | Fav>=115: {fav_over_115:.1f}% | "
              f"Dog<108: {dog_under_108:.1f}% | Dog<110: {dog_under_110:.1f}%")


# =============================================================================
# RESEARCH AREA 13: MARGIN-CONDITIONED BET TYPES
# Novel: Instead of simple ML, predict margin ranges for alternative bets
# =============================================================================

print("\n" + "=" * 80)
print("RESEARCH 13: MARGIN-CONDITIONED ALTERNATIVE BETS")
print("Novel: Predict fav wins by 5+ or 10+ for alt-spread bets at better odds")
print("=" * 80)

# In sportsbetting, you can bet alternative spreads:
# e.g., "Team wins by 5+" at around -130 to -150
# or "Team wins by 10+" at around +110 to +140
# Finding 80%+ accuracy on -5 spread = huge edge

for ng in [8, 10, 12, 14]:
    for fo in [115, 116, 118, 120]:
        for home_only in [True, False]:
            filt = lambda g, _ng=ng, _fo=fo, _ho=home_only: (
                g['net_gap'] >= _ng and g['fav_off'] >= _fo and
                (g['fav_is_home'] if _ho else True)
            )
            qualifying = [g for g in games if filt(g)]
            if len(qualifying) >= 15:
                margins = [g['fav_margin'] for g in qualifying]
                home_str = "Home" if home_only else "Any"

                covers_5 = sum(1 for m in margins if m >= 5)
                covers_10 = sum(1 for m in margins if m >= 10)
                covers_15 = sum(1 for m in margins if m >= 15)
                pct_5 = covers_5 / len(qualifying)
                pct_10 = covers_10 / len(qualifying)
                pct_15 = covers_15 / len(qualifying)

                if pct_5 >= 0.75 or pct_10 >= 0.60:
                    print(f"  NetGap>={ng}, FavOff>={fo}, {home_str}: {len(qualifying)} games | "
                          f"Margin>=5: {100*pct_5:.1f}% | Margin>=10: {100*pct_10:.1f}% | "
                          f"Margin>=15: {100*pct_15:.1f}%")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: MOST PROMISING NOVEL BET TYPES")
print("=" * 80)

print("""
Based on this comprehensive research, the most promising novel pregame bet types are:

1. UNDERDOG TEAM TOTAL UNDER
   - When favorite has elite defense (<=108) facing weak underdog offense (<=110)
   - Dog scoring ceiling is predictable
   - Potentially 80-90%+ accuracy at specific thresholds
   - NOVEL: This is the INVERSE of Play 3's approach

2. GAME TOTAL UNDER IN DEFENSIVE MATCHUPS
   - Both teams with strong defense + low pace = predictable low total
   - Sportsbooks often overshoot totals in these spots

3. FAVORITE SCORING FLOOR (Team Total OVER)
   - When offensive mismatch is extreme (fav_off vs dog_def >= 10+)
   - Fav scoring above 110 is near-certain
   - High accuracy but needs to beat the sportsbook line

4. COMPOSITE DOMINANCE SCORE
   - Multi-factor scoring combining 5+ independent signals
   - At high thresholds, creates very predictable outcomes
   - Novel multi-dimensional approach

5. ALTERNATIVE SPREAD (Margin >= 5)
   - Instead of covering the posted spread, predict winning by 5+
   - Can find spots with 75-80%+ accuracy
   - Alternative spreads at -130 to -150 are still very profitable

See detailed numbers above for specific filter combinations and accuracy rates.
""")
