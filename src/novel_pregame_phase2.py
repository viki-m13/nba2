"""
NOVEL PREGAME BET RESEARCH - PHASE 2: DEEP VALIDATION
======================================================

Phase 1 discovered several blockbuster findings:
  1. NetGap>=12, FavOff>=116, Home → 100% (30/30) score 110+
  2. Composite Score >= 7 → 85.1% ML (67 games) - captures games Play 3 misses
  3. TCI >= 12 + Home → 86.4% ML, avg margin +13.4 (44 games)
  4. Fav OFF>=116 + Fav DEF<=110 + NetGap>=10 → 85% ML, 90% score 110+ (20 games)

Phase 2 Goals:
  1. Estimate sportsbook lines and test RELATIVE accuracy (not just absolute)
  2. Temporal validation (train/test split) to prevent overfitting
  3. Develop the novel COMPOSITE DOMINANCE model
  4. Find the optimal bet type (spread, total, or team total)
  5. Identify games this model captures that Play 3 MISSES (incremental value)
"""

import json
import math
import statistics
from collections import defaultdict

with open('data/pregame_predictions_2025.json') as f:
    games = json.load(f)

# =============================================================================
# FEATURE ENGINEERING (same as phase 1)
# =============================================================================
for g in games:
    if g['fav_is_home']:
        g['fav_off'] = g['home_off']
        g['fav_def'] = g['home_def']
        g['fav_net'] = g['home_net']
        g['fav_win_pct'] = g['home_win_pct']
        g['dog_off'] = g['away_off']
        g['dog_def'] = g['away_def']
        g['dog_net'] = g['away_net']
        g['dog_win_pct'] = g['away_win_pct']
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
    g['net_gap'] = abs(g['net_diff'])
    g['fav_margin'] = g['fav_actual_score'] - g['dog_score']
    g['off_vs_def_mismatch'] = g['fav_off'] - g['dog_def']
    g['win_pct_gap'] = abs(g['fav_win_pct'] - g['dog_win_pct'])
    g['cage_score'] = g['fav_def'] - g['dog_off']
    g['expected_pace'] = (g['home_off'] + g['away_off']) / 2.0

# =============================================================================
# ESTIMATE SPORTSBOOK LINES
# =============================================================================
# In NBA, sportsbooks typically set:
#   - Spread ≈ predicted margin (based on power ratings + home court)
#   - Total ≈ sum of projected team scores (based on pace-adjusted ratings)
#   - Team Total ≈ team's projected score in this matchup

# Our best proxy for the sportsbook spread is the predicted_margin from our model
# Our best proxy for the sportsbook total is (home_off + away_off) adjusted for matchup

print("=" * 80)
print("PHASE 2: ESTIMATING SPORTSBOOK LINES & RELATIVE ACCURACY")
print("=" * 80)

for g in games:
    # Estimated spread (favorite perspective, negative means favorite gives points)
    g['est_spread'] = -abs(g['predicted_margin'])

    # Estimated game total: average of both teams' offensive production
    # Sportsbooks use pace-adjusted metrics; we approximate with raw off ratings
    g['est_total'] = g['home_off'] + g['away_off']  # simple estimate

    # Estimated team totals
    # Sportsbook team total ≈ team's off rating adjusted for opponent defense
    # We use: (team_off + opp_def) / 2 as a matchup-adjusted estimate
    g['est_fav_total'] = (g['fav_off'] + g['dog_def']) / 2.0
    g['est_dog_total'] = (g['dog_off'] + g['fav_def']) / 2.0

    # Did the game go OVER the estimated total?
    g['total_over_est'] = g['total'] > g['est_total']

    # Did the favorite score OVER their estimated team total?
    g['fav_over_est'] = g['fav_actual_score'] > g['est_fav_total']

    # Did the dog score UNDER their estimated team total?
    g['dog_under_est'] = g['dog_score'] < g['est_dog_total']

    # Did the favorite cover the estimated spread?
    g['covers_est_spread'] = g['fav_margin'] > abs(g['est_spread'])


# =============================================================================
# CHECK: How good are our spread/total estimates?
# =============================================================================

print("\nSPREAD ESTIMATION QUALITY:")
print(f"  Avg predicted margin: {statistics.mean([abs(g['predicted_margin']) for g in games]):.1f}")
print(f"  Avg actual margin: {statistics.mean([abs(g['actual_margin']) for g in games]):.1f}")
print(f"  Correlation: predicted_margin vs actual_margin")

# Simple correlation
pred = [g['predicted_margin'] for g in games]
actual = [g['actual_margin'] for g in games]
mean_p = statistics.mean(pred)
mean_a = statistics.mean(actual)
cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(pred, actual)) / len(pred)
std_p = statistics.stdev(pred)
std_a = statistics.stdev(actual)
corr = cov / (std_p * std_a) if std_p * std_a > 0 else 0
print(f"  Pearson r = {corr:.3f}")

# Spread coverage overall
total_covers = sum(1 for g in games if g['covers_est_spread'])
print(f"  Estimated spread coverage: {total_covers}/{len(games)} = {100*total_covers/len(games):.1f}%")
# Should be ~50% if estimate is accurate

print("\nGAME TOTAL ESTIMATION QUALITY:")
est_totals = [g['est_total'] for g in games]
act_totals = [g['total'] for g in games]
print(f"  Avg estimated total: {statistics.mean(est_totals):.1f}")
print(f"  Avg actual total: {statistics.mean(act_totals):.1f}")
mean_et = statistics.mean(est_totals)
mean_at = statistics.mean(act_totals)
cov_t = sum((e - mean_et) * (a - mean_at) for e, a in zip(est_totals, act_totals)) / len(est_totals)
std_et = statistics.stdev(est_totals)
std_at = statistics.stdev(act_totals)
corr_t = cov_t / (std_et * std_at) if std_et * std_at > 0 else 0
print(f"  Pearson r = {corr_t:.3f}")

total_over = sum(1 for g in games if g['total_over_est'])
print(f"  Games going OVER est total: {total_over}/{len(games)} = {100*total_over/len(games):.1f}%")


print("\nTEAM TOTAL ESTIMATION QUALITY:")
fav_over = sum(1 for g in games if g['fav_over_est'])
dog_under = sum(1 for g in games if g['dog_under_est'])
print(f"  Fav over est team total: {fav_over}/{len(games)} = {100*fav_over/len(games):.1f}%")
print(f"  Dog under est team total: {dog_under}/{len(games)} = {100*dog_under/len(games):.1f}%")


# =============================================================================
# TEMPORAL VALIDATION SPLIT
# =============================================================================

print("\n" + "=" * 80)
print("TEMPORAL VALIDATION: Train on first 70%, test on last 30%")
print("=" * 80)

games_sorted = sorted(games, key=lambda g: g['date'])
split_idx = int(len(games_sorted) * 0.7)
train_games = games_sorted[:split_idx]
test_games = games_sorted[split_idx:]

print(f"  Train: {len(train_games)} games ({train_games[0]['date']} to {train_games[-1]['date']})")
print(f"  Test:  {len(test_games)} games ({test_games[0]['date']} to {test_games[-1]['date']})")


def temporal_test(filter_fn, description, bet_type_fn, bet_name):
    """Test a filter on both train and test sets."""
    train_q = [g for g in train_games if filter_fn(g)]
    test_q = [g for g in test_games if filter_fn(g)]

    if len(train_q) < 5 or len(test_q) < 3:
        return None

    train_acc = sum(1 for g in train_q if bet_type_fn(g)) / len(train_q)
    test_acc = sum(1 for g in test_q if bet_type_fn(g)) / len(test_q)

    return {
        'description': description,
        'bet': bet_name,
        'train_n': len(train_q),
        'test_n': len(test_q),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'degradation': train_acc - test_acc,
    }


# =============================================================================
# NOVEL CONCEPT: "SCORING ENVIRONMENT PREDICTOR" (SEP)
# =============================================================================

print("\n" + "=" * 80)
print("NOVEL CONCEPT: SCORING ENVIRONMENT PREDICTOR (SEP)")
print("Predicts GAME TOTAL relative to estimated line, not absolute thresholds")
print("=" * 80)

# The novel insight: when the offensive mismatch is extreme, the ACTUAL
# game total systematically EXCEEDS the estimated total because:
# 1. The favorite scores MORE than their average (facing weak defense)
# 2. The underdog is forced into a faster pace (chasing a deficit)
# 3. More possessions = more total points
# 4. Garbage time scoring inflates the total further

# Test: In high-mismatch games, does the total systematically exceed expectations?

for filter_name, filter_fn in [
    ("NetGap>=10 + FavOff>=118 + Home (Play 3 filter)",
     lambda g: g['net_gap'] >= 10 and g['fav_off'] >= 118 and g['fav_is_home']),
    ("NetGap>=12 + FavOff>=116 + Home",
     lambda g: g['net_gap'] >= 12 and g['fav_off'] >= 116 and g['fav_is_home']),
    ("NetGap>=10 + FavOff>=116 + Home",
     lambda g: g['net_gap'] >= 10 and g['fav_off'] >= 116 and g['fav_is_home']),
    ("NetGap>=8 + FavOff>=118 + Home",
     lambda g: g['net_gap'] >= 8 and g['fav_off'] >= 118 and g['fav_is_home']),
    ("Expected Pace>=118",
     lambda g: g['expected_pace'] >= 118),
    ("FavOff>=120 + Home",
     lambda g: g['fav_off'] >= 120 and g['fav_is_home']),
]:
    qualifying = [g for g in games if filter_fn(g)]
    if len(qualifying) < 10:
        continue

    total_diffs = [g['total'] - g['est_total'] for g in qualifying]
    avg_diff = statistics.mean(total_diffs)
    pct_over = sum(1 for d in total_diffs if d > 0) / len(qualifying)

    # What about a buffer? Total exceeds est_total + buffer
    pct_over_2 = sum(1 for d in total_diffs if d > 2) / len(qualifying)
    pct_over_5 = sum(1 for d in total_diffs if d > 5) / len(qualifying)

    fav_diffs = [g['fav_actual_score'] - g['est_fav_total'] for g in qualifying]
    avg_fav_diff = statistics.mean(fav_diffs)
    pct_fav_over = sum(1 for d in fav_diffs if d > 0) / len(qualifying)

    print(f"\n  {filter_name}: {len(qualifying)} games")
    print(f"    Game Total vs Est: avg diff = {avg_diff:+.1f}, OVER rate = {100*pct_over:.1f}%")
    print(f"    Game Total > Est+2: {100*pct_over_2:.1f}% | > Est+5: {100*pct_over_5:.1f}%")
    print(f"    Fav Score vs Est: avg diff = {avg_fav_diff:+.1f}, OVER rate = {100*pct_fav_over:.1f}%")


# =============================================================================
# NOVEL CONCEPT: "MULTI-DIMENSIONAL DOMINANCE SCORE" (MDDS)
# This is the key patentable idea - a weighted multi-factor scoring system
# =============================================================================

print("\n" + "=" * 80)
print("NOVEL CONCEPT: MULTI-DIMENSIONAL DOMINANCE SCORE (MDDS)")
print("Weighted scoring across 5 independent dominance dimensions")
print("Patentable: Continuous scoring replaces rigid threshold filters")
print("=" * 80)


def compute_mdds(g):
    """
    Multi-Dimensional Dominance Score (MDDS)

    5 independent dimensions of team dominance, each scored 0-3:
      1. Net Rating Gap (overall quality differential)
      2. Offensive Firepower (favorite's raw scoring ability)
      3. Defensive Cage (favorite's ability to suppress opponent scoring)
      4. Win Consistency (record gap reflecting sustained excellence)
      5. Home Court Amplifier

    The NOVEL insight: these dimensions are PARTIALLY INDEPENDENT.
    A team can be dominant via different pathways:
      - Path A: Elite offense (dimension 2) + weak opponent defense (dimension 1)
      - Path B: Elite defense (dimension 3) + high consistency (dimension 4)
      - Path C: Balanced superiority across all dimensions

    The MDDS captures all pathways, unlike rigid filters that only catch Path A.
    """
    score = 0.0

    # Dimension 1: Net Rating Gap (overall quality differential)
    # Range: 0-3, weight reflects primacy of overall quality gap
    ng = g['net_gap']
    if ng >= 14:
        score += 3.0
    elif ng >= 10:
        score += 2.0 + (ng - 10) / 4.0  # linear interpolation 2.0-3.0
    elif ng >= 7:
        score += 1.0 + (ng - 7) / 3.0   # 1.0-2.0
    elif ng >= 4:
        score += (ng - 4) / 3.0          # 0.0-1.0

    # Dimension 2: Offensive Firepower
    fo = g['fav_off']
    if fo >= 122:
        score += 3.0
    elif fo >= 118:
        score += 2.0 + (fo - 118) / 4.0
    elif fo >= 115:
        score += 1.0 + (fo - 115) / 3.0
    elif fo >= 112:
        score += (fo - 112) / 3.0

    # Dimension 3: Defensive Cage (lower cage_score = better fav defense vs dog offense)
    cage = g['cage_score']  # fav_def - dog_off; negative = fav defense better
    if cage <= -6:
        score += 3.0
    elif cage <= -3:
        score += 2.0 + (-3 - cage) / 3.0
    elif cage <= 0:
        score += 1.0 + (0 - cage) / 3.0
    elif cage <= 3:
        score += (3 - cage) / 3.0

    # Dimension 4: Win Consistency Gap
    wpg = g['win_pct_gap']
    if wpg >= 0.5:
        score += 3.0
    elif wpg >= 0.4:
        score += 2.0 + (wpg - 0.4) / 0.1
    elif wpg >= 0.3:
        score += 1.0 + (wpg - 0.3) / 0.1
    elif wpg >= 0.2:
        score += (wpg - 0.2) / 0.1

    # Dimension 5: Home Court Amplifier
    if g['fav_is_home']:
        score += 1.5  # home court is binary but worth 1.5 "points"

    return score


for g in games:
    g['mdds'] = compute_mdds(g)


# Analyze MDDS at various thresholds
print("\nMDDS THRESHOLD ANALYSIS:")
print("-" * 120)
print(f"{'Threshold':>10s} | {'N':>4s} | {'ML%':>6s} | {'Sp.Cov%':>7s} | {'Fav>=110':>8s} | {'Fav>=115':>8s} | "
      f"{'Tot>Est':>7s} | {'Dog<110':>7s} | {'AvgMargin':>9s} | {'AvgTotal':>8s}")
print("-" * 120)

for threshold in [5, 6, 7, 8, 9, 10, 11, 12, 13]:
    qualifying = [g for g in games if g['mdds'] >= threshold]
    if len(qualifying) < 8:
        continue

    n = len(qualifying)
    ml = sum(1 for g in qualifying if g['fav_won']) / n
    sp_cov = sum(1 for g in qualifying if g['covers_est_spread']) / n
    fav_110 = sum(1 for g in qualifying if g['fav_actual_score'] >= 110) / n
    fav_115 = sum(1 for g in qualifying if g['fav_actual_score'] >= 115) / n
    tot_over = sum(1 for g in qualifying if g['total_over_est']) / n
    dog_u110 = sum(1 for g in qualifying if g['dog_score'] < 110) / n
    avg_margin = statistics.mean([g['fav_margin'] for g in qualifying])
    avg_total = statistics.mean([g['total'] for g in qualifying])

    print(f"{threshold:10.0f} | {n:4d} | {100*ml:5.1f}% | {100*sp_cov:6.1f}% | {100*fav_110:7.1f}% | "
          f"{100*fav_115:7.1f}% | {100*tot_over:6.1f}% | {100*dog_u110:6.1f}% | {avg_margin:+8.1f} | {avg_total:7.1f}")


# =============================================================================
# KEY ANALYSIS: MDDS captures games that Play 3 MISSES
# =============================================================================

print("\n" + "=" * 80)
print("MDDS vs PLAY 3: INCREMENTAL VALUE ANALYSIS")
print("Which high-confidence games does MDDS find that Play 3 misses?")
print("=" * 80)

def play3_filter(g):
    return g['net_gap'] >= 10 and g['fav_off'] >= 118 and g['fav_is_home']

play3_games = set(i for i, g in enumerate(games) if play3_filter(g))

for mdds_th in [7, 8, 9, 10]:
    mdds_games = set(i for i, g in enumerate(games) if g['mdds'] >= mdds_th)

    both = play3_games & mdds_games
    mdds_only = mdds_games - play3_games
    play3_only = play3_games - mdds_games

    # Accuracy on MDDS-only games (the incremental picks)
    mdds_only_games = [games[i] for i in mdds_only]
    if len(mdds_only_games) >= 5:
        ml_mdds_only = sum(1 for g in mdds_only_games if g['fav_won']) / len(mdds_only_games)
        fav110_mdds_only = sum(1 for g in mdds_only_games if g['fav_actual_score'] >= 110) / len(mdds_only_games)
        avg_margin_mdds = statistics.mean([g['fav_margin'] for g in mdds_only_games])

        print(f"\n  MDDS >= {mdds_th}:")
        print(f"    Total MDDS games: {len(mdds_games)}, Play 3 games: {len(play3_games)}")
        print(f"    Overlap: {len(both)}, MDDS-only: {len(mdds_only)}, Play3-only: {len(play3_only)}")
        print(f"    MDDS-only accuracy: ML={100*ml_mdds_only:.1f}%, Fav>=110={100*fav110_mdds_only:.1f}%, AvgMargin={avg_margin_mdds:+.1f}")

        # Show what these games look like
        print(f"    MDDS-only game examples:")
        for g in sorted(mdds_only_games, key=lambda x: -x['mdds'])[:5]:
            print(f"      {g['date']} {g['fav']}(home={g['fav_is_home']}) vs {g['dog']}: "
                  f"MDDS={g['mdds']:.1f}, NetGap={g['net_gap']:.1f}, FavOff={g['fav_off']:.1f}, "
                  f"FavDef={g['fav_def']:.1f}, WP%Gap={g['win_pct_gap']:.2f}, "
                  f"Actual: margin={g['fav_margin']:+d}, fav_score={g['fav_actual_score']}")


# =============================================================================
# TEMPORAL VALIDATION OF MDDS
# =============================================================================

print("\n" + "=" * 80)
print("TEMPORAL VALIDATION: MDDS MODEL")
print("=" * 80)

train_sorted = sorted(train_games, key=lambda g: g['date'])
test_sorted = sorted(test_games, key=lambda g: g['date'])

# Compute MDDS for all (already done above)
for mdds_th in [7, 8, 9, 10]:
    result_ml = temporal_test(
        lambda g, th=mdds_th: g['mdds'] >= th,
        f"MDDS >= {mdds_th}",
        lambda g: g['fav_won'],
        "ML"
    )
    result_fav110 = temporal_test(
        lambda g, th=mdds_th: g['mdds'] >= th,
        f"MDDS >= {mdds_th}",
        lambda g: g['fav_actual_score'] >= 110,
        "Fav Score >= 110"
    )
    result_spread = temporal_test(
        lambda g, th=mdds_th: g['mdds'] >= th,
        f"MDDS >= {mdds_th}",
        lambda g: g['covers_est_spread'],
        "Covers Est Spread"
    )

    if result_ml and result_fav110:
        print(f"\n  MDDS >= {mdds_th}:")
        for r in [result_ml, result_fav110, result_spread]:
            if r:
                deg_marker = " *** OVERFITTING ***" if r['degradation'] > 0.15 else ""
                print(f"    {r['bet']:20s}: Train={r['train_n']:3d} games {100*r['train_acc']:.1f}% | "
                      f"Test={r['test_n']:3d} games {100*r['test_acc']:.1f}% | "
                      f"Degradation: {100*r['degradation']:+.1f}%{deg_marker}")


# Also validate Play 3 for comparison
print("\n  PLAY 3 (baseline):")
for bet_name, bet_fn in [
    ("ML", lambda g: g['fav_won']),
    ("Fav Score >= 110", lambda g: g['fav_actual_score'] >= 110),
    ("Covers Est Spread", lambda g: g['covers_est_spread']),
]:
    r = temporal_test(play3_filter, "Play 3", bet_fn, bet_name)
    if r:
        print(f"    {r['bet']:20s}: Train={r['train_n']:3d} games {100*r['train_acc']:.1f}% | "
              f"Test={r['test_n']:3d} games {100*r['test_acc']:.1f}% | "
              f"Degradation: {100*r['degradation']:+.1f}%")


# =============================================================================
# DEEP ANALYSIS: GAME TOTAL OVER AS A BET TYPE
# =============================================================================

print("\n" + "=" * 80)
print("GAME TOTAL OVER: SYSTEMATIC EDGE ANALYSIS")
print("When do game totals SYSTEMATICALLY exceed the estimated line?")
print("=" * 80)

# The key question: can we find conditions where the game total
# consistently EXCEEDS what the sportsbook would set as the line?

# Insight: Sportsbooks set the total based on team averages.
# But in BLOWOUT situations:
# 1. The favorite plays at full pace (not slowing down)
# 2. The underdog is forced to play fast (chasing deficit)
# 3. More fouls = more free throws = more points
# 4. Garbage time = both teams score freely
# This creates a SYSTEMATIC upward bias in total scoring

print("\nAnalysis: Total Over Rate (vs estimated line) by game conditions:")
print("-" * 100)

filters_to_test = [
    # (name, filter_fn)
    ("All games", lambda g: True),
    ("Fav wins", lambda g: g['fav_won']),
    ("Fav wins by 10+", lambda g: g['fav_margin'] >= 10),
    ("Fav wins by 15+", lambda g: g['fav_margin'] >= 15),
    ("NetGap >= 10", lambda g: g['net_gap'] >= 10),
    ("NetGap >= 10 + Home", lambda g: g['net_gap'] >= 10 and g['fav_is_home']),
    ("MDDS >= 7", lambda g: g['mdds'] >= 7),
    ("MDDS >= 8", lambda g: g['mdds'] >= 8),
    ("MDDS >= 9", lambda g: g['mdds'] >= 9),
    ("MDDS >= 10", lambda g: g['mdds'] >= 10),
    ("FavOff >= 118 + Home", lambda g: g['fav_off'] >= 118 and g['fav_is_home']),
    ("FavOff >= 120", lambda g: g['fav_off'] >= 120),
    ("Play 3 filter", play3_filter),
]

for name, filt in filters_to_test:
    qualifying = [g for g in games if filt(g)]
    if len(qualifying) < 10:
        continue

    total_diffs = [g['total'] - g['est_total'] for g in qualifying]
    avg_diff = statistics.mean(total_diffs)
    pct_over = sum(1 for d in total_diffs if d > 0) / len(qualifying)

    # Fav score vs estimate
    fav_diffs = [g['fav_actual_score'] - g['est_fav_total'] for g in qualifying]
    avg_fav_diff = statistics.mean(fav_diffs)
    pct_fav_over = sum(1 for d in fav_diffs if d > 0) / len(qualifying)

    # Dog score vs estimate
    dog_diffs = [g['dog_score'] - g['est_dog_total'] for g in qualifying]
    avg_dog_diff = statistics.mean(dog_diffs)
    pct_dog_under = sum(1 for d in dog_diffs if d < 0) / len(qualifying)

    print(f"  {name:30s} ({len(qualifying):3d} games): "
          f"Total OVER={100*pct_over:.1f}% (avg {avg_diff:+.1f}) | "
          f"Fav OVER={100*pct_fav_over:.1f}% (avg {avg_fav_diff:+.1f}) | "
          f"Dog UNDER={100*pct_dog_under:.1f}% (avg {avg_dog_diff:+.1f})")


# =============================================================================
# DISCOVERY: BLOWOUT SCORING INFLATION EFFECT
# =============================================================================

print("\n" + "=" * 80)
print("DISCOVERY: BLOWOUT SCORING INFLATION EFFECT")
print("When the mismatch is extreme, BOTH teams score more than expected")
print("This creates a systematic OVER on game totals")
print("=" * 80)

# Compute the "inflation" for each game - how much did scoring exceed expectations
for g in games:
    g['scoring_inflation'] = g['total'] - g['est_total']

# Bin by net_gap to see the relationship
print("\nScoring inflation by net rating gap:")
print("-" * 80)
for ng_lo, ng_hi in [(0, 4), (4, 7), (7, 10), (10, 13), (13, 16), (16, 30)]:
    bin_games = [g for g in games if ng_lo <= g['net_gap'] < ng_hi]
    if len(bin_games) < 8:
        continue
    avg_inflation = statistics.mean([g['scoring_inflation'] for g in bin_games])
    pct_over = sum(1 for g in bin_games if g['scoring_inflation'] > 0) / len(bin_games)
    avg_fav_inf = statistics.mean([g['fav_actual_score'] - g['est_fav_total'] for g in bin_games])
    avg_dog_inf = statistics.mean([g['dog_score'] - g['est_dog_total'] for g in bin_games])

    print(f"  NetGap {ng_lo:2d}-{ng_hi:2d}: {len(bin_games):3d} games | "
          f"Avg inflation: {avg_inflation:+.1f} pts | OVER rate: {100*pct_over:.1f}% | "
          f"Fav inflation: {avg_fav_inf:+.1f} | Dog inflation: {avg_dog_inf:+.1f}")

# Bin by MDDS
print("\nScoring inflation by MDDS:")
print("-" * 80)
for mdds_lo, mdds_hi in [(0, 4), (4, 6), (6, 8), (8, 10), (10, 15)]:
    bin_games = [g for g in games if mdds_lo <= g['mdds'] < mdds_hi]
    if len(bin_games) < 8:
        continue
    avg_inflation = statistics.mean([g['scoring_inflation'] for g in bin_games])
    pct_over = sum(1 for g in bin_games if g['scoring_inflation'] > 0) / len(bin_games)

    print(f"  MDDS {mdds_lo:2.0f}-{mdds_hi:2.0f}: {len(bin_games):3d} games | "
          f"Avg inflation: {avg_inflation:+.1f} pts | OVER rate: {100*pct_over:.1f}%")


# =============================================================================
# NOVEL BET TYPE: "DOMINANCE TOTAL OVER"
# =============================================================================

print("\n" + "=" * 80)
print("NOVEL BET TYPE: DOMINANCE TOTAL OVER (DTO)")
print("=" * 80)
print("""
CONCEPT: "Dominance Total Over" (DTO)
======================================
When a pre-game dominance signal fires (MDDS >= threshold),
bet the GAME TOTAL OVER at -110.

THEORETICAL BASIS:
  In dominant matchups, scoring inflates beyond what sportsbooks model because:
  1. Dominant teams don't slow down when leading (pace stays high)
  2. Trailing teams take more 3-point attempts (higher variance, higher average)
  3. Fouls accelerate late (more free throw possessions)
  4. Garbage time scoring inflates both sides

  Sportsbooks set totals based on pace-adjusted team averages.
  But they UNDER-ESTIMATE the inflation effect of blowouts on total scoring.

DIFFERENT FROM PLAY 3:
  - Play 3: Bets SPREAD (who wins and by how much)
  - DTO: Bets TOTAL (combined scoring environment)
  - These are INDEPENDENT bet types on DIFFERENT markets
  - Can be bet on the SAME game for double exposure
""")


# =============================================================================
# VALIDATE DTO WITH TEMPORAL SPLIT
# =============================================================================

print("DTO TEMPORAL VALIDATION:")
print("-" * 100)

for mdds_th in [7, 8, 9, 10]:
    for buffer in [0, 2, 4]:
        r = temporal_test(
            lambda g, th=mdds_th: g['mdds'] >= th,
            f"MDDS >= {mdds_th}",
            lambda g, b=buffer: g['total'] > g['est_total'] + b,
            f"Total > Est+{buffer}"
        )
        if r:
            deg_marker = ""
            if r['degradation'] > 0.15: deg_marker = " *** WATCH ***"
            if r['test_acc'] >= 0.55:
                edge = r['test_acc'] * (100/110) - (1 - r['test_acc'])
                roi = edge * 100
            else:
                roi = 0
            print(f"  MDDS>={mdds_th}, Total>Est+{buffer}: "
                  f"Train={r['train_n']:3d}@{100*r['train_acc']:.1f}% | "
                  f"Test={r['test_n']:3d}@{100*r['test_acc']:.1f}% | "
                  f"Est ROI@-110: {roi:+.1f}%{deg_marker}")


# =============================================================================
# ALTERNATIVE NOVEL CONCEPT: "MATCHUP SCORING FLOOR" (MSF)
# Predicts minimum team score based on specific matchup dynamics
# =============================================================================

print("\n" + "=" * 80)
print("ALTERNATIVE NOVEL CONCEPT: MATCHUP SCORING FLOOR (MSF)")
print("Predicts the MINIMUM score the favorite will achieve")
print("Unlike average-based predictions, this predicts the FLOOR")
print("=" * 80)

# The novel idea: in certain matchups, the favorite's minimum possible score
# is predictable because the opponent CANNOT stop them.
# This is different from predicting the AVERAGE - it's about the worst case.

# Floor = the minimum score across all games matching this profile
# If the floor is consistently above a threshold (e.g., 110), we have a bet.

print("\nFavorite Scoring Floor Analysis:")
print("-" * 100)

for mdds_th in [7, 8, 9, 10, 11, 12]:
    qualifying = [g for g in games if g['mdds'] >= mdds_th]
    if len(qualifying) < 8:
        continue

    fav_scores = sorted([g['fav_actual_score'] for g in qualifying])
    floor = fav_scores[0]
    pct_5 = fav_scores[int(len(fav_scores) * 0.05)] if len(fav_scores) >= 20 else fav_scores[0]  # 5th percentile
    pct_10 = fav_scores[int(len(fav_scores) * 0.10)] if len(fav_scores) >= 10 else fav_scores[0]  # 10th percentile
    median = statistics.median(fav_scores)
    avg = statistics.mean(fav_scores)

    print(f"  MDDS >= {mdds_th:2d}: {len(qualifying):3d} games | "
          f"Floor={floor:3d} | 5th%ile={pct_5:3d} | 10th%ile={pct_10:3d} | "
          f"Median={median:.0f} | Avg={avg:.1f}")


# =============================================================================
# DEEP DIVE: WHAT MAKES MDDS DIFFERENT FROM PLAY 3
# =============================================================================

print("\n" + "=" * 80)
print("CRITICAL ANALYSIS: WHY MDDS IS NOVEL AND COMPLEMENTARY TO PLAY 3")
print("=" * 80)

print("""
PLAY 3 FILTER: net_gap >= 10 AND fav_off >= 118 AND fav_is_home
  - Rigid binary thresholds
  - Only captures ONE pathway to dominance (offense + net gap + home)
  - Misses games where team dominates via defense or consistency

MDDS (Multi-Dimensional Dominance Score):
  - Continuous scoring across 5 dimensions
  - Allows DIFFERENT PATHWAYS TO DOMINANCE:
    * Pathway A: Elite offense + net gap (similar to Play 3)
    * Pathway B: Strong defense + high consistency + home court
    * Pathway C: Moderate offense + elite defense + huge win% gap
    * Pathway D: Extreme net gap alone (compensates for other factors)

  - MATHEMATICALLY: MDDS >= 8 can be achieved by:
    * NetGap=14(3.0) + FavOff=118(2.0) + Home(1.5) + Cage=-2(1.0) + WP%Gap=0.25(0.5) = 8.0
    * NetGap=10(2.0) + FavOff=116(1.3) + Home(1.5) + Cage=-6(3.0) + WP%Gap=0.30(1.0) = 8.8
    * NetGap=7(1.0) + FavOff=122(3.0) + Home(1.5) + Cage=-3(2.0) + WP%Gap=0.40(2.0) = 9.5
""")

# Show the pathways
print("  PATHWAY ANALYSIS (MDDS >= 8 games):")
mdds8_games = [g for g in games if g['mdds'] >= 8]
for g in sorted(mdds8_games, key=lambda x: -x['mdds'])[:15]:
    # Identify dominant pathway
    scores = {
        'NetGap': min(3, max(0, (g['net_gap'] - 4) / 3.33)),
        'Offense': min(3, max(0, (g['fav_off'] - 112) / 3.33)),
        'Defense': min(3, max(0, (3 - g['cage_score']) / 3)),
        'Consistency': min(3, max(0, (g['win_pct_gap'] - 0.2) / 0.1)),
        'Home': 1.5 if g['fav_is_home'] else 0,
    }
    top_dim = max(scores, key=scores.get)
    is_p3 = play3_filter(g)
    p3_str = "P3+MDDS" if is_p3 else "MDDS-only"

    print(f"    {g['date']} {g['fav']:3s} vs {g['dog']:3s}: MDDS={g['mdds']:.1f} [{p3_str:8s}] "
          f"Top={top_dim:12s} | margin={g['fav_margin']:+3d} | "
          f"NG={g['net_gap']:.0f} FO={g['fav_off']:.0f} FD={g['fav_def']:.0f} WP={g['win_pct_gap']:.2f}")


# =============================================================================
# COMBINED PLAY: MDDS SPREAD + TOTAL DUAL BET
# =============================================================================

print("\n" + "=" * 80)
print("COMBINED PLAY: MDDS 'DUAL DOMINANCE' BET")
print("Bet BOTH spread AND total over from a single pregame signal")
print("=" * 80)

for mdds_th in [8, 9, 10]:
    qualifying = [g for g in games if g['mdds'] >= mdds_th]
    if len(qualifying) < 10:
        continue

    # ML accuracy (proxy for spread at -110)
    ml = sum(1 for g in qualifying if g['fav_won']) / len(qualifying)
    # Total over accuracy
    total_over = sum(1 for g in qualifying if g['total'] > g['est_total']) / len(qualifying)
    # Both hit (parlay)
    both = sum(1 for g in qualifying if g['fav_won'] and g['total'] > g['est_total']) / len(qualifying)
    # Either hit (coverage)
    either = sum(1 for g in qualifying if g['fav_won'] or g['total'] > g['est_total']) / len(qualifying)
    # Both miss
    neither = 1 - either

    # EV at -110 for each leg
    ev_ml = ml * (100/110) - (1 - ml)
    ev_total = total_over * (100/110) - (1 - total_over)
    combined_ev = ev_ml + ev_total  # two independent bets

    print(f"\n  MDDS >= {mdds_th} ({len(qualifying)} games):")
    print(f"    Spread (ML): {100*ml:.1f}% → EV per bet: {ev_ml:+.3f} → ROI: {100*ev_ml:+.1f}%")
    print(f"    Total Over:  {100*total_over:.1f}% → EV per bet: {ev_total:+.3f} → ROI: {100*ev_total:+.1f}%")
    print(f"    Both hit:    {100*both:.1f}% | Either hit: {100*either:.1f}% | Both miss: {100*neither:.1f}%")
    print(f"    Combined EV (2 bets): {combined_ev:+.3f} per unit ({100*combined_ev:+.1f}% ROI on total action)")


# =============================================================================
# NOVEL CONCEPT: "OPPONENT DEFENSIVE COLLAPSE" (ODC)
# =============================================================================

print("\n" + "=" * 80)
print("NOVEL CONCEPT: OPPONENT DEFENSIVE COLLAPSE (ODC)")
print("When specific conditions predict the UNDERDOG's defense will fail")
print("=" * 80)

# Key insight: In certain games, the underdog's defense is SO outmatched
# that it "collapses" - allowing extreme scoring. This shows up as:
# 1. The favorite scores way above their average
# 2. The game total exceeds expectations
# 3. The margin is larger than the rating gap suggests

# ODC Score = how badly outmatched the dog's defense is
for g in games:
    # Components:
    # 1. Favorite offense vs dog defense mismatch
    off_mismatch = max(0, g['fav_off'] - g['dog_def'])
    # 2. Dog defense relative to league average (higher = worse)
    dog_def_badness = max(0, g['dog_def'] - 110)
    # 3. Favorite offensive consistency (win pct as proxy)
    fav_consistency = g['fav_win_pct']

    g['odc'] = off_mismatch * 0.5 + dog_def_badness * 0.3 + fav_consistency * 10 * 0.2

print("ODC Score Analysis:")
print("-" * 100)
for odc_th in [4, 5, 6, 7, 8, 9, 10]:
    qualifying = [g for g in games if g['odc'] >= odc_th]
    if len(qualifying) < 10:
        continue

    ml = sum(1 for g in qualifying if g['fav_won']) / len(qualifying)
    avg_fav_score = statistics.mean([g['fav_actual_score'] for g in qualifying])
    avg_total = statistics.mean([g['total'] for g in qualifying])
    fav_over_est = sum(1 for g in qualifying if g['fav_actual_score'] > g['est_fav_total']) / len(qualifying)
    total_over_est = sum(1 for g in qualifying if g['total'] > g['est_total']) / len(qualifying)

    print(f"  ODC >= {odc_th}: {len(qualifying):3d} games | "
          f"ML={100*ml:.1f}% | AvgFavScore={avg_fav_score:.1f} | AvgTotal={avg_total:.1f} | "
          f"FavOverEst={100*fav_over_est:.1f}% | TotalOverEst={100*total_over_est:.1f}%")


# =============================================================================
# FINAL: BEST CANDIDATE MODELS FOR PLAY 4
# =============================================================================

print("\n" + "=" * 80)
print("FINAL: BEST CANDIDATE MODELS FOR NOVEL PREGAME BET (PLAY 4)")
print("=" * 80)

candidates = []

# Candidate 1: MDDS Spread
for th in [7, 8, 9, 10]:
    q = [g for g in games if g['mdds'] >= th]
    if len(q) < 10:
        continue
    ml = sum(1 for g in q if g['fav_won']) / len(q)
    r = temporal_test(lambda g, t=th: g['mdds'] >= t, f"MDDS>={th}", lambda g: g['fav_won'], "ML")
    if r:
        candidates.append({
            'name': f'MDDS>={th} → Spread',
            'games': len(q),
            'full_acc': ml,
            'train_acc': r['train_acc'],
            'test_acc': r['test_acc'],
            'bet': 'Spread at -110',
        })

# Candidate 2: MDDS Game Total Over
for th in [7, 8, 9, 10]:
    q = [g for g in games if g['mdds'] >= th]
    if len(q) < 10:
        continue
    ov = sum(1 for g in q if g['total'] > g['est_total']) / len(q)
    r = temporal_test(lambda g, t=th: g['mdds'] >= t, f"MDDS>={th}", lambda g: g['total'] > g['est_total'], "Total Over")
    if r:
        candidates.append({
            'name': f'MDDS>={th} → Total Over',
            'games': len(q),
            'full_acc': ov,
            'train_acc': r['train_acc'],
            'test_acc': r['test_acc'],
            'bet': 'Game Total Over at -110',
        })

# Candidate 3: All-around dominant + ML
q = [g for g in games if g['fav_off'] >= 116 and g['fav_def'] <= 110 and g['net_gap'] >= 10]
if len(q) >= 10:
    ml = sum(1 for g in q if g['fav_won']) / len(q)
    r = temporal_test(
        lambda g: g['fav_off'] >= 116 and g['fav_def'] <= 110 and g['net_gap'] >= 10,
        "All-around dominant", lambda g: g['fav_won'], "ML")
    if r:
        candidates.append({
            'name': 'All-Around Dominant → Spread',
            'games': len(q),
            'full_acc': ml,
            'train_acc': r['train_acc'],
            'test_acc': r['test_acc'],
            'bet': 'Spread at -110',
        })

# Candidate 4: MDDS >= 8 + enhanced → Fav Score >= 110
for th in [8, 9, 10]:
    q = [g for g in games if g['mdds'] >= th]
    if len(q) < 10:
        continue
    ov = sum(1 for g in q if g['fav_actual_score'] >= 110) / len(q)
    r = temporal_test(lambda g, t=th: g['mdds'] >= t, f"MDDS>={th}", lambda g: g['fav_actual_score'] >= 110, "Fav>=110")
    if r:
        candidates.append({
            'name': f'MDDS>={th} → Fav Score Floor 110',
            'games': len(q),
            'full_acc': ov,
            'train_acc': r['train_acc'],
            'test_acc': r['test_acc'],
            'bet': 'Fav Team Total O109.5',
        })

print(f"\n{'Candidate':45s} | {'Games':>5s} | {'Full':>5s} | {'Train':>5s} | {'Test':>5s} | {'Bet Type':25s}")
print("-" * 120)
for c in sorted(candidates, key=lambda x: -x['test_acc']):
    full_ev, full_roi = profitability_at_minus_110(c['full_acc'])
    test_ev, test_roi = profitability_at_minus_110(c['test_acc'])
    print(f"  {c['name']:43s} | {c['games']:5d} | {100*c['full_acc']:4.1f}% | {100*c['train_acc']:4.1f}% | "
          f"{100*c['test_acc']:4.1f}% | {c['bet']:25s}")


def profitability_at_minus_110(accuracy, unit_size=100):
    profit_per_win = unit_size * (100/110)
    loss_per_loss = unit_size
    ev = accuracy * profit_per_win - (1 - accuracy) * loss_per_loss
    roi = ev / unit_size * 100
    return ev, roi
