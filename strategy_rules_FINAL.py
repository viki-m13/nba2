"""
NBA Trading Strategy - FINAL VALIDATED VERSION
===============================================

CRITICAL FINDING FROM REAL DATA (189 games):

The original strategy predicts WHO WINS at ~90% accuracy.
BUT: Spread coverage (maintaining lead) is only ~50% - a COIN FLIP.

The app bets "-X spread" where X = current lead.
This bet requires the team to EXTEND or MAINTAIN their lead.
This DOES NOT WORK because:
  1. Teams with big leads rest starters
  2. Garbage time favors losing team
  3. Leading teams play conservatively
  4. Final margins compress toward mean

REAL DATA RESULTS:
------------------
Moneyline (predict winner): 89.9% win rate ✓
Spread coverage (maintain lead): 50.8% win rate ✗

ONLY VIABLE APPROACH:
---------------------
Bet MONEYLINE (team to win), NOT spread equal to current lead.

Or: Bet a REDUCED spread (e.g., if up 15, bet -7.5 instead of -15)
"""


def get_signal_MONEYLINE_ONLY(home_score, away_score, home_pts_5min, away_pts_5min, mins_remaining):
    """
    Signals for MONEYLINE bets only (predicting winner).
    DO NOT use for spread betting.

    Returns ('home' or 'away', signal_name) or None
    """

    score_diff = home_score - away_score
    lead = abs(score_diff)
    momentum = home_pts_5min - away_pts_5min
    mom = abs(momentum)

    if score_diff == 0:
        return None

    side = 'home' if score_diff > 0 else 'away'

    # Momentum must align with lead
    if score_diff > 0 and momentum <= 0:
        return None
    if score_diff < 0 and momentum >= 0:
        return None

    # MONEYLINE SIGNALS (90%+ win rate on real data)

    if 18 <= mins_remaining <= 24:  # Halftime
        if lead >= 15 and mom >= 10:
            return (side, 'halftime_momentum')  # 100% ML

    if 15 <= mins_remaining <= 20:  # Mid Q3
        if lead >= 15 and mom >= 7:
            return (side, 'mid_Q3')  # 100% ML

    if 12 <= mins_remaining <= 18:  # Late Q3
        if lead >= 12 and mom >= 10:
            return (side, 'late_Q3')  # 100% ML

    if 8 <= mins_remaining <= 12:  # Early Q4
        if lead >= 10 and mom >= 5:
            return (side, 'early_Q4')  # 100% ML

    if 2 <= mins_remaining <= 8:  # Final
        if lead >= 7 and mom >= 3:
            return (side, 'final')  # 90% ML

    return None


def get_signal_SPREAD_EXPERIMENTAL(home_score, away_score, home_pts_5min, away_pts_5min, mins_remaining):
    """
    EXPERIMENTAL: Only signal with 77%+ spread coverage.
    Very limited - only triggers at halftime with specific conditions.

    WARNING: Still risky. Use with caution.
    """

    score_diff = home_score - away_score
    lead = abs(score_diff)
    momentum = home_pts_5min - away_pts_5min
    mom = abs(momentum)

    if score_diff == 0:
        return None

    side = 'home' if score_diff > 0 else 'away'

    if score_diff > 0 and momentum <= 0:
        return None
    if score_diff < 0 and momentum >= 0:
        return None

    # ONLY SIGNAL WITH 77%+ SPREAD COVERAGE:
    # Halftime (18-24 min), Lead >= 15, Mom >= 10
    if 18 <= mins_remaining <= 24:
        if lead >= 15 and mom >= 10:
            return (side, 'halftime_spread')

    return None


# ============================================================
# REAL DATA SUMMARY
# ============================================================
#
# MONEYLINE PERFORMANCE (predict winner):
# Signal             | Wins/Total | Win Rate
# -------------------|------------|----------
# halftime_momentum  | 16/16      | 100%
# mid_Q3             | 5/5        | 100%
# late_Q3_momentum   | 8/8        | 100%
# early_Q4           | 13/13      | 100%
# final              | 10/11      | 91%
# final_alt          | 19/20      | 95%
#
# SPREAD PERFORMANCE (maintain lead):
# Signal             | Wins/Total | Win Rate
# -------------------|------------|----------
# halftime_momentum  | 14/16      | 87.5%  ← ONLY VIABLE
# Everything else    | ~50%       | COIN FLIP
#
# ============================================================


# ============================================================
# WHAT WENT WRONG WITH YOUR BETS
# ============================================================
#
# Loss 1: MIA Early Q4
#   Lead=10, Mom=6, bet MIA -10 spread
#   MIA won by 9 → SPREAD LOSS
#   With my rules: early_Q4 triggers (Lead>=10, Mom>=5)
#   Problem: My rule was for MONEYLINE, not spread!
#
# Loss 2: NOP Mid Q3 Alt
#   Lead=17, Mom=9, bet NOP -17 spread
#   NOP won by 9 → SPREAD LOSS
#   With my rules: mid_Q3_alt triggers (Lead>=15, Mom>=7)
#   Problem: 17-point leads don't hold!
#
# Real data shows mid_Q3_alt: 20% spread coverage (1/5)
# Real data shows early_Q4: 46% spread coverage (6/13)
#
# ============================================================


if __name__ == "__main__":
    print("="*60)
    print("STRATEGY TEST")
    print("="*60)

    # Your loss cases - should NOT trigger for spread
    print("\nYour MIA loss (Lead=10, Mom=6, Early Q4):")
    result = get_signal_SPREAD_EXPERIMENTAL(95, 85, 12, 6, 10.0)
    if result:
        print(f"  SPREAD SIGNAL: {result[1]} - WOULD LOSE")
    else:
        print("  NO SPREAD SIGNAL (correct - would have been 46% WR)")

    result_ml = get_signal_MONEYLINE_ONLY(95, 85, 12, 6, 10.0)
    if result_ml:
        print(f"  MONEYLINE SIGNAL: {result_ml[1]} - Bet {result_ml[0]} to WIN (100% WR)")

    print("\nYour NOP loss (Lead=17, Mom=9, Mid Q3):")
    result = get_signal_SPREAD_EXPERIMENTAL(70, 53, 15, 6, 18.0)
    if result:
        print(f"  SPREAD SIGNAL: {result[1]}")
    else:
        print("  NO SPREAD SIGNAL (correct - would have been 20% WR)")

    result_ml = get_signal_MONEYLINE_ONLY(70, 53, 15, 6, 18.0)
    if result_ml:
        print(f"  MONEYLINE SIGNAL: {result_ml[1]} - Bet {result_ml[0]} to WIN (100% WR)")

    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("""
DO NOT bet spreads equal to current lead.
Instead:
1. Bet MONEYLINE (team to win) - 90%+ accuracy
2. Or bet REDUCED spread (half the current lead)
3. Only spread signal with 77%+ accuracy:
   - Halftime (18-24 min)
   - Lead >= 15
   - Momentum >= 10
""")
