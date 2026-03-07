"""
NBA Multi-Window Trading Strategy - CONSERVATIVE VERSION
=========================================================

Based on real-world loss feedback:
- Lead=5, Mom=+8 in final minutes → LOST by 9

REVISED RULES (more conservative thresholds):
- Removed low-threshold signals that failed in practice
- Increased minimum lead requirements
- Require BOTH larger lead AND stronger momentum
"""

def get_signal(home_score, away_score, home_pts_5min, away_pts_5min, mins_remaining):
    """
    Conservative signal checker.
    Returns ('home', signal_name) or ('away', signal_name) or None
    """

    score_diff = home_score - away_score
    lead = abs(score_diff)
    momentum = home_pts_5min - away_pts_5min
    mom = abs(momentum)

    if score_diff == 0:
        return None

    side = 'home' if score_diff > 0 else 'away'

    # CRITICAL: Momentum must align with lead
    if score_diff > 0 and momentum <= 0:
        return None
    if score_diff < 0 and momentum >= 0:
        return None

    # ==========================================
    # CONSERVATIVE THRESHOLDS ONLY
    # (removed lower-confidence signals)
    # ==========================================

    # Q2: 30-36 min remaining - NEED BIG LEAD
    if 30 <= mins_remaining <= 36:
        if lead >= 20 and mom >= 5:
            return (side, 'Q2_blowout')

    # LATE Q2: 24-30 min remaining
    if 24 <= mins_remaining <= 30:
        if lead >= 20 and mom >= 5:
            return (side, 'late_Q2')

    # HALFTIME: 18-24 min remaining
    if 18 <= mins_remaining <= 24:
        if lead >= 20 and mom >= 5:
            return (side, 'halftime')

    # MID Q3: 15-20 min remaining
    if 15 <= mins_remaining <= 20:
        if lead >= 18 and mom >= 7:
            return (side, 'mid_Q3')

    # LATE Q3: 12-18 min remaining
    if 12 <= mins_remaining <= 18:
        if lead >= 15 and mom >= 7:
            return (side, 'late_Q3')

    # EARLY Q4: 8-12 min remaining
    if 8 <= mins_remaining <= 12:
        if lead >= 12 and mom >= 5:
            return (side, 'early_Q4')

    # FINAL: 2-8 min remaining
    # REMOVED: Lead>=5 (too risky as proven)
    # REMOVED: Lead>=7 with Mom>=3 (borderline)
    if 2 <= mins_remaining <= 8:
        if lead >= 10 and mom >= 5:
            return (side, 'final')

    return None


# ==========================================
# CONSERVATIVE SIGNAL REFERENCE
# ==========================================
#
# Signal       | Time (min) | Lead | Mom | Notes
# -------------|------------|------|-----|------------------
# Q2_blowout   | 30-36      | ≥20  | ≥5  | Only massive leads
# late_Q2      | 24-30      | ≥20  | ≥5  | Only massive leads
# halftime     | 18-24      | ≥20  | ≥5  | Only massive leads
# mid_Q3       | 15-20      | ≥18  | ≥7  | Tightened momentum
# late_Q3      | 12-18      | ≥15  | ≥7  | Moderate lead + momentum
# early_Q4     | 8-12       | ≥12  | ≥5  | Raised from 10
# final        | 2-8        | ≥10  | ≥5  | Raised from 5/7
#
# REMOVED SIGNALS (too risky):
# - Q2_early (Lead>=15, Mom>=3) - not enough momentum confirmation
# - Q2_selective (Lead>=18, Mom>=3) - not enough momentum
# - late_Q2_alt (Lead>=15, Mom>=7) - lead too small for time
# - halftime_dominant (Lead>=20, Mom>=3) - needs more momentum
# - halftime_momentum (Lead>=15, Mom>=10) - lead too small
# - mid_Q3_alt (Lead>=15, Mom>=7) - borderline
# - late_Q3_momentum (Lead>=12, Mom>=10) - lead too small
# - early_Q4_alt (Lead>=7, Mom>=7) - lead way too small
# - final (Lead>=7, Mom>=3) - PROVEN FAILURE
# - final_alt (Lead>=5, Mom>=5) - PROVEN FAILURE (your screenshot)


def get_signal_aggressive(home_score, away_score, home_pts_5min, away_pts_5min, mins_remaining):
    """
    Original aggressive signals - USE WITH CAUTION.
    These have higher coverage but proven failures exist.
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

    # FINAL: 2-8 min remaining (RISKY)
    if 2 <= mins_remaining <= 8:
        if lead >= 7 and mom >= 3:
            return (side, 'final_risky')  # ~95% WR estimated
        if lead >= 5 and mom >= 5:
            return (side, 'final_very_risky')  # <95% WR - PROVEN LOSS

    return None


# ==========================================
# EXAMPLE
# ==========================================

if __name__ == "__main__":
    # Your loss case: SAS up 5 with +8 momentum
    # Using conservative rules, this would NOT trigger
    result = get_signal(
        home_score=95,  # Would be the leading team's score
        away_score=90,  # 5 point deficit
        home_pts_5min=12,
        away_pts_5min=4,  # +8 momentum
        mins_remaining=5.0
    )

    print("Your loss case (Lead=5, Mom=+8, ~5min left):")
    if result:
        print(f"  Signal: {result[1]} - BET {result[0].upper()}")
    else:
        print("  NO SIGNAL (correctly filtered out)")

    print()

    # Safe case: big lead with momentum
    result2 = get_signal(
        home_score=100,
        away_score=88,  # 12 point lead
        home_pts_5min=14,
        away_pts_5min=6,  # +8 momentum
        mins_remaining=5.0
    )

    print("Safe case (Lead=12, Mom=+8, ~5min left):")
    if result2:
        print(f"  Signal: {result2[1]} - BET {result2[0].upper()}")
    else:
        print("  NO SIGNAL")
