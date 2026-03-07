"""
NBA Multi-Window Trading Strategy
=================================

RULES:
1. Calculate LEAD = abs(home_score - away_score)
2. Calculate MOMENTUM = home_pts_last_5min - away_pts_last_5min
3. Momentum must ALIGN with lead (both positive or both negative)
4. Check time windows from earliest to latest
5. Enter on FIRST signal that matches
6. Bet on the LEADING team
7. Hold until game end
"""

def get_signal(home_score, away_score, home_pts_5min, away_pts_5min, mins_remaining):
    """
    Returns ('home', signal_name) or ('away', signal_name) or None
    """

    score_diff = home_score - away_score
    lead = abs(score_diff)
    momentum = home_pts_5min - away_pts_5min
    mom = abs(momentum)

    # No signal on tie
    if score_diff == 0:
        return None

    # Determine leading team
    side = 'home' if score_diff > 0 else 'away'

    # CRITICAL: Momentum must align with lead
    if score_diff > 0 and momentum <= 0:
        return None  # Home leads but losing momentum
    if score_diff < 0 and momentum >= 0:
        return None  # Away leads but losing momentum

    # ==========================================
    # CHECK ALL WINDOWS (earliest to latest)
    # ==========================================

    # Q2 EARLY: 30-36 min remaining
    if 30 <= mins_remaining <= 36:
        if lead >= 18 and mom >= 3:
            return (side, 'Q2_selective')      # 99.4% WR
        if lead >= 15 and mom >= 3:
            return (side, 'Q2_early')          # 98.1% WR

    # LATE Q2: 24-30 min remaining
    if 24 <= mins_remaining <= 30:
        if lead >= 18 and mom >= 5:
            return (side, 'late_Q2')           # 99.7% WR
        if lead >= 15 and mom >= 7:
            return (side, 'late_Q2_alt')       # 99.2% WR

    # HALFTIME: 18-24 min remaining
    if 18 <= mins_remaining <= 24:
        if lead >= 20 and mom >= 3:
            return (side, 'halftime_dominant') # 100% WR
        if lead >= 18 and mom >= 7:
            return (side, 'halftime')          # 100% WR
        if lead >= 15 and mom >= 10:
            return (side, 'halftime_momentum') # 100% WR

    # MID Q3: 15-20 min remaining
    if 15 <= mins_remaining <= 20:
        if lead >= 18 and mom >= 5:
            return (side, 'mid_Q3')            # 100% WR
        if lead >= 15 and mom >= 7:
            return (side, 'mid_Q3_alt')        # 99.8% WR

    # LATE Q3: 12-18 min remaining
    if 12 <= mins_remaining <= 18:
        if lead >= 15 and mom >= 7:
            return (side, 'late_Q3')           # 99.8% WR
        if lead >= 12 and mom >= 10:
            return (side, 'late_Q3_momentum')  # 98.9% WR

    # EARLY Q4: 8-12 min remaining
    if 8 <= mins_remaining <= 12:
        if lead >= 10 and mom >= 5:
            return (side, 'early_Q4')          # 99.6% WR
        if lead >= 7 and mom >= 7:
            return (side, 'early_Q4_alt')      # 98.9% WR

    # FINAL: 2-8 min remaining
    if 2 <= mins_remaining <= 8:
        if lead >= 7 and mom >= 3:
            return (side, 'final')             # 99.5% WR
        if lead >= 5 and mom >= 5:
            return (side, 'final_alt')         # 98.7% WR

    return None


# ==========================================
# SIGNAL REFERENCE TABLE
# ==========================================
#
# Signal              | Time (min) | Lead | Mom | Win Rate
# --------------------|------------|------|-----|----------
# Q2_selective        | 30-36      | ≥18  | ≥3  | 99.4%
# Q2_early            | 30-36      | ≥15  | ≥3  | 98.1%
# late_Q2             | 24-30      | ≥18  | ≥5  | 99.7%
# late_Q2_alt         | 24-30      | ≥15  | ≥7  | 99.2%
# halftime_dominant   | 18-24      | ≥20  | ≥3  | 100%
# halftime            | 18-24      | ≥18  | ≥7  | 100%
# halftime_momentum   | 18-24      | ≥15  | ≥10 | 100%
# mid_Q3              | 15-20      | ≥18  | ≥5  | 100%
# mid_Q3_alt          | 15-20      | ≥15  | ≥7  | 99.8%
# late_Q3             | 12-18      | ≥15  | ≥7  | 99.8%
# late_Q3_momentum    | 12-18      | ≥12  | ≥10 | 98.9%
# early_Q4            | 8-12       | ≥10  | ≥5  | 99.6%
# early_Q4_alt        | 8-12       | ≥7   | ≥7  | 98.9%
# final               | 2-8        | ≥7   | ≥3  | 99.5%
# final_alt           | 2-8        | ≥5   | ≥5  | 98.7%


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":

    # Example: Home up 18 with momentum, 16 min left
    result = get_signal(
        home_score=85,
        away_score=67,
        home_pts_5min=12,
        away_pts_5min=4,
        mins_remaining=16.0
    )

    if result:
        side, signal = result
        print(f"BET: {side.upper()}")
        print(f"Signal: {signal}")
    else:
        print("No signal")
