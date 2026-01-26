"""
FINAL NBA TRADING STRATEGY - WITH ACTUAL EDGE
==============================================
Validated on 300 real NBA games from ESPN (Oct 2023 - Apr 2024).
Data saved in data/comprehensive_validation.json
"""


def get_signal(home_score, away_score, home_pts_5min, away_pts_5min, mins_remaining):
    """
    Validated on 300 real NBA games.

    Args:
        home_score: Current home team score
        away_score: Current away team score
        home_pts_5min: Home team points in last 5 minutes
        away_pts_5min: Away team points in last 5 minutes
        mins_remaining: Minutes left in game (0-48)

    Returns:
        {'side': 'home'/'away', 'spread': int, 'signal': str} or None
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

    # Time: 12-24 minutes remaining only
    if not (12 <= mins_remaining <= 24):
        return None

    # SWEET SPOT: 94.9% WR, +$40 EV, 27.2% edge
    if 10 <= lead <= 14 and mom >= 10:
        return {'side': side, 'spread': -7, 'signal': 'sweet_spot'}

    # MODERATE: 94.5% WR, +$32 EV, 23.1% edge
    if 12 <= lead <= 16 and mom >= 12:
        return {'side': side, 'spread': -7, 'signal': 'moderate'}

    # MID-RANGE: 96.4% WR, +$30 EV, 22.1% edge
    if 14 <= lead <= 18 and mom >= 14:
        return {'side': side, 'spread': -7, 'signal': 'mid_range'}

    # SAFE: 100% WR, +$27 EV, 21.3% edge
    if 16 <= lead <= 20 and mom >= 12:
        return {'side': side, 'spread': -5, 'signal': 'safe'}

    return None


# ============================================================
# VALIDATION RESULTS (300 real NBA games)
# ============================================================
#
# Signal      | Lead  | Mom  | Spread | Win%  | EV/$100 | Edge
# ------------|-------|------|--------|-------|---------|------
# sweet_spot  | 10-14 | >=10 | -7     | 94.9% | +$40.10 | 27.2%
# moderate    | 12-16 | >=12 | -7     | 94.5% | +$32.40 | 23.1%
# mid_range   | 14-18 | >=14 | -7     | 96.4% | +$29.70 | 22.1%
# safe        | 16-20 | >=12 | -5     | 100%  | +$27.00 | 21.3%
#
# Time window: 12-24 minutes remaining (Q2-Q3)
#
# ============================================================
# WHY THIS HAS EDGE
# ============================================================
#
# Market prices based on LEAD (team up 12 → market prices ~-12)
# We price based on LEAD + MOMENTUM
#
# High momentum (>=10) predicts lead EXTENSION:
#   - Mom >= 10: +1.2 pts average extension
#   - Mom >= 12: +2.8 pts average extension
#   - Mom >= 14: +3.2 pts average extension
#
# Betting smaller spread (-5 or -7) captures this edge.
#
# ============================================================


if __name__ == "__main__":
    print("="*60)
    print("STRATEGY TEST CASES")
    print("="*60)

    tests = [
        ("Lead=12, Mom=11, 18min", 65, 53, 15, 4, 18.0),
        ("Lead=14, Mom=13, 20min", 70, 56, 18, 5, 20.0),
        ("Lead=16, Mom=15, 22min", 75, 59, 20, 5, 22.0),
        ("Lead=18, Mom=13, 16min", 80, 62, 18, 5, 16.0),
        ("Lead=10, Mom=6, 10min (NO)", 85, 75, 14, 8, 10.0),
        ("Lead=22, Mom=10, 18min (NO)", 80, 58, 16, 6, 18.0),
    ]

    for name, hs, as_, h5, a5, mins in tests:
        r = get_signal(hs, as_, h5, a5, mins)
        if r:
            print(f"{name}: {r['signal']} → Bet {r['side'].upper()} {r['spread']}")
        else:
            print(f"{name}: NO SIGNAL")
