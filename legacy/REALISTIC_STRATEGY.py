"""
NBA LIVE BETTING STRATEGY - HONEST ASSESSMENT (v3)
=====================================================

VALIDATED ON: 2,310 real NBA games across 2 full seasons (2021-22, 2022-23)
DATA SOURCE: ESPN play-by-play API, cached locally
METHODOLOGY: Scoring-play-only triggers, one bet per game, no inflated counts

VERDICT: THIS STRATEGY IS NOT PROFITABLE.

The original 100% win rate on 15-27 games (2023-24 season, 4 months) was a
small-sample anomaly. When tested on 2,310 games across 2 full seasons,
EVERY strategy configuration produces NEGATIVE ROI.

RESULTS (2,310 games, 2 seasons, scoring-play-only triggers):

  PRIME (L10-14, M12+, 18-24min):
    Games: 135 | WR: 80.0% (108W/27L) | ROI: -11.68% | P&L: -15.77u
    Market implied: 86.6% --> actual WR is 6.6% BELOW market
    Cross-season: 2021-22: 80.0% WR, -11.0% ROI
                  2022-23: 80.0% WR, -12.4% ROI
    Spread: 30.4% cover rate (needs >52.4% for profit) --> catastrophic

  STANDARD (L10-16, M12+, 18-24min):
    Games: 174 | WR: 78.2% (136W/38L) | ROI: -14.75% | P&L: -25.67u

  BROAD (L10-19, M12+, 18-24min):
    Games: 222 | WR: 81.1% (180W/42L) | ROI: -13.17% | P&L: -29.24u

  GRID SEARCH (hundreds of combinations tested):
    No combination with 20+ games is profitable.
    Best: L12-14 M16+ T15-18 at 100% WR -- but only 8 games (small sample).

WHY IT FAILS:
  1. Teams with big leads + momentum already have high market-implied WP (~87%)
  2. Actual WR (~80%) is BELOW market implied -- momentum doesn't add edge
  3. The vig on heavy favorites (-800 to -4000) is punishing
  4. Leads compress by ~4 points on average (regression to mean)
  5. Even at 2% vig (unrealistically low), strategies remain unprofitable

WHAT THE ORIGINAL ANALYSIS GOT WRONG:
  1. 15-27 game sample from one 4-month window was statistically meaningless
  2. The "100% WR" was expected variance on heavy favorites (~87% implied)
  3. Cross-validated "in both halves" still only meant 7-8 games per half
  4. Fake reduced spreads (-5 when live spread is -38) inflated apparent edge
  5. 5,366 "signals" were 156 games with multiple thresholds per game moment

THE MARKET IS EFFICIENT:
  NBA live betting markets correctly price momentum-aligned leads.
  There is no edge to exploit in these conditions.
"""

import math
from typing import Optional, Dict


# =============================================================================
# MARKET PROBABILITY MODEL (calibrated to NBA historical data)
# =============================================================================

def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    if x > 6: return 0.9999
    if x < -6: return 0.0001
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def estimate_market_win_prob(lead: int, mins_remaining: float) -> float:
    """
    Estimate what the market prices as the leading team's win probability.

    Calibrated to NBA historical benchmarks (sigma=2.6 pts/sqrt(min)).
    """
    if lead <= 0 or mins_remaining <= 0:
        return 0.5
    mins_remaining = max(mins_remaining, 0.5)
    SIGMA = 2.6
    z = lead / (SIGMA * math.sqrt(mins_remaining))
    return max(0.51, min(0.998, _normal_cdf(z)))


def estimate_live_ml_odds(lead: int, mins_remaining: float, vig: float = 0.045) -> float:
    """Estimate live moneyline odds (American format) with standard vig."""
    prob = estimate_market_win_prob(lead, mins_remaining)
    market_prob = min(0.995, prob + vig * prob)

    if market_prob >= 0.5:
        return -(market_prob / (1 - market_prob)) * 100
    else:
        return ((1 - market_prob) / market_prob) * 100


# =============================================================================
# STRATEGY (kept for reference - NOT PROFITABLE)
# =============================================================================

MIN_LEAD = 10
MAX_LEAD = 19
MIN_MOMENTUM = 12
TIME_MIN = 18
TIME_MAX = 24


def get_signal(
    home_score: int,
    away_score: int,
    home_pts_5min: int,
    away_pts_5min: int,
    mins_remaining: float
) -> Optional[Dict]:
    """
    Generate a live betting signal.

    WARNING: This strategy was validated on 2,310 games across 2 seasons
    and is NOT profitable. Every configuration tested has negative ROI.
    The ~80% ML win rate is below the ~87% market-implied probability.

    Kept for reference and further research only.
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

    # Lead bounds
    if lead < MIN_LEAD or lead > MAX_LEAD:
        return None

    # Time window
    if mins_remaining < TIME_MIN or mins_remaining > TIME_MAX:
        return None

    # Minimum momentum
    if mom < MIN_MOMENTUM:
        return None

    # Calculate market pricing
    market_prob = estimate_market_win_prob(lead, mins_remaining)
    market_odds = estimate_live_ml_odds(lead, mins_remaining)

    # Tier classification (for informational purposes)
    if lead <= 14:
        tier = 'prime'
        confidence = 'low'  # 80% WR vs 87% market = negative edge
    elif lead <= 16:
        tier = 'standard'
        confidence = 'low'
    else:
        tier = 'wide'
        confidence = 'low'

    return {
        'side': side,
        'tier': tier,
        'confidence': confidence,
        'bet_type': 'moneyline',
        'spread_also_viable': False,  # 30% cover rate - not viable
        'lead': lead,
        'momentum': mom,
        'mins_remaining': mins_remaining,
        'market_win_prob': round(market_prob * 100, 1),
        'estimated_odds': round(market_odds),
        'warning': 'NOT PROFITABLE - 80% WR on 135 games, -11.7% ROI, negative edge vs market',
    }


# =============================================================================
# VALIDATION SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NBA LIVE BETTING STRATEGY - HONEST ASSESSMENT (v3)")
    print("=" * 70)

    print("""
VALIDATED ON: 2,310 real NBA games, 2 full seasons (2021-22, 2022-23)
METHODOLOGY:  Scoring-play-only triggers, one bet per game

RESULT: NOT PROFITABLE. Every configuration has negative ROI.

  PRIME (L10-14, M12+, 18-24min):
    135 games | 80.0% WR | -11.68% ROI | -15.77 units
    Market implied: 86.6% (actual WR is 6.6% BELOW)
    Both seasons: 80% WR, ~-12% ROI (consistent)

  STANDARD (L10-16, M12+, 18-24min):
    174 games | 78.2% WR | -14.75% ROI | -25.67 units

  BROAD (L10-19, M12+, 18-24min):
    222 games | 81.1% WR | -13.17% ROI | -29.24 units

  Spread: 30% cover rate across the board (needs >52.4%)
  Lead extension: -4 points average (leads SHRINK)

  Grid search: No profitable combination with 20+ games

CONCLUSION: The NBA live market is efficient. Momentum-aligned leads
are already correctly priced. There is no edge to exploit.
""")

    # Run test signals
    tests = [
        ("Lead 12, Mom 14, halftime",
         70, 58, 18, 4, 20.0),
        ("Lead 15, Mom 12, halftime",
         72, 57, 16, 4, 21.0),
        ("Lead 11, Mom 10, halftime (BLOCKED - mom < 12)",
         65, 54, 14, 4, 19.0),
    ]

    for desc, hs, aws, h5, a5, mins in tests:
        print(f"\n{desc}")
        signal = get_signal(hs, aws, h5, a5, mins)
        if signal:
            print(f"  SIGNAL: {signal['tier'].upper()} tier")
            print(f"  Bet: {signal['side'].upper()} MONEYLINE at ~{signal['estimated_odds']}")
            print(f"  Market prob: {signal['market_win_prob']}%")
            print(f"  WARNING: {signal['warning']}")
        else:
            print(f"  NO SIGNAL")

    print(f"\n{'='*70}")
