"""
Odds conversion and EV/ROI math utilities.
Completely self-contained — no external imports beyond stdlib.
"""

import math


def american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds > 0:
        return 1 + odds / 100.0
    if odds < 0:
        return 1 + 100.0 / abs(odds)
    return 2.0  # Even money


def decimal_to_american(dec):
    """Convert decimal odds to American odds."""
    if dec >= 2.0:
        return round((dec - 1) * 100)
    if dec > 1.0:
        return round(-100 / (dec - 1))
    return -100


def implied_probability(odds):
    """Get implied probability from American odds."""
    dec = american_to_decimal(odds)
    return 1.0 / dec if dec > 0 else 1.0


def expected_value(true_prob, odds):
    """Calculate expected value of a bet."""
    dec = american_to_decimal(odds)
    return true_prob * dec - 1.0


def kelly_fraction(true_prob, odds, fraction=0.25):
    """Fractional Kelly criterion for position sizing."""
    dec = american_to_decimal(odds)
    if dec <= 1:
        return 0
    full_kelly = (true_prob * dec - 1) / (dec - 1)
    return max(0, full_kelly * fraction)


def pnl_for_bet(hit, odds, wager):
    """Calculate P&L for a single bet."""
    if hit:
        return round((american_to_decimal(odds) - 1) * wager)
    return -wager


def parlay_decimal_odds(legs_odds):
    """Calculate combined decimal odds for a parlay."""
    result = 1.0
    for odds in legs_odds:
        result *= american_to_decimal(odds)
    return result


def beta_ppf_approx(alpha, beta_param, p):
    """
    Approximate inverse CDF of Beta distribution using normal approximation.
    Used for Bayesian credible intervals without scipy dependency.
    """
    if alpha <= 0 or beta_param <= 0:
        return 0.5
    mean = alpha / (alpha + beta_param)
    variance = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
    std = math.sqrt(variance) if variance > 0 else 0.001
    if p <= 0 or p >= 1:
        return mean
    if p < 0.5:
        t = math.sqrt(-2 * math.log(p))
        z = -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
              (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))
    else:
        t = math.sqrt(-2 * math.log(1 - p))
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / \
            (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
    result = mean + z * std
    return max(0.0, min(1.0, result))


def shannon_entropy(values, n_bins=8):
    """Compute normalized Shannon entropy of a distribution."""
    if len(values) < 3:
        return 1.0
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return 0.0
    bin_width = (max_v - min_v) / n_bins
    counts = [0] * n_bins
    for v in values:
        idx = min(int((v - min_v) / bin_width), n_bins - 1)
        counts[idx] += 1
    n = len(values)
    entropy = 0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log2(p)
    max_entropy = math.log2(n_bins)
    return entropy / max_entropy if max_entropy > 0 else 1.0
