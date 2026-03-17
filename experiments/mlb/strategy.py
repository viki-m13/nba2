"""
MLB ROI-Maximized Strategy
===========================
Self-contained player prop over strategy for MLB, optimized for ROI.

MLB-specific features (adapted from production MLB engine):
- Poisson-Bayesian Fusion (PBF) for discrete stat modeling
- At-Bat Volume Confidence (ABVC)
- Consecutive Game Streak Momentum (CGSM)
- Venue Performance Differential (VPD)
- Opponent-Adjusted Lambda (OAL)

MLB considerations vs NBA:
- Stat categories: hits (h), total bases (tb), runs (r), RBIs (rbi), home runs (hr)
- Discrete stats (0-4 hits typical) need Poisson not Normal modeling
- No minutes filter — uses at-bats (ab) as activity indicator
- Higher variance requires wider confidence bands
- Uses REAL historical odds from The Odds API (no synthetic odds)
- Walk-forward only, Bayesian shrinkage, no look-ahead bias
"""

import math
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.odds_math import (
    american_to_decimal, implied_probability, expected_value,
    kelly_fraction, pnl_for_bet, beta_ppf_approx, shannon_entropy,
    parlay_decimal_odds, decimal_to_american,
)

# =========================================================================
# DEFAULT CONFIG — tuned for MLB ROI maximization
# =========================================================================

MLB_ROI_CONFIG = {
    # Data requirements
    'MIN_GAMES': 10,
    'MIN_AB': 2,
    'WARM_UP_GAMES': 10,

    # Gravitational Floor (adapted for discrete baseball stats)
    'GFT_WINDOWS': [3, 5, 10],
    'GFT_DECAY': 0.90,
    'GFT_GRAVITY': 0.40,
    'GFT_MIN_CLEARANCE': 0.05,
    'GFT_MAX_SPREAD': 3.0,

    # Bayesian Edge
    'BEQ_PRIOR_ALPHA': 1.0,
    'BEQ_PRIOR_BETA': 1.0,
    'BEQ_CI_LEVEL': 0.85,
    'BEQ_MIN_EDGE': 0.03,

    # Stability
    'ESI_BINS': 5,
    'ESI_MAX_ENTROPY': 0.85,
    'ESI_TREND_WEIGHT': 0.25,

    # Poisson-Bayesian Fusion (MLB-specific: discrete stat modeling)
    'PBF_DECAY': 0.92,
    'PBF_MIN_LAMBDA': 0.3,
    'PBF_WEIGHT': 0.12,
    'PBF_MIN_MARGIN': 0.03,

    # At-Bat Volume Confidence (MLB-specific)
    'ABVC_MIN_AVG': 3.0,
    'ABVC_ELITE_AVG': 5.0,
    'ABVC_WEIGHT': 0.08,
    'ABVC_MIN_SCORE': 0.30,

    # Consecutive Game Streak Momentum (MLB-specific)
    'CGSM_MIN_STREAK': 2,
    'CGSM_LOOKBACK': 10,
    'CGSM_WEIGHT': 0.08,

    # Venue Performance Differential (MLB-specific: park factors)
    'VPD_ENABLED': True,
    'VPD_MIN_GAMES': 5,
    'VPD_WEIGHT': 0.05,
    'VPD_MAX_BOOST': 0.10,

    # Opponent-Adjusted Lambda (MLB-specific)
    'OAL_ENABLED': True,
    'OAL_WEIGHT': 0.05,
    'OAL_MAX_ADJ': 0.25,

    # Quality gates — relaxed for plus-money
    'GATE_MIN_GFT': 0.10,
    'GATE_MIN_ESI': 0.08,
    'GATE_MIN_HIT_RATE': 0.35,
    'GATE_MIN_COMBINED': 0.22,

    # ROI gate — require strong EV
    'MIN_EV': 0.08,

    # Bet selection
    'SINGLE_MIN_SCORE': 0.35,
    'PARLAY_LEG_MIN_SCORE': 0.28,

    # Parlay — the ROI engine for MLB
    'PARLAY_MIN_LEGS': 2,
    'PARLAY_MAX_LEGS': 4,
    'PARLAY_MAX_CORR': 0.40,
    'PARLAY_MIN_EDGE': 0.05,

    # Odds filter — TARGET PLUS-MONEY for max ROI
    'MIN_ODDS': -200,
    'MAX_ODDS': 600,

    # Bankroll — aggressive Kelly
    'UNIT_SIZE': 100,
    'KELLY_FRACTION': 0.30,
    'MAX_DAILY_BETS': 8,
}

# Parameter ranges for optimization
MLB_PARAM_RANGES = {
    'GFT_DECAY': (0.80, 0.97),
    'GFT_GRAVITY': (0.10, 0.55),
    'GFT_MIN_CLEARANCE': (0.01, 1.5),
    'GFT_MAX_SPREAD': (1.0, 5.0),
    'BEQ_CI_LEVEL': (0.65, 0.95),
    'BEQ_MIN_EDGE': (0.01, 0.12),
    'ESI_MAX_ENTROPY': (0.50, 0.95),
    'ESI_TREND_WEIGHT': (0.05, 0.45),
    'PBF_DECAY': (0.80, 0.97),
    'PBF_MIN_LAMBDA': (0.1, 0.6),
    'PBF_WEIGHT': (0.03, 0.20),
    'PBF_MIN_MARGIN': (0.01, 0.08),
    'ABVC_MIN_AVG': (2.0, 4.5),
    'ABVC_ELITE_AVG': (4.0, 6.0),
    'ABVC_WEIGHT': (0.02, 0.15),
    'ABVC_MIN_SCORE': (0.15, 0.50),
    'CGSM_MIN_STREAK': (1, 4),
    'CGSM_LOOKBACK': (5, 15),
    'CGSM_WEIGHT': (0.02, 0.15),
    'GATE_MIN_GFT': (0.02, 0.50),
    'GATE_MIN_ESI': (0.02, 0.35),
    'GATE_MIN_HIT_RATE': (0.20, 0.70),
    'GATE_MIN_COMBINED': (0.10, 0.55),
    'MIN_EV': (0.03, 0.25),
    'SINGLE_MIN_SCORE': (0.20, 0.65),
    'PARLAY_LEG_MIN_SCORE': (0.15, 0.50),
    'PARLAY_MAX_CORR': (0.15, 0.55),
    'PARLAY_MIN_EDGE': (0.02, 0.18),
    'MIN_ODDS': (-300, -100),
    'MAX_ODDS': (200, 1000),
    'MIN_GAMES': (6, 16),
    'MIN_AB': (1, 3),
}


# =========================================================================
# PLAYER MODEL — MLB-specific with venue/opponent tracking
# =========================================================================

class MLBPlayerModel:
    """Track MLB batter stat histories with venue and opponent context."""

    def __init__(self, max_history=60):
        self.profiles = {}
        self.max_history = max_history
        self.team_stats_allowed = defaultdict(lambda: defaultdict(list))

    def update(self, name, stats, date, team, home_away, opponent):
        if name not in self.profiles:
            self.profiles[name] = {
                'games': [], 'team': '',
                'home_games': [], 'away_games': [],
            }

        ab = _safe_int(stats.get('ab', 0))
        h = _safe_int(stats.get('h', 0))
        r = _safe_int(stats.get('r', 0))
        rbi = _safe_int(stats.get('rbi', 0))
        hr = _safe_int(stats.get('hr', 0))
        bb = _safe_int(stats.get('bb', 0))
        so = _safe_int(stats.get('so', 0))
        sb = _safe_int(stats.get('sb', 0))
        tb = _safe_int(stats.get('tb', 0))

        game = {
            'date': date, 'team': team, 'opponent': opponent,
            'home_away': home_away,
            'ab': ab, 'h': h, 'r': r, 'rbi': rbi, 'hr': hr,
            'bb': bb, 'so': so, 'sb': sb, 'tb': tb,
            'h_r_rbi': h + r + rbi,
        }

        self.profiles[name]['games'].append(game)
        self.profiles[name]['team'] = team

        # Track home/away games for VPD
        if home_away == 'home':
            self.profiles[name]['home_games'].append(game)
        else:
            self.profiles[name]['away_games'].append(game)

        # Trim
        if len(self.profiles[name]['games']) > self.max_history:
            self.profiles[name]['games'] = self.profiles[name]['games'][-self.max_history:]
        for key in ['home_games', 'away_games']:
            if len(self.profiles[name][key]) > self.max_history // 2:
                self.profiles[name][key] = self.profiles[name][key][-(self.max_history // 2):]

        # Track team defense stats (for OAL)
        self.team_stats_allowed[opponent]['h'].append(h)
        self.team_stats_allowed[opponent]['tb'].append(tb)
        self.team_stats_allowed[opponent]['r'].append(r)

    def get_values(self, name, stat, window=None, min_ab=2):
        p = self.profiles.get(name)
        if not p:
            return None
        games = [g for g in p['games'] if g.get('ab', 0) >= min_ab]
        if window:
            games = games[-window:]
        if not games:
            return None
        return [g.get(stat, 0) for g in games]

    def game_count(self, name):
        p = self.profiles.get(name)
        return len(p['games']) if p else 0

    def avg_ab(self, name, window=5):
        p = self.profiles.get(name)
        if not p:
            return 0
        games = p['games'][-window:]
        return sum(g.get('ab', 0) for g in games) / len(games) if games else 0

    def get_team(self, name):
        p = self.profiles.get(name)
        return p['team'] if p else None

    def get_venue_stats(self, name, stat, venue):
        """Get stat values for home or away games."""
        p = self.profiles.get(name)
        if not p:
            return None
        key = 'home_games' if venue == 'home' else 'away_games'
        games = [g for g in p.get(key, []) if g.get('ab', 0) >= 2]
        if not games:
            return None
        return [g.get(stat, 0) for g in games]

    def get_opp_defense_avg(self, opponent, stat, window=20):
        """Get opponent team's average stats allowed."""
        vals = self.team_stats_allowed.get(opponent, {}).get(stat, [])
        if not vals:
            return None
        recent = vals[-window:]
        return sum(recent) / len(recent)


def _safe_int(val):
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return 0
    return 0


# =========================================================================
# CORE SIGNAL COMPONENTS
# =========================================================================

def compute_gft(model, name, stat, line, config):
    """Gravitational Floor Theory adapted for discrete MLB stats."""
    windows = config['GFT_WINDOWS']
    decay = config['GFT_DECAY']
    gravity = config['GFT_GRAVITY']

    floors, clearances = [], []
    for w in windows:
        values = model.get_values(name, stat, w, min_ab=config['MIN_AB'])
        if values is None or len(values) < min(w, 3):
            return None

        n = len(values)
        weights = [decay ** (n - 1 - i) for i in range(n)]
        tw = sum(weights)
        wmean = sum(v * wt for v, wt in zip(values, weights)) / tw

        sorted_vw = sorted(zip(values, weights), key=lambda x: x[0])
        cum = 0
        p10 = sorted_vw[0][0]
        for val, wt in sorted_vw:
            cum += wt
            if cum / tw >= 0.10:
                p10 = val
                break

        floor = p10 + gravity * (wmean - p10)
        clearance = floor - line
        if clearance < config['GFT_MIN_CLEARANCE']:
            return None
        floors.append(floor)
        clearances.append(clearance)

    spread = max(floors) - min(floors)
    if spread > config['GFT_MAX_SPREAD']:
        return None

    convergence = max(0, 1 - spread / config['GFT_MAX_SPREAD'])
    depth = min(1.0, sum(clearances) / len(clearances) / 4.0)
    score = convergence * 0.5 + depth * 0.5
    return {'score': score, 'convergence': convergence, 'depth': depth}


def compute_beq(model, name, stat, line, market_odds, config):
    """Bayesian Edge Quantification for MLB."""
    values = model.get_values(name, stat, 20, min_ab=config['MIN_AB'])
    if values is None or len(values) < 8:
        return None

    hits = sum(1 for v in values if v > line)
    misses = len(values) - hits
    alpha = config['BEQ_PRIOR_ALPHA'] + hits
    beta_p = config['BEQ_PRIOR_BETA'] + misses
    mean_prob = alpha / (alpha + beta_p)
    ci = 1 - config['BEQ_CI_LEVEL']
    lower = beta_ppf_approx(alpha, beta_p, ci)
    mkt_prob = implied_probability(market_odds)
    edge = lower - mkt_prob
    hit_rate = hits / len(values)

    ext = model.get_values(name, stat, 30, min_ab=config['MIN_AB'])
    ext_hr = None
    if ext and len(ext) >= 15:
        ext_hr = sum(1 for v in ext if v > line) / len(ext)

    return {
        'mean_prob': mean_prob, 'lower_bound': lower,
        'market_prob': mkt_prob, 'edge': edge,
        'hit_rate': hit_rate, 'ext_hit_rate': ext_hr,
    }


def compute_esi(model, name, stat, config):
    """Entropic Stability Index for MLB."""
    values = model.get_values(name, stat, 20, min_ab=config['MIN_AB'])
    if values is None or len(values) < 8:
        return None

    dist_ent = shannon_entropy(values, config['ESI_BINS'])
    mid = len(values) // 2
    if mid == 0:
        return None
    m1 = sum(values[:mid]) / mid
    m2 = sum(values[mid:]) / (len(values) - mid)
    overall = sum(values) / len(values)
    if overall == 0:
        overall = 0.1  # Avoid div/0 for rare stats like HR

    trend_shift = abs(m2 - m1) / overall
    trend_stab = max(0, 1 - trend_shift / 0.5)

    sorted_v = sorted(values)
    p15 = sorted_v[max(0, int(len(sorted_v) * 0.15))]
    recent = values[-5:] if len(values) >= 5 else values
    tail_events = sum(1 for v in recent if v <= p15)
    tail_risk = 1 - tail_events / len(recent)

    ent_score = max(0, 1 - dist_ent / config['ESI_MAX_ENTROPY'])
    tw = config['ESI_TREND_WEIGHT']
    stability = ((1 - tw) * ent_score + tw * trend_stab) * tail_risk
    return {'stability': stability, 'entropy': dist_ent, 'trend_stability': trend_stab}


# =========================================================================
# MLB-SPECIFIC SIGNAL COMPONENTS
# =========================================================================

def compute_pbf(model, name, stat, line, market_odds, config):
    """
    Poisson-Bayesian Fusion (PBF) — MLB-specific discrete stat modeling.
    Models hit/run counts as Poisson process, computes P(X > line) using
    recency-weighted lambda, compares to market implied probability.
    """
    values = model.get_values(name, stat, 15, min_ab=config['MIN_AB'])
    if values is None or len(values) < 6:
        return None

    # Recency-weighted lambda (Poisson rate parameter)
    decay = config['PBF_DECAY']
    n = len(values)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    tw = sum(weights)
    lam = sum(v * w for v, w in zip(values, weights)) / tw

    if lam < config['PBF_MIN_LAMBDA']:
        return None

    # Poisson CDF: P(X <= k) = sum(e^-lam * lam^i / i!) for i=0..k
    k = int(line)
    cdf = 0
    for i in range(k + 1):
        if lam > 0:
            log_pmf = -lam + i * math.log(lam) - sum(math.log(j) for j in range(1, i + 1))
            cdf += math.exp(log_pmf)
        elif i == 0:
            cdf = 1.0

    poisson_prob = 1 - cdf  # P(X > line)
    mkt_prob = implied_probability(market_odds)
    margin = poisson_prob - mkt_prob

    # Lambda stability
    mean_lam = sum(values) / len(values)
    std_lam = math.sqrt(sum((v - mean_lam) ** 2 for v in values) / len(values))
    stability = max(0, 1 - std_lam / max(0.1, mean_lam))

    return {
        'poisson_prob': poisson_prob,
        'margin': margin,
        'lambda': lam,
        'stability': stability,
    }


def compute_cgsm(model, name, stat, line, config):
    """
    Consecutive Game Streak Momentum (CGSM) — Track hitting streaks.
    Rewards players on active streaks of clearing the line.
    """
    values = model.get_values(name, stat, config['CGSM_LOOKBACK'], min_ab=config['MIN_AB'])
    if values is None or len(values) < 3:
        return None

    # Count current consecutive streak (from most recent)
    streak = 0
    for v in reversed(values):
        if v > line:
            streak += 1
        else:
            break

    if streak < config['CGSM_MIN_STREAK']:
        return None

    # Streak score: normalized by lookback window
    window_hr = sum(1 for v in values if v > line) / len(values)
    streak_score = min(1.0, streak / config['CGSM_LOOKBACK']) * 0.6 + window_hr * 0.4

    return {'streak': streak, 'score': streak_score, 'window_hit_rate': window_hr}


def compute_abvc(model, name, config):
    """
    At-Bat Volume Confidence (ABVC) — MLB-specific opportunity filter.
    Players with more at-bats have more opportunities, making their
    props more predictable and their hit rates more reliable.
    """
    avg_ab = model.avg_ab(name, window=10)
    if avg_ab < config['ABVC_MIN_AVG']:
        return None

    # Volume score
    vol_range = config['ABVC_ELITE_AVG'] - config['ABVC_MIN_AVG']
    if vol_range <= 0:
        return None
    volume = min(1.0, (avg_ab - config['ABVC_MIN_AVG']) / vol_range)

    # AB consistency (low CV = good)
    p = model.profiles.get(name)
    if not p:
        return None
    recent = [g.get('ab', 0) for g in p['games'][-10:]]
    if not recent:
        return None
    mean_ab = sum(recent) / len(recent)
    if mean_ab == 0:
        return None
    std_ab = math.sqrt(sum((a - mean_ab) ** 2 for a in recent) / len(recent))
    consistency = max(0, 1 - std_ab / mean_ab)

    score = 0.6 * volume + 0.4 * consistency
    return {'score': score, 'avg_ab': avg_ab, 'consistency': consistency}


def compute_vpd(model, name, stat, home_away, config):
    """
    Venue Performance Differential (VPD) — Park factor adjustment.
    Boosts signal when player historically performs better at current venue.
    One-directional: only provides boost, never penalizes.
    """
    if not config.get('VPD_ENABLED', False):
        return 0

    home_vals = model.get_venue_stats(name, stat, 'home')
    away_vals = model.get_venue_stats(name, stat, 'away')

    if (not home_vals or len(home_vals) < config['VPD_MIN_GAMES'] or
            not away_vals or len(away_vals) < config['VPD_MIN_GAMES']):
        return 0

    home_mean = sum(home_vals) / len(home_vals)
    away_mean = sum(away_vals) / len(away_vals)

    if home_away == 'home' and home_mean > away_mean:
        diff = (home_mean - away_mean) / max(0.1, away_mean)
        return min(config['VPD_MAX_BOOST'], diff * config['VPD_WEIGHT'])
    elif home_away == 'away' and away_mean > home_mean:
        diff = (away_mean - home_mean) / max(0.1, home_mean)
        return min(config['VPD_MAX_BOOST'], diff * config['VPD_WEIGHT'])

    return 0


def compute_oal(model, name, stat, opponent, config):
    """
    Opponent-Adjusted Lambda (OAL) — Adjust based on opponent quality.
    Boosts against weak defenses, slight penalty against elite.
    """
    if not config.get('OAL_ENABLED', False):
        return 0

    opp_avg = model.get_opp_defense_avg(opponent, stat)
    if opp_avg is None:
        return 0

    # Compare to global average (approximate)
    all_opp_avgs = []
    for opp_team in model.team_stats_allowed:
        vals = model.team_stats_allowed[opp_team].get(stat, [])
        if vals:
            all_opp_avgs.append(sum(vals) / len(vals))

    if not all_opp_avgs:
        return 0

    global_avg = sum(all_opp_avgs) / len(all_opp_avgs)
    if global_avg == 0:
        return 0

    # Asymmetric: bigger boost for weak defense, smaller penalty for strong
    diff_pct = (opp_avg - global_avg) / global_avg
    if diff_pct > 0:
        adj = min(config['OAL_MAX_ADJ'], diff_pct * config['OAL_WEIGHT'] * 2)
    else:
        adj = max(-config['OAL_MAX_ADJ'] * 0.5, diff_pct * config['OAL_WEIGHT'])

    return adj


# =========================================================================
# COMBINED SIGNAL
# =========================================================================

def compute_signal(model, name, stat, line, market_odds, config,
                   home_away=None, opponent=None):
    """
    Combined signal with MLB-specific features and ROI-focused scoring.
    10-component cascade: GFT + BEQ + ESI + PBF + ABVC + CGSM + VPD + OAL.
    """
    if model.game_count(name) < config['MIN_GAMES']:
        return None
    if model.avg_ab(name, 5) < config['MIN_AB']:
        return None

    # Core gates (must pass)
    gft = compute_gft(model, name, stat, line, config)
    if not gft or gft['score'] < config['GATE_MIN_GFT']:
        return None

    beq = compute_beq(model, name, stat, line, market_odds, config)
    if not beq or beq['edge'] < config['BEQ_MIN_EDGE']:
        return None

    esi = compute_esi(model, name, stat, config)
    if not esi or esi['stability'] < config['GATE_MIN_ESI']:
        return None

    if beq['hit_rate'] < config['GATE_MIN_HIT_RATE']:
        return None

    # Extended window consistency (anti-overfitting)
    if beq['ext_hit_rate'] is not None:
        if beq['ext_hit_rate'] < beq['hit_rate'] - 0.12:
            return None
        if beq['ext_hit_rate'] < config['GATE_MIN_HIT_RATE'] - 0.05:
            return None

    # MLB-specific components (additive boosters, not hard gates)
    pbf = compute_pbf(model, name, stat, line, market_odds, config)
    pbf_bonus = 0
    if pbf and pbf['margin'] >= config['PBF_MIN_MARGIN']:
        pbf_bonus = pbf['margin'] * config['PBF_WEIGHT']

    abvc = compute_abvc(model, name, config)
    abvc_bonus = 0
    if abvc and abvc['score'] >= config['ABVC_MIN_SCORE']:
        abvc_bonus = abvc['score'] * config['ABVC_WEIGHT']

    cgsm = compute_cgsm(model, name, stat, line, config)
    cgsm_bonus = 0
    if cgsm:
        cgsm_bonus = cgsm['score'] * config['CGSM_WEIGHT']

    vpd_boost = compute_vpd(model, name, stat, home_away, config) if home_away else 0
    oal_adj = compute_oal(model, name, stat, opponent, config) if opponent else 0

    # EV gate
    ev = expected_value(beq['lower_bound'], market_odds)
    if ev < config['MIN_EV']:
        return None

    # Combined score: geometric mean of core + additive MLB bonuses
    components = [
        gft['score'],
        min(1.0, beq['edge'] / 0.15 + 0.5),
        esi['stability'],
    ]
    log_sum = sum(math.log(max(0.001, c)) for c in components)
    combined = math.exp(log_sum / len(components))

    # Add MLB-specific bonuses
    combined += pbf_bonus + abvc_bonus + cgsm_bonus + vpd_boost + oal_adj
    combined = max(0, min(1.0, combined))

    if combined < config['GATE_MIN_COMBINED']:
        return None

    kelly_f = kelly_fraction(beq['lower_bound'], market_odds, config['KELLY_FRACTION'])

    return {
        'player': name, 'stat': stat, 'line': line, 'odds': market_odds,
        'combined_score': combined,
        'gft_score': gft['score'], 'beq_edge': beq['edge'],
        'esi_stability': esi['stability'],
        'pbf_margin': pbf['margin'] if pbf else 0,
        'abvc_score': abvc['score'] if abvc else 0,
        'cgsm_streak': cgsm['streak'] if cgsm else 0,
        'vpd_boost': vpd_boost, 'oal_adj': oal_adj,
        'ev': ev, 'kelly': kelly_f,
        'hit_rate': beq['hit_rate'],
        'bayesian_prob': beq['mean_prob'],
        'true_prob': beq['lower_bound'],
        'market_prob': beq['market_prob'],
        'edge': beq['edge'],
    }


# =========================================================================
# PARLAY CONSTRUCTION
# =========================================================================

def compute_correlation(model, n1, s1, n2, s2, window=20):
    v1 = model.get_values(n1, s1, window, min_ab=1)
    v2 = model.get_values(n2, s2, window, min_ab=1)
    if not v1 or not v2:
        return 0
    n = min(len(v1), len(v2))
    if n < 5:
        return 0
    v1, v2 = v1[-n:], v2[-n:]
    m1, m2 = sum(v1) / n, sum(v2) / n
    cov = sum((a - m1) * (b - m2) for a, b in zip(v1, v2)) / n
    s1v = math.sqrt(sum((a - m1) ** 2 for a in v1) / n)
    s2v = math.sqrt(sum((b - m2) ** 2 for b in v2) / n)
    if s1v == 0 or s2v == 0:
        return 0
    return cov / (s1v * s2v)


def build_parlays(signals, model, config):
    if len(signals) < config['PARLAY_MIN_LEGS']:
        return []

    sorted_sigs = sorted(signals, key=lambda s: s['ev'], reverse=True)
    parlays = []
    used = set()

    for i, anchor in enumerate(sorted_sigs):
        if anchor['player'] in used:
            continue
        legs = [anchor]
        leg_teams = {model.get_team(anchor['player'])}

        for j, cand in enumerate(sorted_sigs):
            if i == j or cand['player'] in {l['player'] for l in legs}:
                continue
            if len(legs) >= config['PARLAY_MAX_LEGS']:
                break
            ct = model.get_team(cand['player'])
            if ct in leg_teams:
                continue
            max_corr = max(abs(compute_correlation(model, l['player'], l['stat'],
                                                    cand['player'], cand['stat']))
                          for l in legs)
            if max_corr > config['PARLAY_MAX_CORR']:
                continue
            legs.append(cand)
            if ct:
                leg_teams.add(ct)

        if len(legs) >= config['PARLAY_MIN_LEGS']:
            combo_prob = 1.0
            total_edge = 0
            odds_list = []
            for l in legs:
                combo_prob *= l['bayesian_prob']
                total_edge += l['edge']
                odds_list.append(l['odds'])

            p_dec = parlay_decimal_odds(odds_list)
            p_ev = combo_prob * p_dec - 1

            if p_ev > 0 and total_edge >= config['PARLAY_MIN_EDGE']:
                parlays.append({
                    'legs': legs, 'n_legs': len(legs),
                    'parlay_decimal': p_dec,
                    'parlay_american': decimal_to_american(p_dec),
                    'combined_prob': combo_prob, 'parlay_ev': p_ev,
                    'total_edge': total_edge,
                })
                for l in legs:
                    used.add(l['player'])

    parlays.sort(key=lambda p: p['parlay_ev'], reverse=True)
    return parlays


# =========================================================================
# WALK-FORWARD BACKTEST
# =========================================================================

def run_backtest(box_scores, odds_data, config, verbose=False):
    """
    Walk-forward backtest for MLB props.
    Uses real historical odds from The Odds API.
    Processes: hitsProps, tbProps, runsProps, rbiProps, hrProps
    """
    model = MLBPlayerModel()
    sorted_games = sorted(box_scores, key=lambda g: g['date'])

    odds_idx = {}
    for od in odds_data:
        d = od.get('date', '')
        k = od.get('gameKey', f"{od.get('awayTeam', '')}@{od.get('homeTeam', '')}")
        odds_idx.setdefault(d, {})[k] = od

    box_idx = {}
    for g in box_scores:
        box_idx.setdefault(g['date'], []).append(g)

    all_picks = []
    daily_results = []
    processed = set()

    for game in sorted_games:
        date = game['date']
        if date in processed:
            continue
        processed.add(date)

        day_odds = odds_idx.get(date, {})
        day_boxes = box_idx.get(date, [])

        if not day_odds or not day_boxes:
            for bg in day_boxes:
                _update_model(model, bg, date)
            continue

        day_signals = []
        for bg in day_boxes:
            gk = f"{bg.get('away', '')}@{bg.get('home', '')}"
            og = day_odds.get(gk)
            if not og:
                continue

            home_t = bg.get('home', '')
            away_t = bg.get('away', '')

            for player in bg.get('players', []):
                name = player['name']
                team = player.get('team', '')
                ha = 'home' if team == home_t else 'away'
                opp = away_t if team == home_t else home_t

                # Hits props (primary MLB market)
                hp = og.get('hitsProps', {}).get(name, {})
                for thr, data in hp.items():
                    _try_signal(model, name, 'h', thr, data, config, ha, opp,
                               player, 'h', day_signals, date, gk)

                # Total bases props
                tbp = og.get('tbProps', {}).get(name, {})
                for thr, data in tbp.items():
                    _try_signal(model, name, 'tb', thr, data, config, ha, opp,
                               player, 'tb', day_signals, date, gk)

                # Runs props
                rp = og.get('runsProps', {}).get(name, {})
                for thr, data in rp.items():
                    _try_signal(model, name, 'r', thr, data, config, ha, opp,
                               player, 'r', day_signals, date, gk)

                # RBI props
                rbip = og.get('rbiProps', {}).get(name, {})
                for thr, data in rbip.items():
                    _try_signal(model, name, 'rbi', thr, data, config, ha, opp,
                               player, 'rbi', day_signals, date, gk)

                # HR props
                hrp = og.get('hrProps', {}).get(name, {})
                for thr, data in hrp.items():
                    _try_signal(model, name, 'hr', thr, data, config, ha, opp,
                               player, 'hr', day_signals, date, gk)

        # Select bets
        if day_signals:
            day_signals.sort(key=lambda s: s['ev'], reverse=True)

            seen = set()
            unique = []
            for s in day_signals:
                if s['player'] not in seen:
                    unique.append(s)
                    seen.add(s['player'])

            singles = [s for s in unique if s['combined_score'] >= config['SINGLE_MIN_SCORE']]
            for s in singles[:config['MAX_DAILY_BETS']]:
                hit = s['hit']
                wager = max(50, min(300, round(config['UNIT_SIZE'] * (1 + s['kelly']))))
                pnl = pnl_for_bet(hit, s['odds'], wager)
                all_picks.append({**s, 'bet_type': 'single', 'pnl': pnl, 'wager': wager})

            parlay_elig = [s for s in unique if s['combined_score'] >= config['PARLAY_LEG_MIN_SCORE']]
            parlays = build_parlays(parlay_elig, model, config)
            for p in parlays[:1]:
                hit = all(l['hit'] for l in p['legs'])
                wager = config['UNIT_SIZE']
                pnl = round((p['parlay_decimal'] - 1) * wager) if hit else -wager
                all_picks.append({
                    'bet_type': 'parlay', 'n_legs': p['n_legs'],
                    'legs': [{'player': l['player'], 'stat': l['stat'],
                              'line': l['line'], 'odds': l['odds'],
                              'hit': l['hit'], 'actual': l.get('actual'),
                              'edge': l['edge']} for l in p['legs']],
                    'hit': hit, 'pnl': pnl, 'wager': wager,
                    'date': date, 'parlay_decimal': p['parlay_decimal'],
                    'parlay_ev': p['parlay_ev'], 'total_edge': p['total_edge'],
                })

            day_picks = [p for p in all_picks if p.get('date') == date]
            if day_picks:
                daily_results.append({
                    'date': date,
                    'n_picks': len(day_picks),
                    'wins': sum(1 for p in day_picks if p.get('hit')),
                    'pnl': sum(p.get('pnl', 0) for p in day_picks),
                })

        for bg in day_boxes:
            _update_model(model, bg, date)

    return _compute_stats(all_picks, daily_results, config)


def _try_signal(model, name, stat, thr_str, data, config, ha, opp,
                player, actual_key, signals, date, gk):
    try:
        line = float(thr_str)
    except (ValueError, TypeError):
        return
    odds_val = data.get('overOdds')
    if odds_val is None:
        return
    if odds_val < config['MIN_ODDS'] or odds_val > config['MAX_ODDS']:
        return

    sig = compute_signal(model, name, stat, line, odds_val, config,
                         home_away=ha, opponent=opp)
    if sig:
        actual = _safe_int(player.get(actual_key, 0))
        sig['actual'] = actual
        sig['hit'] = actual > line
        sig['date'] = date
        sig['game_key'] = gk
        signals.append(sig)


def _update_model(model, game_data, date):
    home = game_data.get('home', '')
    away = game_data.get('away', '')
    for p in game_data.get('players', []):
        ab = _safe_int(p.get('ab', 0))
        if ab < 1:
            continue
        team = p.get('team', '')
        ha = 'home' if team == home else 'away'
        opp = away if team == home else home
        model.update(p['name'], p, date, team, ha, opp)


def _compute_stats(all_picks, daily_results, config):
    """Compute comprehensive backtest statistics."""
    singles = [p for p in all_picks if p.get('bet_type') == 'single']
    parlays = [p for p in all_picks if p.get('bet_type') == 'parlay']

    s_wins = sum(1 for s in singles if s.get('hit'))
    p_wins = sum(1 for p in parlays if p.get('hit'))

    total_pnl = sum(p.get('pnl', 0) for p in all_picks)
    total_wager = sum(p.get('wager', config['UNIT_SIZE']) for p in all_picks)
    total_roi = total_pnl / total_wager if total_wager > 0 else 0

    running, peak, max_dd = 0, 0, 0
    for p in all_picks:
        running += p.get('pnl', 0)
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)

    max_loss_streak, max_win_streak = 0, 0
    cur_loss, cur_win = 0, 0
    for p in all_picks:
        if p.get('hit'):
            cur_win += 1
            cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)

    all_legs = []
    for p in parlays:
        all_legs.extend(p.get('legs', []))

    return {
        'total_picks': len(all_picks),
        'total_wins': s_wins + p_wins,
        'total_pnl': total_pnl,
        'total_wagered': total_wager,
        'total_roi': total_roi,
        'singles_total': len(singles),
        'singles_wins': s_wins,
        'singles_accuracy': s_wins / len(singles) if singles else 0,
        'singles_pnl': sum(p.get('pnl', 0) for p in singles),
        'parlay_total': len(parlays),
        'parlay_wins': p_wins,
        'parlay_accuracy': p_wins / len(parlays) if parlays else 0,
        'parlay_pnl': sum(p.get('pnl', 0) for p in parlays),
        'parlay_leg_total': len(all_legs),
        'parlay_leg_wins': sum(1 for l in all_legs if l.get('hit')),
        'parlay_leg_accuracy': sum(1 for l in all_legs if l.get('hit')) / len(all_legs) if all_legs else 0,
        'individual_accuracy': s_wins / len(singles) if singles else 0,
        'accuracy': (s_wins + p_wins) / len(all_picks) if all_picks else 0,
        'max_drawdown': max_dd,
        'max_loss_streak': max_loss_streak,
        'max_win_streak': max_win_streak,
        'active_days': len(daily_results),
        'avg_daily_pnl': total_pnl / len(daily_results) if daily_results else 0,
        'picks': all_picks,
        'daily': daily_results,
    }
