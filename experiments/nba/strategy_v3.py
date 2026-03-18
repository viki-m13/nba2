"""
Hybrid Ensemble Certainty Engine (HECE) — NBA Strategy v3
==========================================================
Combines V1's proven signal stack (GFT, BEQ, ESI, 5 proprietary boosters)
with V2's multi-gate certainty verification for maximum accuracy on plus-money.

Architecture:
- Layer 1: V1 Signal Stack (GFT floor, Bayesian edge, stability, 5 boosters)
- Layer 2: Certainty Gates (multi-window unanimity, bootstrap CI, streak)
- Layer 3: Ensemble Consensus (both layers must independently agree)
- Layer 4: Market Mispricing Filter (edge, EV, odds constraints)

Target: 90%+ accuracy on positive odds (+100 to +400)
with sufficient volume for statistical validity.

Walk-forward only, real Odds API odds, no leakage.
"""

import math
import random
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.odds_math import (
    american_to_decimal, implied_probability, expected_value,
    kelly_fraction, pnl_for_bet, parlay_decimal_odds, decimal_to_american,
    beta_ppf_approx, shannon_entropy,
)

# =========================================================================
# CONFIG — Hybrid: V1 signals + V2 certainty gates
# =========================================================================

HECE_CONFIG = {
    # === Data requirements ===
    'MIN_GAMES': 12,
    'MIN_MINUTES': 18,

    # === V1 LAYER: Gravitational Floor Theory ===
    'GFT_WINDOWS': [5, 10, 15],
    'GFT_DECAY': 0.93,
    'GFT_GRAVITY': 0.35,
    'GFT_MIN_CLEARANCE': 0.15,
    'GFT_MAX_SPREAD': 5.0,

    # === V1 LAYER: Bayesian Edge Quantification ===
    'BEQ_PRIOR_ALPHA': 1.0,
    'BEQ_PRIOR_BETA': 1.0,
    'BEQ_CI_LEVEL': 0.75,
    'BEQ_MIN_EDGE': 0.12,

    # === V1 LAYER: Entropy Stability Index ===
    'ESI_BINS': 6,
    'ESI_MAX_ENTROPY': 0.90,
    'ESI_TREND_WEIGHT': 0.18,

    # === V1 LAYER: 5 Proprietary Signals ===
    'PAOV_ENABLED': True,
    'PAOV_WEIGHT': 0.10,
    'PAOV_BOOST_THRESHOLD': 1.05,
    'PAOV_MAX_BOOST': 0.15,
    'MSC_ENABLED': True,
    'MSC_WEIGHT': 0.08,
    'MSC_WINDOW': 10,
    'MSC_MAX_CV': 0.22,
    'MSC_ELITE_CV': 0.10,
    'MSC_MIN_AVG_MIN': 25,
    'RDPD_ENABLED': True,
    'RDPD_WEIGHT': 0.06,
    'RDPD_B2B_PENALTY': -0.05,
    'RDPD_REST_BOOST': 0.08,
    'RDPD_MAX_BOOST': 0.12,
    'OSAI_ENABLED': True,
    'OSAI_WEIGHT': 0.10,
    'OSAI_BOOST_MULT': 1.5,
    'OSAI_PENALTY_MULT': 0.55,
    'OSAI_MAX_ADJ': 0.20,
    'HHM_ENABLED': True,
    'HHM_WEIGHT': 0.07,
    'HHM_LOOKBACK': 7,
    'HHM_MIN_STREAK': 2,
    'HHM_MAX_BOOST': 0.10,

    # === V1 LAYER: Quality Gates ===
    'GATE_MIN_GFT': 0.02,
    'GATE_MIN_ESI': 0.08,
    'GATE_MIN_HIT_RATE': 0.55,
    'GATE_MIN_COMBINED': 0.30,
    'MIN_EV': 0.10,
    'SINGLE_MIN_SCORE': 0.40,

    # === V2 LAYER: Certainty Gates ===
    'CG_ENABLED': True,
    'CG_MWU_MIN_HR': 0.70,     # Multi-window unanimity threshold
    'CG_MWU_WINDOWS': [5, 10],  # Windows to check
    'CG_BOOT_ENABLED': True,
    'CG_BOOT_N': 300,
    'CG_BOOT_CI': 0.05,
    'CG_BOOT_MIN_PROB': 0.65,
    'CG_CHS_MIN': 2,           # Consecutive hit streak minimum
    'CG_TSV_ENABLED': True,
    'CG_TSV_MIN_HR': 0.55,     # Temporal split min hit rate

    # === Odds Filter (POSITIVE ONLY) ===
    'MIN_ODDS': 100,
    'MAX_ODDS': 400,

    # === Bet Sizing ===
    'UNIT_SIZE': 100,
    'KELLY_FRACTION': 0.30,
    'MAX_DAILY_BETS': 10,

    # === Parlay ===
    'PARLAY_ENABLED': True,
    'PARLAY_MIN_LEGS': 2,
    'PARLAY_MAX_LEGS': 3,
    'PARLAY_MAX_CORR': 0.35,
    'PARLAY_MIN_EDGE': 0.06,
    'PARLAY_MAX_PER_DAY': 2,
    'PARLAY_WAGER_2LEG': 100,
    'PARLAY_WAGER_3LEG': 60,
}

# Parameter ranges for optimization
HECE_PARAM_RANGES = {
    'MIN_GAMES': (8, 20),
    'MIN_MINUTES': (12, 25),
    'GFT_DECAY': (0.82, 0.98),
    'GFT_GRAVITY': (0.10, 0.55),
    'GFT_MIN_CLEARANCE': (0.05, 2.5),
    'GFT_MAX_SPREAD': (2.0, 10.0),
    'BEQ_CI_LEVEL': (0.60, 0.92),
    'BEQ_MIN_EDGE': (0.05, 0.25),
    'ESI_MAX_ENTROPY': (0.50, 0.98),
    'ESI_TREND_WEIGHT': (0.05, 0.45),
    'GATE_MIN_GFT': (0.01, 0.30),
    'GATE_MIN_ESI': (0.02, 0.30),
    'GATE_MIN_HIT_RATE': (0.40, 0.75),
    'GATE_MIN_COMBINED': (0.15, 0.60),
    'MIN_EV': (0.03, 0.25),
    'SINGLE_MIN_SCORE': (0.25, 0.70),
    'MIN_ODDS': (100, 160),
    'MAX_ODDS': (200, 600),
    'KELLY_FRACTION': (0.10, 0.50),
    'MAX_DAILY_BETS': (4, 15),
    # V1 signal tuning
    'PAOV_WEIGHT': (0.03, 0.20),
    'PAOV_BOOST_THRESHOLD': (1.0, 1.15),
    'MSC_WEIGHT': (0.02, 0.15),
    'MSC_MAX_CV': (0.10, 0.35),
    'MSC_MIN_AVG_MIN': (20, 32),
    'RDPD_WEIGHT': (0.02, 0.12),
    'OSAI_WEIGHT': (0.03, 0.20),
    'OSAI_BOOST_MULT': (1.0, 3.0),
    'OSAI_PENALTY_MULT': (0.2, 0.8),
    'HHM_WEIGHT': (0.02, 0.12),
    'HHM_LOOKBACK': (4, 12),
    # V2 certainty gates
    'CG_MWU_MIN_HR': (0.50, 0.90),
    'CG_BOOT_CI': (0.02, 0.15),
    'CG_BOOT_MIN_PROB': (0.50, 0.85),
    'CG_CHS_MIN': (1, 5),
    'CG_TSV_MIN_HR': (0.40, 0.80),
    # Parlay
    'PARLAY_MAX_LEGS': (2, 4),
    'PARLAY_MAX_CORR': (0.15, 0.55),
    'PARLAY_MIN_EDGE': (0.02, 0.18),
    'PARLAY_WAGER_2LEG': (50, 250),
    'PARLAY_WAGER_3LEG': (25, 150),
}


# =========================================================================
# PLAYER MODEL — full V1 model with team tracking
# =========================================================================

class HECEPlayerModel:
    def __init__(self, max_history=50):
        self.profiles = {}
        self.max_history = max_history
        self.team_pace = defaultdict(list)
        self.team_allowed = defaultdict(lambda: defaultdict(list))
        self.league_stat_avgs = defaultdict(list)

    def update(self, name, stats, date, team, home_away, opponent):
        if name not in self.profiles:
            self.profiles[name] = {'games': [], 'dates': []}

        pts = _si(stats.get('pts', 0))
        reb = _si(stats.get('reb', 0))
        ast = _si(stats.get('ast', 0))
        mins = _si(stats.get('min', 0))
        three_str = stats.get('three', '0-0')
        three_made = _si(three_str.split('-')[0]) if isinstance(three_str, str) and '-' in three_str else _si(three_str)

        game = {
            'date': date, 'team': team, 'opponent': opponent,
            'home_away': home_away,
            'pts': pts, 'reb': reb, 'ast': ast, 'min': mins,
            '3pm': three_made, 'pra': pts + reb + ast,
        }
        self.profiles[name]['games'].append(game)
        self.profiles[name]['dates'].append(date)
        if len(self.profiles[name]['games']) > self.max_history:
            self.profiles[name]['games'] = self.profiles[name]['games'][-self.max_history:]
            self.profiles[name]['dates'] = self.profiles[name]['dates'][-self.max_history:]

        if mins >= 15:
            for sk in ['pts', 'reb', 'ast', 'pra']:
                self.team_allowed[opponent][sk].append(game[sk])
                self.league_stat_avgs[sk].append(game[sk])

    def update_game_pace(self, home_team, away_team, home_pts, away_pts):
        total = home_pts + away_pts
        for t in [home_team, away_team]:
            self.team_pace[t].append(total)
            if len(self.team_pace[t]) > 30:
                self.team_pace[t] = self.team_pace[t][-30:]

    def get_values(self, name, stat, window=None, min_min=15):
        p = self.profiles.get(name)
        if not p: return None
        games = [g for g in p['games'] if g.get('min', 0) >= min_min]
        if window: games = games[-window:]
        return [g.get(stat, 0) for g in games] if games else None

    def game_count(self, name):
        p = self.profiles.get(name)
        return len(p['games']) if p else 0

    def avg_minutes(self, name, window=5):
        vals = self.get_values(name, 'min', window, min_min=0)
        return sum(vals) / len(vals) if vals else 0

    def get_team(self, name):
        p = self.profiles.get(name)
        return p['games'][-1]['team'] if p and p['games'] else None

    def get_minutes_cv(self, name, window=10):
        vals = self.get_values(name, 'min', window, min_min=0)
        if not vals or len(vals) < 5: return None
        mean = sum(vals) / len(vals)
        if mean < 5: return None
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) / mean

    def get_rest_days(self, name, current_date):
        p = self.profiles.get(name)
        if not p or not p['dates']: return None
        last_date = p['dates'][-1]
        try:
            cy, cm, cd = int(current_date[:4]), int(current_date[5:7]), int(current_date[8:10])
            ly, lm, ld = int(last_date[:4]), int(last_date[5:7]), int(last_date[8:10])
            return max(0, (cy * 365 + cm * 30 + cd) - (ly * 365 + lm * 30 + ld))
        except (ValueError, IndexError):
            return None

    def get_opponent_pace(self, opponent, window=15):
        vals = self.team_pace.get(opponent, [])
        if len(vals) < 3: return None
        recent = vals[-window:]
        return sum(recent) / len(recent)

    def get_league_avg_pace(self):
        all_pace = []
        for tv in self.team_pace.values():
            all_pace.extend(tv[-20:])
        return sum(all_pace) / len(all_pace) if all_pace else 210

    def get_opp_defense_avg(self, opponent, stat, window=20):
        vals = self.team_allowed.get(opponent, {}).get(stat, [])
        if not vals: return None
        return sum(vals[-window:]) / len(vals[-window:])

    def get_league_stat_avg(self, stat, window=200):
        vals = self.league_stat_avgs.get(stat, [])
        if not vals: return None
        return sum(vals[-window:]) / len(vals[-window:])

    def get_streak_over_line(self, name, stat, line, lookback=7):
        vals = self.get_values(name, stat, lookback, min_min=10)
        if not vals: return 0, 0
        streak = 0
        for v in reversed(vals):
            if v > line: streak += 1
            else: break
        hr = sum(1 for v in vals if v > line) / len(vals)
        return streak, hr


def _si(val):
    """Safe int conversion."""
    if isinstance(val, (int, float)): return int(val)
    if isinstance(val, str):
        try: return int(val)
        except ValueError: return 0
    return 0


# =========================================================================
# V1 LAYER: Signal Components (GFT, BEQ, ESI, 5 Boosters)
# =========================================================================

def compute_gft(model, name, stat, line, config):
    """Gravitational Floor Theory — multi-window floor detection."""
    windows = config['GFT_WINDOWS']
    decay = config['GFT_DECAY']
    gravity = config['GFT_GRAVITY']
    floors, clearances = [], []
    for w in windows:
        values = model.get_values(name, stat, w)
        if values is None or len(values) < min(w, 5):
            return None
        n = len(values)
        weights = [decay ** (n - 1 - i) for i in range(n)]
        tw = sum(weights)
        wmean = sum(v * wt for v, wt in zip(values, weights)) / tw
        sorted_vw = sorted(zip(values, weights), key=lambda x: x[0])
        cum, p10 = 0, sorted_vw[0][0]
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
    depth = min(1.0, sum(clearances) / len(clearances) / 8.0)
    return {'score': convergence * 0.5 + depth * 0.5, 'convergence': convergence, 'depth': depth}


def compute_beq(model, name, stat, line, market_odds, config):
    """Bayesian Edge Quantification — credible interval edge."""
    values = model.get_values(name, stat, 20)
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
    ext = model.get_values(name, stat, 30)
    ext_hr = sum(1 for v in ext if v > line) / len(ext) if ext and len(ext) >= 15 else None
    return {
        'mean_prob': mean_prob, 'lower_bound': lower,
        'market_prob': mkt_prob, 'edge': edge,
        'hit_rate': hit_rate, 'ext_hit_rate': ext_hr,
    }


def compute_esi(model, name, stat, config):
    """Entropy Stability Index — distribution predictability."""
    values = model.get_values(name, stat, 20)
    if values is None or len(values) < 8:
        return None
    dist_ent = shannon_entropy(values, config['ESI_BINS'])
    mid = len(values) // 2
    m1 = sum(values[:mid]) / mid
    m2 = sum(values[mid:]) / (len(values) - mid)
    overall = sum(values) / len(values)
    if overall == 0: return None
    trend_shift = abs(m2 - m1) / overall
    trend_stab = max(0, 1 - trend_shift / 0.3)
    sorted_v = sorted(values)
    p15 = sorted_v[max(0, int(len(sorted_v) * 0.15))]
    recent = values[-5:]
    tail_events = sum(1 for v in recent if v <= p15)
    tail_risk = 1 - tail_events / len(recent)
    ent_score = max(0, 1 - dist_ent / config['ESI_MAX_ENTROPY'])
    tw = config['ESI_TREND_WEIGHT']
    stability = ((1 - tw) * ent_score + tw * trend_stab) * tail_risk
    return {'stability': stability, 'entropy': dist_ent, 'trend_stability': trend_stab}


# === 5 Proprietary Boosters ===

def compute_paov(model, opponent, config):
    if not config.get('PAOV_ENABLED'): return 0.0
    opp_pace = model.get_opponent_pace(opponent)
    if opp_pace is None: return 0.0
    pace_ratio = opp_pace / max(1, model.get_league_avg_pace())
    if pace_ratio < config['PAOV_BOOST_THRESHOLD']: return 0.0
    excess = pace_ratio - config['PAOV_BOOST_THRESHOLD']
    return min(config['PAOV_MAX_BOOST'], excess * 2.0) * config['PAOV_WEIGHT']


def compute_msc(model, name, config):
    if not config.get('MSC_ENABLED'): return 0.0
    cv = model.get_minutes_cv(name, config.get('MSC_WINDOW', 10))
    if cv is None: return 0.0
    if model.avg_minutes(name, config.get('MSC_WINDOW', 10)) < config.get('MSC_MIN_AVG_MIN', 25):
        return 0.0
    if cv > config['MSC_MAX_CV']: return 0.0
    score = 1.0 if cv <= config['MSC_ELITE_CV'] else max(0, 1 - (cv - config['MSC_ELITE_CV']) / (config['MSC_MAX_CV'] - config['MSC_ELITE_CV']))
    return score * config['MSC_WEIGHT']


def compute_rdpd(model, name, date, config):
    if not config.get('RDPD_ENABLED'): return 0.0
    rest = model.get_rest_days(name, date)
    if rest is None: return 0.0
    if rest <= 1: return config.get('RDPD_B2B_PENALTY', -0.05) * config['RDPD_WEIGHT']
    if rest >= 2:
        boost = min(config.get('RDPD_MAX_BOOST', 0.12), config.get('RDPD_REST_BOOST', 0.08) * min(rest - 1, 3))
        return boost * config['RDPD_WEIGHT']
    return 0.0


def compute_osai(model, stat, opponent, config):
    if not config.get('OSAI_ENABLED'): return 0.0
    opp_avg = model.get_opp_defense_avg(opponent, stat)
    league_avg = model.get_league_stat_avg(stat)
    if opp_avg is None or league_avg is None or league_avg == 0: return 0.0
    ratio = opp_avg / league_avg
    if ratio > 1.0:
        adj = min(config['OSAI_MAX_ADJ'], (ratio - 1.0) * config.get('OSAI_BOOST_MULT', 1.5))
    else:
        adj = -min(config['OSAI_MAX_ADJ'], (1.0 - ratio) * config.get('OSAI_PENALTY_MULT', 0.5))
    return adj * config['OSAI_WEIGHT']


def compute_hhm(model, name, stat, line, config):
    if not config.get('HHM_ENABLED'): return 0.0
    streak, window_hr = model.get_streak_over_line(name, stat, line, config.get('HHM_LOOKBACK', 7))
    if streak < config.get('HHM_MIN_STREAK', 2): return 0.0
    combined = min(1.0, streak / config.get('HHM_LOOKBACK', 7)) * 0.6 + window_hr * 0.4
    return min(config.get('HHM_MAX_BOOST', 0.10), combined * config['HHM_WEIGHT'])


# =========================================================================
# V2 LAYER: Certainty Gates
# =========================================================================

def certainty_gate_check(model, name, stat, line, config):
    """
    V2 certainty gates. Returns True if all enabled gates pass.
    These are ADDITIONAL to V1's signal stack.
    """
    if not config.get('CG_ENABLED', True):
        return True

    all_values = model.get_values(name, stat, None)
    if not all_values or len(all_values) < 8:
        return False

    # Gate: Multi-Window Unanimity
    for w in config.get('CG_MWU_WINDOWS', [5, 10]):
        vals = model.get_values(name, stat, w)
        if not vals or len(vals) < min(w, 3):
            return False
        hr = sum(1 for v in vals if v > line) / len(vals)
        if hr < config.get('CG_MWU_MIN_HR', 0.70):
            return False

    # Gate: Bootstrap Confidence Interval
    if config.get('CG_BOOT_ENABLED', True):
        boot_probs = _bootstrap_hit_prob(
            all_values, line,
            n_samples=int(config.get('CG_BOOT_N', 300)),
            seed=hash((name, stat, str(line))) % (2**31))
        ci_idx = max(0, int(len(boot_probs) * config.get('CG_BOOT_CI', 0.05)) - 1)
        if boot_probs[ci_idx] < config.get('CG_BOOT_MIN_PROB', 0.65):
            return False

    # Gate: Consecutive Hit Streak
    min_streak = int(config.get('CG_CHS_MIN', 2))
    recent = all_values[-min_streak:]
    if len(recent) < min_streak or not all(v > line for v in recent):
        return False

    # Gate: Temporal Split Verification
    if config.get('CG_TSV_ENABLED', True):
        mid = len(all_values) // 2
        if mid >= 3:
            h1_hr = sum(1 for v in all_values[:mid] if v > line) / mid
            h2_hr = sum(1 for v in all_values[mid:] if v > line) / (len(all_values) - mid)
            tsv_min = config.get('CG_TSV_MIN_HR', 0.55)
            if h1_hr < tsv_min or h2_hr < tsv_min:
                return False

    return True


def _bootstrap_hit_prob(values, line, n_samples=300, seed=None):
    rng = random.Random(seed)
    n = len(values)
    hit_rates = []
    for _ in range(n_samples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        hit_rates.append(sum(1 for v in sample if v > line) / n)
    hit_rates.sort()
    return hit_rates


# =========================================================================
# COMBINED SIGNAL — V1 stack + V2 gates
# =========================================================================

def compute_hybrid_signal(model, name, stat, line, market_odds, config,
                          home_away=None, date=None, opponent=None):
    """
    Hybrid signal: V1's full signal stack + V2's certainty gates.
    Both layers must independently agree for a pick to pass.
    """
    # Pre-checks
    if model.game_count(name) < config['MIN_GAMES']:
        return None
    if model.avg_minutes(name, 5) < config['MIN_MINUTES']:
        return None

    # === V1 LAYER ===
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

    # Extended consistency check
    if beq['ext_hit_rate'] is not None:
        if beq['ext_hit_rate'] < beq['hit_rate'] - 0.15:
            return None

    # EV gate
    ev = expected_value(beq['lower_bound'], market_odds)
    if ev < config['MIN_EV']:
        return None

    # Context
    ctx = 0.02 if home_away == 'home' else 0

    # Core combined score (geometric mean)
    components = [gft['score'], min(1.0, beq['edge'] / 0.20 + 0.5), esi['stability']]
    log_sum = sum(math.log(max(0.001, c)) for c in components)
    combined = math.exp(log_sum / len(components)) + ctx
    combined = max(0, min(1.0, combined))

    # Proprietary boosters
    boost = 0.0
    if opponent:
        boost += compute_paov(model, opponent, config)
        boost += compute_osai(model, stat, opponent, config)
    boost += compute_msc(model, name, config)
    if date:
        boost += compute_rdpd(model, name, date, config)
    boost += compute_hhm(model, name, stat, line, config)

    combined = min(1.0, combined + boost)
    if combined < config['GATE_MIN_COMBINED']:
        return None

    # === V2 LAYER: Certainty Gates ===
    if not certainty_gate_check(model, name, stat, line, config):
        return None

    # Final signal
    kelly_val = kelly_fraction(beq['lower_bound'], market_odds, config['KELLY_FRACTION'])

    return {
        'player': name, 'stat': stat, 'line': line, 'odds': market_odds,
        'combined_score': combined,
        'gft_score': gft['score'], 'beq_edge': beq['edge'],
        'esi_stability': esi['stability'],
        'ev': ev, 'kelly': kelly_val,
        'hit_rate': beq['hit_rate'],
        'bayesian_prob': beq['mean_prob'],
        'true_prob': beq['lower_bound'],
        'market_prob': beq['market_prob'],
        'edge': beq['edge'],
        'boost_total': boost,
        'direction': 'over',
    }


# =========================================================================
# PARLAY CONSTRUCTION
# =========================================================================

def compute_correlation(model, n1, s1, n2, s2, window=20):
    v1, v2 = model.get_values(n1, s1, window), model.get_values(n2, s2, window)
    if not v1 or not v2: return 0
    n = min(len(v1), len(v2))
    if n < 5: return 0
    v1, v2 = v1[-n:], v2[-n:]
    m1, m2 = sum(v1) / n, sum(v2) / n
    cov = sum((a - m1) * (b - m2) for a, b in zip(v1, v2)) / n
    s1v = math.sqrt(sum((a - m1) ** 2 for a in v1) / n)
    s2v = math.sqrt(sum((b - m2) ** 2 for b in v2) / n)
    return cov / (s1v * s2v) if s1v > 0 and s2v > 0 else 0


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

        for cand in sorted_sigs:
            if cand['player'] in used or cand['player'] in {l['player'] for l in legs}:
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
            if ct: leg_teams.add(ct)

        if len(legs) >= config['PARLAY_MIN_LEGS']:
            combo_prob = 1.0
            total_edge = 0
            for l in legs:
                combo_prob *= l['bayesian_prob']
                total_edge += l['edge']
            p_dec = parlay_decimal_odds([l['odds'] for l in legs])
            p_ev = combo_prob * p_dec - 1
            if p_ev > 0 and total_edge >= config['PARLAY_MIN_EDGE']:
                parlays.append({
                    'legs': legs, 'n_legs': len(legs),
                    'parlay_decimal': p_dec,
                    'parlay_american': decimal_to_american(p_dec),
                    'combined_prob': combo_prob, 'parlay_ev': p_ev,
                    'total_edge': total_edge,
                })
                for l in legs: used.add(l['player'])

    parlays.sort(key=lambda p: p['parlay_ev'], reverse=True)
    return parlays


# =========================================================================
# WALK-FORWARD BACKTEST
# =========================================================================

def run_backtest(box_scores, odds_data, config, verbose=False):
    model = HECEPlayerModel()
    sorted_games = sorted(box_scores, key=lambda g: g['date'])

    odds_idx = {}
    for od in odds_data:
        d = od.get('date', '')
        k = od.get('gameKey', f"{od.get('awayTeam', '')}@{od.get('homeTeam', '')}")
        odds_idx.setdefault(d, {})[k] = od

    box_idx = {}
    for g in box_scores:
        box_idx.setdefault(g['date'], []).append(g)

    all_picks, daily_results, processed = [], [], set()

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
                opponent = away_t if team == home_t else home_t

                for prop_key, stat_key, actual_key in [
                    ('playerProps', 'pts', 'pts'),
                    ('rebProps', 'reb', 'reb'),
                    ('astProps', 'ast', 'ast'),
                ]:
                    props = og.get(prop_key, {}).get(name, {})
                    for thr, data in props.items():
                        _try_signal(model, name, stat_key, thr, data, config, ha,
                                    player, actual_key, day_signals, date, gk, opponent)

                # PRA
                pra_p = og.get('praProps', {}).get(name, {})
                for thr, data in pra_p.items():
                    _try_signal_pra(model, name, thr, data, config, ha,
                                    player, day_signals, date, gk, opponent)

        if day_signals:
            day_signals.sort(key=lambda s: s['ev'], reverse=True)
            seen = set()
            unique = []
            for s in day_signals:
                if s['player'] not in seen:
                    unique.append(s)
                    seen.add(s['player'])

            # Singles
            singles_selected = [s for s in unique if s['combined_score'] >= config['SINGLE_MIN_SCORE']]
            for s in singles_selected[:int(config['MAX_DAILY_BETS'])]:
                wager = max(50, min(400, round(config['UNIT_SIZE'] * (1 + s['kelly'] * 2))))
                pnl = pnl_for_bet(s['hit'], s['odds'], wager)
                all_picks.append({**s, 'bet_type': 'single', 'pnl': pnl, 'wager': wager})

            # Parlays
            if config.get('PARLAY_ENABLED', True):
                parlay_elig = [s for s in unique if s['combined_score'] >= config.get('PARLAY_LEG_MIN_SCORE', config['SINGLE_MIN_SCORE'] * 0.8)]
                parlays = build_parlays(parlay_elig, model, config)
                for p in parlays[:config.get('PARLAY_MAX_PER_DAY', 2)]:
                    hit = all(l['hit'] for l in p['legs'])
                    n_legs = p['n_legs']
                    wager_key = f'PARLAY_WAGER_{n_legs}LEG'
                    wager = config.get(wager_key, config['UNIT_SIZE'])
                    pnl = round((p['parlay_decimal'] - 1) * wager) if hit else -wager
                    all_picks.append({
                        'bet_type': 'parlay', 'n_legs': n_legs,
                        'legs': [{'player': l['player'], 'stat': l['stat'],
                                  'line': l['line'], 'odds': l['odds'],
                                  'hit': l['hit'], 'actual': l.get('actual'),
                                  'edge': l['edge']} for l in p['legs']],
                        'hit': hit, 'pnl': pnl, 'wager': wager,
                        'date': date, 'parlay_decimal': p['parlay_decimal'],
                        'parlay_ev': p['parlay_ev'], 'total_edge': p['total_edge'],
                    })

        if day_signals:
            day_picks = [p for p in all_picks if p.get('date') == date]
            if day_picks:
                daily_results.append({
                    'date': date, 'n_picks': len(day_picks),
                    'wins': sum(1 for p in day_picks if p.get('hit')),
                    'pnl': sum(p.get('pnl', 0) for p in day_picks),
                })
                if verbose:
                    dp = daily_results[-1]
                    print(f"  {date}: {dp['n_picks']} picks, {dp['wins']} wins, ${dp['pnl']:+d}")

        for bg in day_boxes:
            _update_model(model, bg, date)

    return _compute_stats(all_picks, daily_results, config)


def _try_signal(model, name, stat, thr_str, data, config, ha, player,
                actual_key, signals, date, gk, opponent):
    try:
        line = float(thr_str)
    except (ValueError, TypeError):
        return
    odds_val = data.get('overOdds')
    if odds_val is None:
        return
    if odds_val < config['MIN_ODDS'] or odds_val > config['MAX_ODDS']:
        return
    sig = compute_hybrid_signal(model, name, stat, line, odds_val, config,
                                home_away=ha, date=date, opponent=opponent)
    if sig:
        actual = _si(player.get(actual_key, 0))
        sig['actual'] = actual
        sig['hit'] = actual > line
        sig['date'] = date
        sig['game_key'] = gk
        signals.append(sig)


def _try_signal_pra(model, name, thr_str, data, config, ha, player,
                    signals, date, gk, opponent):
    try:
        line = float(thr_str)
    except (ValueError, TypeError):
        return
    odds_val = data.get('overOdds')
    if odds_val is None:
        return
    if odds_val < config['MIN_ODDS'] or odds_val > config['MAX_ODDS']:
        return
    sig = compute_hybrid_signal(model, name, 'pra', line, odds_val, config,
                                home_away=ha, date=date, opponent=opponent)
    if sig:
        actual = (_si(player.get('pts', 0)) + _si(player.get('reb', 0)) + _si(player.get('ast', 0)))
        sig['actual'] = actual
        sig['hit'] = actual > line
        sig['date'] = date
        sig['game_key'] = gk
        signals.append(sig)


def _update_model(model, game_data, date):
    home, away = game_data.get('home', ''), game_data.get('away', '')
    home_pts = sum(_si(p.get('pts', 0)) for p in game_data.get('players', []) if p.get('team', '') == home)
    away_pts = sum(_si(p.get('pts', 0)) for p in game_data.get('players', []) if p.get('team', '') == away)
    if home_pts > 0 and away_pts > 0:
        model.update_game_pace(home, away, home_pts, away_pts)
    for p in game_data.get('players', []):
        if _si(p.get('min', 0)) < 5:
            continue
        team = p.get('team', '')
        ha = 'home' if team == home else 'away'
        opp = away if team == home else home
        model.update(p['name'], p, date, team, ha, opp)


def _compute_stats(all_picks, daily_results, config):
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
    max_loss_streak, max_win_streak, cur_loss, cur_win = 0, 0, 0, 0
    for p in all_picks:
        if p.get('hit'):
            cur_win += 1; cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_loss += 1; cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)
    all_legs = []
    for p in parlays:
        all_legs.extend(p.get('legs', []))
    parlay_by_legs = {}
    for p in parlays:
        n = p.get('n_legs', 0)
        if n not in parlay_by_legs:
            parlay_by_legs[n] = {'total': 0, 'wins': 0, 'pnl': 0}
        parlay_by_legs[n]['total'] += 1
        if p.get('hit'): parlay_by_legs[n]['wins'] += 1
        parlay_by_legs[n]['pnl'] += p.get('pnl', 0)

    return {
        'total_picks': len(all_picks), 'total_wins': s_wins + p_wins,
        'total_pnl': total_pnl, 'total_wagered': total_wager, 'total_roi': total_roi,
        'singles_total': len(singles), 'singles_wins': s_wins,
        'singles_accuracy': s_wins / len(singles) if singles else 0,
        'singles_pnl': sum(p.get('pnl', 0) for p in singles),
        'parlay_total': len(parlays), 'parlay_wins': p_wins,
        'parlay_accuracy': p_wins / len(parlays) if parlays else 0,
        'parlay_pnl': sum(p.get('pnl', 0) for p in parlays),
        'parlay_leg_total': len(all_legs),
        'parlay_leg_wins': sum(1 for l in all_legs if l.get('hit')),
        'parlay_leg_accuracy': sum(1 for l in all_legs if l.get('hit')) / len(all_legs) if all_legs else 0,
        'parlay_by_legs': parlay_by_legs,
        'individual_accuracy': s_wins / len(singles) if singles else 0,
        'accuracy': (s_wins + p_wins) / len(all_picks) if all_picks else 0,
        'max_drawdown': max_dd, 'max_loss_streak': max_loss_streak,
        'max_win_streak': max_win_streak,
        'active_days': len(daily_results),
        'avg_daily_pnl': total_pnl / len(daily_results) if daily_results else 0,
        'picks': all_picks, 'daily': daily_results,
    }
