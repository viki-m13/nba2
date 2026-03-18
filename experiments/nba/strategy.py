"""
NBA MAX-ROI Strategy v2
========================
Targets 1000%+ ROI through:
1. PLUS-MONEY LINES — each win pays 2-10x wager
2. PARLAY-CENTRIC — 2-5 leg parlays at +500 to +5000
3. 5 PROPRIETARY NBA SIGNALS — pace, minutes stability, rest, defense, momentum
4. AGGRESSIVE KELLY — full fractional Kelly on highest EV parlays
5. VOLUME UNLOCK — proprietary signals boost more picks above threshold

Proprietary Signal Stack:
- Pace-Adjusted Opportunity Volume (PAOV): opponent pace inflates stat ceilings
- Minutes Stability Coefficient (MSC): low-variance minutes = predictable output
- Rest-Day Performance Delta (RDPD): rest advantage detection from game dates
- Opponent Stat Allowance Index (OSAI): exploit weak defenses per stat category
- Hot Hand Momentum (HHM): consecutive-game streak persistence above line

Walk-forward only, real Odds API odds, Bayesian shrinkage, no leakage.
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
# DEFAULT CONFIG — tuned for 1000%+ ROI
# =========================================================================

NBA_ROI_CONFIG = {
    # Data requirements
    'MIN_GAMES': 12,
    'MIN_MINUTES': 18,

    # Gravitational Floor
    'GFT_WINDOWS': [5, 10, 15],
    'GFT_DECAY': 0.93,
    'GFT_GRAVITY': 0.40,
    'GFT_MIN_CLEARANCE': 0.3,
    'GFT_MAX_SPREAD': 6.0,

    # Bayesian Edge — key for plus-money value
    'BEQ_PRIOR_ALPHA': 1.0,
    'BEQ_PRIOR_BETA': 1.0,
    'BEQ_CI_LEVEL': 0.80,
    'BEQ_MIN_EDGE': 0.05,

    # Stability
    'ESI_BINS': 6,
    'ESI_MAX_ENTROPY': 0.90,
    'ESI_TREND_WEIGHT': 0.18,

    # === PROPRIETARY SIGNAL 1: Pace-Adjusted Opportunity Volume ===
    'PAOV_ENABLED': True,
    'PAOV_WEIGHT': 0.10,
    'PAOV_MIN_OPP_GAMES': 5,
    'PAOV_BOOST_THRESHOLD': 1.05,  # opponent pace > 1.05x league avg
    'PAOV_MAX_BOOST': 0.15,

    # === PROPRIETARY SIGNAL 2: Minutes Stability Coefficient ===
    'MSC_ENABLED': True,
    'MSC_WEIGHT': 0.08,
    'MSC_WINDOW': 10,
    'MSC_MAX_CV': 0.20,        # coefficient of variation threshold
    'MSC_ELITE_CV': 0.10,      # elite stability
    'MSC_MIN_AVG_MIN': 25,     # must avg 25+ min for boost

    # === PROPRIETARY SIGNAL 3: Rest-Day Performance Delta ===
    'RDPD_ENABLED': True,
    'RDPD_WEIGHT': 0.06,
    'RDPD_B2B_PENALTY': -0.05, # back-to-back penalty
    'RDPD_REST_BOOST': 0.08,   # 2+ days rest boost
    'RDPD_MAX_BOOST': 0.12,

    # === PROPRIETARY SIGNAL 4: Opponent Stat Allowance Index ===
    'OSAI_ENABLED': True,
    'OSAI_WEIGHT': 0.10,
    'OSAI_MIN_OPP_GAMES': 8,
    'OSAI_BOOST_MULT': 1.5,    # boost multiplier for weak defense
    'OSAI_PENALTY_MULT': 0.5,  # penalty multiplier for strong defense
    'OSAI_MAX_ADJ': 0.20,

    # === PROPRIETARY SIGNAL 5: Hot Hand Momentum ===
    'HHM_ENABLED': True,
    'HHM_WEIGHT': 0.06,
    'HHM_LOOKBACK': 7,
    'HHM_MIN_STREAK': 2,
    'HHM_MAX_BOOST': 0.10,

    # Quality gates — relaxed for plus-money (lower implied prob = lower hit rate OK)
    'GATE_MIN_GFT': 0.10,
    'GATE_MIN_ESI': 0.10,
    'GATE_MIN_HIT_RATE': 0.40,
    'GATE_MIN_COMBINED': 0.25,

    # ROI-focused: require STRONG positive EV
    'MIN_EV': 0.08,

    # Bet selection
    'SINGLE_MIN_SCORE': 0.40,
    'PARLAY_LEG_MIN_SCORE': 0.30,

    # Parlay — the ROI engine (expanded to 5 legs for 1000%+ potential)
    'PARLAY_MIN_LEGS': 2,
    'PARLAY_MAX_LEGS': 5,
    'PARLAY_MAX_CORR': 0.35,
    'PARLAY_MIN_EDGE': 0.06,
    'PARLAY_MAX_PER_DAY': 3,

    # Tiered parlay sizing
    'PARLAY_WAGER_2LEG': 150,
    'PARLAY_WAGER_3LEG': 100,
    'PARLAY_WAGER_4LEG': 75,
    'PARLAY_WAGER_5LEG': 50,

    # Odds filter — TARGET PLUS-MONEY AND NEAR-EVEN
    'MIN_ODDS': -250,
    'MAX_ODDS': 500,

    # Bankroll — aggressive Kelly on high-edge spots
    'UNIT_SIZE': 100,
    'KELLY_FRACTION': 0.35,
    'MAX_DAILY_BETS': 8,

    # === PIPELINE B: Plus-Money Parlay Engine ===
    'PLB_ENABLED': True,
    'PLB_MIN_ODDS': 100,           # Only plus-money legs
    'PLB_MAX_ODDS': 500,           # Cap extreme underdogs
    'PLB_MIN_TRUE_PROB': 0.52,     # Must believe >52% despite plus-money pricing
    'PLB_MIN_EDGE': 0.08,          # Minimum edge for parlay leg
    'PLB_MIN_HIT_RATE': 0.50,      # Must have demonstrated hitting
    'PLB_MIN_EV': 0.10,            # Strong positive EV
    'PLB_PARLAY_MIN_LEGS': 3,      # Minimum 3 legs for multiplier
    'PLB_PARLAY_MAX_LEGS': 5,      # Up to 5 legs for extreme multiplier
    'PLB_PARLAY_MAX_CORR': 0.30,   # Tight independence
    'PLB_MAX_PARLAYS': 3,          # Up to 3 plus-money parlays per day
    'PLB_WAGER_3LEG': 80,          # Wager per 3-leg parlay
    'PLB_WAGER_4LEG': 60,          # Wager per 4-leg parlay
    'PLB_WAGER_5LEG': 40,          # Wager per 5-leg parlay
}

# Parameter ranges for optimization
NBA_PARAM_RANGES = {
    'GFT_DECAY': (0.82, 0.98),
    'GFT_GRAVITY': (0.10, 0.55),
    'GFT_MIN_CLEARANCE': (0.05, 2.5),
    'GFT_MAX_SPREAD': (2.0, 10.0),
    'BEQ_CI_LEVEL': (0.65, 0.92),
    'BEQ_MIN_EDGE': (0.02, 0.20),
    'ESI_MAX_ENTROPY': (0.50, 0.98),
    'ESI_TREND_WEIGHT': (0.05, 0.45),
    'GATE_MIN_GFT': (0.02, 0.50),
    'GATE_MIN_ESI': (0.02, 0.40),
    'GATE_MIN_HIT_RATE': (0.25, 0.65),
    'GATE_MIN_COMBINED': (0.10, 0.55),
    'MIN_EV': (0.03, 0.25),
    'SINGLE_MIN_SCORE': (0.25, 0.70),
    'PARLAY_LEG_MIN_SCORE': (0.12, 0.50),
    'PARLAY_MAX_CORR': (0.15, 0.55),
    'PARLAY_MIN_EDGE': (0.02, 0.20),
    'PARLAY_MAX_LEGS': (2, 5),
    'PARLAY_MAX_PER_DAY': (1, 5),
    'PARLAY_WAGER_2LEG': (75, 250),
    'PARLAY_WAGER_3LEG': (50, 200),
    'PARLAY_WAGER_4LEG': (25, 150),
    'PARLAY_WAGER_5LEG': (25, 100),
    'MIN_ODDS': (-350, -100),
    'MAX_ODDS': (150, 800),
    'MIN_GAMES': (8, 18),
    'MIN_MINUTES': (12, 25),
    'KELLY_FRACTION': (0.15, 0.50),
    'MAX_DAILY_BETS': (4, 12),
    # Proprietary signal tuning
    'PAOV_WEIGHT': (0.03, 0.20),
    'PAOV_BOOST_THRESHOLD': (1.0, 1.15),
    'PAOV_MAX_BOOST': (0.05, 0.25),
    'MSC_WEIGHT': (0.02, 0.15),
    'MSC_MAX_CV': (0.10, 0.35),
    'MSC_ELITE_CV': (0.05, 0.18),
    'MSC_MIN_AVG_MIN': (20, 32),
    'RDPD_WEIGHT': (0.02, 0.12),
    'RDPD_B2B_PENALTY': (-0.12, 0.0),
    'RDPD_REST_BOOST': (0.02, 0.15),
    'OSAI_WEIGHT': (0.03, 0.20),
    'OSAI_BOOST_MULT': (1.0, 3.0),
    'OSAI_PENALTY_MULT': (0.2, 0.8),
    'OSAI_MAX_ADJ': (0.08, 0.30),
    'HHM_WEIGHT': (0.02, 0.12),
    'HHM_LOOKBACK': (4, 12),
    'HHM_MIN_STREAK': (1, 4),
    'HHM_MAX_BOOST': (0.04, 0.18),
    # Pipeline B tuning
    'PLB_MIN_ODDS': (100, 200),
    'PLB_MAX_ODDS': (250, 800),
    'PLB_MIN_TRUE_PROB': (0.45, 0.65),
    'PLB_MIN_EDGE': (0.04, 0.18),
    'PLB_MIN_HIT_RATE': (0.40, 0.65),
    'PLB_MIN_EV': (0.05, 0.25),
    'PLB_PARLAY_MIN_LEGS': (2, 4),
    'PLB_PARLAY_MAX_LEGS': (3, 5),
    'PLB_PARLAY_MAX_CORR': (0.15, 0.45),
    'PLB_MAX_PARLAYS': (1, 5),
    'PLB_WAGER_3LEG': (30, 150),
    'PLB_WAGER_4LEG': (20, 120),
    'PLB_WAGER_5LEG': (15, 80),
}


# =========================================================================
# PLAYER MODEL — enhanced with pace, defense, and rest tracking
# =========================================================================

class NBAPlayerModel:
    def __init__(self, max_history=50):
        self.profiles = {}
        self.max_history = max_history
        # Team-level tracking for PAOV and OSAI
        self.team_pace = defaultdict(list)       # total points in games
        self.team_allowed = defaultdict(lambda: defaultdict(list))  # stats allowed by team
        self.league_stat_avgs = defaultdict(list)  # league-wide stat averages

    def update(self, name, stats, date, team, home_away, opponent):
        if name not in self.profiles:
            self.profiles[name] = {'games': [], 'team': '', 'dates': []}

        pts = _safe_int(stats.get('pts', 0))
        reb = _safe_int(stats.get('reb', 0))
        ast = _safe_int(stats.get('ast', 0))
        mins = _safe_int(stats.get('min', 0))
        three_str = stats.get('three', '0-0')
        if isinstance(three_str, str) and '-' in three_str:
            three_made = _safe_int(three_str.split('-')[0])
        else:
            three_made = _safe_int(three_str)

        game = {
            'date': date, 'team': team, 'opponent': opponent,
            'home_away': home_away,
            'pts': pts, 'reb': reb, 'ast': ast, 'min': mins,
            '3pm': three_made, 'pra': pts + reb + ast,
        }
        self.profiles[name]['games'].append(game)
        self.profiles[name]['team'] = team
        self.profiles[name]['dates'].append(date)
        if len(self.profiles[name]['games']) > self.max_history:
            self.profiles[name]['games'] = self.profiles[name]['games'][-self.max_history:]
            self.profiles[name]['dates'] = self.profiles[name]['dates'][-self.max_history:]

        # Track opponent defense stats (OSAI)
        if mins >= 15:
            self.team_allowed[opponent]['pts'].append(pts)
            self.team_allowed[opponent]['reb'].append(reb)
            self.team_allowed[opponent]['ast'].append(ast)
            self.team_allowed[opponent]['pra'].append(pts + reb + ast)

            # League-wide averages
            self.league_stat_avgs['pts'].append(pts)
            self.league_stat_avgs['reb'].append(reb)
            self.league_stat_avgs['ast'].append(ast)
            self.league_stat_avgs['pra'].append(pts + reb + ast)

    def update_game_pace(self, home_team, away_team, home_pts, away_pts):
        """Track pace from total game points."""
        total = home_pts + away_pts
        self.team_pace[home_team].append(total)
        self.team_pace[away_team].append(total)
        # Trim
        for t in [home_team, away_team]:
            if len(self.team_pace[t]) > 30:
                self.team_pace[t] = self.team_pace[t][-30:]

    def get_values(self, name, stat, window=None, min_min=15):
        p = self.profiles.get(name)
        if not p:
            return None
        games = [g for g in p['games'] if g.get('min', 0) >= min_min]
        if window:
            games = games[-window:]
        return [g.get(stat, 0) for g in games] if games else None

    def game_count(self, name):
        p = self.profiles.get(name)
        return len(p['games']) if p else 0

    def avg_minutes(self, name, window=5):
        vals = self.get_values(name, 'min', window, min_min=0)
        return sum(vals) / len(vals) if vals else 0

    def get_team(self, name):
        p = self.profiles.get(name)
        return p['team'] if p else None

    def get_minutes_cv(self, name, window=10):
        """Coefficient of variation of minutes (MSC)."""
        vals = self.get_values(name, 'min', window, min_min=0)
        if not vals or len(vals) < 5:
            return None
        mean = sum(vals) / len(vals)
        if mean < 5:
            return None
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return math.sqrt(var) / mean

    def get_rest_days(self, name, current_date):
        """Calculate days since last game (RDPD)."""
        p = self.profiles.get(name)
        if not p or not p['dates']:
            return None
        last_date = p['dates'][-1]
        try:
            # Parse YYYY-MM-DD format
            cy, cm, cd = int(current_date[:4]), int(current_date[5:7]), int(current_date[8:10])
            ly, lm, ld = int(last_date[:4]), int(last_date[5:7]), int(last_date[8:10])
            # Approximate day difference
            c_days = cy * 365 + cm * 30 + cd
            l_days = ly * 365 + lm * 30 + ld
            return max(0, c_days - l_days)
        except (ValueError, IndexError):
            return None

    def get_opponent_pace(self, opponent, window=15):
        """Get opponent's average pace (total points per game)."""
        vals = self.team_pace.get(opponent, [])
        if not vals or len(vals) < 3:
            return None
        recent = vals[-window:]
        return sum(recent) / len(recent)

    def get_league_avg_pace(self, window=100):
        """Get league average pace."""
        all_pace = []
        for team_vals in self.team_pace.values():
            all_pace.extend(team_vals[-20:])
        if not all_pace:
            return 210  # NBA average ~210 points/game
        return sum(all_pace) / len(all_pace)

    def get_opp_defense_avg(self, opponent, stat, window=20):
        """Get opponent's average allowed per player for a stat."""
        vals = self.team_allowed.get(opponent, {}).get(stat, [])
        if not vals:
            return None
        recent = vals[-window:]
        return sum(recent) / len(recent)

    def get_league_stat_avg(self, stat, window=200):
        """Get league-wide average for a stat."""
        vals = self.league_stat_avgs.get(stat, [])
        if not vals:
            return None
        recent = vals[-window:]
        return sum(recent) / len(recent)

    def get_streak_over_line(self, name, stat, line, lookback=7):
        """Count consecutive recent games clearing the line (HHM)."""
        vals = self.get_values(name, stat, lookback, min_min=10)
        if not vals:
            return 0, 0
        # Count streak from most recent backward
        streak = 0
        for v in reversed(vals):
            if v > line:
                streak += 1
            else:
                break
        hit_rate = sum(1 for v in vals if v > line) / len(vals)
        return streak, hit_rate


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
    values = model.get_values(name, stat, 20)
    if values is None or len(values) < 8:
        return None
    dist_ent = shannon_entropy(values, config['ESI_BINS'])
    mid = len(values) // 2
    m1 = sum(values[:mid]) / mid
    m2 = sum(values[mid:]) / (len(values) - mid)
    overall = sum(values) / len(values)
    if overall == 0:
        return None
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


# =========================================================================
# PROPRIETARY SIGNAL 1: Pace-Adjusted Opportunity Volume (PAOV)
# =========================================================================

def compute_paov(model, opponent, config):
    """
    Detect high-pace opponents that inflate stat opportunities.
    Games with more possessions produce more points, rebounds, assists.
    One-directional: only boosts, never penalizes.
    """
    if not config.get('PAOV_ENABLED', True):
        return 0.0

    opp_pace = model.get_opponent_pace(opponent)
    if opp_pace is None:
        return 0.0

    league_pace = model.get_league_avg_pace()
    pace_ratio = opp_pace / max(1, league_pace)

    if pace_ratio < config['PAOV_BOOST_THRESHOLD']:
        return 0.0

    # Scale boost by how far above threshold
    excess = pace_ratio - config['PAOV_BOOST_THRESHOLD']
    boost = min(config['PAOV_MAX_BOOST'], excess * 2.0)
    return boost * config['PAOV_WEIGHT']


# =========================================================================
# PROPRIETARY SIGNAL 2: Minutes Stability Coefficient (MSC)
# =========================================================================

def compute_msc(model, name, config):
    """
    Players with stable minutes (low coefficient of variation) produce
    more predictable stat output. Ideal for parlays requiring consistency.
    """
    if not config.get('MSC_ENABLED', True):
        return 0.0

    cv = model.get_minutes_cv(name, config.get('MSC_WINDOW', 10))
    if cv is None:
        return 0.0

    avg_min = model.avg_minutes(name, config.get('MSC_WINDOW', 10))
    if avg_min < config.get('MSC_MIN_AVG_MIN', 25):
        return 0.0

    max_cv = config['MSC_MAX_CV']
    elite_cv = config['MSC_ELITE_CV']

    if cv > max_cv:
        return 0.0

    if cv <= elite_cv:
        score = 1.0
    else:
        score = max(0, 1.0 - (cv - elite_cv) / (max_cv - elite_cv))

    return score * config['MSC_WEIGHT']


# =========================================================================
# PROPRIETARY SIGNAL 3: Rest-Day Performance Delta (RDPD)
# =========================================================================

def compute_rdpd(model, name, date, config):
    """
    NBA players perform differently based on rest.
    Back-to-backs: fatigue reduces output.
    2+ days rest: freshness boosts output.
    Asymmetric: penalize B2B, boost rest.
    """
    if not config.get('RDPD_ENABLED', True):
        return 0.0

    rest_days = model.get_rest_days(name, date)
    if rest_days is None:
        return 0.0

    if rest_days <= 1:
        # Back-to-back or same day (data error)
        return config.get('RDPD_B2B_PENALTY', -0.05) * config['RDPD_WEIGHT']
    elif rest_days >= 2:
        # Well-rested
        boost = min(config.get('RDPD_MAX_BOOST', 0.12),
                     config.get('RDPD_REST_BOOST', 0.08) * min(rest_days - 1, 3))
        return boost * config['RDPD_WEIGHT']
    return 0.0


# =========================================================================
# PROPRIETARY SIGNAL 4: Opponent Stat Allowance Index (OSAI)
# =========================================================================

def compute_osai(model, stat, opponent, config):
    """
    Track what each opponent allows per player for each stat category.
    Weak defenses (allow more than league avg) create exploitable spots.
    Asymmetric: 1.5x boost for weak defense, 0.5x penalty for strong.
    """
    if not config.get('OSAI_ENABLED', True):
        return 0.0

    opp_avg = model.get_opp_defense_avg(opponent, stat)
    league_avg = model.get_league_stat_avg(stat)

    if opp_avg is None or league_avg is None or league_avg == 0:
        return 0.0

    ratio = opp_avg / league_avg

    if ratio > 1.0:
        # Weak defense — boost
        excess = ratio - 1.0
        adj = min(config['OSAI_MAX_ADJ'],
                  excess * config.get('OSAI_BOOST_MULT', 1.5))
    else:
        # Strong defense — small penalty
        deficit = 1.0 - ratio
        adj = -min(config['OSAI_MAX_ADJ'],
                   deficit * config.get('OSAI_PENALTY_MULT', 0.5))

    return adj * config['OSAI_WEIGHT']


# =========================================================================
# PROPRIETARY SIGNAL 5: Hot Hand Momentum (HHM)
# =========================================================================

def compute_hhm(model, name, stat, line, config):
    """
    Players on streaks of clearing a line exhibit momentum persistence.
    Market is slow to adjust to hot streaks. Captures behavioral edge.
    Only boosts — no penalty for cold streaks (already captured by BEQ).
    """
    if not config.get('HHM_ENABLED', True):
        return 0.0

    lookback = config.get('HHM_LOOKBACK', 7)
    min_streak = config.get('HHM_MIN_STREAK', 2)

    streak, window_hr = model.get_streak_over_line(name, stat, line, lookback)

    if streak < min_streak:
        return 0.0

    # Score: weighted blend of streak length and window hit rate
    streak_score = min(1.0, streak / lookback)
    combined = streak_score * 0.6 + window_hr * 0.4

    return min(config.get('HHM_MAX_BOOST', 0.10),
               combined * config['HHM_WEIGHT'])


# =========================================================================
# COMBINED SIGNAL — with proprietary boosters
# =========================================================================

def compute_signal(model, name, stat, line, market_odds, config,
                   home_away=None, date=None, opponent=None):
    """Combined signal — ROI-focused with 5 proprietary NBA boosters."""
    if model.game_count(name) < config['MIN_GAMES']:
        return None
    if model.avg_minutes(name, 5) < config['MIN_MINUTES']:
        return None

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

    # Extended window consistency
    if beq['ext_hit_rate'] is not None:
        if beq['ext_hit_rate'] < beq['hit_rate'] - 0.15:
            return None

    # EV gate — CRITICAL for ROI
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

    # === PROPRIETARY SIGNAL BOOSTERS ===
    # These are additive — they can push borderline picks above thresholds
    boost_total = 0.0

    # PAOV: pace-adjusted opportunity
    if opponent:
        paov_boost = compute_paov(model, opponent, config)
        boost_total += paov_boost

    # MSC: minutes stability
    msc_boost = compute_msc(model, name, config)
    boost_total += msc_boost

    # RDPD: rest advantage
    if date:
        rdpd_boost = compute_rdpd(model, name, date, config)
        boost_total += rdpd_boost

    # OSAI: opponent defensive weakness
    if opponent:
        osai_boost = compute_osai(model, stat, opponent, config)
        boost_total += osai_boost

    # HHM: hot hand momentum
    hhm_boost = compute_hhm(model, name, stat, line, config)
    boost_total += hhm_boost

    # Apply boosters (capped to prevent runaway)
    combined = min(1.0, combined + boost_total)

    if combined < config['GATE_MIN_COMBINED']:
        return None

    kelly = kelly_fraction(beq['lower_bound'], market_odds, config['KELLY_FRACTION'])

    return {
        'player': name, 'stat': stat, 'line': line, 'odds': market_odds,
        'combined_score': combined,
        'gft_score': gft['score'], 'beq_edge': beq['edge'],
        'esi_stability': esi['stability'],
        'ev': ev, 'kelly': kelly,
        'hit_rate': beq['hit_rate'],
        'bayesian_prob': beq['mean_prob'],
        'true_prob': beq['lower_bound'],
        'market_prob': beq['market_prob'],
        'edge': beq['edge'],
        'boost_total': boost_total,
    }


# =========================================================================
# PARLAY CONSTRUCTION — expanded for 1000%+ ROI
# =========================================================================

def compute_correlation(model, n1, s1, n2, s2, window=20):
    v1, v2 = model.get_values(n1, s1, window), model.get_values(n2, s2, window)
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
    return cov / (s1v * s2v) if s1v > 0 and s2v > 0 else 0


def build_parlays(signals, model, config):
    """Build parlays from eligible signals — expanded to 5 legs for massive ROI."""
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
            if i == j or cand['player'] in used:
                continue
            if cand['player'] in {l['player'] for l in legs}:
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
                for l in legs:
                    used.add(l['player'])

    # Sort by EV (highest first) — prioritize highest-ROI parlays
    parlays.sort(key=lambda p: p['parlay_ev'], reverse=True)
    return parlays


# =========================================================================
# PIPELINE B: PLUS-MONEY PARLAY PIPELINE (Patent-Pending)
# =========================================================================
# Separate signal pipeline specifically for plus-money parlay legs.
# Key insight: plus-money lines where our model shows >55% true probability
# represent massive market mispricings. In parlays, these compound:
#   - 4 legs at +200 with 60% true prob each = 12.96% hit rate * 81x = 1050% ROI
#   - 5 legs at +150 with 60% true prob each = 7.78% hit rate * 97x = 655% ROI
# =========================================================================

def compute_plus_money_signal(model, name, stat, line, market_odds, config,
                               home_away=None, date=None, opponent=None):
    """
    Pipeline B signal — ONLY for plus-money lines (odds >= +100).
    Relaxed gates but requires HIGH true probability despite plus-money pricing.
    This is where the market mispricing is largest.
    """
    if market_odds < config.get('PLB_MIN_ODDS', 100):
        return None  # MUST be plus-money

    if model.game_count(name) < config['MIN_GAMES']:
        return None
    if model.avg_minutes(name, 5) < config['MIN_MINUTES']:
        return None

    # Bayesian probability — the core of Pipeline B
    beq = compute_beq(model, name, stat, line, market_odds, config)
    if not beq:
        return None

    # Key gate: true probability must be MUCH higher than market implies
    # Plus-money at +200 implies 33%. If our model says 55%+, that's a 22% edge.
    min_true_prob = config.get('PLB_MIN_TRUE_PROB', 0.52)
    if beq['lower_bound'] < min_true_prob:
        return None

    min_edge = config.get('PLB_MIN_EDGE', 0.08)
    if beq['edge'] < min_edge:
        return None

    # Hit rate gate — must have demonstrated clearing this line
    if beq['hit_rate'] < config.get('PLB_MIN_HIT_RATE', 0.50):
        return None

    # Extended consistency — prevents hot streak exploitation
    if beq['ext_hit_rate'] is not None:
        if beq['ext_hit_rate'] < beq['hit_rate'] - 0.12:
            return None

    # Light GFT check (just needs to be clearing the line)
    gft = compute_gft(model, name, stat, line, config)
    if not gft or gft['score'] < 0.02:
        return None

    # EV must be strong for plus-money
    ev = expected_value(beq['lower_bound'], market_odds)
    if ev < config.get('PLB_MIN_EV', 0.10):
        return None

    # Proprietary boosters
    boost = 0.0
    if opponent:
        boost += compute_paov(model, opponent, config)
        boost += compute_osai(model, stat, opponent, config)
    boost += compute_msc(model, name, config)
    if date:
        boost += compute_rdpd(model, name, date, config)
    boost += compute_hhm(model, name, stat, line, config)

    # Combined score for ranking
    combined = min(1.0, beq['edge'] * 3.0 + gft['score'] * 0.3 + boost)

    kelly = kelly_fraction(beq['lower_bound'], market_odds, config['KELLY_FRACTION'])

    return {
        'player': name, 'stat': stat, 'line': line, 'odds': market_odds,
        'combined_score': combined,
        'gft_score': gft['score'], 'beq_edge': beq['edge'],
        'esi_stability': 0,
        'ev': ev, 'kelly': kelly,
        'hit_rate': beq['hit_rate'],
        'bayesian_prob': beq['mean_prob'],
        'true_prob': beq['lower_bound'],
        'market_prob': beq['market_prob'],
        'edge': beq['edge'],
        'boost_total': boost,
        'pipeline': 'B',
    }


def build_plus_money_parlays(signals, model, config):
    """
    Build high-multiplier parlays from plus-money legs only.
    Targets 3-5 leg parlays for maximum payout multiplier.
    All legs must be from different teams and different games.
    """
    min_legs = config.get('PLB_PARLAY_MIN_LEGS', 3)
    max_legs = config.get('PLB_PARLAY_MAX_LEGS', 5)
    max_corr = config.get('PLB_PARLAY_MAX_CORR', 0.30)

    if len(signals) < min_legs:
        return []

    # Sort by edge * EV product (best value legs first)
    sorted_sigs = sorted(signals, key=lambda s: s['ev'] * (1 + s['edge']), reverse=True)
    parlays = []
    used = set()

    # Try to build multiple parlays
    for attempt in range(config.get('PLB_MAX_PARLAYS', 3)):
        available = [s for s in sorted_sigs if s['player'] not in used]
        if len(available) < min_legs:
            break

        # Greedy construction: pick best uncorrelated legs
        legs = [available[0]]
        leg_teams = {model.get_team(available[0]['player'])}
        leg_games = {available[0].get('game_key', '')}

        for cand in available[1:]:
            if len(legs) >= max_legs:
                break
            ct = model.get_team(cand['player'])
            cg = cand.get('game_key', '')

            # Different team AND different game (true independence)
            if ct in leg_teams:
                continue
            if cg and cg in leg_games:
                continue

            # Correlation check
            max_c = max(abs(compute_correlation(model, l['player'], l['stat'],
                                                 cand['player'], cand['stat']))
                       for l in legs)
            if max_c > max_corr:
                continue

            legs.append(cand)
            if ct:
                leg_teams.add(ct)
            if cg:
                leg_games.add(cg)

        if len(legs) >= min_legs:
            combo_prob = 1.0
            total_edge = 0
            for l in legs:
                combo_prob *= l['true_prob']
                total_edge += l['edge']

            p_dec = parlay_decimal_odds([l['odds'] for l in legs])
            p_ev = combo_prob * p_dec - 1

            if p_ev > 0:
                parlays.append({
                    'legs': legs, 'n_legs': len(legs),
                    'parlay_decimal': p_dec,
                    'parlay_american': decimal_to_american(p_dec),
                    'combined_prob': combo_prob, 'parlay_ev': p_ev,
                    'total_edge': total_edge,
                    'pipeline': 'B',
                })
                for l in legs:
                    used.add(l['player'])

    parlays.sort(key=lambda p: p['parlay_ev'], reverse=True)
    return parlays


def _parlay_wager(n_legs, config):
    """Tiered wager sizing by leg count — smaller bets on longer parlays."""
    key = f'PARLAY_WAGER_{n_legs}LEG'
    return config.get(key, config['UNIT_SIZE'])


# =========================================================================
# WALK-FORWARD BACKTEST
# =========================================================================

def run_backtest(box_scores, odds_data, config, verbose=False):
    model = NBAPlayerModel()
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

                # All prop types
                for prop_key, stat_key, actual_key in [
                    ('playerProps', 'pts', 'pts'),
                    ('rebProps', 'reb', 'reb'),
                    ('astProps', 'ast', 'ast'),
                ]:
                    props = og.get(prop_key, {}).get(name, {})
                    for thr, data in props.items():
                        _try_signal(model, name, stat_key, thr, data, config, ha,
                                   player, actual_key, day_signals, date, gk, opponent)

                # PRA props
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

            # Singles — top EV picks
            singles = [s for s in unique if s['combined_score'] >= config['SINGLE_MIN_SCORE']]
            for s in singles[:config['MAX_DAILY_BETS']]:
                wager = max(50, min(400, round(config['UNIT_SIZE'] * (1 + s['kelly'] * 2))))
                pnl = pnl_for_bet(s['hit'], s['odds'], wager)
                all_picks.append({**s, 'bet_type': 'single', 'pnl': pnl, 'wager': wager})

            # Pipeline A: Standard parlays
            parlay_elig = [s for s in unique if s['combined_score'] >= config['PARLAY_LEG_MIN_SCORE']]
            parlays = build_parlays(parlay_elig, model, config)
            max_parlays = config.get('PARLAY_MAX_PER_DAY', 3)
            for p in parlays[:max_parlays]:
                hit = all(l['hit'] for l in p['legs'])
                wager = _parlay_wager(p['n_legs'], config)
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

        # === PIPELINE B: Plus-Money Parlays ===
        if config.get('PLB_ENABLED', True) and day_signals:
            plb_signals = []
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
                            try:
                                line = float(thr)
                            except (ValueError, TypeError):
                                continue
                            odds_val = data.get('overOdds')
                            if odds_val is None:
                                continue
                            if odds_val < config.get('PLB_MIN_ODDS', 100):
                                continue
                            if odds_val > config.get('PLB_MAX_ODDS', 500):
                                continue
                            sig = compute_plus_money_signal(
                                model, name, stat_key, line, odds_val, config,
                                home_away=ha, date=date, opponent=opponent)
                            if sig:
                                actual = _safe_int(player.get(actual_key, 0))
                                sig['actual'] = actual
                                sig['hit'] = actual > line
                                sig['date'] = date
                                sig['game_key'] = gk
                                plb_signals.append(sig)

                    # PRA for Pipeline B
                    pra_p = og.get('praProps', {}).get(name, {})
                    for thr, data in pra_p.items():
                        try:
                            line = float(thr)
                        except (ValueError, TypeError):
                            continue
                        odds_val = data.get('overOdds')
                        if odds_val is None:
                            continue
                        if odds_val < config.get('PLB_MIN_ODDS', 100):
                            continue
                        if odds_val > config.get('PLB_MAX_ODDS', 500):
                            continue
                        sig = compute_plus_money_signal(
                            model, name, 'pra', line, odds_val, config,
                            home_away=ha, date=date, opponent=opponent)
                        if sig:
                            actual = (_safe_int(player.get('pts', 0)) +
                                      _safe_int(player.get('reb', 0)) +
                                      _safe_int(player.get('ast', 0)))
                            sig['actual'] = actual
                            sig['hit'] = actual > line
                            sig['date'] = date
                            sig['game_key'] = gk
                            plb_signals.append(sig)

            if plb_signals:
                # Deduplicate by player (best signal per player)
                plb_by_player = {}
                for s in plb_signals:
                    if s['player'] not in plb_by_player or s['ev'] > plb_by_player[s['player']]['ev']:
                        plb_by_player[s['player']] = s
                plb_unique = list(plb_by_player.values())

                plb_parlays = build_plus_money_parlays(plb_unique, model, config)
                for p in plb_parlays:
                    hit = all(l['hit'] for l in p['legs'])
                    wager_key = f"PLB_WAGER_{p['n_legs']}LEG"
                    wager = config.get(wager_key, 50)
                    pnl = round((p['parlay_decimal'] - 1) * wager) if hit else -wager
                    all_picks.append({
                        'bet_type': 'parlay_plb', 'n_legs': p['n_legs'],
                        'legs': [{'player': l['player'], 'stat': l['stat'],
                                  'line': l['line'], 'odds': l['odds'],
                                  'hit': l['hit'], 'actual': l.get('actual'),
                                  'edge': l['edge']} for l in p['legs']],
                        'hit': hit, 'pnl': pnl, 'wager': wager,
                        'date': date, 'parlay_decimal': p['parlay_decimal'],
                        'parlay_ev': p['parlay_ev'], 'total_edge': p['total_edge'],
                        'pipeline': 'B',
                    })

        if day_signals or (config.get('PLB_ENABLED') and day_signals):
            day_picks = [p for p in all_picks if p.get('date') == date]
            if day_picks:
                daily_results.append({
                    'date': date, 'n_picks': len(day_picks),
                    'wins': sum(1 for p in day_picks if p.get('hit')),
                    'pnl': sum(p.get('pnl', 0) for p in day_picks),
                })

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
    sig = compute_signal(model, name, stat, line, odds_val, config,
                         home_away=ha, date=date, opponent=opponent)
    if sig:
        actual = _safe_int(player.get(actual_key, 0))
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
    sig = compute_signal(model, name, 'pra', line, odds_val, config,
                         home_away=ha, date=date, opponent=opponent)
    if sig:
        actual = (_safe_int(player.get('pts', 0)) +
                  _safe_int(player.get('reb', 0)) +
                  _safe_int(player.get('ast', 0)))
        sig['actual'] = actual
        sig['hit'] = actual > line
        sig['date'] = date
        sig['game_key'] = gk
        signals.append(sig)


def _update_model(model, game_data, date):
    home, away = game_data.get('home', ''), game_data.get('away', '')

    # Track game pace from total team scores
    home_pts = sum(_safe_int(p.get('pts', 0)) for p in game_data.get('players', [])
                   if p.get('team', '') == home)
    away_pts = sum(_safe_int(p.get('pts', 0)) for p in game_data.get('players', [])
                   if p.get('team', '') == away)
    if home_pts > 0 and away_pts > 0:
        model.update_game_pace(home, away, home_pts, away_pts)

    for p in game_data.get('players', []):
        if _safe_int(p.get('min', 0)) < 5:
            continue
        team = p.get('team', '')
        ha = 'home' if team == home else 'away'
        opp = away if team == home else home
        model.update(p['name'], p, date, team, ha, opp)


def _compute_stats(all_picks, daily_results, config):
    singles = [p for p in all_picks if p.get('bet_type') == 'single']
    parlays = [p for p in all_picks if p.get('bet_type') in ('parlay', 'parlay_plb')]
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

    # Parlay breakdown by leg count
    parlay_by_legs = {}
    for p in parlays:
        n = p.get('n_legs', 0)
        if n not in parlay_by_legs:
            parlay_by_legs[n] = {'total': 0, 'wins': 0, 'pnl': 0}
        parlay_by_legs[n]['total'] += 1
        if p.get('hit'):
            parlay_by_legs[n]['wins'] += 1
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
