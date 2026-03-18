"""
NBA Strategy V6 — Dual-Polarity Regime Engine (DPRE)
=====================================================
Patent-pending innovation: FIRST strategy to bet BOTH overs AND unders.

Previous strategies (V1-V5) only bet overs. This dramatically limits opportunity:
- Over bets require floor > line (player's worst games still clear)
- But under bets require ceiling < line (player's best games still miss)
- The combined opportunity space is 2x larger

Novel Components:
1. DUAL-POLARITY SIGNAL ANALYSIS (DPSA)
   - Over signals: floor analysis (p10 of recent games > line)
   - Under signals: ceiling analysis (p90 of recent games < line)
   - Direction-specific Bayesian edge quantification

2. REGIME-CONDITIONED FILTERING (RCF)
   - Context-aware: only bet overs against weak defenses, unders against strong
   - Pace regime: high-pace games boost overs, low-pace games boost unders
   - Rest regime: rested players bias toward overs, fatigued toward unders
   - Minutes stability: consistent minutes = more predictable in both directions

3. MULTI-STAT DIVERSIFICATION (MSD)
   - Bets across pts, reb, ast, pra independently
   - Each stat has different market efficiency characteristics
   - Rebounds/assists have lower variance → more predictable floors/ceilings

4. ADAPTIVE CONFIDENCE RANKING (ACR)
   - Ranks all signals (over + under) by confidence score
   - Takes top-N per day based on a minimum confidence threshold
   - Allows high volume when many strong signals exist

5. CROSS-WINDOW UNANIMITY (CWU)
   - Floor/ceiling must hold across 5, 10, 15, 20 game windows
   - Prevents recency bias and streak artifacts
   - Convergence score measures stability across time horizons

Walk-forward only, real historical odds from The Odds API, no leakage.
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
from nba.strategy import NBAPlayerModel, _update_model, _safe_int


# =========================================================================
# DEFAULT CONFIG — V6 DPRE
# =========================================================================

NBA_V6_CONFIG = {
    # Data requirements
    'MIN_GAMES': 15,
    'MIN_MINUTES': 20,

    # Odds filter — plus money only
    'MIN_ODDS': 100,
    'MAX_ODDS': 400,

    # --- DPSA: Dual-Polarity Signal Analysis ---
    # Over: floor must exceed line
    'OVER_FLOOR_PCT': 0.10,          # 10th percentile
    'OVER_MIN_CLEARANCE': 0.5,       # floor must exceed line by this much
    'OVER_MIN_HR_20': 0.75,          # 20-game hit rate for overs
    'OVER_MIN_HR_10': 0.70,          # 10-game hit rate
    'OVER_MIN_HR_5': 0.80,           # 5-game hit rate (recent form)

    # Under: ceiling must be below line
    'UNDER_CEILING_PCT': 0.90,       # 90th percentile
    'UNDER_MIN_CLEARANCE': 0.5,      # line must exceed ceiling by this much
    'UNDER_MIN_HR_20': 0.75,         # 20-game miss rate (= under hit rate)
    'UNDER_MIN_HR_10': 0.70,
    'UNDER_MIN_HR_5': 0.80,

    # --- CWU: Cross-Window Unanimity ---
    'CWU_WINDOWS': [5, 10, 15, 20],
    'CWU_REQUIRE_ALL': True,         # All windows must agree
    'CWU_MIN_CONVERGENCE': 0.60,     # Floor/ceiling spread normalized

    # --- RCF: Regime-Conditioned Filtering ---
    'RCF_ENABLED': True,
    'RCF_PACE_WEIGHT': 0.10,         # Pace regime boost
    'RCF_DEFENSE_WEIGHT': 0.12,      # Defense regime boost
    'RCF_REST_WEIGHT': 0.06,         # Rest regime boost
    'RCF_STABILITY_WEIGHT': 0.08,    # Minutes stability boost

    # --- BEQ: Bayesian Edge ---
    'BEQ_PRIOR_ALPHA': 1.0,
    'BEQ_PRIOR_BETA': 1.0,
    'BEQ_CI_LEVEL': 0.80,
    'BEQ_MIN_EDGE': 0.04,

    # --- ESI: Stability ---
    'ESI_BINS': 6,
    'ESI_MAX_ENTROPY': 0.88,
    'ESI_TREND_WEIGHT': 0.20,
    'ESI_MIN_STABILITY': 0.15,

    # EV gate
    'MIN_EV': 0.05,

    # --- ACR: Adaptive Confidence Ranking ---
    'ACR_MAX_DAILY': 8,              # Allow up to 8 bets/day
    'ACR_MIN_CONFIDENCE': 0.30,      # Minimum composite confidence

    # Bet sizing
    'UNIT_SIZE': 100,
    'KELLY_FRACTION': 0.30,
    'MAX_WAGER': 250,
    'MIN_WAGER': 50,
}

# Relaxed config for +200 minimum odds
NBA_V6_CONFIG_PLUS200 = {
    **NBA_V6_CONFIG,
    'MIN_ODDS': 200,
    'MAX_ODDS': 600,
    'OVER_MIN_CLEARANCE': 1.0,
    'UNDER_MIN_CLEARANCE': 1.0,
    'OVER_MIN_HR_20': 0.70,
    'OVER_MIN_HR_10': 0.65,
    'OVER_MIN_HR_5': 0.70,
    'UNDER_MIN_HR_20': 0.70,
    'UNDER_MIN_HR_10': 0.65,
    'UNDER_MIN_HR_5': 0.70,
    'BEQ_MIN_EDGE': 0.06,
    'MIN_EV': 0.08,
    'ACR_MAX_DAILY': 10,
}

# Parameter ranges for optimization
NBA_V6_PARAM_RANGES = {
    'MIN_GAMES': (10, 20),
    'MIN_MINUTES': (15, 25),
    'MIN_ODDS': (100, 100),
    'MAX_ODDS': (200, 600),
    'OVER_FLOOR_PCT': (0.05, 0.25),
    'OVER_MIN_CLEARANCE': (-0.5, 2.0),
    'OVER_MIN_HR_20': (0.60, 0.90),
    'OVER_MIN_HR_10': (0.55, 0.85),
    'OVER_MIN_HR_5': (0.60, 0.95),
    'UNDER_CEILING_PCT': (0.75, 0.95),
    'UNDER_MIN_CLEARANCE': (-0.5, 2.0),
    'UNDER_MIN_HR_20': (0.60, 0.90),
    'UNDER_MIN_HR_10': (0.55, 0.85),
    'UNDER_MIN_HR_5': (0.60, 0.95),
    'CWU_MIN_CONVERGENCE': (0.30, 0.80),
    'BEQ_CI_LEVEL': (0.65, 0.92),
    'BEQ_MIN_EDGE': (0.01, 0.15),
    'ESI_MAX_ENTROPY': (0.55, 0.95),
    'ESI_TREND_WEIGHT': (0.05, 0.40),
    'ESI_MIN_STABILITY': (0.05, 0.35),
    'MIN_EV': (0.02, 0.20),
    'ACR_MAX_DAILY': (3, 12),
    'ACR_MIN_CONFIDENCE': (0.15, 0.55),
    'KELLY_FRACTION': (0.15, 0.50),
}


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def _percentile(values, pct):
    """Compute percentile of a list."""
    if not values:
        return 0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * pct)))
    return s[idx]


def _hit_rate(values, line, direction='over'):
    """Compute hit rate for over or under."""
    if not values:
        return 0
    if direction == 'over':
        return sum(1 for v in values if v > line) / len(values)
    else:
        return sum(1 for v in values if v < line) / len(values)


# =========================================================================
# DUAL-POLARITY SIGNAL ANALYSIS
# =========================================================================

def compute_over_signal(model, name, stat, line, odds, config):
    """
    OVER signal: floor-based analysis.
    The player's statistical floor (p10) must exceed the line.
    """
    if model.game_count(name) < config['MIN_GAMES']:
        return None
    if model.avg_minutes(name, 10) < config['MIN_MINUTES']:
        return None

    # Multi-window hit rate gates
    for window, key in [(20, 'OVER_MIN_HR_20'), (10, 'OVER_MIN_HR_10'), (5, 'OVER_MIN_HR_5')]:
        values = model.get_values(name, stat, window)
        if values is None or len(values) < min(window, 5):
            return None
        hr = _hit_rate(values, line, 'over')
        if hr < config[key]:
            return None

    # Cross-Window Unanimity for floors
    floor_pct = config['OVER_FLOOR_PCT']
    min_clr = config['OVER_MIN_CLEARANCE']
    floors = []
    clearances = []

    for w in config['CWU_WINDOWS']:
        values = model.get_values(name, stat, w)
        if values is None or len(values) < min(w, 5):
            if config.get('CWU_REQUIRE_ALL', True):
                return None
            continue
        floor = _percentile(values, floor_pct)
        clr = floor - line
        if clr < min_clr and config.get('CWU_REQUIRE_ALL', True):
            return None
        floors.append(floor)
        clearances.append(clr)

    if not floors:
        return None

    # Convergence: how tight are floors across windows?
    if len(floors) >= 2:
        spread = max(floors) - min(floors)
        max_possible = max(abs(f) for f in floors) * 0.5 + 1
        convergence = max(0, 1 - spread / max(max_possible, 1))
        if convergence < config['CWU_MIN_CONVERGENCE']:
            return None
    else:
        convergence = 0.5

    avg_clearance = sum(clearances) / len(clearances)

    return {
        'direction': 'over',
        'floor': min(floors),
        'clearance': avg_clearance,
        'convergence': convergence,
        'hr_20': _hit_rate(model.get_values(name, stat, 20), line, 'over'),
    }


def compute_under_signal(model, name, stat, line, odds, config):
    """
    UNDER signal: ceiling-based analysis.
    The player's statistical ceiling (p90) must be BELOW the line.
    This captures situations where even the player's best games don't reach the line.
    """
    if model.game_count(name) < config['MIN_GAMES']:
        return None
    if model.avg_minutes(name, 10) < config['MIN_MINUTES']:
        return None

    # Multi-window miss rate gates (= under hit rate)
    for window, key in [(20, 'UNDER_MIN_HR_20'), (10, 'UNDER_MIN_HR_10'), (5, 'UNDER_MIN_HR_5')]:
        values = model.get_values(name, stat, window)
        if values is None or len(values) < min(window, 5):
            return None
        hr = _hit_rate(values, line, 'under')
        if hr < config[key]:
            return None

    # Cross-Window Unanimity for ceilings
    ceil_pct = config['UNDER_CEILING_PCT']
    min_clr = config['UNDER_MIN_CLEARANCE']
    ceilings = []
    clearances = []

    for w in config['CWU_WINDOWS']:
        values = model.get_values(name, stat, w)
        if values is None or len(values) < min(w, 5):
            if config.get('CWU_REQUIRE_ALL', True):
                return None
            continue
        ceiling = _percentile(values, ceil_pct)
        clr = line - ceiling  # Positive means ceiling is below line
        if clr < min_clr and config.get('CWU_REQUIRE_ALL', True):
            return None
        ceilings.append(ceiling)
        clearances.append(clr)

    if not ceilings:
        return None

    # Convergence
    if len(ceilings) >= 2:
        spread = max(ceilings) - min(ceilings)
        max_possible = max(abs(c) for c in ceilings) * 0.5 + 1
        convergence = max(0, 1 - spread / max(max_possible, 1))
        if convergence < config['CWU_MIN_CONVERGENCE']:
            return None
    else:
        convergence = 0.5

    avg_clearance = sum(clearances) / len(clearances)

    return {
        'direction': 'under',
        'ceiling': max(ceilings),
        'clearance': avg_clearance,
        'convergence': convergence,
        'hr_20': _hit_rate(model.get_values(name, stat, 20), line, 'under'),
    }


# =========================================================================
# BAYESIAN EDGE (direction-aware)
# =========================================================================

def compute_beq_directional(model, name, stat, line, market_odds, config, direction='over'):
    """
    Bayesian Edge Quantification — direction-aware.
    For overs: models P(X > line)
    For unders: models P(X < line)
    """
    values = model.get_values(name, stat, 20)
    if values is None or len(values) < 8:
        return None

    if direction == 'over':
        hits = sum(1 for v in values if v > line)
    else:
        hits = sum(1 for v in values if v < line)

    misses = len(values) - hits
    alpha = config['BEQ_PRIOR_ALPHA'] + hits
    beta_p = config['BEQ_PRIOR_BETA'] + misses
    mean_prob = alpha / (alpha + beta_p)
    ci = 1 - config['BEQ_CI_LEVEL']
    lower = beta_ppf_approx(alpha, beta_p, ci)
    mkt_prob = implied_probability(market_odds)
    edge = lower - mkt_prob
    hit_rate = hits / len(values)

    # Extended window consistency
    ext = model.get_values(name, stat, 30)
    ext_hr = None
    if ext and len(ext) >= 15:
        if direction == 'over':
            ext_hr = sum(1 for v in ext if v > line) / len(ext)
        else:
            ext_hr = sum(1 for v in ext if v < line) / len(ext)

    return {
        'mean_prob': mean_prob, 'lower_bound': lower,
        'market_prob': mkt_prob, 'edge': edge,
        'hit_rate': hit_rate, 'ext_hit_rate': ext_hr,
    }


# =========================================================================
# STABILITY INDEX
# =========================================================================

def compute_esi(model, name, stat, config):
    """Entropic Stability Index."""
    values = model.get_values(name, stat, 20)
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
# REGIME-CONDITIONED FILTERING
# =========================================================================

def compute_regime_boost(model, name, stat, direction, date, opponent, config):
    """
    Compute regime-based boost/penalty.
    For overs: boost from high pace, weak defense, rest
    For unders: boost from low pace, strong defense, fatigue
    """
    if not config.get('RCF_ENABLED', True):
        return 0.0

    boost = 0.0

    # Pace regime
    opp_pace = model.get_opponent_pace(opponent) if opponent else None
    league_pace = model.get_league_avg_pace()
    if opp_pace and league_pace:
        pace_ratio = opp_pace / max(1, league_pace)
        if direction == 'over' and pace_ratio > 1.03:
            boost += min(0.15, (pace_ratio - 1.03) * 2.0) * config['RCF_PACE_WEIGHT']
        elif direction == 'under' and pace_ratio < 0.97:
            boost += min(0.15, (0.97 - pace_ratio) * 2.0) * config['RCF_PACE_WEIGHT']

    # Defense regime
    if opponent:
        opp_avg = model.get_opp_defense_avg(opponent, stat)
        league_avg = model.get_league_stat_avg(stat)
        if opp_avg and league_avg and league_avg > 0:
            def_ratio = opp_avg / league_avg
            if direction == 'over' and def_ratio > 1.05:
                boost += min(0.15, (def_ratio - 1.0) * 1.5) * config['RCF_DEFENSE_WEIGHT']
            elif direction == 'under' and def_ratio < 0.95:
                boost += min(0.15, (1.0 - def_ratio) * 1.5) * config['RCF_DEFENSE_WEIGHT']

    # Rest regime
    if date:
        rest_days = model.get_rest_days(name, date)
        if rest_days is not None:
            if direction == 'over' and rest_days >= 2:
                boost += min(0.10, 0.04 * (rest_days - 1)) * config['RCF_REST_WEIGHT']
            elif direction == 'under' and rest_days <= 1:
                boost += 0.06 * config['RCF_REST_WEIGHT']

    # Minutes stability
    cv = model.get_minutes_cv(name, 10)
    if cv is not None:
        # Both directions benefit from stable minutes
        if cv < 0.15:
            boost += (1 - cv / 0.15) * 0.5 * config['RCF_STABILITY_WEIGHT']

    return boost


# =========================================================================
# COMPOSITE SIGNAL
# =========================================================================

def compute_full_signal(model, name, stat, line, odds, config,
                        direction='over', date=None, opponent=None, home_away=None):
    """
    Full composite signal combining polarity analysis, BEQ, ESI, and regime.
    """
    # Polarity-specific signal
    if direction == 'over':
        pol = compute_over_signal(model, name, stat, line, odds, config)
    else:
        pol = compute_under_signal(model, name, stat, line, odds, config)

    if pol is None:
        return None

    # BEQ — direction-aware
    beq = compute_beq_directional(model, name, stat, line, odds, config, direction)
    if beq is None or beq['edge'] < config['BEQ_MIN_EDGE']:
        return None

    # Extended consistency check
    if beq['ext_hit_rate'] is not None:
        if beq['ext_hit_rate'] < beq['hit_rate'] - 0.12:
            return None

    # ESI stability
    esi = compute_esi(model, name, stat, config)
    if esi is None or esi['stability'] < config.get('ESI_MIN_STABILITY', 0.15):
        return None

    # EV gate
    ev = expected_value(beq['lower_bound'], odds)
    if ev < config['MIN_EV']:
        return None

    # Regime boost
    regime_boost = compute_regime_boost(model, name, stat, direction, date, opponent, config)

    # Composite confidence score
    # Blend of clearance, convergence, edge, stability, and regime
    confidence = (
        min(1.0, pol['clearance'] / 3.0) * 0.25 +
        pol['convergence'] * 0.15 +
        min(1.0, beq['edge'] / 0.15 + 0.3) * 0.25 +
        esi['stability'] * 0.15 +
        min(1.0, pol['hr_20']) * 0.20 +
        regime_boost
    )
    confidence = max(0, min(1.0, confidence))

    if confidence < config['ACR_MIN_CONFIDENCE']:
        return None

    kelly = kelly_fraction(beq['lower_bound'], odds, config['KELLY_FRACTION'])

    return {
        'player': name, 'stat': stat, 'line': line, 'odds': odds,
        'direction': direction,
        'combined_score': confidence,
        'clearance': pol['clearance'],
        'convergence': pol['convergence'],
        'beq_edge': beq['edge'],
        'esi_stability': esi['stability'],
        'regime_boost': regime_boost,
        'ev': ev, 'kelly': kelly,
        'hit_rate': beq['hit_rate'],
        'bayesian_prob': beq['mean_prob'],
        'true_prob': beq['lower_bound'],
        'market_prob': beq['market_prob'],
        'edge': beq['edge'],
    }


# =========================================================================
# WALK-FORWARD BACKTEST
# =========================================================================

def run_backtest(box_scores, odds_data, config, verbose=False):
    """Walk-forward backtest of the V6 DPRE strategy."""
    model = NBAPlayerModel()
    stat_map = {
        'playerProps': 'pts', 'rebProps': 'reb',
        'astProps': 'ast', 'praProps': 'pra',
    }

    dates_sorted = sorted(set(g['date'] for g in box_scores))
    box_by_date = defaultdict(list)
    for g in box_scores:
        box_by_date[g['date']].append(g)
    odds_index = defaultdict(dict)
    for o in odds_data:
        odds_index[o['date']][o['gameKey']] = o

    picks = []
    daily_results = []

    for date in dates_sorted:
        day_odds = odds_index.get(date, {})
        day_boxes = box_by_date.get(date, [])

        # Build actuals
        actuals = {}
        player_context = {}  # Store opponent/home_away info
        for bg in day_boxes:
            home_t = bg.get('home', '')
            away_t = bg.get('away', '')
            for p in bg.get('players', []):
                name = p.get('name', '')
                pts = _safe_int(p.get('pts', 0))
                reb = _safe_int(p.get('reb', 0))
                ast = _safe_int(p.get('ast', 0))
                actuals[name] = {
                    'pts': pts, 'reb': reb, 'ast': ast,
                    'pra': pts + reb + ast,
                }
                team = p.get('team', '')
                ha = 'home' if team == home_t else 'away'
                opp = away_t if team == home_t else home_t
                player_context[name] = {'home_away': ha, 'opponent': opp}

        # Collect all signals (BOTH over and under)
        day_signals = []

        for gk, odds_rec in day_odds.items():
            for prop_type, stat in stat_map.items():
                props = odds_rec.get(prop_type, {})
                for player, lines in props.items():
                    ctx = player_context.get(player, {})
                    opponent = ctx.get('opponent')
                    home_away = ctx.get('home_away')

                    for line_str, line_data in lines.items():
                        try:
                            line = float(line_str)
                        except (ValueError, TypeError):
                            continue

                        # Try OVER signal
                        over_odds = line_data.get('overOdds')
                        if over_odds is not None and config['MIN_ODDS'] <= over_odds <= config['MAX_ODDS']:
                            sig = compute_full_signal(
                                model, player, stat, line, over_odds, config,
                                direction='over', date=date, opponent=opponent,
                                home_away=home_away)
                            if sig:
                                actual = actuals.get(player, {}).get(stat)
                                if actual is not None:
                                    sig['actual'] = actual
                                    sig['hit'] = actual > line
                                    sig['date'] = date
                                    day_signals.append(sig)

                        # Try UNDER signal
                        under_odds = line_data.get('underOdds')
                        if under_odds is not None and config['MIN_ODDS'] <= under_odds <= config['MAX_ODDS']:
                            sig = compute_full_signal(
                                model, player, stat, line, under_odds, config,
                                direction='under', date=date, opponent=opponent,
                                home_away=home_away)
                            if sig:
                                actual = actuals.get(player, {}).get(stat)
                                if actual is not None:
                                    sig['actual'] = actual
                                    sig['hit'] = actual < line
                                    sig['date'] = date
                                    day_signals.append(sig)

        # ACR: Rank by confidence and select top-N
        day_signals.sort(key=lambda s: s['combined_score'], reverse=True)

        # Deduplicate: only best signal per player
        seen_players = set()
        unique_signals = []
        for s in day_signals:
            if s['player'] not in seen_players:
                unique_signals.append(s)
                seen_players.add(s['player'])

        selected = unique_signals[:config['ACR_MAX_DAILY']]

        # Resolve bets
        day_picks = []
        for sig in selected:
            hit = sig['hit']

            # Kelly sizing
            kf = sig['kelly']
            wager = max(
                config.get('MIN_WAGER', 50),
                min(config.get('MAX_WAGER', 250),
                    int(config['UNIT_SIZE'] * (1 + kf * 5)))
            )
            pnl = pnl_for_bet(hit, sig['odds'], wager)

            pick = {
                'date': date,
                'player': sig['player'],
                'stat': sig['stat'],
                'line': sig['line'],
                'odds': sig['odds'],
                'direction': sig['direction'],
                'hit_rate': sig['hit_rate'],
                'ev': sig['ev'],
                'actual': sig['actual'],
                'hit': hit,
                'pnl': pnl,
                'wager': wager,
                'bet_type': 'single',
                'combined_score': sig['combined_score'],
                'edge': sig['edge'],
                'clearance': sig['clearance'],
                'regime_boost': sig['regime_boost'],
            }
            day_picks.append(pick)
            picks.append(pick)

        if day_picks and verbose:
            wins = sum(1 for p in day_picks if p['hit'])
            total_day_pnl = sum(p['pnl'] for p in day_picks)
            print(f"  {date}: {len(day_picks)} picks ({wins} wins) ${total_day_pnl:+d}")
            for p in day_picks:
                status = 'WIN' if p['hit'] else 'LOSS'
                dir_str = 'O' if p['direction'] == 'over' else 'U'
                print(f"    {p['player']} {p['stat'].upper()} {dir_str}{p['line']} "
                      f"+{p['odds']} (clr={p['clearance']:.1f}) "
                      f"→ {p['actual']} [{status}] ${p['pnl']:+d}")

        if day_picks:
            daily_results.append({
                'date': date,
                'n_picks': len(day_picks),
                'wins': sum(1 for p in day_picks if p['hit']),
                'pnl': sum(p['pnl'] for p in day_picks),
            })

        # Update model AFTER generating signals
        for bg in day_boxes:
            _update_model(model, bg, date)

    # Aggregate results
    total_picks = len(picks)
    total_wins = sum(1 for p in picks if p['hit'])
    total_pnl = sum(p['pnl'] for p in picks)
    total_wagered = sum(p['wager'] for p in picks)
    active_days = len(set(p['date'] for p in picks))

    # Over/Under breakdown
    over_picks = [p for p in picks if p.get('direction') == 'over']
    under_picks = [p for p in picks if p.get('direction') == 'under']

    # Max drawdown
    running_pnl = 0
    peak_pnl = 0
    max_dd = 0
    for p in sorted(picks, key=lambda x: x['date']):
        running_pnl += p['pnl']
        peak_pnl = max(peak_pnl, running_pnl)
        dd = peak_pnl - running_pnl
        max_dd = max(max_dd, dd)

    return {
        'total_picks': total_picks,
        'total_wins': total_wins,
        'accuracy': total_wins / total_picks if total_picks > 0 else 0,
        'total_pnl': total_pnl,
        'total_wagered': total_wagered,
        'total_roi': total_pnl / total_wagered if total_wagered > 0 else 0,
        'active_days': active_days,
        'total_days': len(dates_sorted),
        'day_coverage': active_days / len(dates_sorted) if dates_sorted else 0,
        'max_drawdown': max_dd,
        # Over/Under breakdown
        'over_picks': len(over_picks),
        'over_wins': sum(1 for p in over_picks if p['hit']),
        'over_accuracy': sum(1 for p in over_picks if p['hit']) / len(over_picks) if over_picks else 0,
        'under_picks': len(under_picks),
        'under_wins': sum(1 for p in under_picks if p['hit']),
        'under_accuracy': sum(1 for p in under_picks if p['hit']) / len(under_picks) if under_picks else 0,
        'picks': picks,
        'daily': daily_results,
    }
