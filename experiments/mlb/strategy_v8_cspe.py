"""
MLB Strategy V8 — Convergence Score Parlay Engine (CSPE)
=========================================================
Patent-pending: Poisson-Bayesian CSPE for MLB discrete stats.

KEY DESIGN PRINCIPLES:
1. POISSON-BAYESIAN PROBABILITY — baseball stats are discrete (0, 1, 2, 3 hits).
   Poisson distribution models P(hits >= 1) = 1 - e^(-lambda) with exponential
   decay-weighted lambda. This is the mathematical foundation for MLB accuracy.
2. MULTI-SIGNAL CONVERGENCE — only bet when ALL signals agree:
   - Poisson P(over) >= threshold (statistical model)
   - 100% hit rate in recent window (empirical evidence)
   - Active hitting streak (momentum)
   - AB stability (regular playing time)
   - Bayesian lower bound (uncertainty-adjusted)
3. 2-LEG PARLAYS — highest parlay accuracy tier
4. TIGHT ODDS RANGE — only bet moderate favorites (-260 to -120)
   where market inefficiency exists

Walk-forward only. Real odds from The Odds API. No leakage, no overfitting.
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
from mlb.strategy import MLBPlayerModel, _update_model, _safe_int


# =========================================================================
# CONFIG — MLB V8 CSPE with Poisson-Bayesian foundation
# =========================================================================

MLB_V8_CONFIG = {
    # Data requirements
    'MIN_GAMES': 12,
    'MIN_AB': 3,

    # Poisson-Bayesian parameters
    'PBF_DECAY': 0.92,
    'PBF_MIN_PROB': 0.90,     # Poisson P(over) must be >= 90%

    # Bayesian confidence
    'BCF_PRIOR_ALPHA': 1.0,
    'BCF_PRIOR_BETA': 1.0,
    'BCF_CI_LEVEL': 0.80,
    'HARD_MIN_BCF_LB': 0.65,

    # Hit rate gates
    'HARD_MIN_HR': 0.85,      # 85%+ hit rate in window
    'PERFECT_HR_WINDOW': 8,   # Require 100% in last N games
    'PERFECT_HR_REQUIRED': True,

    # Streak requirements
    'MIN_STREAK': 2,          # Must have active streak of 2+ games

    # Scoring weights
    'W_PBF': 0.30,            # Poisson probability (most important)
    'W_BCF': 0.20,            # Bayesian confidence
    'W_HR': 0.15,             # Empirical hit rate
    'W_STREAK': 0.15,         # Momentum
    'W_STABILITY': 0.10,      # Playing time consistency
    'W_EDGE': 0.10,           # Market edge

    # Selection
    'SCORE_MIN': 0.55,
    'ELITE_TOP_N': 6,
    'STATS_ALLOWED': ['h'],   # Hits only
    'MAX_LINE': 0.5,          # Only 0.5 line (at least 1 hit)

    # Odds range — moderate favorites where edge exists
    'LEG_MIN_ODDS': -400,
    'LEG_MAX_ODDS': -110,

    # Multi-window consistency
    'CONSISTENCY_WINDOWS': [5, 8],
    'CONSISTENCY_MIN_HR': 0.80,

    # Extended consistency
    'EXTENDED_WINDOW': 20,
    'EXTENDED_MIN_HR': 0.75,

    # Parlay construction
    'PARLAY_LEGS': [2, 3],
    'PARLAY_REQUIRE_DIFF_TEAMS': True,
    'MAX_PARLAYS_PER_TIER': 1,

    # Wager sizing
    'WAGER_2LEG': 200,
    'WAGER_3LEG': 120,
    'DEFAULT_WAGER': 100,
}


# =========================================================================
# POISSON PROBABILITY
# =========================================================================

def poisson_over_prob(model, name, stat, line, config):
    """
    Compute Poisson probability that player exceeds the line.
    For discrete stats: P(X > line) = 1 - CDF(floor(line))
    Uses decay-weighted lambda (recent games weighted higher).
    """
    window = config.get('EXTENDED_WINDOW', 20)
    min_ab = config.get('MIN_AB', 3)
    values = model.get_values(name, stat, window, min_ab=min_ab)
    if not values or len(values) < 6:
        return None

    # Decay-weighted lambda
    decay = config.get('PBF_DECAY', 0.92)
    n = len(values)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_w = sum(weights)
    lam = sum(v * w for v, w in zip(values, weights)) / total_w

    if lam <= 0.01:
        return None

    # P(X > line) for Poisson
    # For line=0.5: P(X >= 1) = 1 - P(X=0) = 1 - e^(-lambda)
    # For line=1.5: P(X >= 2) = 1 - P(X=0) - P(X=1) = 1 - e^(-lambda) - lambda*e^(-lambda)
    k = int(math.floor(line))
    cdf = 0.0
    for i in range(k + 1):
        cdf += (lam ** i) * math.exp(-lam) / math.factorial(i)
    prob = 1.0 - cdf

    return {'lambda': lam, 'prob': prob, 'k': k}


# =========================================================================
# SCORING
# =========================================================================

def _std_dev(values):
    if len(values) < 2:
        return 0.001
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return max(math.sqrt(var), 0.001)


def compute_convergence_score(model, name, stat, line, odds, config):
    """Compute Convergence Score with Poisson-Bayesian foundation."""
    if model.game_count(name) < config['MIN_GAMES']:
        return None
    if model.avg_ab(name, 10) < config['MIN_AB']:
        return None
    if odds < config.get('LEG_MIN_ODDS', -400) or odds > config.get('LEG_MAX_ODDS', -110):
        return None

    # Line filter
    max_line = config.get('MAX_LINE', 999)
    if max_line < 999 and line > max_line:
        return None

    # === POISSON PROBABILITY (primary signal) ===
    pbf = poisson_over_prob(model, name, stat, line, config)
    if pbf is None:
        return None
    poisson_prob = pbf['prob']
    if poisson_prob < config.get('PBF_MIN_PROB', 0.90):
        return None

    # === EMPIRICAL HIT RATE ===
    window = config.get('EXTENDED_WINDOW', 20)
    values = model.get_values(name, stat, window, min_ab=config['MIN_AB'])
    if not values or len(values) < 8:
        return None

    hits = sum(1 for v in values if v > line)
    n = len(values)
    hr = hits / n
    if hr < config.get('HARD_MIN_HR', 0.85):
        return None

    # === PERFECT HIT RATE WINDOW ===
    if config.get('PERFECT_HR_REQUIRED', True):
        perf_window = config.get('PERFECT_HR_WINDOW', 8)
        recent = model.get_values(name, stat, perf_window, min_ab=config['MIN_AB'])
        if recent and len(recent) >= perf_window:
            recent_hr = sum(1 for v in recent if v > line) / len(recent)
            if recent_hr < 1.0:
                return None

    # === BAYESIAN CONFIDENCE ===
    alpha = config['BCF_PRIOR_ALPHA'] + hits
    beta_p = config['BCF_PRIOR_BETA'] + (n - hits)
    mean_prob = alpha / (alpha + beta_p)
    bcf_lb = beta_ppf_approx(alpha, beta_p, 1 - config['BCF_CI_LEVEL'])
    if bcf_lb < config.get('HARD_MIN_BCF_LB', 0.65):
        return None

    # === STREAK ===
    recent_vals = model.get_values(name, stat, 10, min_ab=config['MIN_AB'])
    streak = 0
    if recent_vals:
        for v in reversed(recent_vals):
            if v > line:
                streak += 1
            else:
                break
    min_streak = config.get('MIN_STREAK', 2)
    if streak < min_streak:
        return None

    # === MULTI-WINDOW CONSISTENCY ===
    consistency_windows = config.get('CONSISTENCY_WINDOWS', [5, 8])
    consistency_min = config.get('CONSISTENCY_MIN_HR', 0.80)
    for w in consistency_windows:
        vw = model.get_values(name, stat, w, min_ab=config['MIN_AB'])
        if vw and len(vw) >= min(w, 3):
            if sum(1 for v in vw if v > line) / len(vw) < consistency_min:
                return None

    # === EXTENDED CONSISTENCY ===
    ext_min = config.get('EXTENDED_MIN_HR', 0.75)
    if n >= 15 and hr < ext_min:
        return None

    # === AB STABILITY ===
    p = model.profiles.get(name)
    stability = 0.0
    if p:
        recent_ab = [g.get('ab', 0) for g in p['games'][-10:]]
        if recent_ab and len(recent_ab) >= 5:
            mean_ab = sum(recent_ab) / len(recent_ab)
            if mean_ab >= config['MIN_AB']:
                std_ab = math.sqrt(sum((a - mean_ab)**2 for a in recent_ab) / len(recent_ab))
                cv = std_ab / mean_ab if mean_ab > 0 else 1.0
                stability = max(0.0, min(1.0, 1.0 - cv / 0.40))

    # === MARKET EDGE ===
    market_prob = implied_probability(odds)
    edge = bcf_lb - market_prob
    edge_norm = min(1.0, max(0.0, edge / 0.25))

    # === COMPOSITE SCORE ===
    score = (
        config['W_PBF'] * poisson_prob +
        config['W_BCF'] * bcf_lb +
        config['W_HR'] * hr +
        config['W_STREAK'] * min(1.0, streak / 8.0) +
        config['W_STABILITY'] * stability +
        config['W_EDGE'] * edge_norm
    )

    return {
        'player': name, 'stat': stat, 'line': line, 'odds': odds,
        'dec_odds': american_to_decimal(odds),
        'score': score, 'bcf_lb': bcf_lb, 'mean_prob': mean_prob,
        'hr_20': hr, 'poisson_prob': poisson_prob, 'poisson_lambda': pbf['lambda'],
        'z_floor': 0, 'clearance': 0,  # kept for compatibility
        'streak': streak, 'stability': stability,
        'market_prob': market_prob, 'edge': edge,
    }


# =========================================================================
# PARLAY CONSTRUCTION
# =========================================================================

def build_parlays(elite_legs, n_legs, config):
    if len(elite_legs) < n_legs:
        return []
    parlays = []
    used = set()
    for _ in range(config.get('MAX_PARLAYS_PER_TIER', 1)):
        avail = [l for l in elite_legs if l['player'] not in used]
        if len(avail) < n_legs:
            break
        sel, teams = [], set()
        for l in avail:
            if config.get('PARLAY_REQUIRE_DIFF_TEAMS', True):
                t = l.get('team', '')
                if t and t in teams:
                    continue
                if t:
                    teams.add(t)
            sel.append(l)
            if len(sel) >= n_legs:
                break
        if len(sel) < n_legs:
            break
        p_dec = parlay_decimal_odds([l['odds'] for l in sel])
        cp = 1.0
        for l in sel:
            cp *= l['bcf_lb']
        parlays.append({
            'legs': sel, 'n_legs': n_legs,
            'parlay_decimal': p_dec,
            'parlay_american': decimal_to_american(p_dec),
            'combined_prob': cp, 'parlay_ev': cp * p_dec - 1,
            'total_edge': sum(l['edge'] for l in sel),
            'avg_score': sum(l['score'] for l in sel) / len(sel),
        })
        for l in sel:
            used.add(l['player'])
    return parlays


# =========================================================================
# WALK-FORWARD BACKTEST
# =========================================================================

def run_backtest(box_scores, odds_data, config, verbose=False):
    model = MLBPlayerModel()
    stat_map = {
        'hitsProps': 'h', 'tbProps': 'tb', 'runsProps': 'r',
        'rbiProps': 'rbi', 'hrProps': 'hr',
    }
    stats_allowed = config.get('STATS_ALLOWED', ['h'])

    sorted_games = sorted(box_scores, key=lambda g: g['date'])
    odds_idx = {}
    for od in odds_data:
        d = od.get('date', '')
        k = od.get('gameKey', f"{od.get('awayTeam', '')}@{od.get('homeTeam', '')}")
        odds_idx.setdefault(d, {})[k] = od
    box_idx = {}
    for g in box_scores:
        box_idx.setdefault(g['date'], []).append(g)

    all_picks, daily_results = [], []
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

        actuals, player_teams = {}, {}
        for bg in day_boxes:
            for p in bg.get('players', []):
                actuals[p['name']] = {k: _safe_int(p.get(k, 0)) for k in ['h', 'tb', 'r', 'rbi', 'hr']}
                player_teams[p['name']] = p.get('team', '')

        day_legs = []
        for bg in day_boxes:
            gk = f"{bg.get('away', '')}@{bg.get('home', '')}"
            og = day_odds.get(gk)
            if not og:
                continue
            for player in bg.get('players', []):
                name = player['name']
                for prop_key, stat_key in stat_map.items():
                    if stat_key not in stats_allowed:
                        continue
                    props = og.get(prop_key, {}).get(name, {})
                    for thr_str, data in props.items():
                        try:
                            line = float(thr_str)
                        except (ValueError, TypeError):
                            continue
                        over_odds = data.get('overOdds')
                        if over_odds is None:
                            continue
                        leg = compute_convergence_score(model, name, stat_key, line, over_odds, config)
                        if leg is None:
                            continue
                        actual = actuals.get(name, {}).get(stat_key)
                        if actual is not None:
                            leg['actual'] = actual
                            leg['hit'] = actual > line
                            leg['team'] = player_teams.get(name, '')
                            leg['date'] = date
                            day_legs.append(leg)

        # Elite selection
        min_score = config.get('SCORE_MIN', 0.55)
        qualified = [l for l in day_legs if l['score'] >= min_score]
        qualified.sort(key=lambda l: l['score'], reverse=True)
        seen, elite = set(), []
        for l in qualified:
            if l['player'] not in seen:
                elite.append(l)
                seen.add(l['player'])
                if len(elite) >= config.get('ELITE_TOP_N', 6):
                    break

        day_picks = []
        for n_legs in config.get('PARLAY_LEGS', [2, 3]):
            for p in build_parlays(elite, n_legs, config):
                hit = all(l['hit'] for l in p['legs'])
                wager = config.get(f'WAGER_{n_legs}LEG', config.get('DEFAULT_WAGER', 100))
                pnl = round((p['parlay_decimal'] - 1) * wager) if hit else -wager
                pick = {
                    'date': date, 'bet_type': f'parlay_{n_legs}L', 'n_legs': n_legs,
                    'legs': [{
                        'player': l['player'], 'stat': l['stat'],
                        'line': l['line'], 'odds': l['odds'],
                        'hit': l['hit'], 'actual': l.get('actual'),
                        'score': l['score'], 'bcf_lb': l['bcf_lb'],
                        'hr_20': l['hr_20'], 'z_floor': l.get('z_floor', 0),
                        'clearance': l.get('clearance', 0), 'streak': l['streak'],
                        'edge': l['edge'],
                        'poisson_prob': l.get('poisson_prob', 0),
                        'poisson_lambda': l.get('poisson_lambda', 0),
                    } for l in p['legs']],
                    'parlay_decimal': p['parlay_decimal'],
                    'parlay_american': p['parlay_american'],
                    'combined_prob': p['combined_prob'],
                    'parlay_ev': p['parlay_ev'],
                    'hit': hit, 'pnl': pnl, 'wager': wager,
                    'combined_score': p['avg_score'],
                    'edge': p['total_edge'],
                }
                day_picks.append(pick)
                all_picks.append(pick)

        if day_picks and verbose:
            for p in day_picks:
                lh = sum(1 for l in p['legs'] if l['hit'])
                legs_info = ', '.join(f"{l['player']}({l.get('poisson_prob',0)*100:.0f}%)" for l in p['legs'])
                print(f"  {date}: {p['n_legs']}L ({p['parlay_american']:+d}) "
                      f"[{lh}/{p['n_legs']}] [{'WIN' if p['hit'] else 'LOSS'}] ${p['pnl']:+d} "
                      f"  [{legs_info}]")
        if day_picks:
            daily_results.append({
                'date': date, 'n_picks': len(day_picks),
                'wins': sum(1 for p in day_picks if p['hit']),
                'pnl': sum(p['pnl'] for p in day_picks),
            })

        for bg in day_boxes:
            _update_model(model, bg, date)

    return _compute_stats(all_picks, daily_results, list(processed), config)


def _compute_stats(all_picks, daily_results, dates, config):
    by_legs = defaultdict(lambda: {
        'wins': 0, 'total': 0, 'pnl': 0, 'wagered': 0,
        'legs_total': 0, 'legs_hits': 0,
    })
    for p in all_picks:
        n = p['n_legs']
        by_legs[n]['total'] += 1
        if p['hit']:
            by_legs[n]['wins'] += 1
        by_legs[n]['pnl'] += p['pnl']
        by_legs[n]['wagered'] += p['wager']
        for l in p.get('legs', []):
            by_legs[n]['legs_total'] += 1
            if l.get('hit'):
                by_legs[n]['legs_hits'] += 1

    total_pnl = sum(p['pnl'] for p in all_picks)
    total_wagered = sum(p['wager'] for p in all_picks)
    return {
        'total_picks': len(all_picks),
        'total_wins': sum(1 for p in all_picks if p['hit']),
        'accuracy': sum(1 for p in all_picks if p['hit']) / len(all_picks) if all_picks else 0,
        'total_pnl': total_pnl,
        'total_wagered': total_wagered,
        'total_roi': total_pnl / total_wagered if total_wagered > 0 else 0,
        'active_days': len(daily_results),
        'total_days': len(dates),
        'by_legs': {n: {**s,
            'accuracy': s['wins'] / s['total'] if s['total'] else 0,
            'roi': s['pnl'] / s['wagered'] if s['wagered'] else 0,
            'leg_accuracy': s['legs_hits'] / s['legs_total'] if s['legs_total'] else 0,
        } for n, s in sorted(by_legs.items())},
        'picks': all_picks, 'daily': daily_results,
    }


def generate_picks_for_date(model, odds_for_date, config, stats_allowed=None):
    if stats_allowed is None:
        stats_allowed = config.get('STATS_ALLOWED', ['h'])
    stat_map = {
        'hitsProps': 'h', 'tbProps': 'tb', 'runsProps': 'r',
        'rbiProps': 'rbi', 'hrProps': 'hr',
    }
    day_legs = []
    for gk, odds_rec in odds_for_date.items():
        for prop_type, stat in stat_map.items():
            if stat not in stats_allowed:
                continue
            for player, lines in odds_rec.get(prop_type, {}).items():
                for line_str, line_data in lines.items():
                    try:
                        line_val = float(line_str)
                    except (ValueError, TypeError):
                        continue
                    over_odds = line_data.get('overOdds')
                    if over_odds is None:
                        continue
                    leg = compute_convergence_score(model, player, stat, line_val, over_odds, config)
                    if leg:
                        leg['team'] = model.get_team(player) or ''
                        day_legs.append(leg)

    min_score = config.get('SCORE_MIN', 0.55)
    qualified = [l for l in day_legs if l['score'] >= min_score]
    qualified.sort(key=lambda l: l['score'], reverse=True)
    seen, elite = set(), []
    for l in qualified:
        if l['player'] not in seen:
            elite.append(l)
            seen.add(l['player'])
            if len(elite) >= config.get('ELITE_TOP_N', 6):
                break

    recs = []
    for n_legs in config.get('PARLAY_LEGS', [2, 3]):
        for p in build_parlays(elite, n_legs, config):
            recs.append({
                'type': f'{n_legs}-leg parlay', 'n_legs': n_legs,
                'parlay_american': p['parlay_american'],
                'parlay_decimal': p['parlay_decimal'],
                'combined_prob': p['combined_prob'],
                'expected_value': p['parlay_ev'],
                'wager': config.get(f'WAGER_{n_legs}LEG', config.get('DEFAULT_WAGER', 100)),
                'legs': [{
                    'player': l['player'], 'stat': l['stat'],
                    'line': l['line'], 'odds': l['odds'], 'direction': 'over',
                    'score': round(l['score'], 3),
                    'bcf_lb': round(l['bcf_lb'], 3),
                    'hit_rate': round(l['hr_20'], 3),
                    'poisson_prob': round(l.get('poisson_prob', 0), 3),
                    'z_floor': round(l.get('z_floor', 0), 2),
                    'clearance': round(l.get('clearance', 0), 1),
                    'streak': l['streak'],
                } for l in p['legs']],
            })
    return recs
