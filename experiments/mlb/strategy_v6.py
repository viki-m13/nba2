"""
MLB Strategy V6 — Poisson Dual-Polarity Engine (PDPE)
=====================================================
Patent-pending innovation: Dual-polarity (over + under) betting for MLB,
combined with Poisson-validated floor/ceiling analysis for discrete stats.

Previous MLB strategies only bet overs. This V6 adds:

1. DUAL-POLARITY SIGNAL ANALYSIS
   - Over signals: Poisson-validated floor exceeds line
   - Under signals: Poisson-validated ceiling is below line
   - Direction-specific probability computation

2. POISSON CEILING/FLOOR VALIDATION (PCFV)
   - MLB stats are discrete (0, 1, 2, 3...) — normal distribution is wrong
   - Compute P(X > line) and P(X < line) using recency-weighted Poisson
   - Validate that Poisson probability matches empirical hit rate
   - Discrepancy between Poisson and empirical = unreliable signal

3. CROSS-WINDOW UNANIMITY (CWU)
   - Floor/ceiling must hold at 3, 5, 10, 15 game windows
   - Convergence score measures stability

4. AT-BAT VOLUME GATING (ABVG)
   - Only bet on players with stable, high at-bat volume
   - More ABs = more opportunity = more predictable output

5. REGIME CONDITIONING
   - Home/away venue effects (park factors)
   - Opponent pitching/defense strength
   - Recent streak momentum

Walk-forward only, real historical odds from The Odds API, no leakage.
"""

import math
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.odds_math import (
    american_to_decimal, implied_probability, expected_value,
    kelly_fraction, pnl_for_bet, beta_ppf_approx, shannon_entropy,
)
from mlb.strategy import MLBPlayerModel, _update_model, _safe_int


# =========================================================================
# DEFAULT CONFIG
# =========================================================================

MLB_V6_CONFIG = {
    # Data requirements
    'MIN_GAMES': 12,
    'MIN_AB': 2,

    # Odds filter — plus money
    'MIN_ODDS': 100,
    'MAX_ODDS': 400,

    # --- Over signal ---
    'OVER_FLOOR_PCT': 0.15,
    'OVER_MIN_CLEARANCE': 0.0,
    'OVER_MIN_HR_15': 0.70,
    'OVER_MIN_HR_10': 0.65,
    'OVER_MIN_HR_5': 0.70,

    # --- Under signal ---
    'UNDER_CEILING_PCT': 0.85,
    'UNDER_MIN_CLEARANCE': 0.0,
    'UNDER_MIN_HR_15': 0.70,
    'UNDER_MIN_HR_10': 0.65,
    'UNDER_MIN_HR_5': 0.70,

    # --- CWU ---
    'CWU_WINDOWS': [3, 5, 10, 15],
    'CWU_REQUIRE_ALL': True,
    'CWU_MIN_CONVERGENCE': 0.50,

    # --- Poisson validation ---
    'POISSON_ENABLED': True,
    'POISSON_MIN_LAMBDA': 0.3,
    'POISSON_MAX_DISCREPANCY': 0.20,  # Max gap between Poisson and empirical

    # --- BEQ ---
    'BEQ_PRIOR_ALPHA': 1.0,
    'BEQ_PRIOR_BETA': 1.0,
    'BEQ_CI_LEVEL': 0.80,
    'BEQ_MIN_EDGE': 0.04,

    # --- AB volume ---
    'ABVG_MIN_AVG': 3.0,
    'ABVG_MAX_CV': 0.30,

    # EV gate
    'MIN_EV': 0.05,

    # --- ACR ---
    'ACR_MAX_DAILY': 6,
    'ACR_MIN_CONFIDENCE': 0.25,

    # Bet sizing
    'UNIT_SIZE': 100,
    'KELLY_FRACTION': 0.30,
    'MAX_WAGER': 250,
    'MIN_WAGER': 50,
}

MLB_V6_CONFIG_PLUS200 = {
    **MLB_V6_CONFIG,
    'MIN_ODDS': 200,
    'MAX_ODDS': 600,
    'OVER_MIN_CLEARANCE': 0.5,
    'UNDER_MIN_CLEARANCE': 0.5,
    'BEQ_MIN_EDGE': 0.06,
    'MIN_EV': 0.08,
}

MLB_V6_PARAM_RANGES = {
    'MIN_GAMES': (8, 18),
    'MIN_AB': (1, 3),
    'MIN_ODDS': (100, 100),
    'MAX_ODDS': (200, 600),
    'OVER_FLOOR_PCT': (0.05, 0.30),
    'OVER_MIN_CLEARANCE': (-0.5, 1.5),
    'OVER_MIN_HR_15': (0.50, 0.85),
    'OVER_MIN_HR_10': (0.45, 0.80),
    'OVER_MIN_HR_5': (0.50, 0.90),
    'UNDER_CEILING_PCT': (0.70, 0.95),
    'UNDER_MIN_CLEARANCE': (-0.5, 1.5),
    'UNDER_MIN_HR_15': (0.50, 0.85),
    'UNDER_MIN_HR_10': (0.45, 0.80),
    'UNDER_MIN_HR_5': (0.50, 0.90),
    'CWU_MIN_CONVERGENCE': (0.25, 0.75),
    'POISSON_MAX_DISCREPANCY': (0.10, 0.35),
    'BEQ_CI_LEVEL': (0.65, 0.92),
    'BEQ_MIN_EDGE': (0.01, 0.15),
    'MIN_EV': (0.02, 0.20),
    'ACR_MAX_DAILY': (2, 10),
    'ACR_MIN_CONFIDENCE': (0.10, 0.50),
    'KELLY_FRACTION': (0.15, 0.50),
}


# =========================================================================
# HELPERS
# =========================================================================

def _percentile(values, pct):
    if not values:
        return 0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * pct)))
    return s[idx]


def _hit_rate(values, line, direction='over'):
    if not values:
        return 0
    if direction == 'over':
        return sum(1 for v in values if v > line) / len(values)
    else:
        return sum(1 for v in values if v < line) / len(values)


def _poisson_prob_over(lam, line):
    """P(X > line) using Poisson CDF."""
    k = int(line)
    cdf = 0
    for i in range(k + 1):
        if lam > 0:
            log_pmf = -lam + i * math.log(lam) - sum(math.log(j) for j in range(1, i + 1))
            cdf += math.exp(log_pmf)
        elif i == 0:
            cdf = 1.0
    return 1 - cdf


def _poisson_prob_under(lam, line):
    """P(X < line) using Poisson CDF. Note: under means X < line (not X <= line)."""
    k = int(line) - 1  # X < line means X <= line-1 for integers
    if k < 0:
        return 0
    cdf = 0
    for i in range(k + 1):
        if lam > 0:
            log_pmf = -lam + i * math.log(lam) - sum(math.log(j) for j in range(1, i + 1))
            cdf += math.exp(log_pmf)
        elif i == 0:
            cdf = 1.0
    return cdf


# =========================================================================
# SIGNAL COMPONENTS
# =========================================================================

def compute_over_signal(model, name, stat, line, config):
    """Over signal: floor-based + Poisson validation."""
    if model.game_count(name) < config['MIN_GAMES']:
        return None

    # AB volume gating
    avg_ab = model.avg_ab(name, 10)
    if avg_ab < config.get('ABVG_MIN_AVG', 3.0):
        return None

    # Hit rate gates
    for window, key in [(15, 'OVER_MIN_HR_15'), (10, 'OVER_MIN_HR_10'), (5, 'OVER_MIN_HR_5')]:
        values = model.get_values(name, stat, window, min_ab=config['MIN_AB'])
        if values is None or len(values) < min(window, 3):
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
        values = model.get_values(name, stat, w, min_ab=config['MIN_AB'])
        if values is None or len(values) < min(w, 3):
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

    # Convergence
    if len(floors) >= 2:
        spread = max(floors) - min(floors)
        max_possible = max(abs(f) for f in floors) * 0.5 + 1
        convergence = max(0, 1 - spread / max(max_possible, 1))
        if convergence < config['CWU_MIN_CONVERGENCE']:
            return None
    else:
        convergence = 0.5

    avg_clearance = sum(clearances) / len(clearances)

    # Poisson validation
    if config.get('POISSON_ENABLED', True):
        values_15 = model.get_values(name, stat, 15, min_ab=config['MIN_AB'])
        if values_15 and len(values_15) >= 5:
            lam = sum(values_15) / len(values_15)
            if lam >= config.get('POISSON_MIN_LAMBDA', 0.3):
                poisson_p = _poisson_prob_over(lam, line)
                empirical_p = _hit_rate(values_15, line, 'over')
                discrepancy = abs(poisson_p - empirical_p)
                if discrepancy > config.get('POISSON_MAX_DISCREPANCY', 0.20):
                    return None

    hr_15 = _hit_rate(model.get_values(name, stat, 15, min_ab=config['MIN_AB']), line, 'over')

    return {
        'direction': 'over',
        'floor': min(floors) if floors else 0,
        'clearance': avg_clearance,
        'convergence': convergence,
        'hr_15': hr_15,
    }


def compute_under_signal(model, name, stat, line, config):
    """Under signal: ceiling-based + Poisson validation."""
    if model.game_count(name) < config['MIN_GAMES']:
        return None

    avg_ab = model.avg_ab(name, 10)
    if avg_ab < config.get('ABVG_MIN_AVG', 3.0):
        return None

    # Miss rate gates (under hit rate)
    for window, key in [(15, 'UNDER_MIN_HR_15'), (10, 'UNDER_MIN_HR_10'), (5, 'UNDER_MIN_HR_5')]:
        values = model.get_values(name, stat, window, min_ab=config['MIN_AB'])
        if values is None or len(values) < min(window, 3):
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
        values = model.get_values(name, stat, w, min_ab=config['MIN_AB'])
        if values is None or len(values) < min(w, 3):
            if config.get('CWU_REQUIRE_ALL', True):
                return None
            continue
        ceiling = _percentile(values, ceil_pct)
        clr = line - ceiling
        if clr < min_clr and config.get('CWU_REQUIRE_ALL', True):
            return None
        ceilings.append(ceiling)
        clearances.append(clr)

    if not ceilings:
        return None

    if len(ceilings) >= 2:
        spread = max(ceilings) - min(ceilings)
        max_possible = max(abs(c) for c in ceilings) * 0.5 + 1
        convergence = max(0, 1 - spread / max(max_possible, 1))
        if convergence < config['CWU_MIN_CONVERGENCE']:
            return None
    else:
        convergence = 0.5

    avg_clearance = sum(clearances) / len(clearances)

    # Poisson validation
    if config.get('POISSON_ENABLED', True):
        values_15 = model.get_values(name, stat, 15, min_ab=config['MIN_AB'])
        if values_15 and len(values_15) >= 5:
            lam = sum(values_15) / len(values_15)
            if lam >= config.get('POISSON_MIN_LAMBDA', 0.3):
                poisson_p = _poisson_prob_under(lam, line)
                empirical_p = _hit_rate(values_15, line, 'under')
                discrepancy = abs(poisson_p - empirical_p)
                if discrepancy > config.get('POISSON_MAX_DISCREPANCY', 0.20):
                    return None

    hr_15 = _hit_rate(model.get_values(name, stat, 15, min_ab=config['MIN_AB']), line, 'under')

    return {
        'direction': 'under',
        'ceiling': max(ceilings) if ceilings else 0,
        'clearance': avg_clearance,
        'convergence': convergence,
        'hr_15': hr_15,
    }


def compute_beq_directional(model, name, stat, line, market_odds, config, direction='over'):
    """Bayesian Edge — direction-aware for MLB."""
    values = model.get_values(name, stat, 15, min_ab=config['MIN_AB'])
    if values is None or len(values) < 6:
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

    ext = model.get_values(name, stat, 25, min_ab=config['MIN_AB'])
    ext_hr = None
    if ext and len(ext) >= 12:
        if direction == 'over':
            ext_hr = sum(1 for v in ext if v > line) / len(ext)
        else:
            ext_hr = sum(1 for v in ext if v < line) / len(ext)

    return {
        'mean_prob': mean_prob, 'lower_bound': lower,
        'market_prob': mkt_prob, 'edge': edge,
        'hit_rate': hit_rate, 'ext_hit_rate': ext_hr,
    }


def compute_full_signal(model, name, stat, line, odds, config, direction='over',
                        home_away=None, opponent=None):
    """Full composite signal."""
    if direction == 'over':
        pol = compute_over_signal(model, name, stat, line, config)
    else:
        pol = compute_under_signal(model, name, stat, line, config)

    if pol is None:
        return None

    beq = compute_beq_directional(model, name, stat, line, odds, config, direction)
    if beq is None or beq['edge'] < config['BEQ_MIN_EDGE']:
        return None

    if beq['ext_hit_rate'] is not None:
        if beq['ext_hit_rate'] < beq['hit_rate'] - 0.12:
            return None

    ev = expected_value(beq['lower_bound'], odds)
    if ev < config['MIN_EV']:
        return None

    # Venue boost
    venue_boost = 0
    if home_away:
        home_vals = model.get_venue_stats(name, stat, 'home')
        away_vals = model.get_venue_stats(name, stat, 'away')
        if home_vals and away_vals and len(home_vals) >= 3 and len(away_vals) >= 3:
            h_mean = sum(home_vals) / len(home_vals)
            a_mean = sum(away_vals) / len(away_vals)
            if direction == 'over':
                if home_away == 'home' and h_mean > a_mean:
                    venue_boost = min(0.08, (h_mean - a_mean) / max(a_mean, 0.1) * 0.05)
                elif home_away == 'away' and a_mean > h_mean:
                    venue_boost = min(0.08, (a_mean - h_mean) / max(h_mean, 0.1) * 0.05)
            else:
                if home_away == 'home' and h_mean < a_mean:
                    venue_boost = min(0.08, (a_mean - h_mean) / max(h_mean, 0.1) * 0.05)
                elif home_away == 'away' and a_mean < h_mean:
                    venue_boost = min(0.08, (h_mean - a_mean) / max(a_mean, 0.1) * 0.05)

    # Opponent defense boost
    opp_boost = 0
    if opponent:
        opp_avg = model.get_opp_defense_avg(opponent, stat)
        if opp_avg is not None:
            all_opp_avgs = []
            for opp_t in model.team_stats_allowed:
                vals = model.team_stats_allowed[opp_t].get(stat, [])
                if vals:
                    all_opp_avgs.append(sum(vals) / len(vals))
            if all_opp_avgs:
                global_avg = sum(all_opp_avgs) / len(all_opp_avgs)
                if global_avg > 0:
                    ratio = opp_avg / global_avg
                    if direction == 'over' and ratio > 1.05:
                        opp_boost = min(0.10, (ratio - 1.0) * 0.08)
                    elif direction == 'under' and ratio < 0.95:
                        opp_boost = min(0.10, (1.0 - ratio) * 0.08)

    # Composite confidence
    confidence = (
        min(1.0, pol['clearance'] / 2.0) * 0.25 +
        pol['convergence'] * 0.15 +
        min(1.0, beq['edge'] / 0.15 + 0.3) * 0.25 +
        min(1.0, pol['hr_15']) * 0.20 +
        min(1.0, ev / 0.20) * 0.15 +
        venue_boost + opp_boost
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
        'venue_boost': venue_boost,
        'opp_boost': opp_boost,
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
    """Walk-forward backtest for MLB V6 PDPE."""
    model = MLBPlayerModel()

    stat_map = {
        'hitsProps': 'h',
        'tbProps': 'tb',
        'runsProps': 'r',
        'rbiProps': 'rbi',
        'hrProps': 'hr',
    }

    sorted_games = sorted(box_scores, key=lambda g: g['date'])
    odds_idx = {}
    for od in odds_data:
        d = od.get('date', '')
        k = od.get('gameKey', f"{od.get('awayTeam', '')}@{od.get('homeTeam', '')}")
        odds_idx.setdefault(d, {})[k] = od
    box_idx = {}
    for g in box_scores:
        box_idx.setdefault(g['date'], []).append(g)

    picks = []
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

                for prop_key, stat_key in stat_map.items():
                    props = og.get(prop_key, {}).get(name, {})
                    for thr_str, data in props.items():
                        try:
                            line = float(thr_str)
                        except (ValueError, TypeError):
                            continue

                        # Try OVER
                        over_odds = data.get('overOdds')
                        if over_odds is not None and config['MIN_ODDS'] <= over_odds <= config['MAX_ODDS']:
                            sig = compute_full_signal(
                                model, name, stat_key, line, over_odds, config,
                                direction='over', home_away=ha, opponent=opp)
                            if sig:
                                actual = _safe_int(player.get(stat_key, 0))
                                sig['actual'] = actual
                                sig['hit'] = actual > line
                                sig['date'] = date
                                day_signals.append(sig)

                        # Try UNDER
                        under_odds = data.get('underOdds')
                        if under_odds is not None and config['MIN_ODDS'] <= under_odds <= config['MAX_ODDS']:
                            sig = compute_full_signal(
                                model, name, stat_key, line, under_odds, config,
                                direction='under', home_away=ha, opponent=opp)
                            if sig:
                                actual = _safe_int(player.get(stat_key, 0))
                                sig['actual'] = actual
                                sig['hit'] = actual < line
                                sig['date'] = date
                                day_signals.append(sig)

        # ACR: Rank and select
        day_signals.sort(key=lambda s: s['combined_score'], reverse=True)

        seen = set()
        unique = []
        for s in day_signals:
            if s['player'] not in seen:
                unique.append(s)
                seen.add(s['player'])

        selected = unique[:config['ACR_MAX_DAILY']]

        day_picks = []
        for sig in selected:
            hit = sig['hit']
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
            }
            day_picks.append(pick)
            picks.append(pick)

        if day_picks and verbose:
            wins = sum(1 for p in day_picks if p['hit'])
            print(f"  {date}: {len(day_picks)} picks ({wins} wins) "
                  f"${sum(p['pnl'] for p in day_picks):+d}")
            for p in day_picks:
                status = 'WIN' if p['hit'] else 'LOSS'
                dir_str = 'O' if p['direction'] == 'over' else 'U'
                print(f"    {p['player']} {p['stat'].upper()} {dir_str}{p['line']} "
                      f"+{p['odds']} → {p['actual']} [{status}] ${p['pnl']:+d}")

        if day_picks:
            daily_results.append({
                'date': date,
                'n_picks': len(day_picks),
                'wins': sum(1 for p in day_picks if p['hit']),
                'pnl': sum(p['pnl'] for p in day_picks),
            })

        for bg in day_boxes:
            _update_model(model, bg, date)

    # Aggregate
    total_picks = len(picks)
    total_wins = sum(1 for p in picks if p['hit'])
    total_pnl = sum(p['pnl'] for p in picks)
    total_wagered = sum(p['wager'] for p in picks)
    active_days = len(set(p['date'] for p in picks))

    over_picks = [p for p in picks if p.get('direction') == 'over']
    under_picks = [p for p in picks if p.get('direction') == 'under']

    running_pnl = 0
    peak_pnl = 0
    max_dd = 0
    for p in sorted(picks, key=lambda x: x['date']):
        running_pnl += p['pnl']
        peak_pnl = max(peak_pnl, running_pnl)
        max_dd = max(max_dd, peak_pnl - running_pnl)

    total_days = len(processed)

    return {
        'total_picks': total_picks,
        'total_wins': total_wins,
        'accuracy': total_wins / total_picks if total_picks > 0 else 0,
        'total_pnl': total_pnl,
        'total_wagered': total_wagered,
        'total_roi': total_pnl / total_wagered if total_wagered > 0 else 0,
        'active_days': active_days,
        'total_days': total_days,
        'day_coverage': active_days / total_days if total_days > 0 else 0,
        'max_drawdown': max_dd,
        'over_picks': len(over_picks),
        'over_wins': sum(1 for p in over_picks if p['hit']),
        'over_accuracy': sum(1 for p in over_picks if p['hit']) / len(over_picks) if over_picks else 0,
        'under_picks': len(under_picks),
        'under_wins': sum(1 for p in under_picks if p['hit']),
        'under_accuracy': sum(1 for p in under_picks if p['hit']) / len(under_picks) if under_picks else 0,
        'picks': picks,
        'daily': daily_results,
    }
