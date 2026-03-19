"""
MLB Strategy V8 — Multi-Gate Certainty Cascade (MGCC) for Parlays
=================================================================
Patent-pending: Same MGCC framework as NBA V8, adapted for MLB discrete stats.

MLB-specific adaptations:
- Uses at-bats instead of minutes for activity filtering
- Lower HR thresholds (MLB stats are more variable)
- Stat categories: h, tb, r, rbi, hr
- Poisson-aware floor percentile (discrete stat-friendly)

The 7 Gates (Hierarchical Evidence Stacking):
1. BCF: Bayesian Credible Floor (LB >= 0.85 for MLB)
2. PFC: Percentile Floor Clearance (p15 > line)
3. MWU: Multi-Window Unanimity (3/5/10/15 games)
4. SC: Streak Continuity (2+ consecutive clears)
5. EC: Extended Consistency (25-game HR >= 75%)
6. MS: AB Stability (consistent at-bat volume)
7. HF: Heavy Favorite (odds <= -200)

Walk-forward only, real Odds API odds, no leakage.
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


MLB_V8_CONFIG = {
    'MIN_GAMES': 12,
    'MIN_AB': 2,

    # Gate 1: Bayesian Credible Floor — relaxed for MLB
    'BCF_PRIOR_ALPHA': 1.0,
    'BCF_PRIOR_BETA': 1.0,
    'BCF_CI_LEVEL': 0.80,
    'BCF_MIN_LB': 0.85,

    # Gate 2: Percentile Floor
    'PFC_PERCENTILE': 0.15,
    'PFC_MIN_CLEARANCE': 0.0,

    # Gate 3: Multi-Window Unanimity
    'MWU_WINDOWS': [3, 5, 10, 15],
    'MWU_MIN_HR': 0.70,

    # Gate 4: Streak Continuity
    'SC_MIN_STREAK': 2,

    # Gate 5: Extended Consistency
    'EC_MIN_GAMES': 12,
    'EC_MIN_HR': 0.75,

    # Gate 6: AB Stability
    'MS_MIN_AVG_AB': 3.0,
    'MS_MAX_CV': 0.30,

    # Gate 7: Heavy Favorite
    'HF_MAX_ODDS': -200,

    # Leg odds
    'LEG_MIN_ODDS': -1500,
    'LEG_MAX_ODDS': -100,

    # CRS
    'CRS_TOP_N': 10,

    # Stats
    'STATS_ALLOWED': ['h', 'tb', 'r', 'rbi', 'hr'],

    # Parlay construction
    'PARLAY_LEGS': [2, 3, 5],
    'PARLAY_REQUIRE_DIFF_TEAMS': True,
    'MAX_PARLAYS_PER_DAY': 3,

    # Wager
    'WAGER_2LEG': 150,
    'WAGER_3LEG': 120,
    'WAGER_5LEG': 100,
    'DEFAULT_WAGER': 100,
}


def _percentile(values, pct):
    s = sorted(values)
    return s[max(0, min(len(s) - 1, int(len(s) * pct)))]


def evaluate_leg_mgcc(model, name, stat, line, odds, config):
    """Multi-Gate Certainty Cascade for MLB legs."""
    if model.game_count(name) < config['MIN_GAMES']:
        return None
    if model.avg_ab(name, 10) < config['MIN_AB']:
        return None
    if odds < config.get('LEG_MIN_ODDS', -1500) or odds > config.get('LEG_MAX_ODDS', -100):
        return None

    v15 = model.get_values(name, stat, 15, min_ab=config['MIN_AB'])
    if not v15 or len(v15) < 6:
        return None

    # Gate 1: BCF
    hits = sum(1 for v in v15 if v > line)
    alpha = config['BCF_PRIOR_ALPHA'] + hits
    beta_p = config['BCF_PRIOR_BETA'] + (len(v15) - hits)
    mean_prob = alpha / (alpha + beta_p)
    lower_bound = beta_ppf_approx(alpha, beta_p, 1 - config['BCF_CI_LEVEL'])
    if lower_bound < config['BCF_MIN_LB']:
        return None
    hr_15 = hits / len(v15)
    gates_passed = 1

    # Gate 2: PFC
    floor = _percentile(v15, config['PFC_PERCENTILE'])
    clearance = floor - line
    g2 = clearance >= config['PFC_MIN_CLEARANCE']
    if g2: gates_passed += 1

    # Gate 3: MWU
    g3 = True
    for w in config['MWU_WINDOWS']:
        vw = model.get_values(name, stat, w, min_ab=config['MIN_AB'])
        if vw and len(vw) >= min(w, 3):
            if sum(1 for v in vw if v > line) / len(vw) < config['MWU_MIN_HR']:
                g3 = False
                break
    if g3: gates_passed += 1

    # Gate 4: SC
    v5 = model.get_values(name, stat, 5, min_ab=config['MIN_AB'])
    streak = 0
    if v5:
        for v in reversed(v5):
            if v > line: streak += 1
            else: break
    g4 = streak >= config['SC_MIN_STREAK']
    if g4: gates_passed += 1

    # Gate 5: EC
    v25 = model.get_values(name, stat, 25, min_ab=config['MIN_AB'])
    g5 = True
    if v25 and len(v25) >= config['EC_MIN_GAMES']:
        g5 = sum(1 for v in v25 if v > line) / len(v25) >= config['EC_MIN_HR']
    if g5: gates_passed += 1

    # Gate 6: AB stability
    p = model.profiles.get(name)
    g6 = False
    if p:
        recent = [g.get('ab', 0) for g in p['games'][-10:]]
        if recent and len(recent) >= 5:
            mean_ab = sum(recent) / len(recent)
            if mean_ab >= config['MS_MIN_AVG_AB']:
                std_ab = math.sqrt(sum((a - mean_ab)**2 for a in recent) / len(recent))
                cv = std_ab / mean_ab if mean_ab > 0 else 1.0
                g6 = cv < config['MS_MAX_CV']
    if g6: gates_passed += 1

    # Gate 7: HF
    g7 = odds <= config['HF_MAX_ODDS']
    if g7: gates_passed += 1

    confidence = (
        lower_bound * 0.30 + hr_15 * 0.25 +
        min(1.0, clearance / 2.0 + 0.3) * 0.15 +
        (0.10 if g3 else 0) + (0.08 if g4 else 0) +
        (0.07 if g5 else 0) + (0.05 if g6 else 0)
    )

    market_prob = implied_probability(odds)
    return {
        'player': name, 'stat': stat, 'line': line, 'odds': odds,
        'dec_odds': american_to_decimal(odds),
        'lower_bound': lower_bound, 'mean_prob': mean_prob,
        'hr_15': hr_15, 'clearance': clearance, 'streak': streak,
        'gates_passed': gates_passed, 'confidence': confidence,
        'market_prob': market_prob, 'edge': lower_bound - market_prob,
        'g2_pfc': g2, 'g3_mwu': g3, 'g4_sc': g4,
        'g5_ec': g5, 'g6_ms': g6, 'g7_hf': g7,
    }


def build_multi_leg_parlays(day_legs, n_legs, config):
    if len(day_legs) < n_legs:
        return []
    parlays = []
    used = set()
    for _ in range(config.get('MAX_PARLAYS_PER_DAY', 3)):
        available = [l for l in day_legs if l['player'] not in used]
        if len(available) < n_legs:
            break
        selected = []
        teams = set()
        for l in available:
            if config.get('PARLAY_REQUIRE_DIFF_TEAMS', True):
                t = l.get('team', '')
                if t and t in teams:
                    continue
                if t: teams.add(t)
            selected.append(l)
            if len(selected) >= n_legs:
                break
        if len(selected) < n_legs:
            break
        p_dec = parlay_decimal_odds([l['odds'] for l in selected])
        combined_prob = 1.0
        for l in selected:
            combined_prob *= l['lower_bound']
        parlays.append({
            'legs': selected, 'n_legs': n_legs,
            'parlay_decimal': p_dec,
            'parlay_american': decimal_to_american(p_dec),
            'combined_prob': combined_prob,
            'parlay_ev': combined_prob * p_dec - 1,
            'total_edge': sum(l['edge'] for l in selected),
        })
        for l in selected:
            used.add(l['player'])
    return parlays


def run_backtest(box_scores, odds_data, config, verbose=False):
    """Walk-forward backtest for MLB V8 MGCC."""
    model = MLBPlayerModel()
    stat_map = {
        'hitsProps': 'h', 'tbProps': 'tb', 'runsProps': 'r',
        'rbiProps': 'rbi', 'hrProps': 'hr',
    }
    stats_allowed = config.get('STATS_ALLOWED', ['h', 'tb', 'r', 'rbi', 'hr'])

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

        actuals = {}
        player_teams = {}
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
                        leg = evaluate_leg_mgcc(model, name, stat_key, line, over_odds, config)
                        if leg:
                            actual = actuals.get(name, {}).get(stat_key)
                            if actual is not None:
                                leg['actual'] = actual
                                leg['hit'] = actual > line
                                leg['team'] = player_teams.get(name, '')
                                leg['date'] = date
                                day_legs.append(leg)

        # CRS
        day_legs.sort(key=lambda l: l['confidence'], reverse=True)
        seen = set()
        elite = []
        for l in day_legs:
            if l['player'] not in seen:
                elite.append(l)
                seen.add(l['player'])
                if len(elite) >= config.get('CRS_TOP_N', 10):
                    break

        day_picks = []
        for n_legs in config.get('PARLAY_LEGS', [2, 3, 5]):
            parlays = build_multi_leg_parlays(elite, n_legs, config)
            for p in parlays:
                hit = all(l['hit'] for l in p['legs'])
                wager = config.get(f'WAGER_{n_legs}LEG', config.get('DEFAULT_WAGER', 100))
                pnl = round((p['parlay_decimal'] - 1) * wager) if hit else -wager
                pick = {
                    'date': date, 'bet_type': f'parlay_{n_legs}L',
                    'n_legs': n_legs,
                    'legs': [{'player': l['player'], 'stat': l['stat'],
                              'line': l['line'], 'odds': l['odds'],
                              'hit': l['hit'], 'actual': l.get('actual'),
                              'lower_bound': l['lower_bound'], 'hr_15': l['hr_15'],
                              'gates': l['gates_passed'], 'clearance': l['clearance'],
                    } for l in p['legs']],
                    'parlay_decimal': p['parlay_decimal'],
                    'parlay_american': p['parlay_american'],
                    'combined_prob': p['combined_prob'],
                    'hit': hit, 'pnl': pnl, 'wager': wager,
                    'combined_score': sum(l['confidence'] for l in p['legs']) / len(p['legs']),
                    'edge': p['total_edge'],
                }
                day_picks.append(pick)
                all_picks.append(pick)

        if day_picks and verbose:
            for p in day_picks:
                lh = sum(1 for l in p['legs'] if l['hit'])
                print(f"  {date}: {p['n_legs']}L ({p['parlay_american']:+d}) "
                      f"[{lh}/{p['n_legs']}] [{'WIN' if p['hit'] else 'LOSS'}] ${p['pnl']:+d}")

        if day_picks:
            daily_results.append({
                'date': date, 'n_picks': len(day_picks),
                'wins': sum(1 for p in day_picks if p['hit']),
                'pnl': sum(p['pnl'] for p in day_picks),
            })

        for bg in day_boxes:
            _update_model(model, bg, date)

    # Stats
    by_legs = defaultdict(lambda: {'total': 0, 'wins': 0, 'pnl': 0, 'wagered': 0, 'legs_total': 0, 'legs_hits': 0})
    for p in all_picks:
        n = p['n_legs']
        by_legs[n]['total'] += 1
        if p['hit']: by_legs[n]['wins'] += 1
        by_legs[n]['pnl'] += p['pnl']
        by_legs[n]['wagered'] += p['wager']
        for l in p.get('legs', []):
            by_legs[n]['legs_total'] += 1
            if l.get('hit'): by_legs[n]['legs_hits'] += 1

    total_pnl = sum(p['pnl'] for p in all_picks)
    total_wag = sum(p['wager'] for p in all_picks)
    total_wins = sum(1 for p in all_picks if p['hit'])

    return {
        'total_picks': len(all_picks), 'total_wins': total_wins,
        'accuracy': total_wins / len(all_picks) if all_picks else 0,
        'total_pnl': total_pnl, 'total_wagered': total_wag,
        'total_roi': total_pnl / total_wag if total_wag > 0 else 0,
        'active_days': len(daily_results), 'total_days': len(processed),
        'day_coverage': len(daily_results) / len(processed) if processed else 0,
        'by_legs': {n: {**s, 'accuracy': s['wins']/s['total'] if s['total'] else 0,
                        'roi': s['pnl']/s['wagered'] if s['wagered'] else 0,
                        'leg_accuracy': s['legs_hits']/s['legs_total'] if s['legs_total'] else 0}
                    for n, s in by_legs.items()},
        'picks': all_picks, 'daily': daily_results,
    }
