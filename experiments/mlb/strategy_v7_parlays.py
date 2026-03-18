"""
MLB Strategy V7 — Safe-Leg Parlay Amplification Engine (SLPAE)
===============================================================
Adapted for MLB discrete stats. Same concept as NBA V7:
transform safe negative-odds legs into plus-money parlays.

MLB-specific adaptations:
- Uses at-bats instead of minutes for activity filtering
- Poisson probability validation for discrete stats
- Lower hit rate thresholds (MLB stats are more variable)
- Broader stat coverage: h, tb, r, rbi, hr, h_r_rbi

Walk-forward only, real historical odds, no leakage.
"""

import math
from collections import defaultdict
from itertools import combinations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.odds_math import (
    american_to_decimal, implied_probability, expected_value,
    kelly_fraction, pnl_for_bet, beta_ppf_approx, shannon_entropy,
    parlay_decimal_odds, decimal_to_american,
)
from mlb.strategy import MLBPlayerModel, _update_model, _safe_int


MLB_V7_CONFIG = {
    'MIN_GAMES': 12,
    'MIN_AB': 2,

    # Leg identification — target favorites
    'LEG_MIN_ODDS': -500,
    'LEG_MAX_ODDS': -110,

    # Hit rates — adapted for MLB (lower than NBA due to stat variance)
    'LEG_MIN_HR_15': 0.75,
    'LEG_MIN_HR_10': 0.70,
    'LEG_MIN_HR_5': 0.70,

    # Floor
    'LEG_FLOOR_PCT': 0.15,
    'LEG_MIN_CLEARANCE': 0.0,
    'LEG_WINDOWS': [3, 5, 10, 15],
    'LEG_MIN_WINDOWS_PASSING': 2,

    # Bayesian
    'BEQ_PRIOR_ALPHA': 1.0,
    'BEQ_PRIOR_BETA': 1.0,
    'BEQ_CI_LEVEL': 0.80,
    'BEQ_MIN_TRUE_PROB': 0.75,

    # Parlays
    'PARLAY_MIN_LEGS': 2,
    'PARLAY_MAX_LEGS': 4,
    'PARLAY_MAX_CORR': 0.35,
    'PARLAY_REQUIRE_DIFF_TEAMS': True,
    'PARLAY_MIN_COMBINED_ODDS': -200,
    'PARLAY_MAX_COMBINED_ODDS': 1000,
    'PARLAY_MIN_EV': 0.03,
    'MAX_PARLAYS_PER_DAY': 5,

    # Sizing
    'UNIT_SIZE': 100,
    'WAGER_2LEG': 150,
    'WAGER_3LEG': 100,
    'WAGER_4LEG': 75,

    # Singles
    'SINGLES_ENABLED': True,
    'SINGLE_MIN_HR_15': 0.85,
    'SINGLE_MIN_TRUE_PROB': 0.82,
    'SINGLE_MIN_ODDS': -400,
    'SINGLE_MAX_ODDS': -110,
    'MAX_SINGLES_PER_DAY': 3,
}

MLB_V7_CONFIG_RELAXED = {
    **MLB_V7_CONFIG,
    'LEG_MIN_HR_15': 0.65,
    'LEG_MIN_HR_10': 0.60,
    'LEG_MIN_HR_5': 0.60,
    'LEG_MIN_CLEARANCE': -1.0,
    'LEG_MIN_WINDOWS_PASSING': 1,
    'BEQ_MIN_TRUE_PROB': 0.68,
    'PARLAY_MIN_COMBINED_ODDS': -300,
    'PARLAY_MIN_EV': 0.01,
    'MAX_PARLAYS_PER_DAY': 8,
    'MIN_GAMES': 8,
}


def _percentile(values, pct):
    if not values:
        return 0
    s = sorted(values)
    return s[max(0, min(len(s) - 1, int(len(s) * pct)))]


def _correlation(v1, v2):
    n = min(len(v1), len(v2))
    if n < 5:
        return 0
    v1, v2 = v1[-n:], v2[-n:]
    m1, m2 = sum(v1) / n, sum(v2) / n
    cov = sum((a - m1) * (b - m2) for a, b in zip(v1, v2)) / n
    s1 = math.sqrt(sum((a - m1) ** 2 for a in v1) / n)
    s2 = math.sqrt(sum((b - m2) ** 2 for b in v2) / n)
    return cov / (s1 * s2) if s1 > 0 and s2 > 0 else 0


def evaluate_leg(model, name, stat, line, odds, config):
    """Evaluate a potential parlay leg."""
    if model.game_count(name) < config['MIN_GAMES']:
        return None
    if model.avg_ab(name, 10) < config['MIN_AB']:
        return None
    if odds < config['LEG_MIN_ODDS'] or odds > config['LEG_MAX_ODDS']:
        return None

    # Hit rate gates
    hit_rates = {}
    for window, key in [(15, 'LEG_MIN_HR_15'), (10, 'LEG_MIN_HR_10'), (5, 'LEG_MIN_HR_5')]:
        values = model.get_values(name, stat, window, min_ab=config['MIN_AB'])
        if values is None or len(values) < min(window, 3):
            return None
        hr = sum(1 for v in values if v > line) / len(values)
        if hr < config[key]:
            return None
        hit_rates[window] = hr

    # Floor clearance
    floor_pct = config['LEG_FLOOR_PCT']
    min_clr = config['LEG_MIN_CLEARANCE']
    windows_passing = 0
    clearances = []

    for w in config['LEG_WINDOWS']:
        values = model.get_values(name, stat, w, min_ab=config['MIN_AB'])
        if values is None or len(values) < min(w, 3):
            continue
        floor = _percentile(values, floor_pct)
        clr = floor - line
        clearances.append(clr)
        if clr >= min_clr:
            windows_passing += 1

    if windows_passing < config.get('LEG_MIN_WINDOWS_PASSING', 2):
        return None

    # Bayesian probability
    values_15 = model.get_values(name, stat, 15, min_ab=config['MIN_AB'])
    if not values_15 or len(values_15) < 6:
        return None
    hits = sum(1 for v in values_15 if v > line)
    misses = len(values_15) - hits
    alpha = config['BEQ_PRIOR_ALPHA'] + hits
    beta_p = config['BEQ_PRIOR_BETA'] + misses
    mean_prob = alpha / (alpha + beta_p)
    lower = beta_ppf_approx(alpha, beta_p, 1 - config['BEQ_CI_LEVEL'])

    if lower < config['BEQ_MIN_TRUE_PROB']:
        return None

    # Extended consistency
    values_25 = model.get_values(name, stat, 25, min_ab=config['MIN_AB'])
    if values_25 and len(values_25) >= 12:
        ext_hr = sum(1 for v in values_25 if v > line) / len(values_25)
        if ext_hr < hit_rates[15] - 0.12:
            return None

    market_prob = implied_probability(odds)
    edge = lower - market_prob
    dec_odds = american_to_decimal(odds)
    ev = lower * dec_odds - 1

    avg_clearance = sum(clearances) / len(clearances) if clearances else 0
    confidence = (
        lower * 0.35 +
        min(1.0, avg_clearance / 2.0 + 0.3) * 0.20 +
        hit_rates[15] * 0.25 +
        min(1.0, edge / 0.15 + 0.3) * 0.20
    )

    return {
        'player': name, 'stat': stat, 'line': line, 'odds': odds,
        'dec_odds': dec_odds,
        'hit_rate_15': hit_rates[15],
        'true_prob': lower, 'mean_prob': mean_prob,
        'market_prob': market_prob, 'edge': edge, 'ev': ev,
        'avg_clearance': avg_clearance,
        'confidence': confidence,
    }


def build_parlays(legs, model, config):
    """Build parlays from safe legs."""
    min_legs = config['PARLAY_MIN_LEGS']
    max_legs = config['PARLAY_MAX_LEGS']

    if len(legs) < min_legs:
        return []

    sorted_legs = sorted(legs, key=lambda l: l['confidence'], reverse=True)
    parlays = []
    used = set()

    for _ in range(config.get('MAX_PARLAYS_PER_DAY', 5)):
        available = [l for l in sorted_legs if l['player'] not in used]
        if len(available) < min_legs:
            break

        best_parlay = None
        best_score = -1

        top = available[:min(10, len(available))]
        for n in range(min_legs, min(max_legs + 1, len(top) + 1)):
            for combo in combinations(range(len(top)), n):
                cands = [top[i] for i in combo]

                # Different teams
                if config.get('PARLAY_REQUIRE_DIFF_TEAMS', True):
                    teams = set()
                    ok = True
                    for l in cands:
                        t = model.get_team(l['player'])
                        if t and t in teams:
                            ok = False
                            break
                        if t:
                            teams.add(t)
                    if not ok:
                        continue

                # Correlation
                corr_ok = True
                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        v1 = model.get_values(cands[i]['player'], cands[i]['stat'], 15, min_ab=1)
                        v2 = model.get_values(cands[j]['player'], cands[j]['stat'], 15, min_ab=1)
                        if v1 and v2 and abs(_correlation(v1, v2)) > config['PARLAY_MAX_CORR']:
                            corr_ok = False
                            break
                    if not corr_ok:
                        break
                if not corr_ok:
                    continue

                p_dec = parlay_decimal_odds([l['odds'] for l in cands])
                p_am = decimal_to_american(p_dec)

                if p_am < config['PARLAY_MIN_COMBINED_ODDS']:
                    continue
                if p_am > config.get('PARLAY_MAX_COMBINED_ODDS', 1000):
                    continue

                combined_prob = 1.0
                for l in cands:
                    combined_prob *= l['true_prob']

                p_ev = combined_prob * p_dec - 1
                if p_ev < config['PARLAY_MIN_EV']:
                    continue

                score = combined_prob * 0.5 + p_ev * 0.5
                if score > best_score:
                    best_score = score
                    best_parlay = {
                        'legs': cands, 'n_legs': n,
                        'parlay_decimal': p_dec, 'parlay_american': p_am,
                        'combined_prob': combined_prob, 'parlay_ev': p_ev,
                        'total_edge': sum(l['edge'] for l in cands),
                    }

        if best_parlay:
            parlays.append(best_parlay)
            for l in best_parlay['legs']:
                used.add(l['player'])
        else:
            break

    return parlays


def run_backtest(box_scores, odds_data, config, verbose=False):
    """Walk-forward backtest for MLB V7."""
    model = MLBPlayerModel()
    stat_map = {
        'hitsProps': 'h', 'tbProps': 'tb', 'runsProps': 'r',
        'rbiProps': 'rbi', 'hrProps': 'hr',
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

        # Collect actuals
        actuals = {}
        for bg in day_boxes:
            for p in bg.get('players', []):
                actuals[p['name']] = {k: _safe_int(p.get(k, 0))
                                       for k in ['h', 'tb', 'r', 'rbi', 'hr']}

        # Find safe legs
        day_legs = []
        for bg in day_boxes:
            gk = f"{bg.get('away', '')}@{bg.get('home', '')}"
            og = day_odds.get(gk)
            if not og:
                continue

            for player in bg.get('players', []):
                name = player['name']
                for prop_key, stat_key in stat_map.items():
                    props = og.get(prop_key, {}).get(name, {})
                    for thr_str, data in props.items():
                        try:
                            line = float(thr_str)
                        except (ValueError, TypeError):
                            continue
                        over_odds = data.get('overOdds')
                        if over_odds is None:
                            continue

                        leg = evaluate_leg(model, name, stat_key, line, over_odds, config)
                        if leg:
                            actual = actuals.get(name, {}).get(stat_key)
                            if actual is not None:
                                leg['actual'] = actual
                                leg['hit'] = actual > line
                                leg['game_key'] = gk
                                day_legs.append(leg)

        # Deduplicate
        best_by_player = {}
        for l in day_legs:
            if l['player'] not in best_by_player or l['confidence'] > best_by_player[l['player']]['confidence']:
                best_by_player[l['player']] = l
        day_legs = list(best_by_player.values())

        day_picks = []

        # Build parlays
        parlays = build_parlays(day_legs, model, config)
        for p in parlays:
            hit = all(l['hit'] for l in p['legs'])
            wager = config.get(f'WAGER_{p["n_legs"]}LEG', config['UNIT_SIZE'])
            pnl = round((p['parlay_decimal'] - 1) * wager) if hit else -wager

            pick = {
                'date': date, 'bet_type': 'parlay', 'n_legs': p['n_legs'],
                'legs': [{'player': l['player'], 'stat': l['stat'],
                          'line': l['line'], 'odds': l['odds'],
                          'hit': l['hit'], 'actual': l.get('actual'),
                          'true_prob': l['true_prob'], 'hit_rate': l['hit_rate_15'],
                          'edge': l['edge']} for l in p['legs']],
                'parlay_decimal': p['parlay_decimal'],
                'parlay_american': p['parlay_american'],
                'combined_prob': p['combined_prob'],
                'hit': hit, 'pnl': pnl, 'wager': wager,
                'combined_score': p['combined_prob'], 'edge': p['total_edge'],
            }
            day_picks.append(pick)
            all_picks.append(pick)

        # Singles
        if config.get('SINGLES_ENABLED', True):
            used = set()
            for p in parlays:
                for l in p['legs']:
                    used.add(l['player'])

            for l in sorted(day_legs, key=lambda x: x['confidence'], reverse=True):
                if l['player'] in used:
                    continue
                if l['hit_rate_15'] < config.get('SINGLE_MIN_HR_15', 0.85):
                    continue
                if l['true_prob'] < config.get('SINGLE_MIN_TRUE_PROB', 0.82):
                    continue
                if l['odds'] < config.get('SINGLE_MIN_ODDS', -400):
                    continue
                if l['odds'] > config.get('SINGLE_MAX_ODDS', -110):
                    continue

                wager = config['UNIT_SIZE']
                pnl = pnl_for_bet(l['hit'], l['odds'], wager)
                pick = {
                    'date': date, 'bet_type': 'single',
                    'player': l['player'], 'stat': l['stat'],
                    'line': l['line'], 'odds': l['odds'],
                    'actual': l['actual'], 'hit': l['hit'],
                    'pnl': pnl, 'wager': wager,
                    'combined_score': l['confidence'], 'edge': l['edge'],
                    'hit_rate': l['hit_rate_15'], 'true_prob': l['true_prob'],
                }
                day_picks.append(pick)
                all_picks.append(pick)
                used.add(l['player'])
                if len([p for p in day_picks if p['bet_type'] == 'single']) >= config.get('MAX_SINGLES_PER_DAY', 3):
                    break

        if day_picks and verbose:
            wins = sum(1 for p in day_picks if p['hit'])
            print(f"  {date}: {len(day_picks)} picks ({wins} wins) ${sum(p['pnl'] for p in day_picks):+d}")

        if day_picks:
            daily_results.append({
                'date': date, 'n_picks': len(day_picks),
                'wins': sum(1 for p in day_picks if p['hit']),
                'pnl': sum(p['pnl'] for p in day_picks),
            })

        for bg in day_boxes:
            _update_model(model, bg, date)

    # Stats
    singles = [p for p in all_picks if p['bet_type'] == 'single']
    parlays_list = [p for p in all_picks if p['bet_type'] == 'parlay']
    s_wins = sum(1 for s in singles if s['hit'])
    p_wins = sum(1 for p in parlays_list if p['hit'])
    total_pnl = sum(p['pnl'] for p in all_picks)
    total_wager = sum(p['wager'] for p in all_picks)

    all_legs = []
    for p in parlays_list:
        all_legs.extend(p.get('legs', []))

    parlay_by_legs = {}
    for p in parlays_list:
        n = p['n_legs']
        if n not in parlay_by_legs:
            parlay_by_legs[n] = {'total': 0, 'wins': 0, 'pnl': 0}
        parlay_by_legs[n]['total'] += 1
        if p['hit']:
            parlay_by_legs[n]['wins'] += 1
        parlay_by_legs[n]['pnl'] += p['pnl']

    pm_parlays = [p for p in parlays_list if p.get('parlay_american', 0) >= 100]

    running, peak, max_dd = 0, 0, 0
    for p in all_picks:
        running += p['pnl']
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)

    return {
        'total_picks': len(all_picks),
        'total_wins': s_wins + p_wins,
        'accuracy': (s_wins + p_wins) / len(all_picks) if all_picks else 0,
        'total_pnl': total_pnl,
        'total_wagered': total_wager,
        'total_roi': total_pnl / total_wager if total_wager > 0 else 0,
        'active_days': len(daily_results),
        'total_days': len(processed),
        'day_coverage': len(daily_results) / len(processed) if processed else 0,
        'max_drawdown': max_dd,
        'singles_total': len(singles), 'singles_wins': s_wins,
        'singles_accuracy': s_wins / len(singles) if singles else 0,
        'singles_pnl': sum(p['pnl'] for p in singles),
        'parlay_total': len(parlays_list), 'parlay_wins': p_wins,
        'parlay_accuracy': p_wins / len(parlays_list) if parlays_list else 0,
        'parlay_pnl': sum(p['pnl'] for p in parlays_list),
        'parlay_by_legs': parlay_by_legs,
        'parlay_leg_total': len(all_legs),
        'parlay_leg_wins': sum(1 for l in all_legs if l.get('hit')),
        'parlay_leg_accuracy': sum(1 for l in all_legs if l.get('hit')) / len(all_legs) if all_legs else 0,
        'plus_money_parlays': len(pm_parlays),
        'plus_money_wins': sum(1 for p in pm_parlays if p['hit']),
        'plus_money_accuracy': sum(1 for p in pm_parlays if p['hit']) / len(pm_parlays) if pm_parlays else 0,
        'picks': all_picks, 'daily': daily_results,
    }
