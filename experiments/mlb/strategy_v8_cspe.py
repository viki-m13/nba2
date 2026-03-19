"""
MLB Strategy V8 — Convergence Score Parlay Engine (CSPE)
=========================================================
Patent-pending: Edge-centric continuous scoring with dual-tier
parlay construction. MLB-specific adaptation with discrete stat handling.

MLB adaptations vs NBA:
- Uses at-bats instead of minutes for activity filtering
- Poisson-aware scoring for discrete stats (h, tb, r, rbi, hr)
- Lower thresholds due to higher variance in baseball
- Adjusted window sizes for 162-game season

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
# CONFIG — MLB-specific tuning
# =========================================================================

MLB_V8_CONFIG = {
    # Data requirements (MLB-specific)
    'MIN_GAMES': 12,
    'MIN_AB': 3,

    # Bayesian parameters
    'BCF_PRIOR_ALPHA': 1.0,
    'BCF_PRIOR_BETA': 1.0,
    'BCF_CI_LEVEL': 0.80,

    # Z-Score Floor Analysis (smaller window for MLB)
    'ZFA_PERCENTILE': 0.15,
    'ZFA_WINDOW': 15,

    # Entropy-Calibrated Confidence
    'ECC_BINS': 5,

    # === CORE TIER — Very tight filtering for MLB ===
    'CORE_MIN_HR': 0.80,
    'CORE_MIN_BCF_LB': 0.70,
    'CORE_MIN_FLOOR_CLEARANCE': -999,
    'CORE_MIN_STREAK': 2,
    'CORE_LEG_MIN_ODDS': -600,
    'CORE_LEG_MAX_ODDS': -200,
    'CORE_SCORE_MIN': 0.60,
    'CORE_TOP_N': 5,
    'CORE_PARLAY_LEGS': [2],
    'CORE_MAX_PARLAYS': 1,
    'CORE_WAGER': 60,

    # === AMPLIFIER TIER — Disabled for MLB (market too efficient) ===
    'AMP_MIN_HR': 0.60,
    'AMP_MIN_BCF_LB': 0.52,
    'AMP_MIN_EDGE': 0.10,
    'AMP_MIN_STREAK': 2,
    'AMP_LEG_MIN_ODDS': -400,
    'AMP_LEG_MAX_ODDS': 350,
    'AMP_EDGE_SCORE_MIN': 0.15,
    'AMP_TOP_N': 8,
    'AMP_PARLAY_LEGS': [],  # Disabled — enable when more data available
    'AMP_MAX_PARLAYS': 2,
    'AMP_WAGER_3LEG': 15,

    # === HYBRID TIER — Disabled for MLB (use core+amp only) ===
    'HYBRID_ENABLED': False,

    # Shared settings — focus on hits only (most predictable MLB stat)
    'STATS_ALLOWED': ['h', 'tb'],
    'PARLAY_REQUIRE_DIFF_TEAMS': True,
}


# =========================================================================
# SCORING FUNCTIONS (MLB-specific)
# =========================================================================

def _percentile(values, pct):
    s = sorted(values)
    return s[max(0, min(len(s) - 1, int(len(s) * pct)))]


def _std_dev(values):
    if len(values) < 2:
        return 0.001
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return max(math.sqrt(var), 0.001)


def _stat_entropy(values, n_bins=5):
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


def score_leg(model, name, stat, line, odds, config, tier='core'):
    """Score a potential MLB parlay leg for Core or Amplifier tier."""
    if model.game_count(name) < config['MIN_GAMES']:
        return None
    if model.avg_ab(name, 10) < config['MIN_AB']:
        return None

    if tier == 'core':
        if odds < config['CORE_LEG_MIN_ODDS'] or odds > config['CORE_LEG_MAX_ODDS']:
            return None
    else:
        if odds < config['AMP_LEG_MIN_ODDS'] or odds > config['AMP_LEG_MAX_ODDS']:
            return None

    window = config.get('ZFA_WINDOW', 15)
    values = model.get_values(name, stat, window, min_ab=config['MIN_AB'])
    if not values or len(values) < 6:
        return None

    hits = sum(1 for v in values if v > line)
    n = len(values)
    hr = hits / n

    min_hr = config['CORE_MIN_HR'] if tier == 'core' else config['AMP_MIN_HR']
    if hr < min_hr:
        return None

    alpha = config['BCF_PRIOR_ALPHA'] + hits
    beta_p = config['BCF_PRIOR_BETA'] + (n - hits)
    mean_prob = alpha / (alpha + beta_p)
    bcf_lb = beta_ppf_approx(alpha, beta_p, 1 - config['BCF_CI_LEVEL'])

    min_bcf = config['CORE_MIN_BCF_LB'] if tier == 'core' else config['AMP_MIN_BCF_LB']
    if bcf_lb < min_bcf:
        return None

    floor_val = _percentile(values, config.get('ZFA_PERCENTILE', 0.15))
    clearance = floor_val - line
    std = _std_dev(values)
    z_floor = clearance / std

    if tier == 'core' and clearance < config.get('CORE_MIN_FLOOR_CLEARANCE', -999):
        return None

    recent = model.get_values(name, stat, 5, min_ab=config['MIN_AB'])
    streak = 0
    if recent:
        for v in reversed(recent):
            if v > line:
                streak += 1
            else:
                break

    min_streak = config['CORE_MIN_STREAK'] if tier == 'core' else config['AMP_MIN_STREAK']
    if streak < min_streak:
        return None

    # Multi-window consistency
    for w in [3, 5, 10]:
        vw = model.get_values(name, stat, w, min_ab=config['MIN_AB'])
        if vw and len(vw) >= min(w, 3):
            whr = sum(1 for v in vw if v > line) / len(vw)
            min_whr = 0.55 if tier == 'core' else 0.35
            if whr < min_whr:
                return None

    entropy = _stat_entropy(values, config.get('ECC_BINS', 5))
    ecc = 1.0 - entropy

    # AB stability (MLB equivalent of minutes CV)
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

    market_prob = implied_probability(odds)
    edge = bcf_lb - market_prob

    if tier == 'amp' and edge < config.get('AMP_MIN_EDGE', 0.05):
        return None

    # Scoring
    if tier == 'core':
        z_norm = min(1.0, max(0.0, z_floor / 2.0))
        score = (
            0.25 * z_norm + 0.20 * bcf_lb + 0.15 * hr + 0.15 * ecc +
            0.10 * min(1.0, streak / 4.0) + 0.10 * stability +
            0.05 * min(1.0, max(0.0, edge / 0.25))
        )
    else:
        edge_norm = min(1.0, max(0.0, edge / 0.25))
        ev = bcf_lb * american_to_decimal(odds) - 1.0
        ev_norm = min(1.0, max(0.0, ev / 0.40))
        score = (
            0.30 * edge_norm + 0.20 * ev_norm + 0.15 * bcf_lb +
            0.10 * hr + 0.10 * ecc + 0.08 * min(1.0, streak / 4.0) +
            0.07 * stability
        )

    return {
        'player': name, 'stat': stat, 'line': line, 'odds': odds,
        'dec_odds': american_to_decimal(odds), 'score': score,
        'bcf_lb': bcf_lb, 'mean_prob': mean_prob, 'hr_20': hr,
        'z_floor': z_floor, 'clearance': clearance, 'entropy': entropy,
        'ecc': ecc, 'streak': streak, 'stability': stability,
        'market_prob': market_prob, 'edge': edge,
        'ev': bcf_lb * american_to_decimal(odds) - 1.0, 'tier': tier,
    }


# =========================================================================
# PARLAY CONSTRUCTION (shared with NBA — same logic)
# =========================================================================

def build_parlays(elite_legs, n_legs, config, max_parlays=1):
    if len(elite_legs) < n_legs:
        return []
    parlays = []
    used_players = set()
    for _ in range(max_parlays):
        available = [l for l in elite_legs if l['player'] not in used_players]
        if len(available) < n_legs:
            break
        selected = []
        used_teams = set()
        for l in available:
            if config.get('PARLAY_REQUIRE_DIFF_TEAMS', True):
                team = l.get('team', '')
                if team and team in used_teams:
                    continue
                if team:
                    used_teams.add(team)
            selected.append(l)
            if len(selected) >= n_legs:
                break
        if len(selected) < n_legs:
            break
        leg_odds = [l['odds'] for l in selected]
        p_dec = parlay_decimal_odds(leg_odds)
        p_am = decimal_to_american(p_dec)
        combined_prob = 1.0
        for l in selected:
            combined_prob *= l['bcf_lb']
        parlay_ev = combined_prob * p_dec - 1
        kelly_raw = parlay_ev / (p_dec - 1) if p_dec > 1 else 0
        kelly_frac = max(0, min(0.5, kelly_raw * 0.25))
        parlays.append({
            'legs': selected, 'n_legs': n_legs,
            'parlay_decimal': p_dec, 'parlay_american': p_am,
            'combined_prob': combined_prob, 'parlay_ev': parlay_ev,
            'total_edge': sum(l['edge'] for l in selected),
            'avg_score': sum(l['score'] for l in selected) / len(selected),
            'kelly_frac': kelly_frac,
        })
        for l in selected:
            used_players.add(l['player'])
    return parlays


# =========================================================================
# WALK-FORWARD BACKTEST
# =========================================================================

def run_backtest(box_scores, odds_data, config, verbose=False):
    """Walk-forward backtest for MLB V8 CSPE."""
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

        core_legs = []
        amp_legs = []
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
                        actual = actuals.get(name, {}).get(stat_key)
                        if actual is None:
                            continue

                        core_leg = score_leg(model, name, stat_key, line, over_odds, config, tier='core')
                        if core_leg:
                            core_leg['actual'] = actual
                            core_leg['hit'] = actual > line
                            core_leg['team'] = player_teams.get(name, '')
                            core_leg['date'] = date
                            core_legs.append(core_leg)

                        amp_leg = score_leg(model, name, stat_key, line, over_odds, config, tier='amp')
                        if amp_leg:
                            amp_leg['actual'] = actual
                            amp_leg['hit'] = actual > line
                            amp_leg['team'] = player_teams.get(name, '')
                            amp_leg['date'] = date
                            amp_legs.append(amp_leg)

        day_picks = []

        # Core tier
        core_legs.sort(key=lambda l: l['score'], reverse=True)
        seen = set()
        core_elite = []
        for l in core_legs:
            if l['player'] not in seen:
                core_elite.append(l)
                seen.add(l['player'])
                if len(core_elite) >= config.get('CORE_TOP_N', 8):
                    break
        core_elite = [l for l in core_elite if l['score'] >= config.get('CORE_SCORE_MIN', 0.50)]

        for n_legs in config.get('CORE_PARLAY_LEGS', [2]):
            parlays = build_parlays(core_elite, n_legs, config, config.get('CORE_MAX_PARLAYS', 1))
            for p in parlays:
                hit = all(l['hit'] for l in p['legs'])
                wager = config.get('CORE_WAGER', 100)
                pnl = round((p['parlay_decimal'] - 1) * wager) if hit else -wager
                day_picks.append(_make_pick(p, date, n_legs, hit, pnl, wager, 'core'))

        # Amplifier tier
        amp_legs.sort(key=lambda l: l['score'], reverse=True)
        seen_amp = set()
        amp_elite = []
        for l in amp_legs:
            if l['player'] not in seen_amp:
                amp_elite.append(l)
                seen_amp.add(l['player'])
                if len(amp_elite) >= config.get('AMP_TOP_N', 12):
                    break
        amp_elite = [l for l in amp_elite if l['score'] >= config.get('AMP_EDGE_SCORE_MIN', 0.08)]

        for n_legs in config.get('AMP_PARLAY_LEGS', [3, 4]):
            wager = config.get(f'AMP_WAGER_{n_legs}LEG', 20)
            parlays = build_parlays(amp_elite, n_legs, config, config.get('AMP_MAX_PARLAYS', 3))
            for p in parlays:
                hit = all(l['hit'] for l in p['legs'])
                pnl = round((p['parlay_decimal'] - 1) * wager) if hit else -wager
                day_picks.append(_make_pick(p, date, n_legs, hit, pnl, wager, 'amp'))

        # Hybrid tier
        if config.get('HYBRID_ENABLED', True):
            n_anchors = config.get('HYBRID_ANCHOR_COUNT', 2)
            anchors = core_elite[:n_anchors + 2]
            anchor_players = set(l['player'] for l in anchors)
            amp_for_hybrid = [l for l in amp_elite if l['player'] not in anchor_players]

            for n_amps in config.get('HYBRID_AMP_COUNT', [2, 3]):
                if len(anchors) < n_anchors or len(amp_for_hybrid) < n_amps:
                    continue
                used_h = set()
                for _ in range(config.get('HYBRID_MAX_PARLAYS', 2)):
                    sel_a, teams_h = [], set()
                    for l in anchors:
                        if l['player'] in used_h:
                            continue
                        t = l.get('team', '')
                        if t and t in teams_h:
                            continue
                        if t:
                            teams_h.add(t)
                        sel_a.append(l)
                        if len(sel_a) >= n_anchors:
                            break
                    if len(sel_a) < n_anchors:
                        break
                    sel_amp = []
                    for l in amp_for_hybrid:
                        if l['player'] in used_h:
                            continue
                        t = l.get('team', '')
                        if t and t in teams_h:
                            continue
                        if t:
                            teams_h.add(t)
                        sel_amp.append(l)
                        if len(sel_amp) >= n_amps:
                            break
                    if len(sel_amp) < n_amps:
                        break
                    all_legs = sel_a + sel_amp
                    total = len(all_legs)
                    p_dec = parlay_decimal_odds([l['odds'] for l in all_legs])
                    cp = 1.0
                    for l in all_legs:
                        cp *= l['bcf_lb']
                    hp = {'legs': all_legs, 'n_legs': total, 'parlay_decimal': p_dec,
                          'parlay_american': decimal_to_american(p_dec),
                          'combined_prob': cp, 'parlay_ev': cp * p_dec - 1,
                          'total_edge': sum(l['edge'] for l in all_legs),
                          'avg_score': sum(l['score'] for l in all_legs) / len(all_legs)}
                    hit = all(l['hit'] for l in all_legs)
                    wager = config.get(f'HYBRID_WAGER_{total}LEG', 10)
                    pnl = round((p_dec - 1) * wager) if hit else -wager
                    day_picks.append(_make_pick(hp, date, total, hit, pnl, wager, 'hybrid'))
                    for l in all_legs:
                        used_h.add(l['player'])

        for pick in day_picks:
            all_picks.append(pick)

        if day_picks and verbose:
            for p in day_picks:
                lh = sum(1 for l in p['legs'] if l['hit'])
                print(f"  {date}: [{p.get('tier','?')}] {p['n_legs']}L ({p['parlay_american']:+d}) "
                      f"[{lh}/{p['n_legs']}] [{'WIN' if p['hit'] else 'LOSS'}] ${p['pnl']:+d}")

        if day_picks:
            daily_results.append({
                'date': date, 'n_picks': len(day_picks),
                'wins': sum(1 for p in day_picks if p['hit']),
                'pnl': sum(p['pnl'] for p in day_picks),
            })

        for bg in day_boxes:
            _update_model(model, bg, date)

    return _compute_stats(all_picks, daily_results, list(processed), config)


def _make_pick(parlay, date, n_legs, hit, pnl, wager, tier):
    return {
        'date': date, 'bet_type': f'{tier}_parlay_{n_legs}L',
        'n_legs': n_legs, 'tier': tier,
        'legs': [{
            'player': l['player'], 'stat': l['stat'],
            'line': l['line'], 'odds': l['odds'],
            'hit': l['hit'], 'actual': l.get('actual'),
            'score': l['score'], 'bcf_lb': l['bcf_lb'],
            'hr_20': l['hr_20'], 'z_floor': l['z_floor'],
            'clearance': l['clearance'], 'streak': l['streak'],
            'edge': l['edge'], 'ev': l.get('ev', 0), 'tier': tier,
        } for l in parlay['legs']],
        'parlay_decimal': parlay['parlay_decimal'],
        'parlay_american': parlay['parlay_american'],
        'combined_prob': parlay['combined_prob'],
        'parlay_ev': parlay['parlay_ev'],
        'hit': hit, 'pnl': pnl, 'wager': wager,
        'combined_score': parlay['avg_score'],
        'edge': parlay['total_edge'],
    }


def _compute_stats(all_picks, daily_results, dates, config):
    by_legs = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0, 'wagered': 0, 'legs_total': 0, 'legs_hits': 0})
    by_tier = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0, 'wagered': 0})
    for p in all_picks:
        tier = p.get('tier', 'unknown')
        key = f"{tier}_{p['n_legs']}L"
        by_legs[key]['total'] += 1
        if p['hit']:
            by_legs[key]['wins'] += 1
        by_legs[key]['pnl'] += p['pnl']
        by_legs[key]['wagered'] += p['wager']
        for l in p.get('legs', []):
            by_legs[key]['legs_total'] += 1
            if l.get('hit'):
                by_legs[key]['legs_hits'] += 1
        by_tier[tier]['total'] += 1
        if p['hit']:
            by_tier[tier]['wins'] += 1
        by_tier[tier]['pnl'] += p['pnl']
        by_tier[tier]['wagered'] += p['wager']

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
        'by_legs': {k: {**v, 'accuracy': v['wins']/v['total'] if v['total'] else 0,
                         'roi': v['pnl']/v['wagered'] if v['wagered'] else 0,
                         'leg_accuracy': v['legs_hits']/v['legs_total'] if v['legs_total'] else 0}
                    for k, v in by_legs.items()},
        'by_tier': {k: {**v, 'accuracy': v['wins']/v['total'] if v['total'] else 0,
                         'roi': v['pnl']/v['wagered'] if v['wagered'] else 0}
                    for k, v in by_tier.items()},
        'picks': all_picks, 'daily': daily_results,
    }


def generate_picks_for_date(model, odds_for_date, config, stats_allowed=None):
    """Generate MLB V8 CSPE picks for a specific date."""
    if stats_allowed is None:
        stats_allowed = config.get('STATS_ALLOWED', ['h', 'tb', 'r', 'rbi', 'hr'])
    stat_map = {
        'hitsProps': 'h', 'tbProps': 'tb', 'runsProps': 'r',
        'rbiProps': 'rbi', 'hrProps': 'hr',
    }
    core_legs, amp_legs = [], []
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
                    cl = score_leg(model, player, stat, line_val, over_odds, config, tier='core')
                    if cl:
                        cl['team'] = model.get_team(player) or ''
                        core_legs.append(cl)
                    al = score_leg(model, player, stat, line_val, over_odds, config, tier='amp')
                    if al:
                        al['team'] = model.get_team(player) or ''
                        amp_legs.append(al)

    recs = []
    # Core
    core_legs.sort(key=lambda l: l['score'], reverse=True)
    seen = set()
    ce = []
    for l in core_legs:
        if l['player'] not in seen:
            ce.append(l)
            seen.add(l['player'])
            if len(ce) >= config.get('CORE_TOP_N', 8):
                break
    ce = [l for l in ce if l['score'] >= config.get('CORE_SCORE_MIN', 0.50)]
    for nl in config.get('CORE_PARLAY_LEGS', [2]):
        for p in build_parlays(ce, nl, config, config.get('CORE_MAX_PARLAYS', 1)):
            recs.append(_make_rec(p, nl, config.get('CORE_WAGER', 100), 'core'))

    # Amplifier
    amp_legs.sort(key=lambda l: l['score'], reverse=True)
    seen2 = set()
    ae = []
    for l in amp_legs:
        if l['player'] not in seen2:
            ae.append(l)
            seen2.add(l['player'])
            if len(ae) >= config.get('AMP_TOP_N', 12):
                break
    ae = [l for l in ae if l['score'] >= config.get('AMP_EDGE_SCORE_MIN', 0.08)]
    for nl in config.get('AMP_PARLAY_LEGS', [3, 4]):
        w = config.get(f'AMP_WAGER_{nl}LEG', 20)
        for p in build_parlays(ae, nl, config, config.get('AMP_MAX_PARLAYS', 3)):
            recs.append(_make_rec(p, nl, w, 'amp'))

    return recs


def _make_rec(parlay, n_legs, wager, tier):
    return {
        'type': f'{tier} {n_legs}-leg parlay', 'tier': tier,
        'n_legs': n_legs,
        'parlay_american': parlay['parlay_american'],
        'parlay_decimal': parlay['parlay_decimal'],
        'combined_prob': parlay['combined_prob'],
        'expected_value': parlay['parlay_ev'],
        'wager': wager,
        'legs': [{
            'player': l['player'], 'stat': l['stat'], 'line': l['line'],
            'odds': l['odds'], 'direction': 'over',
            'score': round(l['score'], 3), 'bcf_lb': round(l['bcf_lb'], 3),
            'hit_rate': round(l['hr_20'], 3), 'edge': round(l['edge'], 3),
            'z_floor': round(l['z_floor'], 2), 'clearance': round(l['clearance'], 1),
            'streak': l['streak'],
        } for l in parlay['legs']],
    }
