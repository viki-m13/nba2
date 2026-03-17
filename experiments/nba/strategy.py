"""
NBA MAX-ROI Strategy
=====================
Targets 1000%+ ROI through:
1. PLUS-MONEY AND NEAR-EVEN LINES — each win pays 1-10x wager
2. ULTRA-SELECTIVE — only massive edges (>15% Bayesian edge)
3. PARLAY-CENTRIC — 2-4 leg parlays at +300 to +2000
4. AGGRESSIVE KELLY — size up on highest-confidence opportunities
5. VALUE HUNTING — find the market's biggest mispricings

Philosophy: A 40% hit rate on +200 lines yields 20% ROI.
A 30% hit rate on 3-leg parlays at +700 yields massive ROI.
We only need to find consistent mispricings, not predict everything.

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
# DEFAULT CONFIG — tuned for EXTREME ROI
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

    # Parlay — the ROI engine
    'PARLAY_MIN_LEGS': 2,
    'PARLAY_MAX_LEGS': 4,
    'PARLAY_MAX_CORR': 0.35,
    'PARLAY_MIN_EDGE': 0.06,

    # Odds filter — TARGET PLUS-MONEY AND NEAR-EVEN
    'MIN_ODDS': -250,
    'MAX_ODDS': 500,

    # Bankroll — aggressive Kelly on high-edge spots
    'UNIT_SIZE': 100,
    'KELLY_FRACTION': 0.35,
    'MAX_DAILY_BETS': 6,
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
    'GATE_MIN_HIT_RATE': (0.25, 0.75),
    'GATE_MIN_COMBINED': (0.10, 0.55),
    'MIN_EV': (0.03, 0.25),
    'SINGLE_MIN_SCORE': (0.25, 0.70),
    'PARLAY_LEG_MIN_SCORE': (0.15, 0.55),
    'PARLAY_MAX_CORR': (0.15, 0.55),
    'PARLAY_MIN_EDGE': (0.02, 0.20),
    'PARLAY_MAX_LEGS': (2, 4),
    'MIN_ODDS': (-350, -100),
    'MAX_ODDS': (150, 800),
    'MIN_GAMES': (8, 18),
    'MIN_MINUTES': (12, 22),
    'KELLY_FRACTION': (0.15, 0.50),
}


# =========================================================================
# PLAYER MODEL
# =========================================================================

class NBAPlayerModel:
    def __init__(self, max_history=50):
        self.profiles = {}
        self.max_history = max_history

    def update(self, name, stats, date, team, home_away, opponent):
        if name not in self.profiles:
            self.profiles[name] = {'games': [], 'team': ''}

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
        if len(self.profiles[name]['games']) > self.max_history:
            self.profiles[name]['games'] = self.profiles[name]['games'][-self.max_history:]

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
# SIGNAL COMPONENTS
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


def compute_signal(model, name, stat, line, market_odds, config, home_away=None):
    """Combined signal — ROI-focused, works with plus-money lines."""
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

    # Combined score
    components = [gft['score'], min(1.0, beq['edge'] / 0.20 + 0.5), esi['stability']]
    log_sum = sum(math.log(max(0.001, c)) for c in components)
    combined = math.exp(log_sum / len(components)) + ctx
    combined = max(0, min(1.0, combined))

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
    }


# =========================================================================
# PARLAY CONSTRUCTION — the ROI engine
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

    parlays.sort(key=lambda p: p['parlay_ev'], reverse=True)
    return parlays


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
            for player in bg.get('players', []):
                name = player['name']
                team = player.get('team', '')
                ha = 'home' if team == home_t else 'away'

                # All prop types
                for prop_key, stat_key, actual_key in [
                    ('playerProps', 'pts', 'pts'),
                    ('rebProps', 'reb', 'reb'),
                    ('astProps', 'ast', 'ast'),
                ]:
                    props = og.get(prop_key, {}).get(name, {})
                    for thr, data in props.items():
                        _try_signal(model, name, stat_key, thr, data, config, ha,
                                   player, actual_key, day_signals, date, gk)

                # PRA props
                pra_p = og.get('praProps', {}).get(name, {})
                for thr, data in pra_p.items():
                    _try_signal_pra(model, name, thr, data, config, ha,
                                    player, day_signals, date, gk)

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

            # Parlays — the real ROI driver
            parlay_elig = [s for s in unique if s['combined_score'] >= config['PARLAY_LEG_MIN_SCORE']]
            parlays = build_parlays(parlay_elig, model, config)
            for p in parlays[:2]:  # Up to 2 parlays per day
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
                    'date': date, 'n_picks': len(day_picks),
                    'wins': sum(1 for p in day_picks if p.get('hit')),
                    'pnl': sum(p.get('pnl', 0) for p in day_picks),
                })

        for bg in day_boxes:
            _update_model(model, bg, date)

    return _compute_stats(all_picks, daily_results, config)


def _try_signal(model, name, stat, thr_str, data, config, ha, player,
                actual_key, signals, date, gk):
    try:
        line = float(thr_str)
    except (ValueError, TypeError):
        return
    odds_val = data.get('overOdds')
    if odds_val is None:
        return
    if odds_val < config['MIN_ODDS'] or odds_val > config['MAX_ODDS']:
        return
    sig = compute_signal(model, name, stat, line, odds_val, config, home_away=ha)
    if sig:
        actual = _safe_int(player.get(actual_key, 0))
        sig['actual'] = actual
        sig['hit'] = actual > line
        sig['date'] = date
        sig['game_key'] = gk
        signals.append(sig)


def _try_signal_pra(model, name, thr_str, data, config, ha, player,
                    signals, date, gk):
    try:
        line = float(thr_str)
    except (ValueError, TypeError):
        return
    odds_val = data.get('overOdds')
    if odds_val is None:
        return
    if odds_val < config['MIN_ODDS'] or odds_val > config['MAX_ODDS']:
        return
    sig = compute_signal(model, name, 'pra', line, odds_val, config, home_away=ha)
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
    for p in game_data.get('players', []):
        if _safe_int(p.get('min', 0)) < 5:
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
        'individual_accuracy': s_wins / len(singles) if singles else 0,
        'accuracy': (s_wins + p_wins) / len(all_picks) if all_picks else 0,
        'max_drawdown': max_dd, 'max_loss_streak': max_loss_streak,
        'max_win_streak': max_win_streak,
        'active_days': len(daily_results),
        'avg_daily_pnl': total_pnl / len(daily_results) if daily_results else 0,
        'picks': all_picks, 'daily': daily_results,
    }
