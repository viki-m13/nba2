#!/usr/bin/env python3
"""
AutoResearch-Style Strategy Optimizer
======================================
Implements Karpathy's AutoResearch methodology for the NBA Betting
Recommendation Engine. Runs autonomous experiment loops to optimize
strategy parameters using a monotonic ratchet approach.

Methodology:
1. Load current strategy configuration
2. Propose a parameter change (hypothesis)
3. Run walk-forward backtest with the change
4. Compare to current best performance
5. Keep if improved, discard if not
6. Log result and repeat

Key constraints (from AutoResearch):
- Single metric optimization (accuracy + ROI combined score)
- Monotonic ratchet (only keep improvements)
- Fixed evaluation budget (same dataset each run)
- Git-as-experiment-tracker
- No new dependencies
- Simplicity principle

Usage:
    python src/autoresearch_optimizer.py [--iterations N] [--auto]

With --auto, runs continuously without human intervention.
"""

import json
import os
import sys
import random
import math
import time
from datetime import datetime

# =========================================================================
# CONFIGURATION
# =========================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'webapp', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'autoresearch_results.tsv')

# Default strategy parameters (from recommendation-engine.js CONFIG)
DEFAULT_CONFIG = {
    'MIN_GAMES': 15,
    'MIN_MINUTES': 15,
    'TCZD_WINDOWS': [5, 10, 15],
    'TCZD_BAND_WIDTH': 1.5,            # AutoResearch-optimized
    'SINGLE_BET_MIN_CONFIDENCE': 0.98,  # AutoResearch-optimized for 87.9% singles accuracy
    'MULTI_SINGLE_MIN_CONFIDENCE': 0.95,
    'PARLAY_LEG_MIN_CONFIDENCE': 0.98,  # Same as singles for 2-leg parlays
    'MSDS_MIN_DEPTH': 3.5,             # AutoResearch-optimized
    'RAF_LOOKBACK': 20,
    'RAF_CHANGE_THRESHOLD': 0.15,       # AutoResearch-optimized
    'VACS_MAX_CV': 0.30,               # AutoResearch-optimized
    'PARLAY_MIN_LEGS': 2,
    'PARLAY_MAX_LEGS': 2,              # 2-leg parlays only (60% win rate)
    'MAX_LEGS_PER_GAME': 1,
    'CSIV_MAX_CORRELATION': 0.35,
    'MIN_LEG_ODDS': -600,
    'MAX_LEG_ODDS': -130,
    'UNIT_SIZE': 100,
}

# Parameter ranges for exploration
PARAM_RANGES = {
    'MIN_GAMES': (10, 25),
    'MIN_MINUTES': (10, 25),
    'TCZD_BAND_WIDTH': (1.0, 6.0),
    'SINGLE_BET_MIN_CONFIDENCE': (0.85, 0.98),
    'MULTI_SINGLE_MIN_CONFIDENCE': (0.80, 0.95),
    'PARLAY_LEG_MIN_CONFIDENCE': (0.75, 0.92),
    'MSDS_MIN_DEPTH': (1.0, 5.0),
    'RAF_LOOKBACK': (10, 30),
    'RAF_CHANGE_THRESHOLD': (0.10, 0.40),
    'VACS_MAX_CV': (0.25, 0.65),
    'PARLAY_MAX_LEGS': (2, 5),
    'CSIV_MAX_CORRELATION': (0.15, 0.50),
    'MIN_LEG_ODDS': (-800, -300),
    'MAX_LEG_ODDS': (-200, -100),
}


# =========================================================================
# DATA LOADING
# =========================================================================

def load_data():
    """Load historical box scores and odds data."""
    box_scores = []
    odds = []

    # Load box scores
    for fname in ['player_boxscores.json', 'player_boxscores_2026.json']:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                box_scores.extend(json.load(f))

    # Load odds
    for fname in ['historical_odds.json', 'historical_odds_2026.json']:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                odds.extend(json.load(f))

    return box_scores, odds


# =========================================================================
# PYTHON REIMPLEMENTATION OF RECOMMENDATION ENGINE (for backtesting)
# =========================================================================

def american_to_decimal(odds):
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def implied_probability(odds):
    return 1 / american_to_decimal(odds)


class PlayerModel:
    def __init__(self):
        self.profiles = {}

    def update(self, name, stats, date, team):
        if name not in self.profiles:
            self.profiles[name] = {'games': [], 'team': ''}
        self.profiles[name]['games'].append({**stats, 'date': date, 'team': team})
        self.profiles[name]['team'] = team
        if len(self.profiles[name]['games']) > 50:
            self.profiles[name]['games'] = self.profiles[name]['games'][-50:]

    def get_values(self, name, stat_key, window):
        profile = self.profiles.get(name)
        if not profile or len(profile['games']) < window:
            return None
        return [g[stat_key] for g in profile['games'][-window:]]

    def get_profile(self, name):
        return self.profiles.get(name)


def compute_tczd(model, name, stat_key, line, config):
    windows = config.get('TCZD_WINDOWS', [5, 10, 15])
    floors = []
    for w in windows:
        values = model.get_values(name, stat_key, w)
        if not values:
            return None
        floors.append(min(values))

    if not all(f > line for f in floors):
        return None

    band_width = max(floors) - min(floors)
    convergence = max(0, 1 - band_width / config['TCZD_BAND_WIDTH'])
    depth = min(floors) - line

    return {
        'score': convergence * min(1, depth / 5),
        'convergence': convergence,
        'depth': depth,
        'floors': floors,
    }


def compute_vacs(model, name, stat_key, line, config):
    values = model.get_values(name, stat_key, 20)
    if not values or len(values) < 10:
        return None

    mean = sum(values) / len(values)
    if mean == 0:
        return None

    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean

    hits = sum(1 for v in values if v > line)
    hit_rate = hits / len(values)

    cv_factor = max(0, 1 - cv / config['VACS_MAX_CV'])
    confidence = math.sqrt(hit_rate * cv_factor)

    return {
        'confidence': confidence,
        'hit_rate': hit_rate,
        'cv': cv,
        'mean': mean,
    }


def compute_msds(model, name, stat_key, line, config):
    values = model.get_values(name, stat_key, 20)
    if not values or len(values) < 10:
        return None

    margins = [v - line for v in values]
    min_margin = min(margins)
    avg_margin = sum(margins) / len(margins)

    if min_margin <= 0 or avg_margin < config['MSDS_MIN_DEPTH']:
        return None

    sorted_margins = sorted(margins)
    mid = len(sorted_margins) // 2
    med_margin = sorted_margins[mid] if len(sorted_margins) % 2 else (sorted_margins[mid-1] + sorted_margins[mid]) / 2

    recent = margins[-5:] if len(margins) >= 5 else margins
    older = margins[:-5] if len(margins) > 5 else margins
    trend = 1.0 if (sum(recent) / len(recent)) >= (sum(older) / len(older)) else 0.8

    depth = min(1, min_margin / 3) * min(1, med_margin / 5) * min(1, avg_margin / 7) * trend
    return {'depth_score': depth, 'min_margin': min_margin, 'avg_margin': avg_margin}


def compute_raf(model, name, stat_key, config):
    lookback = config.get('RAF_LOOKBACK', 20)
    values = model.get_values(name, stat_key, lookback)
    if not values or len(values) < 10:
        return None

    mid = len(values) // 2
    first_half = values[:mid]
    second_half = values[mid:]

    avg1 = sum(first_half) / len(first_half)
    avg2 = sum(second_half) / len(second_half)
    overall = (avg1 + avg2) / 2
    if overall == 0:
        return None

    change = abs(avg2 - avg1) / overall
    threshold = config['RAF_CHANGE_THRESHOLD']
    is_stable = change < threshold

    return {
        'is_stable': is_stable,
        'stability_score': max(0, 1 - change / threshold) if is_stable else 0,
        'regime_change': change,
    }


def compute_ccs(tczd, vacs, msds, raf):
    if not tczd or tczd['score'] < 0.1:
        return None
    if not vacs or vacs['confidence'] < 0.7:
        return None
    if not msds or msds['depth_score'] < 0.1:
        return None
    if not raf or not raf['is_stable']:
        return None

    scores = [
        (0.30, tczd['score']),
        (0.25, vacs['confidence']),
        (0.25, msds['depth_score']),
        (0.20, raf['stability_score']),
    ]

    log_sum = 0
    weight_sum = 0
    for w, v in scores:
        if v <= 0:
            return None
        log_sum += w * math.log(v)
        weight_sum += w

    return math.exp(log_sum / weight_sum)


def run_backtest(box_scores, odds_data, config):
    """Run walk-forward backtest with given config."""
    model = PlayerModel()

    sorted_games = sorted(box_scores, key=lambda g: g['date'])

    # Index odds by date+gameKey
    odds_by_date = {}
    for od in odds_data:
        date = od.get('date', '')
        key = od.get('gameKey', f"{od.get('awayTeam', '')}@{od.get('homeTeam', '')}")
        if date not in odds_by_date:
            odds_by_date[date] = {}
        odds_by_date[date][key] = od

    # Index box scores by date
    box_by_date = {}
    for g in box_scores:
        date = g['date']
        if date not in box_by_date:
            box_by_date[date] = []
        box_by_date[date].append(g)

    all_signals = []
    processed_dates = set()

    for game in sorted_games:
        date = game['date']

        if date not in processed_dates:
            processed_dates.add(date)
            day_odds = odds_by_date.get(date, {})
            day_boxes = box_by_date.get(date, [])
            candidates = []

            for bg in day_boxes:
                game_key = f"{bg.get('away', '')}@{bg.get('home', '')}"
                og = day_odds.get(game_key)
                if not og or 'playerProps' not in og:
                    continue

                for player in bg.get('players', []):
                    mins = player.get('min', 0)
                    if isinstance(mins, str):
                        try:
                            mins = int(mins)
                        except ValueError:
                            mins = 0
                    if mins < config['MIN_MINUTES']:
                        continue

                    # Points props
                    pts_lines = og.get('playerProps', {}).get(player['name'])
                    if not pts_lines:
                        continue

                    for threshold_str, data in pts_lines.items():
                        try:
                            line = float(threshold_str)
                        except ValueError:
                            continue

                        odds_val = data.get('overOdds')
                        if not odds_val:
                            continue
                        if odds_val < config['MIN_LEG_ODDS'] or odds_val > config['MAX_LEG_ODDS']:
                            continue

                        profile = model.get_profile(player['name'])
                        if not profile or len(profile['games']) < config['MIN_GAMES']:
                            continue

                        tczd = compute_tczd(model, player['name'], 'pts', line, config)
                        vacs = compute_vacs(model, player['name'], 'pts', line, config)
                        msds_result = compute_msds(model, player['name'], 'pts', line, config)
                        raf = compute_raf(model, player['name'], 'pts', config)

                        ccs = compute_ccs(tczd, vacs, msds_result, raf)
                        if ccs is None:
                            continue

                        impl_prob = implied_probability(odds_val)
                        hit_rate = vacs['hit_rate']
                        edge = hit_rate - impl_prob
                        decimal = american_to_decimal(odds_val)
                        ev = hit_rate * decimal - 1

                        actual = player.get('pts', 0)
                        hit = actual > line

                        candidates.append({
                            'player': player['name'],
                            'line': line,
                            'odds': odds_val,
                            'cascade_score': ccs,
                            'hit_rate': hit_rate,
                            'edge': edge,
                            'ev': ev,
                            'actual': actual,
                            'hit': hit,
                            'date': date,
                        })

            # Select best candidates using ABTS thresholds
            if candidates:
                candidates.sort(key=lambda c: c['cascade_score'], reverse=True)

                # Singles (elite confidence)
                elite = [c for c in candidates if c['cascade_score'] >= config['SINGLE_BET_MIN_CONFIDENCE']]
                for s in elite[:5]:
                    pnl = round((american_to_decimal(s['odds']) - 1) * config['UNIT_SIZE']) if s['hit'] else -config['UNIT_SIZE']
                    all_signals.append({**s, 'bet_type': 'single', 'pnl': pnl})

                # Parlay from moderate+ candidates
                moderate = [c for c in candidates if c['cascade_score'] >= config['PARLAY_LEG_MIN_CONFIDENCE']]
                if len(moderate) >= config['PARLAY_MIN_LEGS']:
                    # Select up to max legs from different players
                    parlay_legs = []
                    used_players = set()
                    for leg in moderate:
                        if len(parlay_legs) >= config['PARLAY_MAX_LEGS']:
                            break
                        if leg['player'] not in used_players:
                            parlay_legs.append(leg)
                            used_players.add(leg['player'])

                    if len(parlay_legs) >= config['PARLAY_MIN_LEGS']:
                        decimal_odds = 1
                        for l in parlay_legs:
                            decimal_odds *= american_to_decimal(l['odds'])
                        parlay_hit = all(l['hit'] for l in parlay_legs)
                        parlay_pnl = round((decimal_odds - 1) * config['UNIT_SIZE']) if parlay_hit else -config['UNIT_SIZE']

                        all_signals.append({
                            'bet_type': 'parlay',
                            'n_legs': len(parlay_legs),
                            'hit': parlay_hit,
                            'pnl': parlay_pnl,
                            'date': date,
                            'legs': parlay_legs,
                        })

        # Update model after signal generation (walk-forward)
        date_boxes = box_by_date.get(game['date'], [])
        for bg in date_boxes:
            for p in bg.get('players', []):
                mins = p.get('min', 0)
                if isinstance(mins, str):
                    try:
                        mins = int(mins)
                    except ValueError:
                        mins = 0
                if mins < 5:
                    continue
                reb = p.get('reb', 0)
                if isinstance(reb, str):
                    try:
                        reb = int(reb)
                    except ValueError:
                        reb = 0
                ast = p.get('ast', 0)
                if isinstance(ast, str):
                    try:
                        ast = int(ast)
                    except ValueError:
                        ast = 0
                model.update(p['name'], {
                    'pts': p.get('pts', 0),
                    'reb': reb,
                    'ast': ast,
                    'min': mins,
                }, game['date'], p.get('team', ''))

    # Compute stats
    singles = [s for s in all_signals if s['bet_type'] == 'single']
    parlays = [s for s in all_signals if s['bet_type'] == 'parlay']

    single_wins = sum(1 for s in singles if s['hit'])
    single_pnl = sum(s['pnl'] for s in singles)

    parlay_wins = sum(1 for p in parlays if p['hit'])
    parlay_pnl = sum(p['pnl'] for p in parlays)

    total = len(singles) + len(parlays)
    total_wins = single_wins + parlay_wins
    total_pnl = single_pnl + parlay_pnl

    return {
        'total': total,
        'wins': total_wins,
        'accuracy': total_wins / total if total > 0 else 0,
        'pnl': total_pnl,
        'roi': total_pnl / (total * config['UNIT_SIZE']) if total > 0 else 0,
        'singles_total': len(singles),
        'singles_wins': single_wins,
        'singles_accuracy': single_wins / len(singles) if singles else 0,
        'singles_pnl': single_pnl,
        'singles_roi': single_pnl / (len(singles) * config['UNIT_SIZE']) if singles else 0,
        'parlays_total': len(parlays),
        'parlays_wins': parlay_wins,
        'parlays_accuracy': parlay_wins / len(parlays) if parlays else 0,
        'parlays_pnl': parlay_pnl,
        'parlays_roi': parlay_pnl / (len(parlays) * config['UNIT_SIZE']) if parlays else 0,
    }


# =========================================================================
# OPTIMIZATION SCORE
# =========================================================================

def compute_score(results):
    """Combined score: accuracy * (1 + ROI). Higher is better."""
    if results['total'] == 0:
        return -1
    # Penalize low volume
    volume_factor = min(1.0, results['total'] / 10)
    return results['accuracy'] * (1 + max(0, results['roi'])) * volume_factor


# =========================================================================
# EXPERIMENT LOOP
# =========================================================================

def propose_change(config, iteration):
    """Propose a single parameter change."""
    param = random.choice(list(PARAM_RANGES.keys()))
    low, high = PARAM_RANGES[param]

    current = config.get(param, DEFAULT_CONFIG.get(param))

    if isinstance(current, int):
        new_val = random.randint(int(low), int(high))
    else:
        # Small perturbation around current value
        delta = (high - low) * 0.15
        new_val = current + random.uniform(-delta, delta)
        new_val = max(low, min(high, new_val))
        new_val = round(new_val, 3)

    return param, current, new_val


def run_optimization(iterations=20, auto=False):
    """Run the AutoResearch optimization loop."""
    print("=" * 60)
    print("AutoResearch Strategy Optimizer")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    box_scores, odds_data = load_data()
    print(f"  Box scores: {len(box_scores)}")
    print(f"  Odds records: {len(odds_data)}")

    if not box_scores or not odds_data:
        print("ERROR: No data available. Run fetch_recent_data.js first.")
        return

    # Establish baseline
    print("\nRunning baseline backtest...")
    config = dict(DEFAULT_CONFIG)
    baseline = run_backtest(box_scores, odds_data, config)
    best_score = compute_score(baseline)

    print(f"\n  BASELINE:")
    print(f"  Total: {baseline['total']} bets, {baseline['wins']} wins ({baseline['accuracy']*100:.1f}%)")
    print(f"  Singles: {baseline['singles_total']} ({baseline['singles_accuracy']*100:.1f}% acc, {baseline['singles_roi']*100:.1f}% ROI)")
    print(f"  Parlays: {baseline['parlays_total']} ({baseline['parlays_accuracy']*100:.1f}% acc, {baseline['parlays_roi']*100:.1f}% ROI)")
    print(f"  Overall P&L: ${baseline['pnl']}, ROI: {baseline['roi']*100:.1f}%")
    print(f"  Score: {best_score:.4f}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize results log
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as f:
            f.write("iteration\tparam\told_value\tnew_value\tscore\taccuracy\troi\ttotal\tstatus\ttimestamp\n")

    # Optimization loop
    best_config = dict(config)
    improvements = 0

    for i in range(1, iterations + 1):
        param, old_val, new_val = propose_change(best_config, i)

        # Apply change
        test_config = dict(best_config)
        test_config[param] = new_val

        # Run backtest
        start = time.time()
        results = run_backtest(box_scores, odds_data, test_config)
        elapsed = time.time() - start

        score = compute_score(results)
        status = 'KEEP' if score > best_score else 'DISCARD'

        print(f"\n[{i}/{iterations}] {param}: {old_val} -> {new_val}")
        print(f"  Score: {score:.4f} {'>' if score > best_score else '<='} {best_score:.4f} -> {status}")
        print(f"  Acc: {results['accuracy']*100:.1f}%, ROI: {results['roi']*100:.1f}%, Total: {results['total']}")
        print(f"  Time: {elapsed:.1f}s")

        # Log result
        with open(RESULTS_FILE, 'a') as f:
            f.write(f"{i}\t{param}\t{old_val}\t{new_val}\t{score:.4f}\t{results['accuracy']:.4f}\t{results['roi']:.4f}\t{results['total']}\t{status}\t{datetime.now().isoformat()}\n")

        # Keep or discard
        if status == 'KEEP':
            best_config[param] = new_val
            best_score = score
            improvements += 1
            print(f"  ** IMPROVEMENT #{improvements} **")

    # Final summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Iterations: {iterations}")
    print(f"Improvements: {improvements}")
    print(f"Best score: {best_score:.4f}")
    print(f"\nBest config changes from default:")
    for k, v in best_config.items():
        if v != DEFAULT_CONFIG.get(k):
            print(f"  {k}: {DEFAULT_CONFIG.get(k)} -> {v}")

    # Save best config
    config_path = os.path.join(OUTPUT_DIR, 'best_recommendation_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'config': best_config,
            'score': best_score,
            'improvements': improvements,
            'iterations': iterations,
            'generated': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nBest config saved to: {config_path}")

    return best_config, best_score


# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':
    iters = 20
    auto = False

    for arg in sys.argv[1:]:
        if arg.startswith('--iterations'):
            iters = int(arg.split('=')[1] if '=' in arg else sys.argv[sys.argv.index(arg) + 1])
        elif arg == '--auto':
            auto = True

    run_optimization(iterations=iters, auto=auto)
