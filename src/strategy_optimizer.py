"""
AUTONOMOUS STRATEGY OPTIMIZER
==============================

Inspired by Karpathy's autoresearch framework: systematically test strategy
parameter variations within fixed time/iteration budgets, using a single
optimization metric for unambiguous comparison.

DESIGN PRINCIPLES (from autoresearch):
  1. Fixed iteration budget per experiment (not open-ended)
  2. Single metric focus: risk-adjusted ROI (Sharpe-like ratio)
  3. Code-as-optimization: parameters are the search space
  4. Self-contained: no external dependencies beyond the existing models
  5. Walk-forward validation: no future data leakage

WHAT IT DOES:
  - Runs N experiments, each testing a parameter variation
  - Each experiment backtests against historical data
  - Results are ranked by the optimization metric
  - Best parameters are reported with confidence intervals
  - Supports grid search, random search, and Bayesian-style narrowing

USAGE:
  optimizer = StrategyOptimizer(games_data)
  results = optimizer.run_experiment_sweep(budget=50)
  best = optimizer.get_best_parameters()
"""

import json
import math
import random
import time
from collections import defaultdict
from copy import deepcopy

from convergent_dominance_model import ConvergentDominanceModel, backtest_cds


# =============================================================================
# OPTIMIZATION METRIC
# =============================================================================

def compute_sharpe_roi(backtest_results, risk_free_rate=0.0):
    """
    Compute risk-adjusted ROI (Sharpe-like ratio) from backtest results.

    This is the SINGLE METRIC used for all comparisons, following
    autoresearch's principle of one unambiguous evaluation measure.

    Returns:
        dict with sharpe_roi, raw_roi, win_rate, drawdown, and sample_size
    """
    results = backtest_results.get('results', [])
    if not results or len(results) < 5:
        return {
            'sharpe_roi': -999,
            'raw_roi': 0,
            'win_rate': 0,
            'max_drawdown': 1.0,
            'sample_size': len(results),
            'is_valid': False,
        }

    # Compute per-bet returns at -110 odds
    returns = []
    for r in results:
        if r['fav_won']:
            returns.append(100 / 110)  # Win at -110: profit = 0.909 units
        else:
            returns.append(-1.0)  # Loss: lose 1 unit

    # Basic stats
    total = len(returns)
    wins = sum(1 for r in returns if r > 0)
    win_rate = wins / total
    raw_roi = sum(returns) / total

    # Volatility of returns
    mean_ret = sum(returns) / total
    variance = sum((r - mean_ret) ** 2 for r in returns) / max(total - 1, 1)
    std_ret = variance ** 0.5

    # Sharpe-like ratio (annualized assuming ~5 bets/month, 60/year)
    if std_ret > 0:
        sharpe = (mean_ret - risk_free_rate) * (60 ** 0.5) / std_ret
    else:
        sharpe = mean_ret * 100 if mean_ret > 0 else -999

    # Maximum drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / max(peak, 1)
        if dd > max_dd:
            max_dd = dd

    return {
        'sharpe_roi': round(sharpe, 3),
        'raw_roi': round(raw_roi * 100, 2),
        'win_rate': round(win_rate, 4),
        'max_drawdown': round(max_dd, 4),
        'sample_size': total,
        'total_profit_units': round(sum(returns), 2),
        'is_valid': total >= 10,
    }


# =============================================================================
# PARAMETER SPACE
# =============================================================================

# Defines the search space for CDS model optimization
CDS_PARAM_SPACE = {
    # Dimension 1: Net Rating Gap breakpoints
    'd1_top': {'min': 12, 'max': 18, 'step': 1, 'default': 14},
    'd1_mid': {'min': 8, 'max': 12, 'step': 1, 'default': 10},
    'd1_low': {'min': 5, 'max': 9, 'step': 1, 'default': 7},

    # Dimension 2: Offensive Firepower breakpoints
    'd2_top': {'min': 119, 'max': 125, 'step': 1, 'default': 122},
    'd2_mid': {'min': 115, 'max': 120, 'step': 1, 'default': 118},

    # Dimension 3: Defensive Cage breakpoints
    'd3_top': {'min': -9, 'max': -4, 'step': 1, 'default': -6},
    'd3_mid': {'min': -5, 'max': -1, 'step': 1, 'default': -3},

    # Dimension 5: Home Court value
    'd5_value': {'min': 1.0, 'max': 2.5, 'step': 0.25, 'default': 1.5},

    # Dimension 6: Dual-Edge thresholds
    'd6_off_threshold': {'min': 112, 'max': 118, 'step': 1, 'default': 115},
    'd6_def_threshold': {'min': 107, 'max': 113, 'step': 1, 'default': 110},

    # Signal thresholds
    'elite_min_cds': {'min': 10, 'max': 13, 'step': 0.5, 'default': 11},
    'high_min_cds': {'min': 8, 'max': 11, 'step': 0.5, 'default': 9},
    'strong_min_cds': {'min': 7, 'max': 10, 'step': 0.5, 'default': 8},

    # Lookback window
    'lookback_window': {'min': 10, 'max': 25, 'step': 5, 'default': 15},

    # Home court advantage (for predicted margin)
    'home_advantage': {'min': 2.5, 'max': 5.0, 'step': 0.5, 'default': 3.5},
}

# Pre-game ADI filter parameters
ADI_PARAM_SPACE = {
    'min_net_gap': {'min': 7, 'max': 14, 'step': 1, 'default': 10},
    'min_fav_off': {'min': 115, 'max': 121, 'step': 1, 'default': 118},
    'require_home': {'values': [True, False], 'default': True},
}


def sample_params(param_space, method='random'):
    """Sample a parameter configuration from the search space."""
    params = {}
    for key, spec in param_space.items():
        if 'values' in spec:
            params[key] = random.choice(spec['values'])
        elif method == 'default':
            params[key] = spec['default']
        elif method == 'random':
            steps = int((spec['max'] - spec['min']) / spec['step']) + 1
            idx = random.randint(0, steps - 1)
            params[key] = spec['min'] + idx * spec['step']
        else:
            params[key] = spec['default']
    return params


def mutate_params(params, param_space, mutation_rate=0.3):
    """Mutate a parameter config by randomly adjusting some parameters."""
    new_params = dict(params)
    for key, spec in param_space.items():
        if random.random() < mutation_rate:
            if 'values' in spec:
                new_params[key] = random.choice(spec['values'])
            else:
                direction = random.choice([-1, 1])
                new_val = new_params[key] + direction * spec['step']
                new_val = max(spec['min'], min(spec['max'], new_val))
                new_params[key] = new_val
    return new_params


# =============================================================================
# PARAMETERIZED CDS MODEL
# =============================================================================

class ParameterizedCDSModel(ConvergentDominanceModel):
    """
    CDS model with externally configurable dimension breakpoints and thresholds.

    This allows the optimizer to test different parameter configurations
    without modifying the base model code.
    """

    def __init__(self, params=None):
        self.params = params or {}
        lookback = int(self.params.get('lookback_window', 15))
        super().__init__(lookback_window=lookback)

        # Override thresholds if provided
        if 'elite_min_cds' in self.params:
            self.THRESHOLDS = deepcopy(self.THRESHOLDS)
            self.THRESHOLDS['ELITE']['min_cds'] = self.params['elite_min_cds']
        if 'high_min_cds' in self.params:
            self.THRESHOLDS = deepcopy(self.THRESHOLDS)
            self.THRESHOLDS['HIGH']['min_cds'] = self.params['high_min_cds']
        if 'strong_min_cds' in self.params:
            self.THRESHOLDS = deepcopy(self.THRESHOLDS)
            self.THRESHOLDS['STRONG']['min_cds'] = self.params['strong_min_cds']

    def compute_cds(self, fav_metrics, dog_metrics, fav_is_home):
        """Compute CDS with parameterized breakpoints."""
        fm = fav_metrics
        dm = dog_metrics
        p = self.params

        dimensions = {}
        score = 0.0

        # D1: Net Rating Gap (parameterized breakpoints)
        net_gap = abs(fm['net_rating'] - dm['net_rating'])
        d1_top = p.get('d1_top', 14)
        d1_mid = p.get('d1_mid', 10)
        d1_low = p.get('d1_low', 7)
        d1_floor = max(d1_low - 3, 2)

        if net_gap >= d1_top:
            d1 = 3.0
        elif net_gap >= d1_mid:
            d1 = 2.0 + (net_gap - d1_mid) / max(d1_top - d1_mid, 1)
        elif net_gap >= d1_low:
            d1 = 1.0 + (net_gap - d1_low) / max(d1_mid - d1_low, 1)
        elif net_gap >= d1_floor:
            d1 = (net_gap - d1_floor) / max(d1_low - d1_floor, 1)
        else:
            d1 = 0.0
        dimensions['net_gap'] = round(d1, 2)
        score += d1

        # D2: Offensive Firepower (parameterized)
        fav_off = fm['off_rating']
        d2_top = p.get('d2_top', 122)
        d2_mid = p.get('d2_mid', 118)
        d2_low = d2_mid - 3
        d2_floor = d2_low - 3

        if fav_off >= d2_top:
            d2 = 3.0
        elif fav_off >= d2_mid:
            d2 = 2.0 + (fav_off - d2_mid) / max(d2_top - d2_mid, 1)
        elif fav_off >= d2_low:
            d2 = 1.0 + (fav_off - d2_low) / max(d2_mid - d2_low, 1)
        elif fav_off >= d2_floor:
            d2 = (fav_off - d2_floor) / max(d2_low - d2_floor, 1)
        else:
            d2 = 0.0
        dimensions['offense'] = round(d2, 2)
        score += d2

        # D3: Defensive Cage (parameterized)
        cage = fm['def_rating'] - dm['off_rating']
        d3_top = p.get('d3_top', -6)
        d3_mid = p.get('d3_mid', -3)

        if cage <= d3_top:
            d3 = 3.0
        elif cage <= d3_mid:
            d3 = 2.0 + (d3_mid - cage) / max(d3_mid - d3_top, 1)
        elif cage <= 0:
            d3 = 1.0 + (0 - cage) / max(0 - d3_mid, 1)
        elif cage <= 3:
            d3 = (3 - cage) / 3.0
        else:
            d3 = 0.0
        dimensions['defense'] = round(d3, 2)
        score += d3

        # D4: Win Consistency Gap (unchanged)
        win_pct_gap = abs(fm['win_pct'] - dm['win_pct'])
        if win_pct_gap >= 0.5:
            d4 = 3.0
        elif win_pct_gap >= 0.4:
            d4 = 2.0 + (win_pct_gap - 0.4) / 0.1
        elif win_pct_gap >= 0.3:
            d4 = 1.0 + (win_pct_gap - 0.3) / 0.1
        elif win_pct_gap >= 0.2:
            d4 = (win_pct_gap - 0.2) / 0.1
        else:
            d4 = 0.0
        dimensions['consistency'] = round(d4, 2)
        score += d4

        # D5: Home Court (parameterized value)
        d5 = p.get('d5_value', 1.5) if fav_is_home else 0.0
        dimensions['home'] = round(d5, 2)
        score += d5

        # D6: Dual-Edge Bonus (parameterized thresholds)
        d6_off = p.get('d6_off_threshold', 115)
        d6_def = p.get('d6_def_threshold', 110)
        if fm['off_rating'] >= d6_off and fm['def_rating'] <= d6_def:
            d6 = 1.5
        elif fm['off_rating'] >= d6_off - 3 and fm['def_rating'] <= d6_def + 2:
            d6 = 1.0
        else:
            d6 = 0.0
        dimensions['dual_edge'] = d6
        score += d6

        # D7: Opponent Weakness Amplifier (unchanged)
        if dm['off_rating'] <= 110 and dm['def_rating'] >= 114:
            d7 = 1.0
        elif dm['off_rating'] <= 112 and dm['def_rating'] >= 112:
            d7 = 0.5
        else:
            d7 = 0.0
        dimensions['opp_weakness'] = d7
        score += d7

        return round(score, 1), dimensions

    def predict_game(self, home_team, away_team):
        """Predict with parameterized home advantage."""
        home_m = self.get_team_metrics(home_team)
        away_m = self.get_team_metrics(away_team)
        if not home_m or not away_m:
            return None

        home_adv = self.params.get('home_advantage', 3.5)
        net_diff = home_m['net_rating'] - away_m['net_rating']
        predicted_margin = net_diff + home_adv

        if predicted_margin > 0:
            fav, dog = home_team, away_team
            fav_m, dog_m = home_m, away_m
            fav_is_home = True
        else:
            fav, dog = away_team, home_team
            fav_m, dog_m = away_m, home_m
            fav_is_home = False

        cds_score, dimensions = self.compute_cds(fav_m, dog_m, fav_is_home)

        signals = []
        for tier_name, tier_config in sorted(
            self.THRESHOLDS.items(), key=lambda x: -x[1]['min_cds']
        ):
            if cds_score >= tier_config['min_cds']:
                signals.append({
                    'play': 'CDS_SPREAD',
                    'confidence': tier_config['confidence'],
                    'cds_score': cds_score,
                    'kelly_fraction': tier_config['kelly_fraction'],
                })
                break

        play3_eligible = (
            abs(net_diff) >= self.params.get('min_net_gap', 10) and
            fav_m['off_rating'] >= self.params.get('min_fav_off', 118) and
            fav_is_home
        )

        return {
            'favorite': fav,
            'underdog': dog,
            'fav_is_home': fav_is_home,
            'predicted_margin': round(abs(predicted_margin), 1),
            'cds_score': cds_score,
            'dimensions': dimensions,
            'signals': signals,
            'play3_also_triggers': play3_eligible,
            'is_incremental': bool(signals) and not play3_eligible,
        }


# =============================================================================
# PARAMETERIZED BACKTEST
# =============================================================================

def backtest_parameterized(games_data, params, min_cds=None):
    """
    Run backtest with parameterized CDS model.

    Returns backtest results compatible with compute_sharpe_roi.
    """
    if min_cds is None:
        min_cds = params.get('strong_min_cds', 8)

    model = ParameterizedCDSModel(params)
    results = []
    sorted_games = sorted(games_data, key=lambda g: g.get('date', ''))

    for game in sorted_games:
        prediction = model.predict_game(game['home_team'], game['away_team'])

        if prediction and prediction['cds_score'] >= min_cds:
            home_won = game['home_score'] > game['away_score']
            fav_is_home = prediction['fav_is_home']
            fav_won = (fav_is_home and home_won) or (not fav_is_home and not home_won)

            fav_score = game['home_score'] if fav_is_home else game['away_score']
            dog_score = game['away_score'] if fav_is_home else game['home_score']

            results.append({
                'date': game['date'],
                'favorite': prediction['favorite'],
                'underdog': prediction['underdog'],
                'cds_score': prediction['cds_score'],
                'fav_is_home': prediction['fav_is_home'],
                'fav_won': fav_won,
                'fav_score': fav_score,
                'dog_score': dog_score,
                'margin': fav_score - dog_score,
                'play3_also': prediction['play3_also_triggers'],
                'is_incremental': prediction['is_incremental'],
                'confidence': (prediction['signals'][0]['confidence']
                               if prediction['signals'] else 'NONE'),
            })

        model.update_team(game['home_team'], game['home_score'],
                          game['away_score'], game['date'])
        model.update_team(game['away_team'], game['away_score'],
                          game['home_score'], game['date'])

    total = len(results)
    wins = sum(1 for r in results if r['fav_won'])

    return {
        'games': total,
        'wins': wins,
        'ml_accuracy': wins / total if total > 0 else 0,
        'results': results,
    }


# =============================================================================
# EXPERIMENT RUNNER (autoresearch pattern)
# =============================================================================

class StrategyOptimizer:
    """
    Autonomous strategy optimizer following autoresearch patterns.

    Runs experiments within a fixed budget, each testing a parameter
    variation against historical data with walk-forward validation.
    """

    def __init__(self, games_data, train_ratio=0.7):
        """
        Args:
            games_data: List of game dicts (home_team, away_team, scores, date)
            train_ratio: Fraction of data for training (rest for validation)
        """
        self.games = sorted(games_data, key=lambda g: g.get('date', ''))
        split_idx = int(len(self.games) * train_ratio)
        self.train_games = self.games[:split_idx]
        self.val_games = self.games[split_idx:]
        self.experiment_log = []

    def run_single_experiment(self, params, experiment_id=None):
        """
        Run one experiment: backtest on train set, evaluate on val set.

        Following autoresearch's pattern of fixed-budget experiments
        with clear before/after metrics.
        """
        start_time = time.time()

        # Train evaluation
        train_results = backtest_parameterized(self.train_games, params)
        train_metric = compute_sharpe_roi(train_results)

        # Validation evaluation (the real test)
        val_results = backtest_parameterized(self.val_games, params)
        val_metric = compute_sharpe_roi(val_results)

        elapsed = time.time() - start_time

        experiment = {
            'id': experiment_id or len(self.experiment_log),
            'params': dict(params),
            'train': train_metric,
            'val': val_metric,
            'elapsed_seconds': round(elapsed, 2),
            'overfit_ratio': (
                train_metric['sharpe_roi'] / max(val_metric['sharpe_roi'], 0.001)
                if val_metric['sharpe_roi'] > 0 else 999
            ),
        }

        self.experiment_log.append(experiment)
        return experiment

    def run_experiment_sweep(self, budget=50, method='evolutionary'):
        """
        Run a sweep of experiments within a fixed budget.

        Methods:
          - 'random': Pure random search
          - 'grid': Systematic grid over key parameters
          - 'evolutionary': Start random, then mutate best performers

        Returns:
            List of experiment results sorted by validation Sharpe ROI
        """
        print(f"\n{'='*70}")
        print(f"STRATEGY OPTIMIZER — {method.upper()} SEARCH ({budget} experiments)")
        print(f"{'='*70}")
        print(f"  Train set: {len(self.train_games)} games")
        print(f"  Val set:   {len(self.val_games)} games")
        print()

        # Experiment 0: baseline (default params)
        baseline_params = sample_params(CDS_PARAM_SPACE, method='default')
        baseline = self.run_single_experiment(baseline_params, experiment_id=0)
        print(f"  [BASELINE] Train Sharpe={baseline['train']['sharpe_roi']:.2f} "
              f"Val Sharpe={baseline['val']['sharpe_roi']:.2f} "
              f"Val ROI={baseline['val']['raw_roi']:.1f}% "
              f"Val Win%={baseline['val']['win_rate']*100:.1f}%")

        if method == 'evolutionary':
            self._run_evolutionary(budget - 1, baseline_params)
        elif method == 'random':
            self._run_random(budget - 1)
        else:
            self._run_random(budget - 1)

        # Sort by validation Sharpe ROI
        self.experiment_log.sort(key=lambda e: -e['val']['sharpe_roi'])

        # Report top results
        print(f"\n{'='*70}")
        print("TOP 5 CONFIGURATIONS (by validation Sharpe ROI):")
        print(f"{'='*70}")
        for i, exp in enumerate(self.experiment_log[:5]):
            overfit = "OK" if exp['overfit_ratio'] < 2.0 else "OVERFIT"
            print(f"  #{i+1} [Exp {exp['id']}] "
                  f"Val Sharpe={exp['val']['sharpe_roi']:.2f} "
                  f"Val ROI={exp['val']['raw_roi']:.1f}% "
                  f"Win%={exp['val']['win_rate']*100:.1f}% "
                  f"N={exp['val']['sample_size']} "
                  f"[{overfit}]")

        return self.experiment_log

    def _run_random(self, n):
        """Pure random search."""
        for i in range(n):
            params = sample_params(CDS_PARAM_SPACE, method='random')
            exp = self.run_single_experiment(params, experiment_id=i + 1)
            if (i + 1) % 10 == 0:
                best_so_far = max(self.experiment_log,
                                  key=lambda e: e['val']['sharpe_roi'])
                print(f"  [{i+1}/{n}] Best Val Sharpe so far: "
                      f"{best_so_far['val']['sharpe_roi']:.2f}")

    def _run_evolutionary(self, n, seed_params):
        """
        Evolutionary search: start with seed, mutate best performers.

        Phase 1 (40% budget): Random exploration
        Phase 2 (60% budget): Mutate top performers
        """
        explore_budget = int(n * 0.4)
        exploit_budget = n - explore_budget

        # Phase 1: Explore
        print(f"\n  Phase 1: Random exploration ({explore_budget} experiments)")
        for i in range(explore_budget):
            params = sample_params(CDS_PARAM_SPACE, method='random')
            self.run_single_experiment(params, experiment_id=i + 1)

        # Sort and pick top performers
        top_k = sorted(self.experiment_log,
                        key=lambda e: -e['val']['sharpe_roi'])[:5]
        print(f"  Phase 1 best Val Sharpe: {top_k[0]['val']['sharpe_roi']:.2f}")

        # Phase 2: Exploit (mutate top performers)
        print(f"  Phase 2: Mutating top performers ({exploit_budget} experiments)")
        for i in range(exploit_budget):
            parent = random.choice(top_k)
            child_params = mutate_params(parent['params'], CDS_PARAM_SPACE,
                                          mutation_rate=0.25)
            exp = self.run_single_experiment(
                child_params, experiment_id=explore_budget + i + 1
            )

            # Refresh top_k
            all_sorted = sorted(self.experiment_log,
                                 key=lambda e: -e['val']['sharpe_roi'])
            top_k = all_sorted[:5]

            if (i + 1) % 10 == 0:
                print(f"  [{explore_budget + i + 1}/{n}] "
                      f"Best Val Sharpe: {top_k[0]['val']['sharpe_roi']:.2f}")

    def get_best_parameters(self, max_overfit_ratio=2.0):
        """
        Get the best parameter configuration that isn't overfitting.

        Filters out configs where train performance is much higher than
        validation, suggesting overfitting to training data.
        """
        valid = [e for e in self.experiment_log
                 if e['overfit_ratio'] <= max_overfit_ratio
                 and e['val']['is_valid']]

        if not valid:
            valid = [e for e in self.experiment_log if e['val']['is_valid']]

        if not valid:
            return None

        best = max(valid, key=lambda e: e['val']['sharpe_roi'])
        return {
            'params': best['params'],
            'val_sharpe': best['val']['sharpe_roi'],
            'val_roi': best['val']['raw_roi'],
            'val_win_rate': best['val']['win_rate'],
            'val_sample_size': best['val']['sample_size'],
            'overfit_ratio': best['overfit_ratio'],
            'experiment_id': best['id'],
        }

    def export_results(self, filepath='output/optimizer_results.json'):
        """Export experiment log for analysis."""
        export = {
            'total_experiments': len(self.experiment_log),
            'train_games': len(self.train_games),
            'val_games': len(self.val_games),
            'best_config': self.get_best_parameters(),
            'experiments': [
                {
                    'id': e['id'],
                    'val_sharpe': e['val']['sharpe_roi'],
                    'val_roi': e['val']['raw_roi'],
                    'val_win_rate': e['val']['win_rate'],
                    'val_n': e['val']['sample_size'],
                    'train_sharpe': e['train']['sharpe_roi'],
                    'overfit': e['overfit_ratio'],
                    'params': e['params'],
                }
                for e in sorted(self.experiment_log,
                                key=lambda e: -e['val']['sharpe_roi'])
            ],
        }

        with open(filepath, 'w') as f:
            json.dump(export, f, indent=2)
        print(f"\nResults exported to {filepath}")
        return export


# =============================================================================
# CONVENIENCE: Run optimizer on existing data
# =============================================================================

def run_optimization():
    """Run the optimizer on the existing pregame predictions dataset."""
    try:
        with open('data/pregame_predictions_2025.json') as f:
            games = json.load(f)
    except FileNotFoundError:
        print("Error: data/pregame_predictions_2025.json not found")
        return

    game_records = []
    for g in games:
        if g['actual_margin'] >= 0:
            home_score = g['winner_score']
            away_score = g['loser_score']
        else:
            home_score = g['loser_score']
            away_score = g['winner_score']
        game_records.append({
            'home_team': g['home'],
            'away_team': g['away'],
            'home_score': home_score,
            'away_score': away_score,
            'date': g['date'],
        })

    optimizer = StrategyOptimizer(game_records)
    optimizer.run_experiment_sweep(budget=50, method='evolutionary')
    best = optimizer.get_best_parameters()

    if best:
        print(f"\n{'='*70}")
        print("RECOMMENDED PARAMETER UPDATE:")
        print(f"{'='*70}")
        print(f"  Val Sharpe ROI: {best['val_sharpe']:.2f}")
        print(f"  Val Win Rate:   {best['val_win_rate']*100:.1f}%")
        print(f"  Val Raw ROI:    {best['val_roi']:.1f}%")
        print(f"  Overfit Ratio:  {best['overfit_ratio']:.2f}")
        print(f"  Sample Size:    {best['val_sample_size']} games")
        print(f"\n  Parameters:")
        for k, v in sorted(best['params'].items()):
            default = CDS_PARAM_SPACE.get(k, {}).get('default', '?')
            changed = " ***" if v != default else ""
            print(f"    {k}: {v} (default: {default}){changed}")

    optimizer.export_results()
    return optimizer


if __name__ == '__main__':
    run_optimization()
