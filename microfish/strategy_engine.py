"""
Microfish Strategy Engine
==========================
Pure deterministic rules-based strategy engine. NO AI, NO ML at runtime.

Reads strategy rules from strategy_rules.json (produced by the MiroFish
multi-agent development pipeline) and evaluates MLB +200 underdog
opportunities against those rules.

This is the runtime component — fast, deterministic, fully backtestable.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Optional

from config import RULES_FILE, MIN_AMERICAN_ODDS


class StrategyEngine:
    """
    Deterministic rules engine that evaluates +200 underdog opportunities.

    Rules format (from strategy_rules.json):
    {
        "rules": [
            {
                "name": "rule_name",
                "conditions": {
                    "feature": {"operator": ">=", "value": 0.55}
                },
                "weight": 3,
                "min_conditions_met": 2
            }
        ],
        "global_filters": {
            "min_total_weight": 6,
            "min_rules_triggered": 2,
            "max_bets_per_day": 5
        }
    }
    """

    def __init__(self, rules: dict = None, rules_file: str = None):
        if rules is not None:
            self.rules = rules
        elif rules_file is not None:
            with open(rules_file, 'r') as f:
                self.rules = json.load(f)
        elif os.path.exists(RULES_FILE):
            with open(RULES_FILE, 'r') as f:
                self.rules = json.load(f)
        else:
            raise FileNotFoundError(
                f"No strategy rules found. Run development pipeline first: "
                f"python run.py develop"
            )

        self._validate_rules()

    def _validate_rules(self):
        """Ensure rules structure is valid."""
        if 'rules' not in self.rules:
            raise ValueError("Rules file must contain 'rules' key")
        if 'global_filters' not in self.rules:
            self.rules['global_filters'] = {
                'min_total_weight': 1,
                'min_rules_triggered': 1,
                'max_bets_per_day': 10,
            }

    def evaluate(self, game: dict) -> dict:
        """
        Evaluate a single +200 underdog opportunity against all rules.

        Weight=0 rules are HARD PRE-FILTERS — if they don't pass, the
        game is immediately rejected (PASS) without checking other rules.
        This was identified as a critical bug fix by the MiroFish agents.

        Returns:
            {
                'recommendation': 'BET' or 'PASS',
                'total_weight': sum of triggered rule weights,
                'rules_triggered': list of triggered rule names,
                'rule_details': detailed results per rule,
            }
        """
        # Apply feature name aliases so agent-generated rules work with
        # data pipeline feature names
        game = self._apply_feature_aliases(game)

        # PHASE 1: Hard pre-filters (weight=0 rules)
        for rule in self.rules.get('rules', []):
            if rule.get('weight', 1) == 0:
                result = self._evaluate_rule(game, rule)
                if not result['triggered']:
                    return {
                        'recommendation': 'PASS',
                        'total_weight': 0,
                        'rules_triggered': [],
                        'n_rules_triggered': 0,
                        'rule_details': [result],
                        'filtered_by': rule.get('name', 'pre-filter'),
                    }

        # PHASE 2: Weighted rules
        triggered_rules = []
        total_weight = 0
        rule_details = []

        for rule in self.rules.get('rules', []):
            if rule.get('weight', 1) == 0:
                continue  # Already handled in phase 1
            result = self._evaluate_rule(game, rule)
            rule_details.append(result)
            if result['triggered']:
                triggered_rules.append(rule['name'])
                total_weight += rule.get('weight', 1)

        filters = self.rules.get('global_filters', {})
        min_weight = filters.get('min_total_weight', 1)
        min_rules = filters.get('min_rules_triggered', 1)

        recommend = (
            total_weight >= min_weight and
            len(triggered_rules) >= min_rules
        )

        return {
            'recommendation': 'BET' if recommend else 'PASS',
            'total_weight': total_weight,
            'rules_triggered': triggered_rules,
            'n_rules_triggered': len(triggered_rules),
            'rule_details': rule_details,
        }

    @staticmethod
    def _apply_feature_aliases(game: dict) -> dict:
        """
        Map agent-generated feature names to data pipeline feature names.
        The MiroFish agents may use different naming conventions than the
        data pipeline, so we bridge them here.
        """
        g = game.copy()
        # Odds aliases
        if 'bet_odds' in g:
            g.setdefault('dog_odds', g['bet_odds'])
            # For upper bound checks, use the same odds value
            g.setdefault('dog_odds_upper', g['bet_odds'])
        return g

    def _evaluate_rule(self, game: dict, rule: dict) -> dict:
        """Evaluate a single rule against a game."""
        conditions = rule.get('conditions', {})
        min_met = rule.get('min_conditions_met', len(conditions))
        conditions_met = 0
        conditions_checked = 0
        condition_results = {}

        for feature, spec in conditions.items():
            value = game.get(feature)
            if value is None:
                condition_results[feature] = {
                    'status': 'missing',
                    'required': spec,
                }
                continue

            conditions_checked += 1
            try:
                value = float(value)
            except (ValueError, TypeError):
                condition_results[feature] = {
                    'status': 'invalid',
                    'value': value,
                    'required': spec,
                }
                continue

            operator = spec.get('operator', '>=')
            threshold = spec.get('value', 0)

            passed = self._check_condition(value, operator, threshold)
            if passed:
                conditions_met += 1

            condition_results[feature] = {
                'status': 'pass' if passed else 'fail',
                'value': value,
                'required': spec,
            }

        triggered = conditions_met >= min_met and conditions_checked > 0

        return {
            'name': rule.get('name', 'unnamed'),
            'triggered': triggered,
            'conditions_met': conditions_met,
            'conditions_checked': conditions_checked,
            'min_required': min_met,
            'weight': rule.get('weight', 1),
            'condition_results': condition_results,
        }

    @staticmethod
    def _check_condition(value: float, operator: str, threshold: float) -> bool:
        """Check a single condition."""
        if operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '==':
            return abs(value - threshold) < 1e-9
        elif operator == '!=':
            return abs(value - threshold) >= 1e-9
        elif operator == 'between':
            if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
                return threshold[0] <= value <= threshold[1]
            return False
        else:
            return False

    def evaluate_batch(self, opportunities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate a batch of opportunities. Applies max_bets_per_day filter.
        Returns DataFrame with recommendation columns added.
        """
        results = []
        for _, row in opportunities_df.iterrows():
            game = row.to_dict()
            result = self.evaluate(game)
            game.update({
                'recommendation': result['recommendation'],
                'total_weight': result['total_weight'],
                'n_rules_triggered': result['n_rules_triggered'],
                'rules_triggered': ','.join(result['rules_triggered']),
            })
            results.append(game)

        df = pd.DataFrame(results)

        if df.empty:
            return df

        # Apply max_bets_per_day filter
        max_per_day = self.rules.get('global_filters', {}).get(
            'max_bets_per_day', 10
        )

        if 'date' in df.columns:
            # Sort by weight (best first), then apply daily cap
            bets = df[df['recommendation'] == 'BET'].copy()
            if not bets.empty:
                bets = bets.sort_values('total_weight', ascending=False)
                kept_indices = []
                for date, group in bets.groupby('date'):
                    kept_indices.extend(
                        group.head(max_per_day).index.tolist()
                    )
                # Mark excess bets as PASS
                excess = bets.index.difference(kept_indices)
                df.loc[excess, 'recommendation'] = 'PASS'

        return df

    def summary(self) -> str:
        """Print a human-readable summary of the strategy rules."""
        lines = []
        lines.append(f"Strategy: {self.rules.get('version', 'unknown')}")
        lines.append(f"Developed by: {self.rules.get('developed_by', 'unknown')}")
        lines.append(f"Model: {self.rules.get('model_used', 'unknown')}")
        lines.append(f"")

        rules = self.rules.get('rules', [])
        lines.append(f"RULES ({len(rules)} total):")
        for rule in rules:
            lines.append(f"  [{rule.get('weight', '?')}] {rule.get('name', 'unnamed')}")
            lines.append(f"      {rule.get('description', 'no description')}")
            for feat, spec in rule.get('conditions', {}).items():
                op = spec.get('operator', '>=')
                val = spec.get('value', '?')
                lines.append(f"      - {feat} {op} {val}")
            lines.append(f"      min conditions: {rule.get('min_conditions_met', 'all')}")

        filters = self.rules.get('global_filters', {})
        lines.append(f"")
        lines.append(f"GLOBAL FILTERS:")
        lines.append(f"  Min total weight: {filters.get('min_total_weight', '?')}")
        lines.append(f"  Min rules triggered: {filters.get('min_rules_triggered', '?')}")
        lines.append(f"  Max bets per day: {filters.get('max_bets_per_day', '?')}")

        if 'reasoning' in self.rules:
            lines.append(f"")
            lines.append(f"REASONING: {self.rules['reasoning']}")

        return '\n'.join(lines)
