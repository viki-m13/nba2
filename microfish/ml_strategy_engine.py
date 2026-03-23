"""
Microfish Moneyline Strategy Engine
======================================
Pure deterministic limit-price engine. NO AI, NO ML at runtime.

Reads strategy rules from ml_strategy_rules.json (produced by the CFD engine
and refined by MiroFish MinMax agents) and evaluates games against
the Certainty Floor threshold.

This is the runtime component — fast, deterministic, fully backtestable.

Execution model on Polymarket:
1. For each game, check if either team's implied probability >= Certainty Floor
2. If yes, output a LIMIT BUY order at the floor price for that team
3. The order fills if/when the market reaches that price
4. When filled, the bet wins (100% historical accuracy at the floor)
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Optional

from ml_config import ML_RULES_FILE
from ml_data_pipeline import american_to_implied_prob


class MLStrategyEngine:
    """
    Deterministic limit-price engine for Polymarket MLB moneyline bets.

    Strategy: Set limit orders at the Certainty Floor probability.
    When the market reaches this price, the order fills.
    At this price, favorites have won 100% of the time historically.
    """

    def __init__(self, rules: dict = None, rules_file: str = None):
        if rules is not None:
            self.rules = rules
        elif rules_file is not None:
            with open(rules_file, 'r') as f:
                self.rules = json.load(f)
        elif os.path.exists(ML_RULES_FILE):
            with open(ML_RULES_FILE, 'r') as f:
                self.rules = json.load(f)
        else:
            raise FileNotFoundError(
                f"No moneyline strategy rules found. Run: python run.py ml-discover"
            )

    @property
    def base_floor(self) -> float:
        """The base certainty floor as a probability (0-1)."""
        base = self.rules.get('base_rule', {})
        return base.get('floor_pct', 99) / 100.0

    @property
    def limit_price(self) -> float:
        """The limit price in dollars (Polymarket share price)."""
        return self.base_floor

    def evaluate(self, game: dict) -> dict:
        """
        Evaluate a single game for limit-order placement.

        Args:
            game: dict with at minimum:
                - fav_team, dog_team
                - fav_prob_novig (no-vig implied probability)
                - fav_ml, dog_ml (American odds)

        Returns:
            {
                'action': 'LIMIT_BUY' or 'SKIP',
                'team': team to bet on,
                'limit_price': price to set limit order at,
                'current_prob': current implied probability,
                'profit_if_win': profit per $1 share,
                'floor_used': which floor was applied,
                'reasoning': explanation,
            }
        """
        fav_prob = game.get('fav_prob_novig', 0)
        fav_team = game.get('fav_team', '')
        fav_ml = game.get('fav_ml', 0)

        # Check exclusion filters (from MinMax agents)
        for exclusion in self.rules.get('agent_exclusion_filters', []):
            if self._check_exclusion(game, exclusion):
                return {
                    'action': 'SKIP',
                    'team': fav_team,
                    'limit_price': 0,
                    'current_prob': fav_prob,
                    'profit_if_win': 0,
                    'floor_used': 'excluded',
                    'reasoning': f"Excluded by filter: {exclusion.get('name', 'unnamed')}",
                }

        # Check contextual rules (lower floors with conditions)
        applicable_floor = self.base_floor
        floor_source = 'base'

        for ctx_rule in self.rules.get('contextual_rules', []):
            if self._check_context(game, ctx_rule):
                ctx_floor = ctx_rule.get('floor_pct', 100) / 100.0
                if ctx_floor < applicable_floor:
                    applicable_floor = ctx_floor
                    floor_source = ctx_rule.get('name', 'contextual')

        # Also check agent contextual rules
        for ctx_rule in self.rules.get('agent_contextual_rules', []):
            if self._check_context(game, ctx_rule):
                ctx_floor = ctx_rule.get('adjusted_floor_pct', 100) / 100.0
                if ctx_floor < applicable_floor:
                    applicable_floor = ctx_floor
                    floor_source = ctx_rule.get('name', 'agent_contextual')

        # Evaluate: does this game meet the certainty floor?
        if fav_prob >= applicable_floor:
            profit = 1.0 - applicable_floor
            return {
                'action': 'LIMIT_BUY',
                'team': fav_team,
                'limit_price': applicable_floor,
                'current_prob': fav_prob,
                'profit_if_win': round(profit, 4),
                'floor_used': floor_source,
                'reasoning': (
                    f"{fav_team} implied prob {fav_prob:.1%} >= floor {applicable_floor:.0%}. "
                    f"Set limit buy at ${applicable_floor:.2f}. "
                    f"Profit if win: ${profit:.2f}/share."
                ),
            }
        else:
            return {
                'action': 'SKIP',
                'team': fav_team,
                'limit_price': applicable_floor,
                'current_prob': fav_prob,
                'profit_if_win': 0,
                'floor_used': floor_source,
                'reasoning': (
                    f"{fav_team} implied prob {fav_prob:.1%} < floor {applicable_floor:.0%}. "
                    f"No bet — market hasn't reached certainty threshold."
                ),
            }

    def _check_exclusion(self, game: dict, exclusion: dict) -> bool:
        """Check if a game matches an exclusion filter."""
        conditions = exclusion.get('conditions', {})
        for feature, spec in conditions.items():
            value = game.get(feature)
            if value is None:
                continue
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue
            operator = spec.get('operator', '>=')
            threshold = spec.get('value', 0)
            if self._check_op(value, operator, threshold):
                return True
        return False

    def _check_context(self, game: dict, ctx_rule: dict) -> bool:
        """Check if a game matches a contextual rule's conditions."""
        conditions = ctx_rule.get('conditions', {})
        if not conditions:
            return False
        for feature, spec in conditions.items():
            value = game.get(feature)
            if value is None:
                return False
            try:
                value = float(value)
            except (ValueError, TypeError):
                return False
            operator = spec.get('operator', '>=')
            threshold = spec.get('value', 0)
            if not self._check_op(value, operator, threshold):
                return False
        return True

    @staticmethod
    def _check_op(value: float, operator: str, threshold) -> bool:
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
        return False

    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate a batch of games. Apply max_bets_per_day."""
        max_per_day = self.rules.get('execution_rules', {}).get('max_bets_per_day', 5)

        results = []
        for _, row in df.iterrows():
            game = row.to_dict()
            result = self.evaluate(game)
            game.update(result)
            results.append(game)

        result_df = pd.DataFrame(results)
        if result_df.empty:
            return result_df

        # Apply daily cap
        if 'date' in result_df.columns:
            bets = result_df[result_df['action'] == 'LIMIT_BUY'].copy()
            if not bets.empty:
                bets = bets.sort_values('current_prob', ascending=False)
                kept = []
                for date, group in bets.groupby('date'):
                    kept.extend(group.head(max_per_day).index.tolist())
                excess = bets.index.difference(kept)
                result_df.loc[excess, 'action'] = 'SKIP'

        return result_df

    def summary(self) -> str:
        """Print a human-readable strategy summary."""
        lines = []
        base = self.rules.get('base_rule', {})
        lines.append(f"Strategy: {self.rules.get('strategy_name', 'Unknown')}")
        lines.append(f"Version: {self.rules.get('version', '?')}")
        lines.append(f"Developed by: {self.rules.get('developed_by', '?')}")
        lines.append(f"")
        lines.append(f"BASE RULE:")
        lines.append(f"  Certainty Floor: {base.get('floor_pct', '?')}%")
        lines.append(f"  Limit Price: ${base.get('limit_price', '?')}")
        lines.append(f"  Profit per win: ${base.get('profit_per_share', '?')}")
        lines.append(f"  Fill rate: {base.get('fill_rate', 0):.1%}")
        lines.append(f"  Games validated: {base.get('n_games_validated', '?')}")

        ctx = self.rules.get('contextual_rules', [])
        if ctx:
            lines.append(f"")
            lines.append(f"CONTEXTUAL RULES ({len(ctx)}):")
            for r in ctx:
                lines.append(f"  {r.get('name', '?')}: floor={r.get('floor_pct', '?')}%")

        wf = self.rules.get('walk_forward_validation', {})
        lines.append(f"")
        lines.append(f"VALIDATION:")
        lines.append(f"  Walk-forward accuracy: {wf.get('accuracy', 0):.1%}")
        lines.append(f"  Walk-forward bets: {wf.get('total_bets', 0)}")
        lines.append(f"  Walk-forward losses: {wf.get('losses', 0)}")

        return '\n'.join(lines)
