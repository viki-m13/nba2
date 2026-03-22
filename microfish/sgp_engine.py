"""
MiroFish SGP Strategy Engine
================================
Deterministic SGP (Same Game Parlay) construction engine.
NO AI at runtime — reads rules from sgp_strategy_rules.json
and matches available props against SGP templates.

@sgp_vick-inspired architecture:
  1. Load SGP templates (produced by MiroFish multi-agent discovery)
  2. For each game, find available props that match template criteria
  3. Score each potential parlay by correlation lift and edge
  4. Output constructed SGP slips with combined odds

Patent-pending: AI discovers correlation structures between props
that books underprice. Runtime is pure deterministic rule matching.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Optional

from sgp_config import (
    SGP_RULES_FILE, PARLAY_ARCHETYPES, STAKE_SIZE,
    MAX_PARLAYS_PER_DAY, SGP_MIN_LEG_CONFIDENCE,
)
from sgp_data import (
    american_to_decimal, american_to_implied_prob,
    decimal_to_american, parlay_odds, parlay_american,
)


class SGPEngine:
    """
    Deterministic SGP construction engine.

    Reads SGP templates from sgp_strategy_rules.json and constructs
    parlays from available player props.

    Templates define:
      - Which prop markets to combine
      - Player selection criteria per leg
      - Game condition filters
      - Min implied probability per leg
      - Target combined odds range
    """

    def __init__(self, rules: dict = None, rules_file: str = None):
        if rules is not None:
            self.rules = rules
        elif rules_file is not None:
            with open(rules_file, 'r') as f:
                self.rules = json.load(f)
        elif os.path.exists(SGP_RULES_FILE):
            with open(SGP_RULES_FILE, 'r') as f:
                self.rules = json.load(f)
        else:
            raise FileNotFoundError(
                f"No SGP rules found at {SGP_RULES_FILE}. "
                f"Run: python run.py sgp-develop"
            )

        self.templates = self.rules.get('sgp_templates', [])
        self.global_filters = self.rules.get('global_filters', {})
        self.anti_patterns = self.rules.get('anti_patterns', [])

    def construct_sgps(self, props_df: pd.DataFrame,
                       games_df: pd.DataFrame = None) -> list:
        """
        Construct SGP slips from available props using templates.

        Args:
            props_df: DataFrame of available player props with columns:
                game_id, game_date, home_team, away_team, player,
                market, bet_type, line, odds, implied_prob
            games_df: Optional game-level data (moneyline, totals)

        Returns:
            List of constructed SGP dicts, each with:
                template, legs, combined_odds, stake, potential_payout
        """
        if props_df.empty:
            return []

        constructed = []

        # Group props by game for SGP (same game)
        for game_id, game_props in props_df.groupby('game_id'):
            game_info = {
                'game_id': game_id,
                'home_team': game_props.iloc[0].get('home_team', ''),
                'away_team': game_props.iloc[0].get('away_team', ''),
                'game_date': game_props.iloc[0].get('game_date', ''),
            }

            # Add game-level context if available
            if games_df is not None and not games_df.empty:
                game_context = games_df[games_df['game_id'] == game_id]
                if not game_context.empty:
                    game_info['game_total'] = game_context.iloc[0].get('line', None)

            # Try each template against this game's props
            for template in self.templates:
                sgp = self._match_template(template, game_props, game_info)
                if sgp:
                    constructed.append(sgp)

        # Sort by expected value (combined odds * hit rate estimate)
        constructed.sort(
            key=lambda s: s.get('score', 0), reverse=True
        )

        # Apply daily cap
        max_per_day = self.global_filters.get(
            'max_sgps_per_day', MAX_PARLAYS_PER_DAY
        )
        if len(constructed) > max_per_day:
            constructed = constructed[:max_per_day]

        return constructed

    def _match_template(self, template: dict, game_props: pd.DataFrame,
                        game_info: dict) -> Optional[dict]:
        """
        Try to match a template against available props for one game.
        Returns constructed SGP dict or None if template doesn't match.
        """
        template_legs = template.get('legs', [])
        if not template_legs:
            return None

        # Check game conditions
        game_conds = template.get('game_conditions', {})
        if not self._check_game_conditions(game_conds, game_info, game_props):
            return None

        # Match each template leg to best available prop
        matched_legs = []
        used_players = set()

        for leg_spec in template_legs:
            match = self._find_best_leg_match(
                leg_spec, game_props, used_players
            )
            if match is not None:
                matched_legs.append(match)
                used_players.add(match['player'])

        # Check minimum legs met
        min_legs = template.get('min_legs', len(template_legs))
        if len(matched_legs) < min_legs:
            return None

        # Check anti-patterns
        if self._has_anti_pattern(matched_legs):
            return None

        # Calculate combined parlay odds
        leg_odds_american = [leg['odds'] for leg in matched_legs]
        combined_american = parlay_american(leg_odds_american)
        leg_odds_decimal = [american_to_decimal(o) for o in leg_odds_american]
        combined_decimal = parlay_odds(leg_odds_decimal)

        # Check if combined odds fall in template's target range
        target_min = template.get('target_odds_min', 0)
        target_max = template.get('target_odds_max', 999999)
        if combined_american < target_min or combined_american > target_max:
            return None

        # Calculate hit probability estimate (product of individual probs)
        individual_probs = [leg['implied_prob'] for leg in matched_legs]
        naive_hit_prob = 1.0
        for p in individual_probs:
            naive_hit_prob *= p

        # Apply correlation lift if template specifies one
        correlation_lift = template.get('expected_correlation_lift', 1.0)
        adjusted_hit_prob = naive_hit_prob * correlation_lift

        # Expected value per dollar
        ev_per_dollar = (adjusted_hit_prob * combined_decimal) - 1.0

        # Score for ranking
        score = ev_per_dollar * 100

        potential_payout = STAKE_SIZE * combined_decimal

        return {
            'template': template.get('name', 'unknown'),
            'template_description': template.get('description', ''),
            'game_id': game_info['game_id'],
            'game_date': str(game_info.get('game_date', '')),
            'home_team': game_info.get('home_team', ''),
            'away_team': game_info.get('away_team', ''),
            'n_legs': len(matched_legs),
            'legs': matched_legs,
            'combined_odds_american': int(combined_american),
            'combined_odds_decimal': round(combined_decimal, 2),
            'naive_hit_prob': round(naive_hit_prob, 6),
            'adjusted_hit_prob': round(adjusted_hit_prob, 6),
            'ev_per_dollar': round(ev_per_dollar, 4),
            'stake': STAKE_SIZE,
            'potential_payout': round(potential_payout, 2),
            'score': round(score, 2),
            'correlation_edge': template.get('correlation_edge_source', ''),
        }

    def _find_best_leg_match(self, leg_spec: dict,
                              game_props: pd.DataFrame,
                              used_players: set) -> Optional[dict]:
        """
        Find the best available prop that matches a leg specification.

        Leg spec format:
        {
            "market": "pitcher_strikeouts",
            "direction": "over",
            "line_criteria": {"operator": ">=", "value": 5.5},
            "min_implied_prob": 0.65,
            "player_criteria": "starting pitcher with 7+ K/9",
            "correlation_role": "ace dominance driver"
        }
        """
        market = leg_spec.get('market', '')
        direction = leg_spec.get('direction', 'over').capitalize()
        min_prob = leg_spec.get('min_implied_prob',
                                self.global_filters.get('min_leg_implied_prob',
                                                         SGP_MIN_LEG_CONFIDENCE))

        # Filter to matching market and direction
        candidates = game_props[
            (game_props['market'] == market) &
            (game_props['bet_type'].str.lower() == direction.lower())
        ].copy()

        if candidates.empty:
            return None

        # Exclude already-used players
        if used_players:
            candidates = candidates[~candidates['player'].isin(used_players)]

        if candidates.empty:
            return None

        # Apply line criteria
        line_criteria = leg_spec.get('line_criteria', {})
        if line_criteria and 'line' in candidates.columns:
            op = line_criteria.get('operator', '>=')
            val = line_criteria.get('value', 0)
            candidates = candidates[candidates['line'].notna()]
            if op == '>=':
                candidates = candidates[candidates['line'] >= val]
            elif op == '<=':
                candidates = candidates[candidates['line'] <= val]
            elif op == '==':
                candidates = candidates[candidates['line'] == val]

        if candidates.empty:
            return None

        # Filter by minimum implied probability
        candidates = candidates[candidates['implied_prob'] >= min_prob]

        if candidates.empty:
            return None

        # Pick the best candidate (highest implied prob = most likely to hit)
        best = candidates.sort_values('implied_prob', ascending=False).iloc[0]

        return {
            'player': best.get('player', ''),
            'market': market,
            'direction': direction,
            'line': float(best.get('line', 0)) if pd.notna(best.get('line')) else None,
            'odds': float(best.get('odds', 0)),
            'implied_prob': float(best.get('implied_prob', 0)),
            'bookmaker': best.get('bookmaker', ''),
            'correlation_role': leg_spec.get('correlation_role', ''),
        }

    def _check_game_conditions(self, conditions: dict,
                                game_info: dict,
                                game_props: pd.DataFrame) -> bool:
        """Check if a game meets template's game conditions."""
        if not conditions:
            return True

        for cond_name, spec in conditions.items():
            if isinstance(spec, dict) and 'operator' in spec:
                value = game_info.get(cond_name)
                if value is None:
                    # Try to derive from props
                    if cond_name == 'n_props_available':
                        value = len(game_props)
                    elif cond_name == 'n_unique_players':
                        value = game_props['player'].nunique() if 'player' in game_props.columns else 0
                    else:
                        continue

                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

                op = spec.get('operator', '>=')
                threshold = spec.get('value', 0)

                if op == '>=' and value < threshold:
                    return False
                elif op == '<=' and value > threshold:
                    return False
                elif op == '>' and value <= threshold:
                    return False
                elif op == '<' and value >= threshold:
                    return False

        return True

    def _has_anti_pattern(self, legs: list) -> bool:
        """Check if matched legs contain any anti-pattern combinations."""
        if not self.anti_patterns:
            return False

        leg_markets = {leg['market'] for leg in legs}
        leg_descriptions = [
            f"{leg['player']}_{leg['market']}_{leg['direction']}"
            for leg in legs
        ]

        for pattern in self.anti_patterns:
            if isinstance(pattern, str):
                # Simple string match against any leg description
                for desc in leg_descriptions:
                    if pattern.lower() in desc.lower():
                        return True
            elif isinstance(pattern, dict):
                # Dict pattern: {"markets": ["a", "b"], "rule": "never_combine"}
                forbidden = set(pattern.get('markets', []))
                if forbidden.issubset(leg_markets):
                    return True

        return False

    def construct_sgp_plus(self, props_df: pd.DataFrame,
                            max_games: int = 3) -> list:
        """
        Construct SGP+ (cross-game SGPs) by combining legs from multiple games.
        This implements @sgp_vick's Pattern 4.

        Picks the best legs from different games and combines them.
        """
        if props_df.empty:
            return []

        # Get best individual legs across all games
        all_legs = []
        for game_id, game_props in props_df.groupby('game_id'):
            game_info = {
                'game_id': game_id,
                'home_team': game_props.iloc[0].get('home_team', ''),
                'away_team': game_props.iloc[0].get('away_team', ''),
            }

            # Find high-confidence legs from any template
            for template in self.templates:
                for leg_spec in template.get('legs', []):
                    match = self._find_best_leg_match(
                        leg_spec, game_props, set()
                    )
                    if match:
                        match['game_id'] = game_id
                        match['game_info'] = game_info
                        all_legs.append(match)

        if len(all_legs) < 3:
            return []

        # Sort by implied prob (highest confidence first)
        all_legs.sort(key=lambda l: l['implied_prob'], reverse=True)

        # Build SGP+ by taking top legs from different games
        sgp_plus_slips = []
        archetype = PARLAY_ARCHETYPES.get('sgp_plus', {})
        min_legs = archetype.get('min_legs', 6)
        max_legs = archetype.get('max_legs', 12)

        # Try building a slip with legs from multiple games
        selected = []
        games_used = set()
        players_used = set()

        for leg in all_legs:
            if len(games_used) >= max_games and leg['game_id'] not in games_used:
                continue
            if leg['player'] in players_used:
                continue

            selected.append(leg)
            games_used.add(leg['game_id'])
            players_used.add(leg['player'])

            if len(selected) >= max_legs:
                break

        if len(selected) >= min_legs and len(games_used) >= 2:
            leg_odds = [l['odds'] for l in selected]
            combined = parlay_american(leg_odds)
            combined_dec = parlay_odds([american_to_decimal(o) for o in leg_odds])

            naive_prob = 1.0
            for l in selected:
                naive_prob *= l['implied_prob']

            sgp_plus_slips.append({
                'template': 'sgp_plus_cross_game',
                'template_description': f'SGP+ across {len(games_used)} games',
                'n_legs': len(selected),
                'n_games': len(games_used),
                'games': list(games_used),
                'legs': selected,
                'combined_odds_american': int(combined),
                'combined_odds_decimal': round(combined_dec, 2),
                'naive_hit_prob': round(naive_prob, 6),
                'stake': STAKE_SIZE,
                'potential_payout': round(STAKE_SIZE * combined_dec, 2),
                'score': round((naive_prob * combined_dec - 1) * 100, 2),
            })

        return sgp_plus_slips

    def summary(self) -> str:
        """Human-readable summary of SGP strategy."""
        lines = []
        lines.append(f"SGP Strategy: {self.rules.get('version', 'unknown')}")
        lines.append(f"Developed by: {self.rules.get('developed_by', 'unknown')}")
        lines.append(f"Model: {self.rules.get('model_used', 'unknown')}")
        lines.append(f"Target: {self.rules.get('target', '$10 SGPs')}")
        lines.append("")

        lines.append(f"TEMPLATES ({len(self.templates)} total):")
        for t in self.templates:
            name = t.get('name', 'unnamed')
            desc = t.get('description', '')
            n_legs = len(t.get('legs', []))
            odds_range = f"+{t.get('target_odds_min', '?')} to +{t.get('target_odds_max', '?')}"
            hit_rate = t.get('expected_hit_rate', '?')
            lines.append(f"  [{name}] {desc}")
            lines.append(f"    Legs: {n_legs}, Odds: {odds_range}, Hit rate: {hit_rate}")
            for leg in t.get('legs', []):
                market = leg.get('market', '?')
                direction = leg.get('direction', '?')
                min_prob = leg.get('min_implied_prob', '?')
                lines.append(f"      - {market} {direction} (min prob: {min_prob})")
            lines.append(f"    Edge: {t.get('correlation_edge_source', 'not specified')}")

        lines.append("")
        gf = self.global_filters
        lines.append(f"GLOBAL FILTERS:")
        lines.append(f"  Min leg implied prob: {gf.get('min_leg_implied_prob', '?')}")
        lines.append(f"  Max SGPs per day: {gf.get('max_sgps_per_day', '?')}")
        lines.append(f"  Min correlation lift: {gf.get('min_correlation_lift', '?')}")
        lines.append(f"  Stake size: ${gf.get('stake_size', STAKE_SIZE)}")

        if self.anti_patterns:
            lines.append(f"\nANTI-PATTERNS ({len(self.anti_patterns)}):")
            for ap in self.anti_patterns[:5]:
                lines.append(f"  - {ap}")

        if 'reasoning' in self.rules:
            lines.append(f"\nREASONING: {self.rules['reasoning']}")

        return '\n'.join(lines)
