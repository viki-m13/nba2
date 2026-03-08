"""
STRATEGY VERIFICATION FRAMEWORK
=================================

Inspired by everything-claude-code's verification loops and structured
evaluation patterns. Implements checkpoint-based validation rather than
trusting single-pass generation.

DESIGN PRINCIPLES (from ECC):
  1. Verification loops: Test every strategy change against multiple criteria
  2. Grading rubrics: Explicit pass/fail thresholds for each quality dimension
  3. Iterative refinement: Progressive confidence adjustment based on recent performance
  4. Research-first: Analyze recent data before generating picks

WHAT IT DOES:
  - Validates strategy changes against a rubric before deployment
  - Detects regime changes (when recent accuracy diverges from historical)
  - Implements progressive confidence scaling based on rolling performance
  - Provides structured go/no-go evaluation for each night's picks
"""

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta


# =============================================================================
# VERIFICATION RUBRIC
# =============================================================================

class StrategyRubric:
    """
    Structured evaluation rubric for strategy quality.

    Each dimension has a pass threshold and a weight.
    Overall grade is weighted average; strategy passes if grade >= threshold.
    """

    DIMENSIONS = {
        'accuracy': {
            'description': 'Recent ML win rate (L20 picks)',
            'pass_threshold': 0.75,
            'weight': 3.0,
        },
        'sample_size': {
            'description': 'Minimum picks in evaluation window',
            'pass_threshold': 10,
            'weight': 1.0,
        },
        'roi': {
            'description': 'ROI at -110 odds over evaluation window',
            'pass_threshold': 0.0,  # Must be profitable
            'weight': 2.0,
        },
        'drawdown': {
            'description': 'Maximum consecutive losses',
            'pass_threshold': 5,  # Max 5 consecutive losses
            'weight': 1.5,
        },
        'consistency': {
            'description': 'Win rate standard deviation across weeks',
            'pass_threshold': 0.20,  # Max 20% weekly variance
            'weight': 1.0,
        },
        'edge_stability': {
            'description': 'Recent edge vs historical edge ratio',
            'pass_threshold': 0.7,  # Recent must be >= 70% of historical
            'weight': 2.0,
        },
    }

    def evaluate(self, metrics):
        """
        Evaluate a strategy against the rubric.

        Args:
            metrics: dict with keys matching DIMENSIONS

        Returns:
            dict with per-dimension scores and overall grade
        """
        results = {}
        weighted_sum = 0
        total_weight = 0

        for dim_name, dim_config in self.DIMENSIONS.items():
            value = metrics.get(dim_name)
            threshold = dim_config['pass_threshold']
            weight = dim_config['weight']

            if value is None:
                score = 0.0
                passed = False
            elif dim_name in ('drawdown', 'consistency'):
                # Lower is better for these
                passed = value <= threshold
                score = min(1.0, threshold / max(value, 0.001))
            elif dim_name == 'sample_size':
                passed = value >= threshold
                score = min(1.0, value / threshold)
            else:
                passed = value >= threshold
                score = min(1.0, value / max(threshold, 0.001))

            results[dim_name] = {
                'value': value,
                'threshold': threshold,
                'passed': passed,
                'score': round(score, 3),
                'weighted_score': round(score * weight, 3),
            }

            weighted_sum += score * weight
            total_weight += weight

        overall_grade = weighted_sum / total_weight if total_weight > 0 else 0
        all_critical_pass = all(
            results[d]['passed'] for d in ('accuracy', 'roi', 'edge_stability')
            if d in results and results[d]['value'] is not None
        )

        return {
            'dimensions': results,
            'overall_grade': round(overall_grade, 3),
            'all_critical_pass': all_critical_pass,
            'recommendation': 'DEPLOY' if overall_grade >= 0.7 and all_critical_pass else 'HOLD',
        }


# =============================================================================
# REGIME DETECTION
# =============================================================================

class RegimeDetector:
    """
    Detects when recent performance diverges from historical norms.

    This catches situations where a strategy that worked historically
    has stopped working due to market changes, rule changes, or
    other regime shifts.
    """

    def __init__(self, lookback_short=20, lookback_long=100):
        self.short = lookback_short
        self.long = lookback_long

    def detect(self, results):
        """
        Check for regime change in a sequence of bet results.

        Args:
            results: List of dicts with 'fav_won' (bool) and 'date' keys

        Returns:
            dict with regime status and confidence
        """
        if len(results) < self.short:
            return {
                'regime': 'INSUFFICIENT_DATA',
                'confidence': 0,
                'message': f'Need at least {self.short} results',
            }

        recent = results[-self.short:]
        recent_wr = sum(1 for r in recent if r['fav_won']) / len(recent)

        if len(results) >= self.long:
            historical = results[:-self.short]
            hist_wr = sum(1 for r in historical if r['fav_won']) / len(historical)
        else:
            historical = results
            hist_wr = sum(1 for r in historical if r['fav_won']) / len(historical)

        # Statistical test: is recent win rate significantly different?
        delta = recent_wr - hist_wr
        n = len(recent)
        se = (hist_wr * (1 - hist_wr) / n) ** 0.5 if hist_wr > 0 and n > 0 else 0.1

        z_score = delta / se if se > 0 else 0

        if z_score < -2.0:
            regime = 'DEGRADING'
            confidence = min(1.0, abs(z_score) / 3.0)
            message = (f'Recent win rate ({recent_wr:.1%}) significantly below '
                       f'historical ({hist_wr:.1%}), z={z_score:.2f}')
        elif z_score > 2.0:
            regime = 'IMPROVING'
            confidence = min(1.0, z_score / 3.0)
            message = (f'Recent win rate ({recent_wr:.1%}) significantly above '
                       f'historical ({hist_wr:.1%}), z={z_score:.2f}')
        else:
            regime = 'STABLE'
            confidence = 1.0 - abs(z_score) / 2.0
            message = (f'Recent ({recent_wr:.1%}) consistent with '
                       f'historical ({hist_wr:.1%})')

        # Check for losing streaks
        max_streak = 0
        current_streak = 0
        for r in recent:
            if not r['fav_won']:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return {
            'regime': regime,
            'confidence': round(confidence, 3),
            'message': message,
            'recent_win_rate': round(recent_wr, 4),
            'historical_win_rate': round(hist_wr, 4),
            'z_score': round(z_score, 2),
            'max_losing_streak': max_streak,
            'recommendation': self._get_recommendation(regime, max_streak),
        }

    def _get_recommendation(self, regime, max_streak):
        """Get actionable recommendation based on regime."""
        if regime == 'DEGRADING':
            return {
                'action': 'REDUCE_EXPOSURE',
                'kelly_multiplier': 0.5,
                'message': 'Cut position sizes by 50% until regime stabilizes',
            }
        elif max_streak >= 4:
            return {
                'action': 'PAUSE_AND_REVIEW',
                'kelly_multiplier': 0.25,
                'message': 'Extended losing streak — reduce to 25% Kelly and review model',
            }
        elif regime == 'IMPROVING':
            return {
                'action': 'MAINTAIN',
                'kelly_multiplier': 1.0,
                'message': 'Strategy performing above baseline — maintain current sizing',
            }
        else:
            return {
                'action': 'NORMAL',
                'kelly_multiplier': 1.0,
                'message': 'Strategy within normal parameters',
            }


# =============================================================================
# PROGRESSIVE CONFIDENCE SCALING
# =============================================================================

class AdaptiveConfidence:
    """
    Adjusts signal confidence based on rolling recent performance.

    Instead of fixed Kelly fractions, scales the fraction based on
    how well the strategy has been performing recently.

    This follows ECC's iterative refinement pattern: extract patterns
    from completed sessions and evolve confidence accordingly.
    """

    def __init__(self, base_kelly_fractions=None, lookback=20, min_scale=0.25, max_scale=1.5):
        self.base_kelly = base_kelly_fractions or {
            'ELITE': 0.10,
            'HIGH': 0.08,
            'STRONG': 0.06,
            'DIAMOND': 0.15,
            'PLATINUM': 0.10,
            'GOLD': 0.07,
        }
        self.lookback = lookback
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.history = []

    def record_outcome(self, tier, won, date=None):
        """Record a bet outcome for confidence tracking."""
        self.history.append({
            'tier': tier,
            'won': won,
            'date': date or datetime.now().strftime('%Y%m%d'),
        })

    def get_adjusted_kelly(self, tier):
        """
        Get Kelly fraction adjusted for recent performance.

        Scale factor based on recent win rate vs expected:
          - Winning more than expected → scale up (max 1.5x)
          - Winning less than expected → scale down (min 0.25x)
          - On track → 1.0x
        """
        base = self.base_kelly.get(tier, 0.05)

        # Get recent results for this tier
        tier_results = [h for h in self.history if h['tier'] == tier]
        recent = tier_results[-self.lookback:] if tier_results else []

        if len(recent) < 5:
            return base  # Not enough data to adjust

        recent_wr = sum(1 for r in recent if r['won']) / len(recent)

        # Expected win rates by tier
        expected_wr = {
            'ELITE': 0.90, 'HIGH': 0.82, 'STRONG': 0.80,
            'DIAMOND': 1.00, 'PLATINUM': 0.97, 'GOLD': 0.95,
        }
        expected = expected_wr.get(tier, 0.80)

        # Scale factor: ratio of actual to expected, bounded
        if expected > 0:
            scale = recent_wr / expected
        else:
            scale = 1.0

        scale = max(self.min_scale, min(self.max_scale, scale))

        return round(base * scale, 4)

    def get_all_adjusted_kellys(self):
        """Get adjusted Kelly fractions for all tiers."""
        return {tier: self.get_adjusted_kelly(tier) for tier in self.base_kelly}

    def get_confidence_report(self):
        """Generate a confidence report across all tiers."""
        report = {}
        for tier in self.base_kelly:
            tier_results = [h for h in self.history if h['tier'] == tier]
            recent = tier_results[-self.lookback:]
            if recent:
                wr = sum(1 for r in recent if r['won']) / len(recent)
                adjusted = self.get_adjusted_kelly(tier)
                report[tier] = {
                    'base_kelly': self.base_kelly[tier],
                    'adjusted_kelly': adjusted,
                    'scale_factor': round(adjusted / self.base_kelly[tier], 2),
                    'recent_win_rate': round(wr, 3),
                    'recent_n': len(recent),
                }
        return report


# =============================================================================
# NIGHTLY PICK VERIFICATION
# =============================================================================

class NightlyVerifier:
    """
    Structured go/no-go evaluation for each night's picks.

    Follows ECC's checkpoint pattern: verify before deploying,
    not after problems emerge.
    """

    def __init__(self):
        self.rubric = StrategyRubric()
        self.regime_detector = RegimeDetector()
        self.adaptive_confidence = AdaptiveConfidence()

    def verify_picks(self, picks, historical_results):
        """
        Run full verification pipeline on tonight's picks.

        Args:
            picks: List of tonight's proposed picks
            historical_results: List of past results for context

        Returns:
            Verification report with go/no-go recommendation
        """
        # Step 1: Regime check
        regime = self.regime_detector.detect(historical_results)

        # Step 2: Calculate rubric metrics
        recent_n = min(20, len(historical_results))
        recent = historical_results[-recent_n:] if historical_results else []

        recent_wins = sum(1 for r in recent if r['fav_won'])
        recent_wr = recent_wins / len(recent) if recent else 0

        # Calculate weekly consistency
        weekly_wrs = []
        if len(historical_results) >= 7:
            for i in range(0, len(historical_results) - 6, 7):
                week = historical_results[i:i+7]
                if week:
                    wwr = sum(1 for r in week if r['fav_won']) / len(week)
                    weekly_wrs.append(wwr)

        consistency = 0
        if len(weekly_wrs) >= 2:
            mean_wwr = sum(weekly_wrs) / len(weekly_wrs)
            variance = sum((w - mean_wwr) ** 2 for w in weekly_wrs) / len(weekly_wrs)
            consistency = variance ** 0.5

        # Max consecutive losses
        max_streak = 0
        streak = 0
        for r in recent:
            if not r['fav_won']:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        # ROI
        if recent:
            profit = sum(100/110 if r['fav_won'] else -1 for r in recent)
            roi = profit / len(recent)
        else:
            roi = 0

        # Historical vs recent edge
        if len(historical_results) > recent_n:
            older = historical_results[:-recent_n]
            older_wr = sum(1 for r in older if r['fav_won']) / len(older) if older else 0
            edge_stability = recent_wr / older_wr if older_wr > 0 else 1.0
        else:
            edge_stability = 1.0

        metrics = {
            'accuracy': recent_wr,
            'sample_size': len(recent),
            'roi': roi,
            'drawdown': max_streak,
            'consistency': consistency,
            'edge_stability': edge_stability,
        }

        # Step 3: Rubric evaluation
        rubric_result = self.rubric.evaluate(metrics)

        # Step 4: Adjust picks based on regime
        kelly_mult = regime['recommendation']['kelly_multiplier']
        adjusted_picks = []
        for pick in picks:
            adj = dict(pick)
            tier = pick.get('confidence', pick.get('tier', 'STRONG'))
            base_kelly = self.adaptive_confidence.get_adjusted_kelly(tier)
            adj['adjusted_kelly'] = round(base_kelly * kelly_mult, 4)
            adj['kelly_multiplier'] = kelly_mult
            adjusted_picks.append(adj)

        # Step 5: Final recommendation
        if regime['regime'] == 'DEGRADING' and rubric_result['recommendation'] == 'HOLD':
            final_rec = 'SKIP_TONIGHT'
            reason = 'Both regime and rubric signal caution'
        elif regime['regime'] == 'DEGRADING':
            final_rec = 'REDUCED_SIZE'
            reason = 'Regime degrading — use reduced position sizes'
        elif rubric_result['recommendation'] == 'HOLD':
            final_rec = 'REDUCED_SIZE'
            reason = 'Rubric below threshold — use reduced position sizes'
        else:
            final_rec = 'FULL_SIZE'
            reason = 'All checks pass — deploy at full size'

        return {
            'recommendation': final_rec,
            'reason': reason,
            'regime': regime,
            'rubric': rubric_result,
            'metrics': metrics,
            'picks': adjusted_picks,
            'kelly_multiplier': kelly_mult,
            'timestamp': datetime.now().isoformat(),
        }

    def load_and_verify(self, picks, results_file='output/historical_results.json'):
        """Load historical results from file and run verification."""
        try:
            with open(results_file) as f:
                historical = json.load(f)
        except FileNotFoundError:
            historical = []

        return self.verify_picks(picks, historical)


# =============================================================================
# BACK-TO-BACK GAME DETECTION
# =============================================================================

class BackToBackDetector:
    """
    Detects back-to-back games and adjusts confidence accordingly.

    NBA teams playing on consecutive days show measurably worse
    performance, particularly on the road.
    """

    # Historical B2B performance penalty (points)
    B2B_PENALTY = 2.5
    B2B_ROAD_PENALTY = 4.0

    def __init__(self):
        self.team_schedule = defaultdict(list)

    def update_schedule(self, team, date):
        """Record a game date for a team."""
        self.team_schedule[team].append(date)
        # Keep last 10 dates
        if len(self.team_schedule[team]) > 10:
            self.team_schedule[team] = self.team_schedule[team][-10:]

    def is_back_to_back(self, team, game_date):
        """Check if team is playing on a back-to-back."""
        dates = self.team_schedule.get(team, [])
        if not dates:
            return False

        # Parse dates (format: YYYYMMDD)
        try:
            gd = datetime.strptime(str(game_date), '%Y%m%d')
            yesterday = (gd - timedelta(days=1)).strftime('%Y%m%d')
            return yesterday in [str(d) for d in dates]
        except (ValueError, TypeError):
            return False

    def get_adjustment(self, team, game_date, is_home):
        """
        Get confidence adjustment for B2B status.

        Returns:
            dict with is_b2b flag and adjustment factors
        """
        b2b = self.is_back_to_back(team, game_date)
        if not b2b:
            return {
                'is_b2b': False,
                'margin_penalty': 0,
                'kelly_multiplier': 1.0,
            }

        penalty = self.B2B_PENALTY if is_home else self.B2B_ROAD_PENALTY
        return {
            'is_b2b': True,
            'margin_penalty': penalty,
            'kelly_multiplier': 0.6 if not is_home else 0.8,
            'note': f'B2B {"home" if is_home else "road"} game: '
                    f'{penalty:.1f}pt penalty applied',
        }


# =============================================================================
# CORRELATION-AWARE PARLAY EVALUATION
# =============================================================================

class CorrelationAdjuster:
    """
    Adjusts parlay probability estimates for leg correlation.

    Standard parlay math assumes independence between legs.
    In reality, NBA games on the same night have correlations
    (schedule, travel, referee assignments, etc.).
    """

    # Empirical correlation estimates
    SAME_CONFERENCE_CORR = 0.05
    SAME_DIVISION_CORR = 0.08
    INDEPENDENT_CORR = 0.02

    NBA_DIVISIONS = {
        'ATL': 'SE', 'CHA': 'SE', 'MIA': 'SE', 'ORL': 'SE', 'WSH': 'SE',
        'BOS': 'AT', 'BKN': 'AT', 'NY': 'AT', 'PHI': 'AT', 'TOR': 'AT',
        'CHI': 'CE', 'CLE': 'CE', 'DET': 'CE', 'IND': 'CE', 'MIL': 'CE',
        'DAL': 'SW', 'HOU': 'SW', 'MEM': 'SW', 'NO': 'SW', 'SA': 'SW',
        'DEN': 'NW', 'MIN': 'NW', 'OKC': 'NW', 'POR': 'NW', 'UTAH': 'NW',
        'GS': 'PA', 'LAC': 'PA', 'LAL': 'PA', 'PHX': 'PA', 'SAC': 'PA',
    }

    NBA_CONFERENCES = {
        'SE': 'EAST', 'AT': 'EAST', 'CE': 'EAST',
        'SW': 'WEST', 'NW': 'WEST', 'PA': 'WEST',
    }

    def estimate_correlation(self, team1, team2):
        """Estimate correlation between two games."""
        div1 = self.NBA_DIVISIONS.get(team1, '')
        div2 = self.NBA_DIVISIONS.get(team2, '')

        if div1 and div2 and div1 == div2:
            return self.SAME_DIVISION_CORR

        conf1 = self.NBA_CONFERENCES.get(div1, '')
        conf2 = self.NBA_CONFERENCES.get(div2, '')

        if conf1 and conf2 and conf1 == conf2:
            return self.SAME_CONFERENCE_CORR

        return self.INDEPENDENT_CORR

    def adjust_parlay_probability(self, legs):
        """
        Adjust combined parlay probability for correlations.

        Uses a simplified correlation model:
        P(all win) ≈ Π(P_i) × (1 + avg_correlation × n_pairs)

        This is conservative — correlation can only reduce combined
        probability (positive correlation between wins means when one
        leg fails, others are more likely to fail too).
        """
        if len(legs) < 2:
            return legs[0].get('win_probability', 0.5) if legs else 0

        # Independent probability
        indep_prob = 1.0
        for leg in legs:
            indep_prob *= leg.get('win_probability', leg.get('hitRate', 0.85))

        # Calculate average pairwise correlation
        teams = [leg.get('team', '') for leg in legs]
        total_corr = 0
        n_pairs = 0
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                total_corr += self.estimate_correlation(teams[i], teams[j])
                n_pairs += 1

        avg_corr = total_corr / max(n_pairs, 1)

        # Correlation penalty: reduces combined probability
        # Higher correlation → bigger penalty
        penalty = 1.0 - avg_corr * n_pairs * 0.5
        penalty = max(0.5, min(1.0, penalty))

        adjusted_prob = indep_prob * penalty

        return {
            'independent_probability': round(indep_prob, 6),
            'adjusted_probability': round(adjusted_prob, 6),
            'correlation_penalty': round(1 - penalty, 4),
            'avg_pairwise_correlation': round(avg_corr, 4),
            'n_pairs': n_pairs,
        }


# =============================================================================
# CONVENIENCE: Run verification on tonight's picks
# =============================================================================

def verify_tonight(picks_file='output/tonight_picks.json',
                   results_file='output/historical_results.json'):
    """Run full verification pipeline on tonight's picks."""
    verifier = NightlyVerifier()

    try:
        with open(picks_file) as f:
            picks = json.load(f)
    except FileNotFoundError:
        print(f"No picks file found at {picks_file}")
        return

    result = verifier.load_and_verify(picks, results_file)

    print(f"\n{'='*70}")
    print(f"NIGHTLY VERIFICATION REPORT")
    print(f"{'='*70}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Reason: {result['reason']}")
    print(f"\n  Regime: {result['regime']['regime']} "
          f"(confidence: {result['regime']['confidence']:.1%})")
    print(f"  {result['regime']['message']}")
    print(f"\n  Rubric Grade: {result['rubric']['overall_grade']:.2f} "
          f"({'PASS' if result['rubric']['all_critical_pass'] else 'FAIL'})")

    for dim, score in result['rubric']['dimensions'].items():
        status = "PASS" if score['passed'] else "FAIL"
        print(f"    {dim:20s}: {score['value']:.3f} "
              f"(threshold: {score['threshold']:.3f}) [{status}]")

    print(f"\n  Kelly Multiplier: {result['kelly_multiplier']:.2f}x")
    print(f"  Adjusted Picks: {len(result['picks'])}")
    for pick in result['picks']:
        team = pick.get('favorite', pick.get('team', '???'))
        kelly = pick.get('adjusted_kelly', 0)
        conf = pick.get('confidence', '???')
        print(f"    {team} [{conf}] Kelly={kelly:.3f}")

    return result


if __name__ == '__main__':
    verify_tonight()
