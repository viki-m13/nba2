"""
MiroFish MinMax Adversarial Strategy Development Pipeline
============================================================
Patent-Pending: MinMax Adversarial Certainty Floor Refinement

Uses Claude AI agents in an adversarial MIN/MAX debate to stress-test
and refine the Certainty Floor discovered by the CFD engine.

Architecture:
  MAX Agent (Exploiter)   — Finds the LOWEST floor that still wins 100%
  MIN Agent (Attacker)    — Finds scenarios where the floor FAILS
  Forensic Agent          — Deep-dives into every loss to find root causes
  Regime Agent            — Tests floor stability across time periods
  Synthesis Director      — Resolves debates, produces final rules

The agents DON'T make bets. They refine the LIMIT PRICE threshold.
Once refined, the strategy executes as pure deterministic rules.
"""

import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import anthropic

from ml_config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, ML_RULES_FILE,
    ML_RESEARCH_DIR, RESULTS_DIR, MAX_DEV_ITERATIONS,
    MIN_AGENTS_CONSENSUS, ADVERSARIAL_ROUNDS
)


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ============================================================================
# MINMAX AGENT SYSTEM PROMPTS
# ============================================================================

AGENT_PROMPTS = {
    'max_exploiter': """You are the MAX Agent (Exploiter) in a MiroFish MinMax adversarial system.

Your goal: Find the LOWEST possible Polymarket limit price that still achieves 100% win rate.
Lower price = more profit per bet AND more games that fill at that price.

You will receive:
1. Certainty floor analysis showing win rates at each probability threshold
2. Contextual analysis showing floors under different conditions
3. Historical upset data showing when heavy favorites lost

Your task:
1. Identify the lowest probability threshold with 100% win rate AND adequate sample size
2. Find contextual conditions that LOWER the floor (more profitable)
3. Propose multi-condition rules that capture more value
4. Calculate expected profit per share and daily fill rate at each proposed floor

Be AGGRESSIVE but honest. Push for the lowest floor supported by data.
Every percentage point lower = more profit. But one loss = strategy death.

CRITICAL: Base your analysis ONLY on the data provided. Do not invent statistics.""",

    'min_attacker': """You are the MIN Agent (Attacker) in a MiroFish MinMax adversarial system.

Your goal: Find WEAKNESSES in the proposed certainty floor. Find scenarios where
the floor will FAIL and the favorite will lose.

You will receive:
1. Proposed certainty floor from the MAX Agent and CFD engine
2. Historical data showing upset patterns
3. Walk-forward validation results

Your task:
1. Identify games where the favorite lost at high implied probabilities
2. Find patterns in upsets — are there conditions that make upsets MORE likely?
3. Challenge sample size sufficiency — is n large enough to trust 100%?
4. Identify regime risks — rule changes, market evolution, black swans
5. Propose HIGHER floor thresholds with safety margins

Be ADVERSARIAL. Your job is to break the strategy. Every proposed floor is
guilty until proven innocent. Find the edge cases, the corner cases, the
scenarios where baseball's randomness defeats probability.

Use statistical reasoning: if true win rate is 97% and n=30, there's a
(0.97)^30 = 40% chance of seeing zero losses. That's NOT proof of 100%.""",

    'forensic_agent': """You are the Forensic Agent in a MiroFish MinMax adversarial system.

Your goal: Deep-dive into EVERY upset (favorite loss) to find root causes
and identify patterns that could predict future upsets.

You will receive:
1. Full list of upsets at various probability thresholds
2. Game context (odds, teams, scores, dates)
3. Feature data for upset games vs non-upset games

Your task:
1. Categorize each upset by root cause:
   - Starting pitcher scratched/injured after odds set?
   - Weather event (rain delay, wind)?
   - Team had nothing to play for (late season)?
   - Historic parity (division rival, etc.)?
   - Genuine randomness
2. Identify ACTIONABLE patterns — conditions that predict upsets
3. Propose exclusion filters that remove high-upset-risk games
4. Calculate how many upsets each filter would have prevented

Be THOROUGH. Every upset that can be explained is an upset that can be filtered.""",

    'regime_agent': """You are the Regime Agent in a MiroFish MinMax adversarial system.

Your goal: Test whether the certainty floor is STABLE across different
time periods, contexts, and market regimes.

You will receive:
1. Certainty floor data across different months and conditions
2. Walk-forward validation results by window
3. MLB context (2023 rule changes, market evolution)

Your task:
1. Test floor stability by month — does it hold in April vs September?
2. Test floor stability by odds bracket — does it hold for -150 vs -300 favorites?
3. Identify regime changes that could invalidate the floor
4. Propose time-decay adjustments (more recent data weighted higher)
5. Calculate confidence intervals for the true win rate

Be SKEPTICAL of any pattern that doesn't hold across multiple time periods.
A floor that works in June but fails in September is worthless.""",

    'synthesis_director': """You are the Synthesis Director in a MiroFish MinMax adversarial system.

Your goal: Resolve the MIN/MAX debate and produce FINAL strategy rules.

You receive analyses from:
- MAX Agent (wants lower floor = more profit)
- MIN Agent (wants higher floor = more safety)
- Forensic Agent (upset analysis and filters)
- Regime Agent (stability analysis)

Your task:
1. Weigh the evidence from all agents
2. Set the FINAL certainty floor that balances profit vs safety
3. Incorporate contextual rules that LOWER the floor under safe conditions
4. Add exclusion filters from the Forensic Agent
5. Ensure the strategy produces DAILY bet opportunities

OUTPUT FORMAT — You MUST output ONLY valid JSON:
{
    "certainty_floor_pct": number (50-99),
    "limit_price": number (0.50-0.99),
    "contextual_rules": [
        {
            "name": "rule_name",
            "context_description": "when this condition is true",
            "adjusted_floor_pct": number,
            "conditions": {"feature": {"operator": ">=", "value": number}},
            "evidence": "why this lower floor is safe"
        }
    ],
    "exclusion_filters": [
        {
            "name": "filter_name",
            "description": "games to SKIP even above the floor",
            "conditions": {"feature": {"operator": ">=", "value": number}},
            "upsets_prevented": number
        }
    ],
    "confidence_assessment": {
        "statistical_confidence": "high/medium/low",
        "sample_size_adequate": true/false,
        "regime_stable": true/false,
        "recommended_action": "deploy/test_more/kill"
    },
    "reasoning": "synthesis explanation"
}"""
}


def _call_agent(client: anthropic.Anthropic, agent_type: str,
                user_content: str, max_tokens: int = 4000) -> str:
    """Call a single MiroFish MinMax agent."""
    system_prompt = AGENT_PROMPTS[agent_type]

    for attempt in range(4):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.3 if agent_type != 'synthesis_director' else 0.1,
            )
            return response.content[0].text
        except Exception as e:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"  Agent {agent_type} error: {e}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"  Agent {agent_type} failed after retries: {e}")
            return f"ERROR: {e}"


def run_minmax_refinement(strategy: dict, floor_data: dict,
                           df: pd.DataFrame) -> dict:
    """
    Run the full MinMax adversarial refinement pipeline.

    Takes the CFD engine's initial strategy and puts it through
    adversarial debate to find weaknesses and optimize the floor.
    """
    if not ANTHROPIC_API_KEY:
        print("WARNING: No ANTHROPIC_API_KEY — skipping MinMax refinement")
        print("  Using CFD engine results directly")
        return strategy

    client = get_client()

    # Build data summaries for agents
    floor_summary = _build_floor_summary(floor_data, df)
    strategy_summary = json.dumps(strategy, indent=2, default=str)

    # Upset summary
    upset_summary = _build_upset_summary(floor_data, df)

    print(f"\n{'='*70}")
    print(f"MIROFISH MINMAX ADVERSARIAL REFINEMENT")
    print(f"{'='*70}")
    print(f"Model: {CLAUDE_MODEL}")

    for round_num in range(1, ADVERSARIAL_ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"ADVERSARIAL ROUND {round_num}/{ADVERSARIAL_ROUNDS}")
        print(f"{'='*70}")

        # ---- MAX Agent ----
        print(f"\n  MAX Agent (Exploiter) — finding lowest safe floor...")
        max_input = (
            f"Analyze this certainty floor data to find the LOWEST safe limit price.\n\n"
            f"CURRENT STRATEGY:\n{strategy_summary}\n\n"
            f"FLOOR DATA:\n{floor_summary}\n\n"
            f"UPSET EXAMPLES:\n{upset_summary}"
        )
        max_result = _call_agent(client, 'max_exploiter', max_input)
        print(f"  MAX Agent complete ({len(max_result)} chars)")

        # ---- MIN Agent ----
        print(f"\n  MIN Agent (Attacker) — finding weaknesses...")
        min_input = (
            f"Attack the proposed certainty floor. Find where it will FAIL.\n\n"
            f"MAX AGENT PROPOSAL:\n{max_result}\n\n"
            f"CURRENT STRATEGY:\n{strategy_summary}\n\n"
            f"FLOOR DATA:\n{floor_summary}\n\n"
            f"UPSET EXAMPLES:\n{upset_summary}"
        )
        min_result = _call_agent(client, 'min_attacker', min_input)
        print(f"  MIN Agent complete ({len(min_result)} chars)")

        # ---- Forensic Agent ----
        print(f"\n  Forensic Agent — analyzing upsets...")
        forensic_input = (
            f"Deep-dive into every upset. Find patterns and propose filters.\n\n"
            f"UPSET EXAMPLES:\n{upset_summary}\n\n"
            f"FLOOR DATA:\n{floor_summary}\n\n"
            f"MAX AGENT ANALYSIS:\n{max_result}\n\n"
            f"MIN AGENT CONCERNS:\n{min_result}"
        )
        forensic_result = _call_agent(client, 'forensic_agent', forensic_input)
        print(f"  Forensic Agent complete ({len(forensic_result)} chars)")

        # ---- Regime Agent ----
        print(f"\n  Regime Agent — testing stability...")
        regime_input = (
            f"Test floor stability across time periods and conditions.\n\n"
            f"STRATEGY:\n{strategy_summary}\n\n"
            f"FLOOR DATA:\n{floor_summary}\n\n"
            f"MAX FINDINGS:\n{max_result}\n\n"
            f"MIN CONCERNS:\n{min_result}\n\n"
            f"FORENSIC FINDINGS:\n{forensic_result}"
        )
        regime_result = _call_agent(client, 'regime_agent', regime_input)
        print(f"  Regime Agent complete ({len(regime_result)} chars)")

        # ---- Synthesis Director ----
        print(f"\n  Synthesis Director — resolving debate...")
        synthesis_input = (
            f"Resolve the MinMax debate. Produce FINAL strategy rules.\n\n"
            f"MAX AGENT (wants lower floor):\n{max_result}\n\n"
            f"MIN AGENT (wants higher floor):\n{min_result}\n\n"
            f"FORENSIC AGENT (upset analysis):\n{forensic_result}\n\n"
            f"REGIME AGENT (stability):\n{regime_result}\n\n"
            f"CURRENT STRATEGY:\n{strategy_summary}\n\n"
            f"OUTPUT ONLY VALID JSON. No markdown, no explanation outside JSON."
        )
        synthesis_result = _call_agent(client, 'synthesis_director', synthesis_input,
                                        max_tokens=6000)
        print(f"  Synthesis Director complete ({len(synthesis_result)} chars)")

        # Parse synthesis result
        refined = _parse_rules_json(synthesis_result)
        if refined:
            # Update strategy with agent refinements
            strategy = _merge_refinements(strategy, refined)
            strategy_summary = json.dumps(strategy, indent=2, default=str)
            print(f"\n  Round {round_num} refinement applied")
            print(f"    Floor: {refined.get('certainty_floor_pct', 'unchanged')}%")
            confidence = refined.get('confidence_assessment', {})
            print(f"    Confidence: {confidence.get('statistical_confidence', 'N/A')}")
            print(f"    Action: {confidence.get('recommended_action', 'N/A')}")

        # Save round log
        _save_round_log(round_num, {
            'max': max_result,
            'min': min_result,
            'forensic': forensic_result,
            'regime': regime_result,
            'synthesis': synthesis_result,
            'parsed': refined,
        })

    return strategy


def _build_floor_summary(floor_data: dict, df: pd.DataFrame) -> str:
    """Build a data summary for agents."""
    lines = []
    lines.append(f"=== CERTAINTY FLOOR DATA (REAL MLB 2025) ===")
    lines.append(f"Total games: {len(df)}")
    lines.append(f"Favorite win rate overall: {df['fav_won'].mean():.1%}")
    lines.append(f"")
    lines.append(f"WIN RATE BY PROBABILITY THRESHOLD:")

    for threshold in range(50, 100):
        data = floor_data.get(threshold, {})
        n = data.get('n_games', 0)
        if n == 0:
            continue
        wr = data.get('win_rate', 0)
        losses = data.get('fav_losses', 0)
        lines.append(f"  >= {threshold}%: {n} games, {wr:.1%} win rate, {losses} losses")

    # Add distribution info
    if not df.empty:
        lines.append(f"")
        lines.append(f"ODDS DISTRIBUTION:")
        brackets = [
            ('Pick-em (-100 to -110)', -110, -100),
            ('Slight fav (-111 to -150)', -150, -111),
            ('Moderate fav (-151 to -200)', -200, -151),
            ('Heavy fav (-201 to -300)', -300, -201),
            ('Extreme fav (-301+)', -9999, -301),
        ]
        for name, ml_min, ml_max in brackets:
            bracket = df[(df['fav_ml'] >= ml_min) & (df['fav_ml'] <= ml_max)]
            if len(bracket) > 0:
                wr = bracket['fav_won'].mean()
                lines.append(f"  {name}: {len(bracket)} games, {wr:.1%} fav wins")

    return '\n'.join(lines)


def _build_upset_summary(floor_data: dict, df: pd.DataFrame) -> str:
    """Build upset summary for agents."""
    lines = []
    lines.append(f"=== UPSET ANALYSIS (FAVORITES THAT LOST) ===")

    # Collect all upsets at high probabilities
    all_upsets = []
    for threshold in range(60, 100):
        data = floor_data.get(threshold, {})
        for ex in data.get('loss_examples', []):
            if ex not in all_upsets:
                all_upsets.append(ex)

    # Deduplicate by game
    seen = set()
    unique_upsets = []
    for u in all_upsets:
        key = (u.get('date', ''), u.get('game', ''))
        if key not in seen:
            seen.add(key)
            unique_upsets.append(u)

    # Sort by fav_prob descending
    unique_upsets.sort(key=lambda x: x.get('fav_prob', 0), reverse=True)

    lines.append(f"Total unique upsets at >= 60% implied: {len(unique_upsets)}")
    lines.append(f"")

    for u in unique_upsets[:30]:
        lines.append(
            f"  {u.get('date', '?')}: {u.get('game', '?')} — "
            f"{u.get('fav_team', '?')} ({u.get('fav_prob', 0):.1%} implied, "
            f"ML {u.get('fav_ml', '?')}) LOST, score: {u.get('score', '?')}"
        )

    return '\n'.join(lines)


def _parse_rules_json(text: str) -> dict:
    """Extract and parse JSON from agent output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for marker in ['```json', '```']:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index('```', start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    if '{' in text and '}' in text:
        depth = 0
        start = None
        for i, c in enumerate(text):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        continue

    return None


def _merge_refinements(strategy: dict, refined: dict) -> dict:
    """Merge agent refinements into the strategy."""
    merged = strategy.copy()

    # Update floor if agents propose a different one
    if 'certainty_floor_pct' in refined:
        new_floor = refined['certainty_floor_pct']
        merged['base_rule']['floor_pct'] = new_floor
        merged['base_rule']['limit_price'] = new_floor / 100.0
        merged['base_rule']['profit_per_share'] = round(1 - new_floor / 100.0, 4)

    # Add contextual rules
    if 'contextual_rules' in refined:
        merged['agent_contextual_rules'] = refined['contextual_rules']

    # Add exclusion filters
    if 'exclusion_filters' in refined:
        merged['agent_exclusion_filters'] = refined['exclusion_filters']

    # Add confidence assessment
    if 'confidence_assessment' in refined:
        merged['agent_confidence'] = refined['confidence_assessment']

    # Add reasoning
    if 'reasoning' in refined:
        merged['agent_reasoning'] = refined['reasoning']

    return merged


def _save_round_log(round_num: int, data: dict):
    """Save adversarial round outputs."""
    log_dir = os.path.join(RESULTS_DIR, 'minmax_logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'round_{round_num}_{timestamp}.json')

    clean_data = {}
    for k, v in data.items():
        if isinstance(v, str):
            clean_data[k] = v
        else:
            clean_data[k] = json.dumps(v, default=str)

    with open(log_file, 'w') as f:
        json.dump(clean_data, f, indent=2, default=str)
    print(f"  Round log saved: {log_file}")
