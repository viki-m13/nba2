"""
MiroFish Multi-Agent Strategy Development Pipeline
====================================================
Adapts the MiroFish architecture to MLB +200 underdog strategy development.

This is the DEVELOPMENT tool — Claude AI agents collaborate to discover,
analyze, and refine strategy rules. The output is a deterministic rule set
(strategy_rules.json) that the strategy engine executes WITHOUT AI.

MiroFish Architecture Mapping:
  Stage 1 (Ontology)    → Data Ontology Agent: identifies what factors matter
  Stage 2 (Graph Build)  → Pattern Discovery Agent: finds predictive patterns in data
  Stage 3 (Simulation)   → Strategy Architect Agent: proposes rule structures
  Stage 4 (Report)       → Validation Agent: stress-tests rules for robustness
  Stage 5 (Synthesis)    → Synthesis Director: combines all into final rule set

Patent-Pending Innovation:
  Multi-agent AI swarm used as a strategy DEVELOPMENT tool, not for runtime
  prediction. Agents don't make bets — they design the bet-selection rules
  through collaborative analysis, debate, and iterative refinement.
"""

import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import anthropic

from config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, RULES_FILE,
    DATA_DIR, RESULTS_DIR, MAX_DEV_ITERATIONS,
    MIN_AGENTS_CONSENSUS, ROLLING_WINDOWS
)


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ============================================================================
# AGENT SYSTEM PROMPTS — Each agent has a distinct analytical perspective
# ============================================================================

AGENT_PROMPTS = {
    'data_ontology': """You are the Data Ontology Agent in a MiroFish multi-agent strategy development system.

Your role: Analyze MLB baseball data to identify which factors are most predictive of +200 underdog wins.

You will receive historical MLB game data with features. Your task is to:
1. Identify which features have the strongest signal for predicting underdog upsets
2. Determine optimal thresholds for each feature
3. Identify feature interactions (combinations that are more predictive than individual features)
4. Flag any features that might introduce bias or overfitting risk

Focus on CAUSALLY meaningful factors:
- Pitcher matchup quality differentials
- Team form/momentum (short vs medium term)
- Market mispricing signals (when odds don't reflect recent performance)
- Situational edges (rest, home/away, streaks)

You must be RIGOROUS. Only identify patterns supported by statistical evidence in the data.
Do not speculate. Every claim must reference specific data points.""",

    'pattern_discovery': """You are the Pattern Discovery Agent in a MiroFish multi-agent strategy development system.

Your role: Find specific, repeatable patterns in MLB data that predict +200 underdog wins at high accuracy.

You will receive historical data about +200 underdog opportunities and their outcomes.

Your task:
1. Segment the data to find HIGH-ACCURACY subsets (where underdogs win at elevated rates)
2. Identify multi-condition filters that isolate winning underdogs
3. Calculate the win rate, sample size, and statistical significance of each pattern
4. Rank patterns by reliability (consistency across different time periods)

CRITICAL CONSTRAINTS:
- Every pattern must have sufficient sample size (n >= 15 for any claimed win rate)
- Patterns must hold across multiple seasons (not just one hot streak)
- Report exact numbers: "X wins out of Y opportunities = Z% win rate"
- Flag any pattern that might be data-mined (found by chance in noise)

The breakeven win rate at +200 odds is 33.3%. We need patterns that SIGNIFICANTLY exceed this.
A pattern with 45%+ win rate at +200 odds is highly profitable. 50%+ is exceptional.""",

    'strategy_architect': """You are the Strategy Architect Agent in a MiroFish multi-agent strategy development system.

Your role: Design a deterministic rule-based strategy for MLB +200 underdog betting.

You will receive:
1. Pattern analysis from other agents
2. Historical data statistics
3. Previous strategy rules (if iterating)
4. Backtest results from previous iterations (if any)

Your task:
1. Design a set of DETERMINISTIC rules (no randomness, no AI at runtime)
2. Each rule must be expressible as: IF condition_1 AND condition_2 AND ... THEN BET
3. Rules should be ordered by confidence level
4. Include thresholds that can be validated in walk-forward backtesting
5. Balance accuracy vs frequency — we want near-daily bets at 90%+ accuracy

OUTPUT FORMAT — You MUST output valid JSON:
{
    "rules": [
        {
            "name": "rule_name",
            "description": "what this rule detects",
            "conditions": {
                "feature_name": {"operator": ">=", "value": 0.55},
                ...
            },
            "weight": 1-5 (confidence weight),
            "min_conditions_met": number
        }
    ],
    "global_filters": {
        "min_total_weight": minimum sum of triggered rule weights to place bet,
        "min_rules_triggered": minimum number of rules that must fire,
        "max_bets_per_day": maximum bets in a single day
    },
    "reasoning": "explanation of the strategy logic"
}""",

    'validation': """You are the Validation Agent in a MiroFish multi-agent strategy development system.

Your role: Stress-test proposed strategy rules for robustness, overfitting, and bias.

You will receive:
1. Proposed strategy rules
2. Backtest results
3. Historical data statistics

Your task:
1. Identify potential OVERFITTING: Are rules too specific to historical data?
2. Check for DATA LEAKAGE: Do any rules use information not available pre-game?
3. Assess REGIME SENSITIVITY: Would rules work in different market conditions?
4. Evaluate SAMPLE SIZE: Are win rates based on too few observations?
5. Check for SURVIVORSHIP BIAS: Are we only looking at winners?
6. Assess FREQUENCY: Can we achieve near-daily bet recommendations?

For each issue found, suggest SPECIFIC modifications to make rules more robust.

Be adversarial — your job is to find weaknesses. The strategy is only as good as
its weakest assumption.""",

    'synthesis_director': """You are the Synthesis Director in a MiroFish multi-agent strategy development system.

Your role: Combine insights from all specialist agents into a FINAL strategy rule set.

You receive analyses from:
- Data Ontology Agent (feature importance)
- Pattern Discovery Agent (predictive patterns)
- Strategy Architect Agent (proposed rules)
- Validation Agent (robustness assessment)

Your task:
1. Resolve disagreements between agents
2. Keep only rules that MULTIPLE agents support (consensus >= 3 of 4)
3. Adjust thresholds based on validation feedback
4. Ensure the final strategy is:
   a. DETERMINISTIC (no randomness)
   b. ANTI-FRAGILE (robust to changing conditions)
   c. FREQUENT (produces near-daily recommendations)
   d. ACCURATE (targets 90%+ win rate)
   e. PROFITABLE (positive ROI at +200 odds)

OUTPUT FORMAT — You MUST output ONLY valid JSON matching this exact structure:
{
    "version": "iteration number",
    "developed_by": "microfish_swarm",
    "development_date": "YYYY-MM-DD",
    "model_used": "claude-opus-4-6",
    "rules": [
        {
            "name": "rule_identifier",
            "description": "what this rule detects and why it works",
            "conditions": {
                "feature_name": {"operator": ">=|<=|>|<|==", "value": number}
            },
            "weight": 1-5,
            "min_conditions_met": number_of_conditions_that_must_be_true
        }
    ],
    "global_filters": {
        "min_total_weight": number,
        "min_rules_triggered": number,
        "max_bets_per_day": number
    },
    "agent_consensus": {
        "ontology_supported": ["list of rule names supported"],
        "pattern_supported": ["list of rule names supported"],
        "architect_supported": ["list of rule names supported"],
        "validation_approved": ["list of rule names approved"]
    },
    "reasoning": "high-level strategy explanation",
    "risk_notes": "known limitations and assumptions"
}"""
}


def _format_data_summary(df: pd.DataFrame) -> str:
    """Create a comprehensive data summary for agents to analyze."""
    if df.empty:
        return "No data available."

    lines = []
    lines.append(f"=== MLB +200 UNDERDOG DATA SUMMARY ===")
    lines.append(f"Total opportunities: {len(df)}")

    if 'bet_won' in df.columns:
        total_wins = df['bet_won'].sum()
        total = len(df)
        lines.append(f"Overall win rate: {total_wins}/{total} = {total_wins/total:.1%}")
        lines.append(f"Breakeven at +200: 33.3%")
        lines.append(f"")

    if 'season' in df.columns and 'bet_won' in df.columns:
        lines.append("BY SEASON:")
        for season in sorted(df['season'].unique()):
            s = df[df['season'] == season]
            w = s['bet_won'].sum()
            lines.append(f"  {int(season)}: {w}/{len(s)} = {w/len(s):.1%} win rate")
        lines.append("")

    # Feature distributions for winning vs losing underdogs
    if 'bet_won' in df.columns:
        wins = df[df['bet_won'] == 1]
        losses = df[df['bet_won'] == 0]

        lines.append("FEATURE ANALYSIS (Winners vs Losers):")
        feature_cols = [c for c in df.columns if any(
            c.startswith(p) for p in ['dog_', 'fav_']
        ) and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

        for col in sorted(feature_cols):
            if col in wins.columns and col in losses.columns:
                w_mean = wins[col].mean()
                l_mean = losses[col].mean()
                diff = w_mean - l_mean
                if abs(diff) > 0.01:
                    lines.append(
                        f"  {col}: winners={w_mean:.3f}, losers={l_mean:.3f}, "
                        f"diff={diff:+.3f}"
                    )

        lines.append("")
        lines.append("HIGH WIN-RATE SEGMENTS:")
        # Find high-accuracy subsets
        for col in feature_cols:
            if df[col].dtype in ['float64', 'float32']:
                for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                    subset = df[df[col] >= threshold]
                    if len(subset) >= 15:
                        wr = subset['bet_won'].mean()
                        if wr >= 0.40:
                            lines.append(
                                f"  {col} >= {threshold}: "
                                f"{subset['bet_won'].sum()}/{len(subset)} = "
                                f"{wr:.1%} ({len(subset)} games)"
                            )
            elif df[col].dtype in ['int64', 'int32']:
                for threshold in range(1, 6):
                    subset = df[df[col] >= threshold]
                    if len(subset) >= 10:
                        wr = subset['bet_won'].mean()
                        if wr >= 0.40:
                            lines.append(
                                f"  {col} >= {threshold}: "
                                f"{subset['bet_won'].sum()}/{len(subset)} = "
                                f"{wr:.1%} ({len(subset)} games)"
                            )

        # Multi-condition segments
        lines.append("")
        lines.append("MULTI-CONDITION SEGMENTS:")
        if 'dog_win_pct_7d' in df.columns and 'fav_win_pct_7d' in df.columns:
            for dog_thresh in [0.50, 0.55, 0.60]:
                for fav_thresh in [0.55, 0.50, 0.45, 0.40]:
                    subset = df[
                        (df['dog_win_pct_7d'] >= dog_thresh) &
                        (df['fav_win_pct_7d'] <= fav_thresh)
                    ]
                    if len(subset) >= 10:
                        wr = subset['bet_won'].mean()
                        if wr >= 0.38:
                            lines.append(
                                f"  dog_7d>={dog_thresh} AND fav_7d<={fav_thresh}: "
                                f"{subset['bet_won'].sum()}/{len(subset)} = "
                                f"{wr:.1%}"
                            )

        if 'dog_streak_7d' in df.columns:
            for streak in [2, 3, 4]:
                for wp in [0.50, 0.55, 0.60]:
                    if 'dog_win_pct_7d' in df.columns:
                        subset = df[
                            (df['dog_streak_7d'] >= streak) &
                            (df['dog_win_pct_7d'] >= wp)
                        ]
                        if len(subset) >= 8:
                            wr = subset['bet_won'].mean()
                            if wr >= 0.38:
                                lines.append(
                                    f"  dog_streak>={streak} AND dog_7d>={wp}: "
                                    f"{subset['bet_won'].sum()}/{len(subset)} = "
                                    f"{wr:.1%}"
                                )

    # Odds distribution
    if 'bet_odds' in df.columns:
        lines.append("")
        lines.append("ODDS DISTRIBUTION:")
        for bracket_min, bracket_max in [(200, 250), (250, 350), (350, 500), (500, 1000)]:
            subset = df[(df['bet_odds'] >= bracket_min) & (df['bet_odds'] < bracket_max)]
            if len(subset) >= 5 and 'bet_won' in df.columns:
                wr = subset['bet_won'].mean()
                lines.append(
                    f"  +{bracket_min} to +{bracket_max}: "
                    f"{subset['bet_won'].sum()}/{len(subset)} = {wr:.1%}"
                )

    return '\n'.join(lines)


def _call_agent(client: anthropic.Anthropic, agent_type: str,
                user_content: str, max_tokens: int = 4000) -> str:
    """Call a single MiroFish development agent."""
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


def run_development_pipeline(opportunities_df: pd.DataFrame,
                              previous_rules: dict = None,
                              backtest_results: dict = None,
                              iteration: int = 1) -> dict:
    """
    Run the full MiroFish multi-agent strategy development pipeline.

    Stage 1: Data Ontology Agent analyzes features
    Stage 2: Pattern Discovery Agent finds predictive patterns
    Stage 3: Strategy Architect Agent proposes rules
    Stage 4: Validation Agent stress-tests rules
    Stage 5: Synthesis Director produces final rule set

    Returns: strategy_rules dict ready for strategy_engine.py
    """
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY required for MiroFish development pipeline")
        return {}

    client = get_client()
    data_summary = _format_data_summary(opportunities_df)

    print(f"\n{'='*70}")
    print(f"MICROFISH DEVELOPMENT PIPELINE — ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"Model: {CLAUDE_MODEL}")
    print(f"Opportunities analyzed: {len(opportunities_df)}")

    # Build context with previous iteration results if available
    iteration_context = ""
    if previous_rules:
        iteration_context += f"\n\nPREVIOUS STRATEGY RULES (iteration {iteration-1}):\n"
        iteration_context += json.dumps(previous_rules, indent=2)
    if backtest_results:
        iteration_context += f"\n\nPREVIOUS BACKTEST RESULTS:\n"
        iteration_context += json.dumps(backtest_results, indent=2)
        iteration_context += "\n\nIMPORTANT: Analyze why the previous strategy "
        iteration_context += "succeeded or failed and propose improvements."

    # ---- STAGE 1: Data Ontology Agent ----
    print(f"\n  STAGE 1: Data Ontology Agent analyzing features...")
    ontology_input = (
        f"Analyze this MLB +200 underdog dataset to identify the most "
        f"predictive features for underdog wins:\n\n{data_summary}"
        f"{iteration_context}"
    )
    ontology_result = _call_agent(client, 'data_ontology', ontology_input)
    print(f"  Stage 1 complete ({len(ontology_result)} chars)")

    # ---- STAGE 2: Pattern Discovery Agent ----
    print(f"\n  STAGE 2: Pattern Discovery Agent finding patterns...")
    pattern_input = (
        f"Find high-accuracy patterns in this MLB +200 underdog data.\n\n"
        f"DATA:\n{data_summary}\n\n"
        f"ONTOLOGY AGENT FINDINGS:\n{ontology_result}"
        f"{iteration_context}"
    )
    pattern_result = _call_agent(client, 'pattern_discovery', pattern_input)
    print(f"  Stage 2 complete ({len(pattern_result)} chars)")

    # ---- STAGE 3: Strategy Architect Agent ----
    print(f"\n  STAGE 3: Strategy Architect Agent designing rules...")
    architect_input = (
        f"Design a deterministic rule-based strategy based on these analyses.\n\n"
        f"DATA SUMMARY:\n{data_summary}\n\n"
        f"ONTOLOGY ANALYSIS:\n{ontology_result}\n\n"
        f"PATTERN ANALYSIS:\n{pattern_result}"
        f"{iteration_context}"
    )
    architect_result = _call_agent(client, 'strategy_architect', architect_input,
                                    max_tokens=6000)
    print(f"  Stage 3 complete ({len(architect_result)} chars)")

    # ---- STAGE 4: Validation Agent ----
    print(f"\n  STAGE 4: Validation Agent stress-testing rules...")
    validation_input = (
        f"Stress-test these proposed strategy rules for robustness.\n\n"
        f"PROPOSED RULES:\n{architect_result}\n\n"
        f"DATA SUMMARY:\n{data_summary}\n\n"
        f"ONTOLOGY FINDINGS:\n{ontology_result}\n\n"
        f"PATTERN FINDINGS:\n{pattern_result}"
        f"{iteration_context}"
    )
    validation_result = _call_agent(client, 'validation', validation_input)
    print(f"  Stage 4 complete ({len(validation_result)} chars)")

    # ---- STAGE 5: Synthesis Director ----
    print(f"\n  STAGE 5: Synthesis Director producing final rules...")
    synthesis_input = (
        f"Combine all agent analyses into a FINAL strategy rule set.\n\n"
        f"DATA ONTOLOGY AGENT:\n{ontology_result}\n\n"
        f"PATTERN DISCOVERY AGENT:\n{pattern_result}\n\n"
        f"STRATEGY ARCHITECT AGENT:\n{architect_result}\n\n"
        f"VALIDATION AGENT:\n{validation_result}\n\n"
        f"DATA SUMMARY:\n{data_summary}"
        f"{iteration_context}\n\n"
        f"OUTPUT ONLY VALID JSON. No markdown, no explanation outside the JSON."
    )
    synthesis_result = _call_agent(client, 'synthesis_director', synthesis_input,
                                    max_tokens=8000)
    print(f"  Stage 5 complete ({len(synthesis_result)} chars)")

    # Parse the synthesis result into strategy rules
    rules = _parse_rules_json(synthesis_result)
    if not rules:
        print("  WARNING: Could not parse synthesis output, saving raw response")
        rules = {
            'version': str(iteration),
            'error': 'parse_failed',
            'raw_synthesis': synthesis_result,
            'raw_architect': architect_result,
        }

    # Save agent outputs for audit trail
    _save_development_log(iteration, {
        'ontology': ontology_result,
        'patterns': pattern_result,
        'architect': architect_result,
        'validation': validation_result,
        'synthesis': synthesis_result,
        'parsed_rules': rules,
    })

    return rules


def _parse_rules_json(text: str) -> dict:
    """Extract and parse JSON rules from agent output."""
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    for marker in ['```json', '```']:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index('```', start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    # Try finding JSON object boundaries
    if '{' in text and '}' in text:
        # Find the outermost { }
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


def _save_development_log(iteration: int, data: dict):
    """Save development pipeline outputs for audit trail."""
    log_dir = os.path.join(RESULTS_DIR, 'dev_logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'iteration_{iteration}_{timestamp}.json')

    # Make serializable
    clean_data = {}
    for k, v in data.items():
        if isinstance(v, str):
            clean_data[k] = v
        else:
            clean_data[k] = json.dumps(v, default=str)

    with open(log_file, 'w') as f:
        json.dump(clean_data, f, indent=2, default=str)
    print(f"  Development log saved: {log_file}")


def save_strategy_rules(rules: dict, filepath: str = None):
    """Save finalized strategy rules to JSON file."""
    if filepath is None:
        filepath = RULES_FILE

    with open(filepath, 'w') as f:
        json.dump(rules, f, indent=2, default=str)
    print(f"\nStrategy rules saved to: {filepath}")


def load_strategy_rules(filepath: str = None) -> dict:
    """Load strategy rules from JSON file."""
    if filepath is None:
        filepath = RULES_FILE

    if not os.path.exists(filepath):
        print(f"No strategy rules found at {filepath}")
        print("Run the development pipeline first: python run.py develop")
        return {}

    with open(filepath, 'r') as f:
        return json.load(f)


def iterative_development(opportunities_df: pd.DataFrame,
                           max_iterations: int = None) -> dict:
    """
    Run the full iterative MiroFish development loop:
    1. Develop initial rules via multi-agent pipeline
    2. Backtest the rules
    3. Feed results back to agents for refinement
    4. Repeat until convergence or max iterations

    This mirrors MiroFish's knowledge graph feedback loop where
    simulation results are written back for the next analysis cycle.
    """
    if max_iterations is None:
        max_iterations = MAX_DEV_ITERATIONS

    # Import here to avoid circular dependency
    from backtest import run_backtest_with_rules
    from strategy_engine import StrategyEngine

    current_rules = None
    backtest_results = None
    best_rules = None
    best_accuracy = 0

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'#'*70}")
        print(f"# DEVELOPMENT ITERATION {iteration}/{max_iterations}")
        print(f"{'#'*70}")

        # Run MiroFish development pipeline
        rules = run_development_pipeline(
            opportunities_df,
            previous_rules=current_rules,
            backtest_results=backtest_results,
            iteration=iteration,
        )

        if not rules or 'error' in rules:
            print(f"  Iteration {iteration} failed to produce valid rules")
            continue

        current_rules = rules

        # Save rules for this iteration
        iter_rules_file = os.path.join(
            RESULTS_DIR, f'strategy_rules_iter{iteration}.json'
        )
        save_strategy_rules(rules, iter_rules_file)

        # Backtest the rules
        print(f"\n  Backtesting iteration {iteration} rules...")
        try:
            engine = StrategyEngine(rules)
            backtest_results = run_backtest_with_rules(
                opportunities_df, engine
            )

            accuracy = backtest_results.get('accuracy', 0)
            roi = backtest_results.get('roi', 0)
            n_bets = backtest_results.get('total_bets', 0)
            frequency = backtest_results.get('daily_frequency', 0)

            print(f"\n  ITERATION {iteration} RESULTS:")
            print(f"    Accuracy: {accuracy:.1%}")
            print(f"    ROI: {roi:.1f}%")
            print(f"    Total bets: {n_bets}")
            print(f"    Daily frequency: {frequency:.1%}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_rules = rules.copy()
                save_strategy_rules(best_rules)
                print(f"    NEW BEST — saved to {RULES_FILE}")

        except Exception as e:
            print(f"  Backtest failed: {e}")
            backtest_results = {'error': str(e)}

    if best_rules:
        print(f"\n{'='*70}")
        print(f"DEVELOPMENT COMPLETE")
        print(f"Best accuracy achieved: {best_accuracy:.1%}")
        print(f"Rules saved to: {RULES_FILE}")
        print(f"{'='*70}")
        return best_rules

    return current_rules or {}
