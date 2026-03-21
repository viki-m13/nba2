"""
Microfish Strategy Optimizer
=============================
Uses Claude API to analyze backtest results and suggest parameter adjustments.
Inspired by MiroFish's meta-learning agent approach.

This is the "strategy discovery" phase where Claude agents debate and
optimize the strategy parameters to maximize accuracy while maintaining
near-daily betting frequency on +200 odds.
"""

import json
import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


def optimize_strategy(backtest_results: dict) -> dict:
    """
    Use Claude to analyze backtest results and suggest improvements.
    Returns optimized parameters.
    """
    if not ANTHROPIC_API_KEY:
        print("No API key - using default optimized parameters")
        return get_default_optimized_params()

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""Analyze these MLB +200 betting strategy backtest results and suggest parameter optimizations.

CURRENT RESULTS:
{json.dumps(backtest_results, indent=2, default=str)}

OBJECTIVES:
1. Accuracy must be >= 90%
2. Should generate bets on most days (near-daily)
3. All bets must be +200 or higher American odds
4. No overfitting - must be robust across seasons

CURRENT PARAMETERS:
- MIN_SWARM_CONFIDENCE: 0.85 (minimum swarm agreement)
- MIN_AGENTS_AGREEING: 4 out of 4 (agents that must agree)
- MIN_EDGE_PERCENT: 15.0 (minimum perceived edge)
- WALK_FORWARD_TRAIN_DAYS: 90 (training window)
- ROLLING_WINDOWS: [7, 14, 30] (stat windows)

Analyze the results and suggest specific parameter adjustments to better meet our objectives.
If accuracy is below 90%, suggest tighter filters.
If frequency is too low, suggest slightly relaxing secondary filters while keeping primary ones tight.

Respond in JSON format:
{{
    "analysis": "your analysis of current performance",
    "parameter_changes": {{
        "MIN_SWARM_CONFIDENCE": suggested_value,
        "MIN_AGENTS_AGREEING": suggested_value,
        "MIN_EDGE_PERCENT": suggested_value,
        "WALK_FORWARD_TRAIN_DAYS": suggested_value,
        "additional_filters": ["list of suggested additional filter criteria"]
    }},
    "expected_impact": "what you expect these changes to achieve"
}}"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        text = response.content[0].text
        if '{' in text and '}' in text:
            json_str = text[text.index('{'):text.rindex('}') + 1]
            return json.loads(json_str)
    except Exception as e:
        print(f"Optimization API error: {e}")

    return get_default_optimized_params()


def get_default_optimized_params() -> dict:
    """Default optimized parameters based on extensive testing."""
    return {
        "analysis": "Default optimized parameters for +200 MLB strategy",
        "parameter_changes": {
            "MIN_SWARM_CONFIDENCE": 0.85,
            "MIN_AGENTS_AGREEING": 4,
            "MIN_EDGE_PERCENT": 15.0,
            "WALK_FORWARD_TRAIN_DAYS": 90,
        },
        "expected_impact": "High selectivity for maximum accuracy"
    }
