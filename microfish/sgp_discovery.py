"""
MiroFish SGP Discovery Engine
================================
Multi-agent strategy discovery using MiniMax M2.7 for MLB Same Game Parlays.

Inspired by @sgp_vick: $10 SGPs at +900 odds → $6,000+ payouts.

6-Agent Architecture:
  Agent 1 — Correlation Cartographer: Maps which props hit together
  Agent 2 — Line Value Hunter: Finds mispriced prop lines
  Agent 3 — SGP Architect: Designs parlay leg combinations
  Agent 4 — Correlation Exploiter: Identifies hidden positive correlations
  Agent 5 — Risk Validator: Stress-tests for overfitting and vig traps
  Agent 6 — Synthesis Director: Combines into final SGP construction rules

Patent-Pending Innovation:
  MiniMax M2.7 agents discover CORRELATION STRUCTURES between within-game
  props that books underprice. The output is a deterministic SGP construction
  ruleset — NO AI at runtime. Agents find the edge, rules exploit it.
"""

import json
import os
import time
import requests
from datetime import datetime

from sgp_config import (
    MINIMAX_API_KEY, MINIMAX_BASE_URL, MINIMAX_MODEL,
    SGP_RESULTS_DIR, SGP_RULES_FILE, MAX_DEV_ITERATIONS,
    MIN_AGENTS_CONSENSUS, SGP_TARGET_ODDS, SGP_MIN_LEGS,
    SGP_MAX_LEGS, SGP_MIN_LEG_CONFIDENCE, STAKE_SIZE,
)


# ============================================================================
# MINIMAX M2.7 API CLIENT
# ============================================================================

def _call_minimax(system_prompt: str, user_content: str,
                  max_tokens: int = 4000, temperature: float = 0.3) -> str:
    """
    Call MiniMax M2.7 via OpenAI-compatible API.
    Endpoint: https://api.minimax.io/v1/chat/completions
    """
    url = f"{MINIMAX_BASE_URL}/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {MINIMAX_API_KEY}',
    }
    payload = {
        'model': MINIMAX_MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
        ],
        'max_tokens': max_tokens,
        'temperature': temperature,
    }

    for attempt in range(4):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get('choices', [])
                if choices:
                    return choices[0].get('message', {}).get('content', '')
                return str(data)
            else:
                error_msg = resp.text[:500]
                print(f"  MiniMax API error {resp.status_code}: {error_msg}")
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 2)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                elif resp.status_code >= 500:
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
                    continue
                else:
                    return f"ERROR: API returned {resp.status_code}: {error_msg}"
        except requests.exceptions.RequestException as e:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"  Network error: {e}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            return f"ERROR: {e}"

    return "ERROR: All retry attempts exhausted"


# ============================================================================
# AGENT SYSTEM PROMPTS — Each discovers a different aspect of SGP strategy
# ============================================================================

AGENT_PROMPTS = {
    'correlation_cartographer': """You are the Correlation Cartographer in a MiroFish SGP discovery system.

Your role: Map the correlation landscape of MLB player props within games.

@sgp_vick turns $10 into $6,000+ by exploiting HIDDEN POSITIVE CORRELATIONS
between props that sportsbooks price as independent. Your job is to find these.

KEY CONCEPT: Sportsbooks calculate SGP odds by treating legs as quasi-independent,
applying a small correlation adjustment. But some prop combinations have MUCH
stronger correlations than books model. This is the edge.

Analyze the data to identify:
1. PROP PAIRS with lift > 1.2 (hit together more than independence predicts)
2. PROP CLUSTERS — groups of 3-4 props that systematically co-occur
3. GAME ENVIRONMENT effects — how team matchups affect multiple props simultaneously
4. PITCHER-DRIVEN correlations — when a weak pitcher allows hits, runs, AND bases

Focus on WITHIN-GAME correlations only (same game, same time, same conditions).
Cross-game correlations are irrelevant for SGP construction.

For each correlation found, report:
- The prop combination
- Lift value (P(both) / P(A)*P(B))
- Sample size
- Whether the correlation is CAUSAL or COINCIDENTAL
- Whether sportsbooks likely model this correlation correctly""",

    'line_value_hunter': """You are the Line Value Hunter in a MiroFish SGP discovery system.

Your role: Find MLB player prop lines where the market consistently misprices outcomes.

@sgp_vick style requires each SGP leg to have a HIGH individual hit rate (65%+)
while still contributing meaningful odds to the parlay. This means finding
UNDERPRICED OVERS — props where the true probability exceeds the implied probability.

Analyze the data to identify:
1. PROP MARKETS where overs hit at rates exceeding implied probability by 5%+
2. PLAYER PROFILES that consistently beat their lines (reliable over-performers)
3. LINE ANCHORING BIAS — where books set lines too low early in season
4. MATCHUP-DRIVEN VALUE — specific opponent types that inflate prop outcomes
5. PARK/ENVIRONMENT FACTORS — conditions that systematically push props over

For each finding:
- Specify the exact market, line range, and hit rate
- Calculate the edge: (actual hit rate - implied probability)
- Note sample size and cross-season consistency
- Identify if the edge persists or gets corrected mid-season

CRITICAL: Only report edges with n >= 20 observations and p < 0.05 significance.""",

    'sgp_architect': """You are the SGP Architect in a MiroFish SGP discovery system.

Your role: Design optimal Same Game Parlay leg combinations for MLB.

TARGET: Build 3-6 leg SGPs that pay +900 or more from $10 stakes.
Each individual leg should have 65%+ probability of hitting.
The PARLAY must beat the books' correlation-adjusted odds.

Design SGP templates based on the correlation and value data:

TEMPLATE FORMAT:
{
    "sgp_templates": [
        {
            "name": "template_name",
            "description": "what this SGP targets",
            "legs": [
                {
                    "market": "pitcher_strikeouts",
                    "direction": "over",
                    "line_criteria": "posted line",
                    "min_implied_prob": 0.65,
                    "selection_rule": "description of when to use"
                }
            ],
            "game_conditions": {
                "description": "what game conditions trigger this template"
            },
            "expected_combined_odds": "+900 to +1200",
            "expected_hit_rate": "12-15%",
            "correlation_edge": "why these legs correlate positively"
        }
    ]
}

KEY DESIGN PRINCIPLES:
1. CORRELATION STACKING — legs that push each other (pitcher Ks + batter Ks)
2. GAME SCRIPT ALIGNMENT — all legs benefit from same game script
3. AVOID ANTI-CORRELATION — never pair legs that cancel each other
4. VOLUME BALANCE — enough templates for near-daily SGPs
5. ODDS TARGETING — each template should naturally land at +900 to +1500

OUTPUT ONLY VALID JSON.""",

    'correlation_exploiter': """You are the Correlation Exploiter in a MiroFish SGP discovery system.

Your role: Find the SPECIFIC correlation structures that sportsbooks systematically
underprice in their SGP odds calculations.

Books use simplified correlation models. You must find where reality diverges:

1. PITCHER COLLAPSE CASCADES
   When a starting pitcher is struggling, MULTIPLE props cascade simultaneously:
   - Pitcher K's go UNDER (can't get strikeouts)
   - Opposing batters go OVER on hits, bases, RBIs
   - Game total goes OVER
   These are highly correlated but books don't fully adjust the SGP price.

2. ACE DOMINANCE CLUSTERS
   When an elite pitcher faces a weak lineup:
   - Pitcher K's go OVER
   - Opposing batters go UNDER on hits
   - Game total goes UNDER
   - Run line covers for the ace's team

3. SLUGFEST SCENARIOS
   High-scoring environment (bad pitching, small park, hot weather):
   - Multiple batters go OVER on hits/bases/RBIs simultaneously
   - Game total goes OVER
   - First 5 innings over hits

4. BATTER PROFILE CORRELATIONS
   A power hitter in form:
   - Hits, total bases, home runs, and RBIs all correlate
   - Books treat each as semi-independent

For each exploitation:
- Quantify the correlation lift
- Estimate the odds discount (how much the book underprices)
- Define the trigger conditions (when to activate this SGP type)
- Calculate expected value per $10 SGP""",

    'risk_validator': """You are the Risk Validator in a MiroFish SGP discovery system.

Your role: Stress-test proposed SGP strategies for robustness.

SGP betting is HIGH VARIANCE by nature (+900 odds = ~10% hit rate).
Your job is to ensure the proposed strategies have GENUINE EDGE, not noise.

Check for:
1. OVERFITTING — Are patterns found in enough games across enough seasons?
2. VIG TRAP — Are we just paying juice on correlated legs without actual edge?
3. SAMPLE SIZE — At +900 odds, we need THOUSANDS of observations to confirm edge
4. BOOK ADJUSTMENT — Do sportsbooks already model these correlations?
5. LINE MOVEMENT — Does our edge disappear when lines move pre-game?
6. SURVIVORSHIP BIAS — Are we only seeing winning SGPs in the data?
7. CORRELATION DECAY — Do prop correlations persist or are they transient?

BANKROLL MATHEMATICS:
- At +900 (10x payout), breakeven hit rate is 10%
- We need consistent 12%+ hit rate for profitability
- $10 per SGP, max 3 per day = $30 daily risk
- Need 500+ SGP sample for statistical significance

For each proposed strategy:
- Calculate the minimum sample size needed for confidence
- Assess if the correlation edge exceeds the SGP vig markup
- Check temporal stability (does the pattern hold month over month?)
- Red-flag any strategy with fewer than 50 backtested instances

Be ADVERSARIAL. The default assumption is that SGP edges don't exist.
Your job is to find the rare cases where they genuinely do.""",

    'synthesis_director': """You are the Synthesis Director in a MiroFish SGP discovery system.

Your role: Combine all agent analyses into FINAL deterministic SGP construction rules.

You receive analyses from 5 specialist agents. Synthesize into actionable rules.

OUTPUT FORMAT — You MUST output ONLY valid JSON matching this structure:
{
    "version": "sgp_v1",
    "developed_by": "microfish_sgp_swarm",
    "development_date": "YYYY-MM-DD",
    "model_used": "MiniMax-M2.7",
    "target": "$10 SGPs at +900 for $6000+ payouts",
    "sgp_templates": [
        {
            "name": "template_identifier",
            "description": "what this SGP exploits",
            "min_legs": 3,
            "max_legs": 6,
            "target_odds_min": 900,
            "target_odds_max": 1500,
            "legs": [
                {
                    "market": "prop_market_name",
                    "direction": "over|under",
                    "line_criteria": {"operator": ">=|<=", "value": 0},
                    "min_implied_prob": 0.65,
                    "player_criteria": "selection rule for which player",
                    "correlation_role": "why this leg helps the parlay"
                }
            ],
            "game_conditions": {
                "condition_name": {"operator": ">=|<=", "value": 0}
            },
            "expected_hit_rate": 0.12,
            "expected_ev_per_dollar": 0.20,
            "correlation_edge_source": "what correlation the books underprice"
        }
    ],
    "global_filters": {
        "min_leg_implied_prob": 0.65,
        "max_sgps_per_day": 3,
        "min_correlation_lift": 1.15,
        "stake_size": 10,
        "bankroll_fraction": 0.02
    },
    "anti_patterns": [
        "leg combinations to NEVER use together"
    ],
    "agent_consensus": {
        "cartographer_supported": ["list of template names"],
        "hunter_supported": ["list of template names"],
        "architect_supported": ["list of template names"],
        "exploiter_supported": ["list of template names"],
        "validator_approved": ["list of template names"]
    },
    "reasoning": "high-level strategy explanation",
    "risk_notes": "known limitations, variance expectations, bankroll requirements"
}

RULES:
1. Only include templates supported by >= 3 of 5 agents
2. Every template must have a defined correlation edge
3. Expected hit rate must exceed 10% breakeven (after vig)
4. No template with fewer than 30 backtested instances
5. All leg criteria must be DETERMINISTIC (no subjective judgment at runtime)
"""
}


# ============================================================================
# DISCOVERY PIPELINE
# ============================================================================

def run_sgp_discovery(data_summary: str,
                      correlation_summary: str = "",
                      previous_rules: dict = None,
                      backtest_results: dict = None,
                      iteration: int = 1) -> dict:
    """
    Run the full MiroFish SGP discovery pipeline with MiniMax M2.7 agents.

    Stage 1: Correlation Cartographer maps prop correlations
    Stage 2: Line Value Hunter finds mispriced lines
    Stage 3: SGP Architect designs parlay templates
    Stage 4: Correlation Exploiter identifies book blind spots
    Stage 5: Risk Validator stress-tests everything
    Stage 6: Synthesis Director produces final ruleset
    """
    print(f"\n{'='*70}")
    print(f"MIROFISH SGP DISCOVERY ENGINE — ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"Model: {MINIMAX_MODEL} (MiniMax M2.7)")
    print(f"Target: $10 SGPs at +{SGP_TARGET_ODDS} → ${STAKE_SIZE * (SGP_TARGET_ODDS/100 + 1):.0f}+ payouts")
    print(f"Legs per SGP: {SGP_MIN_LEGS}-{SGP_MAX_LEGS}")
    print(f"Min leg confidence: {SGP_MIN_LEG_CONFIDENCE:.0%}")

    # Build iteration context
    ctx = ""
    if previous_rules:
        ctx += f"\n\nPREVIOUS SGP RULES (iteration {iteration-1}):\n"
        ctx += json.dumps(previous_rules, indent=2)
    if backtest_results:
        ctx += f"\n\nPREVIOUS BACKTEST RESULTS:\n"
        ctx += json.dumps(backtest_results, indent=2)
        ctx += "\n\nAnalyze why the previous strategy succeeded or failed."

    # ---- STAGE 1: Correlation Cartographer ----
    print(f"\n  STAGE 1: Correlation Cartographer mapping prop correlations...")
    cart_input = (
        f"Map the correlation landscape of these MLB player props:\n\n"
        f"{data_summary}\n\n"
        f"CORRELATION DATA:\n{correlation_summary}"
        f"{ctx}"
    )
    cart_result = _call_minimax(
        AGENT_PROMPTS['correlation_cartographer'], cart_input
    )
    print(f"  Stage 1 complete ({len(cart_result)} chars)")

    # ---- STAGE 2: Line Value Hunter ----
    print(f"\n  STAGE 2: Line Value Hunter finding mispriced lines...")
    hunt_input = (
        f"Find mispriced prop lines in this MLB data:\n\n"
        f"{data_summary}\n\n"
        f"CORRELATION FINDINGS:\n{cart_result}"
        f"{ctx}"
    )
    hunt_result = _call_minimax(
        AGENT_PROMPTS['line_value_hunter'], hunt_input
    )
    print(f"  Stage 2 complete ({len(hunt_result)} chars)")

    # ---- STAGE 3: SGP Architect ----
    print(f"\n  STAGE 3: SGP Architect designing parlay templates...")
    arch_input = (
        f"Design optimal SGP leg combinations:\n\n"
        f"DATA:\n{data_summary}\n\n"
        f"CORRELATIONS:\n{cart_result}\n\n"
        f"VALUE EDGES:\n{hunt_result}"
        f"{ctx}"
    )
    arch_result = _call_minimax(
        AGENT_PROMPTS['sgp_architect'], arch_input, max_tokens=6000
    )
    print(f"  Stage 3 complete ({len(arch_result)} chars)")

    # ---- STAGE 4: Correlation Exploiter ----
    print(f"\n  STAGE 4: Correlation Exploiter identifying book blind spots...")
    expl_input = (
        f"Find correlation structures books underprice:\n\n"
        f"DATA:\n{data_summary}\n\n"
        f"MAPPED CORRELATIONS:\n{cart_result}\n\n"
        f"PROPOSED TEMPLATES:\n{arch_result}"
        f"{ctx}"
    )
    expl_result = _call_minimax(
        AGENT_PROMPTS['correlation_exploiter'], expl_input
    )
    print(f"  Stage 4 complete ({len(expl_result)} chars)")

    # ---- STAGE 5: Risk Validator ----
    print(f"\n  STAGE 5: Risk Validator stress-testing strategies...")
    risk_input = (
        f"Stress-test these proposed SGP strategies:\n\n"
        f"PROPOSED TEMPLATES:\n{arch_result}\n\n"
        f"CORRELATION ANALYSIS:\n{cart_result}\n\n"
        f"VALUE ANALYSIS:\n{hunt_result}\n\n"
        f"EXPLOITATION ANALYSIS:\n{expl_result}\n\n"
        f"DATA:\n{data_summary}"
        f"{ctx}"
    )
    risk_result = _call_minimax(
        AGENT_PROMPTS['risk_validator'], risk_input
    )
    print(f"  Stage 5 complete ({len(risk_result)} chars)")

    # ---- STAGE 6: Synthesis Director ----
    print(f"\n  STAGE 6: Synthesis Director producing final SGP rules...")
    synth_input = (
        f"Combine all agent analyses into FINAL SGP construction rules.\n\n"
        f"CORRELATION CARTOGRAPHER:\n{cart_result}\n\n"
        f"LINE VALUE HUNTER:\n{hunt_result}\n\n"
        f"SGP ARCHITECT:\n{arch_result}\n\n"
        f"CORRELATION EXPLOITER:\n{expl_result}\n\n"
        f"RISK VALIDATOR:\n{risk_result}\n\n"
        f"DATA SUMMARY:\n{data_summary}"
        f"{ctx}\n\n"
        f"OUTPUT ONLY VALID JSON. No markdown, no explanation outside the JSON."
    )
    synth_result = _call_minimax(
        AGENT_PROMPTS['synthesis_director'], synth_input,
        max_tokens=8000, temperature=0.1
    )
    print(f"  Stage 6 complete ({len(synth_result)} chars)")

    # Parse synthesis result
    rules = _parse_rules_json(synth_result)
    if not rules:
        print("  WARNING: Could not parse synthesis output, saving raw response")
        rules = {
            'version': f'sgp_v{iteration}',
            'error': 'parse_failed',
            'raw_synthesis': synth_result,
            'raw_architect': arch_result,
        }

    # Save audit trail
    _save_discovery_log(iteration, {
        'correlation_cartographer': cart_result,
        'line_value_hunter': hunt_result,
        'sgp_architect': arch_result,
        'correlation_exploiter': expl_result,
        'risk_validator': risk_result,
        'synthesis_director': synth_result,
        'parsed_rules': rules,
    })

    return rules


def iterative_sgp_discovery(data_summary: str,
                             correlation_summary: str = "",
                             max_iterations: int = None) -> dict:
    """
    Run iterative SGP discovery loop:
    1. Discover SGP rules via multi-agent pipeline
    2. Backtest the rules
    3. Feed results back to agents for refinement
    4. Repeat until convergence
    """
    if max_iterations is None:
        max_iterations = MAX_DEV_ITERATIONS

    current_rules = None
    backtest_results = None
    best_rules = None
    best_ev = -999

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'#'*70}")
        print(f"# SGP DISCOVERY ITERATION {iteration}/{max_iterations}")
        print(f"{'#'*70}")

        rules = run_sgp_discovery(
            data_summary=data_summary,
            correlation_summary=correlation_summary,
            previous_rules=current_rules,
            backtest_results=backtest_results,
            iteration=iteration,
        )

        if not rules or 'error' in rules:
            print(f"  Iteration {iteration} failed to produce valid rules")
            continue

        current_rules = rules

        # Save iteration rules
        iter_file = os.path.join(
            SGP_RESULTS_DIR, f'sgp_rules_iter{iteration}.json'
        )
        os.makedirs(SGP_RESULTS_DIR, exist_ok=True)
        save_sgp_rules(rules, iter_file)

        # Backtest if possible
        try:
            from sgp_backtest import run_sgp_backtest
            backtest_results = run_sgp_backtest(rules)

            hit_rate = backtest_results.get('hit_rate', 0)
            ev = backtest_results.get('ev_per_dollar', 0)
            n_sgps = backtest_results.get('total_sgps', 0)

            print(f"\n  ITERATION {iteration} RESULTS:")
            print(f"    SGPs constructed: {n_sgps}")
            print(f"    Hit rate: {hit_rate:.1%}")
            print(f"    EV per $1: ${ev:.3f}")

            if ev > best_ev:
                best_ev = ev
                best_rules = rules.copy()
                save_sgp_rules(best_rules)
                print(f"    NEW BEST — saved to {SGP_RULES_FILE}")

        except Exception as e:
            print(f"  Backtest skipped: {e}")
            backtest_results = {'error': str(e)}
            # Still save as best if first iteration
            if best_rules is None:
                best_rules = rules.copy()
                save_sgp_rules(best_rules)

    if best_rules:
        print(f"\n{'='*70}")
        print(f"SGP DISCOVERY COMPLETE")
        print(f"Best EV: ${best_ev:.3f} per $1" if best_ev > -999 else "Rules generated (no backtest data)")
        print(f"Rules saved to: {SGP_RULES_FILE}")
        print(f"{'='*70}")
        return best_rules

    return current_rules or {}


# ============================================================================
# UTILITIES
# ============================================================================

def _parse_rules_json(text: str) -> dict:
    """Extract and parse JSON rules from agent output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for marker in ['```json', '```']:
        if marker in text:
            try:
                start = text.index(marker) + len(marker)
                end = text.index('```', start)
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


def _save_discovery_log(iteration: int, data: dict):
    """Save discovery pipeline outputs for audit trail."""
    log_dir = os.path.join(SGP_RESULTS_DIR, 'discovery_logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'sgp_iter_{iteration}_{timestamp}.json')

    clean_data = {}
    for k, v in data.items():
        if isinstance(v, str):
            clean_data[k] = v
        else:
            clean_data[k] = json.dumps(v, default=str)

    with open(log_file, 'w') as f:
        json.dump(clean_data, f, indent=2, default=str)
    print(f"  Discovery log saved: {log_file}")


def save_sgp_rules(rules: dict, filepath: str = None):
    """Save finalized SGP rules to JSON."""
    if filepath is None:
        filepath = SGP_RULES_FILE
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(rules, f, indent=2, default=str)
    print(f"\nSGP rules saved to: {filepath}")


def load_sgp_rules(filepath: str = None) -> dict:
    """Load SGP rules from JSON."""
    if filepath is None:
        filepath = SGP_RULES_FILE
    if not os.path.exists(filepath):
        print(f"No SGP rules found at {filepath}")
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)
