# Microfish — MiroFish Multi-Agent Strategy Development System

MLB +200 underdog betting strategy developed by a Claude AI agent swarm,
executed as pure deterministic rules.

## Architecture

Adapted from the MiroFish multi-agent architecture:

```
┌─────────────────────────────────────────────────────────┐
│               DEVELOPMENT PHASE (Claude AI)              │
│                                                          │
│  Stage 1: Data Ontology Agent                           │
│    → Identifies predictive features in MLB data          │
│                                                          │
│  Stage 2: Pattern Discovery Agent                       │
│    → Finds high-accuracy patterns in underdog outcomes   │
│                                                          │
│  Stage 3: Strategy Architect Agent                      │
│    → Designs deterministic rule structures               │
│                                                          │
│  Stage 4: Validation Agent                              │
│    → Stress-tests rules for overfitting/bias            │
│                                                          │
│  Stage 5: Synthesis Director                            │
│    → Combines all agents into final rule set            │
│                                                          │
│  Output: strategy_rules.json                            │
├─────────────────────────────────────────────────────────┤
│               RUNTIME PHASE (No AI)                      │
│                                                          │
│  Strategy Engine reads strategy_rules.json               │
│    → Evaluates games against deterministic rules         │
│    → Outputs BET/PASS recommendations                    │
│    → Fully backtestable, no API calls                   │
└─────────────────────────────────────────────────────────┘
```

## Usage

```bash
# 1. Develop strategy rules (uses Claude Opus agents)
python run.py develop

# 2. Backtest rules against historical data (no AI)
python run.py backtest

# 3. Generate today's picks (no AI)
python run.py picks

# 4. Show current strategy rules
python run.py show

# 5. Full pipeline (develop + backtest)
python run.py full
```

## Key Design Principles

- **No ML** — Pure rules-based strategy engine
- **No AI at runtime** — Claude AI used only during development
- **No synthetic data** — Real odds from The Odds API
- **No leakage** — All features lagged by 1+ days
- **Walk-forward validation** — Never look ahead in backtesting
- **MiroFish architecture** — Multi-agent collaborative analysis

## Proprietary Innovation

The patentable innovation is using a multi-agent AI swarm as a strategy
DEVELOPMENT tool rather than for runtime prediction. The agents don't
make bets — they design the bet-selection rules through collaborative
analysis, cross-validation debate, and iterative refinement.
