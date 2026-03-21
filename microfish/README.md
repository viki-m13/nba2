# Microfish - MLB +200 Underdog Strategy

Multi-agent swarm intelligence system for identifying profitable MLB +200 underdog bets.
Inspired by [MiroFish](https://github.com/666ghj/MiroFish/) multi-agent architecture.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   MICROFISH PIPELINE                 │
├─────────────────────────────────────────────────────┤
│  1. Data Pipeline (data_pipeline.py)                │
│     - Real odds from The Odds API                   │
│     - Strict temporal features (all lagged)         │
│     - No leakage, no forward bias                   │
├─────────────────────────────────────────────────────┤
│  2. ML Ensemble (prediction_model.py)               │
│     - XGBoost + LightGBM + Logistic Regression      │
│     - Walk-forward validation with purge gap        │
│     - Conservative hyperparameters (anti-overfit)   │
├─────────────────────────────────────────────────────┤
│  3. Claude Swarm Model (swarm_model.py)             │
│     - 4 Specialist Agents + 1 Synthesis Agent       │
│     - Statistical, Matchup, Market, Momentum views  │
│     - Agents vote; synthesis requires consensus     │
├─────────────────────────────────────────────────────┤
│  4. Combined Filter                                 │
│     - ML + Swarm must BOTH agree                    │
│     - Only +200 or higher odds                      │
│     - Regime shift detection for edge               │
└─────────────────────────────────────────────────────┘
```

## Swarm Agents (Claude API)

| Agent | Focus | Key Signals |
|-------|-------|-------------|
| Statistical Analyst | Team performance metrics | Win%, trends, splits |
| Matchup Specialist | H2H context | Underdog win rate, home/away |
| Market Analyst | Line value | Odds mispricing, implied prob gap |
| Momentum Tracker | Streaks & form | Hot/cold streaks, fatigue |
| **Synthesis Agent** | Final decision | Requires multi-agent consensus |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full backtest
python run.py backtest

# Generate today's picks (uses Claude API)
python run.py picks

# Generate picks without API (rule-based)
python run.py picks --no-api

# Run optimizer + backtest + picks
python run.py full
```

## Configuration

Copy `.env.example` to `.env` and set:
- `ANTHROPIC_API_KEY` - Claude API key for swarm model
- `THE_ODDS_API_KEY` - The Odds API key for real odds data

## Anti-Overfitting Measures

1. **Walk-forward validation** - Train/test split advances through time
2. **Purge gap** - 1-day gap between train and test periods
3. **Feature lag** - All features use data from at least 1 day prior
4. **Conservative models** - Low depth, high regularization
5. **Dual filter** - Both ML and Swarm must independently agree
6. **Cross-season validation** - Performance tracked per season

## Key Metrics

- **Win Rate**: Raw percentage of winning bets
- **ROI**: Return on investment per unit wagered
- **EV Accuracy**: Percentage of bets with positive expected value
- **CLV Accuracy**: Percentage of bets beating closing line value
- **Selectivity**: How selective the strategy is (lower = more selective)
