"""
Microfish MLB Moneyline Strategy Configuration
================================================
MiroFish MinMax Adversarial Architecture for moneyline strategy DEVELOPMENT.

Patent-Pending Innovation: Certainty Floor Discovery (CFD)
==========================================================
Instead of predicting WHO wins (impossible at 100%), we determine the optimal
LIMIT PRICE on prediction markets (Polymarket) where:
  1. Our limit orders get filled frequently enough to generate daily bets
  2. When filled, the bet wins 100% of the time

Core Insight: On Polymarket, you buy shares at a price (e.g. $0.76 = 76%)
and receive $1 if correct. The strategy finds the MINIMUM price threshold
where outcomes become deterministic — the "Certainty Floor."

Architecture:
  - MiroFish MinMax agents analyze historical data to discover the Certainty Floor
  - MAX agents find the lowest price that still wins 100%
  - MIN agents attack by finding counterexamples (games lost above that price)
  - Only thresholds surviving adversarial attack become rules
  - The strategy runs as pure deterministic rules — no AI at runtime

Key Innovation: We DON'T just pick heavy favorites. We find the LIMIT PRICE
at which ANY game becomes a certainty, including games that start close.
A 55% pre-game favorite can reach 90%+ during the game — we set limit
orders that fill when the market reaches our Certainty Floor.

Claude AI MinMax agents are used ONLY during strategy development.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# API Keys
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY', '')

# Claude Model (used ONLY in MiroFish MinMax development pipeline)
CLAUDE_MODEL = "claude-opus-4-6"

# ==========================================================================
# CERTAINTY FLOOR DISCOVERY (CFD) PARAMETERS
# ==========================================================================

# Probability thresholds to test for the Certainty Floor
# The CFD engine finds the minimum % where win rate = 100%
CFD_PROBABILITY_GRID = list(range(50, 100))  # Test 50% through 99%

# Context-dependent thresholds (different situations may have different floors)
CFD_CONTEXTS = [
    'pre_game',             # Before first pitch
    'early_inning_lead',    # Leading in innings 1-3
    'mid_inning_lead',      # Leading in innings 4-6
    'late_inning_lead',     # Leading in innings 7-9
    'large_lead',           # Leading by 3+ runs
    'pitching_dominant',    # Starter has low hits/walks through 4+
]

# Minimum sample size to trust a threshold
MIN_SAMPLE_SIZE = 30        # Need 30+ fills at a price to validate 100%
MIN_FILL_RATE = 0.15        # At least 15% of games must fill at this price

# Maximum bets per day
MAX_BETS_PER_DAY = 5        # Can be higher since we're highly selective on price

# ==========================================================================
# ANTI-LEAKAGE / ANTI-OVERFITTING
# ==========================================================================

WALK_FORWARD_TRAIN_DAYS = 60
WALK_FORWARD_STEP_DAYS = 7
PURGE_DAYS = 1

FEATURE_LAG_DAYS = 1
ROLLING_WINDOWS = [3, 7, 14, 30]

MIN_SEASONS_CONSISTENT = 3
REGIME_SPLIT_YEAR = 2023

# ==========================================================================
# DATA PATHS
# ==========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
ML_DATA_DIR = os.path.join(DATA_DIR, 'moneyline')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'moneyline')
ML_RULES_FILE = os.path.join(BASE_DIR, 'ml_strategy_rules.json')
ML_RESEARCH_DIR = os.path.join(RESULTS_DIR, 'research')

# MLB seasons for backtesting
BACKTEST_SEASONS = [2021, 2022, 2023, 2024]

# ==========================================================================
# MIROFISH MINMAX DEVELOPMENT PIPELINE
# ==========================================================================

MAX_DEV_ITERATIONS = 5
MIN_AGENTS_CONSENSUS = 4        # Out of 6 agents must agree (stricter)
ADVERSARIAL_ROUNDS = 3          # MAX/MIN debate rounds per iteration

# ==========================================================================
# POLYMARKET INTEGRATION
# ==========================================================================

# Polymarket MLB game market structure:
# - Each game has YES/NO shares for each team
# - Price = market-implied probability (e.g., $0.76 = 76%)
# - Buy at limit price, receive $1 if team wins
# - Profit = $1 - buy_price (e.g., buy at $0.92, win = $0.08 profit)
# - Loss = buy_price (e.g., buy at $0.92, lose = -$0.92 loss)
#
# The strategy sets LIMIT ORDERS at the Certainty Floor price.
# Orders fill when the market reaches that probability level.
# At the Certainty Floor, filled orders win 100% of the time.

POLYMARKET_FEE_RATE = 0.02      # ~2% trading fee on Polymarket
MIN_PROFIT_PER_SHARE = 0.03     # Minimum profit after fees ($0.03/share)
