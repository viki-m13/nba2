"""
Microfish MLB +200 Strategy Configuration
==========================================
MiroFish Multi-Agent Architecture for strategy DEVELOPMENT.
The strategy itself runs as pure deterministic rules — no AI at runtime.

Claude AI agents are used ONLY during strategy development to discover,
analyze, and refine the rules. Once rules are finalized, they're saved
to strategy_rules.json and the strategy engine executes them without AI.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# API Keys
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY', '')

# Claude Model (used ONLY in MiroFish development pipeline)
CLAUDE_MODEL = "claude-opus-4-6"

# Strategy Parameters (applied by deterministic rules engine)
MIN_AMERICAN_ODDS = 200          # Only bet on +200 or higher underdogs

# Backtesting
WALK_FORWARD_TRAIN_DAYS = 60     # Training window for rule calibration
WALK_FORWARD_STEP_DAYS = 7       # Step forward by 1 week
PURGE_DAYS = 1                   # Gap between train and test to prevent leakage

# Feature Engineering
FEATURE_LAG_DAYS = 1             # All features use data from at least 1 day prior
ROLLING_WINDOWS = [7, 14, 30]    # Rolling stat windows

# Data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RULES_FILE = os.path.join(BASE_DIR, 'strategy_rules.json')

# MLB seasons for backtesting
BACKTEST_SEASONS = [2021, 2022, 2023, 2024]

# MiroFish Development Pipeline
MAX_DEV_ITERATIONS = 5           # Max refinement cycles
MIN_AGENTS_CONSENSUS = 3         # Out of 4 agents must agree on a rule
