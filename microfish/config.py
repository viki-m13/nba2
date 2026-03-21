"""
Microfish MLB +200 Strategy Configuration
==========================================
All configuration constants in one place. No magic numbers in code.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# API Keys
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY', '')

# Strategy Parameters
MIN_AMERICAN_ODDS = 200          # Only bet on +200 or higher
MIN_SWARM_CONFIDENCE = 0.50      # Minimum swarm agreement to place bet
MIN_AGENTS_AGREEING = 3          # Out of 4 agents must agree
MIN_EDGE_PERCENT = 10.0          # Minimum perceived edge over implied prob

# Backtesting
WALK_FORWARD_TRAIN_DAYS = 90     # Training window size
WALK_FORWARD_STEP_DAYS = 7       # Step forward by 1 week
PURGE_DAYS = 1                   # Gap between train and test to prevent leakage

# Model
FEATURE_LAG_DAYS = 1             # All features use data from at least 1 day prior
ROLLING_WINDOWS = [7, 14, 30]    # Rolling stat windows

# Claude Swarm
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_SWARM_RETRIES = 2

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# MLB seasons for backtesting
BACKTEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
