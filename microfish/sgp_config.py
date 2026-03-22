"""
MiroFish Parlay Discovery Engine — Configuration
===================================================
High-odds player prop parlay discovery using MiniMax M2.7 multi-agent swarm.

@sgp_vick Strategy Patterns (studied from actual bet slips):

PATTERN 1 — "First Basket Bombs" (3 legs, +50,000-60,000)
  3 first-basket scorers across games, each +500-900
  $10 → $6,000

PATTERN 2 — "Milestone Stacks" (3-5 legs, +3,000-10,000)
  Cross-game player milestones: 20+ pts, double-double, 8+ ast
  Each leg +150-460

PATTERN 3 — "Floor Finder SGP" (6+ legs, +1,000-2,000)
  Single-game SGP picking EASY milestones for role players
  10+ points for guys who avg 14+, 15+ for guys who avg 22+
  Uses ALTERNATE LINES below the player's floor

PATTERN 4 — "SGP+" (3-12 legs, +2,000-25,000)
  Combines SGPs from 2-3 games into one slip
  e.g. 4-leg hockey SGP (+793) × 8-leg NBA SGP (+244) = +2,981

PATTERN 5 — "Alt Line Stacking" (8-12 legs, +2,000-5,000)
  Uses ALT (alternate) lower lines across many players
  ALT PTS+REB, ALT POINTS, ALT PTS+AST, ALT PTS+REB+AST
  Each leg is high-confidence but combined odds are massive

KEY INSIGHT: The edge is in ALTERNATE LINES — books set standard lines
at 50/50, but alt lines below a player's floor have actual 70-85% hit
rates while still contributing + odds to the parlay multiplier.

MLB ADAPTATION:
  - First HR in game → First Basket equivalent
  - Pitcher 5+ Ks (alt line) → Easy milestone for aces
  - Batter 1+ hits (alt line) → Floor finder for .280+ hitters
  - Batter 2+ total bases → Alt line milestone
  - Team 4+ runs → Game environment leg
  - Batter 1+ RBI → Floor finder for cleanup hitters

MiniMax M2.7 agents discover which MLB player/prop combinations
systematically hit together. Output is deterministic rules. NO AI at runtime.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# MiniMax M2.7 API
MINIMAX_API_KEY = os.getenv(
    'MINIMAX_API_KEY',
    'sk-api-eYkJhLY8htQ1SPFFxGp94_0wh8rRiEGFDMYBVxR1USzpBL4C63vJHjxeHFAkoUUwhHyTp2dxcTktKAKAZv5_0uJPVRAts77fZ3B9kGgz-f7GcKahorH-ew0'
)
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODEL = "MiniMax-M2.7"

# The Odds API
THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY', '')

# ============================================================================
# @SGP_VICK PARLAY ARCHETYPES (adapted for MLB)
# ============================================================================

PARLAY_ARCHETYPES = {
    # PATTERN 1: "First Event Bombs" — equivalent of First Basket
    'first_event_bomb': {
        'min_legs': 3,
        'max_legs': 3,
        'target_odds_min': 5000,
        'target_odds_max': 60000,
        'leg_odds_min': 400,
        'leg_odds_max': 1500,
        'description': 'First HR / First Hit across games',
        'mlb_markets': ['batter_home_runs', 'batter_first_home_run'],
    },

    # PATTERN 2: "Milestone Stacks" — 20+ pts equivalent
    'milestone_stack': {
        'min_legs': 3,
        'max_legs': 5,
        'target_odds_min': 3000,
        'target_odds_max': 15000,
        'leg_odds_min': 150,
        'leg_odds_max': 500,
        'description': 'Player milestones across games (3+ hits, 7+ Ks, 3+ RBI)',
        'mlb_markets': ['batter_hits', 'pitcher_strikeouts', 'batter_rbis',
                        'batter_total_bases'],
    },

    # PATTERN 3: "Floor Finder SGP" — easy milestones in one game
    'floor_finder_sgp': {
        'min_legs': 4,
        'max_legs': 8,
        'target_odds_min': 800,
        'target_odds_max': 3000,
        'leg_odds_min': -200,
        'leg_odds_max': 200,
        'description': 'Alt lines below player floors in single game',
        'mlb_markets': ['batter_hits', 'batter_total_bases', 'pitcher_strikeouts',
                        'pitcher_outs', 'batter_runs_scored'],
    },

    # PATTERN 4: "SGP+" — combine SGPs across 2-3 games
    'sgp_plus': {
        'min_legs': 6,
        'max_legs': 12,
        'target_odds_min': 2000,
        'target_odds_max': 25000,
        'leg_odds_min': -250,
        'leg_odds_max': 500,
        'description': 'Multiple SGPs combined across games',
        'mlb_markets': ['batter_hits', 'batter_total_bases', 'pitcher_strikeouts',
                        'batter_rbis', 'batter_runs_scored', 'batter_home_runs',
                        'totals', 'h2h'],
    },

    # PATTERN 5: "Alt Line Stacking" — many alt lines
    'alt_line_stack': {
        'min_legs': 6,
        'max_legs': 12,
        'target_odds_min': 1500,
        'target_odds_max': 8000,
        'leg_odds_min': -180,
        'leg_odds_max': 250,
        'description': 'Alt lines below floors for many players',
        'mlb_markets': ['batter_hits', 'batter_total_bases', 'pitcher_strikeouts',
                        'batter_rbis', 'pitcher_outs', 'batter_runs_scored',
                        'batter_walks', 'batter_stolen_bases'],
    },
}

STAKE_SIZE = 10                    # $10 per parlay
MAX_PARLAYS_PER_DAY = 5            # Max parlay slips per day

# SGP Discovery Parameters (used by sgp_discovery.py)
SGP_TARGET_ODDS = 900              # Target combined parlay odds (+900)
SGP_MIN_LEGS = 3                   # Minimum legs per parlay
SGP_MAX_LEGS = 8                   # Maximum legs per parlay
SGP_MIN_LEG_CONFIDENCE = 0.65     # Min implied probability per leg

# ============================================================================
# MLB PROP MARKETS
# ============================================================================

# All prop markets (used by sgp_data.py for fetching)
PROP_MARKETS = [
    'batter_hits', 'batter_total_bases', 'batter_rbis',
    'batter_runs_scored', 'batter_home_runs', 'batter_singles',
    'batter_doubles', 'batter_walks', 'batter_stolen_bases',
    'batter_strikeouts', 'pitcher_strikeouts', 'pitcher_hits_allowed',
    'pitcher_walks', 'pitcher_outs',
]

# High-odds "first" markets (Pattern 1)
FIRST_MARKETS = [
    'batter_home_runs',            # To hit a HR (O/U 0.5)
    'batter_first_home_run',       # First HR of game
]

# Milestone/performance markets (Patterns 2-5)
MILESTONE_MARKETS = [
    'batter_hits',                 # Hits O/U
    'batter_total_bases',          # Total bases O/U
    'batter_rbis',                 # RBIs O/U
    'batter_runs_scored',          # Runs scored O/U
    'batter_home_runs',            # HRs O/U
    'batter_singles',              # Singles O/U
    'batter_doubles',              # Doubles O/U
    'batter_walks',                # Walks O/U
    'batter_stolen_bases',         # SBs O/U
    'batter_strikeouts',           # Batter Ks O/U
    'pitcher_strikeouts',          # Pitcher Ks O/U
    'pitcher_hits_allowed',        # Hits allowed O/U
    'pitcher_walks',               # BB allowed O/U
    'pitcher_outs',                # Outs recorded O/U
]

# Game-level markets
GAME_MARKETS = [
    'h2h',                         # Moneyline
    'totals',                      # Over/under runs
    'spreads',                     # Run line
]

# ============================================================================
# ALT LINE STRATEGY PARAMETERS
# ============================================================================

# The core edge: alt lines set BELOW a player's statistical floor
# Example: .300 hitter gets 1+ hits at +100, but actually hits 70-80% of games
ALT_LINE_MIN_EDGE = 0.05          # Min 5% edge over implied prob
ALT_LINE_MIN_SAMPLE = 20          # Min games to establish floor
ALT_LINE_FLOOR_PERCENTILE = 25    # Use 25th percentile as "floor"

# ============================================================================
# BACKTESTING & VALIDATION
# ============================================================================

BACKTEST_SEASONS = [2023, 2024, 2025]
WALK_FORWARD_TRAIN_DAYS = 30
WALK_FORWARD_STEP_DAYS = 7
PURGE_DAYS = 1
FEATURE_LAG_DAYS = 1
ROLLING_WINDOWS = [7, 14, 30]

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SGP_DATA_DIR = os.path.join(BASE_DIR, 'data', 'sgp')
SGP_CACHE_DIR = os.path.join(SGP_DATA_DIR, 'cache')
SGP_RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'sgp')
SGP_RULES_FILE = os.path.join(BASE_DIR, 'sgp_strategy_rules.json')

# MiroFish Pipeline
MAX_DEV_ITERATIONS = 3
MIN_AGENTS_CONSENSUS = 3
