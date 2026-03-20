#!/bin/bash
# =============================================================================
# ULTRA BETTING ENGINE v1.0 — Daily Pick Generation
# =============================================================================
# Schedule this to run daily at 12:00 PM (noon) EST via cron:
#   0 12 * * * cd /path/to/nba2 && TZ=America/New_York bash scripts/daily-picks.sh >> logs/daily.log 2>&1
# NOTE: Noon EST = 5 PM UTC matches the timestamp used by The Odds API historical
# endpoint (T17:00:00Z) when collecting odds for backtesting, ensuring consistency
# between live picks and backtest results.
#
# This script:
# 1. Resolves any pending picks from previous days (via ESPN box scores)
# 2. Fetches live odds from The Odds API
# 3. Generates tonight's picks using the Ultra Engine
# 4. Updates webapp signals and stats
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "ULTRA BETTING ENGINE v1.0 — Daily Run"
echo "Date: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

# Step 1: Fetch latest box scores and odds from ESPN + The Odds API
echo ""
echo "--- Step 1: Fetch latest box scores + odds ---"
cd "$PROJECT_DIR"
node scripts/fetch-current-season.js 2>&1

# Step 2: Resolve pending picks and export updated backtest signals (Python)
echo ""
echo "--- Step 2: Resolve pending picks + export backtest ---"
python3 src/generate_nightly_picks.py 2>&1

# Step 3: Generate tonight's live picks using Odds API (Node.js)
echo ""
echo "--- Step 3: Generate tonight's picks from live odds ---"
node scripts/seed-live-picks.js 2>&1

echo ""
echo "============================================================"
echo "Daily run complete: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"
