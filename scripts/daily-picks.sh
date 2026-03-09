#!/bin/bash
# =============================================================================
# ULTRA BETTING ENGINE v1.0 — Daily Pick Generation
# =============================================================================
# Schedule this to run daily at 9:00 AM EST via cron:
#   0 9 * * * cd /path/to/nba2 && TZ=America/New_York bash scripts/daily-picks.sh >> logs/daily.log 2>&1
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

# Step 1: Resolve pending picks and export updated backtest signals (Python)
echo ""
echo "--- Step 1: Resolve pending picks + export backtest ---"
cd "$PROJECT_DIR"
python3 src/generate_nightly_picks.py 2>&1

# Step 2: Generate tonight's live picks using Odds API (Node.js)
echo ""
echo "--- Step 2: Generate tonight's picks from live odds ---"
node scripts/seed-live-picks.js 2>&1

echo ""
echo "============================================================"
echo "Daily run complete: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"
