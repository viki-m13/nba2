#!/bin/bash
# =============================================================================
# MLB ULTRA BETTING ENGINE v1.0 — Daily Pick Generation
# =============================================================================
# Schedule this to run daily at 10:00 AM EST via cron:
#   0 10 * * * cd /path/to/nba2 && TZ=America/New_York bash mlb/scripts/daily-picks-mlb.sh >> logs/mlb-daily.log 2>&1
#
# This script:
# 1. Fetches latest box scores and odds data
# 2. Generates tonight's picks using the Ultra Engine
# 3. Updates webapp signals and stats
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "MLB ULTRA BETTING ENGINE v1.0 — Daily Run"
echo "Date: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

# Step 1: Generate tonight's live picks using Odds API (Node.js)
echo ""
echo "--- Step 1: Generate tonight's MLB picks from live odds ---"
cd "$PROJECT_DIR"
node mlb/scripts/seed-live-picks-mlb.js 2>&1

echo ""
echo "============================================================"
echo "MLB Daily run complete: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"
