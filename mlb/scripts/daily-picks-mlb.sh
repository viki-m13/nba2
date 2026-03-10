#!/bin/bash
# =============================================================================
# MLB ULTRA BETTING ENGINE v3.0 — Daily Pick Generation
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
echo "MLB ULTRA BETTING ENGINE v3.0 — Daily Run"
echo "Date: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

# Step 1: Copy latest v3 config to webapp data dir
echo ""
echo "--- Step 1: Sync v3 config to webapp ---"
cd "$PROJECT_DIR"
if [ -f "mlb/output/mlb_ultra_engine_v3_config.json" ]; then
  cp mlb/output/mlb_ultra_engine_v3_config.json mlb/webapp/data/mlb_ultra_engine_v3_config.json
  echo "Config synced."
fi

# Step 2: Generate tonight's live picks using Odds API (Node.js)
echo ""
echo "--- Step 2: Resolve results + generate tonight's MLB picks ---"
node mlb/scripts/seed-live-picks-mlb.js 2>&1

# Step 3: Sync data files to webapp serving directory
echo ""
echo "--- Step 3: Sync data to webapp directory ---"
if [ -d "$PROJECT_DIR/webapp/mlb/data" ]; then
  cp mlb/webapp/data/mlb_ultra_signals_v3.json webapp/mlb/data/ 2>/dev/null
  cp mlb/webapp/data/mlb_ultra_backtest_stats_v3.json webapp/mlb/data/ 2>/dev/null
  cp mlb/webapp/data/mlb_ultra_recommendations.json webapp/mlb/data/ 2>/dev/null
  cp mlb/webapp/data/mlb_ultra_engine_v3_config.json webapp/mlb/data/ 2>/dev/null
  echo "Data synced to webapp."
fi

echo ""
echo "============================================================"
echo "MLB Daily run complete: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"
