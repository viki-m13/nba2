#!/usr/bin/env bash
#
# Positive Odds Models — Daily Pick Generation
# =============================================
# Separate cron for the positive-odds HECE models.
# Does NOT interfere with the main daily-picks.sh cron.
#
# Crontab example (runs at 12:30 PM EST daily, 30 min after main cron):
#   30 12 * * * cd /path/to/nba2 && TZ=America/New_York bash experiments/scripts/daily-picks.sh >> experiments/logs/daily.log 2>&1
#
# What it does (mirrors original cron flow):
#   1. Fetches latest box scores (reuses main fetch scripts)
#   2. Regenerates backtest history from all historical data
#   3. Resolves pending live picks via ESPN + fetches live odds + generates tonight's picks (NBA)
#   4. Same for MLB
#   5. Saves results to webapp/positive-odds/data/
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
LOG_DIR="$EXPERIMENTS_DIR/logs"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "POSITIVE ODDS MODELS — Daily Run"
echo "Date: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

cd "$PROJECT_ROOT"

# Step 1: Fetch latest box scores (reuse main fetch scripts)
# These update the shared data files that both original and experiment models use
echo ""
echo "--- Step 1: Fetch latest box scores ---"
node scripts/fetch-current-season.js 2>&1 || echo "  (NBA fetch completed with warnings)"
node mlb/scripts/fetch-mlb-season.js 2>&1 || echo "  (MLB fetch completed with warnings)"

# Step 2: Regenerate backtest history from all historical data
# This updates nba_signals.json and mlb_signals.json with the full backtest history
echo ""
echo "--- Step 2: Regenerate backtest history ---"
python3 experiments/scripts/generate_picks.py 2>&1

# Step 3: Generate NBA live picks (resolve pending + fetch live odds + generate tonight's)
echo ""
echo "--- Step 3: NBA live picks ---"
python3 experiments/scripts/seed_live_picks.py 2>&1

# Step 4: Generate MLB live picks (resolve pending + fetch live odds + generate tonight's)
echo ""
echo "--- Step 4: MLB live picks ---"
python3 experiments/scripts/seed_live_picks_mlb.py 2>&1

echo ""
echo "============================================================"
echo "Positive odds daily run complete: $(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Data saved to webapp/positive-odds/data/"
echo "============================================================"
