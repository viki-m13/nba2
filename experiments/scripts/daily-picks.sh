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
# What it does:
#   1. Runs the positive-odds backtest for NBA and MLB
#   2. Generates historical signals and today's picks
#   3. Saves results to experiments/webapp/data/
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
LOG_DIR="$EXPERIMENTS_DIR/logs"

mkdir -p "$LOG_DIR"

echo "========================================"
echo "POSITIVE ODDS DAILY PICKS"
echo "Date: $(date)"
echo "========================================"

# Run the Python pick generator
cd "$PROJECT_ROOT"
python3 experiments/scripts/generate_picks.py 2>&1

echo ""
echo "Positive odds picks generated at $(date)"
echo "Data saved to experiments/webapp/data/"
echo "========================================"
