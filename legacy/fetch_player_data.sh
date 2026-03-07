#!/bin/bash
# Fetch player box scores for all games in the season data
# Output: webapp/data/player_boxscores.json

SEASON_FILE="webapp/data/espn_full_season_2025.json"
OUTPUT_FILE="webapp/data/player_boxscores.json"

# Extract all event IDs
EVENT_IDS=$(python3 -c "
import json
with open('$SEASON_FILE') as f:
    data = json.load(f)
for g in data:
    print(g['event_id'])
")

echo "[]" > "$OUTPUT_FILE"

TOTAL=$(echo "$EVENT_IDS" | wc -l)
COUNT=0
RESULTS="["

for EVENT_ID in $EVENT_IDS; do
    COUNT=$((COUNT + 1))

    # Fetch game summary
    RESP=$(curl -s --connect-timeout 10 "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=$EVENT_ID" 2>/dev/null)

    if [ -z "$RESP" ]; then
        echo "[$COUNT/$TOTAL] Failed: $EVENT_ID"
        continue
    fi

    # Extract player data using python
    GAME_DATA=$(echo "$RESP" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    players = data.get('boxscore', {}).get('players', [])
    result = []
    for team in players:
        abbr = team['team']['abbreviation']
        for cat in team.get('statistics', []):
            labels = cat.get('labels', [])
            for p in cat.get('athletes', []):
                name = p.get('athlete', {}).get('displayName', '?')
                pid = p.get('athlete', {}).get('id', '?')
                stats = p.get('stats', [])
                if len(stats) >= 14 and stats[0] != '0':  # has minutes
                    result.append({
                        'team': abbr,
                        'name': name,
                        'pid': pid,
                        'min': stats[0],
                        'pts': int(stats[1]),
                        'fg': stats[2],
                        'three': stats[3],
                        'ft': stats[4],
                        'reb': int(stats[5]),
                        'ast': int(stats[6]),
                        'to': int(stats[7]),
                        'stl': int(stats[8]),
                        'blk': int(stats[9]),
                    })
    print(json.dumps(result))
except:
    print('[]')
" 2>/dev/null)

    if [ "$GAME_DATA" != "[]" ] && [ -n "$GAME_DATA" ]; then
        if [ "$COUNT" -gt 1 ]; then
            RESULTS="${RESULTS},"
        fi

        # Get game date
        DATE=$(python3 -c "
import json
with open('$SEASON_FILE') as f:
    data = json.load(f)
for g in data:
    if g['event_id'] == '$EVENT_ID':
        print(g['date'])
        break
")
        HOME=$(python3 -c "
import json
with open('$SEASON_FILE') as f:
    data = json.load(f)
for g in data:
    if g['event_id'] == '$EVENT_ID':
        print(g['home_team'])
        break
")
        AWAY=$(python3 -c "
import json
with open('$SEASON_FILE') as f:
    data = json.load(f)
for g in data:
    if g['event_id'] == '$EVENT_ID':
        print(g['away_team'])
        break
")

        RESULTS="${RESULTS}{\"event_id\":\"$EVENT_ID\",\"date\":\"$DATE\",\"home\":\"$HOME\",\"away\":\"$AWAY\",\"players\":$GAME_DATA}"
    fi

    if [ $((COUNT % 20)) -eq 0 ]; then
        echo "[$COUNT/$TOTAL] Fetched..."
    fi

    # Rate limit
    sleep 0.3
done

RESULTS="${RESULTS}]"
echo "$RESULTS" | python3 -m json.tool > "$OUTPUT_FILE" 2>/dev/null || echo "$RESULTS" > "$OUTPUT_FILE"

echo "Done! Fetched $COUNT games, saved to $OUTPUT_FILE"
