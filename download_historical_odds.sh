#!/bin/bash
# Download historical FanDuel odds for all season game dates
# Uses The Odds API historical endpoints via curl

API_KEY="3879c3373a31421d8ef7d428b8758cd8"
BASE="https://api.the-odds-api.com/v4"
OUTPUT_DIR="/home/user/nba2/webapp/data/historical_odds_raw"
mkdir -p "$OUTPUT_DIR"

DATES=(
20241022 20241023 20241024 20241025 20241026 20241027 20241028 20241029 20241030 20241031
20241101 20241104 20241106 20241108 20241110 20241112 20241115 20241118 20241120 20241122
20241125 20241127 20241129 20241201 20241204 20241206 20241209 20241211 20241213 20241216
20241220 20241223 20241225 20241227 20241230
20250101 20250103 20250106 20250108 20250110 20250113 20250115 20250117 20250120 20250122
20250124 20250127 20250129 20250131 20250201 20250203 20250205 20250207 20250210 20250212
20250219 20250221 20250224
)

echo "Downloading historical FanDuel odds for ${#DATES[@]} dates..."

for DATE in "${DATES[@]}"; do
  ISO_DATE="${DATE:0:4}-${DATE:4:2}-${DATE:6:2}T12:00:00Z"
  OUTFILE="$OUTPUT_DIR/game_odds_${DATE}.json"

  # Skip if already downloaded
  if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
    echo "  $DATE: already downloaded, skipping"
    continue
  fi

  # Fetch game-level odds (h2h, spreads, totals)
  curl -s "$BASE/historical/sports/basketball_nba/odds?apiKey=$API_KEY&regions=us&markets=h2h,spreads,totals&oddsFormat=american&date=$ISO_DATE&bookmakers=fanduel" -o "$OUTFILE"

  # Check for valid response
  if [ ! -s "$OUTFILE" ]; then
    echo "  $DATE: empty response, removing"
    rm -f "$OUTFILE"
    continue
  fi

  # Check for error
  if grep -q '"message"' "$OUTFILE" 2>/dev/null; then
    echo "  $DATE: API error: $(cat $OUTFILE)"
    rm -f "$OUTFILE"
    continue
  fi

  SIZE=$(wc -c < "$OUTFILE")
  echo "  $DATE: downloaded ($SIZE bytes)"

  # Rate limit: 200ms between calls
  sleep 0.2
done

echo ""
echo "Game-level odds download complete."
echo "Now downloading player props per event..."

# Second pass: download player props for each event
for DATE in "${DATES[@]}"; do
  ISO_DATE="${DATE:0:4}-${DATE:4:2}-${DATE:6:2}T12:00:00Z"
  GAME_FILE="$OUTPUT_DIR/game_odds_${DATE}.json"

  if [ ! -f "$GAME_FILE" ]; then
    continue
  fi

  # Extract event IDs from the game odds file
  EVENT_IDS=$(node -e "
    const d = JSON.parse(require('fs').readFileSync('$GAME_FILE','utf8'));
    const events = d.data || [];
    events.forEach(e => console.log(e.id));
  " 2>/dev/null)

  if [ -z "$EVENT_IDS" ]; then
    continue
  fi

  for EVENT_ID in $EVENT_IDS; do
    PROPS_FILE="$OUTPUT_DIR/props_${DATE}_${EVENT_ID}.json"

    # Skip if already downloaded
    if [ -f "$PROPS_FILE" ] && [ -s "$PROPS_FILE" ]; then
      continue
    fi

    curl -s "$BASE/historical/sports/basketball_nba/events/$EVENT_ID/odds?apiKey=$API_KEY&regions=us&markets=player_points_alternate&oddsFormat=american&date=$ISO_DATE&bookmakers=fanduel" -o "$PROPS_FILE"

    # Remove if empty or error
    if [ ! -s "$PROPS_FILE" ]; then
      rm -f "$PROPS_FILE"
      continue
    fi

    if grep -q '"message"' "$PROPS_FILE" 2>/dev/null; then
      rm -f "$PROPS_FILE"
      continue
    fi

    sleep 0.2
  done

  PROP_COUNT=$(ls "$OUTPUT_DIR"/props_${DATE}_*.json 2>/dev/null | wc -l)
  echo "  $DATE: $PROP_COUNT event player props downloaded"
done

echo ""
echo "All downloads complete!"
echo "Files in $OUTPUT_DIR:"
ls "$OUTPUT_DIR" | wc -l
