#!/usr/bin/env python3
"""Fetch player box scores from ESPN for all season games."""
import json
import subprocess
import time
import sys

SEASON_FILE = "webapp/data/espn_full_season_2025.json"
OUTPUT_FILE = "webapp/data/player_boxscores.json"

with open(SEASON_FILE) as f:
    games = json.load(f)

print(f"Total games to fetch: {len(games)}")

results = []
failed = 0

for i, game in enumerate(games):
    event_id = game["event_id"]
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"

    try:
        resp = subprocess.run(
            ["curl", "-s", "--connect-timeout", "10", url],
            capture_output=True, text=True, timeout=15
        )
        data = json.loads(resp.stdout)

        players_data = []
        players = data.get("boxscore", {}).get("players", [])
        for team in players:
            abbr = team["team"]["abbreviation"]
            for cat in team.get("statistics", []):
                for p in cat.get("athletes", []):
                    name = p.get("athlete", {}).get("displayName", "?")
                    pid = p.get("athlete", {}).get("id", "?")
                    stats = p.get("stats", [])
                    if len(stats) >= 14 and stats[0] != "0":
                        players_data.append({
                            "team": abbr,
                            "name": name,
                            "pid": pid,
                            "min": stats[0],
                            "pts": int(stats[1]),
                            "fg": stats[2],
                            "three": stats[3],
                            "ft": stats[4],
                            "reb": int(stats[5]),
                            "ast": int(stats[6]),
                            "to": int(stats[7]),
                            "stl": int(stats[8]),
                            "blk": int(stats[9]),
                        })

        if players_data:
            results.append({
                "event_id": event_id,
                "date": game["date"],
                "home": game["home_team"],
                "away": game["away_team"],
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                "players": players_data,
            })
    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"  Failed {event_id}: {e}", file=sys.stderr)

    if (i + 1) % 25 == 0:
        print(f"[{i+1}/{len(games)}] Fetched {len(results)} games ({failed} failed)")
        # Save intermediate
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f)

    time.sleep(0.25)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f)

print(f"\nDone! {len(results)} games saved, {failed} failed")
