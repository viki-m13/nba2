#!/usr/bin/env python3
"""
Player Prop Parlay Analysis — Walk-Forward Backtesting

Common player prop bets at sportsbooks:
- Points Over/Under (e.g., Jayson Tatum Over 26.5 pts at -115)
- Rebounds Over/Under
- Assists Over/Under
- PRA (Points + Rebounds + Assists) Over/Under
- 3-Pointers Made Over/Under

Standard player prop odds: -115 (win $87 on $100 bet, break-even ~53.5%)
Some books offer -110 on props.

Parlay math:
- 2-leg at -115/-115: ~+251 American (break-even ~28.5%)
- 2-leg at -110/-110: +264 American (break-even ~27.5%)
- 3-leg at -115: ~+570 (break-even ~14.9%)

Strategy: Find players whose scoring/rebounding/assists are predictable
from recent history, then combine independent legs.
"""

import json
from collections import defaultdict

with open("webapp/data/player_boxscores.json") as f:
    games = json.load(f)

# Sort by date
games.sort(key=lambda g: g["date"])

print(f"=== Player Prop Parlay Analysis ===")
print(f"Games: {len(games)}")
print(f"Date range: {games[0]['date']} to {games[-1]['date']}")
print()

# ─── Build player history ───
# For each player, track game-by-game stats
player_games = defaultdict(list)

for game in games:
    for p in game["players"]:
        # Parse minutes
        try:
            mins = int(p["min"])
        except:
            try:
                mins = int(p["min"].split(":")[0]) if ":" in p["min"] else int(float(p["min"]))
            except:
                mins = 0

        if mins < 15:  # Skip low-minute players
            continue

        player_games[p["name"]].append({
            "date": game["date"],
            "team": p["team"],
            "pts": p["pts"],
            "reb": p["reb"],
            "ast": p["ast"],
            "pra": p["pts"] + p["reb"] + p["ast"],
            "min": mins,
            "opp_home": game["home"],
            "opp_away": game["away"],
        })

print(f"Players with 15+ min games: {len(player_games)}")

# Filter to players with enough games
qualified = {k: v for k, v in player_games.items() if len(v) >= 15}
print(f"Players with 15+ games: {len(qualified)}")
print()

# ─── Strategy: Over/Under Points based on L5 average ───
# For each game, predict "over L5_avg" or "under L5_avg" and check if the
# player goes over/under. Use walk-forward only.

def analyze_prop(stat_key, window_sizes=[5, 7, 10]):
    """Analyze over/under predictions for a given stat using rolling averages."""

    results_by_window = {}

    for window in window_sizes:
        over_results = []  # (date, player, predicted_line, actual, hit)
        under_results = []

        for player, pgames in qualified.items():
            for i in range(window, len(pgames)):
                # Walk-forward: use only data before this game
                recent = pgames[i - window:i]
                avg = sum(g[stat_key] for g in recent) / len(recent)
                actual = pgames[i][stat_key]
                date = pgames[i]["date"]

                # Standard: line is set at the average (which is what books approximate)
                # We need a signal to decide OVER or UNDER

                # Signal 1: Trend — if recent 3 > recent window, go OVER (momentum)
                if i >= 3:
                    recent3 = pgames[i - 3:i]
                    avg3 = sum(g[stat_key] for g in recent3) / 3

                    if avg3 > avg * 1.05:  # Recent trend UP → OVER
                        hit = actual > avg
                        over_results.append((date, player, avg, actual, hit))
                    elif avg3 < avg * 0.95:  # Recent trend DOWN → UNDER
                        hit = actual < avg
                        under_results.append((date, player, avg, actual, hit))

        over_hits = sum(1 for r in over_results if r[4])
        under_hits = sum(1 for r in under_results if r[4])

        results_by_window[window] = {
            "over": {"total": len(over_results), "hits": over_hits},
            "under": {"total": len(under_results), "hits": under_hits},
        }

    return results_by_window

print("=" * 60)
print("STRATEGY 1: Momentum-Based Over/Under")
print("Signal: Recent 3-game avg > L{window} avg by 5% → OVER")
print("Signal: Recent 3-game avg < L{window} avg by 5% → UNDER")
print("=" * 60)

for stat, label in [("pts", "Points"), ("reb", "Rebounds"), ("ast", "Assists"), ("pra", "PRA")]:
    print(f"\n--- {label} ---")
    res = analyze_prop(stat)
    for w, data in res.items():
        o = data["over"]
        u = data["under"]
        o_pct = o["hits"] / o["total"] * 100 if o["total"] > 0 else 0
        u_pct = u["hits"] / u["total"] * 100 if u["total"] > 0 else 0
        comb_hits = o["hits"] + u["hits"]
        comb_total = o["total"] + u["total"]
        comb_pct = comb_hits / comb_total * 100 if comb_total > 0 else 0
        print(f"  L{w}: OVER {o['hits']}/{o['total']} ({o_pct:.1f}%) | UNDER {u['hits']}/{u['total']} ({u_pct:.1f}%) | Combined {comb_hits}/{comb_total} ({comb_pct:.1f}%)")

# ─── Strategy 2: High-Volume Scorers Over in Favorable Matchups ───
print()
print("=" * 60)
print("STRATEGY 2: Star Player Consistency — Always OVER on hot streaks")
print("Signal: Player L5 avg > season avg AND L3 all above season avg")
print("=" * 60)

for stat, label in [("pts", "Points"), ("reb", "Rebounds"), ("ast", "Assists"), ("pra", "PRA")]:
    hits = 0
    total = 0
    monthly = defaultdict(lambda: {"hits": 0, "total": 0})

    for player, pgames in qualified.items():
        # Need at least 20 games for a season average
        if len(pgames) < 20:
            continue

        for i in range(20, len(pgames)):
            season_avg = sum(g[stat] for g in pgames[:i]) / i
            l5 = pgames[i-5:i]
            l3 = pgames[i-3:i]
            l5_avg = sum(g[stat] for g in l5) / 5

            # All of last 3 above season average AND L5 avg above season avg
            if l5_avg > season_avg and all(g[stat] > season_avg for g in l3):
                actual = pgames[i][stat]
                hit = actual > season_avg
                total += 1
                if hit:
                    hits += 1
                month = pgames[i]["date"][:6]
                monthly[month]["total"] += 1
                if hit:
                    monthly[month]["hits"] += 1

    pct = hits / total * 100 if total > 0 else 0
    print(f"\n--- {label}: {hits}/{total} ({pct:.1f}%) ---")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        mp = d["hits"] / d["total"] * 100 if d["total"] > 0 else 0
        print(f"  {m}: {d['hits']}/{d['total']} ({mp:.1f}%)")

# ─── Strategy 3: Combined filters — high floor players ───
print()
print("=" * 60)
print("STRATEGY 3: High-Floor Players — Consistent scorers going OVER")
print("Signal: Player has gone OVER their L10 avg in 7+ of last 10 games")
print("        AND L5 avg > L10 avg (trending up)")
print("=" * 60)

for stat, label in [("pts", "Points"), ("reb", "Rebounds"), ("pra", "PRA")]:
    hits = 0
    total = 0
    monthly = defaultdict(lambda: {"hits": 0, "total": 0})

    for player, pgames in qualified.items():
        if len(pgames) < 15:
            continue

        for i in range(10, len(pgames)):
            l10 = pgames[i-10:i]
            l5 = pgames[i-5:i]
            l10_avg = sum(g[stat] for g in l10) / 10
            l5_avg = sum(g[stat] for g in l5) / 5

            # Count how many of L10 went over the L10 avg
            over_count = sum(1 for g in l10 if g[stat] > l10_avg)

            if over_count >= 7 and l5_avg > l10_avg * 1.03:
                actual = pgames[i][stat]
                line = l10_avg  # Approximate sportsbook line
                hit = actual > line
                total += 1
                if hit:
                    hits += 1
                month = pgames[i]["date"][:6]
                monthly[month]["total"] += 1
                if hit:
                    monthly[month]["hits"] += 1

    pct = hits / total * 100 if total > 0 else 0
    print(f"\n--- {label}: {hits}/{total} ({pct:.1f}%) ---")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        mp = d["hits"] / d["total"] * 100 if d["total"] > 0 else 0
        print(f"  {m}: {d['hits']}/{d['total']} ({mp:.1f}%)")

# ─── Strategy 4: UNDER on regression candidates ───
print()
print("=" * 60)
print("STRATEGY 4: Regression UNDER — Players on unsustainable hot streaks")
print("Signal: L3 avg > L10 avg by 25%+ (unsustainable spike)")
print("=" * 60)

for stat, label in [("pts", "Points"), ("pra", "PRA")]:
    hits = 0
    total = 0

    for player, pgames in qualified.items():
        if len(pgames) < 15:
            continue

        for i in range(10, len(pgames)):
            l10 = pgames[i-10:i]
            l3 = pgames[i-3:i]
            l10_avg = sum(g[stat] for g in l10) / 10
            l3_avg = sum(g[stat] for g in l3) / 3

            if l10_avg > 0 and l3_avg > l10_avg * 1.25:
                actual = pgames[i][stat]
                # Line would be closer to L5 avg (books adjust for hot streaks)
                l5 = pgames[i-5:i]
                l5_avg = sum(g[stat] for g in l5) / 5
                line = l5_avg  # Books use recent average
                hit = actual < line
                total += 1
                if hit:
                    hits += 1

    pct = hits / total * 100 if total > 0 else 0
    print(f"  {label}: {hits}/{total} ({pct:.1f}%)")

# ─── Strategy 5: Compound filter — multiple signals converging ───
print()
print("=" * 60)
print("STRATEGY 5: Compound Player Props — Multiple signals agree")
print("For OVER: L5 > L10 AND L3 > L5 AND consistency > 60%")
print("=" * 60)

for stat, label in [("pts", "Points"), ("reb", "Rebounds"), ("ast", "Assists"), ("pra", "PRA")]:
    hits = 0
    total = 0
    monthly = defaultdict(lambda: {"hits": 0, "total": 0})
    details = []

    for player, pgames in qualified.items():
        if len(pgames) < 15:
            continue

        for i in range(10, len(pgames)):
            l10 = pgames[i-10:i]
            l5 = pgames[i-5:i]
            l3 = pgames[i-3:i]

            l10_avg = sum(g[stat] for g in l10) / 10
            l5_avg = sum(g[stat] for g in l5) / 5
            l3_avg = sum(g[stat] for g in l3) / 3

            # Consistency: how often over l10_avg in last 10
            consistency = sum(1 for g in l10 if g[stat] > l10_avg) / 10

            # Compound: all windows trending up AND high consistency
            if l3_avg > l5_avg and l5_avg > l10_avg and consistency >= 0.6:
                actual = pgames[i][stat]
                line = l10_avg  # Conservative line
                hit = actual > line
                total += 1
                if hit:
                    hits += 1
                month = pgames[i]["date"][:6]
                monthly[month]["total"] += 1
                if hit:
                    monthly[month]["hits"] += 1
                if len(details) < 20:
                    details.append(f"  {pgames[i]['date']} {player}: L10avg={l10_avg:.1f} actual={actual} {'HIT' if hit else 'MISS'}")

    pct = hits / total * 100 if total > 0 else 0
    print(f"\n--- {label}: {hits}/{total} ({pct:.1f}%) ---")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        mp = d["hits"] / d["total"] * 100 if d["total"] > 0 else 0
        print(f"  {m}: {d['hits']}/{d['total']} ({mp:.1f}%)")

# ─── Strategy 6: Ultra-selective — top N most consistent + trending ───
print()
print("=" * 60)
print("STRATEGY 6: Ultra-Selective Daily Picks")
print("Per day: pick top 2 OVER signals (strongest trend + consistency)")
print("Then parlay them as 2-leg at -115/-115 = +251")
print("=" * 60)

# Group games by date
games_by_date = defaultdict(list)
for game in games:
    games_by_date[game["date"]].append(game)

# Build day-by-day walk-forward
player_history = defaultdict(list)  # pid -> list of stat dicts
parlay_results = []

for date in sorted(games_by_date.keys()):
    day_games = games_by_date[date]

    # Generate candidate OVER picks for today
    candidates = []

    for game in day_games:
        for p in game["players"]:
            try:
                mins = int(p["min"])
            except:
                try:
                    mins = int(p["min"].split(":")[0]) if ":" in p["min"] else int(float(p["min"]))
                except:
                    mins = 0

            if mins < 20:
                continue

            pid = p["name"]
            hist = player_history.get(pid, [])

            if len(hist) < 10:
                continue

            l10 = hist[-10:]
            l5 = hist[-5:]
            l3 = hist[-3:]

            for stat in ["pts"]:
                l10_avg = sum(g[stat] for g in l10) / 10
                l5_avg = sum(g[stat] for g in l5) / 5
                l3_avg = sum(g[stat] for g in l3) / 3
                consistency = sum(1 for g in l10 if g[stat] > l10_avg) / 10

                # Compound filter
                if l3_avg > l5_avg and l5_avg > l10_avg and consistency >= 0.6 and l10_avg >= 15:
                    # Strength score
                    trend_strength = (l3_avg - l10_avg) / l10_avg if l10_avg > 0 else 0
                    score = trend_strength * 100 + consistency * 50

                    actual = p[stat]
                    hit = actual > l10_avg

                    candidates.append({
                        "player": pid,
                        "stat": stat,
                        "line": l10_avg,
                        "actual": actual,
                        "hit": hit,
                        "score": score,
                        "game": f"{game['away']}@{game['home']}",
                    })

    # Select top 2 from different games
    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected = []
    used_games = set()

    for c in candidates:
        if c["game"] in used_games:
            continue
        selected.append(c)
        used_games.add(c["game"])
        if len(selected) >= 2:
            break

    if len(selected) >= 2:
        both_hit = selected[0]["hit"] and selected[1]["hit"]
        parlay_results.append({
            "date": date,
            "leg1": selected[0],
            "leg2": selected[1],
            "won": both_hit,
            # 2-leg at -115 each = +251 (decimal 3.51)
            "pnl": 251 if both_hit else -100,
        })

    # Update history for all players in today's games
    for game in day_games:
        for p in game["players"]:
            try:
                mins = int(p["min"])
            except:
                try:
                    mins = int(p["min"].split(":")[0]) if ":" in p["min"] else int(float(p["min"]))
                except:
                    mins = 0
            if mins < 10:
                continue
            player_history[p["name"]].append({
                "date": date,
                "pts": p["pts"],
                "reb": p["reb"],
                "ast": p["ast"],
                "pra": p["pts"] + p["reb"] + p["ast"],
            })

# Results
wins = sum(1 for r in parlay_results if r["won"])
total = len(parlay_results)
pnl = sum(r["pnl"] for r in parlay_results)
hit_rate = wins / total * 100 if total > 0 else 0

print(f"\nPoints OVER 2-Leg Parlay Results:")
print(f"  Record: {wins}-{total - wins} ({hit_rate:.1f}%)")
print(f"  PnL: ${pnl:+d}")
print(f"  ROI: {pnl / (total * 100) * 100:.1f}%")
print(f"  Break-even: 28.5% (at -115 legs)")

# Monthly breakdown
monthly = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0})
for r in parlay_results:
    m = r["date"][:6]
    monthly[m]["total"] += 1
    monthly[m]["pnl"] += r["pnl"]
    if r["won"]:
        monthly[m]["wins"] += 1

print(f"\nMonthly:")
for m in sorted(monthly.keys()):
    d = monthly[m]
    mp = d["wins"] / d["total"] * 100 if d["total"] > 0 else 0
    print(f"  {m}: {d['wins']}-{d['total'] - d['wins']} ({mp:.1f}%) PnL: ${d['pnl']:+d}")

# Show recent results
print(f"\nLast 20 parlays:")
for r in parlay_results[-20:]:
    l1 = r["leg1"]
    l2 = r["leg2"]
    result = "WIN" if r["won"] else "LOSS"
    print(f"  {r['date']} | {l1['player']} {l1['stat'].upper()} O{l1['line']:.1f} (actual:{l1['actual']}) {'✓' if l1['hit'] else '✗'} + {l2['player']} {l2['stat'].upper()} O{l2['line']:.1f} (actual:{l2['actual']}) {'✓' if l2['hit'] else '✗'} → {result} ${r['pnl']:+d}")

# ─── Also test PRA (Points + Rebounds + Assists) compound ───
print()
print("=" * 60)
print("STRATEGY 7: PRA OVER 2-Leg Parlay")
print("Same compound filter but on PRA stat")
print("=" * 60)

player_history2 = defaultdict(list)
parlay_results2 = []

for date in sorted(games_by_date.keys()):
    day_games = games_by_date[date]
    candidates = []

    for game in day_games:
        for p in game["players"]:
            try:
                mins = int(p["min"])
            except:
                try:
                    mins = int(p["min"].split(":")[0]) if ":" in p["min"] else int(float(p["min"]))
                except:
                    mins = 0
            if mins < 20:
                continue

            pid = p["name"]
            hist = player_history2.get(pid, [])
            if len(hist) < 10:
                continue

            l10 = hist[-10:]
            l5 = hist[-5:]
            l3 = hist[-3:]

            stat = "pra"
            l10_avg = sum(g[stat] for g in l10) / 10
            l5_avg = sum(g[stat] for g in l5) / 5
            l3_avg = sum(g[stat] for g in l3) / 3
            consistency = sum(1 for g in l10 if g[stat] > l10_avg) / 10

            if l3_avg > l5_avg and l5_avg > l10_avg and consistency >= 0.6 and l10_avg >= 25:
                trend_strength = (l3_avg - l10_avg) / l10_avg if l10_avg > 0 else 0
                score = trend_strength * 100 + consistency * 50

                actual_pra = p["pts"] + p["reb"] + p["ast"]
                hit = actual_pra > l10_avg

                candidates.append({
                    "player": pid,
                    "stat": "PRA",
                    "line": l10_avg,
                    "actual": actual_pra,
                    "hit": hit,
                    "score": score,
                    "game": f"{game['away']}@{game['home']}",
                })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected = []
    used_games = set()
    for c in candidates:
        if c["game"] in used_games:
            continue
        selected.append(c)
        used_games.add(c["game"])
        if len(selected) >= 2:
            break

    if len(selected) >= 2:
        both_hit = selected[0]["hit"] and selected[1]["hit"]
        parlay_results2.append({
            "date": date,
            "leg1": selected[0],
            "leg2": selected[1],
            "won": both_hit,
            "pnl": 251 if both_hit else -100,
        })

    for game in day_games:
        for p in game["players"]:
            try:
                mins = int(p["min"])
            except:
                try:
                    mins = int(p["min"].split(":")[0]) if ":" in p["min"] else int(float(p["min"]))
                except:
                    mins = 0
            if mins < 10:
                continue
            player_history2[p["name"]].append({
                "date": date,
                "pts": p["pts"],
                "reb": p["reb"],
                "ast": p["ast"],
                "pra": p["pts"] + p["reb"] + p["ast"],
            })

wins2 = sum(1 for r in parlay_results2 if r["won"])
total2 = len(parlay_results2)
pnl2 = sum(r["pnl"] for r in parlay_results2)
hit_rate2 = wins2 / total2 * 100 if total2 > 0 else 0

print(f"\nPRA OVER 2-Leg Parlay Results:")
print(f"  Record: {wins2}-{total2 - wins2} ({hit_rate2:.1f}%)")
print(f"  PnL: ${pnl2:+d}")
print(f"  ROI: {pnl2 / (total2 * 100) * 100:.1f}%")

monthly2 = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0})
for r in parlay_results2:
    m = r["date"][:6]
    monthly2[m]["total"] += 1
    monthly2[m]["pnl"] += r["pnl"]
    if r["won"]:
        monthly2[m]["wins"] += 1

print(f"\nMonthly:")
for m in sorted(monthly2.keys()):
    d = monthly2[m]
    mp = d["wins"] / d["total"] * 100 if d["total"] > 0 else 0
    print(f"  {m}: {d['wins']}-{d['total'] - d['wins']} ({mp:.1f}%) PnL: ${d['pnl']:+d}")
