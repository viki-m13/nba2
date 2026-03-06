#!/usr/bin/env python3
"""
Deep Player Prop Analysis — Focus on highest-accuracy single legs
then test combining them into parlays.

Key findings from analysis 1:
- Regression UNDER (L3 > L10 by 25%+): 59.2% on pts, 60.3% on PRA
- Star OVER (L5 > season AND all L3 above season): 58.2% pts, 63.8% PRA
- Compound rebounds OVER: 58.7%

Need: Individual legs at 55%+ to make parlays profitable at -115.
At -115 odds: break-even is ~53.5% per leg.
2-leg parlay at -115: +251, break-even 28.5%
If each leg hits 58%: parlay hits 33.6% → ROI = (0.336*351 - 100)/100 = +18%
If each leg hits 60%: parlay hits 36% → ROI = (0.36*351 - 100)/100 = +26.4%
"""

import json
from collections import defaultdict

with open("webapp/data/player_boxscores.json") as f:
    games = json.load(f)

games.sort(key=lambda g: g["date"])

# Build per-game player lookups
games_by_date = defaultdict(list)
for game in games:
    games_by_date[game["date"]].append(game)

def parse_mins(m):
    try:
        return int(m)
    except:
        try:
            return int(m.split(":")[0]) if ":" in m else int(float(m))
        except:
            return 0

# ─── Test individual leg accuracy with tighter filters ───

print("=" * 70)
print("DEEP ANALYSIS: Individual Leg Accuracy at Various Thresholds")
print("=" * 70)

# Test: Regression UNDER with varying spike thresholds
print("\n--- Regression UNDER (Points) ---")
print("Filter: L3_avg > L10_avg * threshold → bet UNDER L5_avg")
for threshold in [1.15, 1.20, 1.25, 1.30, 1.35, 1.40]:
    player_hist = defaultdict(list)
    hits = 0
    total = 0
    for date in sorted(games_by_date.keys()):
        for game in games_by_date[date]:
            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 20:
                    continue
                pid = p["name"]
                hist = player_hist.get(pid, [])
                if len(hist) >= 10:
                    l10 = hist[-10:]
                    l3 = hist[-3:]
                    l10_avg = sum(g["pts"] for g in l10) / 10
                    l3_avg = sum(g["pts"] for g in l3) / 3
                    l5 = hist[-5:]
                    l5_avg = sum(g["pts"] for g in l5) / 5

                    if l10_avg > 15 and l3_avg > l10_avg * threshold:
                        hit = p["pts"] < l5_avg
                        total += 1
                        if hit:
                            hits += 1

        for game in games_by_date[date]:
            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 10:
                    continue
                player_hist[p["name"]].append({"pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "pra": p["pts"] + p["reb"] + p["ast"]})

    pct = hits / total * 100 if total > 0 else 0
    print(f"  Threshold {threshold:.2f}: {hits}/{total} ({pct:.1f}%)")

# Test: Star PRA OVER with varying filters
print("\n--- Star PRA OVER ---")
print("Filter: L5_avg > season_avg AND all L3 > season_avg, min games threshold")
for min_games in [15, 20, 25, 30]:
    for min_pra in [25, 30, 35]:
        player_hist = defaultdict(list)
        hits = 0
        total = 0
        for date in sorted(games_by_date.keys()):
            for game in games_by_date[date]:
                for p in game["players"]:
                    mins = parse_mins(p["min"])
                    if mins < 20:
                        continue
                    pid = p["name"]
                    hist = player_hist.get(pid, [])
                    if len(hist) >= min_games:
                        season_avg = sum(g["pra"] for g in hist) / len(hist)
                        if season_avg < min_pra:
                            continue
                        l5 = hist[-5:]
                        l3 = hist[-3:]
                        l5_avg = sum(g["pra"] for g in l5) / 5
                        if l5_avg > season_avg and all(g["pra"] > season_avg for g in l3):
                            actual_pra = p["pts"] + p["reb"] + p["ast"]
                            hit = actual_pra > season_avg
                            total += 1
                            if hit:
                                hits += 1

            for game in games_by_date[date]:
                for p in game["players"]:
                    mins = parse_mins(p["min"])
                    if mins < 10:
                        continue
                    player_hist[p["name"]].append({"pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "pra": p["pts"] + p["reb"] + p["ast"]})

        pct = hits / total * 100 if total > 0 else 0
        print(f"  minGames={min_games} minPRA={min_pra}: {hits}/{total} ({pct:.1f}%)")

# ─── Best Signal: Assists OVER for high-assist players ───
print("\n--- Assists OVER for high-assist players ---")
print("Filter: Season avg 7+ AST, L5_avg > season AND all L3 > season")
for min_ast in [5, 6, 7, 8]:
    player_hist = defaultdict(list)
    hits = 0
    total = 0
    monthly = defaultdict(lambda: {"h": 0, "t": 0})
    for date in sorted(games_by_date.keys()):
        for game in games_by_date[date]:
            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 20:
                    continue
                pid = p["name"]
                hist = player_hist.get(pid, [])
                if len(hist) >= 15:
                    season_avg = sum(g["ast"] for g in hist) / len(hist)
                    if season_avg < min_ast:
                        continue
                    l5 = hist[-5:]
                    l3 = hist[-3:]
                    l5_avg = sum(g["ast"] for g in l5) / 5
                    if l5_avg > season_avg and all(g["ast"] > season_avg for g in l3):
                        hit = p["ast"] > season_avg
                        total += 1
                        if hit:
                            hits += 1
                        m = date[:6]
                        monthly[m]["t"] += 1
                        if hit:
                            monthly[m]["h"] += 1

        for game in games_by_date[date]:
            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 10:
                    continue
                player_hist[p["name"]].append({"pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "pra": p["pts"] + p["reb"] + p["ast"]})

    pct = hits / total * 100 if total > 0 else 0
    print(f"  minAST={min_ast}: {hits}/{total} ({pct:.1f}%)")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        mp = d["h"] / d["t"] * 100 if d["t"] > 0 else 0
        print(f"    {m}: {d['h']}/{d['t']} ({mp:.1f}%)")

# ─── BEST PARLAY TEST: Combine strongest independent signals ───
print()
print("=" * 70)
print("PARLAY TEST A: UNDER regression (pts) + OVER star PRA")
print("2-leg parlay at -115/-115 = +251")
print("=" * 70)

player_hist = defaultdict(list)
parlay_results = []

for date in sorted(games_by_date.keys()):
    day_games = games_by_date[date]

    # Generate UNDER candidates (regression)
    under_candidates = []
    over_candidates = []

    for game in day_games:
        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 20:
                continue
            pid = p["name"]
            hist = player_hist.get(pid, [])
            if len(hist) < 15:
                continue

            # UNDER: pts regression
            l10 = hist[-10:]
            l3 = hist[-3:]
            l10_avg = sum(g["pts"] for g in l10) / 10
            l3_avg = sum(g["pts"] for g in l3) / 3
            l5 = hist[-5:]
            l5_avg = sum(g["pts"] for g in l5) / 5

            if l10_avg > 15 and l3_avg > l10_avg * 1.25:
                hit = p["pts"] < l5_avg
                under_candidates.append({
                    "player": pid,
                    "type": "PTS UNDER",
                    "line": round(l5_avg, 1),
                    "actual": p["pts"],
                    "hit": hit,
                    "strength": (l3_avg - l10_avg) / l10_avg,
                    "game_key": f"{game['away']}@{game['home']}",
                })

            # OVER: PRA star
            season_avg_pra = sum(g["pra"] for g in hist) / len(hist)
            if season_avg_pra >= 30:
                l5_pra = sum(g["pra"] for g in l5) / 5
                if l5_pra > season_avg_pra and all(g["pra"] > season_avg_pra for g in l3):
                    actual_pra = p["pts"] + p["reb"] + p["ast"]
                    hit = actual_pra > season_avg_pra
                    over_candidates.append({
                        "player": pid,
                        "type": "PRA OVER",
                        "line": round(season_avg_pra, 1),
                        "actual": actual_pra,
                        "hit": hit,
                        "strength": (l5_pra - season_avg_pra) / season_avg_pra,
                        "game_key": f"{game['away']}@{game['home']}",
                    })

    # Try to build a parlay: 1 UNDER + 1 OVER from different games
    under_candidates.sort(key=lambda c: c["strength"], reverse=True)
    over_candidates.sort(key=lambda c: c["strength"], reverse=True)

    for uc in under_candidates:
        for oc in over_candidates:
            if uc["game_key"] != oc["game_key"]:  # Different games
                both_hit = uc["hit"] and oc["hit"]
                parlay_results.append({
                    "date": date,
                    "leg1": uc,
                    "leg2": oc,
                    "won": both_hit,
                    "pnl": 251 if both_hit else -100,
                })
                break  # Only one parlay per day
        else:
            continue
        break

    # Update history
    for game in day_games:
        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 10:
                continue
            player_hist[p["name"]].append({
                "pts": p["pts"], "reb": p["reb"], "ast": p["ast"],
                "pra": p["pts"] + p["reb"] + p["ast"],
            })

wins = sum(1 for r in parlay_results if r["won"])
total = len(parlay_results)
pnl = sum(r["pnl"] for r in parlay_results)
hit_rate = wins / total * 100 if total > 0 else 0

print(f"\nUNDER + OVER Mixed Parlay:")
print(f"  Record: {wins}-{total - wins} ({hit_rate:.1f}%)")
print(f"  PnL: ${pnl:+d} on ${total * 100} wagered")
print(f"  ROI: {pnl / (total * 100) * 100:.1f}%")
print(f"  Break-even: 28.5%")

monthly = defaultdict(lambda: {"w": 0, "t": 0, "pnl": 0})
for r in parlay_results:
    m = r["date"][:6]
    monthly[m]["t"] += 1
    monthly[m]["pnl"] += r["pnl"]
    if r["won"]:
        monthly[m]["w"] += 1

print(f"\nMonthly:")
for m in sorted(monthly.keys()):
    d = monthly[m]
    mp = d["w"] / d["t"] * 100 if d["t"] > 0 else 0
    print(f"  {m}: {d['w']}-{d['t'] - d['w']} ({mp:.1f}%) PnL: ${d['pnl']:+d}")

for r in parlay_results:
    l1, l2 = r["leg1"], r["leg2"]
    res = "WIN" if r["won"] else "LOSS"
    print(f"  {r['date']} | {l1['player']} {l1['type']} {l1['line']} (actual:{l1['actual']}) {'✓' if l1['hit'] else '✗'} + {l2['player']} {l2['type']} {l2['line']} (actual:{l2['actual']}) {'✓' if l2['hit'] else '✗'} → {res}")

# ─── PARLAY TEST B: Same-signal parlay — 2 regression UNDERs ───
print()
print("=" * 70)
print("PARLAY TEST B: 2x Regression UNDER (pts) from different games")
print("2-leg parlay at -115/-115 = +251")
print("=" * 70)

player_hist2 = defaultdict(list)
parlay_results2 = []

for date in sorted(games_by_date.keys()):
    day_games = games_by_date[date]
    candidates = []

    for game in day_games:
        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 20:
                continue
            pid = p["name"]
            hist = player_hist2.get(pid, [])
            if len(hist) < 10:
                continue

            l10 = hist[-10:]
            l3 = hist[-3:]
            l5 = hist[-5:]
            l10_avg = sum(g["pts"] for g in l10) / 10
            l3_avg = sum(g["pts"] for g in l3) / 3
            l5_avg = sum(g["pts"] for g in l5) / 5

            if l10_avg > 15 and l3_avg > l10_avg * 1.25:
                hit = p["pts"] < l5_avg
                candidates.append({
                    "player": pid,
                    "line": round(l5_avg, 1),
                    "actual": p["pts"],
                    "hit": hit,
                    "strength": (l3_avg - l10_avg) / l10_avg,
                    "game_key": f"{game['away']}@{game['home']}",
                })

    candidates.sort(key=lambda c: c["strength"], reverse=True)
    selected = []
    used = set()
    for c in candidates:
        if c["game_key"] in used:
            continue
        selected.append(c)
        used.add(c["game_key"])
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
            mins = parse_mins(p["min"])
            if mins < 10:
                continue
            player_hist2[p["name"]].append({"pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "pra": p["pts"] + p["reb"] + p["ast"]})

wins2 = sum(1 for r in parlay_results2 if r["won"])
total2 = len(parlay_results2)
pnl2 = sum(r["pnl"] for r in parlay_results2)

print(f"\n2x UNDER Regression Parlay:")
print(f"  Record: {wins2}-{total2 - wins2} ({wins2/total2*100:.1f}%)")
print(f"  PnL: ${pnl2:+d}")
print(f"  ROI: {pnl2 / (total2 * 100) * 100:.1f}%")

monthly2 = defaultdict(lambda: {"w": 0, "t": 0, "pnl": 0})
for r in parlay_results2:
    m = r["date"][:6]
    monthly2[m]["t"] += 1
    monthly2[m]["pnl"] += r["pnl"]
    if r["won"]:
        monthly2[m]["w"] += 1
print(f"\nMonthly:")
for m in sorted(monthly2.keys()):
    d = monthly2[m]
    mp = d["w"] / d["t"] * 100 if d["t"] > 0 else 0
    print(f"  {m}: {d['w']}-{d['t'] - d['w']} ({mp:.1f}%) PnL: ${d['pnl']:+d}")

# ─── PARLAY TEST C: Higher threshold regression (30%+ spike) ───
print()
print("=" * 70)
print("PARLAY TEST C: Ultra-selective regression UNDER (30%+ spike)")
print("=" * 70)

for spike in [1.30, 1.35, 1.40]:
    player_hist3 = defaultdict(list)
    results3 = []

    for date in sorted(games_by_date.keys()):
        candidates = []
        for game in games_by_date[date]:
            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 20:
                    continue
                pid = p["name"]
                hist = player_hist3.get(pid, [])
                if len(hist) < 10:
                    continue
                l10 = hist[-10:]
                l3 = hist[-3:]
                l5 = hist[-5:]
                l10_avg = sum(g["pts"] for g in l10) / 10
                l3_avg = sum(g["pts"] for g in l3) / 3
                l5_avg = sum(g["pts"] for g in l5) / 5

                if l10_avg > 15 and l3_avg > l10_avg * spike:
                    hit = p["pts"] < l5_avg
                    candidates.append({
                        "player": pid, "line": round(l5_avg, 1), "actual": p["pts"],
                        "hit": hit, "strength": l3_avg / l10_avg,
                        "game_key": f"{game['away']}@{game['home']}",
                    })

        candidates.sort(key=lambda c: c["strength"], reverse=True)
        selected = []
        used = set()
        for c in candidates:
            if c["game_key"] in used:
                continue
            selected.append(c)
            used.add(c["game_key"])
            if len(selected) >= 2:
                break

        if len(selected) >= 2:
            both_hit = selected[0]["hit"] and selected[1]["hit"]
            results3.append({"date": date, "won": both_hit, "pnl": 251 if both_hit else -100})

        for game in games_by_date[date]:
            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 10:
                    continue
                player_hist3[p["name"]].append({"pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "pra": p["pts"] + p["reb"] + p["ast"]})

    w3 = sum(1 for r in results3 if r["won"])
    t3 = len(results3)
    p3 = sum(r["pnl"] for r in results3)
    print(f"  Spike {spike:.0%}: {w3}-{t3-w3} ({w3/t3*100:.1f}%) PnL: ${p3:+d} ROI: {p3/(t3*100)*100:.1f}%" if t3 > 0 else f"  Spike {spike:.0%}: No parlays")

# ─── PARLAY TEST D: Mix UNDER pts + OVER rebounds ───
print()
print("=" * 70)
print("PARLAY TEST D: UNDER pts (regression) + OVER reb (compound)")
print("=" * 70)

player_hist4 = defaultdict(list)
parlay_results4 = []

for date in sorted(games_by_date.keys()):
    under_cands = []
    over_cands = []

    for game in games_by_date[date]:
        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 20:
                continue
            pid = p["name"]
            hist = player_hist4.get(pid, [])
            if len(hist) < 10:
                continue

            l10 = hist[-10:]
            l5 = hist[-5:]
            l3 = hist[-3:]

            # UNDER pts regression
            l10_pts = sum(g["pts"] for g in l10) / 10
            l3_pts = sum(g["pts"] for g in l3) / 3
            l5_pts = sum(g["pts"] for g in l5) / 5
            if l10_pts > 15 and l3_pts > l10_pts * 1.25:
                hit = p["pts"] < l5_pts
                under_cands.append({
                    "player": pid, "type": "PTS U", "line": round(l5_pts, 1),
                    "actual": p["pts"], "hit": hit,
                    "strength": l3_pts / l10_pts,
                    "game_key": f"{game['away']}@{game['home']}",
                })

            # OVER rebounds compound
            l10_reb = sum(g["reb"] for g in l10) / 10
            l5_reb = sum(g["reb"] for g in l5) / 5
            l3_reb = sum(g["reb"] for g in l3) / 3
            consistency = sum(1 for g in l10 if g["reb"] > l10_reb) / 10
            if l3_reb > l5_reb and l5_reb > l10_reb and consistency >= 0.6 and l10_reb >= 5:
                hit = p["reb"] > l10_reb
                over_cands.append({
                    "player": pid, "type": "REB O", "line": round(l10_reb, 1),
                    "actual": p["reb"], "hit": hit,
                    "strength": (l3_reb - l10_reb) / l10_reb if l10_reb > 0 else 0,
                    "game_key": f"{game['away']}@{game['home']}",
                })

    under_cands.sort(key=lambda c: c["strength"], reverse=True)
    over_cands.sort(key=lambda c: c["strength"], reverse=True)

    for uc in under_cands:
        for oc in over_cands:
            if uc["game_key"] != oc["game_key"] and uc["player"] != oc["player"]:
                both_hit = uc["hit"] and oc["hit"]
                parlay_results4.append({
                    "date": date, "leg1": uc, "leg2": oc,
                    "won": both_hit, "pnl": 251 if both_hit else -100,
                })
                break
        else:
            continue
        break

    for game in games_by_date[date]:
        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 10:
                continue
            player_hist4[p["name"]].append({"pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "pra": p["pts"] + p["reb"] + p["ast"]})

w4 = sum(1 for r in parlay_results4 if r["won"])
t4 = len(parlay_results4)
p4 = sum(r["pnl"] for r in parlay_results4)
print(f"\nUNDER pts + OVER reb Parlay:")
print(f"  Record: {w4}-{t4-w4} ({w4/t4*100:.1f}%)" if t4 > 0 else "  No parlays")
print(f"  PnL: ${p4:+d} ROI: {p4/(t4*100)*100:.1f}%" if t4 > 0 else "")

monthly4 = defaultdict(lambda: {"w": 0, "t": 0, "pnl": 0})
for r in parlay_results4:
    m = r["date"][:6]
    monthly4[m]["t"] += 1
    monthly4[m]["pnl"] += r["pnl"]
    if r["won"]:
        monthly4[m]["w"] += 1
print(f"\nMonthly:")
for m in sorted(monthly4.keys()):
    d = monthly4[m]
    mp = d["w"] / d["t"] * 100 if d["t"] > 0 else 0
    print(f"  {m}: {d['w']}-{d['t'] - d['w']} ({mp:.1f}%) PnL: ${d['pnl']:+d}")

# ─── SINGLE LEG ANALYSIS: What's the absolute best individual signal? ───
print()
print("=" * 70)
print("INDIVIDUAL LEG ACCURACY SUMMARY")
print("Need 55%+ for profitable -115 single legs")
print("Need 58%+ for each leg to make 2-leg parlay profitable")
print("=" * 70)

# Test: PRA UNDER regression
player_hist5 = defaultdict(list)
for stat, label, min_val in [("pts", "Points", 15), ("reb", "Rebounds", 5), ("ast", "Assists", 4), ("pra", "PRA", 25)]:
    for direction, threshold_fn in [
        ("OVER (compound)", lambda l3, l5, l10, cons: l3 > l5 and l5 > l10 and cons >= 0.6),
        ("UNDER (regression 25%)", lambda l3, l5, l10, cons: l3 > l10 * 1.25),
        ("UNDER (regression 30%)", lambda l3, l5, l10, cons: l3 > l10 * 1.30),
    ]:
        player_hist5 = defaultdict(list)
        hits = 0
        total = 0
        for date in sorted(games_by_date.keys()):
            for game in games_by_date[date]:
                for p in game["players"]:
                    mins = parse_mins(p["min"])
                    if mins < 20:
                        continue
                    pid = p["name"]
                    hist = player_hist5.get(pid, [])
                    if len(hist) < 10:
                        continue
                    l10 = hist[-10:]
                    l5 = hist[-5:]
                    l3 = hist[-3:]
                    l10_avg = sum(g[stat] for g in l10) / 10
                    l5_avg = sum(g[stat] for g in l5) / 5
                    l3_avg = sum(g[stat] for g in l3) / 3
                    cons = sum(1 for g in l10 if g[stat] > l10_avg) / 10

                    if l10_avg < min_val:
                        continue

                    if threshold_fn(l3_avg, l5_avg, l10_avg, cons):
                        if "OVER" in direction:
                            hit = p[stat] if stat != "pra" else (p["pts"] + p["reb"] + p["ast"])
                            hit = hit > l10_avg
                        else:
                            actual = p[stat] if stat != "pra" else (p["pts"] + p["reb"] + p["ast"])
                            hit = actual < l5_avg
                        total += 1
                        if hit:
                            hits += 1

            for game in games_by_date[date]:
                for p in game["players"]:
                    mins = parse_mins(p["min"])
                    if mins < 10:
                        continue
                    player_hist5[p["name"]].append({"pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "pra": p["pts"] + p["reb"] + p["ast"]})

        pct = hits / total * 100 if total > 0 else 0
        marker = " ✓" if pct >= 55 else " ✗"
        print(f"  {label:10s} {direction:25s}: {hits}/{total:4d} ({pct:.1f}%){marker}")
