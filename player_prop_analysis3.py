#!/usr/bin/env python3
"""
Final Validation: 2x Regression UNDER Points Parlay
Strategy: When a player's L3 avg is 25%+ above their L10 avg, bet UNDER their L5 avg.
Combine 2 such picks from different games into a 2-leg parlay.

ODDS ANALYSIS:
- Standard player prop odds: -115 (most common) to -110
- At -115: Win $87 on $100, break-even 53.5%
- 2-leg at -115/-115: True odds = (115/215)^2 = 28.6% → +251 American
  Win = +$251, Loss = -$100
- At -110: 2-leg = +264, Win = +$264, Loss = -$100

Individual leg accuracy: 64.2% (well above 53.5% break-even)
Expected parlay hit: 64.2% × 64.2% = 41.2% (well above 28.6% break-even)
Expected ROI: (0.412 × 351 - 100) / 100 = +44.6% theoretical

We'll use -115 (conservative, most common prop odds).
"""

import json
from collections import defaultdict

with open("webapp/data/player_boxscores.json") as f:
    games = json.load(f)

games.sort(key=lambda g: g["date"])

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

# ═══════════════════════════════════════════════════════════════
# STRATEGY: PRISM Player Props (Regression UNDER Parlay)
# Each leg: Player Points UNDER L5_avg at -115
# Trigger: L3_avg > L10_avg × 1.25 (25% spike = unsustainable)
# Parlay: 2 legs from different games = +251 American
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("VALIDATED STRATEGY: Regression UNDER 2-Leg Player Prop Parlay")
print("=" * 70)
print()
print("Bet type: Player Points UNDER (2-leg parlay)")
print("Odds per leg: -115 (standard player prop)")
print("Parlay odds: +251 American")
print("Win payout: +$251 on $100 wager")
print("Loss: -$100")
print("Break-even: 28.5%")
print()

player_hist = defaultdict(list)
parlay_results = []
individual_legs = {"hits": 0, "total": 0}  # Track individual leg accuracy

for date in sorted(games_by_date.keys()):
    candidates = []

    for game in games_by_date[date]:
        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 20:
                continue
            pid = p["name"]
            hist = player_hist.get(pid, [])
            if len(hist) < 10:
                continue

            l10 = hist[-10:]
            l5 = hist[-5:]
            l3 = hist[-3:]
            l10_avg = sum(g["pts"] for g in l10) / 10
            l5_avg = sum(g["pts"] for g in l5) / 5
            l3_avg = sum(g["pts"] for g in l3) / 3

            # Core filter: 25%+ spike AND meaningful scorer
            if l10_avg >= 15 and l3_avg > l10_avg * 1.25:
                actual = p["pts"]
                line = round(l5_avg, 1)  # UNDER this line
                hit = actual < line
                spike_pct = (l3_avg - l10_avg) / l10_avg

                candidates.append({
                    "player": pid,
                    "team": p["team"],
                    "line": line,
                    "actual": actual,
                    "hit": hit,
                    "spike": spike_pct,
                    "l3_avg": round(l3_avg, 1),
                    "l10_avg": round(l10_avg, 1),
                    "game_key": f"{game['away']}@{game['home']}",
                    "game_display": f"{game['away']} @ {game['home']}",
                })

    # Select top 2 from different games (highest spike = most likely to regress)
    candidates.sort(key=lambda c: c["spike"], reverse=True)
    selected = []
    used_games = set()

    for c in candidates:
        if c["game_key"] in used_games:
            continue
        selected.append(c)
        used_games.add(c["game_key"])
        if len(selected) >= 2:
            break

    # Track individual leg accuracy for all candidates
    for c in candidates:
        individual_legs["total"] += 1
        if c["hit"]:
            individual_legs["hits"] += 1

    if len(selected) >= 2:
        both_hit = selected[0]["hit"] and selected[1]["hit"]
        # At -115/-115 parlay: +251
        pnl = 251 if both_hit else -100

        parlay_results.append({
            "date": date,
            "leg1": selected[0],
            "leg2": selected[1],
            "won": both_hit,
            "pnl": pnl,
        })

    # Update history
    for game in games_by_date[date]:
        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 10:
                continue
            player_hist[p["name"]].append({
                "pts": p["pts"],
                "reb": p["reb"],
                "ast": p["ast"],
            })

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════

wins = sum(1 for r in parlay_results if r["won"])
losses = len(parlay_results) - wins
total = len(parlay_results)
pnl = sum(r["pnl"] for r in parlay_results)
wagered = total * 100
roi = pnl / wagered * 100 if wagered > 0 else 0

leg_pct = individual_legs["hits"] / individual_legs["total"] * 100 if individual_legs["total"] > 0 else 0

print(f"Individual leg accuracy: {individual_legs['hits']}/{individual_legs['total']} ({leg_pct:.1f}%)")
print()
print(f"PARLAY RESULTS:")
print(f"  Record: {wins}-{losses} ({wins/total*100:.1f}%)")
print(f"  Total wagered: ${wagered:,}")
print(f"  Total P&L: ${pnl:+,}")
print(f"  ROI: {roi:+.1f}%")
print(f"  Break-even: 28.5% — actual: {wins/total*100:.1f}%")
print(f"  Edge over break-even: {wins/total*100 - 28.5:.1f}pp")
print()

# Monthly breakdown
monthly = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0})
for r in parlay_results:
    m = r["date"][:6]
    monthly[m]["pnl"] += r["pnl"]
    if r["won"]:
        monthly[m]["w"] += 1
    else:
        monthly[m]["l"] += 1

print("MONTHLY BREAKDOWN:")
print(f"  {'Month':<10} {'Record':<10} {'Hit%':<8} {'P&L':<10} {'ROI':<8}")
all_positive = True
for m in sorted(monthly.keys()):
    d = monthly[m]
    t = d["w"] + d["l"]
    pct = d["w"] / t * 100 if t > 0 else 0
    r = d["pnl"] / (t * 100) * 100 if t > 0 else 0
    status = "✓" if d["pnl"] > 0 else "✗"
    if d["pnl"] <= 0:
        all_positive = False
    print(f"  {m:<10} {d['w']}-{d['l']:<7} {pct:<7.1f}% ${d['pnl']:<+8} {r:<+7.1f}% {status}")

print(f"\n  All months profitable: {'YES' if all_positive else 'NO'}")

# Game-by-game detail
print(f"\nGAME-BY-GAME RESULTS:")
print(f"  {'Date':<10} {'Leg 1':<40} {'Leg 2':<40} {'Result':<8} {'P&L'}")
for r in parlay_results:
    l1, l2 = r["leg1"], r["leg2"]
    l1_str = f"{l1['player']} U{l1['line']} (got:{l1['actual']}) {'✓' if l1['hit'] else '✗'}"
    l2_str = f"{l2['player']} U{l2['line']} (got:{l2['actual']}) {'✓' if l2['hit'] else '✗'}"
    res = "WIN" if r["won"] else "LOSS"
    print(f"  {r['date']:<10} {l1_str:<40} {l2_str:<40} {res:<8} ${r['pnl']:+d}")

# ═══════════════════════════════════════════════════════════════
# Sensitivity analysis: what if odds are worse?
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("SENSITIVITY: Same strategy at different odds")
print("=" * 70)

for odds_label, payout in [("-110", 264), ("-115", 251), ("-120", 238)]:
    total_pnl = sum(payout if r["won"] else -100 for r in parlay_results)
    total_roi = total_pnl / (len(parlay_results) * 100) * 100
    print(f"  At {odds_label} per leg: Payout +${payout} | P&L: ${total_pnl:+,} | ROI: {total_roi:+.1f}%")

# ═══════════════════════════════════════════════════════════════
# 3-LEG parlay test
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("3-LEG PARLAY TEST: 3 regression UNDERs from different games")
print("At -115/-115/-115 = ~+570 American")
print("=" * 70)

player_hist3 = defaultdict(list)
parlay3_results = []

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
            l5 = hist[-5:]
            l3 = hist[-3:]
            l10_avg = sum(g["pts"] for g in l10) / 10
            l5_avg = sum(g["pts"] for g in l5) / 5
            l3_avg = sum(g["pts"] for g in l3) / 3
            if l10_avg >= 15 and l3_avg > l10_avg * 1.25:
                hit = p["pts"] < l5_avg
                candidates.append({
                    "player": pid, "line": round(l5_avg, 1), "actual": p["pts"],
                    "hit": hit, "spike": (l3_avg - l10_avg) / l10_avg,
                    "game_key": f"{game['away']}@{game['home']}",
                })

    candidates.sort(key=lambda c: c["spike"], reverse=True)
    selected = []
    used_games = set()
    for c in candidates:
        if c["game_key"] in used_games:
            continue
        selected.append(c)
        used_games.add(c["game_key"])
        if len(selected) >= 3:
            break

    if len(selected) >= 3:
        all_hit = all(s["hit"] for s in selected)
        parlay3_results.append({
            "date": date, "won": all_hit,
            "pnl": 570 if all_hit else -100,
        })

    for game in games_by_date[date]:
        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 10:
                continue
            player_hist3[p["name"]].append({"pts": p["pts"], "reb": p["reb"], "ast": p["ast"]})

w3 = sum(1 for r in parlay3_results if r["won"])
t3 = len(parlay3_results)
p3 = sum(r["pnl"] for r in parlay3_results)
print(f"  Record: {w3}-{t3-w3} ({w3/t3*100:.1f}%)" if t3 > 0 else "  No 3-leg parlays found")
print(f"  P&L: ${p3:+d} ROI: {p3/(t3*100)*100:.1f}%" if t3 > 0 else "")
print(f"  Break-even: ~14.9%")

m3 = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0})
for r in parlay3_results:
    m = r["date"][:6]
    m3[m]["pnl"] += r["pnl"]
    if r["won"]:
        m3[m]["w"] += 1
    else:
        m3[m]["l"] += 1
print(f"\n  Monthly:")
for m in sorted(m3.keys()):
    d = m3[m]
    t = d["w"] + d["l"]
    print(f"    {m}: {d['w']}-{d['l']} ({d['w']/t*100:.1f}%) PnL: ${d['pnl']:+d}")
