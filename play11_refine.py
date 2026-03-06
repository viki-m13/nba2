#!/usr/bin/env python3
"""
Play 11 Refinement: Optimize for 90%+ parlay accuracy with high payout
Focus on confidence-filtered floor bets
"""

import json
from collections import defaultdict

with open("webapp/data/player_boxscores.json") as f:
    games = json.load(f)

games.sort(key=lambda g: g["date"])
games_by_date = defaultdict(list)
for game in games:
    games_by_date[game["date"]].append(game)

dates = sorted(games_by_date.keys())

def parse_mins(m):
    try:
        return int(m)
    except:
        try:
            return int(m.split(":")[0]) if ":" in m else int(float(m))
        except:
            return 0

def american_to_decimal(am):
    return 1 + am/100 if am > 0 else 1 + 100/abs(am)

def parlay_american(legs):
    dec = 1.0
    for o in legs:
        dec *= american_to_decimal(o)
    if dec >= 2.0:
        return round((dec - 1) * 100)
    else:
        return round(-100 / (dec - 1))

print("=" * 80)
print("PLAY 11 REFINEMENT: Confidence-filtered floor bets")
print("=" * 80)

# Test configurations: (floor_pct, min_confidence, leg_odds, min_legs, max_legs, label)
configs = [
    # Floor %, min confidence (L10_min/floor ratio), odds, min/max legs
    (0.40, 1.5, -275, 4, 6, "40% floor, conf>=1.5"),
    (0.40, 2.0, -275, 4, 6, "40% floor, conf>=2.0"),
    (0.40, 2.5, -300, 3, 6, "40% floor, conf>=2.5"),
    (0.45, 1.5, -225, 4, 6, "45% floor, conf>=1.5"),
    (0.45, 2.0, -250, 4, 6, "45% floor, conf>=2.0"),
    (0.45, 2.5, -275, 3, 6, "45% floor, conf>=2.5"),
    (0.35, 2.0, -350, 4, 7, "35% floor, conf>=2.0"),
    (0.35, 2.5, -400, 4, 7, "35% floor, conf>=2.5"),
    (0.35, 3.0, -450, 4, 7, "35% floor, conf>=3.0"),
    (0.40, 1.0, -250, 5, 8, "40% floor, conf>=1.0, 5-8legs"),
    (0.35, 1.5, -300, 5, 8, "35% floor, conf>=1.5, 5-8legs"),
    (0.35, 2.0, -350, 5, 8, "35% floor, conf>=2.0, 5-8legs"),
    # Very conservative
    (0.30, 2.0, -500, 5, 8, "30% floor, conf>=2.0"),
    (0.30, 2.5, -550, 5, 8, "30% floor, conf>=2.5"),
    (0.30, 3.0, -600, 4, 7, "30% floor, conf>=3.0"),
]

best_config = None
best_score = 0

for floor_pct, min_conf, leg_odds, min_legs, max_legs, label in configs:
    player_hist = defaultdict(list)
    parlay_results = []
    leg_tracker = {"hits": 0, "total": 0}

    for date in dates:
        candidates = []

        for game in games_by_date[date]:
            game_key = f"{game['away']}@{game['home']}"
            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 1:
                    continue
                pid = p["name"]
                hist = player_hist.get(pid, [])
                if len(hist) < 10:
                    continue

                l10 = hist[-10:]
                l10_avg_min = sum(h["min"] for h in l10) / 10
                l10_avg_pts = sum(h["pts"] for h in l10) / 10
                l10_min_pts = min(h["pts"] for h in l10)

                if l10_avg_min < 28 or l10_avg_pts < 15:
                    continue

                floor_line = round(l10_avg_pts * floor_pct, 1)
                if floor_line < 4:
                    continue

                confidence = l10_min_pts / floor_line if floor_line > 0 else 0
                if confidence < min_conf:
                    continue

                hit = p["pts"] > floor_line
                candidates.append({
                    "player": pid, "game_key": game_key,
                    "hit": hit, "floor": floor_line, "actual": p["pts"],
                    "confidence": confidence, "l10_avg": l10_avg_pts, "l10_min": l10_min_pts
                })

        for c in candidates:
            leg_tracker["total"] += 1
            if c["hit"]:
                leg_tracker["hits"] += 1

        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        selected = []
        used_games = set()
        for c in candidates:
            if c["game_key"] in used_games:
                continue
            selected.append(c)
            used_games.add(c["game_key"])
            if len(selected) >= max_legs:
                break

        if len(selected) >= min_legs:
            n = len(selected)
            all_hit = all(s["hit"] for s in selected)
            odds = parlay_american([leg_odds] * n)
            pnl = odds if all_hit else -100
            parlay_results.append({
                "date": date, "won": all_hit, "pnl": pnl,
                "legs": selected, "num_legs": n, "odds": odds
            })

        for game in games_by_date[date]:
            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 10:
                    continue
                player_hist[p["name"]].append({
                    "pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "min": mins
                })

    if not parlay_results:
        continue

    wins = sum(1 for r in parlay_results if r["won"])
    total = len(parlay_results)
    total_pnl = sum(r["pnl"] for r in parlay_results)
    roi = total_pnl / (total * 100) * 100
    leg_acc = leg_tracker["hits"] / leg_tracker["total"] * 100 if leg_tracker["total"] > 0 else 0
    avg_legs = sum(r["num_legs"] for r in parlay_results) / total
    avg_odds = sum(r["odds"] for r in parlay_results) / total
    hit_rate = wins / total * 100
    fire_rate = total / len(dates) * 100

    # Score: want 90%+ hit rate, high fire rate, high ROI
    score = 0
    if hit_rate >= 90:
        score = roi * (fire_rate / 100) * (hit_rate / 100)
    elif hit_rate >= 85:
        score = roi * (fire_rate / 100) * (hit_rate / 100) * 0.5

    flag = "★" if hit_rate >= 90 and roi > 50 and fire_rate > 50 else ""

    print(f"  {label:<35} | {wins}/{total} ({hit_rate:.0f}%) | "
          f"legs:{avg_legs:.1f} | +{avg_odds:.0f} | "
          f"ROI:{roi:+.0f}% | fire:{fire_rate:.0f}% | leg:{leg_acc:.1f}% {flag}")

    if score > best_score:
        best_score = score
        best_config = (floor_pct, min_conf, leg_odds, min_legs, max_legs, label, parlay_results, leg_tracker)

if best_config:
    floor_pct, min_conf, leg_odds, min_legs, max_legs, label, parlay_results, leg_tracker = best_config
    print(f"\n{'='*80}")
    print(f"BEST CONFIG: {label}")
    print(f"{'='*80}")

    wins = sum(1 for r in parlay_results if r["won"])
    total = len(parlay_results)
    total_pnl = sum(r["pnl"] for r in parlay_results)
    roi = total_pnl / (total * 100) * 100
    leg_acc = leg_tracker["hits"] / leg_tracker["total"] * 100
    avg_legs = sum(r["num_legs"] for r in parlay_results) / total

    print(f"  Parlay record: {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"  Leg accuracy: {leg_tracker['hits']}/{leg_tracker['total']} ({leg_acc:.1f}%)")
    print(f"  Avg legs per parlay: {avg_legs:.1f}")
    print(f"  Total P&L: ${total_pnl:+,}")
    print(f"  ROI: {roi:+.1f}%")

    monthly = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0})
    for r in parlay_results:
        m = r["date"][:6]
        monthly[m]["pnl"] += r["pnl"]
        if r["won"]: monthly[m]["w"] += 1
        else: monthly[m]["l"] += 1

    print(f"\n  Monthly:")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        t = d["w"] + d["l"]
        print(f"    {m}: {d['w']}-{d['l']} ({d['w']/t*100:.1f}%) PnL: ${d['pnl']:+,}")

    # Show some example parlays
    print(f"\n  Recent parlays:")
    for r in parlay_results[-10:]:
        legs_str = " | ".join(f"{l['player']} O{l['floor']}pts (got:{l['actual']}){'✓' if l['hit'] else '✗'}" for l in r["legs"][:4])
        if len(r["legs"]) > 4:
            legs_str += f" +{len(r['legs'])-4}more"
        res = "WIN" if r["won"] else "LOSS"
        print(f"    {r['date']} [{r['num_legs']}leg +{r['odds']}] {res} ${r['pnl']:+d}: {legs_str}")

# ═══════════════════════════════════════════════════════════════
# Now test the BEST approach but with more conservative odds
# to make sure it's profitable even with worse pricing
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("SENSITIVITY: Best config at various odds levels")
print(f"{'='*80}")

if best_config:
    _, _, _, _, _, _, results, _ = best_config
    for test_odds_label, test_odds in [("-200", -200), ("-250", -250), ("-300", -300), ("-350", -350), ("-400", -400), ("-500", -500)]:
        total_pnl = 0
        for r in results:
            n = r["num_legs"]
            odds = parlay_american([int(test_odds)] * n)
            pnl = odds if r["won"] else -100
            total_pnl += pnl
        roi = total_pnl / (len(results) * 100) * 100
        print(f"  At {test_odds_label}/leg: P&L: ${total_pnl:+,} | ROI: {roi:+.1f}%")
