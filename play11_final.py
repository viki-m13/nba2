#!/usr/bin/env python3
"""
Play 11 FINAL: VAULT Strategy
Multi-leg player prop parlay with 90%+ accuracy and high payout.
Signal: Player Points OVER a very conservative floor (30-40% of L10 avg)
Confidence filter: L10 minimum must be well above the floor
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

# ═══════════════════════════════════════════════════════════════
# FINAL SWEEP: Comprehensive parameter search
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("PLAY 11 FINAL SWEEP")
print("=" * 80)

results_table = []

for floor_pct in [0.25, 0.30, 0.33, 0.35, 0.38, 0.40]:
    for min_conf in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        for min_avg_pts in [12, 15, 18]:
            for min_avg_min in [25, 28, 30]:
                for min_legs, max_legs in [(3,6), (4,7), (5,8), (4,8)]:
                    # Use appropriate odds based on floor
                    if floor_pct <= 0.25:
                        leg_odds = -600
                    elif floor_pct <= 0.30:
                        leg_odds = -450
                    elif floor_pct <= 0.35:
                        leg_odds = -300
                    elif floor_pct <= 0.40:
                        leg_odds = -250
                    else:
                        leg_odds = -200

                    player_hist = defaultdict(list)
                    parlay_results = []
                    leg_hits = 0
                    leg_total = 0

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
                                avg_min = sum(h["min"] for h in l10) / 10
                                avg_pts = sum(h["pts"] for h in l10) / 10
                                min_pts = min(h["pts"] for h in l10)

                                if avg_min < min_avg_min or avg_pts < min_avg_pts:
                                    continue

                                floor = round(avg_pts * floor_pct, 1)
                                if floor < 3:
                                    continue

                                conf = min_pts / floor if floor > 0 else 0
                                if conf < min_conf:
                                    continue

                                hit = p["pts"] > floor
                                candidates.append({
                                    "player": pid, "game_key": game_key,
                                    "hit": hit, "floor": floor, "actual": p["pts"],
                                    "confidence": conf
                                })

                        for c in candidates:
                            leg_total += 1
                            if c["hit"]:
                                leg_hits += 1

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
                                "num_legs": n, "odds": odds, "legs": selected
                            })

                        for game in games_by_date[date]:
                            for p in game["players"]:
                                mins = parse_mins(p["min"])
                                if mins < 10:
                                    continue
                                player_hist[p["name"]].append({
                                    "pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "min": mins
                                })

                    if not parlay_results or len(parlay_results) < 10:
                        continue

                    wins = sum(1 for r in parlay_results if r["won"])
                    total = len(parlay_results)
                    total_pnl = sum(r["pnl"] for r in parlay_results)
                    hit_rate = wins / total * 100
                    roi = total_pnl / (total * 100) * 100
                    fire_rate = total / len(dates) * 100
                    leg_acc = leg_hits / leg_total * 100 if leg_total > 0 else 0
                    avg_legs = sum(r["num_legs"] for r in parlay_results) / total
                    avg_odds = sum(r["odds"] for r in parlay_results) / total

                    # Filter: only show configs with 90%+ accuracy
                    if hit_rate >= 90 and fire_rate >= 30:
                        # Score by ROI * fire_rate
                        score = roi * fire_rate / 100
                        results_table.append({
                            "floor": floor_pct, "conf": min_conf,
                            "avg_pts": min_avg_pts, "avg_min": min_avg_min,
                            "legs": f"{min_legs}-{max_legs}", "odds": leg_odds,
                            "record": f"{wins}/{total}", "hit": hit_rate,
                            "roi": roi, "fire": fire_rate, "leg_acc": leg_acc,
                            "avg_legs": avg_legs, "avg_odds": avg_odds,
                            "pnl": total_pnl, "score": score,
                            "parlay_results": parlay_results, "leg_tracker": (leg_hits, leg_total)
                        })

# Sort by score
results_table.sort(key=lambda r: r["score"], reverse=True)

print(f"\nTop 20 configs with 90%+ accuracy and 30%+ fire rate:")
print(f"{'Floor':<6} {'Conf':<5} {'AvgPts':<7} {'AvgMin':<7} {'Legs':<6} {'Odds':<6} {'Record':<8} {'Hit%':<6} {'ROI':<8} {'Fire%':<7} {'LegAcc':<7} {'AvgLegs':<8} {'P&L':<10}")
print("-" * 100)

for r in results_table[:20]:
    print(f"  {r['floor']:<5} {r['conf']:<5} {r['avg_pts']:<7} {r['avg_min']:<7} {r['legs']:<6} {r['odds']:<6} "
          f"{r['record']:<8} {r['hit']:<5.0f}% {r['roi']:<+7.0f}% {r['fire']:<6.0f}% {r['leg_acc']:<6.1f}% "
          f"{r['avg_legs']:<8.1f} ${r['pnl']:+,}")

# Deep dive on top config
if results_table:
    best = results_table[0]
    pr = best["parlay_results"]
    print(f"\n{'='*80}")
    print(f"BEST CONFIG DEEP DIVE")
    print(f"Floor: {best['floor']*100:.0f}% of L10avg | Confidence: {best['conf']}+ | "
          f"MinAvgPts: {best['avg_pts']} | MinAvgMin: {best['avg_min']} | Legs: {best['legs']}")
    print(f"{'='*80}")

    wins = sum(1 for r in pr if r["won"])
    total = len(pr)
    print(f"  Record: {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"  Leg accuracy: {best['leg_tracker'][0]}/{best['leg_tracker'][1]} ({best['leg_acc']:.1f}%)")
    print(f"  Avg legs: {best['avg_legs']:.1f} | Avg payout: +{best['avg_odds']:.0f}")
    print(f"  P&L: ${best['pnl']:+,} | ROI: {best['roi']:+.1f}%")

    monthly = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0})
    for r in pr:
        m = r["date"][:6]
        monthly[m]["pnl"] += r["pnl"]
        if r["won"]: monthly[m]["w"] += 1
        else: monthly[m]["l"] += 1

    print(f"\n  Monthly:")
    all_pos = True
    for m in sorted(monthly.keys()):
        d = monthly[m]
        t = d["w"] + d["l"]
        if d["pnl"] <= 0: all_pos = False
        print(f"    {m}: {d['w']}-{d['l']} ({d['w']/t*100:.1f}%) PnL: ${d['pnl']:+,}")
    print(f"  All months profitable: {'YES' if all_pos else 'NO'}")

    print(f"\n  Game-by-game:")
    for r in pr:
        legs_str = " + ".join(f"{l['player']} O{l['floor']}({l['actual']}){'✓' if l['hit'] else '✗'}" for l in r["legs"][:3])
        if len(r["legs"]) > 3:
            legs_str += f" +{len(r['legs'])-3}more"
        res = "WIN" if r["won"] else "LOSS"
        print(f"    {r['date']} [{r['num_legs']}L +{r['odds']}] {res} ${r['pnl']:+d}: {legs_str}")

    # Sensitivity
    print(f"\n  Sensitivity (same picks, different odds):")
    for test_odds in [-200, -250, -300, -350, -400, -500, -600]:
        tp = sum(parlay_american([test_odds]*r["num_legs"]) if r["won"] else -100 for r in pr)
        print(f"    At {test_odds}/leg: P&L ${tp:+,} | ROI {tp/(len(pr)*100)*100:+.0f}%")

    # Find the 2nd and 3rd best configs that are substantially different
    print(f"\n{'='*80}")
    print("RUNNER-UP CONFIGS (different parameters):")
    shown = 0
    seen_floors = {best["floor"]}
    for r in results_table[1:]:
        if r["floor"] in seen_floors and abs(r["conf"] - best["conf"]) < 0.5:
            continue
        seen_floors.add(r["floor"])
        print(f"  Floor:{r['floor']*100:.0f}% Conf:{r['conf']} Legs:{r['legs']} | "
              f"{r['record']} ({r['hit']:.0f}%) | ROI:{r['roi']:+.0f}% | Fire:{r['fire']:.0f}% | "
              f"P&L:${r['pnl']:+,}")
        shown += 1
        if shown >= 3:
            break
