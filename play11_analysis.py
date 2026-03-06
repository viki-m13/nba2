#!/usr/bin/env python3
"""
Play 11 Analysis: High-Accuracy Multi-Leg Player Prop Parlays
Goal: 3-10 legs, 90%+ parlay accuracy, fires almost every night, high payout
Each leg can be less favorable than -110 (e.g., -130, -200, -300)
"""

import json
from collections import defaultdict
import math

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

def american_to_decimal(american):
    """Convert American odds to decimal odds"""
    if american > 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)

def decimal_to_american(decimal):
    """Convert decimal odds to American"""
    if decimal >= 2.0:
        return round((decimal - 1) * 100)
    else:
        return round(-100 / (decimal - 1))

def parlay_decimal_odds(leg_odds_american):
    """Calculate parlay decimal odds from list of American odds per leg"""
    product = 1.0
    for odds in leg_odds_american:
        product *= american_to_decimal(odds)
    return product

def parlay_american(leg_odds_american):
    """Calculate parlay American odds"""
    dec = parlay_decimal_odds(leg_odds_american)
    return decimal_to_american(dec)

dates = sorted(games_by_date.keys())

# ═══════════════════════════════════════════════════════════════
# SIGNAL EXPLORATION: Find ultra-high accuracy player prop signals
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("PLAY 11 SIGNAL EXPLORATION: Ultra-High Accuracy Player Props")
print("=" * 80)

# Track player histories for walk-forward
player_hist = defaultdict(list)

# Signal definitions - each returns (hit, odds_american) or None
signals_results = defaultdict(lambda: {"hits": 0, "total": 0, "by_date": defaultdict(list)})

for date in dates:
    day_candidates = defaultdict(list)  # signal_name -> list of candidates

    for game in games_by_date[date]:
        game_key = f"{game['away']}@{game['home']}"

        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 1:
                continue

            pid = p["name"]
            hist = player_hist.get(pid, [])

            pts = p["pts"]
            reb = p["reb"]
            ast = p["ast"]
            pra = pts + reb + ast

            # ── SIGNAL 1: Starter (25+ min history) scores OVER 4.5 pts ──
            # Very low bar for any starter
            if len(hist) >= 5:
                recent_mins = [h["min"] for h in hist[-5:]]
                avg_mins = sum(recent_mins) / len(recent_mins)
                avg_pts = sum(h["pts"] for h in hist[-5:]) / 5

                if avg_mins >= 25 and avg_pts >= 12:
                    hit = pts > 4.5
                    signals_results["starter_over_4.5pts"]["total"] += 1
                    if hit:
                        signals_results["starter_over_4.5pts"]["hits"] += 1
                    day_candidates["starter_over_4.5pts"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "avg_pts": avg_pts, "avg_mins": avg_mins, "actual": pts
                    })

            # ── SIGNAL 2: Starter OVER 7.5 pts ──
            if len(hist) >= 5:
                recent_mins = [h["min"] for h in hist[-5:]]
                avg_mins = sum(recent_mins) / len(recent_mins)
                avg_pts = sum(h["pts"] for h in hist[-5:]) / 5

                if avg_mins >= 28 and avg_pts >= 16:
                    hit = pts > 7.5
                    signals_results["star_over_7.5pts"]["total"] += 1
                    if hit:
                        signals_results["star_over_7.5pts"]["hits"] += 1
                    day_candidates["star_over_7.5pts"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "avg_pts": avg_pts, "avg_mins": avg_mins, "actual": pts
                    })

            # ── SIGNAL 3: Star OVER 9.5 pts ──
            if len(hist) >= 5:
                recent_mins = [h["min"] for h in hist[-5:]]
                avg_mins = sum(recent_mins) / len(recent_mins)
                avg_pts = sum(h["pts"] for h in hist[-5:]) / 5

                if avg_mins >= 30 and avg_pts >= 20:
                    hit = pts > 9.5
                    signals_results["star_over_9.5pts"]["total"] += 1
                    if hit:
                        signals_results["star_over_9.5pts"]["hits"] += 1
                    day_candidates["star_over_9.5pts"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "avg_pts": avg_pts, "avg_mins": avg_mins, "actual": pts
                    })

            # ── SIGNAL 4: Star OVER 11.5 pts ──
            if len(hist) >= 5:
                recent_mins = [h["min"] for h in hist[-5:]]
                avg_mins = sum(recent_mins) / len(recent_mins)
                avg_pts = sum(h["pts"] for h in hist[-5:]) / 5

                if avg_mins >= 30 and avg_pts >= 22:
                    hit = pts > 11.5
                    signals_results["star_over_11.5pts"]["total"] += 1
                    if hit:
                        signals_results["star_over_11.5pts"]["hits"] += 1
                    day_candidates["star_over_11.5pts"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "avg_pts": avg_pts, "avg_mins": avg_mins, "actual": pts
                    })

            # ── SIGNAL 5: Star OVER 14.5 pts ──
            if len(hist) >= 5:
                recent_mins = [h["min"] for h in hist[-5:]]
                avg_mins = sum(recent_mins) / len(recent_mins)
                avg_pts = sum(h["pts"] for h in hist[-5:]) / 5

                if avg_mins >= 32 and avg_pts >= 24:
                    hit = pts > 14.5
                    signals_results["elite_over_14.5pts"]["total"] += 1
                    if hit:
                        signals_results["elite_over_14.5pts"]["hits"] += 1
                    day_candidates["elite_over_14.5pts"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "avg_pts": avg_pts, "avg_mins": avg_mins, "actual": pts
                    })

            # ── SIGNAL 6: PRA (pts+reb+ast) OVER floors for stars ──
            if len(hist) >= 5:
                recent_mins = [h["min"] for h in hist[-5:]]
                avg_mins = sum(recent_mins) / len(recent_mins)
                avg_pra = sum(h["pts"] + h["reb"] + h["ast"] for h in hist[-5:]) / 5

                if avg_mins >= 28 and avg_pra >= 25:
                    hit = pra > 9.5
                    signals_results["starter_pra_over_9.5"]["total"] += 1
                    if hit:
                        signals_results["starter_pra_over_9.5"]["hits"] += 1
                    day_candidates["starter_pra_over_9.5"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "avg_pra": avg_pra, "actual": pra
                    })

            # ── SIGNAL 7: Starter OVER 0.5 assists ──
            if len(hist) >= 5:
                recent_mins = [h["min"] for h in hist[-5:]]
                avg_mins = sum(recent_mins) / len(recent_mins)
                avg_ast = sum(h["ast"] for h in hist[-5:]) / 5

                if avg_mins >= 25 and avg_ast >= 3:
                    hit = ast > 0.5
                    signals_results["starter_over_0.5ast"]["total"] += 1
                    if hit:
                        signals_results["starter_over_0.5ast"]["hits"] += 1
                    day_candidates["starter_over_0.5ast"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "avg_ast": avg_ast, "actual": ast
                    })

            # ── SIGNAL 8: Starter OVER 0.5 rebounds ──
            if len(hist) >= 5:
                recent_mins = [h["min"] for h in hist[-5:]]
                avg_mins = sum(recent_mins) / len(recent_mins)
                avg_reb = sum(h["reb"] for h in hist[-5:]) / 5

                if avg_mins >= 25 and avg_reb >= 4:
                    hit = reb > 0.5
                    signals_results["starter_over_0.5reb"]["total"] += 1
                    if hit:
                        signals_results["starter_over_0.5reb"]["hits"] += 1
                    day_candidates["starter_over_0.5reb"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "avg_reb": avg_reb, "actual": reb
                    })

            # ── SIGNAL 9: Consistent scorer - L10 min >= 10pts, bet OVER 5.5 ──
            if len(hist) >= 10:
                l10 = hist[-10:]
                l10_pts = [h["pts"] for h in l10]
                l10_min_pts = min(l10_pts)
                l10_avg = sum(l10_pts) / 10
                recent_mins_avg = sum(h["min"] for h in l10) / 10

                if recent_mins_avg >= 25 and l10_min_pts >= 10 and l10_avg >= 18:
                    hit = pts > 5.5
                    signals_results["consistent_over_5.5"]["total"] += 1
                    if hit:
                        signals_results["consistent_over_5.5"]["hits"] += 1
                    day_candidates["consistent_over_5.5"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l10_min": l10_min_pts, "l10_avg": l10_avg, "actual": pts
                    })

            # ── SIGNAL 10: Consistent scorer - L10 min >= 10pts, bet OVER 7.5 ──
            if len(hist) >= 10:
                l10 = hist[-10:]
                l10_pts = [h["pts"] for h in l10]
                l10_min_pts = min(l10_pts)
                l10_avg = sum(l10_pts) / 10
                recent_mins_avg = sum(h["min"] for h in l10) / 10

                if recent_mins_avg >= 25 and l10_min_pts >= 10 and l10_avg >= 18:
                    hit = pts > 7.5
                    signals_results["consistent_over_7.5"]["total"] += 1
                    if hit:
                        signals_results["consistent_over_7.5"]["hits"] += 1
                    day_candidates["consistent_over_7.5"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l10_min": l10_min_pts, "l10_avg": l10_avg, "actual": pts
                    })

            # ── SIGNAL 11: Consistent scorer - L5 all >= 12pts, bet OVER 9.5 ──
            if len(hist) >= 5:
                l5 = hist[-5:]
                l5_pts = [h["pts"] for h in l5]
                l5_min_pts = min(l5_pts)
                l5_avg = sum(l5_pts) / 5

                if l5_min_pts >= 12 and l5_avg >= 20:
                    hit = pts > 9.5
                    signals_results["hot_streak_over_9.5"]["total"] += 1
                    if hit:
                        signals_results["hot_streak_over_9.5"]["hits"] += 1
                    day_candidates["hot_streak_over_9.5"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l5_min": l5_min_pts, "l5_avg": l5_avg, "actual": pts
                    })

            # ── SIGNAL 12: Player averaging 20+ in L10 -> OVER 9.5 ──
            if len(hist) >= 10:
                l10 = hist[-10:]
                l10_avg = sum(h["pts"] for h in l10) / 10
                recent_mins_avg = sum(h["min"] for h in l10) / 10

                if recent_mins_avg >= 30 and l10_avg >= 20:
                    hit = pts > 9.5
                    signals_results["avg20_over_9.5"]["total"] += 1
                    if hit:
                        signals_results["avg20_over_9.5"]["hits"] += 1
                    day_candidates["avg20_over_9.5"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l10_avg": l10_avg, "actual": pts
                    })

            # ── SIGNAL 13: Player with L10 PRA avg >= 30 -> PRA OVER 14.5 ──
            if len(hist) >= 10:
                l10 = hist[-10:]
                l10_pra = sum(h["pts"] + h["reb"] + h["ast"] for h in l10) / 10
                recent_mins_avg = sum(h["min"] for h in l10) / 10

                if recent_mins_avg >= 28 and l10_pra >= 30:
                    hit = pra > 14.5
                    signals_results["pra30_over_14.5"]["total"] += 1
                    if hit:
                        signals_results["pra30_over_14.5"]["hits"] += 1
                    day_candidates["pra30_over_14.5"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l10_pra": l10_pra, "actual": pra
                    })

            # ── SIGNAL 14: Consistent L5 all >= 15pts -> OVER 11.5 ──
            if len(hist) >= 5:
                l5 = hist[-5:]
                l5_pts = [h["pts"] for h in l5]
                l5_min_pts = min(l5_pts)
                l5_avg = sum(l5_pts) / 5

                if l5_min_pts >= 15 and l5_avg >= 22:
                    hit = pts > 11.5
                    signals_results["hot15_over_11.5"]["total"] += 1
                    if hit:
                        signals_results["hot15_over_11.5"]["hits"] += 1
                    day_candidates["hot15_over_11.5"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l5_min": l5_min_pts, "l5_avg": l5_avg, "actual": pts
                    })

            # ── SIGNAL 15: Player L5 avg PRA >= 35, OVER 19.5 PRA ──
            if len(hist) >= 5:
                l5 = hist[-5:]
                l5_pra = sum(h["pts"] + h["reb"] + h["ast"] for h in l5) / 5
                l5_min_pra = min(h["pts"] + h["reb"] + h["ast"] for h in l5)

                if l5_pra >= 35 and l5_min_pra >= 20:
                    hit = pra > 19.5
                    signals_results["pra35_over_19.5"]["total"] += 1
                    if hit:
                        signals_results["pra35_over_19.5"]["hits"] += 1
                    day_candidates["pra35_over_19.5"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l5_pra": l5_pra, "actual": pra
                    })

            # ── SIGNAL 16: Dynamic floor - OVER (L10_avg * 0.35) ──
            if len(hist) >= 10:
                l10 = hist[-10:]
                l10_avg = sum(h["pts"] for h in l10) / 10
                recent_mins_avg = sum(h["min"] for h in l10) / 10
                floor_line = round(l10_avg * 0.35, 1)

                if recent_mins_avg >= 28 and l10_avg >= 18 and floor_line >= 5:
                    hit = pts > floor_line
                    signals_results["dynamic_floor_35pct"]["total"] += 1
                    if hit:
                        signals_results["dynamic_floor_35pct"]["hits"] += 1
                    day_candidates["dynamic_floor_35pct"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l10_avg": l10_avg, "floor": floor_line, "actual": pts
                    })

            # ── SIGNAL 17: Dynamic floor - OVER (L10_avg * 0.4) ──
            if len(hist) >= 10:
                l10 = hist[-10:]
                l10_avg = sum(h["pts"] for h in l10) / 10
                recent_mins_avg = sum(h["min"] for h in l10) / 10
                floor_line = round(l10_avg * 0.4, 1)

                if recent_mins_avg >= 28 and l10_avg >= 18 and floor_line >= 5:
                    hit = pts > floor_line
                    signals_results["dynamic_floor_40pct"]["total"] += 1
                    if hit:
                        signals_results["dynamic_floor_40pct"]["hits"] += 1
                    day_candidates["dynamic_floor_40pct"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l10_avg": l10_avg, "floor": floor_line, "actual": pts
                    })

            # ── SIGNAL 18: Dynamic floor - OVER (L10_avg * 0.45) ──
            if len(hist) >= 10:
                l10 = hist[-10:]
                l10_avg = sum(h["pts"] for h in l10) / 10
                recent_mins_avg = sum(h["min"] for h in l10) / 10
                floor_line = round(l10_avg * 0.45, 1)

                if recent_mins_avg >= 28 and l10_avg >= 18 and floor_line >= 5:
                    hit = pts > floor_line
                    signals_results["dynamic_floor_45pct"]["total"] += 1
                    if hit:
                        signals_results["dynamic_floor_45pct"]["hits"] += 1
                    day_candidates["dynamic_floor_45pct"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l10_avg": l10_avg, "floor": floor_line, "actual": pts
                    })

            # ── SIGNAL 19: Dynamic floor - OVER (L10_avg * 0.5) ──
            if len(hist) >= 10:
                l10 = hist[-10:]
                l10_avg = sum(h["pts"] for h in l10) / 10
                recent_mins_avg = sum(h["min"] for h in l10) / 10
                floor_line = round(l10_avg * 0.5, 1)

                if recent_mins_avg >= 28 and l10_avg >= 18 and floor_line >= 5:
                    hit = pts > floor_line
                    signals_results["dynamic_floor_50pct"]["total"] += 1
                    if hit:
                        signals_results["dynamic_floor_50pct"]["hits"] += 1
                    day_candidates["dynamic_floor_50pct"].append({
                        "player": pid, "hit": hit, "game_key": game_key,
                        "l10_avg": l10_avg, "floor": floor_line, "actual": pts
                    })

    # Track daily availability for each signal
    for sig_name, cands in day_candidates.items():
        signals_results[sig_name]["by_date"][date] = cands

    # Update history
    for game in games_by_date[date]:
        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 10:
                continue
            player_hist[p["name"]].append({
                "pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "min": mins
            })

# Print signal results
print(f"\n{'Signal':<30} {'Accuracy':<12} {'Total':<8} {'Avail Days':<12} {'Avg/Day':<8}")
print("-" * 75)
for sig_name in sorted(signals_results.keys(), key=lambda s: signals_results[s]["hits"]/max(signals_results[s]["total"],1), reverse=True):
    s = signals_results[sig_name]
    if s["total"] == 0:
        continue
    acc = s["hits"] / s["total"] * 100
    days_avail = len(s["by_date"])
    avg_per_day = s["total"] / max(days_avail, 1)
    print(f"  {sig_name:<28} {acc:<11.1f}% {s['total']:<8} {days_avail:<12} {avg_per_day:<7.1f}")

# ═══════════════════════════════════════════════════════════════
# PARLAY SIMULATION: Best signals combined
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PARLAY SIMULATION: Combining high-accuracy signals")
print("=" * 80)

# For the parlay, we want signals with 90%+ individual accuracy
# and enough volume to fire most nights
# We'll test various combinations

# First, let's identify which signals have 90%+ accuracy
high_acc_signals = []
for sig_name, s in signals_results.items():
    if s["total"] > 50:  # Need meaningful sample
        acc = s["hits"] / s["total"] * 100
        if acc >= 88:  # 88%+ threshold
            high_acc_signals.append((sig_name, acc, s))

print(f"\nSignals with 88%+ accuracy and 50+ samples:")
for name, acc, s in sorted(high_acc_signals, key=lambda x: x[1], reverse=True):
    print(f"  {name}: {acc:.1f}% ({s['hits']}/{s['total']})")

# ═══════════════════════════════════════════════════════════════
# PARLAY TEST: Use best signal for multi-leg parlays
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("MULTI-LEG PARLAY BACKTESTS")
print("=" * 80)

# For each high-accuracy signal, simulate parlays
for sig_name, acc, sig_data in sorted(high_acc_signals, key=lambda x: x[1], reverse=True)[:5]:
    # Determine realistic odds for this accuracy level
    # Sportsbooks price high-probability props with heavy juice
    # 95%+ hit → odds around -500 to -800
    # 90-95% hit → odds around -300 to -500
    # 88-90% hit → odds around -200 to -300
    if acc >= 95:
        leg_odds = -500
    elif acc >= 93:
        leg_odds = -400
    elif acc >= 90:
        leg_odds = -300
    else:
        leg_odds = -200

    for num_legs in [3, 4, 5, 6]:
        parlay_wins = 0
        parlay_total = 0
        total_pnl = 0

        for date in sorted(sig_data["by_date"].keys()):
            cands = sig_data["by_date"][date]
            # Select from different games
            selected = []
            used_games = set()
            # Sort by strongest signal
            for c in cands:
                if c["game_key"] in used_games:
                    continue
                selected.append(c)
                used_games.add(c["game_key"])
                if len(selected) >= num_legs:
                    break

            if len(selected) >= num_legs:
                all_hit = all(s["hit"] for s in selected)
                parlay_odds_am = parlay_american([leg_odds] * num_legs)
                payout = parlay_odds_am  # per $100
                pnl = payout if all_hit else -100

                parlay_wins += 1 if all_hit else 0
                parlay_total += 1
                total_pnl += pnl

        if parlay_total > 0:
            parlays_odds = parlay_american([leg_odds] * num_legs)
            hit_rate = parlay_wins / parlay_total * 100
            roi = total_pnl / (parlay_total * 100) * 100
            be = 100 / (parlays_odds + 100) * 100 if parlays_odds > 0 else abs(parlays_odds) / (abs(parlays_odds) + 100) * 100

            flag = "✓✓✓" if roi > 30 else ("✓✓" if roi > 10 else ("✓" if roi > 0 else ""))
            print(f"  {sig_name} | {num_legs}-leg @{leg_odds} = +{parlays_odds} | "
                  f"{parlay_wins}/{parlay_total} ({hit_rate:.1f}%) | "
                  f"P&L: ${total_pnl:+,} | ROI: {roi:+.1f}% | BE: {be:.1f}% {flag}")

# ═══════════════════════════════════════════════════════════════
# MIXED SIGNAL PARLAYS: Combine different high-accuracy signals
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("MIXED SIGNAL PARLAY: Combine different signal types per night")
print("=" * 80)

# Reset player history for clean walk-forward
player_hist2 = defaultdict(list)

# We'll use the dynamic floor approach - pick legs where we're very confident
# For each player on each night, generate the best "floor" bet
# Use various stat types and thresholds

parlay_configs = [
    ("CONSERVATIVE", 0.35, -350, 4, 8),  # floor at 35% of avg, assume -350, 4-8 legs
    ("MODERATE", 0.40, -275, 4, 7),       # floor at 40% of avg
    ("BALANCED", 0.45, -225, 3, 6),       # floor at 45% of avg
    ("AGGRESSIVE", 0.50, -180, 3, 5),     # floor at 50% of avg
]

for config_name, floor_pct, leg_odds, min_legs, max_legs in parlay_configs:
    player_hist2 = defaultdict(list)
    parlay_results = []
    leg_tracker = {"hits": 0, "total": 0}

    for date in sorted(games_by_date.keys()):
        candidates = []

        for game in games_by_date[date]:
            game_key = f"{game['away']}@{game['home']}"

            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 1:
                    continue
                pid = p["name"]
                hist = player_hist2.get(pid, [])

                if len(hist) < 10:
                    continue

                l10 = hist[-10:]
                l10_avg_pts = sum(h["pts"] for h in l10) / 10
                l10_avg_min = sum(h["min"] for h in l10) / 10
                l10_min_pts = min(h["pts"] for h in l10)

                # Only consider established starters
                if l10_avg_min < 28 or l10_avg_pts < 15:
                    continue

                # Floor line
                floor_line = round(l10_avg_pts * floor_pct, 1)
                if floor_line < 5:
                    continue

                hit = p["pts"] > floor_line
                confidence = l10_min_pts / floor_line if floor_line > 0 else 0

                candidates.append({
                    "player": pid,
                    "game_key": game_key,
                    "hit": hit,
                    "floor_line": floor_line,
                    "actual": p["pts"],
                    "l10_avg": l10_avg_pts,
                    "l10_min": l10_min_pts,
                    "confidence": confidence,
                })

        # Track individual legs
        for c in candidates:
            leg_tracker["total"] += 1
            if c["hit"]:
                leg_tracker["hits"] += 1

        # Select best legs from different games, sorted by confidence
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
            actual_legs = len(selected)
            all_hit = all(s["hit"] for s in selected)
            odds = parlay_american([leg_odds] * actual_legs)
            pnl = odds if all_hit else -100

            parlay_results.append({
                "date": date, "won": all_hit, "pnl": pnl,
                "legs": selected, "num_legs": actual_legs, "odds": odds
            })

        # Update history
        for game in games_by_date[date]:
            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 10:
                    continue
                player_hist2[p["name"]].append({
                    "pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "min": mins
                })

    if not parlay_results:
        continue

    wins = sum(1 for r in parlay_results if r["won"])
    total = len(parlay_results)
    total_pnl = sum(r["pnl"] for r in parlay_results)
    wagered = total * 100
    roi = total_pnl / wagered * 100
    leg_acc = leg_tracker["hits"] / leg_tracker["total"] * 100 if leg_tracker["total"] > 0 else 0
    avg_legs = sum(r["num_legs"] for r in parlay_results) / total
    avg_odds = sum(r["odds"] for r in parlay_results) / total

    print(f"\n  {config_name} (floor={floor_pct*100:.0f}% of L10avg, @{leg_odds}/leg)")
    print(f"    Leg accuracy: {leg_tracker['hits']}/{leg_tracker['total']} ({leg_acc:.1f}%)")
    print(f"    Parlays: {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"    Avg legs: {avg_legs:.1f} | Avg odds: +{avg_odds:.0f}")
    print(f"    P&L: ${total_pnl:+,} | ROI: {roi:+.1f}%")
    print(f"    Fires: {total}/{len(dates)} nights ({total/len(dates)*100:.1f}%)")

    # Monthly
    monthly = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0})
    for r in parlay_results:
        m = r["date"][:6]
        monthly[m]["pnl"] += r["pnl"]
        if r["won"]:
            monthly[m]["w"] += 1
        else:
            monthly[m]["l"] += 1

    print(f"    Monthly:")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        t = d["w"] + d["l"]
        print(f"      {m}: {d['w']}-{d['l']} ({d['w']/t*100:.1f}%) PnL: ${d['pnl']:+,}")

# ═══════════════════════════════════════════════════════════════
# ADVANCED: Multi-stat mixed legs (pts, reb, ast floors)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("MULTI-STAT MIXED LEGS: Points + Rebounds + Assists floors")
print("=" * 80)

player_hist3 = defaultdict(list)
for floor_pct, leg_odds in [(0.35, -350), (0.40, -275), (0.45, -225)]:
    player_hist3 = defaultdict(list)
    parlay_results = []
    leg_tracker = {"hits": 0, "total": 0}

    for date in sorted(games_by_date.keys()):
        candidates = []

        for game in games_by_date[date]:
            game_key = f"{game['away']}@{game['home']}"

            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 1:
                    continue
                pid = p["name"]
                hist = player_hist3.get(pid, [])
                if len(hist) < 10:
                    continue

                l10 = hist[-10:]
                l10_avg_min = sum(h["min"] for h in l10) / 10
                if l10_avg_min < 25:
                    continue

                # Points floor
                l10_avg_pts = sum(h["pts"] for h in l10) / 10
                if l10_avg_pts >= 15:
                    floor = round(l10_avg_pts * floor_pct, 1)
                    if floor >= 4:
                        l10_min = min(h["pts"] for h in l10)
                        hit = p["pts"] > floor
                        confidence = l10_min / floor if floor > 0 else 0
                        candidates.append({
                            "player": pid, "stat": "PTS", "game_key": game_key,
                            "hit": hit, "floor": floor, "actual": p["pts"],
                            "confidence": confidence, "l10_avg": l10_avg_pts
                        })

                # Rebounds floor
                l10_avg_reb = sum(h["reb"] for h in l10) / 10
                if l10_avg_reb >= 6:
                    floor = round(l10_avg_reb * floor_pct, 1)
                    if floor >= 1.5:
                        l10_min = min(h["reb"] for h in l10)
                        hit = p["reb"] > floor
                        confidence = l10_min / floor if floor > 0 else 0
                        candidates.append({
                            "player": pid, "stat": "REB", "game_key": game_key,
                            "hit": hit, "floor": floor, "actual": p["reb"],
                            "confidence": confidence, "l10_avg": l10_avg_reb
                        })

                # Assists floor
                l10_avg_ast = sum(h["ast"] for h in l10) / 10
                if l10_avg_ast >= 4:
                    floor = round(l10_avg_ast * floor_pct, 1)
                    if floor >= 1:
                        l10_min = min(h["ast"] for h in l10)
                        hit = p["ast"] > floor
                        confidence = l10_min / floor if floor > 0 else 0
                        candidates.append({
                            "player": pid, "stat": "AST", "game_key": game_key,
                            "hit": hit, "floor": floor, "actual": p["ast"],
                            "confidence": confidence, "l10_avg": l10_avg_ast
                        })

        # Track leg accuracy
        for c in candidates:
            leg_tracker["total"] += 1
            if c["hit"]:
                leg_tracker["hits"] += 1

        # Select best legs: max 1 per player, from different games
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        selected = []
        used_players = set()
        used_games_count = defaultdict(int)  # Allow 2 legs per game max

        for c in candidates:
            if c["player"] in used_players:
                continue
            if used_games_count[c["game_key"]] >= 2:
                continue
            selected.append(c)
            used_players.add(c["player"])
            used_games_count[c["game_key"]] += 1
            if len(selected) >= 8:
                break

        if len(selected) >= 4:
            actual_legs = len(selected)
            all_hit = all(s["hit"] for s in selected)
            odds = parlay_american([leg_odds] * actual_legs)
            pnl = odds if all_hit else -100

            parlay_results.append({
                "date": date, "won": all_hit, "pnl": pnl,
                "legs": selected, "num_legs": actual_legs, "odds": odds
            })

        # Update history
        for game in games_by_date[date]:
            for p in game["players"]:
                mins = parse_mins(p["min"])
                if mins < 10:
                    continue
                player_hist3[p["name"]].append({
                    "pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "min": mins
                })

    if not parlay_results:
        continue

    wins = sum(1 for r in parlay_results if r["won"])
    total = len(parlay_results)
    total_pnl = sum(r["pnl"] for r in parlay_results)
    roi = total_pnl / (total * 100) * 100
    leg_acc = leg_tracker["hits"] / leg_tracker["total"] * 100
    avg_legs = sum(r["num_legs"] for r in parlay_results) / total

    print(f"\n  Mixed stats floor={floor_pct*100:.0f}% @{leg_odds}/leg")
    print(f"    Leg accuracy: {leg_tracker['hits']}/{leg_tracker['total']} ({leg_acc:.1f}%)")
    print(f"    Parlays: {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"    Avg legs: {avg_legs:.1f}")
    print(f"    P&L: ${total_pnl:+,} | ROI: {roi:+.1f}%")
    print(f"    Fires: {total}/{len(dates)} nights ({total/len(dates)*100:.1f}%)")

    monthly = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0})
    for r in parlay_results:
        m = r["date"][:6]
        monthly[m]["pnl"] += r["pnl"]
        if r["won"]: monthly[m]["w"] += 1
        else: monthly[m]["l"] += 1

    print(f"    Monthly:")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        t = d["w"] + d["l"]
        print(f"      {m}: {d['w']}-{d['l']} ({d['w']/t*100:.1f}%) PnL: ${d['pnl']:+,}")
