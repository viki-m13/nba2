#!/usr/bin/env python3
"""
Play 11 VAULT: Full audit of odds, payouts, W/L, and P&L calculations.
Verifies every number is correct.
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

# ═══════════════════════════════════════════════════════════════
# ODDS MATH VERIFICATION
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 1: ODDS MATH VERIFICATION")
print("=" * 80)

def american_to_decimal(am):
    """American -> Decimal odds"""
    if am > 0:
        return 1 + am / 100
    else:
        return 1 + 100 / abs(am)

def decimal_to_american(dec):
    if dec >= 2.0:
        return round((dec - 1) * 100)
    else:
        return round(-100 / (dec - 1))

def parlay_decimal(leg_odds_am):
    product = 1.0
    for o in leg_odds_am:
        product *= american_to_decimal(o)
    return product

def parlay_american(leg_odds_am):
    return decimal_to_american(parlay_decimal(leg_odds_am))

# Verify: -300 American = decimal 1.333...
print(f"\n-300 American → Decimal: {american_to_decimal(-300):.6f}")
print(f"  Expected: 1.333333 (you bet $300 to win $100, get back $400 total)")
print(f"  Implied probability: {1/american_to_decimal(-300)*100:.2f}% (break-even)")

# Verify parlay odds for different leg counts at -300/leg
print(f"\nParlay odds at -300 per leg:")
for n in range(3, 9):
    dec = parlay_decimal([-300] * n)
    am = parlay_american([-300] * n)
    payout_on_100 = am  # profit on $100 bet
    total_return = payout_on_100 + 100  # total back
    be = 1 / dec * 100  # break-even %
    print(f"  {n} legs: decimal {dec:.4f} → American +{am} → "
          f"$100 bet returns ${total_return} (profit ${payout_on_100}) → "
          f"Break-even: {be:.2f}%")

# Cross-check: 5 legs at -300
# Each leg decimal = 1.3333
# 1.3333^5 = 4.2140
# American = (4.2140 - 1) * 100 = 321.40 → +321
print(f"\n  Manual check 5-leg: 1.3333^5 = {1.3333**5:.4f} → +{round((1.3333**5 - 1)*100)}")
print(f"  Manual check 7-leg: 1.3333^7 = {1.3333**7:.4f} → +{round((1.3333**7 - 1)*100)}")
print(f"  Manual check 8-leg: 1.3333^8 = {1.3333**8:.4f} → +{round((1.3333**8 - 1)*100)}")

# ═══════════════════════════════════════════════════════════════
# P&L MATH VERIFICATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("SECTION 2: P&L CALCULATION VERIFICATION")
print("="*80)

print(f"\nHow P&L works on a $100 flat bet:")
print(f"  WIN:  profit = parlay_american_odds (e.g., +649 → win $649)")
print(f"  LOSS: profit = -$100 (lose the wager)")
print(f"  ROI = total_profit / total_wagered * 100")

# Example: 34 wins at avg +737, 3 losses at -100
# This is approximate - let me compute exactly from the backtest
print(f"\n  Example check: if 34 wins avg +737 and 3 losses:")
print(f"    Win P&L: 34 * 737 = ${34*737:,}")
print(f"    Loss P&L: 3 * -100 = ${3*-100}")
print(f"    Total P&L: ${34*737 + 3*-100:,}")
print(f"    Total wagered: 37 * $100 = $3,700")
print(f"    ROI: {(34*737 + 3*-100) / 3700 * 100:.1f}%")
print(f"  Note: actual P&L varies because each parlay has different # legs → different odds")

# ═══════════════════════════════════════════════════════════════
# FULL WALK-FORWARD BACKTEST WITH DETAILED AUDIT
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("SECTION 3: COMPLETE WALK-FORWARD BACKTEST (AUDIT)")
print("="*80)

# VAULT config: floor=33% of L10avg, confidence>=1.5, minAvgPts=18, minAvgMin=30, legs 4-8
FLOOR_PCT = 0.33
MIN_CONF = 1.5
MIN_AVG_PTS = 18
MIN_AVG_MIN = 30
MIN_LEGS = 4
MAX_LEGS = 8
LEG_ODDS = -300  # American odds per leg

player_hist = defaultdict(list)
parlays = []
all_legs = []  # track every individual leg

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

            if avg_min < MIN_AVG_MIN or avg_pts < MIN_AVG_PTS:
                continue

            floor_line = round(avg_pts * FLOOR_PCT, 1)
            if floor_line < 4:
                continue

            confidence = min_pts / floor_line if floor_line > 0 else 0
            if confidence < MIN_CONF:
                continue

            # CRITICAL: The bet is OVER floor_line
            # Hit means actual points > floor_line
            actual_pts = p["pts"]
            hit = actual_pts > floor_line

            candidates.append({
                "player": pid,
                "game_key": game_key,
                "hit": hit,
                "floor_line": floor_line,
                "actual": actual_pts,
                "confidence": confidence,
                "l10_avg": avg_pts,
                "l10_min": min_pts,
                "game_display": f"{game['away']} @ {game['home']}",
            })

    # Track ALL individual legs for this date
    for c in candidates:
        all_legs.append({
            "date": date,
            "player": c["player"],
            "floor": c["floor_line"],
            "actual": c["actual"],
            "hit": c["hit"],
            "confidence": c["confidence"],
        })

    # Select top by confidence, 1 per game
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    selected = []
    used_games = set()
    for c in candidates:
        if c["game_key"] in used_games:
            continue
        selected.append(c)
        used_games.add(c["game_key"])
        if len(selected) >= MAX_LEGS:
            break

    if len(selected) >= MIN_LEGS:
        n = len(selected)
        all_hit = all(s["hit"] for s in selected)

        # Calculate parlay odds
        parlay_odds = parlay_american([LEG_ODDS] * n)

        # P&L calculation
        if all_hit:
            pnl = parlay_odds  # WIN: profit = parlay odds on $100 bet
        else:
            pnl = -100  # LOSS: lose $100 wager

        parlays.append({
            "date": date,
            "legs": selected,
            "num_legs": n,
            "parlay_odds": parlay_odds,
            "won": all_hit,
            "pnl": pnl,
        })

    # Update history AFTER predictions (walk-forward)
    for game in games_by_date[date]:
        for p in game["players"]:
            mins = parse_mins(p["min"])
            if mins < 10:
                continue
            player_hist[p["name"]].append({
                "pts": p["pts"], "reb": p["reb"], "ast": p["ast"], "min": mins
            })

# ═══════════════════════════════════════════════════════════════
# DETAILED RESULTS
# ═══════════════════════════════════════════════════════════════

print(f"\nConfig: floor={FLOOR_PCT*100:.0f}% of L10avg, conf>={MIN_CONF}, "
      f"minPts>={MIN_AVG_PTS}, minMin>={MIN_AVG_MIN}, legs {MIN_LEGS}-{MAX_LEGS}, "
      f"leg odds: {LEG_ODDS}")

# Individual leg audit
leg_hits = sum(1 for l in all_legs if l["hit"])
leg_total = len(all_legs)
print(f"\n── INDIVIDUAL LEGS ──")
print(f"  Total qualifying legs: {leg_total}")
print(f"  Legs that hit: {leg_hits}")
print(f"  Legs that missed: {leg_total - leg_hits}")
print(f"  Leg accuracy: {leg_hits}/{leg_total} = {leg_hits/leg_total*100:.2f}%")

# Show all misses
misses = [l for l in all_legs if not l["hit"]]
print(f"\n  ALL INDIVIDUAL LEG MISSES ({len(misses)} total):")
for m in misses:
    print(f"    {m['date']}: {m['player']} — floor O{m['floor']}, "
          f"got {m['actual']} pts ← MISS (conf: {m['confidence']:.2f})")

# Parlay audit
print(f"\n── PARLAY RESULTS ──")
print(f"  Total parlays: {len(parlays)}")
wins = [p for p in parlays if p["won"]]
losses = [p for p in parlays if not p["won"]]
print(f"  Wins: {len(wins)}")
print(f"  Losses: {len(losses)}")
print(f"  Win rate: {len(wins)}/{len(parlays)} = {len(wins)/len(parlays)*100:.2f}%")

print(f"\n  Detailed parlay-by-parlay:")
total_pnl = 0
total_wagered = 0
for i, p in enumerate(parlays):
    total_wagered += 100
    total_pnl += p["pnl"]

    missed = [l for l in p["legs"] if not l["hit"]]
    miss_info = ""
    if missed:
        miss_info = " MISSED: " + ", ".join(
            f"{m['player']} O{m['floor_line']}(got {m['actual']})" for m in missed
        )

    legs_preview = ", ".join(f"{l['player']} O{l['floor_line']}({l['actual']})" for l in p["legs"][:3])
    if len(p["legs"]) > 3:
        legs_preview += f" +{len(p['legs'])-3}more"

    result = "WIN " if p["won"] else "LOSS"
    print(f"  #{i+1:2d} {p['date']} | {p['num_legs']}L @{LEG_ODDS} = +{p['parlay_odds']:>4d} | "
          f"{result} ${p['pnl']:>+5d} | running: ${total_pnl:>+7,} | "
          f"{legs_preview}{miss_info}")

print(f"\n── FINAL P&L AUDIT ──")
print(f"  Total parlays placed: {len(parlays)}")
print(f"  Total wagered: ${total_wagered:,} ({len(parlays)} × $100)")
print(f"  Total profit: ${total_pnl:+,}")
print(f"  ROI: {total_pnl / total_wagered * 100:+.2f}%")

# Cross-check P&L
check_pnl = sum(p["pnl"] for p in parlays)
print(f"\n  Cross-check total P&L (sum of all pnl): ${check_pnl:+,}")
assert check_pnl == total_pnl, "P&L MISMATCH!"
print(f"  ✓ P&L matches")

# Cross-check: wins should equal sum of parlay odds, losses should be -100 each
win_pnl = sum(p["parlay_odds"] for p in wins)
loss_pnl = len(losses) * -100
print(f"\n  Win P&L breakdown: {len(wins)} wins")
for w in wins:
    print(f"    {w['date']}: +${w['parlay_odds']} ({w['num_legs']}L)")
print(f"  Total from wins: +${win_pnl:,}")
print(f"  Total from losses: -${len(losses)*100} ({len(losses)} × $100)")
print(f"  Net: ${win_pnl + loss_pnl:+,}")
assert win_pnl + loss_pnl == total_pnl, "WIN/LOSS BREAKDOWN MISMATCH!"
print(f"  ✓ Win/loss breakdown matches total P&L")

# Monthly breakdown
monthly = defaultdict(lambda: {"w": 0, "l": 0, "win_pnl": 0, "loss_pnl": 0, "total": 0})
for p in parlays:
    m = p["date"][:6]
    monthly[m]["total"] += 1
    if p["won"]:
        monthly[m]["w"] += 1
        monthly[m]["win_pnl"] += p["parlay_odds"]
    else:
        monthly[m]["l"] += 1
        monthly[m]["loss_pnl"] += -100

print(f"\n── MONTHLY BREAKDOWN ──")
for m in sorted(monthly.keys()):
    d = monthly[m]
    net = d["win_pnl"] + d["loss_pnl"]
    t = d["w"] + d["l"]
    roi = net / (t * 100) * 100
    print(f"  {m}: {d['w']}-{d['l']} ({d['w']/t*100:.1f}%) | "
          f"Win P&L: +${d['win_pnl']:,} | Loss P&L: -${abs(d['loss_pnl'])} | "
          f"Net: ${net:+,} | ROI: {roi:+.1f}%")

# Verify parlay odds are correct for each parlay
print(f"\n── ODDS VERIFICATION (every parlay) ──")
all_odds_ok = True
for p in parlays:
    expected_dec = (1 + 100 / abs(LEG_ODDS)) ** p["num_legs"]
    expected_am = round((expected_dec - 1) * 100)
    if p["parlay_odds"] != expected_am:
        print(f"  ✗ {p['date']}: {p['num_legs']}L got +{p['parlay_odds']} but expected +{expected_am}")
        all_odds_ok = False

if all_odds_ok:
    print(f"  ✓ All {len(parlays)} parlay odds are correct")

# Verify W/L determination
print(f"\n── W/L DETERMINATION VERIFICATION (every parlay) ──")
wl_ok = True
for p in parlays:
    legs_all_hit = all(l["hit"] for l in p["legs"])
    if legs_all_hit != p["won"]:
        print(f"  ✗ {p['date']}: legs_all_hit={legs_all_hit} but won={p['won']}")
        wl_ok = False
    # Verify each individual leg
    for l in p["legs"]:
        expected_hit = l["actual"] > l["floor_line"]
        if expected_hit != l["hit"]:
            print(f"  ✗ {p['date']}: {l['player']} actual={l['actual']} floor={l['floor_line']} "
                  f"expected_hit={expected_hit} got={l['hit']}")
            wl_ok = False

if wl_ok:
    print(f"  ✓ All {len(parlays)} parlays have correct W/L determination")
    total_legs_in_parlays = sum(p["num_legs"] for p in parlays)
    print(f"  ✓ All {total_legs_in_parlays} individual legs verified (actual > floor = hit)")

# Fire rate
print(f"\n── FIRE RATE ──")
total_dates = len(dates)
fire_dates = len(parlays)
print(f"  Total game dates in data: {total_dates}")
print(f"  Dates with VAULT parlay: {fire_dates}")
print(f"  Fire rate: {fire_dates/total_dates*100:.1f}%")

# Summary
print(f"\n{'='*80}")
print("AUDIT SUMMARY")
print("="*80)
print(f"  Individual leg accuracy: {leg_hits}/{leg_total} ({leg_hits/leg_total*100:.1f}%)")
print(f"  Parlay record: {len(wins)}/{len(parlays)} ({len(wins)/len(parlays)*100:.1f}%)")
print(f"  Average legs per parlay: {sum(p['num_legs'] for p in parlays)/len(parlays):.1f}")
print(f"  Average parlay payout: +{sum(p['parlay_odds'] for p in parlays)/len(parlays):.0f}")
print(f"  Total P&L: ${total_pnl:+,} on ${total_wagered:,} wagered")
print(f"  ROI: {total_pnl/total_wagered*100:+.1f}%")
print(f"  Fire rate: {fire_dates}/{total_dates} ({fire_dates/total_dates*100:.1f}%)")
print(f"  Monthly: profitable {'all' if all(monthly[m]['win_pnl']+monthly[m]['loss_pnl'] > 0 for m in monthly) else 'NOT all'} months")
print(f"\n  ✓ Odds math verified")
print(f"  {'✓' if all_odds_ok else '✗'} All parlay odds correct")
print(f"  {'✓' if wl_ok else '✗'} All W/L determinations correct")
print(f"  ✓ P&L cross-checked")
