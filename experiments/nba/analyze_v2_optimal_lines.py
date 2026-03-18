#!/usr/bin/env python3
"""
V2 Analysis: Optimal line selection + floor detection + under bets.
Key insight: instead of evaluating ALL lines, find the BEST line per player.
Also explore under bets for ceiling detection.
"""
import sys, os, json, math
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_loader import load_nba_boxscores, load_nba_odds
from nba.strategy import NBAPlayerModel, _update_model, _safe_int
from shared.odds_math import american_to_decimal, implied_probability

box_scores = load_nba_boxscores()
odds_data = load_nba_odds()
box_scores = [g for g in box_scores if g.get('home','') not in ['STARS','WORLD'] and g.get('away','') not in ['STARS','WORLD']]

model = NBAPlayerModel()
dates_sorted = sorted(set(g['date'] for g in box_scores))
box_by_date = defaultdict(list)
for g in box_scores:
    box_by_date[g['date']].append(g)
odds_index = defaultdict(dict)
for o in odds_data:
    odds_index[o['date']][o['gameKey']] = o

stat_map = {'playerProps': 'pts', 'rebProps': 'reb', 'astProps': 'ast', 'praProps': 'pra'}

# For each player-stat-date: find the LOWEST plus-money over line
# and compare to player's floor
all_optimal = []

for date in dates_sorted:
    day_odds = odds_index.get(date, {})
    day_boxes = box_by_date.get(date, [])

    actuals = {}
    for bg in day_boxes:
        for p in bg.get('players', []):
            name = p.get('name', '')
            pts = _safe_int(p.get('pts', 0))
            reb = _safe_int(p.get('reb', 0))
            ast = _safe_int(p.get('ast', 0))
            actuals[name] = {'pts': pts, 'reb': reb, 'ast': ast, 'pra': pts + reb + ast}

    # Collect all lines per player-stat
    player_lines = defaultdict(list)  # (player, stat) -> [(line, odds)]

    for gk, odds_rec in day_odds.items():
        for prop_type, stat in stat_map.items():
            props = odds_rec.get(prop_type, {})
            for player, lines in props.items():
                for line_str, line_data in lines.items():
                    over_odds = line_data.get('overOdds', -999)
                    line = float(line_str)
                    player_lines[(player, stat)].append((line, over_odds))

    for (player, stat), lines_list in player_lines.items():
        profile = model.profiles.get(player)
        if not profile or len(profile['games']) < 10:
            continue

        games = profile['games']
        window = games[-20:]
        avg_min = sum(g['min'] for g in window) / len(window)
        if avg_min < 18:
            continue

        values = [g.get(stat, 0) for g in window]
        sorted_vals = sorted(values)
        p10 = sorted_vals[max(0, int(len(sorted_vals) * 0.1))]
        p25 = sorted_vals[max(0, int(len(sorted_vals) * 0.25))]
        median = sorted_vals[len(sorted_vals) // 2]
        p75 = sorted_vals[min(len(sorted_vals)-1, int(len(sorted_vals) * 0.75))]
        p90 = sorted_vals[min(len(sorted_vals)-1, int(len(sorted_vals) * 0.9))]
        mean = sum(values) / len(values)

        # Strategy A: Find LOWEST plus-money over line
        plus_money_lines = [(l, o) for l, o in lines_list if 100 <= o <= 400]
        if plus_money_lines:
            # Sort by line ascending (lowest line = safest bet)
            plus_money_lines.sort(key=lambda x: x[0])
            best_line, best_odds = plus_money_lines[0]

            hits = sum(1 for v in values if v > best_line)
            hit_rate = hits / len(values)
            clearance = p10 - best_line
            clearance_25 = p25 - best_line

            actual = actuals.get(player, {}).get(stat)
            hit = actual > best_line if actual is not None else None

            # Multi-window hit rates
            hr_10 = sum(1 for v in values[-10:] if v > best_line) / min(10, len(values))
            hr_5 = sum(1 for v in values[-5:] if v > best_line) / min(5, len(values))

            all_optimal.append({
                'date': date, 'player': player, 'stat': stat,
                'line': best_line, 'odds': best_odds,
                'hit_rate': hit_rate, 'hr_10': hr_10, 'hr_5': hr_5,
                'clearance': clearance, 'clearance_25': clearance_25,
                'p10': p10, 'p25': p25, 'median': median, 'mean': mean,
                'hit': hit, 'direction': 'over',
                'n_games': len(values),
            })

        # Strategy B: Find LOWEST plus-money UNDER line (highest line where under is plus-money)
        # Under odds aren't in our data typically, but we can infer:
        # if over is very negative (like -300), under might be plus money
        # Actually, we only have overOdds. Let's look for lines where OVER is very
        # negative, implying UNDER is plus money. But we don't have under odds directly.
        # Skip for now.

    for bg in day_boxes:
        _update_model(model, bg, date)

# Analysis
sigs = [s for s in all_optimal if s['hit'] is not None]
print(f"{'='*70}")
print(f"OPTIMAL LINE ANALYSIS — Lowest Plus-Money Over Line Per Player")
print(f"{'='*70}")
print(f"Total player-stat-date signals: {len(sigs)}")
print(f"Overall accuracy: {sum(1 for s in sigs if s['hit'])/len(sigs)*100:.1f}%")

# Floor clearance analysis (the money signal)
print(f"\n{'='*70}")
print(f"FLOOR CLEARANCE ON OPTIMAL LINES")
print(f"{'='*70}")
print(f"{'Clr(p10)':<10} {'Sigs':<8} {'Wins':<8} {'Acc%':<8} {'Days':<8} {'Day%':<8} {'AvgHR':<8} {'AvgOdds':<8}")
print("-" * 66)
for clr in [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5]:
    f = [s for s in sigs if s['clearance'] >= clr]
    if not f:
        continue
    w = sum(1 for s in f if s['hit'])
    days = len(set(s['date'] for s in f))
    avg_hr = sum(s['hit_rate'] for s in f) / len(f)
    avg_odds = sum(s['odds'] for s in f) / len(f)
    marker = " <-- 90%+" if w/len(f) >= 0.90 else ""
    print(f"{clr:<10.1f} {len(f):<8} {w:<8} {w/len(f)*100:<8.1f} {days:<8} {days/len(dates_sorted)*100:<8.1f} {avg_hr:<8.2f} {avg_odds:<8.0f}{marker}")

# P25 clearance
print(f"\n{'='*70}")
print(f"25th PERCENTILE CLEARANCE ON OPTIMAL LINES")
print(f"{'='*70}")
print(f"{'Clr(p25)':<10} {'Sigs':<8} {'Wins':<8} {'Acc%':<8} {'Days':<8} {'Day%':<8}")
print("-" * 50)
for clr in [-2, -1, 0, 0.5, 1, 2, 3, 4, 5]:
    f = [s for s in sigs if s['clearance_25'] >= clr]
    if not f:
        continue
    w = sum(1 for s in f if s['hit'])
    days = len(set(s['date'] for s in f))
    marker = " <-- 90%+" if w/len(f) >= 0.90 else ""
    print(f"{clr:<10.1f} {len(f):<8} {w:<8} {w/len(f)*100:<8.1f} {days:<8} {days/len(dates_sorted)*100:<8.1f}{marker}")

# Combined: floor + hit rate + multi-window
print(f"\n{'='*70}")
print(f"COMBINED FILTERS: Floor + Hit Rate + Multi-Window Agreement")
print(f"{'='*70}")
combos = [
    (0, 0.70, 0.60, "Relaxed"),
    (0, 0.75, 0.70, "Moderate"),
    (0, 0.80, 0.70, "Strict HR"),
    (0.5, 0.70, 0.60, "Floor+0.5"),
    (1.0, 0.65, 0.50, "Floor+1.0 loose"),
    (1.0, 0.70, 0.60, "Floor+1.0 mod"),
    (-0.5, 0.80, 0.70, "Near floor, high HR"),
    (-1.0, 0.85, 0.80, "Below floor, very high HR"),
    (0, 0.90, 0.80, "90% HR multi"),
]
print(f"{'Label':<25} {'Clr':<6} {'HR':<6} {'MW':<6} {'Sigs':<8} {'Wins':<8} {'Acc%':<8} {'Days':<8} {'Day%':<8}")
print("-" * 81)
for clr, hr, mw, label in combos:
    f = [s for s in sigs if s['clearance'] >= clr and s['hit_rate'] >= hr and s['hr_10'] >= mw]
    if not f:
        print(f"{label:<25} {clr:<6.1f} {hr:<6.2f} {mw:<6.2f} {'0':<8} {'—':<8} {'—':<8} {'—':<8} {'—':<8}")
        continue
    w = sum(1 for s in f if s['hit'])
    days = len(set(s['date'] for s in f))
    marker = " ***" if w/len(f) >= 0.85 and days/len(dates_sorted) >= 0.30 else ""
    print(f"{label:<25} {clr:<6.1f} {hr:<6.2f} {mw:<6.2f} {len(f):<8} {w:<8} {w/len(f)*100:<8.1f} {days:<8} {days/len(dates_sorted)*100:<8.1f}{marker}")

# Top-N per day with optimal lines
print(f"\n{'='*70}")
print(f"TOP-N PER DAY — Optimal Lines, HR >= 0.70, ranked by clearance*hr")
print(f"{'='*70}")
for top_n in [1, 2, 3]:
    by_date = defaultdict(list)
    for s in sigs:
        if s['hit_rate'] >= 0.70:
            by_date[s['date']].append(s)
    picks = []
    for d in dates_sorted:
        dsigs = by_date.get(d, [])
        if not dsigs:
            continue
        dsigs.sort(key=lambda s: (s['clearance'] + 5) * s['hit_rate'], reverse=True)
        picks.extend(dsigs[:top_n])
    if picks:
        w = sum(1 for p in picks if p['hit'])
        days = len(set(p['date'] for p in picks))
        print(f"  Top-{top_n}: {len(picks)} picks, {w} wins, {w/len(picks)*100:.1f}% acc, {days} days ({days/len(dates_sorted)*100:.1f}%)")

# Show example picks with high clearance
print(f"\n{'='*70}")
print(f"SAMPLE HIGH-CLEARANCE PICKS")
print(f"{'='*70}")
high_clr = sorted([s for s in sigs if s['clearance'] >= 0], key=lambda s: -s['clearance'])[:20]
print(f"{'Date':<12} {'Player':<22} {'Stat':<6} {'Line':<8} {'Odds':<8} {'P10':<8} {'Clr':<8} {'HR':<8} {'Hit':<6}")
print("-" * 86)
for s in high_clr:
    print(f"{s['date']:<12} {s['player'][:20]:<22} {s['stat']:<6} {s['line']:<8.1f} +{s['odds']:<7} {s['p10']:<8.1f} {s['clearance']:<8.1f} {s['hit_rate']:<8.2f} {'Y' if s['hit'] else 'N':<6}")

print("\nDone.")
