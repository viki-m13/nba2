#!/usr/bin/env python3
"""
Analyze signal distribution — NO gates, just raw hit rates vs actual outcomes.
"""
import sys, os, json, math
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_loader import load_nba_boxscores, load_nba_odds
from nba.strategy import NBAPlayerModel, _update_model, _safe_int
from shared.odds_math import american_to_decimal, implied_probability, expected_value

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
all_signals = []

for date in dates_sorted:
    day_odds = odds_index.get(date, {})
    day_boxes = box_by_date.get(date, [])

    # Build actuals
    actuals = {}
    for bg in day_boxes:
        for p in bg.get('players', []):
            name = p.get('name', '')
            pts = _safe_int(p.get('pts', 0))
            reb = _safe_int(p.get('reb', 0))
            ast = _safe_int(p.get('ast', 0))
            actuals[name] = {'pts': pts, 'reb': reb, 'ast': ast, 'pra': pts + reb + ast}

    for gk, odds_rec in day_odds.items():
        for prop_type, stat in stat_map.items():
            props = odds_rec.get(prop_type, {})
            for player, lines in props.items():
                for line_str, line_data in lines.items():
                    over_odds = line_data.get('overOdds', -999)
                    if over_odds < 100 or over_odds > 500:
                        continue

                    line = float(line_str)
                    profile = model.profiles.get(player)
                    if not profile or len(profile['games']) < 8:
                        continue

                    games = profile['games']
                    window = games[-20:]
                    avg_min = sum(g['min'] for g in window) / len(window)
                    if avg_min < 15:
                        continue

                    values = [g.get(stat, 0) for g in window]
                    hits = sum(1 for v in values if v > line)
                    hit_rate = hits / len(values)

                    # Also compute 10-game and 5-game hit rates
                    recent_10 = values[-10:]
                    recent_5 = values[-5:]
                    hr_10 = sum(1 for v in recent_10 if v > line) / len(recent_10) if recent_10 else 0
                    hr_5 = sum(1 for v in recent_5 if v > line) / len(recent_5) if recent_5 else 0

                    # Median and floor
                    sorted_vals = sorted(values)
                    median = sorted_vals[len(sorted_vals)//2]
                    p10 = sorted_vals[max(0, int(len(sorted_vals)*0.1))]
                    clearance = p10 - line  # positive = floor > line

                    dec_odds = american_to_decimal(over_odds)
                    market_prob = implied_probability(over_odds)
                    ev = expected_value(hit_rate, dec_odds)

                    actual = actuals.get(player, {}).get(stat)
                    hit = actual > line if actual is not None else None

                    all_signals.append({
                        'date': date, 'player': player, 'stat': stat,
                        'line': line, 'odds': over_odds,
                        'hit_rate': hit_rate, 'hr_10': hr_10, 'hr_5': hr_5,
                        'ev': ev, 'clearance': clearance, 'median': median,
                        'p10': p10, 'market_prob': market_prob,
                        'n_games': len(values), 'avg_min': avg_min,
                        'hit': hit,
                    })

    for bg in day_boxes:
        _update_model(model, bg, date)

# Analysis
sigs = [s for s in all_signals if s['hit'] is not None]
print(f"Total plus-money signals with results: {len(sigs)}")
print(f"Overall hit rate: {sum(1 for s in sigs if s['hit'])/len(sigs)*100:.1f}%")

# Sweep: hit rate threshold
print(f"\n{'='*70}")
print(f"HIT RATE THRESHOLD SWEEP (20-game window)")
print(f"{'='*70}")
print(f"{'HR>=':<8} {'Sigs':<8} {'Wins':<8} {'Acc%':<8} {'Days':<8} {'Day%':<8} {'AvgEV':<8} {'AvgOdds':<8}")
print("-" * 64)
for hr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    f = [s for s in sigs if s['hit_rate'] >= hr]
    if not f:
        continue
    w = sum(1 for s in f if s['hit'])
    days = len(set(s['date'] for s in f))
    avg_ev = sum(s['ev'] for s in f) / len(f)
    avg_odds = sum(s['odds'] for s in f) / len(f)
    print(f"{hr:<8.2f} {len(f):<8} {w:<8} {w/len(f)*100:<8.1f} {days:<8} {days/len(dates_sorted)*100:<8.1f} {avg_ev:<8.2f} {avg_odds:<8.0f}")

# Multi-window agreement: 20g AND 10g AND 5g all >= threshold
print(f"\n{'='*70}")
print(f"MULTI-WINDOW AGREEMENT (20g + 10g + 5g all >= threshold)")
print(f"{'='*70}")
print(f"{'HR>=':<8} {'Sigs':<8} {'Wins':<8} {'Acc%':<8} {'Days':<8} {'Day%':<8}")
print("-" * 48)
for hr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    f = [s for s in sigs if s['hit_rate'] >= hr and s['hr_10'] >= hr and s['hr_5'] >= hr]
    if not f:
        continue
    w = sum(1 for s in f if s['hit'])
    days = len(set(s['date'] for s in f))
    print(f"{hr:<8.2f} {len(f):<8} {w:<8} {w/len(f)*100:<8.1f} {days:<8} {days/len(dates_sorted)*100:<8.1f}")

# Floor clearance: p10 > line
print(f"\n{'='*70}")
print(f"FLOOR CLEARANCE (10th percentile above line)")
print(f"{'='*70}")
print(f"{'Clr>=':<8} {'HR>=':<8} {'Sigs':<8} {'Wins':<8} {'Acc%':<8} {'Days':<8} {'Day%':<8}")
print("-" * 56)
for clr in [0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    for hr in [0.60, 0.70, 0.80]:
        f = [s for s in sigs if s['clearance'] >= clr and s['hit_rate'] >= hr]
        if not f:
            continue
        w = sum(1 for s in f if s['hit'])
        days = len(set(s['date'] for s in f))
        print(f"{clr:<8.1f} {hr:<8.2f} {len(f):<8} {w:<8} {w/len(f)*100:<8.1f} {days:<8} {days/len(dates_sorted)*100:<8.1f}")

# Top-N per day by score
print(f"\n{'='*70}")
print(f"TOP-N PER DAY STRATEGY")
print(f"{'='*70}")
for min_hr in [0.60, 0.65, 0.70, 0.75]:
    print(f"\n  --- Min HR: {min_hr} ---")
    print(f"  {'TopN':<8} {'Picks':<8} {'Wins':<8} {'Acc%':<8} {'Days':<8} {'Day%':<8}")
    print(f"  " + "-" * 48)
    for top_n in [1, 2, 3, 5]:
        picks = []
        by_date = defaultdict(list)
        for s in sigs:
            if s['hit_rate'] >= min_hr:
                by_date[s['date']].append(s)
        for d in dates_sorted:
            dsigs = by_date.get(d, [])
            if not dsigs:
                continue
            dsigs.sort(key=lambda s: s['hit_rate'] * (1 + s['ev']) * (1 + max(0, s['clearance'])), reverse=True)
            picks.extend(dsigs[:top_n])
        if picks:
            w = sum(1 for p in picks if p['hit'])
            days = len(set(p['date'] for p in picks))
            print(f"  {top_n:<8} {len(picks):<8} {w:<8} {w/len(picks)*100:<8.1f} {days:<8} {days/len(dates_sorted)*100:<8.1f}")

# Stat category analysis
print(f"\n{'='*70}")
print(f"BY STAT CATEGORY (HR >= 0.70)")
print(f"{'='*70}")
for stat in ['pts', 'reb', 'ast', 'pra']:
    f = [s for s in sigs if s['stat'] == stat and s['hit_rate'] >= 0.70]
    if not f:
        continue
    w = sum(1 for s in f if s['hit'])
    days = len(set(s['date'] for s in f))
    print(f"  {stat:<6} {len(f):>5} sigs, {w:>5} wins, {w/len(f)*100:.1f}% acc, {days} days ({days/len(dates_sorted)*100:.1f}%)")

# Cross-stat convergence
print(f"\n{'='*70}")
print(f"CROSS-STAT CONVERGENCE (player has HR>=0.65 in 2+ stats)")
print(f"{'='*70}")
by_date_player = defaultdict(lambda: defaultdict(list))
for s in sigs:
    if s['hit_rate'] >= 0.65:
        by_date_player[s['date']][s['player']].append(s)

convergence_picks = []
for d, players in by_date_player.items():
    for player, player_sigs in players.items():
        unique_stats = set(s['stat'] for s in player_sigs)
        if len(unique_stats) >= 2:
            # Pick the signal with best hit_rate
            best = max(player_sigs, key=lambda s: s['hit_rate'])
            convergence_picks.append(best)

if convergence_picks:
    w = sum(1 for p in convergence_picks if p['hit'])
    days = len(set(p['date'] for p in convergence_picks))
    print(f"Picks: {len(convergence_picks)}, Wins: {w}, Accuracy: {w/len(convergence_picks)*100:.1f}%")
    print(f"Days: {days}/{len(dates_sorted)} ({days/len(dates_sorted)*100:.1f}%)")

    # With 3+ stats
    conv3 = []
    for d, players in by_date_player.items():
        for player, player_sigs in players.items():
            unique_stats = set(s['stat'] for s in player_sigs)
            if len(unique_stats) >= 3:
                best = max(player_sigs, key=lambda s: s['hit_rate'])
                conv3.append(best)
    if conv3:
        w = sum(1 for p in conv3 if p['hit'])
        days = len(set(p['date'] for p in conv3))
        print(f"\n3+ stat convergence: {len(conv3)} picks, {w} wins, {w/len(conv3)*100:.1f}% acc, {days} days")

print("\nDone.")
