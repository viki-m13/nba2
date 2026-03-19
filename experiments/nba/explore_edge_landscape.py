#!/usr/bin/env python3
"""Explore the edge landscape: what accuracy is achievable at different odds ranges?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from shared.data_loader import load_nba_boxscores, load_nba_odds
from shared.odds_math import american_to_decimal, beta_ppf_approx
from nba.strategy import NBAPlayerModel, _update_model, _safe_int

def explore():
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores if g.get('home','') not in ['STARS','WORLD'] and g.get('away','') not in ['STARS','WORLD']]

    model = NBAPlayerModel()
    stat_map = {'playerProps': 'pts', 'rebProps': 'reb', 'astProps': 'ast', 'praProps': 'pra'}

    dates_sorted = sorted(set(g['date'] for g in box_scores))
    box_by_date = defaultdict(list)
    for g in box_scores:
        box_by_date[g['date']].append(g)
    odds_index = defaultdict(dict)
    for o in odds_data:
        odds_index[o['date']][o['gameKey']] = o

    # Track results by odds bucket and by direction (over/under)
    buckets = [(-500,-300), (-300,-200), (-200,-150), (-150,-120), (-120,-100), (-100,100), (100,200), (200,500)]
    results = {b: {'over_hits': 0, 'over_total': 0, 'under_hits': 0, 'under_total': 0} for b in buckets}

    # Also track by BCF lower bound threshold
    bcf_results = {}
    for bcf_thresh in [0.70, 0.75, 0.80, 0.85, 0.90]:
        bcf_results[bcf_thresh] = {b: {'over_hits': 0, 'over_total': 0, 'under_hits': 0, 'under_total': 0} for b in buckets}

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

        for gk, odds_rec in day_odds.items():
            for prop_type, stat in stat_map.items():
                props = odds_rec.get(prop_type, {})
                for player, lines in props.items():
                    for line_str, line_data in lines.items():
                        try:
                            line_val = float(line_str)
                        except (ValueError, TypeError):
                            continue

                        over_odds = line_data.get('overOdds')
                        under_odds = line_data.get('underOdds')
                        actual = actuals.get(player, {}).get(stat)
                        if actual is None:
                            continue

                        # Check model has enough data
                        if model.game_count(player) < 12:
                            continue
                        if model.avg_minutes(player, 10) < 15:
                            continue

                        v20 = model.get_values(player, stat, 20)
                        if not v20 or len(v20) < 8:
                            continue

                        # Compute BCF lower bound for OVER
                        over_hits_20 = sum(1 for v in v20 if v > line_val)
                        alpha_o = 1.0 + over_hits_20
                        beta_o = 1.0 + (len(v20) - over_hits_20)
                        lb_over = beta_ppf_approx(alpha_o, beta_o, 0.20)

                        # Compute BCF lower bound for UNDER
                        under_hits_20 = sum(1 for v in v20 if v < line_val)
                        alpha_u = 1.0 + under_hits_20
                        beta_u = 1.0 + (len(v20) - under_hits_20)
                        lb_under = beta_ppf_approx(alpha_u, beta_u, 0.20)

                        # OVER analysis
                        if over_odds is not None:
                            over_hit = actual > line_val
                            for (lo, hi) in buckets:
                                if lo <= over_odds < hi:
                                    results[(lo,hi)]['over_total'] += 1
                                    if over_hit:
                                        results[(lo,hi)]['over_hits'] += 1
                                    for bcf_thresh in bcf_results:
                                        if lb_over >= bcf_thresh:
                                            bcf_results[bcf_thresh][(lo,hi)]['over_total'] += 1
                                            if over_hit:
                                                bcf_results[bcf_thresh][(lo,hi)]['over_hits'] += 1
                                    break

                        # UNDER analysis
                        if under_odds is not None:
                            under_hit = actual < line_val
                            for (lo, hi) in buckets:
                                if lo <= under_odds < hi:
                                    results[(lo,hi)]['under_total'] += 1
                                    if under_hit:
                                        results[(lo,hi)]['under_hits'] += 1
                                    for bcf_thresh in bcf_results:
                                        if lb_under >= bcf_thresh:
                                            bcf_results[bcf_thresh][(lo,hi)]['under_total'] += 1
                                            if under_hit:
                                                bcf_results[bcf_thresh][(lo,hi)]['under_hits'] += 1
                                    break

        # Update model AFTER signals
        for bg in day_boxes:
            _update_model(model, bg, date)

    # Print results
    print("\n=== RAW ACCURACY BY ODDS BUCKET (no filtering) ===")
    print(f"{'Odds Range':<15} | {'Over Acc':>8} {'(n)':>6} | {'Under Acc':>9} {'(n)':>6}")
    print("-" * 55)
    for b in buckets:
        ot, oh = results[b]['over_total'], results[b]['over_hits']
        ut, uh = results[b]['under_total'], results[b]['under_hits']
        oa = oh/ot*100 if ot > 0 else 0
        ua = uh/ut*100 if ut > 0 else 0
        print(f"{b[0]:>+4} to {b[1]:>+4}   | {oa:>7.1f}% {ot:>5} | {ua:>8.1f}% {ut:>5}")

    for bcf_thresh in sorted(bcf_results.keys()):
        print(f"\n=== ACCURACY WITH BCF LB >= {bcf_thresh:.2f} ===")
        print(f"{'Odds Range':<15} | {'Over Acc':>8} {'(n)':>6} | {'Under Acc':>9} {'(n)':>6}")
        print("-" * 55)
        for b in buckets:
            r = bcf_results[bcf_thresh][b]
            ot, oh = r['over_total'], r['over_hits']
            ut, uh = r['under_total'], r['under_hits']
            oa = oh/ot*100 if ot > 0 else 0
            ua = uh/ut*100 if ut > 0 else 0
            if ot > 0 or ut > 0:
                print(f"{b[0]:>+4} to {b[1]:>+4}   | {oa:>7.1f}% {ot:>5} | {ua:>8.1f}% {ut:>5}")

if __name__ == '__main__':
    explore()
