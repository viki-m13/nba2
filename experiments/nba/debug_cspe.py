#!/usr/bin/env python3
"""Debug CSPE: check how many legs pass each filter stage."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from shared.data_loader import load_nba_boxscores, load_nba_odds
from shared.odds_math import beta_ppf_approx
from nba.strategy import NBAPlayerModel, _update_model, _safe_int
from nba.strategy_v8_cspe import _percentile, _std_dev, _stat_entropy

def debug():
    box_scores = load_nba_boxscores()
    odds_data = load_nba_odds()
    box_scores = [g for g in box_scores if g.get('home','') not in ['STARS','WORLD'] and g.get('away','') not in ['STARS','WORLD']]

    model = NBAPlayerModel()
    stat_map = {'playerProps': 'pts'}

    dates_sorted = sorted(set(g['date'] for g in box_scores))
    box_by_date = defaultdict(list)
    for g in box_scores:
        box_by_date[g['date']].append(g)
    odds_index = defaultdict(dict)
    for o in odds_data:
        odds_index[o['date']][o['gameKey']] = o

    counters = defaultdict(int)

    for date in dates_sorted:
        day_odds = odds_index.get(date, {})
        day_boxes = box_by_date.get(date, [])

        actuals = {}
        for bg in day_boxes:
            for p in bg.get('players', []):
                name = p.get('name', '')
                actuals[name] = {'pts': _safe_int(p.get('pts', 0))}

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
                        if over_odds is None:
                            continue
                        actual = actuals.get(player, {}).get(stat)
                        if actual is None:
                            continue

                        counters['total'] += 1

                        # Check each filter stage
                        if model.game_count(player) < 15:
                            counters['fail_min_games'] += 1
                            continue
                        counters['pass_min_games'] += 1

                        if model.avg_minutes(player, 10) < 20:
                            counters['fail_min_minutes'] += 1
                            continue
                        counters['pass_min_minutes'] += 1

                        if over_odds < -600 or over_odds > -105:
                            counters['fail_odds_range'] += 1
                            continue
                        counters['pass_odds_range'] += 1

                        v20 = model.get_values(player, stat, 20)
                        if not v20 or len(v20) < 10:
                            counters['fail_data'] += 1
                            continue
                        counters['pass_data'] += 1

                        hits = sum(1 for v in v20 if v > line_val)
                        hr = hits / len(v20)

                        if hr < 0.80:
                            counters['fail_hr80'] += 1
                            continue
                        counters['pass_hr80'] += 1

                        alpha = 1.0 + hits
                        beta_p = 1.0 + (len(v20) - hits)
                        bcf_lb = beta_ppf_approx(alpha, beta_p, 0.20)

                        if bcf_lb < 0.78:
                            counters['fail_bcf78'] += 1
                            continue
                        counters['pass_bcf78'] += 1

                        floor_val = _percentile(v20, 0.10)
                        clearance = floor_val - line_val
                        if clearance < 0:
                            counters['fail_floor'] += 1
                            continue
                        counters['pass_floor'] += 1

                        recent = model.get_values(player, stat, 5)
                        streak = 0
                        if recent:
                            for v in reversed(recent):
                                if v > line_val: streak += 1
                                else: break
                        if streak < 2:
                            counters['fail_streak'] += 1
                            continue
                        counters['pass_streak'] += 1

                        # Multi-window check
                        mw_fail = False
                        for w in [5, 10]:
                            vw = model.get_values(player, stat, w)
                            if vw and len(vw) >= min(w, 3):
                                whr = sum(1 for v in vw if v > line_val) / len(vw)
                                if whr < 0.70:
                                    mw_fail = True
                                    break
                        if mw_fail:
                            counters['fail_mw'] += 1
                            continue
                        counters['pass_mw'] += 1

                        # Extended consistency
                        v30 = model.get_values(player, stat, 30)
                        if v30 and len(v30) >= 15:
                            hr_30 = sum(1 for v in v30 if v > line_val) / len(v30)
                            if hr_30 < 0.75:
                                counters['fail_ec'] += 1
                                continue
                        counters['pass_ec'] += 1

                        # Hit check
                        hit = actual > line_val
                        counters['final_total'] += 1
                        if hit:
                            counters['final_hits'] += 1

        for bg in day_boxes:
            _update_model(model, bg, date)

    print("\n=== FILTER FUNNEL ===")
    for key in ['total', 'pass_min_games', 'fail_min_games',
                'pass_min_minutes', 'fail_min_minutes',
                'pass_odds_range', 'fail_odds_range',
                'pass_data', 'fail_data',
                'pass_hr80', 'fail_hr80',
                'pass_bcf78', 'fail_bcf78',
                'pass_floor', 'fail_floor',
                'pass_streak', 'fail_streak',
                'pass_mw', 'fail_mw',
                'pass_ec', 'fail_ec',
                'final_total', 'final_hits']:
        val = counters.get(key, 0)
        print(f"  {key:<25}: {val:>6}")

    total = counters['final_total']
    hits = counters['final_hits']
    if total > 0:
        print(f"\n  FINAL ACCURACY: {hits}/{total} = {hits/total*100:.1f}%")

if __name__ == '__main__':
    debug()
