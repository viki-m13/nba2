#!/usr/bin/env python3
"""Explore accuracy with COMBINED filters at different odds ranges."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from shared.data_loader import load_nba_boxscores, load_nba_odds
from shared.odds_math import american_to_decimal, beta_ppf_approx, shannon_entropy
from nba.strategy import NBAPlayerModel, _update_model, _safe_int

def _percentile(values, pct):
    s = sorted(values)
    return s[max(0, min(len(s)-1, int(len(s)*pct)))]

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

    # Track all qualifying legs with their filter pass/fail
    all_legs = []

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
                        if over_odds is None or over_odds > -100 or over_odds < -600:
                            continue
                        actual = actuals.get(player, {}).get(stat)
                        if actual is None:
                            continue

                        if model.game_count(player) < 15:
                            continue
                        if model.avg_minutes(player, 10) < 18:
                            continue

                        v20 = model.get_values(player, stat, 20)
                        if not v20 or len(v20) < 10:
                            continue

                        # Compute all filters
                        hits_20 = sum(1 for v in v20 if v > line_val)
                        alpha = 1.0 + hits_20
                        beta_p = 1.0 + (len(v20) - hits_20)
                        lb = beta_ppf_approx(alpha, beta_p, 0.20)
                        hr_20 = hits_20 / len(v20)

                        # PFC
                        floor_p10 = _percentile(v20, 0.10)
                        clearance = floor_p10 - line_val
                        pfc_pass = clearance >= 0

                        # MWU
                        mwu_passes = 0
                        for w in [5, 10, 15]:
                            vw = model.get_values(player, stat, w)
                            if vw and len(vw) >= min(w, 3):
                                whr = sum(1 for v in vw if v > line_val) / len(vw)
                                if whr >= 0.75:
                                    mwu_passes += 1
                        mwu_pass = mwu_passes >= 2

                        # SC (streak)
                        v5 = model.get_values(player, stat, 5)
                        streak = 0
                        if v5:
                            for v in reversed(v5):
                                if v > line_val: streak += 1
                                else: break
                        sc_pass = streak >= 3

                        # EC (extended)
                        v30 = model.get_values(player, stat, 30)
                        ec_pass = True
                        if v30 and len(v30) >= 15:
                            ec_pass = sum(1 for v in v30 if v > line_val) / len(v30) >= 0.80

                        # MS (minutes CV)
                        cv = model.get_minutes_cv(player, 10)
                        ms_pass = cv is not None and cv < 0.18

                        over_hit = actual > line_val

                        all_legs.append({
                            'date': date, 'player': player, 'stat': stat,
                            'line': line_val, 'odds': over_odds, 'hit': over_hit,
                            'lb': lb, 'hr_20': hr_20, 'clearance': clearance,
                            'pfc': pfc_pass, 'mwu': mwu_pass, 'mwu_count': mwu_passes,
                            'sc': sc_pass, 'streak': streak, 'ec': ec_pass,
                            'ms': ms_pass,
                        })

        for bg in day_boxes:
            _update_model(model, bg, date)

    # Analyze combined filter effectiveness
    print(f"\nTotal qualifying legs (BCF>=0, odds -600 to -100): {len(all_legs)}")

    for bcf_thresh in [0.70, 0.75, 0.80, 0.85]:
        filtered = [l for l in all_legs if l['lb'] >= bcf_thresh]
        print(f"\n{'='*80}")
        print(f"BCF LB >= {bcf_thresh:.2f}: {len(filtered)} legs")

        if not filtered:
            continue

        # Try different filter combinations
        combos = [
            ("BCF only", lambda l: True),
            ("BCF + PFC", lambda l: l['pfc']),
            ("BCF + PFC + MWU(2/3)", lambda l: l['pfc'] and l['mwu']),
            ("BCF + PFC + MWU + SC(3)", lambda l: l['pfc'] and l['mwu'] and l['sc']),
            ("BCF + PFC + MWU + SC + EC", lambda l: l['pfc'] and l['mwu'] and l['sc'] and l['ec']),
            ("BCF + PFC + MWU + SC + EC + MS", lambda l: l['pfc'] and l['mwu'] and l['sc'] and l['ec'] and l['ms']),
            # Relaxed combos
            ("BCF + PFC + MWU(2/3)", lambda l: l['pfc'] and l['mwu']),
            ("BCF + PFC + SC(3)", lambda l: l['pfc'] and l['sc']),
            ("BCF + MWU + SC(3)", lambda l: l['mwu'] and l['sc']),
            ("BCF + PFC + MWU + MS", lambda l: l['pfc'] and l['mwu'] and l['ms']),
        ]

        print(f"{'Combo':<40} | {'Legs':>5} | {'Hits':>4} | {'Acc%':>6} | {'Days':>4} | {'Legs/Day':>8}")
        print("-" * 80)

        for name, filt in combos:
            subset = [l for l in filtered if filt(l)]
            if not subset:
                continue
            hits = sum(1 for l in subset if l['hit'])
            acc = hits / len(subset) * 100
            days = len(set(l['date'] for l in subset))
            lpd = len(subset) / days if days > 0 else 0

            # Breakdown by odds range
            heavy = [l for l in subset if l['odds'] <= -300]
            medium = [l for l in subset if -300 < l['odds'] <= -150]
            light = [l for l in subset if l['odds'] > -150]

            h_acc = sum(1 for l in heavy if l['hit']) / len(heavy) * 100 if heavy else 0
            m_acc = sum(1 for l in medium if l['hit']) / len(medium) * 100 if medium else 0
            l_acc = sum(1 for l in light if l['hit']) / len(light) * 100 if light else 0

            print(f"{name:<40} | {len(subset):>5} | {hits:>4} | {acc:>6.1f} | {days:>4} | {lpd:>8.1f}")
            if heavy: print(f"  └ -600 to -300: {len(heavy)} legs, {h_acc:.1f}% acc")
            if medium: print(f"  └ -300 to -150: {len(medium)} legs, {m_acc:.1f}% acc")
            if light: print(f"  └ -150 to -100: {len(light)} legs, {l_acc:.1f}% acc")

    # Simulate parlay ROI for best combos
    print(f"\n{'='*80}")
    print("PARLAY ROI SIMULATION")
    print(f"{'='*80}")

    for bcf_thresh in [0.75, 0.80]:
        for combo_name, combo_filt in [
            ("BCF+PFC+MWU+SC", lambda l: l['pfc'] and l['mwu'] and l['sc']),
            ("BCF+PFC+MWU+SC+EC", lambda l: l['pfc'] and l['mwu'] and l['sc'] and l['ec']),
            ("BCF+PFC+MWU+SC+EC+MS", lambda l: l['pfc'] and l['mwu'] and l['sc'] and l['ec'] and l['ms']),
        ]:
            subset = [l for l in all_legs if l['lb'] >= bcf_thresh and combo_filt(l)]
            if len(subset) < 5:
                continue

            # Group by date
            by_date = defaultdict(list)
            for l in subset:
                by_date[l['date']].append(l)

            # Sort by confidence within each day and build parlays
            for n_legs in [3, 4, 5, 8, 10]:
                total_pnl = 0
                total_wagered = 0
                total_parlays = 0
                total_wins = 0
                wager = 100

                for date in sorted(by_date.keys()):
                    day_legs = by_date[date]
                    # Sort by lb (higher = better)
                    day_legs.sort(key=lambda l: l['lb'], reverse=True)

                    # Dedup by player
                    seen = set()
                    unique = []
                    for l in day_legs:
                        if l['player'] not in seen:
                            unique.append(l)
                            seen.add(l['player'])

                    if len(unique) >= n_legs:
                        selected = unique[:n_legs]
                        all_hit = all(l['hit'] for l in selected)
                        combined_dec = 1.0
                        for l in selected:
                            combined_dec *= american_to_decimal(l['odds'])

                        pnl = round((combined_dec - 1) * wager) if all_hit else -wager
                        total_pnl += pnl
                        total_wagered += wager
                        total_parlays += 1
                        if all_hit:
                            total_wins += 1

                if total_parlays >= 3:
                    roi = total_pnl / total_wagered * 100 if total_wagered > 0 else 0
                    acc = total_wins / total_parlays * 100
                    print(f"BCF>={bcf_thresh} {combo_name} | {n_legs}L: {total_parlays} parlays, "
                          f"{total_wins}W, {acc:.1f}% acc, ${total_pnl:+d}, {roi:+.1f}% ROI")

if __name__ == '__main__':
    explore()
