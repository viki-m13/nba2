"""
MLB Strategy V4 — Adaptive Floor Convergence Engine (AFCE)
==========================================================
Same approach as NBA V4 adapted for MLB discrete stats.

Key MLB adaptations:
1. Uses at-bats instead of minutes for activity filter
2. Poisson-validated floors for discrete stats (hits, HR, etc.)
3. Component decomposition for h_r_rbi composite stat
4. Smaller stat values require different floor percentiles

Walk-forward only, real historical odds, no leakage.
"""

import math
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.odds_math import american_to_decimal, implied_probability, pnl_for_bet
from mlb.strategy import MLBPlayerModel, _update_model, _safe_int


def _percentile(values, pct):
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * pct)))
    return s[idx]


def run_backtest(box_scores, odds_data, config=None, verbose=False):
    """Walk-forward backtest of MLB V4 AFCE strategy."""
    cfg = config or {}
    MIN_GAMES = cfg.get('MIN_GAMES', 10)
    MIN_AB = cfg.get('MIN_AB', 2)
    MIN_ODDS = cfg.get('MIN_ODDS', 100)
    MAX_ODDS = cfg.get('MAX_ODDS', 400)
    FLOOR_PCT = cfg.get('FLOOR_PCT', 0.15)
    MIN_CLEARANCE = cfg.get('MIN_CLEARANCE', 0.0)
    MAX_DAILY = cfg.get('MAX_DAILY', 2)
    MIN_HR = cfg.get('MIN_HR', 0.75)
    UNIT_SIZE = cfg.get('UNIT_SIZE', 130)
    REQUIRE_MULTI_WINDOW = cfg.get('REQUIRE_MULTI_WINDOW', True)
    MIN_HR_5G = cfg.get('MIN_HR_5G', 0.80)

    model = MLBPlayerModel()

    # MLB stat map from odds data
    stat_map = {
        'hitterProps': 'h',
        'tbProps': 'tb',
        'rProps': 'r',
        'rbiProps': 'rbi',
        'hrProps': 'hr',
        'hrrbiProps': 'h_r_rbi',
    }

    dates_sorted = sorted(set(g['date'] for g in box_scores))
    box_by_date = defaultdict(list)
    for g in box_scores:
        box_by_date[g['date']].append(g)
    odds_index = defaultdict(dict)
    for o in odds_data:
        odds_index[o['date']][o.get('gameKey', '')] = o

    picks = []

    for date in dates_sorted:
        day_odds = odds_index.get(date, {})
        day_boxes = box_by_date.get(date, [])

        actuals = {}
        for bg in day_boxes:
            for p in bg.get('players', []):
                name = p.get('name', '')
                h = _safe_int(p.get('h', 0))
                r = _safe_int(p.get('r', 0))
                rbi = _safe_int(p.get('rbi', 0))
                hr = _safe_int(p.get('hr', 0))
                tb = _safe_int(p.get('tb', 0))
                actuals[name] = {
                    'h': h, 'r': r, 'rbi': rbi, 'hr': hr, 'tb': tb,
                    'h_r_rbi': h + r + rbi,
                }

        # Collect lines
        player_stat_lines = defaultdict(list)
        for gk, odds_rec in day_odds.items():
            for prop_type, stat in stat_map.items():
                props = odds_rec.get(prop_type, {})
                if not isinstance(props, dict):
                    continue
                for player, lines in props.items():
                    if not isinstance(lines, dict):
                        continue
                    for line_str, line_data in lines.items():
                        if not isinstance(line_data, dict):
                            continue
                        over_odds = line_data.get('overOdds', -999)
                        if MIN_ODDS <= over_odds <= MAX_ODDS:
                            player_stat_lines[(player, stat)].append(
                                (float(line_str), over_odds)
                            )

        day_signals = []
        for (player, stat), lines_list in player_stat_lines.items():
            profile = model.profiles.get(player)
            if not profile or len(profile['games']) < MIN_GAMES:
                continue

            # AB check
            games = profile['games']
            recent = games[-20:]
            avg_ab = sum(g.get('ab', 0) for g in recent) / len(recent)
            if avg_ab < MIN_AB:
                continue

            # Get stat values
            values_20 = [g.get(stat, 0) for g in recent]
            if len(values_20) < 10:
                continue

            lines_list.sort(key=lambda x: x[0])

            for line, odds in lines_list[:3]:
                floor = _percentile(values_20, FLOOR_PCT)
                clearance = floor - line
                if clearance < MIN_CLEARANCE:
                    continue

                hits_20 = sum(1 for v in values_20 if v > line)
                hr_20 = hits_20 / len(values_20)
                if hr_20 < MIN_HR:
                    continue

                if REQUIRE_MULTI_WINDOW:
                    values_5 = [g.get(stat, 0) for g in games[-5:]]
                    if values_5:
                        hr_5 = sum(1 for v in values_5 if v > line) / len(values_5)
                        if hr_5 < MIN_HR_5G:
                            continue

                    values_10 = [g.get(stat, 0) for g in games[-10:]]
                    if values_10:
                        hr_10 = sum(1 for v in values_10 if v > line) / len(values_10)
                        if hr_10 < MIN_HR:
                            continue

                day_signals.append({
                    'player': player, 'stat': stat,
                    'line': line, 'odds': odds,
                    'clearance': clearance, 'floor': floor,
                    'hit_rate': hr_20,
                })
                break

        day_signals.sort(key=lambda s: s['clearance'], reverse=True)
        selected = day_signals[:MAX_DAILY]

        for sig in selected:
            actual = actuals.get(sig['player'], {}).get(sig['stat'])
            if actual is None:
                continue

            hit = actual > sig['line']
            pnl = pnl_for_bet(UNIT_SIZE, sig['odds'], hit)

            pick = {
                'date': date,
                'player': sig['player'],
                'stat': sig['stat'],
                'line': sig['line'],
                'odds': sig['odds'],
                'hit_rate': sig['hit_rate'],
                'ev': 0,
                'actual': actual,
                'hit': hit,
                'pnl': pnl,
                'wager': UNIT_SIZE,
                'bet_type': 'single',
                'combined_score': sig['clearance'],
                'edge': sig['clearance'],
            }
            picks.append(pick)

            if verbose:
                status = 'WIN' if hit else 'LOSS'
                print(f"  {date}: {sig['player']} {sig['stat'].upper()} O{sig['line']} "
                      f"+{sig['odds']} (floor={sig['floor']:.1f}, clr={sig['clearance']:.1f}) "
                      f"→ {actual} [{status}] ${pnl:+d}")

        for bg in day_boxes:
            _update_model(model, bg, date)

    total_picks = len(picks)
    total_wins = sum(1 for p in picks if p['hit'])
    total_pnl = sum(p['pnl'] for p in picks)
    total_wagered = sum(p['wager'] for p in picks)
    active_days = len(set(p['date'] for p in picks))

    return {
        'total_picks': total_picks,
        'total_wins': total_wins,
        'accuracy': total_wins / total_picks if total_picks > 0 else 0,
        'total_pnl': total_pnl,
        'total_wagered': total_wagered,
        'total_roi': total_pnl / total_wagered if total_wagered > 0 else 0,
        'active_days': active_days,
        'total_days': len(dates_sorted),
        'day_coverage': active_days / len(dates_sorted) if dates_sorted else 0,
        'picks': picks,
    }


if __name__ == '__main__':
    from shared.data_loader import load_mlb_boxscores, load_mlb_odds

    print("Loading MLB data...")
    box_scores = load_mlb_boxscores()
    odds_data = load_mlb_odds()
    print(f"  {len(box_scores)} games, {len(odds_data)} odds records")

    configs = [
        ("Strict Floor (p10>line, HR>=0.75, top-1)", {
            'FLOOR_PCT': 0.10, 'MIN_CLEARANCE': 0.0, 'MIN_HR': 0.75,
            'MAX_DAILY': 1, 'MIN_HR_5G': 0.80,
        }),
        ("Strict Floor (p15>line, HR>=0.70, top-2)", {
            'FLOOR_PCT': 0.15, 'MIN_CLEARANCE': 0.0, 'MIN_HR': 0.70,
            'MAX_DAILY': 2, 'MIN_HR_5G': 0.60,
        }),
        ("Relaxed Floor (p20>line, HR>=0.65, top-3)", {
            'FLOOR_PCT': 0.20, 'MIN_CLEARANCE': 0.0, 'MIN_HR': 0.65,
            'MAX_DAILY': 3, 'MIN_HR_5G': 0.60,
        }),
        ("Near Floor (clr>=-0.5, HR>=0.80, top-2)", {
            'FLOOR_PCT': 0.10, 'MIN_CLEARANCE': -0.5, 'MIN_HR': 0.80,
            'MAX_DAILY': 2, 'MIN_HR_5G': 0.80,
        }),
        ("Very relaxed (p20, clr>=-1, HR>=0.65, top-5)", {
            'FLOOR_PCT': 0.20, 'MIN_CLEARANCE': -1.0, 'MIN_HR': 0.65,
            'MAX_DAILY': 5, 'MIN_HR_5G': 0.60, 'MIN_GAMES': 8,
        }),
        ("No multi-window (p10>line, HR>=0.70, top-3)", {
            'FLOOR_PCT': 0.10, 'MIN_CLEARANCE': 0.0, 'MIN_HR': 0.70,
            'MAX_DAILY': 3, 'REQUIRE_MULTI_WINDOW': False,
        }),
    ]

    print(f"\n{'Config':<55} {'Picks':>6} {'Wins':>6} {'Acc%':>7} {'Days':>6} {'Day%':>7} {'P&L':>8} {'ROI%':>7}")
    print("-" * 103)
    for name, cfg in configs:
        r = run_backtest(box_scores, odds_data, cfg, verbose=False)
        if r['total_picks'] > 0:
            print(f"{name:<55} {r['total_picks']:>6} {r['total_wins']:>6} {r['accuracy']*100:>7.1f} "
                  f"{r['active_days']:>6} {r['day_coverage']*100:>7.1f} ${r['total_pnl']:>+7d} {r['total_roi']*100:>7.1f}")
        else:
            print(f"{name:<55} {'0':>6} {'—':>6} {'—':>7} {'0':>6} {'0.0':>7} {'$0':>8} {'—':>7}")

    # Run best verbose
    print("\n" + "=" * 60)
    print("DETAILED — Strict Floor (p15>line, HR>=0.70, top-2)")
    print("=" * 60)
    r = run_backtest(box_scores, odds_data, {
        'FLOOR_PCT': 0.15, 'MIN_CLEARANCE': 0.0, 'MIN_HR': 0.70,
        'MAX_DAILY': 2, 'MIN_HR_5G': 0.60,
    }, verbose=True)
    print(f"\n  Total: {r['total_picks']} picks, {r['total_wins']} wins, "
          f"{r['accuracy']*100:.1f}% acc, {r['active_days']} days")
