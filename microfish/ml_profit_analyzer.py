"""
Profit-Optimized Live Entry Point Analyzer
=============================================
Finds the MOST PROFITABLE entry points during live MLB games.
Instead of buying at $0.95-$0.99 (safe but tiny profit), we find
spots where we can buy at $0.70-$0.85 with 90%+ accuracy.

Key insight: A team with 85% WP in the 6th inning with a 3-run lead
might be buyable at $0.80 and wins 93% of the time → $0.20 profit/share
vs $0.05 profit at $0.95. Even with some losses, the EV is far higher.

EXPECTED VALUE FORMULA:
  EV = (accuracy × profit_per_share) - ((1 - accuracy) × buy_price)

  At $0.95, 100% accuracy: EV = 1.0 × $0.05 = $0.050
  At $0.80, 93% accuracy:  EV = 0.93 × $0.20 - 0.07 × $0.80 = $0.130
  At $0.75, 91% accuracy:  EV = 0.91 × $0.25 - 0.09 × $0.75 = $0.160

The profit-optimized sweet spot is NOT 100% accuracy — it's where EV peaks.
"""

import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'moneyline')
CACHE_FILE = os.path.join(DATA_DIR, 'espn_live_winprob_cache.json')


def load_cache():
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    valid = {k: v for k, v in cache.items()
             if v.get('n_plays', 0) > 0 and not v.get('error')}
    return valid


def analyze_entry_points():
    """
    For every game, at every play, compute:
    - inning, run lead, win probability
    - whether the leading team won

    Then aggregate by (inning_bucket, lead_bucket, wp_bucket) to find
    the most profitable entry points.
    """
    games = load_cache()
    print(f"Analyzing {len(games)} games for profit-optimized entry points...\n")

    # Collect all entry point observations
    # Key: (min_inning, min_lead, wp_bucket) → list of (won, wp)
    entries = defaultdict(lambda: {'wins': 0, 'losses': 0, 'wps': []})

    # Also track individual game entries for detailed analysis
    game_entries = []

    for eid, game in games.items():
        timeline = game.get('timeline', [])
        if not timeline:
            continue

        home_score_final = game.get('home_score', 0)
        away_score_final = game.get('away_score', 0)
        if home_score_final == away_score_final:
            continue

        home_won = home_score_final > away_score_final

        # Track the FIRST time each (inning, lead) combo reaches each WP threshold
        # for each side (home/away) — this is when the alert would fire
        seen_combos_home = set()
        seen_combos_away = set()

        for play in timeline:
            inning = play.get('inning', 0)
            home_score = play.get('homeScore', 0)
            away_score = play.get('awayScore', 0)
            home_wp = play.get('homeWP', 0.5)
            away_wp = play.get('awayWP', 0.5)

            if inning < 1:
                continue

            # Check home side
            home_lead = home_score - away_score
            if home_lead > 0 and home_wp > 0.6:
                for wp_thresh in range(60, 100):
                    wp_pct = wp_thresh / 100.0
                    if home_wp >= wp_pct:
                        # For each inning threshold
                        for inn_thresh in range(1, min(inning + 1, 10)):
                            for lead_thresh in range(1, min(home_lead + 1, 11)):
                                key = (inn_thresh, lead_thresh, wp_thresh)
                                combo_key = (key, 'home')
                                if combo_key not in seen_combos_home:
                                    seen_combos_home.add(combo_key)
                                    bucket = entries[key]
                                    if home_won:
                                        bucket['wins'] += 1
                                    else:
                                        bucket['losses'] += 1
                                    bucket['wps'].append(round(home_wp, 3))

            # Check away side
            away_lead = away_score - home_score
            if away_lead > 0 and away_wp > 0.6:
                for wp_thresh in range(60, 100):
                    wp_pct = wp_thresh / 100.0
                    if away_wp >= wp_pct:
                        for inn_thresh in range(1, min(inning + 1, 10)):
                            for lead_thresh in range(1, min(away_lead + 1, 11)):
                                key = (inn_thresh, lead_thresh, wp_thresh)
                                combo_key = (key, 'away')
                                if combo_key not in seen_combos_away:
                                    seen_combos_away.add(combo_key)
                                    bucket = entries[key]
                                    if not home_won:
                                        bucket['wins'] += 1
                                    else:
                                        bucket['losses'] += 1
                                    bucket['wps'].append(round(away_wp, 3))

    return entries


def find_profit_sweet_spots(entries):
    """
    For each entry point combo, compute:
    - Accuracy (win rate)
    - Expected value at various buy prices
    - Profit per bet
    - Volume (bets per game day)

    Find the combos that maximize TOTAL DAILY EV.
    """
    GAME_DAYS = 178  # approximate MLB game days in a season
    BET_SIZE = 100   # dollars per bet

    results = []

    for (inn_thresh, lead_thresh, wp_thresh), data in entries.items():
        total = data['wins'] + data['losses']
        if total < 10:  # need statistical significance
            continue

        accuracy = data['wins'] / total
        if accuracy < 0.88:  # minimum accuracy floor
            continue

        # The buy price should be at or below the WP threshold
        # On Polymarket, if WP is at 85%, market price is ~$0.85
        # We place limit buy at various prices below the market
        buy_price = wp_thresh / 100.0

        # Also consider buying at a discount (5-10 cents below WP)
        # since limit orders may not fill at exact market price
        for discount in [0, 0.03, 0.05, 0.08, 0.10]:
            actual_buy = buy_price - discount
            if actual_buy < 0.50:
                continue

            profit_if_win = 1.0 - actual_buy
            loss_if_lose = actual_buy

            ev_per_share = (accuracy * profit_if_win) - ((1 - accuracy) * loss_if_lose)

            if ev_per_share <= 0:
                continue

            bets_per_day = total / GAME_DAYS
            daily_ev = ev_per_share * BET_SIZE * bets_per_day
            season_ev = daily_ev * GAME_DAYS

            results.append({
                'min_inning': inn_thresh,
                'min_lead': lead_thresh,
                'min_wp': wp_thresh / 100.0,
                'buy_price': round(actual_buy, 2),
                'discount': round(discount, 2),
                'accuracy': round(accuracy, 4),
                'total_games': total,
                'wins': data['wins'],
                'losses': data['losses'],
                'profit_if_win': round(profit_if_win, 2),
                'loss_if_lose': round(loss_if_lose, 2),
                'ev_per_share': round(ev_per_share, 4),
                'ev_per_bet': round(ev_per_share * BET_SIZE, 2),
                'bets_per_day': round(bets_per_day, 2),
                'daily_ev': round(daily_ev, 2),
                'season_ev': round(season_ev, 0),
            })

    # Sort by daily EV (total profit per day)
    results.sort(key=lambda x: x['daily_ev'], reverse=True)
    return results


def find_non_overlapping_tiers(results, max_tiers=8):
    """
    Select the best non-overlapping tiers that maximize total daily EV.
    A tier "overlaps" if a more profitable tier already covers the same games.

    We want tiers that are DISTINCT entry points, not subsets of each other.
    """
    # Group by unique (inning, lead, wp) combo first
    best_by_combo = {}
    for r in results:
        key = (r['min_inning'], r['min_lead'], r['min_wp'])
        # Pick the discount level that maximizes daily EV for each combo
        if key not in best_by_combo or r['daily_ev'] > best_by_combo[key]['daily_ev']:
            best_by_combo[key] = r

    # Sort combos by daily EV
    combos = sorted(best_by_combo.values(), key=lambda x: x['daily_ev'], reverse=True)

    selected = []
    for combo in combos:
        if len(selected) >= max_tiers:
            break

        # Check if this combo is a strict subset of an already selected tier
        # A is subset of B if A.inning >= B.inning AND A.lead >= B.lead AND A.wp >= B.wp
        is_subset = False
        is_superset = False
        for sel in selected:
            if (combo['min_inning'] >= sel['min_inning'] and
                combo['min_lead'] >= sel['min_lead'] and
                combo['min_wp'] >= sel['min_wp']):
                is_subset = True
                break
            if (combo['min_inning'] <= sel['min_inning'] and
                combo['min_lead'] <= sel['min_lead'] and
                combo['min_wp'] <= sel['min_wp']):
                is_superset = True
                break

        if not is_subset and not is_superset:
            selected.append(combo)

    return selected


def print_profit_report(results, tiers):
    """Print the profit-optimized analysis."""
    print("=" * 85)
    print("PROFIT-OPTIMIZED LIVE MONEYLINE STRATEGY")
    print("Target: 90%+ accuracy with MAXIMUM profitability")
    print("=" * 85)

    print("\n TOP 30 ENTRY POINTS BY DAILY EXPECTED VALUE:")
    print(f"{'#':>3} {'Inn':>4} {'Lead':>5} {'WP':>5} {'Buy$':>6} {'Acc':>7} "
          f"{'Games':>6} {'W':>5} {'L':>3} {'EV/bet':>8} {'Bets/d':>7} {'Daily$':>8} {'Season$':>9}")
    print("-" * 85)

    for i, r in enumerate(results[:30]):
        print(f"{i+1:>3} {r['min_inning']:>4}+ {r['min_lead']:>4}+ "
              f"{r['min_wp']:>4.0%} ${r['buy_price']:>5.2f} "
              f"{r['accuracy']:>6.1%} {r['total_games']:>6} "
              f"{r['wins']:>5} {r['losses']:>3} "
              f"${r['ev_per_bet']:>6.2f} {r['bets_per_day']:>6.1f} "
              f"${r['daily_ev']:>7.2f} ${r['season_ev']:>8,.0f}")

    print("\n\n" + "=" * 85)
    print("SELECTED NON-OVERLAPPING TIERS (Profit-Maximized)")
    print("=" * 85)

    total_daily_ev = 0
    total_bets = 0

    for i, t in enumerate(tiers):
        tier_name = f"PROFIT TIER {i+1}"
        print(f"\n  {tier_name}")
        print(f"    Rule: Inning {t['min_inning']}+ AND {t['min_lead']}+ run lead AND {t['min_wp']:.0%}+ WP")
        print(f"    Buy at: ${t['buy_price']:.2f} (limit order)")
        print(f"    Accuracy: {t['accuracy']:.1%} ({t['wins']}W / {t['losses']}L in {t['total_games']} games)")
        print(f"    Profit/win: ${t['profit_if_win']:.2f}/share | Loss/lose: ${t['loss_if_lose']:.2f}/share")
        print(f"    EV per $100 bet: ${t['ev_per_bet']:.2f}")
        print(f"    Bets/day: ~{t['bets_per_day']:.1f}")
        print(f"    Daily EV: ${t['daily_ev']:.2f}")
        print(f"    Season EV: ${t['season_ev']:,.0f}")
        total_daily_ev += t['daily_ev']
        total_bets += t['bets_per_day']

    print(f"\n  {'=' * 50}")
    print(f"  COMBINED DAILY EV: ${total_daily_ev:.2f} (~{total_bets:.1f} bets/day)")
    print(f"  COMBINED SEASON EV: ${total_daily_ev * 178:,.0f}")
    print(f"  Assuming $100/bet")

    # Compare to old strategy
    old_daily = sum([1.7 * 0.01 * 100, 0.8 * 0.04 * 100, 0.6 * 0.05 * 100])
    print(f"\n  vs OLD STRATEGY (100% accuracy, $0.95-$0.99 buys):")
    print(f"    Old daily EV: ${old_daily:.2f}")
    print(f"    Old season EV: ${old_daily * 178:,.0f}")
    print(f"    IMPROVEMENT: {total_daily_ev / old_daily:.1f}x more profitable!")

    return {
        'total_daily_ev': total_daily_ev,
        'total_bets_per_day': total_bets,
        'season_ev': total_daily_ev * 178,
        'tiers': tiers,
    }


def run_profit_analysis():
    """Full profit optimization pipeline."""
    print("Loading ESPN live win probability cache...")
    entries = analyze_entry_points()
    print(f"Computed {len(entries)} unique entry point combinations\n")

    print("Finding profit sweet spots...")
    results = find_profit_sweet_spots(entries)
    print(f"Found {len(results)} profitable entry points (90%+ accuracy)\n")

    print("Selecting non-overlapping tiers...")
    tiers = find_non_overlapping_tiers(results)

    summary = print_profit_report(results, tiers)

    # Save results
    output = {
        'strategy': 'profit_optimized_live_moneyline',
        'target_accuracy': '90%+',
        'optimization': 'maximize_daily_ev',
        'top_entries': results[:50],
        'selected_tiers': tiers,
        'summary': summary,
    }

    output_file = os.path.join(DATA_DIR, 'profit_optimized_tiers.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results, tiers


if __name__ == '__main__':
    run_profit_analysis()
