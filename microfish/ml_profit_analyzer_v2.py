"""
Profit-Optimized Live Entry Point Analyzer v2
================================================
REALISTIC VERSION: Buy price = actual ESPN WP at entry point (≈ market price).

Key insight: On Polymarket, the live market price roughly tracks ESPN WP.
So if ESPN says 78% WP, you can buy at ~$0.78. We want to find game
situations where the REAL win rate is significantly higher than the ESPN WP,
creating a mispricing edge.

Example: In the 7th inning with a 3-run lead, ESPN might show 82% WP,
but historically teams in that spot win 95%. That's a buy at $0.82 with
$0.18 profit/share and 95% accuracy → massive EV.
"""

import json
import os
from collections import defaultdict
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'moneyline')
CACHE_FILE = os.path.join(DATA_DIR, 'espn_live_winprob_cache.json')


def load_cache():
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    valid = {k: v for k, v in cache.items()
             if v.get('n_plays', 0) > 0 and not v.get('error')}
    return valid


def analyze_realistic_entries():
    """
    For each game, find the FIRST play that meets each condition.
    Record the ACTUAL ESPN WP at that moment — that's the realistic buy price.
    Then compute win rate and profit based on real buy prices.
    """
    games = load_cache()
    print(f"Analyzing {len(games)} games with realistic buy prices...\n")

    # Define conditions to test
    # Each condition: (min_inning, min_lead, description)
    conditions = []
    for inn in range(1, 10):
        for lead in range(1, 11):
            conditions.append((inn, lead))

    # For each condition, collect: list of (actual_wp_at_entry, did_team_win)
    cond_results = defaultdict(list)

    for eid, game in games.items():
        timeline = game.get('timeline', [])
        if not timeline:
            continue

        home_score_final = game.get('home_score', 0)
        away_score_final = game.get('away_score', 0)
        if home_score_final == away_score_final:
            continue

        home_won = home_score_final > away_score_final

        # Track first trigger per condition per side
        triggered_home = set()
        triggered_away = set()

        for play in timeline:
            inning = play.get('inning', 0)
            home_score = play.get('homeScore', 0)
            away_score = play.get('awayScore', 0)
            home_wp = play.get('homeWP', 0.5)
            away_wp = play.get('awayWP', 0.5)

            if inning < 1:
                continue

            # Home side leading
            home_lead = home_score - away_score
            if home_lead >= 1:
                for inn_thresh in range(1, inning + 1):
                    for lead_thresh in range(1, home_lead + 1):
                        key = (inn_thresh, lead_thresh)
                        if key not in triggered_home:
                            triggered_home.add(key)
                            cond_results[key].append({
                                'wp': round(home_wp, 4),
                                'won': home_won,
                                'inning_actual': inning,
                                'lead_actual': home_lead,
                            })

            # Away side leading
            away_lead = away_score - home_score
            if away_lead >= 1:
                for inn_thresh in range(1, inning + 1):
                    for lead_thresh in range(1, away_lead + 1):
                        key = (inn_thresh, lead_thresh)
                        if key not in triggered_away:
                            triggered_away.add(key)
                            cond_results[key].append({
                                'wp': round(away_wp, 4),
                                'won': not home_won,
                                'inning_actual': inning,
                                'lead_actual': away_lead,
                            })

    return cond_results


def compute_tier_stats(entries):
    """
    For each condition, compute:
    - Win rate (accuracy)
    - Average buy price (avg ESPN WP at entry)
    - Median buy price
    - EV per bet using actual buy prices
    """
    GAME_DAYS = 178
    BET_SIZE = 100

    results = []

    for (inn_thresh, lead_thresh), plays in entries.items():
        if len(plays) < 20:
            continue

        wins = sum(1 for p in plays if p['won'])
        losses = len(plays) - wins
        accuracy = wins / len(plays)

        if accuracy < 0.88:
            continue

        # Actual buy prices (ESPN WP at entry)
        wps = sorted([p['wp'] for p in plays])
        avg_wp = sum(wps) / len(wps)
        median_wp = wps[len(wps) // 2]
        p10_wp = wps[int(len(wps) * 0.10)]  # 10th percentile (cheapest entries)
        p25_wp = wps[int(len(wps) * 0.25)]  # 25th percentile

        # Compute EV using ACTUAL individual buy prices
        total_ev = 0
        for p in plays:
            buy_price = p['wp']
            if p['won']:
                total_ev += (1.0 - buy_price)  # profit
            else:
                total_ev -= buy_price  # loss
        ev_per_play = total_ev / len(plays)

        bets_per_day = len(plays) / GAME_DAYS
        daily_ev = ev_per_play * BET_SIZE * bets_per_day
        season_ev = daily_ev * GAME_DAYS

        # Also compute: what if we only take entries where WP < some cap?
        # This means we only buy when the price is "cheap enough"
        wp_cap_results = {}
        for cap in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            capped = [p for p in plays if p['wp'] <= cap]
            if len(capped) < 5:
                continue
            c_wins = sum(1 for p in capped if p['won'])
            c_acc = c_wins / len(capped) if capped else 0
            c_ev = 0
            for p in capped:
                if p['won']:
                    c_ev += (1.0 - p['wp'])
                else:
                    c_ev -= p['wp']
            c_ev_per = c_ev / len(capped) if capped else 0
            c_bpd = len(capped) / GAME_DAYS
            wp_cap_results[cap] = {
                'cap': cap,
                'n': len(capped),
                'accuracy': round(c_acc, 4),
                'ev_per_play': round(c_ev_per, 4),
                'bets_per_day': round(c_bpd, 2),
                'daily_ev': round(c_ev_per * BET_SIZE * c_bpd, 2),
            }

        results.append({
            'min_inning': inn_thresh,
            'min_lead': lead_thresh,
            'total_games': len(plays),
            'wins': wins,
            'losses': losses,
            'accuracy': round(accuracy, 4),
            'avg_buy_price': round(avg_wp, 3),
            'median_buy_price': round(median_wp, 3),
            'p10_buy_price': round(p10_wp, 3),
            'p25_buy_price': round(p25_wp, 3),
            'ev_per_play': round(ev_per_play, 4),
            'ev_per_bet': round(ev_per_play * BET_SIZE, 2),
            'bets_per_day': round(bets_per_day, 2),
            'daily_ev': round(daily_ev, 2),
            'season_ev': round(season_ev, 0),
            'wp_cap_results': wp_cap_results,
        })

    results.sort(key=lambda x: x['daily_ev'], reverse=True)
    return results


def find_best_capped_tiers(results):
    """
    Find the best tiers when we add a WP cap (max buy price).
    This is the key: only buy when the market underprices the situation.

    For each condition + WP cap combo, find highest daily EV with 90%+ accuracy.
    """
    all_capped = []

    for r in results:
        for cap, cr in r.get('wp_cap_results', {}).items():
            if cr['accuracy'] < 0.90 or cr['n'] < 10:
                continue
            all_capped.append({
                'min_inning': r['min_inning'],
                'min_lead': r['min_lead'],
                'max_buy_price': cap,
                'accuracy': cr['accuracy'],
                'n_games': cr['n'],
                'ev_per_play': cr['ev_per_play'],
                'bets_per_day': cr['bets_per_day'],
                'daily_ev': cr['daily_ev'],
                'season_ev': round(cr['daily_ev'] * 178, 0),
            })

    all_capped.sort(key=lambda x: x['daily_ev'], reverse=True)

    # Select non-overlapping
    selected = []
    for c in all_capped:
        if len(selected) >= 10:
            break
        # Check overlap
        overlap = False
        for s in selected:
            # Same or stricter condition
            if (c['min_inning'] >= s['min_inning'] and
                c['min_lead'] >= s['min_lead'] and
                c['max_buy_price'] >= s['max_buy_price']):
                overlap = True
                break
            if (c['min_inning'] <= s['min_inning'] and
                c['min_lead'] <= s['min_lead'] and
                c['max_buy_price'] <= s['max_buy_price']):
                overlap = True
                break
        if not overlap:
            selected.append(c)

    return all_capped, selected


def print_report(results, all_capped, selected_tiers):
    """Print comprehensive profit report."""
    print("=" * 90)
    print("PROFIT-OPTIMIZED LIVE MONEYLINE STRATEGY v2 (Realistic Buy Prices)")
    print("Buy price = ESPN WP at entry ≈ Polymarket market price")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("SECTION 1: BEST CONDITIONS BY DAILY EV (no buy price cap)")
    print("=" * 90)
    print(f"{'#':>3} {'Inn':>4} {'Lead':>5} {'Games':>6} {'Acc':>7} {'AvgBuy':>7} "
          f"{'MedBuy':>7} {'EV/bet':>8} {'Bets/d':>7} {'Daily$':>8} {'Season$':>9}")
    print("-" * 80)

    for i, r in enumerate(results[:20]):
        print(f"{i+1:>3} {r['min_inning']:>4}+ {r['min_lead']:>4}+ "
              f"{r['total_games']:>6} {r['accuracy']:>6.1%} "
              f"${r['avg_buy_price']:>5.2f} ${r['median_buy_price']:>5.2f} "
              f"${r['ev_per_bet']:>6.2f} {r['bets_per_day']:>6.1f} "
              f"${r['daily_ev']:>7.2f} ${r['season_ev']:>8,.0f}")

    print("\n\n" + "=" * 90)
    print("SECTION 2: BEST CONDITIONS WITH BUY PRICE CAP (only buy when cheap)")
    print("This is where the REAL edge is — situations where market underprices")
    print("=" * 90)
    print(f"{'#':>3} {'Inn':>4} {'Lead':>5} {'MaxBuy':>7} {'Games':>6} {'Acc':>7} "
          f"{'EV/play':>8} {'Bets/d':>7} {'Daily$':>8} {'Season$':>9}")
    print("-" * 75)

    for i, c in enumerate(all_capped[:40]):
        print(f"{i+1:>3} {c['min_inning']:>4}+ {c['min_lead']:>4}+ "
              f"${c['max_buy_price']:>5.2f} {c['n_games']:>6} "
              f"{c['accuracy']:>6.1%} ${c['ev_per_play']:>6.4f} "
              f"{c['bets_per_day']:>6.1f} ${c['daily_ev']:>7.2f} "
              f"${c['season_ev']:>8,.0f}")

    print("\n\n" + "=" * 90)
    print("SECTION 3: SELECTED ALERT TIERS (Non-Overlapping, Profit-Maximized)")
    print("=" * 90)

    total_daily = 0
    total_bets = 0

    for i, t in enumerate(selected_tiers):
        print(f"\n  ALERT TIER {i+1}:")
        print(f"    Trigger: Inning {t['min_inning']}+ AND {t['min_lead']}+ run lead")
        print(f"    Max buy price: ${t['max_buy_price']:.2f}")
        print(f"    Accuracy: {t['accuracy']:.1%} ({t['n_games']} games validated)")
        print(f"    EV per $100 bet: ${t['ev_per_play'] * 100:.2f}")
        print(f"    Bets/day: ~{t['bets_per_day']:.1f}")
        print(f"    Daily EV: ${t['daily_ev']:.2f}")
        print(f"    Season EV: ${t['season_ev']:,.0f}")
        total_daily += t['daily_ev']
        total_bets += t['bets_per_day']

    print(f"\n  {'=' * 55}")
    print(f"  COMBINED DAILY EV: ${total_daily:.2f} (~{total_bets:.1f} bets/day)")
    print(f"  COMBINED SEASON EV: ${total_daily * 178:,.0f}")
    print(f"  Assuming $100/bet")

    return selected_tiers


def analyze_wp_distribution(entries):
    """
    Deep dive: for the most promising conditions, show the distribution
    of actual buy prices and win rates at each price bucket.
    """
    print("\n\n" + "=" * 90)
    print("SECTION 4: WIN RATE BY ACTUAL BUY PRICE (Key Mispricings)")
    print("These show where ESPN WP underestimates true win probability")
    print("=" * 90)

    # Most interesting conditions
    interesting = [
        (7, 2, "7th+ inning, 2+ run lead"),
        (6, 3, "6th+ inning, 3+ run lead"),
        (5, 4, "5th+ inning, 4+ run lead"),
        (8, 1, "8th+ inning, any lead"),
        (9, 1, "9th inning, any lead"),
        (7, 3, "7th+ inning, 3+ run lead"),
        (6, 2, "6th+ inning, 2+ run lead"),
        (5, 3, "5th+ inning, 3+ run lead"),
        (4, 4, "4th+ inning, 4+ run lead"),
        (3, 5, "3rd+ inning, 5+ run lead"),
    ]

    for inn, lead, desc in interesting:
        key = (inn, lead)
        plays = entries.get(key, [])
        if len(plays) < 20:
            continue

        print(f"\n  {desc} ({len(plays)} games):")
        print(f"  {'WP Range':>12} {'Games':>7} {'Wins':>6} {'Losses':>7} {'WinRate':>8} "
              f"{'AvgProfit':>10} {'Edge':>8}")
        print(f"  {'-'*65}")

        # Bucket by WP
        buckets = defaultdict(list)
        for p in plays:
            bucket = int(p['wp'] * 20) * 5  # 5% buckets
            buckets[bucket].append(p)

        for bucket in sorted(buckets.keys()):
            bp = buckets[bucket]
            n = len(bp)
            wins = sum(1 for p in bp if p['won'])
            wr = wins / n if n else 0
            avg_wp = sum(p['wp'] for p in bp) / n
            avg_profit = sum((1-p['wp']) if p['won'] else -p['wp'] for p in bp) / n
            edge = wr - avg_wp  # how much better we do than expected

            wp_lo = bucket / 100
            wp_hi = (bucket + 5) / 100
            marker = " ◀ EDGE" if edge > 0.05 and wr >= 0.90 and n >= 10 else ""
            print(f"  {wp_lo:.0%}-{wp_hi:.0%}   {n:>7} {wins:>6} {n-wins:>7} "
                  f"{wr:>7.1%} ${avg_profit:>9.4f} {edge:>+7.1%}{marker}")

    return


def run_analysis():
    """Full analysis pipeline."""
    entries = analyze_realistic_entries()
    print(f"Computed {len(entries)} unique (inning, lead) conditions\n")

    results = compute_tier_stats(entries)
    print(f"Found {len(results)} conditions with 88%+ accuracy\n")

    all_capped, selected = find_best_capped_tiers(results)
    print(f"Found {len(all_capped)} profitable capped conditions\n")

    print_report(results, all_capped, selected)
    analyze_wp_distribution(entries)

    # Save
    output = {
        'strategy': 'profit_optimized_live_v2',
        'date': datetime.now().isoformat(),
        'method': 'realistic_buy_prices_from_espn_wp',
        'selected_tiers': selected,
        'all_capped_top50': all_capped[:50],
        'top_conditions': results[:30],
    }

    output_file = os.path.join(DATA_DIR, 'profit_optimized_v2.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {output_file}")

    return entries, results, selected


if __name__ == '__main__':
    run_analysis()
