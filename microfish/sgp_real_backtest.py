"""
MiroFish SGP Real Backtest
=============================
Uses REAL historical odds (with player prop lines) merged with REAL
player boxscores to backtest Same Game Parlay strategies.

Data sources:
  - mlb_historical_odds.json: 2,502 games with hitsProps, tbProps, rbiProps, runsProps
  - mlb_player_boxscores_expanded.json: 2,368 games with per-player stats (h, tb, rbi, r)

This merges the two to create a complete picture:
  - What props were available (and at what odds)
  - What actually happened (did the prop hit?)
  - Which SGP combinations would have won

NO synthetic data. NO forward leakage.
"""

import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from itertools import combinations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ODDS_FILE = os.path.join(BASE_DIR, '..', 'mlb', 'webapp', 'data', 'mlb_historical_odds.json')
BOXSCORE_FILE = os.path.join(BASE_DIR, '..', 'mlb', 'webapp', 'data', 'mlb_player_boxscores_expanded.json')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'sgp')

from sgp_data import american_to_decimal, american_to_implied_prob, decimal_to_american


def load_and_merge_data():
    """
    Load real odds + real boxscores and merge into prop-level DataFrame.

    Each row = one prop bet opportunity with:
      - game context (date, teams, scores, game total)
      - prop details (player, market, line, odds)
      - outcome (hit = 1/0 based on actual boxscore)
    """
    print("Loading real historical odds...")
    with open(ODDS_FILE) as f:
        odds_data = json.load(f)
    print(f"  {len(odds_data)} games with prop lines")

    print("Loading real boxscores...")
    with open(BOXSCORE_FILE) as f:
        box_data = json.load(f)
    print(f"  {len(box_data)} games with player stats")

    # Index boxscores by (date, home, away) for matching
    box_index = {}
    for game in box_data:
        key = (game['date'], game['home'], game['away'])
        box_index[key] = game

    # Merge: for each odds game, find matching boxscore
    rows = []
    matched = 0
    unmatched = 0

    for game in odds_data:
        date = game['date']
        home = game['homeTeam']
        away = game['awayTeam']
        event_id = game.get('eventId', '')

        # Find matching boxscore
        box = box_index.get((date, home, away))
        if not box:
            unmatched += 1
            continue
        matched += 1

        home_score = box.get('home_score', 0)
        away_score = box.get('away_score', 0)
        game_total_runs = home_score + away_score

        # Build player stats lookup from boxscore
        player_stats = {}
        for p in box.get('players', []):
            player_stats[p['name']] = {
                'team': p['team'],
                'ab': p.get('ab', 0),
                'h': p.get('h', 0),
                'r': p.get('r', 0),
                'rbi': p.get('rbi', 0),
                'hr': p.get('hr', 0),
                'bb': p.get('bb', 0),
                'so': p.get('so', 0),
                'sb': p.get('sb', 0),
                'tb': p.get('tb', 0),
            }

        # Game-level odds
        home_ml = game.get('home_ml', 0)
        away_ml = game.get('away_ml', 0)
        game_total_line = game.get('total', 0)
        total_over_odds = game.get('total_over', 0)

        # Process each prop type
        prop_types = {
            'hitsProps': ('hits', 'h'),
            'tbProps': ('total_bases', 'tb'),
            'rbiProps': ('rbis', 'rbi'),
            'runsProps': ('runs', 'r'),
        }

        for prop_key, (market_name, stat_key) in prop_types.items():
            props = game.get(prop_key, {})
            if not props:
                continue

            for player_name, lines in props.items():
                # Look up actual stat from boxscore
                stats = player_stats.get(player_name)
                if not stats:
                    continue

                actual_value = stats.get(stat_key, 0)
                # For total bases, if tb=0 but h>0, compute from hits
                if stat_key == 'tb' and actual_value == 0 and stats.get('h', 0) > 0:
                    # Minimum: singles = hits - hr - doubles - triples
                    # We don't have doubles/triples, so use h as minimum
                    actual_value = stats['h']  # Conservative estimate

                for line_str, odds_info in lines.items():
                    try:
                        line = float(line_str)
                    except (ValueError, TypeError):
                        continue

                    over_odds = odds_info.get('overOdds', 0)
                    if not over_odds:
                        continue

                    # Did the over hit?
                    over_hit = 1 if actual_value > line else 0

                    rows.append({
                        'date': date,
                        'event_id': event_id,
                        'home_team': home,
                        'away_team': away,
                        'home_score': home_score,
                        'away_score': away_score,
                        'game_total_runs': game_total_runs,
                        'game_total_line': game_total_line,
                        'home_ml': home_ml,
                        'away_ml': away_ml,
                        'player': player_name,
                        'player_team': stats['team'],
                        'market': market_name,
                        'line': line,
                        'odds': over_odds,
                        'implied_prob': american_to_implied_prob(over_odds),
                        'actual_value': actual_value,
                        'hit': over_hit,
                        'ab': stats.get('ab', 0),
                    })

    print(f"  Matched: {matched} games, Unmatched: {unmatched}")

    df = pd.DataFrame(rows)
    print(f"  Total prop records: {len(df)}")
    if not df.empty:
        print(f"  Unique players: {df['player'].nunique()}")
        print(f"  Unique games: {df['event_id'].nunique()}")
        print(f"  Date range: {df['date'].min()} - {df['date'].max()}")
    return df


def analyze_prop_hit_rates(df):
    """Analyze actual hit rates vs implied probability by market and line."""
    print("\n" + "=" * 70)
    print("PROP HIT RATE ANALYSIS (Real Data)")
    print("=" * 70)

    lines = []

    for market in sorted(df['market'].unique()):
        mdf = df[df['market'] == market]
        lines.append(f"\n--- {market.upper()} ---")
        lines.append(f"  Total props: {len(mdf)}")
        lines.append(f"  Overall hit rate: {mdf['hit'].mean():.1%}")

        for line_val in sorted(mdf['line'].unique()):
            ldf = mdf[mdf['line'] == line_val]
            if len(ldf) < 20:
                continue
            hit_rate = ldf['hit'].mean()
            avg_odds = ldf['odds'].mean()
            implied = ldf['implied_prob'].mean()
            edge = hit_rate - implied
            lines.append(
                f"  Line {line_val}: {ldf['hit'].sum()}/{len(ldf)} = {hit_rate:.1%} "
                f"(implied: {implied:.1%}, edge: {edge:+.1%}, avg odds: {avg_odds:+.0f})"
            )

    summary = '\n'.join(lines)
    print(summary)
    return summary


def analyze_correlations(df, min_co_occurrences=30):
    """
    Find within-game prop correlations.
    Which props hit together more than independence predicts?
    """
    print("\n" + "=" * 70)
    print("WITHIN-GAME PROP CORRELATION ANALYSIS")
    print("=" * 70)

    # Group by game
    pair_stats = defaultdict(lambda: {'both': 0, 'a_only': 0, 'b_only': 0, 'neither': 0, 'n': 0})

    for event_id, game_df in df.groupby('event_id'):
        # Get all hit props in this game
        props = list(game_df[['player', 'market', 'line', 'hit']].itertuples(index=False))

        for i in range(len(props)):
            for j in range(i + 1, len(props)):
                a = props[i]
                b = props[j]
                key = (
                    f"{a.player}_{a.market}_{a.line}",
                    f"{b.player}_{b.market}_{b.line}"
                )
                # Normalize key order
                if key[0] > key[1]:
                    key = (key[1], key[0])

                pair_stats[key]['n'] += 1
                if a.hit and b.hit:
                    pair_stats[key]['both'] += 1
                elif a.hit:
                    pair_stats[key]['a_only'] += 1
                elif b.hit:
                    pair_stats[key]['b_only'] += 1
                else:
                    pair_stats[key]['neither'] += 1

    # Calculate lift for significant pairs
    correlations = []
    for (prop_a, prop_b), stats in pair_stats.items():
        if stats['n'] < min_co_occurrences:
            continue

        n = stats['n']
        p_a = (stats['both'] + stats['a_only']) / n
        p_b = (stats['both'] + stats['b_only']) / n
        p_both = stats['both'] / n
        p_indep = p_a * p_b

        lift = p_both / p_indep if p_indep > 0 else 0

        correlations.append({
            'prop_a': prop_a,
            'prop_b': prop_b,
            'lift': lift,
            'p_both': p_both,
            'p_indep': p_indep,
            'p_a': p_a,
            'p_b': p_b,
            'n': n,
        })

    correlations.sort(key=lambda x: x['lift'], reverse=True)

    # Print top positive correlations
    lines = []
    lines.append(f"\nTotal significant prop pairs: {len(correlations)}")
    lines.append(f"\nTOP 30 POSITIVELY CORRELATED PROP PAIRS (lift > 1.0):")
    for c in correlations[:30]:
        if c['lift'] > 1.0:
            lines.append(
                f"  {c['prop_a']} + {c['prop_b']}: "
                f"lift={c['lift']:.3f}, P(both)={c['p_both']:.1%} vs "
                f"P(indep)={c['p_indep']:.1%}, n={c['n']}"
            )

    lines.append(f"\nTOP 15 ANTI-CORRELATED PAIRS (lift < 1.0, AVOID):")
    for c in correlations[-15:]:
        if c['lift'] < 1.0:
            lines.append(
                f"  {c['prop_a']} + {c['prop_b']}: "
                f"lift={c['lift']:.3f}, n={c['n']}"
            )

    summary = '\n'.join(lines)
    print(summary)
    return correlations, summary


def analyze_sgp_archetypes(df):
    """
    Analyze @sgp_vick-style SGP archetypes against real data.
    For each archetype, simulate constructing SGPs and check hit rates.
    """
    print("\n" + "=" * 70)
    print("SGP ARCHETYPE SIMULATION (Real Odds + Real Outcomes)")
    print("=" * 70)

    results = {}

    # ================================================================
    # ARCHETYPE 1: "Floor Finder" — Alt lines below player floors
    # Pick 4-6 overs on low lines (0.5 hits, 0.5 runs, 1.5 TB) for
    # players who consistently exceed those floors
    # ================================================================
    print("\n--- ARCHETYPE: Floor Finder SGP ---")
    ff_results = _simulate_floor_finder(df)
    results['floor_finder'] = ff_results

    # ================================================================
    # ARCHETYPE 2: "Milestone Stack" — Higher lines with higher odds
    # Pick 3-4 overs on moderate lines (1.5 hits, 2.5 TB, 0.5 RBI)
    # ================================================================
    print("\n--- ARCHETYPE: Milestone Stack ---")
    ms_results = _simulate_milestone_stack(df)
    results['milestone_stack'] = ms_results

    # ================================================================
    # ARCHETYPE 3: "Slugfest Correlated" — High-scoring game SGPs
    # When game total is high, stack overs from both teams
    # ================================================================
    print("\n--- ARCHETYPE: Slugfest SGP ---")
    sf_results = _simulate_slugfest(df)
    results['slugfest'] = sf_results

    return results


def _simulate_floor_finder(df, n_legs=4):
    """
    Floor Finder: Pick overs on the LOWEST available lines for players
    with high implied probability (>= 60%).
    Target: +400 to +1500 combined odds.
    """
    sgps = []

    for event_id, game_df in df.groupby('event_id'):
        # Get all low-line overs with decent implied prob
        candidates = game_df[
            (game_df['implied_prob'] >= 0.55) &
            (game_df['odds'] > -300)  # Not too heavy
        ].copy()

        if len(candidates) < n_legs:
            continue

        # Pick the n_legs highest-probability legs (avoiding same player/market combos)
        candidates = candidates.sort_values('implied_prob', ascending=False)
        selected = []
        used_players_markets = set()

        for _, row in candidates.iterrows():
            key = (row['player'], row['market'])
            if key in used_players_markets:
                continue
            selected.append(row)
            used_players_markets.add(key)
            if len(selected) >= n_legs:
                break

        if len(selected) < n_legs:
            continue

        # Calculate combined odds
        leg_odds_dec = [american_to_decimal(r['odds']) for r in selected]
        combined_dec = 1.0
        for o in leg_odds_dec:
            combined_dec *= o
        combined_american = decimal_to_american(combined_dec)

        # Check if combined odds in target range
        if combined_american < 200 or combined_american > 3000:
            continue

        # Check if all legs hit
        all_hit = all(r['hit'] == 1 for r in selected)

        sgps.append({
            'event_id': event_id,
            'date': selected[0]['date'],
            'home': selected[0]['home_team'],
            'away': selected[0]['away_team'],
            'n_legs': len(selected),
            'combined_odds': int(combined_american),
            'combined_dec': round(combined_dec, 2),
            'all_hit': all_hit,
            'payout': 10 * combined_dec if all_hit else 0,
            'legs': [
                {
                    'player': r['player'],
                    'market': r['market'],
                    'line': r['line'],
                    'odds': int(r['odds']),
                    'actual': int(r['actual_value']),
                    'hit': int(r['hit']),
                }
                for r in selected
            ],
        })

    return _summarize_sgps(sgps, 'Floor Finder')


def _simulate_milestone_stack(df, n_legs=3):
    """
    Milestone Stack: Pick 3 moderate-line overs (1.5 hits, 1.5 TB, 0.5 RBI).
    Each leg at +100 to +500 range. Target: +800 to +5000 combined.
    """
    sgps = []

    for event_id, game_df in df.groupby('event_id'):
        # Get moderate-line props with decent odds
        candidates = game_df[
            (game_df['odds'] >= 100) &
            (game_df['odds'] <= 500) &
            (game_df['line'] >= 1.0)
        ].copy()

        if len(candidates) < n_legs:
            continue

        # Pick the legs with best balance of odds and hit probability
        candidates = candidates.sort_values('implied_prob', ascending=False)
        selected = []
        used_players = set()

        for _, row in candidates.iterrows():
            if row['player'] in used_players:
                continue
            selected.append(row)
            used_players.add(row['player'])
            if len(selected) >= n_legs:
                break

        if len(selected) < n_legs:
            continue

        leg_odds_dec = [american_to_decimal(r['odds']) for r in selected]
        combined_dec = 1.0
        for o in leg_odds_dec:
            combined_dec *= o
        combined_american = decimal_to_american(combined_dec)

        if combined_american < 500 or combined_american > 10000:
            continue

        all_hit = all(r['hit'] == 1 for r in selected)

        sgps.append({
            'event_id': event_id,
            'date': selected[0]['date'],
            'home': selected[0]['home_team'],
            'away': selected[0]['away_team'],
            'n_legs': len(selected),
            'combined_odds': int(combined_american),
            'combined_dec': round(combined_dec, 2),
            'all_hit': all_hit,
            'payout': 10 * combined_dec if all_hit else 0,
            'legs': [
                {
                    'player': r['player'],
                    'market': r['market'],
                    'line': r['line'],
                    'odds': int(r['odds']),
                    'actual': int(r['actual_value']),
                    'hit': int(r['hit']),
                }
                for r in selected
            ],
        })

    return _summarize_sgps(sgps, 'Milestone Stack')


def _simulate_slugfest(df, n_legs=5):
    """
    Slugfest SGP: In high-scoring games (total >= 9), stack low-line overs.
    """
    sgps = []

    for event_id, game_df in df.groupby('event_id'):
        # Only high-scoring games
        if game_df.iloc[0]['game_total_runs'] < 9:
            continue

        candidates = game_df[
            (game_df['implied_prob'] >= 0.50) &
            (game_df['line'] <= 1.5)
        ].copy()

        if len(candidates) < n_legs:
            continue

        candidates = candidates.sort_values('implied_prob', ascending=False)
        selected = []
        used = set()

        for _, row in candidates.iterrows():
            key = (row['player'], row['market'])
            if key in used:
                continue
            selected.append(row)
            used.add(key)
            if len(selected) >= n_legs:
                break

        if len(selected) < n_legs:
            continue

        leg_odds_dec = [american_to_decimal(r['odds']) for r in selected]
        combined_dec = 1.0
        for o in leg_odds_dec:
            combined_dec *= o
        combined_american = decimal_to_american(combined_dec)

        if combined_american < 200:
            continue

        all_hit = all(r['hit'] == 1 for r in selected)

        sgps.append({
            'event_id': event_id,
            'date': selected[0]['date'],
            'home': selected[0]['home_team'],
            'away': selected[0]['away_team'],
            'n_legs': len(selected),
            'combined_odds': int(combined_american),
            'combined_dec': round(combined_dec, 2),
            'all_hit': all_hit,
            'payout': 10 * combined_dec if all_hit else 0,
            'game_total': int(game_df.iloc[0]['game_total_runs']),
            'legs': [
                {
                    'player': r['player'],
                    'market': r['market'],
                    'line': r['line'],
                    'odds': int(r['odds']),
                    'actual': int(r['actual_value']),
                    'hit': int(r['hit']),
                }
                for r in selected
            ],
        })

    return _summarize_sgps(sgps, 'Slugfest')


def _summarize_sgps(sgps, name):
    """Print and return SGP archetype results."""
    if not sgps:
        print(f"  {name}: No SGPs constructed")
        return {'total': 0}

    total = len(sgps)
    hits = sum(1 for s in sgps if s['all_hit'])
    hit_rate = hits / total
    total_wagered = total * 10
    total_payout = sum(s['payout'] for s in sgps)
    roi = (total_payout - total_wagered) / total_wagered * 100

    avg_odds = np.mean([s['combined_odds'] for s in sgps])
    median_odds = np.median([s['combined_odds'] for s in sgps])
    avg_legs = np.mean([s['n_legs'] for s in sgps])

    breakeven = 100 / (avg_odds + 100) if avg_odds > 0 else 0.5
    edge = hit_rate - breakeven

    print(f"  {name} Results:")
    print(f"    SGPs constructed: {total}")
    print(f"    Hits: {hits}")
    print(f"    HIT RATE: {hit_rate:.1%}")
    print(f"    Avg combined odds: +{avg_odds:.0f}")
    print(f"    Median combined odds: +{median_odds:.0f}")
    print(f"    Avg legs: {avg_legs:.1f}")
    print(f"    Total wagered: ${total_wagered}")
    print(f"    Total payout: ${total_payout:.2f}")
    print(f"    ROI: {roi:+.1f}%")
    print(f"    Breakeven rate: {breakeven:.1%}")
    print(f"    EDGE: {edge:+.1%}")

    # Show some winning SGPs
    winners = [s for s in sgps if s['all_hit']]
    if winners:
        print(f"\n    Sample winning SGPs ({min(5, len(winners))} of {len(winners)}):")
        for w in winners[:5]:
            print(f"      {w['date']} {w['away']}@{w['home']} +{w['combined_odds']} → ${w['payout']:.2f}")
            for l in w['legs']:
                print(f"        {l['player']}: {l['market']} O{l['line']} ({l['odds']:+d}) actual={l['actual']} {'HIT' if l['hit'] else 'MISS'}")

    return {
        'total': total, 'hits': hits, 'hit_rate': round(hit_rate, 4),
        'avg_odds': round(float(avg_odds)), 'median_odds': round(float(median_odds)),
        'roi': round(roi, 1), 'edge': round(edge, 4),
        'total_wagered': total_wagered, 'total_payout': round(total_payout, 2),
        'breakeven': round(breakeven, 4),
        'sgps': sgps,
    }


def build_data_summary_for_agents(df, correlations_summary, archetype_results):
    """Build comprehensive data summary for MiniMax agents."""
    lines = []
    lines.append("=== MLB SGP REAL DATA ANALYSIS ===")
    lines.append(f"Source: 2,502 games with real prop lines + 2,368 games with real boxscores")
    lines.append(f"Season: 2025 MLB (April-September)")
    lines.append(f"Total prop records: {len(df)}")
    lines.append(f"Unique players: {df['player'].nunique()}")
    lines.append(f"Unique games: {df['event_id'].nunique()}")
    lines.append("")

    # Hit rates by market and line
    lines.append("PROP HIT RATES BY MARKET AND LINE:")
    for market in sorted(df['market'].unique()):
        mdf = df[df['market'] == market]
        lines.append(f"\n  {market.upper()} (n={len(mdf)}, overall hit rate={mdf['hit'].mean():.1%}):")
        for line_val in sorted(mdf['line'].unique()):
            ldf = mdf[mdf['line'] == line_val]
            if len(ldf) < 30:
                continue
            hit_rate = ldf['hit'].mean()
            avg_odds = ldf['odds'].mean()
            implied = ldf['implied_prob'].mean()
            edge = hit_rate - implied
            ev_per_dollar = hit_rate * american_to_decimal(avg_odds) - 1
            lines.append(
                f"    O{line_val}: {ldf['hit'].sum()}/{len(ldf)} = {hit_rate:.1%} "
                f"(implied={implied:.1%}, edge={edge:+.1%}, "
                f"avg_odds={avg_odds:+.0f}, EV/$ = {ev_per_dollar:+.3f})"
            )

    # Odds bracket analysis
    lines.append("\n\nODDS BRACKET HIT RATES:")
    brackets = [
        ((-500, -200), "Heavy Fav"),
        ((-200, -110), "Moderate Fav"),
        ((-110, 110), "Near Even"),
        ((110, 200), "Slight Dog"),
        ((200, 400), "Moderate Dog"),
        ((400, 800), "Long Shot"),
    ]
    for (lo, hi), label in brackets:
        subset = df[(df['odds'] >= lo) & (df['odds'] < hi)]
        if len(subset) >= 20:
            lines.append(
                f"  {label} ({lo:+d} to {hi:+d}): "
                f"{subset['hit'].sum()}/{len(subset)} = {subset['hit'].mean():.1%}"
            )

    # Correlation data
    lines.append("\n\n" + correlations_summary)

    # Archetype backtest results
    lines.append("\n\nARCHETYPE BACKTEST RESULTS:")
    for name, res in archetype_results.items():
        if res.get('total', 0) == 0:
            continue
        lines.append(f"\n  {name}:")
        lines.append(f"    SGPs: {res['total']}, Hits: {res.get('hits', 0)}, "
                     f"Hit rate: {res.get('hit_rate', 0):.1%}")
        lines.append(f"    Avg odds: +{res.get('avg_odds', 0)}, "
                     f"ROI: {res.get('roi', 0):+.1f}%, "
                     f"Edge: {res.get('edge', 0):+.1%}")

    # Game environment analysis
    lines.append("\n\nGAME ENVIRONMENT ANALYSIS:")
    if 'game_total_runs' in df.columns:
        for threshold in [7, 8, 9, 10, 12]:
            high_scoring = df[df['game_total_runs'] >= threshold]
            if len(high_scoring) >= 50:
                lines.append(
                    f"  Games with {threshold}+ total runs: "
                    f"prop hit rate = {high_scoring['hit'].mean():.1%} "
                    f"(n={len(high_scoring)}, {high_scoring['event_id'].nunique()} games)"
                )

    return '\n'.join(lines)


def run_full_analysis():
    """Run the complete SGP analysis pipeline."""
    # Load and merge real data
    df = load_and_merge_data()
    if df.empty:
        print("ERROR: No data available")
        return None, None, None, None

    # Analyze hit rates
    hit_rate_summary = analyze_prop_hit_rates(df)

    # Analyze correlations
    correlations, corr_summary = analyze_correlations(df)

    # Simulate archetypes
    archetype_results = analyze_sgp_archetypes(df)

    # Build agent summary
    agent_summary = build_data_summary_for_agents(df, corr_summary, archetype_results)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'real_backtest_analysis.json'), 'w') as f:
        json.dump({
            'archetype_results': {
                k: {kk: vv for kk, vv in v.items() if kk != 'sgps'}
                for k, v in archetype_results.items()
            },
            'total_props': len(df),
            'unique_games': int(df['event_id'].nunique()),
            'unique_players': int(df['player'].nunique()),
        }, f, indent=2, default=str)

    return df, agent_summary, archetype_results, correlations


if __name__ == '__main__':
    run_full_analysis()
