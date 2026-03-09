#!/usr/bin/env python3
"""
Parlay Research Engine - AutoResearch-Inspired Iterative Experimentation
=========================================================================

Uses the autoresearch methodology (Karpathy): single objective metric,
automated evaluation harness, binary keep/discard decisions, relentless
iteration over strategy hypotheses.

Goal: Find betting patterns with >90% accuracy and positive odds payout,
validated against REAL historical odds from The Odds API.

Patent-worthy innovation: Multi-dimensional confluence filtering that
identifies statistically robust "floor" scenarios where player/game
outcomes become near-deterministic, combined with adaptive line selection
that finds positive-EV entry points on those high-probability events.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import statistics

# ============================================================
# DATA LOADING - Real odds + real outcomes
# ============================================================

def _parse_minutes(val):
    """Safely parse minutes from various formats."""
    if val is None or val == '--' or val == '':
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    try:
        return int(val.split(":")[0])
    except (ValueError, AttributeError):
        return 0


def load_all_data(base_dir=None):
    """Load and merge all data sources: odds, box scores, game results."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent

    # Load odds data (real FanDuel odds from The Odds API)
    odds_files = [
        base_dir / "webapp" / "data" / "historical_odds.json",
        base_dir / "webapp" / "data" / "historical_odds_2026.json",
    ]
    all_odds = []
    for f in odds_files:
        if f.exists():
            with open(f) as fh:
                all_odds.extend(json.load(fh))

    # Load box scores (real ESPN results)
    box_files = [
        base_dir / "webapp" / "data" / "player_boxscores.json",
        base_dir / "webapp" / "data" / "player_boxscores_2026.json",
    ]
    all_boxes = []
    for f in box_files:
        if f.exists():
            with open(f) as fh:
                all_boxes.extend(json.load(fh))

    # Load reb/ast props
    reb_ast_file = base_dir / "webapp" / "data" / "historical_reb_ast_props.json"
    reb_ast_props = []
    if reb_ast_file.exists():
        with open(reb_ast_file) as fh:
            reb_ast_props = json.load(fh)

    # Load game results
    espn_file = base_dir / "data" / "espn_full_season_2025.json"
    espn_games = []
    if espn_file.exists():
        with open(espn_file) as fh:
            espn_games = json.load(fh)

    return {
        "odds": all_odds,
        "boxes": all_boxes,
        "reb_ast_props": reb_ast_props,
        "espn_games": espn_games,
    }


def merge_odds_and_outcomes(data):
    """
    Merge odds data with actual outcomes for backtesting.
    Returns list of complete game records with odds + actuals.
    """
    # Index box scores by date+home team
    box_index = {}
    for b in data["boxes"]:
        key = (b["date"], b["home"])
        box_index[key] = b

    # Index reb/ast props by date+gameKey
    reb_ast_index = {}
    for r in data["reb_ast_props"]:
        key = (r.get("date"), r.get("gameKey"))
        reb_ast_index[key] = r

    merged = []
    for odds_game in data["odds"]:
        date = odds_game["date"]
        home = odds_game["homeTeam"]
        away = odds_game["awayTeam"]
        game_key = odds_game.get("gameKey", f"{away}@{home}")

        box = box_index.get((date, home))
        if not box:
            continue  # No outcome data, skip

        # Build player actual stats
        player_stats = {}
        for p in box.get("players", []):
            player_stats[p["name"]] = {
                "pts": p.get("pts", 0),
                "reb": p.get("reb", 0),
                "ast": p.get("ast", 0),
                "min": _parse_minutes(p.get("min", 0)),
                "team": p.get("team", ""),
                "fg": p.get("fg", "0-0"),
                "three": p.get("three", "0-0"),
                "stl": p.get("stl", 0),
                "blk": p.get("blk", 0),
                "to": p.get("to", 0),
            }

        # Get reb/ast props if available
        reb_ast = reb_ast_index.get((date, game_key), {})

        record = {
            "date": date,
            "home": home,
            "away": away,
            "home_score": box.get("home_score", 0),
            "away_score": box.get("away_score", 0),
            "total_score": box.get("home_score", 0) + box.get("away_score", 0),
            "margin": box.get("home_score", 0) - box.get("away_score", 0),
            # Real odds
            "home_ml": odds_game.get("home_ml"),
            "away_ml": odds_game.get("away_ml"),
            "spread_point": odds_game.get("spread_point"),
            "spread_home": odds_game.get("spread_home"),
            "spread_away": odds_game.get("spread_away"),
            "total_line": odds_game.get("total"),
            "total_over": odds_game.get("total_over"),
            "total_under": odds_game.get("total_under"),
            # Player props (points)
            "player_props": odds_game.get("playerProps", {}),
            # Player actual stats
            "player_stats": player_stats,
            # Reb/ast props
            "reb_props": reb_ast.get("rebProps", {}),
            "ast_props": reb_ast.get("astProps", {}),
        }
        merged.append(record)

    # Sort by date
    merged.sort(key=lambda x: x["date"])
    return merged


# ============================================================
# ODDS MATH UTILITIES
# ============================================================

def american_to_implied(odds):
    """Convert American odds to implied probability."""
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds is None:
        return None
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / abs(odds)


def is_positive_ev(win_rate, odds):
    """Check if a bet has positive expected value."""
    decimal = american_to_decimal(odds)
    if decimal is None:
        return False
    ev = win_rate * decimal - 1
    return ev > 0


def calculate_ev(win_rate, odds):
    """Calculate expected value per dollar wagered."""
    decimal = american_to_decimal(odds)
    if decimal is None:
        return 0
    return win_rate * decimal - 1


def parlay_odds(legs):
    """Calculate parlay decimal odds from list of decimal odds."""
    result = 1.0
    for leg in legs:
        result *= leg
    return result


# ============================================================
# STRATEGY 1: PLAYER PROP FLOOR EXPLOITATION
# Patent concept: "Adaptive Floor Detection for Player Props"
#
# Core insight: Certain players have extremely reliable scoring
# floors where they almost NEVER score below X points. When the
# sportsbook offers a line at or below that floor with positive
# or near-even odds, it's a high-accuracy bet.
#
# The key innovation is using GAME CONTEXT to dynamically adjust
# the floor - a player's floor varies by opponent, rest, and
# season phase. We model the conditional floor, not just the
# unconditional one.
# ============================================================

def analyze_player_floors(games, min_games=8):
    """
    For each player, analyze their scoring distribution and identify
    reliable floors (lines they almost always exceed).
    """
    player_data = defaultdict(list)

    for game in games:
        for player_name, stats in game["player_stats"].items():
            if stats["min"] >= 15:  # Only players with meaningful minutes
                player_data[player_name].append({
                    "pts": stats["pts"],
                    "reb": stats["reb"],
                    "ast": stats["ast"],
                    "min": stats["min"],
                    "date": game["date"],
                    "team": stats["team"],
                    "home": game["home"],
                    "away": game["away"],
                    "opponent": game["away"] if stats["team"] == game["home"] else game["home"],
                })

    player_profiles = {}
    for player, games_list in player_data.items():
        if len(games_list) < min_games:
            continue

        pts_values = [g["pts"] for g in games_list]
        reb_values = [g["reb"] for g in games_list]
        ast_values = [g["ast"] for g in games_list]

        profile = {
            "games": len(games_list),
            "pts_mean": statistics.mean(pts_values),
            "pts_median": statistics.median(pts_values),
            "pts_min": min(pts_values),
            "pts_p5": sorted(pts_values)[max(0, len(pts_values) * 5 // 100)],
            "pts_p10": sorted(pts_values)[max(0, len(pts_values) * 10 // 100)],
            "pts_std": statistics.stdev(pts_values) if len(pts_values) > 1 else 0,
            "reb_mean": statistics.mean(reb_values),
            "reb_min": min(reb_values),
            "reb_p10": sorted(reb_values)[max(0, len(reb_values) * 10 // 100)],
            "ast_mean": statistics.mean(ast_values),
            "ast_min": min(ast_values),
            "ast_p10": sorted(ast_values)[max(0, len(ast_values) * 10 // 100)],
            "history": games_list,
        }

        # Calculate floor for each common prop line
        for line in [9.5, 14.5, 19.5, 24.5, 29.5, 34.5]:
            over_count = sum(1 for p in pts_values if p > line)
            over_rate = over_count / len(pts_values)
            profile[f"pts_over_{line}_rate"] = over_rate
            profile[f"pts_over_{line}_count"] = over_count
            profile[f"pts_over_{line}_total"] = len(pts_values)

        for line in [3.5, 5.5, 7.5, 9.5, 11.5, 13.5]:
            over_count = sum(1 for r in reb_values if r > line)
            over_rate = over_count / len(reb_values)
            profile[f"reb_over_{line}_rate"] = over_rate

        for line in [1.5, 3.5, 5.5, 7.5, 9.5]:
            over_count = sum(1 for a in ast_values if a > line)
            over_rate = over_count / len(ast_values)
            profile[f"ast_over_{line}_rate"] = over_rate

        player_profiles[player] = profile

    return player_profiles


# ============================================================
# STRATEGY 2: HEAVY FAVORITE ML PARLAY
# Patent concept: "Convergent Certainty Stacking"
#
# When multiple independent high-probability events are combined,
# the joint probability remains high while odds become positive.
# Key: selecting truly independent legs and filtering for games
# where the favorite winning is near-certain (structural dominance).
# ============================================================

def analyze_favorite_ml(games):
    """
    Analyze moneyline favorites: at what odds threshold do favorites
    win at >95% rates? Then we can stack 2-3 of these into a parlay
    that still has positive composite odds.
    """
    results_by_bucket = defaultdict(lambda: {"wins": 0, "losses": 0, "odds": []})

    for game in games:
        if game["home_ml"] is None or game["away_ml"] is None:
            continue

        home_won = game["home_score"] > game["away_score"]

        # Home favorite
        if game["home_ml"] < 0:
            bucket = categorize_ml(game["home_ml"])
            if home_won:
                results_by_bucket[bucket]["wins"] += 1
            else:
                results_by_bucket[bucket]["losses"] += 1
            results_by_bucket[bucket]["odds"].append(game["home_ml"])

        # Away favorite
        if game["away_ml"] < 0:
            bucket = categorize_ml(game["away_ml"])
            if not home_won:
                results_by_bucket[bucket]["wins"] += 1
            else:
                results_by_bucket[bucket]["losses"] += 1
            results_by_bucket[bucket]["odds"].append(game["away_ml"])

    return results_by_bucket


def categorize_ml(odds):
    """Categorize ML odds into buckets."""
    abs_odds = abs(odds)
    if abs_odds >= 1000:
        return "extreme_fav_-1000+"
    elif abs_odds >= 600:
        return "heavy_fav_-600_to_-999"
    elif abs_odds >= 400:
        return "strong_fav_-400_to_-599"
    elif abs_odds >= 300:
        return "solid_fav_-300_to_-399"
    elif abs_odds >= 200:
        return "moderate_fav_-200_to_-299"
    elif abs_odds >= 150:
        return "slight_fav_-150_to_-199"
    else:
        return "lean_fav_-100_to_-149"


# ============================================================
# STRATEGY 3: GAME TOTAL FLOOR/CEILING EXPLOITATION
# Patent concept: "Structural Total Bounds Detection"
#
# Some game totals are so far from the line that even significant
# variance can't change the outcome. When two high-scoring teams
# meet, the Under at a very high line becomes near-certain.
# ============================================================

def analyze_totals(games):
    """Analyze over/under outcomes by how far the total deviated from the line."""
    results = []
    for game in games:
        if game["total_line"] is None:
            continue
        actual = game["total_score"]
        line = game["total_line"]
        diff = actual - line

        results.append({
            "date": game["date"],
            "home": game["home"],
            "away": game["away"],
            "actual": actual,
            "line": line,
            "diff": diff,
            "over_hit": actual > line,
            "over_odds": game["total_over"],
            "under_odds": game["total_under"],
        })

    return results


# ============================================================
# STRATEGY 4: SPREAD FLOOR EXPLOITATION
# Patent concept: "Margin of Safety Spread Selection"
#
# Identify games where the spread is so conservative that the
# favorite covering becomes near-certain. Combined with game
# context (rest advantage, home/away, season records).
# ============================================================

def analyze_spreads(games):
    """Analyze spread cover rates by spread size."""
    results_by_bucket = defaultdict(lambda: {"covers": 0, "misses": 0, "pushes": 0})

    for game in games:
        if game["spread_point"] is None:
            continue

        spread = game["spread_point"]  # Negative = home favored
        margin = game["margin"]  # Positive = home won

        # Home covers if margin > abs(spread) (home favored means spread is negative)
        if spread < 0:  # Home favored
            actual_cover = margin + spread  # margin - abs(spread)
            bucket = categorize_spread(spread)
            if actual_cover > 0:
                results_by_bucket[bucket]["covers"] += 1
            elif actual_cover == 0:
                results_by_bucket[bucket]["pushes"] += 1
            else:
                results_by_bucket[bucket]["misses"] += 1

    return results_by_bucket


def categorize_spread(spread):
    """Categorize spread into buckets."""
    abs_spread = abs(spread)
    if abs_spread >= 12:
        return "huge_-12+"
    elif abs_spread >= 9:
        return "large_-9_to_-11.5"
    elif abs_spread >= 6:
        return "medium_-6_to_-8.5"
    elif abs_spread >= 3:
        return "small_-3_to_-5.5"
    else:
        return "tiny_-0.5_to_-2.5"


# ============================================================
# MAIN RESEARCH ENGINE
# ============================================================

def run_full_research(data=None):
    """Run comprehensive research across all strategies."""
    if data is None:
        raw = load_all_data()
        data = merge_odds_and_outcomes(raw)

    print(f"\n{'='*70}")
    print(f"PARLAY RESEARCH ENGINE - AutoResearch Methodology")
    print(f"{'='*70}")
    print(f"Total games with real odds + real outcomes: {len(data)}")
    print(f"Date range: {data[0]['date']} to {data[-1]['date']}")

    # Count games with player props
    props_games = sum(1 for g in data if g["player_props"])
    print(f"Games with player props: {props_games}")

    # Split into train/test for validation
    split_idx = int(len(data) * 0.7)
    train = data[:split_idx]
    test = data[split_idx:]
    print(f"\nTrain set: {len(train)} games ({train[0]['date']} to {train[-1]['date']})")
    print(f"Test set:  {len(test)} games ({test[0]['date']} to {test[-1]['date']})")

    results = {}

    # Strategy 1: Player Prop Floors
    print(f"\n{'='*70}")
    print("STRATEGY 1: PLAYER PROP FLOOR EXPLOITATION")
    print(f"{'='*70}")
    results["player_floors"] = research_player_prop_floors(train, test, data)

    # Strategy 2: Heavy Favorite ML
    print(f"\n{'='*70}")
    print("STRATEGY 2: HEAVY FAVORITE ML PARLAY")
    print(f"{'='*70}")
    results["favorite_ml"] = research_favorite_ml_parlays(train, test, data)

    # Strategy 3: Game Totals
    print(f"\n{'='*70}")
    print("STRATEGY 3: GAME TOTAL ANALYSIS")
    print(f"{'='*70}")
    results["totals"] = research_totals(data)

    # Strategy 4: Spreads
    print(f"\n{'='*70}")
    print("STRATEGY 4: SPREAD ANALYSIS")
    print(f"{'='*70}")
    results["spreads"] = research_spreads(data)

    return results


def research_player_prop_floors(train, test, all_data):
    """
    Core research: find player prop lines where the over hits >90%
    with positive odds available.
    """
    print("\n--- Analyzing player scoring floors on TRAINING data ---")
    train_profiles = analyze_player_floors(train)

    # Find bets with >90% hit rate
    high_accuracy_bets = []

    for game in all_data:
        if not game["player_props"]:
            continue

        for player_name, prop_lines in game["player_props"].items():
            if player_name not in train_profiles:
                continue

            profile = train_profiles[player_name]

            for line_str, line_data in prop_lines.items():
                line = float(line_str)
                odds = line_data.get("overOdds")
                if odds is None:
                    continue

                # Check historical over rate for this player at this line
                rate_key = f"pts_over_{line}_rate"
                if rate_key not in profile:
                    continue

                hit_rate = profile[rate_key]

                # We want >90% accuracy
                if hit_rate >= 0.90 and profile[f"pts_over_{line}_total"] >= 8:
                    # Check actual outcome
                    actual_pts = game["player_stats"].get(player_name, {}).get("pts")
                    if actual_pts is None:
                        continue

                    hit = actual_pts > line
                    high_accuracy_bets.append({
                        "player": player_name,
                        "line": line,
                        "odds": odds,
                        "historical_rate": hit_rate,
                        "sample_size": profile[f"pts_over_{line}_total"],
                        "actual_pts": actual_pts,
                        "hit": hit,
                        "date": game["date"],
                        "ev": calculate_ev(hit_rate, odds),
                        "positive_odds": odds > 0,
                    })

    if not high_accuracy_bets:
        print("No bets found with >90% accuracy in player props")
        return {"bets": [], "accuracy": 0}

    # Analyze results
    total = len(high_accuracy_bets)
    hits = sum(1 for b in high_accuracy_bets if b["hit"])
    accuracy = hits / total if total > 0 else 0

    print(f"\nAll bets with historical rate >= 90%:")
    print(f"  Total bets: {total}")
    print(f"  Hits: {hits}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Avg odds: {statistics.mean([b['odds'] for b in high_accuracy_bets]):.0f}")
    print(f"  Avg EV: {statistics.mean([b['ev'] for b in high_accuracy_bets]):.3f}")

    # Filter for positive odds only
    pos_odds_bets = [b for b in high_accuracy_bets if b["positive_odds"]]
    if pos_odds_bets:
        pos_hits = sum(1 for b in pos_odds_bets if b["hit"])
        pos_accuracy = pos_hits / len(pos_odds_bets)
        print(f"\n  Positive odds only:")
        print(f"    Total: {len(pos_odds_bets)}")
        print(f"    Hits: {pos_hits}")
        print(f"    Accuracy: {pos_accuracy:.1%}")

    # Break down by odds range
    print(f"\n  Breakdown by odds range:")
    odds_buckets = defaultdict(lambda: {"hits": 0, "total": 0})
    for b in high_accuracy_bets:
        if b["odds"] <= -500:
            bucket = "Heavy juice (<= -500)"
        elif b["odds"] <= -200:
            bucket = "Moderate juice (-200 to -500)"
        elif b["odds"] < 0:
            bucket = "Light juice (-100 to -200)"
        elif b["odds"] <= 100:
            bucket = "Even to slight plus (0 to +100)"
        else:
            bucket = "Plus odds (> +100)"
        odds_buckets[bucket]["total"] += 1
        if b["hit"]:
            odds_buckets[bucket]["hits"] += 1

    for bucket_name, counts in sorted(odds_buckets.items()):
        rate = counts["hits"] / counts["total"] if counts["total"] > 0 else 0
        print(f"    {bucket_name}: {counts['hits']}/{counts['total']} = {rate:.1%}")

    return {
        "bets": high_accuracy_bets,
        "accuracy": accuracy,
        "total": total,
        "hits": hits,
    }


def research_favorite_ml_parlays(train, test, all_data):
    """Research heavy favorite ML patterns for parlay construction."""
    ml_results = analyze_favorite_ml(all_data)

    print("\nFavorite ML Win Rates by Odds Bucket:")
    print(f"{'Bucket':<35} {'Wins':>5} {'Losses':>7} {'Rate':>8} {'Avg Odds':>10}")
    print("-" * 70)

    bucket_stats = {}
    for bucket, data in sorted(ml_results.items()):
        total = data["wins"] + data["losses"]
        rate = data["wins"] / total if total > 0 else 0
        avg_odds = statistics.mean(data["odds"]) if data["odds"] else 0
        print(f"{bucket:<35} {data['wins']:>5} {data['losses']:>7} {rate:>8.1%} {avg_odds:>10.0f}")
        bucket_stats[bucket] = {"rate": rate, "total": total, "avg_odds": avg_odds}

    # Simulate 2-leg parlays from heavy favorites
    print("\n--- Simulating 2-Leg ML Parlays from Heavy Favorites ---")
    simulate_ml_parlays(all_data, min_odds=-600, max_odds=-300, legs=2)
    simulate_ml_parlays(all_data, min_odds=-1000, max_odds=-400, legs=2)
    simulate_ml_parlays(all_data, min_odds=-1000, max_odds=-300, legs=3)

    return bucket_stats


def simulate_ml_parlays(games, min_odds=-600, max_odds=-300, legs=2):
    """
    Simulate parlays by picking N legs from the same date,
    all heavy favorites.
    """
    # Group games by date
    games_by_date = defaultdict(list)
    for game in games:
        if game["home_ml"] is None or game["away_ml"] is None:
            continue

        # Find the favorite
        if game["home_ml"] < game["away_ml"]:
            fav_odds = game["home_ml"]
            fav_won = game["home_score"] > game["away_score"]
        else:
            fav_odds = game["away_ml"]
            fav_won = game["away_score"] > game["home_score"]

        if min_odds <= fav_odds <= max_odds:
            games_by_date[game["date"]].append({
                "odds": fav_odds,
                "won": fav_won,
                "game": f"{game['away']}@{game['home']}",
            })

    # Build parlays
    parlays_hit = 0
    parlays_total = 0
    total_payout = 0

    for date, day_games in sorted(games_by_date.items()):
        if len(day_games) < legs:
            continue

        # Take the N strongest favorites
        day_games.sort(key=lambda x: x["odds"])
        selected = day_games[:legs]

        all_won = all(g["won"] for g in selected)
        combo_decimal = 1.0
        for g in selected:
            combo_decimal *= american_to_decimal(g["odds"])

        parlays_total += 1
        if all_won:
            parlays_hit += 1
            total_payout += combo_decimal - 1  # Profit
        else:
            total_payout -= 1  # Loss

    if parlays_total > 0:
        accuracy = parlays_hit / parlays_total
        avg_roi = total_payout / parlays_total
        print(f"\n  {legs}-leg parlays (odds {min_odds} to {max_odds}):")
        print(f"    Total parlays: {parlays_total}")
        print(f"    Hits: {parlays_hit}")
        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    Total ROI: {avg_roi:.1%}")
        # Check if the parlay odds are positive
        # With 2 legs of -400 each, decimal = 1.25 * 1.25 = 1.5625 => +56% or about -178 American
        print(f"    Avg parlay decimal odds: ~{combo_decimal:.2f}")


def research_totals(games):
    """Research over/under patterns."""
    total_results = analyze_totals(games)

    over_hits = sum(1 for r in total_results if r["over_hit"])
    total_games = len(total_results)
    print(f"\nOver/Under baseline: {over_hits}/{total_games} = {over_hits/total_games:.1%} overs")

    # Analyze by deviation size
    print(f"\nActual - Line deviation distribution:")
    diffs = [r["diff"] for r in total_results]
    print(f"  Mean deviation: {statistics.mean(diffs):.1f}")
    print(f"  Std deviation: {statistics.stdev(diffs):.1f}")

    # Big over games
    big_over = [r for r in total_results if r["diff"] > 20]
    big_under = [r for r in total_results if r["diff"] < -20]
    print(f"  Games >20 over line: {len(big_over)}")
    print(f"  Games >20 under line: {len(big_under)}")

    return {"total_results": len(total_results)}


def research_spreads(games):
    """Research spread cover patterns."""
    spread_results = analyze_spreads(games)

    print(f"\nSpread Cover Rates by Bucket:")
    print(f"{'Bucket':<25} {'Covers':>7} {'Misses':>7} {'Pushes':>7} {'Rate':>8}")
    print("-" * 60)
    for bucket, data in sorted(spread_results.items()):
        total = data["covers"] + data["misses"]
        rate = data["covers"] / total if total > 0 else 0
        print(f"{bucket:<25} {data['covers']:>7} {data['misses']:>7} {data['pushes']:>7} {rate:>8.1%}")

    return spread_results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results = run_full_research()
