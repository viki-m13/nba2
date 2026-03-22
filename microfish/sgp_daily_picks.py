"""
MiroFish SGP Daily Picks Generator
======================================
Generates today's SGP (Same Game Parlay) recommendations using the
deterministic template engine.

Workflow:
1. Fetch today's player props from The Odds API
2. Load SGP strategy rules (sgp_strategy_rules.json)
3. Construct SGPs by matching templates to available props
4. Output ranked SGP slips with combined odds and stake info

NO AI at runtime — pure template matching against live prop lines.
"""

import os
import json
from datetime import datetime

from sgp_config import (
    SGP_RESULTS_DIR, STAKE_SIZE, MAX_PARLAYS_PER_DAY,
)
from sgp_data import (
    fetch_player_props, fetch_game_odds,
    parse_props_to_dataframe, parse_games_to_dataframe,
)
from sgp_engine import SGPEngine


def generate_sgp_picks(game_date: str = None) -> list:
    """
    Generate today's SGP pick recommendations.

    Returns list of constructed SGP slips sorted by score.
    """
    if game_date is None:
        game_date = datetime.now().strftime('%Y-%m-%d')

    print("=" * 70)
    print(f"MIROFISH SGP DAILY PICKS — {game_date}")
    print("=" * 70)

    # Load engine
    try:
        engine = SGPEngine()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return []

    print(f"Strategy loaded: {engine.rules.get('version', 'unknown')}")
    print(f"Templates: {len(engine.templates)}")
    print()

    # Fetch today's props
    print("STEP 1: Fetching player props...")
    raw_props = fetch_player_props(game_date)
    if not raw_props:
        print("  No prop data available. Check API key or try later.")
        return []

    props_df = parse_props_to_dataframe(raw_props)
    print(f"  Props loaded: {len(props_df)} lines")
    if 'player' in props_df.columns:
        print(f"  Players: {props_df['player'].nunique()}")
    if 'game_id' in props_df.columns:
        print(f"  Games: {props_df['game_id'].nunique()}")

    # Fetch game odds for context
    print("\nSTEP 2: Fetching game odds...")
    raw_games = fetch_game_odds(game_date)
    games_df = parse_games_to_dataframe(raw_games) if raw_games else None
    if games_df is not None and not games_df.empty:
        print(f"  Game odds loaded: {len(games_df)} entries")

    # Construct SGPs
    print("\nSTEP 3: Constructing SGPs from templates...")
    sgps = engine.construct_sgps(props_df, games_df)
    print(f"  SGPs constructed: {len(sgps)}")

    # Also try SGP+ (cross-game)
    print("\nSTEP 4: Constructing SGP+ (cross-game)...")
    sgp_plus = engine.construct_sgp_plus(props_df)
    print(f"  SGP+ constructed: {len(sgp_plus)}")

    # Combine and rank
    all_sgps = sgps + sgp_plus
    all_sgps.sort(key=lambda s: s.get('score', 0), reverse=True)

    # Apply daily cap
    if len(all_sgps) > MAX_PARLAYS_PER_DAY:
        all_sgps = all_sgps[:MAX_PARLAYS_PER_DAY]

    # Print picks
    if all_sgps:
        print(f"\n{'='*70}")
        print(f"TODAY'S SGP PICKS ({len(all_sgps)} slips)")
        print(f"{'='*70}")

        total_stake = 0
        total_potential = 0

        for i, sgp in enumerate(all_sgps, 1):
            print(f"\n--- SLIP {i}: {sgp.get('template', 'unknown')} ---")
            print(f"  Game: {sgp.get('away_team', '?')} @ {sgp.get('home_team', '?')}")
            print(f"  Combined odds: +{sgp.get('combined_odds_american', 0)}")
            print(f"  Stake: ${sgp.get('stake', STAKE_SIZE)}")
            print(f"  Potential payout: ${sgp.get('potential_payout', 0):.2f}")
            if sgp.get('ev_per_dollar') is not None:
                print(f"  EV/dollar: ${sgp['ev_per_dollar']:.4f}")
            print(f"  Legs ({sgp.get('n_legs', 0)}):")

            for leg in sgp.get('legs', []):
                player = leg.get('player', '?')
                market = leg.get('market', '?')
                direction = leg.get('direction', '?')
                line = leg.get('line', '?')
                odds = leg.get('odds', 0)
                book = leg.get('bookmaker', '')
                print(f"    - {player}: {market} {direction} {line} ({odds:+d}) [{book}]")

            if sgp.get('correlation_edge'):
                print(f"  Edge: {sgp['correlation_edge']}")

            total_stake += sgp.get('stake', STAKE_SIZE)
            total_potential += sgp.get('potential_payout', 0)

        print(f"\n{'='*70}")
        print(f"TOTAL STAKE: ${total_stake}")
        print(f"MAX POTENTIAL PAYOUT: ${total_potential:.2f}")
        print(f"{'='*70}")
    else:
        print("\n  No SGPs match today's available props.")
        print("  This is normal — not every day produces actionable slips.")

    # Save picks
    _save_sgp_picks(all_sgps, game_date)

    return all_sgps


def _save_sgp_picks(sgps: list, game_date: str):
    """Save SGP picks to results directory."""
    os.makedirs(SGP_RESULTS_DIR, exist_ok=True)

    picks_file = os.path.join(SGP_RESULTS_DIR, f'sgp_picks_{game_date}.json')

    # Clean for JSON serialization
    clean_sgps = []
    for sgp in sgps:
        clean = {}
        for k, v in sgp.items():
            if k == 'legs':
                clean[k] = [
                    {
                        'player': l.get('player', ''),
                        'market': l.get('market', ''),
                        'direction': l.get('direction', ''),
                        'line': l.get('line'),
                        'odds': l.get('odds', 0),
                        'implied_prob': l.get('implied_prob', 0),
                        'bookmaker': l.get('bookmaker', ''),
                    }
                    for l in v
                ]
            else:
                clean[k] = v
        clean_sgps.append(clean)

    output = {
        'date': game_date,
        'generated_at': datetime.now().isoformat(),
        'n_sgps': len(clean_sgps),
        'total_stake': sum(s.get('stake', STAKE_SIZE) for s in sgps),
        'sgps': clean_sgps,
    }

    with open(picks_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Picks saved to {picks_file}")
