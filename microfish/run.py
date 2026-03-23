#!/usr/bin/env python3
"""
Microfish — MiroFish Multi-Agent Strategy Development System
=============================================================
MLB +200 Underdog Strategy developed by Claude AI agent swarm,
executed as pure deterministic rules.

Usage:
    python run.py develop           Run MiroFish agent swarm to develop/refine strategy rules
    python run.py develop --iter N  Run N development iterations (default: 3)
    python run.py backtest          Backtest strategy_rules.json against historical data
    python run.py picks             Generate today's picks using rules engine
    python run.py show              Show current strategy rules
    python run.py full              Develop + backtest + show results

    python run.py sgp-develop       Run SGP discovery engine (MiniMax M2.7 agents)
    python run.py sgp-develop --iter N  Run N SGP iterations (default: 3)
    python run.py sgp-backtest      Backtest SGP rules against historical prop data
    python run.py sgp-picks         Generate today's SGP parlay picks
    python run.py sgp-show          Show current SGP strategy rules

    python run.py ml-discover       Run Certainty Floor Discovery on real MLB data
    python run.py ml-refine         Run MiroFish MinMax agents to refine the floor
    python run.py ml-backtest       Walk-forward backtest of limit-price strategy
    python run.py ml-picks          Generate today's Polymarket limit orders
    python run.py ml-show           Show current moneyline strategy rules
    python run.py ml-full           Discover + refine + backtest (complete pipeline)

    python run.py ml-live-download  Download ESPN live win probability data (2,364 games)
    python run.py ml-live-analyze   Analyze live data for in-game certainty floors
    python run.py ml-live-report    Show live strategy tiers and projections

Workflow:
    +200 Underdogs:
      1. 'develop'  — Claude Opus agents produce strategy_rules.json
      2. 'backtest' — Walk-forward validation (pure deterministic)
      3. 'picks'    — Apply rules to today's games

    Moneyline (Polymarket Certainty Floor):
      1. 'ml-discover' — CFD engine analyzes real data for certainty floors
      2. 'ml-refine'   — MinMax agents adversarially refine the floor
      3. 'ml-backtest' — Walk-forward validation of limit-price strategy
      4. 'ml-picks'    — Generate today's Polymarket limit orders

    SGP Parlays (@sgp_vick style):
      1. 'sgp-develop'  — MiniMax M2.7 agents discover correlation rules
      2. 'sgp-backtest' — Walk-forward SGP validation
      3. 'sgp-picks'    — Construct today's SGP slips ($10 → $6,000+)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == 'develop':
        from data_pipeline import load_all_seasons, build_features, get_plus200_opportunities
        from microfish_dev import iterative_development

        # Parse iteration count
        n_iter = 3
        for i, arg in enumerate(sys.argv):
            if arg == '--iter' and i + 1 < len(sys.argv):
                n_iter = int(sys.argv[i + 1])

        print("PHASE 1: Loading historical data...")
        raw = load_all_seasons()
        if raw.empty:
            print("ERROR: No data. Set THE_ODDS_API_KEY or add cached CSV data.")
            sys.exit(1)

        print(f"\nPHASE 2: Building features...")
        features = build_features(raw)

        print(f"\nPHASE 3: Extracting +200 opportunities...")
        opportunities = get_plus200_opportunities(features)
        print(f"  {len(opportunities)} opportunities found")

        if opportunities.empty:
            print("ERROR: No +200 opportunities in data")
            sys.exit(1)

        print(f"\nPHASE 4: MiroFish Development Pipeline ({n_iter} iterations)...")
        rules = iterative_development(opportunities, max_iterations=n_iter)

        if rules:
            print("\nDevelopment complete. Strategy rules saved.")
            print("Next: python run.py backtest")
        else:
            print("\nDevelopment failed to produce valid rules.")

    elif command == 'backtest':
        from backtest import run_full_backtest
        results = run_full_backtest()

        if results.get('accuracy', 0) >= 0.90:
            print(f"\nACCURACY TARGET MET: {results['accuracy']:.1%} >= 90%")
        else:
            print(f"\nAccuracy {results.get('accuracy', 0):.1%} — below 90% target")

        if results.get('daily_frequency', 0) >= 0.30:
            print(f"FREQUENCY TARGET MET: {results['daily_frequency']:.1%} >= 30%")
        else:
            print(f"Frequency {results.get('daily_frequency', 0):.1%} — below 30% target")

    elif command == 'picks':
        from daily_picks import generate_daily_picks
        generate_daily_picks()

    elif command == 'show':
        from strategy_engine import StrategyEngine
        try:
            engine = StrategyEngine()
            print(engine.summary())
        except FileNotFoundError as e:
            print(f"ERROR: {e}")

    elif command == 'full':
        # Run develop then backtest
        os.system(f'{sys.executable} {__file__} develop')
        print("\n\n")
        os.system(f'{sys.executable} {__file__} backtest')

    # ================================================================
    # SGP COMMANDS — @sgp_vick-style Same Game Parlay system
    # ================================================================

    elif command == 'sgp-develop':
        from sgp_data import (
            load_all_sgp_data, build_sgp_features,
            compute_correlation_matrix, format_sgp_data_summary
        )
        from sgp_discovery import iterative_sgp_discovery

        n_iter = 3
        for i, arg in enumerate(sys.argv):
            if arg == '--iter' and i + 1 < len(sys.argv):
                n_iter = int(sys.argv[i + 1])

        print("PHASE 1: Loading historical SGP data...")
        data = load_all_sgp_data()
        props_df = data['props']
        results_df = data['results']

        if props_df.empty:
            print("ERROR: No prop data. Cache data to data/sgp/cache/")
            sys.exit(1)

        print(f"  Props: {len(props_df)} records")

        print("\nPHASE 2: Building SGP features...")
        features = build_sgp_features(props_df, results_df)
        print(f"  Features: {len(features)} game-level records")

        print("\nPHASE 3: Computing prop correlations...")
        correlations = compute_correlation_matrix(props_df)
        print(f"  Significant correlations: {len(correlations)}")

        print("\nPHASE 4: Building data summary for agents...")
        data_summary = format_sgp_data_summary(props_df, correlations=correlations)

        # Format correlation summary
        corr_lines = []
        sorted_corrs = sorted(correlations.items(),
                              key=lambda x: x[1]['lift'], reverse=True)
        for (a, b), stats in sorted_corrs[:50]:
            corr_lines.append(
                f"{a} + {b}: lift={stats['lift']:.2f}, "
                f"P(both)={stats['p_both']:.3f}, n={stats['co_occurrences']}"
            )
        corr_summary = '\n'.join(corr_lines)

        print(f"\nPHASE 5: MiroFish SGP Discovery ({n_iter} iterations)...")
        rules = iterative_sgp_discovery(
            data_summary=data_summary,
            correlation_summary=corr_summary,
            max_iterations=n_iter,
        )

        if rules:
            print("\nSGP development complete. Rules saved.")
            print("Next: python run.py sgp-backtest")
        else:
            print("\nSGP development failed to produce valid rules.")

    elif command == 'sgp-backtest':
        from sgp_backtest import run_sgp_backtest
        results = run_sgp_backtest()

        hit_rate = results.get('hit_rate', 0)
        roi = results.get('roi', 0)
        total = results.get('total_sgps', 0)
        if total > 0:
            print(f"\nSGP Hit rate: {hit_rate:.1%}")
            print(f"SGP ROI: {roi:+.1f}%")

    elif command == 'sgp-picks':
        from sgp_daily_picks import generate_sgp_picks
        sgps = generate_sgp_picks()
        if sgps:
            print(f"\n{len(sgps)} SGP slips generated for today.")
        else:
            print("\nNo SGPs available today.")

    elif command == 'sgp-show':
        from sgp_engine import SGPEngine
        try:
            engine = SGPEngine()
            print(engine.summary())
        except FileNotFoundError as e:
            print(f"ERROR: {e}")

    elif command == 'sgp-full':
        os.system(f'{sys.executable} {__file__} sgp-develop')
        print("\n\n")
        os.system(f'{sys.executable} {__file__} sgp-backtest')

    # ================================================================
    # MONEYLINE COMMANDS — Polymarket Certainty Floor Strategy
    # ================================================================

    elif command == 'ml-discover':
        from ml_certainty_engine import run_full_pipeline
        strategy = run_full_pipeline()

        if strategy:
            base = strategy.get('base_rule', {})
            print(f"\nCertainty Floor: {base.get('floor_pct', '?')}%")
            print(f"Limit Price: ${base.get('limit_price', 0):.2f}")
            print("Next: python run.py ml-refine (optional) or python run.py ml-backtest")
        else:
            print("\nCFD pipeline failed.")

    elif command == 'ml-refine':
        from ml_certainty_engine import run_certainty_floor_discovery
        from ml_microfish_dev import run_minmax_refinement
        from ml_data_pipeline import load_real_data, build_features, compute_certainty_floor_data
        import json

        print("PHASE 1: Loading data and computing floors...")
        df = load_real_data()
        featured = build_features(df)
        floor_data = compute_certainty_floor_data(featured)

        # Load existing strategy
        from ml_config import ML_RULES_FILE
        if os.path.exists(ML_RULES_FILE):
            with open(ML_RULES_FILE, 'r') as f:
                strategy = json.load(f)
        else:
            print("No strategy found. Run ml-discover first.")
            sys.exit(1)

        print("\nPHASE 2: MiroFish MinMax Adversarial Refinement...")
        refined = run_minmax_refinement(strategy, floor_data, featured)

        if refined:
            with open(ML_RULES_FILE, 'w') as f:
                json.dump(refined, f, indent=2, default=str)
            print(f"\nRefined strategy saved to {ML_RULES_FILE}")
            print("Next: python run.py ml-backtest")

    elif command == 'ml-backtest':
        from ml_backtest import run_full_backtest
        results = run_full_backtest()

        accuracy = results.get('accuracy', 0)
        if accuracy == 1.0:
            print(f"\n100% ACCURACY ACHIEVED: {results['wins']}/{results['total_bets']} bets won")
        else:
            print(f"\nAccuracy: {accuracy:.1%} ({results.get('losses', 0)} losses)")

        fill = results.get('fill_rate', 0)
        print(f"Fill rate: {fill:.1%}")
        print(f"ROI: {results.get('roi', 0):.1f}%")

    elif command == 'ml-picks':
        from ml_daily_picks import generate_daily_picks
        orders = generate_daily_picks()
        if orders:
            print(f"\n{len(orders)} limit orders generated for today.")
        else:
            print("\nNo orders today.")

    elif command == 'ml-show':
        from ml_strategy_engine import MLStrategyEngine
        try:
            engine = MLStrategyEngine()
            print(engine.summary())
        except FileNotFoundError as e:
            print(f"ERROR: {e}")

    elif command == 'ml-full':
        # Full pipeline: discover + refine + backtest
        os.system(f'{sys.executable} {__file__} ml-discover')
        print("\n\n")
        os.system(f'{sys.executable} {__file__} ml-refine')
        print("\n\n")
        os.system(f'{sys.executable} {__file__} ml-backtest')

    # ================================================================
    # LIVE MONEYLINE — In-game Polymarket limit orders
    # ================================================================

    elif command == 'ml-live-download':
        from ml_live_winprob import download_all
        print("Downloading ESPN live win probability data...")
        cache = download_all()
        valid = sum(1 for v in cache.values()
                    if v.get('n_plays', 0) > 0 and not v.get('error'))
        print(f"\nDone: {valid} games with win probability data cached.")

    elif command == 'ml-live-analyze':
        from ml_live_winprob import run_full_download_and_analysis
        results, floor = run_full_download_and_analysis()

    elif command == 'ml-live-report':
        from ml_live_strategy import generate_live_strategy_report
        generate_live_strategy_report()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
