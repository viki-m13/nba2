#!/usr/bin/env python3
"""
Microfish — MiroFish Multi-Agent Strategy Development System
=============================================================
MLB +200 Underdog Strategy developed by Claude AI agent swarm,
executed as pure deterministic rules.

Usage:
    python run.py develop         Run MiroFish agent swarm to develop/refine strategy rules
    python run.py develop --iter N  Run N development iterations (default: 3)
    python run.py backtest        Backtest strategy_rules.json against historical data
    python run.py picks           Generate today's picks using rules engine
    python run.py show            Show current strategy rules
    python run.py full            Develop + backtest + show results

Workflow:
    1. 'develop' — Claude Opus agents analyze data and produce strategy_rules.json
    2. 'backtest' — Walk-forward validation of rules (no AI, pure deterministic)
    3. 'picks'   — Apply rules to today's games (no AI, pure deterministic)
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

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
