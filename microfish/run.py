#!/usr/bin/env python3
"""
Microfish - Main Entry Point
=============================
MLB +200 Underdog Strategy using Multi-Agent Swarm Intelligence

Usage:
    python run.py backtest          Run full backtest with walk-forward validation
    python run.py picks             Generate today's picks
    python run.py picks --no-api    Generate picks without Claude API (rule-based only)
    python run.py optimize          Run optimizer on backtest results
    python run.py full              Run backtest + optimize + picks
"""

import sys
import os

# Add microfish dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == 'backtest':
        from backtest import run_full_backtest
        results = run_full_backtest()
        if results.get('accuracy', 0) >= 0.90:
            print("\n✓ ACCURACY TARGET MET (>= 90%)")
        else:
            print(f"\n✗ Accuracy {results.get('accuracy', 0):.1%} below 90% target")

    elif command == 'picks':
        from daily_picks import generate_daily_picks
        use_api = '--no-api' not in sys.argv
        generate_daily_picks(use_claude_api=use_api)

    elif command == 'optimize':
        import json
        from optimizer import optimize_strategy
        from config import RESULTS_DIR

        results_file = os.path.join(RESULTS_DIR, 'backtest_summary.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            optimized = optimize_strategy(results)
            print("\nOptimization suggestions:")
            print(json.dumps(optimized, indent=2))
        else:
            print("Run backtest first: python run.py backtest")

    elif command == 'full':
        from backtest import run_full_backtest
        from optimizer import optimize_strategy
        from daily_picks import generate_daily_picks

        print("PHASE 1: BACKTEST")
        results = run_full_backtest()

        print("\n\nPHASE 2: OPTIMIZE")
        optimized = optimize_strategy(results)
        print(f"Optimizer analysis: {optimized.get('analysis', 'N/A')}")

        print("\n\nPHASE 3: DAILY PICKS")
        generate_daily_picks(use_claude_api=True)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
