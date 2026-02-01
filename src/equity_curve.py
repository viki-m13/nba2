"""
Equity Curve Simulation for Q3 O/U Strategy
=============================================
Plots cumulative P&L for out-of-sample signals, comparing:
  - vs Opening line (illusory edge — line is stale by Q3 end)
  - vs Live line estimate (honest edge — what you'd actually face)

Loads signals from generate_historical_signals.py output.
"""

import json
from pathlib import Path


SIGNALS_PATH = Path("/home/user/nba2/output/q3_ou_historical_signals.json")
WIN_PAYOUT = 100 / 110  # 0.9091


def load_signals():
    """Load historical signals from JSON."""
    with open(SIGNALS_PATH) as f:
        data = json.load(f)
    return data


def build_equity_curves(signals, oos_only=True):
    """Build equity curves from signal list.

    Returns list of dicts with cumulative P&L for both opening and live line.
    Only uses OOS (2022-23) signals by default to avoid in-sample bias.
    """
    if oos_only:
        signals = [s for s in signals if s.get('season') == '2022-23']

    # Sort by date
    signals = sorted(signals, key=lambda x: x.get('date', ''))

    cum_opening = 0.0
    cum_live = 0.0
    curve = []

    for s in signals:
        # Opening line P&L
        if s.get('openingCorrect', s.get('correct', False)):
            opening_pnl = WIN_PAYOUT
        elif s.get('isPushOpening', s.get('isPush', False)):
            opening_pnl = 0.0
        else:
            opening_pnl = -1.0
        cum_opening += opening_pnl

        # Live line P&L
        if s.get('liveCorrect', False):
            live_pnl = WIN_PAYOUT
        else:
            live_pnl = -1.0
        cum_live += live_pnl

        curve.append({
            'date': s.get('date', ''),
            'tier': s.get('tier', ''),
            'teams': f"{s.get('awayTeam', '')}@{s.get('homeTeam', '')}",
            'direction': s.get('direction', ''),
            'opening_correct': s.get('openingCorrect', s.get('correct', False)),
            'live_correct': s.get('liveCorrect', False),
            'opening_pnl': opening_pnl,
            'live_pnl': live_pnl,
            'cum_opening': cum_opening,
            'cum_live': cum_live,
        })

    return curve


def print_trade_log(curve, max_trades=50):
    """Print trade-by-trade log."""
    print(f"{'#':<4} {'Date':<11} {'Game':<10} {'Tier':<9} {'Dir':<6} "
          f"{'Open':<5} {'Live':<5} {'OpenPnL':>8} {'LivePnL':>8} "
          f"{'CumOpen':>9} {'CumLive':>9}")
    print("-" * 95)

    for i, t in enumerate(curve[:max_trades]):
        o_wl = "W" if t['opening_correct'] else "L"
        l_wl = "W" if t['live_correct'] else "L"
        print(f"{i+1:<4} {t['date']:<11} {t['teams']:<10} {t['tier']:<9} "
              f"{t['direction']:<6} {o_wl:<5} {l_wl:<5} "
              f"{t['opening_pnl']:>+8.2f} {t['live_pnl']:>+8.2f} "
              f"{t['cum_opening']:>+9.2f} {t['cum_live']:>+9.2f}")

    if len(curve) > max_trades:
        print(f"  ... ({len(curve) - max_trades} more trades)")


def render_ascii_chart(values, label, width=70, height=20):
    """Render a proper ASCII line chart (not filled area).

    Each column shows a single character at the equity value's position.
    """
    if not values:
        print("  (no data)")
        return

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val

    if val_range == 0:
        val_range = 1.0

    # Downsample if more points than width
    if len(values) > width:
        step = len(values) / width
        sampled = []
        for i in range(width):
            idx = int(i * step)
            sampled.append(values[min(idx, len(values) - 1)])
        values = sampled

    n_cols = len(values)

    # Build the chart grid
    grid = [[' ' for _ in range(n_cols)] for _ in range(height + 1)]

    # Place each point
    for col, val in enumerate(values):
        row = round((val - min_val) / val_range * height)
        row = max(0, min(height, row))
        grid[row][col] = '*'

    # Connect adjacent points with vertical lines where needed
    for col in range(1, n_cols):
        prev_val = values[col - 1]
        curr_val = values[col]
        prev_row = round((prev_val - min_val) / val_range * height)
        curr_row = round((curr_val - min_val) / val_range * height)
        prev_row = max(0, min(height, prev_row))
        curr_row = max(0, min(height, curr_row))

        if abs(curr_row - prev_row) > 1:
            lo = min(prev_row, curr_row) + 1
            hi = max(prev_row, curr_row)
            for r in range(lo, hi):
                if grid[r][col] == ' ':
                    grid[r][col] = '|'

    # Print top to bottom
    print(f"\n  {label}")
    print(f"  {max_val:>+8.1f} |", end="")
    for c in range(n_cols):
        print(grid[height][c], end="")
    print()

    # Print middle rows (skip some if height is large)
    for row in range(height - 1, 0, -1):
        # Show label at a few key rows
        frac = row / height
        val_at_row = min_val + frac * val_range
        if row == height // 2:
            print(f"  {val_at_row:>+8.1f} |", end="")
        else:
            print(f"           |", end="")
        for c in range(n_cols):
            print(grid[row][c], end="")
        print()

    print(f"  {min_val:>+8.1f} |", end="")
    for c in range(n_cols):
        print(grid[0][c], end="")
    print()
    print(f"           +{'─' * n_cols}")

    # X-axis labels
    if n_cols > 10:
        label_line = "            "
        for i in range(n_cols):
            if i == 0:
                label_line += "1"
            elif i == n_cols - 1:
                label_line += str(len(values))
            elif i == n_cols // 2:
                mid = str(len(values) // 2)
                label_line += mid[0]
            else:
                label_line += " "
        print(label_line)
    print(f"           Trade # (1 to {len(values)})")


def print_summary(curve):
    """Print summary statistics."""
    if not curve:
        print("  No trades to summarize.")
        return

    total = len(curve)
    opening_wins = sum(1 for t in curve if t['opening_correct'])
    live_wins = sum(1 for t in curve if t['live_correct'])

    final_opening = curve[-1]['cum_opening']
    final_live = curve[-1]['cum_live']

    opening_wr = opening_wins / total
    live_wr = live_wins / total

    opening_roi = final_opening / total * 100
    live_roi = final_live / total * 100

    # Max drawdown for opening
    peak = 0
    max_dd_opening = 0
    for t in curve:
        peak = max(peak, t['cum_opening'])
        dd = peak - t['cum_opening']
        max_dd_opening = max(max_dd_opening, dd)

    # Max drawdown for live
    peak = 0
    max_dd_live = 0
    for t in curve:
        peak = max(peak, t['cum_live'])
        dd = peak - t['cum_live']
        max_dd_live = max(max_dd_live, dd)

    print(f"\n  {'Metric':<25} {'vs Opening Line':>16} {'vs Live Line':>16}")
    print(f"  {'─'*57}")
    print(f"  {'Total Trades':<25} {total:>16}")
    print(f"  {'Wins':<25} {opening_wins:>16} {live_wins:>16}")
    print(f"  {'Win Rate':<25} {opening_wr:>15.1%} {live_wr:>15.1%}")
    print(f"  {'Breakeven':<25} {'52.4%':>16} {'52.4%':>16}")
    print(f"  {'Cumulative P&L':<25} {final_opening:>+15.1f}u {final_live:>+15.1f}u")
    print(f"  {'ROI':<25} {opening_roi:>+15.1f}% {live_roi:>+15.1f}%")
    print(f"  {'Max Drawdown':<25} {max_dd_opening:>15.1f}u {max_dd_live:>15.1f}u")
    print(f"  {'Avg P&L per trade':<25} {final_opening/total:>+15.3f}u {final_live/total:>+15.3f}u")


def main():
    print("=" * 70)
    print("Q3 O/U EQUITY CURVE — OPENING LINE vs LIVE LINE")
    print("Out-of-Sample (2022-23 Season) Only")
    print("=" * 70)

    data = load_signals()
    signals = data.get('signals', [])
    print(f"\nLoaded {len(signals)} total signals")

    oos = [s for s in signals if s.get('season') == '2022-23']
    insample = [s for s in signals if s.get('season') == '2021-22']
    print(f"  OOS (2022-23): {len(oos)}")
    print(f"  In-sample (2021-22): {len(insample)} (excluded)")

    # Build equity curves
    curve = build_equity_curves(signals, oos_only=True)

    if not curve:
        print("\nNo OOS signals found. Run generate_historical_signals.py first.")
        return

    # Trade log (first 30)
    print(f"\n{'='*70}")
    print("TRADE LOG (first 30 OOS signals)")
    print(f"{'='*70}\n")
    print_trade_log(curve, max_trades=30)

    # Summary stats
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print_summary(curve)

    # ASCII equity curves
    print(f"\n{'='*70}")
    print("EQUITY CURVES (ASCII)")
    print(f"{'='*70}")

    opening_equity = [t['cum_opening'] for t in curve]
    live_equity = [t['cum_live'] for t in curve]

    render_ascii_chart(
        opening_equity,
        "vs OPENING LINE (illusory — stale line, not tradeable)",
        width=70, height=18
    )

    print()

    render_ascii_chart(
        live_equity,
        "vs LIVE LINE (honest — what you'd actually face)",
        width=70, height=18
    )

    # Final verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    final_live_roi = curve[-1]['cum_live'] / len(curve) * 100
    if final_live_roi > 5:
        print("\n  The model shows potential live-market edge. Validate with")
        print("  more data before risking capital.")
    elif final_live_roi > -5:
        print("\n  The model shows approximately ZERO edge against the live market.")
        print("  The high opening-line accuracy is real but not tradeable —")
        print("  the live line has already moved to reflect the game state.")
        print("  Our 9-feature model predicts Q4 about as well as a simple")
        print("  3-feature market proxy (RMSE 8.64 vs 8.65).")
    else:
        print("\n  The model shows NEGATIVE edge against the live market.")
        print("  Do not trade this strategy.")


if __name__ == "__main__":
    main()
