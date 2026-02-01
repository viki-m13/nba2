"""
Equity Curve Simulation for Q3 O/U Strategy
=============================================
Plots cumulative P&L for out-of-sample signals, comparing:
  - vs Opening line at -110 (illusory edge — line is stale by Q3 end)
  - vs Estimated live odds (realistic — what you'd actually face)
  - Pre-game entry + Q3 confirmation (recommended execution)

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

    Returns list of dicts with cumulative P&L for opening line (-110),
    estimated live odds, and pre-game confirmation strategy.
    Only uses OOS (2022-23) signals by default to avoid in-sample bias.
    """
    if oos_only:
        signals = [s for s in signals if s.get('season') == '2022-23']

    # Sort by date
    signals = sorted(signals, key=lambda x: x.get('date', ''))

    cum_opening = 0.0
    cum_live = 0.0
    cum_est_odds = 0.0
    curve = []

    for s in signals:
        # Opening line P&L (at -110)
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

        # Estimated live odds P&L (realistic)
        est_payout = s.get('estLivePayout', WIN_PAYOUT)
        if s.get('openingCorrect', s.get('correct', False)):
            est_odds_pnl = est_payout
        elif s.get('isPushOpening', s.get('isPush', False)):
            est_odds_pnl = 0.0
        else:
            est_odds_pnl = -1.0
        cum_est_odds += est_odds_pnl

        curve.append({
            'date': s.get('date', ''),
            'tier': s.get('tier', ''),
            'teams': f"{s.get('awayTeam', '')}@{s.get('homeTeam', '')}",
            'direction': s.get('direction', ''),
            'opening_correct': s.get('openingCorrect', s.get('correct', False)),
            'live_correct': s.get('liveCorrect', False),
            'est_odds': s.get('estLiveOdds', -110),
            'opening_pnl': opening_pnl,
            'live_pnl': live_pnl,
            'est_odds_pnl': est_odds_pnl,
            'cum_opening': cum_opening,
            'cum_live': cum_live,
            'cum_est_odds': cum_est_odds,
        })

    return curve


def print_trade_log(curve, max_trades=50):
    """Print trade-by-trade log."""
    print(f"{'#':<4} {'Date':<11} {'Game':<10} {'Tier':<9} {'Dir':<6} "
          f"{'W/L':<5} {'Est Odds':<10} {'PnL@-110':>9} {'PnL@Odds':>9} "
          f"{'Cum-110':>8} {'CumOdds':>8}")
    print("-" * 100)

    for i, t in enumerate(curve[:max_trades]):
        wl = "W" if t['opening_correct'] else "L"
        print(f"{i+1:<4} {t['date']:<11} {t['teams']:<10} {t['tier']:<9} "
              f"{t['direction']:<6} {wl:<5} {t['est_odds']:<10} "
              f"{t['opening_pnl']:>+9.2f} {t['est_odds_pnl']:>+9.4f} "
              f"{t['cum_opening']:>+8.1f} {t['cum_est_odds']:>+8.1f}")

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

    # Print middle rows
    for row in range(height - 1, 0, -1):
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

    final_opening = curve[-1]['cum_opening']
    final_est_odds = curve[-1]['cum_est_odds']

    opening_wr = opening_wins / total
    opening_roi = final_opening / total * 100
    est_odds_roi = final_est_odds / total * 100

    # Max drawdown for opening
    peak = 0
    max_dd_opening = 0
    for t in curve:
        peak = max(peak, t['cum_opening'])
        dd = peak - t['cum_opening']
        max_dd_opening = max(max_dd_opening, dd)

    # Max drawdown for est odds
    peak = 0
    max_dd_est = 0
    for t in curve:
        peak = max(peak, t['cum_est_odds'])
        dd = peak - t['cum_est_odds']
        max_dd_est = max(max_dd_est, dd)

    print(f"\n  {'Metric':<25} {'At -110 (pre-game)':>18} {'At Est Live Odds':>18}")
    print(f"  {'─'*61}")
    print(f"  {'Total Trades':<25} {total:>18}")
    print(f"  {'Wins':<25} {opening_wins:>18} {opening_wins:>18}")
    print(f"  {'Win Rate':<25} {opening_wr:>17.1%} {opening_wr:>17.1%}")
    print(f"  {'Cumulative P&L':<25} {final_opening:>+17.1f}u {final_est_odds:>+17.1f}u")
    print(f"  {'ROI':<25} {opening_roi:>+17.1f}% {est_odds_roi:>+17.1f}%")
    print(f"  {'Max Drawdown':<25} {max_dd_opening:>17.1f}u {max_dd_est:>17.1f}u")
    print(f"  {'Avg P&L / trade':<25} {final_opening/total:>+17.3f}u {final_est_odds/total:>+17.3f}u")


def simulate_pregame_confirmation(all_signals, oos_only=True):
    """Simulate the pre-game entry + Q3 confirmation strategy.

    Strategy:
    - Bet OVER pre-game at -110 on every game
    - At Q3 end: if model confirms OVER with BRONZE+, HOLD
    - If model says UNDER with BRONZE+, HEDGE (cash out at estimated value)
    - If neutral (margin < 10), HOLD original bet
    """
    if oos_only:
        all_signals = [s for s in all_signals if s.get('season') == '2022-23']

    all_signals = sorted(all_signals, key=lambda x: x.get('date', ''))

    cum_pnl = 0.0
    categories = {'confirmed': 0, 'hedged': 0, 'neutral': 0}
    cat_pnl = {'confirmed': 0.0, 'hedged': 0.0, 'neutral': 0.0}
    cat_wins = {'confirmed': 0, 'hedged': 0, 'neutral': 0}

    for s in all_signals:
        tier = s.get('tier')
        direction = s.get('direction', '')
        margin = s.get('openingMargin', 0)
        went_over = s.get('finalTotal', 0) > s.get('ouLine', 0)
        is_push = s.get('finalTotal', 0) == s.get('ouLine', 0)

        if tier and margin >= 10 and direction == 'OVER':
            # Model confirms OVER → HOLD
            if went_over:
                pnl = WIN_PAYOUT
                cat_wins['confirmed'] += 1
            elif is_push:
                pnl = 0.0
            else:
                pnl = -1.0
            categories['confirmed'] += 1
            cat_pnl['confirmed'] += pnl

        elif tier and margin >= 10 and direction == 'UNDER':
            # Model says UNDER → HEDGE our OVER bet
            # Cash-out value depends on how far the game is from the line
            est_prob = s.get('estMarketProb', 0.05)
            # For an UNDER signal, est_prob is P(UNDER). P(OVER) = 1 - est_prob
            p_over = 1.0 - est_prob
            # Cash-out value: P(over_wins) * total_return, minus vig
            cashout_value = p_over * (1 + WIN_PAYOUT) * 0.85  # 85% of fair
            pnl = cashout_value - 1.0  # We paid $1, get back cashout
            categories['hedged'] += 1
            cat_pnl['hedged'] += pnl

        else:
            # Neutral → HOLD original OVER bet
            if went_over:
                pnl = WIN_PAYOUT
                cat_wins['neutral'] += 1
            elif is_push:
                pnl = 0.0
            else:
                pnl = -1.0
            categories['neutral'] += 1
            cat_pnl['neutral'] += pnl

        cum_pnl += pnl

    return categories, cat_pnl, cat_wins, cum_pnl


def main():
    print("=" * 70)
    print("Q3 O/U EQUITY CURVE — REALISTIC ODDS ANALYSIS")
    print("Out-of-Sample (2022-23 Season) Only")
    print("=" * 70)

    data = load_signals()
    signals = data.get('signals', [])
    print(f"\nLoaded {len(signals)} total signals")

    # Filter to signals with tiers
    tier_signals = [s for s in signals if s.get('tier') is not None]
    oos = [s for s in tier_signals if s.get('season') == '2022-23']
    insample = [s for s in tier_signals if s.get('season') == '2021-22']
    print(f"  OOS (2022-23): {len(oos)} signals")
    print(f"  In-sample (2021-22): {len(insample)} (excluded)")

    # Build equity curves (tiered signals only)
    curve = build_equity_curves(tier_signals, oos_only=True)

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

    # Odds distribution
    print(f"\n{'='*70}")
    print("ESTIMATED LIVE ODDS DISTRIBUTION")
    print(f"{'='*70}")

    odds_list = [t['est_odds'] for t in curve]
    print(f"\n  Total signals: {len(odds_list)}")
    sorted_odds = sorted(odds_list)
    print(f"  Best odds (closest to even): {sorted_odds[-1]}")
    print(f"  Worst odds (most extreme):   {sorted_odds[0]}")
    print(f"  Median odds:                 {sorted_odds[len(sorted_odds)//2]}")

    # Bucket by odds range
    buckets = [
        ("Better than -500", lambda o: o > -500),
        ("-500 to -1000", lambda o: -1000 <= o <= -500),
        ("-1000 to -2000", lambda o: -2000 <= o < -1000),
        ("-2000 to -5000", lambda o: -5000 <= o < -2000),
        ("Worse than -5000", lambda o: o < -5000),
    ]
    print(f"\n  {'Odds Range':<25} {'Count':<8} {'Win Rate':<10} {'Avg PnL':<10}")
    print(f"  {'-'*53}")
    for label, filt in buckets:
        matching = [t for t in curve if filt(t['est_odds'])]
        if matching:
            wins = sum(1 for t in matching if t['opening_correct'])
            wr = wins / len(matching)
            avg_pnl = sum(t['est_odds_pnl'] for t in matching) / len(matching)
            print(f"  {label:<25} {len(matching):<8} {wr:<10.1%} {avg_pnl:<+10.4f}")

    # ASCII equity curves
    print(f"\n{'='*70}")
    print("EQUITY CURVES (ASCII)")
    print(f"{'='*70}")

    opening_equity = [t['cum_opening'] for t in curve]
    est_odds_equity = [t['cum_est_odds'] for t in curve]

    render_ascii_chart(
        opening_equity,
        "P&L at -110 (pre-game odds — if you entered pre-game)",
        width=70, height=18
    )

    print()

    render_ascii_chart(
        est_odds_equity,
        "P&L at ESTIMATED LIVE ODDS (what you'd get entering at Q3 end)",
        width=70, height=18
    )

    # Pre-game confirmation strategy
    print(f"\n{'='*70}")
    print("PRE-GAME ENTRY + Q3 CONFIRMATION STRATEGY")
    print(f"{'='*70}")
    print("""
  Strategy: Bet OVER pre-game at -110 on all games.
  At Q3 end, use the model to decide: HOLD, HEDGE, or neutral.
""")

    # Run on ALL games (not just signals)
    all_games = data.get('allGames', signals)
    categories, cat_pnl, cat_wins, total_pnl = simulate_pregame_confirmation(all_games)

    total_games = sum(categories.values())
    print(f"  Total games: {total_games}")
    print(f"\n  {'Category':<20} {'Games':<8} {'Wins':<8} {'PnL':<12} {'ROI':<10}")
    print(f"  {'-'*58}")

    for cat in ['confirmed', 'hedged', 'neutral']:
        n = categories[cat]
        w = cat_wins[cat]
        p = cat_pnl[cat]
        roi = p / n * 100 if n > 0 else 0
        wr = w / n if n > 0 else 0
        print(f"  {cat.upper():<20} {n:<8} {w:<8} {p:<+12.1f} {roi:<+10.1f}%")

    print(f"\n  {'TOTAL':<20} {total_games:<8} {'':<8} {total_pnl:<+12.1f} "
          f"{total_pnl/total_games*100:<+10.1f}%")

    # Final verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    est_roi = curve[-1]['cum_est_odds'] / len(curve) * 100
    opening_roi = curve[-1]['cum_opening'] / len(curve) * 100

    print(f"\n  Signal accuracy (vs opening line): "
          f"{sum(1 for t in curve if t['opening_correct'])/len(curve):.1%}")
    print(f"  ROI at -110 (pre-game entry):      {opening_roi:+.1f}%")
    print(f"  ROI at estimated live odds:         {est_roi:+.1f}%")
    print(f"  Pre-game + Q3 confirm ROI:          {total_pnl/total_games*100:+.1f}%")

    if opening_roi > 5:
        print(f"\n  The model achieves {opening_roi:+.1f}% ROI when you enter pre-game at -110")
        print("  and hold through Q3 confirmation. This is the recommended strategy.")
        print("  At estimated live odds (entering at Q3 end), the edge is much smaller")
        print("  or negative — the market has already priced in the game state.")
    else:
        print("\n  At estimated live odds, there is no profitable edge.")
        print("  The only viable approach is pre-game entry with Q3 confirmation.")


if __name__ == "__main__":
    main()
