"""
Simulate equity curve betting $100 on each signal.
Uses conservative payout estimates for alternate spreads.
"""

import json
from datetime import datetime


def load_data():
    with open('data/comprehensive_validation.json', 'r') as f:
        return json.load(f)


def get_signal_from_record(r):
    """Check if record matches any of our 4 signals."""
    lead = r['actual_lead']
    mom = r['actual_mom']

    # Must have won the game (our signals require momentum alignment)
    if not r['ml_won']:
        return None

    # SWEET SPOT: Lead 10-14, Mom >= 10, Bet -7
    if 10 <= lead <= 14 and mom >= 10:
        return {'signal': 'sweet_spot', 'spread': 7, 'lead': lead, 'mom': mom}

    # MODERATE: Lead 12-16, Mom >= 12, Bet -7
    if 12 <= lead <= 16 and mom >= 12:
        return {'signal': 'moderate', 'spread': 7, 'lead': lead, 'mom': mom}

    # MID-RANGE: Lead 14-18, Mom >= 14, Bet -7
    if 14 <= lead <= 18 and mom >= 14:
        return {'signal': 'mid_range', 'spread': 7, 'lead': lead, 'mom': mom}

    # SAFE: Lead 16-20, Mom >= 12, Bet -5
    if 16 <= lead <= 20 and mom >= 12:
        return {'signal': 'safe', 'spread': 5, 'lead': lead, 'mom': mom}

    return None


def estimate_odds(lead, bet_spread):
    """
    Conservative odds estimate for alternate spreads.
    Market has ~-lead, we're betting -bet_spread.
    """
    points_bought = lead - bet_spread

    # Very conservative: -25 per point bought
    base_odds = -110
    adjustment = points_bought * 25

    return base_odds - adjustment


def calculate_payout(bet_amount, american_odds):
    """Calculate payout for a winning bet."""
    if american_odds < 0:
        return bet_amount * (100 / abs(american_odds))
    else:
        return bet_amount * (american_odds / 100)


def simulate_equity_curve(data, bet_amount=100):
    """Simulate equity curve betting fixed amount on each signal."""

    # Dedupe and filter to valid signals
    seen = set()
    trades = []

    for r in data:
        # Create unique key for each game
        key = (r['date'], r['home_team'], r['away_team'])
        if key in seen:
            continue

        signal = get_signal_from_record(r)
        if signal:
            seen.add(key)

            # Determine if bet won (final margin >= bet spread)
            won = r['final_margin'] >= signal['spread']

            # Calculate odds and payout
            odds = estimate_odds(signal['lead'], signal['spread'])

            if won:
                payout = calculate_payout(bet_amount, odds)
                profit = payout
            else:
                profit = -bet_amount

            trades.append({
                'date': r['date'],
                'teams': f"{r['away_team']}@{r['home_team']}",
                'signal': signal['signal'],
                'lead': signal['lead'],
                'mom': signal['mom'],
                'spread_bet': signal['spread'],
                'final_margin': r['final_margin'],
                'won': won,
                'odds': odds,
                'profit': profit,
            })

    # Sort by date
    trades.sort(key=lambda x: x['date'])

    return trades


def print_equity_curve(trades):
    """Print equity curve and statistics."""

    print("="*80)
    print("EQUITY CURVE SIMULATION")
    print("Bet: $100 per signal | Conservative odds estimates")
    print("="*80)

    cumulative = 0
    wins = 0
    losses = 0

    print(f"\n{'#':<4} {'Date':<10} {'Game':<12} {'Signal':<12} {'L/M':<6} {'Bet':<5} {'Final':<6} {'W/L':<4} {'Odds':<6} {'P/L':<8} {'Cumul':<10}")
    print("-"*95)

    for i, t in enumerate(trades):
        cumulative += t['profit']
        if t['won']:
            wins += 1
            wl = "WIN"
        else:
            losses += 1
            wl = "LOSS"

        print(f"{i+1:<4} {t['date']:<10} {t['teams']:<12} {t['signal']:<12} "
              f"{t['lead']}/{t['mom']:<3} -{t['spread_bet']:<4} {t['final_margin']:<6} "
              f"{wl:<4} {t['odds']:<6.0f} ${t['profit']:>+7.2f} ${cumulative:>+9.2f}")

    # Summary
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    total_wagered = total_trades * 100
    roi = (cumulative / total_wagered) * 100 if total_wagered > 0 else 0

    print("-"*95)
    print(f"\n{'SUMMARY':=^80}")
    print(f"""
Total Trades:     {total_trades}
Wins:             {wins}
Losses:           {losses}
Win Rate:         {win_rate:.1%}

Total Wagered:    ${total_wagered:,.2f}
Total Profit:     ${cumulative:+,.2f}
ROI:              {roi:+.1f}%

Average per trade: ${cumulative/total_trades:+.2f}
""")

    # Equity curve visualization (ASCII)
    print("="*80)
    print("EQUITY CURVE (ASCII)")
    print("="*80)

    equity = []
    running = 0
    for t in trades:
        running += t['profit']
        equity.append(running)

    if equity:
        min_eq = min(equity)
        max_eq = max(equity)
        range_eq = max_eq - min_eq if max_eq != min_eq else 1

        # Normalize to 40 characters height
        height = 20

        print(f"\n${max_eq:>+8.0f} |")

        for row in range(height, -1, -1):
            threshold = min_eq + (range_eq * row / height)
            line = "           |"
            for eq in equity:
                if eq >= threshold:
                    line += "*"
                else:
                    line += " "
            print(line)

        print(f"${min_eq:>+8.0f} |" + "-" * len(equity))
        print("            " + "".join([str(i % 10) for i in range(len(equity))]))

    return cumulative, wins, losses


def main():
    print("Loading historical data...")
    data = load_data()
    print(f"Total records: {len(data)}")

    print("\nSimulating equity curve...\n")
    trades = simulate_equity_curve(data, bet_amount=100)

    print_equity_curve(trades)


if __name__ == "__main__":
    main()
