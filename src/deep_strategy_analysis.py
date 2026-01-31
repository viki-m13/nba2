"""
DEEP NBA LIVE BETTING STRATEGY ANALYSIS
=========================================

Goes beyond the surface-level "do they win?" question to find ACTIONABLE,
CONSISTENTLY PROFITABLE strategies against realistic live market pricing.

KEY ANALYSES:
1. Lead extension analysis: Do momentum-aligned teams EXTEND their lead?
   (This is what matters for live spread profitability)
2. Score margin distributions by condition
3. Temporal consistency (does edge persist across different time periods?)
4. Realistic P&L simulation with proper vig/juice
5. Optimal entry conditions grid search
6. Risk-adjusted returns (Sharpe, drawdown, streak analysis)
"""

import json
import math
import os
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class Signal:
    signal: str
    window: str
    min_lead: int
    min_mom: int
    actual_lead: int
    actual_mom: int
    mins_remaining: float
    side: str
    spread_covered: bool
    ml_won: bool
    final_margin: int
    home_team: str
    away_team: str
    final_home: int
    final_away: int
    date: str

    @property
    def lead_extension(self) -> int:
        """How much the lead changed from signal to final. Positive = lead grew."""
        return self.final_margin - self.actual_lead

    @property
    def month(self) -> str:
        return self.date[:6]  # YYYYMM


def load_data(filepath: str) -> List[Signal]:
    with open(filepath) as f:
        data = json.load(f)
    records = []
    for item in data:
        records.append(Signal(
            signal=item['signal'],
            window=item.get('window', ''),
            min_lead=item.get('min_lead', 0),
            min_mom=item.get('min_mom', 0),
            actual_lead=item['actual_lead'],
            actual_mom=item['actual_mom'],
            mins_remaining=item['mins_remaining'],
            side=item['side'],
            spread_covered=item.get('spread_covered', False),
            ml_won=item.get('ml_won', False),
            final_margin=item['final_margin'],
            home_team=item.get('home_team', ''),
            away_team=item.get('away_team', ''),
            final_home=item.get('final_home', 0),
            final_away=item.get('final_away', 0),
            date=item.get('date', ''),
        ))
    return records


# =============================================================================
# MARKET MODEL (calibrated)
# =============================================================================

def normal_cdf(x: float) -> float:
    if x > 6: return 0.9999
    if x < -6: return 0.0001
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def market_win_prob(lead: int, mins_remaining: float) -> float:
    """Calibrated NBA win probability (sigma=2.6)."""
    if lead <= 0 or mins_remaining <= 0: return 0.5
    z = lead / (2.6 * math.sqrt(max(mins_remaining, 0.5)))
    return max(0.51, min(0.998, normal_cdf(z)))


def live_ml_odds(lead: int, mins_remaining: float, vig: float = 0.045) -> float:
    """American ML odds with vig."""
    prob = market_win_prob(lead, mins_remaining)
    mp = min(0.995, prob + vig * prob)
    if mp >= 0.5:
        return -(mp / (1 - mp)) * 100
    return ((1 - mp) / mp) * 100


def ml_payout(odds: float) -> float:
    """Payout per unit risked."""
    if odds < 0:
        return 100 / abs(odds)
    return odds / 100


# =============================================================================
# ANALYSIS 1: LEAD EXTENSION - The core question for live spread betting
# =============================================================================

def analyze_lead_extension(records: List[Signal]):
    """
    The KEY question for live spread betting:
    Does the leading team EXTEND their lead after the signal?

    lead_extension = final_margin - lead_at_signal
    - Positive: lead grew (team would have covered the live spread + extension)
    - Zero: lead held exactly (push on live spread)
    - Negative: lead compressed (team lost ground, didn't cover live spread)

    If momentum predicts positive lead_extension, there's a genuine live spread edge.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 1: LEAD EXTENSION BY CONDITION")
    print("=" * 80)
    print("\nlead_extension = final_margin - lead_at_signal")
    print("Positive = lead GREW (covers live spread), Negative = lead COMPRESSED")

    # Filter to realistic signal conditions
    filtered = [r for r in records if 12 <= r.mins_remaining <= 24 and r.actual_lead >= 10 and r.actual_mom >= 8]
    print(f"\nTotal qualifying signals (lead 10+, mom 8+, 12-24 min): {len(filtered)}")

    # Overall
    extensions = [r.lead_extension for r in filtered]
    if extensions:
        avg_ext = statistics.mean(extensions)
        med_ext = statistics.median(extensions)
        pct_positive = sum(1 for e in extensions if e > 0) / len(extensions) * 100
        pct_positive_1 = sum(1 for e in extensions if e > 1) / len(extensions) * 100
        print(f"\n  OVERALL (N={len(extensions)}):")
        print(f"    Avg lead extension:  {avg_ext:+.1f} pts")
        print(f"    Median extension:    {med_ext:+.1f} pts")
        print(f"    % lead grew (>0):    {pct_positive:.1f}%")
        print(f"    % lead grew by 2+:   {pct_positive_1:.1f}%")
        print(f"    Std dev:             {statistics.stdev(extensions):.1f} pts")

    # By momentum tier
    tiers = [
        ("Mom 14+ (ELITE)", lambda r: r.actual_mom >= 14),
        ("Mom 12-13",       lambda r: 12 <= r.actual_mom <= 13),
        ("Mom 10-11",       lambda r: 10 <= r.actual_mom <= 11),
        ("Mom 8-9",         lambda r: 8 <= r.actual_mom <= 9),
    ]

    print(f"\n  BY MOMENTUM TIER:")
    for name, cond in tiers:
        subset = [r for r in filtered if cond(r)]
        if not subset:
            continue
        exts = [r.lead_extension for r in subset]
        avg = statistics.mean(exts)
        med = statistics.median(exts)
        pct_pos = sum(1 for e in exts if e > 0) / len(exts) * 100
        ml_wr = sum(1 for r in subset if r.ml_won) / len(subset) * 100
        print(f"    {name} (N={len(subset)}):")
        print(f"      Avg extension: {avg:+.1f} | Median: {med:+.1f} | Lead grew: {pct_pos:.0f}% | ML WR: {ml_wr:.1f}%")

    # By lead range
    print(f"\n  BY LEAD RANGE:")
    lead_ranges = [
        ("Lead 10-12", 10, 12),
        ("Lead 13-15", 13, 15),
        ("Lead 16-19", 16, 19),
        ("Lead 20-24", 20, 24),
        ("Lead 25-29", 25, 29),
        ("Lead 30+",   30, 99),
    ]

    for name, lo, hi in lead_ranges:
        subset = [r for r in filtered if lo <= r.actual_lead <= hi]
        if not subset:
            continue
        exts = [r.lead_extension for r in subset]
        avg = statistics.mean(exts)
        pct_pos = sum(1 for e in exts if e > 0) / len(exts) * 100
        pct_neg = sum(1 for e in exts if e < -3) / len(exts) * 100
        ml_wr = sum(1 for r in subset if r.ml_won) / len(subset) * 100
        print(f"    {name} (N={len(subset)}):")
        print(f"      Avg extension: {avg:+.1f} | Lead grew: {pct_pos:.0f}% | Collapsed(>3): {pct_neg:.0f}% | ML WR: {ml_wr:.1f}%")

    # By time window
    print(f"\n  BY TIME WINDOW:")
    time_ranges = [
        ("18-24 min (Halftime)", 18, 24),
        ("15-18 min (Early Q3)", 15, 18),
        ("12-15 min (Mid Q3)",   12, 15),
    ]

    for name, lo, hi in time_ranges:
        subset = [r for r in filtered if lo <= r.mins_remaining <= hi]
        if not subset:
            continue
        exts = [r.lead_extension for r in subset]
        avg = statistics.mean(exts)
        pct_pos = sum(1 for e in exts if e > 0) / len(exts) * 100
        print(f"    {name} (N={len(subset)}):")
        print(f"      Avg extension: {avg:+.1f} | Lead grew: {pct_pos:.0f}%")

    # CRITICAL CROSS: Lead range x Momentum
    print(f"\n  CROSS-ANALYSIS: LEAD RANGE x MOMENTUM:")
    for lname, llo, lhi in [("Lead 10-14", 10, 14), ("Lead 15-19", 15, 19), ("Lead 20+", 20, 99)]:
        for mname, mcond in [("Mom 14+", lambda r: r.actual_mom >= 14),
                              ("Mom 12+", lambda r: r.actual_mom >= 12),
                              ("Mom 10+", lambda r: r.actual_mom >= 10)]:
            subset = [r for r in filtered if llo <= r.actual_lead <= lhi and mcond(r)]
            if len(subset) < 10:
                continue
            exts = [r.lead_extension for r in subset]
            avg = statistics.mean(exts)
            pct_pos = sum(1 for e in exts if e > 0) / len(exts) * 100
            ml_wr = sum(1 for r in subset if r.ml_won) / len(subset) * 100
            marker = "***" if pct_pos > 55 and avg > 0 else "   "
            print(f"    {marker} {lname} x {mname} (N={len(subset)}):")
            print(f"          Avg ext: {avg:+.1f} | Lead grew: {pct_pos:.0f}% | ML WR: {ml_wr:.1f}%")


# =============================================================================
# ANALYSIS 2: REALISTIC P&L SIMULATION
# =============================================================================

def simulate_pnl(records: List[Signal]):
    """
    Full P&L simulation with realistic market pricing.

    For EACH signal:
    1. Estimate live ML odds based on lead + time
    2. Simulate ML bet result
    3. Estimate live spread (≈ -lead) and check coverage
    4. Track cumulative P&L, drawdown, streaks
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 2: REALISTIC P&L SIMULATION")
    print("=" * 80)

    # Sort by date for time-series analysis
    records_sorted = sorted(records, key=lambda r: r.date)

    # Test multiple strategies
    strategies = [
        ("A: Lead 10-14, Mom 12+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 14 and r.actual_mom >= 12 and 18 <= r.mins_remaining <= 24),
        ("B: Lead 10-14, Mom 10+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 14 and r.actual_mom >= 10 and 18 <= r.mins_remaining <= 24),
        ("C: Lead 10-19, Mom 12+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 19 and r.actual_mom >= 12 and 18 <= r.mins_remaining <= 24),
        ("D: Lead 10-19, Mom 14+, 12-24 min",
         lambda r: 10 <= r.actual_lead <= 19 and r.actual_mom >= 14 and 12 <= r.mins_remaining <= 24),
        ("E: Lead 10-14, Mom 14+, 12-24 min",
         lambda r: 10 <= r.actual_lead <= 14 and r.actual_mom >= 14 and 12 <= r.mins_remaining <= 24),
        ("F: Lead 10-14, Mom 10+, 12-24 min (broad)",
         lambda r: 10 <= r.actual_lead <= 14 and r.actual_mom >= 10 and 12 <= r.mins_remaining <= 24),
        # Live spread strategies (bet the actual live spread)
        ("G: LIVE SPREAD Lead 10-14, Mom 12+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 14 and r.actual_mom >= 12 and 18 <= r.mins_remaining <= 24),
        ("H: LIVE SPREAD Lead 10-14, Mom 10+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 14 and r.actual_mom >= 10 and 18 <= r.mins_remaining <= 24),
    ]

    for strat_name, condition in strategies:
        is_spread = strat_name.startswith("G:") or strat_name.startswith("H:")
        qualifying = [r for r in records_sorted if condition(r)]

        if len(qualifying) < 10:
            continue

        # Simulate P&L
        bets = []
        for r in qualifying:
            if is_spread:
                # Live spread bet at -110 odds (standard)
                # Win if final_margin > actual_lead (team extended lead)
                won = r.final_margin > r.actual_lead
                payout = 100 / 110 if won else -1.0  # -110 both sides
            else:
                # ML bet
                odds = live_ml_odds(r.actual_lead, r.mins_remaining)
                pay = ml_payout(odds)
                won = r.ml_won
                payout = pay if won else -1.0

            bets.append({
                'date': r.date,
                'won': won,
                'payout': payout,
                'lead': r.actual_lead,
                'mom': r.actual_mom,
                'mins': r.mins_remaining,
                'final_margin': r.final_margin,
            })

        # Calculate metrics
        total_pnl = sum(b['payout'] for b in bets)
        n = len(bets)
        wins = sum(1 for b in bets if b['won'])
        losses = n - wins
        wr = wins / n * 100

        # Cumulative P&L for drawdown
        cum_pnl = []
        running = 0
        for b in bets:
            running += b['payout']
            cum_pnl.append(running)

        max_dd = 0
        peak = 0
        for p in cum_pnl:
            if p > peak:
                peak = p
            dd = peak - p
            if dd > max_dd:
                max_dd = dd

        roi = total_pnl / n * 100 if n > 0 else 0

        # Streak analysis
        max_loss_streak = 0
        current_streak = 0
        for b in bets:
            if not b['won']:
                current_streak += 1
                max_loss_streak = max(max_loss_streak, current_streak)
            else:
                current_streak = 0

        # Monthly consistency
        monthly = defaultdict(lambda: {'pnl': 0, 'n': 0})
        for b in bets:
            month = b['date'][:6]
            monthly[month]['pnl'] += b['payout']
            monthly[month]['n'] += 1

        profitable_months = sum(1 for m in monthly.values() if m['pnl'] > 0)
        total_months = len(monthly)

        avg_payout_per_bet = total_pnl / n if n > 0 else 0

        print(f"\n  {'='*70}")
        print(f"  STRATEGY {strat_name}")
        print(f"  {'='*70}")
        print(f"  Bets: {n}  |  Wins: {wins}  |  Losses: {losses}  |  WR: {wr:.1f}%")
        print(f"  Total P&L:       {total_pnl:+.2f} units")
        print(f"  ROI per bet:     {roi:+.2f}%")
        print(f"  Avg P&L/bet:     {avg_payout_per_bet:+.4f} units")
        print(f"  Max drawdown:    {max_dd:.2f} units")
        print(f"  Max loss streak: {max_loss_streak}")
        print(f"  Monthly: {profitable_months}/{total_months} profitable months")

        if total_months > 0:
            print(f"  Monthly breakdown:")
            for month in sorted(monthly.keys()):
                m = monthly[month]
                print(f"    {month}: {m['pnl']:+.2f}u ({m['n']} bets)")


# =============================================================================
# ANALYSIS 3: GRID SEARCH FOR OPTIMAL CONDITIONS
# =============================================================================

def grid_search_conditions(records: List[Signal]):
    """
    Exhaustive grid search over lead, momentum, and time conditions
    to find the most profitable actionable strategy.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 3: GRID SEARCH FOR OPTIMAL ML CONDITIONS")
    print("=" * 80)

    results = []

    for lead_min in [10, 11, 12, 13, 14, 15]:
        for lead_max in [14, 16, 19, 24, 99]:
            if lead_max < lead_min:
                continue
            for mom_min in [8, 10, 12, 14]:
                for time_min in [12, 15, 18]:
                    for time_max in [18, 21, 24]:
                        if time_max <= time_min:
                            continue

                        subset = [r for r in records
                                  if lead_min <= r.actual_lead <= lead_max
                                  and r.actual_mom >= mom_min
                                  and time_min <= r.mins_remaining <= time_max]

                        if len(subset) < 20:
                            continue

                        n = len(subset)
                        ml_wins = sum(1 for r in subset if r.ml_won)
                        ml_wr = ml_wins / n

                        # Calculate realistic P&L
                        total_pnl = 0
                        for r in subset:
                            odds = live_ml_odds(r.actual_lead, r.mins_remaining)
                            pay = ml_payout(odds)
                            if r.ml_won:
                                total_pnl += pay
                            else:
                                total_pnl -= 1.0

                        roi = total_pnl / n * 100

                        # Average market prob
                        avg_mkt = sum(market_win_prob(r.actual_lead, r.mins_remaining) for r in subset) / n
                        edge = ml_wr - avg_mkt

                        # Lead extension stats
                        exts = [r.lead_extension for r in subset]
                        avg_ext = statistics.mean(exts)
                        pct_ext = sum(1 for e in exts if e > 0) / n * 100

                        results.append({
                            'label': f"L{lead_min}-{lead_max} M{mom_min}+ T{time_min}-{time_max}",
                            'n': n,
                            'ml_wr': ml_wr * 100,
                            'roi': roi,
                            'pnl': total_pnl,
                            'edge': edge * 100,
                            'avg_ext': avg_ext,
                            'pct_ext': pct_ext,
                        })

    # Sort by ROI
    results.sort(key=lambda x: x['roi'], reverse=True)

    print(f"\n  TOP 20 ML STRATEGIES BY ROI (min 20 signals):")
    print(f"  {'Conditions':<30} {'N':>5} {'WR%':>6} {'Edge%':>6} {'ROI%':>7} {'P&L':>8} {'AvgExt':>7} {'%Ext':>5}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*5}")

    for r in results[:20]:
        print(f"  {r['label']:<30} {r['n']:>5} {r['ml_wr']:>5.1f}% {r['edge']:>+5.1f}% {r['roi']:>+6.2f}% {r['pnl']:>+7.2f}u {r['avg_ext']:>+6.1f} {r['pct_ext']:>4.0f}%")

    # Also show top live spread strategies
    print(f"\n  TOP 15 LIVE SPREAD STRATEGIES BY COVER RATE (min 20 signals):")
    spread_results = []

    for lead_min in [10, 11, 12, 13]:
        for lead_max in [12, 13, 14, 15, 16, 19]:
            if lead_max < lead_min:
                continue
            for mom_min in [8, 10, 12, 14]:
                for time_min in [12, 15, 18]:
                    for time_max in [18, 21, 24]:
                        if time_max <= time_min:
                            continue

                        subset = [r for r in records
                                  if lead_min <= r.actual_lead <= lead_max
                                  and r.actual_mom >= mom_min
                                  and time_min <= r.mins_remaining <= time_max]

                        if len(subset) < 20:
                            continue

                        n = len(subset)
                        # Live spread coverage: final_margin > actual_lead
                        covers = sum(1 for r in subset if r.final_margin > r.actual_lead)
                        cover_rate = covers / n

                        # P&L at -110
                        pnl = covers * (100/110) - (n - covers) * 1.0
                        roi = pnl / n * 100

                        if cover_rate > 0.524:  # Profitable at -110
                            spread_results.append({
                                'label': f"L{lead_min}-{lead_max} M{mom_min}+ T{time_min}-{time_max}",
                                'n': n,
                                'cover_rate': cover_rate * 100,
                                'roi': roi,
                                'pnl': pnl,
                            })

    spread_results.sort(key=lambda x: x['roi'], reverse=True)

    print(f"  {'Conditions':<30} {'N':>5} {'Cover%':>7} {'ROI%':>7} {'P&L':>8}")
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*7} {'-'*8}")

    for r in spread_results[:15]:
        print(f"  {r['label']:<30} {r['n']:>5} {r['cover_rate']:>6.1f}% {r['roi']:>+6.2f}% {r['pnl']:>+7.2f}u")

    if not spread_results:
        print(f"  No conditions found with >52.4% live spread coverage rate.")


# =============================================================================
# ANALYSIS 4: TEMPORAL CONSISTENCY
# =============================================================================

def analyze_temporal_consistency(records: List[Signal]):
    """Check if edge persists across different time periods."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: TEMPORAL CONSISTENCY")
    print("=" * 80)

    # Best strategies from grid search
    strategies = [
        ("Lead 10-14, Mom 12+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 14 and r.actual_mom >= 12 and 18 <= r.mins_remaining <= 24),
        ("Lead 10-14, Mom 10+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 14 and r.actual_mom >= 10 and 18 <= r.mins_remaining <= 24),
        ("Lead 10-19, Mom 12+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 19 and r.actual_mom >= 12 and 18 <= r.mins_remaining <= 24),
    ]

    for strat_name, condition in strategies:
        qualifying = [r for r in records if condition(r)]

        if len(qualifying) < 10:
            continue

        # Group by month
        by_month = defaultdict(list)
        for r in qualifying:
            by_month[r.month].append(r)

        # Group by date (game-level)
        by_date = defaultdict(list)
        for r in qualifying:
            by_date[r.date].append(r)

        print(f"\n  STRATEGY: {strat_name}")
        print(f"  Total signals: {len(qualifying)} across {len(by_date)} game dates")

        print(f"\n  Monthly breakdown:")
        months = sorted(by_month.keys())
        for month in months:
            recs = by_month[month]
            n = len(recs)
            wins = sum(1 for r in recs if r.ml_won)
            wr = wins / n * 100

            # P&L
            pnl = 0
            for r in recs:
                odds = live_ml_odds(r.actual_lead, r.mins_remaining)
                pay = ml_payout(odds)
                pnl += pay if r.ml_won else -1.0

            exts = [r.lead_extension for r in recs]
            avg_ext = statistics.mean(exts) if exts else 0

            print(f"    {month}: N={n:>3}, WR={wr:>5.1f}%, ML P&L={pnl:>+6.2f}u, AvgExt={avg_ext:>+5.1f}")

        # Loss analysis
        losses = [r for r in qualifying if not r.ml_won]
        if losses:
            print(f"\n  LOSS ANALYSIS ({len(losses)} losses):")
            for r in losses:
                mkt_prob = market_win_prob(r.actual_lead, r.mins_remaining) * 100
                print(f"    {r.date} {r.home_team} vs {r.away_team}: "
                      f"Lead={r.actual_lead}, Mom={r.actual_mom}, "
                      f"Final={r.final_home}-{r.final_away} (margin={r.final_margin}), "
                      f"MktProb={mkt_prob:.0f}%")


# =============================================================================
# ANALYSIS 5: WHAT BET IS ACTUALLY AVAILABLE?
# =============================================================================

def analyze_actionable_bets(records: List[Signal]):
    """
    For each condition, show EXACTLY what bet would be placed and expected return.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 5: ACTIONABLE BET DETAILS")
    print("=" * 80)

    strategies = [
        ("STRATEGY A: Lead 10-14, Mom 12+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 14 and r.actual_mom >= 12 and 18 <= r.mins_remaining <= 24),
        ("STRATEGY B: Lead 10-14, Mom 10+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 14 and r.actual_mom >= 10 and 18 <= r.mins_remaining <= 24),
        ("STRATEGY C: Lead 10-19, Mom 12+, 18-24 min",
         lambda r: 10 <= r.actual_lead <= 19 and r.actual_mom >= 12 and 18 <= r.mins_remaining <= 24),
    ]

    for strat_name, condition in strategies:
        qualifying = [r for r in records if condition(r)]
        if not qualifying:
            continue

        n = len(qualifying)
        wins = sum(1 for r in qualifying if r.ml_won)
        losses = n - wins
        wr = wins / n * 100

        # Odds distribution
        odds_list = [live_ml_odds(r.actual_lead, r.mins_remaining) for r in qualifying]
        avg_odds = statistics.mean(odds_list)
        min_odds = min(odds_list)
        max_odds = max(odds_list)

        # P&L
        total_pnl = 0
        bet_details = []
        for r in qualifying:
            odds = live_ml_odds(r.actual_lead, r.mins_remaining)
            pay = ml_payout(odds)
            result = pay if r.ml_won else -1.0
            total_pnl += result
            bet_details.append({
                'odds': odds,
                'payout': pay,
                'result': result,
                'lead': r.actual_lead,
            })

        roi = total_pnl / n * 100
        avg_payout_on_win = statistics.mean([b['payout'] for b in bet_details if b['result'] > 0]) if wins > 0 else 0

        print(f"\n  {'='*70}")
        print(f"  {strat_name}")
        print(f"  {'='*70}")
        print(f"\n  SIGNAL CONDITIONS:")
        print(f"    Lead: {min(r.actual_lead for r in qualifying)}-{max(r.actual_lead for r in qualifying)} pts")
        print(f"    Momentum: {min(r.actual_mom for r in qualifying)}+ pts (aligned with lead)")
        print(f"    Time: 18-24 min remaining (around halftime)")
        print(f"\n  WHAT BET TO PLACE:")
        print(f"    Type: MONEYLINE on the leading team")
        print(f"    Expected odds range: {max_odds:.0f} to {min_odds:.0f} (avg {avg_odds:.0f})")
        print(f"    Example: Team up 12 at halftime with +12 mom → ML at ~{live_ml_odds(12, 20):.0f}")
        print(f"\n  RESULTS ({n} historical bets):")
        print(f"    Win Rate:          {wr:.1f}% ({wins}W / {losses}L)")
        print(f"    Avg payout on win: {avg_payout_on_win:.4f} units per unit risked")
        print(f"    Total P&L:         {total_pnl:+.2f} units")
        print(f"    ROI:               {roi:+.2f}%")
        print(f"\n  EXAMPLE BETS:")
        print(f"    $100 per bet on {n} signals = ${n * 100} total wagered")
        print(f"    Estimated return:   ${total_pnl * 100:+.0f}")
        print(f"    Per winning bet:    Win ${avg_payout_on_win * 100:.2f} on $100")
        print(f"    Per losing bet:     Lose $100")

        # Break-even analysis
        break_even_wr = 1 / (1 + avg_payout_on_win)
        print(f"\n  BREAK-EVEN ANALYSIS:")
        print(f"    Break-even WR at avg odds: {break_even_wr * 100:.1f}%")
        print(f"    Actual WR:                 {wr:.1f}%")
        print(f"    WR cushion:                {wr - break_even_wr * 100:+.1f}%")

        # Show individual losses if any
        loss_recs = [r for r in qualifying if not r.ml_won]
        if loss_recs:
            print(f"\n  LOSSES ({len(loss_recs)}):")
            for r in loss_recs:
                print(f"    {r.date}: {r.home_team} {r.final_home} - {r.away_team} {r.final_away} "
                      f"(Lead was {r.actual_lead}, mom {r.actual_mom}, "
                      f"final margin {r.final_margin})")
        else:
            print(f"\n  NO LOSSES in sample period")


# =============================================================================
# MAIN
# =============================================================================

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comp_path = os.path.join(base_dir, 'data', 'comprehensive_validation.json')

    print("=" * 80)
    print("DEEP NBA LIVE BETTING STRATEGY ANALYSIS")
    print("=" * 80)

    records = load_data(comp_path)
    print(f"\nLoaded {len(records)} signal occurrences")

    # Deduplicate by game date to avoid counting the same game moment multiple times
    # Multiple signals per game are OK if they're at different times,
    # but the same game at the same lead/momentum is a duplicate
    unique_games = len(set(f"{r.date}_{r.home_team}_{r.away_team}" for r in records))
    print(f"From {unique_games} unique games")

    # Run all analyses
    analyze_lead_extension(records)
    simulate_pnl(records)
    grid_search_conditions(records)
    analyze_temporal_consistency(records)
    analyze_actionable_bets(records)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
