"""
DEDUPLICATED NBA LIVE BETTING STRATEGY ANALYSIS
=================================================

CRITICAL FIX: The comprehensive_validation.json contains 5,366 records from
only 156 games. Each game has multiple records for different threshold combinations
(e.g., L14_M14, L12_M10, etc.) at the SAME game moment. This inflates sample
counts dramatically (e.g., "174 bets" is really ~13 games).

This analysis deduplicates to ONE BET PER GAME:
- For each strategy, find which games qualify
- Within each game, take the FIRST qualifying entry (highest mins_remaining)
- Calculate honest game-level win rates, P&L, and consistency

This is the ONLY honest way to evaluate these strategies.
"""

import json
import math
import os
import statistics
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass


# =============================================================================
# MARKET MODEL (calibrated to NBA benchmarks, sigma=2.6)
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
    """NBA win probability calibrated to sigma=2.6."""
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
    """Payout per unit risked on ML."""
    if odds < 0:
        return 100 / abs(odds)
    return odds / 100


# =============================================================================
# DATA LOADING AND DEDUPLICATION
# =============================================================================

@dataclass
class GameSignal:
    """One bet per game - the FIRST qualifying entry point."""
    game_key: str
    date: str
    home_team: str
    away_team: str
    side: str
    actual_lead: int
    actual_mom: int
    mins_remaining: float
    ml_won: bool
    final_margin: int
    final_home: int
    final_away: int

    @property
    def lead_extension(self) -> int:
        return self.final_margin - self.actual_lead

    @property
    def month(self) -> str:
        return self.date[:6]


def load_raw_data(filepath: str) -> List[dict]:
    with open(filepath) as f:
        return json.load(f)


def deduplicate_for_strategy(
    raw_data: List[dict],
    lead_min: int, lead_max: int,
    mom_min: int,
    time_min: float, time_max: float,
) -> List[GameSignal]:
    """
    For a given strategy condition, find qualifying games and take the
    FIRST entry point (highest mins_remaining) per game.

    This simulates a real bettor: you see the first qualifying signal
    and place ONE bet on that game.
    """
    # Group by game
    by_game = defaultdict(list)
    for r in raw_data:
        lead = r['actual_lead']
        mom = r['actual_mom']
        mins = r['mins_remaining']

        # Check strategy conditions
        if not (lead_min <= lead <= lead_max): continue
        if mom < mom_min: continue
        if not (time_min <= mins <= time_max): continue

        game_key = f"{r['date']}_{r['home_team']}_{r['away_team']}"
        by_game[game_key].append(r)

    # Take FIRST entry per game (highest mins_remaining = earliest in game)
    signals = []
    for game_key, records in by_game.items():
        # Sort by mins_remaining descending (first opportunity)
        records.sort(key=lambda x: x['mins_remaining'], reverse=True)
        first = records[0]
        signals.append(GameSignal(
            game_key=game_key,
            date=first['date'],
            home_team=first['home_team'],
            away_team=first['away_team'],
            side=first['side'],
            actual_lead=first['actual_lead'],
            actual_mom=first['actual_mom'],
            mins_remaining=first['mins_remaining'],
            ml_won=first['ml_won'],
            final_margin=first['final_margin'],
            final_home=first['final_home'],
            final_away=first['final_away'],
        ))

    # Sort by date
    signals.sort(key=lambda s: s.date)
    return signals


# =============================================================================
# ANALYSIS: FULL STRATEGY EVALUATION (one bet per game)
# =============================================================================

def evaluate_strategy(
    name: str,
    signals: List[GameSignal],
    min_games: int = 5,
) -> Optional[dict]:
    """
    Full evaluation of a strategy with ONE bet per game.
    Returns metrics dict or None if insufficient data.
    """
    n = len(signals)
    if n < min_games:
        return None

    # ML Bet Simulation
    ml_wins = sum(1 for s in signals if s.ml_won)
    ml_losses = n - ml_wins
    ml_wr = ml_wins / n

    ml_pnl = 0.0
    ml_bets = []
    for s in signals:
        odds = live_ml_odds(s.actual_lead, s.mins_remaining)
        pay = ml_payout(odds)
        result = pay if s.ml_won else -1.0
        ml_pnl += result
        ml_bets.append(result)

    ml_roi = ml_pnl / n * 100

    # Average market prob and edge
    avg_market_prob = sum(market_win_prob(s.actual_lead, s.mins_remaining) for s in signals) / n
    ml_edge = ml_wr - avg_market_prob

    # Average odds
    avg_odds = statistics.mean([live_ml_odds(s.actual_lead, s.mins_remaining) for s in signals])

    # Live Spread Simulation
    # Bet: leading team covers the live spread (≈ -lead)
    # Win if final_margin > actual_lead (lead grew)
    spread_covers = sum(1 for s in signals if s.final_margin > s.actual_lead)
    spread_pushes = sum(1 for s in signals if s.final_margin == s.actual_lead)
    spread_losses = n - spread_covers - spread_pushes
    spread_cover_rate = spread_covers / n

    # P&L at -110 both sides
    spread_pnl = spread_covers * (100/110) - spread_losses * 1.0  # pushes = no action
    spread_roi = spread_pnl / (n - spread_pushes) * 100 if (n - spread_pushes) > 0 else 0

    # Lead extension stats
    extensions = [s.lead_extension for s in signals]
    avg_ext = statistics.mean(extensions)
    med_ext = statistics.median(extensions)
    pct_grew = sum(1 for e in extensions if e > 0) / n * 100

    # Cumulative P&L and drawdown (ML)
    cum_pnl = []
    running = 0
    for b in ml_bets:
        running += b
        cum_pnl.append(running)

    max_dd = 0
    peak = 0
    for p in cum_pnl:
        if p > peak: peak = p
        dd = peak - p
        if dd > max_dd: max_dd = dd

    # Monthly consistency
    by_month = defaultdict(lambda: {'pnl': 0, 'n': 0, 'wins': 0})
    for s, result in zip(signals, ml_bets):
        month = s.month
        by_month[month]['pnl'] += result
        by_month[month]['n'] += 1
        by_month[month]['wins'] += 1 if s.ml_won else 0

    profitable_months = sum(1 for m in by_month.values() if m['pnl'] > 0)
    total_months = len(by_month)

    # Max loss streak
    max_loss_streak = 0
    current = 0
    for s in signals:
        if not s.ml_won:
            current += 1
            max_loss_streak = max(max_loss_streak, current)
        else:
            current = 0

    return {
        'name': name,
        'n_games': n,
        'ml_wins': ml_wins,
        'ml_losses': ml_losses,
        'ml_wr': ml_wr * 100,
        'ml_pnl': ml_pnl,
        'ml_roi': ml_roi,
        'ml_edge': ml_edge * 100,
        'avg_odds': avg_odds,
        'avg_market_prob': avg_market_prob * 100,
        'spread_covers': spread_covers,
        'spread_pushes': spread_pushes,
        'spread_losses': spread_losses,
        'spread_cover_rate': spread_cover_rate * 100,
        'spread_pnl': spread_pnl,
        'spread_roi': spread_roi,
        'avg_extension': avg_ext,
        'med_extension': med_ext,
        'pct_lead_grew': pct_grew,
        'max_dd': max_dd,
        'max_loss_streak': max_loss_streak,
        'profitable_months': profitable_months,
        'total_months': total_months,
        'by_month': dict(by_month),
        'signals': signals,
    }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def print_strategy_report(result: dict):
    """Print detailed report for a strategy."""
    r = result
    print(f"\n{'='*75}")
    print(f"  {r['name']}")
    print(f"{'='*75}")
    print(f"\n  SAMPLE: {r['n_games']} unique games (ONE bet per game)")
    print(f"  Date range: {r['signals'][0].date} to {r['signals'][-1].date}")
    print(f"  Months: {r['total_months']}")

    print(f"\n  --- MONEYLINE RESULTS ---")
    print(f"  Win Rate:     {r['ml_wr']:.1f}% ({r['ml_wins']}W / {r['ml_losses']}L)")
    print(f"  Avg Mkt Prob: {r['avg_market_prob']:.1f}%")
    print(f"  Edge:         {r['ml_edge']:+.1f}% (WR minus market implied)")
    print(f"  Avg Odds:     {r['avg_odds']:.0f}")
    print(f"  Total P&L:    {r['ml_pnl']:+.2f} units")
    print(f"  ROI:          {r['ml_roi']:+.2f}%")
    print(f"  Max Drawdown: {r['max_dd']:.2f} units")
    print(f"  Max Loss Run: {r['max_loss_streak']}")

    print(f"\n  --- LIVE SPREAD RESULTS ---")
    print(f"  (Bet: leading team covers live spread ≈ -lead)")
    print(f"  Cover Rate:   {r['spread_cover_rate']:.1f}% ({r['spread_covers']}W / {r['spread_losses']}L / {r['spread_pushes']}P)")
    print(f"  Break-even:   52.4% (at -110 odds)")
    profitable_spread = "YES" if r['spread_cover_rate'] > 52.4 else "NO"
    print(f"  Profitable:   {profitable_spread}")
    print(f"  Total P&L:    {r['spread_pnl']:+.2f} units")
    print(f"  ROI:          {r['spread_roi']:+.2f}%")

    print(f"\n  --- LEAD EXTENSION ---")
    print(f"  Avg extension:  {r['avg_extension']:+.1f} pts")
    print(f"  Med extension:  {r['med_extension']:+.1f} pts")
    print(f"  Lead grew (%):  {r['pct_lead_grew']:.1f}%")

    print(f"\n  --- MONTHLY CONSISTENCY (ML) ---")
    print(f"  Profitable months: {r['profitable_months']}/{r['total_months']}")
    for month in sorted(r['by_month'].keys()):
        m = r['by_month'][month]
        wr = m['wins'] / m['n'] * 100 if m['n'] > 0 else 0
        print(f"    {month}: {m['n']:>2} games, WR={wr:>5.1f}%, P&L={m['pnl']:>+6.2f}u")

    # Show losses
    losses = [s for s in r['signals'] if not s.ml_won]
    if losses:
        print(f"\n  --- ML LOSSES ({len(losses)}) ---")
        for s in losses:
            mkt = market_win_prob(s.actual_lead, s.mins_remaining) * 100
            print(f"    {s.date}: {s.home_team} vs {s.away_team} | "
                  f"Lead={s.actual_lead}, Mom={s.actual_mom}, Mins={s.mins_remaining:.0f} | "
                  f"Final={s.final_home}-{s.final_away} (margin={s.final_margin}) | "
                  f"MktProb={mkt:.0f}%")
    else:
        print(f"\n  --- NO ML LOSSES in sample ---")

    # Show worst lead collapses (for spread)
    worst_ext = sorted(r['signals'], key=lambda s: s.lead_extension)[:5]
    print(f"\n  --- WORST LEAD COLLAPSES (spread perspective) ---")
    for s in worst_ext:
        print(f"    {s.date}: {s.home_team} vs {s.away_team} | "
              f"Lead={s.actual_lead} → Margin={s.final_margin} (ext={s.lead_extension:+d})")


def grid_search_deduplicated(raw_data: List[dict]):
    """
    Exhaustive grid search with proper deduplication.
    Every result is GAMES, not inflated signals.
    """
    print("\n" + "=" * 80)
    print("GRID SEARCH: ALL CONDITIONS (DEDUPLICATED - one bet per game)")
    print("=" * 80)

    ml_results = []
    spread_results = []

    for lead_min in [8, 10, 12]:
        for lead_max in [12, 14, 16, 19, 24]:
            if lead_max < lead_min: continue
            for mom_min in [8, 10, 12, 14, 16]:
                for time_min in [12, 15, 18]:
                    for time_max in [18, 21, 24]:
                        if time_max <= time_min: continue

                        signals = deduplicate_for_strategy(
                            raw_data, lead_min, lead_max,
                            mom_min, time_min, time_max
                        )

                        n = len(signals)
                        if n < 5: continue

                        label = f"L{lead_min}-{lead_max} M{mom_min}+ T{time_min}-{time_max}"

                        # ML evaluation
                        ml_wins = sum(1 for s in signals if s.ml_won)
                        ml_wr = ml_wins / n

                        ml_pnl = 0
                        for s in signals:
                            odds = live_ml_odds(s.actual_lead, s.mins_remaining)
                            pay = ml_payout(odds)
                            ml_pnl += pay if s.ml_won else -1.0

                        ml_roi = ml_pnl / n * 100
                        avg_mkt = sum(market_win_prob(s.actual_lead, s.mins_remaining) for s in signals) / n
                        ml_edge = (ml_wr - avg_mkt) * 100

                        # Monthly consistency
                        by_month = defaultdict(float)
                        for s in signals:
                            odds = live_ml_odds(s.actual_lead, s.mins_remaining)
                            pay = ml_payout(odds)
                            by_month[s.month] += pay if s.ml_won else -1.0
                        prof_months = sum(1 for v in by_month.values() if v > 0)
                        total_months = len(by_month)

                        ml_results.append({
                            'label': label,
                            'n': n,
                            'ml_wr': ml_wr * 100,
                            'ml_roi': ml_roi,
                            'ml_pnl': ml_pnl,
                            'ml_edge': ml_edge,
                            'prof_months': prof_months,
                            'total_months': total_months,
                            'ml_losses': n - ml_wins,
                        })

                        # Live spread evaluation
                        covers = sum(1 for s in signals if s.final_margin > s.actual_lead)
                        pushes = sum(1 for s in signals if s.final_margin == s.actual_lead)
                        sp_losses = n - covers - pushes
                        cover_rate = covers / n
                        sp_pnl = covers * (100/110) - sp_losses * 1.0
                        active_bets = n - pushes
                        sp_roi = sp_pnl / active_bets * 100 if active_bets > 0 else 0

                        # Extension stats
                        exts = [s.lead_extension for s in signals]
                        avg_ext = statistics.mean(exts)
                        pct_grew = sum(1 for e in exts if e > 0) / n * 100

                        if cover_rate > 0.524:  # Profitable at -110
                            # Monthly spread consistency
                            sp_by_month = defaultdict(float)
                            for s in signals:
                                if s.final_margin > s.actual_lead:
                                    sp_by_month[s.month] += 100/110
                                elif s.final_margin < s.actual_lead:
                                    sp_by_month[s.month] -= 1.0
                            sp_prof_months = sum(1 for v in sp_by_month.values() if v > 0)

                            spread_results.append({
                                'label': label,
                                'n': n,
                                'cover_rate': cover_rate * 100,
                                'sp_roi': sp_roi,
                                'sp_pnl': sp_pnl,
                                'avg_ext': avg_ext,
                                'pct_grew': pct_grew,
                                'sp_prof_months': sp_prof_months,
                                'sp_total_months': len(sp_by_month),
                            })

    # Print ML results
    print(f"\n  TOP 25 ML STRATEGIES BY ROI (min 5 games):")
    print(f"  {'Conditions':<28} {'Games':>5} {'WR%':>6} {'Edge%':>6} {'ROI%':>7} {'P&L':>7} {'L':>3} {'M+/M':>5}")
    print(f"  {'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*3} {'-'*5}")

    ml_results.sort(key=lambda x: x['ml_roi'], reverse=True)
    for r in ml_results[:25]:
        print(f"  {r['label']:<28} {r['n']:>5} {r['ml_wr']:>5.1f}% {r['ml_edge']:>+5.1f}% "
              f"{r['ml_roi']:>+6.2f}% {r['ml_pnl']:>+6.2f}u {r['ml_losses']:>2} {r['prof_months']}/{r['total_months']}")

    # Print ML with minimum 10 games
    print(f"\n  TOP 20 ML STRATEGIES BY ROI (min 10 games, more reliable):")
    print(f"  {'Conditions':<28} {'Games':>5} {'WR%':>6} {'Edge%':>6} {'ROI%':>7} {'P&L':>7} {'L':>3} {'M+/M':>5}")
    print(f"  {'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*3} {'-'*5}")

    ml_10 = [r for r in ml_results if r['n'] >= 10]
    ml_10.sort(key=lambda x: x['ml_roi'], reverse=True)
    for r in ml_10[:20]:
        print(f"  {r['label']:<28} {r['n']:>5} {r['ml_wr']:>5.1f}% {r['ml_edge']:>+5.1f}% "
              f"{r['ml_roi']:>+6.2f}% {r['ml_pnl']:>+6.2f}u {r['ml_losses']:>2} {r['prof_months']}/{r['total_months']}")

    # Print ML with minimum 20 games
    print(f"\n  TOP 15 ML STRATEGIES BY ROI (min 20 games, most reliable):")
    print(f"  {'Conditions':<28} {'Games':>5} {'WR%':>6} {'Edge%':>6} {'ROI%':>7} {'P&L':>7} {'L':>3} {'M+/M':>5}")
    print(f"  {'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*3} {'-'*5}")

    ml_20 = [r for r in ml_results if r['n'] >= 20]
    ml_20.sort(key=lambda x: x['ml_roi'], reverse=True)
    for r in ml_20[:15]:
        print(f"  {r['label']:<28} {r['n']:>5} {r['ml_wr']:>5.1f}% {r['ml_edge']:>+5.1f}% "
              f"{r['ml_roi']:>+6.2f}% {r['ml_pnl']:>+6.2f}u {r['ml_losses']:>2} {r['prof_months']}/{r['total_months']}")

    # Print spread results
    if spread_results:
        print(f"\n  TOP 20 LIVE SPREAD STRATEGIES (>52.4% cover, min 5 games):")
        print(f"  {'Conditions':<28} {'Games':>5} {'Cover%':>7} {'ROI%':>7} {'P&L':>7} {'AvgExt':>7} {'%Grew':>5} {'M+/M':>5}")
        print(f"  {'-'*28} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5}")

        spread_results.sort(key=lambda x: x['sp_roi'], reverse=True)
        for r in spread_results[:20]:
            print(f"  {r['label']:<28} {r['n']:>5} {r['cover_rate']:>6.1f}% "
                  f"{r['sp_roi']:>+6.2f}% {r['sp_pnl']:>+6.2f}u {r['avg_ext']:>+6.1f} {r['pct_grew']:>4.0f}% "
                  f"{r['sp_prof_months']}/{r['sp_total_months']}")

        # Also show with min 10 games
        spread_10 = [r for r in spread_results if r['n'] >= 10]
        if spread_10:
            print(f"\n  TOP 15 LIVE SPREAD STRATEGIES (>52.4% cover, min 10 games):")
            print(f"  {'Conditions':<28} {'Games':>5} {'Cover%':>7} {'ROI%':>7} {'P&L':>7} {'AvgExt':>7} {'%Grew':>5} {'M+/M':>5}")
            print(f"  {'-'*28} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5}")
            spread_10.sort(key=lambda x: x['sp_roi'], reverse=True)
            for r in spread_10[:15]:
                print(f"  {r['label']:<28} {r['n']:>5} {r['cover_rate']:>6.1f}% "
                      f"{r['sp_roi']:>+6.2f}% {r['sp_pnl']:>+6.2f}u {r['avg_ext']:>+6.1f} {r['pct_grew']:>4.0f}% "
                      f"{r['sp_prof_months']}/{r['sp_total_months']}")
    else:
        print(f"\n  No live spread strategies with >52.4% cover rate found.")

    return ml_results, spread_results


def combined_strategy_analysis(raw_data: List[dict]):
    """
    Test COMBINED strategies: ML bet on high-edge signals + spread on others.
    Also test alternative sizing approaches.
    """
    print("\n" + "=" * 80)
    print("COMBINED / ALTERNATIVE STRATEGIES")
    print("=" * 80)

    # Strategy: Bet ML when lead is lower (better odds), skip extreme leads
    strategies = [
        # (name, lead_min, lead_max, mom_min, time_min, time_max)
        ("CONSERVATIVE: L10-12, M12+, 18-24min", 10, 12, 12, 18, 24),
        ("MODERATE: L10-14, M12+, 18-24min", 10, 14, 12, 18, 24),
        ("MODERATE-B: L10-14, M10+, 18-24min", 10, 14, 10, 18, 24),
        ("WIDER: L10-16, M12+, 18-24min", 10, 16, 12, 18, 24),
        ("BROADEST: L10-19, M12+, 18-24min", 10, 19, 12, 18, 24),
        ("HIGH MOM: L10-14, M14+, 12-24min", 10, 14, 14, 12, 24),
        ("HIGH MOM WIDE: L10-19, M14+, 12-24min", 10, 19, 14, 12, 24),
        ("ELITE: L10-12, M14+, 18-24min", 10, 12, 14, 18, 24),
        ("TIGHT: L10-14, M12+, 18-21min", 10, 14, 12, 18, 21),
        ("Q3 ONLY: L10-14, M12+, 15-18min", 10, 14, 12, 15, 18),
        ("LOWER LEAD: L8-12, M12+, 18-24min", 8, 12, 12, 18, 24),
        ("LOWER LEAD+MOM: L8-14, M10+, 18-24min", 8, 14, 10, 18, 24),
        ("SUPER HIGH MOM: L10-14, M16+, 12-24min", 10, 14, 16, 12, 24),
        ("SUPER HIGH MOM WIDE: L10-19, M16+, 12-24min", 10, 19, 16, 12, 24),
    ]

    results = []
    for name, lmin, lmax, mmin, tmin, tmax in strategies:
        signals = deduplicate_for_strategy(raw_data, lmin, lmax, mmin, tmin, tmax)
        result = evaluate_strategy(name, signals)
        if result:
            results.append(result)

    # Sort by ML ROI
    results.sort(key=lambda x: x['ml_roi'], reverse=True)

    print(f"\n  {'Strategy':<42} {'Games':>5} {'WR':>5} {'Edge':>5} {'ROI':>6} {'P&L':>6} {'L':>2} {'M+/M':>4} {'SpCov':>5} {'SpROI':>6}")
    print(f"  {'-'*42} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*2} {'-'*4} {'-'*5} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<42} {r['n_games']:>5} {r['ml_wr']:>4.0f}% {r['ml_edge']:>+4.1f}% "
              f"{r['ml_roi']:>+5.1f}% {r['ml_pnl']:>+5.1f}u {r['ml_losses']:>2} {r['profitable_months']}/{r['total_months']} "
              f"{r['spread_cover_rate']:>4.0f}% {r['spread_roi']:>+5.1f}%")

    # Detailed reports for top strategies
    print("\n\n" + "#" * 80)
    print("DETAILED REPORTS FOR TOP STRATEGIES")
    print("#" * 80)

    # Show top 3 by ML ROI
    for r in results[:3]:
        print_strategy_report(r)

    # Also show best spread strategy
    best_spread = max(results, key=lambda x: x['spread_roi'])
    if best_spread not in results[:3]:
        print(f"\n\n  === BEST SPREAD STRATEGY ===")
        print_strategy_report(best_spread)

    return results


def cross_validation_test(raw_data: List[dict]):
    """
    Simple cross-validation: train on first half of data, test on second half.
    With only 156 games over 4 months, this is limited but still informative.
    """
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION: FIRST HALF vs SECOND HALF")
    print("=" * 80)

    # Get all unique game dates
    game_dates = sorted(set(r['date'] for r in raw_data))
    mid = len(game_dates) // 2
    first_dates = set(game_dates[:mid])
    second_dates = set(game_dates[mid:])

    print(f"\n  Total game dates: {len(game_dates)}")
    print(f"  First half:  {game_dates[0]} to {game_dates[mid-1]} ({len(first_dates)} dates)")
    print(f"  Second half: {game_dates[mid]} to {game_dates[-1]} ({len(second_dates)} dates)")

    first_data = [r for r in raw_data if r['date'] in first_dates]
    second_data = [r for r in raw_data if r['date'] in second_dates]

    strategies = [
        ("L10-14, M12+, 18-24min", 10, 14, 12, 18, 24),
        ("L10-14, M10+, 18-24min", 10, 14, 10, 18, 24),
        ("L10-16, M12+, 18-24min", 10, 16, 12, 18, 24),
        ("L10-19, M12+, 18-24min", 10, 19, 12, 18, 24),
        ("L10-14, M14+, 12-24min", 10, 14, 14, 12, 24),
        ("L10-12, M12+, 18-24min", 10, 12, 12, 18, 24),
        ("L8-14, M10+, 18-24min", 8, 14, 10, 18, 24),
    ]

    print(f"\n  {'Strategy':<30} {'1st Half':>12}  {'2nd Half':>12}  {'Full':>12}")
    print(f"  {'':<30} {'G  WR  ROI':>12}  {'G  WR  ROI':>12}  {'G  WR  ROI':>12}")
    print(f"  {'-'*30} {'-'*12}  {'-'*12}  {'-'*12}")

    for name, lmin, lmax, mmin, tmin, tmax in strategies:
        parts = []
        for subset in [first_data, second_data, raw_data]:
            signals = deduplicate_for_strategy(subset, lmin, lmax, mmin, tmin, tmax)
            if len(signals) < 2:
                parts.append(f"{'--':>3} {'--':>4} {'--':>5}")
                continue
            n = len(signals)
            wr = sum(1 for s in signals if s.ml_won) / n * 100
            pnl = 0
            for s in signals:
                odds = live_ml_odds(s.actual_lead, s.mins_remaining)
                pay = ml_payout(odds)
                pnl += pay if s.ml_won else -1.0
            roi = pnl / n * 100
            parts.append(f"{n:>3} {wr:>3.0f}% {roi:>+5.1f}%")

        print(f"  {name:<30} {parts[0]}  {parts[1]}  {parts[2]}")


def sensitivity_analysis(raw_data: List[dict]):
    """
    Test how sensitive the best strategies are to small parameter changes.
    A robust strategy should not collapse with minor threshold adjustments.
    """
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS: How robust are the top strategies?")
    print("=" * 80)

    # Base parameters for the most promising strategies
    base_configs = [
        ("L10-14, M12+, 18-24min", 10, 14, 12, 18, 24),
        ("L10-14, M10+, 18-24min", 10, 14, 10, 18, 24),
    ]

    for base_name, b_lmin, b_lmax, b_mmin, b_tmin, b_tmax in base_configs:
        print(f"\n  BASE: {base_name}")
        print(f"  Testing ±1 adjustments to each parameter...\n")

        variations = [
            ("Base", b_lmin, b_lmax, b_mmin, b_tmin, b_tmax),
            ("Lead min +1", b_lmin+1, b_lmax, b_mmin, b_tmin, b_tmax),
            ("Lead min -1", b_lmin-1, b_lmax, b_mmin, b_tmin, b_tmax),
            ("Lead max +2", b_lmin, b_lmax+2, b_mmin, b_tmin, b_tmax),
            ("Lead max -2", b_lmin, b_lmax-2, b_mmin, b_tmin, b_tmax),
            ("Mom min +2", b_lmin, b_lmax, b_mmin+2, b_tmin, b_tmax),
            ("Mom min -2", b_lmin, b_lmax, max(b_mmin-2, 6), b_tmin, b_tmax),
            ("Time min +3", b_lmin, b_lmax, b_mmin, b_tmin+3, b_tmax),
            ("Time min -3", b_lmin, b_lmax, b_mmin, max(b_tmin-3, 6), b_tmax),
            ("Time max +3", b_lmin, b_lmax, b_mmin, b_tmin, min(b_tmax+3, 30)),
            ("Time max -3", b_lmin, b_lmax, b_mmin, b_tmin, b_tmax-3),
        ]

        print(f"  {'Variation':<20} {'Games':>5} {'WR%':>5} {'ROI%':>6} {'P&L':>6} {'SpCov%':>6}")
        print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*6}")

        for vname, lmin, lmax, mmin, tmin, tmax in variations:
            if lmax < lmin or tmax <= tmin:
                continue
            signals = deduplicate_for_strategy(raw_data, lmin, lmax, mmin, tmin, tmax)
            n = len(signals)
            if n < 3:
                print(f"  {vname:<20} {n:>5} {'--':>5} {'--':>6} {'--':>6} {'--':>6}")
                continue
            wr = sum(1 for s in signals if s.ml_won) / n * 100
            pnl = 0
            for s in signals:
                odds = live_ml_odds(s.actual_lead, s.mins_remaining)
                pay = ml_payout(odds)
                pnl += pay if s.ml_won else -1.0
            roi = pnl / n * 100
            sp_cov = sum(1 for s in signals if s.final_margin > s.actual_lead) / n * 100
            marker = " <-- BASE" if vname == "Base" else ""
            print(f"  {vname:<20} {n:>5} {wr:>4.0f}% {roi:>+5.1f}% {pnl:>+5.1f}u {sp_cov:>5.1f}%{marker}")


def game_by_game_detail(raw_data: List[dict]):
    """Show every single game for the top strategy so the user can inspect."""
    print("\n" + "=" * 80)
    print("GAME-BY-GAME DETAIL: L10-14, M12+, 18-24min (FIRST ENTRY PER GAME)")
    print("=" * 80)

    signals = deduplicate_for_strategy(raw_data, 10, 14, 12, 18, 24)

    print(f"\n  {'Date':<10} {'Matchup':<16} {'Lead':>4} {'Mom':>4} {'Mins':>5} {'Odds':>6} "
          f"{'Final':>10} {'Margin':>6} {'Ext':>4} {'ML':>3} {'Spread':>6}")
    print(f"  {'-'*10} {'-'*16} {'-'*4} {'-'*4} {'-'*5} {'-'*6} "
          f"{'-'*10} {'-'*6} {'-'*4} {'-'*3} {'-'*6}")

    total_ml_pnl = 0
    total_sp_pnl = 0

    for s in signals:
        odds = live_ml_odds(s.actual_lead, s.mins_remaining)
        pay = ml_payout(odds)
        ml_result = "W" if s.ml_won else "L"
        sp_result = "W" if s.final_margin > s.actual_lead else ("P" if s.final_margin == s.actual_lead else "L")

        ml_bet_pnl = pay if s.ml_won else -1.0
        sp_bet_pnl = 100/110 if sp_result == "W" else (-1.0 if sp_result == "L" else 0)

        total_ml_pnl += ml_bet_pnl
        total_sp_pnl += sp_bet_pnl

        matchup = f"{s.home_team}v{s.away_team}"
        final = f"{s.final_home}-{s.final_away}"

        print(f"  {s.date:<10} {matchup:<16} {s.actual_lead:>4} {s.actual_mom:>4} {s.mins_remaining:>5.1f} {odds:>6.0f} "
              f"{final:>10} {s.final_margin:>6} {s.lead_extension:>+4} {ml_result:>3} {sp_result:>6}")

    n = len(signals)
    print(f"\n  TOTALS: {n} games")
    print(f"  ML P&L:     {total_ml_pnl:>+.2f} units (ROI: {total_ml_pnl/n*100:+.1f}%)")
    print(f"  Spread P&L: {total_sp_pnl:>+.2f} units (ROI: {total_sp_pnl/n*100:+.1f}%)")


def analyze_vig_sensitivity(raw_data: List[dict]):
    """Test how different vig levels affect profitability."""
    print("\n" + "=" * 80)
    print("VIG SENSITIVITY: How much vig can the strategy survive?")
    print("=" * 80)

    strategies = [
        ("L10-14, M12+, 18-24min", 10, 14, 12, 18, 24),
        ("L10-14, M10+, 18-24min", 10, 14, 10, 18, 24),
        ("L10-16, M12+, 18-24min", 10, 16, 12, 18, 24),
    ]

    vig_levels = [0.02, 0.03, 0.045, 0.06, 0.08, 0.10]

    for name, lmin, lmax, mmin, tmin, tmax in strategies:
        signals = deduplicate_for_strategy(raw_data, lmin, lmax, mmin, tmin, tmax)
        n = len(signals)
        if n < 5: continue

        print(f"\n  {name} ({n} games):")
        print(f"  {'Vig':>6} {'Avg Odds':>9} {'ML P&L':>8} {'ML ROI':>8}")
        print(f"  {'-'*6} {'-'*9} {'-'*8} {'-'*8}")

        for vig in vig_levels:
            pnl = 0
            odds_list = []
            for s in signals:
                odds = live_ml_odds(s.actual_lead, s.mins_remaining, vig=vig)
                pay = ml_payout(odds)
                pnl += pay if s.ml_won else -1.0
                odds_list.append(odds)
            roi = pnl / n * 100
            avg_odds = statistics.mean(odds_list)
            marker = " <-- standard" if vig == 0.045 else (" <-- aggressive" if vig == 0.08 else "")
            print(f"  {vig:>5.1%} {avg_odds:>8.0f} {pnl:>+7.2f}u {roi:>+6.2f}%{marker}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comp_path = os.path.join(base_dir, 'data', 'comprehensive_validation.json')

    print("=" * 80)
    print("DEDUPLICATED NBA LIVE BETTING STRATEGY ANALYSIS")
    print("(ONE BET PER GAME - Honest Sample Counts)")
    print("=" * 80)

    raw_data = load_raw_data(comp_path)
    total_signals = len(raw_data)
    unique_games = len(set(f"{r['date']}_{r['home_team']}_{r['away_team']}" for r in raw_data))

    print(f"\nData: {total_signals} raw signal records from {unique_games} unique games")
    print(f"Season: 2023-24 (Oct 2023 - Jan 2024)")
    print(f"\nCRITICAL: All analysis below uses ONE BET PER GAME.")
    print(f"The old analysis counted {total_signals} 'signals' but these were really")
    print(f"just {unique_games} games with multiple threshold combinations per game.")

    # Run all analyses
    grid_search_deduplicated(raw_data)
    combined_strategy_analysis(raw_data)
    cross_validation_test(raw_data)
    sensitivity_analysis(raw_data)
    game_by_game_detail(raw_data)
    analyze_vig_sensitivity(raw_data)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL HONEST ASSESSMENT")
    print("=" * 80)
    print(f"""
  DATA LIMITATIONS:
  - Only {unique_games} unique games over ~4 months (Oct 2023 - Jan 2024)
  - Single season, no out-of-sample validation possible
  - All strategies are fit to this specific dataset

  KEY FINDINGS (DEDUPLICATED):
  - Sample sizes are 5-40 games, NOT hundreds of "signals"
  - At these sample sizes, even 100% win rates don't guarantee future performance
  - The market model (sigma=2.6) is calibrated but could be off by 10-20%
  - Live vig on ML favorites varies (4-10%); higher vig kills edge

  WHAT THE DATA DOES SHOW:
  - Momentum alignment at moderate leads (10-14) DOES predict lead extension
  - The leading team with strong momentum wins the game at very high rates
  - Whether this exceeds the vig-adjusted market price is the real question
  - Live spread coverage (>52.4% needed) is the hardest edge to find

  HONEST RECOMMENDATION:
  - These results are SUGGESTIVE, not conclusive
  - Need 2+ seasons of data for real validation
  - If betting, use SMALL stakes until more data confirms the edge
  - Monitor actual live odds vs model odds - if real odds are worse, no edge
    """)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
