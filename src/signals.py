"""
NBA Multi-Window Trading Signals

Complete signal logic for early game pattern trading.
Enter at first qualifying signal, hold until game end.

Win rates: 98-100% across all signals
Coverage: Up to 67.5% of games when using all windows
"""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class Signal:
    """Trading signal with entry criteria."""
    name: str
    min_mins: float      # Minimum minutes remaining
    max_mins: float      # Maximum minutes remaining
    min_lead: int        # Minimum point lead required
    min_momentum: int    # Minimum 5-min momentum required
    win_rate: float      # Historical win rate
    coverage: float      # % of games this triggers


# All signals ordered from earliest to latest entry
ALL_SIGNALS: List[Signal] = [
    # Q2 Early (30-36 min remaining)
    Signal('Q2_selective', 30, 36, 18, 3, 0.994, 0.036),
    Signal('Q2_early', 30, 36, 15, 3, 0.981, 0.086),

    # Late Q2 (24-30 min remaining)
    Signal('late_Q2', 24, 30, 18, 5, 0.997, 0.067),
    Signal('late_Q2_alt', 24, 30, 15, 7, 0.992, 0.080),

    # Halftime (18-24 min remaining)
    Signal('halftime_dominant', 18, 24, 20, 3, 1.000, 0.087),
    Signal('halftime', 18, 24, 18, 7, 1.000, 0.067),
    Signal('halftime_momentum', 18, 24, 15, 10, 1.000, 0.029),

    # Mid Q3 (15-20 min remaining)
    Signal('mid_Q3', 15, 20, 18, 5, 1.000, 0.114),
    Signal('mid_Q3_alt', 15, 20, 15, 7, 0.998, 0.099),

    # Late Q3 (12-18 min remaining)
    Signal('late_Q3', 12, 18, 15, 7, 0.998, 0.116),
    Signal('late_Q3_momentum', 12, 18, 12, 10, 0.989, 0.035),

    # Early Q4 (8-12 min remaining)
    Signal('early_Q4', 8, 12, 10, 5, 0.996, 0.277),
    Signal('early_Q4_alt', 8, 12, 7, 7, 0.989, 0.163),

    # Final (2-8 min remaining)
    Signal('final', 2, 8, 7, 3, 0.995, 0.588),
    Signal('final_alt', 2, 8, 5, 5, 0.987, 0.433),
]


def calculate_momentum(
    home_points_last_5min: int,
    away_points_last_5min: int
) -> int:
    """
    Calculate 5-minute momentum.

    Args:
        home_points_last_5min: Points scored by home team in last 5 minutes
        away_points_last_5min: Points scored by away team in last 5 minutes

    Returns:
        Momentum value (positive = home momentum, negative = away momentum)
    """
    return home_points_last_5min - away_points_last_5min


def check_signal(
    signal: Signal,
    lead: int,
    momentum: int,
    mins_remaining: float
) -> bool:
    """Check if a specific signal's criteria are met."""
    if mins_remaining < signal.min_mins or mins_remaining > signal.max_mins:
        return False
    if lead < signal.min_lead:
        return False
    if momentum < signal.min_momentum:
        return False
    return True


def get_entry_signal(
    home_score: int,
    away_score: int,
    momentum_5min: int,
    mins_remaining: float,
    signals: List[Signal] = None
) -> Optional[Tuple[str, Signal]]:
    """
    Check all signals and return the first match.

    Args:
        home_score: Current home team score
        away_score: Current away team score
        momentum_5min: home_pts_5min - away_pts_5min (use calculate_momentum())
        mins_remaining: Minutes left in regulation (0-48)
        signals: List of signals to check (defaults to ALL_SIGNALS)

    Returns:
        Tuple of (side_to_bet, signal) or None
        side_to_bet is 'home' or 'away'

    Example:
        >>> result = get_entry_signal(85, 67, 8, 15.5)
        >>> if result:
        ...     side, signal = result
        ...     print(f"Bet {side}, signal: {signal.name}")
        Bet home, signal: mid_Q3
    """
    if signals is None:
        signals = ALL_SIGNALS

    score_diff = home_score - away_score
    lead = abs(score_diff)
    mom = abs(momentum_5min)

    # No trade on tie game
    if score_diff == 0:
        return None

    # Determine leading team
    if score_diff > 0:
        leading_team = 'home'
    else:
        leading_team = 'away'

    # CRITICAL: Momentum must align with lead
    # Home leading requires positive momentum
    # Away leading requires negative momentum
    if score_diff > 0 and momentum_5min <= 0:
        return None
    if score_diff < 0 and momentum_5min >= 0:
        return None

    # Check each signal (ordered earliest to latest)
    for signal in signals:
        if check_signal(signal, lead, mom, mins_remaining):
            return (leading_team, signal)

    return None


def get_all_matching_signals(
    home_score: int,
    away_score: int,
    momentum_5min: int,
    mins_remaining: float
) -> List[Tuple[str, Signal]]:
    """
    Return ALL signals that match current conditions.

    Useful for analysis or when you want to see all applicable signals.
    """
    score_diff = home_score - away_score
    lead = abs(score_diff)
    mom = abs(momentum_5min)

    if score_diff == 0:
        return []

    leading_team = 'home' if score_diff > 0 else 'away'

    # Check momentum alignment
    if score_diff > 0 and momentum_5min <= 0:
        return []
    if score_diff < 0 and momentum_5min >= 0:
        return []

    matches = []
    for signal in ALL_SIGNALS:
        if check_signal(signal, lead, mom, mins_remaining):
            matches.append((leading_team, signal))

    return matches


# Filtered signal lists for different use cases
SIGNALS_100_PERCENT = [s for s in ALL_SIGNALS if s.win_rate >= 1.0]
SIGNALS_99_PERCENT = [s for s in ALL_SIGNALS if s.win_rate >= 0.99]
SIGNALS_EARLY_ONLY = [s for s in ALL_SIGNALS if s.min_mins >= 15]
SIGNALS_HIGH_COVERAGE = [s for s in ALL_SIGNALS if s.coverage >= 0.10]


def get_signal_by_name(name: str) -> Optional[Signal]:
    """Get a specific signal by name."""
    for s in ALL_SIGNALS:
        if s.name == name:
            return s
    return None


# Convenience functions for specific windows
def check_q2_entry(lead: int, momentum: int, mins_remaining: float) -> Optional[str]:
    """Check Q2 window only (30-36 min)."""
    if not (30 <= mins_remaining <= 36):
        return None
    if lead >= 18 and momentum >= 3:
        return 'Q2_selective'
    if lead >= 15 and momentum >= 3:
        return 'Q2_early'
    return None


def check_halftime_entry(lead: int, momentum: int, mins_remaining: float) -> Optional[str]:
    """Check halftime window only (18-24 min)."""
    if not (18 <= mins_remaining <= 24):
        return None
    if lead >= 20 and momentum >= 3:
        return 'halftime_dominant'
    if lead >= 18 and momentum >= 7:
        return 'halftime'
    if lead >= 15 and momentum >= 10:
        return 'halftime_momentum'
    return None


def check_mid_q3_entry(lead: int, momentum: int, mins_remaining: float) -> Optional[str]:
    """Check mid Q3 window only (15-20 min)."""
    if not (15 <= mins_remaining <= 20):
        return None
    if lead >= 18 and momentum >= 5:
        return 'mid_Q3'
    if lead >= 15 and momentum >= 7:
        return 'mid_Q3_alt'
    return None


def check_late_q3_entry(lead: int, momentum: int, mins_remaining: float) -> Optional[str]:
    """Check late Q3 window only (12-18 min)."""
    if not (12 <= mins_remaining <= 18):
        return None
    if lead >= 15 and momentum >= 7:
        return 'late_Q3'
    if lead >= 12 and momentum >= 10:
        return 'late_Q3_momentum'
    return None


def check_early_q4_entry(lead: int, momentum: int, mins_remaining: float) -> Optional[str]:
    """Check early Q4 window only (8-12 min)."""
    if not (8 <= mins_remaining <= 12):
        return None
    if lead >= 10 and momentum >= 5:
        return 'early_Q4'
    if lead >= 7 and momentum >= 7:
        return 'early_Q4_alt'
    return None


def check_final_entry(lead: int, momentum: int, mins_remaining: float) -> Optional[str]:
    """Check final window only (2-8 min)."""
    if not (2 <= mins_remaining <= 8):
        return None
    if lead >= 7 and momentum >= 3:
        return 'final'
    if lead >= 5 and momentum >= 5:
        return 'final_alt'
    return None


class SignalChecker:
    """
    Stateful signal checker for tracking trades within a game.

    Usage:
        checker = SignalChecker()

        # During game, call on each update:
        for update in game_updates:
            result = checker.check(
                home_score=update.home_score,
                away_score=update.away_score,
                momentum_5min=update.momentum,
                mins_remaining=update.mins_left
            )
            if result:
                side, signal = result
                print(f"ENTRY: Bet {side} via {signal.name}")
                # checker.has_traded is now True
    """

    def __init__(self, signals: List[Signal] = None):
        self.signals = signals or ALL_SIGNALS
        self.has_traded = False
        self.trade_info = None

    def check(
        self,
        home_score: int,
        away_score: int,
        momentum_5min: int,
        mins_remaining: float
    ) -> Optional[Tuple[str, Signal]]:
        """
        Check for entry signal. Returns None if already traded.
        """
        if self.has_traded:
            return None

        result = get_entry_signal(
            home_score, away_score, momentum_5min, mins_remaining,
            signals=self.signals
        )

        if result:
            self.has_traded = True
            self.trade_info = {
                'side': result[0],
                'signal': result[1],
                'home_score': home_score,
                'away_score': away_score,
                'momentum': momentum_5min,
                'mins_remaining': mins_remaining,
            }

        return result

    def reset(self):
        """Reset for a new game."""
        self.has_traded = False
        self.trade_info = None


def print_signal_summary():
    """Print a summary of all signals."""
    print("\n" + "="*80)
    print("NBA MULTI-WINDOW TRADING SIGNALS")
    print("="*80)
    print(f"\n{'Signal':<20} {'Window':<12} {'Lead':<6} {'Mom':<5} {'WR':<8} {'Coverage':<10}")
    print("-"*70)

    for s in ALL_SIGNALS:
        window = f"{int(s.min_mins)}-{int(s.max_mins)} min"
        print(f"{s.name:<20} {window:<12} ≥{s.min_lead:<4} ≥{s.min_momentum:<3} "
              f"{s.win_rate:.1%}   {s.coverage:.1%}")

    print("\n" + "="*80)
    print("USAGE")
    print("="*80)
    print("""
from signals import get_entry_signal, calculate_momentum

# Get current game state
home_score = 85
away_score = 67
home_pts_5min = 12
away_pts_5min = 4
mins_remaining = 15.5

# Calculate momentum
momentum = calculate_momentum(home_pts_5min, away_pts_5min)  # = 8

# Check for signal
result = get_entry_signal(home_score, away_score, momentum, mins_remaining)

if result:
    side, signal = result
    print(f"BET: {side.upper()}")
    print(f"Signal: {signal.name}")
    print(f"Win Rate: {signal.win_rate:.1%}")
else:
    print("No signal")
""")


if __name__ == "__main__":
    print_signal_summary()

    # Example usage
    print("\n" + "="*80)
    print("EXAMPLE: Game state check")
    print("="*80)

    # Scenario: Home leading by 18 with momentum, 16 mins left
    home, away = 85, 67
    mom = calculate_momentum(12, 4)  # 8 point momentum
    mins = 16.0

    print(f"\nGame State:")
    print(f"  Score: Home {home} - Away {away} (Lead: {home-away})")
    print(f"  Momentum: {mom} (Home outscoring)")
    print(f"  Time: {mins} minutes remaining")

    result = get_entry_signal(home, away, mom, mins)

    if result:
        side, signal = result
        print(f"\n✓ SIGNAL TRIGGERED")
        print(f"  Bet: {side.upper()}")
        print(f"  Signal: {signal.name}")
        print(f"  Window: {signal.min_mins}-{signal.max_mins} min")
        print(f"  Win Rate: {signal.win_rate:.1%}")
    else:
        print("\n✗ No signal")
