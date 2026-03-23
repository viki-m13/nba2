#!/usr/bin/env python3
"""
Microfish LIVE Moneyline Alert Engine
========================================
Continuously monitors all active MLB games via ESPN API and fires
real-time alerts when profit-optimized entry points are detected.

STRATEGY: Buy moneylines when the market (ESPN WP) underprices a team's
true win probability based on game situation (inning + run lead).

The alert tiers are calibrated from 2,364 real 2025 MLB games:
- Each tier has 90%+ actual win rate
- Buy price is BELOW the true win probability (positive edge)
- Profit per share = $1.00 - buy_price when the team wins

USAGE:
  python ml_live_alerts.py              # Run live monitor
  python ml_live_alerts.py --test       # Test with sample data
  python ml_live_alerts.py --backtest   # Backtest on 2025 season
"""

import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta
from collections import defaultdict

# ─── PROFIT-OPTIMIZED ALERT TIERS ──────────────────────────────────────────
# Calibrated on 2,364 real 2025 MLB games
# Each tier: game condition where actual win rate >> ESPN win probability
#
# Priority: Highest EV per bet first, then highest volume
# All tiers validated at 90%+ accuracy with positive expected value

# REFINED TIERS — all validated at 90%+ accuracy with positive ROI
# on full 2025 season backtest (2,364 games, 1,810 bets)
ALERT_TIERS = [
    {
        'id': 'CHARLIE',
        'name': 'CHARLIE — Late Innings Grip',
        'min_inning': 7,
        'min_lead': 3,
        'max_buy': 0.90,
        'validated_accuracy': 0.912,
        'validated_games': 434,
        'validated_pnl': 1649.30,
        'validated_roi': 0.0435,
        'description': '3+ run lead in 7th+ inning, buy ≤ $0.90',
        'alert_emoji': '🟡',
        'priority': 1,
    },
    {
        'id': 'DELTA',
        'name': 'DELTA — Late Comfortable',
        'min_inning': 8,
        'min_lead': 3,
        'max_buy': 0.95,
        'validated_accuracy': 0.944,
        'validated_games': 521,
        'validated_pnl': 915.20,
        'validated_roi': 0.019,
        'description': '3+ run lead in 8th+ inning, buy ≤ $0.95',
        'alert_emoji': '🟢',
        'priority': 2,
    },
    {
        'id': 'YANKEE',
        'name': 'YANKEE — 9th Inning Lock',
        'min_inning': 9,
        'min_lead': 3,
        'max_buy': 0.95,
        'validated_accuracy': 0.947,
        'validated_games': 170,
        'validated_pnl': 376.70,
        'validated_roi': 0.024,
        'description': '3+ run lead in 9th, buy ≤ $0.95',
        'alert_emoji': '🔵',
        'priority': 3,
    },
    {
        'id': 'XRAY',
        'name': 'XRAY — Absolute Blowout',
        'min_inning': 8,
        'min_lead': 5,
        'max_buy': 0.99,
        'validated_accuracy': 0.994,
        'validated_games': 326,
        'validated_pnl': 309.80,
        'validated_roi': 0.0097,
        'description': '5+ run lead in 8th+, buy ≤ $0.99 (99.4% accuracy)',
        'alert_emoji': '💎',
        'priority': 4,
    },
    {
        'id': 'BRAVO',
        'name': 'BRAVO — 7th+ Commanding',
        'min_inning': 7,
        'min_lead': 4,
        'max_buy': 0.95,
        'validated_accuracy': 0.944,
        'validated_games': 359,
        'validated_pnl': 269.20,
        'validated_roi': 0.008,
        'description': '4+ run lead in 7th+ inning, buy ≤ $0.95',
        'alert_emoji': '🟠',
        'priority': 5,
    },
]


# ─── ESPN API ──────────────────────────────────────────────────────────────

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
ESPN_GAME_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={event_id}"


def fetch_active_games():
    """Fetch all active MLB games from ESPN scoreboard."""
    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        events = data.get('events', [])

        games = []
        for event in events:
            competition = event.get('competitions', [{}])[0]
            status = competition.get('status', {})
            status_type = status.get('type', {})
            state = status_type.get('state', '')

            # We want in-progress games
            competitors = competition.get('competitors', [])
            home = away = None
            for c in competitors:
                if c.get('homeAway') == 'home':
                    home = c
                elif c.get('homeAway') == 'away':
                    away = c

            if not home or not away:
                continue

            game_info = {
                'event_id': event.get('id', ''),
                'state': state,
                'status_detail': status.get('type', {}).get('shortDetail', ''),
                'home_team': home.get('team', {}).get('abbreviation', ''),
                'away_team': away.get('team', {}).get('abbreviation', ''),
                'home_score': int(home.get('score', '0')),
                'away_score': int(away.get('score', '0')),
                'inning': status.get('period', 0),
                'inning_half': status.get('type', {}).get('shortDetail', ''),
            }

            games.append(game_info)

        return games

    except Exception as e:
        print(f"  [ERROR] Failed to fetch scoreboard: {e}")
        return []


def fetch_game_win_probability(event_id):
    """Fetch live win probability for a specific game."""
    try:
        url = ESPN_GAME_URL.format(event_id=event_id)
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()
        wp_data = data.get('winprobability', [])

        if not wp_data:
            # Try to get from predictor
            predictor = data.get('predictor', {})
            home_wp = predictor.get('homeTeam', {}).get('gameProjection', 50) / 100
            return {
                'homeWP': home_wp,
                'awayWP': 1 - home_wp,
                'source': 'predictor',
            }

        # Get the latest win probability
        latest = wp_data[-1]
        home_wp = latest.get('homeWinPercentage', 0.5)
        return {
            'homeWP': round(home_wp, 4),
            'awayWP': round(1 - home_wp, 4),
            'source': 'winprobability',
        }

    except Exception as e:
        return None


def evaluate_game(game, home_wp, away_wp):
    """
    Evaluate a game state against all alert tiers.
    Returns list of triggered alerts, sorted by priority.
    """
    alerts = []

    inning = game.get('inning', 0)
    home_score = game.get('home_score', 0)
    away_score = game.get('away_score', 0)
    home_team = game.get('home_team', '???')
    away_team = game.get('away_team', '???')

    # Check home team leading
    home_lead = home_score - away_score
    if home_lead >= 1:
        for tier in ALERT_TIERS:
            if (inning >= tier['min_inning'] and
                home_lead >= tier['min_lead'] and
                home_wp <= tier['max_buy']):
                alerts.append({
                    'tier': tier,
                    'team': home_team,
                    'opponent': away_team,
                    'side': 'home',
                    'buy_price': round(home_wp, 3),
                    'inning': inning,
                    'lead': home_lead,
                    'score': f"{away_team} {away_score} - {home_team} {home_score}",
                    'profit_per_share': round(1.0 - home_wp, 3),
                    'ev_per_bet': round(
                        (tier['validated_accuracy'] * (1.0 - home_wp)) -
                        ((1 - tier['validated_accuracy']) * home_wp), 4
                    ),
                })

    # Check away team leading
    away_lead = away_score - home_score
    if away_lead >= 1:
        for tier in ALERT_TIERS:
            if (inning >= tier['min_inning'] and
                away_lead >= tier['min_lead'] and
                away_wp <= tier['max_buy']):
                alerts.append({
                    'tier': tier,
                    'team': away_team,
                    'opponent': home_team,
                    'side': 'away',
                    'buy_price': round(away_wp, 3),
                    'inning': inning,
                    'lead': away_lead,
                    'score': f"{away_team} {away_score} @ {home_team} {home_score}",
                    'profit_per_share': round(1.0 - away_wp, 3),
                    'ev_per_bet': round(
                        (tier['validated_accuracy'] * (1.0 - away_wp)) -
                        ((1 - tier['validated_accuracy']) * away_wp), 4
                    ),
                })

    # Sort by EV per bet (highest first), then by priority
    alerts.sort(key=lambda a: (-a['ev_per_bet'], a['tier']['priority']))
    return alerts


def format_alert(alert, is_new=True):
    """Format a single alert for display."""
    tier = alert['tier']
    prefix = "🚨 NEW ALERT" if is_new else "   ACTIVE"

    lines = [
        f"",
        f"  {prefix} — {tier['alert_emoji']} {tier['name']}",
        f"  ┌─────────────────────────────────────────────────────────────",
        f"  │ BUY: {alert['team']} moneyline",
        f"  │ Price: ${alert['buy_price']:.2f} (limit buy)",
        f"  │ Profit if win: ${alert['profit_per_share']:.2f}/share "
        f"(+{alert['profit_per_share']/alert['buy_price']*100:.0f}% return)",
        f"  │ EV per $100 bet: ${alert['ev_per_bet'] * 100:.2f}",
        f"  │ Accuracy: {tier['validated_accuracy']:.1%} "
        f"({tier['validated_games']} games validated)",
        f"  │ Score: {alert['score']} (Inn {alert['inning']}, "
        f"{alert['lead']}-run lead)",
        f"  │ Rule: {tier['description']}",
        f"  └─────────────────────────────────────────────────────────────",
    ]
    return '\n'.join(lines)


def format_game_line(game, home_wp=None, away_wp=None, has_alert=False):
    """Format a single game for the scoreboard."""
    state = game.get('state', '')
    detail = game.get('status_detail', '')
    home = game.get('home_team', '???')
    away = game.get('away_team', '???')
    hs = game.get('home_score', 0)
    as_ = game.get('away_score', 0)

    if state == 'in':
        status = f"Inn {game.get('inning', '?')}"
        wp_str = ""
        if home_wp is not None:
            wp_str = f" | WP: {away}={away_wp:.0%} {home}={home_wp:.0%}"
        marker = " ◀ ALERT" if has_alert else ""
        return f"  {away:>4} {as_:>2} @ {home:<4} {hs:<2}  [{status}]{wp_str}{marker}"
    elif state == 'post':
        return f"  {away:>4} {as_:>2} @ {home:<4} {hs:<2}  [FINAL]"
    else:
        return f"  {away:>4}    @ {home:<4}     [{detail}]"


# ─── LIVE MONITOR ──────────────────────────────────────────────────────────

TRACKING_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'moneyline', 'live_session_tracking.json'
)


class LiveAlertMonitor:
    """Continuous live game monitor with profit-optimized alerts and real-time tracking."""

    def __init__(self, poll_interval=30, bet_size=100):
        self.poll_interval = poll_interval
        self.bet_size = bet_size
        self.active_alerts = {}   # event_id+team+tier → alert
        self.alert_history = []   # all alerts fired this session
        self.resolved_bets = []   # bets with known outcomes (won/lost)
        self.pending_bets = {}    # event_id+team → alert (awaiting game end)
        self.session_stats = {
            'alerts_fired': 0,
            'bets_won': 0,
            'bets_lost': 0,
            'bets_pending': 0,
            'total_wagered': 0,
            'total_profit': 0,
            'session_start': datetime.now().isoformat(),
            'games_monitored': 0,
            'tier_stats': {t['id']: {'wins': 0, 'losses': 0, 'profit': 0} for t in ALERT_TIERS},
        }
        self._load_tracking()

    def run(self):
        """Main monitoring loop."""
        print("=" * 70)
        print("  MICROFISH LIVE MONEYLINE ALERT ENGINE")
        print("  Profit-Optimized Strategy (90%+ accuracy, max EV)")
        print(f"  Poll interval: {self.poll_interval}s | Bet size: ${self.bet_size}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print("\n  Monitoring active MLB games...\n")

        self._print_tier_summary()

        try:
            while True:
                self._poll_cycle()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self._print_session_summary()

    def _load_tracking(self):
        """Load persistent tracking data from disk."""
        if os.path.exists(TRACKING_FILE):
            try:
                with open(TRACKING_FILE, 'r') as f:
                    saved = json.load(f)
                self.resolved_bets = saved.get('resolved_bets', [])
                # Restore cumulative stats
                for bet in self.resolved_bets:
                    tid = bet.get('tier_id', '')
                    if bet.get('won'):
                        self.session_stats['bets_won'] += 1
                        if tid in self.session_stats['tier_stats']:
                            self.session_stats['tier_stats'][tid]['wins'] += 1
                            self.session_stats['tier_stats'][tid]['profit'] += bet.get('profit', 0)
                    elif bet.get('won') is False:
                        self.session_stats['bets_lost'] += 1
                        if tid in self.session_stats['tier_stats']:
                            self.session_stats['tier_stats'][tid]['losses'] += 1
                            self.session_stats['tier_stats'][tid]['profit'] += bet.get('profit', 0)
                    self.session_stats['total_wagered'] += bet.get('wager', 0)
                    self.session_stats['total_profit'] += bet.get('profit', 0)
                print(f"  Loaded {len(self.resolved_bets)} historical bets from tracking file")
            except Exception:
                pass

    def _save_tracking(self):
        """Save tracking data to disk."""
        os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
        data = {
            'last_updated': datetime.now().isoformat(),
            'session_stats': self.session_stats,
            'resolved_bets': self.resolved_bets[-500:],  # Keep last 500
            'pending_bets': [
                {k: v for k, v in bet.items() if k != 'tier'}
                for bet in self.pending_bets.values()
            ],
        }
        with open(TRACKING_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _resolve_finished_games(self, finished_games):
        """Check if any pending bets have resolved (game finished)."""
        finished_ids = {g['event_id'] for g in finished_games}

        resolved_keys = []
        for key, bet in self.pending_bets.items():
            eid = key.split('_')[0]
            if eid in finished_ids:
                # Find the finished game to get final score
                game = next((g for g in finished_games if g['event_id'] == eid), None)
                if not game:
                    continue

                home_score = game.get('home_score', 0)
                away_score = game.get('away_score', 0)
                home_won = home_score > away_score

                # Did our team win?
                won = (bet['side'] == 'home' and home_won) or \
                      (bet['side'] == 'away' and not home_won)

                buy_price = bet['buy_price']
                profit = (1.0 - buy_price) * self.bet_size if won else -buy_price * self.bet_size
                wager = buy_price * self.bet_size

                # Record result
                result = {
                    'time': bet.get('time', ''),
                    'team': bet['team'],
                    'opponent': bet.get('opponent', ''),
                    'tier_id': bet['tier']['id'] if isinstance(bet.get('tier'), dict) else bet.get('tier_id', ''),
                    'buy_price': buy_price,
                    'won': won,
                    'profit': round(profit, 2),
                    'wager': round(wager, 2),
                    'final_score': f"{game.get('away_team', '')} {away_score} @ {game.get('home_team', '')} {home_score}",
                    'resolved_at': datetime.now().isoformat(),
                }
                self.resolved_bets.append(result)

                # Update stats
                tid = result['tier_id']
                self.session_stats['total_wagered'] += wager
                self.session_stats['total_profit'] += profit
                if won:
                    self.session_stats['bets_won'] += 1
                    if tid in self.session_stats['tier_stats']:
                        self.session_stats['tier_stats'][tid]['wins'] += 1
                        self.session_stats['tier_stats'][tid]['profit'] += profit
                    print(f"\n  ✅ WIN: {bet['team']} — bought ${buy_price:.2f}, "
                          f"profit +${(1-buy_price)*self.bet_size:.2f} "
                          f"({result['final_score']})")
                else:
                    self.session_stats['bets_lost'] += 1
                    if tid in self.session_stats['tier_stats']:
                        self.session_stats['tier_stats'][tid]['losses'] += 1
                        self.session_stats['tier_stats'][tid]['profit'] += profit
                    print(f"\n  ❌ LOSS: {bet['team']} — bought ${buy_price:.2f}, "
                          f"lost -${buy_price*self.bet_size:.2f} "
                          f"({result['final_score']})")

                resolved_keys.append(key)

        for key in resolved_keys:
            del self.pending_bets[key]
            self.session_stats['bets_pending'] = len(self.pending_bets)

        if resolved_keys:
            self._save_tracking()

    def _print_tier_summary(self):
        """Print the alert tiers being monitored."""
        print("  ALERT TIERS:")
        for tier in ALERT_TIERS:
            print(f"    {tier['alert_emoji']} {tier['id']}: Inn {tier['min_inning']}+ "
                  f"Lead {tier['min_lead']}+ Buy≤${tier['max_buy']:.2f} "
                  f"({tier['validated_accuracy']:.0%} acc, "
                  f"ROI +{tier.get('validated_roi', 0)*100:.1f}%, "
                  f"P&L ${tier.get('validated_pnl', 0):,.0f})")
        print()

    def _poll_cycle(self):
        """Single poll cycle: fetch games, evaluate, alert."""
        now = datetime.now().strftime('%H:%M:%S')
        games = fetch_active_games()

        if not games:
            print(f"  [{now}] No games found on ESPN scoreboard")
            return

        in_progress = [g for g in games if g.get('state') == 'in']
        scheduled = [g for g in games if g.get('state') == 'pre']
        finished = [g for g in games if g.get('state') == 'post']

        self.session_stats['games_monitored'] = len(games)

        # Header
        print(f"\n  [{now}] {len(in_progress)} live | "
              f"{len(scheduled)} scheduled | {len(finished)} final")

        if not in_progress:
            if scheduled:
                print(f"  Next games starting soon...")
                for g in scheduled[:3]:
                    print(format_game_line(g))
            return

        # Resolve any finished games that had pending bets
        if finished:
            self._resolve_finished_games(finished)

        # Check each live game
        new_alerts = []
        current_alert_keys = set()

        for game in in_progress:
            eid = game['event_id']

            # Fetch live win probability
            wp = fetch_game_win_probability(eid)
            if not wp:
                print(format_game_line(game))
                continue

            home_wp = wp['homeWP']
            away_wp = wp['awayWP']

            # Evaluate against all tiers
            alerts = evaluate_game(game, home_wp, away_wp)

            has_alert = len(alerts) > 0
            print(format_game_line(game, home_wp, away_wp, has_alert))

            for alert in alerts:
                alert_key = f"{eid}_{alert['team']}_{alert['tier']['id']}"
                current_alert_keys.add(alert_key)

                if alert_key not in self.active_alerts:
                    # New alert!
                    alert['time'] = now
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append({
                        'time': now,
                        **alert,
                        'tier_name': alert['tier']['name'],
                    })
                    self.session_stats['alerts_fired'] += 1
                    new_alerts.append(alert)

                    # Add to pending bets for resolution tracking
                    pending_key = f"{eid}_{alert['team']}"
                    if pending_key not in self.pending_bets:
                        self.pending_bets[pending_key] = alert
                        self.session_stats['bets_pending'] = len(self.pending_bets)
                else:
                    # Update existing alert
                    self.active_alerts[alert_key] = alert

        # Print new alerts prominently
        for alert in new_alerts:
            print(format_alert(alert, is_new=True))

        # Clean up expired alerts (game ended or condition no longer met)
        expired = [k for k in self.active_alerts if k not in current_alert_keys]
        for k in expired:
            old = self.active_alerts.pop(k)
            print(f"\n  ⏹ EXPIRED: {old['tier']['name']} — {old['team']} "
                  f"(condition no longer met)")

        # Print live dashboard
        self._print_live_dashboard()

        # Rate limit ESPN API calls
        time.sleep(0.5)

    def _print_live_dashboard(self):
        """Print real-time P&L and accuracy dashboard."""
        s = self.session_stats
        total_resolved = s['bets_won'] + s['bets_lost']

        if total_resolved == 0 and s['alerts_fired'] == 0:
            return

        acc = s['bets_won'] / total_resolved * 100 if total_resolved > 0 else 0
        roi = s['total_profit'] / s['total_wagered'] * 100 if s['total_wagered'] > 0 else 0

        print(f"\n  {'─' * 60}")
        print(f"  📊 LIVE DASHBOARD | Alerts: {s['alerts_fired']} | "
              f"Pending: {s['bets_pending']} | "
              f"Resolved: {total_resolved}")
        if total_resolved > 0:
            pnl_color = '+' if s['total_profit'] >= 0 else ''
            print(f"  📈 Record: {s['bets_won']}W / {s['bets_lost']}L "
                  f"({acc:.1f}% accuracy) | "
                  f"P&L: ${pnl_color}{s['total_profit']:.2f} | "
                  f"ROI: {roi:+.2f}%")

            # Per-tier breakdown
            tier_lines = []
            for t in ALERT_TIERS:
                ts = s['tier_stats'].get(t['id'], {})
                tw = ts.get('wins', 0)
                tl = ts.get('losses', 0)
                tp = ts.get('profit', 0)
                if tw + tl > 0:
                    ta = tw / (tw + tl) * 100
                    tier_lines.append(
                        f"  {t['alert_emoji']} {t['id']}: {tw}W/{tl}L "
                        f"({ta:.0f}%) ${'+' if tp >= 0 else ''}{tp:.0f}"
                    )
            if tier_lines:
                for line in tier_lines:
                    print(line)

        print(f"  {'─' * 60}")

    def _print_session_summary(self):
        """Print comprehensive summary when monitor is stopped."""
        s = self.session_stats
        start = datetime.fromisoformat(s['session_start']) if isinstance(s['session_start'], str) else s['session_start']
        duration = datetime.now() - start
        total_resolved = s['bets_won'] + s['bets_lost']

        print("\n\n" + "=" * 70)
        print("  SESSION SUMMARY")
        print("=" * 70)
        print(f"  Duration: {duration}")
        print(f"  Games monitored: {s['games_monitored']}")
        print(f"  Total alerts fired: {s['alerts_fired']}")
        print(f"  Pending resolution: {s['bets_pending']}")

        if total_resolved > 0:
            acc = s['bets_won'] / total_resolved * 100
            roi = s['total_profit'] / s['total_wagered'] * 100 if s['total_wagered'] > 0 else 0
            print(f"\n  RESULTS:")
            print(f"    Record: {s['bets_won']}W / {s['bets_lost']}L ({acc:.1f}%)")
            print(f"    Total wagered: ${s['total_wagered']:,.2f}")
            print(f"    Total profit: ${'+' if s['total_profit'] >= 0 else ''}{s['total_profit']:,.2f}")
            print(f"    ROI: {roi:+.2f}%")

            print(f"\n  PER-TIER BREAKDOWN:")
            for t in ALERT_TIERS:
                ts = s['tier_stats'].get(t['id'], {})
                tw = ts.get('wins', 0)
                tl = ts.get('losses', 0)
                tp = ts.get('profit', 0)
                if tw + tl > 0:
                    ta = tw / (tw + tl) * 100
                    print(f"    {t['alert_emoji']} {t['id']}: {tw}W / {tl}L "
                          f"({ta:.1f}%) | P&L ${'+' if tp >= 0 else ''}{tp:.2f}")

        if self.alert_history:
            print(f"\n  ALERT LOG ({len(self.alert_history)} alerts):")
            for i, a in enumerate(self.alert_history[-20:]):  # Show last 20
                tier = a.get('tier_name', a.get('tier', {}).get('name', '?'))
                print(f"    {i+1}. [{a['time']}] {tier} — "
                      f"BUY {a['team']} @ ${a['buy_price']:.2f} "
                      f"(EV ${a['ev_per_bet']*100:.2f}/$100)")

        # Save final tracking
        self._save_tracking()
        print(f"\n  Tracking data saved to {TRACKING_FILE}")
        print("  Goodbye!")


# ─── BACKTEST ──────────────────────────────────────────────────────────────

def run_backtest():
    """Backtest the profit-optimized tiers on full 2025 season data."""
    cache_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'moneyline', 'espn_live_winprob_cache.json'
    )
    with open(cache_file, 'r') as f:
        cache = json.load(f)

    valid = {k: v for k, v in cache.items()
             if v.get('n_plays', 0) > 0 and not v.get('error')}

    print("=" * 80)
    print("BACKTEST: Profit-Optimized Live Moneyline Strategy")
    print(f"Data: {len(valid)} MLB games, 2025 season")
    print("=" * 80)

    BET_SIZE = 100
    GAME_DAYS = 178

    tier_results = {t['id']: {
        'tier': t,
        'bets': [],
        'wins': 0,
        'losses': 0,
        'total_wagered': 0,
        'total_profit': 0,
        'biggest_win': 0,
        'biggest_loss': 0,
        'loss_details': [],
    } for t in ALERT_TIERS}

    for eid, game in valid.items():
        timeline = game.get('timeline', [])
        if not timeline:
            continue

        home_score_final = game.get('home_score', 0)
        away_score_final = game.get('away_score', 0)
        if home_score_final == away_score_final:
            continue

        home_won = home_score_final > away_score_final

        # Track first trigger per tier per side
        triggered = set()

        for play in timeline:
            inning = play.get('inning', 0)
            home_score = play.get('homeScore', 0)
            away_score = play.get('awayScore', 0)
            home_wp = play.get('homeWP', 0.5)
            away_wp = play.get('awayWP', 0.5)

            if inning < 1:
                continue

            # Home leading
            home_lead = home_score - away_score
            if home_lead >= 1:
                for tier in ALERT_TIERS:
                    key = (tier['id'], 'home')
                    if key in triggered:
                        continue
                    if (inning >= tier['min_inning'] and
                        home_lead >= tier['min_lead'] and
                        home_wp <= tier['max_buy']):
                        triggered.add(key)
                        buy_price = home_wp
                        won = home_won
                        profit = (1.0 - buy_price) if won else -buy_price
                        r = tier_results[tier['id']]
                        r['bets'].append({
                            'buy_price': buy_price,
                            'won': won,
                            'profit': profit,
                            'profit_dollars': profit * BET_SIZE,
                        })
                        r['total_wagered'] += buy_price * BET_SIZE
                        r['total_profit'] += profit * BET_SIZE
                        if won:
                            r['wins'] += 1
                            r['biggest_win'] = max(r['biggest_win'],
                                                    profit * BET_SIZE)
                        else:
                            r['losses'] += 1
                            r['biggest_loss'] = min(r['biggest_loss'],
                                                     profit * BET_SIZE)
                            r['loss_details'].append({
                                'date': game.get('date', ''),
                                'game': f"{game.get('away', '')}@{game.get('home', '')}",
                                'buy_price': round(buy_price, 3),
                                'inning': inning,
                                'lead': home_lead,
                                'final': f"{away_score_final}-{home_score_final}",
                            })

            # Away leading
            away_lead = away_score - home_score
            if away_lead >= 1:
                for tier in ALERT_TIERS:
                    key = (tier['id'], 'away')
                    if key in triggered:
                        continue
                    if (inning >= tier['min_inning'] and
                        away_lead >= tier['min_lead'] and
                        away_wp <= tier['max_buy']):
                        triggered.add(key)
                        buy_price = away_wp
                        won = not home_won
                        profit = (1.0 - buy_price) if won else -buy_price
                        r = tier_results[tier['id']]
                        r['bets'].append({
                            'buy_price': buy_price,
                            'won': won,
                            'profit': profit,
                            'profit_dollars': profit * BET_SIZE,
                        })
                        r['total_wagered'] += buy_price * BET_SIZE
                        r['total_profit'] += profit * BET_SIZE
                        if won:
                            r['wins'] += 1
                            r['biggest_win'] = max(r['biggest_win'],
                                                    profit * BET_SIZE)
                        else:
                            r['losses'] += 1
                            r['biggest_loss'] = min(r['biggest_loss'],
                                                     profit * BET_SIZE)
                            r['loss_details'].append({
                                'date': game.get('date', ''),
                                'game': f"{game.get('away', '')}@{game.get('home', '')}",
                                'buy_price': round(buy_price, 3),
                                'inning': inning,
                                'lead': away_lead,
                                'final': f"{away_score_final}-{home_score_final}",
                            })

    # Print results
    total_profit = 0
    total_bets = 0
    total_wagered = 0

    for tier in ALERT_TIERS:
        r = tier_results[tier['id']]
        n = len(r['bets'])
        if n == 0:
            continue

        accuracy = r['wins'] / n if n else 0
        avg_buy = sum(b['buy_price'] for b in r['bets']) / n if n else 0
        avg_profit = r['total_profit'] / n if n else 0
        roi = r['total_profit'] / r['total_wagered'] * 100 if r['total_wagered'] else 0
        bets_per_day = n / GAME_DAYS
        daily_profit = r['total_profit'] / GAME_DAYS

        print(f"\n  {tier['alert_emoji']} {tier['name']}")
        print(f"    Bets: {n} ({r['wins']}W / {r['losses']}L)")
        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    Avg buy price: ${avg_buy:.3f}")
        print(f"    Total wagered: ${r['total_wagered']:,.0f}")
        print(f"    Total profit: ${r['total_profit']:,.2f}")
        print(f"    ROI: {roi:+.2f}%")
        print(f"    Avg profit/bet: ${avg_profit:.2f}")
        print(f"    Bets/day: {bets_per_day:.1f}")
        print(f"    Daily profit: ${daily_profit:.2f}")
        print(f"    Season projection: ${r['total_profit']:,.0f}")

        if r['biggest_win'] > 0:
            print(f"    Biggest win: +${r['biggest_win']:.2f}")
        if r['biggest_loss'] < 0:
            print(f"    Biggest loss: -${abs(r['biggest_loss']):.2f}")

        if r['loss_details']:
            print(f"    Losses ({len(r['loss_details'])}):")
            for ld in r['loss_details'][:5]:
                print(f"      {ld['date']}: {ld['game']} — bought ${ld['buy_price']:.2f} "
                      f"inn {ld['inning']} lead {ld['lead']} → LOST {ld['final']}")

        total_profit += r['total_profit']
        total_bets += n
        total_wagered += r['total_wagered']

    print(f"\n\n  {'=' * 60}")
    print(f"  COMBINED BACKTEST RESULTS ($100/bet)")
    print(f"  {'=' * 60}")
    print(f"  Total bets: {total_bets}")
    print(f"  Total wagered: ${total_wagered:,.0f}")
    print(f"  Total profit: ${total_profit:,.2f}")
    overall_roi = total_profit / total_wagered * 100 if total_wagered else 0
    print(f"  Overall ROI: {overall_roi:+.2f}%")
    print(f"  Bets/day: {total_bets / GAME_DAYS:.1f}")
    print(f"  Daily profit: ${total_profit / GAME_DAYS:.2f}")
    print(f"  Season profit: ${total_profit:,.0f}")

    # Compare to old strategy
    # Old: 3.1 bets/day × $0.03 avg profit = $9.30/day
    old_season = 7.90 * 178
    print(f"\n  vs OLD STRATEGY (100% acc, $0.95-$0.99):")
    print(f"    Old season: ${old_season:,.0f}")
    print(f"    IMPROVEMENT: {total_profit / old_season:.1f}x more profitable!")

    # Save backtest results
    output = {
        'strategy': 'profit_optimized_live_alerts',
        'date': datetime.now().isoformat(),
        'bet_size': BET_SIZE,
        'total_bets': total_bets,
        'total_wagered': total_wagered,
        'total_profit': round(total_profit, 2),
        'overall_roi': round(overall_roi, 4),
        'daily_profit': round(total_profit / GAME_DAYS, 2),
        'tiers': {
            tid: {
                'name': r['tier']['name'],
                'bets': len(r['bets']),
                'wins': r['wins'],
                'losses': r['losses'],
                'accuracy': round(r['wins'] / len(r['bets']), 4) if r['bets'] else 0,
                'total_profit': round(r['total_profit'], 2),
                'roi': round(r['total_profit'] / r['total_wagered'] * 100, 4)
                    if r['total_wagered'] else 0,
            }
            for tid, r in tier_results.items()
            if r['bets']
        },
    }

    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'moneyline', 'profit_backtest_results.json'
    )
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Backtest results saved to {output_file}")

    return tier_results


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    if '--backtest' in sys.argv:
        run_backtest()
    elif '--test' in sys.argv:
        # Quick test with fake data
        print("Testing alert evaluation...")
        test_game = {
            'event_id': 'test123',
            'state': 'in',
            'home_team': 'NYY',
            'away_team': 'BOS',
            'home_score': 6,
            'away_score': 2,
            'inning': 7,
        }
        alerts = evaluate_game(test_game, home_wp=0.88, away_wp=0.12)
        for a in alerts:
            print(format_alert(a))
        if not alerts:
            print("  No alerts triggered (all conditions not met or price too high)")
    else:
        monitor = LiveAlertMonitor(
            poll_interval=int(sys.argv[1]) if len(sys.argv) > 1 else 30,
        )
        monitor.run()


if __name__ == '__main__':
    main()
