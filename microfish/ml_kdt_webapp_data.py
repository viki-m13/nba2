#!/usr/bin/env python3
"""
Generate webapp data files from KDT backtest results.
Produces signals, stats, and tier details for the live-alerts webapp view.
"""

import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'moneyline')
KDT_RESULTS_FILE = os.path.join(DATA_DIR, 'kdt_backtest_results.json')
WEBAPP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'webapp', 'live-alerts', 'data')

KDT_TIERS_META = {
    'APEX': {'color': '#f59e0b', 'name': 'APEX — Structural Lock',
             'rule': 'Lead 3+ | KD ≥ 10% | IDS Entry'},
    'WAVE': {'color': '#8b5cf6', 'name': 'WAVE — Rally Survivor',
             'rule': 'Lead 2+ | KD ≥ 10% | IDS Entry'},
    'DRIFT': {'color': '#06b6d4', 'name': 'DRIFT — Deep Divergence',
              'rule': 'Lead 1+ | KD ≥ 10% | IDS Entry'},
}


def generate_webapp_files():
    """Generate all webapp data files from KDT backtest results."""
    with open(KDT_RESULTS_FILE, 'r') as f:
        results = json.load(f)

    os.makedirs(WEBAPP_DIR, exist_ok=True)

    signals = results.get('signals', [])

    # Transform signals to webapp format
    webapp_signals = []
    for s in signals:
        webapp_signals.append({
            'date': s['date'],
            'betType': 'live_moneyline',
            'team': s['team'],
            'opponent': s['opponent'],
            'tierId': s['tier_id'],
            'tierName': s['tier_name'],
            'inning': s['inning'],
            'lead': s['lead'],
            'buyPrice': s['buy_price'],
            'side': s['side'],
            'hit': s['won'],
            'pnl': s['pnl'],
            'wager': s['wager'],
            'score': s.get('score', ''),
            'finalScore': s.get('final_score', ''),
            'kd': s.get('kd', 0),
            'peakKd': s.get('peak_kd', 0),
            'swp': s.get('swp', 0),
            'profitPerShare': round(1.0 - s['buy_price'], 4) if s['won'] else 0,
            'roiIfWin': round((1.0 - s['buy_price']) / s['buy_price'] * 100, 1) if s['buy_price'] > 0 else 0,
            'bet': f"{s['team']} ML @ ${s['buy_price']:.2f} (Inn {s['inning']}, +{s['lead']}, KD {s.get('peak_kd', 0):.0%})",
            'engine': 'kdt_ids',
            'source': 'backtest',
        })

    # Sort by date descending
    webapp_signals.sort(key=lambda s: s['date'], reverse=True)

    # Build tier stats
    tier_stats = {}
    for tid, meta in KDT_TIERS_META.items():
        tier_sigs = [s for s in webapp_signals if s['tierId'] == tid]
        if not tier_sigs:
            continue
        wins = sum(1 for s in tier_sigs if s['hit'])
        losses = len(tier_sigs) - wins
        pnl = sum(s['pnl'] for s in tier_sigs)
        wagered = sum(s['wager'] for s in tier_sigs)
        avg_buy = sum(s['buyPrice'] for s in tier_sigs) / len(tier_sigs)

        tier_stats[tid] = {
            'id': tid,
            'name': meta['name'],
            'color': meta['color'],
            'rule': meta['rule'],
            'total': len(tier_sigs),
            'wins': wins,
            'losses': losses,
            'accuracy': round(wins / len(tier_sigs), 4),
            'pnl': round(pnl, 2),
            'wagered': round(wagered, 2),
            'roi': round(pnl / wagered, 4) if wagered else 0,
            'avg_buy': round(avg_buy, 4),
            'avg_profit_per_bet': round(pnl / len(tier_sigs), 2),
        }

    # Overall stats
    total_bets = len(webapp_signals)
    total_wins = sum(1 for s in webapp_signals if s['hit'])
    total_pnl = sum(s['pnl'] for s in webapp_signals)
    total_wagered = sum(s['wager'] for s in webapp_signals)

    stats = {
        'overall': {
            'total': total_bets,
            'wins': total_wins,
            'losses': total_bets - total_wins,
            'accuracy': round(total_wins / total_bets, 4) if total_bets else 0,
            'pnl': round(total_pnl, 2),
            'wagered': round(total_wagered, 2),
            'roi': round(total_pnl / total_wagered, 4) if total_wagered else 0,
        },
        'tiers': tier_stats,
        'daily': {
            'bets_per_day': round(total_bets / 178, 1),
            'profit_per_day': round(total_pnl / 178, 2),
        },
        'strategy_description': {
            'name': 'Kinetic Divergence Theory (KDT)',
            'type': 'Live in-game moneyline — IDS entry method',
            'data_source': '2,364 real 2025 MLB games (ESPN play-by-play)',
            'target': 'Buy when market underprices a leading team (KD > 10%)',
            'method': (
                'Separates win probability into Structural (empirical, discrete) '
                'and Kinetic (model, continuous). When Kinetic drops below Structural '
                'by 10%+ during an opponent rally, and the rally fails (inning ends), '
                'buy at the start of the next inning. The market is slow to reprice '
                'after a failed rally — this captures the structural edge.'
            ),
        },
    }

    # Save files
    signals_file = os.path.join(WEBAPP_DIR, 'mlb_live_signals.json')
    stats_file = os.path.join(WEBAPP_DIR, 'mlb_live_stats.json')
    tiers_file = os.path.join(WEBAPP_DIR, 'mlb_live_tiers.json')

    with open(signals_file, 'w') as f:
        json.dump(webapp_signals, f, indent=2)

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    tiers_data = {
        'tiers': [
            {
                'id': tid,
                'name': meta['name'],
                'color': meta['color'],
                'stats': tier_stats.get(tid, {}),
            }
            for tid, meta in KDT_TIERS_META.items()
        ],
    }
    with open(tiers_file, 'w') as f:
        json.dump(tiers_data, f, indent=2)

    print(f"Generated KDT webapp data files in {WEBAPP_DIR}:")
    print(f"  {signals_file}: {len(webapp_signals)} signals")
    print(f"  {stats_file}: overall stats")
    print(f"  {tiers_file}: {len(KDT_TIERS_META)} tiers")

    print(f"\n{'Tier':<25} {'Bets':>6} {'W':>5} {'L':>4} {'Acc':>7} {'P&L':>10} {'ROI':>8}")
    print('-' * 70)
    for tid, ts in tier_stats.items():
        print(f"{ts['name']:<25} {ts['total']:>6} {ts['wins']:>5} {ts['losses']:>4} "
              f"{ts['accuracy']:>6.1%} ${ts['pnl']:>9,.2f} {ts['roi']:>+7.2%}")

    print(f"\n{'TOTAL':<25} {stats['overall']['total']:>6} {stats['overall']['wins']:>5} "
          f"{stats['overall']['losses']:>4} {stats['overall']['accuracy']:>6.1%} "
          f"${stats['overall']['pnl']:>9,.2f} {stats['overall']['roi']:>+7.2%}")

    return stats


if __name__ == '__main__':
    generate_webapp_files()
