"""
Final Strategy Optimization

Key findings so far:
1. spread_dev >= 3.0 gives 90%+ win rate but fewer trades
2. Lower thresholds give more trades but lower win rate
3. Multiple signals can increase frequency

Strategy: Use tight take profit with selective entry
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


def generate_games(n_games: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate game data quickly."""
    np.random.seed(seed)
    all_states = []

    for g in range(n_games):
        gid = f"G{g:05d}"
        home_edge = np.random.normal(2, 6)
        vol = np.random.uniform(0.8, 1.2)

        h_score, a_score = 0, 0
        h_hist, a_hist = [], []
        last_spread = -home_edge / 2

        for secs in range(0, 48*60+1, 15):
            mins_rem = max((48*60 - secs) / 60, 0)

            h_pts = np.random.poisson(0.55 * vol * (1 + home_edge/100))
            a_pts = np.random.poisson(0.55 * vol * (1 - home_edge/100))
            h_score += h_pts
            a_score += a_pts
            h_hist.append(h_pts)
            a_hist.append(a_pts)

            mom_2 = sum(h_hist[-8:]) - sum(a_hist[-8:])

            score_diff = h_score - a_score
            hca = 3.0 * (mins_rem / 48)
            fair = -score_diff - hca
            market = fair - mom_2 * 0.25 + np.random.normal(0, 0.2)

            all_states.append({
                'gid': gid, 'secs': secs, 'mins': mins_rem,
                'diff': score_diff, 'mom': mom_2,
                'mkt': market, 'fair': fair, 'dev': market - fair,
            })

    return pd.DataFrame(all_states)


@dataclass
class Trade:
    gid: str
    pnl: float
    reason: str


def backtest(df: pd.DataFrame, dev_th: float = 2.5, mom_th: float = 5,
             tp: float = 0.4, sl: float = 5.0, ts: float = 90, cd: float = 20) -> Dict:
    """Run backtest."""
    trades = []
    positions = {}
    last_entry = {}

    for gid, gdf in df.groupby('gid'):
        for _, r in gdf.sort_values('secs').iterrows():
            secs, mins, mkt = r['secs'], r['mins'], r['mkt']

            # Check exit
            if gid in positions:
                entry_mkt, entry_secs, side = positions[gid]
                pnl = (mkt - entry_mkt) if side == 'away' else (entry_mkt - mkt)

                reason = None
                if pnl >= tp: reason = 'tp'
                elif pnl <= -sl: reason = 'sl'
                elif secs - entry_secs >= ts: reason = 'ts'

                if reason:
                    trades.append(Trade(gid, pnl, reason))
                    del positions[gid]
                continue

            # Check entry
            if mins < 4 or mins > 44: continue
            if abs(r['diff']) > 20: continue
            if secs - last_entry.get(gid, -999) < cd: continue

            # Signal: deviation OR momentum
            signal = None
            if abs(r['dev']) >= dev_th:
                signal = 'away' if r['dev'] < 0 else 'home'
            elif abs(r['mom']) >= mom_th:
                signal = 'away' if r['mom'] > 0 else 'home'

            if signal:
                positions[gid] = (mkt, secs, signal)
                last_entry[gid] = secs

        # Close at game end
        if gid in positions:
            final = gdf.iloc[-1]
            entry_mkt, _, side = positions[gid]
            pnl = (final['mkt'] - entry_mkt) if side == 'away' else (entry_mkt - final['mkt'])
            trades.append(Trade(gid, pnl, 'end'))
            del positions[gid]

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'cov': 0}

    total = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    pnls = [t.pnl for t in trades]
    games = len(set(t.gid for t in trades))
    n_games = df['gid'].nunique()
    tp_pct = sum(1 for t in trades if t.reason == 'tp') / total

    return {
        'trades': total,
        'wins': wins,
        'wr': wins / total,
        'pnl': sum(pnls),
        'cov': games / n_games,
        'tpg': total / games if games > 0 else 0,
        'tp_pct': tp_pct,
    }


def main():
    print("="*70)
    print("FINAL STRATEGY OPTIMIZATION")
    print("="*70)

    print("\nGenerating 5000 games...")
    df = generate_games(5000)
    print(f"Generated {len(df):,} states")

    # Focused grid search
    print("\n" + "-"*60)
    print("GRID SEARCH - Focus on 90%+ WR with high coverage")
    print("-"*60)

    results = []
    for dev in [2.0, 2.5, 3.0, 3.5]:
        for mom in [4, 5, 6, 7, 8]:
            for tp in [0.25, 0.3, 0.35, 0.4, 0.5]:
                for sl in [4.0, 5.0, 6.0]:
                    for ts in [60, 90, 120]:
                        for cd in [15, 20, 30]:
                            res = backtest(df, dev, mom, tp, sl, ts, cd)
                            if res['trades'] < 100:
                                continue
                            results.append({
                                'dev': dev, 'mom': mom, 'tp': tp,
                                'sl': sl, 'ts': ts, 'cd': cd, **res
                            })

    results = pd.DataFrame(results)
    print(f"Tested {len(results)} valid configurations")

    # Analyze
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # 90%+ win rate
    wr90 = results[results['wr'] >= 0.90]
    print(f"\n90%+ win rate strategies: {len(wr90)}")

    if len(wr90) > 0:
        print("\nTop 10 with 90%+ WR (by coverage):")
        top = wr90.sort_values('cov', ascending=False).head(10)
        print(top[['dev', 'mom', 'tp', 'sl', 'ts', 'cd', 'wr', 'trades', 'cov', 'tpg', 'pnl']].to_string(index=False))

    # 85%+ win rate with high coverage
    wr85 = results[(results['wr'] >= 0.85) & (results['cov'] >= 0.9)]
    print(f"\n85%+ WR with 90%+ coverage: {len(wr85)}")

    if len(wr85) > 0:
        print("\nTop 10 balanced (85%+ WR, 90%+ coverage):")
        top85 = wr85.sort_values('wr', ascending=False).head(10)
        print(top85[['dev', 'mom', 'tp', 'sl', 'ts', 'cd', 'wr', 'trades', 'cov', 'tpg', 'pnl']].to_string(index=False))

    # Best overall (Sharpe-like metric: WR * sqrt(trades) * pnl_sign)
    results['score'] = results['wr'] * np.sqrt(results['trades']) * np.sign(results['pnl'] + 0.01)
    print("\nTop 10 by composite score:")
    top_score = results.sort_values('score', ascending=False).head(10)
    print(top_score[['dev', 'mom', 'tp', 'sl', 'ts', 'cd', 'wr', 'trades', 'cov', 'pnl']].to_string(index=False))

    # THE BEST STRATEGY
    best = wr90.sort_values('cov', ascending=False).iloc[0] if len(wr90) > 0 else \
           results.sort_values('wr', ascending=False).iloc[0]

    print(f"\n{'='*70}")
    print("FINAL OPTIMAL STRATEGY")
    print("="*70)
    print(f"""
ENTRY RULES:
  Spread Deviation >= {best['dev']} points from fair value
  OR Momentum >= {best['mom']} points in 2 minutes
  Time: 4-44 minutes remaining
  Score diff <= 20 points
  Cooldown: {best['cd']} seconds

EXIT RULES:
  Take Profit: {best['tp']} points
  Stop Loss: {best['sl']} points
  Time Stop: {best['ts']} seconds

PERFORMANCE:
  Win Rate: {best['wr']:.1%}
  Total Trades: {int(best['trades']):,}
  Games Traded: {best['cov']:.0%}
  Trades/Game: {best['tpg']:.1f}
  Total P&L: {best['pnl']:+.0f} points
  Take Profit %: {best['tp_pct']:.0%}
""")

    # Save
    results.to_csv('/home/user/nba2/output/final_optimization.csv', index=False)
    print(f"Saved {len(results)} results to output/final_optimization.csv")

    return results


if __name__ == "__main__":
    main()
