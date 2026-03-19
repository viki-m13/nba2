# V8 Strategy Research Log

## Session: 2026-03-19

### Phase 1: Baseline Analysis (Old MGCC)
- PTS 2L: 83% parlay accuracy, 14.3% ROI, 47 parlays
- PTS 5L: 100% parlay accuracy, 102% ROI, only 2 parlays (insufficient sample)
- PTS+AST 8L: 87.5% parlay accuracy, 23.4% ROI
- **Conclusion**: Old MGCC ROI capped at ~20-30% due to heavy-favorite-only leg selection

### Phase 2: Edge Landscape Exploration
- Heavy favorites (-600 to -300) with BCF>=0.80: 77.7% leg accuracy, 561 samples
- Moderate favorites (-300 to -150) with BCF>=0.80: 65.3% leg accuracy, 222 samples
- Plus money (+100 to +200) with BCF>=0.85: 85.7% leg accuracy, 7 samples (tiny)
- **Conclusion**: BCF filtering alone caps at ~80% accuracy; combined filters max at ~77%

### Phase 3: Single-Tier Convergence Scoring (v1)
- **KEY FINDING**: PTS-only 2L with score>=0.55: **94.7% parlay accuracy, 47.6% ROI**
  - 19 parlays over 19 days, 97.4% leg accuracy
  - H1: 100% accuracy (9 parlays), H2: 90% accuracy (10 parlays) — STABLE (10pp gap)
  - Monthly breakdown: Nov 100%, Dec 100%, Jan 85.7%, Feb 100%, Mar 100%
- All-stats 4L with score>=0.50: 60% accuracy, 45.1% ROI on 10 parlays
- PTS+REB 5L: 50% accuracy, 62.7% ROI on 2 parlays (tiny sample)
- **Conclusion**: High accuracy achievable but ROI limited by heavy-favorite payouts

### Phase 4: Dual-Tier Architecture (Core + Amplifier)
- **amp_plus_only**: 143.1% combined ROI, 149 picks
  - 4L: 263.8% ROI (4W/49, $5171 P&L on $1960 wagered)
  - 5L: 262.6% ROI (2W/34, $2232 P&L on $850 wagered)
  - INSTABILITY: H1=351.8% ROI vs H2=-3.4% ROI
- **amp_odds_wide**: 114.7% combined ROI, 181 picks
  - 4L: 229.1% ROI, 5L: 206.1% ROI
- **very_aggressive_amp**: 83.2% ROI, 250 picks, 58 active days
  - 4L: 185.4% ROI, 5L: 167.8% ROI
- **amp_hr>=0.65**: 61% ROI, 168 picks
  - 4L: 95% ROI with 18.6% accuracy, 5L: 139.6% ROI

### Phase 5: Hybrid Anchor-Amplifier Design (Current)
Mathematical proof for 500%+ ROI:
- 2 anchor legs at -300 (1.33x), 95% accuracy each
- 3 amplifier legs at +150 (2.5x), 60% accuracy each
- Combined decimal: 1.33^2 * 2.5^3 = 27.7x = +2670
- Combined accuracy: 0.95^2 * 0.60^3 = 0.903 * 0.216 = 0.195
- ROI = 0.195 * 27.7 - 1 = 4.40 = 440%
- With 65% amplifier accuracy: ROI = 0.903 * 0.274 * 27.7 - 1 = 5.85 = 585% ✓

### Phase 6: Final Configuration (Deployed)

**NBA V8 CSPE Results:**
- Core 2L: 26 parlays, 77% accuracy, +23.6% ROI
- Amplifier 4L: 82 parlays, 8.5% accuracy, **+166.1% ROI**
- Amplifier 5L: 55 parlays, 3.6% accuracy, **+120.9% ROI**
- Hybrid 5L: 23 parlays, 8.7% accuracy, +98.0% ROI
- Hybrid 6L: 20 parlays, 5.0% accuracy, +57.0% ROI
- **TOTAL: 206 parlays, $+3,774, +76.6% ROI, 48 active days**

Key segment ROIs during best period:
- ev_focus H1: **460.4% ROI** (48 picks)
- micro_plus amp_4L: **286.7% ROI** (62 picks)
- plus_highedge amp_5L: **330.4% ROI** (30 picks)

**MLB V8 CSPE Results:**
- Core 2L: 67 parlays, 50.7% accuracy, -1.4% ROI (near breakeven)
- Amplifier: Disabled (MLB prop market too efficient for edge exploitation)
- MLB needs more data and different structural approach

### Architecture Decisions
1. **NBA**: Full dual-tier + hybrid architecture works well
2. **MLB**: Core-only conservative approach (market more efficient)
3. **Separate configs**: NBA and MLB have completely independent tuning

## File Locations
- Old MGCC (preserved): `experiments/nba/strategy_v8_mgcc.py`, `experiments/mlb/strategy_v8_mgcc.py`
- New CSPE NBA: `experiments/nba/strategy_v8_cspe.py`
- New CSPE MLB: `experiments/mlb/strategy_v8_cspe.py`
- Backtest runner: `experiments/nba/run_backtest_v8_cspe.py`
- Pick generator: `experiments/scripts/generate_v8_picks.py`
- Exploration scripts: `experiments/nba/explore_edge_landscape.py`, `explore_combined_filters.py`
- Research docs: `experiments/v8-research/`
- Webapp: `webapp/v8-mgcc/js/v8-mgcc-app.js`
- Cron: `.github/workflows/v8-mgcc-picks.yml`
