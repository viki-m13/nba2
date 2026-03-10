#!/usr/bin/env node
// =============================================================================
// AutoResearch v3.0 — MLB Ultra Engine v3.0 Optimizer
// Inspired by karpathy/autoresearch monotonic ratchet + ECC instinct learning
//
// Target: 90%+ accuracy, 100%+ ROI, 50+ bets with parlays
// Key insight from NBA: parlays at 90%+ leg accuracy drive massive ROI
// =============================================================================

const fs = require('fs');
const path = require('path');

global.window = global;
global.document = { readyState: 'complete', addEventListener: () => {}, querySelectorAll: () => [] };
global.localStorage = { _data: {}, getItem(k) { return this._data[k] || null; }, setItem(k, v) { this._data[k] = v; } };

// Load v3.0 engine
eval(fs.readFileSync(path.join(__dirname, '../webapp/js/recommendation-engine-mlb-v3.js'), 'utf8'));

const ENGINE = global.window.MLBRecommendationEngineV3;
const DATA_DIR = path.join(__dirname, '../webapp/data');
const OUTPUT_DIR = path.join(__dirname, '../output');

if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const boxScores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_player_boxscores.json'), 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_historical_odds.json'), 'utf8'));

console.log('=== AutoResearch v3.0 — MLB Ultra Engine v3.0 Optimizer ===');
console.log(`Box scores: ${boxScores.length} games`);
console.log(`Historical odds: ${historicalOdds.length} records`);

// =============================================================================
// OBJECTIVE FUNCTION — Balances accuracy, ROI, volume, and parlay performance
// =============================================================================

function objective(results) {
  if (!results || !results.stats) return -Infinity;
  const s = results.stats.overall;
  const p = results.stats.parlays;
  const sg = results.stats.singles;

  // HARD FLOORS
  if (s.total < 8) return -Infinity;
  if (s.accuracy < 0.80) return s.accuracy * 20;
  if (s.roi < -0.20) return s.accuracy * 20 + s.roi * 10;

  let score = 0;

  // === PRIMARY: ROI * accuracy * sqrt(volume) ===
  // ROI is now the DOMINANT factor — Kelly sizing makes this the true measure of edge
  const volumeFactor = Math.sqrt(Math.min(s.total, 120));
  const roiFactor = Math.max(0.1, 1 + Math.min(s.roi, 10));
  score = s.accuracy * volumeFactor * roiFactor * 60;

  // === ACCURACY TIERS ===
  if (s.accuracy >= 0.97) score += 120;
  else if (s.accuracy >= 0.95) score += 90;
  else if (s.accuracy >= 0.93) score += 65;
  else if (s.accuracy >= 0.90) score += 45;
  else if (s.accuracy >= 0.87) score += 28;
  else if (s.accuracy >= 0.85) score += 15;

  // === PARLAY-FIRST STRATEGY ===
  // Parlays are THE path to 100%+ ROI. Singles at heavy-fav odds max out at ~50% ROI.
  // Parlay count and parlay ROI are the most important factors.
  const parlayRatio = s.total > 0 ? p.total / s.total : 0;

  if (p.total >= 10) {
    score += 350;  // Massive reward for 10+ parlays
    score += Math.min(p.total * 20, 200);
    if (p.accuracy >= 0.90) score += 150;
    else if (p.accuracy >= 0.80) score += 90;
    else if (p.accuracy >= 0.70) score += 45;
    if (p.legAccuracy >= 0.95) score += 100;
    else if (p.legAccuracy >= 0.90) score += 65;
    else if (p.legAccuracy >= 0.85) score += 35;
    if (p.roi > 3.0) score += 200;
    else if (p.roi > 2.0) score += 150;
    else if (p.roi > 1.0) score += 100;
    else if (p.roi > 0.5) score += 60;
    else if (p.roi > 0) score += 30;
  } else if (p.total >= 5) {
    score += 220;
    score += Math.min(p.total * 18, 120);
    if (p.accuracy >= 0.90) score += 100;
    else if (p.accuracy >= 0.80) score += 60;
    else if (p.accuracy >= 0.70) score += 30;
    if (p.legAccuracy >= 0.95) score += 80;
    else if (p.legAccuracy >= 0.90) score += 50;
    if (p.roi > 2.0) score += 130;
    else if (p.roi > 1.0) score += 90;
    else if (p.roi > 0.5) score += 50;
    else if (p.roi > 0) score += 25;
  } else if (p.total >= 3) {
    score += 120;
    score += p.total * 12;
    if (p.accuracy >= 0.80) score += 40;
    if (p.roi > 0) score += 30;
  } else if (p.total >= 1) {
    score += 40;
  } else {
    score -= 200;  // Even heavier penalty for no parlays
  }

  // Parlay ratio bonus — reward higher proportion of parlays
  if (parlayRatio >= 0.40) score += 80;
  else if (parlayRatio >= 0.30) score += 50;
  else if (parlayRatio >= 0.20) score += 30;
  else if (parlayRatio >= 0.15) score += 15;

  // === VOLUME ===
  if (s.total >= 60 && s.total <= 120) score += 60;
  else if (s.total >= 40) score += 45;
  else if (s.total >= 25) score += 30;
  else if (s.total >= 15) score += 10;
  else score -= 30;

  if (s.total > 200) score -= 40;
  if (s.total < 12) score -= 50;

  // === ROI — THE KEY METRIC ===
  if (s.roi > 3.0) score += 200;
  else if (s.roi > 2.0) score += 150;
  else if (s.roi > 1.0) score += 120;
  else if (s.roi > 0.7) score += 80;
  else if (s.roi > 0.5) score += 50;
  else if (s.roi > 0.3) score += 30;
  else if (s.roi > 0.1) score += 15;
  else if (s.roi > 0) score += 5;
  else score += Math.max(-80, s.roi * 100);

  // === PnL absolute ===
  if (s.pnl > 10000) score += 80;
  else if (s.pnl > 5000) score += 50;
  else if (s.pnl > 3000) score += 35;
  else if (s.pnl > 2000) score += 25;
  else if (s.pnl > 1000) score += 15;

  // === Singles accuracy ===
  if (sg.total > 0) {
    if (sg.accuracy >= 0.95) score += 25;
    else if (sg.accuracy >= 0.92) score += 15;
    else if (sg.accuracy >= 0.90) score += 8;
  }

  return score;
}

// =============================================================================
// RUNNER
// =============================================================================

function runWithConfig(configOverrides) {
  const origConfig = {};
  for (const k of Object.keys(configOverrides)) {
    if (ENGINE.CONFIG.hasOwnProperty(k)) {
      origConfig[k] = ENGINE.CONFIG[k];
      ENGINE.CONFIG[k] = configOverrides[k];
    }
  }
  let results;
  try {
    results = ENGINE.runBacktest(boxScores, historicalOdds);
  } catch (e) {
    results = null;
  }
  for (const k of Object.keys(origConfig)) {
    ENGINE.CONFIG[k] = origConfig[k];
  }
  return results;
}

// =============================================================================
// MONOTONIC RATCHET OPTIMIZATION
// =============================================================================

function ratchetOptimize(startConfig, maxRounds) {
  let bestConfig = { ...startConfig };
  let bestResults = runWithConfig(bestConfig);
  let bestScore = objective(bestResults);
  let totalImprovements = 0;

  const r = bestResults ? bestResults.stats.overall : { total: 0, accuracy: 0, pnl: 0, roi: 0 };
  const p = bestResults ? bestResults.stats.parlays : { total: 0 };
  console.log(`Start: ${r.total} sigs (${p.total}P), ${(r.accuracy * 100).toFixed(1)}% acc, $${r.pnl}, ${(r.roi * 100).toFixed(1)}% ROI | score=${bestScore.toFixed(1)}`);

  // Parameter sweep ranges
  const sweeps = {
    // Gate thresholds (most important)
    GATE_MIN_HIT_RATE: [0.78, 0.80, 0.82, 0.85, 0.87, 0.88, 0.90, 0.92, 0.93, 0.95, 0.97, 1.0],
    GATE_MIN_COMBINED: [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85],
    GATE_MIN_GFT_SCORE: [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    GATE_MIN_BEQ_EDGE: [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.17, 0.20, 0.22, 0.25],
    GATE_MIN_STREAK: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    GATE_MIN_ABVC: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    GATE_MIN_PBF_MARGIN: [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10],
    GATE_MIN_ESI_STABILITY: [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    GATE_MIN_IMAD_SCORE: [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10],

    // Odds range (critical for quality)
    MIN_ODDS: [-550, -500, -450, -400, -350, -300, -280, -260, -250, -240, -220, -200],
    MAX_ODDS: [-100, -110, -120, -130, -140, -150, -160, -170, -180, -190, -200],

    // BEQ
    BEQ_CREDIBLE_LEVEL: [0.75, 0.80, 0.83, 0.85, 0.87, 0.90, 0.92, 0.95],
    BEQ_MIN_EDGE: [0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20],

    // GFT
    GFT_DECAY_RATE: [0.82, 0.85, 0.87, 0.88, 0.90, 0.91, 0.92, 0.93, 0.95, 0.97, 0.98],
    GFT_GRAVITY_STRENGTH: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    GFT_MIN_CLEARANCE: [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0],
    GFT_CONVERGENCE_MAX_SPREAD: [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    GFT_WINDOWS: [[3, 5, 8], [3, 5, 10], [3, 7, 12], [3, 7, 15], [5, 8, 12], [5, 10, 15]],

    // PBF
    PBF_DECAY_RATE: [0.85, 0.88, 0.90, 0.92, 0.94, 0.96],
    PBF_MIN_LAMBDA: [0.2, 0.3, 0.4, 0.5, 0.6],
    PBF_WEIGHT: [0.05, 0.08, 0.10, 0.12, 0.15],

    // CGSM
    CGSM_MIN_STREAK: [0, 1, 2, 3, 4],
    CGSM_LOOKBACK: [5, 8, 10, 12, 15],
    CGSM_WEIGHT: [0.03, 0.05, 0.06, 0.08, 0.10],

    // ABVC
    ABVC_MIN_AVG_AB: [2.0, 2.5, 3.0, 3.5, 4.0],
    ABVC_ELITE_AB: [4.0, 4.5, 5.0, 5.5, 6.0],
    ABVC_WEIGHT: [0.03, 0.05, 0.07, 0.10],

    // ESI
    ESI_MAX_ENTROPY: [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.83, 0.85, 0.90, 1.0],
    ESI_TREND_WEIGHT: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    ESI_BINS: [4, 5, 6, 7, 8],

    // Bet type thresholds
    SINGLE_MIN_SCORE: [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90],
    MULTI_SINGLE_MIN_SCORE: [0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75],
    PARLAY_LEG_MIN_SCORE: [0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70],

    // Parlay config
    PARLAY_MAX_CORRELATION: [0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    PARLAY_SAME_GAME_ALLOWED: [true, false],
    PARLAY_MIN_COMBINED_EDGE: [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15],
    PARLAY_MAX_LEGS: [2, 3, 4],
    PARLAY_MAX_PER_DAY: [1, 2, 3, 4],

    // Eligibility
    MIN_GAMES: [5, 6, 7, 8, 9, 10, 12, 14, 15],
    MIN_AB: [1, 2, 3, 4],
    WARM_UP_GAMES: [6, 7, 8, 10, 12, 14, 15, 18],

    // v3.0 new features
    SWTA_ENABLED: [true, false],
    SWTA_WEIGHT: [0.03, 0.05, 0.06, 0.08, 0.10],
    SWTA_SHORT_WEIGHT: [0.40, 0.50, 0.60],
    SWTA_MED_WEIGHT: [0.20, 0.30, 0.35],
    SWTA_SHORT_DECAY: [0.90, 0.93, 0.95, 0.97],
    SWTA_MED_DECAY: [0.85, 0.88, 0.90, 0.92],
    SWTA_LONG_DECAY: [0.80, 0.83, 0.85, 0.88],

    ASC_ENABLED: [true, false],
    ASC_ARBITER_MIN_CONSENSUS: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    ASC_WEIGHT: [0.02, 0.04, 0.06, 0.08],

    DMCT_ENABLED: [true, false],
    DMCT_WEIGHT: [0.03, 0.05, 0.07, 0.10],
    DMCT_MIN_PROB: [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    DMCT_ORDER: [1, 2, 3],

    TRD_ENABLED: [true, false],
    TRD_HOT_THRESHOLD: [1.08, 1.10, 1.12, 1.15, 1.20, 1.25],
    TRD_COLD_THRESHOLD: [0.75, 0.80, 0.85, 0.88, 0.90],
    TRD_BOOST: [0.01, 0.02, 0.03, 0.04, 0.05],
    TRD_PENALTY: [0.02, 0.03, 0.05, 0.07, 0.10],

    PSI_ENABLED: [true, false],
    PSI_WEIGHT: [0.02, 0.03, 0.05, 0.07],

    // Optional enhancers
    OAL_ENABLED: [true, false],
    OAL_WEIGHT: [0.02, 0.03, 0.05, 0.07],
    OAL_MAX_ADJUSTMENT: [0.10, 0.15, 0.20, 0.25, 0.30],

    VPD_ENABLED: [true, false],
    VPD_MIN_GAMES: [3, 5, 7, 10],
    VPD_WEIGHT: [0.03, 0.04, 0.05, 0.07],
    VPD_MAX_BOOST: [0.05, 0.08, 0.10, 0.12, 0.15],

    HOME_BOOST: [0.0, 0.01, 0.015, 0.02, 0.025, 0.03],
    CAMPD_MAX_SAME_STAT: [1, 2, 3, 4, 5],
    CAMPD_CROSS_STAT_BONUS: [0.0, 0.01, 0.02, 0.03],

    // Kelly-Criterion Proportional Sizing
    KELLY_SIZING: [true, false],
    KELLY_MIN_UNITS: [0.3, 0.5, 0.7, 1.0],
    KELLY_MAX_UNITS: [2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
    KELLY_SIZING_MULTIPLIER: [1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
    PARLAY_UNIT_BONUS: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
  };

  // Optimization order: most impactful parameters first
  const paramOrder = [
    // Gate thresholds first (biggest impact on accuracy/volume)
    'GATE_MIN_HIT_RATE', 'GATE_MIN_COMBINED', 'GATE_MIN_GFT_SCORE',
    'GATE_MIN_BEQ_EDGE', 'GATE_MIN_PBF_MARGIN', 'GATE_MIN_STREAK',
    'GATE_MIN_ABVC', 'GATE_MIN_ESI_STABILITY', 'GATE_MIN_IMAD_SCORE',
    // Odds range (volume/quality)
    'MIN_ODDS', 'MAX_ODDS',
    // Bet type and parlay (ROI)
    'SINGLE_MIN_SCORE', 'MULTI_SINGLE_MIN_SCORE', 'PARLAY_LEG_MIN_SCORE',
    'PARLAY_MAX_CORRELATION', 'PARLAY_SAME_GAME_ALLOWED',
    'PARLAY_MIN_COMBINED_EDGE', 'PARLAY_MAX_LEGS', 'PARLAY_MAX_PER_DAY',
    // Core signal params
    'BEQ_CREDIBLE_LEVEL', 'BEQ_MIN_EDGE',
    'GFT_DECAY_RATE', 'GFT_GRAVITY_STRENGTH', 'GFT_MIN_CLEARANCE',
    'GFT_CONVERGENCE_MAX_SPREAD', 'GFT_WINDOWS',
    'PBF_DECAY_RATE', 'PBF_MIN_LAMBDA', 'PBF_WEIGHT',
    'CGSM_MIN_STREAK', 'CGSM_LOOKBACK', 'CGSM_WEIGHT',
    'ABVC_MIN_AVG_AB', 'ABVC_ELITE_AB', 'ABVC_WEIGHT',
    'ESI_MAX_ENTROPY', 'ESI_TREND_WEIGHT', 'ESI_BINS',
    // Eligibility
    'MIN_GAMES', 'MIN_AB', 'WARM_UP_GAMES',
    // v3.0 new features
    'SWTA_ENABLED', 'SWTA_WEIGHT', 'SWTA_SHORT_WEIGHT', 'SWTA_MED_WEIGHT',
    'SWTA_SHORT_DECAY', 'SWTA_MED_DECAY', 'SWTA_LONG_DECAY',
    'ASC_ENABLED', 'ASC_ARBITER_MIN_CONSENSUS', 'ASC_WEIGHT',
    'DMCT_ENABLED', 'DMCT_WEIGHT', 'DMCT_MIN_PROB', 'DMCT_ORDER',
    'TRD_ENABLED', 'TRD_HOT_THRESHOLD', 'TRD_COLD_THRESHOLD', 'TRD_BOOST', 'TRD_PENALTY',
    'PSI_ENABLED', 'PSI_WEIGHT',
    'OAL_ENABLED', 'OAL_WEIGHT', 'OAL_MAX_ADJUSTMENT',
    'VPD_ENABLED', 'VPD_MIN_GAMES', 'VPD_WEIGHT', 'VPD_MAX_BOOST',
    'HOME_BOOST', 'CAMPD_MAX_SAME_STAT', 'CAMPD_CROSS_STAT_BONUS',
    // Kelly-Criterion sizing (ROI amplifier)
    'KELLY_SIZING', 'KELLY_SIZING_MULTIPLIER', 'KELLY_MAX_UNITS',
    'KELLY_MIN_UNITS', 'PARLAY_UNIT_BONUS',
  ];

  for (let round = 0; round < maxRounds; round++) {
    let roundImprovements = 0;
    console.log(`\n--- Round ${round + 1}/${maxRounds} ---`);

    for (const param of paramOrder) {
      if (!sweeps[param]) continue;

      for (const val of sweeps[param]) {
        if (JSON.stringify(val) === JSON.stringify(bestConfig[param])) continue;

        const testConfig = { ...bestConfig, [param]: val };
        const results = runWithConfig(testConfig);
        const score = objective(results);

        if (score > bestScore) {
          bestConfig[param] = val;
          bestScore = score;
          bestResults = results;
          totalImprovements++;
          roundImprovements++;

          const rs = results.stats.overall;
          const ps = results.stats.parlays;
          console.log(`  [+] ${param}: ${JSON.stringify(val)} | ${rs.total} sigs (${ps.total}P), ${(rs.accuracy * 100).toFixed(1)}% acc, $${rs.pnl}, ${(rs.roi * 100).toFixed(1)}% ROI | score=${score.toFixed(1)}`);
        }
      }
    }

    console.log(`Round ${round + 1} complete: ${roundImprovements} improvements`);
    if (roundImprovements === 0) {
      console.log('Converged — no further improvements found');
      break;
    }
  }

  return { config: bestConfig, results: bestResults, score: bestScore, improvements: totalImprovements };
}

// =============================================================================
// STARTING CONFIGURATIONS
// =============================================================================

const starts = [
  // Start A: NBA-calibrated balanced (looser gates)
  {
    MIN_GAMES: 8, MIN_AB: 2, WARM_UP_GAMES: 10,
    GFT_WINDOWS: [3, 5, 10], GFT_DECAY_RATE: 0.91, GFT_GRAVITY_STRENGTH: 0.36,
    GFT_MIN_CLEARANCE: 0.05, GFT_CONVERGENCE_MAX_SPREAD: 4.0,
    BEQ_CREDIBLE_LEVEL: 0.87, BEQ_MIN_EDGE: 0.03,
    ESI_BINS: 6, ESI_MAX_ENTROPY: 0.83, ESI_TREND_WEIGHT: 0.30,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 1, CGSM_LOOKBACK: 8, CGSM_WEIGHT: 0.06,
    ABVC_MIN_AVG_AB: 2.5, ABVC_ELITE_AB: 5.0, ABVC_WEIGHT: 0.05,
    PBF_DECAY_RATE: 0.92, PBF_MIN_LAMBDA: 0.3, PBF_WEIGHT: 0.10,
    PBF_MIN_PROB_MARGIN: 0.03,
    OAL_ENABLED: true, OAL_WEIGHT: 0.03, OAL_MAX_ADJUSTMENT: 0.20,
    VPD_ENABLED: true, VPD_MIN_GAMES: 3, VPD_WEIGHT: 0.04, VPD_MAX_BOOST: 0.12,
    SWTA_ENABLED: true, SWTA_WEIGHT: 0.06,
    SWTA_SHORT_WEIGHT: 0.50, SWTA_MED_WEIGHT: 0.30,
    SWTA_SHORT_DECAY: 0.95, SWTA_MED_DECAY: 0.90, SWTA_LONG_DECAY: 0.85,
    ASC_ENABLED: true, ASC_ARBITER_MIN_CONSENSUS: 0.65, ASC_WEIGHT: 0.04,
    DMCT_ENABLED: true, DMCT_WEIGHT: 0.05, DMCT_MIN_PROB: 0.60, DMCT_ORDER: 2,
    TRD_ENABLED: true, TRD_HOT_THRESHOLD: 1.15, TRD_COLD_THRESHOLD: 0.85,
    TRD_BOOST: 0.03, TRD_PENALTY: 0.05,
    PSI_ENABLED: true, PSI_WEIGHT: 0.03,
    GATE_MIN_GFT_SCORE: 0.35, GATE_MIN_BEQ_EDGE: 0.04,
    GATE_MIN_ESI_STABILITY: 0.12, GATE_MIN_IMAD_SCORE: 0.015,
    GATE_MIN_HIT_RATE: 0.78, GATE_MIN_COMBINED: 0.52,
    GATE_MIN_STREAK: 1, GATE_MIN_ABVC: 0.25,
    GATE_MIN_PBF_MARGIN: 0.02,
    SINGLE_MIN_SCORE: 0.70, MULTI_SINGLE_MIN_SCORE: 0.60,
    PARLAY_LEG_MIN_SCORE: 0.52,
    PARLAY_MAX_CORRELATION: 0.35, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.05, PARLAY_MAX_PER_DAY: 2, PARLAY_MAX_LEGS: 4,
    MIN_ODDS: -500, MAX_ODDS: -110,
    HOME_BOOST: 0.02,
    CAMPD_MAX_SAME_STAT: 2, CAMPD_CROSS_STAT_BONUS: 0.02,
    KELLY_SIZING: true, KELLY_MIN_UNITS: 0.5, KELLY_MAX_UNITS: 5.0,
    KELLY_SIZING_MULTIPLIER: 3.0, PARLAY_UNIT_BONUS: 1.5,
  },

  // Start B: Tight gates (high accuracy target, v2.0-like)
  {
    MIN_GAMES: 8, MIN_AB: 2, WARM_UP_GAMES: 10,
    GFT_WINDOWS: [3, 5, 10], GFT_DECAY_RATE: 0.90, GFT_GRAVITY_STRENGTH: 0.40,
    GFT_MIN_CLEARANCE: 0.10, GFT_CONVERGENCE_MAX_SPREAD: 3.0,
    BEQ_CREDIBLE_LEVEL: 0.87, BEQ_MIN_EDGE: 0.05,
    ESI_BINS: 5, ESI_MAX_ENTROPY: 0.85, ESI_TREND_WEIGHT: 0.25,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 2, CGSM_LOOKBACK: 10, CGSM_WEIGHT: 0.08,
    ABVC_MIN_AVG_AB: 3.0, ABVC_ELITE_AB: 5.0, ABVC_WEIGHT: 0.07,
    PBF_DECAY_RATE: 0.92, PBF_MIN_LAMBDA: 0.3, PBF_WEIGHT: 0.10,
    PBF_MIN_PROB_MARGIN: 0.05,
    OAL_ENABLED: true, OAL_WEIGHT: 0.05, OAL_MAX_ADJUSTMENT: 0.25,
    VPD_ENABLED: true, VPD_MIN_GAMES: 5, VPD_WEIGHT: 0.05, VPD_MAX_BOOST: 0.10,
    SWTA_ENABLED: true, SWTA_WEIGHT: 0.06,
    SWTA_SHORT_WEIGHT: 0.50, SWTA_MED_WEIGHT: 0.30,
    SWTA_SHORT_DECAY: 0.95, SWTA_MED_DECAY: 0.90, SWTA_LONG_DECAY: 0.85,
    ASC_ENABLED: true, ASC_ARBITER_MIN_CONSENSUS: 0.65, ASC_WEIGHT: 0.04,
    DMCT_ENABLED: true, DMCT_WEIGHT: 0.05, DMCT_MIN_PROB: 0.60, DMCT_ORDER: 2,
    TRD_ENABLED: true, TRD_HOT_THRESHOLD: 1.15, TRD_COLD_THRESHOLD: 0.85,
    TRD_BOOST: 0.03, TRD_PENALTY: 0.05,
    PSI_ENABLED: true, PSI_WEIGHT: 0.03,
    GATE_MIN_GFT_SCORE: 0.60, GATE_MIN_BEQ_EDGE: 0.15,
    GATE_MIN_ESI_STABILITY: 0.30, GATE_MIN_IMAD_SCORE: 0.05,
    GATE_MIN_HIT_RATE: 0.90, GATE_MIN_COMBINED: 0.68,
    GATE_MIN_STREAK: 3, GATE_MIN_ABVC: 0.35,
    GATE_MIN_PBF_MARGIN: 0.05,
    SINGLE_MIN_SCORE: 0.75, MULTI_SINGLE_MIN_SCORE: 0.65,
    PARLAY_LEG_MIN_SCORE: 0.60,
    PARLAY_MAX_CORRELATION: 0.30, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.10, PARLAY_MAX_PER_DAY: 2, PARLAY_MAX_LEGS: 3,
    MIN_ODDS: -350, MAX_ODDS: -150,
    HOME_BOOST: 0.02,
    CAMPD_MAX_SAME_STAT: 2, CAMPD_CROSS_STAT_BONUS: 0.02,
    KELLY_SIZING: true, KELLY_MIN_UNITS: 0.5, KELLY_MAX_UNITS: 5.0,
    KELLY_SIZING_MULTIPLIER: 3.0, PARLAY_UNIT_BONUS: 1.5,
  },

  // Start C: Heavy favorites focus (exploit high-odds accuracy)
  {
    MIN_GAMES: 8, MIN_AB: 2, WARM_UP_GAMES: 10,
    GFT_WINDOWS: [3, 5, 10], GFT_DECAY_RATE: 0.92, GFT_GRAVITY_STRENGTH: 0.35,
    GFT_MIN_CLEARANCE: 0.08, GFT_CONVERGENCE_MAX_SPREAD: 3.0,
    BEQ_CREDIBLE_LEVEL: 0.90, BEQ_MIN_EDGE: 0.05,
    ESI_BINS: 6, ESI_MAX_ENTROPY: 0.80, ESI_TREND_WEIGHT: 0.30,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 2, CGSM_LOOKBACK: 8, CGSM_WEIGHT: 0.06,
    ABVC_MIN_AVG_AB: 3.0, ABVC_ELITE_AB: 5.0, ABVC_WEIGHT: 0.05,
    PBF_DECAY_RATE: 0.92, PBF_MIN_LAMBDA: 0.3, PBF_WEIGHT: 0.10,
    PBF_MIN_PROB_MARGIN: 0.04,
    OAL_ENABLED: true, OAL_WEIGHT: 0.03, OAL_MAX_ADJUSTMENT: 0.20,
    VPD_ENABLED: true, VPD_MIN_GAMES: 5, VPD_WEIGHT: 0.04, VPD_MAX_BOOST: 0.10,
    SWTA_ENABLED: true, SWTA_WEIGHT: 0.06,
    SWTA_SHORT_WEIGHT: 0.50, SWTA_MED_WEIGHT: 0.30,
    SWTA_SHORT_DECAY: 0.95, SWTA_MED_DECAY: 0.90, SWTA_LONG_DECAY: 0.85,
    ASC_ENABLED: true, ASC_ARBITER_MIN_CONSENSUS: 0.70, ASC_WEIGHT: 0.04,
    DMCT_ENABLED: true, DMCT_WEIGHT: 0.05, DMCT_MIN_PROB: 0.60, DMCT_ORDER: 2,
    TRD_ENABLED: true, TRD_HOT_THRESHOLD: 1.15, TRD_COLD_THRESHOLD: 0.85,
    TRD_BOOST: 0.03, TRD_PENALTY: 0.05,
    PSI_ENABLED: true, PSI_WEIGHT: 0.03,
    GATE_MIN_GFT_SCORE: 0.50, GATE_MIN_BEQ_EDGE: 0.10,
    GATE_MIN_ESI_STABILITY: 0.20, GATE_MIN_IMAD_SCORE: 0.03,
    GATE_MIN_HIT_RATE: 0.87, GATE_MIN_COMBINED: 0.60,
    GATE_MIN_STREAK: 2, GATE_MIN_ABVC: 0.30,
    GATE_MIN_PBF_MARGIN: 0.03,
    SINGLE_MIN_SCORE: 0.72, MULTI_SINGLE_MIN_SCORE: 0.62,
    PARLAY_LEG_MIN_SCORE: 0.55,
    PARLAY_MAX_CORRELATION: 0.35, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.07, PARLAY_MAX_PER_DAY: 2, PARLAY_MAX_LEGS: 4,
    MIN_ODDS: -400, MAX_ODDS: -180,
    HOME_BOOST: 0.02,
    CAMPD_MAX_SAME_STAT: 2, CAMPD_CROSS_STAT_BONUS: 0.02,
    KELLY_SIZING: true, KELLY_MIN_UNITS: 0.5, KELLY_MAX_UNITS: 5.0,
    KELLY_SIZING_MULTIPLIER: 3.0, PARLAY_UNIT_BONUS: 1.5,
  },

  // Start D: Parlay-Heavy ROI Maximizer with Kelly Sizing
  // Focuses on generating maximum parlays with Kelly-proportional sizing
  {
    MIN_GAMES: 8, MIN_AB: 2, WARM_UP_GAMES: 10,
    GFT_WINDOWS: [3, 5, 10], GFT_DECAY_RATE: 0.91, GFT_GRAVITY_STRENGTH: 0.36,
    GFT_MIN_CLEARANCE: 0.05, GFT_CONVERGENCE_MAX_SPREAD: 4.0,
    BEQ_CREDIBLE_LEVEL: 0.87, BEQ_MIN_EDGE: 0.03,
    ESI_BINS: 6, ESI_MAX_ENTROPY: 0.83, ESI_TREND_WEIGHT: 0.30,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 1, CGSM_LOOKBACK: 8, CGSM_WEIGHT: 0.06,
    ABVC_MIN_AVG_AB: 2.5, ABVC_ELITE_AB: 4.0, ABVC_WEIGHT: 0.05,
    PBF_DECAY_RATE: 0.94, PBF_MIN_LAMBDA: 0.3, PBF_WEIGHT: 0.10,
    PBF_MIN_PROB_MARGIN: 0.03,
    OAL_ENABLED: true, OAL_WEIGHT: 0.03, OAL_MAX_ADJUSTMENT: 0.20,
    VPD_ENABLED: false, VPD_MIN_GAMES: 3, VPD_WEIGHT: 0.04, VPD_MAX_BOOST: 0.12,
    SWTA_ENABLED: true, SWTA_WEIGHT: 0.06,
    SWTA_SHORT_WEIGHT: 0.50, SWTA_MED_WEIGHT: 0.30,
    SWTA_SHORT_DECAY: 0.95, SWTA_MED_DECAY: 0.90, SWTA_LONG_DECAY: 0.85,
    ASC_ENABLED: true, ASC_ARBITER_MIN_CONSENSUS: 0.65, ASC_WEIGHT: 0.04,
    DMCT_ENABLED: true, DMCT_WEIGHT: 0.05, DMCT_MIN_PROB: 0.60, DMCT_ORDER: 2,
    TRD_ENABLED: true, TRD_HOT_THRESHOLD: 1.08, TRD_COLD_THRESHOLD: 0.85,
    TRD_BOOST: 0.05, TRD_PENALTY: 0.05,
    PSI_ENABLED: true, PSI_WEIGHT: 0.03,
    // Looser gates to generate more parlay candidates
    GATE_MIN_GFT_SCORE: 0.30, GATE_MIN_BEQ_EDGE: 0.03,
    GATE_MIN_ESI_STABILITY: 0.10, GATE_MIN_IMAD_SCORE: 0.01,
    GATE_MIN_HIT_RATE: 0.85, GATE_MIN_COMBINED: 0.55,
    GATE_MIN_STREAK: 1, GATE_MIN_ABVC: 0.20,
    GATE_MIN_PBF_MARGIN: 0.02,
    // Higher single threshold, lower parlay threshold = more parlays, fewer singles
    SINGLE_MIN_SCORE: 0.82, MULTI_SINGLE_MIN_SCORE: 0.70,
    PARLAY_LEG_MIN_SCORE: 0.48,
    PARLAY_MAX_CORRELATION: 0.40, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.03, PARLAY_MAX_PER_DAY: 4, PARLAY_MAX_LEGS: 4,
    MIN_ODDS: -260, MAX_ODDS: -150,
    HOME_BOOST: 0.01,
    CAMPD_MAX_SAME_STAT: 3, CAMPD_CROSS_STAT_BONUS: 0.02,
    // Kelly sizing for ROI amplification
    KELLY_SIZING: true, KELLY_MIN_UNITS: 0.5, KELLY_MAX_UNITS: 5.0,
    KELLY_SIZING_MULTIPLIER: 5.0, PARLAY_UNIT_BONUS: 2.5,
  },

  // Start E: Previous best config as starting point (warm start)
  {
    MIN_GAMES: 8, MIN_AB: 4, WARM_UP_GAMES: 10,
    GFT_WINDOWS: [3, 5, 10], GFT_DECAY_RATE: 0.91, GFT_GRAVITY_STRENGTH: 0.36,
    GFT_MIN_CLEARANCE: 0.05, GFT_CONVERGENCE_MAX_SPREAD: 4.0,
    BEQ_CREDIBLE_LEVEL: 0.87, BEQ_MIN_EDGE: 0.03,
    ESI_BINS: 6, ESI_MAX_ENTROPY: 0.83, ESI_TREND_WEIGHT: 0.30,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 1, CGSM_LOOKBACK: 8, CGSM_WEIGHT: 0.06,
    ABVC_MIN_AVG_AB: 2.5, ABVC_ELITE_AB: 4.0, ABVC_WEIGHT: 0.05,
    PBF_DECAY_RATE: 0.94, PBF_MIN_LAMBDA: 0.3, PBF_WEIGHT: 0.10,
    PBF_MIN_PROB_MARGIN: 0.03,
    OAL_ENABLED: true, OAL_WEIGHT: 0.03, OAL_MAX_ADJUSTMENT: 0.20,
    VPD_ENABLED: false, VPD_MIN_GAMES: 3, VPD_WEIGHT: 0.04, VPD_MAX_BOOST: 0.12,
    SWTA_ENABLED: true, SWTA_WEIGHT: 0.06,
    SWTA_SHORT_WEIGHT: 0.50, SWTA_MED_WEIGHT: 0.30,
    SWTA_SHORT_DECAY: 0.95, SWTA_MED_DECAY: 0.90, SWTA_LONG_DECAY: 0.85,
    ASC_ENABLED: true, ASC_ARBITER_MIN_CONSENSUS: 0.65, ASC_WEIGHT: 0.04,
    DMCT_ENABLED: true, DMCT_WEIGHT: 0.05, DMCT_MIN_PROB: 0.60, DMCT_ORDER: 2,
    TRD_ENABLED: true, TRD_HOT_THRESHOLD: 1.08, TRD_COLD_THRESHOLD: 0.85,
    TRD_BOOST: 0.05, TRD_PENALTY: 0.05,
    PSI_ENABLED: true, PSI_WEIGHT: 0.03,
    GATE_MIN_GFT_SCORE: 0.35, GATE_MIN_BEQ_EDGE: 0.04,
    GATE_MIN_ESI_STABILITY: 0.50, GATE_MIN_IMAD_SCORE: 0.015,
    GATE_MIN_HIT_RATE: 1.0, GATE_MIN_COMBINED: 0.80,
    GATE_MIN_STREAK: 1, GATE_MIN_ABVC: 0.25,
    GATE_MIN_PBF_MARGIN: 0.02,
    SINGLE_MIN_SCORE: 0.70, MULTI_SINGLE_MIN_SCORE: 0.60,
    PARLAY_LEG_MIN_SCORE: 0.52,
    PARLAY_MAX_CORRELATION: 0.35, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.05, PARLAY_MAX_PER_DAY: 2, PARLAY_MAX_LEGS: 4,
    MIN_ODDS: -260, MAX_ODDS: -150,
    HOME_BOOST: 0.01,
    CAMPD_MAX_SAME_STAT: 2, CAMPD_CROSS_STAT_BONUS: 0.02,
    KELLY_SIZING: true, KELLY_MIN_UNITS: 0.5, KELLY_MAX_UNITS: 5.0,
    KELLY_SIZING_MULTIPLIER: 3.0, PARLAY_UNIT_BONUS: 1.5,
  },
];

// =============================================================================
// RUN OPTIMIZATION
// =============================================================================

let overallBest = null;

for (let i = 0; i < starts.length; i++) {
  console.log(`\n${'#'.repeat(60)}`);
  console.log(`START POINT ${i + 1}/${starts.length}`);
  console.log(`${'#'.repeat(60)}`);

  const result = ratchetOptimize(starts[i], 8);

  if (!overallBest || result.score > overallBest.score) {
    overallBest = result;
    console.log(`*** New overall best: score=${result.score.toFixed(1)} ***`);
  }
}

const best = overallBest;
const f = best.results.stats;

console.log('\n' + '='.repeat(60));
console.log('FINAL RESULTS — MLB Ultra Engine v3.0');
console.log('='.repeat(60));
console.log(`Singles: ${f.singles.total} (${(f.singles.accuracy * 100).toFixed(1)}% acc, $${f.singles.pnl}, ${(f.singles.roi * 100).toFixed(1)}% ROI)`);
console.log(`Parlays: ${f.parlays.total} (${(f.parlays.accuracy * 100).toFixed(1)}% acc, $${f.parlays.pnl}, ${(f.parlays.roi * 100).toFixed(1)}% ROI)`);
if (f.parlays.totalLegs > 0) console.log(`  Legs: ${f.parlays.hitLegs}/${f.parlays.totalLegs} (${(f.parlays.legAccuracy * 100).toFixed(1)}%)`);
console.log(`Overall: ${f.overall.total} (${(f.overall.accuracy * 100).toFixed(1)}% acc, $${f.overall.pnl}, ${(f.overall.roi * 100).toFixed(1)}% ROI)`);
console.log(`Improvements: ${best.improvements}`);

// =============================================================================
// SAVE RESULTS
// =============================================================================

const configOutput = {
  config: best.config,
  score: best.score,
  improvements: best.improvements,
  results: {
    total: f.overall.total, accuracy: f.overall.accuracy,
    roi: f.overall.roi, pnl: f.overall.pnl,
    singles: f.singles, parlays: f.parlays,
  },
  optimized_at: new Date().toISOString(),
  version: '3.0',
};

fs.writeFileSync(path.join(OUTPUT_DIR, 'mlb_ultra_engine_v3_config.json'), JSON.stringify(configOutput, null, 2));
fs.writeFileSync(path.join(DATA_DIR, 'mlb_ultra_signals_v3.json'), JSON.stringify(best.results.signals, null, 2));
fs.writeFileSync(path.join(DATA_DIR, 'mlb_ultra_backtest_stats_v3.json'), JSON.stringify(best.results.stats, null, 2));

console.log('\nConfig saved to', path.join(OUTPUT_DIR, 'mlb_ultra_engine_v3_config.json'));

// Signal details
console.log('\n=== SIGNAL DETAILS ===');
for (const sig of best.results.signals) {
  if (sig.betType === 'single') {
    console.log(`  ${sig.date} ${sig.hit ? 'W' : 'L'} | ${sig.player} O${sig.line} ${sig.statLabel || sig.statType} | ${sig.odds} | actual: ${sig.actual}`);
  } else {
    console.log(`  ${sig.date} ${sig.hit ? 'W' : 'L'} | PARLAY ${sig.n_legs}L @ ${sig.parlay_american}`);
    for (const l of (sig.legs || [])) console.log(`    ${l.hit ? 'W' : 'L'} | ${l.player} O${l.line} ${l.statLabel || l.stat} | actual: ${l.actual}`);
  }
}

// Key params
console.log('\n=== KEY OPTIMIZED PARAMS ===');
const kp = ['MIN_ODDS', 'MAX_ODDS', 'BEQ_CREDIBLE_LEVEL', 'GATE_MIN_HIT_RATE',
  'GATE_MIN_COMBINED', 'GATE_MIN_GFT_SCORE', 'GATE_MIN_BEQ_EDGE', 'GATE_MIN_PBF_MARGIN',
  'GATE_MIN_STREAK', 'GATE_MIN_ESI_STABILITY', 'PBF_WEIGHT', 'SINGLE_MIN_SCORE',
  'PARLAY_LEG_MIN_SCORE', 'PARLAY_MAX_PER_DAY', 'GFT_MIN_CLEARANCE', 'GFT_CONVERGENCE_MAX_SPREAD',
  'SWTA_ENABLED', 'ASC_ENABLED', 'DMCT_ENABLED', 'TRD_ENABLED', 'PSI_ENABLED',
  'MIN_GAMES', 'WARM_UP_GAMES'];
for (const p of kp) console.log(`  ${p}: ${JSON.stringify(best.config[p])}`);

// Cross-validation
console.log('\n=== PURGED WALK-FORWARD CROSS-VALIDATION ===');
for (const k of Object.keys(best.config)) {
  if (ENGINE.CONFIG.hasOwnProperty(k)) ENGINE.CONFIG[k] = best.config[k];
}
const cvResults = ENGINE.runCrossValidation(boxScores, historicalOdds, 5);
if (cvResults && cvResults.folds) {
  for (let i = 0; i < cvResults.folds.length; i++) {
    const fold = cvResults.folds[i];
    console.log(`  Fold ${i}: ${fold.total} sigs, ${(fold.accuracy * 100).toFixed(1)}% acc, ${(fold.roi * 100).toFixed(1)}% ROI, $${fold.pnl}`);
  }
  console.log(`  CV Average: ${(cvResults.avgAccuracy * 100).toFixed(1)}% acc, ${(cvResults.avgROI * 100).toFixed(1)}% ROI`);
  const overfit = Math.abs(f.overall.accuracy - cvResults.avgAccuracy);
  console.log(`  Overfitting gap: ${(overfit * 100).toFixed(1)}% (threshold: <5% = safe)`);
  configOutput.crossValidation = cvResults;
}

// Sensitivity analysis
console.log('\n=== PARAMETER SENSITIVITY ANALYSIS ===');
const sensitivity = ENGINE.runSensitivityAnalysis(boxScores, historicalOdds);
if (sensitivity && sensitivity.sensitivities) {
  const sorted = Object.entries(sensitivity.sensitivities).sort((a, b) => b[1] - a[1]);
  for (const [param, sens] of sorted.slice(0, 15)) {
    console.log(`  ${param}: ${(sens * 100).toFixed(1)}% sensitivity`);
  }
  const maxSens = sorted[0] ? sorted[0][1] : 0;
  console.log(`  Max sensitivity: ${(maxSens * 100).toFixed(1)}%`);
  console.log(`  Overfitting risk: ${maxSens < 0.30 ? 'LOW' : maxSens < 0.50 ? 'MEDIUM' : 'HIGH'}`);
  configOutput.sensitivity = sensitivity;
}

// Final save with CV + sensitivity
fs.writeFileSync(path.join(OUTPUT_DIR, 'mlb_ultra_engine_v3_config.json'), JSON.stringify(configOutput, null, 2));

console.log('\n=== OPTIMIZATION COMPLETE ===');
