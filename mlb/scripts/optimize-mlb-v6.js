#!/usr/bin/env node
// =============================================================================
// AutoResearch v6 — MLB Ultra Engine v2.0 Optimizer
// Monotonic Ratchet with Poisson-Bayesian Fusion, OAL, VPD parameter sweeps
// Target: NBA-level accuracy (95%+) with profitable parlays
// =============================================================================

const fs = require('fs');
const path = require('path');

global.window = global;
global.document = { readyState: 'complete', addEventListener: () => {}, querySelectorAll: () => [] };
global.localStorage = { _data: {}, getItem(k) { return this._data[k] || null; }, setItem(k, v) { this._data[k] = v; } };

eval(fs.readFileSync(path.join(__dirname, '../webapp/js/recommendation-engine-mlb.js'), 'utf8'));

const ENGINE = global.window.MLBRecommendationEngine;
const DATA_DIR = path.join(__dirname, '../webapp/data');
const OUTPUT_DIR = path.join(__dirname, '../output');

const boxScores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_player_boxscores.json'), 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_historical_odds.json'), 'utf8'));

console.log('=== AutoResearch v6 — MLB v2.0 Optimizer ===');
console.log(`Data: ${boxScores.length} games, ${historicalOdds.length} odds records`);

// Objective function: mirrors NBA model's success criteria
// Prioritizes: high accuracy > profitable parlays > volume > ROI
function objective(results) {
  if (!results || results.stats.overall.total < 3) return -Infinity;
  const acc = results.stats.overall.accuracy;
  const roi = results.stats.overall.roi;
  const count = results.stats.overall.total;
  const pnl = results.stats.overall.pnl;

  // Hard floor: must achieve minimum accuracy
  if (acc < 0.75) return acc * 20 + roi * 5;

  // Base: accuracy is king (like NBA model)
  let score = acc * 200;

  // Volume tiers: strongly prefer 15-80 signals (like NBA's 168)
  if (count >= 30 && count <= 100) score += 50;
  else if (count >= 20) score += 40;
  else if (count >= 15) score += 30;
  else if (count >= 10) score += 15;
  else if (count >= 7) score += 5;
  else score -= 20;

  // ROI: must be positive
  if (roi > 0) score += Math.min(roi * 25, 80);
  else score += roi * 50; // Heavily penalize losses

  // Parlay bonus (critical for NBA-level ROI)
  const parlays = results.stats.parlays;
  if (parlays.total > 0) {
    score += parlays.total * 8; // Each parlay opportunity is valuable
    if (parlays.accuracy >= 0.90) score += 30;
    else if (parlays.accuracy >= 0.80) score += 15;
    if (parlays.roi > 0) score += Math.min(parlays.roi * 15, 60);
    // Leg accuracy like NBA's 98.2%
    if (parlays.legAccuracy >= 0.95) score += 25;
    else if (parlays.legAccuracy >= 0.90) score += 15;
  }

  // Accuracy bonuses (tiered like NBA performance)
  if (acc >= 0.97) score += 80;
  else if (acc >= 0.95) score += 60;
  else if (acc >= 0.93) score += 40;
  else if (acc >= 0.90) score += 25;
  else if (acc >= 0.87) score += 10;

  // PnL bonus (absolute profit matters)
  if (pnl > 1000) score += 20;
  else if (pnl > 500) score += 10;

  return score;
}

function runWithConfig(configOverrides) {
  const origConfig = {};
  for (const k of Object.keys(configOverrides)) {
    origConfig[k] = ENGINE.CONFIG[k];
  }
  Object.assign(ENGINE.CONFIG, configOverrides);
  const results = ENGINE.runBacktest(boxScores, historicalOdds);
  Object.assign(ENGINE.CONFIG, origConfig);
  return results;
}

function ratchetOptimize(startConfig, maxRounds) {
  const origConfig = {};
  for (const k of Object.keys(ENGINE.CONFIG)) origConfig[k] = ENGINE.CONFIG[k];
  Object.assign(ENGINE.CONFIG, startConfig);

  let bestConfig = { ...ENGINE.CONFIG };
  let bestResults = ENGINE.runBacktest(boxScores, historicalOdds);
  let bestScore = objective(bestResults);
  let totalImprovements = 0;

  Object.assign(ENGINE.CONFIG, origConfig);

  const r = bestResults.stats.overall;
  const p = bestResults.stats.parlays;
  console.log(`Start: ${r.total} signals (${p.total}P), ${(r.accuracy * 100).toFixed(1)}% acc, $${r.pnl}, ${(r.roi * 100).toFixed(1)}% ROI | score=${bestScore.toFixed(1)}`);

  // Comprehensive parameter sweeps including all v2.0 innovations
  const sweeps = {
    // Priority 1: Odds filter (biggest blocker in v1)
    MIN_ODDS: [-800, -700, -600, -500, -450, -400, -350, -300, -260, -220, -200, -180, -160],
    MAX_ODDS: [-100, -110, -115, -120, -125, -130, -135, -140, -145, -150, -160, -170, -180, -200],

    // Priority 2: Quality gates
    GATE_MIN_HIT_RATE: [0.75, 0.78, 0.80, 0.82, 0.85, 0.87, 0.88, 0.90, 0.92, 0.95, 1.0],
    GATE_MIN_COMBINED: [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75],
    GATE_MIN_GFT_SCORE: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    GATE_MIN_BEQ_EDGE: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
    GATE_MIN_ESI_STABILITY: [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35],
    GATE_MIN_IMAD_SCORE: [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10],
    GATE_MIN_STREAK: [1, 2, 3, 4, 5, 6],
    GATE_MIN_ABVC: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60],
    GATE_MIN_PBF_MARGIN: [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15],

    // Priority 3: BEQ calibration (key insight from NBA: 87% credible)
    BEQ_CREDIBLE_LEVEL: [0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95],
    BEQ_MIN_EDGE: [0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20],

    // Priority 4: GFT parameters
    GFT_DECAY_RATE: [0.80, 0.82, 0.85, 0.87, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97],
    GFT_GRAVITY_STRENGTH: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    GFT_MIN_CLEARANCE: [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
    GFT_CONVERGENCE_MAX_SPREAD: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    GFT_WINDOWS: [[3, 5, 8], [3, 5, 10], [3, 7, 12], [5, 8, 12], [5, 10, 15], [3, 7, 15]],

    // Priority 5: Poisson-Bayesian Fusion (NEW v2.0)
    PBF_DECAY_RATE: [0.85, 0.87, 0.90, 0.92, 0.94, 0.95, 0.97],
    PBF_MIN_LAMBDA: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
    PBF_WEIGHT: [0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18],

    // Priority 6: CGSM and ABVC
    CGSM_MIN_STREAK: [1, 2, 3, 4, 5],
    CGSM_LOOKBACK: [5, 7, 8, 10, 12, 15],
    CGSM_WEIGHT: [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15],
    ABVC_MIN_AVG_AB: [2.0, 2.5, 3.0, 3.2, 3.5, 4.0],
    ABVC_ELITE_AB: [3.5, 4.0, 4.5, 5.0, 5.5],
    ABVC_WEIGHT: [0.03, 0.05, 0.07, 0.08, 0.10, 0.12],

    // Priority 7: ESI
    ESI_MAX_ENTROPY: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
    ESI_TREND_WEIGHT: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    ESI_BINS: [4, 5, 6, 7, 8],

    // Priority 8: Player eligibility
    MIN_GAMES: [5, 6, 7, 8, 9, 10, 12, 14],
    MIN_AB: [1, 2, 3],
    WARM_UP_GAMES: [5, 6, 7, 8, 9, 10, 12],

    // Priority 9: OAL (NEW v2.0)
    OAL_ENABLED: [true, false],
    OAL_WEIGHT: [0.02, 0.03, 0.05, 0.07, 0.10],
    OAL_MAX_ADJUSTMENT: [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],

    // Priority 10: VPD (NEW v2.0)
    VPD_ENABLED: [true, false],
    VPD_MIN_GAMES: [3, 4, 5, 7, 10],
    VPD_WEIGHT: [0.02, 0.03, 0.05, 0.07, 0.10],
    VPD_MAX_BOOST: [0.05, 0.07, 0.10, 0.12, 0.15],

    // Priority 11: Bet type selection
    SINGLE_MIN_SCORE: [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80],
    MULTI_SINGLE_MIN_SCORE: [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75],
    PARLAY_LEG_MIN_SCORE: [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68],

    // Priority 12: Parlay construction
    PARLAY_MAX_CORRELATION: [0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.50, 0.60, 0.70],
    PARLAY_SAME_GAME_ALLOWED: [true, false],
    PARLAY_MIN_COMBINED_EDGE: [0.03, 0.05, 0.08, 0.10, 0.12, 0.15],
    PARLAY_MAX_LEGS: [2, 3, 4],

    // Priority 13: Contextual
    HOME_BOOST: [0.0, 0.01, 0.015, 0.02, 0.025, 0.03],
  };

  // Order sweeps from most impactful to least
  const paramOrder = [
    // Fix the biggest blockers first
    'MIN_ODDS', 'MAX_ODDS',
    // Quality gates (high impact)
    'GATE_MIN_HIT_RATE', 'GATE_MIN_COMBINED', 'GATE_MIN_BEQ_EDGE',
    'GATE_MIN_GFT_SCORE', 'GATE_MIN_PBF_MARGIN',
    'GATE_MIN_STREAK', 'GATE_MIN_ABVC',
    // BEQ calibration (critical)
    'BEQ_CREDIBLE_LEVEL', 'BEQ_MIN_EDGE',
    // GFT
    'GFT_DECAY_RATE', 'GFT_GRAVITY_STRENGTH', 'GFT_MIN_CLEARANCE',
    'GFT_CONVERGENCE_MAX_SPREAD',
    // Poisson-Bayesian Fusion (new)
    'PBF_DECAY_RATE', 'PBF_MIN_LAMBDA', 'PBF_WEIGHT',
    // CGSM, ABVC
    'CGSM_MIN_STREAK', 'CGSM_LOOKBACK', 'CGSM_WEIGHT',
    'ABVC_MIN_AVG_AB', 'ABVC_ELITE_AB', 'ABVC_WEIGHT',
    // ESI
    'GATE_MIN_ESI_STABILITY', 'GATE_MIN_IMAD_SCORE',
    'ESI_MAX_ENTROPY', 'ESI_TREND_WEIGHT', 'ESI_BINS',
    // Player eligibility
    'MIN_GAMES', 'MIN_AB', 'WARM_UP_GAMES',
    // OAL, VPD (new)
    'OAL_ENABLED', 'OAL_WEIGHT', 'OAL_MAX_ADJUSTMENT',
    'VPD_ENABLED', 'VPD_MIN_GAMES', 'VPD_WEIGHT', 'VPD_MAX_BOOST',
    // Bet type
    'SINGLE_MIN_SCORE', 'MULTI_SINGLE_MIN_SCORE', 'PARLAY_LEG_MIN_SCORE',
    // Parlay
    'PARLAY_MAX_CORRELATION', 'PARLAY_SAME_GAME_ALLOWED',
    'PARLAY_MIN_COMBINED_EDGE', 'PARLAY_MAX_LEGS',
    // Contextual
    'HOME_BOOST',
    // Windows last (expensive)
    'GFT_WINDOWS',
  ];

  for (let round = 0; round < maxRounds; round++) {
    let roundImprovements = 0;
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
          const s = results.stats.overall;
          const p = results.stats.parlays;
          console.log(`  [+] ${param}: ${JSON.stringify(val)} | ${s.total} sigs (${p.total}P), ${(s.accuracy * 100).toFixed(1)}% acc, $${s.pnl}, ${(s.roi * 100).toFixed(1)}% ROI | score=${score.toFixed(1)}`);
        }
      }
    }
    console.log(`Round ${round + 1}: ${roundImprovements} improvements`);
    if (roundImprovements === 0) break;
  }

  return { config: bestConfig, results: bestResults, score: bestScore, improvements: totalImprovements };
}

// ============================================================================
// Multiple starting points for broader search
// ============================================================================

const starts = [
  // Start A: Moderate selectivity, wide odds, NBA-calibrated BEQ
  {
    MIN_GAMES: 8, MIN_AB: 2, WARM_UP_GAMES: 10,
    GFT_WINDOWS: [3, 5, 10], GFT_DECAY_RATE: 0.90, GFT_GRAVITY_STRENGTH: 0.40,
    GFT_MIN_CLEARANCE: 0.05, GFT_CONVERGENCE_MAX_SPREAD: 3.0,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.87, BEQ_MIN_EDGE: 0.05,
    ESI_BINS: 5, ESI_MAX_ENTROPY: 0.85, ESI_TREND_WEIGHT: 0.25,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 2, CGSM_LOOKBACK: 10, CGSM_WEIGHT: 0.08,
    ABVC_MIN_AVG_AB: 3.0, ABVC_ELITE_AB: 5.0, ABVC_WEIGHT: 0.07,
    PBF_DECAY_RATE: 0.92, PBF_MIN_LAMBDA: 0.3, PBF_WEIGHT: 0.10,
    OAL_ENABLED: true, OAL_WEIGHT: 0.05, OAL_MAX_ADJUSTMENT: 0.25,
    VPD_ENABLED: true, VPD_MIN_GAMES: 5, VPD_WEIGHT: 0.05, VPD_MAX_BOOST: 0.10,
    GATE_MIN_GFT_SCORE: 0.40, GATE_MIN_BEQ_EDGE: 0.05,
    GATE_MIN_ESI_STABILITY: 0.15, GATE_MIN_IMAD_SCORE: 0.02,
    GATE_MIN_HIT_RATE: 0.82, GATE_MIN_COMBINED: 0.55,
    GATE_MIN_STREAK: 2, GATE_MIN_ABVC: 0.30,
    GATE_MIN_PBF_MARGIN: 0.03,
    SINGLE_MIN_SCORE: 0.72, MULTI_SINGLE_MIN_SCORE: 0.65,
    PARLAY_LEG_MIN_SCORE: 0.58, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.40, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.08,
    MIN_ODDS: -450, MAX_ODDS: -115,
    PREFERRED_ODDS_RANGE: [-500, -110],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
    HOME_BOOST: 0.015, BACK_TO_BACK_PENALTY: 0.00,
  },
  // Start B: Volume-first with relaxed gates
  {
    MIN_GAMES: 7, MIN_AB: 2, WARM_UP_GAMES: 8,
    GFT_WINDOWS: [3, 7, 12], GFT_DECAY_RATE: 0.88, GFT_GRAVITY_STRENGTH: 0.35,
    GFT_MIN_CLEARANCE: 0.02, GFT_CONVERGENCE_MAX_SPREAD: 4.0,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.82, BEQ_MIN_EDGE: 0.03,
    ESI_BINS: 5, ESI_MAX_ENTROPY: 0.90, ESI_TREND_WEIGHT: 0.20,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 1, CGSM_LOOKBACK: 8, CGSM_WEIGHT: 0.10,
    ABVC_MIN_AVG_AB: 2.5, ABVC_ELITE_AB: 4.5, ABVC_WEIGHT: 0.08,
    PBF_DECAY_RATE: 0.90, PBF_MIN_LAMBDA: 0.2, PBF_WEIGHT: 0.12,
    OAL_ENABLED: true, OAL_WEIGHT: 0.03, OAL_MAX_ADJUSTMENT: 0.20,
    VPD_ENABLED: true, VPD_MIN_GAMES: 4, VPD_WEIGHT: 0.03, VPD_MAX_BOOST: 0.08,
    GATE_MIN_GFT_SCORE: 0.30, GATE_MIN_BEQ_EDGE: 0.03,
    GATE_MIN_ESI_STABILITY: 0.10, GATE_MIN_IMAD_SCORE: 0.01,
    GATE_MIN_HIT_RATE: 0.78, GATE_MIN_COMBINED: 0.48,
    GATE_MIN_STREAK: 1, GATE_MIN_ABVC: 0.20,
    GATE_MIN_PBF_MARGIN: 0.02,
    SINGLE_MIN_SCORE: 0.65, MULTI_SINGLE_MIN_SCORE: 0.58,
    PARLAY_LEG_MIN_SCORE: 0.52, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.50, PARLAY_SAME_GAME_ALLOWED: true,
    PARLAY_MIN_COMBINED_EDGE: 0.05,
    MIN_ODDS: -600, MAX_ODDS: -110,
    PREFERRED_ODDS_RANGE: [-600, -110],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
    HOME_BOOST: 0.02, BACK_TO_BACK_PENALTY: 0.00,
  },
  // Start C: Tight selectivity for ultra-high accuracy
  {
    MIN_GAMES: 10, MIN_AB: 3, WARM_UP_GAMES: 12,
    GFT_WINDOWS: [5, 10, 15], GFT_DECAY_RATE: 0.92, GFT_GRAVITY_STRENGTH: 0.45,
    GFT_MIN_CLEARANCE: 0.10, GFT_CONVERGENCE_MAX_SPREAD: 2.0,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.90, BEQ_MIN_EDGE: 0.07,
    ESI_BINS: 6, ESI_MAX_ENTROPY: 0.75, ESI_TREND_WEIGHT: 0.30,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 3, CGSM_LOOKBACK: 12, CGSM_WEIGHT: 0.07,
    ABVC_MIN_AVG_AB: 3.5, ABVC_ELITE_AB: 5.0, ABVC_WEIGHT: 0.05,
    PBF_DECAY_RATE: 0.94, PBF_MIN_LAMBDA: 0.4, PBF_WEIGHT: 0.08,
    OAL_ENABLED: true, OAL_WEIGHT: 0.05, OAL_MAX_ADJUSTMENT: 0.20,
    VPD_ENABLED: false, VPD_MIN_GAMES: 7, VPD_WEIGHT: 0.03, VPD_MAX_BOOST: 0.07,
    GATE_MIN_GFT_SCORE: 0.50, GATE_MIN_BEQ_EDGE: 0.07,
    GATE_MIN_ESI_STABILITY: 0.20, GATE_MIN_IMAD_SCORE: 0.03,
    GATE_MIN_HIT_RATE: 0.85, GATE_MIN_COMBINED: 0.60,
    GATE_MIN_STREAK: 3, GATE_MIN_ABVC: 0.35,
    GATE_MIN_PBF_MARGIN: 0.05,
    SINGLE_MIN_SCORE: 0.75, MULTI_SINGLE_MIN_SCORE: 0.68,
    PARLAY_LEG_MIN_SCORE: 0.62, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 3,
    PARLAY_MAX_CORRELATION: 0.30, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.10,
    MIN_ODDS: -350, MAX_ODDS: -130,
    PREFERRED_ODDS_RANGE: [-400, -120],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
    HOME_BOOST: 0.01, BACK_TO_BACK_PENALTY: 0.00,
  },
];

let overallBest = null;

for (let i = 0; i < starts.length; i++) {
  console.log(`\n${'#'.repeat(60)}`);
  console.log(`START POINT ${i + 1}/${starts.length}`);
  console.log(`${'#'.repeat(60)}`);
  const result = ratchetOptimize(starts[i], 10);
  if (!overallBest || result.score > overallBest.score) {
    overallBest = result;
    console.log(`*** New overall best: score=${result.score.toFixed(1)} ***`);
  }
}

const best = overallBest;
const f = best.results.stats;

console.log('\n' + '='.repeat(60));
console.log('FINAL RESULTS — MLB Ultra Engine v2.0');
console.log('='.repeat(60));
console.log(`Singles: ${f.singles.total} (${(f.singles.accuracy * 100).toFixed(1)}% acc, $${f.singles.pnl}, ${(f.singles.roi * 100).toFixed(1)}% ROI)`);
console.log(`Parlays: ${f.parlays.total} (${(f.parlays.accuracy * 100).toFixed(1)}% acc, $${f.parlays.pnl}, ${(f.parlays.roi * 100).toFixed(1)}% ROI)`);
if (f.parlays.totalLegs > 0) console.log(`  Legs: ${f.parlays.hitLegs}/${f.parlays.totalLegs} (${(f.parlays.legAccuracy * 100).toFixed(1)}%)`);
console.log(`Overall: ${f.overall.total} (${(f.overall.accuracy * 100).toFixed(1)}% acc, $${f.overall.pnl}, ${(f.overall.roi * 100).toFixed(1)}% ROI)`);
console.log(`Optimization score: ${best.score.toFixed(1)} (${best.improvements} improvements)`);

// Save optimized config
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
  version: '2.0',
};

if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.writeFileSync(path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json'), JSON.stringify(configOutput, null, 2));
console.log(`\nConfig saved to ${path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json')}`);

// Save signals and stats
const signalsPath = path.join(DATA_DIR, 'mlb_ultra_signals.json');
const statsPath = path.join(DATA_DIR, 'mlb_ultra_backtest_stats.json');
fs.writeFileSync(signalsPath, JSON.stringify(best.results.signals, null, 2));
fs.writeFileSync(statsPath, JSON.stringify(best.results.stats, null, 2));

// Also save to webapp directory
const webappDir = path.join(__dirname, '../../webapp/mlb/data');
if (fs.existsSync(webappDir)) {
  fs.writeFileSync(path.join(webappDir, 'mlb_ultra_signals.json'), JSON.stringify(best.results.signals, null, 2));
  fs.writeFileSync(path.join(webappDir, 'mlb_ultra_backtest_stats.json'), JSON.stringify(best.results.stats, null, 2));
}

// Print all signal details
console.log('\n=== SIGNAL DETAILS ===');
for (const sig of best.results.signals) {
  if (sig.betType === 'single') {
    console.log(`  ${sig.date} ${sig.hit ? 'W' : 'L'} | ${sig.player} O${sig.line} ${sig.statLabel || sig.statType} | ${sig.odds} | actual: ${sig.actual} | cascade: ${sig.cascadeScore?.toFixed(3) || '?'} | streak: ${sig.streak || '?'}`);
  } else {
    const status = sig.hit ? 'W' : 'L';
    console.log(`  ${sig.date} ${status} | PARLAY ${sig.n_legs}L @ ${sig.parlay_american} | EV: ${sig.ev}`);
    for (const l of (sig.legs || [])) {
      console.log(`    ${l.hit ? 'W' : 'L'} | ${l.player} O${l.line} ${l.statLabel || l.stat} | actual: ${l.actual}`);
    }
  }
}

// Print key optimized parameters
console.log('\n=== KEY OPTIMIZED PARAMETERS ===');
const keyParams = ['MIN_ODDS', 'MAX_ODDS', 'BEQ_CREDIBLE_LEVEL', 'GATE_MIN_HIT_RATE',
  'GATE_MIN_COMBINED', 'GATE_MIN_GFT_SCORE', 'GATE_MIN_BEQ_EDGE', 'GATE_MIN_PBF_MARGIN',
  'PBF_DECAY_RATE', 'PBF_MIN_LAMBDA', 'PBF_WEIGHT',
  'OAL_ENABLED', 'VPD_ENABLED', 'SINGLE_MIN_SCORE', 'PARLAY_LEG_MIN_SCORE'];
for (const p of keyParams) {
  console.log(`  ${p}: ${JSON.stringify(best.config[p])}`);
}
