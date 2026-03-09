#!/usr/bin/env node
// =============================================================================
// AutoResearch v4 — Optimizer for Enhanced 8-Gate MLB Engine
// With CGSM (streak momentum) and ABVC (at-bat volume confidence)
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

console.log('=== AutoResearch v4 — 8-Gate MLB Engine Optimizer ===');
console.log(`Data: ${boxScores.length} games, ${historicalOdds.length} odds records`);

// Balanced objective: accuracy^2 * count_factor * roi_factor
function objective(results) {
  if (!results || results.stats.overall.total < 3) return -Infinity;
  const acc = results.stats.overall.accuracy;
  const roi = results.stats.overall.roi;
  const count = results.stats.overall.total;
  const pnl = results.stats.overall.pnl;

  // Must have positive ROI for good score
  if (roi < 0) return acc * 50 + roi * 80;

  // Accuracy is king (squared)
  let score = (acc * acc) * 200;

  // ROI bonus (capped)
  score += Math.min(roi * 25, 80);

  // Count: prefer 10-60 range
  if (count >= 10 && count <= 60) score += 20;
  else if (count >= 5) score += 8;
  else score -= 15;

  // Accuracy bonuses
  if (acc >= 0.97) score += 100;
  else if (acc >= 0.95) score += 60;
  else if (acc >= 0.93) score += 35;
  else if (acc >= 0.90) score += 15;

  // P&L bonus
  if (pnl > 500) score += 5;
  if (pnl > 1000) score += 10;

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
  // Set full config
  const origConfig = {};
  for (const k of Object.keys(ENGINE.CONFIG)) {
    origConfig[k] = ENGINE.CONFIG[k];
  }
  Object.assign(ENGINE.CONFIG, startConfig);

  let bestConfig = { ...ENGINE.CONFIG };
  let bestResults = ENGINE.runBacktest(boxScores, historicalOdds);
  let bestScore = objective(bestResults);
  let totalImprovements = 0;

  Object.assign(ENGINE.CONFIG, origConfig);

  const r = bestResults.stats.overall;
  console.log(`Start: ${r.total} signals, ${(r.accuracy * 100).toFixed(1)}% acc, $${r.pnl}, ${(r.roi * 100).toFixed(1)}% ROI | score=${bestScore.toFixed(1)}`);

  const sweeps = {
    // Core quality gates
    GATE_MIN_HIT_RATE: [0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.87, 0.88, 0.90, 0.92, 0.95, 1.0],
    GATE_MIN_COMBINED: [0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80],
    GATE_MIN_GFT_SCORE: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    GATE_MIN_BEQ_EDGE: [0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    GATE_MIN_ESI_STABILITY: [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50],
    GATE_MIN_IMAD_SCORE: [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15],

    // NEW: Streak and ABVC gates
    GATE_MIN_STREAK: [1, 2, 3, 4, 5, 6, 7, 8],
    GATE_MIN_ABVC: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    CGSM_MIN_STREAK: [1, 2, 3, 4, 5],
    CGSM_LOOKBACK: [5, 7, 8, 10, 12, 15],
    CGSM_WEIGHT: [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    ABVC_MIN_AVG_AB: [2.0, 2.5, 3.0, 3.2, 3.5, 4.0],
    ABVC_ELITE_AB: [3.5, 4.0, 4.2, 4.5, 5.0],
    ABVC_WEIGHT: [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],

    // Odds range
    MIN_ODDS: [-800, -600, -500, -450, -400, -350, -300, -280, -260, -250, -240, -220, -200],
    MAX_ODDS: [-100, -110, -120, -125, -130, -135, -140, -145, -150, -155, -160, -170, -180, -200],

    // Eligibility
    MIN_GAMES: [5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
    MIN_AB: [1, 2, 3, 4],
    WARM_UP_GAMES: [6, 8, 10, 12, 14, 15, 18],

    // GFT
    GFT_WINDOWS: [[3, 5, 8], [3, 5, 10], [3, 7, 12], [5, 8, 12], [5, 10, 15], [3, 7, 15]],
    GFT_DECAY_RATE: [0.78, 0.80, 0.82, 0.85, 0.87, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97, 0.98],
    GFT_GRAVITY_STRENGTH: [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
    GFT_MIN_CLEARANCE: [0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5],
    GFT_CONVERGENCE_MAX_SPREAD: [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],

    // BEQ
    BEQ_CREDIBLE_LEVEL: [0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95],
    BEQ_MIN_EDGE: [0.01, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25],

    // ESI
    ESI_MAX_ENTROPY: [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
    ESI_TREND_WEIGHT: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
    ESI_BINS: [4, 5, 6, 7, 8],

    // Bet selection
    SINGLE_MIN_SCORE: [0.55, 0.6, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85],
    MULTI_SINGLE_MIN_SCORE: [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.67, 0.70, 0.75, 0.80],
  };

  const paramOrder = [
    'GATE_MIN_STREAK', 'GATE_MIN_ABVC', 'GATE_MIN_HIT_RATE',
    'MIN_ODDS', 'MAX_ODDS',
    'GATE_MIN_BEQ_EDGE', 'GATE_MIN_COMBINED', 'GATE_MIN_GFT_SCORE',
    'CGSM_MIN_STREAK', 'CGSM_LOOKBACK', 'CGSM_WEIGHT',
    'ABVC_MIN_AVG_AB', 'ABVC_ELITE_AB', 'ABVC_WEIGHT',
    'GFT_MIN_CLEARANCE', 'BEQ_MIN_EDGE', 'BEQ_CREDIBLE_LEVEL',
    'MIN_GAMES', 'MIN_AB', 'WARM_UP_GAMES',
    'GFT_DECAY_RATE', 'GFT_GRAVITY_STRENGTH', 'GFT_CONVERGENCE_MAX_SPREAD',
    'GATE_MIN_ESI_STABILITY', 'GATE_MIN_IMAD_SCORE',
    'ESI_MAX_ENTROPY', 'ESI_TREND_WEIGHT', 'ESI_BINS',
    'SINGLE_MIN_SCORE', 'MULTI_SINGLE_MIN_SCORE',
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
          console.log(`  ${param}: ${JSON.stringify(val)} | ${s.total} sigs, ${(s.accuracy * 100).toFixed(1)}% acc, $${s.pnl}, ${(s.roi * 100).toFixed(1)}% ROI | score=${score.toFixed(1)}`);
        }
      }
    }

    console.log(`Round ${round + 1}: ${roundImprovements} improvements`);
    if (roundImprovements === 0) break;
  }

  return { config: bestConfig, results: bestResults, score: bestScore, improvements: totalImprovements };
}

// Multiple starting points
const starts = [
  // Start A: Moderate selectivity with streak focus
  {
    MIN_GAMES: 10, MIN_AB: 3, WARM_UP_GAMES: 12,
    GFT_WINDOWS: [5, 10, 15], GFT_DECAY_RATE: 0.92, GFT_GRAVITY_STRENGTH: 0.45,
    GFT_MIN_CLEARANCE: 0.25, GFT_CONVERGENCE_MAX_SPREAD: 1.5,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.85, BEQ_MIN_EDGE: 0.08,
    ESI_BINS: 5, ESI_MAX_ENTROPY: 0.75, ESI_TREND_WEIGHT: 0.25,
    IMAD_MIN_ASYMMETRY: 0.1, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 3, CGSM_LOOKBACK: 10, CGSM_WEIGHT: 0.15,
    ABVC_MIN_AVG_AB: 3.0, ABVC_ELITE_AB: 4.0, ABVC_WEIGHT: 0.10,
    GATE_MIN_GFT_SCORE: 0.5, GATE_MIN_BEQ_EDGE: 0.08,
    GATE_MIN_ESI_STABILITY: 0.20, GATE_MIN_IMAD_SCORE: 0.02,
    GATE_MIN_HIT_RATE: 0.85, GATE_MIN_COMBINED: 0.60,
    GATE_MIN_STREAK: 3, GATE_MIN_ABVC: 0.5,
    SINGLE_MIN_SCORE: 0.72, MULTI_SINGLE_MIN_SCORE: 0.65,
    PARLAY_LEG_MIN_SCORE: 0.60, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.33, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.1,
    MIN_ODDS: -350, MAX_ODDS: -130,
    PREFERRED_ODDS_RANGE: [-400, -140],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
  },
  // Start B: Aggressive streak + wider odds
  {
    MIN_GAMES: 8, MIN_AB: 3, WARM_UP_GAMES: 10,
    GFT_WINDOWS: [3, 7, 12], GFT_DECAY_RATE: 0.88, GFT_GRAVITY_STRENGTH: 0.5,
    GFT_MIN_CLEARANCE: 0.15, GFT_CONVERGENCE_MAX_SPREAD: 2.0,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.82, BEQ_MIN_EDGE: 0.05,
    ESI_BINS: 5, ESI_MAX_ENTROPY: 0.8, ESI_TREND_WEIGHT: 0.3,
    IMAD_MIN_ASYMMETRY: 0.1, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 4, CGSM_LOOKBACK: 8, CGSM_WEIGHT: 0.20,
    ABVC_MIN_AVG_AB: 3.5, ABVC_ELITE_AB: 4.5, ABVC_WEIGHT: 0.10,
    GATE_MIN_GFT_SCORE: 0.45, GATE_MIN_BEQ_EDGE: 0.05,
    GATE_MIN_ESI_STABILITY: 0.15, GATE_MIN_IMAD_SCORE: 0.02,
    GATE_MIN_HIT_RATE: 0.88, GATE_MIN_COMBINED: 0.55,
    GATE_MIN_STREAK: 4, GATE_MIN_ABVC: 0.4,
    SINGLE_MIN_SCORE: 0.65, MULTI_SINGLE_MIN_SCORE: 0.60,
    PARLAY_LEG_MIN_SCORE: 0.55, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.33, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.1,
    MIN_ODDS: -600, MAX_ODDS: -120,
    PREFERRED_ODDS_RANGE: [-600, -120],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
  },
  // Start C: High streak req + tight BEQ
  {
    MIN_GAMES: 7, MIN_AB: 2, WARM_UP_GAMES: 8,
    GFT_WINDOWS: [3, 5, 10], GFT_DECAY_RATE: 0.90, GFT_GRAVITY_STRENGTH: 0.4,
    GFT_MIN_CLEARANCE: 0.2, GFT_CONVERGENCE_MAX_SPREAD: 1.5,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.87, BEQ_MIN_EDGE: 0.10,
    ESI_BINS: 6, ESI_MAX_ENTROPY: 0.7, ESI_TREND_WEIGHT: 0.2,
    IMAD_MIN_ASYMMETRY: 0.1, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 5, CGSM_LOOKBACK: 10, CGSM_WEIGHT: 0.18,
    ABVC_MIN_AVG_AB: 3.2, ABVC_ELITE_AB: 4.2, ABVC_WEIGHT: 0.12,
    GATE_MIN_GFT_SCORE: 0.55, GATE_MIN_BEQ_EDGE: 0.10,
    GATE_MIN_ESI_STABILITY: 0.25, GATE_MIN_IMAD_SCORE: 0.03,
    GATE_MIN_HIT_RATE: 0.90, GATE_MIN_COMBINED: 0.65,
    GATE_MIN_STREAK: 5, GATE_MIN_ABVC: 0.5,
    SINGLE_MIN_SCORE: 0.70, MULTI_SINGLE_MIN_SCORE: 0.65,
    PARLAY_LEG_MIN_SCORE: 0.60, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.33, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.1,
    MIN_ODDS: -500, MAX_ODDS: -140,
    PREFERRED_ODDS_RANGE: [-500, -140],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
  },
];

let overallBest = null;

for (let i = 0; i < starts.length; i++) {
  console.log(`\n${'#'.repeat(60)}`);
  console.log(`START POINT ${i + 1}/${starts.length}`);
  console.log(`${'#'.repeat(60)}`);

  const result = ratchetOptimize(starts[i], 8);

  if (!overallBest || result.score > overallBest.score) {
    overallBest = result;
    console.log(`*** New overall best: ${(result.results.stats.overall.accuracy * 100).toFixed(1)}% acc, ${result.results.stats.overall.total} signals ***`);
  }
}

const best = overallBest;
const f = best.results.stats;

console.log('\n' + '='.repeat(60));
console.log('FINAL OPTIMIZED RESULTS');
console.log('='.repeat(60));
console.log(`Singles: ${f.singles.total} (${(f.singles.accuracy * 100).toFixed(1)}% acc, $${f.singles.pnl}, ${(f.singles.roi * 100).toFixed(1)}% ROI)`);
console.log(`Parlays: ${f.parlays.total} (${(f.parlays.accuracy * 100).toFixed(1)}% acc)`);
if (f.parlays.totalLegs > 0) {
  console.log(`  Leg accuracy: ${(f.parlays.legAccuracy * 100).toFixed(1)}% (${f.parlays.hitLegs}/${f.parlays.totalLegs})`);
}
console.log(`Overall: ${f.overall.total} (${(f.overall.accuracy * 100).toFixed(1)}% acc, $${f.overall.pnl}, ${(f.overall.roi * 100).toFixed(1)}% ROI)`);

// Save config
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
};

if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.writeFileSync(path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json'), JSON.stringify(configOutput, null, 2));
console.log(`\nConfig saved to ${OUTPUT_DIR}/mlb_ultra_engine_config.json`);

// Also print which signals hit/missed for analysis
console.log('\n=== SIGNAL DETAILS ===');
for (const sig of best.results.signals) {
  if (sig.betType === 'single') {
    console.log(`  ${sig.date} ${sig.hit ? 'WIN' : 'LOSS'} | ${sig.player} O${sig.line} ${sig.statLabel || sig.statType} | odds ${sig.odds} | actual: ${sig.actual} | streak: ${sig.streak || '?'}`);
  } else if (sig.betType === 'parlay') {
    console.log(`  ${sig.date} ${sig.hit ? 'WIN' : 'LOSS'} | PARLAY ${sig.n_legs}L`);
    for (const l of (sig.legs || [])) {
      console.log(`    ${l.hit ? 'WIN' : 'LOSS'} | ${l.player} O${l.line} ${l.statLabel || l.stat} | actual: ${l.actual}`);
    }
  }
}
