#!/usr/bin/env node
// =============================================================================
// AutoResearch v5 — Volume-Optimized with Parlay Focus
// Target: 20+ signals at 95%+ accuracy to enable parlays for high ROI
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

console.log('=== AutoResearch v5 — Volume + Parlay Optimizer ===');

// Objective: reward accuracy AND volume for parlay construction
function objective(results) {
  if (!results || results.stats.overall.total < 3) return -Infinity;
  const acc = results.stats.overall.accuracy;
  const roi = results.stats.overall.roi;
  const count = results.stats.overall.total;
  const pnl = results.stats.overall.pnl;

  // Accuracy must be >= 90% minimum
  if (acc < 0.80) return acc * 30 + roi * 10;

  // Base score from accuracy
  let score = acc * 150;

  // Volume reward: strongly prefer 15-50 signals
  if (count >= 15 && count <= 50) score += 40;
  else if (count >= 10) score += 25;
  else if (count >= 7) score += 10;
  else score -= 10;

  // ROI must be positive
  if (roi > 0) score += Math.min(roi * 20, 60);
  else score += roi * 40;

  // Parlay bonus (more signals = more parlay opportunities)
  const parlays = results.stats.parlays;
  if (parlays.total > 0) {
    score += parlays.total * 5;
    if (parlays.accuracy >= 0.90) score += 20;
    if (parlays.roi > 0) score += Math.min(parlays.roi * 10, 40);
  }

  // High accuracy bonuses
  if (acc >= 0.97) score += 60;
  else if (acc >= 0.95) score += 40;
  else if (acc >= 0.93) score += 25;
  else if (acc >= 0.90) score += 10;

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
  console.log(`Start: ${r.total} signals, ${(r.accuracy * 100).toFixed(1)}% acc, $${r.pnl}, ${(r.roi * 100).toFixed(1)}% ROI | score=${bestScore.toFixed(1)}`);

  const sweeps = {
    GATE_MIN_STREAK: [1, 2, 3, 4, 5, 6, 7, 8],
    GATE_MIN_ABVC: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    GATE_MIN_HIT_RATE: [0.75, 0.78, 0.80, 0.82, 0.85, 0.87, 0.88, 0.90, 0.92, 0.95, 1.0],
    GATE_MIN_COMBINED: [0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80],
    GATE_MIN_GFT_SCORE: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
    GATE_MIN_BEQ_EDGE: [0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
    GATE_MIN_ESI_STABILITY: [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40],
    GATE_MIN_IMAD_SCORE: [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10],
    CGSM_MIN_STREAK: [1, 2, 3, 4, 5],
    CGSM_LOOKBACK: [5, 7, 8, 10, 12, 15],
    CGSM_WEIGHT: [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
    ABVC_MIN_AVG_AB: [2.0, 2.5, 3.0, 3.2, 3.5, 4.0],
    ABVC_ELITE_AB: [3.5, 4.0, 4.2, 4.5, 5.0],
    ABVC_WEIGHT: [0.05, 0.08, 0.10, 0.12, 0.15],
    MIN_ODDS: [-800, -600, -500, -450, -400, -350, -300, -260, -250, -220, -200],
    MAX_ODDS: [-110, -120, -125, -130, -135, -140, -145, -150, -155, -160, -170, -180, -200],
    MIN_GAMES: [5, 6, 7, 8, 9, 10, 12],
    MIN_AB: [1, 2, 3, 4],
    WARM_UP_GAMES: [6, 7, 8, 9, 10, 12, 14],
    GFT_WINDOWS: [[3, 5, 8], [3, 5, 10], [3, 7, 12], [5, 8, 12], [5, 10, 15]],
    GFT_DECAY_RATE: [0.80, 0.82, 0.85, 0.87, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97, 0.98],
    GFT_GRAVITY_STRENGTH: [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7],
    GFT_MIN_CLEARANCE: [0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    GFT_CONVERGENCE_MAX_SPREAD: [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
    BEQ_CREDIBLE_LEVEL: [0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95],
    BEQ_MIN_EDGE: [0.01, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20],
    ESI_MAX_ENTROPY: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
    ESI_TREND_WEIGHT: [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    ESI_BINS: [4, 5, 6, 7, 8],
    SINGLE_MIN_SCORE: [0.55, 0.6, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80],
    MULTI_SINGLE_MIN_SCORE: [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75],
    PARLAY_LEG_MIN_SCORE: [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70],
    PARLAY_MAX_CORRELATION: [0.20, 0.25, 0.30, 0.33, 0.40, 0.50, 0.60, 0.70, 0.80],
    PARLAY_SAME_GAME_ALLOWED: [true, false],
    PARLAY_MIN_COMBINED_EDGE: [0.05, 0.08, 0.10, 0.12, 0.15],
  };

  const paramOrder = [
    'GATE_MIN_STREAK', 'GATE_MIN_ABVC', 'GATE_MIN_HIT_RATE',
    'MIN_ODDS', 'MAX_ODDS', 'GATE_MIN_BEQ_EDGE', 'GATE_MIN_COMBINED',
    'GATE_MIN_GFT_SCORE', 'CGSM_MIN_STREAK', 'CGSM_LOOKBACK',
    'ABVC_MIN_AVG_AB', 'MIN_GAMES', 'MIN_AB', 'WARM_UP_GAMES',
    'GFT_MIN_CLEARANCE', 'BEQ_MIN_EDGE', 'BEQ_CREDIBLE_LEVEL',
    'GFT_DECAY_RATE', 'GFT_GRAVITY_STRENGTH', 'GFT_CONVERGENCE_MAX_SPREAD',
    'GATE_MIN_ESI_STABILITY', 'GATE_MIN_IMAD_SCORE',
    'ESI_MAX_ENTROPY', 'ESI_TREND_WEIGHT', 'ESI_BINS',
    'CGSM_WEIGHT', 'ABVC_WEIGHT', 'ABVC_ELITE_AB',
    'SINGLE_MIN_SCORE', 'MULTI_SINGLE_MIN_SCORE',
    'PARLAY_LEG_MIN_SCORE', 'PARLAY_MAX_CORRELATION',
    'PARLAY_SAME_GAME_ALLOWED', 'PARLAY_MIN_COMBINED_EDGE',
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
          console.log(`  ${param}: ${JSON.stringify(val)} | ${s.total} sigs (${p.total}P), ${(s.accuracy * 100).toFixed(1)}% acc, $${s.pnl}, ${(s.roi * 100).toFixed(1)}% ROI | score=${score.toFixed(1)}`);
        }
      }
    }
    console.log(`Round ${round + 1}: ${roundImprovements} improvements`);
    if (roundImprovements === 0) break;
  }

  return { config: bestConfig, results: bestResults, score: bestScore, improvements: totalImprovements };
}

// Starting configs focused on moderate selectivity
const starts = [
  // A: Moderate gates, relaxed streak
  {
    MIN_GAMES: 8, MIN_AB: 3, WARM_UP_GAMES: 10,
    GFT_WINDOWS: [3, 7, 12], GFT_DECAY_RATE: 0.90, GFT_GRAVITY_STRENGTH: 0.45,
    GFT_MIN_CLEARANCE: 0.15, GFT_CONVERGENCE_MAX_SPREAD: 2.0,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.82, BEQ_MIN_EDGE: 0.05,
    ESI_BINS: 5, ESI_MAX_ENTROPY: 0.80, ESI_TREND_WEIGHT: 0.25,
    IMAD_MIN_ASYMMETRY: 0.1, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 2, CGSM_LOOKBACK: 8, CGSM_WEIGHT: 0.12,
    ABVC_MIN_AVG_AB: 3.0, ABVC_ELITE_AB: 4.0, ABVC_WEIGHT: 0.10,
    GATE_MIN_GFT_SCORE: 0.45, GATE_MIN_BEQ_EDGE: 0.05,
    GATE_MIN_ESI_STABILITY: 0.15, GATE_MIN_IMAD_SCORE: 0.02,
    GATE_MIN_HIT_RATE: 0.82, GATE_MIN_COMBINED: 0.58,
    GATE_MIN_STREAK: 2, GATE_MIN_ABVC: 0.4,
    SINGLE_MIN_SCORE: 0.68, MULTI_SINGLE_MIN_SCORE: 0.60,
    PARLAY_LEG_MIN_SCORE: 0.55, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.50, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.08,
    MIN_ODDS: -400, MAX_ODDS: -130,
    PREFERRED_ODDS_RANGE: [-400, -130],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
  },
  // B: Volume-first approach
  {
    MIN_GAMES: 7, MIN_AB: 2, WARM_UP_GAMES: 8,
    GFT_WINDOWS: [3, 5, 10], GFT_DECAY_RATE: 0.88, GFT_GRAVITY_STRENGTH: 0.5,
    GFT_MIN_CLEARANCE: 0.10, GFT_CONVERGENCE_MAX_SPREAD: 2.5,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.80, BEQ_MIN_EDGE: 0.03,
    ESI_BINS: 5, ESI_MAX_ENTROPY: 0.85, ESI_TREND_WEIGHT: 0.2,
    IMAD_MIN_ASYMMETRY: 0.1, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 2, CGSM_LOOKBACK: 10, CGSM_WEIGHT: 0.10,
    ABVC_MIN_AVG_AB: 2.5, ABVC_ELITE_AB: 4.0, ABVC_WEIGHT: 0.08,
    GATE_MIN_GFT_SCORE: 0.40, GATE_MIN_BEQ_EDGE: 0.03,
    GATE_MIN_ESI_STABILITY: 0.12, GATE_MIN_IMAD_SCORE: 0.01,
    GATE_MIN_HIT_RATE: 0.80, GATE_MIN_COMBINED: 0.50,
    GATE_MIN_STREAK: 2, GATE_MIN_ABVC: 0.3,
    SINGLE_MIN_SCORE: 0.60, MULTI_SINGLE_MIN_SCORE: 0.55,
    PARLAY_LEG_MIN_SCORE: 0.50, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.60, PARLAY_SAME_GAME_ALLOWED: true,
    PARLAY_MIN_COMBINED_EDGE: 0.05,
    MIN_ODDS: -500, MAX_ODDS: -120,
    PREFERRED_ODDS_RANGE: [-500, -120],
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
    console.log(`*** New overall best ***`);
  }
}

const best = overallBest;
const f = best.results.stats;

console.log('\n' + '='.repeat(60));
console.log('FINAL RESULTS');
console.log('='.repeat(60));
console.log(`Singles: ${f.singles.total} (${(f.singles.accuracy * 100).toFixed(1)}% acc, $${f.singles.pnl}, ${(f.singles.roi * 100).toFixed(1)}% ROI)`);
console.log(`Parlays: ${f.parlays.total} (${(f.parlays.accuracy * 100).toFixed(1)}% acc, $${f.parlays.pnl}, ${(f.parlays.roi * 100).toFixed(1)}% ROI)`);
if (f.parlays.totalLegs > 0) console.log(`  Legs: ${f.parlays.hitLegs}/${f.parlays.totalLegs} (${(f.parlays.legAccuracy * 100).toFixed(1)}%)`);
console.log(`Overall: ${f.overall.total} (${(f.overall.accuracy * 100).toFixed(1)}% acc, $${f.overall.pnl}, ${(f.overall.roi * 100).toFixed(1)}% ROI)`);

const configOutput = {
  config: best.config,
  score: best.score,
  improvements: best.improvements,
  results: { total: f.overall.total, accuracy: f.overall.accuracy, roi: f.overall.roi, pnl: f.overall.pnl, singles: f.singles, parlays: f.parlays },
  optimized_at: new Date().toISOString(),
};

if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.writeFileSync(path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json'), JSON.stringify(configOutput, null, 2));
console.log(`\nConfig saved`);

console.log('\n=== SIGNAL DETAILS ===');
for (const sig of best.results.signals) {
  if (sig.betType === 'single') {
    console.log(`  ${sig.date} ${sig.hit ? 'W' : 'L'} | ${sig.player} O${sig.line} ${sig.statLabel || sig.statType} | ${sig.odds} | actual: ${sig.actual} | streak: ${sig.streak || '?'}`);
  } else {
    const status = sig.hit ? 'W' : 'L';
    console.log(`  ${sig.date} ${status} | PARLAY ${sig.n_legs}L @ ${sig.parlay_american}`);
    for (const l of (sig.legs || [])) console.log(`    ${l.hit ? 'W' : 'L'} | ${l.player} O${l.line} ${l.statLabel || l.stat} | actual: ${l.actual}`);
  }
}
