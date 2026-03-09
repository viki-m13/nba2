#!/usr/bin/env node
// =============================================================================
// AutoResearch v2 — Extreme Selectivity Optimizer for MLB Ultra Engine
// Focus: Achieve 97%+ accuracy by being ultra-selective
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

console.log('=== AutoResearch v2 — Extreme Selectivity ===');
console.log(`Data: ${boxScores.length} games, ${historicalOdds.length} odds records`);

function objective(results) {
  if (!results || results.stats.overall.total < 3) return -Infinity;
  const acc = results.stats.overall.accuracy;
  const roi = results.stats.overall.roi;
  const count = results.stats.overall.total;

  // Primary: accuracy (heavily weighted)
  // Secondary: ROI must be positive
  // Tertiary: want reasonable signal count
  let score = acc * 200;  // Double weight on accuracy
  if (roi > 0) score += Math.min(roi * 15, 50);
  else score -= Math.abs(roi) * 30;
  if (count < 5) score -= 30;
  if (count >= 10) score += 5;
  if (count >= 20) score += 5;

  // Big bonus for hitting accuracy targets
  if (acc >= 0.97) score += 50;
  else if (acc >= 0.95) score += 30;
  else if (acc >= 0.90) score += 15;

  return score;
}

function runWithConfig(configOverrides) {
  const origConfig = { ...ENGINE.CONFIG };
  Object.assign(ENGINE.CONFIG, configOverrides);
  const results = ENGINE.runBacktest(boxScores, historicalOdds);
  Object.assign(ENGINE.CONFIG, origConfig);
  return results;
}

// Aggressive parameter space focused on extreme selectivity
const PARAM_SPACE = {
  MIN_GAMES: [5, 8, 10, 12, 15, 18, 20],
  MIN_AB: [2, 3, 4],
  WARM_UP_GAMES: [8, 10, 12, 15, 18, 20],
  GFT_WINDOWS: [[3, 5, 10], [5, 10, 15], [3, 7, 12], [5, 8, 12], [3, 5, 8]],
  GFT_DECAY_RATE: [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98],
  GFT_GRAVITY_STRENGTH: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
  GFT_MIN_CLEARANCE: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2],
  GFT_CONVERGENCE_MAX_SPREAD: [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0],
  BEQ_CREDIBLE_LEVEL: [0.75, 0.80, 0.85, 0.87, 0.90, 0.93, 0.95],
  BEQ_MIN_EDGE: [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
  ESI_MAX_ENTROPY: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
  ESI_TREND_WEIGHT: [0.1, 0.2, 0.3, 0.4, 0.5],
  GATE_MIN_GFT_SCORE: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
  GATE_MIN_BEQ_EDGE: [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
  GATE_MIN_ESI_STABILITY: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
  GATE_MIN_IMAD_SCORE: [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
  GATE_MIN_HIT_RATE: [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 1.0],
  GATE_MIN_COMBINED: [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
  SINGLE_MIN_SCORE: [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
  MULTI_SINGLE_MIN_SCORE: [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
  MIN_ODDS: [-800, -600, -500, -400, -350, -300, -250, -200],
  MAX_ODDS: [-100, -110, -120, -130, -140, -150, -160, -170, -180, -200],
};

function ratchetOptimize(startConfig, maxRounds) {
  let bestConfig = { ...startConfig };
  let bestResults = runWithConfig(bestConfig);
  let bestScore = objective(bestResults);
  let totalImprovements = 0;

  console.log(`Start: ${bestResults.stats.overall.total} signals, ${(bestResults.stats.overall.accuracy * 100).toFixed(1)}% acc, ${(bestResults.stats.overall.roi * 100).toFixed(1)}% ROI, score=${bestScore.toFixed(1)}`);

  const paramOrder = [
    'GATE_MIN_HIT_RATE', 'GATE_MIN_COMBINED', 'MIN_ODDS', 'MAX_ODDS',
    'GATE_MIN_GFT_SCORE', 'GATE_MIN_BEQ_EDGE', 'GFT_MIN_CLEARANCE',
    'BEQ_MIN_EDGE', 'SINGLE_MIN_SCORE', 'MULTI_SINGLE_MIN_SCORE',
    'MIN_GAMES', 'WARM_UP_GAMES', 'MIN_AB',
    'GFT_DECAY_RATE', 'GFT_GRAVITY_STRENGTH', 'GFT_CONVERGENCE_MAX_SPREAD',
    'GATE_MIN_ESI_STABILITY', 'GATE_MIN_IMAD_SCORE',
    'ESI_MAX_ENTROPY', 'ESI_TREND_WEIGHT', 'BEQ_CREDIBLE_LEVEL',
    'GFT_WINDOWS',
  ];

  for (let round = 0; round < maxRounds; round++) {
    let roundImprovements = 0;

    for (const param of paramOrder) {
      if (!PARAM_SPACE[param]) continue;

      for (const val of PARAM_SPACE[param]) {
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
          console.log(`  ${param}: ${JSON.stringify(val)} | ${bestResults.stats.overall.total} signals, ${(bestResults.stats.overall.accuracy * 100).toFixed(1)}% acc, ${(bestResults.stats.overall.roi * 100).toFixed(1)}% ROI, score=${bestScore.toFixed(1)}`);
        }
      }
    }

    console.log(`Round ${round + 1}: ${roundImprovements} improvements`);
    if (roundImprovements === 0) break;
  }

  return { config: bestConfig, results: bestResults, score: bestScore, improvements: totalImprovements };
}

// Try multiple starting points to avoid local optima
function multiStart() {
  const starts = [
    // Start 1: Heavy favorites focus
    {
      MIN_GAMES: 10, MIN_AB: 3, WARM_UP_GAMES: 12,
      GFT_WINDOWS: [5, 10, 15], GFT_DECAY_RATE: 0.85, GFT_GRAVITY_STRENGTH: 0.4,
      GFT_MIN_CLEARANCE: 0.3, GFT_CONVERGENCE_MAX_SPREAD: 1.0,
      BEQ_CREDIBLE_LEVEL: 0.87, BEQ_MIN_EDGE: 0.05,
      ESI_BINS: 6, ESI_MAX_ENTROPY: 0.83, ESI_TREND_WEIGHT: 0.3,
      IMAD_MIN_ASYMMETRY: 0.1, IMAD_VOLUME_DISCOUNT: 0.02,
      GATE_MIN_GFT_SCORE: 0.6, GATE_MIN_BEQ_EDGE: 0.05,
      GATE_MIN_ESI_STABILITY: 0.4, GATE_MIN_IMAD_SCORE: 0.02,
      GATE_MIN_HIT_RATE: 0.90, GATE_MIN_COMBINED: 0.75,
      SINGLE_MIN_SCORE: 0.75, MULTI_SINGLE_MIN_SCORE: 0.70,
      PARLAY_LEG_MIN_SCORE: 0.6, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
      PARLAY_MAX_CORRELATION: 0.33, PARLAY_SAME_GAME_ALLOWED: false,
      PARLAY_MIN_COMBINED_EDGE: 0.1,
      MIN_ODDS: -600, MAX_ODDS: -130,
      PREFERRED_ODDS_RANGE: [-600, -150],
      UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
    },
    // Start 2: Ultra-tight gates
    {
      MIN_GAMES: 8, MIN_AB: 2, WARM_UP_GAMES: 10,
      GFT_WINDOWS: [3, 5, 10], GFT_DECAY_RATE: 0.90, GFT_GRAVITY_STRENGTH: 0.5,
      GFT_MIN_CLEARANCE: 0.5, GFT_CONVERGENCE_MAX_SPREAD: 2.0,
      BEQ_CREDIBLE_LEVEL: 0.90, BEQ_MIN_EDGE: 0.10,
      ESI_BINS: 6, ESI_MAX_ENTROPY: 0.7, ESI_TREND_WEIGHT: 0.3,
      IMAD_MIN_ASYMMETRY: 0.1, IMAD_VOLUME_DISCOUNT: 0.02,
      GATE_MIN_GFT_SCORE: 0.7, GATE_MIN_BEQ_EDGE: 0.10,
      GATE_MIN_ESI_STABILITY: 0.5, GATE_MIN_IMAD_SCORE: 0.05,
      GATE_MIN_HIT_RATE: 0.95, GATE_MIN_COMBINED: 0.80,
      SINGLE_MIN_SCORE: 0.80, MULTI_SINGLE_MIN_SCORE: 0.75,
      PARLAY_LEG_MIN_SCORE: 0.7, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
      PARLAY_MAX_CORRELATION: 0.33, PARLAY_SAME_GAME_ALLOWED: false,
      PARLAY_MIN_COMBINED_EDGE: 0.1,
      MIN_ODDS: -500, MAX_ODDS: -150,
      PREFERRED_ODDS_RANGE: [-500, -150],
      UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
    },
    // Start 3: Perfect hit rate focus
    {
      MIN_GAMES: 12, MIN_AB: 3, WARM_UP_GAMES: 15,
      GFT_WINDOWS: [5, 10, 15], GFT_DECAY_RATE: 0.92, GFT_GRAVITY_STRENGTH: 0.3,
      GFT_MIN_CLEARANCE: 0.2, GFT_CONVERGENCE_MAX_SPREAD: 1.5,
      BEQ_CREDIBLE_LEVEL: 0.85, BEQ_MIN_EDGE: 0.15,
      ESI_BINS: 6, ESI_MAX_ENTROPY: 0.8, ESI_TREND_WEIGHT: 0.3,
      IMAD_MIN_ASYMMETRY: 0.1, IMAD_VOLUME_DISCOUNT: 0.02,
      GATE_MIN_GFT_SCORE: 0.5, GATE_MIN_BEQ_EDGE: 0.15,
      GATE_MIN_ESI_STABILITY: 0.3, GATE_MIN_IMAD_SCORE: 0.03,
      GATE_MIN_HIT_RATE: 1.0, GATE_MIN_COMBINED: 0.70,
      SINGLE_MIN_SCORE: 0.70, MULTI_SINGLE_MIN_SCORE: 0.65,
      PARLAY_LEG_MIN_SCORE: 0.6, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
      PARLAY_MAX_CORRELATION: 0.33, PARLAY_SAME_GAME_ALLOWED: false,
      PARLAY_MIN_COMBINED_EDGE: 0.1,
      MIN_ODDS: -800, MAX_ODDS: -160,
      PREFERRED_ODDS_RANGE: [-800, -150],
      UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
    },
  ];

  let overallBest = null;

  for (let i = 0; i < starts.length; i++) {
    console.log(`\n${'#'.repeat(60)}`);
    console.log(`START POINT ${i + 1}/${starts.length}`);
    console.log(`${'#'.repeat(60)}`);

    const result = ratchetOptimize(starts[i], 6);

    if (!overallBest || result.score > overallBest.score) {
      overallBest = result;
      console.log(`*** New overall best: ${(result.results.stats.overall.accuracy * 100).toFixed(1)}% acc ***`);
    }
  }

  return overallBest;
}

const best = multiStart();

console.log('\n' + '='.repeat(60));
console.log('FINAL OPTIMIZED RESULTS');
console.log('='.repeat(60));
console.log(`Total signals: ${best.results.stats.overall.total}`);
console.log(`Accuracy: ${(best.results.stats.overall.accuracy * 100).toFixed(1)}%`);
console.log(`P&L: $${best.results.stats.overall.pnl}`);
console.log(`ROI: ${(best.results.stats.overall.roi * 100).toFixed(1)}%`);
console.log(`Singles: ${best.results.stats.singles.total} (${(best.results.stats.singles.accuracy * 100).toFixed(1)}%)`);
console.log(`Parlays: ${best.results.stats.parlays.total}`);
console.log(`Improvements: ${best.improvements}`);

// Save
const configOutput = {
  config: best.config,
  score: best.score,
  improvements: best.improvements,
  results: {
    total: best.results.stats.overall.total,
    accuracy: best.results.stats.overall.accuracy,
    roi: best.results.stats.overall.roi,
    pnl: best.results.stats.overall.pnl,
  },
  optimized_at: new Date().toISOString(),
};

if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.writeFileSync(path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json'), JSON.stringify(configOutput, null, 2));
console.log(`\nConfig saved to ${path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json')}`);

console.log('\n=== OPTIMIZED CONFIG ===');
for (const [k, v] of Object.entries(best.config)) {
  console.log(`  ${k}: ${JSON.stringify(v)}`);
}
