#!/usr/bin/env node
// =============================================================================
// AutoResearch-Style Monotonic Ratchet Optimizer for MLB Ultra Engine
// =============================================================================
// Implements the monotonic ratchet optimization from karpathy/autoresearch:
// 1. Start with current parameters
// 2. Try systematic variations of each parameter
// 3. Keep changes that improve the objective (accuracy + ROI)
// 4. Repeat until convergence
// =============================================================================

const fs = require('fs');
const path = require('path');

// Shim browser globals
global.window = global;
global.document = { readyState: 'complete', addEventListener: () => {}, querySelectorAll: () => [] };
global.localStorage = { _data: {}, getItem(k) { return this._data[k] || null; }, setItem(k, v) { this._data[k] = v; } };

// Load MLB Ultra Engine
eval(fs.readFileSync(path.join(__dirname, '../webapp/js/recommendation-engine-mlb.js'), 'utf8'));

const ENGINE = global.window.MLBRecommendationEngine;
const DATA_DIR = path.join(__dirname, '../webapp/data');
const OUTPUT_DIR = path.join(__dirname, '../output');

// Load data once
const boxScores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_player_boxscores.json'), 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_historical_odds.json'), 'utf8'));

console.log('=== AutoResearch MLB Engine Optimizer ===');
console.log(`Data: ${boxScores.length} games, ${historicalOdds.length} odds records`);

// Objective function: maximize accuracy with ROI constraint
function objective(results) {
  if (!results || results.signals.length < 5) return -Infinity;

  const acc = results.stats.overall.accuracy;
  const roi = results.stats.overall.roi;
  const count = results.stats.overall.total;

  // Target: 97%+ accuracy, 200%+ ROI, minimum 10 signals
  // Score = accuracy * 100 + ROI bonus + count bonus
  // Heavy penalty if accuracy < 0.90
  // Heavy penalty if count < 10

  let score = acc * 100;

  // ROI bonus (capped)
  if (roi > 0) score += Math.min(roi * 20, 60);
  else score += roi * 50; // Heavy penalty for negative ROI

  // Count bonus - want at least 10-30 signals for statistical significance
  if (count < 5) score -= 50;
  else if (count < 10) score -= 20;
  else if (count >= 15) score += 5;
  else if (count >= 25) score += 10;

  return score;
}

function runWithConfig(configOverrides) {
  // Reset config to base and apply overrides
  const origConfig = { ...ENGINE.CONFIG };
  Object.assign(ENGINE.CONFIG, configOverrides);

  const results = ENGINE.runBacktest(boxScores, historicalOdds);

  // Restore original config
  Object.assign(ENGINE.CONFIG, origConfig);

  return results;
}

// Parameter search space for MLB-specific optimization
const PARAM_SPACE = {
  // Player eligibility - baseball has fewer games in our dataset
  MIN_GAMES: [5, 8, 10, 12, 15],
  MIN_AB: [1, 2, 3],
  WARM_UP_GAMES: [8, 10, 12, 15, 18],

  // GFT - adjusted for MLB stat scale (hits 0-4 vs NBA points 15-45)
  GFT_WINDOWS: [[3, 5, 10], [5, 10, 15], [3, 7, 12], [5, 8, 12]],
  GFT_DECAY_RATE: [0.85, 0.90, 0.92, 0.95, 0.98],
  GFT_GRAVITY_STRENGTH: [0.2, 0.3, 0.4, 0.5, 0.6],
  GFT_MIN_CLEARANCE: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
  GFT_CONVERGENCE_MAX_SPREAD: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],

  // BEQ
  BEQ_CREDIBLE_LEVEL: [0.80, 0.85, 0.87, 0.90, 0.95],
  BEQ_MIN_EDGE: [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],

  // ESI
  ESI_MAX_ENTROPY: [0.6, 0.7, 0.8, 0.9, 1.0],
  ESI_TREND_WEIGHT: [0.2, 0.3, 0.4, 0.5],

  // IMAD
  IMAD_MIN_ASYMMETRY: [0.05, 0.1, 0.15, 0.2],

  // Quality Gates - these are the most important for accuracy
  GATE_MIN_GFT_SCORE: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  GATE_MIN_BEQ_EDGE: [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
  GATE_MIN_ESI_STABILITY: [0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
  GATE_MIN_IMAD_SCORE: [0.01, 0.02, 0.05, 0.08, 0.1],
  GATE_MIN_HIT_RATE: [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
  GATE_MIN_COMBINED: [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],

  // Bet Type Selection - higher = more selective
  SINGLE_MIN_SCORE: [0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
  MULTI_SINGLE_MIN_SCORE: [0.55, 0.6, 0.65, 0.7, 0.75],

  // Odds Filters - critical for MLB
  MIN_ODDS: [-600, -500, -400, -324, -250, -200],
  MAX_ODDS: [-100, -110, -120, -130, -150],
};

// Phase 1: Coarse grid search for the most impactful parameters
function coarseSearch() {
  console.log('\n=== PHASE 1: Coarse Grid Search ===');

  // Start with MLB-optimized defaults
  let bestConfig = {
    MIN_GAMES: 10,
    MIN_AB: 2,
    WARM_UP_GAMES: 12,
    GFT_WINDOWS: [5, 10, 15],
    GFT_DECAY_RATE: 0.92,
    GFT_GRAVITY_STRENGTH: 0.4,
    GFT_MIN_CLEARANCE: 0.3,
    GFT_CONVERGENCE_MAX_SPREAD: 4.0,
    BEQ_CREDIBLE_LEVEL: 0.87,
    BEQ_MIN_EDGE: 0.05,
    ESI_BINS: 6,
    ESI_MAX_ENTROPY: 0.83,
    ESI_TREND_WEIGHT: 0.3,
    IMAD_MIN_ASYMMETRY: 0.1,
    IMAD_VOLUME_DISCOUNT: 0.02,
    GATE_MIN_GFT_SCORE: 0.4,
    GATE_MIN_BEQ_EDGE: 0.05,
    GATE_MIN_ESI_STABILITY: 0.15,
    GATE_MIN_IMAD_SCORE: 0.02,
    GATE_MIN_HIT_RATE: 0.85,
    GATE_MIN_COMBINED: 0.55,
    SINGLE_MIN_SCORE: 0.75,
    MULTI_SINGLE_MIN_SCORE: 0.67,
    PARLAY_LEG_MIN_SCORE: 0.6,
    PARLAY_MIN_LEGS: 2,
    PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.33,
    PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.1,
    MIN_ODDS: -500,
    MAX_ODDS: -110,
    PREFERRED_ODDS_RANGE: [-500, -150],
    UNIT_SIZE: 100,
    MAX_DAILY_UNITS: 5,
    KELLY_FRACTION: 0.25,
  };

  let bestResults = runWithConfig(bestConfig);
  let bestScore = objective(bestResults);

  console.log(`Initial: ${bestResults.stats.overall.total} signals, ${(bestResults.stats.overall.accuracy * 100).toFixed(1)}% acc, ${(bestResults.stats.overall.roi * 100).toFixed(1)}% ROI, score=${bestScore.toFixed(1)}`);

  return { config: bestConfig, results: bestResults, score: bestScore };
}

// Phase 2: Monotonic ratchet - iterate through each parameter
function ratchetOptimize(initial) {
  console.log('\n=== PHASE 2: Monotonic Ratchet Optimization ===');

  let bestConfig = { ...initial.config };
  let bestScore = initial.score;
  let bestResults = initial.results;
  let improvements = 0;

  // Priority order: most impactful parameters first
  const paramOrder = [
    'GATE_MIN_HIT_RATE',
    'MIN_ODDS',
    'MAX_ODDS',
    'GATE_MIN_COMBINED',
    'GATE_MIN_GFT_SCORE',
    'GATE_MIN_BEQ_EDGE',
    'GFT_MIN_CLEARANCE',
    'BEQ_MIN_EDGE',
    'MIN_GAMES',
    'WARM_UP_GAMES',
    'GFT_DECAY_RATE',
    'GFT_GRAVITY_STRENGTH',
    'GFT_CONVERGENCE_MAX_SPREAD',
    'GATE_MIN_ESI_STABILITY',
    'GATE_MIN_IMAD_SCORE',
    'ESI_MAX_ENTROPY',
    'ESI_TREND_WEIGHT',
    'BEQ_CREDIBLE_LEVEL',
    'SINGLE_MIN_SCORE',
    'MULTI_SINGLE_MIN_SCORE',
    'MIN_AB',
    'GFT_WINDOWS',
  ];

  const MAX_ROUNDS = 5;

  for (let round = 0; round < MAX_ROUNDS; round++) {
    console.log(`\n--- Round ${round + 1} ---`);
    let roundImprovements = 0;

    for (const param of paramOrder) {
      if (!PARAM_SPACE[param]) continue;

      const values = PARAM_SPACE[param];
      let paramBest = bestConfig[param];
      let paramBestScore = bestScore;
      let paramBestResults = bestResults;

      for (const val of values) {
        // Skip if same as current
        if (JSON.stringify(val) === JSON.stringify(bestConfig[param])) continue;

        const testConfig = { ...bestConfig, [param]: val };
        const results = runWithConfig(testConfig);
        const score = objective(results);

        if (score > paramBestScore) {
          paramBest = val;
          paramBestScore = score;
          paramBestResults = results;
        }
      }

      if (paramBestScore > bestScore) {
        const oldVal = bestConfig[param];
        bestConfig[param] = paramBest;
        bestScore = paramBestScore;
        bestResults = paramBestResults;
        improvements++;
        roundImprovements++;
        console.log(`  ${param}: ${JSON.stringify(oldVal)} -> ${JSON.stringify(paramBest)} | ${bestResults.stats.overall.total} signals, ${(bestResults.stats.overall.accuracy * 100).toFixed(1)}% acc, ${(bestResults.stats.overall.roi * 100).toFixed(1)}% ROI, score=${bestScore.toFixed(1)}`);
      }
    }

    console.log(`Round ${round + 1}: ${roundImprovements} improvements`);
    if (roundImprovements === 0) {
      console.log('Converged!');
      break;
    }
  }

  return { config: bestConfig, results: bestResults, score: bestScore, improvements };
}

// Phase 3: Fine-tuning with smaller step sizes
function fineTune(initial) {
  console.log('\n=== PHASE 3: Fine-Tuning ===');

  let bestConfig = { ...initial.config };
  let bestScore = initial.score;
  let bestResults = initial.results;
  let improvements = 0;

  // Fine-tune numeric parameters with smaller steps
  const fineParams = {
    GATE_MIN_HIT_RATE: [-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05],
    GATE_MIN_COMBINED: [-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05],
    GATE_MIN_GFT_SCORE: [-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05],
    GATE_MIN_BEQ_EDGE: [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03],
    GFT_MIN_CLEARANCE: [-0.1, -0.05, 0.05, 0.1],
    BEQ_MIN_EDGE: [-0.02, -0.01, 0.01, 0.02],
    GFT_DECAY_RATE: [-0.02, -0.01, 0.01, 0.02],
    GFT_GRAVITY_STRENGTH: [-0.05, -0.02, 0.02, 0.05],
    MIN_ODDS: [-50, -25, 25, 50],
    MAX_ODDS: [-10, -5, 5, 10],
  };

  for (let round = 0; round < 3; round++) {
    let roundImprovements = 0;

    for (const [param, deltas] of Object.entries(fineParams)) {
      for (const delta of deltas) {
        const val = bestConfig[param] + delta;
        const testConfig = { ...bestConfig, [param]: val };
        const results = runWithConfig(testConfig);
        const score = objective(results);

        if (score > bestScore) {
          bestConfig[param] = val;
          bestScore = score;
          bestResults = results;
          improvements++;
          roundImprovements++;
          console.log(`  ${param}: ${(val - delta).toFixed(4)} -> ${val.toFixed(4)} | ${bestResults.stats.overall.total} signals, ${(bestResults.stats.overall.accuracy * 100).toFixed(1)}% acc, ${(bestResults.stats.overall.roi * 100).toFixed(1)}% ROI`);
        }
      }
    }

    if (roundImprovements === 0) break;
  }

  return { config: bestConfig, results: bestResults, score: bestScore, improvements: initial.improvements + improvements };
}

// Main optimization loop
function main() {
  const phase1 = coarseSearch();
  const phase2 = ratchetOptimize(phase1);
  const phase3 = fineTune(phase2);

  const finalConfig = phase3.config;
  const finalResults = phase3.results;

  console.log('\n' + '='.repeat(60));
  console.log('OPTIMIZATION COMPLETE');
  console.log('='.repeat(60));
  console.log(`Total improvements: ${phase3.improvements}`);
  console.log(`\nFinal Results:`);
  console.log(`  Total signals: ${finalResults.stats.overall.total}`);
  console.log(`  Accuracy: ${(finalResults.stats.overall.accuracy * 100).toFixed(1)}%`);
  console.log(`  P&L: $${finalResults.stats.overall.pnl}`);
  console.log(`  ROI: ${(finalResults.stats.overall.roi * 100).toFixed(1)}%`);
  console.log(`  Singles: ${finalResults.stats.singles.total} (${(finalResults.stats.singles.accuracy * 100).toFixed(1)}%)`);
  console.log(`  Parlays: ${finalResults.stats.parlays.total}`);

  // Save optimized config
  const configOutput = {
    config: finalConfig,
    score: phase3.score,
    improvements: phase3.improvements,
    results: {
      total: finalResults.stats.overall.total,
      accuracy: finalResults.stats.overall.accuracy,
      roi: finalResults.stats.overall.roi,
      pnl: finalResults.stats.overall.pnl,
    },
    optimized_at: new Date().toISOString(),
  };

  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json'), JSON.stringify(configOutput, null, 2));
  console.log(`\nConfig saved to ${path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json')}`);

  // Print final config
  console.log('\n=== OPTIMIZED CONFIG ===');
  for (const [k, v] of Object.entries(finalConfig)) {
    console.log(`  ${k}: ${JSON.stringify(v)}`);
  }
}

main();
