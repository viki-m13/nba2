#!/usr/bin/env node
// =============================================================================
// AutoResearch v7 — Accuracy-First Optimizer
// Target: 90%+ accuracy to match NBA-level parlay performance
// Strategy: Accept lower volume for much higher precision
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

console.log('=== AutoResearch v7 — Accuracy-First Optimizer ===');

// Accuracy-first objective: mirrors NBA model's 95%+ accuracy target
function objective(results) {
  if (!results || results.stats.overall.total < 5) return -Infinity;
  const acc = results.stats.overall.accuracy;
  const roi = results.stats.overall.roi;
  const count = results.stats.overall.total;
  const pnl = results.stats.overall.pnl;

  // Below 80% is unacceptable
  if (acc < 0.80) return acc * 50;

  // Accuracy is EVERYTHING (like NBA model)
  let score = acc * 300;

  // Volume: need enough for parlays but don't sacrifice accuracy
  if (count >= 10 && count <= 80) score += 20;
  else if (count >= 7) score += 10;
  else score -= 5;

  // ROI must be positive
  if (roi > 0) score += Math.min(roi * 15, 40);
  else score += roi * 80; // Very heavy penalty for losses

  // Parlay bonus
  const parlays = results.stats.parlays;
  if (parlays.total > 0) {
    score += parlays.total * 5;
    if (parlays.accuracy >= 0.90) score += 30;
    if (parlays.legAccuracy >= 0.95) score += 25;
    if (parlays.roi > 0) score += Math.min(parlays.roi * 10, 30);
  }

  // AGGRESSIVE accuracy bonuses
  if (acc >= 0.97) score += 120;
  else if (acc >= 0.95) score += 90;
  else if (acc >= 0.93) score += 65;
  else if (acc >= 0.90) score += 45;
  else if (acc >= 0.87) score += 25;
  else if (acc >= 0.85) score += 10;

  // Absolute PnL
  if (pnl > 500) score += 10;

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

  const sweeps = {
    // High-impact accuracy controls
    GATE_MIN_HIT_RATE: [0.80, 0.82, 0.85, 0.87, 0.88, 0.90, 0.92, 0.95, 1.0],
    GATE_MIN_COMBINED: [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80],
    GATE_MIN_GFT_SCORE: [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    GATE_MIN_BEQ_EDGE: [0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    GATE_MIN_STREAK: [2, 3, 4, 5, 6, 7, 8],
    GATE_MIN_ABVC: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    GATE_MIN_PBF_MARGIN: [0.03, 0.05, 0.07, 0.10, 0.12, 0.15],
    GATE_MIN_ESI_STABILITY: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    GATE_MIN_IMAD_SCORE: [0.01, 0.02, 0.03, 0.05, 0.07, 0.10],

    // Odds filtering (tighter = more selective)
    MIN_ODDS: [-500, -450, -400, -350, -300, -260, -240, -220],
    MAX_ODDS: [-150, -160, -170, -180, -190, -200, -210, -220],

    // BEQ calibration
    BEQ_CREDIBLE_LEVEL: [0.82, 0.85, 0.87, 0.90, 0.92, 0.95],
    BEQ_MIN_EDGE: [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20],

    // GFT
    GFT_DECAY_RATE: [0.82, 0.85, 0.88, 0.90, 0.92, 0.95],
    GFT_GRAVITY_STRENGTH: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    GFT_MIN_CLEARANCE: [0.02, 0.05, 0.08, 0.10, 0.15, 0.20],
    GFT_CONVERGENCE_MAX_SPREAD: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    GFT_WINDOWS: [[3, 5, 8], [3, 5, 10], [3, 7, 12], [5, 8, 12], [5, 10, 15]],

    // PBF
    PBF_DECAY_RATE: [0.88, 0.90, 0.92, 0.94, 0.95, 0.97],
    PBF_MIN_LAMBDA: [0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
    PBF_WEIGHT: [0.05, 0.08, 0.10, 0.12, 0.15, 0.18],

    // CGSM, ABVC
    CGSM_MIN_STREAK: [2, 3, 4, 5, 6],
    CGSM_LOOKBACK: [5, 7, 8, 10, 12],
    CGSM_WEIGHT: [0.03, 0.05, 0.07, 0.08, 0.10],
    ABVC_MIN_AVG_AB: [2.5, 3.0, 3.2, 3.5, 4.0],
    ABVC_ELITE_AB: [4.0, 4.5, 5.0, 5.5],
    ABVC_WEIGHT: [0.03, 0.05, 0.07, 0.08, 0.10],

    // ESI
    ESI_MAX_ENTROPY: [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    ESI_TREND_WEIGHT: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    ESI_BINS: [4, 5, 6, 7],

    // Player eligibility
    MIN_GAMES: [7, 8, 9, 10, 12, 14, 16],
    MIN_AB: [2, 3, 4],
    WARM_UP_GAMES: [8, 10, 12, 14],

    // OAL, VPD
    OAL_ENABLED: [true, false],
    OAL_WEIGHT: [0.02, 0.03, 0.05, 0.07],
    OAL_MAX_ADJUSTMENT: [0.10, 0.15, 0.20, 0.25],
    VPD_ENABLED: [true, false],
    VPD_MIN_GAMES: [3, 5, 7],
    VPD_WEIGHT: [0.03, 0.05, 0.07, 0.10],
    VPD_MAX_BOOST: [0.05, 0.08, 0.10, 0.12, 0.15],

    // Bet selection (higher = more selective)
    SINGLE_MIN_SCORE: [0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85],
    MULTI_SINGLE_MIN_SCORE: [0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80],
    PARLAY_LEG_MIN_SCORE: [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70],

    // Parlay construction
    PARLAY_MAX_CORRELATION: [0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    PARLAY_SAME_GAME_ALLOWED: [true, false],
    PARLAY_MIN_COMBINED_EDGE: [0.05, 0.08, 0.10, 0.12],
    PARLAY_MAX_LEGS: [2, 3, 4],

    // Contextual
    HOME_BOOST: [0.0, 0.01, 0.015, 0.02, 0.025, 0.03],
  };

  const paramOrder = [
    'GATE_MIN_HIT_RATE', 'GATE_MIN_COMBINED', 'GATE_MIN_BEQ_EDGE',
    'GATE_MIN_GFT_SCORE', 'GATE_MIN_PBF_MARGIN', 'GATE_MIN_STREAK',
    'GATE_MIN_ABVC', 'MIN_ODDS', 'MAX_ODDS',
    'BEQ_CREDIBLE_LEVEL', 'BEQ_MIN_EDGE',
    'GFT_DECAY_RATE', 'GFT_GRAVITY_STRENGTH', 'GFT_MIN_CLEARANCE',
    'GFT_CONVERGENCE_MAX_SPREAD',
    'PBF_DECAY_RATE', 'PBF_MIN_LAMBDA', 'PBF_WEIGHT',
    'CGSM_MIN_STREAK', 'CGSM_LOOKBACK', 'CGSM_WEIGHT',
    'ABVC_MIN_AVG_AB', 'ABVC_ELITE_AB', 'ABVC_WEIGHT',
    'GATE_MIN_ESI_STABILITY', 'GATE_MIN_IMAD_SCORE',
    'ESI_MAX_ENTROPY', 'ESI_TREND_WEIGHT', 'ESI_BINS',
    'MIN_GAMES', 'MIN_AB', 'WARM_UP_GAMES',
    'OAL_ENABLED', 'OAL_WEIGHT', 'OAL_MAX_ADJUSTMENT',
    'VPD_ENABLED', 'VPD_MIN_GAMES', 'VPD_WEIGHT', 'VPD_MAX_BOOST',
    'SINGLE_MIN_SCORE', 'MULTI_SINGLE_MIN_SCORE', 'PARLAY_LEG_MIN_SCORE',
    'PARLAY_MAX_CORRELATION', 'PARLAY_SAME_GAME_ALLOWED',
    'PARLAY_MIN_COMBINED_EDGE', 'PARLAY_MAX_LEGS',
    'HOME_BOOST', 'GFT_WINDOWS',
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

// Start from the best v6 optimized config + tight accuracy configs
const starts = [
  // Start A: From best v6 result, tighten for accuracy
  {
    MIN_GAMES: 10, MIN_AB: 3, WARM_UP_GAMES: 12,
    GFT_WINDOWS: [3, 7, 12], GFT_DECAY_RATE: 0.90, GFT_GRAVITY_STRENGTH: 0.45,
    GFT_MIN_CLEARANCE: 0.10, GFT_CONVERGENCE_MAX_SPREAD: 2.0,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.92, BEQ_MIN_EDGE: 0.10,
    ESI_BINS: 5, ESI_MAX_ENTROPY: 0.70, ESI_TREND_WEIGHT: 0.30,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 3, CGSM_LOOKBACK: 10, CGSM_WEIGHT: 0.05,
    ABVC_MIN_AVG_AB: 3.5, ABVC_ELITE_AB: 5.0, ABVC_WEIGHT: 0.05,
    PBF_DECAY_RATE: 0.92, PBF_MIN_LAMBDA: 0.4, PBF_WEIGHT: 0.12,
    PBF_MIN_PROB_MARGIN: 0.05,
    OAL_ENABLED: true, OAL_WEIGHT: 0.03, OAL_MAX_ADJUSTMENT: 0.20,
    VPD_ENABLED: true, VPD_MIN_GAMES: 5, VPD_WEIGHT: 0.05, VPD_MAX_BOOST: 0.10,
    GATE_MIN_GFT_SCORE: 0.60, GATE_MIN_BEQ_EDGE: 0.10,
    GATE_MIN_ESI_STABILITY: 0.20, GATE_MIN_IMAD_SCORE: 0.03,
    GATE_MIN_HIT_RATE: 0.85, GATE_MIN_COMBINED: 0.60,
    GATE_MIN_STREAK: 3, GATE_MIN_ABVC: 0.40,
    GATE_MIN_PBF_MARGIN: 0.05,
    SINGLE_MIN_SCORE: 0.75, MULTI_SINGLE_MIN_SCORE: 0.70,
    PARLAY_LEG_MIN_SCORE: 0.62, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 3,
    PARLAY_MAX_CORRELATION: 0.35, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.08,
    MIN_ODDS: -300, MAX_ODDS: -200,
    PREFERRED_ODDS_RANGE: [-400, -120],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
    HOME_BOOST: 0.015, BACK_TO_BACK_PENALTY: 0.00,
  },
  // Start B: Ultra-selective from v6 best
  {
    MIN_GAMES: 12, MIN_AB: 3, WARM_UP_GAMES: 14,
    GFT_WINDOWS: [5, 10, 15], GFT_DECAY_RATE: 0.92, GFT_GRAVITY_STRENGTH: 0.50,
    GFT_MIN_CLEARANCE: 0.15, GFT_CONVERGENCE_MAX_SPREAD: 1.5,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.95, BEQ_MIN_EDGE: 0.15,
    ESI_BINS: 6, ESI_MAX_ENTROPY: 0.65, ESI_TREND_WEIGHT: 0.35,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 5, CGSM_LOOKBACK: 12, CGSM_WEIGHT: 0.05,
    ABVC_MIN_AVG_AB: 3.5, ABVC_ELITE_AB: 5.0, ABVC_WEIGHT: 0.05,
    PBF_DECAY_RATE: 0.94, PBF_MIN_LAMBDA: 0.5, PBF_WEIGHT: 0.10,
    PBF_MIN_PROB_MARGIN: 0.05,
    OAL_ENABLED: false, OAL_WEIGHT: 0.03, OAL_MAX_ADJUSTMENT: 0.20,
    VPD_ENABLED: true, VPD_MIN_GAMES: 7, VPD_WEIGHT: 0.05, VPD_MAX_BOOST: 0.08,
    GATE_MIN_GFT_SCORE: 0.70, GATE_MIN_BEQ_EDGE: 0.15,
    GATE_MIN_ESI_STABILITY: 0.25, GATE_MIN_IMAD_SCORE: 0.05,
    GATE_MIN_HIT_RATE: 0.90, GATE_MIN_COMBINED: 0.70,
    GATE_MIN_STREAK: 5, GATE_MIN_ABVC: 0.50,
    GATE_MIN_PBF_MARGIN: 0.07,
    SINGLE_MIN_SCORE: 0.80, MULTI_SINGLE_MIN_SCORE: 0.75,
    PARLAY_LEG_MIN_SCORE: 0.68, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 3,
    PARLAY_MAX_CORRELATION: 0.30, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.10,
    MIN_ODDS: -260, MAX_ODDS: -200,
    PREFERRED_ODDS_RANGE: [-300, -150],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
    HOME_BOOST: 0.02, BACK_TO_BACK_PENALTY: 0.00,
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
console.log('FINAL RESULTS — MLB Ultra Engine v2.0 (Accuracy-Optimized)');
console.log('='.repeat(60));
console.log(`Singles: ${f.singles.total} (${(f.singles.accuracy * 100).toFixed(1)}% acc, $${f.singles.pnl}, ${(f.singles.roi * 100).toFixed(1)}% ROI)`);
console.log(`Parlays: ${f.parlays.total} (${(f.parlays.accuracy * 100).toFixed(1)}% acc, $${f.parlays.pnl}, ${(f.parlays.roi * 100).toFixed(1)}% ROI)`);
if (f.parlays.totalLegs > 0) console.log(`  Legs: ${f.parlays.hitLegs}/${f.parlays.totalLegs} (${(f.parlays.legAccuracy * 100).toFixed(1)}%)`);
console.log(`Overall: ${f.overall.total} (${(f.overall.accuracy * 100).toFixed(1)}% acc, $${f.overall.pnl}, ${(f.overall.roi * 100).toFixed(1)}% ROI)`);
console.log(`Optimization score: ${best.score.toFixed(1)} (${best.improvements} improvements)`);

// Compare to v6 results
const v6Path = path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json');
if (fs.existsSync(v6Path)) {
  const v6 = JSON.parse(fs.readFileSync(v6Path, 'utf8'));
  console.log(`\n--- Comparison to v6 ---`);
  console.log(`v6: ${v6.results.total} sigs, ${(v6.results.accuracy * 100).toFixed(1)}% acc, $${v6.results.pnl}, ${(v6.results.roi * 100).toFixed(1)}% ROI`);
}

// Only save if better than current
const currentConfig = fs.existsSync(v6Path) ? JSON.parse(fs.readFileSync(v6Path, 'utf8')) : null;
const shouldSave = !currentConfig || best.score > currentConfig.score;

if (shouldSave) {
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
    version: '2.0-accuracy',
  };

  fs.writeFileSync(v6Path, JSON.stringify(configOutput, null, 2));

  // Save signals
  fs.writeFileSync(path.join(DATA_DIR, 'mlb_ultra_signals.json'), JSON.stringify(best.results.signals, null, 2));
  fs.writeFileSync(path.join(DATA_DIR, 'mlb_ultra_backtest_stats.json'), JSON.stringify(best.results.stats, null, 2));

  const webappDir = path.join(__dirname, '../../webapp/mlb/data');
  if (fs.existsSync(webappDir)) {
    fs.writeFileSync(path.join(webappDir, 'mlb_ultra_signals.json'), JSON.stringify(best.results.signals, null, 2));
    fs.writeFileSync(path.join(webappDir, 'mlb_ultra_backtest_stats.json'), JSON.stringify(best.results.stats, null, 2));
  }
  console.log('\nConfig saved (better than v6)');
} else {
  console.log('\nv6 config retained (better score)');
}

// Print signal details
console.log('\n=== SIGNAL DETAILS ===');
for (const sig of best.results.signals) {
  if (sig.betType === 'single') {
    console.log(`  ${sig.date} ${sig.hit ? 'W' : 'L'} | ${sig.player} O${sig.line} ${sig.statLabel || sig.statType} | ${sig.odds} | actual: ${sig.actual}`);
  } else {
    console.log(`  ${sig.date} ${sig.hit ? 'W' : 'L'} | PARLAY ${sig.n_legs}L @ ${sig.parlay_american}`);
    for (const l of (sig.legs || [])) console.log(`    ${l.hit ? 'W' : 'L'} | ${l.player} O${l.line} ${l.statLabel || l.stat} | actual: ${l.actual}`);
  }
}
