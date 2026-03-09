#!/usr/bin/env node
// =============================================================================
// AutoResearch v8 — Balanced Accuracy + Parlays
// Target: 85%+ accuracy WITH parlays for maximum ROI
// Key insight from NBA: parlays at 90%+ accuracy drive massive ROI
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

console.log('=== AutoResearch v8 — Balanced Accuracy + Parlays ===');

// Balanced objective: requires 85%+ accuracy AND parlays
function objective(results) {
  if (!results || results.stats.overall.total < 5) return -Infinity;
  const acc = results.stats.overall.accuracy;
  const roi = results.stats.overall.roi;
  const count = results.stats.overall.total;
  const pnl = results.stats.overall.pnl;
  const parlays = results.stats.parlays;

  // Hard floor: 80% minimum
  if (acc < 0.80) return acc * 30;

  // Core: accuracy × PnL together
  let score = acc * 200;

  // REQUIRE parlays (this is what drives NBA-level ROI)
  if (parlays.total >= 3) {
    score += 100; // Big bonus for having parlays
    score += parlays.total * 10;
    if (parlays.accuracy >= 0.90) score += 50;
    if (parlays.legAccuracy >= 0.95) score += 40;
    if (parlays.roi > 0) score += Math.min(parlays.roi * 20, 80);
  } else if (parlays.total >= 1) {
    score += 30;
  } else {
    score -= 50; // Penalize no parlays
  }

  // Volume: need enough for parlays
  if (count >= 15 && count <= 80) score += 30;
  else if (count >= 10) score += 15;
  else score -= 10;

  // ROI must be positive
  if (roi > 0) score += Math.min(roi * 20, 50);
  else score += roi * 60;

  // Accuracy bonuses
  if (acc >= 0.95) score += 80;
  else if (acc >= 0.93) score += 55;
  else if (acc >= 0.90) score += 35;
  else if (acc >= 0.87) score += 20;
  else if (acc >= 0.85) score += 10;

  // PnL bonus
  if (pnl > 2000) score += 30;
  else if (pnl > 1000) score += 15;
  else if (pnl > 500) score += 5;

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
  console.log(`Start: ${r.total} sigs (${p.total}P), ${(r.accuracy * 100).toFixed(1)}% acc, $${r.pnl}, ${(r.roi * 100).toFixed(1)}% ROI | score=${bestScore.toFixed(1)}`);

  const sweeps = {
    GATE_MIN_HIT_RATE: [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95, 1.0],
    GATE_MIN_COMBINED: [0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80],
    GATE_MIN_GFT_SCORE: [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    GATE_MIN_BEQ_EDGE: [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20],
    GATE_MIN_STREAK: [1, 2, 3, 4, 5, 6, 7, 8],
    GATE_MIN_ABVC: [0.20, 0.30, 0.40, 0.50, 0.60],
    GATE_MIN_PBF_MARGIN: [0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
    GATE_MIN_ESI_STABILITY: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    GATE_MIN_IMAD_SCORE: [0.005, 0.01, 0.02, 0.03, 0.05],
    MIN_ODDS: [-500, -400, -350, -300, -260, -220],
    MAX_ODDS: [-140, -150, -160, -170, -180, -200],
    BEQ_CREDIBLE_LEVEL: [0.80, 0.85, 0.87, 0.90, 0.92, 0.95],
    BEQ_MIN_EDGE: [0.03, 0.05, 0.07, 0.10, 0.15],
    GFT_DECAY_RATE: [0.82, 0.85, 0.88, 0.90, 0.92, 0.95],
    GFT_GRAVITY_STRENGTH: [0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    GFT_MIN_CLEARANCE: [0.01, 0.02, 0.05, 0.10, 0.15],
    GFT_CONVERGENCE_MAX_SPREAD: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    GFT_WINDOWS: [[3, 5, 8], [3, 5, 10], [3, 7, 12], [5, 8, 12], [5, 10, 15]],
    PBF_DECAY_RATE: [0.88, 0.90, 0.92, 0.94, 0.95, 0.97],
    PBF_MIN_LAMBDA: [0.2, 0.3, 0.4, 0.5],
    PBF_WEIGHT: [0.05, 0.08, 0.10, 0.12, 0.15],
    CGSM_MIN_STREAK: [1, 2, 3, 4, 5],
    CGSM_LOOKBACK: [5, 8, 10, 12],
    CGSM_WEIGHT: [0.03, 0.05, 0.07, 0.08, 0.10],
    ABVC_MIN_AVG_AB: [2.0, 2.5, 3.0, 3.5, 4.0],
    ABVC_ELITE_AB: [4.0, 4.5, 5.0, 5.5],
    ABVC_WEIGHT: [0.03, 0.05, 0.07, 0.10],
    ESI_MAX_ENTROPY: [0.55, 0.65, 0.75, 0.85, 0.95],
    ESI_TREND_WEIGHT: [0.15, 0.25, 0.30, 0.40],
    MIN_GAMES: [7, 8, 10, 12],
    MIN_AB: [1, 2, 3],
    WARM_UP_GAMES: [8, 10, 12],
    OAL_ENABLED: [true, false],
    OAL_WEIGHT: [0.02, 0.05, 0.07],
    OAL_MAX_ADJUSTMENT: [0.15, 0.20, 0.25],
    VPD_ENABLED: [true, false],
    VPD_MIN_GAMES: [3, 5, 7],
    VPD_WEIGHT: [0.03, 0.05, 0.10],
    VPD_MAX_BOOST: [0.05, 0.10, 0.15],
    SINGLE_MIN_SCORE: [0.68, 0.72, 0.75, 0.78, 0.80],
    MULTI_SINGLE_MIN_SCORE: [0.62, 0.65, 0.68, 0.72, 0.75],
    PARLAY_LEG_MIN_SCORE: [0.50, 0.55, 0.58, 0.60, 0.65],
    PARLAY_MAX_CORRELATION: [0.25, 0.30, 0.40, 0.50, 0.60],
    PARLAY_SAME_GAME_ALLOWED: [true, false],
    PARLAY_MIN_COMBINED_EDGE: [0.03, 0.05, 0.08, 0.10],
    PARLAY_MAX_LEGS: [2, 3, 4],
    HOME_BOOST: [0.0, 0.01, 0.02, 0.03],
  };

  const paramOrder = [
    'MIN_ODDS', 'MAX_ODDS',
    'GATE_MIN_HIT_RATE', 'GATE_MIN_COMBINED', 'GATE_MIN_BEQ_EDGE',
    'GATE_MIN_GFT_SCORE', 'GATE_MIN_PBF_MARGIN', 'GATE_MIN_STREAK',
    'GATE_MIN_ABVC',
    'BEQ_CREDIBLE_LEVEL', 'BEQ_MIN_EDGE',
    'PARLAY_LEG_MIN_SCORE', 'PARLAY_MAX_CORRELATION', 'PARLAY_SAME_GAME_ALLOWED',
    'PARLAY_MIN_COMBINED_EDGE', 'PARLAY_MAX_LEGS',
    'SINGLE_MIN_SCORE', 'MULTI_SINGLE_MIN_SCORE',
    'GFT_DECAY_RATE', 'GFT_GRAVITY_STRENGTH', 'GFT_MIN_CLEARANCE',
    'GFT_CONVERGENCE_MAX_SPREAD',
    'PBF_DECAY_RATE', 'PBF_MIN_LAMBDA', 'PBF_WEIGHT',
    'CGSM_MIN_STREAK', 'CGSM_LOOKBACK', 'CGSM_WEIGHT',
    'ABVC_MIN_AVG_AB', 'ABVC_ELITE_AB', 'ABVC_WEIGHT',
    'GATE_MIN_ESI_STABILITY', 'GATE_MIN_IMAD_SCORE',
    'ESI_MAX_ENTROPY', 'ESI_TREND_WEIGHT',
    'MIN_GAMES', 'MIN_AB', 'WARM_UP_GAMES',
    'OAL_ENABLED', 'OAL_WEIGHT', 'OAL_MAX_ADJUSTMENT',
    'VPD_ENABLED', 'VPD_MIN_GAMES', 'VPD_WEIGHT', 'VPD_MAX_BOOST',
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

// Starting configs that balance accuracy with parlay opportunity
const starts = [
  // Start A: Medium selectivity with wide enough for parlays
  {
    MIN_GAMES: 8, MIN_AB: 2, WARM_UP_GAMES: 10,
    GFT_WINDOWS: [3, 5, 10], GFT_DECAY_RATE: 0.90, GFT_GRAVITY_STRENGTH: 0.45,
    GFT_MIN_CLEARANCE: 0.05, GFT_CONVERGENCE_MAX_SPREAD: 2.5,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.90, BEQ_MIN_EDGE: 0.07,
    ESI_BINS: 5, ESI_MAX_ENTROPY: 0.75, ESI_TREND_WEIGHT: 0.25,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 2, CGSM_LOOKBACK: 10, CGSM_WEIGHT: 0.07,
    ABVC_MIN_AVG_AB: 3.0, ABVC_ELITE_AB: 5.0, ABVC_WEIGHT: 0.07,
    PBF_DECAY_RATE: 0.92, PBF_MIN_LAMBDA: 0.3, PBF_WEIGHT: 0.10,
    PBF_MIN_PROB_MARGIN: 0.05,
    OAL_ENABLED: true, OAL_WEIGHT: 0.05, OAL_MAX_ADJUSTMENT: 0.20,
    VPD_ENABLED: true, VPD_MIN_GAMES: 5, VPD_WEIGHT: 0.05, VPD_MAX_BOOST: 0.10,
    GATE_MIN_GFT_SCORE: 0.55, GATE_MIN_BEQ_EDGE: 0.07,
    GATE_MIN_ESI_STABILITY: 0.15, GATE_MIN_IMAD_SCORE: 0.02,
    GATE_MIN_HIT_RATE: 0.85, GATE_MIN_COMBINED: 0.55,
    GATE_MIN_STREAK: 3, GATE_MIN_ABVC: 0.35,
    GATE_MIN_PBF_MARGIN: 0.03,
    SINGLE_MIN_SCORE: 0.72, MULTI_SINGLE_MIN_SCORE: 0.65,
    PARLAY_LEG_MIN_SCORE: 0.58, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.40, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.05,
    MIN_ODDS: -350, MAX_ODDS: -170,
    PREFERRED_ODDS_RANGE: [-400, -130],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
    HOME_BOOST: 0.015, BACK_TO_BACK_PENALTY: 0.00,
  },
  // Start B: From the v6 best (which had good parlays) but tighter
  {
    MIN_GAMES: 7, MIN_AB: 2, WARM_UP_GAMES: 8,
    GFT_WINDOWS: [3, 7, 12], GFT_DECAY_RATE: 0.88, GFT_GRAVITY_STRENGTH: 0.45,
    GFT_MIN_CLEARANCE: 0.05, GFT_CONVERGENCE_MAX_SPREAD: 3.0,
    BEQ_PRIOR_ALPHA: 1.0, BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.87, BEQ_MIN_EDGE: 0.05,
    ESI_BINS: 5, ESI_MAX_ENTROPY: 0.70, ESI_TREND_WEIGHT: 0.30,
    IMAD_MIN_ASYMMETRY: 0.05, IMAD_VOLUME_DISCOUNT: 0.02,
    CGSM_MIN_STREAK: 2, CGSM_LOOKBACK: 8, CGSM_WEIGHT: 0.08,
    ABVC_MIN_AVG_AB: 2.5, ABVC_ELITE_AB: 4.5, ABVC_WEIGHT: 0.06,
    PBF_DECAY_RATE: 0.92, PBF_MIN_LAMBDA: 0.3, PBF_WEIGHT: 0.12,
    PBF_MIN_PROB_MARGIN: 0.05,
    OAL_ENABLED: false, OAL_WEIGHT: 0.03, OAL_MAX_ADJUSTMENT: 0.20,
    VPD_ENABLED: true, VPD_MIN_GAMES: 5, VPD_WEIGHT: 0.05, VPD_MAX_BOOST: 0.12,
    GATE_MIN_GFT_SCORE: 0.50, GATE_MIN_BEQ_EDGE: 0.05,
    GATE_MIN_ESI_STABILITY: 0.12, GATE_MIN_IMAD_SCORE: 0.01,
    GATE_MIN_HIT_RATE: 0.82, GATE_MIN_COMBINED: 0.55,
    GATE_MIN_STREAK: 2, GATE_MIN_ABVC: 0.30,
    GATE_MIN_PBF_MARGIN: 0.02,
    SINGLE_MIN_SCORE: 0.70, MULTI_SINGLE_MIN_SCORE: 0.62,
    PARLAY_LEG_MIN_SCORE: 0.55, PARLAY_MIN_LEGS: 2, PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.50, PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.05,
    MIN_ODDS: -400, MAX_ODDS: -160,
    PREFERRED_ODDS_RANGE: [-500, -120],
    UNIT_SIZE: 100, MAX_DAILY_UNITS: 5, KELLY_FRACTION: 0.25,
    HOME_BOOST: 0.02, BACK_TO_BACK_PENALTY: 0.00,
  },
];

let overallBest = null;

for (let i = 0; i < starts.length; i++) {
  console.log(`\n${'#'.repeat(60)}`);
  console.log(`START POINT ${i + 1}/${starts.length}`);
  console.log(`${'#'.repeat(60)}`);
  const result = ratchetOptimize(starts[i], 12);
  if (!overallBest || result.score > overallBest.score) {
    overallBest = result;
    console.log(`*** New overall best: score=${result.score.toFixed(1)} ***`);
  }
}

const best = overallBest;
const f = best.results.stats;

console.log('\n' + '='.repeat(60));
console.log('FINAL RESULTS — MLB Ultra Engine v2.0 (Balanced)');
console.log('='.repeat(60));
console.log(`Singles: ${f.singles.total} (${(f.singles.accuracy * 100).toFixed(1)}% acc, $${f.singles.pnl}, ${(f.singles.roi * 100).toFixed(1)}% ROI)`);
console.log(`Parlays: ${f.parlays.total} (${(f.parlays.accuracy * 100).toFixed(1)}% acc, $${f.parlays.pnl}, ${(f.parlays.roi * 100).toFixed(1)}% ROI)`);
if (f.parlays.totalLegs > 0) console.log(`  Legs: ${f.parlays.hitLegs}/${f.parlays.totalLegs} (${(f.parlays.legAccuracy * 100).toFixed(1)}%)`);
console.log(`Overall: ${f.overall.total} (${(f.overall.accuracy * 100).toFixed(1)}% acc, $${f.overall.pnl}, ${(f.overall.roi * 100).toFixed(1)}% ROI)`);

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
  version: '2.0-balanced',
};

fs.writeFileSync(path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json'), JSON.stringify(configOutput, null, 2));
fs.writeFileSync(path.join(DATA_DIR, 'mlb_ultra_signals.json'), JSON.stringify(best.results.signals, null, 2));
fs.writeFileSync(path.join(DATA_DIR, 'mlb_ultra_backtest_stats.json'), JSON.stringify(best.results.stats, null, 2));
const webappDir = path.join(__dirname, '../../webapp/mlb/data');
if (fs.existsSync(webappDir)) {
  fs.writeFileSync(path.join(webappDir, 'mlb_ultra_signals.json'), JSON.stringify(best.results.signals, null, 2));
  fs.writeFileSync(path.join(webappDir, 'mlb_ultra_backtest_stats.json'), JSON.stringify(best.results.stats, null, 2));
}
console.log('\nConfig saved');

console.log('\n=== SIGNAL DETAILS ===');
for (const sig of best.results.signals) {
  if (sig.betType === 'single') {
    console.log(`  ${sig.date} ${sig.hit ? 'W' : 'L'} | ${sig.player} O${sig.line} ${sig.statLabel || sig.statType} | ${sig.odds} | actual: ${sig.actual}`);
  } else {
    console.log(`  ${sig.date} ${sig.hit ? 'W' : 'L'} | PARLAY ${sig.n_legs}L @ ${sig.parlay_american}`);
    for (const l of (sig.legs || [])) console.log(`    ${l.hit ? 'W' : 'L'} | ${l.player} O${l.line} ${l.statLabel || l.stat} | actual: ${l.actual}`);
  }
}

console.log('\n=== KEY PARAMS ===');
const kp = ['MIN_ODDS', 'MAX_ODDS', 'BEQ_CREDIBLE_LEVEL', 'GATE_MIN_HIT_RATE',
  'GATE_MIN_COMBINED', 'GATE_MIN_GFT_SCORE', 'GATE_MIN_BEQ_EDGE', 'GATE_MIN_PBF_MARGIN',
  'GATE_MIN_STREAK', 'PBF_WEIGHT', 'SINGLE_MIN_SCORE', 'PARLAY_LEG_MIN_SCORE'];
for (const p of kp) console.log(`  ${p}: ${JSON.stringify(best.config[p])}`);
