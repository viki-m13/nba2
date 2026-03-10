#!/usr/bin/env node
// =============================================================================
// Run MLB Ultra Engine v3.0 backtest on real historical data
// Generates signals and stats files for the webapp
// =============================================================================

const fs = require('fs');
const path = require('path');

// Shim browser globals
global.window = global;
global.document = { readyState: 'complete', addEventListener: () => {}, querySelectorAll: () => [] };
global.localStorage = { _data: {}, getItem(k) { return this._data[k] || null; }, setItem(k, v) { this._data[k] = v; } };

// Load MLB Ultra Engine v3.0
eval(fs.readFileSync(path.join(__dirname, '../webapp/js/recommendation-engine-mlb-v3.js'), 'utf8'));

const ENGINE = global.window.MLBRecommendationEngineV3;
const DATA_DIR = path.join(__dirname, '../webapp/data');
const WEBAPP_DATA_DIR = path.join(__dirname, '../../webapp/mlb/data');
const OUTPUT_DIR = path.join(__dirname, '../output');

// Load optimized config if available
try {
  const configPath = path.join(OUTPUT_DIR, 'mlb_ultra_engine_v3_config.json');
  if (fs.existsSync(configPath)) {
    const configData = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    ENGINE.loadConfig(configData);
  }
} catch (e) { console.log('Using default config'); }

// Load data
const boxScores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_player_boxscores.json'), 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_historical_odds.json'), 'utf8'));

console.log('=== MLB Ultra Engine v3.0 Backtest ===');
console.log(`Box scores: ${boxScores.length} games`);
console.log(`Historical odds: ${historicalOdds.length} records`);
console.log(`  With hits props: ${historicalOdds.filter(o => o.hitsProps).length}`);
console.log(`  With TB props: ${historicalOdds.filter(o => o.tbProps).length}`);
console.log(`  With RBI props: ${historicalOdds.filter(o => o.rbiProps).length}`);
console.log(`  With runs props: ${historicalOdds.filter(o => o.runsProps).length}`);

// Run backtest
console.log('\nRunning walk-forward backtest...');
const results = ENGINE.runBacktest(boxScores, historicalOdds);

if (!results) {
  console.log('Backtest returned no results.');
  process.exit(1);
}

console.log('\n=== BACKTEST RESULTS ===');
console.log(`Total signals: ${results.signals.length}`);
console.log(`\nSingles: ${results.stats.singles.total}`);
console.log(`  Wins: ${results.stats.singles.wins}`);
console.log(`  Accuracy: ${(results.stats.singles.accuracy * 100).toFixed(1)}%`);
console.log(`  P&L: $${results.stats.singles.pnl}`);
console.log(`  ROI: ${(results.stats.singles.roi * 100).toFixed(1)}%`);

console.log(`\nParlays: ${results.stats.parlays.total}`);
console.log(`  Wins: ${results.stats.parlays.wins}`);
console.log(`  Accuracy: ${(results.stats.parlays.accuracy * 100).toFixed(1)}%`);
console.log(`  Leg accuracy: ${(results.stats.parlays.legAccuracy * 100).toFixed(1)}% (${results.stats.parlays.hitLegs}/${results.stats.parlays.totalLegs})`);
console.log(`  P&L: $${results.stats.parlays.pnl}`);
console.log(`  ROI: ${(results.stats.parlays.roi * 100).toFixed(1)}%`);

console.log(`\nOverall: ${results.stats.overall.total}`);
console.log(`  Wins: ${results.stats.overall.wins}`);
console.log(`  Accuracy: ${(results.stats.overall.accuracy * 100).toFixed(1)}%`);
console.log(`  P&L: $${results.stats.overall.pnl}`);
console.log(`  ROI: ${(results.stats.overall.roi * 100).toFixed(1)}%`);

// Save signals and stats
const signalsPath1 = path.join(DATA_DIR, 'mlb_ultra_signals_v3.json');
const statsPath1 = path.join(DATA_DIR, 'mlb_ultra_backtest_stats_v3.json');

fs.writeFileSync(signalsPath1, JSON.stringify(results.signals, null, 2));
fs.writeFileSync(statsPath1, JSON.stringify(results.stats, null, 2));

if (fs.existsSync(WEBAPP_DATA_DIR)) {
  fs.writeFileSync(path.join(WEBAPP_DATA_DIR, 'mlb_ultra_signals_v3.json'), JSON.stringify(results.signals, null, 2));
  fs.writeFileSync(path.join(WEBAPP_DATA_DIR, 'mlb_ultra_backtest_stats_v3.json'), JSON.stringify(results.stats, null, 2));
}

console.log(`\nSignals saved to ${signalsPath1}`);
console.log(`Stats saved to ${statsPath1}`);

// Run cross-validation if available
if (typeof ENGINE.runCrossValidation === 'function') {
  console.log('\n=== PURGED WALK-FORWARD CROSS-VALIDATION (5-fold) ===');
  const cvResults = ENGINE.runCrossValidation(boxScores, historicalOdds, 5);
  if (cvResults && cvResults.folds) {
    for (let i = 0; i < cvResults.folds.length; i++) {
      const fold = cvResults.folds[i];
      console.log(`  Fold ${i+1}: ${fold.totalSignals || 0} sigs (${fold.singles || 0}S, ${fold.parlays || 0}P), ${(fold.accuracy * 100).toFixed(1)}% acc, ${(fold.roi * 100).toFixed(1)}% ROI, $${fold.pnl}`);
    }
    if (cvResults.aggregate) {
      const agg = cvResults.aggregate;
      console.log(`  CV Overall: ${agg.totalSignals} sigs, ${(agg.overallAccuracy * 100).toFixed(1)}% acc, ${(agg.overallROI * 100).toFixed(1)}% ROI, $${agg.overallPnl}`);
      console.log(`  CV Mean: ${(agg.meanAccuracy * 100).toFixed(1)}% acc (std: ${(agg.stdAccuracy * 100).toFixed(1)}%), ${(agg.meanROI * 100).toFixed(1)}% ROI (std: ${(agg.stdROI * 100).toFixed(1)}%)`);
      const overfit = Math.abs(results.stats.overall.accuracy - agg.overallAccuracy);
      console.log(`  Overfitting gap: ${(overfit * 100).toFixed(1)}% (threshold: <5% = safe)`);
    }
  }
}

// Run sensitivity analysis if available
if (typeof ENGINE.runSensitivityAnalysis === 'function') {
  console.log('\n=== PARAMETER SENSITIVITY ANALYSIS ===');
  const sensitivity = ENGINE.runSensitivityAnalysis(boxScores, historicalOdds);
  if (sensitivity && sensitivity.sensitivities) {
    const sorted = Object.entries(sensitivity.sensitivities)
      .filter(([k, v]) => typeof v === 'number' && isFinite(v))
      .sort((a, b) => b[1] - a[1]);
    for (const [param, sens] of sorted.slice(0, 10)) {
      console.log(`  ${param}: ${(sens * 100).toFixed(1)}% sensitivity`);
    }
    const maxSens = sorted[0] ? sorted[0][1] : 0;
    console.log(`  Max sensitivity: ${(maxSens * 100).toFixed(1)}% (threshold: <30% = safe)`);
    console.log(`  Overfitting risk: ${maxSens < 0.30 ? 'LOW' : maxSens < 0.50 ? 'MEDIUM' : 'HIGH'}`);
  }
}

// Print signal details
console.log('\n=== SIGNAL DETAILS ===');
for (const sig of results.signals) {
  if (sig.betType === 'single') {
    console.log(`  ${sig.date} ${sig.hit ? 'W' : 'L'} | ${sig.player} O${sig.line} ${sig.statLabel || sig.statType} | ${sig.odds} | actual: ${sig.actual} | score: ${(sig.cascadeScore || 0).toFixed(3)}`);
  } else {
    console.log(`  ${sig.date} ${sig.hit ? 'W' : 'L'} | PARLAY ${sig.n_legs}L @ ${sig.parlay_american}`);
    for (const l of (sig.legs || [])) {
      console.log(`    ${l.hit ? 'W' : 'L'} | ${l.player} O${l.line} ${l.statLabel || l.stat} | actual: ${l.actual}`);
    }
  }
}
