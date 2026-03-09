#!/usr/bin/env node
// =============================================================================
// Run MLB Ultra Engine backtest on real historical data
// Generates signals and stats files for the webapp
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
const WEBAPP_DATA_DIR = path.join(__dirname, '../../webapp/mlb/data');
const OUTPUT_DIR = path.join(__dirname, '../output');

// Load optimized config
try {
  const configPath = path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json');
  if (fs.existsSync(configPath)) {
    const configData = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    ENGINE.loadConfig(configData);
  }
} catch (e) { console.log('Using default config'); }

// Load data
const boxScores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_player_boxscores.json'), 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_historical_odds.json'), 'utf8'));

console.log('=== MLB Ultra Engine Backtest ===');
console.log(`Box scores: ${boxScores.length} games`);
console.log(`Historical odds: ${historicalOdds.length} records`);
console.log(`  With hits props: ${historicalOdds.filter(o => o.hitsProps).length}`);
console.log(`  With TB props: ${historicalOdds.filter(o => o.tbProps).length}`);

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
const signalsPath1 = path.join(DATA_DIR, 'mlb_ultra_signals.json');
const signalsPath2 = path.join(WEBAPP_DATA_DIR, 'mlb_ultra_signals.json');
const statsPath1 = path.join(DATA_DIR, 'mlb_ultra_backtest_stats.json');
const statsPath2 = path.join(WEBAPP_DATA_DIR, 'mlb_ultra_backtest_stats.json');

fs.writeFileSync(signalsPath1, JSON.stringify(results.signals, null, 2));
fs.writeFileSync(signalsPath2, JSON.stringify(results.signals, null, 2));
fs.writeFileSync(statsPath1, JSON.stringify(results.stats, null, 2));
fs.writeFileSync(statsPath2, JSON.stringify(results.stats, null, 2));

console.log(`\nSignals saved to ${signalsPath1}`);
console.log(`Stats saved to ${statsPath1}`);
