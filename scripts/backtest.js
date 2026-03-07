#!/usr/bin/env node
// Run backtest using the v4 engine
const fs = require('fs');
const path = require('path');

global.window = global;
global.localStorage = { _data: {}, getItem(k) { return this._data[k] || null; }, setItem(k, v) { this._data[k] = v; } };
global.fetch = async () => ({ ok: false, headers: { get: () => null }, json: async () => null });

eval(fs.readFileSync(path.join(__dirname, '../webapp/js/engine.js'), 'utf8'));

const DATA_DIR = path.join(__dirname, '../webapp/data');
const seasonData = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'espn_full_season_2025.json'), 'utf8'));
const boxScores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'player_boxscores.json'), 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'historical_odds.json'), 'utf8'));
const rebAstProps = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'historical_reb_ast_props.json'), 'utf8'));

console.log(`Data: ${seasonData.length} games, ${boxScores.length} box scores, ${historicalOdds.length} odds, ${rebAstProps.length} reb/ast`);

const results = window.BettingEngine.runBacktest(seasonData, boxScores, historicalOdds, rebAstProps);

console.log('\n=== BACKTEST RESULTS (v4 Engine) ===\n');
console.log('Overall:', results.stats.overall);
console.log('Singles:', results.stats.singles);
console.log('Parlays:', results.stats.parlays);
console.log('Points:', results.stats.points);
console.log('Rebounds:', results.stats.rebounds);
console.log('Assists:', results.stats.assists);
console.log(`Days: ${results.stats.daysWithPicks}/${results.stats.totalDays}`);

if (results.stats.recent30) console.log('Recent 30:', results.stats.recent30);
if (results.stats.recent14) console.log('Recent 14:', results.stats.recent14);
if (results.stats.recent7) console.log('Recent 7:', results.stats.recent7);

// Parlay breakdown by leg count
const byLegs = {};
for (const p of results.parlays) {
  const n = p.numLegs || p.legs.length;
  if (!byLegs[n]) byLegs[n] = { w: 0, t: 0, pnl: 0 };
  byLegs[n].t++;
  byLegs[n].pnl += p.pnl;
  if (p.won) byLegs[n].w++;
}

console.log('\n=== BY LEG COUNT ===');
for (const n of Object.keys(byLegs).sort()) {
  const s = byLegs[n];
  const wr = (s.w / s.t * 100).toFixed(1);
  const roi = (s.pnl / (s.t * 100) * 100).toFixed(1);
  console.log(`  ${n}-leg: ${s.w}-${s.t - s.w} (${wr}%), P&L: $${s.pnl >= 0 ? '+' : ''}${s.pnl}, ROI: ${roi}%`);
}

// Train/test split analysis
const allDates = [...new Set(results.parlays.map(p => p.date))].sort();
const splitIdx = Math.floor(allDates.length * 0.75);
const trainDates = new Set(allDates.slice(0, splitIdx));
const testDates = new Set(allDates.slice(splitIdx));

const trainParlays = results.parlays.filter(p => trainDates.has(p.date));
const testParlays = results.parlays.filter(p => testDates.has(p.date));

const summarize = (parlays, label) => {
  const w = parlays.filter(p => p.won).length;
  const t = parlays.length;
  const pnl = parlays.reduce((s, p) => s + p.pnl, 0);
  const wr = t > 0 ? (w / t * 100).toFixed(1) : '0.0';
  const roi = t > 0 ? (pnl / (t * 100) * 100).toFixed(1) : '0.0';
  console.log(`${label}: ${w}-${t - w} (${wr}%), P&L: $${pnl >= 0 ? '+' : ''}${pnl}, ROI: ${roi}%`);
};

console.log('\n=== TRAIN/TEST SPLIT (75/25) ===');
console.log(`Train dates: ${allDates[0]} to ${allDates[splitIdx - 1]} (${trainDates.size} dates)`);
console.log(`Test dates:  ${allDates[splitIdx]} to ${allDates[allDates.length - 1]} (${testDates.size} dates)`);
summarize(trainParlays, 'TRAIN');
summarize(testParlays, 'TEST ');

// Test by leg count
console.log('\nTest set by leg count:');
for (const n of Object.keys(byLegs).sort()) {
  const parlays = testParlays.filter(p => (p.numLegs || p.legs.length) == n);
  if (parlays.length === 0) continue;
  const w = parlays.filter(p => p.won).length;
  const pnl = parlays.reduce((s, p) => s + p.pnl, 0);
  const roi = (pnl / (parlays.length * 100) * 100).toFixed(1);
  console.log(`  ${n}-leg: ${w}-${parlays.length - w} (${(w/parlays.length*100).toFixed(1)}%), P&L: $${pnl >= 0 ? '+' : ''}${pnl}, ROI: ${roi}%`);
}

// Sample recent parlays
console.log('\n=== SAMPLE RECENT PARLAYS ===');
const recent = results.parlays.slice(-10);
for (const p of recent) {
  const legs = p.legs.map(l => `${l.player} ${l.statLabel} O${l.line}→${l.actual}`).join(', ');
  console.log(`${p.date} ${p.won ? 'WIN' : 'LOSS'} ${p.odds > 0 ? '+' : ''}${p.odds} | ${legs}`);
}
