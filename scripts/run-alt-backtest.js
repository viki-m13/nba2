#!/usr/bin/env node
// =============================================================================
// Run Alt-Line Walk-Forward Backtest
// =============================================================================

global.window = {};
global.localStorage = { getItem: () => null, setItem: () => {} };
global.fetch = () => Promise.resolve({ ok: false });
const fs = require('fs');

// Load modules in order
eval(fs.readFileSync('webapp/js/nba-role-engine.js', 'utf8'));
eval(fs.readFileSync('webapp/js/nba-alt-rung-evaluator.js', 'utf8'));
eval(fs.readFileSync('webapp/js/nba-correlation-engine.js', 'utf8'));
eval(fs.readFileSync('webapp/js/nba-screenshot-slip-builder.js', 'utf8'));
eval(fs.readFileSync('webapp/js/engine.js', 'utf8'));
eval(fs.readFileSync('webapp/js/nba-alt-backtest.js', 'utf8'));

const boxScores = JSON.parse(fs.readFileSync('webapp/data/player_boxscores.json', 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync('webapp/data/historical_odds.json', 'utf8'));
let rebAstProps = [];
try { rebAstProps = JSON.parse(fs.readFileSync('webapp/data/historical_reb_ast_props.json', 'utf8')); } catch(e) {}

console.log('Data loaded:', boxScores.length, 'box scores,', historicalOdds.length, 'odds records,', rebAstProps.length, 'reb/ast records');

const results = window.NbaAltBacktest.runBacktest(boxScores, historicalOdds, rebAstProps);

console.log('\n========================================');
console.log('  ALT-LINE WALK-FORWARD BACKTEST');
console.log('========================================');
console.log('Dates processed:', results.dates.length);
console.log('Total legs evaluated:', results.legs.length);
console.log('Total parlays built:', results.parlays.length);

const s = results.stats;
console.log('\n--- OVERALL ---');
console.log('Legs:', JSON.stringify(s.overall.legs, null, 2));
console.log('Parlays:', JSON.stringify(s.overall.parlays, null, 2));

console.log('\n--- BY MARKET ---');
for (const [k, v] of Object.entries(s.byMarket)) {
  if (v.total === 0) continue;
  console.log(`  ${k}: ${v.wins}-${v.losses} (${(v.hitRate*100).toFixed(1)}%) | P&L: $${v.pnl} | ROI: ${v.roi}%`);
}

console.log('\n--- BY ROLE ---');
for (const [k, v] of Object.entries(s.byRole)) {
  if (v.total === 0) continue;
  console.log(`  ${k}: ${v.wins}-${v.losses} (${(v.hitRate*100).toFixed(1)}%) | P&L: $${v.pnl} | ROI: ${v.roi}%`);
}

console.log('\n--- BY BUILDER ---');
for (const [k, v] of Object.entries(s.byBuilder)) {
  if (v.total === 0) continue;
  console.log(`  ${k}: ${v.wins}-${v.losses} (${(v.hitRate*100).toFixed(1)}%) | P&L: $${v.pnl} | ROI: ${v.roi}%`);
}

console.log('\n--- BY EDGE BUCKET ---');
for (const [k, v] of Object.entries(s.byEdge)) {
  if (v.total === 0) continue;
  console.log(`  ${k}: ${v.wins}-${v.losses} (${(v.hitRate*100).toFixed(1)}%) | P&L: $${v.pnl} | ROI: ${v.roi}%`);
}

console.log('\n--- BY NUM LEGS ---');
for (const [k, v] of Object.entries(s.byNumLegs)) {
  if (v.total === 0) continue;
  console.log(`  ${k}-leg: ${v.wins}-${v.losses} (${(v.hitRate*100).toFixed(1)}%) | P&L: $${v.pnl} | ROI: ${v.roi}%`);
}

// Bias check: compare first half vs second half
const midDate = results.dates[Math.floor(results.dates.length / 2)] || '';
const firstHalf = results.legs.filter(l => l.date <= midDate);
const secondHalf = results.legs.filter(l => l.date > midDate);
const calcGroup = (items) => {
  if (items.length === 0) return { total: 0, hitRate: 0, roi: 0 };
  const wins = items.filter(i => i.won).length;
  const pnl = items.reduce((s, i) => s + (i.pnl || 0), 0);
  return { total: items.length, hitRate: (wins/items.length*100).toFixed(1), roi: (pnl/(items.length*100)*100).toFixed(1) };
};

console.log('\n--- BIAS CHECK (first half vs second half) ---');
console.log('  First half:', JSON.stringify(calcGroup(firstHalf)));
console.log('  Second half:', JSON.stringify(calcGroup(secondHalf)));

// Sample legs
console.log('\n--- SAMPLE WINNING LEGS ---');
const sampleWins = results.legs.filter(l => l.won).slice(0, 5);
for (const l of sampleWins) {
  console.log(`  WIN: ${l.player} ${l.marketLabel} OVER ${l.line} | ${l.roleTag} | actual: ${l.actual} | prob: ${l.modelProb} | edge: ${l.edge}`);
}

console.log('\n--- SAMPLE LOSING LEGS ---');
const sampleLosses = results.legs.filter(l => l.won === false).slice(0, 5);
for (const l of sampleLosses) {
  console.log(`  LOSS: ${l.player} ${l.marketLabel} OVER ${l.line} | ${l.roleTag} | actual: ${l.actual} | prob: ${l.modelProb} | edge: ${l.edge}`);
}

// Sample parlays
console.log('\n--- SAMPLE PARLAYS ---');
const sampleParlays = results.parlays.filter(p => p.resolved).slice(0, 5);
for (const p of sampleParlays) {
  const legSummary = p.legs.map(l => `${l.player} ${l.marketLabel} O${l.line}(${l.won ? 'W' : 'L'})`).join(' + ');
  console.log(`  ${p.builder} ${p.numLegs}-leg ${p.won ? 'WIN' : 'LOSS'} @ ${p.odds > 0 ? '+' : ''}${p.odds} | P&L: $${p.pnl}`);
  console.log(`    ${legSummary}`);
}
