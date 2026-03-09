#!/usr/bin/env node
// =============================================================================
// Validate MLB Ultra Engine on expanded dataset
// Runs detailed analysis comparing in-sample vs out-of-sample performance
// =============================================================================

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'webapp', 'data');

const signals = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_ultra_signals.json'), 'utf8'));
const boxscores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_player_boxscores.json'), 'utf8'));
const odds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_historical_odds.json'), 'utf8'));

const boxDates = [...new Set(boxscores.map(b => b.date))].sort();
const oddsDates = [...new Set(odds.map(o => o.date))].sort();
const propsDates = [...new Set(odds.filter(o => o.hitsProps).map(o => o.date))].sort();

console.log('╔══════════════════════════════════════════════════════════════╗');
console.log('║       MLB ULTRA ENGINE — EXPANDED VALIDATION REPORT        ║');
console.log('╚══════════════════════════════════════════════════════════════╝');
console.log();

console.log('=== DATA COVERAGE ===');
console.log(`  Boxscores: ${boxscores.length} games, ${boxDates.length} dates`);
console.log(`  Range: ${boxDates[0]} - ${boxDates[boxDates.length - 1]}`);
console.log(`  Odds: ${odds.length} records, ${oddsDates.length} dates`);
console.log(`  With batter props: ${odds.filter(o => o.hitsProps).length} records, ${propsDates.length} dates`);
console.log();

console.log('=== OVERALL PERFORMANCE ===');
console.log(`  Total signals: ${signals.length}`);
console.log(`  Wins: ${signals.filter(s => s.hit).length}`);
console.log(`  Losses: ${signals.filter(s => !s.hit).length}`);
const acc = (signals.filter(s => s.hit).length / signals.length * 100).toFixed(1);
console.log(`  Accuracy: ${acc}%`);
const totalPnl = signals.reduce((a, s) => a + (s.pnl || 0), 0);
console.log(`  Total P&L: $${totalPnl.toFixed(0)}`);
console.log(`  ROI: ${(totalPnl / (signals.length * 100) * 100).toFixed(1)}%`);
console.log(`  Break-even accuracy needed (at avg -205 odds): ~67.2%`);
console.log();

// Monthly breakdown
console.log('=== MONTHLY BREAKDOWN ===');
const byMonth = {};
for (const s of signals) {
  const m = s.date.substring(0, 6);
  if (!byMonth[m]) byMonth[m] = { total: 0, wins: 0, pnl: 0 };
  byMonth[m].total++;
  if (s.hit) byMonth[m].wins++;
  byMonth[m].pnl += (s.pnl || 0);
}
for (const [m, d] of Object.entries(byMonth).sort()) {
  const mAcc = (d.wins / d.total * 100).toFixed(1);
  const label = m.substring(0, 4) + '-' + m.substring(4);
  const sign = d.pnl >= 0 ? '+' : '';
  console.log(`  ${label}: ${String(d.total).padStart(2)} signals, ${String(d.wins).padStart(2)} wins (${mAcc.padStart(5)}%), P&L: $${sign}${d.pnl.toFixed(0)}`);
}
console.log();

// In-sample vs out-of-sample
const inSample = signals.filter(s => s.date >= '20250815' && s.date <= '20250915');
const outOfSample = signals.filter(s => s.date < '20250815' || s.date > '20250915');

console.log('=== IN-SAMPLE vs OUT-OF-SAMPLE ===');
console.log(`  In-sample (Aug 15-Sep 15, original training period):`);
console.log(`    Signals: ${inSample.length}, Wins: ${inSample.filter(s => s.hit).length}, Accuracy: ${inSample.length > 0 ? (inSample.filter(s => s.hit).length / inSample.length * 100).toFixed(1) : 'N/A'}%`);
console.log(`    P&L: $${inSample.reduce((a, s) => a + (s.pnl || 0), 0).toFixed(0)}`);
console.log(`  Out-of-sample (rest of season):`);
console.log(`    Signals: ${outOfSample.length}, Wins: ${outOfSample.filter(s => s.hit).length}, Accuracy: ${outOfSample.length > 0 ? (outOfSample.filter(s => s.hit).length / outOfSample.length * 100).toFixed(1) : 'N/A'}%`);
console.log(`    P&L: $${outOfSample.reduce((a, s) => a + (s.pnl || 0), 0).toFixed(0)}`);
console.log();

// Signal details
console.log('=== ALL SIGNALS ===');
for (const s of signals) {
  const result = s.hit ? 'WIN ' : 'LOSS';
  const pnlSign = (s.pnl || 0) >= 0 ? '+' : '';
  console.log(`  ${s.date} ${(s.player || '').padEnd(22)} ${s.statType || s.stat} O${s.line} (${s.odds}) ${result} ${pnlSign}$${(s.pnl || 0).toFixed(0)}`);
}
console.log();

// Cascade scores
const winScores = signals.filter(s => s.hit).map(s => s.cascadeScore);
const lossScores = signals.filter(s => !s.hit).map(s => s.cascadeScore);
console.log('=== CASCADE SCORE: WINS vs LOSSES ===');
console.log(`  Win avg:  ${(winScores.reduce((a, b) => a + b, 0) / winScores.length).toFixed(3)}`);
console.log(`  Loss avg: ${(lossScores.reduce((a, b) => a + b, 0) / lossScores.length).toFixed(3)}`);
console.log();

// Verdict
const overallAcc = signals.filter(s => s.hit).length / signals.length;
console.log('=== VERDICT ===');
if (overallAcc >= 0.80) {
  console.log('  STRONG: Model achieves 80%+ accuracy across full season');
} else if (overallAcc >= 0.70) {
  console.log('  MODERATE: 70-80% accuracy. Profitable at -200/-210 odds but below 100% claim.');
} else if (overallAcc >= 0.67) {
  console.log('  MARGINAL: Barely breaks even at these odds');
} else {
  console.log('  UNPROFITABLE: Below break-even threshold');
}
