// Run backtest using REAL historical FanDuel odds
const fs = require('fs');

global.window = global;
global.fetch = () => Promise.resolve({ ok: false });
global.localStorage = { getItem: () => null, setItem: () => {} };

eval(fs.readFileSync('webapp/js/engine.js', 'utf8'));

const seasonData = JSON.parse(fs.readFileSync('data/espn_full_season_2025.json', 'utf8'));
const boxScores = JSON.parse(fs.readFileSync('webapp/data/player_boxscores.json', 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync('webapp/data/historical_odds.json', 'utf8'));

console.log(`Season: ${seasonData.length} games | Box scores: ${boxScores.length} | Historical odds: ${historicalOdds.length} (${historicalOdds.filter(g => g.playerProps).length} with props)`);
console.log('');

const results = window.BettingEngine.runBacktest(seasonData, boxScores, historicalOdds);
const s = results.stats;

console.log('=== REAL FANDUEL ODDS BACKTEST ===');
console.log('');
console.log('SINGLES (Player Points OVER):');
console.log(`  Record: ${s.singles.wins}-${s.singles.losses} (${(s.singles.hitRate * 100).toFixed(1)}%)`);
console.log(`  P&L: $${s.singles.pnl} | ROI: ${s.singles.roi}%`);
console.log('');
console.log('PARLAYS (2-3 leg):');
console.log(`  Record: ${s.parlays.wins}-${s.parlays.losses} (${(s.parlays.hitRate * 100).toFixed(1)}%)`);
console.log(`  P&L: $${s.parlays.pnl} | ROI: ${s.parlays.roi}%`);
console.log('');
console.log('OVERALL:');
console.log(`  Record: ${s.overall.wins}-${s.overall.losses} (${(s.overall.hitRate * 100).toFixed(1)}%)`);
console.log(`  P&L: $${s.overall.pnl} | ROI: ${s.overall.roi}%`);
console.log(`  Days with picks: ${s.daysWithPicks}/${s.totalDays}`);
console.log('');

// Show all picks
console.log('=== ALL SINGLES ===');
for (const p of results.singles) {
  console.log(`  ${p.date} ${p.player} OVER ${p.line} (${p.odds}) | Actual: ${p.actual} | Floor: ${p.l10Min}/${p.l15Min} | ${p.won ? 'WIN' : 'LOSS'} | P&L: $${p.pnl}`);
}

console.log('');
console.log('=== ALL PARLAYS ===');
for (const p of results.parlays) {
  console.log(`  ${p.date} ${p.numLegs}-leg (${p.odds}) | ${p.won ? 'WIN' : 'LOSS'} | P&L: $${p.pnl}`);
  for (const leg of p.legs) {
    console.log(`    ${leg.player} OVER ${leg.line} (${leg.odds}) | Actual: ${leg.actual} | ${leg.won ? 'HIT' : 'MISS'}`);
  }
}

// Monthly breakdown
console.log('');
console.log('=== MONTHLY BREAKDOWN ===');
const months = {};
for (const p of [...results.singles, ...results.parlays]) {
  const m = p.date.slice(0, 6);
  if (!months[m]) months[m] = { picks: 0, wins: 0, pnl: 0 };
  months[m].picks++;
  if (p.won) months[m].wins++;
  months[m].pnl += p.pnl;
}
for (const [m, d] of Object.entries(months).sort()) {
  console.log(`  ${m}: ${d.wins}/${d.picks} (${(d.wins/d.picks*100).toFixed(0)}%) | P&L: $${d.pnl}`);
}
