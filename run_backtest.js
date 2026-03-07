// Run the Master Bet Recommender backtest with real data
const fs = require('fs');

// Polyfill window for browser-targeted scripts
global.window = global;
global.fetch = () => Promise.resolve({ ok: false });

// Load the engine
eval(fs.readFileSync('webapp/js/parlay-engine.js', 'utf8'));
eval(fs.readFileSync('webapp/js/master-bet-engine.js', 'utf8'));

// Load season data
const seasonData = JSON.parse(fs.readFileSync('data/espn_full_season_2025.json', 'utf8'));
console.log('Season data:', seasonData.length, 'games');

// Load player box scores (check if file has content)
let boxScores = [];
try {
  const raw = fs.readFileSync('webapp/data/player_boxscores.json', 'utf8').trim();
  if (raw && raw.length > 2) {
    boxScores = JSON.parse(raw);
  }
} catch (e) { console.log('No pre-loaded box scores'); }
console.log('Box scores:', boxScores.length, 'games');

// Run the backtest
const results = MasterBetEngine.runOfflineBacktest(seasonData, boxScores);
const s = results.stats;

console.log('\n=== MASTER BET RECOMMENDER BACKTEST RESULTS ===');
console.log('\nIRONLOCK (Heavy Favorite ML Singles):');
console.log('  Total:', s.ironlock.total, '| Wins:', s.ironlock.wins, '| Losses:', s.ironlock.losses);
console.log('  Hit Rate:', (s.ironlock.hitRate * 100).toFixed(1) + '%');
console.log('  P&L:', '$' + s.ironlock.pnl, '| ROI:', s.ironlock.roi + '%');

console.log('\nBEDROCK (Ultra-Safe Player Prop OVER Singles):');
console.log('  Total:', s.bedrock.total, '| Wins:', s.bedrock.wins, '| Losses:', s.bedrock.losses);
console.log('  Hit Rate:', (s.bedrock.hitRate * 100).toFixed(1) + '%');
console.log('  P&L:', '$' + s.bedrock.pnl, '| ROI:', s.bedrock.roi + '%');

console.log('\nAPEX (Confluence Parlays):');
console.log('  Total:', s.apex.total, '| Wins:', s.apex.wins, '| Losses:', s.apex.losses);
console.log('  Hit Rate:', (s.apex.hitRate * 100).toFixed(1) + '%');
console.log('  P&L:', '$' + s.apex.pnl, '| ROI:', s.apex.roi + '%');

console.log('\nOVERALL:');
console.log('  Total:', s.overall.total, '| Wins:', s.overall.wins, '| Losses:', s.overall.losses);
console.log('  Hit Rate:', (s.overall.hitRate * 100).toFixed(1) + '%');
console.log('  P&L:', '$' + s.overall.pnl, '| ROI:', s.overall.roi + '%');
console.log('  Days with picks:', s.daysWithPicks, 'of', s.totalDays, 'total');

// Show some sample IRONLOCK picks
if (results.ironlockPicks.length > 0) {
  console.log('\nSample IRONLOCK picks (last 10):');
  for (const p of results.ironlockPicks.slice(-10)) {
    console.log('  ', p.date, p.team, 'ML', p.odds, '| CDS:', p.cdsScore, '| Edge:', (p.edge*100).toFixed(1)+'%', '| Won:', p.won, '| Score:', p.awayTeam, p.awayScore, '-', p.homeTeam, p.homeScore);
  }
}

// Show sample BEDROCK picks
if (results.bedrockPicks.length > 0) {
  console.log('\nSample BEDROCK picks (last 10):');
  for (const p of results.bedrockPicks.slice(-10)) {
    console.log('  ', p.date, p.player, p.displayLine, p.odds, '| Actual:', p.actual, '| Floor:', p.l10Min, '| Hit:', p.won);
  }
}

// Show sample APEX parlays
if (results.apexParlays.length > 0) {
  console.log('\nSample APEX parlays (last 5):');
  for (const p of results.apexParlays.slice(-5)) {
    console.log('  ', p.date, p.numLegs + '-leg', 'Odds:', p.odds, '| Won:', p.won, '| P&L:', p.pnl);
    for (const l of p.legs) {
      console.log('    ', l.legType, l.team || l.player, l.displayLine || 'ML', '| Won:', l.won);
    }
  }
}

// IRONLOCK losses analysis
console.log('\nIRONLOCK Losses:');
for (const p of results.ironlockPicks.filter(p => !p.won)) {
  console.log('  ', p.date, p.team, 'ML', p.odds, '| CDS:', p.cdsScore, '| NetGap:', p.netGap, '| WinPctGap:', p.winPctGap, '| Edge:', (p.edge*100).toFixed(1)+'%', '| Implied:', (p.impliedProb*100).toFixed(1)+'%', '| Model:', (p.modelProb*100).toFixed(1)+'%');
  console.log('    Score:', p.awayTeam, p.awayScore, '-', p.homeTeam, p.homeScore);
}

// BEDROCK losses analysis
console.log('\nBEDROCK Losses:');
for (const p of results.bedrockPicks.filter(p => !p.won)) {
  console.log('  ', p.date, p.player, p.displayLine, '| Actual:', p.actual, '| L10Avg:', p.l10Avg, '| Floor:', p.l10Min, '| FloorRatio:', p.floorRatio, '| HitRate:', p.hitRate);
}

// APEX losses analysis
console.log('\nAPEX Losses:');
for (const p of results.apexParlays.filter(p => !p.won)) {
  console.log('  ', p.date, p.numLegs + '-leg', 'Odds:', p.odds);
  for (const l of p.legs) {
    const status = l.won ? 'HIT' : 'MISS';
    console.log('    ', l.legType, l.team || l.player, l.displayLine || 'ML', '|', status, l.actual !== undefined ? '| Actual: ' + l.actual : '');
  }
}

// Monthly breakdown
const monthlyData = {};
for (const p of [...results.ironlockPicks, ...results.bedrockPicks, ...results.apexParlays]) {
  const month = p.date.slice(0, 6);
  if (!monthlyData[month]) monthlyData[month] = { total: 0, wins: 0, pnl: 0 };
  monthlyData[month].total++;
  if (p.won) monthlyData[month].wins++;
  monthlyData[month].pnl += p.pnl;
}

console.log('\nMonthly Breakdown:');
for (const [month, data] of Object.entries(monthlyData).sort()) {
  const hitRate = data.total > 0 ? (data.wins / data.total * 100).toFixed(1) : '0.0';
  console.log(`  ${month}: ${data.wins}/${data.total} (${hitRate}%) | P&L: $${data.pnl}`);
}
