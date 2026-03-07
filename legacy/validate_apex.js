// Quick validation of APEX backtest (OVER-only)
const fs = require('fs');
const window = {};
const code = fs.readFileSync('./webapp/js/parlay-engine.js', 'utf8');
eval(code);
const APEXModel = window.ParlayEngine.APEXModel;

const data = JSON.parse(fs.readFileSync('./webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = [...data].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

const apex = Object.create(APEXModel);
apex.teamHistory = {};

let totalPicks = 0, wins = 0, pnl = 0;
const tiers = { ELITE: {w:0,l:0,pnl:0}, HIGH: {w:0,l:0,pnl:0}, STRONG: {w:0,l:0,pnl:0} };

for (const game of sorted) {
  const picks = apex.predictGame(game.home_team, game.away_team);
  if (picks) {
    const actualTotal = game.home_score + game.away_score;
    for (const pick of picks) {
      totalPicks++;
      const hit = actualTotal > pick.predTotal;
      if (hit) { wins++; pnl += 91; }
      else { pnl -= 100; }
      tiers[pick.tier][hit ? 'w' : 'l']++;
      tiers[pick.tier].pnl += hit ? 91 : -100;
    }
  }
  apex.updateTeam(game.home_team, game.home_score, game.away_score, game.date, true);
  apex.updateTeam(game.away_team, game.away_score, game.home_score, game.date, false);
}

console.log('=== APEX OVER-Only Validation ===');
console.log(`Total: ${wins}-${totalPicks - wins} (${(wins/totalPicks*100).toFixed(1)}%) PnL: $${pnl} ROI: ${(pnl/(totalPicks*100)*100).toFixed(1)}%`);
for (const tier of ['ELITE','HIGH','STRONG']) {
  const t = tiers[tier];
  const total = t.w + t.l;
  console.log(`  ${tier}: ${t.w}-${t.l} (${total > 0 ? (t.w/total*100).toFixed(1) : 0}%) PnL: $${t.pnl}`);
}
