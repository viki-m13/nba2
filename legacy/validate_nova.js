// Validate NOVA model — final version with tiered bonus signals
const fs = require('fs');
const data = JSON.parse(fs.readFileSync('./webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = [...data].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

const teams = {};
function updateTeam(t, pf, pa, date) {
  if (!teams[t]) teams[t] = [];
  teams[t].push({ pf, pa, total: pf + pa, date });
  if (teams[t].length > 30) teams[t] = teams[t].slice(-20);
}
function getWindow(t, n) {
  const g = (teams[t] || []).slice(-n);
  if (g.length < n) return null;
  return { ppg: g.reduce((s,x)=>s+x.pf,0)/n, oppg: g.reduce((s,x)=>s+x.pa,0)/n, avgTotal: g.reduce((s,x)=>s+x.total,0)/n };
}

const results = {};
function ensureTier(t) { if (!results[t]) results[t] = {w:0,l:0,months:{}}; }

for (const game of sorted) {
  const h = game.home_team, a = game.away_team;
  const h3 = getWindow(h,3), h5 = getWindow(h,5), h10 = getWindow(h,10);
  const a3 = getWindow(a,3), a5 = getWindow(a,5), a10 = getWindow(a,10);

  if (h3 && h5 && h10 && a3 && a5 && a10) {
    const predTotal = Math.round(((h10.ppg + a10.ppg + h10.oppg + a10.oppg) / 2) * 10) / 10;
    const actualTotal = game.home_score + game.away_score;
    const month = game.date.slice(0, 6);

    // Core compound filter: both PPG 118+ L5 AND both avg total 225+ L3
    if (h5.ppg > 118 && a5.ppg > 118 && h3.avgTotal > 225 && a3.avgTotal > 225) {
      let bonusCount = 0;
      if (h5.oppg > 113 || a5.oppg > 113) bonusCount++;
      if (h3.ppg > h5.ppg) bonusCount++;
      if (a3.ppg > a5.ppg) bonusCount++;
      if (h5.ppg > 120 || a5.ppg > 120) bonusCount++;
      if (h3.avgTotal > 230 || a3.avgTotal > 230) bonusCount++;

      let tier;
      if (bonusCount >= 3) tier = 'ELITE';
      else if (bonusCount >= 1) tier = 'HIGH';
      else tier = 'STRONG';

      const hit = actualTotal > predTotal;
      ensureTier(tier);
      ensureTier('ALL');
      results[tier][hit?'w':'l']++;
      results['ALL'][hit?'w':'l']++;
      if (!results[tier].months[month]) results[tier].months[month] = {w:0,l:0};
      if (!results['ALL'].months[month]) results['ALL'].months[month] = {w:0,l:0};
      results[tier].months[month][hit?'w':'l']++;
      results['ALL'].months[month][hit?'w':'l']++;

      if (!hit) {
        console.log(`  LOSS: ${game.date} ${a}@${h} pred:${predTotal} actual:${actualTotal} tier:${tier} bonus:${bonusCount}`);
      }
    }
  }

  updateTeam(h, game.home_score, game.away_score, game.date);
  updateTeam(a, game.away_score, game.home_score, game.date);
}

console.log('\n=== NOVA MODEL FINAL VALIDATION ===\n');
const months = ['202410', '202411', '202412', '202501', '202502'];
for (const tier of ['ELITE', 'HIGH', 'STRONG', 'ALL']) {
  const d = results[tier];
  if (!d) continue;
  const t = d.w + d.l;
  console.log(`${tier}: ${d.w}-${d.l} (${(d.w/t*100).toFixed(1)}%) PnL: $${d.w*91-d.l*100} ROI: ${((d.w*91-d.l*100)/(t*100)*100).toFixed(1)}%`);
  for (const m of months) {
    const md = d.months[m];
    if (md) { const mt=md.w+md.l; console.log(`  ${m.slice(0,4)}-${m.slice(4)}: ${md.w}-${md.l} (${(md.w/mt*100).toFixed(0)}%) PnL: $${md.w*91-md.l*100}`); }
  }
  console.log('');
}
