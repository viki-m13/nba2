// Test C5 as HIGH tier, excluding ELITE games
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

const r = { ELITE: {w:0,l:0,m:{}}, HIGH_C5_residual: {w:0,l:0,m:{}} };

for (const game of sorted) {
  const h = game.home_team, a = game.away_team;
  const h3 = getWindow(h,3), h5 = getWindow(h,5), h10 = getWindow(h,10);
  const a3 = getWindow(a,3), a5 = getWindow(a,5), a10 = getWindow(a,10);

  if (h3 && h5 && h10 && a3 && a5 && a10) {
    const predTotal = Math.round(((h10.ppg + a10.ppg + h10.oppg + a10.oppg) / 2) * 10) / 10;
    const actualTotal = game.home_score + game.away_score;
    const hit = actualTotal > predTotal;
    const month = game.date.slice(0, 6);

    const isElite = h5.ppg > 118 && a5.ppg > 118 && h3.avgTotal > 225 && a3.avgTotal > 225;
    const isC5 = h3.avgTotal > 224 && a3.avgTotal > 224 && h5.avgTotal > 222 && a5.avgTotal > 222 && h10.avgTotal > 220 && a10.avgTotal > 220;

    if (isElite) {
      r.ELITE[hit?'w':'l']++;
      if (!r.ELITE.m[month]) r.ELITE.m[month] = {w:0,l:0};
      r.ELITE.m[month][hit?'w':'l']++;
    } else if (isC5) {
      r.HIGH_C5_residual[hit?'w':'l']++;
      if (!r.HIGH_C5_residual.m[month]) r.HIGH_C5_residual.m[month] = {w:0,l:0};
      r.HIGH_C5_residual.m[month][hit?'w':'l']++;
    }
  }

  updateTeam(h, game.home_score, game.away_score, game.date);
  updateTeam(a, game.away_score, game.home_score, game.date);
}

const months = ['202410', '202411', '202412', '202501', '202502'];
for (const [name, d] of Object.entries(r)) {
  const t = d.w + d.l;
  console.log(`${name}: ${d.w}-${d.l} (${t>0?(d.w/t*100).toFixed(1):'—'}%) PnL: $${d.w*91-d.l*100} ROI: ${t>0?((d.w*91-d.l*100)/(t*100)*100).toFixed(1):'—'}%`);
  for (const m of months) {
    const md = d.m[m];
    if (md) { const mt=md.w+md.l; console.log(`  ${m.slice(0,4)}-${m.slice(4)}: ${md.w}-${md.l} (${(md.w/mt*100).toFixed(0)}%) PnL: $${md.w*91-md.l*100}`); }
  }
  console.log('');
}
