// ===========================================================================
// FINAL PARLAY STRATEGY VALIDATION
// ===========================================================================
// Focus on the "Selective Multi-Type" approach:
// When 2+ independent signal types fire on the same day (different games),
// parlay them at -110 per leg.
//
// VALIDATE: Exact payout, odds, W/L per bet. Monthly consistency.
// ===========================================================================

const fs = require('fs');
const data = JSON.parse(fs.readFileSync('./webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = [...data].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

// ── Team History ─────────────────────────────────────────────────────────

const prismTeams = {};
const novaTeams = {};

function pu(t, pf, pa, d) {
  if (!prismTeams[t]) prismTeams[t] = [];
  prismTeams[t].push({ pf, pa, margin: pf - pa, total: pf + pa, d });
  if (prismTeams[t].length > 30) prismTeams[t] = prismTeams[t].slice(-15);
}
function pw(t, n) {
  const g = (prismTeams[t] || []).slice(-n);
  if (g.length < Math.max(5, Math.floor(n/2))) return null;
  const l = g.length;
  return { off: g.reduce((s,x)=>s+x.pf,0)/l, def: g.reduce((s,x)=>s+x.pa,0)/l, margin: g.reduce((s,x)=>s+x.margin,0)/l, pace: (g.reduce((s,x)=>s+x.pf,0)+g.reduce((s,x)=>s+x.pa,0))/l, wpct: g.filter(x=>x.margin>0).length/l, n: l };
}
function nu(t, pf, pa, d) {
  if (!novaTeams[t]) novaTeams[t] = [];
  novaTeams[t].push({ pf, pa, total: pf + pa, d });
  if (novaTeams[t].length > 30) novaTeams[t] = novaTeams[t].slice(-20);
}
function nw(t, n) {
  const g = (novaTeams[t] || []).slice(-n);
  if (g.length < n) return null;
  return { ppg: g.reduce((s,x)=>s+x.pf,0)/n, oppg: g.reduce((s,x)=>s+x.pa,0)/n, avgTotal: g.reduce((s,x)=>s+x.total,0)/n };
}

// ── Collect Daily Picks ──────────────────────────────────────────────────

const dailyPicks = {};

for (const game of sorted) {
  const d = game.date;
  if (!dailyPicks[d]) dailyPicks[d] = [];
  const h5 = pw(game.home_team, 5), h10 = pw(game.home_team, 10), h15 = pw(game.home_team, 15);
  const a5 = pw(game.away_team, 5), a10 = pw(game.away_team, 10), a15 = pw(game.away_team, 15);

  if (h10 && a10) {
    const predTotal = (h10.pace + a10.pace) / 2;
    const predMargin = 3.5 + (h10.margin - a10.margin) / 2;
    const actualTotal = game.home_score + game.away_score;
    const actualHM = game.home_score - game.away_score;

    // PRISM Totals
    let overScore = 0, underScore = 0;
    if (h10.def > 114 && a10.def > 114) overScore += 2; else if (h10.def > 112 && a10.def > 112) overScore += 1;
    if (h10.def < 105 || a10.def < 105) underScore += 2; else if (h10.def < 108 || a10.def < 108) underScore += 1;
    if (h5 && a5) {
      const pd = (h5.pace + a5.pace) / 2 - predTotal;
      if (pd < -4) underScore += 2; else if (pd < -2) underScore += 1;
      if (pd > 4) overScore += 2; else if (pd > 2) overScore += 1;
      const ht = h5.off - h10.off, at = a5.off - a10.off;
      if (ht < -3 && at < -3) underScore += 2; else if (ht < -2 && at < -2) underScore += 1;
      if (ht > 3 && at > 3) overScore += 2; else if (ht > 2 && at > 2) overScore += 1;
    }
    if (h10.pace < 215 && a10.pace < 215) underScore += 1;
    if (h10.pace > 230 && a10.pace > 230) overScore += 1;

    if (overScore >= 3 && underScore === 0) {
      dailyPicks[d].push({ type: 'prism_over', hit: actualTotal > predTotal, strength: overScore, game: `${game.away_team}@${game.home_team}`, detail: `OVER ${Math.round(predTotal*10)/10} (actual: ${actualTotal})` });
    }
    if (underScore >= 4 && overScore === 0) {
      dailyPicks[d].push({ type: 'prism_under', hit: actualTotal < predTotal, strength: underScore, game: `${game.away_team}@${game.home_team}`, detail: `UNDER ${Math.round(predTotal*10)/10} (actual: ${actualTotal})` });
    }

    // PRISM Spread
    const hE = h10.margin / 20 + 0.5, aE = a10.margin / 20 + 0.5;
    const hL = h10.wpct - hE, aL = a10.wpct - aE;
    if (hL > 0.1 && aL < -0.1) {
      dailyPicks[d].push({ type: 'prism_spread', hit: actualHM < predMargin, strength: +(hL - aL).toFixed(2), game: `${game.away_team}@${game.home_team}`, detail: `${game.away_team} covers (spread ${predMargin.toFixed(1)}, actual ${actualHM})` });
    }

    // PRISM Trend
    if (h5 && h15 && a5 && a15) {
      const hD = h5.margin < h10.margin && h10.margin < h15.margin;
      const aI = a5.margin > a10.margin && a10.margin > a15.margin;
      if (hD && aI) {
        const hasSpr = dailyPicks[d].some(p => p.type === 'prism_spread' && p.game === `${game.away_team}@${game.home_team}`);
        if (!hasSpr) dailyPicks[d].push({ type: 'prism_trend', hit: actualHM < predMargin, game: `${game.away_team}@${game.home_team}`, detail: `${game.away_team} covers via trend` });
      }
    }
  }

  // NOVA OVER
  const nh3 = nw(game.home_team, 3), nh5 = nw(game.home_team, 5), nh10 = nw(game.home_team, 10);
  const na3 = nw(game.away_team, 3), na5 = nw(game.away_team, 5), na10 = nw(game.away_team, 10);
  if (nh3 && nh5 && nh10 && na3 && na5 && na10) {
    if (nh5.ppg > 118 && na5.ppg > 118 && nh3.avgTotal > 225 && na3.avgTotal > 225) {
      const pt = Math.round(((nh10.ppg + na10.ppg + nh10.oppg + na10.oppg) / 2) * 10) / 10;
      const at = game.home_score + game.away_score;
      dailyPicks[d].push({ type: 'nova_over', hit: at > pt, game: `${game.away_team}@${game.home_team}`, detail: `NOVA OVER ${pt} (actual: ${at})` });
    }
  }

  pu(game.home_team, game.home_score, game.away_score, d);
  pu(game.away_team, game.away_score, game.home_score, d);
  nu(game.home_team, game.home_score, game.away_score, d);
  nu(game.away_team, game.away_score, game.home_score, d);
}

// ── Strategy A: "PRISM+NOVA Convergence Parlay" ──────────────────────────
// When NOVA OVER fires AND at least 1 PRISM pick fires on same day
// (different games), parlay them as 2-leg at +264

console.log('==========================================================');
console.log('STRATEGY A: NOVA + PRISM 2-Leg Parlay (+264 payout)');
console.log('When NOVA OVER + any PRISM pick fire on same day (diff game)');
console.log('==========================================================\n');

const stratA = [];
for (const [date, picks] of Object.entries(dailyPicks)) {
  const novas = picks.filter(p => p.type === 'nova_over');
  const prisms = picks.filter(p => p.type.startsWith('prism_'));
  if (novas.length >= 1 && prisms.length >= 1) {
    for (const nova of novas) {
      for (const prism of prisms) {
        if (nova.game === prism.game) continue; // must be different games
        const won = nova.hit && prism.hit;
        stratA.push({ date, nova: nova.detail, prism: `${prism.type}: ${prism.detail}`, won, pnl: won ? 264 : -100 });
      }
    }
  }
}
const aW = stratA.filter(b => b.won).length, aL = stratA.filter(b => !b.won).length;
const aPnl = stratA.reduce((s, b) => s + b.pnl, 0);
console.log(`Record: ${aW}-${aL} (${(aW/(aW+aL)*100).toFixed(1)}%)`);
console.log(`PnL: $${aPnl} | ROI: ${(aPnl/((aW+aL)*100)*100).toFixed(1)}%`);
console.log(`Bet type: 2-leg parlay at -110/-110 = +264 American`);
console.log(`Win payout: +$264 on $100 wager | Loss: -$100`);
console.log(`\nDetailed bets:`);
stratA.forEach(b => console.log(`  ${b.date}: ${b.nova} + ${b.prism} → ${b.won ? 'W +$264' : 'L -$100'}`));

// ── Strategy B: Best 2-leg per day (highest confidence) ──────────────────

console.log('\n==========================================================');
console.log('STRATEGY B: Best 2-Leg Per Day (+264 payout)');
console.log('Take the single best pick from 2 different signal types');
console.log('==========================================================\n');

const stratB = [];
for (const [date, picks] of Object.entries(dailyPicks)) {
  // Group by type, take best per type
  const byType = {};
  for (const p of picks) {
    const cat = p.type === 'prism_trend' ? 'prism_spread' : p.type; // merge trend into spread category
    if (!byType[cat]) byType[cat] = p;
  }
  const types = Object.keys(byType);
  if (types.length < 2) continue;

  // Prioritize: nova_over > prism_spread > prism_over > prism_under
  const priority = ['nova_over', 'prism_spread', 'prism_over', 'prism_under'];
  const selected = [];
  for (const t of priority) {
    if (byType[t] && selected.length < 2) {
      // Check different game
      if (selected.length === 0 || selected[0].game !== byType[t].game) {
        selected.push(byType[t]);
      }
    }
  }
  if (selected.length < 2) continue;

  const won = selected.every(p => p.hit);
  stratB.push({ date, legs: selected.map(s => `${s.type}: ${s.detail}`), won, pnl: won ? 264 : -100 });
}

const bW = stratB.filter(b => b.won).length, bL = stratB.filter(b => !b.won).length;
const bPnl = stratB.reduce((s, b) => s + b.pnl, 0);
console.log(`Record: ${bW}-${bL} (${(bW/(bW+bL)*100).toFixed(1)}%)`);
console.log(`PnL: $${bPnl} | ROI: ${(bPnl/((bW+bL)*100)*100).toFixed(1)}%`);
console.log(`\nMonthly breakdown:`);
const bMonths = {};
stratB.forEach(b => {
  const m = b.date.slice(0, 6);
  if (!bMonths[m]) bMonths[m] = { w: 0, l: 0 };
  bMonths[m][b.won ? 'w' : 'l']++;
});
for (const [m, r] of Object.entries(bMonths).sort()) {
  const t = r.w + r.l;
  const pnl = r.w * 264 - r.l * 100;
  console.log(`  ${m}: ${r.w}-${r.l} (${(r.w/t*100).toFixed(0)}%) PnL: $${pnl}`);
}
console.log(`\nDetailed bets:`);
stratB.forEach(b => console.log(`  ${b.date}: ${b.legs.join(' + ')} → ${b.won ? 'W +$264' : 'L -$100'}`));

// ── Strategy C: NOVA-only days (single OVER bet, not parlay) ─────────────

console.log('\n==========================================================');
console.log('COMPARISON: NOVA Singles vs NOVA-involved Parlays');
console.log('==========================================================\n');

let novaSinglePnl = 0, novaSingleW = 0, novaSingleL = 0;
for (const picks of Object.values(dailyPicks)) {
  for (const p of picks) {
    if (p.type === 'nova_over') {
      novaSinglePnl += p.hit ? 91 : -100;
      p.hit ? novaSingleW++ : novaSingleL++;
    }
  }
}
console.log(`NOVA Single (-110): ${novaSingleW}-${novaSingleL} (${(novaSingleW/(novaSingleW+novaSingleL)*100).toFixed(1)}%) PnL: $${novaSinglePnl}`);

const novaParlay = stratA.filter(b => b.won).length;
const novaParlayL = stratA.filter(b => !b.won).length;
console.log(`NOVA+PRISM Parlay (+264): ${novaParlay}-${novaParlayL} (${(novaParlay/(novaParlay+novaParlayL)*100).toFixed(1)}%) PnL: $${aPnl}`);
console.log(`\nKey: parlays sacrifice some hit rate but gain MUCH higher payout per win`);
console.log(`NOVA single: win +$91, loss -$100`);
console.log(`NOVA parlay: win +$264, loss -$100 (2.9x higher payout per win)`);
