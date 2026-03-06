// ===========================================================================
// PARLAY STRATEGY ANALYSIS: -110 LEGS ONLY
// ===========================================================================
// All legs at -110. Parlay payouts:
//   2-leg: +264 ($100 → $264 profit)
//   3-leg: +596 ($100 → $596 profit)
//   4-leg: +1228 ($100 → $1228 profit)
//
// Break-even accuracy:
//   2-leg at +264: need 27.5% hit rate
//   3-leg at +596: need 14.4% hit rate
//
// We test combinations of:
//   - NOVA OVER (81.8% singles)
//   - PRISM totals (OVER/UNDER)
//   - PRISM spreads
//   - Additional -110 filters
// ===========================================================================

const fs = require('fs');
const data = JSON.parse(fs.readFileSync('./webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = [...data].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

// ── Team History Tracking ────────────────────────────────────────────────

const prismTeams = {};
const novaTeams = {};

function prismUpdate(t, pf, pa, d) {
  if (!prismTeams[t]) prismTeams[t] = [];
  prismTeams[t].push({ pf, pa, margin: pf - pa, total: pf + pa, d });
  if (prismTeams[t].length > 30) prismTeams[t] = prismTeams[t].slice(-15);
}
function prismWindow(t, n) {
  const g = (prismTeams[t] || []).slice(-n);
  if (g.length < Math.max(5, Math.floor(n/2))) return null;
  const len = g.length;
  return {
    off: g.reduce((s,x)=>s+x.pf,0)/len, def: g.reduce((s,x)=>s+x.pa,0)/len,
    margin: g.reduce((s,x)=>s+x.margin,0)/len, pace: (g.reduce((s,x)=>s+x.pf,0)+g.reduce((s,x)=>s+x.pa,0))/len,
    wpct: g.filter(x=>x.margin>0).length/len, n: len,
  };
}

function novaUpdate(t, pf, pa, d) {
  if (!novaTeams[t]) novaTeams[t] = [];
  novaTeams[t].push({ pf, pa, total: pf + pa, d });
  if (novaTeams[t].length > 30) novaTeams[t] = novaTeams[t].slice(-20);
}
function novaWindow(t, n) {
  const g = (novaTeams[t] || []).slice(-n);
  if (g.length < n) return null;
  return { ppg: g.reduce((s,x)=>s+x.pf,0)/n, oppg: g.reduce((s,x)=>s+x.pa,0)/n, avgTotal: g.reduce((s,x)=>s+x.total,0)/n };
}

// ── Collect Daily Picks ──────────────────────────────────────────────────

const dailyPicks = {};

for (const game of sorted) {
  const d = game.date;
  if (!dailyPicks[d]) dailyPicks[d] = [];

  const h5 = prismWindow(game.home_team, 5);
  const h10 = prismWindow(game.home_team, 10);
  const h15 = prismWindow(game.home_team, 15);
  const a5 = prismWindow(game.away_team, 5);
  const a10 = prismWindow(game.away_team, 10);
  const a15 = prismWindow(game.away_team, 15);

  if (h10 && a10) {
    const predTotal = (h10.pace + a10.pace) / 2;
    const HOME_ADV = 3.5;
    const predMargin = HOME_ADV + (h10.margin - a10.margin) / 2;
    const actualTotal = game.home_score + game.away_score;
    const actualHM = game.home_score - game.away_score;

    // ── PRISM Totals ──
    let overScore = 0, underScore = 0;
    if (h10.def > 114 && a10.def > 114) overScore += 2;
    else if (h10.def > 112 && a10.def > 112) overScore += 1;
    if (h10.def < 105 || a10.def < 105) underScore += 2;
    else if (h10.def < 108 || a10.def < 108) underScore += 1;

    if (h5 && a5) {
      const recentPace = (h5.pace + a5.pace) / 2;
      const paceDiff = recentPace - predTotal;
      if (paceDiff < -4) underScore += 2; else if (paceDiff < -2) underScore += 1;
      if (paceDiff > 4) overScore += 2; else if (paceDiff > 2) overScore += 1;
      const hTrend = h5.off - h10.off, aTrend = a5.off - a10.off;
      if (hTrend < -3 && aTrend < -3) underScore += 2; else if (hTrend < -2 && aTrend < -2) underScore += 1;
      if (hTrend > 3 && aTrend > 3) overScore += 2; else if (hTrend > 2 && aTrend > 2) overScore += 1;
    }
    if (h10.pace < 215 && a10.pace < 215) underScore += 1;
    if (h10.pace > 230 && a10.pace > 230) overScore += 1;

    if (overScore >= 3 && underScore === 0) {
      const hit = actualTotal > predTotal;
      dailyPicks[d].push({ type: 'prism_over', predTotal: Math.round(predTotal*10)/10, hit, overScore, game: `${game.away_team}@${game.home_team}` });
    }
    if (underScore >= 4 && overScore === 0) {
      const hit = actualTotal < predTotal;
      dailyPicks[d].push({ type: 'prism_under', predTotal: Math.round(predTotal*10)/10, hit, underScore, game: `${game.away_team}@${game.home_team}` });
    }

    // ── PRISM Spreads ──
    const hExpWpct = h10.margin / 20 + 0.5;
    const aExpWpct = a10.margin / 20 + 0.5;
    const hLuck = h10.wpct - hExpWpct;
    const aLuck = a10.wpct - aExpWpct;
    if (hLuck > 0.1 && aLuck < -0.1) {
      const hit = actualHM < predMargin;
      const luckGap = hLuck - aLuck;
      dailyPicks[d].push({ type: 'prism_spread', hit, luckGap, game: `${game.away_team}@${game.home_team}` });
    }

    // ── PRISM Trend Divergence ──
    if (h5 && h15 && a5 && a15) {
      const hDeclining = h5.margin < h10.margin && h10.margin < h15.margin;
      const aImproving = a5.margin > a10.margin && a10.margin > a15.margin;
      if (hDeclining && aImproving) {
        const hasSpread = dailyPicks[d].some(p => p.type === 'prism_spread' && p.game === `${game.away_team}@${game.home_team}`);
        if (!hasSpread) {
          const hit = actualHM < predMargin;
          dailyPicks[d].push({ type: 'prism_trend', hit, game: `${game.away_team}@${game.home_team}` });
        }
      }
    }
  }

  // ── NOVA OVER ──
  const nh3 = novaWindow(game.home_team, 3), nh5 = novaWindow(game.home_team, 5), nh10 = novaWindow(game.home_team, 10);
  const na3 = novaWindow(game.away_team, 3), na5 = novaWindow(game.away_team, 5), na10 = novaWindow(game.away_team, 10);
  if (nh3 && nh5 && nh10 && na3 && na5 && na10) {
    if (nh5.ppg > 118 && na5.ppg > 118 && nh3.avgTotal > 225 && na3.avgTotal > 225) {
      const predTotal = Math.round(((nh10.ppg + na10.ppg + nh10.oppg + na10.oppg) / 2) * 10) / 10;
      const actualTotal = game.home_score + game.away_score;
      dailyPicks[d].push({ type: 'nova_over', predTotal, hit: actualTotal > predTotal, game: `${game.away_team}@${game.home_team}` });
    }
  }

  prismUpdate(game.home_team, game.home_score, game.away_score, d);
  prismUpdate(game.away_team, game.away_score, game.home_score, d);
  novaUpdate(game.home_team, game.home_score, game.away_score, d);
  novaUpdate(game.away_team, game.away_score, game.home_score, d);
}

// ── Analyze Singles (benchmark) ──────────────────────────────────────────

console.log('==========================================================');
console.log('SINGLES BENCHMARK (each at -110)');
console.log('==========================================================\n');

const singleStats = {};
for (const picks of Object.values(dailyPicks)) {
  for (const p of picks) {
    if (!singleStats[p.type]) singleStats[p.type] = { w: 0, l: 0 };
    singleStats[p.type][p.hit ? 'w' : 'l']++;
  }
}
for (const [type, r] of Object.entries(singleStats)) {
  const t = r.w + r.l;
  const pnl = r.w * 91 - r.l * 100;
  console.log(`${type}: ${r.w}-${r.l} (${(r.w/t*100).toFixed(1)}%) | PnL: $${pnl} | ROI: ${(pnl/(t*100)*100).toFixed(1)}%`);
}

// ── 2-Leg Parlays: All combinations on same day ──────────────────────────

console.log('\n==========================================================');
console.log('2-LEG PARLAYS (each leg -110, payout +264)');
console.log('Break-even: 27.5% | Win = +$264, Loss = -$100');
console.log('==========================================================\n');

function testParlay(legs, filter) {
  const r = { w: 0, l: 0, pnl: 0, months: {} };
  for (const [date, picks] of Object.entries(dailyPicks)) {
    const eligible = filter(picks);
    if (eligible.length < legs) continue;

    // Generate all unique combinations
    const combos = [];
    for (let i = 0; i < eligible.length; i++) {
      for (let j = i + 1; j < eligible.length; j++) {
        if (legs === 2) combos.push([eligible[i], eligible[j]]);
        else {
          for (let k = j + 1; k < eligible.length; k++) {
            combos.push([eligible[i], eligible[j], eligible[k]]);
          }
        }
      }
    }

    // Must be different games
    for (const combo of combos) {
      const games = combo.map(p => p.game);
      if (new Set(games).size !== combo.length) continue;

      const allWon = combo.every(p => p.hit);
      const payout = legs === 2 ? 264 : 596;
      const month = date.slice(0, 6);
      if (!r.months[month]) r.months[month] = { w: 0, l: 0 };
      r.pnl += allWon ? payout : -100;
      r[allWon ? 'w' : 'l']++;
      r.months[month][allWon ? 'w' : 'l']++;
    }
  }
  return r;
}

// Test: PRISM spread + NOVA OVER
const combo1 = testParlay(2, picks => {
  const spreads = picks.filter(p => p.type === 'prism_spread' || p.type === 'prism_trend');
  const novas = picks.filter(p => p.type === 'nova_over');
  return [...spreads, ...novas];
});
const c1t = combo1.w + combo1.l;
if (c1t > 0) {
  console.log(`PRISM Spread + NOVA OVER:`);
  console.log(`  ${combo1.w}-${combo1.l} (${(combo1.w/c1t*100).toFixed(1)}%) | PnL: $${combo1.pnl} | ROI: ${(combo1.pnl/(c1t*100)*100).toFixed(1)}%`);
  for (const [m, r] of Object.entries(combo1.months).sort()) {
    const mt = r.w + r.l; if (mt > 0) console.log(`  ${m}: ${r.w}-${r.l} (${(r.w/mt*100).toFixed(0)}%)`);
  }
}

// Test: PRISM totals + NOVA OVER
const combo2 = testParlay(2, picks => {
  const prismTotals = picks.filter(p => p.type === 'prism_over' || p.type === 'prism_under');
  const novas = picks.filter(p => p.type === 'nova_over');
  return [...prismTotals, ...novas];
});
const c2t = combo2.w + combo2.l;
if (c2t > 0) {
  console.log(`\nPRISM Total + NOVA OVER:`);
  console.log(`  ${combo2.w}-${combo2.l} (${(combo2.w/c2t*100).toFixed(1)}%) | PnL: $${combo2.pnl} | ROI: ${(combo2.pnl/(c2t*100)*100).toFixed(1)}%`);
}

// Test: Any 2 different-type picks on same day
const combo3 = testParlay(2, picks => picks);
const c3t = combo3.w + combo3.l;
if (c3t > 0) {
  console.log(`\nAny 2-Leg (different games):`);
  console.log(`  ${combo3.w}-${combo3.l} (${(combo3.w/c3t*100).toFixed(1)}%) | PnL: $${combo3.pnl} | ROI: ${(combo3.pnl/(c3t*100)*100).toFixed(1)}%`);
  for (const [m, r] of Object.entries(combo3.months).sort()) {
    const mt = r.w + r.l; if (mt > 0) console.log(`  ${m}: ${r.w}-${r.l} (${(r.w/mt*100).toFixed(0)}%)`);
  }
}

// Test: PRISM spread + PRISM OVER
const combo4 = testParlay(2, picks => {
  const spreads = picks.filter(p => p.type === 'prism_spread' || p.type === 'prism_trend');
  const overs = picks.filter(p => p.type === 'prism_over');
  return [...spreads, ...overs];
});
const c4t = combo4.w + combo4.l;
if (c4t > 0) {
  console.log(`\nPRISM Spread + PRISM OVER:`);
  console.log(`  ${combo4.w}-${combo4.l} (${(combo4.w/c4t*100).toFixed(1)}%) | PnL: $${combo4.pnl} | ROI: ${(combo4.pnl/(c4t*100)*100).toFixed(1)}%`);
  for (const [m, r] of Object.entries(combo4.months).sort()) {
    const mt = r.w + r.l; if (mt > 0) console.log(`  ${m}: ${r.w}-${r.l} (${(r.w/mt*100).toFixed(0)}%)`);
  }
}

// Test: 2x PRISM spreads
const combo5 = testParlay(2, picks => picks.filter(p => p.type === 'prism_spread' || p.type === 'prism_trend'));
const c5t = combo5.w + combo5.l;
if (c5t > 0) {
  console.log(`\n2x PRISM Spread:`);
  console.log(`  ${combo5.w}-${combo5.l} (${(combo5.w/c5t*100).toFixed(1)}%) | PnL: $${combo5.pnl} | ROI: ${(combo5.pnl/(c5t*100)*100).toFixed(1)}%`);
}

// Test: 2x PRISM OVER
const combo6 = testParlay(2, picks => picks.filter(p => p.type === 'prism_over'));
const c6t = combo6.w + combo6.l;
if (c6t > 0) {
  console.log(`\n2x PRISM OVER:`);
  console.log(`  ${combo6.w}-${combo6.l} (${(combo6.w/c6t*100).toFixed(1)}%) | PnL: $${combo6.pnl} | ROI: ${(combo6.pnl/(c6t*100)*100).toFixed(1)}%`);
}

// ── 3-Leg Parlays ────────────────────────────────────────────────────────

console.log('\n==========================================================');
console.log('3-LEG PARLAYS (each leg -110, payout +596)');
console.log('Break-even: 14.4% | Win = +$596, Loss = -$100');
console.log('==========================================================\n');

const combo3leg = testParlay(3, picks => picks);
const c3lt = combo3leg.w + combo3leg.l;
if (c3lt > 0) {
  console.log(`Any 3-Leg (different games):`);
  console.log(`  ${combo3leg.w}-${combo3leg.l} (${(combo3leg.w/c3lt*100).toFixed(1)}%) | PnL: $${combo3leg.pnl} | ROI: ${(combo3leg.pnl/(c3lt*100)*100).toFixed(1)}%`);
  for (const [m, r] of Object.entries(combo3leg.months).sort()) {
    const mt = r.w + r.l; if (mt > 0) console.log(`  ${m}: ${r.w}-${r.l} (${(r.w/mt*100).toFixed(0)}%)`);
  }
}

// 3-leg with at least 1 spread + 1 OVER
const combo3legMix = testParlay(3, picks => {
  const spreads = picks.filter(p => p.type === 'prism_spread' || p.type === 'prism_trend');
  const overs = picks.filter(p => p.type === 'prism_over' || p.type === 'nova_over');
  if (spreads.length === 0 || overs.length === 0) return [];
  return [...spreads, ...overs];
});
const c3lmt = combo3legMix.w + combo3legMix.l;
if (c3lmt > 0) {
  console.log(`\n3-Leg (spread+over mix):`);
  console.log(`  ${combo3legMix.w}-${combo3legMix.l} (${(combo3legMix.w/c3lmt*100).toFixed(1)}%) | PnL: $${combo3legMix.pnl} | ROI: ${(combo3legMix.pnl/(c3lmt*100)*100).toFixed(1)}%`);
}

// ── Best Single-Day Combos ───────────────────────────────────────────────

console.log('\n==========================================================');
console.log('BEST STRATEGY: Selective Day Parlays');
console.log('==========================================================\n');

// Only parlay on days with 3+ qualifying picks AND at least 2 types
const selectiveResults = { w: 0, l: 0, pnl: 0, bets: [] };
for (const [date, picks] of Object.entries(dailyPicks)) {
  const types = new Set(picks.map(p => p.type));
  if (picks.length >= 3 && types.size >= 2) {
    // Take the best pick of each type
    const bestByType = {};
    for (const p of picks) {
      if (!bestByType[p.type]) bestByType[p.type] = p;
    }
    const legs = Object.values(bestByType).slice(0, 3);
    if (legs.length >= 2) {
      const games = legs.map(l => l.game);
      if (new Set(games).size === legs.length) {
        const allWon = legs.every(l => l.hit);
        const payout = legs.length === 2 ? 264 : 596;
        selectiveResults.pnl += allWon ? payout : -100;
        selectiveResults[allWon ? 'w' : 'l']++;
        selectiveResults.bets.push({ date, legs: legs.length, types: legs.map(l => l.type), won: allWon });
      }
    }
  }
}

const srt = selectiveResults.w + selectiveResults.l;
if (srt > 0) {
  console.log(`Selective Multi-Type Parlays:`);
  console.log(`  ${selectiveResults.w}-${selectiveResults.l} (${(selectiveResults.w/srt*100).toFixed(1)}%) | PnL: $${selectiveResults.pnl} | ROI: ${(selectiveResults.pnl/(srt*100)*100).toFixed(1)}%`);
  console.log(`\nAll bets:`);
  selectiveResults.bets.forEach(b => {
    const payStr = b.legs === 2 ? '+264' : '+596';
    console.log(`  ${b.date}: ${b.legs}-leg (${b.types.join(' + ')}) → ${b.won ? 'W' : 'L'} ${b.won ? '+$' + (b.legs === 2 ? 264 : 596) : '-$100'}`);
  });
}
