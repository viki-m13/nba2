// Final optimization: find the highest-accuracy -110 strategy
// Test weighted signals, additional filters, and combined approaches

const fs = require('fs');
const data = JSON.parse(fs.readFileSync('webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = data.sort((a, b) => a.date.localeCompare(b.date));

class TeamTracker {
  constructor() { this.teams = {}; }
  update(team, pf, pa, date, isHome, opp) {
    if (!this.teams[team]) this.teams[team] = [];
    this.teams[team].push({ pf, pa, margin: pf-pa, date, isHome, opp, total: pf+pa });
    if (this.teams[team].length > 40) this.teams[team] = this.teams[team].slice(-30);
  }
  getWindow(team, n) { return (this.teams[team] || []).slice(-n); }
  getStats(team, n) {
    const w = this.getWindow(team, n);
    if (w.length < Math.min(n, 8)) return null;
    const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
    const std = arr => { const m=avg(arr); return Math.sqrt(arr.reduce((s,x)=>s+(x-m)**2,0)/arr.length); };
    const pf=w.map(g=>g.pf), pa=w.map(g=>g.pa), margins=w.map(g=>g.margin), totals=w.map(g=>g.total);
    const homeGames=w.filter(g=>g.isHome), awayGames=w.filter(g=>!g.isHome);
    return {
      offRating: avg(pf), defRating: avg(pa), netRating: avg(margins),
      winPct: margins.filter(m=>m>0).length/w.length,
      avgTotal: avg(totals), stdTotal: std(totals), stdMargin: std(margins),
      homeWinPct: homeGames.length>=3 ? homeGames.filter(g=>g.margin>0).length/homeGames.length : null,
      awayWinPct: awayGames.length>=3 ? awayGames.filter(g=>g.margin>0).length/awayGames.length : null,
      avgPace: avg(totals)/2,
      last3Margins: w.slice(-3).map(g=>g.margin),
      last5Off: w.length>=5 ? avg(w.slice(-5).map(g=>g.pf)) : null,
      last5Def: w.length>=5 ? avg(w.slice(-5).map(g=>g.pa)) : null,
      last5Total: w.length>=5 ? avg(w.slice(-5).map(g=>g.total)) : null,
      lastGameTotal: w[w.length-1].total,
      last3Totals: w.slice(-3).map(g=>g.total),
      offTrend: w.length>=10 ? avg(w.slice(-5).map(g=>g.pf)) - avg(w.slice(-10,-5).map(g=>g.pf)) : 0,
      defTrend: w.length>=10 ? avg(w.slice(-5).map(g=>g.pa)) - avg(w.slice(-10,-5).map(g=>g.pa)) : 0,
      games: w.length,
    };
  }
  getLastDate(team) {
    const h = this.teams[team] || [];
    return h.length > 0 ? h[h.length-1].date : null;
  }
}

function daysBetween(d1, d2) {
  const date1 = new Date(d1.slice(0,4)+'-'+d1.slice(4,6)+'-'+d1.slice(6,8));
  const date2 = new Date(d2.slice(0,4)+'-'+d2.slice(4,6)+'-'+d2.slice(6,8));
  return Math.round((date2 - date1) / 86400000);
}

const tracker = new TeamTracker();
const strategies = {};

function addPick(name, date, hit, factors, detail) {
  if (!strategies[name]) strategies[name] = [];
  strategies[name].push({ date, hit, pnl: hit ? 91 : -100, factors, detail });
}

for (const game of sorted) {
  const home = game.home_team, away = game.away_team;
  const actualHM = game.home_score - game.away_score;
  const actualTotal = game.home_score + game.away_score;

  const h5 = tracker.getStats(home, 5);
  const a5 = tracker.getStats(away, 5);
  const h10 = tracker.getStats(home, 10);
  const a10 = tracker.getStats(away, 10);
  const h15 = tracker.getStats(home, 15);
  const a15 = tracker.getStats(away, 15);
  const h7 = tracker.getStats(home, 7);
  const a7 = tracker.getStats(away, 7);

  if (h10 && a10 && h5 && a5 && h15 && a15 && h7 && a7) {
    const predTotal = (h10.offRating + a10.offRating + h10.defRating + a10.defRating) / 2;

    // ═══════════════════════════════════════════════════════
    // ENHANCED OVER SIGNAL SYSTEM with weighted scoring
    // ═══════════════════════════════════════════════════════
    let overScore = 0;
    let overFactors = [];

    // S1: Both teams fast pace (weight: 1.5)
    if (h10.avgPace > 113 && a10.avgPace > 113) { overScore += 1.5; overFactors.push('elite_pace'); }
    else if (h10.avgPace > 111 && a10.avgPace > 111) { overScore += 1.0; overFactors.push('fast_pace'); }

    // S2: Both offenses heating up recently (weight: 1.0)
    if (h5.offRating > h10.offRating + 3 && a5.offRating > a10.offRating + 3) { overScore += 1.5; overFactors.push('both_surging'); }
    else if (h5.offRating > h10.offRating + 1 && a5.offRating > a10.offRating + 1) { overScore += 1.0; overFactors.push('both_heating'); }

    // S3: Recent game totals high (weight: 1.5 - very predictive)
    if (h5.last5Total > 228 && a5.last5Total > 228) { overScore += 2.0; overFactors.push('extreme_recent_high'); }
    else if (h5.last5Total > 224 && a5.last5Total > 224) { overScore += 1.5; overFactors.push('recent_high'); }
    else if (h5.last5Total > 220 && a5.last5Total > 220) { overScore += 1.0; overFactors.push('recent_elevated'); }

    // S4: Multi-window consistency (weight: 2.0 - most predictive)
    const allWindowsHigh = h5.last5Total > 224 && a5.last5Total > 224 &&
                           h10.avgTotal > 223 && a10.avgTotal > 223 &&
                           h15.avgTotal > 222 && a15.avgTotal > 222;
    if (allWindowsHigh) { overScore += 2.0; overFactors.push('all_windows_high'); }
    else if (h10.avgTotal > 223 && a10.avgTotal > 223 && h15.avgTotal > 222 && a15.avgTotal > 222) {
      overScore += 1.5; overFactors.push('multi_window_high');
    }

    // S5: Scoring asymmetry - both matchups favor offense (weight: 1.5)
    const homeOffVsAwayDef = h10.offRating - a10.defRating;
    const awayOffVsHomeDef = a10.offRating - h10.defRating;
    if (homeOffVsAwayDef > 7 && awayOffVsHomeDef > 7) { overScore += 2.0; overFactors.push('extreme_asym'); }
    else if (homeOffVsAwayDef > 4 && awayOffVsHomeDef > 4) { overScore += 1.0; overFactors.push('scoring_asym'); }

    // S6: Both defenses poor (weight: 1.0)
    if (h10.defRating > 116 && a10.defRating > 116) { overScore += 1.5; overFactors.push('terrible_def'); }
    else if (h10.defRating > 113 && a10.defRating > 113) { overScore += 1.0; overFactors.push('poor_def'); }

    // S7: Defensive trend declining (weight: 1.0)
    if (h5.defRating > h10.defRating + 3 && a5.defRating > a10.defRating + 3) { overScore += 1.5; overFactors.push('def_collapsing'); }
    else if (h5.defRating > h10.defRating + 1 && a5.defRating > a10.defRating + 1) { overScore += 0.5; overFactors.push('def_softening'); }

    // S8: Last game for both teams was high-scoring (weight: 1.0)
    if (h10.lastGameTotal > 230 && a10.lastGameTotal > 230) { overScore += 1.0; overFactors.push('last_game_high'); }

    // S9: Both teams' last 3 games all over 215 (consistency signal)
    const hLast3AllHigh = h10.last3Totals.every(t => t > 215);
    const aLast3AllHigh = a10.last3Totals.every(t => t > 215);
    if (hLast3AllHigh && aLast3AllHigh) { overScore += 1.0; overFactors.push('consistent_high_3'); }

    // Test various thresholds
    const overHit = actualTotal > predTotal;
    if (overScore >= 3) addPick('over_3pt', game.date, overHit, overFactors, `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal} score:${overScore.toFixed(1)}`);
    if (overScore >= 4) addPick('over_4pt', game.date, overHit, overFactors, `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal} score:${overScore.toFixed(1)}`);
    if (overScore >= 5) addPick('over_5pt', game.date, overHit, overFactors, `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal} score:${overScore.toFixed(1)}`);
    if (overScore >= 6) addPick('over_6pt', game.date, overHit, overFactors, `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal} score:${overScore.toFixed(1)}`);
    if (overScore >= 7) addPick('over_7pt', game.date, overHit, overFactors, `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal} score:${overScore.toFixed(1)}`);
    if (overScore >= 8) addPick('over_8pt', game.date, overHit, overFactors, `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal} score:${overScore.toFixed(1)}`);
    if (overScore >= 9) addPick('over_9pt', game.date, overHit, overFactors, `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal} score:${overScore.toFixed(1)}`);
    if (overScore >= 10) addPick('over_10pt', game.date, overHit, overFactors, `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal} score:${overScore.toFixed(1)}`);

    // ═══════════════════════════════════════════════════════
    // HOME DOMINANCE SYSTEM (for ML/spread at variable odds)
    // Score home signals with weights
    // ═══════════════════════════════════════════════════════
    let homeScore = 0;
    let homeFactors = [];

    if (h10.homeWinPct !== null && h10.homeWinPct >= 0.7) { homeScore += 1.5; homeFactors.push('fortress_home'); }
    if (a10.awayWinPct !== null && a10.awayWinPct <= 0.3) { homeScore += 1.5; homeFactors.push('away_roadkill'); }
    if (h10.netRating > a10.netRating + 5) { homeScore += 2.0; homeFactors.push('net_dominance'); }
    else if (h10.netRating > a10.netRating + 3) { homeScore += 1.0; homeFactors.push('net_edge'); }
    if (h5.netRating > h10.netRating + 3) { homeScore += 1.5; homeFactors.push('home_surging'); }
    else if (h5.netRating > h10.netRating + 1) { homeScore += 0.5; homeFactors.push('home_improving'); }
    if (a5.netRating < a10.netRating - 3) { homeScore += 1.5; homeFactors.push('away_cratering'); }
    else if (a5.netRating < a10.netRating - 1) { homeScore += 0.5; homeFactors.push('away_fading'); }
    if (h10.offRating > a10.defRating + 6) { homeScore += 1.5; homeFactors.push('off_vs_def_crush'); }
    else if (h10.offRating > a10.defRating + 4) { homeScore += 1.0; homeFactors.push('off_vs_def_edge'); }
    if (h10.winPct - a10.winPct > 0.4) { homeScore += 2.0; homeFactors.push('huge_wpct_gap'); }
    else if (h10.winPct - a10.winPct > 0.25) { homeScore += 1.0; homeFactors.push('wpct_gap'); }
    if (h5.last3Margins.every(m => m > 0)) { homeScore += 1.0; homeFactors.push('home_hot'); }
    if (a5.last3Margins.every(m => m < 0)) { homeScore += 1.0; homeFactors.push('away_cold'); }

    // Home ML tests at various thresholds
    const homeWin = actualHM > 0;
    if (homeScore >= 5) addPick('home_ml_5pt', game.date, homeWin, homeFactors, `${home} vs ${away} pred:${(h10.netRating-a10.netRating+3.5).toFixed(1)} act:${actualHM} score:${homeScore.toFixed(1)}`);
    if (homeScore >= 6) addPick('home_ml_6pt', game.date, homeWin, homeFactors, `${home} vs ${away} pred:${(h10.netRating-a10.netRating+3.5).toFixed(1)} act:${actualHM} score:${homeScore.toFixed(1)}`);
    if (homeScore >= 7) addPick('home_ml_7pt', game.date, homeWin, homeFactors, `${home} vs ${away} pred:${(h10.netRating-a10.netRating+3.5).toFixed(1)} act:${actualHM} score:${homeScore.toFixed(1)}`);
    if (homeScore >= 8) addPick('home_ml_8pt', game.date, homeWin, homeFactors, `${home} vs ${away} pred:${(h10.netRating-a10.netRating+3.5).toFixed(1)} act:${actualHM} score:${homeScore.toFixed(1)}`);
    if (homeScore >= 9) addPick('home_ml_9pt', game.date, homeWin, homeFactors, `${home} vs ${away} pred:${(h10.netRating-a10.netRating+3.5).toFixed(1)} act:${actualHM} score:${homeScore.toFixed(1)}`);
    if (homeScore >= 10) addPick('home_ml_10pt', game.date, homeWin, homeFactors, `${home} vs ${away} pred:${(h10.netRating-a10.netRating+3.5).toFixed(1)} act:${actualHM} score:${homeScore.toFixed(1)}`);

    // ═══════════════════════════════════════════════════════
    // COMBINED: OVER when BOTH over signals AND home dominance
    // (high-scoring blowout game = over)
    // ═══════════════════════════════════════════════════════
    if (overScore >= 3 && homeScore >= 4) {
      addPick('combo_over3_home4', game.date, overHit, [...overFactors, ...homeFactors],
        `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal}`);
    }
    if (overScore >= 3 && homeScore >= 5) {
      addPick('combo_over3_home5', game.date, overHit, [...overFactors, ...homeFactors],
        `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal}`);
    }
    if (overScore >= 4 && homeScore >= 4) {
      addPick('combo_over4_home4', game.date, overHit, [...overFactors, ...homeFactors],
        `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal}`);
    }

    // ═══════════════════════════════════════════════════════
    // MASTER STRATEGY: Use overScore to determine OVER bets,
    // and homeScore to determine "home dominance game total OVER"
    // Key insight: when home team dominates, they run up the score
    // ═══════════════════════════════════════════════════════
    // When home dominance is strong, does total tend to go OVER?
    if (homeScore >= 6) {
      addPick('home_dom6_over', game.date, overHit, homeFactors,
        `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal} homeScore:${homeScore.toFixed(1)}`);
    }
    if (homeScore >= 7) {
      addPick('home_dom7_over', game.date, overHit, homeFactors,
        `${away}@${home} pred:${predTotal.toFixed(0)} act:${actualTotal} homeScore:${homeScore.toFixed(1)}`);
    }

    // ═══════════════════════════════════════════════════════
    // APEX FINAL: Combined over + home + all types
    // Pick the BEST bet per game
    // ═══════════════════════════════════════════════════════
    const overQualifies = overScore >= 5;
    const homeQualifies = homeScore >= 6;

    if (overQualifies || homeQualifies) {
      let bestBet, bestHit;
      if (overQualifies && homeQualifies) {
        // Both qualify - bet OVER (strongest signal)
        bestBet = 'over'; bestHit = overHit;
      } else if (overQualifies) {
        bestBet = 'over'; bestHit = overHit;
      } else {
        bestBet = 'home_ml'; bestHit = homeWin;
      }
      addPick('apex_final', game.date, bestHit,
        overQualifies ? overFactors : homeFactors,
        `${bestBet}: ${away}@${home} | overScore:${overScore.toFixed(1)} homeScore:${homeScore.toFixed(1)} | act total:${actualTotal} actHM:${actualHM}`);
    }
  }

  tracker.update(home, game.home_score, game.away_score, game.date, true, away);
  tracker.update(away, game.away_score, game.home_score, game.date, false, home);
}

// ═══════════════════════════════════════════════════════════
// RESULTS
// ═══════════════════════════════════════════════════════════
console.log('═══════════════════════════════════════════════════════════════════');
console.log('WEIGHTED SIGNAL RESULTS (sorted by accuracy, min 10 picks)');
console.log('═══════════════════════════════════════════════════════════════════');
console.log(`${'Strategy'.padEnd(30)} | ${'W-L'.padEnd(8)} | ${'Acc%'.padEnd(7)} | ${'PnL'.padEnd(9)} | ${'ROI%'.padEnd(8)} | Picks`);
console.log('─'.repeat(80));

const results = Object.entries(strategies)
  .map(([name, picks]) => {
    const w = picks.filter(p => p.hit).length;
    const t = picks.length;
    const pnl = picks.reduce((s,p) => s+p.pnl, 0);
    return { name, w, l: t-w, t, acc: w/t*100, pnl, roi: pnl/(t*100)*100 };
  })
  .filter(r => r.t >= 10)
  .sort((a,b) => b.acc - a.acc);

for (const r of results) {
  const pnlStr = (r.pnl>=0?'+':'')+'$'+r.pnl;
  const roiStr = (r.roi>=0?'+':'')+r.roi.toFixed(1)+'%';
  // Mark if pnl computation assumes -110 or variable
  const note = r.name.includes('home_ml') ? ' (ML)' : ' (-110)';
  console.log(`${r.name.padEnd(30)} | ${(r.w+'-'+r.l).padEnd(8)} | ${r.acc.toFixed(1).padEnd(7)} | ${pnlStr.padEnd(9)} | ${roiStr.padEnd(8)} | ${r.t}${note}`);
}

// Game-by-game for best strategy
console.log('\n═══════════════════════════════════════════════');
console.log('GAME-BY-GAME: over_7pt (OVER at 7+ weighted score)');
console.log('═══════════════════════════════════════════════');
if (strategies.over_7pt) {
  for (const p of strategies.over_7pt) {
    console.log(`[${p.hit?'W':'L'}] ${p.detail} | ${p.factors.join(', ')}`);
  }
}

console.log('\n═══════════════════════════════════════════════');
console.log('GAME-BY-GAME: over_8pt');
console.log('═══════════════════════════════════════════════');
if (strategies.over_8pt) {
  for (const p of strategies.over_8pt) {
    console.log(`[${p.hit?'W':'L'}] ${p.detail} | ${p.factors.join(', ')}`);
  }
}

console.log('\n═══════════════════════════════════════════════');
console.log('GAME-BY-GAME: home_ml_8pt');
console.log('═══════════════════════════════════════════════');
if (strategies.home_ml_8pt) {
  for (const p of strategies.home_ml_8pt) {
    console.log(`[${p.hit?'W':'L'}] ${p.detail} | ${p.factors.join(', ')}`);
  }
}

console.log('\n═══════════════════════════════════════════════');
console.log('GAME-BY-GAME: apex_final');
console.log('═══════════════════════════════════════════════');
if (strategies.apex_final) {
  for (const p of strategies.apex_final) {
    console.log(`[${p.hit?'W':'L'}] ${p.detail} | ${p.factors.join(', ')}`);
  }
}

// Monthly breakdown for apex_final
console.log('\n═══════════════════════════════════════════════');
console.log('MONTHLY: apex_final');
console.log('═══════════════════════════════════════════════');
if (strategies.apex_final) {
  const months = {};
  for (const p of strategies.apex_final) {
    const m = p.date.slice(0,6);
    if (!months[m]) months[m] = {w:0,l:0};
    if (p.hit) months[m].w++; else months[m].l++;
  }
  for (const [m,d] of Object.entries(months).sort()) {
    console.log(`  ${m}: ${d.w}-${d.l} (${(d.w/(d.w+d.l)*100).toFixed(0)}%)`);
  }
}
