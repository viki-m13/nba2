// Deep dive into top signals + combine for ultimate strategy
// Focus on: stacked_home_edge (73.9%), pace_explosion (67.6%), all_windows_over (64.7%)
// Plus: scoring_asymmetry_over (90% small sample) and b2b effects

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
    const pf = w.map(g=>g.pf), pa = w.map(g=>g.pa), margins = w.map(g=>g.margin), totals = w.map(g=>g.total);
    const homeGames = w.filter(g=>g.isHome), awayGames = w.filter(g=>!g.isHome);
    return {
      offRating: avg(pf), defRating: avg(pa), netRating: avg(margins),
      winPct: margins.filter(m=>m>0).length/w.length,
      avgTotal: avg(totals), stdTotal: std(totals), stdMargin: std(margins),
      homeWinPct: homeGames.length>=3 ? homeGames.filter(g=>g.margin>0).length/homeGames.length : null,
      awayWinPct: awayGames.length>=3 ? awayGames.filter(g=>g.margin>0).length/awayGames.length : null,
      homeAvgMargin: homeGames.length>=3 ? avg(homeGames.map(g=>g.margin)) : null,
      awayAvgMargin: awayGames.length>=3 ? avg(awayGames.map(g=>g.margin)) : null,
      avgPace: avg(totals)/2,
      last3Margins: w.slice(-3).map(g=>g.margin),
      last5Off: w.length>=5 ? avg(w.slice(-5).map(g=>g.pf)) : null,
      last5Def: w.length>=5 ? avg(w.slice(-5).map(g=>g.pa)) : null,
      last5Total: w.length>=5 ? avg(w.slice(-5).map(g=>g.total)) : null,
      games: w.length,
    };
  }
  // Get last game date for b2b detection
  getLastDate(team) {
    const h = this.teams[team] || [];
    return h.length > 0 ? h[h.length-1].date : null;
  }
}

// ═══════════════════════════════════════════════════════════
// B2B detection helper
// ═══════════════════════════════════════════════════════════
function daysBetween(d1, d2) {
  // Dates as YYYYMMDD strings
  const date1 = new Date(d1.slice(0,4)+'-'+d1.slice(4,6)+'-'+d1.slice(6,8));
  const date2 = new Date(d2.slice(0,4)+'-'+d2.slice(4,6)+'-'+d2.slice(6,8));
  return Math.round((date2 - date1) / 86400000);
}

const tracker = new TeamTracker();
const picks = [];

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

  if (h10 && a10 && h5 && a5 && h15 && a15) {
    const predTotal = (h10.offRating + a10.offRating + h10.defRating + a10.defRating) / 2;
    const predHM = h10.netRating - a10.netRating + 3.5;

    // B2B detection
    const hLastDate = tracker.getLastDate(home);
    const aLastDate = tracker.getLastDate(away);
    const homeRest = hLastDate ? daysBetween(hLastDate, game.date) : 3;
    const awayRest = aLastDate ? daysBetween(aLastDate, game.date) : 3;
    const homeB2B = homeRest <= 1;
    const awayB2B = awayRest <= 1;

    // ═════════════════════════════════════════════════════════
    // ULTIMATE STRATEGY EXPLORATION
    // ═════════════════════════════════════════════════════════

    // ── SIGNAL SCORING SYSTEM ──
    // Count independent signals for HOME WIN
    let homeWinSignals = 0;
    let homeWinFactors = [];

    // S1: Strong home court advantage (historical)
    if (h10.homeWinPct !== null && h10.homeWinPct >= 0.7) {
      homeWinSignals++; homeWinFactors.push('strong_home_court');
    }
    // S2: Away team bad on road
    if (a10.awayWinPct !== null && a10.awayWinPct <= 0.3) {
      homeWinSignals++; homeWinFactors.push('away_bad_road');
    }
    // S3: Home better net rating
    if (h10.netRating > a10.netRating + 3) {
      homeWinSignals++; homeWinFactors.push('net_rating_edge');
    }
    // S4: Home improving recently
    if (h5.netRating > h10.netRating + 2) {
      homeWinSignals++; homeWinFactors.push('home_improving');
    }
    // S5: Away declining recently
    if (a5.netRating < a10.netRating - 2) {
      homeWinSignals++; homeWinFactors.push('away_declining');
    }
    // S6: Home offense > away defense + 5
    if (h10.offRating > a10.defRating + 5) {
      homeWinSignals++; homeWinFactors.push('off_vs_def_mismatch');
    }
    // S7: Home stronger win % gap
    if (h10.winPct - a10.winPct > 0.3) {
      homeWinSignals++; homeWinFactors.push('win_pct_gap');
    }
    // S8: Home last 3 all wins
    if (h5.last3Margins.every(m => m > 0)) {
      homeWinSignals++; homeWinFactors.push('home_3_streak');
    }
    // S9: Away last 3 all losses
    if (a5.last3Margins.every(m => m < 0)) {
      homeWinSignals++; homeWinFactors.push('away_3_losing');
    }
    // S10: Rest advantage for home
    if (awayB2B && !homeB2B) {
      homeWinSignals++; homeWinFactors.push('rest_advantage');
    }

    // Count OVER signals
    let overSignals = 0;
    let overFactors = [];

    // O1: Both teams high pace
    if (h10.avgPace > 112 && a10.avgPace > 112) {
      overSignals++; overFactors.push('both_fast_pace');
    }
    // O2: Both offenses heating up
    if (h5.offRating > h10.offRating + 2 && a5.offRating > a10.offRating + 2) {
      overSignals++; overFactors.push('both_off_heating');
    }
    // O3: Both recent totals high
    if (h5.last5Total > 225 && a5.last5Total > 225) {
      overSignals++; overFactors.push('recent_totals_high');
    }
    // O4: Multi-window high totals
    if (h10.avgTotal > 224 && a10.avgTotal > 224 && h15.avgTotal > 223 && a15.avgTotal > 223) {
      overSignals++; overFactors.push('multi_window_high');
    }
    // O5: Scoring asymmetry (offense >> defense on both sides)
    if (h10.offRating - a10.defRating > 5 && a10.offRating - h10.defRating > 5) {
      overSignals++; overFactors.push('scoring_asymmetry');
    }
    // O6: Both defenses poor
    if (h10.defRating > 114 && a10.defRating > 114) {
      overSignals++; overFactors.push('both_poor_def');
    }
    // O7: Both defenses getting worse
    if (h5.defRating > h10.defRating + 2 && a5.defRating > a10.defRating + 2) {
      overSignals++; overFactors.push('def_declining');
    }

    // Count UNDER signals
    let underSignals = 0;
    let underFactors = [];

    // U1: Both teams low pace
    if (h10.avgPace < 108 && a10.avgPace < 108) {
      underSignals++; underFactors.push('both_slow_pace');
    }
    // U2: Both defenses strong
    if (h10.defRating < 110 && a10.defRating < 110) {
      underSignals++; underFactors.push('both_good_def');
    }
    // U3: Both offenses cooling
    if (h5.offRating < h10.offRating - 2 && a5.offRating < a10.offRating - 2) {
      underSignals++; underFactors.push('both_off_cooling');
    }
    // U4: Multi-window low totals
    if (h10.avgTotal < 216 && a10.avgTotal < 216 && h15.avgTotal < 217 && a15.avgTotal < 217) {
      underSignals++; underFactors.push('multi_window_low');
    }
    // U5: Scoring suppression (offense << defense)
    if (a10.defRating - h10.offRating > 3 && h10.defRating - a10.offRating > 3) {
      underSignals++; underFactors.push('scoring_suppression');
    }
    // U6: B2B factor (tired = less scoring)
    if (homeB2B || awayB2B) {
      underSignals++; underFactors.push('b2b_tired');
    }

    // ═════════════════════════════════════════════════════════
    // GENERATE PICKS
    // ═════════════════════════════════════════════════════════

    // HOME WIN picks at various thresholds
    if (homeWinSignals >= 3) {
      picks.push({
        strat: 'home_ml_' + homeWinSignals,
        date: game.date, type: 'home_ml',
        signals: homeWinSignals, factors: homeWinFactors,
        hit: actualHM > 0, pnl: actualHM > 0 ? 91 : -100,
        detail: `${home} vs ${away} | pred margin: ${predHM.toFixed(1)} | actual: ${actualHM}`
      });
    }

    // OVER picks
    if (overSignals >= 2) {
      picks.push({
        strat: 'over_' + overSignals,
        date: game.date, type: 'over',
        signals: overSignals, factors: overFactors,
        hit: actualTotal > predTotal, pnl: (actualTotal > predTotal) ? 91 : -100,
        detail: `${away}@${home} | pred total: ${predTotal.toFixed(1)} | actual: ${actualTotal}`
      });
    }

    // UNDER picks
    if (underSignals >= 2) {
      picks.push({
        strat: 'under_' + underSignals,
        date: game.date, type: 'under',
        signals: underSignals, factors: underFactors,
        hit: actualTotal < predTotal, pnl: (actualTotal < predTotal) ? 91 : -100,
        detail: `${away}@${home} | pred total: ${predTotal.toFixed(1)} | actual: ${actualTotal}`
      });
    }

    // ═══════════════════════════════════════════════════════
    // COMBINED MEGA-STRATEGY: Pick best signal per game
    // ═══════════════════════════════════════════════════════
    const bestSignal = Math.max(homeWinSignals, overSignals, underSignals);
    if (bestSignal >= 3) {
      let megaType, megaHit, megaFactors;
      if (homeWinSignals >= overSignals && homeWinSignals >= underSignals && homeWinSignals >= 3) {
        megaType = 'home_ml'; megaHit = actualHM > 0; megaFactors = homeWinFactors;
      } else if (overSignals >= underSignals && overSignals >= 3) {
        megaType = 'over'; megaHit = actualTotal > predTotal; megaFactors = overFactors;
      } else if (underSignals >= 3) {
        megaType = 'under'; megaHit = actualTotal < predTotal; megaFactors = underFactors;
      }
      if (megaType) {
        picks.push({
          strat: 'mega_' + bestSignal,
          date: game.date, type: megaType,
          signals: bestSignal, factors: megaFactors,
          hit: megaHit, pnl: megaHit ? 91 : -100,
          detail: `${megaType}: ${away}@${home}`
        });
      }
    }
  }

  tracker.update(home, game.home_score, game.away_score, game.date, true, away);
  tracker.update(away, game.away_score, game.home_score, game.date, false, home);
}

// ═══════════════════════════════════════════════════════════
// RESULTS
// ═══════════════════════════════════════════════════════════

console.log('═'.repeat(80));
console.log('SIGNAL-BASED STRATEGY RESULTS');
console.log('═'.repeat(80));

const groupBy = {};
for (const p of picks) {
  if (!groupBy[p.strat]) groupBy[p.strat] = [];
  groupBy[p.strat].push(p);
}

const results = Object.entries(groupBy).map(([name, picks]) => {
  const wins = picks.filter(p => p.hit).length;
  const total = picks.length;
  const pnl = picks.reduce((s,p) => s+p.pnl, 0);
  return { name, wins, losses: total-wins, total, acc: wins/total*100, pnl, roi: pnl/(total*100)*100 };
}).sort((a,b) => b.acc - a.acc);

console.log(`${'Strategy'.padEnd(25)} | ${'W-L'.padEnd(8)} | ${'Acc%'.padEnd(7)} | ${'PnL'.padEnd(9)} | ${'ROI%'.padEnd(8)} | Picks`);
console.log('─'.repeat(80));
for (const r of results) {
  console.log(`${r.name.padEnd(25)} | ${(r.wins+'-'+r.losses).padEnd(8)} | ${r.acc.toFixed(1).padEnd(7)} | ${((r.pnl>=0?'+':'')+'$'+r.pnl).padEnd(9)} | ${((r.roi>=0?'+':'')+r.roi.toFixed(1)+'%').padEnd(8)} | ${r.total}`);
}

// Best combo: home_ml at 4+ signals
console.log('\n═══════════════════════════════════════');
console.log('HOME ML at 4+ signals: Game-by-game');
console.log('═══════════════════════════════════════');
const best4 = picks.filter(p => p.strat === 'home_ml_4' || p.strat === 'home_ml_5' || p.strat === 'home_ml_6' || p.strat === 'home_ml_7');
for (const p of best4) {
  const res = p.hit ? 'W' : 'L';
  console.log(`[${res}] ${p.detail} | factors: ${p.factors.join(', ')}`);
}

// Best combo: OVER at 3+ signals
console.log('\n═══════════════════════════════════════');
console.log('OVER at 3+ signals: Game-by-game');
console.log('═══════════════════════════════════════');
const bestOver = picks.filter(p => (p.strat === 'over_3' || p.strat === 'over_4' || p.strat === 'over_5') );
for (const p of bestOver) {
  const res = p.hit ? 'W' : 'L';
  console.log(`[${res}] ${p.detail} | factors: ${p.factors.join(', ')}`);
}

// Combined: ALL picks with 4+ signals (any type)
console.log('\n═══════════════════════════════════════');
console.log('ALL TYPES at 4+ signals combined');
console.log('═══════════════════════════════════════');
const ultra = picks.filter(p => p.signals >= 4 && !p.strat.startsWith('mega'));
const uWins = ultra.filter(p => p.hit).length;
const uTotal = ultra.length;
const uPnl = ultra.reduce((s,p) => s+p.pnl, 0);
console.log(`W-L: ${uWins}-${uTotal-uWins} | Acc: ${(uWins/uTotal*100).toFixed(1)}% | PnL: $${uPnl} | ROI: ${(uPnl/(uTotal*100)*100).toFixed(1)}%`);

// By type within 4+
for (const type of ['home_ml', 'over', 'under']) {
  const typed = ultra.filter(p => p.type === type);
  if (typed.length > 0) {
    const tw = typed.filter(p => p.hit).length;
    const tpnl = typed.reduce((s,p) => s+p.pnl, 0);
    console.log(`  ${type}: ${tw}-${typed.length-tw} (${(tw/typed.length*100).toFixed(1)}%) PnL: $${tpnl}`);
  }
}

// Most common winning factor combinations
console.log('\n═══════════════════════════════════════');
console.log('FACTOR FREQUENCY IN WINS (4+ signals)');
console.log('═══════════════════════════════════════');
const factorCounts = {};
for (const p of ultra.filter(p => p.hit)) {
  for (const f of p.factors) {
    factorCounts[f] = (factorCounts[f] || 0) + 1;
  }
}
Object.entries(factorCounts).sort((a,b) => b[1]-a[1]).forEach(([f,c]) => {
  console.log(`  ${f}: ${c}`);
});

// FINAL: What if we COMBINE home_ml 3+ AND over 3+ as a single Play?
console.log('\n═══════════════════════════════════════════════');
console.log('COMBINED PLAY: home_ml 3+ OR over 3+ OR under 3+');
console.log('═══════════════════════════════════════════════');
const combined = picks.filter(p => p.signals >= 3 && !p.strat.startsWith('mega'));
const cWins = combined.filter(p => p.hit).length;
const cTotal = combined.length;
const cPnl = combined.reduce((s,p) => s+p.pnl, 0);
console.log(`W-L: ${cWins}-${cTotal-cWins} | Acc: ${(cWins/cTotal*100).toFixed(1)}% | PnL: $${cPnl} | ROI: ${(cPnl/(cTotal*100)*100).toFixed(1)}%`);

// Try adjusting HOME ML to only use -110 (moneyline favorites at short odds)
// For home_ml, we need to determine what the odds would be
console.log('\n═══════════════════════════════════════════════');
console.log('HOME ML PICKS - Filtered by predicted margin');
console.log('═══════════════════════════════════════════════');
const homeMLPicks = picks.filter(p => p.type === 'home_ml' && !p.strat.startsWith('mega'));
for (const range of [{min:-2,max:2,label:'pick-em (-2 to 2)'}, {min:2,max:5,label:'slight fav (2-5)'}, {min:5,max:10,label:'moderate fav (5-10)'}, {min:10,max:99,label:'big fav (10+)'}]) {
  const filtered = homeMLPicks.filter(p => {
    const predHM = parseFloat(p.detail.match(/pred margin: ([-\d.]+)/)?.[1] || 0);
    return predHM >= range.min && predHM < range.max;
  });
  if (filtered.length > 0) {
    const fw = filtered.filter(p => p.hit).length;
    const fpnl = filtered.reduce((s,p) => s+p.pnl, 0);
    console.log(`  ${range.label}: ${fw}-${filtered.length-fw} (${(fw/filtered.length*100).toFixed(1)}%) PnL: $${fpnl}`);
  }
}
