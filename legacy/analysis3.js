// Final calibration: test at -110 spread/total lines
// Key: Home ML picks need to be converted to spread bets at -110
// OVER picks are already at -110

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
      avgTotal: avg(totals), stdTotal: std(totals),
      homeWinPct: homeGames.length>=3 ? homeGames.filter(g=>g.margin>0).length/homeGames.length : null,
      awayWinPct: awayGames.length>=3 ? awayGames.filter(g=>g.margin>0).length/awayGames.length : null,
      avgPace: avg(totals)/2,
      last3Margins: w.slice(-3).map(g=>g.margin),
      last5Off: w.length>=5 ? avg(w.slice(-5).map(g=>g.pf)) : null,
      last5Def: w.length>=5 ? avg(w.slice(-5).map(g=>g.pa)) : null,
      last5Total: w.length>=5 ? avg(w.slice(-5).map(g=>g.total)) : null,
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

// Collect all signal-enriched games
const allGames = [];

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

    const hLastDate = tracker.getLastDate(home);
    const aLastDate = tracker.getLastDate(away);
    const homeRest = hLastDate ? daysBetween(hLastDate, game.date) : 3;
    const awayRest = aLastDate ? daysBetween(aLastDate, game.date) : 3;
    const homeB2B = homeRest <= 1;
    const awayB2B = awayRest <= 1;

    // HOME WIN SIGNALS
    let homeSignals = 0, homeFactors = [];
    if (h10.homeWinPct !== null && h10.homeWinPct >= 0.7) { homeSignals++; homeFactors.push('strong_home_court'); }
    if (a10.awayWinPct !== null && a10.awayWinPct <= 0.3) { homeSignals++; homeFactors.push('away_bad_road'); }
    if (h10.netRating > a10.netRating + 3) { homeSignals++; homeFactors.push('net_edge'); }
    if (h5.netRating > h10.netRating + 2) { homeSignals++; homeFactors.push('home_improving'); }
    if (a5.netRating < a10.netRating - 2) { homeSignals++; homeFactors.push('away_declining'); }
    if (h10.offRating > a10.defRating + 5) { homeSignals++; homeFactors.push('off_mismatch'); }
    if (h10.winPct - a10.winPct > 0.3) { homeSignals++; homeFactors.push('win_pct_gap'); }
    if (h5.last3Margins.every(m => m > 0)) { homeSignals++; homeFactors.push('home_streak'); }
    if (a5.last3Margins.every(m => m < 0)) { homeSignals++; homeFactors.push('away_losing'); }
    if (awayB2B && !homeB2B) { homeSignals++; homeFactors.push('rest_adv'); }

    // OVER SIGNALS
    let overSignals = 0, overFactors = [];
    if (h10.avgPace > 112 && a10.avgPace > 112) { overSignals++; overFactors.push('fast_pace'); }
    if (h5.offRating > h10.offRating + 2 && a5.offRating > a10.offRating + 2) { overSignals++; overFactors.push('both_heating'); }
    if (h5.last5Total > 225 && a5.last5Total > 225) { overSignals++; overFactors.push('recent_high'); }
    if (h10.avgTotal > 224 && a10.avgTotal > 224 && h15.avgTotal > 223 && a15.avgTotal > 223) { overSignals++; overFactors.push('multi_window'); }
    if (h10.offRating - a10.defRating > 5 && a10.offRating - h10.defRating > 5) { overSignals++; overFactors.push('scoring_asym'); }
    if (h10.defRating > 114 && a10.defRating > 114) { overSignals++; overFactors.push('both_poor_def'); }
    if (h5.defRating > h10.defRating + 2 && a5.defRating > a10.defRating + 2) { overSignals++; overFactors.push('def_declining'); }

    allGames.push({
      date: game.date, home, away, actualHM, actualTotal, predTotal, predHM,
      homeSignals, homeFactors, overSignals, overFactors,
    });
  }

  tracker.update(home, game.home_score, game.away_score, game.date, true, away);
  tracker.update(away, game.away_score, game.home_score, game.date, false, home);
}

// ═══════════════════════════════════════════════════════════
// TEST 1: Home team spread bets at -110
// For home signal picks, test covering various spreads
// ═══════════════════════════════════════════════════════════
console.log('═══════════════════════════════════════════════');
console.log('HOME SPREAD AT -110: Various lines & signal thresholds');
console.log('═══════════════════════════════════════════════\n');

for (const minSignals of [3, 4, 5, 6]) {
  const homePicks = allGames.filter(g => g.homeSignals >= minSignals);
  console.log(`\n--- ${minSignals}+ Home Win Signals (${homePicks.length} picks) ---`);

  // Home ML (win outright)
  {
    const w = homePicks.filter(g => g.actualHM > 0).length;
    console.log(`  ML (win outright): ${w}-${homePicks.length-w} (${(w/homePicks.length*100).toFixed(1)}%)`);
  }

  // Home spread at various lines
  for (const spread of [-1.5, -3.5, -5.5, -7.5, -9.5]) {
    const w = homePicks.filter(g => g.actualHM > Math.abs(spread)).length;
    const pnl = w * 91 - (homePicks.length - w) * 100;
    const roi = pnl / (homePicks.length * 100) * 100;
    console.log(`  Home ${spread}: ${w}-${homePicks.length-w} (${(w/homePicks.length*100).toFixed(1)}%) PnL: ${pnl>=0?'+':''}$${pnl} ROI: ${roi>=0?'+':''}${roi.toFixed(1)}%`);
  }

  // AWAY SPREAD (home is favorite, but what if we bet against?)
  // Actually: let's test AWAY +spread (underdog covers)
  console.log('  --- Away +spread (home favorite, away covers):');
  for (const spread of [3.5, 5.5, 7.5, 9.5]) {
    const w = homePicks.filter(g => g.actualHM < spread).length; // away covers if actual < spread
    const pnl = w * 91 - (homePicks.length - w) * 100;
    const roi = pnl / (homePicks.length * 100) * 100;
    console.log(`  Away +${spread}: ${w}-${homePicks.length-w} (${(w/homePicks.length*100).toFixed(1)}%) PnL: ${pnl>=0?'+':''}$${pnl} ROI: ${roi>=0?'+':''}${roi.toFixed(1)}%`);
  }
}

// ═══════════════════════════════════════════════════════════
// TEST 2: OVER bets at -110
// ═══════════════════════════════════════════════════════════
console.log('\n═══════════════════════════════════════════════');
console.log('OVER AT -110: Various signal thresholds');
console.log('═══════════════════════════════════════════════\n');

for (const minSignals of [2, 3, 4, 5]) {
  const overPicks = allGames.filter(g => g.overSignals >= minSignals);
  console.log(`\n--- ${minSignals}+ Over Signals (${overPicks.length} picks) ---`);

  // OVER vs predicted total
  {
    const w = overPicks.filter(g => g.actualTotal > g.predTotal).length;
    const pnl = w * 91 - (overPicks.length - w) * 100;
    console.log(`  OVER pred total: ${w}-${overPicks.length-w} (${(w/overPicks.length*100).toFixed(1)}%) PnL: $${pnl}`);
  }

  // OVER at various offsets from predicted total
  for (const offset of [-5, -3, 0, 3, 5]) {
    const line = 'predTotal' + (offset >= 0 ? '+' : '') + offset;
    const w = overPicks.filter(g => g.actualTotal > g.predTotal + offset).length;
    const pnl = w * 91 - (overPicks.length - w) * 100;
    const roi = pnl / (overPicks.length * 100) * 100;
    console.log(`  OVER (pred${offset>=0?'+':''}${offset}): ${w}-${overPicks.length-w} (${(w/overPicks.length*100).toFixed(1)}%) PnL: ${pnl>=0?'+':''}$${pnl} ROI: ${roi>=0?'+':''}${roi.toFixed(1)}%`);
  }
}

// ═══════════════════════════════════════════════════════════
// TEST 3: COMBINED STRATEGY - Best of each type
// ═══════════════════════════════════════════════════════════
console.log('\n═══════════════════════════════════════════════');
console.log('OPTIMAL COMBINED PLAY 8 CANDIDATES');
console.log('═══════════════════════════════════════════════\n');

// Candidate A: Home ML at 5+ signals (79.4%, but need -110 equivalent)
// For ML bets, we'd use the actual ML odds. But user wants -110 minimum.
// So for HOME picks, we should use SPREAD bets at -110.
// Best spread seems to be -5.5 at 5+ signals

// Candidate B: OVER at 4+ signals at -110

// Candidate C: Combined approach
const strategies = [
  { name: 'Home -5.5 @ 5+ signals', filter: g => g.homeSignals >= 5, test: g => g.actualHM > 5.5, betType: 'home_spread' },
  { name: 'Home -3.5 @ 5+ signals', filter: g => g.homeSignals >= 5, test: g => g.actualHM > 3.5, betType: 'home_spread' },
  { name: 'Home -7.5 @ 5+ signals', filter: g => g.homeSignals >= 5, test: g => g.actualHM > 7.5, betType: 'home_spread' },
  { name: 'Home ML @ 5+ signals', filter: g => g.homeSignals >= 5, test: g => g.actualHM > 0, betType: 'home_ml' },
  { name: 'Home -5.5 @ 4+ signals', filter: g => g.homeSignals >= 4, test: g => g.actualHM > 5.5, betType: 'home_spread' },
  { name: 'Home ML @ 4+ signals', filter: g => g.homeSignals >= 4, test: g => g.actualHM > 0, betType: 'home_ml' },
  { name: 'OVER @ 4+ signals', filter: g => g.overSignals >= 4, test: g => g.actualTotal > g.predTotal, betType: 'over' },
  { name: 'OVER @ 3+ signals', filter: g => g.overSignals >= 3, test: g => g.actualTotal > g.predTotal, betType: 'over' },
  { name: 'Home ML@5 + OVER@4', filter: g => g.homeSignals >= 5 || g.overSignals >= 4,
    test: g => {
      if (g.homeSignals >= 5 && g.overSignals >= 4) return g.actualHM > 0 && g.actualTotal > g.predTotal; // both need to hit? No, they're separate bets
      if (g.homeSignals >= 5) return g.actualHM > 0;
      return g.actualTotal > g.predTotal;
    }, betType: 'combined' },
];

for (const s of strategies) {
  const picks = allGames.filter(s.filter);
  const wins = picks.filter(s.test).length;
  const total = picks.length;
  if (total === 0) continue;
  // For ML bets, approximate pnl based on spread-to-ML conversion
  let pnl;
  if (s.betType === 'home_ml') {
    // Each ML bet has variable odds. For now, use actual margin to compute ML odds
    pnl = 0;
    for (const g of picks) {
      const won = s.test(g);
      if (won) {
        // Estimate ML odds from predicted margin
        const m = Math.abs(g.predHM);
        let odds;
        if (m <= 1.5) odds = -120;
        else if (m <= 2.5) odds = -135;
        else if (m <= 3.5) odds = -155;
        else if (m <= 4.5) odds = -185;
        else if (m <= 5.5) odds = -210;
        else if (m <= 6.5) odds = -250;
        else if (m <= 7.5) odds = -300;
        else if (m <= 8.5) odds = -350;
        else if (m <= 9.5) odds = -420;
        else if (m <= 10.5) odds = -500;
        else if (m <= 12.5) odds = -650;
        else if (m <= 14.5) odds = -850;
        else odds = -1000;
        pnl += Math.round(100 / Math.abs(odds) * 100);
      } else {
        pnl -= 100;
      }
    }
  } else {
    pnl = wins * 91 - (total - wins) * 100;
  }
  const roi = pnl / (total * 100) * 100;
  const mlNote = s.betType === 'home_ml' ? ' (variable odds)' : ' (at -110)';
  console.log(`${s.name.padEnd(30)} | ${(wins+'-'+(total-wins)).padEnd(8)} | ${(wins/total*100).toFixed(1)}% | ${(pnl>=0?'+':'')+'$'+pnl} | ROI: ${roi>=0?'+':''}${roi.toFixed(1)}%${mlNote} | ${total} picks`);
}

// ═══════════════════════════════════════════════════════════
// GAME-BY-GAME for the best -110 strategies
// ═══════════════════════════════════════════════════════════
console.log('\n═══════════════════════════════════════════════');
console.log('MONTHLY BREAKDOWN: Home ML@5+ (pure win rate)');
console.log('═══════════════════════════════════════════════');
const homeML5 = allGames.filter(g => g.homeSignals >= 5);
const months = {};
for (const g of homeML5) {
  const m = g.date.slice(0,6);
  if (!months[m]) months[m] = { wins: 0, total: 0 };
  months[m].total++;
  if (g.actualHM > 0) months[m].wins++;
}
for (const [m, d] of Object.entries(months).sort()) {
  console.log(`  ${m}: ${d.wins}-${d.total-d.wins} (${(d.wins/d.total*100).toFixed(0)}%)`);
}

console.log('\n═══════════════════════════════════════════════');
console.log('MONTHLY BREAKDOWN: OVER@4+');
console.log('═══════════════════════════════════════════════');
const over4 = allGames.filter(g => g.overSignals >= 4);
const oMonths = {};
for (const g of over4) {
  const m = g.date.slice(0,6);
  if (!oMonths[m]) oMonths[m] = { wins: 0, total: 0 };
  oMonths[m].total++;
  if (g.actualTotal > g.predTotal) oMonths[m].wins++;
}
for (const [m, d] of Object.entries(oMonths).sort()) {
  console.log(`  ${m}: ${d.wins}-${d.total-d.wins} (${(d.wins/d.total*100).toFixed(0)}%)`);
}

// ═══════════════════════════════════════════════════════════
// WHAT IF: Home picks at -110 spread where predHM gives spread
// Use dynamic spread: bet home -(predHM - 2) at -110
// This way the model has a 2-point buffer
// ═══════════════════════════════════════════════════════════
console.log('\n═══════════════════════════════════════════════');
console.log('DYNAMIC SPREAD: Home -(predHM - buffer) at -110');
console.log('═══════════════════════════════════════════════');

for (const minSig of [4, 5, 6]) {
  for (const buffer of [2, 3, 4, 5, 7]) {
    const picks = allGames.filter(g => g.homeSignals >= minSig && g.predHM > buffer);
    if (picks.length < 10) continue;
    const wins = picks.filter(g => g.actualHM > g.predHM - buffer).length;
    const pnl = wins * 91 - (picks.length - wins) * 100;
    const roi = pnl / (picks.length * 100) * 100;
    console.log(`  ${minSig}+ signals, buffer ${buffer}: ${wins}-${picks.length-wins} (${(wins/picks.length*100).toFixed(1)}%) PnL: ${pnl>=0?'+':''}$${pnl} ROI: ${roi>=0?'+':''}${roi.toFixed(1)}% | ${picks.length} picks`);
  }
}
