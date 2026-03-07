// ===========================================================================
// COMPREHENSIVE PARLAY & PROP STRATEGY ANALYSIS
// ===========================================================================
// Tests:
// 1. Multi-game ML parlays (CDS picks combined)
// 2. Same-game parlays (ML + OVER in same game)
// 3. Player prop analysis (fetch box scores from ESPN)
// 4. Mixed parlays (team picks + player props)
//
// CRITICAL: Tracks exact odds, payout, W/L for every bet
// ===========================================================================

const fs = require('fs');
const https = require('https');

const data = JSON.parse(fs.readFileSync('./webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = [...data].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

// ── Utility Functions ─────────────────────────────────────────────────────

function spreadToML(margin) {
  const m = Math.abs(margin);
  if (m <= 1.5) return -120;
  if (m <= 2.5) return -135;
  if (m <= 3.5) return -155;
  if (m <= 4.5) return -185;
  if (m <= 5.5) return -210;
  if (m <= 6.5) return -250;
  if (m <= 7.5) return -300;
  if (m <= 8.5) return -350;
  if (m <= 9.5) return -420;
  if (m <= 10.5) return -500;
  if (m <= 12.5) return -650;
  if (m <= 14.5) return -850;
  return -1000;
}

function americanToDecimal(odds) {
  if (odds < 0) return 1 + (100 / Math.abs(odds));
  return 1 + (odds / 100);
}

function decimalToAmerican(dec) {
  if (dec >= 2) return Math.round((dec - 1) * 100);
  return Math.round(-100 / (dec - 1));
}

function parlayDecimalOdds(legOdds) {
  return legOdds.reduce((acc, o) => acc * americanToDecimal(o), 1);
}

function parlayPayout(legOdds, wager) {
  const dec = parlayDecimalOdds(legOdds);
  return Math.round((dec - 1) * wager * 100) / 100;
}

// ── CDS Model (for ML picks) ─────────────────────────────────────────────

const teams = {};
function updateTeam(team, pf, pa, date) {
  if (!teams[team]) teams[team] = [];
  teams[team].push({ pf, pa, margin: pf - pa, date });
  if (teams[team].length > 30) teams[team] = teams[team].slice(-15);
}
function getMetrics(team) {
  const h = (teams[team] || []).slice(-15);
  if (h.length < 8) return null;
  const pf = h.map(g => g.pf);
  const pa = h.map(g => g.pa);
  const margins = h.map(g => g.margin);
  return {
    offRating: pf.reduce((a, b) => a + b, 0) / h.length,
    defRating: pa.reduce((a, b) => a + b, 0) / h.length,
    netRating: margins.reduce((a, b) => a + b, 0) / h.length,
    winPct: margins.filter(m => m > 0).length / h.length,
  };
}
function computeCDS(favM, dogM, favIsHome) {
  let score = 0;
  const ng = Math.abs(favM.netRating - dogM.netRating);
  if (ng >= 14) score += 3; else if (ng >= 10) score += 2 + (ng - 10) / 4; else if (ng >= 7) score += 1 + (ng - 7) / 3; else if (ng >= 4) score += (ng - 4) / 3;
  const fo = favM.offRating;
  if (fo >= 122) score += 3; else if (fo >= 118) score += 2 + (fo - 118) / 4; else if (fo >= 115) score += 1 + (fo - 115) / 3; else if (fo >= 112) score += (fo - 112) / 3;
  const cage = favM.defRating - dogM.offRating;
  if (cage <= -6) score += 3; else if (cage <= -3) score += 2 + (-3 - cage) / 3; else if (cage <= 0) score += 1 + (-cage) / 3; else if (cage <= 3) score += (3 - cage) / 3;
  const wpg = Math.abs(favM.winPct - dogM.winPct);
  if (wpg >= 0.5) score += 3; else if (wpg >= 0.4) score += 2 + (wpg - 0.4) / 0.1; else if (wpg >= 0.3) score += 1 + (wpg - 0.3) / 0.1; else if (wpg >= 0.2) score += (wpg - 0.2) / 0.1;
  score += favIsHome ? 1.5 : 0;
  if (favM.offRating >= 115 && favM.defRating <= 110) score += 1.5; else if (favM.offRating >= 112 && favM.defRating <= 108) score += 1;
  if (dogM.offRating <= 110 && dogM.defRating >= 114) score += 1; else if (dogM.offRating <= 112 && dogM.defRating >= 112) score += 0.5;
  return +score.toFixed(1);
}

// ── NOVA Model (for OVER picks) ──────────────────────────────────────────

const novaTeams = {};
function novaUpdate(t, pf, pa, date) {
  if (!novaTeams[t]) novaTeams[t] = [];
  novaTeams[t].push({ pf, pa, total: pf + pa, date });
  if (novaTeams[t].length > 30) novaTeams[t] = novaTeams[t].slice(-20);
}
function novaWindow(t, n) {
  const g = (novaTeams[t] || []).slice(-n);
  if (g.length < n) return null;
  return { ppg: g.reduce((s, x) => s + x.pf, 0) / n, oppg: g.reduce((s, x) => s + x.pa, 0) / n, avgTotal: g.reduce((s, x) => s + x.total, 0) / n };
}

// ── Strategy 1: Multi-Game ML Parlays ────────────────────────────────────
// Combine 2-4 CDS ML picks on the same day into parlays

const dailyPicks = {}; // date → [{ fav, dog, favIsHome, cds, margin, mlOdds, won }]

for (const game of sorted) {
  const hM = getMetrics(game.home_team);
  const aM = getMetrics(game.away_team);

  if (hM && aM) {
    const HOME_ADV = 3.5;
    const netDiff = hM.netRating - aM.netRating;
    const predMargin = netDiff + HOME_ADV;
    let fav, dog, favM, dogM, favIsHome;
    if (predMargin > 0) { fav = game.home_team; dog = game.away_team; favM = hM; dogM = aM; favIsHome = true; }
    else { fav = game.away_team; dog = game.home_team; favM = aM; dogM = hM; favIsHome = false; }
    const cds = computeCDS(favM, dogM, favIsHome);
    const mlOdds = spreadToML(predMargin);
    const actualMargin = game.home_score - game.away_score;
    const won = (favIsHome && actualMargin > 0) || (!favIsHome && actualMargin < 0);

    if (cds >= 8) { // CDS qualifying
      if (!dailyPicks[game.date]) dailyPicks[game.date] = [];
      dailyPicks[game.date].push({ fav, dog, favIsHome, cds, margin: Math.abs(predMargin), mlOdds, won, date: game.date });
    }

    // NOVA check
    const h3 = novaWindow(game.home_team, 3), h5 = novaWindow(game.home_team, 5), h10 = novaWindow(game.home_team, 10);
    const a3 = novaWindow(game.away_team, 3), a5 = novaWindow(game.away_team, 5), a10 = novaWindow(game.away_team, 10);
    if (h3 && h5 && h10 && a3 && a5 && a10) {
      if (h5.ppg > 118 && a5.ppg > 118 && h3.avgTotal > 225 && a3.avgTotal > 225) {
        const predTotal = Math.round(((h10.ppg + a10.ppg + h10.oppg + a10.oppg) / 2) * 10) / 10;
        const actualTotal = game.home_score + game.away_score;
        const overHit = actualTotal > predTotal;
        if (!dailyPicks[game.date]) dailyPicks[game.date] = [];
        dailyPicks[game.date].push({ type: 'nova_over', predTotal, actualTotal, overHit, overOdds: -110, date: game.date });
      }
    }
  }

  updateTeam(game.home_team, game.home_score, game.away_score, game.date);
  updateTeam(game.away_team, game.away_score, game.home_score, game.date);
  novaUpdate(game.home_team, game.home_score, game.away_score, game.date);
  novaUpdate(game.away_team, game.away_score, game.home_score, game.date);
}

console.log('==========================================================');
console.log('STRATEGY 1: Multi-Game ML Parlays (CDS picks)');
console.log('==========================================================\n');

// Test 2-leg, 3-leg parlays from CDS picks
for (const legs of [2, 3]) {
  const results = { ELITE: { w: 0, l: 0, totalPayout: 0, totalWagered: 0 }, ALL: { w: 0, l: 0, totalPayout: 0, totalWagered: 0 } };

  for (const [date, picks] of Object.entries(dailyPicks)) {
    const mlPicks = picks.filter(p => !p.type && p.cds >= 8).sort((a, b) => b.cds - a.cds);
    const elitePicks = mlPicks.filter(p => p.cds >= 11);

    // ALL: top N picks by CDS
    if (mlPicks.length >= legs) {
      const selected = mlPicks.slice(0, legs);
      const allWon = selected.every(p => p.won);
      const legOdds = selected.map(p => p.mlOdds);
      const payout = parlayPayout(legOdds, 100);
      results.ALL.totalWagered += 100;
      if (allWon) { results.ALL.w++; results.ALL.totalPayout += payout; }
      else { results.ALL.l++; results.ALL.totalPayout -= 100; }
    }

    // ELITE: only CDS >= 11
    if (elitePicks.length >= legs) {
      const selected = elitePicks.slice(0, legs);
      const allWon = selected.every(p => p.won);
      const legOdds = selected.map(p => p.mlOdds);
      const payout = parlayPayout(legOdds, 100);
      results.ELITE.totalWagered += 100;
      if (allWon) { results.ELITE.w++; results.ELITE.totalPayout += payout; }
      else { results.ELITE.l++; results.ELITE.totalPayout -= 100; }
    }
  }

  for (const [tier, r] of Object.entries(results)) {
    const total = r.w + r.l;
    if (total === 0) continue;
    const acc = (r.w / total * 100).toFixed(1);
    const avgPayout = total > 0 ? (r.totalPayout / total).toFixed(0) : 0;
    const roi = r.totalWagered > 0 ? (r.totalPayout / r.totalWagered * 100).toFixed(1) : 0;
    // Calculate avg parlay odds
    console.log(`${legs}-Leg ML Parlay [${tier}]: ${r.w}-${r.l} (${acc}%) | Net P&L: $${r.totalPayout.toFixed(0)} | ROI: ${roi}% | Avg payout per bet: $${avgPayout}`);
  }
  console.log('');
}

console.log('==========================================================');
console.log('STRATEGY 2: Same-Game Parlay (ML + OVER)');
console.log('==========================================================\n');

// Find games where CDS ML pick AND NOVA OVER both fire
const sgpResults = { w: 0, l: 0, totalPayout: 0, details: [] };

for (const [date, picks] of Object.entries(dailyPicks)) {
  const mlPicks = picks.filter(p => !p.type && p.cds >= 9);
  const overPicks = picks.filter(p => p.type === 'nova_over');

  // Can't easily match same game without more data, so skip SGP for now
  // Instead test ML + any OVER on same day
}

console.log('(Same-game parlays require matching game IDs — testing cross-game instead)\n');

console.log('==========================================================');
console.log('STRATEGY 3: ML + NOVA OVER Cross-Game Parlay');
console.log('==========================================================\n');

// Combine best CDS ML + best NOVA OVER on same day
const crossResults = { w: 0, l: 0, totalPayout: 0, parlays: [] };

for (const [date, picks] of Object.entries(dailyPicks)) {
  const mlPicks = picks.filter(p => !p.type && p.cds >= 9).sort((a, b) => b.cds - a.cds);
  const overPicks = picks.filter(p => p.type === 'nova_over');

  if (mlPicks.length >= 1 && overPicks.length >= 1) {
    const ml = mlPicks[0];
    const over = overPicks[0];
    const legOdds = [ml.mlOdds, -110];
    const parlayDec = parlayDecimalOdds(legOdds);
    const parlayAmerican = decimalToAmerican(parlayDec);
    const allWon = ml.won && over.overHit;
    const payout = parlayPayout(legOdds, 100);

    crossResults.totalPayout += allWon ? payout : -100;
    crossResults[allWon ? 'w' : 'l']++;
    crossResults.parlays.push({ date, ml: `${ml.fav} ML ${ml.mlOdds}`, over: `OVER ${over.predTotal}`, parlayOdds: parlayAmerican, won: allWon, payout: allWon ? payout : -100 });
  }
}

const ct = crossResults.w + crossResults.l;
console.log(`ML + NOVA OVER (2-leg): ${crossResults.w}-${crossResults.l} (${(crossResults.w / ct * 100).toFixed(1)}%)`);
console.log(`Net P&L: $${crossResults.totalPayout.toFixed(0)} | ROI: ${(crossResults.totalPayout / (ct * 100) * 100).toFixed(1)}%`);
console.log(`Sample parlays:`);
crossResults.parlays.slice(-10).forEach(p => {
  console.log(`  ${p.date}: ${p.ml} + ${p.over} → Parlay ${p.parlayOdds > 0 ? '+' : ''}${p.parlayOdds} → ${p.won ? 'W' : 'L'} ($${p.payout})`);
});

console.log('\n==========================================================');
console.log('STRATEGY 4: 2-Leg CDS ELITE ML + OVER 225 (high-scoring filter)');
console.log('==========================================================\n');

// Use CDS ELITE (≥11) ML + simple OVER filter on same day (diff games)
// The OVER filter: game has both teams recent total > 222, bet OVER 225 at -110

const novaTeams2 = {};
function nU2(t, pf, pa) {
  if (!novaTeams2[t]) novaTeams2[t] = [];
  novaTeams2[t].push(pf + pa);
  if (novaTeams2[t].length > 10) novaTeams2[t] = novaTeams2[t].slice(-10);
}
function avgTotals(t, n) {
  const g = (novaTeams2[t] || []).slice(-n);
  if (g.length < n) return null;
  return g.reduce((s, x) => s + x, 0) / n;
}

// Reset and re-run with totals tracking
const dailyPicks2 = {};
const teams2 = {};
function uT(t, pf, pa, d) {
  if (!teams2[t]) teams2[t] = [];
  teams2[t].push({ pf, pa, margin: pf - pa, d });
  if (teams2[t].length > 30) teams2[t] = teams2[t].slice(-15);
}
function gM(t) {
  const h = (teams2[t] || []).slice(-15);
  if (h.length < 8) return null;
  const margins = h.map(g => g.margin);
  return {
    offRating: h.reduce((s, g) => s + g.pf, 0) / h.length,
    defRating: h.reduce((s, g) => s + g.pa, 0) / h.length,
    netRating: margins.reduce((s, m) => s + m, 0) / h.length,
    winPct: margins.filter(m => m > 0).length / h.length,
  };
}

for (const game of sorted) {
  const hM = gM(game.home_team), aM = gM(game.away_team);
  const hTot = avgTotals(game.home_team, 5), aTot = avgTotals(game.away_team, 5);

  if (!dailyPicks2[game.date]) dailyPicks2[game.date] = [];

  if (hM && aM) {
    const predMargin = hM.netRating - aM.netRating + 3.5;
    let fav, dog, favM, dogM, favIsHome;
    if (predMargin > 0) { fav = game.home_team; dog = game.away_team; favM = hM; dogM = aM; favIsHome = true; }
    else { fav = game.away_team; dog = game.home_team; favM = aM; dogM = hM; favIsHome = false; }
    const cds = computeCDS(favM, dogM, favIsHome);
    const mlOdds = spreadToML(predMargin);
    const actualMargin = game.home_score - game.away_score;
    const won = (favIsHome && actualMargin > 0) || (!favIsHome && actualMargin < 0);
    if (cds >= 8) dailyPicks2[game.date].push({ type: 'ml', fav, cds, mlOdds, won, game });
  }

  // Simple OVER: both teams averaging 222+ total in last 5
  if (hTot && aTot && hTot > 222 && aTot > 222) {
    const predTotal = (hTot + aTot) / 2;
    const actualTotal = game.home_score + game.away_score;
    dailyPicks2[game.date].push({ type: 'over', predTotal, actualTotal, hit: actualTotal > predTotal, game });
  }

  uT(game.home_team, game.home_score, game.away_score, game.date);
  uT(game.away_team, game.away_score, game.home_score, game.date);
  nU2(game.home_team, game.home_score, game.away_score);
  nU2(game.away_team, game.away_score, game.home_score);
}

// Strategy 4a: 2-leg (CDS ≥ 11 ML + OVER)
const s4a = { w: 0, l: 0, pnl: 0, bets: [] };
for (const [date, picks] of Object.entries(dailyPicks2)) {
  const mls = picks.filter(p => p.type === 'ml' && p.cds >= 11).sort((a, b) => b.cds - a.cds);
  const overs = picks.filter(p => p.type === 'over');
  if (mls.length >= 1 && overs.length >= 1) {
    const ml = mls[0];
    const over = overs[0];
    const won = ml.won && over.hit;
    const legOdds = [ml.mlOdds, -110];
    const payout = parlayPayout(legOdds, 100);
    s4a.pnl += won ? payout : -100;
    s4a[won ? 'w' : 'l']++;
    s4a.bets.push({ date, mlOdds: ml.mlOdds, won, payout: won ? payout : -100, parlayOdds: decimalToAmerican(parlayDecimalOdds(legOdds)) });
  }
}
const s4t = s4a.w + s4a.l;
if (s4t > 0) {
  console.log(`2-Leg (ELITE ML + OVER): ${s4a.w}-${s4a.l} (${(s4a.w/s4t*100).toFixed(1)}%)`);
  console.log(`Net P&L: $${s4a.pnl.toFixed(0)} | ROI: ${(s4a.pnl/(s4t*100)*100).toFixed(1)}%`);
  const avgOdds = s4a.bets.reduce((s, b) => s + b.parlayOdds, 0) / s4a.bets.length;
  console.log(`Avg parlay odds: ${avgOdds > 0 ? '+' : ''}${avgOdds.toFixed(0)}`);
  console.log(`Sample bets:`);
  s4a.bets.slice(-8).forEach(b => console.log(`  ${b.date}: ML ${b.mlOdds} + OVER -110 → ${b.parlayOdds > 0 ? '+' : ''}${b.parlayOdds} → ${b.won ? 'W' : 'L'} $${b.payout.toFixed(0)}`));
}

console.log('\n==========================================================');
console.log('STRATEGY 5: Pure CDS High-Confidence Singles Benchmark');
console.log('==========================================================\n');

// Benchmark: what do individual CDS picks actually return?
const cdsResults = { 8: { w: 0, l: 0, pnl: 0 }, 9: { w: 0, l: 0, pnl: 0 }, 10: { w: 0, l: 0, pnl: 0 }, 11: { w: 0, l: 0, pnl: 0 } };
for (const picks of Object.values(dailyPicks)) {
  for (const p of picks) {
    if (p.type) continue;
    for (const thresh of [8, 9, 10, 11]) {
      if (p.cds >= thresh) {
        const profit = p.won ? Math.round(100 / Math.abs(p.mlOdds) * 100) : -100;
        cdsResults[thresh].pnl += profit;
        cdsResults[thresh][p.won ? 'w' : 'l']++;
      }
    }
  }
}
for (const [thresh, r] of Object.entries(cdsResults)) {
  const t = r.w + r.l;
  if (t === 0) continue;
  console.log(`CDS ≥ ${thresh}: ${r.w}-${r.l} (${(r.w/t*100).toFixed(1)}%) | P&L: $${r.pnl} | ROI: ${(r.pnl/(t*100)*100).toFixed(1)}% | Avg payout per win: $${Math.round(r.pnl > 0 ? r.pnl / r.w : 0)}`);
}

console.log('\n==========================================================');
console.log('STRATEGY 6: 3-Leg CDS ML Parlay (Top 3 by CDS on each day)');
console.log('==========================================================\n');

for (const minCds of [8, 9, 10]) {
  const r = { w: 0, l: 0, pnl: 0, bets: [] };
  for (const [date, picks] of Object.entries(dailyPicks)) {
    const mls = picks.filter(p => !p.type && p.cds >= minCds).sort((a, b) => b.cds - a.cds);
    if (mls.length >= 3) {
      const sel = mls.slice(0, 3);
      const allWon = sel.every(p => p.won);
      const legOdds = sel.map(p => p.mlOdds);
      const payout = parlayPayout(legOdds, 100);
      const parlayOdds = decimalToAmerican(parlayDecimalOdds(legOdds));
      r.pnl += allWon ? payout : -100;
      r[allWon ? 'w' : 'l']++;
      r.bets.push({ date, odds: parlayOdds, won: allWon });
    }
  }
  const t = r.w + r.l;
  if (t > 0) {
    const avgOdds = r.bets.reduce((s, b) => s + b.odds, 0) / r.bets.length;
    console.log(`3-Leg ML (CDS ≥ ${minCds}): ${r.w}-${r.l} (${(r.w/t*100).toFixed(1)}%) | P&L: $${r.pnl.toFixed(0)} | ROI: ${(r.pnl/(t*100)*100).toFixed(1)}% | Avg odds: ${avgOdds > 0 ? '+' : ''}${avgOdds.toFixed(0)}`);
  }
}

console.log('\n==========================================================');
console.log('NOTE ON PLAYER PROPS');
console.log('==========================================================\n');
console.log('Player prop parlays require individual box score data.');
console.log('We have event_id for every game — can fetch from ESPN API.');
console.log('To test: fetch box scores → build player stat models →');
console.log('predict over/under on points/rebounds/assists → combine in parlays.');
console.log('This requires HTTP fetches and will be done separately.\n');
