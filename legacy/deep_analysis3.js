// Deep analysis 3: Ultra-selective compound strategies
// Goal: Find combinations that hold up in Feb AND have 70%+ accuracy
const fs = require('fs');
const data = JSON.parse(fs.readFileSync('./webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = [...data].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

const teams = {};
function initTeam(t) {
  if (!teams[t]) teams[t] = { games: [], homeGames: [], awayGames: [] };
}
function getStats(t, n) {
  const g = teams[t]?.games || [];
  if (g.length < n) return null;
  const r = g.slice(-n);
  const ppg = r.reduce((s, x) => s + x.pf, 0) / n;
  const oppg = r.reduce((s, x) => s + x.pa, 0) / n;
  const totals = r.map(x => x.pf + x.pa);
  const avgTotal = totals.reduce((s, x) => s + x, 0) / n;
  const margins = r.map(x => x.pf - x.pa);
  const avgMargin = margins.reduce((s, x) => s + x, 0) / n;
  const wins = r.filter(x => x.pf > x.pa).length;
  return { ppg, oppg, avgTotal, avgMargin, winPct: wins/n, totals, margins };
}
function getRestDays(t, date) {
  const g = teams[t]?.games || [];
  if (g.length === 0) return 99;
  const ld = g[g.length - 1].date;
  const d1 = new Date(date.slice(0,4)+'-'+date.slice(4,6)+'-'+date.slice(6,8));
  const d2 = new Date(ld.slice(0,4)+'-'+ld.slice(4,6)+'-'+ld.slice(6,8));
  return Math.round((d1 - d2) / 86400000);
}

// Track by month
const strats = {};
function addPick(name, hit, date) {
  const month = date.slice(0, 6);
  if (!strats[name]) strats[name] = { all: {w:0,l:0}, months: {} };
  if (!strats[name].months[month]) strats[name].months[month] = {w:0,l:0};
  strats[name].all[hit ? 'w' : 'l']++;
  strats[name].months[month][hit ? 'w' : 'l']++;
}

for (const game of sorted) {
  const h = game.home_team, a = game.away_team;
  initTeam(h); initTeam(a);

  const actualTotal = game.home_score + game.away_score;
  const homeMargin = game.home_score - game.away_score;
  const h10 = getStats(h, 10), a10 = getStats(a, 10);
  const h5 = getStats(h, 5), a5 = getStats(a, 5);
  const h3 = getStats(h, 3), a3 = getStats(a, 3);
  const h15 = getStats(h, 15), a15 = getStats(a, 15);
  const hRest = getRestDays(h, game.date);
  const aRest = getRestDays(a, game.date);

  if (h10 && a10 && h5 && a5 && h3 && a3) {
    const predTotal = (h10.ppg + a10.ppg + h10.oppg + a10.oppg) / 2;

    // ===== ULTRA-SELECTIVE COMPOUND OVERS =====

    // C1: Both PPG 118+ last 5 AND both recent totals high
    if (h5.ppg > 118 && a5.ppg > 118 && h3.avgTotal > 225 && a3.avgTotal > 225) {
      addPick('C1_ppg118_plus_recent225_over', actualTotal > predTotal, game.date);
    }

    // C2: Both PPG 115+ last 5 AND both defense bad AND last game high
    const hLast = teams[h].games.length > 0 ? teams[h].games[teams[h].games.length-1] : null;
    const aLast = teams[a].games.length > 0 ? teams[a].games[teams[a].games.length-1] : null;
    if (h5.ppg > 115 && a5.ppg > 115 && h5.oppg > 112 && a5.oppg > 112 && hLast && aLast) {
      const hlt = hLast.pf + hLast.pa, alt = aLast.pf + aLast.pa;
      if (hlt > 225 && alt > 225) {
        addPick('C2_ppg115_def112_lastgame225_over', actualTotal > predTotal, game.date);
      }
    }

    // C3: Both avg total 225+ last 3 AND both ppg rising (3 > 5)
    if (h3.avgTotal > 225 && a3.avgTotal > 225 && h3.ppg > h5.ppg && a3.ppg > a5.ppg) {
      addPick('C3_total225_rising_scoring_over', actualTotal > predTotal, game.date);
    }

    // C4: Both PPG 116+ last 3 AND both defense 112+ last 5 AND combined total 235+
    if (h3.ppg > 116 && a3.ppg > 116 && h5.oppg > 112 && a5.oppg > 112 && h5.ppg + a5.ppg > 232) {
      addPick('C4_ppg116_def112_combined232_over', actualTotal > predTotal, game.date);
    }

    // C5: Multiple windows all high (relaxed vs all_windows_high but require last 3 also)
    if (h3.avgTotal > 224 && a3.avgTotal > 224 && h5.avgTotal > 222 && a5.avgTotal > 222 && h10.avgTotal > 220 && a10.avgTotal > 220) {
      addPick('C5_triple_window_high_over', actualTotal > predTotal, game.date);
    }

    // C6: Extreme combined scoring + bad defense double-check
    if (h5.ppg + a5.ppg > 235 && h10.oppg > 110 && a10.oppg > 110) {
      addPick('C6_extreme_combined_bad_def_over', actualTotal > predTotal, game.date);
    }

    // C7: Both PPG 118+ last 3 AND at least one team bad defense
    if (h3.ppg > 118 && a3.ppg > 118 && (h5.oppg > 113 || a5.oppg > 113)) {
      addPick('C7_ppg118_last3_one_bad_def_over', actualTotal > predTotal, game.date);
    }

    // C8: Combined PPG 240+ AND both avg total 225+ last 3
    if (h5.ppg + a5.ppg > 240 && h3.avgTotal > 225 && a3.avgTotal > 225) {
      addPick('C8_combined240_recent225_over', actualTotal > predTotal, game.date);
    }

    // ===== SPREAD STRATEGIES (at -110) =====
    // Use predicted margin from net rating; bet "home covers" when spread edge is large
    const netGap = h10.avgMargin - a10.avgMargin;
    const predMargin = netGap + 3.5; // 3.5 HCA

    // S1: Big net gap + home winning + bet home covers the market spread
    // "Home covers" = home wins by more than (predMargin * 0.5) — adjusted spread
    if (netGap > 10 && h5.winPct > 0.6) {
      const coverLine = predMargin * 0.4; // conservative
      addPick('S1_big_gap_home_wins_covers', homeMargin > coverLine, game.date);
    }

    // S2: Away team significantly better, bet away covers
    if (netGap < -10 && a5.winPct > 0.6) {
      const awayLine = -predMargin * 0.4;
      addPick('S2_big_gap_away_covers', homeMargin < -awayLine, game.date);
    }

    // S3: Home with streak advantage + net rating
    const hStreak = (() => {
      const g = teams[h].games; let len = 0;
      for (let i = g.length-1; i >= 0 && g[i].pf > g[i].pa; i--) len++;
      return len;
    })();
    const aStreak = (() => {
      const g = teams[a].games; let len = 0;
      for (let i = g.length-1; i >= 0 && g[i].pf < g[i].pa; i--) len++;
      return len;
    })();

    if (netGap > 8 && hStreak >= 3 && aStreak >= 2) {
      addPick('S3_net8_home3W_away2L_covers', homeMargin > predMargin * 0.3, game.date);
    }

    // ===== UNDER STRATEGIES =====
    // U1: Both defenses elite AND slow pace AND low recent totals
    if (h10.oppg < 108 && a10.oppg < 108 && h10.avgTotal < 215 && a10.avgTotal < 215) {
      addPick('U1_elite_def_low_totals_under', actualTotal < predTotal, game.date);
    }

    // U2: One team cold AND defensive game
    if ((h3.ppg < 105 || a3.ppg < 105) && h10.oppg < 110 && a10.oppg < 110) {
      addPick('U2_cold_team_good_def_under', actualTotal < predTotal, game.date);
    }

    // ===== REST-BASED STRATEGIES =====
    // R1: Away on B2B, home rested 2+ days, home net rating advantage
    if (aRest <= 1 && hRest >= 2 && netGap > 3) {
      addPick('R1_away_b2b_home_rested_advantage', homeMargin > 0, game.date);
    }

    // ===== COMBINATION: Multiple independent edges =====
    // MEGA1: High scoring + bad defense + recent high totals (triple convergence)
    if (h5.ppg > 115 && a5.ppg > 115 && h5.oppg > 113 && a5.oppg > 113 && h3.avgTotal > 225 && a3.avgTotal > 225) {
      addPick('MEGA1_triple_convergence_over', actualTotal > predTotal, game.date);
    }

    // MEGA2: Even stricter — PPG 117+, oppg 114+, recent total 228+
    if (h5.ppg > 117 && a5.ppg > 117 && h5.oppg > 114 && a5.oppg > 114 && h3.avgTotal > 228 && a3.avgTotal > 228) {
      addPick('MEGA2_ultra_convergence_over', actualTotal > predTotal, game.date);
    }

    // MEGA3: Best features from analysis: PPG 116+, last 3 total 224+, combined 236+
    if (h3.ppg > 116 && a3.ppg > 116 && h3.avgTotal > 224 && a3.avgTotal > 224 && h5.ppg + a5.ppg > 236) {
      addPick('MEGA3_scoring_total_combined_over', actualTotal > predTotal, game.date);
    }
  }

  teams[h].games.push({ date: game.date, pf: game.home_score, pa: game.away_score });
  teams[a].games.push({ date: game.date, pf: game.away_score, pa: game.home_score });
}

// Print results
console.log('=== ULTRA-SELECTIVE STRATEGY RESULTS (monthly) ===\n');
const months = ['202410', '202411', '202412', '202501', '202502'];

const entries = Object.entries(strats)
  .filter(([_, d]) => d.all.w + d.all.l >= 8)
  .sort((a, b) => {
    const aa = a[1].all, bb = b[1].all;
    return (bb.w / (bb.w + bb.l)) - (aa.w / (aa.w + aa.l));
  });

for (const [name, data] of entries) {
  const all = data.all;
  const total = all.w + all.l;
  const acc = (all.w / total * 100).toFixed(1);
  const pnl = all.w * 91 - all.l * 100;
  const roi = (pnl / (total * 100) * 100).toFixed(1);

  // Check Feb specifically
  const feb = data.months['202502'];
  const febStr = feb ? `Feb: ${feb.w}-${feb.l} (${(feb.w/(feb.w+feb.l)*100).toFixed(0)}%)` : 'Feb: —';

  const marker = acc >= 70 ? '***' : acc >= 65 ? '**' : acc >= 60 ? '*' : '';
  console.log(`${marker} ${name}: ${all.w}-${all.l} (${acc}%) PnL: $${pnl} ROI: ${roi}% | ${febStr}`);

  for (const m of months) {
    const md = data.months[m];
    if (md) {
      const mt = md.w + md.l;
      const label = m.slice(0,4)+'-'+m.slice(4);
      console.log(`    ${label}: ${md.w}-${md.l} (${(md.w/mt*100).toFixed(0)}%) PnL: $${md.w*91-md.l*100}`);
    }
  }
  console.log('');
}
