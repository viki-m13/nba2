// Deep analysis part 2: Monthly breakdown of top strategies
const fs = require('fs');
const window = {};
eval(fs.readFileSync('./webapp/js/parlay-engine.js', 'utf8'));
const data = JSON.parse(fs.readFileSync('./webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = [...data].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

const teams = {};
function initTeam(t) {
  if (!teams[t]) teams[t] = { games: [], homeGames: [], awayGames: [] };
}
function getTeamStats(t, n) {
  const g = teams[t]?.games || [];
  if (g.length < n) return null;
  const recent = g.slice(-n);
  const ppg = recent.reduce((s, x) => s + x.pf, 0) / n;
  const oppg = recent.reduce((s, x) => s + x.pa, 0) / n;
  const totals = recent.map(x => x.pf + x.pa);
  const avgTotal = totals.reduce((s, x) => s + x, 0) / n;
  const margins = recent.map(x => x.pf - x.pa);
  const avgMargin = margins.reduce((s, x) => s + x, 0) / n;
  const wins = recent.filter(x => x.pf > x.pa).length;
  return { ppg, oppg, avgTotal, avgMargin, winPct: wins/n, totals, margins };
}

// Track strategies by month
const monthlyResults = {};
function addPick(name, hit, date) {
  const month = date.slice(0, 6); // YYYYMM
  if (!monthlyResults[name]) monthlyResults[name] = {};
  if (!monthlyResults[name][month]) monthlyResults[name][month] = { w: 0, l: 0 };
  if (!monthlyResults[name]['all']) monthlyResults[name]['all'] = { w: 0, l: 0 };
  monthlyResults[name][month][hit ? 'w' : 'l']++;
  monthlyResults[name]['all'][hit ? 'w' : 'l']++;
}

for (const game of sorted) {
  const h = game.home_team;
  const a = game.away_team;
  initTeam(h); initTeam(a);

  const actualTotal = game.home_score + game.away_score;
  const h10 = getTeamStats(h, 10);
  const a10 = getTeamStats(a, 10);
  const h5 = getTeamStats(h, 5);
  const a5 = getTeamStats(a, 5);
  const h3 = getTeamStats(h, 3);
  const a3 = getTeamStats(a, 3);
  const h15 = getTeamStats(h, 15);
  const a15 = getTeamStats(a, 15);

  if (h10 && a10 && h5 && a5 && h3 && a3) {
    const predTotal = (h10.ppg + a10.ppg + h10.oppg + a10.oppg) / 2;
    const combinedPace = h10.avgTotal / 2 + a10.avgTotal / 2;

    // Top OVER strategies from analysis 1
    // Strategy 1: all_windows_high_over
    if (h15 && a15) {
      const h5t = h5.avgTotal, a5t = a5.avgTotal;
      const h10t = h10.avgTotal, a10t = a10.avgTotal;
      const h15t = h15.avgTotal, a15t = a15.avgTotal;
      if (h5t > 224 && a5t > 224 && h10t > 223 && a10t > 223 && h15t > 222 && a15t > 222) {
        addPick('all_windows_high_over', actualTotal > predTotal, game.date);
      }
    }

    // Strategy 2: both_avg_total_225+_last3_over
    if (h3.avgTotal > 225 && a3.avgTotal > 225) {
      addPick('both_avg_total_225+_last3_over', actualTotal > predTotal, game.date);
    }

    // Strategy 3: both_avg_total_230+_last3_over
    if (h3.avgTotal > 230 && a3.avgTotal > 230) {
      addPick('both_avg_total_230+_last3_over', actualTotal > predTotal, game.date);
    }

    // Strategy 4: 4+_signals_plus_bad_def_over
    let overSignals = 0;
    if (h5.ppg > 112 && a5.ppg > 112) overSignals++;
    if (h5.oppg > 112 && a5.oppg > 112) overSignals++;
    if (combinedPace > 198) overSignals++;
    if (h3.avgTotal > 220 && a3.avgTotal > 220) overSignals++;
    if (h5.avgTotal > 218 && a5.avgTotal > 218) overSignals++;
    if (overSignals >= 4 && h10.oppg > 112 && a10.oppg > 112) {
      addPick('4+_signals_plus_bad_def_over', actualTotal > predTotal, game.date);
    }

    // Strategy 5: combined_ppg_240+_over
    if (h5.ppg + a5.ppg > 240) {
      addPick('combined_ppg_240+_over', actualTotal > predTotal, game.date);
    }

    // Strategy 6: both_ppg_118+_last5_over (high accuracy, small sample)
    if (h5.ppg > 118 && a5.ppg > 118) {
      addPick('both_ppg_118+_last5_over', actualTotal > predTotal, game.date);
    }

    // Strategy 7: COMPOUND_scoring_def_recent_all_over
    if (h5.ppg > 112 && a5.ppg > 112 && h5.oppg > 112 && a5.oppg > 112 && h3.avgTotal > 222 && a3.avgTotal > 222) {
      addPick('COMPOUND_scoring_def_recent_all_over', actualTotal > predTotal, game.date);
    }

    // Strategy 8: 5_over_signals_converge
    if (overSignals >= 5) {
      addPick('5_over_signals_converge_over', actualTotal > predTotal, game.date);
    }

    // Strategy 9: both_last_game_230+_over
    const hLastTotal = teams[h].games.length > 0 ? teams[h].games[teams[h].games.length-1].pf + teams[h].games[teams[h].games.length-1].pa : 0;
    const aLastTotal = teams[a].games.length > 0 ? teams[a].games[teams[a].games.length-1].pf + teams[a].games[teams[a].games.length-1].pa : 0;
    if (hLastTotal > 230 && aLastTotal > 230) {
      addPick('both_last_game_230+_over', actualTotal > predTotal, game.date);
    }

    // Strategy 10: both_ppg_118+_last3_over
    if (h3.ppg > 118 && a3.ppg > 118) {
      addPick('both_ppg_118+_last3_over', actualTotal > predTotal, game.date);
    }
  }

  const pace = game.home_score + game.away_score;
  teams[h].games.push({ date: game.date, pf: game.home_score, pa: game.away_score, pace });
  teams[a].games.push({ date: game.date, pf: game.away_score, pa: game.home_score, pace });
}

// Print monthly breakdown for each strategy
console.log('=== MONTHLY BREAKDOWN OF TOP OVER STRATEGIES ===\n');
const months = ['202410', '202411', '202412', '202501', '202502'];

for (const [name, data] of Object.entries(monthlyResults)) {
  const all = data['all'];
  const total = all.w + all.l;
  if (total < 10) continue;
  const acc = (all.w / total * 100).toFixed(1);
  const pnl = all.w * 91 - all.l * 100;

  console.log(`${name}: ${all.w}-${all.l} (${acc}%) PnL: $${pnl}`);
  for (const m of months) {
    const md = data[m];
    if (md) {
      const mt = md.w + md.l;
      const ma = (md.w / mt * 100).toFixed(1);
      const mp = md.w * 91 - md.l * 100;
      const label = m.slice(0,4) + '-' + m.slice(4);
      console.log(`  ${label}: ${md.w}-${md.l} (${ma}%) PnL: $${mp}`);
    }
  }
  console.log('');
}
