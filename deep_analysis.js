// Deep analysis: find high-accuracy -110 strategies
const fs = require('fs');
const window = {};
eval(fs.readFileSync('./webapp/js/parlay-engine.js', 'utf8'));
const data = JSON.parse(fs.readFileSync('./webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = [...data].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

// Build rich team history with many features
const teams = {};

function initTeam(t) {
  if (!teams[t]) teams[t] = {
    games: [], wins: 0, losses: 0,
    homeGames: [], awayGames: [],
    homeWins: 0, homeLosses: 0, awayWins: 0, awayLosses: 0,
    last10: [], streakType: null, streakLen: 0,
  };
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
  const pace = recent.reduce((s, x) => s + x.pace, 0) / n;
  return { ppg, oppg, avgTotal, avgMargin, winPct: wins/n, pace, totals, margins };
}

function getHomeStats(t, n) {
  const g = teams[t]?.homeGames || [];
  if (g.length < n) return null;
  const recent = g.slice(-n);
  const wins = recent.filter(x => x.pf > x.pa).length;
  return { winPct: wins/n, count: recent.length };
}

function getAwayStats(t, n) {
  const g = teams[t]?.awayGames || [];
  if (g.length < n) return null;
  const recent = g.slice(-n);
  const wins = recent.filter(x => x.pf > x.pa).length;
  return { winPct: wins/n, count: recent.length };
}

function getStreak(t) {
  const g = teams[t]?.games || [];
  if (g.length === 0) return { type: null, len: 0 };
  let type = g[g.length-1].pf > g[g.length-1].pa ? 'W' : 'L';
  let len = 0;
  for (let i = g.length - 1; i >= 0; i--) {
    const w = g[i].pf > g[i].pa;
    if ((type === 'W' && w) || (type === 'L' && !w)) len++;
    else break;
  }
  return { type, len };
}

function getRestDays(t, date) {
  const g = teams[t]?.games || [];
  if (g.length === 0) return 99;
  const lastDate = g[g.length - 1].date;
  // Simple date diff
  const d1 = new Date(date.slice(0,4) + '-' + date.slice(4,6) + '-' + date.slice(6,8));
  const d2 = new Date(lastDate.slice(0,4) + '-' + lastDate.slice(4,6) + '-' + lastDate.slice(6,8));
  return Math.round((d1 - d2) / (1000 * 60 * 60 * 24));
}

function isB2B(t, date) {
  return getRestDays(t, date) <= 1;
}

// Strategy testing framework
const strategies = {};

function addPick(name, hit) {
  if (!strategies[name]) strategies[name] = { w: 0, l: 0 };
  strategies[name][hit ? 'w' : 'l']++;
}

// Process games
for (const game of sorted) {
  const h = game.home_team;
  const a = game.away_team;
  initTeam(h);
  initTeam(a);

  const actualTotal = game.home_score + game.away_score;
  const homeMargin = game.home_score - game.away_score;
  const homeWon = homeMargin > 0;

  // Get pre-game stats (BEFORE updating)
  const h10 = getTeamStats(h, 10);
  const a10 = getTeamStats(a, 10);
  const h5 = getTeamStats(h, 5);
  const a5 = getTeamStats(a, 5);
  const h3 = getTeamStats(h, 3);
  const a3 = getTeamStats(a, 3);
  const h15 = getTeamStats(h, 15);
  const a15 = getTeamStats(a, 15);
  const hHome5 = getHomeStats(h, 5);
  const aAway5 = getAwayStats(a, 5);
  const hStreak = getStreak(h);
  const aStreak = getStreak(a);
  const hRest = getRestDays(h, game.date);
  const aRest = getRestDays(a, game.date);
  const hB2B = isB2B(h, game.date);
  const aB2B = isB2B(a, game.date);

  if (h10 && a10) {
    const predTotal = (h10.ppg + a10.ppg + h10.oppg + a10.oppg) / 2;
    const combinedPace = h10.pace + a10.pace;

    // ========== OVER STRATEGIES ==========

    // 1. Extreme pace convergence
    if (h10.pace > 100 && a10.pace > 100) {
      addPick('both_pace_100+_over', actualTotal > predTotal);
    }
    if (h5.pace > 102 && a5.pace > 102) {
      addPick('both_pace_102+_last5_over', actualTotal > predTotal);
    }

    // 2. Both teams hot scoring recently
    if (h5.ppg > 115 && a5.ppg > 115) {
      addPick('both_ppg_115+_last5_over', actualTotal > predTotal);
    }
    if (h3.ppg > 118 && a3.ppg > 118) {
      addPick('both_ppg_118+_last3_over', actualTotal > predTotal);
    }
    if (h5.ppg > 118 && a5.ppg > 118) {
      addPick('both_ppg_118+_last5_over', actualTotal > predTotal);
    }

    // 3. Both teams poor defense recently
    if (h5.oppg > 115 && a5.oppg > 115) {
      addPick('both_oppg_115+_last5_over', actualTotal > predTotal);
    }
    if (h10.oppg > 115 && a10.oppg > 115) {
      addPick('both_oppg_115+_last10_over', actualTotal > predTotal);
    }
    if (h5.oppg > 118 && a5.oppg > 118) {
      addPick('both_oppg_118+_last5_over', actualTotal > predTotal);
    }

    // 4. Recent games going high
    if (h3 && a3) {
      const hRecentAvgTotal = h3.avgTotal;
      const aRecentAvgTotal = a3.avgTotal;
      if (hRecentAvgTotal > 225 && aRecentAvgTotal > 225) {
        addPick('both_avg_total_225+_last3_over', actualTotal > predTotal);
      }
      if (hRecentAvgTotal > 230 && aRecentAvgTotal > 230) {
        addPick('both_avg_total_230+_last3_over', actualTotal > predTotal);
      }
    }

    // 5. High scoring with bad defense combo
    if (h5.ppg > 115 && a5.oppg > 115) {
      addPick('home_hot_away_leaky_over', actualTotal > predTotal);
    }
    if (a5.ppg > 115 && h5.oppg > 115) {
      addPick('away_hot_home_leaky_over', actualTotal > predTotal);
    }
    if ((h5.ppg > 115 && a5.oppg > 115) && (a5.ppg > 115 && h5.oppg > 115)) {
      addPick('BOTH_hot_and_leaky_over', actualTotal > predTotal);
    }

    // 6. Pace + defense combo
    if (combinedPace > 200 && h10.oppg > 112 && a10.oppg > 112) {
      addPick('high_pace_both_bad_def_over', actualTotal > predTotal);
    }
    if (combinedPace > 202 && (h10.oppg + a10.oppg) / 2 > 113) {
      addPick('very_high_pace_bad_avg_def_over', actualTotal > predTotal);
    }

    // 7. Multi-window consistency (all windows show high)
    if (h3 && a3 && h5 && a5) {
      const allWindowsHigh = h3.avgTotal > 220 && h5.avgTotal > 218 && h10.avgTotal > 215 &&
                             a3.avgTotal > 220 && a5.avgTotal > 218 && a10.avgTotal > 215;
      if (allWindowsHigh) {
        addPick('all_windows_high_over', actualTotal > predTotal);
      }
    }

    // 8. Scoring trajectory (getting hotter)
    if (h3 && h5 && a3 && a5) {
      const hHeating = h3.ppg > h5.ppg && h5.ppg > h10.ppg;
      const aHeating = a3.ppg > a5.ppg && a5.ppg > a10.ppg;
      if (hHeating && aHeating) {
        addPick('both_heating_trajectory_over', actualTotal > predTotal);
      }
    }

    // 9. Rest-based: both rested = higher scoring
    if (hRest >= 3 && aRest >= 3) {
      addPick('both_well_rested_over', actualTotal > predTotal);
    }

    // 10. B2B team UNDER (fatigue)
    if (hB2B && !aB2B) {
      addPick('home_b2b_away_rested_under', actualTotal < predTotal);
      addPick('home_b2b_away_rested_away_ml', homeMargin < 0);
    }
    if (aB2B && !hB2B) {
      addPick('away_b2b_home_rested_under', actualTotal < predTotal);
      addPick('away_b2b_home_rested_home_ml', homeMargin > 0);
    }

    // ========== SPREAD / ML STRATEGIES ==========

    // 11. Streak-based: hot team vs cold team
    if (hStreak.type === 'W' && hStreak.len >= 5 && aStreak.type === 'L' && aStreak.len >= 3) {
      addPick('home_5W_streak_away_3L_home_ml', homeWon);
    }
    if (hStreak.type === 'W' && hStreak.len >= 3 && aStreak.type === 'L' && aStreak.len >= 3) {
      addPick('home_3W_away_3L_home_ml', homeWon);
    }
    if (aStreak.type === 'W' && aStreak.len >= 5 && hStreak.type === 'L' && hStreak.len >= 3) {
      addPick('away_5W_home_3L_away_ml', !homeWon);
    }

    // 12. Net rating gap
    const netGap = h10.avgMargin - a10.avgMargin;
    if (netGap > 10) {
      addPick('net_rating_gap_10+_home_ml', homeWon);
      addPick('net_rating_gap_10+_home_spread', homeMargin > (netGap * 0.5));
    }
    if (netGap > 15) {
      addPick('net_rating_gap_15+_home_ml', homeWon);
    }
    if (netGap < -10) {
      addPick('net_rating_gap_10+_away_ml', !homeWon);
    }
    if (netGap < -15) {
      addPick('net_rating_gap_15+_away_ml', !homeWon);
    }

    // 13. Win% gap
    const wpctGap = h10.winPct - a10.winPct;
    if (wpctGap >= 0.6) {
      addPick('winpct_gap_60+_home_ml', homeWon);
    }
    if (wpctGap >= 0.5 && h10.avgMargin > 5) {
      addPick('winpct_50+_net5+_home_ml', homeWon);
    }
    if (wpctGap <= -0.6) {
      addPick('winpct_gap_60+_away_ml', !homeWon);
    }

    // 14. Home dominance at home
    if (hHome5 && hHome5.winPct >= 0.8 && aAway5 && aAway5.winPct <= 0.4) {
      addPick('home_dominance_away_weak_home_ml', homeWon);
    }

    // 15. Combined: multiple signals agree on OVER
    let overSignals = 0;
    if (h5.ppg > 112 && a5.ppg > 112) overSignals++;
    if (h5.oppg > 112 && a5.oppg > 112) overSignals++;
    if (combinedPace > 198) overSignals++;
    if (h3 && a3 && h3.avgTotal > 220 && a3.avgTotal > 220) overSignals++;
    if (h5.avgTotal > 218 && a5.avgTotal > 218) overSignals++;

    if (overSignals >= 4) {
      addPick('4+_over_signals_converge_over', actualTotal > predTotal);
    }
    if (overSignals >= 5) {
      addPick('5_over_signals_converge_over', actualTotal > predTotal);
    }
    if (overSignals >= 3) {
      addPick('3+_over_signals_converge_over', actualTotal > predTotal);
    }

    // 16. Extreme: when EVERYTHING points OVER
    if (overSignals >= 4 && h10.oppg > 112 && a10.oppg > 112) {
      addPick('4+_signals_plus_bad_def_over', actualTotal > predTotal);
    }

    // ========== CREATIVE STRATEGIES ==========

    // 17. "Regression to mean" — teams that scored very low recently bounce back
    if (h3 && a3) {
      if (h3.ppg < 100 || a3.ppg < 100) {
        addPick('one_team_cold_last3_over', actualTotal > predTotal);
      }
    }

    // 18. High variance teams (inconsistent scoring = exploitable)
    if (h5) {
      const hVar = h5.totals.reduce((s, t) => s + Math.pow(t - h5.avgTotal, 2), 0) / 5;
      const aVar = a5.totals.reduce((s, t) => s + Math.pow(t - a5.avgTotal, 2), 0) / 5;
      // Low variance + high scoring = reliable OVER
      if (hVar < 100 && aVar < 100 && h5.avgTotal > 220 && a5.avgTotal > 220) {
        addPick('low_var_high_total_over', actualTotal > predTotal);
      }
    }

    // 19. Blowout regression: after a blowout loss, teams play harder
    if (h3) {
      const lastHMargin = h3.margins[h3.margins.length - 1];
      const lastAMargin = a3.margins[a3.margins.length - 1];
      if (lastHMargin < -15 && a10.winPct < 0.5) {
        addPick('home_blowout_loss_bounce_home_ml', homeWon);
      }
      if (lastAMargin < -15 && h10.winPct < 0.5) {
        addPick('away_blowout_loss_bounce_away_ml', !homeWon);
      }
    }

    // 20. Combined NET + streak + home
    if (netGap > 8 && hStreak.type === 'W' && hStreak.len >= 2) {
      addPick('net8+_home_winning_home_ml', homeWon);
    }
    if (netGap > 8 && hStreak.type === 'W' && hStreak.len >= 2 && aStreak.type === 'L') {
      addPick('net8+_home_W_away_L_home_ml', homeWon);
    }

    // 21. Pace differential (fast vs slow)
    if (h10.pace > 100 && a10.pace < 96) {
      addPick('home_fast_away_slow_under', actualTotal < predTotal);
    }
    if (h10.pace < 96 && a10.pace > 100) {
      addPick('home_slow_away_fast_under', actualTotal < predTotal);
    }

    // 22. Defense-offense mismatch
    if (h10.oppg < 106 && a10.ppg < 106) {
      addPick('home_good_def_away_weak_off_under', actualTotal < predTotal);
    }
    if (a10.oppg < 106 && h10.ppg < 106) {
      addPick('away_good_def_home_weak_off_under', actualTotal < predTotal);
    }
    if ((h10.oppg < 106 && a10.ppg < 106) && (a10.oppg < 106 && h10.ppg < 106)) {
      addPick('both_good_def_both_weak_off_under', actualTotal < predTotal);
    }

    // 23. Rest advantage spread
    if (hRest >= 3 && aB2B) {
      addPick('home_rested_away_b2b_home_spread', homeMargin > 3);
      addPick('home_rested_away_b2b_home_ml', homeWon);
    }
    if (aRest >= 3 && hB2B) {
      addPick('away_rested_home_b2b_away_spread', homeMargin < -3);
      addPick('away_rested_home_b2b_away_ml', !homeWon);
    }

    // 24. COMPOUND: Multiple independent edges stacking
    // Over with: pace + scoring + defense all bad
    if (h5.pace > 99 && a5.pace > 99 && h5.ppg > 112 && a5.ppg > 112 && h5.oppg > 112 && a5.oppg > 112) {
      addPick('COMPOUND_pace_off_def_all_high_over', actualTotal > predTotal);
    }

    // 25. Score both high AND defense both bad AND last 3 totals high
    if (h3 && a3) {
      if (h5.ppg > 112 && a5.ppg > 112 && h5.oppg > 112 && a5.oppg > 112 && h3.avgTotal > 222 && a3.avgTotal > 222) {
        addPick('COMPOUND_scoring_def_recent_all_over', actualTotal > predTotal);
      }
    }

    // 26. Multi-window scoring trend + pace
    if (h3 && h5 && a3 && a5 && h15 && a15) {
      const hAllHot = h3.ppg > h5.ppg && h5.ppg > h10.ppg;
      const aAllHot = a3.ppg > a5.ppg && a5.ppg > a10.ppg;
      if (hAllHot && aAllHot && combinedPace > 198) {
        addPick('COMPOUND_heating_pace_over', actualTotal > predTotal);
      }
    }

    // 27. Defensive collapse + high pace
    if (h5.oppg > h10.oppg && a5.oppg > a10.oppg && combinedPace > 200) {
      addPick('both_def_declining_high_pace_over', actualTotal > predTotal);
    }

    // 28. Extreme scoring environment
    if (h5.ppg + a5.ppg > 235) {
      addPick('combined_ppg_235+_over', actualTotal > predTotal);
    }
    if (h5.ppg + a5.ppg > 240) {
      addPick('combined_ppg_240+_over', actualTotal > predTotal);
    }

    // 29. Back to back team scoring OVER (fatigue = less defense)
    if (hB2B) {
      addPick('home_b2b_over', actualTotal > predTotal);
    }
    if (aB2B) {
      addPick('away_b2b_over', actualTotal > predTotal);
    }
    if (hB2B || aB2B) {
      addPick('either_b2b_over', actualTotal > predTotal);
    }

    // 30. High totals last game for BOTH teams
    const hLastTotal = teams[h].games.length > 0 ? teams[h].games[teams[h].games.length-1].pf + teams[h].games[teams[h].games.length-1].pa : 0;
    const aLastTotal = teams[a].games.length > 0 ? teams[a].games[teams[a].games.length-1].pf + teams[a].games[teams[a].games.length-1].pa : 0;
    if (hLastTotal > 230 && aLastTotal > 230) {
      addPick('both_last_game_230+_over', actualTotal > predTotal);
    }

    // 31. Combine rest + scoring
    if (hRest >= 2 && aRest >= 2 && h5.ppg > 112 && a5.ppg > 112) {
      addPick('both_rested_both_scoring_over', actualTotal > predTotal);
    }

    // 32. Extreme net rating + home court
    if (h10.avgMargin > 7 && a10.avgMargin < -3) {
      addPick('strong_home_weak_away_net_home_ml', homeWon);
    }
    if (h10.avgMargin > 7 && a10.avgMargin < -5) {
      addPick('strong_home_very_weak_away_home_ml', homeWon);
    }

    // 33. Combined multiple spread signals
    let homeEdgeSignals = 0;
    if (h10.avgMargin > 5) homeEdgeSignals++;
    if (a10.avgMargin < -2) homeEdgeSignals++;
    if (h10.winPct > 0.6) homeEdgeSignals++;
    if (a10.winPct < 0.4) homeEdgeSignals++;
    if (hStreak.type === 'W' && hStreak.len >= 2) homeEdgeSignals++;
    if (aStreak.type === 'L' && aStreak.len >= 2) homeEdgeSignals++;
    if (hHome5 && hHome5.winPct >= 0.6) homeEdgeSignals++;
    if (aAway5 && aAway5.winPct <= 0.4) homeEdgeSignals++;
    if (hRest > aRest) homeEdgeSignals++;

    if (homeEdgeSignals >= 7) {
      addPick('7+_home_edge_signals_home_spread', homeMargin > 3);
      addPick('7+_home_edge_signals_home_ml', homeWon);
    }
    if (homeEdgeSignals >= 8) {
      addPick('8+_home_edge_signals_home_ml', homeWon);
    }
    if (homeEdgeSignals >= 6) {
      addPick('6+_home_edge_signals_home_ml', homeWon);
    }

    // 34. Conference/record mismatch
    if (h10.winPct >= 0.8 && a10.winPct <= 0.3) {
      addPick('wpct_80v30_home_ml', homeWon);
    }
    if (h10.winPct >= 0.7 && a10.winPct <= 0.3) {
      addPick('wpct_70v30_home_ml', homeWon);
    }

    // 35. Both teams B2B
    if (hB2B && aB2B) {
      addPick('both_b2b_over', actualTotal > predTotal);
    }

    // 36. UNDER strategies with strong defense
    if (h10.oppg < 105 && a10.oppg < 105) {
      addPick('both_elite_def_under', actualTotal < predTotal);
    }
    if (h10.oppg < 108 && a10.oppg < 108 && combinedPace < 196) {
      addPick('both_good_def_slow_pace_under', actualTotal < predTotal);
    }

    // 37. Rebound after low-scoring game
    if (hLastTotal < 200 && aLastTotal < 200) {
      addPick('both_last_game_low_over', actualTotal > predTotal);
    }

    // 38. Combined: away team on losing streak + B2B + home team rested
    if (aStreak.type === 'L' && aStreak.len >= 2 && aB2B && hRest >= 2) {
      addPick('away_losing_b2b_home_rested_home_ml', homeWon);
    }

    // 39. Home team on win streak at home
    if (hHome5 && hHome5.winPct >= 0.8 && hStreak.type === 'W' && hStreak.len >= 3) {
      addPick('home_dominant_at_home_streaking_home_ml', homeWon);
    }
  }

  // Update team history
  const pace = (game.home_score + game.away_score); // simplified pace proxy
  teams[h].games.push({ date: game.date, pf: game.home_score, pa: game.away_score, pace, isHome: true });
  teams[a].games.push({ date: game.date, pf: game.away_score, pa: game.home_score, pace, isHome: false });
  teams[h].homeGames.push({ date: game.date, pf: game.home_score, pa: game.away_score });
  teams[a].awayGames.push({ date: game.date, pf: game.away_score, pa: game.home_score });
}

// Print results sorted by accuracy (only strategies with 10+ picks)
console.log('\n=== All Strategies (10+ picks, sorted by accuracy) ===');
const results = Object.entries(strategies)
  .filter(([_, v]) => v.w + v.l >= 10)
  .map(([name, v]) => {
    const total = v.w + v.l;
    const acc = v.w / total;
    const pnl = v.w * 91 - v.l * 100;
    const roi = pnl / (total * 100) * 100;
    return { name, w: v.w, l: v.l, total, acc, pnl, roi };
  })
  .sort((a, b) => b.acc - a.acc);

for (const r of results) {
  const marker = r.acc >= 0.65 ? '***' : r.acc >= 0.60 ? '**' : r.acc >= 0.55 ? '*' : '';
  console.log(`${marker} ${r.name}: ${r.w}-${r.l} (${(r.acc*100).toFixed(1)}%) PnL: $${r.pnl} ROI: ${r.roi.toFixed(1)}% ${marker}`);
}

// Print strategies that beat 65%
console.log('\n=== TOP STRATEGIES (65%+) ===');
const top = results.filter(r => r.acc >= 0.65);
for (const r of top) {
  console.log(`${r.name}: ${r.w}-${r.l} (${(r.acc*100).toFixed(1)}%) PnL: $${r.pnl} ROI: ${r.roi.toFixed(1)}%`);
}

console.log('\n=== PROFITABLE at -110 (>52.4%, 20+ picks) ===');
const profitable = results.filter(r => r.acc > 0.524 && r.total >= 20).sort((a, b) => b.roi - a.roi);
for (const r of profitable) {
  console.log(`${r.name}: ${r.w}-${r.l} (${(r.acc*100).toFixed(1)}%) PnL: $${r.pnl} ROI: ${r.roi.toFixed(1)}%`);
}
