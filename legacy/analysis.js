// Deep analysis script to find high-accuracy betting signals
// Walk-forward backtesting with correct margin computation

const fs = require('fs');
const data = JSON.parse(fs.readFileSync('webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = data.sort((a, b) => a.date.localeCompare(b.date));

console.log(`Total games: ${sorted.length}`);
console.log(`Date range: ${sorted[0].date} to ${sorted[sorted.length-1].date}`);

// ═══════════════════════════════════════════════════════════════════
// TEAM STATE TRACKER - tracks multiple windows of stats per team
// ═══════════════════════════════════════════════════════════════════

class TeamTracker {
  constructor() { this.teams = {}; }

  update(team, pf, pa, date, isHome, oppTeam) {
    if (!this.teams[team]) this.teams[team] = [];
    this.teams[team].push({ pf, pa, margin: pf - pa, date, isHome, opp: oppTeam, total: pf + pa });
    if (this.teams[team].length > 40) this.teams[team] = this.teams[team].slice(-30);
  }

  getWindow(team, n) {
    const h = this.teams[team] || [];
    return h.slice(-n);
  }

  getStats(team, n) {
    const w = this.getWindow(team, n);
    if (w.length < Math.min(n, 8)) return null;
    const pf = w.map(g => g.pf);
    const pa = w.map(g => g.pa);
    const margins = w.map(g => g.margin);
    const totals = w.map(g => g.total);
    const avg = arr => arr.reduce((a,b) => a+b, 0) / arr.length;
    const std = arr => { const m = avg(arr); return Math.sqrt(arr.reduce((s,x) => s + (x-m)**2, 0) / arr.length); };

    const homeGames = w.filter(g => g.isHome);
    const awayGames = w.filter(g => !g.isHome);

    return {
      offRating: avg(pf),
      defRating: avg(pa),
      netRating: avg(margins),
      winPct: margins.filter(m => m > 0).length / w.length,
      avgTotal: avg(totals),
      stdMargin: std(margins),
      stdTotal: std(totals),
      homeWinPct: homeGames.length >= 3 ? homeGames.filter(g => g.margin > 0).length / homeGames.length : null,
      awayWinPct: awayGames.length >= 3 ? awayGames.filter(g => g.margin > 0).length / awayGames.length : null,
      homeAvgMargin: homeGames.length >= 3 ? avg(homeGames.map(g => g.margin)) : null,
      awayAvgMargin: awayGames.length >= 3 ? avg(awayGames.map(g => g.margin)) : null,
      last3: w.slice(-3).map(g => g.margin),
      last5PF: avg(w.slice(-5).map(g => g.pf)),
      last5PA: avg(w.slice(-5).map(g => g.pa)),
      games: w.length,
      recentTotals: w.slice(-5).map(g => g.total),
      // Pace proxy
      avgPace: avg(totals) / 2,
      // ATS-like: how often team covers spread (positive margin games above their average)
      coverRate: w.filter(g => g.margin > avg(margins)).length / w.length,
      // Scoring trends
      last5Off: w.length >= 5 ? avg(w.slice(-5).map(g => g.pf)) : null,
      first5Off: w.length >= 10 ? avg(w.slice(0, 5).map(g => g.pf)) : null,
      last5Def: w.length >= 5 ? avg(w.slice(-5).map(g => g.pa)) : null,
      first5Def: w.length >= 10 ? avg(w.slice(0, 5).map(g => g.pa)) : null,
      // Blowout frequency
      blowoutWinPct: w.filter(g => g.margin >= 10).length / w.length,
      closeGamePct: w.filter(g => Math.abs(g.margin) <= 5).length / w.length,
      closeLossPct: w.filter(g => g.margin >= -5 && g.margin < 0).length / w.length,
    };
  }
}

const tracker = new TeamTracker();
const results = {
  // Track different strategy results
  strategies: {}
};

function addResult(name, date, hit, betType) {
  if (!results.strategies[name]) results.strategies[name] = [];
  results.strategies[name].push({ date, hit, pnl: hit ? 91 : -100, betType });
}

// ═══════════════════════════════════════════════════════════════════
// WALK-FORWARD BACKTESTING
// ═══════════════════════════════════════════════════════════════════

for (const game of sorted) {
  const home = game.home_team;
  const away = game.away_team;
  const actualHomeMargin = game.home_score - game.away_score;
  const actualTotal = game.home_score + game.away_score;

  // Get current stats BEFORE updating (walk-forward)
  const h10 = tracker.getStats(home, 10);
  const a10 = tracker.getStats(away, 10);
  const h15 = tracker.getStats(home, 15);
  const a15 = tracker.getStats(away, 15);
  const h5 = tracker.getStats(home, 5);
  const a5 = tracker.getStats(away, 5);
  const h20 = tracker.getStats(home, 20);
  const a20 = tracker.getStats(away, 20);

  if (h10 && a10 && h15 && a15 && h5 && a5) {
    const predTotal = (h10.offRating + a10.offRating + h10.defRating + a10.defRating) / 2;
    const predHomeMargin = h10.netRating - a10.netRating + 3.5; // 3.5 HCA

    // ═════════════════════════════════════════════════════════
    // STRATEGY 1: "Defensive Squeeze" - UNDER when both teams defend well
    // and recent totals are trending down
    // ═════════════════════════════════════════════════════════
    {
      const bothGoodDef = h10.defRating <= 110 && a10.defRating <= 110;
      const bothAvgDef = h10.defRating <= 112 && a10.defRating <= 112;
      const recentHomeTotals = h5.avgTotal;
      const recentAwayTotals = a5.avgTotal;
      const lowRecentTotals = recentHomeTotals < 215 && recentAwayTotals < 215;
      const lowPace = h10.avgPace < 108 && a10.avgPace < 108;

      if (bothGoodDef && lowPace) {
        addResult('def_squeeze_strong', game.date, actualTotal < predTotal, 'under');
      }
      if (bothAvgDef && lowPace && lowRecentTotals) {
        addResult('def_squeeze_converge', game.date, actualTotal < predTotal, 'under');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 2: "Pace Explosion" - OVER when both teams play fast
    // and recent scoring is trending up
    // ═════════════════════════════════════════════════════════
    {
      const bothFast = h10.avgPace > 112 && a10.avgPace > 112;
      const bothHotOff = h5.offRating > h10.offRating && a5.offRating > a10.offRating;
      const highRecent = h5.avgTotal > 220 && a5.avgTotal > 220;

      if (bothFast && bothHotOff) {
        addResult('pace_explosion', game.date, actualTotal > predTotal, 'over');
      }
      if (bothFast && highRecent) {
        addResult('pace_explosion_recent', game.date, actualTotal > predTotal, 'over');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 3: "Home Dog Regression" - Bet on home underdogs
    // when they're undervalued (good recent form, slight underdog)
    // ═════════════════════════════════════════════════════════
    {
      const homeIsUnderdog = predHomeMargin < 0;
      const homeSlightDog = predHomeMargin > -5 && predHomeMargin < 0;
      const homeGoodRecent = h5.winPct >= 0.6;
      const homeStrongAtHome = h10.homeWinPct !== null && h10.homeWinPct >= 0.6;
      const awayBadAway = a10.awayWinPct !== null && a10.awayWinPct <= 0.4;

      if (homeSlightDog && homeGoodRecent) {
        addResult('home_dog_form', game.date, actualHomeMargin > 0, 'home_ml');
      }
      if (homeSlightDog && homeStrongAtHome) {
        addResult('home_dog_strong_home', game.date, actualHomeMargin > 0, 'home_ml');
      }
      if (homeSlightDog && homeGoodRecent && homeStrongAtHome) {
        addResult('home_dog_converge', game.date, actualHomeMargin > 0, 'home_ml');
      }
      if (homeSlightDog && awayBadAway) {
        addResult('home_dog_opp_road_weak', game.date, actualHomeMargin > 0, 'home_ml');
      }
      if (homeSlightDog && homeGoodRecent && awayBadAway) {
        addResult('home_dog_full_converge', game.date, actualHomeMargin > 0, 'home_ml');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 4: "Regression to Mean" (fade hot/cold streaks)
    // Teams that won/lost many recent games but underlying stats don't support it
    // ═════════════════════════════════════════════════════════
    {
      // Home team overperforming: high win% but net rating doesn't support it
      const homeOverperf = h10.winPct >= 0.7 && h10.netRating < 3;
      const awayOverperf = a10.winPct >= 0.7 && a10.netRating < 3;
      const homeUnderperf = h10.winPct <= 0.3 && h10.netRating > -3;
      const awayUnderperf = a10.winPct <= 0.3 && a10.netRating > -3;

      // Fade home overperformer
      if (homeOverperf && !awayUnderperf) {
        addResult('fade_home_hot', game.date, actualHomeMargin < predHomeMargin, 'away_cover');
      }
      // Fade away overperformer
      if (awayOverperf && !homeUnderperf) {
        addResult('fade_away_hot', game.date, actualHomeMargin > predHomeMargin, 'home_cover');
      }
      // Back home underperformer (regression up)
      if (homeUnderperf && !awayOverperf) {
        addResult('back_home_cold', game.date, actualHomeMargin > predHomeMargin, 'home_cover');
      }
      // Combined: home underperforming + away overperforming
      if (homeUnderperf && awayOverperf) {
        addResult('double_regression', game.date, actualHomeMargin > predHomeMargin, 'home_cover');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 5: "Consistency Exploiter" - bet UNDER when both
    // teams have low variance in their scoring
    // ═════════════════════════════════════════════════════════
    {
      const bothLowVar = h10.stdTotal < 15 && a10.stdTotal < 15;
      const bothVeryLowVar = h10.stdTotal < 12 && a10.stdTotal < 12;
      const totalBelowRecent = predTotal < (h10.avgTotal + a10.avgTotal) / 2;

      if (bothLowVar && h10.defRating <= 112 && a10.defRating <= 112) {
        addResult('consistency_under', game.date, actualTotal < predTotal, 'under');
      }
      if (bothVeryLowVar) {
        addResult('very_consistent_under', game.date, actualTotal < predTotal, 'under');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 6: "Short Window Momentum Shift" -
    // When 5-game and 15-game windows diverge significantly
    // ═════════════════════════════════════════════════════════
    {
      const homeImproving = h5.netRating - h15.netRating > 5;
      const homeDeclining = h15.netRating - h5.netRating > 5;
      const awayImproving = a5.netRating - a15.netRating > 5;
      const awayDeclining = a15.netRating - a5.netRating > 5;

      // Home improving + away declining = strong home edge
      if (homeImproving && awayDeclining) {
        addResult('momentum_home', game.date, actualHomeMargin > 0, 'home_ml');
      }
      // Away improving + home declining = away edge
      if (awayImproving && homeDeclining) {
        addResult('momentum_away', game.date, actualHomeMargin < 0, 'away_ml');
      }

      // Bigger thresholds
      const homeHotStreak = h5.netRating - h15.netRating > 8;
      const awayHotStreak = a5.netRating - a15.netRating > 8;
      const homeColdStreak = h15.netRating - h5.netRating > 8;
      const awayColdStreak = a15.netRating - a5.netRating > 8;

      if (homeHotStreak && awayColdStreak) {
        addResult('momentum_home_strong', game.date, actualHomeMargin > 0, 'home_ml');
      }
      if (awayHotStreak && homeColdStreak) {
        addResult('momentum_away_strong', game.date, actualHomeMargin < 0, 'away_ml');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 7: "Blowout Predictor" - When a dominant team faces a weak team
    // Bet the favorite to win by a large margin (spread)
    // ═════════════════════════════════════════════════════════
    {
      const netDiff = h10.netRating - a10.netRating;
      const hugeGap = Math.abs(netDiff) > 12;
      const favIsHome = netDiff > 0;

      if (hugeGap) {
        // Favorite (whoever has better net rating) covers a spread
        if (favIsHome) {
          addResult('blowout_fav_home', game.date, actualHomeMargin > predHomeMargin * 0.5, 'fav_cover');
        } else {
          addResult('blowout_fav_away', game.date, actualHomeMargin < predHomeMargin * 0.5, 'fav_cover');
        }
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 8: "Total Stability" - When both teams' recent
    // totals are very stable and cluster around a specific range
    // ═════════════════════════════════════════════════════════
    {
      const hRecentTotals = h5.recentTotals || [];
      const aRecentTotals = a5.recentTotals || [];

      if (hRecentTotals.length >= 5 && aRecentTotals.length >= 5) {
        const allBelow220 = hRecentTotals.every(t => t < 220) && aRecentTotals.every(t => t < 220);
        const allAbove220 = hRecentTotals.every(t => t > 220) && aRecentTotals.every(t => t > 220);

        if (allBelow220 && h10.defRating < 112 && a10.defRating < 112) {
          addResult('total_cluster_low', game.date, actualTotal < predTotal, 'under');
        }
        if (allAbove220 && h10.offRating > 112 && a10.offRating > 112) {
          addResult('total_cluster_high', game.date, actualTotal > predTotal, 'over');
        }
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 9: "Close Game Expert" - Teams that play lots of
    // close games (high variance in outcomes)
    // ═════════════════════════════════════════════════════════
    {
      const homeCloseGameTeam = h10.closeGamePct >= 0.5;
      const awayCloseGameTeam = a10.closeGamePct >= 0.5;
      const homeCloseLosses = h10.closeLossPct >= 0.3;
      const awayCloseLosses = a10.closeLossPct >= 0.3;

      // Teams that lose close games are "unlucky" - expect regression
      if (homeCloseLosses && !awayCloseLosses) {
        addResult('close_loss_regression_home', game.date, actualHomeMargin > predHomeMargin, 'home_cover');
      }
      if (awayCloseLosses && !homeCloseLosses) {
        addResult('close_loss_regression_away', game.date, actualHomeMargin < predHomeMargin, 'away_cover');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 10: "Defensive Mismatch Total" - One team scores a lot,
    // other can't defend → OVER. One team defends well, other can't score → UNDER
    // ═════════════════════════════════════════════════════════
    {
      const homeOffStrong = h10.offRating >= 115;
      const awayDefWeak = a10.defRating >= 115;
      const awayOffStrong = a10.offRating >= 115;
      const homeDefWeak = h10.defRating >= 115;

      // Both matchups favor scoring
      if ((homeOffStrong && awayDefWeak) || (awayOffStrong && homeDefWeak)) {
        addResult('def_mismatch_over', game.date, actualTotal > predTotal, 'over');
      }
      if ((homeOffStrong && awayDefWeak) && (awayOffStrong && homeDefWeak)) {
        addResult('double_mismatch_over', game.date, actualTotal > predTotal, 'over');
      }

      const homeDefStrong = h10.defRating <= 108;
      const awayOffWeak = a10.offRating <= 108;
      const awayDefStrong = a10.defRating <= 108;
      const homeOffWeak = h10.offRating <= 108;

      if ((homeDefStrong && awayOffWeak) || (awayDefStrong && homeOffWeak)) {
        addResult('def_mismatch_under', game.date, actualTotal < predTotal, 'under');
      }
      if ((homeDefStrong && awayOffWeak) && (awayDefStrong && homeOffWeak)) {
        addResult('double_mismatch_under', game.date, actualTotal < predTotal, 'under');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 11: "Expected Win % Regression" - Pythagorean W%
    // vs actual W% - fade teams with big gap
    // ═════════════════════════════════════════════════════════
    {
      const pythag = (off, def) => {
        const pf = Math.pow(off, 13.91);
        const pa = Math.pow(def, 13.91);
        return pf / (pf + pa);
      };

      const hExpWin = pythag(h15.offRating, h15.defRating);
      const aExpWin = pythag(a15.offRating, a15.defRating);
      const hLuckGap = h15.winPct - hExpWin;
      const aLuckGap = a15.winPct - aExpWin;

      // Home is "lucky" (winning more than expected) AND away is "unlucky"
      if (hLuckGap > 0.15 && aLuckGap < -0.15) {
        addResult('pythag_fade_home', game.date, actualHomeMargin < predHomeMargin, 'away_cover');
      }
      if (hLuckGap < -0.15 && aLuckGap > 0.15) {
        addResult('pythag_fade_away', game.date, actualHomeMargin > predHomeMargin, 'home_cover');
      }

      // Extreme luck gaps
      if (hLuckGap > 0.2 && aLuckGap < -0.1) {
        addResult('pythag_extreme_fade_home', game.date, actualHomeMargin < predHomeMargin, 'away_cover');
      }
      if (hLuckGap < -0.2 && aLuckGap > 0.1) {
        addResult('pythag_extreme_fade_away', game.date, actualHomeMargin > predHomeMargin, 'home_cover');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 12: "Offensive Cooling" - When high-scoring teams'
    // last 5 games show declining scoring → UNDER
    // ═════════════════════════════════════════════════════════
    {
      if (h10.last5Off && h10.first5Off && a10.last5Off && a10.first5Off) {
        const homeCooling = h10.first5Off - h10.last5Off > 4;
        const awayCooling = a10.first5Off - a10.last5Off > 4;
        const homeHeating = h10.last5Off - h10.first5Off > 4;
        const awayHeating = a10.last5Off - a10.first5Off > 4;

        if (homeCooling && awayCooling) {
          addResult('both_cooling_under', game.date, actualTotal < predTotal, 'under');
        }
        if (homeHeating && awayHeating) {
          addResult('both_heating_over', game.date, actualTotal > predTotal, 'over');
        }
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 13: "Road Warrior" - Away team with significantly
    // better road record than home team's home record
    // ═════════════════════════════════════════════════════════
    {
      if (a10.awayWinPct !== null && h10.homeWinPct !== null) {
        const awayBetterOnRoad = a10.awayWinPct > 0.6 && h10.homeWinPct < 0.5;
        const awayMuchBetter = a10.awayWinPct > 0.7 && h10.homeWinPct < 0.4;
        const awayBetterOverall = a10.netRating > h10.netRating + 5;

        if (awayBetterOnRoad && awayBetterOverall) {
          addResult('road_warrior', game.date, actualHomeMargin < 0, 'away_ml');
        }
        if (awayMuchBetter) {
          addResult('road_warrior_strong', game.date, actualHomeMargin < 0, 'away_ml');
        }
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 14: "Convergent Confidence" - Multiple independent
    // signals all pointing the same direction on the total
    // ═════════════════════════════════════════════════════════
    {
      let underSignals = 0;
      let overSignals = 0;

      // Signal 1: Both teams good defense
      if (h10.defRating < 110 && a10.defRating < 110) underSignals++;
      if (h10.defRating > 114 && a10.defRating > 114) overSignals++;

      // Signal 2: Recent totals trending
      if (h5.avgTotal < h10.avgTotal - 3 && a5.avgTotal < a10.avgTotal - 3) underSignals++;
      if (h5.avgTotal > h10.avgTotal + 3 && a5.avgTotal > a10.avgTotal + 3) overSignals++;

      // Signal 3: Pace
      if (h10.avgPace < 108 && a10.avgPace < 108) underSignals++;
      if (h10.avgPace > 114 && a10.avgPace > 114) overSignals++;

      // Signal 4: Scoring consistency
      if (h10.stdTotal < 12 && a10.stdTotal < 12) underSignals++;

      // Signal 5: Defensive trend
      if (h5.defRating < h10.defRating && a5.defRating < a10.defRating) underSignals++;
      if (h5.defRating > h10.defRating && a5.defRating > a10.defRating) overSignals++;

      // Signal 6: Offensive cooling/heating
      if (h5.offRating < h10.offRating - 2 && a5.offRating < a10.offRating - 2) underSignals++;
      if (h5.offRating > h10.offRating + 2 && a5.offRating > a10.offRating + 2) overSignals++;

      if (underSignals >= 3) {
        addResult('convergent_under_3', game.date, actualTotal < predTotal, 'under');
      }
      if (underSignals >= 4) {
        addResult('convergent_under_4', game.date, actualTotal < predTotal, 'under');
      }
      if (overSignals >= 3) {
        addResult('convergent_over_3', game.date, actualTotal > predTotal, 'over');
      }
      if (overSignals >= 4) {
        addResult('convergent_over_4', game.date, actualTotal > predTotal, 'over');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 15: "Spread Compression" - When the predicted
    // margin is small but one team has much better quality metrics
    // ═════════════════════════════════════════════════════════
    {
      const netDiff = h10.netRating - a10.netRating;
      const offDiff = h10.offRating - a10.offRating;
      const defDiff = a10.defRating - h10.defRating; // reversed: lower is better for def

      // Close game expected but quality disparity
      if (Math.abs(predHomeMargin) < 4) {
        if (netDiff > 6 && offDiff > 3) {
          addResult('compressed_home_quality', game.date, actualHomeMargin > 0, 'home_ml');
        }
        if (netDiff < -6 && offDiff < -3) {
          addResult('compressed_away_quality', game.date, actualHomeMargin < 0, 'away_ml');
        }
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 16: "Multi-Window Total Convergence" - When 5, 10,
    // and 15 game windows all agree on total direction
    // ═════════════════════════════════════════════════════════
    {
      const h5total = h5.avgTotal;
      const a5total = a5.avgTotal;
      const h10total = h10.avgTotal;
      const a10total = a10.avgTotal;
      const h15total = h15.avgTotal;
      const a15total = a15.avgTotal;

      const allWindowsLow = h5total < 215 && a5total < 215 &&
                             h10total < 216 && a10total < 216 &&
                             h15total < 217 && a15total < 217;
      const allWindowsHigh = h5total > 225 && a5total > 225 &&
                              h10total > 224 && a10total > 224 &&
                              h15total > 223 && a15total > 223;

      if (allWindowsLow) {
        addResult('all_windows_under', game.date, actualTotal < predTotal, 'under');
      }
      if (allWindowsHigh) {
        addResult('all_windows_over', game.date, actualTotal > predTotal, 'over');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 17: "Last 3 Games Streak" - All last 3 games
    // were wins/losses for a team
    // ═════════════════════════════════════════════════════════
    {
      const hLast3 = h5.last3;
      const aLast3 = a5.last3;

      const home3Wins = hLast3.every(m => m > 0);
      const home3Losses = hLast3.every(m => m < 0);
      const away3Wins = aLast3.every(m => m > 0);
      const away3Losses = aLast3.every(m => m < 0);

      // Fade teams on 3-game win streaks when net rating is mediocre
      if (home3Wins && h10.netRating < 2 && away3Losses && a10.netRating > -2) {
        addResult('streak_fade_home', game.date, actualHomeMargin < predHomeMargin, 'away_cover');
      }
      if (away3Wins && a10.netRating < 2 && home3Losses && h10.netRating > -2) {
        addResult('streak_fade_away', game.date, actualHomeMargin > predHomeMargin, 'home_cover');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 18: "Defensive Recovery" - Team's defense is getting
    // significantly better over recent games
    // ═════════════════════════════════════════════════════════
    {
      if (h5.defRating && h15.defRating && a5.defRating && a15.defRating) {
        const homeDefImprove = h15.defRating - h5.defRating;
        const awayDefImprove = a15.defRating - a5.defRating;

        // Both defenses improving → UNDER
        if (homeDefImprove > 3 && awayDefImprove > 3) {
          addResult('both_def_improving_under', game.date, actualTotal < predTotal, 'under');
        }
        // Both defenses getting worse → OVER
        if (homeDefImprove < -3 && awayDefImprove < -3) {
          addResult('both_def_declining_over', game.date, actualTotal > predTotal, 'over');
        }
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 19: "Combined Home Edge" - Multiple home advantages stack
    // ═════════════════════════════════════════════════════════
    {
      let homeEdgeSignals = 0;
      if (h10.homeWinPct !== null && h10.homeWinPct >= 0.7) homeEdgeSignals++;
      if (a10.awayWinPct !== null && a10.awayWinPct <= 0.3) homeEdgeSignals++;
      if (h10.netRating > a10.netRating + 3) homeEdgeSignals++;
      if (h5.netRating > h10.netRating + 2) homeEdgeSignals++; // home improving
      if (a5.netRating < a10.netRating - 2) homeEdgeSignals++; // away declining

      if (homeEdgeSignals >= 3) {
        addResult('stacked_home_edge_3', game.date, actualHomeMargin > 0, 'home_ml');
      }
      if (homeEdgeSignals >= 4) {
        addResult('stacked_home_edge_4', game.date, actualHomeMargin > 0, 'home_ml');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 20: "Upset Alert" - Away team much better but
    // home team has strong home court advantage
    // ═════════════════════════════════════════════════════════
    {
      if (h10.homeAvgMargin !== null && a10.awayAvgMargin !== null) {
        // Away team better overall but home team dominant at home
        const awayBetter = a10.netRating > h10.netRating + 3;
        const homeStrongAtHome = h10.homeAvgMargin > 5;
        const awayWeakOnRoad = a10.awayAvgMargin < 0;

        if (awayBetter && homeStrongAtHome && awayWeakOnRoad) {
          addResult('upset_alert_home', game.date, actualHomeMargin > 0, 'home_ml');
        }
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 21: "Score Suppression" - Both teams averaging
    // under 105 points in recent games → Strong UNDER
    // ═════════════════════════════════════════════════════════
    {
      if (h5.offRating < 105 && a5.offRating < 105) {
        addResult('score_suppression', game.date, actualTotal < predTotal, 'under');
      }
      if (h5.offRating < 105 && a5.offRating < 105 && h5.defRating < 110 && a5.defRating < 110) {
        addResult('score_suppression_def', game.date, actualTotal < predTotal, 'under');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 22: "Scoring Asymmetry" - One team's offense vs
    // other team's defense creates predictable matchup
    // ═════════════════════════════════════════════════════════
    {
      // Home offense vs Away defense
      const homeOffVsAwayDef = h10.offRating - a10.defRating;
      // Away offense vs Home defense
      const awayOffVsHomeDef = a10.offRating - h10.defRating;

      // Both matchups favor low scoring
      if (homeOffVsAwayDef < -5 && awayOffVsHomeDef < -5) {
        addResult('scoring_asymmetry_under', game.date, actualTotal < predTotal, 'under');
      }
      // Both matchups favor high scoring
      if (homeOffVsAwayDef > 5 && awayOffVsHomeDef > 5) {
        addResult('scoring_asymmetry_over', game.date, actualTotal > predTotal, 'over');
      }
    }

    // ═════════════════════════════════════════════════════════
    // STRATEGY 23: "Rest Advantage Proxy" - Back-to-back detection
    // via same-day or consecutive-day games in history
    // ═════════════════════════════════════════════════════════
    {
      const hHistory = tracker.getWindow(home, 2);
      const aHistory = tracker.getWindow(away, 2);

      if (hHistory.length >= 2 && aHistory.length >= 2) {
        const hLastDate = hHistory[hHistory.length - 1].date;
        const aLastDate = aHistory[aHistory.length - 1].date;

        // Check if team played yesterday (back-to-back)
        const gameDate = parseInt(game.date);
        const hLastNum = parseInt(hLastDate);
        const aLastNum = parseInt(aLastDate);

        const homeB2B = (gameDate - hLastNum) <= 1;
        const awayB2B = (gameDate - aLastNum) <= 1;

        // Away on B2B, home rested
        if (awayB2B && !homeB2B && (gameDate - hLastNum) >= 2) {
          addResult('rest_adv_home', game.date, actualHomeMargin > 0, 'home_ml');
        }
        // Home on B2B, away rested
        if (homeB2B && !awayB2B && (gameDate - aLastNum) >= 2) {
          addResult('rest_adv_away', game.date, actualHomeMargin < 0, 'away_ml');
        }

        // B2B team plays UNDER (tired = less offense)
        if (homeB2B || awayB2B) {
          addResult('b2b_under', game.date, actualTotal < predTotal, 'under');
        }
      }
    }
  }

  // UPDATE TRACKER (after predictions)
  tracker.update(home, game.home_score, game.away_score, game.date, true, away);
  tracker.update(away, game.away_score, game.home_score, game.date, false, home);
}

// ═══════════════════════════════════════════════════════════════════
// RESULTS ANALYSIS
// ═══════════════════════════════════════════════════════════════════

console.log('\n' + '═'.repeat(80));
console.log('STRATEGY RESULTS (sorted by accuracy, min 20 picks)');
console.log('═'.repeat(80));
console.log(`${'Strategy'.padEnd(35)} | ${'W-L'.padEnd(8)} | ${'Acc%'.padEnd(7)} | ${'PnL'.padEnd(8)} | ${'ROI%'.padEnd(7)} | Picks`);
console.log('─'.repeat(80));

const stratResults = Object.entries(results.strategies)
  .map(([name, picks]) => {
    const wins = picks.filter(p => p.hit).length;
    const total = picks.length;
    const acc = (wins / total * 100);
    const pnl = picks.reduce((s, p) => s + p.pnl, 0);
    const roi = pnl / (total * 100) * 100;
    return { name, wins, losses: total - wins, total, acc, pnl, roi, betType: picks[0].betType };
  })
  .filter(s => s.total >= 20)
  .sort((a, b) => b.acc - a.acc);

for (const s of stratResults) {
  const accStr = s.acc.toFixed(1) + '%';
  const pnlStr = (s.pnl >= 0 ? '+' : '') + '$' + s.pnl;
  const roiStr = (s.roi >= 0 ? '+' : '') + s.roi.toFixed(1) + '%';
  console.log(`${s.name.padEnd(35)} | ${(s.wins + '-' + s.losses).padEnd(8)} | ${accStr.padEnd(7)} | ${pnlStr.padEnd(8)} | ${roiStr.padEnd(7)} | ${s.total} (${s.betType})`);
}

console.log('\n' + '═'.repeat(80));
console.log('STRATEGIES ABOVE 55% ACCURACY (sorted by ROI)');
console.log('═'.repeat(80));

const profitable = stratResults.filter(s => s.acc > 55);
profitable.sort((a, b) => b.roi - a.roi);

for (const s of profitable) {
  const accStr = s.acc.toFixed(1) + '%';
  const pnlStr = (s.pnl >= 0 ? '+' : '') + '$' + s.pnl;
  const roiStr = (s.roi >= 0 ? '+' : '') + s.roi.toFixed(1) + '%';
  console.log(`${s.name.padEnd(35)} | ${(s.wins + '-' + s.losses).padEnd(8)} | ${accStr.padEnd(7)} | ${pnlStr.padEnd(8)} | ${roiStr.padEnd(7)} | ${s.total} (${s.betType})`);
}

// Show strategies with 10-19 picks that have very high accuracy
console.log('\n' + '═'.repeat(80));
console.log('HIGH ACCURACY SMALL SAMPLE (10-19 picks, >60% accuracy)');
console.log('═'.repeat(80));

const smallSample = Object.entries(results.strategies)
  .map(([name, picks]) => {
    const wins = picks.filter(p => p.hit).length;
    const total = picks.length;
    return { name, wins, total, acc: wins/total*100, pnl: picks.reduce((s,p) => s+p.pnl, 0), betType: picks[0].betType };
  })
  .filter(s => s.total >= 10 && s.total < 20 && s.acc > 60)
  .sort((a, b) => b.acc - a.acc);

for (const s of smallSample) {
  console.log(`${s.name.padEnd(35)} | ${(s.wins + '-' + (s.total-s.wins)).padEnd(8)} | ${s.acc.toFixed(1)}% | +$${s.pnl} | ${s.total} (${s.betType})`);
}
