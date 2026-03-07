// =============================================================================
// NBA ROLE ENGINE — Player Classification for Alt-Line Prop Betting
// =============================================================================
//
// Classifies players into role types that determine which alt-line markets
// are appropriate and how to weight their evaluation.
//
// Role tags:
//   primary_creator    — Lead ballhandler/playmaker (assists ladder)
//   secondary_creator  — Secondary playmaker (moderate assists ladder)
//   high_usage_scorer  — High-volume scorer (points ladder)
//   support_scorer     — Consistent mid-range scorer (points ladder)
//   rebound_heavy_big  — Center/PF with rebounding role (rebounds ladder)
//   volatile_bench     — Bench player with unstable minutes
//   low_minute_rotation — Under 20 min, high variance
//   injury_elevated    — Player with expanded role due to teammate injury
// =============================================================================

window.NbaRoleEngine = (function () {
  'use strict';

  // Role type definitions with market suitability
  const ROLE_MARKET_FIT = {
    primary_creator:    { points: 0.8, assists: 1.0, rebounds: 0.4, combo: 0.9 },
    secondary_creator:  { points: 0.7, assists: 0.8, rebounds: 0.5, combo: 0.8 },
    high_usage_scorer:  { points: 1.0, assists: 0.4, rebounds: 0.5, combo: 0.8 },
    support_scorer:     { points: 0.8, assists: 0.3, rebounds: 0.4, combo: 0.6 },
    rebound_heavy_big:  { points: 0.6, assists: 0.2, rebounds: 1.0, combo: 0.7 },
    volatile_bench:     { points: 0.2, assists: 0.1, rebounds: 0.2, combo: 0.1 },
    low_minute_rotation:{ points: 0.3, assists: 0.2, rebounds: 0.3, combo: 0.2 },
    injury_elevated:    { points: 0.7, assists: 0.6, rebounds: 0.6, combo: 0.7 },
  };

  // Classify a player based on recent game logs
  function classifyPlayer(gameLog) {
    if (!gameLog || gameLog.length < 5) return { role: 'low_minute_rotation', confidence: 0.3 };

    const recent = gameLog.slice(-10);
    const n = recent.length;

    const avgMin = recent.reduce((s, g) => s + g.min, 0) / n;
    const avgPts = recent.reduce((s, g) => s + g.pts, 0) / n;
    const avgAst = recent.reduce((s, g) => s + g.ast, 0) / n;
    const avgReb = recent.reduce((s, g) => s + g.reb, 0) / n;

    // Minutes stability (coefficient of variation)
    const minStd = Math.sqrt(recent.reduce((s, g) => s + Math.pow(g.min - avgMin, 2), 0) / n);
    const minCV = avgMin > 0 ? minStd / avgMin : 1;

    // Starter check: played 28+ min in 70%+ of recent games
    const starterGames = recent.filter(g => g.min >= 28).length;
    const isStarter = starterGames / n >= 0.7;

    // Low minute filter
    if (avgMin < 18) {
      return {
        role: avgMin < 12 ? 'low_minute_rotation' : 'volatile_bench',
        confidence: 0.5,
        avgMin, avgPts, avgAst, avgReb, minCV, isStarter,
      };
    }

    // Volatile bench: non-starter with high minute variance
    if (!isStarter && minCV > 0.25) {
      return {
        role: 'volatile_bench',
        confidence: 0.6,
        avgMin, avgPts, avgAst, avgReb, minCV, isStarter,
      };
    }

    // Primary creator: high assists, lead ballhandler
    if (avgAst >= 6.0) {
      return {
        role: 'primary_creator',
        confidence: Math.min(0.95, 0.7 + (avgAst - 6) * 0.05),
        avgMin, avgPts, avgAst, avgReb, minCV, isStarter,
      };
    }

    // Secondary creator: moderate assists
    if (avgAst >= 4.0) {
      return {
        role: 'secondary_creator',
        confidence: Math.min(0.9, 0.65 + (avgAst - 4) * 0.05),
        avgMin, avgPts, avgAst, avgReb, minCV, isStarter,
      };
    }

    // Rebound-heavy big: high rebounds, moderate/low assists
    if (avgReb >= 8.0 && avgAst < 4.0) {
      return {
        role: 'rebound_heavy_big',
        confidence: Math.min(0.95, 0.7 + (avgReb - 8) * 0.04),
        avgMin, avgPts, avgAst, avgReb, minCV, isStarter,
      };
    }

    // High-usage scorer: high points
    if (avgPts >= 20.0) {
      return {
        role: 'high_usage_scorer',
        confidence: Math.min(0.95, 0.7 + (avgPts - 20) * 0.02),
        avgMin, avgPts, avgAst, avgReb, minCV, isStarter,
      };
    }

    // Support scorer: moderate points, stable minutes
    if (avgPts >= 10.0 && isStarter) {
      return {
        role: 'support_scorer',
        confidence: Math.min(0.85, 0.6 + (avgPts - 10) * 0.02),
        avgMin, avgPts, avgAst, avgReb, minCV, isStarter,
      };
    }

    // Default: low minute rotation if nothing else fits
    return {
      role: isStarter ? 'support_scorer' : 'volatile_bench',
      confidence: 0.5,
      avgMin, avgPts, avgAst, avgReb, minCV, isStarter,
    };
  }

  // Get market suitability score for a role + market type
  function getMarketFit(role, marketType) {
    const fits = ROLE_MARKET_FIT[role];
    if (!fits) return 0.3;

    if (marketType === 'player_points_alternate') return fits.points;
    if (marketType === 'player_assists_alternate') return fits.assists;
    if (marketType === 'player_rebounds_alternate') return fits.rebounds;
    // Combo markets
    return fits.combo;
  }

  // Calculate minutes certainty score (0-1)
  function getMinutesCertainty(gameLog) {
    if (!gameLog || gameLog.length < 5) return 0.2;

    const recent = gameLog.slice(-10);
    const n = recent.length;
    const avgMin = recent.reduce((s, g) => s + g.min, 0) / n;
    const minStd = Math.sqrt(recent.reduce((s, g) => s + Math.pow(g.min - avgMin, 2), 0) / n);
    const minCV = avgMin > 0 ? minStd / avgMin : 1;

    // Recent trend: are minutes trending up or down?
    const last3Avg = recent.slice(-3).reduce((s, g) => s + g.min, 0) / Math.min(3, recent.length);
    const trendRatio = avgMin > 0 ? last3Avg / avgMin : 1;

    // Floor check: minimum recent minutes
    const minFloor = Math.min(...recent.map(g => g.min));

    let certainty = 0.5;

    // Low CV = stable minutes
    if (minCV < 0.1) certainty += 0.3;
    else if (minCV < 0.15) certainty += 0.2;
    else if (minCV < 0.2) certainty += 0.1;
    else if (minCV > 0.3) certainty -= 0.2;

    // High average minutes
    if (avgMin >= 32) certainty += 0.15;
    else if (avgMin >= 28) certainty += 0.1;
    else if (avgMin < 20) certainty -= 0.15;

    // Stable trend
    if (trendRatio >= 0.95 && trendRatio <= 1.05) certainty += 0.05;
    else if (trendRatio < 0.85) certainty -= 0.15; // Minutes dropping

    // Good floor
    if (minFloor >= 24) certainty += 0.1;
    else if (minFloor < 15) certainty -= 0.15;

    return Math.max(0, Math.min(1, certainty));
  }

  // Detect if player has injury-elevated role
  // Simple heuristic: recent minutes/usage spike compared to longer history
  function detectInjuryElevation(gameLog) {
    if (!gameLog || gameLog.length < 15) return false;

    const recent5 = gameLog.slice(-5);
    const prior = gameLog.slice(-20, -5);
    if (prior.length < 5) return false;

    const recentAvgMin = recent5.reduce((s, g) => s + g.min, 0) / recent5.length;
    const priorAvgMin = prior.reduce((s, g) => s + g.min, 0) / prior.length;

    // 15%+ minutes increase suggests elevated role
    return recentAvgMin > priorAvgMin * 1.15 && recentAvgMin >= 25;
  }

  return {
    classifyPlayer,
    getMarketFit,
    getMinutesCertainty,
    detectInjuryElevation,
    ROLE_MARKET_FIT,
  };
})();
