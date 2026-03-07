// =============================================================================
// NBA ALT-RUNG EVALUATOR — Market-Specific Alternate Line Evaluation
// =============================================================================
//
// Evaluates every rung on every alternate ladder for a player, using
// market-specific logic rather than one generic formula.
//
// For each player-market-rung combination, produces:
//   - modelProb: estimated true hit probability
//   - fairOdds: what the odds should be
//   - edge: modelProb - impliedProb
//   - clearance: how far above the line the player's expected output sits
//   - fragility: how unstable/risky the leg is (0=solid, 1=fragile)
//   - stability: inverse of fragility
// =============================================================================

window.NbaAltRungEvaluator = (function () {
  'use strict';

  // --- Odds math ---
  function americanToDecimal(odds) {
    return odds > 0 ? 1 + odds / 100 : 1 + 100 / Math.abs(odds);
  }
  function decimalToAmerican(dec) {
    if (dec >= 2.0) return Math.round((dec - 1) * 100);
    return Math.round(-100 / (dec - 1));
  }
  function impliedProb(odds) { return 1 / americanToDecimal(odds); }

  // --- Stat key mapping ---
  function getStatKeys(marketType) {
    switch (marketType) {
      case 'player_points_alternate': return ['pts'];
      case 'player_assists_alternate': return ['ast'];
      case 'player_rebounds_alternate': return ['reb'];
      case 'player_points_assists_alternate': return ['pts', 'ast'];
      case 'player_points_rebounds_alternate': return ['pts', 'reb'];
      case 'player_points_rebounds_assists_alternate': return ['pts', 'reb', 'ast'];
      default: return ['pts'];
    }
  }

  function getStatValue(game, statKeys) {
    return statKeys.reduce((sum, key) => sum + (game[key] || 0), 0);
  }

  // --- Core evaluation: compute hit rates and model probability ---

  function computeHitRates(gameLog, statKeys, line) {
    if (!gameLog || gameLog.length < 5) return null;

    const values = gameLog.map(g => getStatValue(g, statKeys));
    const n = values.length;

    // L5, L10, L20 hit rates
    const l5 = values.slice(-5);
    const l10 = values.slice(-10);
    const l20 = values.slice(-20);

    const l5Hit = l5.filter(v => v > line).length / l5.length;
    const l10Hit = l10.filter(v => v > line).length / l10.length;
    const l20Hit = l20.filter(v => v > line).length / Math.min(20, n);

    // Full season hit rate
    const fullHit = values.filter(v => v > line).length / n;

    // Average and floor
    const avg = values.reduce((s, v) => s + v, 0) / n;
    const recentAvg = l10.reduce((s, v) => s + v, 0) / l10.length;
    const l10Floor = Math.min(...l10);
    const l5Floor = Math.min(...l5);

    // Standard deviation
    const std = Math.sqrt(values.reduce((s, v) => s + Math.pow(v - avg, 2), 0) / n);
    const cv = avg > 0 ? std / avg : 1;

    // Clearance: how far above the line is the player's expected output
    const clearance = avg > 0 ? (recentAvg - line) / avg : 0;
    const floorClearance = l10Floor - line;

    return {
      l5Hit, l10Hit, l20Hit, fullHit,
      avg, recentAvg, l10Floor, l5Floor,
      std, cv, clearance, floorClearance,
      games: n,
    };
  }

  // --- Market-specific model probability estimators ---

  function modelProbPoints(stats, roleInfo, envContext) {
    if (!stats) return 0;

    // Base: weighted hit rate blend, shrunk toward 0.85 to avoid overconfidence
    const rawProb = stats.l10Hit * 0.40 + stats.l20Hit * 0.35 + stats.fullHit * 0.25;
    let prob = rawProb * 0.9 + 0.85 * 0.1; // 10% shrinkage toward base rate

    // Floor bonus: if L10 floor is above the line, safer — but capped conservatively
    if (stats.floorClearance > 0) {
      prob += Math.min(0.04, stats.floorClearance * 0.006);
    } else if (stats.floorClearance < -2) {
      prob -= 0.06;
    }

    // Low variance bonus
    if (stats.cv < 0.2) prob += 0.02;
    else if (stats.cv > 0.35) prob -= 0.05;

    // Role adjustment
    if (roleInfo) {
      if (roleInfo.role === 'high_usage_scorer') prob += 0.02;
      if (roleInfo.role === 'rebound_heavy_big') prob -= 0.03; // Bigs are weaker on points
      if (roleInfo.role === 'volatile_bench') prob -= 0.07;
      if (roleInfo.role === 'low_minute_rotation') prob -= 0.09;
    }

    // Game environment
    if (envContext) {
      if (envContext.teamImpliedTotal >= 115) prob += 0.015;
      else if (envContext.teamImpliedTotal <= 105) prob -= 0.02;
      if (envContext.blowoutRisk > 0.6) prob -= 0.05;
    }

    return Math.max(0, Math.min(0.97, prob)); // Cap at 97% — never assign near-certainty
  }

  function modelProbAssists(stats, roleInfo, envContext) {
    if (!stats) return 0;

    // Assists are more role-dependent, weight recent performance higher
    const rawProb = stats.l10Hit * 0.45 + stats.l20Hit * 0.30 + stats.fullHit * 0.25;
    let prob = rawProb * 0.88 + 0.82 * 0.12; // 12% shrinkage — assists are more volatile

    // Floor bonus — conservative
    if (stats.floorClearance > 0) {
      prob += Math.min(0.04, stats.floorClearance * 0.01);
    } else if (stats.floorClearance < -1) {
      prob -= 0.07;
    }

    // Assists have higher variance, so CV penalty is stronger
    if (stats.cv < 0.25) prob += 0.02;
    else if (stats.cv > 0.4) prob -= 0.07;

    // Role is critical for assists
    if (roleInfo) {
      if (roleInfo.role === 'primary_creator') prob += 0.03;
      else if (roleInfo.role === 'secondary_creator') prob += 0.015;
      else if (roleInfo.role === 'high_usage_scorer') prob -= 0.03;
      else if (roleInfo.role === 'rebound_heavy_big') prob -= 0.05;
      else if (roleInfo.role === 'volatile_bench') prob -= 0.09;
    }

    // Game environment
    if (envContext) {
      if (envContext.total >= 230) prob += 0.015;
      if (Math.abs(envContext.spread || 0) <= 5) prob += 0.01;
      if (envContext.blowoutRisk > 0.6) prob -= 0.06;
    }

    return Math.max(0, Math.min(0.96, prob)); // Cap at 96%
  }

  function modelProbRebounds(stats, roleInfo, envContext) {
    if (!stats) return 0;

    // Rebounds weight consistency higher — but shrink more aggressively
    // Backtest showed rebound-heavy bigs have negative ROI
    const rawProb = stats.l10Hit * 0.35 + stats.l20Hit * 0.40 + stats.fullHit * 0.25;
    let prob = rawProb * 0.85 + 0.80 * 0.15; // 15% shrinkage — rebounds overestimated

    // Floor bonus — very conservative for rebounds
    if (stats.floorClearance > 0) {
      prob += Math.min(0.03, stats.floorClearance * 0.008);
    } else if (stats.floorClearance < -2) {
      prob -= 0.06;
    }

    // CV penalty
    if (stats.cv < 0.25) prob += 0.02;
    else if (stats.cv > 0.4) prob -= 0.05;

    // Role — bigs get a small bonus but not as much as before
    if (roleInfo) {
      if (roleInfo.role === 'rebound_heavy_big') prob += 0.01; // Reduced from 0.04
      else if (roleInfo.role === 'primary_creator') prob -= 0.03;
      else if (roleInfo.role === 'volatile_bench') prob -= 0.07;
    }

    // Environment
    if (envContext) {
      if (envContext.total >= 230) prob += 0.01;
      if (envContext.blowoutRisk > 0.6) prob -= 0.04;
    }

    return Math.max(0, Math.min(0.95, prob)); // Cap at 95% — rebounds are less predictable
  }

  function modelProbCombo(stats, roleInfo, envContext, marketType) {
    if (!stats) return 0;

    // Combo stats use direct historical hit rate + component adjustments
    const rawProb = stats.l10Hit * 0.40 + stats.l20Hit * 0.35 + stats.fullHit * 0.25;
    let prob = rawProb * 0.87 + 0.82 * 0.13; // 13% shrinkage — combos are harder

    // Floor bonus — conservative for combos
    if (stats.floorClearance > 0) {
      prob += Math.min(0.03, stats.floorClearance * 0.005);
    } else if (stats.floorClearance < -3) {
      prob -= 0.06;
    }

    // Combo stats have inherently higher variance
    if (stats.cv < 0.2) prob += 0.02;
    else if (stats.cv > 0.3) prob -= 0.05;

    // Role adjustments for combos
    if (roleInfo) {
      if (roleInfo.role === 'primary_creator') prob += 0.02;
      if (roleInfo.role === 'high_usage_scorer' && marketType.includes('points')) prob += 0.015;
      if (roleInfo.role === 'rebound_heavy_big' && marketType.includes('rebounds')) prob += 0.005;
      if (roleInfo.role === 'volatile_bench') prob -= 0.08;
    }

    // Environment
    if (envContext) {
      if (envContext.teamImpliedTotal >= 115) prob += 0.015;
      if (envContext.blowoutRisk > 0.6) prob -= 0.05;
    }

    return Math.max(0, Math.min(0.95, prob)); // Cap at 95%
  }

  // --- Fragility score (0 = solid, 1 = fragile) ---

  function computeFragility(stats, roleInfo, minutesCertainty) {
    if (!stats) return 1;

    let fragility = 0.3; // base

    // High CV = more fragile
    fragility += Math.min(0.25, Math.max(0, (stats.cv - 0.15) * 0.8));

    // Low floor clearance = more fragile
    if (stats.floorClearance < 0) fragility += Math.min(0.2, Math.abs(stats.floorClearance) * 0.03);
    else if (stats.floorClearance > 3) fragility -= 0.1;

    // Inconsistent hit rate across windows = fragile
    const hitRateSpread = Math.abs(stats.l5Hit - stats.l20Hit);
    if (hitRateSpread > 0.15) fragility += 0.1;

    // Role volatility
    if (roleInfo) {
      if (roleInfo.role === 'volatile_bench') fragility += 0.2;
      if (roleInfo.role === 'low_minute_rotation') fragility += 0.25;
      if (roleInfo.role === 'injury_elevated') fragility += 0.1;
    }

    // Minutes uncertainty
    if (minutesCertainty !== undefined) {
      fragility += (1 - minutesCertainty) * 0.2;
    }

    // Few games = uncertain
    if (stats.games < 10) fragility += 0.1;
    if (stats.games < 7) fragility += 0.1;

    return Math.max(0, Math.min(1, fragility));
  }

  // --- Evaluate a single rung for a player-market combo ---

  function evaluateRung(gameLog, marketType, line, bookOdds, roleInfo, envContext, minutesCertainty) {
    const statKeys = getStatKeys(marketType);
    const stats = computeHitRates(gameLog, statKeys, line);
    if (!stats || stats.games < 5) return null;

    // Market-specific model probability
    let modelProb;
    if (marketType === 'player_points_alternate') {
      modelProb = modelProbPoints(stats, roleInfo, envContext);
    } else if (marketType === 'player_assists_alternate') {
      modelProb = modelProbAssists(stats, roleInfo, envContext);
    } else if (marketType === 'player_rebounds_alternate') {
      modelProb = modelProbRebounds(stats, roleInfo, envContext);
    } else {
      modelProb = modelProbCombo(stats, roleInfo, envContext, marketType);
    }

    const implied = impliedProb(bookOdds);
    const edge = modelProb - implied;
    const fairDecimal = modelProb > 0 ? 1 / modelProb : 100;
    const fairOdds = decimalToAmerican(fairDecimal);
    const fragility = computeFragility(stats, roleInfo, minutesCertainty);

    return {
      marketType,
      line,
      bookOdds,
      modelProb: Math.round(modelProb * 1000) / 1000,
      impliedProb: Math.round(implied * 1000) / 1000,
      fairOdds,
      edge: Math.round(edge * 1000) / 1000,
      clearance: Math.round(stats.clearance * 1000) / 1000,
      floorClearance: stats.floorClearance,
      fragility: Math.round(fragility * 1000) / 1000,
      stability: Math.round((1 - fragility) * 1000) / 1000,
      l5Hit: stats.l5Hit,
      l10Hit: stats.l10Hit,
      l20Hit: stats.l20Hit,
      avg: Math.round(stats.recentAvg * 10) / 10,
      floor: stats.l10Floor,
      cv: Math.round(stats.cv * 1000) / 1000,
      games: stats.games,
      ev: Math.round((modelProb * americanToDecimal(bookOdds) - 1) * 1000) / 1000,
    };
  }

  // --- Evaluate full ladder for a player + market ---
  // Returns array of evaluated rungs, sorted by line descending (highest safe rung first)

  function evaluateLadder(gameLog, marketType, availableLines, roleInfo, envContext, minutesCertainty) {
    const results = [];

    for (const [threshold, data] of Object.entries(availableLines)) {
      const line = parseFloat(threshold);
      const odds = data.overOdds;
      if (!odds) continue;

      const rung = evaluateRung(gameLog, marketType, line, odds, roleInfo, envContext, minutesCertainty);
      if (rung) results.push(rung);
    }

    // Sort by line descending — we want to find the HIGHEST safe rung
    results.sort((a, b) => b.line - a.line);
    return results;
  }

  // --- Select the best rung from a ladder ---
  // Strategy: find the highest rung that meets quality thresholds

  function selectBestRung(ladder, opts = {}) {
    const {
      minModelProb = 0.82,
      minEdge = 0.03,
      maxFragility = 0.55,
      minFloorClearance = -1,
      preferHigherRung = true,
    } = opts;

    // Filter to qualifying rungs
    const qualifying = ladder.filter(r =>
      r.modelProb >= minModelProb &&
      r.edge >= minEdge &&
      r.fragility <= maxFragility &&
      r.floorClearance >= minFloorClearance
    );

    if (qualifying.length === 0) return null;

    if (preferHigherRung) {
      // Return highest line that qualifies (already sorted desc)
      return qualifying[0];
    }

    // Alternative: return the rung with best edge
    return qualifying.reduce((best, r) => r.edge > best.edge ? r : best, qualifying[0]);
  }

  return {
    evaluateRung,
    evaluateLadder,
    selectBestRung,
    computeHitRates,
    getStatKeys,
    getStatValue,
    americanToDecimal,
    decimalToAmerican,
    impliedProb,
  };
})();
