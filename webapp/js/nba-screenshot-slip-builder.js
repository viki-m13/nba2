// =============================================================================
// NBA SCREENSHOT SLIP BUILDER — Core / Screenshot / Nuke Parlay Constructors
// =============================================================================
//
// Three output modes designed for screenshot-style NBA alt-line parlays:
//
// A. Core Builder (3-4 legs) — Highest quality edges, disciplined, sustainable
// B. Screenshot Builder (5-8 legs) — Resembles the screenshot style: low alt
//    thresholds, strong role support, SGP correlation where applicable
// C. Nuke Builder (8-12 legs) — Long parlays only when every leg clears strict
//    quality thresholds. Rare and not forced.
// =============================================================================

window.NbaScreenshotSlipBuilder = (function () {
  'use strict';

  const { americanToDecimal, decimalToAmerican } = window.NbaAltRungEvaluator;
  const { scoreCluster, buildSGPClusters } = window.NbaCorrelationEngine;

  // --- Quality thresholds per builder ---

  const CORE_THRESHOLDS = {
    minModelProb: 0.88,
    minEdge: 0.05,
    maxFragility: 0.40,
    minStability: 0.60,
    minMinutesCertainty: 0.60,
    minMarketFit: 0.5,
    targetLegs: [3, 4],
    maxLegsPerGame: 2,
    label: 'CORE',
  };

  const SCREENSHOT_THRESHOLDS = {
    minModelProb: 0.84,
    minEdge: 0.03,
    maxFragility: 0.50,
    minStability: 0.50,
    minMinutesCertainty: 0.50,
    minMarketFit: 0.4,
    targetLegs: [5, 6, 7, 8],
    maxLegsPerGame: 3,
    label: 'SCREENSHOT',
  };

  const NUKE_THRESHOLDS = {
    minModelProb: 0.86,
    minEdge: 0.04,
    maxFragility: 0.40,
    minStability: 0.60,
    minMinutesCertainty: 0.55,
    minMarketFit: 0.4,
    targetLegs: [8, 9, 10, 11, 12],
    maxLegsPerGame: 3,
    label: 'NUKE',
  };

  // --- Filter candidates by threshold ---

  function filterCandidates(featureTable, thresholds) {
    return featureTable.filter(leg =>
      leg.modelProb >= thresholds.minModelProb &&
      leg.edge >= thresholds.minEdge &&
      leg.fragility <= thresholds.maxFragility &&
      leg.stability >= thresholds.minStability &&
      (leg.minutesCertainty === undefined || leg.minutesCertainty >= thresholds.minMinutesCertainty) &&
      (leg.marketFit === undefined || leg.marketFit >= thresholds.minMarketFit) &&
      // Reject volatile bench and low-minute rotation
      leg.roleTag !== 'volatile_bench' &&
      leg.roleTag !== 'low_minute_rotation'
    );
  }

  // --- Score a candidate leg for ranking ---

  function scoreLeg(leg) {
    // Composite score: edge-weighted, stability-adjusted, role-fit-adjusted
    const edgeScore = Math.max(0, leg.edge) * 3;
    const probScore = Math.max(0, leg.modelProb - 0.8) * 5;
    const stabilityScore = (leg.stability || 0.5) * 1.5;
    const fitScore = (leg.marketFit || 0.5) * 0.5;
    const evScore = Math.max(0, leg.ev || 0) * 2;

    return edgeScore + probScore + stabilityScore + fitScore + evScore;
  }

  // --- Build a parlay from filtered candidates ---

  function buildParlay(candidates, thresholds) {
    if (candidates.length < Math.min(...thresholds.targetLegs)) return null;

    // Score and sort candidates
    const scored = candidates.map(c => ({ ...c, _score: scoreLeg(c) }));
    scored.sort((a, b) => b._score - a._score);

    // Greedy selection with diversification
    const selected = [];
    const gameCount = {};
    const usedPlayerMarkets = new Set();

    for (const leg of scored) {
      const targetMax = Math.max(...thresholds.targetLegs);
      if (selected.length >= targetMax) break;

      // One leg per player+market
      const pmKey = `${leg.player}_${leg.marketType}`;
      if (usedPlayerMarkets.has(pmKey)) continue;

      // Game diversification
      const gk = leg.gameId || leg.gameKey || 'unknown';
      const gc = gameCount[gk] || 0;
      if (gc >= thresholds.maxLegsPerGame) continue;

      selected.push(leg);
      usedPlayerMarkets.add(pmKey);
      gameCount[gk] = gc + 1;
    }

    // Check if we hit minimum target
    const minTarget = Math.min(...thresholds.targetLegs);
    if (selected.length < minTarget) return null;

    // Calculate parlay odds
    const decimalOdds = selected.reduce((d, l) => d * americanToDecimal(l.bookOdds), 1);
    const americanOdds = decimalToAmerican(decimalOdds);
    const combinedProb = selected.reduce((p, l) => p * l.modelProb, 1);
    const ev = combinedProb * decimalOdds - 1;

    // Score correlation for SGP legs
    const gameGroups = {};
    for (const leg of selected) {
      const gk = leg.gameId || leg.gameKey;
      if (!gameGroups[gk]) gameGroups[gk] = [];
      gameGroups[gk].push(leg);
    }

    const sgpClusters = [];
    for (const [gk, legs] of Object.entries(gameGroups)) {
      if (legs.length >= 2) {
        sgpClusters.push({
          gameId: gk,
          legs: legs.length,
          correlation: scoreCluster(legs),
        });
      }
    }

    return {
      builder: thresholds.label,
      legs: selected.map(l => ({
        player: l.player,
        team: l.team,
        opponent: l.opponent,
        gameId: l.gameId,
        gameKey: l.gameKey,
        gameDisplay: l.gameDisplay || `${l.team} vs ${l.opponent}`,
        marketType: l.marketType,
        marketLabel: l.marketLabel,
        line: l.line,
        bookOdds: l.bookOdds,
        odds: l.bookOdds, // compat alias
        statType: marketToStatType(l.marketType),
        statLabel: l.marketLabel,
        modelProb: l.modelProb,
        edge: l.edge,
        fragility: l.fragility,
        stability: l.stability,
        roleTag: l.roleTag,
        avg: l.avg,
        floor: l.floor,
        hitRate: l.l10Hit,
      })),
      numLegs: selected.length,
      odds: americanOdds,
      decimalOdds: Math.round(decimalOdds * 100) / 100,
      combinedHitRate: Math.round(combinedProb * 10000) / 10000,
      ev: Math.round(ev * 1000) / 1000,
      sgpClusters,
      hasSGP: sgpClusters.length > 0,
    };
  }

  // --- Public builders ---

  function buildCore(featureTable) {
    const candidates = filterCandidates(featureTable, CORE_THRESHOLDS);
    const parlays = [];

    for (const size of CORE_THRESHOLDS.targetLegs) {
      const t = { ...CORE_THRESHOLDS, targetLegs: [size, size] };
      const p = buildParlay(candidates, t);
      if (p) parlays.push(p);
    }

    return parlays;
  }

  function buildScreenshot(featureTable) {
    const candidates = filterCandidates(featureTable, SCREENSHOT_THRESHOLDS);
    const parlays = [];

    // Try to build SGP-aware parlays
    const sgpClusters = buildSGPClusters(candidates);
    const sgpLegs = [];
    const standalonLegs = [];

    for (const cluster of sgpClusters) {
      if (cluster.type === 'sgp' && cluster.correlation.avgCorr >= 0) {
        sgpLegs.push(...cluster.legs);
      } else {
        standalonLegs.push(...cluster.legs);
      }
    }

    // Combine SGP legs with standalone legs, prioritizing SGP clusters
    const combinedPool = [...sgpLegs, ...standalonLegs];

    for (const size of [6, 7, 8, 5]) {
      const t = { ...SCREENSHOT_THRESHOLDS, targetLegs: [size, size] };
      const p = buildParlay(combinedPool.length > 0 ? combinedPool : candidates, t);
      if (p) {
        parlays.push(p);
        break; // One screenshot parlay per run
      }
    }

    // Also try a pure SGP if we have enough legs from one game
    for (const cluster of sgpClusters) {
      if (cluster.type === 'sgp' && cluster.legs.length >= 3) {
        const sgpParlay = buildParlay(cluster.legs, {
          ...SCREENSHOT_THRESHOLDS,
          targetLegs: [3, 4],
          maxLegsPerGame: 4,
          label: 'SGP',
        });
        if (sgpParlay) parlays.push(sgpParlay);
      }
    }

    return parlays;
  }

  function buildNuke(featureTable) {
    const candidates = filterCandidates(featureTable, NUKE_THRESHOLDS);

    // Nuke requires strict quality — don't force it
    if (candidates.length < 8) return [];

    const parlays = [];
    for (const size of [10, 9, 8, 11, 12]) {
      const t = { ...NUKE_THRESHOLDS, targetLegs: [size, size] };
      const p = buildParlay(candidates, t);
      if (p) {
        parlays.push(p);
        break; // One nuke parlay max
      }
    }

    return parlays;
  }

  // Build all three types
  function buildAll(featureTable) {
    return {
      core: buildCore(featureTable),
      screenshot: buildScreenshot(featureTable),
      nuke: buildNuke(featureTable),
    };
  }

  // --- Market label helpers ---

  const MARKET_LABELS = {
    'player_points_alternate': 'ALT PTS',
    'player_assists_alternate': 'ALT AST',
    'player_rebounds_alternate': 'ALT REB',
    'player_points_assists_alternate': 'ALT PTS+AST',
    'player_points_rebounds_alternate': 'ALT PTS+REB',
    'player_points_rebounds_assists_alternate': 'ALT PTS+REB+AST',
  };

  function marketLabel(marketType) {
    return MARKET_LABELS[marketType] || marketType;
  }

  function marketToStatType(marketType) {
    if (marketType === 'player_points_alternate') return 'points';
    if (marketType === 'player_assists_alternate') return 'assists';
    if (marketType === 'player_rebounds_alternate') return 'rebounds';
    if (marketType === 'player_points_assists_alternate') return 'points_assists';
    if (marketType === 'player_points_rebounds_alternate') return 'points_rebounds';
    if (marketType === 'player_points_rebounds_assists_alternate') return 'points_rebounds_assists';
    return 'points';
  }

  return {
    buildCore,
    buildScreenshot,
    buildNuke,
    buildAll,
    filterCandidates,
    scoreLeg,
    marketLabel,
    marketToStatType,
    CORE_THRESHOLDS,
    SCREENSHOT_THRESHOLDS,
    NUKE_THRESHOLDS,
  };
})();
