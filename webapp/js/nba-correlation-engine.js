// =============================================================================
// NBA CORRELATION ENGINE — Same-Game Parlay Correlation Logic
// =============================================================================
//
// Scores same-game relationships between candidate legs to build
// intentional SGP clusters, not random stacking.
//
// Positive correlations:
//   - Lead guard assists + teammate points
//   - Star assists + secondary scorer points
//   - Multiple low alt point overs in high-pace game
//   - High-usage scorer points + secondary creator assists
//
// Negative / avoid:
//   - Two players competing for same stat share
//   - Correlated only through unsustainably hot shooting
//   - Legs depending on exact same event chain
// =============================================================================

window.NbaCorrelationEngine = (function () {
  'use strict';

  // Correlation types
  const CORR_POSITIVE = 'positive';
  const CORR_NEUTRAL = 'neutral';
  const CORR_NEGATIVE = 'negative';

  // Score a pair of legs from the same game
  function scorePairCorrelation(legA, legB) {
    // Different teams in same game — mostly independent
    if (legA.team !== legB.team) {
      return {
        type: CORR_NEUTRAL,
        score: 0,
        reason: 'different_teams',
      };
    }

    // Same team correlations
    const roleA = legA.roleTag || '';
    const roleB = legB.roleTag || '';
    const mktA = legA.marketType || '';
    const mktB = legB.marketType || '';

    // Same player, different markets — mild positive (if he plays well, all stats benefit)
    if (legA.player === legB.player) {
      return {
        type: CORR_POSITIVE,
        score: 0.3,
        reason: 'same_player_multi_stat',
      };
    }

    // Creator assists + teammate points — strong positive
    if (isCreatorRole(roleA) && isAssistsMarket(mktA) && isPointsMarket(mktB)) {
      return {
        type: CORR_POSITIVE,
        score: 0.6,
        reason: 'creator_assists_teammate_points',
      };
    }
    if (isCreatorRole(roleB) && isAssistsMarket(mktB) && isPointsMarket(mktA)) {
      return {
        type: CORR_POSITIVE,
        score: 0.6,
        reason: 'creator_assists_teammate_points',
      };
    }

    // Two scorers on same team, same market — mild negative (competing for shots)
    if (isPointsMarket(mktA) && isPointsMarket(mktB)) {
      // In high-pace games, both can eat — less negative
      if (legA.envContext && legA.envContext.total >= 230) {
        return {
          type: CORR_NEUTRAL,
          score: 0.1,
          reason: 'same_team_points_high_pace',
        };
      }
      return {
        type: CORR_NEGATIVE,
        score: -0.15,
        reason: 'same_team_competing_points',
      };
    }

    // Two assisters on same team — negative (competing for assists)
    if (isAssistsMarket(mktA) && isAssistsMarket(mktB)) {
      return {
        type: CORR_NEGATIVE,
        score: -0.2,
        reason: 'same_team_competing_assists',
      };
    }

    // Points + assists on same team (different players, not creator-to-scorer)
    if ((isPointsMarket(mktA) && isAssistsMarket(mktB)) ||
        (isAssistsMarket(mktA) && isPointsMarket(mktB))) {
      return {
        type: CORR_POSITIVE,
        score: 0.3,
        reason: 'same_team_points_assists_synergy',
      };
    }

    // Rebounds + points on same team — mild positive
    if ((isReboundsMarket(mktA) && isPointsMarket(mktB)) ||
        (isPointsMarket(mktA) && isReboundsMarket(mktB))) {
      return {
        type: CORR_NEUTRAL,
        score: 0.05,
        reason: 'same_team_rebounds_points_neutral',
      };
    }

    // Combo markets with component overlap — mild positive
    if (isComboMarket(mktA) || isComboMarket(mktB)) {
      return {
        type: CORR_NEUTRAL,
        score: 0.1,
        reason: 'combo_market_overlap',
      };
    }

    return {
      type: CORR_NEUTRAL,
      score: 0,
      reason: 'default_neutral',
    };
  }

  // Score a full SGP cluster
  function scoreCluster(legs) {
    if (legs.length < 2) return { totalCorr: 0, avgCorr: 0, pairs: [] };

    const pairs = [];
    let totalCorr = 0;

    for (let i = 0; i < legs.length; i++) {
      for (let j = i + 1; j < legs.length; j++) {
        const corr = scorePairCorrelation(legs[i], legs[j]);
        pairs.push({ legA: i, legB: j, ...corr });
        totalCorr += corr.score;
      }
    }

    const numPairs = pairs.length;
    return {
      totalCorr,
      avgCorr: numPairs > 0 ? totalCorr / numPairs : 0,
      pairs,
      hasNegative: pairs.some(p => p.type === CORR_NEGATIVE),
      positiveCount: pairs.filter(p => p.type === CORR_POSITIVE).length,
    };
  }

  // Build same-game clusters from a pool of candidate legs
  function buildSGPClusters(legs, maxLegsPerCluster = 4) {
    // Group legs by gameId
    const byGame = {};
    for (const leg of legs) {
      const gid = leg.gameId || leg.gameKey || 'unknown';
      if (!byGame[gid]) byGame[gid] = [];
      byGame[gid].push(leg);
    }

    const clusters = [];

    for (const [gameId, gameLeg] of Object.entries(byGame)) {
      if (gameLeg.length < 2) {
        // Single leg — not an SGP, just a standalone
        clusters.push({
          gameId,
          type: 'single',
          legs: gameLeg,
          correlation: { totalCorr: 0, avgCorr: 0, pairs: [] },
        });
        continue;
      }

      // Try to build the best cluster from this game's legs
      // Sort by quality (modelProb * stability * (1 + edge))
      const sorted = [...gameLeg].sort((a, b) => {
        const scoreA = (a.modelProb || 0) * (a.stability || 0.5) * (1 + (a.edge || 0));
        const scoreB = (b.modelProb || 0) * (b.stability || 0.5) * (1 + (b.edge || 0));
        return scoreB - scoreA;
      });

      // Greedy cluster build: add legs that improve or maintain correlation
      const cluster = [sorted[0]];
      const usedPlayers = new Set([sorted[0].player]);

      for (let i = 1; i < sorted.length && cluster.length < maxLegsPerCluster; i++) {
        const candidate = sorted[i];

        // Don't duplicate same player + same market
        const dupKey = `${candidate.player}_${candidate.marketType}`;
        if (usedPlayers.has(dupKey)) continue;

        // Check correlation with existing cluster
        const testCluster = [...cluster, candidate];
        const corr = scoreCluster(testCluster);

        // Only add if no strong negative correlation
        if (corr.avgCorr >= -0.1) {
          cluster.push(candidate);
          usedPlayers.add(dupKey);
          usedPlayers.add(candidate.player);
        }
      }

      clusters.push({
        gameId,
        type: cluster.length >= 2 ? 'sgp' : 'single',
        legs: cluster,
        correlation: scoreCluster(cluster),
      });
    }

    return clusters;
  }

  // Helper functions
  function isCreatorRole(role) {
    return role === 'primary_creator' || role === 'secondary_creator';
  }
  function isPointsMarket(mkt) {
    return mkt === 'player_points_alternate';
  }
  function isAssistsMarket(mkt) {
    return mkt === 'player_assists_alternate';
  }
  function isReboundsMarket(mkt) {
    return mkt === 'player_rebounds_alternate';
  }
  function isComboMarket(mkt) {
    return mkt && (mkt.includes('points_assists') || mkt.includes('points_rebounds'));
  }

  return {
    scorePairCorrelation,
    scoreCluster,
    buildSGPClusters,
    CORR_POSITIVE,
    CORR_NEUTRAL,
    CORR_NEGATIVE,
  };
})();
