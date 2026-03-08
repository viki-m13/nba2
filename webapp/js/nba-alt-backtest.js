// =============================================================================
// NBA ALT-LINE BACKTEST — Walk-Forward Validation
// =============================================================================
//
// Validates the alt-line engine using historical box scores and odds data.
// Walk-forward: for each game date, the model ONLY sees prior game data.
// No future data leakage. No parameter tuning on test data.
//
// Produces per-leg and per-parlay results broken down by:
//   - market type (points, assists, rebounds, combo)
//   - builder type (core, screenshot, nuke)
//   - number of legs
//   - role tag
//   - edge bucket
// =============================================================================

window.NbaAltBacktest = (function () {
  'use strict';

  const RungEval = window.NbaAltRungEvaluator;
  const RoleEngine = window.NbaRoleEngine;
  const SlipBuilder = window.NbaScreenshotSlipBuilder;

  // Run walk-forward backtest on historical data
  function runBacktest(boxScores, historicalOdds, rebAstProps) {
    if (!RungEval || !RoleEngine || !SlipBuilder) {
      console.error('[ALT-BACKTEST] Required modules not loaded');
      return null;
    }

    const model = Object.create(window.BettingEngine.PlayerModel);
    model.history = {};

    // Sort box scores by date for walk-forward
    const sortedBoxes = [...boxScores].sort((a, b) => a.date.localeCompare(b.date));

    // Index odds by date + gameKey
    const oddsByDate = {};
    for (const od of (historicalOdds || [])) {
      if (!oddsByDate[od.date]) oddsByDate[od.date] = {};
      oddsByDate[od.date][od.gameKey] = od;
    }

    // Index reb/ast props
    const rebAstByKey = {};
    for (const ra of (rebAstProps || [])) {
      rebAstByKey[`${ra.date}_${ra.gameKey}`] = ra;
    }

    const results = {
      legs: [],       // Every evaluated leg with outcome
      parlays: [],    // Every constructed parlay with outcome
      dates: [],
    };

    const processedDates = new Set();

    for (const boxGame of sortedBoxes) {
      const date = boxGame.date;

      if (!processedDates.has(date)) {
        processedDates.add(date);

        const dayOdds = oddsByDate[date] || {};
        const dayLegs = [];

        // For each game with odds on this date
        for (const [gameKey, oddsRecord] of Object.entries(dayOdds)) {
          // Build environment context from game odds
          const envContext = {
            spread: oddsRecord.spread_point || 0,
            total: oddsRecord.total || 220,
            teamImpliedTotal: (oddsRecord.total || 220) / 2,
            blowoutRisk: Math.min(1, Math.abs(oddsRecord.spread_point || 0) / 15),
          };

          // Evaluate all available markets for all players
          const marketSources = [
            { key: 'playerProps', market: 'player_points_alternate' },
            { key: 'playerRebProps', market: 'player_rebounds_alternate' },
            { key: 'playerAstProps', market: 'player_assists_alternate' },
            { key: 'playerPtsAstProps', market: 'player_points_assists_alternate' },
            { key: 'playerPtsRebProps', market: 'player_points_rebounds_alternate' },
            { key: 'playerPRAProps', market: 'player_points_rebounds_assists_alternate' },
          ];

          // Also check rebAstProps for backward compat
          const raRecord = rebAstByKey[`${date}_${gameKey}`];

          for (const { key, market } of marketSources) {
            let playerLines = oddsRecord[key];

            // Fallback for reb/ast from legacy format
            if (!playerLines && raRecord) {
              if (market === 'player_rebounds_alternate') playerLines = raRecord.rebProps;
              if (market === 'player_assists_alternate') playerLines = raRecord.astProps;
            }

            if (!playerLines) continue;

            for (const [playerName, lines] of Object.entries(playerLines)) {
              const hist = model.history[playerName];
              if (!hist || hist.length < 10) continue;

              const gameLog = hist.map(g => ({
                pts: g.pts || 0,
                reb: g.reb || 0,
                ast: g.ast || 0,
                min: g.min || 0,
              }));

              const roleInfo = RoleEngine.classifyPlayer(gameLog);
              const minutesCertainty = RoleEngine.getMinutesCertainty(gameLog);
              const marketFit = RoleEngine.getMarketFit(roleInfo.role, market);

              const ladder = RungEval.evaluateLadder(gameLog, market, lines, roleInfo, envContext, minutesCertainty);
              const bestRung = RungEval.selectBestRung(ladder);
              if (!bestRung) continue;

              const team = hist[hist.length - 1].team || '';
              dayLegs.push({
                player: playerName,
                team,
                opponent: gameKey.includes('@') ?
                  (gameKey.split('@')[1] === team ? gameKey.split('@')[0] : gameKey.split('@')[1]) : '',
                gameId: oddsRecord.eventId || gameKey,
                gameKey,
                gameDisplay: gameKey.replace('@', ' @ '),
                marketType: market,
                marketLabel: SlipBuilder.marketLabel(market),
                line: bestRung.line,
                bookOdds: bestRung.bookOdds,
                roleTag: roleInfo.role,
                minutesCertainty,
                marketFit,
                expectedMinutes: roleInfo.avgMin || 0,
                l5Hit: bestRung.l5Hit,
                l10Hit: bestRung.l10Hit,
                l20Hit: bestRung.l20Hit,
                avg: bestRung.avg,
                floor: bestRung.floor,
                cv: bestRung.cv,
                modelProb: bestRung.modelProb,
                impliedProb: bestRung.impliedProb,
                fairOdds: bestRung.fairOdds,
                edge: bestRung.edge,
                ev: bestRung.ev,
                fragility: bestRung.fragility,
                stability: bestRung.stability,
                clearance: bestRung.clearance,
                floorClearance: bestRung.floorClearance,
                games: bestRung.games,
                envContext,
                date,
              });
            }
          }
        }

        // Build parlays from today's legs
        if (dayLegs.length > 0) {
          const slips = SlipBuilder.buildAll(dayLegs);

          // Now resolve: find actual outcomes from box scores
          // Find all box score entries for this date
          const dateBoxes = sortedBoxes.filter(b => b.date === date);
          const playerActuals = {};

          for (const bg of dateBoxes) {
            for (const p of (bg.players || [])) {
              playerActuals[p.name] = {
                pts: p.pts || 0,
                reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
                ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
                min: typeof p.min === 'number' ? p.min : parseInt(p.min) || 0,
              };
            }
          }

          // Resolve each leg
          for (const leg of dayLegs) {
            const actual = playerActuals[leg.player];
            if (!actual) continue;

            const statKeys = RungEval.getStatKeys(leg.marketType);
            const actualValue = statKeys.reduce((s, k) => s + (actual[k] || 0), 0);
            const won = actualValue > leg.line;

            results.legs.push({
              ...leg,
              actual: actualValue,
              won,
              pnl: won
                ? Math.round((RungEval.americanToDecimal(leg.bookOdds) - 1) * 100)
                : -100,
            });
          }

          // Resolve parlays
          const resolveParlay = (parlay, builderType) => {
            const resolvedLegs = parlay.legs.map(leg => {
              const actual = playerActuals[leg.player];
              if (!actual) return { ...leg, actual: null, won: null };

              const statKeys = RungEval.getStatKeys(leg.marketType);
              const actualValue = statKeys.reduce((s, k) => s + (actual[k] || 0), 0);
              return { ...leg, actual: actualValue, won: actualValue > leg.line };
            });

            const allResolved = resolvedLegs.every(l => l.actual !== null);
            const allWon = allResolved && resolvedLegs.every(l => l.won);

            return {
              ...parlay,
              builder: builderType,
              date,
              legs: resolvedLegs,
              resolved: allResolved,
              won: allWon,
              pnl: allResolved
                ? (allWon ? Math.round((parlay.decimalOdds - 1) * 100) : -100)
                : 0,
            };
          };

          for (const p of (slips.core || [])) results.parlays.push(resolveParlay(p, 'CORE'));
          for (const p of (slips.screenshot || [])) results.parlays.push(resolveParlay(p, 'SCREENSHOT'));
          for (const p of (slips.nuke || [])) results.parlays.push(resolveParlay(p, 'NUKE'));

          // Super Parlay backtest
          if (window.NbaSuperParlayBuilder) {
            const superSlips = window.NbaSuperParlayBuilder.buildAllWithFallback(dayLegs);
            // v6 autoresearch-optimized strategies
            for (const p of (superSlips.safeParlay || [])) results.parlays.push(resolveParlay(p, p.builder || 'SAFE PARLAY'));
            for (const p of (superSlips.screenshot || [])) results.parlays.push(resolveParlay(p, p.builder || 'SCREENSHOT SPECIAL'));
            for (const p of (superSlips.mini || [])) results.parlays.push(resolveParlay(p, p.builder || 'MINI FORTRESS'));
            for (const p of (superSlips.fortressSGP || [])) results.parlays.push(resolveParlay(p, p.builder || 'FORTRESS SGP'));
            for (const p of (superSlips.fortressSGPPlus || [])) results.parlays.push(resolveParlay(p, p.builder || 'FORTRESS SGP+'));
            for (const p of (superSlips.moonshot || [])) results.parlays.push(resolveParlay(p, p.builder || 'MOONSHOT SGP+'));
            for (const p of (superSlips.assistSniper || [])) results.parlays.push(resolveParlay(p, p.builder || 'ASSIST SNIPER'));
          }
        }

        results.dates.push(date);
      }

      // Walk-forward: update model AFTER predictions
      for (const p of (boxGame.players || [])) {
        const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
        if (mins < 5) continue;
        model.update(p.name, {
          pts: p.pts || 0,
          reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
          ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
          min: mins,
        }, boxGame.date, p.team);
      }
    }

    // Compute summary statistics
    results.stats = computeStats(results);
    return results;
  }

  function computeStats(results) {
    const calcGroup = (items) => {
      const resolved = items.filter(i => i.won !== null && i.won !== undefined);
      if (resolved.length === 0) return { total: 0, wins: 0, hitRate: 0, pnl: 0, roi: 0 };
      const wins = resolved.filter(i => i.won).length;
      const pnl = resolved.reduce((s, i) => s + (i.pnl || 0), 0);
      return {
        total: resolved.length,
        wins,
        losses: resolved.length - wins,
        hitRate: Math.round(wins / resolved.length * 1000) / 1000,
        pnl,
        roi: Math.round(pnl / (resolved.length * 100) * 10000) / 100,
      };
    };

    const legs = results.legs;
    const parlays = results.parlays.filter(p => p.resolved);

    // By market type
    const byMarket = {};
    for (const leg of legs) {
      const m = leg.marketType || 'unknown';
      if (!byMarket[m]) byMarket[m] = [];
      byMarket[m].push(leg);
    }

    // By role
    const byRole = {};
    for (const leg of legs) {
      const r = leg.roleTag || 'unknown';
      if (!byRole[r]) byRole[r] = [];
      byRole[r].push(leg);
    }

    // By builder
    const byBuilder = {};
    for (const p of parlays) {
      const b = p.builder || 'unknown';
      if (!byBuilder[b]) byBuilder[b] = [];
      byBuilder[b].push(p);
    }

    // By number of legs
    const byNumLegs = {};
    for (const p of parlays) {
      const n = p.numLegs || p.legs.length;
      if (!byNumLegs[n]) byNumLegs[n] = [];
      byNumLegs[n].push(p);
    }

    // By edge bucket
    const byEdge = { 'low (0-3%)': [], 'mid (3-6%)': [], 'high (6%+)': [] };
    for (const leg of legs) {
      const e = leg.edge || 0;
      if (e < 0.03) byEdge['low (0-3%)'].push(leg);
      else if (e < 0.06) byEdge['mid (3-6%)'].push(leg);
      else byEdge['high (6%+)'].push(leg);
    }

    return {
      overall: {
        legs: calcGroup(legs),
        parlays: calcGroup(parlays),
      },
      byMarket: Object.fromEntries(Object.entries(byMarket).map(([k, v]) => [k, calcGroup(v)])),
      byRole: Object.fromEntries(Object.entries(byRole).map(([k, v]) => [k, calcGroup(v)])),
      byBuilder: Object.fromEntries(Object.entries(byBuilder).map(([k, v]) => [k, calcGroup(v)])),
      byNumLegs: Object.fromEntries(Object.entries(byNumLegs).map(([k, v]) => [k, calcGroup(v)])),
      byEdge: Object.fromEntries(Object.entries(byEdge).map(([k, v]) => [k, calcGroup(v)])),
      totalDates: results.dates.length,
    };
  }

  return {
    runBacktest,
    computeStats,
  };
})();
