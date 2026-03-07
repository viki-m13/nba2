// =============================================================================
// NBA PROP BETTING ENGINE v6.0 — Alt-Line Parlay Engine
// =============================================================================
//
// Multi-market alternate prop engine supporting:
//   - player_points_alternate
//   - player_assists_alternate
//   - player_rebounds_alternate
//   - player_points_assists_alternate
//   - player_points_rebounds_alternate
//   - player_points_rebounds_assists_alternate
//
// Uses market-specific evaluators, role-aware logic, alt-rung ladder
// optimization, same-game correlation, and three parlay constructors
// (Core / Screenshot / Nuke) to produce screenshot-style FanDuel slips.
//
// Integrates with:
//   - NbaRoleEngine — player role classification
//   - NbaAltRungEvaluator — market-specific rung evaluation
//   - NbaCorrelationEngine — SGP correlation scoring
//   - NbaScreenshotSlipBuilder — parlay construction
//
// =============================================================================

window.BettingEngine = (function () {
  'use strict';

  // --- Configuration (minimal to avoid overfitting) ---

  const CONFIG = {
    MIN_GAMES: 10,          // Minimum games of history
    FLOOR_WINDOW: 10,       // Recent window for floor confirmation
    CONSISTENCY_WINDOW: 20, // Longer window for hit rate consistency
    MIN_MINUTES: 10,        // Skip DNP/garbage-time players
    MIN_HIT_RATE: 0.85,     // L20 hit rate threshold
    MIN_LEG_ODDS: -500,     // Minimum per-leg odds (sweet spot lower bound)
    MAX_LEG_ODDS: -150,     // Maximum per-leg odds (sweet spot upper bound)
    UNIT_SIZE: 100,         // Dollars per bet
    PARLAY_MIN_LEGS: 3,
    PARLAY_MAX_LEGS: 6,
    MAX_LEGS_PER_GAME: 2,   // Diversification
    MIN_PARLAY_ODDS: 100,   // Positive odds parlays only
  };

  const ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8';
  const ODDS_API_BASE = 'https://api.the-odds-api.com/v4';

  const TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GS', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NO',
    'New York Knicks': 'NY', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SA',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTAH',
    'Washington Wizards': 'WSH',
  };

  function teamAbbr(name) { return TEAM_MAP[name] || name; }

  // --- Odds Utilities ---

  function americanToDecimal(odds) {
    return odds > 0 ? 1 + odds / 100 : 1 + 100 / Math.abs(odds);
  }

  function decimalToAmerican(decimal) {
    if (decimal >= 2.0) return Math.round((decimal - 1) * 100);
    return Math.round(-100 / (decimal - 1));
  }

  function formatOdds(odds) {
    return odds > 0 ? `+${odds}` : `${odds}`;
  }

  const STAT_LABELS = { points: 'PTS', rebounds: 'REB', assists: 'AST' };
  function statLabel(statType) { return STAT_LABELS[statType] || statType.toUpperCase(); }

  // --- Player Model ---

  const PlayerModel = {
    history: {},

    reset() { this.history = {}; },

    update(name, stats, date, team) {
      if (!this.history[name]) this.history[name] = [];
      this.history[name].push({ ...stats, date, team });
      // Keep last 50 games
      if (this.history[name].length > 50) {
        this.history[name] = this.history[name].slice(-40);
      }
    },

    // Evaluate a single prop using dual confirmation:
    // 1. L10 floor > line (recent form)
    // 2. L20 hit rate >= 85% (long-term consistency)
    // 3. Odds in sweet spot range (-500 to -150)
    evaluate(playerName, statType, line, odds) {
      const hist = this.history[playerName];
      if (!hist || hist.length < CONFIG.MIN_GAMES) return null;

      const statKey = statType === 'points' ? 'pts' : statType === 'rebounds' ? 'reb' : 'ast';

      // Recent activity check
      const lastGame = hist[hist.length - 1];
      if (lastGame.min < 5) return null;

      // Odds sweet spot filter
      if (odds < CONFIG.MIN_LEG_ODDS || odds > CONFIG.MAX_LEG_ODDS) return null;

      // CONFIRMATION 1: L10 floor must be strictly above the line
      const floorWindow = Math.min(CONFIG.FLOOR_WINDOW, hist.length);
      const recentShort = hist.slice(-floorWindow);
      const shortValues = recentShort.map(g => g[statKey]);
      const floor = Math.min(...shortValues);
      if (floor <= line) return null;

      // CONFIRMATION 2: L20 hit rate must meet threshold
      const consistWindow = Math.min(CONFIG.CONSISTENCY_WINDOW, hist.length);
      const recentLong = hist.slice(-consistWindow);
      const longValues = recentLong.map(g => g[statKey]);
      const hits = longValues.filter(v => v > line).length;
      const hitRate = hits / longValues.length;
      if (hitRate < CONFIG.MIN_HIT_RATE) return null;

      const avg = longValues.reduce((s, v) => s + v, 0) / longValues.length;
      const decimal = americanToDecimal(odds);
      const impliedProb = 1 / decimal;
      const edge = hitRate - impliedProb;

      return {
        player: playerName,
        team: recentLong[recentLong.length - 1].team,
        statType,
        statLabel: statLabel(statType),
        line,
        odds,
        hitRate: Math.round(hitRate * 1000) / 1000,
        avg: Math.round(avg * 10) / 10,
        floor,
        edge: Math.round(edge * 1000) / 1000,
        ev: Math.round((hitRate * decimal - 1) * 1000) / 1000,
        games: consistWindow,
      };
    },

    // Find the BEST line for a player+stat from available FanDuel lines
    // "Best" = line with highest edge (hit rate - implied probability)
    findBestProp(playerName, statType, fdLines) {
      if (!fdLines || typeof fdLines !== 'object') return null;

      let best = null;
      for (const [threshold, data] of Object.entries(fdLines)) {
        const line = parseFloat(threshold);
        const odds = data.overOdds;
        if (!odds) continue;
        const result = this.evaluate(playerName, statType, line, odds);
        if (!result) continue;

        // Pick the line with highest edge
        if (!best || result.edge > best.edge) {
          best = result;
        }
      }
      return best;
    },
  };

  // --- Parlay Builder ---

  function buildParlays(legs) {
    if (legs.length < CONFIG.PARLAY_MIN_LEGS) return [];

    // Sort by edge descending (highest-edge legs first)
    const sorted = [...legs].sort((a, b) => b.edge - a.edge);

    // Deduplicate: one leg per player+stat (best edge wins)
    const seen = new Set();
    const unique = [];
    for (const leg of sorted) {
      const key = `${leg.player}_${leg.statType}`;
      if (!seen.has(key)) {
        seen.add(key);
        unique.push(leg);
      }
    }

    if (unique.length < CONFIG.PARLAY_MIN_LEGS) return [];

    const parlays = [];
    const maxLegs = Math.min(CONFIG.PARLAY_MAX_LEGS, unique.length);

    for (let size = CONFIG.PARLAY_MIN_LEGS; size <= maxLegs; size++) {
      const parlay = buildBestParlay(unique, size);
      if (parlay && parlay.odds >= CONFIG.MIN_PARLAY_ODDS) parlays.push(parlay);
    }

    return parlays;
  }

  function buildBestParlay(legs, targetSize) {
    // Greedy selection: pick highest-edge legs with game diversification
    const selected = [];
    const gameCount = {};
    const usedPlayers = new Set();

    for (const leg of legs) {
      if (selected.length >= targetSize) break;
      if (usedPlayers.has(leg.player)) continue;
      const gc = gameCount[leg.gameKey] || 0;
      if (gc >= CONFIG.MAX_LEGS_PER_GAME) continue;

      selected.push(leg);
      usedPlayers.add(leg.player);
      gameCount[leg.gameKey] = gc + 1;
    }

    if (selected.length < targetSize) return null;

    const decimal = selected.reduce((d, l) => d * americanToDecimal(l.odds), 1);
    const american = decimalToAmerican(decimal);
    const combinedHitRate = selected.reduce((p, l) => p * l.hitRate, 1);
    const ev = combinedHitRate * decimal - 1;

    return {
      legs: selected.map(l => ({ ...l })),
      numLegs: targetSize,
      odds: american,
      decimalOdds: Math.round(decimal * 100) / 100,
      combinedHitRate: Math.round(combinedHitRate * 1000) / 1000,
      ev: Math.round(ev * 1000) / 1000,
    };
  }

  // --- Backtest ---

  function runBacktest(seasonData, boxScores, historicalOdds, rebAstProps) {
    const model = Object.create(PlayerModel);
    model.history = {};

    const sortedGames = [...seasonData].sort((a, b) => a.date.localeCompare(b.date));

    // Index box scores, odds, and reb/ast by date
    const boxByDate = {};
    for (const g of boxScores) {
      (boxByDate[g.date] || (boxByDate[g.date] = [])).push(g);
    }
    const oddsByDate = {};
    for (const od of (historicalOdds || [])) {
      if (!oddsByDate[od.date]) oddsByDate[od.date] = {};
      oddsByDate[od.date][od.gameKey] = od;
    }
    const rebAstByKey = {};
    for (const ra of (rebAstProps || [])) {
      rebAstByKey[`${ra.date}_${ra.gameKey}`] = ra;
    }

    const results = { singles: [], parlays: [], dailySummaries: [], dates: [] };
    const processedDates = new Set();

    for (const game of sortedGames) {
      const date = game.date;

      if (!processedDates.has(date)) {
        processedDates.add(date);

        const dayOdds = oddsByDate[date] || {};
        const dayBoxes = boxByDate[date] || [];
        const daySingles = [];

        for (const bg of dayBoxes) {
          const gameKey = `${bg.away}@${bg.home}`;
          const og = dayOdds[gameKey];
          const ra = rebAstByKey[`${date}_${gameKey}`];

          for (const player of (bg.players || [])) {
            const mins = typeof player.min === 'number' ? player.min : parseInt(player.min) || 0;
            if (mins < CONFIG.MIN_MINUTES) continue;

            const actualReb = typeof player.reb === 'number' ? player.reb : parseInt(player.reb) || 0;
            const actualAst = typeof player.ast === 'number' ? player.ast : parseInt(player.ast) || 0;

            // Points
            const ptsLines = og && og.playerProps && og.playerProps[player.name];
            if (ptsLines) {
              const prop = model.findBestProp(player.name, 'points', ptsLines);
              if (prop) {
                const won = player.pts > prop.line;
                daySingles.push({ ...prop, date, gameKey, actual: player.pts, won,
                  pnl: won ? Math.round((americanToDecimal(prop.odds) - 1) * CONFIG.UNIT_SIZE) : -CONFIG.UNIT_SIZE });
              }
            }

            // Rebounds
            const rebLines = (ra && ra.rebProps && ra.rebProps[player.name])
              || (og && og.playerRebProps && og.playerRebProps[player.name]);
            if (rebLines) {
              const prop = model.findBestProp(player.name, 'rebounds', rebLines);
              if (prop) {
                const won = actualReb > prop.line;
                daySingles.push({ ...prop, date, gameKey, actual: actualReb, won,
                  pnl: won ? Math.round((americanToDecimal(prop.odds) - 1) * CONFIG.UNIT_SIZE) : -CONFIG.UNIT_SIZE });
              }
            }

            // Assists
            const astLines = (ra && ra.astProps && ra.astProps[player.name])
              || (og && og.playerAstProps && og.playerAstProps[player.name]);
            if (astLines) {
              const prop = model.findBestProp(player.name, 'assists', astLines);
              if (prop) {
                const won = actualAst > prop.line;
                daySingles.push({ ...prop, date, gameKey, actual: actualAst, won,
                  pnl: won ? Math.round((americanToDecimal(prop.odds) - 1) * CONFIG.UNIT_SIZE) : -CONFIG.UNIT_SIZE });
              }
            }
          }
        }

        // Build parlays
        const dayParlays = buildParlays(daySingles);
        for (const parlay of dayParlays) {
          parlay.date = date;
          parlay.won = parlay.legs.every(leg => {
            const match = daySingles.find(s =>
              s.player === leg.player && s.line === leg.line && s.statType === leg.statType
            );
            return match && match.won;
          });
          parlay.pnl = parlay.won
            ? Math.round((parlay.decimalOdds - 1) * CONFIG.UNIT_SIZE)
            : -CONFIG.UNIT_SIZE;
          parlay.legs = parlay.legs.map(leg => {
            const match = daySingles.find(s =>
              s.player === leg.player && s.line === leg.line && s.statType === leg.statType
            );
            return { ...leg, won: match ? match.won : false, actual: match ? match.actual : 0 };
          });
        }

        results.singles.push(...daySingles);
        results.parlays.push(...dayParlays);
        if (daySingles.length > 0) {
          results.dailySummaries.push({
            date, singles: daySingles.length, parlays: dayParlays.length,
            wins: [...daySingles, ...dayParlays].filter(p => p.won).length,
            total: daySingles.length + dayParlays.length,
            pnl: [...daySingles, ...dayParlays].reduce((s, p) => s + p.pnl, 0),
          });
        }
        results.dates.push(date);
      }

      // Walk-forward: update model AFTER making predictions for this date
      const dateBoxes = boxByDate[game.date] || [];
      for (const bg of dateBoxes) {
        if (bg.home === game.home_team && bg.away === game.away_team) {
          for (const p of (bg.players || [])) {
            const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
            if (mins < CONFIG.MIN_MINUTES) continue;
            model.update(p.name, {
              pts: p.pts,
              reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
              ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
              min: mins,
            }, game.date, p.team);
          }
        }
      }
    }

    // Calculate summary stats
    const calcGroup = (picks) => {
      if (picks.length === 0) return { total: 0, wins: 0, losses: 0, hitRate: 0, pnl: 0, roi: 0 };
      const wins = picks.filter(p => p.won).length;
      const pnl = picks.reduce((s, p) => s + p.pnl, 0);
      return {
        total: picks.length, wins, losses: picks.length - wins,
        hitRate: Math.round(wins / picks.length * 1000) / 1000,
        pnl, roi: Math.round(pnl / (picks.length * CONFIG.UNIT_SIZE) * 10000) / 100,
      };
    };

    results.stats = {
      overall: calcGroup([...results.singles, ...results.parlays]),
      singles: calcGroup(results.singles),
      parlays: calcGroup(results.parlays),
      points: calcGroup(results.singles.filter(s => s.statType === 'points')),
      rebounds: calcGroup(results.singles.filter(s => s.statType === 'rebounds')),
      assists: calcGroup(results.singles.filter(s => s.statType === 'assists')),
      totalDays: results.dates.length,
      daysWithPicks: results.dailySummaries.length,
    };

    // Rolling recent accuracy
    const allDates = [...new Set(results.parlays.map(p => p.date))].sort();
    if (allDates.length > 0) {
      const last30 = allDates[Math.max(0, allDates.length - 30)] || '';
      const last14 = allDates[Math.max(0, allDates.length - 14)] || '';
      const last7 = allDates[Math.max(0, allDates.length - 7)] || '';
      results.stats.recent30 = calcGroup(results.parlays.filter(p => p.date >= last30));
      results.stats.recent14 = calcGroup(results.parlays.filter(p => p.date >= last14));
      results.stats.recent7 = calcGroup(results.parlays.filter(p => p.date >= last7));
    }

    return results;
  }

  // --- Live Odds Fetching ---

  async function fetchLiveOdds() {
    const result = { events: [], playerProps: {} };

    try {
      const eventsUrl = `${ODDS_API_BASE}/sports/basketball_nba/events?apiKey=${ODDS_API_KEY}`;
      const eventsRes = await fetch(eventsUrl);
      const remaining = eventsRes.headers.get('x-requests-remaining');
      console.log(`[ENGINE] Odds API: remaining=${remaining}`);

      if (!eventsRes.ok) {
        if (eventsRes.status === 401 || remaining === '0') result.quotaExhausted = true;
        return result;
      }

      const events = await eventsRes.json();
      result.events = events;

      // All supported alternate markets including combo stats
      const ALL_ALT_MARKETS = [
        'player_points_alternate',
        'player_rebounds_alternate',
        'player_assists_alternate',
        'player_points_assists_alternate',
        'player_points_rebounds_alternate',
        'player_points_rebounds_assists_alternate',
      ];
      const markets = ALL_ALT_MARKETS.join(',');

      for (const event of events) {
        try {
          const propsUrl = `${ODDS_API_BASE}/sports/basketball_nba/events/${event.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=${markets}&oddsFormat=american&bookmakers=fanduel`;
          const propsRes = await fetch(propsUrl);
          if (!propsRes.ok) continue;
          const propsData = await propsRes.json();

          const homeAbbr = teamAbbr(event.home_team);
          const awayAbbr = teamAbbr(event.away_team);
          const gameKey = `${awayAbbr}@${homeAbbr}`;

          const fd = propsData.bookmakers && propsData.bookmakers.find(b => b.key === 'fanduel');
          if (!fd) continue;

          const gameProps = {
            points: {}, rebounds: {}, assists: {},
            points_assists: {}, points_rebounds: {}, points_rebounds_assists: {},
          };
          const marketMap = {
            'player_points_alternate': 'points',
            'player_rebounds_alternate': 'rebounds',
            'player_assists_alternate': 'assists',
            'player_points_assists_alternate': 'points_assists',
            'player_points_rebounds_alternate': 'points_rebounds',
            'player_points_rebounds_assists_alternate': 'points_rebounds_assists',
          };

          for (const mkt of (fd.markets || [])) {
            const st = marketMap[mkt.key];
            if (!st) continue;
            for (const outcome of (mkt.outcomes || [])) {
              const player = outcome.description;
              const threshold = outcome.point;
              if (!gameProps[st][player]) gameProps[st][player] = {};
              if (!gameProps[st][player][threshold]) gameProps[st][player][threshold] = {};
              if (outcome.name === 'Over') {
                gameProps[st][player][threshold].overOdds = outcome.price;
              }
            }
          }

          result.playerProps[gameKey] = {
            lines: gameProps.points,
            rebLines: gameProps.rebounds,
            astLines: gameProps.assists,
            ptsAstLines: gameProps.points_assists,
            ptsRebLines: gameProps.points_rebounds,
            praLines: gameProps.points_rebounds_assists,
            eventId: event.id,
            home: homeAbbr,
            away: awayAbbr,
          };
        } catch (e) {
          console.warn('[ENGINE] Error fetching props:', event.id);
        }
      }

      console.log(`[ENGINE] Fetched props for ${Object.keys(result.playerProps).length} games`);
    } catch (e) {
      console.error('[ENGINE] Error fetching live odds:', e);
    }

    return result;
  }

  // --- Live Pick Tracking ---

  const LIVE_PICKS_KEY = 'nba_live_picks';
  let basePicksData = null;

  function setBasePicksData(data) {
    basePicksData = data;
    console.log(`[ENGINE] Base picks loaded: ${(data.parlays || []).length} parlays`);
  }

  function loadLivePicks() {
    const base = basePicksData || { parlays: [] };
    let local;
    try {
      local = JSON.parse(localStorage.getItem(LIVE_PICKS_KEY) || '{"parlays":[]}');
    } catch (e) {
      local = { parlays: [] };
    }

    // Normalize: flatten spParlays into parlays for backward compat
    const baseParlays = [...(base.parlays || []), ...(base.spParlays || [])];
    const localParlays = [...(local.parlays || []), ...(local.spParlays || [])];

    const localDates = new Set(localParlays.map(p => p.date));

    return {
      parlays: [
        ...baseParlays.filter(p => !localDates.has(p.date)),
        ...localParlays,
      ],
    };
  }

  function saveLivePicks(data) {
    const baseDates = new Set();
    if (basePicksData) {
      for (const p of (basePicksData.parlays || [])) baseDates.add(p.date);
      for (const p of (basePicksData.spParlays || [])) baseDates.add(p.date);
    }

    const localOnly = {
      parlays: (data.parlays || []).filter(p => !baseDates.has(p.date)),
    };

    try {
      localStorage.setItem(LIVE_PICKS_KEY, JSON.stringify(localOnly));
    } catch (e) {
      console.warn('[ENGINE] Could not save live picks:', e);
    }
  }

  function saveTodayParlays(parlays, spParlays, date) {
    const data = loadLivePicks();

    // Remove existing entries for this date
    data.parlays = data.parlays.filter(p => p.date !== date);

    // Merge both standard and SP parlays into one list
    const allParlays = [...(parlays || []), ...(spParlays || [])];
    for (const parlay of allParlays) {
      data.parlays.push({
        date,
        numLegs: parlay.numLegs || parlay.legs.length,
        odds: parlay.odds,
        decimalOdds: parlay.decimalOdds,
        combinedHitRate: parlay.combinedHitRate,
        ev: parlay.ev,
        legs: parlay.legs.map(l => ({
          player: l.player, team: l.team,
          statType: l.statType || 'points',
          statLabel: l.statLabel || statLabel(l.statType || 'points'),
          line: l.line, odds: l.odds,
          gameKey: l.gameKey || '', gameDisplay: l.gameDisplay || '',
          actual: null, won: null,
        })),
        resolved: false, won: null, pnl: null,
        savedAt: new Date().toISOString(),
      });
    }

    saveLivePicks(data);
    console.log(`[ENGINE] Saved ${allParlays.length} parlays for ${date}`);
  }

  async function resolveLivePicks() {
    const data = loadLivePicks();
    const unresolvedDates = [...new Set(
      data.parlays.filter(p => !p.resolved).map(p => p.date)
    )];

    const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    const datesToResolve = unresolvedDates.filter(d => d < today);
    if (datesToResolve.length === 0) return data;

    console.log(`[ENGINE] Resolving picks for ${datesToResolve.length} dates`);

    for (const date of datesToResolve) {
      const { games, eventIds } = await window.NbaApi.fetchESPNScoreboardForDate(date);
      if (!games.length) continue;

      const playerStats = {};
      let finalCount = 0;
      for (const [teamKey, eventId] of Object.entries(eventIds)) {
        const game = games.find(g => g.id === eventId || g.espnId === eventId);
        if (game && game.status !== 'final') continue;
        finalCount++;
        const boxScore = await window.NbaApi.fetchESPNBoxScore(eventId);
        if (!boxScore) continue;
        for (const p of boxScore) playerStats[p.name.toLowerCase()] = p;
        await new Promise(r => setTimeout(r, 200));
      }

      const allFinal = finalCount === Object.keys(eventIds).length;
      if (Object.keys(playerStats).length === 0) continue;

      const findPlayer = (name) => {
        const lower = name.toLowerCase();
        if (playerStats[lower]) return playerStats[lower];
        const parts = lower.split(' ');
        if (parts.length >= 2) {
          const found = Object.entries(playerStats).find(([n]) =>
            n.includes(parts[0]) && n.includes(parts[parts.length - 1])
          );
          if (found) return found[1];
        }
        return null;
      };

      for (const parlay of data.parlays) {
        if (parlay.date !== date || parlay.resolved) continue;

        for (const leg of parlay.legs) {
          if (leg.actual !== null && leg.actual !== undefined) continue;
          const pStats = findPlayer(leg.player);
          if (pStats) {
            let actual;
            if (leg.statType === 'rebounds') actual = pStats.reb;
            else if (leg.statType === 'assists') actual = pStats.ast;
            else if (leg.statType === 'points_assists') actual = (pStats.pts || 0) + (pStats.ast || 0);
            else if (leg.statType === 'points_rebounds') actual = (pStats.pts || 0) + (pStats.reb || 0);
            else if (leg.statType === 'points_rebounds_assists') actual = (pStats.pts || 0) + (pStats.reb || 0) + (pStats.ast || 0);
            else actual = pStats.pts;
            leg.actual = actual;
            leg.won = leg.actual > leg.line;
          } else if (allFinal) {
            leg.actual = 0;
            leg.won = false;
          }
        }

        if (parlay.legs.every(l => l.actual !== null && l.actual !== undefined)) {
          parlay.resolved = true;
          parlay.won = parlay.legs.every(l => l.won);
          parlay.pnl = parlay.won
            ? Math.round((parlay.decimalOdds - 1) * CONFIG.UNIT_SIZE)
            : -CONFIG.UNIT_SIZE;
        }
      }
    }

    saveLivePicks(data);
    return data;
  }

  // --- Historical Odds Fetching ---

  async function fetchHistoricalOddsForDate(dateStr) {
    const isoDate = `${dateStr.slice(0,4)}-${dateStr.slice(4,6)}-${dateStr.slice(6,8)}T12:00:00Z`;
    const result = [];

    try {
      const gamesUrl = `${ODDS_API_BASE}/historical/sports/basketball_nba/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&date=${isoDate}&bookmakers=fanduel`;
      const gamesRes = await fetch(gamesUrl);
      if (!gamesRes.ok) return result;
      const gamesData = await gamesRes.json();
      const events = gamesData.data || [];

      for (const event of events) {
        const homeAbbr = teamAbbr(event.home_team);
        const awayAbbr = teamAbbr(event.away_team);
        const gameKey = `${awayAbbr}@${homeAbbr}`;
        const fd = (event.bookmakers || []).find(b => b.key === 'fanduel');
        if (!fd) continue;

        const record = {
          date: dateStr, gameKey, eventId: event.id,
          homeTeam: homeAbbr, awayTeam: awayAbbr, commenceTime: event.commence_time,
        };

        for (const mkt of (fd.markets || [])) {
          if (mkt.key === 'h2h') {
            const ho = mkt.outcomes.find(o => o.name === event.home_team);
            const ao = mkt.outcomes.find(o => o.name === event.away_team);
            if (ho) record.home_ml = ho.price;
            if (ao) record.away_ml = ao.price;
          } else if (mkt.key === 'spreads') {
            const ho = mkt.outcomes.find(o => o.name === event.home_team);
            if (ho) { record.spread_home = ho.price; record.spread_point = ho.point; }
          } else if (mkt.key === 'totals') {
            const ov = mkt.outcomes.find(o => o.name === 'Over');
            if (ov) { record.total = ov.point; record.total_over = ov.price; }
          }
        }

        // Fetch player props — all alternate markets including combos
        const propMarkets = 'player_points_alternate,player_rebounds_alternate,player_assists_alternate,player_points_assists_alternate,player_points_rebounds_alternate,player_points_rebounds_assists_alternate';
        try {
          const propsUrl = `${ODDS_API_BASE}/historical/sports/basketball_nba/events/${event.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=${propMarkets}&oddsFormat=american&date=${isoDate}&bookmakers=fanduel`;
          const propsRes = await fetch(propsUrl);
          if (propsRes.ok) {
            const propsData = await propsRes.json();
            const eventData = propsData.data || propsData;
            const fdBook = (eventData.bookmakers || []).find(b => b.key === 'fanduel');
            if (fdBook) {
              const mmap = {
                'player_points_alternate': 'playerProps',
                'player_rebounds_alternate': 'playerRebProps',
                'player_assists_alternate': 'playerAstProps',
                'player_points_assists_alternate': 'playerPtsAstProps',
                'player_points_rebounds_alternate': 'playerPtsRebProps',
                'player_points_rebounds_assists_alternate': 'playerPRAProps',
              };
              for (const mkt of (fdBook.markets || [])) {
                const propKey = mmap[mkt.key];
                if (!propKey) continue;
                const playerLines = {};
                for (const outcome of (mkt.outcomes || [])) {
                  const player = outcome.description;
                  const threshold = outcome.point;
                  if (!playerLines[player]) playerLines[player] = {};
                  if (!playerLines[player][threshold]) playerLines[player][threshold] = {};
                  if (outcome.name === 'Over') playerLines[player][threshold].overOdds = outcome.price;
                }
                record[propKey] = playerLines;
              }
            }
          }
        } catch (e) { /* skip props */ }

        result.push(record);
      }
    } catch (e) {
      console.warn(`[ENGINE] Error fetching historical odds for ${dateStr}:`, e);
    }

    return result;
  }

  // Incremental daily odds cache
  const DAILY_ODDS_KEY = 'nba_daily_odds_cache';

  function loadCachedDailyOdds() {
    try { return JSON.parse(localStorage.getItem(DAILY_ODDS_KEY) || '{}'); }
    catch (e) { return {}; }
  }

  function saveCachedDailyOdds(cache) {
    try { localStorage.setItem(DAILY_ODDS_KEY, JSON.stringify(cache)); }
    catch (e) {}
  }

  async function fetchMissingDailyOdds(historicalOdds, maxDates) {
    const existingDates = new Set(historicalOdds.map(o => o.date));
    const cache = loadCachedDailyOdds();
    const newOdds = [];
    let fetchedCount = 0;

    const lastDate = [...existingDates].sort().pop() || '20250224';
    const today = new Date();
    const dates = [];
    const startDate = new Date(`${lastDate.slice(0,4)}-${lastDate.slice(4,6)}-${lastDate.slice(6,8)}`);
    startDate.setDate(startDate.getDate() + 1);
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    for (let d = new Date(startDate); d <= yesterday; d.setDate(d.getDate() + 1)) {
      const ds = `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}`;
      if (!existingDates.has(ds)) dates.push(ds);
    }

    for (const date of dates) {
      if (cache[date]) { newOdds.push(...cache[date]); continue; }
      if (fetchedCount >= (maxDates || 5)) break;

      console.log(`[ENGINE] Fetching odds for ${date}...`);
      const dateOdds = await fetchHistoricalOddsForDate(date);
      cache[date] = dateOdds;
      if (dateOdds.length > 0) newOdds.push(...dateOdds);
      saveCachedDailyOdds(cache);
      fetchedCount++;
      await new Promise(r => setTimeout(r, 300));
    }

    return newOdds;
  }

  // Legacy no-ops for backward compatibility
  function saveOddsToHistory() {}

  // =========================================================================
  // ALT-LINE PARLAY ENGINE — Feature Table + Slip Generation
  // =========================================================================

  // All supported markets and their line accessors from fetched odds
  const MARKET_LINE_KEYS = {
    'player_points_alternate':                   'lines',
    'player_assists_alternate':                  'astLines',
    'player_rebounds_alternate':                 'rebLines',
    'player_points_assists_alternate':           'ptsAstLines',
    'player_points_rebounds_alternate':          'ptsRebLines',
    'player_points_rebounds_assists_alternate':  'praLines',
  };

  const MARKET_LABELS = {
    'player_points_alternate': 'ALT PTS',
    'player_assists_alternate': 'ALT AST',
    'player_rebounds_alternate': 'ALT REB',
    'player_points_assists_alternate': 'ALT PTS+AST',
    'player_points_rebounds_alternate': 'ALT PTS+REB',
    'player_points_rebounds_assists_alternate': 'ALT PTS+REB+AST',
  };

  // Build game environment context from game odds data
  function buildEnvContext(gameOdds) {
    if (!gameOdds) return null;
    const spread = gameOdds.spread_point || 0;
    const total = gameOdds.total || 220;
    const homeML = gameOdds.home_ml || 0;
    const awayML = gameOdds.away_ml || 0;

    // Approximate team implied totals
    const teamImpliedTotal = total / 2 + (spread / 2);
    const oppImpliedTotal = total / 2 - (spread / 2);

    // Blowout risk: large spread = more blowout risk
    const blowoutRisk = Math.min(1, Math.abs(spread) / 15);

    return {
      spread,
      total,
      teamImpliedTotal: Math.round(teamImpliedTotal * 10) / 10,
      oppImpliedTotal: Math.round(oppImpliedTotal * 10) / 10,
      blowoutRisk: Math.round(blowoutRisk * 100) / 100,
      homeML,
      awayML,
    };
  }

  // Generate the nightly player-market-rung feature table
  // This is the CORE decision object for the alt-line engine
  function generateFeatureTable(liveOdds, gameOddsMap) {
    if (!liveOdds || !liveOdds.playerProps) return [];
    if (!window.NbaRoleEngine || !window.NbaAltRungEvaluator) {
      console.warn('[ENGINE] Alt-line modules not loaded');
      return [];
    }

    const RoleEngine = window.NbaRoleEngine;
    const RungEval = window.NbaAltRungEvaluator;
    const featureTable = [];

    for (const [gameKey, gameProps] of Object.entries(liveOdds.playerProps)) {
      const envContext = gameOddsMap ? buildEnvContext(gameOddsMap[gameKey]) : null;

      // Process each market type
      for (const [marketType, lineKey] of Object.entries(MARKET_LINE_KEYS)) {
        const playerLines = gameProps[lineKey];
        if (!playerLines) continue;

        for (const [playerName, lines] of Object.entries(playerLines)) {
          // Get player history from model
          const hist = PlayerModel.history[playerName];
          if (!hist || hist.length < CONFIG.MIN_GAMES) continue;

          // Build game log in the format expected by evaluators
          const gameLog = hist.map(g => ({
            pts: g.pts || 0,
            reb: g.reb || 0,
            ast: g.ast || 0,
            min: g.min || 0,
          }));

          // Classify player role
          const roleInfo = RoleEngine.classifyPlayer(gameLog);
          const marketFit = RoleEngine.getMarketFit(roleInfo.role, marketType);
          const minutesCertainty = RoleEngine.getMinutesCertainty(gameLog);
          const isElevated = RoleEngine.detectInjuryElevation(gameLog);

          if (isElevated) roleInfo.role = 'injury_elevated';

          // Evaluate the full ladder for this player + market
          const ladder = RungEval.evaluateLadder(
            gameLog, marketType, lines, roleInfo, envContext, minutesCertainty
          );

          // Select the best rung (highest safe rung)
          const bestRung = RungEval.selectBestRung(ladder);
          if (!bestRung) continue;

          // Determine player team
          const team = hist[hist.length - 1].team || '';

          featureTable.push({
            player: playerName,
            team,
            opponent: gameKey.includes('@') ?
              (gameKey.split('@')[1] === team ? gameKey.split('@')[0] : gameKey.split('@')[1]) : '',
            gameId: gameProps.eventId || gameKey,
            gameKey,
            gameDisplay: gameKey.replace('@', ' @ '),
            marketType,
            marketLabel: MARKET_LABELS[marketType] || marketType,
            line: bestRung.line,
            bookOdds: bestRung.bookOdds,
            roleTag: roleInfo.role,
            roleConfidence: roleInfo.confidence,
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
            // Full ladder for inspection
            ladderSize: ladder.length,
          });
        }
      }
    }

    // Sort by composite score
    featureTable.sort((a, b) => {
      const scoreA = a.edge * 3 + a.stability * 2 + a.modelProb;
      const scoreB = b.edge * 3 + b.stability * 2 + b.modelProb;
      return scoreB - scoreA;
    });

    console.log(`[ENGINE] Feature table: ${featureTable.length} candidate legs across ${Object.keys(liveOdds.playerProps).length} games`);
    return featureTable;
  }

  // Generate all parlay slips from the feature table
  function generateSlips(featureTable) {
    if (!window.NbaScreenshotSlipBuilder) {
      console.warn('[ENGINE] Slip builder not loaded');
      return { core: [], screenshot: [], nuke: [] };
    }
    return window.NbaScreenshotSlipBuilder.buildAll(featureTable);
  }

  // Full pipeline: fetch odds → build model → generate feature table → build slips
  async function runAltLinePipeline(gameOddsMap) {
    console.log('[ENGINE] Running alt-line pipeline...');
    const liveOdds = await fetchLiveOdds();

    if (liveOdds.quotaExhausted) {
      console.warn('[ENGINE] API quota exhausted');
      return { featureTable: [], slips: { core: [], screenshot: [], nuke: [] }, quotaExhausted: true };
    }

    const featureTable = generateFeatureTable(liveOdds, gameOddsMap);
    const slips = generateSlips(featureTable);

    console.log(`[ENGINE] Pipeline complete: ${slips.core.length} core, ${slips.screenshot.length} screenshot, ${slips.nuke.length} nuke parlays`);
    return { featureTable, slips, liveOdds };
  }

  // --- Public API ---

  return {
    CONFIG,
    PlayerModel,
    buildParlays,
    runBacktest,
    fetchLiveOdds,
    saveOddsToHistory,
    saveTodayParlays,
    loadLivePicks,
    setBasePicksData,
    resolveLivePicks,
    fetchMissingDailyOdds,
    fetchHistoricalOddsForDate,
    loadCachedDailyOdds,
    americanToDecimal,
    decimalToAmerican,
    formatOdds,
    statLabel,
    teamAbbr,
    TEAM_MAP,
    // Alt-line engine exports
    generateFeatureTable,
    generateSlips,
    runAltLinePipeline,
    buildEnvContext,
    MARKET_LINE_KEYS,
    MARKET_LABELS,
  };
})();
