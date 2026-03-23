/**
 * KDT Live Engine — Client-Side Kinetic Divergence Theory Monitor
 * =================================================================
 * Polls ESPN for live MLB games, computes Kinetic Divergence (KD),
 * detects Intra-Inning Divergence Spike (IDS) entries, and fires
 * real-time alerts when KDT tiers trigger.
 *
 * This runs entirely in the browser — no server needed beyond the
 * ESPN API proxy (or direct ESPN API calls).
 */
(function () {
  'use strict';

  // ── Configuration ──────────────────────────────────────────────
  const POLL_INTERVAL_MS = 30000; // 30 seconds
  const ESPN_BASE = '/api/mlb';   // Vercel proxy
  const ESPN_DIRECT = 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb';
  const SWP_DATA_URL = '/live-alerts/data/kdt_swp_table.json';

  // ── KDT Alert Tiers ────────────────────────────────────────────
  const KDT_TIERS = [
    {
      id: 'APEX',
      name: 'APEX — Structural Lock',
      minLead: 3,
      minKd: 0.10,
      description: 'Leading by 3+ runs with 10%+ kinetic divergence',
      color: '#f59e0b',
      priority: 1,
    },
    {
      id: 'WAVE',
      name: 'WAVE — Rally Survivor',
      minLead: 2,
      minKd: 0.10,
      description: 'Leading by 2+ runs with 10%+ kinetic divergence',
      color: '#8b5cf6',
      priority: 2,
    },
    {
      id: 'DRIFT',
      name: 'DRIFT — Deep Divergence',
      minLead: 1,
      minKd: 0.10,
      description: 'Leading by 1+ runs with 10%+ kinetic divergence',
      color: '#06b6d4',
      priority: 3,
    },
  ];

  // ── State ──────────────────────────────────────────────────────
  let swpTable = null;
  let pollTimer = null;
  let isRunning = false;

  // Per-game tracking: gameId -> { home: [...plays], away: [...plays] }
  const gamePlayHistory = {};

  // Active alerts: key -> alert object
  const activeAlerts = {};

  // All alerts fired this session
  const alertHistory = [];

  // Resolved bets (game finished)
  const resolvedBets = [];

  // Callbacks
  let onUpdate = null;  // Called after each poll cycle
  let onAlert = null;   // Called when new alert fires

  // ── SWP Table ──────────────────────────────────────────────────

  function lookupSwp(inning, lead) {
    if (!swpTable) return null;
    const key = `${inning},${lead}`;
    if (swpTable[key]) return swpTable[key].win_rate;
    // Fallback: clamp lead to max 8
    const clamped = Math.min(lead, 8);
    const clampedKey = `${inning},${clamped}`;
    if (swpTable[clampedKey]) return swpTable[clampedKey].win_rate;
    return null;
  }

  function computeKd(swpValue, kineticWp) {
    return swpValue - kineticWp;
  }

  // ── ESPN API ───────────────────────────────────────────────────

  async function fetchJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  }

  async function fetchScoreboard() {
    // Try Vercel proxy first, fall back to direct ESPN
    try {
      return await fetchJson(`${ESPN_BASE}?endpoint=scoreboard`);
    } catch (e) {
      return await fetchJson(`${ESPN_DIRECT}/scoreboard`);
    }
  }

  async function fetchGameSummary(eventId) {
    try {
      return await fetchJson(`${ESPN_BASE}?endpoint=summary&eventId=${eventId}`);
    } catch (e) {
      return await fetchJson(`${ESPN_DIRECT}/summary?event=${eventId}`);
    }
  }

  // ── Game Parsing ───────────────────────────────────────────────

  function parseScoreboard(data) {
    const events = data.events || [];
    const games = [];

    for (const event of events) {
      const comp = (event.competitions || [])[0];
      if (!comp) continue;

      const status = comp.status || {};
      const statusType = status.type || {};
      const state = statusType.state || '';

      let home = null, away = null;
      for (const c of comp.competitors || []) {
        if (c.homeAway === 'home') home = c;
        else if (c.homeAway === 'away') away = c;
      }
      if (!home || !away) continue;

      games.push({
        eventId: event.id || '',
        shortName: event.shortName || '',
        state,
        statusDetail: statusType.shortDetail || '',
        period: status.period || 0,
        periodPrefix: statusType.shortDetail || '',
        homeTeam: (home.team || {}).abbreviation || '???',
        awayTeam: (away.team || {}).abbreviation || '???',
        homeFullName: (home.team || {}).displayName || '',
        awayFullName: (away.team || {}).displayName || '',
        homeLogo: (home.team || {}).logo || '',
        awayLogo: (away.team || {}).logo || '',
        homeScore: parseInt(home.score || '0', 10),
        awayScore: parseInt(away.score || '0', 10),
      });
    }

    return games;
  }

  function parseWinProbability(summaryData) {
    const wp = summaryData.winprobability || [];
    if (wp.length > 0) {
      const latest = wp[wp.length - 1];
      return {
        homeWP: latest.homeWinPercentage || 0.5,
        awayWP: 1 - (latest.homeWinPercentage || 0.5),
        source: 'winprobability',
        timeline: wp,
      };
    }

    // Fallback to predictor
    const predictor = summaryData.predictor || {};
    const homeProj = (predictor.homeTeam || {}).gameProjection;
    if (homeProj != null) {
      return {
        homeWP: homeProj / 100,
        awayWP: 1 - homeProj / 100,
        source: 'predictor',
        timeline: [],
      };
    }

    return null;
  }

  function parseCurrentInning(summaryData) {
    const header = summaryData.header || {};
    const comp = (header.competitions || [])[0];
    if (!comp) return { inning: 0, half: '' };
    const status = comp.status || {};
    return {
      inning: status.period || 0,
      half: (status.periodPrefix || '').toLowerCase(),
    };
  }

  // ── IDS Entry Detection ────────────────────────────────────────

  function evaluateIdsEntry(playHist, currentInning, currentLead, currentKwp) {
    if (!playHist || playHist.length < 2) return null;
    if (currentLead < 1) return null;

    const prevPlay = playHist[playHist.length - 1];
    const prevInning = prevPlay.inning;

    // Only trigger at inning transitions
    if (currentInning <= prevInning) return null;

    const swpVal = lookupSwp(currentInning, currentLead);
    if (swpVal == null) return null;

    const currentKd = Math.max(0, computeKd(swpVal, currentKwp));

    // Get KD values from the previous inning
    const prevInningKds = playHist
      .filter(h => h.inning === prevInning && h.kd > 0)
      .map(h => h.kd);

    if (prevInningKds.length === 0) return null;

    const peakKd = Math.max(...prevInningKds);

    // Find best tier match
    for (const tier of KDT_TIERS) {
      if (currentLead >= tier.minLead && peakKd >= tier.minKd) {
        return {
          tier,
          kd: currentKd,
          peakKd,
          swp: swpVal,
          buyPrice: currentKwp,
          inning: currentInning,
          lead: currentLead,
        };
      }
    }

    return null;
  }

  // ── Also check simple KD threshold (non-IDS) for display ──────

  function evaluateCurrentKd(inning, lead, kwp) {
    if (inning < 1 || lead < 1) return null;
    const swpVal = lookupSwp(inning, lead);
    if (swpVal == null) return null;
    const kd = Math.max(0, computeKd(swpVal, kwp));
    return { swp: swpVal, kd, inning, lead };
  }

  // ── Poll Cycle ─────────────────────────────────────────────────

  async function pollCycle() {
    if (!swpTable) return { games: [], error: 'SWP table not loaded' };

    let scoreboard;
    try {
      scoreboard = await fetchScoreboard();
    } catch (e) {
      return { games: [], error: `Scoreboard fetch failed: ${e.message}` };
    }

    const allGames = parseScoreboard(scoreboard);

    // Separate games by state
    const liveGames = allGames.filter(g => g.state === 'in');
    const preGames = allGames.filter(g => g.state === 'pre');
    const postGames = allGames.filter(g => g.state === 'post');

    // Fetch win probability for all live games in parallel
    const liveResults = await Promise.allSettled(
      liveGames.map(async (game) => {
        let summary;
        try {
          summary = await fetchGameSummary(game.eventId);
        } catch (e) {
          return { ...game, wpError: e.message };
        }

        const wpData = parseWinProbability(summary);
        const inningInfo = parseCurrentInning(summary);

        if (!wpData) {
          return { ...game, wpError: 'No WP data' };
        }

        const homeWP = wpData.homeWP;
        const awayWP = wpData.awayWP;
        const inning = inningInfo.inning || game.period;
        const half = inningInfo.half;

        // Compute KD for both sides
        const homeLead = game.homeScore - game.awayScore;
        const awayLead = game.awayScore - game.homeScore;

        const homeKdInfo = evaluateCurrentKd(inning, homeLead, homeWP);
        const awayKdInfo = evaluateCurrentKd(inning, awayLead, awayWP);

        // Track play history for IDS detection
        if (!gamePlayHistory[game.eventId]) {
          gamePlayHistory[game.eventId] = { home: [], away: [] };
        }
        const hist = gamePlayHistory[game.eventId];

        // Add current state to history (dedup by inning)
        const homeState = {
          inning,
          lead: homeLead,
          kwp: homeWP,
          kd: homeKdInfo ? homeKdInfo.kd : 0,
        };
        const awayState = {
          inning,
          lead: awayLead,
          kwp: awayWP,
          kd: awayKdInfo ? awayKdInfo.kd : 0,
        };

        // Check IDS entries BEFORE adding current state
        const homeIds = evaluateIdsEntry(hist.home, inning, homeLead, homeWP);
        const awayIds = evaluateIdsEntry(hist.away, inning, awayLead, awayWP);

        // Now add current state to history
        hist.home.push(homeState);
        hist.away.push(awayState);

        // Process IDS alerts
        const newAlerts = [];

        if (homeIds) {
          const alertKey = `${game.eventId}_${game.homeTeam}_${homeIds.tier.id}`;
          if (!activeAlerts[alertKey]) {
            const alert = {
              key: alertKey,
              eventId: game.eventId,
              tier: homeIds.tier,
              team: game.homeTeam,
              opponent: game.awayTeam,
              side: 'home',
              buyPrice: homeIds.buyPrice,
              swp: homeIds.swp,
              kd: homeIds.kd,
              peakKd: homeIds.peakKd,
              inning: homeIds.inning,
              lead: homeIds.lead,
              score: `${game.awayTeam} ${game.awayScore} @ ${game.homeTeam} ${game.homeScore}`,
              profitIfWin: +(1 - homeIds.buyPrice).toFixed(4),
              timestamp: new Date().toISOString(),
            };
            activeAlerts[alertKey] = alert;
            alertHistory.push(alert);
            newAlerts.push(alert);
          }
        }

        if (awayIds) {
          const alertKey = `${game.eventId}_${game.awayTeam}_${awayIds.tier.id}`;
          if (!activeAlerts[alertKey]) {
            const alert = {
              key: alertKey,
              eventId: game.eventId,
              tier: awayIds.tier,
              team: game.awayTeam,
              opponent: game.homeTeam,
              side: 'away',
              buyPrice: awayIds.buyPrice,
              swp: awayIds.swp,
              kd: awayIds.kd,
              peakKd: awayIds.peakKd,
              inning: awayIds.inning,
              lead: awayIds.lead,
              score: `${game.awayTeam} ${game.awayScore} @ ${game.homeTeam} ${game.homeScore}`,
              profitIfWin: +(1 - awayIds.buyPrice).toFixed(4),
              timestamp: new Date().toISOString(),
            };
            activeAlerts[alertKey] = alert;
            alertHistory.push(alert);
            newAlerts.push(alert);
          }
        }

        return {
          ...game,
          homeWP,
          awayWP,
          wpSource: wpData.source,
          inning,
          half,
          homeKd: homeKdInfo,
          awayKd: awayKdInfo,
          newAlerts,
        };
      })
    );

    const enrichedLive = liveResults
      .filter(r => r.status === 'fulfilled')
      .map(r => r.value);

    // Resolve finished games
    for (const game of postGames) {
      resolveFinishedGame(game);
    }

    const result = {
      live: enrichedLive,
      pre: preGames,
      post: postGames,
      activeAlerts: Object.values(activeAlerts),
      alertHistory,
      resolvedBets,
      lastPoll: new Date().toISOString(),
      error: null,
    };

    if (onUpdate) onUpdate(result);

    // Fire new alert callbacks
    for (const game of enrichedLive) {
      if (game.newAlerts) {
        for (const a of game.newAlerts) {
          if (onAlert) onAlert(a);
        }
      }
    }

    return result;
  }

  function resolveFinishedGame(game) {
    // Check if any active alerts are for this game
    const keysToResolve = Object.keys(activeAlerts).filter(
      k => k.startsWith(game.eventId + '_')
    );

    for (const key of keysToResolve) {
      const alert = activeAlerts[key];
      const won = (alert.side === 'home' && game.homeScore > game.awayScore) ||
                  (alert.side === 'away' && game.awayScore > game.homeScore);

      resolvedBets.push({
        ...alert,
        won,
        finalScore: `${game.awayTeam} ${game.awayScore} @ ${game.homeTeam} ${game.homeScore}`,
        pnl: won ? +((1 - alert.buyPrice) * 100).toFixed(2) : +((-alert.buyPrice) * 100).toFixed(2),
        resolvedAt: new Date().toISOString(),
      });

      delete activeAlerts[key];
    }
  }

  // ── Public API ─────────────────────────────────────────────────

  async function init() {
    // Load SWP table
    try {
      console.log('[KDT] Loading SWP table from', SWP_DATA_URL);
      swpTable = await fetchJson(SWP_DATA_URL);
      const keys = Object.keys(swpTable);
      console.log('[KDT] SWP table loaded:', keys.length, 'states');
      return keys.length > 0;
    } catch (e) {
      console.error('[KDT] Failed to load SWP table:', e);
      return false;
    }
  }

  function start(updateCb, alertCb) {
    if (isRunning) return;
    isRunning = true;
    onUpdate = updateCb || null;
    onAlert = alertCb || null;

    // Immediate first poll
    pollCycle().catch(e => console.error('[KDT] Poll error:', e));

    // Schedule recurring polls
    pollTimer = setInterval(() => {
      pollCycle().catch(e => console.error('[KDT] Poll error:', e));
    }, POLL_INTERVAL_MS);
  }

  function stop() {
    isRunning = false;
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function getState() {
    return {
      isRunning,
      swpLoaded: !!swpTable,
      activeAlerts: Object.values(activeAlerts),
      alertHistory,
      resolvedBets,
      gamePlayHistory,
    };
  }

  function getTiers() {
    return KDT_TIERS;
  }

  // Force a single poll (for manual refresh)
  async function poll() {
    return pollCycle();
  }

  // Expose public API
  window.KDTLiveEngine = {
    init,
    start,
    stop,
    poll,
    getState,
    getTiers,
    lookupSwp,
    computeKd,
  };
})();
