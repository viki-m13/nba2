// =============================================================================
// NBA API - Fetch live game data from NBA.com public endpoints
// =============================================================================

window.NbaApi = (function() {

  const NBA_SCOREBOARD_URL = 'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json';
  const NBA_PLAYBYPLAY_URL = (gameId) => `https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_${gameId}.json`;
  const ESPN_SCOREBOARD_URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard';
  const ESPN_SUMMARY_URL = (eventId) => `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=${eventId}`;

  // Vercel proxy endpoint (auto-detected if deployed on Vercel)
  const VERCEL_PROXY_SCOREBOARD = '/api/nba?endpoint=scoreboard';
  const VERCEL_PROXY_PBP = (gameId) => `/api/nba?endpoint=playbyplay&gameId=${gameId}`;
  const VERCEL_PROXY_ESPN = '/api/nba?endpoint=espn_scoreboard';
  const VERCEL_PROXY_ESPN_SUMMARY = (eventId) => `/api/nba?endpoint=espn_summary&eventId=${eventId}`;

  // CORS proxy (configurable)
  let corsProxy = '';
  let useVercelProxy = false;

  function setCorsProxy(proxy) {
    corsProxy = proxy || '';
  }

  function getCorsProxy() {
    return corsProxy;
  }

  function setUseVercelProxy(val) {
    useVercelProxy = val;
  }

  // Auto-detect if running on Vercel (has /api endpoint)
  async function detectVercelProxy() {
    try {
      const resp = await fetch('/api/nba?endpoint=scoreboard', { method: 'HEAD' });
      if (resp.ok || resp.status === 200) {
        useVercelProxy = true;
        console.log('[NBA API] Vercel proxy detected, using /api/nba');
        return true;
      }
    } catch (e) {
      // Not on Vercel
    }
    return false;
  }

  function getScoreboardUrl() {
    if (useVercelProxy) return VERCEL_PROXY_SCOREBOARD;
    if (corsProxy) return proxyUrl(NBA_SCOREBOARD_URL);
    return NBA_SCOREBOARD_URL;
  }

  function getPlayByPlayUrl(gameId) {
    if (useVercelProxy) return VERCEL_PROXY_PBP(gameId);
    if (corsProxy) return proxyUrl(NBA_PLAYBYPLAY_URL(gameId));
    return NBA_PLAYBYPLAY_URL(gameId);
  }

  function proxyUrl(url) {
    if (!corsProxy) return url;
    // Handle different proxy formats
    if (corsProxy.endsWith('?') || corsProxy.endsWith('=')) {
      return corsProxy + encodeURIComponent(url);
    }
    if (corsProxy.endsWith('/')) {
      return corsProxy + encodeURIComponent(url);
    }
    return corsProxy + '/' + encodeURIComponent(url);
  }

  // =========================================================================
  // NBA TEAM DATA
  // =========================================================================
  const NBA_TEAMS = {
    'ATL': { name: 'Hawks', city: 'Atlanta', color: '#E03A3E' },
    'BOS': { name: 'Celtics', city: 'Boston', color: '#007A33' },
    'BKN': { name: 'Nets', city: 'Brooklyn', color: '#000000' },
    'CHA': { name: 'Hornets', city: 'Charlotte', color: '#1D1160' },
    'CHI': { name: 'Bulls', city: 'Chicago', color: '#CE1141' },
    'CLE': { name: 'Cavaliers', city: 'Cleveland', color: '#860038' },
    'DAL': { name: 'Mavericks', city: 'Dallas', color: '#00538C' },
    'DEN': { name: 'Nuggets', city: 'Denver', color: '#0E2240' },
    'DET': { name: 'Pistons', city: 'Detroit', color: '#C8102E' },
    'GSW': { name: 'Warriors', city: 'Golden State', color: '#1D428A' },
    'HOU': { name: 'Rockets', city: 'Houston', color: '#CE1141' },
    'IND': { name: 'Pacers', city: 'Indiana', color: '#002D62' },
    'LAC': { name: 'Clippers', city: 'LA', color: '#C8102E' },
    'LAL': { name: 'Lakers', city: 'LA', color: '#552583' },
    'MEM': { name: 'Grizzlies', city: 'Memphis', color: '#5D76A9' },
    'MIA': { name: 'Heat', city: 'Miami', color: '#98002E' },
    'MIL': { name: 'Bucks', city: 'Milwaukee', color: '#00471B' },
    'MIN': { name: 'Timberwolves', city: 'Minnesota', color: '#0C2340' },
    'NOP': { name: 'Pelicans', city: 'New Orleans', color: '#0C2340' },
    'NYK': { name: 'Knicks', city: 'New York', color: '#006BB6' },
    'OKC': { name: 'Thunder', city: 'OKC', color: '#007AC1' },
    'ORL': { name: 'Magic', city: 'Orlando', color: '#0077C0' },
    'PHI': { name: '76ers', city: 'Philadelphia', color: '#006BB6' },
    'PHX': { name: 'Suns', city: 'Phoenix', color: '#1D1160' },
    'POR': { name: 'Trail Blazers', city: 'Portland', color: '#E03A3E' },
    'SAC': { name: 'Kings', city: 'Sacramento', color: '#5A2D81' },
    'SAS': { name: 'Spurs', city: 'San Antonio', color: '#C4CED4' },
    'TOR': { name: 'Raptors', city: 'Toronto', color: '#CE1141' },
    'UTA': { name: 'Jazz', city: 'Utah', color: '#002B5C' },
    'WAS': { name: 'Wizards', city: 'Washington', color: '#002B5C' },
  };

  // =========================================================================
  // ESPN TRICODE NORMALIZATION
  // =========================================================================
  // ESPN uses different abbreviations than NBA.com for some teams
  const ESPN_TO_NBA_TRICODE = {
    'GS': 'GSW',
    'NY': 'NYK',
    'NO': 'NOP',
    'SA': 'SAS',
    'WSH': 'WAS',
    'UTAH': 'UTA',
  };

  function normalizeTricode(tricode) {
    return ESPN_TO_NBA_TRICODE[tricode] || tricode;
  }

  // =========================================================================
  // O/U LINE CACHE (localStorage)
  // =========================================================================
  const OU_CACHE_KEY = 'q3ou-ouline-cache';

  function loadOUCache() {
    try {
      const cached = JSON.parse(localStorage.getItem(OU_CACHE_KEY) || '{}');
      // Expire entries older than 24 hours
      const now = Date.now();
      const valid = {};
      for (const [key, entry] of Object.entries(cached)) {
        if (now - entry.ts < 24 * 60 * 60 * 1000) {
          valid[key] = entry;
        }
      }
      return valid;
    } catch (e) {
      return {};
    }
  }

  function saveOUCache(cache) {
    try {
      localStorage.setItem(OU_CACHE_KEY, JSON.stringify(cache));
    } catch (e) {}
  }

  function cacheOULine(teamKey, ouLine) {
    const cache = loadOUCache();
    cache[teamKey] = { ou: ouLine, ts: Date.now() };
    saveOUCache(cache);
  }

  function getCachedOULine(teamKey) {
    const cache = loadOUCache();
    return cache[teamKey]?.ou || null;
  }

  // =========================================================================
  // FETCH ESPN ODDS (O/U lines + event IDs from scoreboard)
  // =========================================================================
  async function fetchESPNOdds() {
    try {
      let url;
      if (useVercelProxy) {
        url = VERCEL_PROXY_ESPN;
      } else if (corsProxy) {
        url = proxyUrl(ESPN_SCOREBOARD_URL);
      } else {
        url = ESPN_SCOREBOARD_URL;
      }

      const response = await fetch(url, {
        headers: { 'Accept': 'application/json' },
      });

      if (!response.ok) {
        console.warn('[NBA API] ESPN odds fetch failed:', response.status);
        return { odds: {}, eventIds: {} };
      }

      const data = await response.json();
      const odds = {};
      const eventIds = {};

      for (const event of (data.events || [])) {
        const comp = event.competitions?.[0];
        if (!comp) continue;

        let homeTeam = '', awayTeam = '';
        for (const c of (comp.competitors || [])) {
          const tricode = normalizeTricode(c.team?.abbreviation || '');
          if (c.homeAway === 'home') homeTeam = tricode;
          else awayTeam = tricode;
        }

        if (!homeTeam || !awayTeam) continue;
        const teamKey = `${homeTeam}-${awayTeam}`;

        // Always store ESPN event ID for summary fallback
        eventIds[teamKey] = event.id;

        // Extract O/U line from odds array (only available for scheduled games)
        const ouLine = comp.odds?.[0]?.overUnder;
        if (ouLine && ouLine > 0) {
          odds[teamKey] = ouLine;
          cacheOULine(teamKey, ouLine);
          console.log(`[ESPN Odds] ${awayTeam} @ ${homeTeam}: O/U ${ouLine}`);
        }
      }

      console.log(`[ESPN Odds] Scoreboard: ${Object.keys(odds).length} O/U lines, ${Object.keys(eventIds).length} event IDs`);
      return { odds, eventIds };
    } catch (error) {
      console.warn('[NBA API] ESPN odds fetch error:', error.message);
      return { odds: {}, eventIds: {} };
    }
  }

  // =========================================================================
  // FETCH ESPN GAME SUMMARY (for live games where scoreboard O/U is missing)
  // =========================================================================
  async function fetchESPNGameOdds(eventId) {
    try {
      let url;
      if (useVercelProxy) {
        url = VERCEL_PROXY_ESPN_SUMMARY(eventId);
      } else if (corsProxy) {
        url = proxyUrl(ESPN_SUMMARY_URL(eventId));
      } else {
        url = ESPN_SUMMARY_URL(eventId);
      }

      const response = await fetch(url, {
        headers: { 'Accept': 'application/json' },
      });

      if (!response.ok) {
        console.warn(`[ESPN Summary] Fetch failed for event ${eventId}:`, response.status);
        return null;
      }

      const data = await response.json();
      const pickcenter = data.pickcenter || [];

      for (const pc of pickcenter) {
        const ou = pc.overUnder ?? pc.total?.alternateDisplayValue;
        if (ou && parseFloat(ou) > 0) {
          return parseFloat(ou);
        }
      }

      return null;
    } catch (error) {
      console.warn(`[ESPN Summary] Error for event ${eventId}:`, error.message);
      return null;
    }
  }

  // =========================================================================
  // FETCH SCOREBOARD
  // =========================================================================
  async function fetchScoreboard() {
    try {
      const url = getScoreboardUrl();
      const response = await fetch(url, {
        headers: { 'Accept': 'application/json' },
      });

      if (!response.ok) {
        console.warn('[NBA API] Scoreboard fetch failed:', response.status);
        return { games: [], error: null };
      }

      const data = await response.json();
      const rawGames = data.scoreboard?.games || [];

      const games = rawGames.map(g => {
        const homeTricode = g.homeTeam.teamTricode;
        const awayTricode = g.awayTeam.teamTricode;
        const homeInfo = NBA_TEAMS[homeTricode] || { name: homeTricode, city: '', color: '#333' };
        const awayInfo = NBA_TEAMS[awayTricode] || { name: awayTricode, city: '', color: '#333' };

        // Parse game clock
        let clockDisplay = g.gameStatusText || '';
        let quarter = g.period || 0;
        let gameClock = g.gameClock || '';

        // Parse PT format clock
        let clockMins = 0, clockSecs = 0;
        const clockMatch = gameClock.match(/PT(\d+)M([\d.]+)S/);
        if (clockMatch) {
          clockMins = parseInt(clockMatch[1]);
          clockSecs = Math.floor(parseFloat(clockMatch[2]));
        }

        // Status mapping
        let status = 'scheduled';
        if (g.gameStatus === 2) {
          status = 'live';
          if (quarter === 2 && clockMins === 0 && clockSecs === 0) status = 'halftime';
        } else if (g.gameStatus === 3) {
          status = 'final';
        }

        return {
          id: g.gameId,
          homeTeam: homeTricode,
          awayTeam: awayTricode,
          homeName: homeInfo.name,
          awayName: awayInfo.name,
          homeColor: homeInfo.color,
          awayColor: awayInfo.color,
          homeScore: g.homeTeam.score || 0,
          awayScore: g.awayTeam.score || 0,
          quarter,
          gameClock: `${clockMins}:${clockSecs.toString().padStart(2, '0')}`,
          status,
          statusText: clockDisplay,
          gameTime: g.gameEt || '',
          possessions: [], // Will be populated by play-by-play
        };
      });

      return { games, error: null };
    } catch (error) {
      console.warn('[NBA API] Error fetching scoreboard:', error.message);
      return { games: [], error: error.message };
    }
  }

  // =========================================================================
  // FETCH PLAY-BY-PLAY
  // =========================================================================
  async function fetchPlayByPlay(gameId) {
    try {
      const url = getPlayByPlayUrl(gameId);
      const response = await fetch(url, {
        headers: { 'Accept': 'application/json' },
      });

      if (!response.ok) {
        return [];
      }

      const data = await response.json();
      const actions = data.game?.actions || [];

      if (actions.length === 0) return [];

      const possessions = [];
      let lastHomeScore = 0;
      let lastAwayScore = 0;

      for (const action of actions) {
        const homeScore = parseInt(action.scoreHome) || 0;
        const awayScore = parseInt(action.scoreAway) || 0;

        const scoreChanged = homeScore !== lastHomeScore || awayScore !== lastAwayScore;
        const isNewPeriod = possessions.length === 0 ||
          (possessions.length > 0 && action.period !== possessions[possessions.length - 1].quarter);

        if (scoreChanged || isNewPeriod) {
          let quarterTime = '12:00';
          const clockMatch = action.clock?.match(/PT(\d+)M([\d.]+)S/);
          if (clockMatch) {
            const minutes = clockMatch[1];
            const seconds = Math.floor(parseFloat(clockMatch[2]));
            quarterTime = `${minutes}:${seconds.toString().padStart(2, '0')}`;
          }

          const periodSeconds = 12 * 60;
          const clockParts = quarterTime.split(':');
          const minutesRemaining = parseInt(clockParts[0]) || 0;
          const secondsRemaining = parseInt(clockParts[1]) || 0;
          const timeElapsedInPeriod = periodSeconds - (minutesRemaining * 60 + secondsRemaining);
          const timestamp = (action.period - 1) * periodSeconds + timeElapsedInPeriod;

          possessions.push({
            id: `pos-${action.actionNumber}`,
            timestamp,
            quarter: action.period,
            quarterTime,
            team: homeScore > lastHomeScore ? 'home' : 'away',
            homeScore,
            awayScore,
            differential: homeScore - awayScore,
            fairDifferential: homeScore - awayScore,
            event: 'score',
          });

          lastHomeScore = homeScore;
          lastAwayScore = awayScore;
        }
      }

      return possessions;
    } catch (error) {
      console.warn(`[NBA API] Error fetching play-by-play for ${gameId}:`, error.message);
      return [];
    }
  }

  // =========================================================================
  // FETCH FULL GAME DATA (scoreboard + play-by-play for live games)
  // =========================================================================
  async function fetchAllGames() {
    const { games, error } = await fetchScoreboard();

    if (error || games.length === 0) {
      return { games, error };
    }

    // Fetch play-by-play for all live games in parallel
    const liveGames = games.filter(g => g.status === 'live' || g.status === 'halftime');

    const pbpPromises = liveGames.map(async (game) => {
      const possessions = await fetchPlayByPlay(game.id);
      game.possessions = possessions;

      // Add team info to possessions
      possessions.forEach(p => {
        p.homeTeam = game.homeTeam;
        p.awayTeam = game.awayTeam;
      });
    });

    // Fetch ESPN odds (O/U lines + event IDs) in parallel with play-by-play
    const [, espnData] = await Promise.all([
      Promise.all(pbpPromises),
      fetchESPNOdds(),
    ]);

    const { odds: espnOdds, eventIds: espnEventIds } = espnData;

    // Phase 1: Merge O/U lines from ESPN scoreboard
    for (const game of games) {
      const key = `${game.homeTeam}-${game.awayTeam}`;
      if (espnOdds[key]) {
        game.ouLine = espnOdds[key];
      }
    }

    // Phase 2: For live games still missing O/U, try localStorage cache
    for (const game of liveGames) {
      if (game.ouLine) continue;
      const key = `${game.homeTeam}-${game.awayTeam}`;
      const cached = getCachedOULine(key);
      if (cached) {
        game.ouLine = cached;
        console.log(`[O/U] ${game.awayTeam} @ ${game.homeTeam}: using cached O/U ${cached}`);
      }
    }

    // Phase 3: For live games STILL missing O/U, fetch ESPN game summary
    const summaryPromises = liveGames
      .filter(g => !g.ouLine)
      .map(async (game) => {
        const key = `${game.homeTeam}-${game.awayTeam}`;
        const eventId = espnEventIds[key];
        if (!eventId) {
          console.warn(`[O/U] ${game.awayTeam} @ ${game.homeTeam}: no ESPN event ID, cannot fetch summary`);
          return;
        }
        console.log(`[O/U] ${game.awayTeam} @ ${game.homeTeam}: fetching ESPN summary (event ${eventId})...`);
        const ou = await fetchESPNGameOdds(eventId);
        if (ou) {
          game.ouLine = ou;
          cacheOULine(key, ou);
          console.log(`[O/U] ${game.awayTeam} @ ${game.homeTeam}: got O/U ${ou} from ESPN summary`);
        } else {
          console.warn(`[O/U] ${game.awayTeam} @ ${game.homeTeam}: ESPN summary had no O/U line`);
        }
      });

    await Promise.all(summaryPromises);

    const gamesWithOU = games.filter(g => g.ouLine > 0).length;
    const liveCount = liveGames.length;
    if (liveCount > 0) {
      console.log(`[NBA API] ${gamesWithOU}/${games.length} games have O/U lines, ${liveCount} live`);
    }

    return { games, error: null };
  }

  // =========================================================================
  // PUBLIC API
  // =========================================================================
  return {
    NBA_TEAMS,
    setCorsProxy,
    getCorsProxy,
    setUseVercelProxy,
    detectVercelProxy,
    fetchScoreboard,
    fetchPlayByPlay,
    fetchESPNOdds,
    fetchESPNGameOdds,
    fetchAllGames,
  };

})();
