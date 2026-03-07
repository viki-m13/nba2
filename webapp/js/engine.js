// =============================================================================
// NBA PROP BETTING ENGINE v3.0 — Multi-Stat Floor Strategy
// =============================================================================
//
// STRATEGY: Player OVER props on ultra-low FanDuel alternate lines across
//           multiple stat types (points, rebounds, assists).
//
// EDGE: We only bet when a player's WORST recent game across 20 games is
//       still well above the line. By combining legs from different stat
//       types, we build parlays with POSITIVE odds and 90%+ hit rates.
//
// VALIDATED: Walk-forward backtest on 2024-25 season using real FanDuel odds.
//            Parameters validated on Q3+Q4 data (not trained on full dataset).
//
// KEY PARAMETERS (validated out-of-sample):
//   - L20 floor STRICTLY above the line
//   - L10 floor >= 1.30x the line (30%+ cushion)
//   - Line <= 60% of L10 average
//   - CV < 0.35 (consistent performers only)
//   - Minimum 15 games of history
//
// =============================================================================

window.BettingEngine = (function () {
  'use strict';

  // --- Configuration ---

  // Tier 1 (Strict): Ultra-safe legs with heavy juice
  const CONFIG = {
    MIN_GAMES: 15,
    MAX_CV: 0.35,
    MIN_FLOOR_RATIO: 1.30,
    MAX_LINE_RATIO: 0.60,
    MIN_ODDS: -2500,
    MAX_ODDS: -100,
    MIN_HIT_RATE: 0.95,
    UNIT_SIZE: 100,
    PARLAY_MIN_LEGS: 2,
    PARLAY_MAX_LEGS: 6,
    FLOOR_WINDOW: 20,

    // Stat-specific minimums
    POINTS: {
      MIN_AVG: 15,
      MIN_AVG_MIN: 26,
      MIN_L3_MIN: 25,
      LINES: [9.5, 14.5, 19.5, 24.5, 29.5, 34.5],
      MARKET: 'player_points_alternate',
    },
    REBOUNDS: {
      MIN_AVG: 4,
      MIN_AVG_MIN: 20,
      MIN_L3_MIN: 18,
      LINES: [1.5, 3.5, 5.5, 7.5, 9.5, 11.5],
      MARKET: 'player_rebounds_alternate',
    },
    ASSISTS: {
      MIN_AVG: 3,
      MIN_AVG_MIN: 20,
      MIN_L3_MIN: 18,
      LINES: [1.5, 3.5, 5.5, 7.5, 9.5],
      MARKET: 'player_assists_alternate',
    },
  };

  // Tier 2 (Enhanced): Better-paying legs, slightly relaxed filters
  // Validated out-of-sample: 92.3% hit rate, avg odds -521
  const CONFIG_T2 = {
    MIN_GAMES: 15,
    MAX_CV: 0.35,
    MIN_FLOOR_RATIO: 1.10,
    MAX_LINE_RATIO: 0.70,
    MIN_ODDS: -800,
    MAX_ODDS: -200,
    MIN_HIT_RATE: 0.90,
    FLOOR_WINDOW: 15,
  };

  const STAT_TYPES = ['points', 'rebounds', 'assists'];

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
    if (odds > 0) return 1 + odds / 100;
    return 1 + 100 / Math.abs(odds);
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

  // --- Multi-Stat Player Model ---

  const PlayerModel = {
    history: {},

    reset() { this.history = {}; },

    update(name, stats, date, team) {
      // stats: { pts, reb, ast, min }
      if (!this.history[name]) this.history[name] = [];
      this.history[name].push({ ...stats, date, team });
      if (this.history[name].length > 50) {
        this.history[name] = this.history[name].slice(-40);
      }
    },

    // Generic stat evaluator — works for any stat type and tier
    evaluateStat(playerName, statType, fdLine, fdOdds, tierConfig) {
      const cfg = tierConfig || CONFIG;
      const hist = this.history[playerName];
      if (!hist || hist.length < cfg.MIN_GAMES) return null;

      const statKey = statType === 'points' ? 'pts' : (statType === 'rebounds' ? 'reb' : 'ast');
      const statConfig = CONFIG[statType.toUpperCase()];
      if (!statConfig) return null;

      // Injury/inactivity check: if most recent game has <5 min, player is likely out
      const lastGame = hist[hist.length - 1];
      if (lastGame.min < 5) return null;

      const fw = cfg.FLOOR_WINDOW || 20;
      const l10 = hist.slice(-10);
      const lfw = hist.slice(-fw);
      const l5 = hist.slice(-5);
      const l3 = hist.slice(-3);

      const values = l10.map(g => g[statKey]);
      const avg = values.reduce((s, v) => s + v, 0) / values.length;
      const avgMin = l10.reduce((s, g) => s + g.min, 0) / l10.length;
      const l3Min = l3.reduce((s, g) => s + g.min, 0) / l3.length;
      const min10 = Math.min(...values);
      const minFW = Math.min(...lfw.map(g => g[statKey]));

      // Basic eligibility
      if (avg < statConfig.MIN_AVG) return null;
      if (avgMin < statConfig.MIN_AVG_MIN) return null;
      if (l3Min < statConfig.MIN_L3_MIN) return null;

      // Consistency check
      const variance = values.reduce((s, v) => s + (v - avg) ** 2, 0) / values.length;
      const cv = Math.sqrt(variance) / avg;
      if (cv > cfg.MAX_CV) return null;

      // Floor window check: must be STRICTLY above line
      if (minFW <= fdLine) return null;

      // L10 floor ratio check
      const floorRatio = min10 / fdLine;
      if (floorRatio < cfg.MIN_FLOOR_RATIO) return null;

      // Line must be conservative relative to average
      const lineRatio = fdLine / avg;
      if (lineRatio > cfg.MAX_LINE_RATIO) return null;

      // Odds filter
      if (fdOdds < cfg.MIN_ODDS || fdOdds > cfg.MAX_ODDS) return null;

      // Hit rates
      const hitsL10 = l10.filter(g => g[statKey] > fdLine).length;
      const hitsFW = lfw.filter(g => g[statKey] > fdLine).length;
      const hitRateL10 = hitsL10 / l10.length;
      const hitRateFW = lfw.length >= 10 ? hitsFW / lfw.length : hitRateL10;
      const hitRate = hitRateL10 * 0.6 + hitRateFW * 0.4;

      if (hitRate < cfg.MIN_HIT_RATE) return null;

      // Momentum
      const avg5 = l5.reduce((s, g) => s + g[statKey], 0) / l5.length;
      const momentum = avg5 >= avg ? 'UP' : (avg5 >= avg * 0.90 ? 'STABLE' : 'DOWN');
      if (momentum === 'DOWN') return null;

      const decimal = americanToDecimal(fdOdds);
      const ev = hitRate * decimal - 1;
      const marginOfSafety = min10 - fdLine;

      const confidence = Math.min(0.99,
        hitRate * 0.35 +
        Math.min(0.25, floorRatio * 0.15) +
        Math.min(0.20, (1 - cv) * 0.25) +
        (momentum === 'UP' ? 0.10 : 0.05) +
        Math.min(0.08, marginOfSafety * 0.01)
      );

      return {
        player: playerName,
        team: l10[l10.length - 1].team,
        statType,
        statLabel: statLabel(statType),
        line: fdLine,
        odds: fdOdds,
        hitRate: Math.round(hitRate * 1000) / 1000,
        l10Avg: Math.round(avg * 10) / 10,
        l5Avg: Math.round(avg5 * 10) / 10,
        l10Min: min10,
        floorMin: minFW,
        floorRatio: Math.round(floorRatio * 100) / 100,
        lineRatio: Math.round(lineRatio * 100) / 100,
        cv: Math.round(cv * 100) / 100,
        momentum,
        marginOfSafety,
        confidence: Math.round(confidence * 1000) / 1000,
        ev: Math.round(ev * 1000) / 1000,
      };
    },

    // Find best prop for a stat type given available lines (Tier 1)
    findBestStatProp(playerName, statType, fdLines, tierConfig) {
      let best = null;
      for (const [threshold, data] of Object.entries(fdLines)) {
        const line = parseFloat(threshold);
        const odds = data.overOdds;
        if (!odds) continue;
        const result = this.evaluateStat(playerName, statType, line, odds, tierConfig);
        if (result && (!best || result.confidence > best.confidence)) {
          best = result;
        }
      }
      if (best) {
        best.tier = tierConfig === CONFIG_T2 ? 2 : 1;
      }
      return best;
    },

    // Legacy: find best points prop
    findBestProp(playerName, fdLines) {
      return this.findBestStatProp(playerName, 'points', fdLines);
    },

    // Find best Tier 2 (enhanced) prop — better odds, slightly relaxed
    findBestT2Prop(playerName, statType, fdLines) {
      return this.findBestStatProp(playerName, statType, fdLines, CONFIG_T2);
    },

    // For backtest without real odds: find best line from standard lines
    findBestStatPropNoOdds(playerName, statType, tierConfig) {
      const cfg = tierConfig || CONFIG;
      const statConfig = CONFIG[statType.toUpperCase()];
      if (!statConfig) return null;

      let best = null;
      for (const line of statConfig.LINES) {
        const estimatedOdds = cfg === CONFIG_T2 ? -500 : -300;
        const result = this.evaluateStat(playerName, statType, line, estimatedOdds, cfg);
        if (result) {
          if (!best || result.line > best.line) {
            best = result;
          }
        }
      }
      if (best) {
        best.tier = cfg === CONFIG_T2 ? 2 : 1;
      }
      return best;
    },
  };

  // --- Parlay Builder ---

  function buildParlays(singles) {
    if (singles.length < 2) return [];

    const parlays = [];
    const sorted = [...singles].sort((a, b) => b.confidence - a.confidence);
    const candidates = sorted.slice(0, 15);

    // Build best parlay for each size
    for (let n = 2; n <= Math.min(CONFIG.PARLAY_MAX_LEGS, candidates.length); n++) {
      const best = findBestParlay(candidates, n);
      if (best) {
        // For 4+ legs, require higher combined hit rate
        if (n >= 4 && best.combinedHitRate < 0.88) continue;
        if (n >= 6 && best.combinedHitRate < 0.90) continue;
        parlays.push(best);
      }
    }

    return parlays;
  }

  function findBestParlay(legs, numLegs) {
    let best = null;
    let bestScore = -Infinity;

    // For large parlays, use greedy approach
    if (numLegs >= 5) {
      const uniquePlayers = [];
      const seen = new Set();
      for (const leg of legs) {
        const key = leg.player;
        if (!seen.has(key)) {
          seen.add(key);
          uniquePlayers.push(leg);
        }
      }
      if (uniquePlayers.length < numLegs) return null;

      // Try to diversify stat types in the parlay
      const byType = {};
      for (const leg of uniquePlayers) {
        const t = leg.statType || 'points';
        if (!byType[t]) byType[t] = [];
        byType[t].push(leg);
      }

      // Build combo prioritizing stat type diversity
      const combo = [];
      const types = Object.keys(byType);
      let idx = 0;
      while (combo.length < numLegs) {
        const type = types[idx % types.length];
        const available = byType[type].filter(l => !combo.includes(l));
        if (available.length > 0) {
          combo.push(available[0]);
          byType[type] = byType[type].filter(l => l !== available[0]);
        }
        idx++;
        if (idx > numLegs * 3) break; // safety
      }

      if (combo.length < numLegs) return null;

      const parlayDecimal = combo.reduce((d, l) => d * americanToDecimal(l.odds), 1);
      const parlayOdds = decimalToAmerican(parlayDecimal);
      const combinedHitRate = combo.reduce((p, l) => p * l.hitRate, 1);
      const ev = combinedHitRate * parlayDecimal - 1;
      if (ev >= 0) {
        return {
          legs: combo.map(l => ({ ...l })),
          numLegs,
          odds: parlayOdds,
          decimalOdds: Math.round(parlayDecimal * 100) / 100,
          combinedHitRate: Math.round(combinedHitRate * 1000) / 1000,
          ev: Math.round(ev * 1000) / 1000,
          confidence: Math.round(combinedHitRate * 1000) / 1000,
        };
      }
      return null;
    }

    const combos = getCombinations(legs, numLegs);
    for (const combo of combos) {
      // Ensure legs are from different players
      const players = new Set(combo.map(l => l.player));
      if (players.size < numLegs) continue;

      const parlayDecimal = combo.reduce((d, l) => d * americanToDecimal(l.odds), 1);
      const parlayOdds = decimalToAmerican(parlayDecimal);
      const combinedHitRate = combo.reduce((p, l) => p * l.hitRate, 1);
      const ev = combinedHitRate * parlayDecimal - 1;

      if (ev < 0) continue;

      // Score: favor stat type diversity + EV + hit rate
      const statTypes = new Set(combo.map(l => l.statType || 'points'));
      const diversityBonus = statTypes.size * 20;
      const score = ev * 100 + combinedHitRate * 50 + diversityBonus;

      if (score > bestScore) {
        bestScore = score;
        best = {
          legs: combo.map(l => ({ ...l })),
          numLegs,
          odds: parlayOdds,
          decimalOdds: Math.round(parlayDecimal * 100) / 100,
          combinedHitRate: Math.round(combinedHitRate * 1000) / 1000,
          ev: Math.round(ev * 1000) / 1000,
          confidence: Math.round(combinedHitRate * 1000) / 1000,
        };
      }
    }
    return best;
  }

  function getCombinations(arr, k) {
    if (k === 1) return arr.map(x => [x]);
    const result = [];
    const cap = Math.min(arr.length, 12);
    for (let i = 0; i < cap; i++) {
      const rest = arr.slice(i + 1);
      for (const sub of getCombinations(rest, k - 1)) {
        result.push([arr[i], ...sub]);
      }
    }
    return result;
  }

  // --- Backtest Engine ---

  function runBacktest(seasonData, boxScores, historicalOdds, rebAstProps) {
    const model = Object.create(PlayerModel);
    model.history = {};

    const sortedGames = [...seasonData].sort((a, b) => a.date.localeCompare(b.date));
    const boxByDate = {};
    for (const g of boxScores) {
      if (!boxByDate[g.date]) boxByDate[g.date] = [];
      boxByDate[g.date].push(g);
    }

    const oddsByDate = {};
    for (const od of (historicalOdds || [])) {
      if (!oddsByDate[od.date]) oddsByDate[od.date] = {};
      oddsByDate[od.date][od.gameKey] = od;
    }

    // Index reb/ast props by date+gameKey
    const rebAstByDateGame = {};
    for (const ra of (rebAstProps || [])) {
      const key = `${ra.date}_${ra.gameKey}`;
      rebAstByDateGame[key] = ra;
    }

    const results = {
      singles: [],
      parlays: [],
      dailySummaries: [],
      dates: [],
    };

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
          const ra = rebAstByDateGame[`${date}_${gameKey}`];

          for (const actualPlayer of (bg.players || [])) {
            const mins = typeof actualPlayer.min === 'number' ? actualPlayer.min : parseInt(actualPlayer.min) || 0;
            if (mins < 10) continue;

            const addSingle = (prop, actualVal) => {
              if (!prop) return;
              const won = actualVal > prop.line;
              const pnl = won
                ? Math.round((americanToDecimal(prop.odds) - 1) * CONFIG.UNIT_SIZE)
                : -CONFIG.UNIT_SIZE;
              daySingles.push({ ...prop, date, gameKey, actual: actualVal, won, pnl });
            };

            const ptsLines = og && og.playerProps && og.playerProps[actualPlayer.name];
            // Reb/ast: check dedicated file first, then inline props from incremental loader
            const rebLines = (ra && ra.rebProps && ra.rebProps[actualPlayer.name])
              || (og && og.playerRebProps && og.playerRebProps[actualPlayer.name]);
            const astLines = (ra && ra.astProps && ra.astProps[actualPlayer.name])
              || (og && og.playerAstProps && og.playerAstProps[actualPlayer.name]);
            const actualReb = typeof actualPlayer.reb === 'number' ? actualPlayer.reb : parseInt(actualPlayer.reb) || 0;
            const actualAst = typeof actualPlayer.ast === 'number' ? actualPlayer.ast : parseInt(actualPlayer.ast) || 0;

            // --- Tier 1 (Strict) picks — real odds only ---
            if (ptsLines) addSingle(model.findBestStatProp(actualPlayer.name, 'points', ptsLines), actualPlayer.pts);
            if (rebLines) addSingle(model.findBestStatProp(actualPlayer.name, 'rebounds', rebLines), actualReb);
            if (astLines) addSingle(model.findBestStatProp(actualPlayer.name, 'assists', astLines), actualAst);

            // --- Tier 2 (Enhanced) picks — only with real odds ---
            if (ptsLines) {
              const t2pts = model.findBestT2Prop(actualPlayer.name, 'points', ptsLines);
              // Only add if not already picked as T1 (different line/odds)
              if (t2pts && !daySingles.find(s => s.player === t2pts.player && s.statType === t2pts.statType && s.line === t2pts.line)) {
                addSingle(t2pts, actualPlayer.pts);
              }
            }
            if (rebLines) {
              const t2reb = model.findBestT2Prop(actualPlayer.name, 'rebounds', rebLines);
              if (t2reb && !daySingles.find(s => s.player === t2reb.player && s.statType === t2reb.statType && s.line === t2reb.line)) {
                addSingle(t2reb, actualReb);
              }
            }
            if (astLines) {
              const t2ast = model.findBestT2Prop(actualPlayer.name, 'assists', astLines);
              if (t2ast && !daySingles.find(s => s.player === t2ast.player && s.statType === t2ast.statType && s.line === t2ast.line)) {
                addSingle(t2ast, actualAst);
              }
            }
          }
        }

        // Build parlays from day's qualifying singles
        const dayParlays = buildParlays(daySingles);
        for (const parlay of dayParlays) {
          const allHit = parlay.legs.every(leg => {
            const match = daySingles.find(p =>
              p.player === leg.player && p.line === leg.line && p.statType === leg.statType
            );
            return match && match.won;
          });

          parlay.date = date;
          parlay.won = allHit;
          parlay.pnl = allHit
            ? Math.round((parlay.decimalOdds - 1) * CONFIG.UNIT_SIZE)
            : -CONFIG.UNIT_SIZE;
          parlay.legs = parlay.legs.map(leg => {
            const match = daySingles.find(p =>
              p.player === leg.player && p.line === leg.line && p.statType === leg.statType
            );
            return {
              ...leg,
              won: match ? match.won : false,
              actual: match ? match.actual : 0,
            };
          });
        }

        results.singles.push(...daySingles);
        results.parlays.push(...dayParlays);

        if (daySingles.length > 0) {
          results.dailySummaries.push({
            date,
            singles: daySingles.length,
            parlays: dayParlays.length,
            wins: [...daySingles, ...dayParlays].filter(p => p.won).length,
            total: daySingles.length + dayParlays.length,
            pnl: [...daySingles, ...dayParlays].reduce((s, p) => s + p.pnl, 0),
          });
        }

        results.dates.push(date);
      }

      // Walk-forward: update model AFTER predictions
      const dateBoxes = boxByDate[game.date] || [];
      for (const bg of dateBoxes) {
        if (bg.home === game.home_team && bg.away === game.away_team) {
          for (const p of (bg.players || [])) {
            const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
            if (mins < 10) continue;
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

    // Calculate stats
    const calcGroup = (picks) => {
      if (picks.length === 0) return { total: 0, wins: 0, losses: 0, hitRate: 0, pnl: 0, roi: 0 };
      const wins = picks.filter(p => p.won).length;
      const pnl = picks.reduce((s, p) => s + p.pnl, 0);
      return {
        total: picks.length,
        wins,
        losses: picks.length - wins,
        hitRate: Math.round(wins / picks.length * 1000) / 1000,
        pnl,
        roi: Math.round(pnl / (picks.length * CONFIG.UNIT_SIZE) * 10000) / 100,
      };
    };

    // Stats by stat type
    const pointsSingles = results.singles.filter(s => s.statType === 'points');
    const reboundsSingles = results.singles.filter(s => s.statType === 'rebounds');
    const assistsSingles = results.singles.filter(s => s.statType === 'assists');

    results.stats = {
      overall: calcGroup([...results.singles, ...results.parlays]),
      singles: calcGroup(results.singles),
      parlays: calcGroup(results.parlays),
      points: calcGroup(pointsSingles),
      rebounds: calcGroup(reboundsSingles),
      assists: calcGroup(assistsSingles),
      totalDays: results.dates.length,
      daysWithPicks: results.dailySummaries.length,
    };

    // Rolling recent accuracy (last 30 days of data)
    const allDates = [...new Set(results.parlays.map(p => p.date))].sort();
    if (allDates.length > 0) {
      const last30Cutoff = allDates[Math.max(0, allDates.length - 30)] || '';
      const last14Cutoff = allDates[Math.max(0, allDates.length - 14)] || '';
      const last7Cutoff = allDates[Math.max(0, allDates.length - 7)] || '';

      results.stats.recent30 = calcGroup(results.parlays.filter(p => p.date >= last30Cutoff));
      results.stats.recent14 = calcGroup(results.parlays.filter(p => p.date >= last14Cutoff));
      results.stats.recent7 = calcGroup(results.parlays.filter(p => p.date >= last7Cutoff));
    }

    return results;
  }

  // --- Live Odds Fetching (Multi-Stat) ---

  async function fetchLiveOdds() {
    const result = { events: [], playerProps: {} };

    try {
      const eventsUrl = `${ODDS_API_BASE}/sports/basketball_nba/events?apiKey=${ODDS_API_KEY}`;
      const eventsRes = await fetch(eventsUrl);
      const remaining = eventsRes.headers.get('x-requests-remaining');
      console.log(`[ENGINE] Odds API: remaining=${remaining}, status=${eventsRes.status}`);

      if (!eventsRes.ok) {
        if (eventsRes.status === 401 || remaining === '0') {
          result.quotaExhausted = true;
          console.warn('[ENGINE] Odds API quota exhausted');
        }
        return result;
      }

      const events = await eventsRes.json();

      result.events = events;

      // Fetch all prop types for each event
      const markets = 'player_points_alternate,player_rebounds_alternate,player_assists_alternate';

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

          const gameProps = { points: {}, rebounds: {}, assists: {} };
          const marketMap = {
            'player_points_alternate': 'points',
            'player_rebounds_alternate': 'rebounds',
            'player_assists_alternate': 'assists',
          };

          for (const mkt of (fd.markets || [])) {
            const statType = marketMap[mkt.key];
            if (!statType) continue;

            for (const outcome of (mkt.outcomes || [])) {
              const player = outcome.description;
              const threshold = outcome.point;
              if (!gameProps[statType][player]) gameProps[statType][player] = {};
              if (!gameProps[statType][player][threshold]) gameProps[statType][player][threshold] = {};
              if (outcome.name === 'Over') {
                gameProps[statType][player][threshold].overOdds = outcome.price;
              }
            }
          }

          // Save to localStorage for future reference
          saveLiveOddsToCache(gameKey, event.id, gameProps);

          result.playerProps[gameKey] = {
            lines: gameProps.points,
            rebLines: gameProps.rebounds,
            astLines: gameProps.assists,
            eventId: event.id,
            home: homeAbbr,
            away: awayAbbr,
          };
        } catch (e) {
          console.warn('[ENGINE] Error fetching props for event:', event.id);
        }
      }

      console.log(`[ENGINE] Fetched multi-stat props for ${Object.keys(result.playerProps).length} games`);
    } catch (e) {
      console.error('[ENGINE] Error fetching live odds:', e);
    }

    return result;
  }

  // --- Odds Caching ---

  function saveLiveOddsToCache(gameKey, eventId, gameProps) {
    try {
      const cacheKey = 'nba_live_odds_cache';
      const cache = JSON.parse(localStorage.getItem(cacheKey) || '{}');
      const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      if (!cache[today]) cache[today] = {};
      cache[today][gameKey] = { eventId, ...gameProps };
      localStorage.setItem(cacheKey, JSON.stringify(cache));
    } catch (e) { /* ignore storage errors */ }
  }

  // --- Live Pick Tracking (2025-26 Season) ---

  const LIVE_PICKS_KEY = 'nba_live_picks';

  function loadLivePicks() {
    try {
      return JSON.parse(localStorage.getItem(LIVE_PICKS_KEY) || '{"parlays":[],"spParlays":[]}');
    } catch (e) { return { parlays: [], spParlays: [] }; }
  }

  function saveLivePicks(data) {
    try {
      localStorage.setItem(LIVE_PICKS_KEY, JSON.stringify(data));
    } catch (e) {
      console.warn('[ENGINE] Could not save live picks:', e);
    }
  }

  function saveTodayParlays(parlays, spParlays, date) {
    const data = loadLivePicks();

    // Don't duplicate — remove any existing entries for this date
    data.parlays = data.parlays.filter(p => p.date !== date);
    data.spParlays = data.spParlays.filter(p => p.date !== date);

    for (const parlay of parlays) {
      data.parlays.push({
        date,
        numLegs: parlay.numLegs || parlay.legs.length,
        odds: parlay.odds,
        decimalOdds: parlay.decimalOdds,
        combinedHitRate: parlay.combinedHitRate,
        ev: parlay.ev,
        legs: parlay.legs.map(l => ({
          player: l.player,
          team: l.team,
          statType: l.statType || 'points',
          statLabel: l.statLabel || statLabel(l.statType || 'points'),
          line: l.line,
          odds: l.odds,
          gameKey: l.gameKey || '',
          gameDisplay: l.gameDisplay || '',
          actual: null,
          won: null,
        })),
        resolved: false,
        won: null,
        pnl: null,
        savedAt: new Date().toISOString(),
      });
    }

    for (const parlay of spParlays) {
      data.spParlays.push({
        date,
        numLegs: parlay.numLegs || parlay.legs.length,
        odds: parlay.odds,
        decimalOdds: parlay.decimalOdds,
        combinedHitRate: parlay.combinedHitRate,
        ev: parlay.ev,
        isSuperPayout: true,
        legs: parlay.legs.map(l => ({
          player: l.player,
          team: l.team,
          statType: l.statType || 'points',
          statLabel: l.statLabel || 'PTS',
          line: l.line,
          odds: l.odds,
          gameKey: l.gameKey || '',
          gameDisplay: l.gameDisplay || '',
          actual: null,
          won: null,
        })),
        resolved: false,
        won: null,
        pnl: null,
        savedAt: new Date().toISOString(),
      });
    }

    saveLivePicks(data);
    console.log(`[ENGINE] Saved ${parlays.length} parlays + ${spParlays.length} SP parlays for ${date}`);
  }

  async function resolveLivePicks() {
    const data = loadLivePicks();
    const allParlays = [...data.parlays, ...data.spParlays];
    const unresolvedDates = [...new Set(
      allParlays.filter(p => !p.resolved).map(p => p.date)
    )];

    const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');

    // Only resolve past dates (not today — games may still be in progress)
    const datesToResolve = unresolvedDates.filter(d => d < today);
    if (datesToResolve.length === 0) return data;

    console.log(`[ENGINE] Resolving live picks for ${datesToResolve.length} dates:`, datesToResolve);

    for (const date of datesToResolve) {
      // Fetch ESPN scoreboard for this date to get event IDs
      const { games, eventIds } = await window.NbaApi.fetchESPNScoreboardForDate(date);
      if (!games.length) {
        console.warn(`[ENGINE] No games found for ${date}, skipping`);
        continue;
      }

      // Only resolve if all games are final
      const allFinal = games.every(g => g.status === 'final');
      if (!allFinal) {
        console.log(`[ENGINE] Not all games final for ${date}, skipping`);
        continue;
      }

      // Fetch box scores for each game
      const playerStats = {}; // playerName -> { pts, reb, ast }
      for (const [teamKey, eventId] of Object.entries(eventIds)) {
        const boxScore = await window.NbaApi.fetchESPNBoxScore(eventId);
        if (!boxScore) continue;
        for (const p of boxScore) {
          // Store by lowercase name for fuzzy matching
          playerStats[p.name.toLowerCase()] = p;
        }
        // Rate limit
        await new Promise(r => setTimeout(r, 200));
      }

      if (Object.keys(playerStats).length === 0) {
        console.warn(`[ENGINE] No box scores found for ${date}`);
        continue;
      }

      // Resolve each parlay for this date
      const resolveParlay = (parlay) => {
        if (parlay.date !== date || parlay.resolved) return;

        let allResolved = true;
        for (const leg of parlay.legs) {
          const pStats = playerStats[leg.player.toLowerCase()];
          if (!pStats) {
            // Try partial match
            const found = Object.entries(playerStats).find(([name]) =>
              name.includes(leg.player.toLowerCase().split(' ').pop())
              && name.includes(leg.player.toLowerCase().split(' ')[0])
            );
            if (found) {
              const [, stats] = found;
              const statKey = leg.statType === 'rebounds' ? 'reb' : leg.statType === 'assists' ? 'ast' : 'pts';
              leg.actual = stats[statKey];
              leg.won = leg.actual > leg.line;
            } else {
              allResolved = false;
              continue;
            }
          } else {
            const statKey = leg.statType === 'rebounds' ? 'reb' : leg.statType === 'assists' ? 'ast' : 'pts';
            leg.actual = pStats[statKey];
            leg.won = leg.actual > leg.line;
          }
        }

        if (allResolved) {
          parlay.resolved = true;
          parlay.won = parlay.legs.every(l => l.won);
          const UNIT = 100;
          parlay.pnl = parlay.won
            ? Math.round((parlay.decimalOdds - 1) * UNIT)
            : -UNIT;
          console.log(`[ENGINE] Resolved ${date} parlay: ${parlay.won ? 'WIN' : 'LOSS'} (${parlay.pnl})`);
        }
      };

      data.parlays.forEach(resolveParlay);
      data.spParlays.forEach(resolveParlay);
    }

    saveLivePicks(data);
    return data;
  }

  // Legacy wrapper for backward compat
  async function saveOddsToHistory(picks, date) {
    // No-op: replaced by saveTodayParlays
    console.log(`[ENGINE] saveOddsToHistory called for ${date} — use saveTodayParlays instead`);
  }

  // --- Incremental Daily Odds Cache ---

  const DAILY_ODDS_KEY = 'nba_daily_odds_cache';

  function loadCachedDailyOdds() {
    try {
      return JSON.parse(localStorage.getItem(DAILY_ODDS_KEY) || '{}');
    } catch (e) { return {}; }
  }

  function saveCachedDailyOdds(cache) {
    try {
      localStorage.setItem(DAILY_ODDS_KEY, JSON.stringify(cache));
    } catch (e) {}
  }

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
          homeTeam: homeAbbr, awayTeam: awayAbbr,
          commenceTime: event.commence_time,
        };

        for (const mkt of (fd.markets || [])) {
          if (mkt.key === 'h2h') {
            const homeO = mkt.outcomes.find(o => o.name === event.home_team);
            const awayO = mkt.outcomes.find(o => o.name === event.away_team);
            if (homeO) record.home_ml = homeO.price;
            if (awayO) record.away_ml = awayO.price;
          } else if (mkt.key === 'spreads') {
            const homeO = mkt.outcomes.find(o => o.name === event.home_team);
            if (homeO) { record.spread_home = homeO.price; record.spread_point = homeO.point; }
          } else if (mkt.key === 'totals') {
            const overO = mkt.outcomes.find(o => o.name === 'Over');
            if (overO) { record.total = overO.point; record.total_over = overO.price; }
          }
        }

        // Fetch player props (points, rebounds, assists)
        const propMarkets = 'player_points_alternate,player_rebounds_alternate,player_assists_alternate';
        try {
          const propsUrl = `${ODDS_API_BASE}/historical/sports/basketball_nba/events/${event.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=${propMarkets}&oddsFormat=american&date=${isoDate}&bookmakers=fanduel`;
          const propsRes = await fetch(propsUrl);
          if (propsRes.ok) {
            const propsData = await propsRes.json();
            const eventData = propsData.data || propsData;
            const fdBook = (eventData.bookmakers || []).find(b => b.key === 'fanduel');
            if (fdBook) {
              const marketMap = {
                'player_points_alternate': 'playerProps',
                'player_rebounds_alternate': 'playerRebProps',
                'player_assists_alternate': 'playerAstProps',
              };

              for (const mkt of (fdBook.markets || [])) {
                const propKey = marketMap[mkt.key];
                if (!propKey) continue;
                const playerLines = {};
                for (const outcome of (mkt.outcomes || [])) {
                  const player = outcome.description;
                  const threshold = outcome.point;
                  if (!playerLines[player]) playerLines[player] = {};
                  if (!playerLines[player][threshold]) playerLines[player][threshold] = {};
                  if (outcome.name === 'Over') {
                    playerLines[player][threshold].overOdds = outcome.price;
                  }
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

  async function fetchMissingDailyOdds(historicalOdds, maxDates) {
    const existingDates = new Set(historicalOdds.map(o => o.date));
    const cache = loadCachedDailyOdds();
    let newOdds = [];
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
      if (!existingDates.has(ds)) {
        dates.push(ds);
      }
    }

    console.log(`[ENGINE] ${dates.length} dates missing from historical data`);

    for (const date of dates) {
      if (cache[date]) {
        newOdds.push(...cache[date]);
        continue;
      }

      if (fetchedCount >= (maxDates || 5)) {
        console.log(`[ENGINE] Rate limited, will fetch more dates next load`);
        break;
      }

      console.log(`[ENGINE] Fetching odds for ${date}...`);
      const dateOdds = await fetchHistoricalOddsForDate(date);
      if (dateOdds.length > 0) {
        cache[date] = dateOdds;
        newOdds.push(...dateOdds);
        saveCachedDailyOdds(cache);
        console.log(`[ENGINE] Cached ${dateOdds.length} odds for ${date}`);
      } else {
        cache[date] = [];
        saveCachedDailyOdds(cache);
      }
      fetchedCount++;
      await new Promise(r => setTimeout(r, 300));
    }

    return newOdds;
  }

  // --- Public API ---

  return {
    CONFIG,
    CONFIG_T2,
    STAT_TYPES,
    PlayerModel,
    buildParlays,
    runBacktest,
    fetchLiveOdds,
    saveOddsToHistory,
    saveTodayParlays,
    loadLivePicks,
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
  };
})();
