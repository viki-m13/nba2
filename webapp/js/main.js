// =============================================================================
// MAIN APP CONTROLLER — NBA Dominance System (Play 3 ADI + Play 4 CDS)
// =============================================================================

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────────────
  let currentView = 'picks';
  let todayGames = [];          // All games today
  let todayPicks = [];          // Games flagged by ADI model (Play 3: moneyline)
  let todayCdsPicks = [];       // Games flagged by CDS model (Play 4: moneyline)
  let historyPicks = [];        // Historical Play 3 picks with results
  let cdsHistoryPicks = [];     // Historical Play 4 CDS picks with results
  let parlayHistory = [];        // Historical daily parlays with results
  let todayPactPicks = [];       // Games flagged by PACT model (Play 6: totals)
  let pactHistoryPicks = [];     // Historical Play 6 PACT picks with results
  let seasonData = [];          // Full season game data for model training
  let modelReady = false;
  let currentHistoryPeriod = 'all';
  let currentCdsHistoryPeriod = 'all';
  let currentParlayHistoryPeriod = 'all';
  let currentPactHistoryPeriod = 'all';
  let useProxy = false;        // True when running on Vercel (CORS proxy available)

  const Model = window.ParlayEngine.PreGameModel;
  const PACTModel = window.ParlayEngine.PACTModel;

  // NBA team full names for display
  const TEAM_NAMES = {
    ATL: 'Atlanta Hawks', BOS: 'Boston Celtics', BKN: 'Brooklyn Nets',
    CHA: 'Charlotte Hornets', CHI: 'Chicago Bulls', CLE: 'Cleveland Cavaliers',
    DAL: 'Dallas Mavericks', DEN: 'Denver Nuggets', DET: 'Detroit Pistons',
    GS: 'Golden State Warriors', GSW: 'Golden State Warriors',
    HOU: 'Houston Rockets', IND: 'Indiana Pacers',
    LAC: 'LA Clippers', LAL: 'Los Angeles Lakers',
    MEM: 'Memphis Grizzlies', MIA: 'Miami Heat', MIL: 'Milwaukee Bucks',
    MIN: 'Minnesota Timberwolves', NO: 'New Orleans Pelicans', NOP: 'New Orleans Pelicans',
    NY: 'New York Knicks', NYK: 'New York Knicks',
    OKC: 'Oklahoma City Thunder', ORL: 'Orlando Magic',
    PHI: 'Philadelphia 76ers', PHX: 'Phoenix Suns',
    POR: 'Portland Trail Blazers', SA: 'San Antonio Spurs', SAS: 'San Antonio Spurs',
    SAC: 'Sacramento Kings', TOR: 'Toronto Raptors',
    UTAH: 'Utah Jazz', UTA: 'Utah Jazz',
    WSH: 'Washington Wizards', WAS: 'Washington Wizards',
  };

  function teamName(abbr) {
    return TEAM_NAMES[abbr] || abbr;
  }

  // Convert predicted margin (spread) to approximate moneyline odds
  // Based on standard NBA spread-to-ML conversion tables
  function spreadToMoneyline(margin) {
    const m = Math.abs(margin);
    // Approximate mapping: spread → ML favorite odds (negative American)
    if (m <= 1.5) return -120;
    if (m <= 2.5) return -135;
    if (m <= 3.5) return -155;
    if (m <= 4.5) return -185;
    if (m <= 5.5) return -210;
    if (m <= 6.5) return -250;
    if (m <= 7.5) return -300;
    if (m <= 8.5) return -350;
    if (m <= 9.5) return -420;
    if (m <= 10.5) return -500;
    if (m <= 12.5) return -650;
    if (m <= 14.5) return -850;
    return -1000;
  }

  // Calculate profit on a $100 moneyline bet at given American odds
  // e.g. -200 → $50 profit, -300 → $33.33 profit
  function moneylineProfit(odds) {
    return Math.round((100 / Math.abs(odds)) * 100);
  }

  // Convert American odds to decimal odds (e.g. -200 → 1.50, -300 → 1.333)
  function americanToDecimal(odds) {
    return 1 + (100 / Math.abs(odds));
  }

  // Calculate parlay decimal odds from array of American odds
  function parlayDecimalOdds(oddsArray) {
    return oddsArray.reduce((acc, odds) => acc * americanToDecimal(odds), 1);
  }

  // Calculate parlay payout on $100 bet (profit only, not including stake)
  function parlayPayout(oddsArray) {
    const decimal = parlayDecimalOdds(oddsArray);
    return Math.round((decimal - 1) * 100);
  }

  // Normalize ESPN abbreviations to our format
  function norm(abbr) {
    const map = { GSW: 'GS', NYK: 'NY', NOP: 'NO', SAS: 'SA', UTA: 'UTAH', WAS: 'WSH' };
    return map[abbr] || abbr;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // CONVERGENT DOMINANCE SCORE (CDS) — PLAY 4 MODEL
  // ══════════════════════════════════════════════════════════════════════════

  const CDSModel = {
    LOOKBACK: 15,
    teamHistory: {},

    reset() {
      this.teamHistory = {};
    },

    updateTeam(team, pointsFor, pointsAgainst, date) {
      if (!this.teamHistory[team]) this.teamHistory[team] = [];
      this.teamHistory[team].push({
        pf: pointsFor,
        pa: pointsAgainst,
        margin: pointsFor - pointsAgainst,
        date: date,
      });
      if (this.teamHistory[team].length > this.LOOKBACK * 2) {
        this.teamHistory[team] = this.teamHistory[team].slice(-this.LOOKBACK);
      }
    },

    getMetrics(team) {
      const history = (this.teamHistory[team] || []).slice(-this.LOOKBACK);
      if (history.length < 8) return null;
      const pf = history.map(g => g.pf);
      const pa = history.map(g => g.pa);
      const margins = history.map(g => g.margin);
      const avgMargin = margins.reduce((a, b) => a + b, 0) / margins.length;
      return {
        offRating: pf.reduce((a, b) => a + b, 0) / pf.length,
        defRating: pa.reduce((a, b) => a + b, 0) / pa.length,
        netRating: avgMargin,
        winPct: margins.filter(m => m > 0).length / margins.length,
        games: history.length,
      };
    },

    computeCDS(favM, dogM, favIsHome) {
      let score = 0;
      const dims = {};

      // D1: Net Rating Gap
      const ng = Math.abs(favM.netRating - dogM.netRating);
      let d1 = 0;
      if (ng >= 14) d1 = 3.0;
      else if (ng >= 10) d1 = 2.0 + (ng - 10) / 4.0;
      else if (ng >= 7) d1 = 1.0 + (ng - 7) / 3.0;
      else if (ng >= 4) d1 = (ng - 4) / 3.0;
      dims.net_gap = +d1.toFixed(2);
      score += d1;

      // D2: Offensive Firepower
      const fo = favM.offRating;
      let d2 = 0;
      if (fo >= 122) d2 = 3.0;
      else if (fo >= 118) d2 = 2.0 + (fo - 118) / 4.0;
      else if (fo >= 115) d2 = 1.0 + (fo - 115) / 3.0;
      else if (fo >= 112) d2 = (fo - 112) / 3.0;
      dims.offense = +d2.toFixed(2);
      score += d2;

      // D3: Defensive Cage
      const cage = favM.defRating - dogM.offRating;
      let d3 = 0;
      if (cage <= -6) d3 = 3.0;
      else if (cage <= -3) d3 = 2.0 + (-3 - cage) / 3.0;
      else if (cage <= 0) d3 = 1.0 + (0 - cage) / 3.0;
      else if (cage <= 3) d3 = (3 - cage) / 3.0;
      dims.defense = +d3.toFixed(2);
      score += d3;

      // D4: Win Consistency Gap
      const wpg = Math.abs(favM.winPct - dogM.winPct);
      let d4 = 0;
      if (wpg >= 0.5) d4 = 3.0;
      else if (wpg >= 0.4) d4 = 2.0 + (wpg - 0.4) / 0.1;
      else if (wpg >= 0.3) d4 = 1.0 + (wpg - 0.3) / 0.1;
      else if (wpg >= 0.2) d4 = (wpg - 0.2) / 0.1;
      dims.consistency = +d4.toFixed(2);
      score += d4;

      // D5: Home Court
      const d5 = favIsHome ? 1.5 : 0;
      dims.home = d5;
      score += d5;

      // D6: Dual-Edge Bonus
      let d6 = 0;
      if (favM.offRating >= 115 && favM.defRating <= 110) d6 = 1.5;
      else if (favM.offRating >= 112 && favM.defRating <= 108) d6 = 1.0;
      dims.dual_edge = d6;
      score += d6;

      // D7: Opponent Weakness Amplifier
      let d7 = 0;
      if (dogM.offRating <= 110 && dogM.defRating >= 114) d7 = 1.0;
      else if (dogM.offRating <= 112 && dogM.defRating >= 112) d7 = 0.5;
      dims.opp_weakness = d7;
      score += d7;

      return { score: +score.toFixed(1), dimensions: dims };
    },

    predictGame(homeTeam, awayTeam) {
      const homeM = this.getMetrics(homeTeam);
      const awayM = this.getMetrics(awayTeam);
      if (!homeM || !awayM) return null;

      const HOME_ADV = 3.5;
      const netDiff = homeM.netRating - awayM.netRating;
      const predictedMargin = netDiff + HOME_ADV;

      let fav, dog, favM, dogM, favIsHome;
      if (predictedMargin > 0) {
        fav = homeTeam; dog = awayTeam; favM = homeM; dogM = awayM; favIsHome = true;
      } else {
        fav = awayTeam; dog = homeTeam; favM = awayM; dogM = homeM; favIsHome = false;
      }

      const { score, dimensions } = this.computeCDS(favM, dogM, favIsHome);

      // Determine tier
      let tier = null, kelly = 0;
      if (score >= 11) { tier = 'ELITE'; kelly = 0.10; }
      else if (score >= 9) { tier = 'HIGH'; kelly = 0.08; }
      else if (score >= 8) { tier = 'STRONG'; kelly = 0.06; }

      if (!tier) return null;

      // Top pathway
      const topDim = Object.entries(dimensions).sort((a, b) => b[1] - a[1])[0];
      const pathway = topDim ? topDim[0] : 'none';

      // Check if Play 3 would also flag this game
      const netGap = Math.abs(netDiff);
      const play3Also = netGap >= 10 && favM.offRating >= 118 && favIsHome;

      return {
        favorite: fav,
        underdog: dog,
        favIsHome,
        predictedMargin: +Math.abs(predictedMargin).toFixed(1),
        cdsScore: score,
        tier,
        dimensions,
        pathway,
        netGap: +netGap.toFixed(1),
        favOffRating: +favM.offRating.toFixed(1),
        favDefRating: +favM.defRating.toFixed(1),
        winPctGap: +Math.abs(favM.winPct - dogM.winPct).toFixed(2),
        kellyFraction: kelly,
        play3Also,
        isIncremental: !play3Also,
      };
    },
  };

  // ── Initialization ─────────────────────────────────────────────────────────

  async function init() {
    console.log('[APP] Initializing...');
    setupNavigation();
    setupHistoryFilters();
    setStatus('loading', 'Loading model data...');

    // Detect Vercel proxy for CORS
    try {
      const probe = await fetch('/api/nba?endpoint=espn_scoreboard', { method: 'HEAD' });
      useProxy = probe.ok || probe.status === 405 || probe.status === 200;
    } catch (e) { useProxy = false; }
    console.log('[APP] Proxy available:', useProxy);

    try {
      await loadSeasonData();
      await fetchTodayGames();
      runPredictions();
      runCDSPredictions();
      runPACTPredictions();
      buildHistory();
      buildCDSHistory();
      buildParlayHistory();
      buildPACTHistory();
      renderPicks();
      renderAllGames();
      renderHistory();
      updateMetrics();

      setStatus('online', 'Models Active');
      modelReady = true;

      // Poll for live score updates every 30s
      setInterval(refreshScores, 30000);

    } catch (err) {
      console.error('[APP] Init error:', err);
      setStatus('error', 'Failed to load');
    }
  }

  // ── Data Loading ───────────────────────────────────────────────────────────

  async function loadSeasonData() {
    console.log('[APP] Loading season data...');

    try {
      const resp = await fetch('data/espn_full_season_2025.json');
      if (resp.ok) {
        seasonData = await resp.json();
        console.log(`[APP] Loaded ${seasonData.length} games from season file`);
      }
    } catch (e) {
      console.warn('[APP] Could not load season file, will fetch from ESPN');
    }

    // Fetch recent games from ESPN to fill gaps
    const recentDates = getRecentDates(30);
    let fetched = 0;
    const existingDates = new Set(seasonData.map(g => g.date));

    for (const dateStr of recentDates) {
      if (existingDates.has(dateStr)) continue;
      try {
        const games = await fetchESPNGamesForDate(dateStr);
        for (const g of games) {
          if (!existingDates.has(g.date + '_' + g.home_team + '_' + g.away_team)) {
            seasonData.push(g);
          }
        }
        fetched++;
        if (fetched % 5 === 0) console.log(`[APP] Fetched ${fetched} recent dates...`);
        await sleep(300);
      } catch (e) { /* skip failed dates */ }
    }

    // Feed all data into all models
    seasonData.sort((a, b) => (a.date || '').localeCompare(b.date || ''));
    Model.reset();
    CDSModel.reset();
    PACTModel.teamHistory = {};
    for (const g of seasonData) {
      Model.updateTeam(g.home_team, g.home_score, g.away_score, g.date);
      Model.updateTeam(g.away_team, g.away_score, g.home_score, g.date);
      CDSModel.updateTeam(g.home_team, g.home_score, g.away_score, g.date);
      CDSModel.updateTeam(g.away_team, g.away_score, g.home_score, g.date);
      PACTModel.updateTeam(g.home_team, g.home_score, g.away_score, g.date);
      PACTModel.updateTeam(g.away_team, g.away_score, g.home_score, g.date);
    }

    console.log(`[APP] Models trained on ${seasonData.length} games, ${Object.keys(Model.teamHistory).length} teams`);
  }

  async function fetchESPNGamesForDate(dateStr) {
    const url = useProxy
      ? `/api/nba?endpoint=espn_scoreboard&dates=${dateStr}`
      : `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=${dateStr}`;
    const resp = await fetch(url);
    if (!resp.ok) return [];
    const data = await resp.json();
    const games = [];

    for (const event of (data.events || [])) {
      const comp = (event.competitions || [{}])[0];
      const competitors = comp.competitors || [];
      const status = (comp.status || {}).type || {};
      if (status.name !== 'STATUS_FINAL' || competitors.length < 2) continue;

      const home = competitors.find(c => c.homeAway === 'home') || competitors[0];
      const away = competitors.find(c => c.homeAway === 'away') || competitors[1];

      games.push({
        date: dateStr,
        home_team: norm(home.team.abbreviation),
        away_team: norm(away.team.abbreviation),
        home_score: parseInt(home.score) || 0,
        away_score: parseInt(away.score) || 0,
        winner_score: Math.max(parseInt(home.score) || 0, parseInt(away.score) || 0),
      });
    }
    return games;
  }

  async function fetchTodayGames() {
    console.log('[APP] Fetching today\'s games...');
    const url = useProxy
      ? '/api/nba?endpoint=espn_scoreboard'
      : 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard';

    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error('ESPN fetch failed');
      const data = await resp.json();

      todayGames = [];
      for (const event of (data.events || [])) {
        const comp = (event.competitions || [{}])[0];
        const competitors = comp.competitors || [];
        if (competitors.length < 2) continue;

        const home = competitors.find(c => c.homeAway === 'home') || competitors[0];
        const away = competitors.find(c => c.homeAway === 'away') || competitors[1];
        const status = (comp.status || {}).type || {};
        const clock = (comp.status || {}).displayClock || '';
        const period = (comp.status || {}).period || 0;

        const startTime = new Date(comp.date || event.date);
        const timeStr = startTime.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });

        todayGames.push({
          id: event.id,
          home_team: norm(home.team.abbreviation),
          away_team: norm(away.team.abbreviation),
          home_score: parseInt(home.score) || 0,
          away_score: parseInt(away.score) || 0,
          status: status.name || 'STATUS_SCHEDULED',
          statusDesc: status.description || 'Scheduled',
          clock,
          period,
          time: timeStr,
          startTime,
        });
      }

      console.log(`[APP] Found ${todayGames.length} games today`);
    } catch (e) {
      console.error('[APP] Error fetching today:', e);
    }
  }

  // ── ADI Predictions (Play 3) ─────────────────────────────────────────────

  function runPredictions() {
    todayPicks = [];

    for (const game of todayGames) {
      const pred = Model.predictGame(game.home_team, game.away_team);

      if (pred && pred.signals && pred.signals.length > 0) {
        todayPicks.push({
          game,
          prediction: pred,
          confidence: pred.signals[0].confidence,
          betTeam: pred.favorite,
          isHome: pred.favIsHome,
          predictedMargin: pred.predictedMargin,
          mlOdds: spreadToMoneyline(pred.predictedMargin),
          netGap: pred.netGap,
          favOffRating: pred.favOffRating,
          dogDefRating: pred.dogDefRating,
          bps: pred.bps,
          kellyFraction: pred.betRecommendation.kellyFraction,
        });
      }
    }

    const confOrder = { HIGH: 0, STRONG: 1, MODERATE: 2 };
    todayPicks.sort((a, b) => {
      const co = (confOrder[a.confidence] || 9) - (confOrder[b.confidence] || 9);
      if (co !== 0) return co;
      return b.predictedMargin - a.predictedMargin;
    });

    console.log(`[ADI] ${todayPicks.length} picks from ${todayGames.length} games`);
  }

  // ── CDS Predictions (Play 4) ─────────────────────────────────────────────

  function runCDSPredictions() {
    todayCdsPicks = [];

    for (const game of todayGames) {
      const pred = CDSModel.predictGame(game.home_team, game.away_team);
      if (pred) {
        todayCdsPicks.push({
          game,
          prediction: pred,
          betTeam: pred.favorite,
          isHome: pred.favIsHome,
          predictedMargin: pred.predictedMargin,
          mlOdds: spreadToMoneyline(pred.predictedMargin),
          cdsScore: pred.cdsScore,
          tier: pred.tier,
          pathway: pred.pathway,
          dimensions: pred.dimensions,
          netGap: pred.netGap,
          favOffRating: pred.favOffRating,
          favDefRating: pred.favDefRating,
          winPctGap: pred.winPctGap,
          kellyFraction: pred.kellyFraction,
          play3Also: pred.play3Also,
          isIncremental: pred.isIncremental,
        });
      }
    }

    // Sort: ELITE first, then HIGH, then STRONG, then by CDS score
    const tierOrder = { ELITE: 0, HIGH: 1, STRONG: 2 };
    todayCdsPicks.sort((a, b) => {
      const to = (tierOrder[a.tier] || 9) - (tierOrder[b.tier] || 9);
      if (to !== 0) return to;
      return b.cdsScore - a.cdsScore;
    });

    console.log(`[CDS] ${todayCdsPicks.length} picks from ${todayGames.length} games`);
  }

  // ── PACT Predictions (Play 6) ──────────────────────────────────────────

  function runPACTPredictions() {
    todayPactPicks = [];

    for (const game of todayGames) {
      const pred = PACTModel.predictGame(game.home_team, game.away_team);
      if (pred) {
        todayPactPicks.push({
          game,
          prediction: pred,
          direction: pred.direction,
          predTotal: pred.predTotal,
          pactStrength: pred.pactStrength,
          tier: pred.tier,
          factors: pred.factors,
          combinedDef: pred.combinedDef,
          minPace: pred.minPace,
          hTrend: pred.hTrend,
          aTrend: pred.aTrend,
          homeOff: pred.homeOff,
          homeDef: pred.homeDef,
          awayOff: pred.awayOff,
          awayDef: pred.awayDef,
        });
      }
    }

    // Sort: ELITE first, then by strength
    const tierOrder = { ELITE: 0, HIGH: 1, STRONG: 2 };
    todayPactPicks.sort((a, b) => {
      const to = (tierOrder[a.tier] || 9) - (tierOrder[b.tier] || 9);
      if (to !== 0) return to;
      return b.pactStrength - a.pactStrength;
    });

    console.log(`[PACT] ${todayPactPicks.length} picks from ${todayGames.length} games`);
  }

  // ── History Builders ─────────────────────────────────────────────────────

  function buildHistory() {
    historyPicks = [];
    const tempModel = {
      teamHistory: {},
      lookbackWindow: 15,
      updateTeam: ParlayEngine.PreGameModel.updateTeam,
      getMetrics: ParlayEngine.PreGameModel.getMetrics,
      predictGame: ParlayEngine.PreGameModel.predictGame,
    };

    const sorted = [...seasonData].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    for (const game of sorted) {
      const pred = tempModel.predictGame(game.home_team, game.away_team);

      if (pred && pred.signals && pred.signals.length > 0) {
        const actualMargin = game.home_score - game.away_score;
        const actualMarginAbs = Math.abs(actualMargin);
        const favWon = (pred.favIsHome && actualMargin > 0) || (!pred.favIsHome && actualMargin < 0);
        const mlOdds = spreadToMoneyline(pred.predictedMargin);
        const profit = moneylineProfit(mlOdds);

        historyPicks.push({
          date: game.date,
          favorite: pred.favorite,
          underdog: pred.underdog,
          favIsHome: pred.favIsHome,
          confidence: pred.signals[0].confidence,
          predictedMargin: pred.predictedMargin,
          actualMargin: actualMarginAbs,
          favWon,
          mlOdds,
          homeScore: game.home_score,
          awayScore: game.away_score,
          pnl: favWon ? profit : -100,
        });
      }

      tempModel.updateTeam(game.home_team, game.home_score, game.away_score, game.date);
      tempModel.updateTeam(game.away_team, game.away_score, game.home_score, game.date);
    }

    console.log(`[ADI] Built history: ${historyPicks.length} picks`);
  }

  function buildCDSHistory() {
    cdsHistoryPicks = [];

    // Independent CDS model instance for walk-forward backtesting
    const cds = Object.create(CDSModel);
    cds.teamHistory = {};

    const sorted = [...seasonData].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    for (const game of sorted) {
      const pred = cds.predictGame(game.home_team, game.away_team);

      if (pred && pred.isIncremental) {
        // Only show incremental picks (ones Play 3 misses)
        const actualMargin = game.home_score - game.away_score;
        const favWon = (pred.favIsHome && actualMargin > 0) || (!pred.favIsHome && actualMargin < 0);
        const mlOdds = spreadToMoneyline(pred.predictedMargin);
        const profit = moneylineProfit(mlOdds);

        cdsHistoryPicks.push({
          date: game.date,
          favorite: pred.favorite,
          underdog: pred.underdog,
          favIsHome: pred.favIsHome,
          cdsScore: pred.cdsScore,
          tier: pred.tier,
          pathway: pred.pathway,
          dimensions: pred.dimensions,
          predictedMargin: pred.predictedMargin,
          actualMargin: Math.abs(actualMargin),
          favWon,
          mlOdds,
          pnl: favWon ? profit : -100,
        });
      }

      cds.updateTeam(game.home_team, game.home_score, game.away_score, game.date);
      cds.updateTeam(game.away_team, game.away_score, game.home_score, game.date);
    }

    console.log(`[CDS] Built history: ${cdsHistoryPicks.length} incremental picks`);
  }

  function buildParlayHistory() {
    parlayHistory = [];

    // Combine all history picks (Play 3 + Play 4) and group by date
    const allPicks = [
      ...historyPicks.map(p => ({ ...p, source: 'P3' })),
      ...cdsHistoryPicks.map(p => ({ ...p, source: 'P4' })),
    ];

    const byDate = {};
    for (const p of allPicks) {
      if (!byDate[p.date]) byDate[p.date] = [];
      byDate[p.date].push(p);
    }

    const dates = Object.keys(byDate).sort();
    for (const date of dates) {
      const legs = byDate[date];
      if (legs.length < 2) continue; // Need 2+ legs for a parlay

      const oddsArray = legs.map(l => l.mlOdds);
      const allWon = legs.every(l => l.favWon);
      const losses = legs.filter(l => !l.favWon).length;
      const payout = parlayPayout(oddsArray);

      parlayHistory.push({
        date,
        legs,
        legCount: legs.length,
        oddsArray,
        parlayDecimal: parlayDecimalOdds(oddsArray),
        payout,       // potential profit on $100
        allWon,
        losses,
        pnl: allWon ? payout : -100,
      });
    }

    console.log(`[PARLAY] Built history: ${parlayHistory.length} daily parlays`);
  }

  function buildPACTHistory() {
    pactHistoryPicks = [];

    const pact = Object.create(PACTModel);
    pact.teamHistory = {};

    const sorted = [...seasonData].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    for (const game of sorted) {
      const pred = pact.predictGame(game.home_team, game.away_team);

      if (pred) {
        const actualTotal = game.home_score + game.away_score;
        const isOver = pred.direction === 'OVER';
        const hit = isOver ? actualTotal > pred.predTotal : actualTotal < pred.predTotal;

        pactHistoryPicks.push({
          date: game.date,
          home: game.home_team,
          away: game.away_team,
          direction: pred.direction,
          predTotal: pred.predTotal,
          actualTotal,
          pactStrength: pred.pactStrength,
          tier: pred.tier,
          factors: pred.factors,
          hit,
          pnl: hit ? 91 : -100,
        });
      }

      pact.updateTeam(game.home_team, game.home_score, game.away_score, game.date);
      pact.updateTeam(game.away_team, game.away_score, game.home_score, game.date);
    }

    console.log(`[PACT] Built history: ${pactHistoryPicks.length} total picks`);
  }

  // ── Rendering: Today's Picks ───────────────────────────────────────────────

  function renderPicks() {
    const loading = document.getElementById('picks-loading');
    const container = document.getElementById('picks-container');
    const cdsContainer = document.getElementById('cds-container');
    const empty = document.getElementById('picks-empty');
    const allSection = document.getElementById('all-games-section');

    loading.style.display = 'none';

    const hasPicks = todayPicks.length > 0 || todayCdsPicks.length > 0;

    if (!hasPicks) {
      container.style.display = 'none';
      if (cdsContainer) cdsContainer.style.display = 'none';
      empty.style.display = 'block';
    } else {
      empty.style.display = 'none';

      // Play 3 picks
      if (todayPicks.length > 0) {
        container.style.display = '';
        container.innerHTML = '<h3 class="section-title">Play 3 — ADI Moneyline Picks</h3>' +
          '<div class="picks-grid">' + todayPicks.map(renderPickCard).join('') + '</div>';
      } else {
        container.style.display = 'none';
      }

      // Play 4 CDS picks (only show incremental ones that Play 3 misses)
      if (cdsContainer && todayCdsPicks.length > 0) {
        // Filter to show all CDS picks, but mark incremental ones
        cdsContainer.style.display = '';
        cdsContainer.innerHTML = '<h3 class="section-title">Play 4 — CDS Moneyline Picks</h3>' +
          '<div class="picks-grid">' + todayCdsPicks.map(renderCDSCard).join('') + '</div>';
      } else if (cdsContainer) {
        cdsContainer.style.display = 'none';
      }

      // Play 5 — Daily Parlay (all picks combined)
      const parlayContainer = document.getElementById('parlay-container');
      const allLegs = [...todayPicks, ...todayCdsPicks];
      if (parlayContainer && allLegs.length >= 2) {
        parlayContainer.style.display = '';
        parlayContainer.innerHTML = renderTodayParlayCard(allLegs);
      } else if (parlayContainer) {
        parlayContainer.style.display = 'none';
      }

      // Play 6 — PACT Totals
      const pactContainer = document.getElementById('pact-container');
      if (pactContainer && todayPactPicks.length > 0) {
        pactContainer.style.display = '';
        pactContainer.innerHTML = '<h3 class="section-title">Play 6 — PACT Over/Under Picks <span class="pact-badge">-110</span></h3>' +
          '<div class="picks-grid">' + todayPactPicks.map(renderPACTCard).join('') + '</div>';
      } else if (pactContainer) {
        pactContainer.style.display = 'none';
      }
    }

    if (todayGames.length > 0) {
      allSection.style.display = '';
    }
  }

  function renderPickCard(pick) {
    const g = pick.game;
    const conf = pick.confidence;
    const confClass = conf === 'HIGH' ? 'conf-high' : conf === 'STRONG' ? 'conf-strong' : 'conf-moderate';
    const confLabel = conf === 'HIGH' ? 'HIGH CONFIDENCE' : conf === 'STRONG' ? 'STRONG' : 'MODERATE';

    const favFull = teamName(pick.betTeam);
    const oppTeam = pick.isHome ? g.away_team : g.home_team;
    const oppFull = teamName(oppTeam);
    const homeAway = pick.isHome ? 'Home' : 'Away';

    let liveHtml = '';
    if (g.status === 'STATUS_FINAL') {
      const favScore = pick.isHome ? g.home_score : g.away_score;
      const oppScore = pick.isHome ? g.away_score : g.home_score;
      const won = favScore > oppScore;
      liveHtml = `
        <div class="pick-live ${won ? 'live-win' : 'live-loss'}">
          <span class="live-label">FINAL</span>
          <span class="live-score">${pick.betTeam} ${favScore} — ${oppTeam} ${oppScore}</span>
          <span class="live-result">${won ? 'W' : 'L'}</span>
        </div>`;
    } else if (g.status === 'STATUS_IN_PROGRESS' || g.status === 'STATUS_HALFTIME') {
      const favScore = pick.isHome ? g.home_score : g.away_score;
      const oppScore = pick.isHome ? g.away_score : g.home_score;
      liveHtml = `
        <div class="pick-live live-active">
          <span class="live-label">LIVE Q${g.period} ${g.clock}</span>
          <span class="live-score">${pick.betTeam} ${favScore} — ${oppTeam} ${oppScore}</span>
        </div>`;
    } else {
      liveHtml = `
        <div class="pick-live live-scheduled">
          <span class="live-label">${g.time}</span>
          <span class="live-status">Pre-Game — Bet Before Tip-Off</span>
        </div>`;
    }

    return `
      <div class="pick-card ${confClass}">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 3</span>
          <span class="pick-conf">${confLabel}</span>
        </div>
        <div class="pick-bet-line">
          ${pick.betTeam} ML ${pick.mlOdds}
        </div>
        <div class="pick-matchup">
          ${favFull} vs ${oppFull}
        </div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Side</span>
            <span class="detail-value">${homeAway}</span>
          </div>
          <div class="detail">
            <span class="detail-label">ML Odds</span>
            <span class="detail-value">${pick.mlOdds}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Net Gap</span>
            <span class="detail-value">${pick.netGap.toFixed(1)}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Off Rating</span>
            <span class="detail-value">${pick.favOffRating.toFixed(1)}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Kelly</span>
            <span class="detail-value">${(pick.kellyFraction * 100).toFixed(0)}%</span>
          </div>
        </div>
        ${liveHtml}
      </div>`;
  }

  function renderCDSCard(pick) {
    const g = pick.game;
    const tierClass = pick.tier === 'ELITE' ? 'conf-diamond' : pick.tier === 'HIGH' ? 'conf-high' : 'conf-strong';
    const tierLabel = pick.tier === 'ELITE' ? 'ELITE' : pick.tier === 'HIGH' ? 'HIGH' : 'STRONG';

    const favFull = teamName(pick.betTeam);
    const oppTeam = pick.isHome ? g.away_team : g.home_team;
    const oppFull = teamName(oppTeam);
    const homeAway = pick.isHome ? 'Home' : 'Away';

    const PATHWAY_LABELS = {
      net_gap: 'Net Rating Gap',
      offense: 'Offensive Firepower',
      defense: 'Defensive Cage',
      consistency: 'Win Consistency',
      home: 'Home Court',
      dual_edge: 'Two-Way Dominance',
      opp_weakness: 'Opponent Weakness',
    };
    const pathwayLabel = PATHWAY_LABELS[pick.pathway] || pick.pathway;

    const incrementalBadge = pick.isIncremental
      ? '<span class="pick-incremental">INCREMENTAL — Play 3 misses this game</span>'
      : '<span class="pick-overlap">Also flagged by Play 3</span>';

    let liveHtml = '';
    if (g.status === 'STATUS_FINAL') {
      const favScore = pick.isHome ? g.home_score : g.away_score;
      const oppScore = pick.isHome ? g.away_score : g.home_score;
      const won = favScore > oppScore;
      liveHtml = `
        <div class="pick-live ${won ? 'live-win' : 'live-loss'}">
          <span class="live-label">FINAL</span>
          <span class="live-score">${pick.betTeam} ${favScore} — ${oppTeam} ${oppScore}</span>
          <span class="live-result">${won ? 'W' : 'L'}</span>
        </div>`;
    } else if (g.status === 'STATUS_IN_PROGRESS' || g.status === 'STATUS_HALFTIME') {
      const favScore = pick.isHome ? g.home_score : g.away_score;
      const oppScore = pick.isHome ? g.away_score : g.home_score;
      liveHtml = `
        <div class="pick-live live-active">
          <span class="live-label">LIVE Q${g.period} ${g.clock}</span>
          <span class="live-score">${pick.betTeam} ${favScore} — ${oppTeam} ${oppScore}</span>
        </div>`;
    } else {
      liveHtml = `
        <div class="pick-live live-scheduled">
          <span class="live-label">${g.time}</span>
          <span class="live-status">Pre-Game — Bet Before Tip-Off</span>
        </div>`;
    }

    return `
      <div class="pick-card ${tierClass}">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 4</span>
          <span class="pick-conf">${tierLabel} — CDS ${pick.cdsScore}</span>
        </div>
        <div class="pick-bet-line">
          ${pick.betTeam} ML ${pick.mlOdds}
        </div>
        <div class="pick-matchup">
          ${favFull} vs ${oppFull}
        </div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Side</span>
            <span class="detail-value">${homeAway}</span>
          </div>
          <div class="detail">
            <span class="detail-label">ML Odds</span>
            <span class="detail-value">${pick.mlOdds}</span>
          </div>
          <div class="detail">
            <span class="detail-label">CDS Score</span>
            <span class="detail-value diamond-value">${pick.cdsScore}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Top Path</span>
            <span class="detail-value">${pathwayLabel}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Net Gap</span>
            <span class="detail-value">${pick.netGap}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Off/Def</span>
            <span class="detail-value">${pick.favOffRating}/${pick.favDefRating}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Kelly</span>
            <span class="detail-value">${(pick.kellyFraction * 100).toFixed(0)}%</span>
          </div>
        </div>
        ${incrementalBadge}
        ${liveHtml}
      </div>`;
  }

  function renderTodayParlayCard(allLegs) {
    const oddsArray = allLegs.map(l => l.mlOdds);
    const payout = parlayPayout(oddsArray);
    const decimalOdds = parlayDecimalOdds(oddsArray);
    const impliedProb = (1 / decimalOdds * 100).toFixed(1);

    // Check live results
    const finalLegs = allLegs.filter(l => l.game.status === 'STATUS_FINAL');
    const bustedLegs = finalLegs.filter(l => {
      const favScore = l.isHome ? l.game.home_score : l.game.away_score;
      const oppScore = l.isHome ? l.game.away_score : l.game.home_score;
      return favScore <= oppScore;
    });
    const allFinal = finalLegs.length === allLegs.length;
    const busted = bustedLegs.length > 0;

    let statusHtml = '';
    if (allFinal && !busted) {
      statusHtml = `<div class="pick-live live-win"><span class="live-label">FINAL</span><span class="live-result">PARLAY HITS! +$${payout}</span></div>`;
    } else if (busted) {
      const bustedNames = bustedLegs.map(l => l.betTeam).join(', ');
      statusHtml = `<div class="pick-live live-loss"><span class="live-label">${allFinal ? 'FINAL' : 'BUSTED'}</span><span class="live-result">Lost on: ${bustedNames}</span></div>`;
    } else if (finalLegs.length > 0) {
      statusHtml = `<div class="pick-live live-active"><span class="live-label">LIVE</span><span class="live-result">${finalLegs.length}/${allLegs.length} legs hit so far</span></div>`;
    } else {
      statusHtml = `<div class="pick-live live-scheduled"><span class="live-label">PRE-GAME</span><span class="live-status">All legs must win</span></div>`;
    }

    const legsHtml = allLegs.map(l => {
      const oppTeam = l.isHome ? l.game.away_team : l.game.home_team;
      let legStatus = '';
      if (l.game.status === 'STATUS_FINAL') {
        const favScore = l.isHome ? l.game.home_score : l.game.away_score;
        const oppScore = l.isHome ? l.game.away_score : l.game.home_score;
        const won = favScore > oppScore;
        legStatus = `<span class="badge ${won ? 'result-win' : 'result-loss'}">${won ? 'W' : 'L'}</span>`;
      }
      return `<div class="parlay-leg">${l.betTeam} ML (${l.mlOdds}) vs ${oppTeam} ${legStatus}</div>`;
    }).join('');

    return `
      <h3 class="section-title">Play 5 — Daily Parlay</h3>
      <div class="pick-card conf-parlay">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 5</span>
          <span class="pick-conf">${allLegs.length}-LEG PARLAY</span>
        </div>
        <div class="pick-bet-line">
          $100 to win $${payout}
        </div>
        <div class="parlay-legs">
          ${legsHtml}
        </div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Legs</span>
            <span class="detail-value">${allLegs.length}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Combined Odds</span>
            <span class="detail-value">${decimalOdds.toFixed(2)}x</span>
          </div>
          <div class="detail">
            <span class="detail-label">Implied Prob</span>
            <span class="detail-value">${impliedProb}%</span>
          </div>
          <div class="detail">
            <span class="detail-label">Payout</span>
            <span class="detail-value">+$${payout}</span>
          </div>
        </div>
        ${statusHtml}
      </div>`;
  }

  function renderPACTCard(pick) {
    const g = pick.game;
    const tierClass = pick.tier === 'ELITE' ? 'conf-pact-elite' : pick.tier === 'HIGH' ? 'conf-pact-high' : 'conf-pact';
    const dirClass = pick.direction === 'OVER' ? 'pact-over' : 'pact-under';
    const dirLabel = pick.direction;

    const FACTOR_LABELS = {
      high_total: 'High Matchup Total',
      low_total: 'Low Matchup Total',
      both_trending_up: 'Both Teams Scoring More',
      both_trending_down: 'Both Teams Scoring Less',
      strong_defense: 'Strong Combined Defense',
      weak_defense: 'Weak Combined Defense',
      fast_pace: 'Fast Pace Matchup',
      slow_pace: 'Slow Pace Matchup',
    };
    const factorList = pick.factors.map(f => FACTOR_LABELS[f] || f).join(', ');

    let liveHtml = '';
    if (g.status === 'STATUS_FINAL') {
      const actualTotal = g.home_score + g.away_score;
      const isOver = pick.direction === 'OVER';
      const won = isOver ? actualTotal > pick.predTotal : actualTotal < pick.predTotal;
      liveHtml = `
        <div class="pick-live ${won ? 'live-win' : 'live-loss'}">
          <span class="live-label">FINAL</span>
          <span class="live-score">Total: ${actualTotal} (pred: ${pick.predTotal})</span>
          <span class="live-result">${won ? 'W' : 'L'}</span>
        </div>`;
    } else if (g.status === 'STATUS_IN_PROGRESS' || g.status === 'STATUS_HALFTIME') {
      const currentTotal = g.home_score + g.away_score;
      liveHtml = `
        <div class="pick-live live-active">
          <span class="live-label">LIVE Q${g.period} ${g.clock}</span>
          <span class="live-score">Running total: ${currentTotal}</span>
        </div>`;
    } else {
      liveHtml = `
        <div class="pick-live live-scheduled">
          <span class="live-label">${g.time}</span>
          <span class="live-status">Pre-Game — Bet ${dirLabel} at -110</span>
        </div>`;
    }

    return `
      <div class="pick-card ${tierClass}">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 6</span>
          <span class="pick-conf">${pick.tier} — PACT ${pick.pactStrength}</span>
        </div>
        <div class="pick-bet-line ${dirClass}">
          ${dirLabel} ${pick.predTotal} at -110
        </div>
        <div class="pick-matchup">
          ${teamName(g.away_team)} @ ${teamName(g.home_team)}
        </div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Direction</span>
            <span class="detail-value ${dirClass}">${dirLabel}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Pred Total</span>
            <span class="detail-value">${pick.predTotal}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Strength</span>
            <span class="detail-value">${pick.pactStrength}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Factors</span>
            <span class="detail-value">${pick.factors.length}</span>
          </div>
        </div>
        <div class="pact-factors">${factorList}</div>
        ${liveHtml}
      </div>`;
  }

  // ── Rendering: All Games Grid ──────────────────────────────────────────────

  function renderAllGames() {
    const grid = document.getElementById('all-games-grid');
    if (!grid || todayGames.length === 0) return;

    grid.innerHTML = todayGames.map(g => {
      const isP3 = todayPicks.some(p => p.game.id === g.id);
      const isP4 = todayCdsPicks.some(p => p.game.id === g.id);
      const pickClass = (isP3 || isP4) ? 'game-picked' : '';

      let statusHtml;
      if (g.status === 'STATUS_FINAL') {
        statusHtml = `<span class="game-status final">Final: ${g.home_score}-${g.away_score}</span>`;
      } else if (g.status === 'STATUS_IN_PROGRESS' || g.status === 'STATUS_HALFTIME') {
        statusHtml = `<span class="game-status live">Q${g.period} ${g.clock}: ${g.home_score}-${g.away_score}</span>`;
      } else {
        statusHtml = `<span class="game-status scheduled">${g.time}</span>`;
      }

      let badge = '';
      if (isP3 && isP4) badge = '<span class="game-badge">P3 + P4</span>';
      else if (isP3) badge = '<span class="game-badge">P3 PICK</span>';
      else if (isP4) badge = '<span class="game-badge">P4 CDS</span>';

      return `
        <div class="game-card ${pickClass}">
          <div class="game-teams">${g.away_team} @ ${g.home_team}</div>
          ${statusHtml}
          ${badge}
        </div>`;
    }).join('');
  }

  // ── Rendering: History ─────────────────────────────────────────────────────

  function renderHistory() {
    renderSpreadHistory();
    renderCDSHistory();
    renderParlayHistory();
    renderPACTHistory();
  }

  function renderSpreadHistory() {
    const tbody = document.getElementById('history-body');
    if (!tbody) return;

    let filtered = historyPicks;
    if (currentHistoryPeriod !== 'all') {
      const cutoff = getCutoffDate(parseInt(currentHistoryPeriod));
      filtered = historyPicks.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="muted">No picks in this period</td></tr>';
    } else {
      tbody.innerHTML = filtered.map(p => {
        const resClass = p.favWon ? 'result-win' : 'result-loss';
        const resText = p.favWon ? 'W' : 'L';
        const pnlText = p.favWon ? '+$' + moneylineProfit(p.mlOdds) : '-$100';
        const confClass = p.confidence === 'HIGH' ? 'conf-high' : 'conf-strong';
        const dateFormatted = formatDate(p.date);

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td><strong>${p.favorite}</strong> ML ${p.mlOdds}</td>
            <td>${p.favorite} ${p.favIsHome ? 'vs' : '@'} ${p.underdog}</td>
            <td><span class="badge ${confClass}">${p.confidence}</span></td>
            <td>${p.predictedMargin.toFixed(1)}</td>
            <td>${p.actualMargin}</td>
            <td><span class="badge ${resClass}">${resText}</span></td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderHistorySummary(filtered);
  }

  function renderCDSHistory() {
    const tbody = document.getElementById('cds-history-body');
    if (!tbody) return;

    let picks = cdsHistoryPicks;
    if (currentCdsHistoryPeriod !== 'all') {
      const cutoff = getCutoffDate(parseInt(currentCdsHistoryPeriod));
      picks = cdsHistoryPicks.filter(p => p.date >= cutoff);
    }

    const filtered = [...picks].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" class="muted">No CDS picks found</td></tr>';
    } else {
      const PATHWAY_LABELS = {
        net_gap: 'Net Gap',
        offense: 'Offense',
        defense: 'Defense',
        consistency: 'Consistency',
        home: 'Home',
        dual_edge: 'Two-Way',
        opp_weakness: 'Opp Weak',
      };

      tbody.innerHTML = filtered.map(p => {
        const resClass = p.favWon ? 'result-win' : 'result-loss';
        const resText = p.favWon ? 'W' : 'L';
        const pnlText = p.favWon ? '+$' + moneylineProfit(p.mlOdds) : '-$100';
        const tierClass = p.tier === 'ELITE' ? 'conf-diamond' : p.tier === 'HIGH' ? 'conf-high' : 'conf-strong';
        const dateFormatted = formatDate(p.date);
        const pathwayLabel = PATHWAY_LABELS[p.pathway] || p.pathway;

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td><strong>${p.favorite}</strong> ML ${p.mlOdds}</td>
            <td>${p.favorite} ${p.favIsHome ? 'vs' : '@'} ${p.underdog}</td>
            <td><strong>${p.cdsScore}</strong></td>
            <td><span class="badge ${tierClass}">${p.tier}</span></td>
            <td>${pathwayLabel}</td>
            <td>${p.predictedMargin.toFixed(1)}</td>
            <td>${p.actualMargin}</td>
            <td><span class="badge ${resClass}">${resText}</span></td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderCDSSummary(filtered);
  }

  function renderHistorySummary(filtered) {
    const summary = document.getElementById('history-summary');
    if (!summary) return;

    const wins = filtered.filter(p => p.favWon).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const acc = total > 0 ? ((wins / total) * 100).toFixed(1) : '0';

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card">
          <div class="summary-title">Play 3 — Moneyline Bets</div>
          <div class="summary-stat">
            <span class="summary-record">${wins}-${total - wins}</span>
            <span class="summary-pct">${acc}%</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl}
          </div>
        </div>
      </div>`;
  }

  function renderCDSSummary(filtered) {
    const summary = document.getElementById('cds-history-summary');
    if (!summary) return;

    const wins = filtered.filter(p => p.favWon).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const acc = total > 0 ? ((wins / total) * 100).toFixed(1) : '—';

    const eliteGames = filtered.filter(p => p.tier === 'ELITE');
    const eliteWins = eliteGames.filter(p => p.favWon).length;
    const eliteAcc = eliteGames.length > 0 ? ((eliteWins / eliteGames.length) * 100).toFixed(1) : '—';

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-cyan">
          <div class="summary-title">Play 4 — CDS Incremental</div>
          <div class="summary-stat">
            <span class="summary-record">${total > 0 ? wins + '-' + (total - wins) : '—'}</span>
            <span class="summary-pct">${acc}${total > 0 ? '%' : ''}</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${total > 0 ? (pnl >= 0 ? '+' : '') + '$' + pnl : '—'}
          </div>
        </div>
        <div class="summary-card summary-diamond">
          <div class="summary-title">ELITE Tier Only</div>
          <div class="summary-stat">
            <span class="summary-record">${eliteGames.length > 0 ? eliteWins + '-' + (eliteGames.length - eliteWins) : '—'}</span>
            <span class="summary-pct">${eliteAcc}${eliteGames.length > 0 ? '%' : ''}</span>
          </div>
        </div>
        <div class="summary-card summary-combined">
          <div class="summary-title">Combined (P3 + P4)</div>
          <div class="summary-pnl-big ${(pnl + historyPicks.reduce((s, p) => s + p.pnl, 0)) >= 0 ? 'result-win' : 'result-loss'}">
            ${(() => {
              const combinedPnl = pnl + historyPicks.reduce((s, p) => s + p.pnl, 0);
              return (combinedPnl >= 0 ? '+' : '') + '$' + combinedPnl;
            })()}
          </div>
        </div>
      </div>`;
  }

  function renderParlayHistory() {
    const tbody = document.getElementById('parlay-history-body');
    if (!tbody) return;

    let filtered = parlayHistory;
    if (currentParlayHistoryPeriod !== 'all') {
      const cutoff = getCutoffDate(parseInt(currentParlayHistoryPeriod));
      filtered = parlayHistory.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="muted">No parlays in this period</td></tr>';
    } else {
      tbody.innerHTML = filtered.map(p => {
        const resClass = p.allWon ? 'result-win' : 'result-loss';
        const resText = p.allWon ? 'HIT' : 'MISS';
        const pnlText = p.allWon ? '+$' + p.payout : '-$100';
        const dateFormatted = formatDate(p.date);
        const legsDetail = p.legs.map(l => l.favorite + ' (' + l.mlOdds + ')').join(', ');
        const lossDetail = p.allWon ? '' : ` — ${p.losses} leg${p.losses > 1 ? 's' : ''} lost`;

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td>${p.legCount}-leg</td>
            <td class="parlay-legs-cell" title="${legsDetail}">${legsDetail}</td>
            <td>${p.parlayDecimal.toFixed(2)}x</td>
            <td>+$${p.payout}</td>
            <td><span class="badge ${resClass}">${resText}</span>${lossDetail}</td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderParlaySummary(filtered);
  }

  function renderParlaySummary(filtered) {
    const summary = document.getElementById('parlay-history-summary');
    if (!summary) return;

    const hits = filtered.filter(p => p.allWon).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';
    const totalWagered = total * 100;
    const roi = totalWagered > 0 ? ((pnl / totalWagered) * 100).toFixed(0) : '0';
    const avgPayout = hits > 0 ? Math.round(filtered.filter(p => p.allWon).reduce((s, p) => s + p.payout, 0) / hits) : 0;
    const avgLegs = total > 0 ? (filtered.reduce((s, p) => s + p.legCount, 0) / total).toFixed(1) : '0';

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-parlay">
          <div class="summary-title">Play 5 — Daily Parlay</div>
          <div class="summary-stat">
            <span class="summary-record">${hits}-${total - hits}</span>
            <span class="summary-pct">${hitRate}% hit rate</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl}
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">Parlay Stats</div>
          <div class="summary-stat">
            <span class="summary-record">Avg ${avgLegs} legs</span>
            <span class="summary-pct">Avg +$${avgPayout} payout</span>
          </div>
          <div class="summary-pnl ${roi >= 0 ? 'result-win' : 'result-loss'}">
            ROI: ${roi >= 0 ? '+' : ''}${roi}%
          </div>
        </div>
      </div>`;
  }

  function renderPACTHistory() {
    const tbody = document.getElementById('pact-history-body');
    if (!tbody) return;

    let filtered = pactHistoryPicks;
    if (currentPactHistoryPeriod !== 'all') {
      const cutoff = getCutoffDate(parseInt(currentPactHistoryPeriod));
      filtered = pactHistoryPicks.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="muted">No PACT picks in this period</td></tr>';
    } else {
      const FACTOR_SHORT = {
        high_total: 'High Total',
        low_total: 'Low Total',
        both_trending_up: 'Trend Up',
        both_trending_down: 'Trend Down',
        strong_defense: 'Strong Def',
        weak_defense: 'Weak Def',
        fast_pace: 'Fast Pace',
        slow_pace: 'Slow Pace',
      };

      tbody.innerHTML = filtered.map(p => {
        const resClass = p.hit ? 'result-win' : 'result-loss';
        const resText = p.hit ? 'W' : 'L';
        const pnlText = p.hit ? '+$91' : '-$100';
        const tierClass = p.tier === 'ELITE' ? 'conf-pact-elite' : p.tier === 'HIGH' ? 'conf-pact-high' : 'conf-pact';
        const dirClass = p.direction === 'OVER' ? 'pact-over' : 'pact-under';
        const dateFormatted = formatDate(p.date);
        const factorStr = p.factors.map(f => FACTOR_SHORT[f] || f).join(', ');

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td><span class="${dirClass}">${p.direction}</span></td>
            <td>${p.away} @ ${p.home}</td>
            <td><span class="badge ${tierClass}">${p.tier}</span></td>
            <td>${p.pactStrength}</td>
            <td>${p.predTotal}</td>
            <td>${p.actualTotal}</td>
            <td><span class="badge ${resClass}">${resText}</span></td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderPACTSummary(filtered);
  }

  function renderPACTSummary(filtered) {
    const summary = document.getElementById('pact-history-summary');
    if (!summary) return;

    const hits = filtered.filter(p => p.hit).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';
    const totalWagered = total * 100;
    const roi = totalWagered > 0 ? ((pnl / totalWagered) * 100).toFixed(0) : '0';

    const overs = filtered.filter(p => p.direction === 'OVER');
    const unders = filtered.filter(p => p.direction === 'UNDER');
    const overHits = overs.filter(p => p.hit).length;
    const underHits = unders.filter(p => p.hit).length;
    const overPnl = overs.reduce((s, p) => s + p.pnl, 0);
    const underPnl = unders.reduce((s, p) => s + p.pnl, 0);

    const elites = filtered.filter(p => p.tier === 'ELITE');
    const eliteHits = elites.filter(p => p.hit).length;
    const eliteRate = elites.length > 0 ? ((eliteHits / elites.length) * 100).toFixed(1) : '—';

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-pact">
          <div class="summary-title">Play 6 — PACT Totals (at -110)</div>
          <div class="summary-stat">
            <span class="summary-record">${hits}-${total - hits}</span>
            <span class="summary-pct">${hitRate}%</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl} (ROI: ${roi >= 0 ? '+' : ''}${roi}%)
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">OVER / UNDER Breakdown</div>
          <div class="summary-stat">
            <span class="summary-record">OVER: ${overHits}-${overs.length - overHits}</span>
            <span class="summary-pct">${overs.length > 0 ? (overHits / overs.length * 100).toFixed(1) + '%' : '—'}</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">UNDER: ${underHits}-${unders.length - underHits}</span>
            <span class="summary-pct">${unders.length > 0 ? (underHits / unders.length * 100).toFixed(1) + '%' : '—'}</span>
          </div>
        </div>
        <div class="summary-card summary-pact-elite">
          <div class="summary-title">ELITE Tier Only</div>
          <div class="summary-stat">
            <span class="summary-record">${elites.length > 0 ? eliteHits + '-' + (elites.length - eliteHits) : '—'}</span>
            <span class="summary-pct">${eliteRate}${elites.length > 0 ? '%' : ''}</span>
          </div>
        </div>
      </div>`;
  }

  // ── Metrics ────────────────────────────────────────────────────────────────

  function updateMetrics() {
    const el = (id) => document.getElementById(id);

    el('metric-picks').textContent = todayPicks.length + todayCdsPicks.length;
    el('metric-games').textContent = todayGames.length;

    if (historyPicks.length > 0) {
      const wins = historyPicks.filter(p => p.favWon).length;
      const total = historyPicks.length;
      const accuracy = ((wins / total) * 100).toFixed(1);
      const totalPnl = historyPicks.reduce((s, p) => s + p.pnl, 0);

      el('metric-accuracy').textContent = accuracy + '%';
      el('metric-record').textContent = `${wins}-${total - wins}`;

      // CDS accuracy
      if (cdsHistoryPicks.length > 0) {
        const cdsWins = cdsHistoryPicks.filter(p => p.favWon).length;
        const cdsTotal = cdsHistoryPicks.length;
        const cdsAcc = ((cdsWins / cdsTotal) * 100).toFixed(1);
        const cdsPnl = cdsHistoryPicks.reduce((s, p) => s + p.pnl, 0);
        el('metric-cds-accuracy').textContent = cdsAcc + '%';

        // Combined ROI
        const combinedBets = total + cdsTotal;
        const combinedPnl = totalPnl + cdsPnl;
        const roi = combinedBets > 0 ? ((combinedPnl / (combinedBets * 100)) * 100).toFixed(0) : '0';
        el('metric-roi').textContent = (roi >= 0 ? '+' : '') + roi + '%';

        // Combined record
        const combinedWins = wins + cdsWins;
        const combinedTotal = total + cdsTotal;
        el('metric-record').textContent = `${combinedWins}-${combinedTotal - combinedWins}`;
      } else {
        const roi = total > 0 ? ((totalPnl / (total * 100)) * 100).toFixed(0) : '0';
        el('metric-roi').textContent = (roi >= 0 ? '+' : '') + roi + '%';
      }
    }
  }

  // ── Live Score Updates ─────────────────────────────────────────────────────

  async function refreshScores() {
    if (!modelReady) return;
    try {
      await fetchTodayGames();
      renderPicks();
      renderAllGames();
      updateMetrics();
    } catch (e) {
      console.warn('[APP] Score refresh failed:', e);
    }
  }

  // ── Navigation ─────────────────────────────────────────────────────────────

  function setupNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        if (view === currentView) return;

        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        const target = document.getElementById('view-' + view);
        if (target) target.classList.add('active');

        currentView = view;
      });
    });
  }

  function setupHistoryFilters() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.cds-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.cds-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentCdsHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.parlay-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.parlay-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentParlayHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.pact-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.pact-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentPactHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });
  }

  // ── Status ─────────────────────────────────────────────────────────────────

  function setStatus(state, text) {
    const dot = document.getElementById('status-dot');
    const label = document.getElementById('status-text');
    dot.className = 'status-dot ' + state;
    label.textContent = text;
  }

  // ── Utilities ──────────────────────────────────────────────────────────────

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  function getRecentDates(days) {
    const dates = [];
    const now = new Date();
    for (let i = 1; i <= days; i++) {
      const d = new Date(now);
      d.setDate(d.getDate() - i);
      dates.push(d.toISOString().slice(0, 10).replace(/-/g, ''));
    }
    return dates;
  }

  function getCutoffDate(days) {
    const d = new Date();
    d.setDate(d.getDate() - days);
    return d.toISOString().slice(0, 10).replace(/-/g, '');
  }

  function formatDate(dateStr) {
    if (!dateStr || dateStr.length !== 8) return dateStr;
    return dateStr.slice(4, 6) + '/' + dateStr.slice(6, 8) + '/' + dateStr.slice(0, 4);
  }

  // ── Boot ───────────────────────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
