// =============================================================================
// MAIN APP CONTROLLER — NBA Dominance System (Play 4 CDS + Play 7 PRISM + Play 8 NOVA)
// =============================================================================

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────────────
  let currentView = 'picks';
  let todayGames = [];          // All games today
  let todayCdsPicks = [];       // Games flagged by CDS model (Play 4: moneyline)
  let cdsHistoryPicks = [];     // Historical Play 4 CDS picks with results
  let todayPrismPicks = [];      // Games flagged by PRISM model (Play 7: convergent -110)
  let prismHistoryPicks = [];    // Historical Play 7 PRISM picks with results
  let todayNovaPicks = [];       // Games flagged by NOVA model (Play 8: compound OVER -110)
  let novaHistoryPicks = [];     // Historical Play 8 NOVA picks with results
  let todayFusionParlays = [];   // Play 9: FUSION parlays (2-leg -110/-110 = +264)
  let fusionHistoryPicks = [];   // Historical Play 9 FUSION parlay results
  let todayPulseParlays = [];    // Play 10: PULSE player prop UNDER parlays (+251)
  let pulseHistoryPicks = [];    // Historical Play 10 PULSE parlay results
  let todayVaultParlays = [];    // Play 11: VAULT multi-leg floor parlays
  let vaultHistoryPicks = [];    // Historical Play 11 VAULT parlay results
  let todayFortressParlays = []; // Play 12: FORTRESS higher-floor parlays
  let fortressHistoryPicks = []; // Historical Play 12 FORTRESS parlay results
  let todaySiegeParlays = [];    // Play 13: SIEGE sportsbook-aligned parlays
  let siegeHistoryPicks = [];    // Historical Play 13 SIEGE parlay results
  let playerBoxScores = [];      // Player box score data for PULSE/VAULT/FORTRESS/SIEGE models
  let seasonData = [];          // Full season game data for model training
  let modelReady = false;
  let currentCdsHistoryPeriod = 'all';
  let currentPrismHistoryPeriod = 'all';
  let currentNovaHistoryPeriod = 'all';
  let currentFusionHistoryPeriod = 'all';
  let currentPulseHistoryPeriod = 'all';
  let currentVaultHistoryPeriod = 'all';
  let currentFortressHistoryPeriod = 'all';
  let currentSiegeHistoryPeriod = 'all';
  let trainedPulseModel = null;  // PULSE model trained on all historical data
  let trainedVaultModel = null;  // VAULT model trained on all historical data
  let trainedFortressModel = null; // FORTRESS model trained on all historical data
  let trainedSiegeModel = null;  // SIEGE model trained on all historical data
  let playerTeamMap = {};        // player name -> most recent team abbreviation
  let useProxy = false;        // True when running on Vercel (CORS proxy available)
  let masterBacktestResults = null; // Master Bet Recommender backtest results
  let masterTodayPicks = [];       // Master Bet Recommender today's picks
  let currentMasterHistoryPeriod = 'all';

  const PRISMModel = window.ParlayEngine.PRISMModel;
  const NOVAModel = window.ParlayEngine.NOVAModel;
  const PULSEModel = window.ParlayEngine.PULSEModel;
  const VAULTModel = window.ParlayEngine.VAULTModel;
  const FORTRESSModel = window.ParlayEngine.FORTRESSModel;
  const SIEGEModel = window.ParlayEngine.SIEGEModel;

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

      const netGap = Math.abs(netDiff);

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
      runCDSPredictions();
      runPRISMPredictions();
      runNOVAPredictions();
      buildFusionParlays();
      await loadPlayerData();
      buildPulseHistory();
      buildVaultHistory();
      buildFortressHistory();
      buildSiegeHistory();
      generateTodayPulsePicks();
      generateTodayVaultPicks();
      generateTodayFortressPicks();
      await generateTodaySiegePicks();
      runMasterBacktest();
      await generateMasterTodayPicks();
      buildCDSHistory();
      buildPRISMHistory();
      buildNOVAHistory();
      buildFusionHistory();
      renderPicks();
      renderAllGames();
      renderHistory();
      renderMasterView();
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
    CDSModel.reset();
    PRISMModel.teamHistory = {};
    NOVAModel.teamHistory = {};
    for (const g of seasonData) {
      CDSModel.updateTeam(g.home_team, g.home_score, g.away_score, g.date);
      CDSModel.updateTeam(g.away_team, g.away_score, g.home_score, g.date);
      PRISMModel.updateTeam(g.home_team, g.home_score, g.away_score, g.date);
      PRISMModel.updateTeam(g.away_team, g.away_score, g.home_score, g.date);
      NOVAModel.updateTeam(g.home_team, g.home_score, g.away_score, g.date);
      NOVAModel.updateTeam(g.away_team, g.away_score, g.home_score, g.date);
    }

    console.log(`[APP] Models trained on ${seasonData.length} games, ${Object.keys(CDSModel.teamHistory).length} teams`);
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

  // ── PRISM Predictions (Play 7) ──────────────────────────────────────────

  function runPRISMPredictions() {
    todayPrismPicks = [];

    for (const game of todayGames) {
      const picks = PRISMModel.predictGame(game.home_team, game.away_team);
      if (picks) {
        for (const pick of picks) {
          todayPrismPicks.push({ game, ...pick });
        }
      }
    }

    const tierOrder = { ELITE: 0, HIGH: 1, STRONG: 2 };
    todayPrismPicks.sort((a, b) => {
      const to = (tierOrder[a.tier] || 9) - (tierOrder[b.tier] || 9);
      if (to !== 0) return to;
      return b.strength - a.strength;
    });

    console.log(`[PRISM] ${todayPrismPicks.length} picks from ${todayGames.length} games`);
  }

  // ── History Builders ─────────────────────────────────────────────────────

  function buildCDSHistory() {
    cdsHistoryPicks = [];

    // Independent CDS model instance for walk-forward backtesting
    const cds = Object.create(CDSModel);
    cds.teamHistory = {};

    const sorted = [...seasonData].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    for (const game of sorted) {
      const pred = cds.predictGame(game.home_team, game.away_team);

      if (pred) {
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

    console.log(`[CDS] Built history: ${cdsHistoryPicks.length} picks`);
  }

  function buildPRISMHistory() {
    prismHistoryPicks = [];

    const prism = Object.create(PRISMModel);
    prism.teamHistory = {};

    const sorted = [...seasonData].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    for (const game of sorted) {
      const picks = prism.predictGame(game.home_team, game.away_team);

      if (picks) {
        const actualTotal = game.home_score + game.away_score;
        const actualHomeMargin = game.home_score - game.away_score;

        for (const pick of picks) {
          let hit = false;
          if (pick.type === 'total') {
            hit = pick.direction === 'OVER'
              ? actualTotal > pick.predTotal
              : actualTotal < pick.predTotal;
          } else if (pick.type === 'spread') {
            // "AWAY covers" means actual home margin < predicted home margin
            hit = actualHomeMargin < pick.predMargin;
          }

          prismHistoryPicks.push({
            date: game.date,
            home: game.home_team,
            away: game.away_team,
            type: pick.type,
            direction: pick.direction,
            predTotal: pick.predTotal || null,
            predMargin: pick.predMargin || null,
            betTeam: pick.betTeam || null,
            strength: pick.strength,
            tier: pick.tier,
            factors: pick.factors,
            actualTotal,
            actualHomeMargin,
            hit,
            pnl: hit ? 91 : -100,
          });
        }
      }

      prism.updateTeam(game.home_team, game.home_score, game.away_score, game.date);
      prism.updateTeam(game.away_team, game.away_score, game.home_score, game.date);
    }

    console.log(`[PRISM] Built history: ${prismHistoryPicks.length} total picks`);
  }

  // ── NOVA Predictions (Play 8) ──────────────────────────────────────────────

  function runNOVAPredictions() {
    todayNovaPicks = [];

    for (const game of todayGames) {
      const picks = NOVAModel.predictGame(game.home_team, game.away_team);
      if (picks) {
        for (const pick of picks) {
          todayNovaPicks.push({ game, ...pick });
        }
      }
    }

    const tierOrder = { ELITE: 0, HIGH: 1, STRONG: 2 };
    todayNovaPicks.sort((a, b) => {
      const to = (tierOrder[a.tier] || 9) - (tierOrder[b.tier] || 9);
      if (to !== 0) return to;
      return b.strength - a.strength;
    });

    console.log(`[NOVA] ${todayNovaPicks.length} picks from ${todayGames.length} games`);
  }

  function buildNOVAHistory() {
    novaHistoryPicks = [];

    const nova = Object.create(NOVAModel);
    nova.teamHistory = {};

    const sorted = [...seasonData].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    for (const game of sorted) {
      const picks = nova.predictGame(game.home_team, game.away_team);

      if (picks) {
        const actualTotal = game.home_score + game.away_score;

        for (const pick of picks) {
          const hit = actualTotal > pick.predTotal;

          novaHistoryPicks.push({
            date: game.date,
            home: game.home_team,
            away: game.away_team,
            predTotal: pick.predTotal,
            strength: pick.strength,
            tier: pick.tier,
            factors: pick.factors,
            hPpg5: pick.hPpg5,
            aPpg5: pick.aPpg5,
            hAvgTotal3: pick.hAvgTotal3,
            aAvgTotal3: pick.aAvgTotal3,
            actualTotal,
            hit,
            pnl: hit ? 91 : -100,
          });
        }
      }

      nova.updateTeam(game.home_team, game.home_score, game.away_score, game.date);
      nova.updateTeam(game.away_team, game.away_score, game.home_score, game.date);
    }

    console.log(`[NOVA] Built history: ${novaHistoryPicks.length} total picks`);
  }

  // ── FUSION Parlays (Play 9) ────────────────────────────────────────────────
  // Combines 2 independent -110 picks from different models/games into a
  // 2-leg parlay at +264 American odds. Win = +$264, Loss = -$100.
  // Priority: NOVA OVER > PRISM Spread > PRISM OVER > PRISM UNDER

  function buildFusionParlays() {
    todayFusionParlays = [];

    // Group picks by signal type
    const novaOvers = todayNovaPicks.filter(p => p.type === 'total' && p.direction === 'OVER');
    const prismSpreads = todayPrismPicks.filter(p => p.type === 'spread');
    const prismOvers = todayPrismPicks.filter(p => p.type === 'total' && p.direction === 'OVER');
    const prismUnders = todayPrismPicks.filter(p => p.type === 'total' && p.direction === 'UNDER');

    // Priority-ordered pick pool: NOVA > spread > over > under
    const pool = [
      ...novaOvers.map(p => ({ ...p, source: 'NOVA', label: `OVER ${p.predTotal}`, gameKey: p.game.id })),
      ...prismSpreads.map(p => ({ ...p, source: 'PRISM', label: `${p.betTeam} covers`, gameKey: p.game.id })),
      ...prismOvers.map(p => ({ ...p, source: 'PRISM', label: `OVER ${p.predTotal}`, gameKey: p.game.id })),
      ...prismUnders.map(p => ({ ...p, source: 'PRISM', label: `UNDER ${p.predTotal}`, gameKey: p.game.id })),
    ];

    if (pool.length < 2) return;

    // Select best 2 from different games AND different signal types
    const selected = [];
    const usedGames = new Set();
    const usedSources = new Set();

    for (const pick of pool) {
      if (usedGames.has(pick.gameKey)) continue;
      // Prefer cross-model parlays (NOVA+PRISM), then cross-type
      if (selected.length === 1 && usedSources.has(pick.source) && pool.some(p => !usedGames.has(p.gameKey) && !usedSources.has(p.source))) continue;
      selected.push(pick);
      usedGames.add(pick.gameKey);
      usedSources.add(pick.source);
      if (selected.length >= 2) break;
    }

    if (selected.length < 2) return;

    todayFusionParlays.push({
      leg1: selected[0],
      leg2: selected[1],
      parlayOdds: '+264',
      parlayDecimal: 3.64,
      winPayout: 264,
    });

    console.log(`[FUSION] ${todayFusionParlays.length} parlays built`);
  }

  function buildFusionHistory() {
    fusionHistoryPicks = [];

    const prism = Object.create(PRISMModel);
    prism.teamHistory = {};
    const nova = Object.create(NOVAModel);
    nova.teamHistory = {};

    const sorted = [...seasonData].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    // Group games by date for parlay building
    const gamesByDate = {};
    for (const game of sorted) {
      if (!gamesByDate[game.date]) gamesByDate[game.date] = [];
      gamesByDate[game.date].push(game);
    }

    const processedDates = new Set();

    for (const game of sorted) {
      const date = game.date;

      // After updating all games for this date, build parlays
      if (!processedDates.has(date)) {
        // First, generate all picks for all games on this date
        const dayGames = gamesByDate[date] || [];
        const dayPicks = [];

        for (const dg of dayGames) {
          // PRISM picks
          const prismPicks = prism.predictGame(dg.home_team, dg.away_team);
          if (prismPicks) {
            const actualTotal = dg.home_score + dg.away_score;
            const actualHM = dg.home_score - dg.away_score;
            for (const pick of prismPicks) {
              let hit = false;
              if (pick.type === 'total') {
                hit = pick.direction === 'OVER' ? actualTotal > pick.predTotal : actualTotal < pick.predTotal;
              } else if (pick.type === 'spread') {
                hit = actualHM < pick.predMargin;
              }
              dayPicks.push({ ...pick, source: 'PRISM', hit, game: dg, label: pick.type === 'total' ? `${pick.direction} ${pick.predTotal}` : `${pick.betTeam} covers` });
            }
          }

          // NOVA picks
          const novaPicks = nova.predictGame(dg.home_team, dg.away_team);
          if (novaPicks) {
            const actualTotal = dg.home_score + dg.away_score;
            for (const pick of novaPicks) {
              const hit = actualTotal > pick.predTotal;
              dayPicks.push({ ...pick, source: 'NOVA', hit, game: dg, label: `OVER ${pick.predTotal}` });
            }
          }
        }

        // Build best 2-leg parlay from different games + preferably different sources
        const pool = [
          ...dayPicks.filter(p => p.source === 'NOVA'),
          ...dayPicks.filter(p => p.source === 'PRISM' && p.type === 'spread'),
          ...dayPicks.filter(p => p.source === 'PRISM' && p.type === 'total' && p.direction === 'OVER'),
          ...dayPicks.filter(p => p.source === 'PRISM' && p.type === 'total' && p.direction === 'UNDER'),
        ];

        if (pool.length >= 2) {
          const selected = [];
          const usedGames = new Set();
          const usedSources = new Set();

          for (const pick of pool) {
            const gk = pick.game.home_team + pick.game.away_team;
            if (usedGames.has(gk)) continue;
            if (selected.length === 1 && usedSources.has(pick.source)) {
              // Check if a cross-source pick is available
              const crossAvail = pool.some(p => {
                const pk = p.game.home_team + p.game.away_team;
                return !usedGames.has(pk) && !usedSources.has(p.source);
              });
              if (crossAvail) continue;
            }
            selected.push(pick);
            usedGames.add(gk);
            usedSources.add(pick.source);
            if (selected.length >= 2) break;
          }

          if (selected.length >= 2) {
            const allHit = selected.every(p => p.hit);
            fusionHistoryPicks.push({
              date,
              leg1Source: selected[0].source,
              leg1Label: selected[0].label,
              leg1Game: `${selected[0].game.away_team}@${selected[0].game.home_team}`,
              leg1Hit: selected[0].hit,
              leg2Source: selected[1].source,
              leg2Label: selected[1].label,
              leg2Game: `${selected[1].game.away_team}@${selected[1].game.home_team}`,
              leg2Hit: selected[1].hit,
              hit: allHit,
              pnl: allHit ? 264 : -100,
            });
          }
        }

        processedDates.add(date);
      }

      // Update models with this game
      prism.updateTeam(game.home_team, game.home_score, game.away_score, game.date);
      prism.updateTeam(game.away_team, game.away_score, game.home_score, game.date);
      nova.updateTeam(game.home_team, game.home_score, game.away_score, game.date);
      nova.updateTeam(game.away_team, game.away_score, game.home_score, game.date);
    }

    console.log(`[FUSION] Built history: ${fusionHistoryPicks.length} parlays`);
  }

  // ── PULSE Player Props (Play 10) ──────────────────────────────────────────

  async function loadPlayerData() {
    try {
      const resp = await fetch('data/player_boxscores.json');
      if (resp.ok) {
        playerBoxScores = await resp.json();
        console.log(`[PULSE] Loaded ${playerBoxScores.length} games with player data`);
      }
    } catch (e) {
      console.warn('[PULSE] Could not load player box scores');
    }

    // Fetch recent player box scores from ESPN to keep PULSE/VAULT models current
    await fetchRecentPlayerBoxScores();
  }

  async function fetchRecentPlayerBoxScores() {
    const existingDates = new Set((playerBoxScores || []).map(g => g.date));
    const recentDates = getRecentDates(30);
    let fetched = 0;

    for (const dateStr of recentDates) {
      if (existingDates.has(dateStr)) continue;

      try {
        // First get event IDs from scoreboard for this date
        const sbUrl = useProxy
          ? `/api/nba?endpoint=espn_scoreboard&dates=${dateStr}`
          : `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=${dateStr}`;
        const sbResp = await fetch(sbUrl);
        if (!sbResp.ok) continue;
        const sbData = await sbResp.json();

        for (const event of (sbData.events || [])) {
          const comp = (event.competitions || [{}])[0];
          const status = (comp.status || {}).type || {};
          if (status.name !== 'STATUS_FINAL') continue;

          const eventId = event.id;
          try {
            const sumUrl = useProxy
              ? `/api/nba?endpoint=espn_summary&eventId=${eventId}`
              : `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=${eventId}`;
            const sumResp = await fetch(sumUrl);
            if (!sumResp.ok) continue;
            const sumData = await sumResp.json();

            const boxscore = sumData.boxscore;
            if (!boxscore || !boxscore.teams || !boxscore.players) continue;

            const competitors = comp.competitors || [];
            const homeComp = competitors.find(c => c.homeAway === 'home') || competitors[0];
            const awayComp = competitors.find(c => c.homeAway === 'away') || competitors[1];
            const homeTeam = norm(homeComp.team.abbreviation);
            const awayTeam = norm(awayComp.team.abbreviation);
            const homeScore = parseInt(homeComp.score) || 0;
            const awayScore = parseInt(awayComp.score) || 0;

            const players = [];
            for (const teamPlayers of boxscore.players) {
              const teamAbbr = norm(teamPlayers.team.abbreviation);
              for (const statGroup of (teamPlayers.statistics || [])) {
                // Find column indices for pts, reb, ast, min
                const labels = (statGroup.labels || []).map(l => l.toLowerCase());
                const minIdx = labels.indexOf('min');
                const ptsIdx = labels.indexOf('pts');
                const rebIdx = labels.indexOf('reb');
                const astIdx = labels.indexOf('ast');

                for (const athlete of (statGroup.athletes || [])) {
                  const stats = athlete.stats || [];
                  if (stats.length === 0) continue;
                  const name = athlete.athlete ? athlete.athlete.displayName : '';
                  if (!name) continue;

                  const min = minIdx >= 0 ? stats[minIdx] : '0';
                  const pts = ptsIdx >= 0 ? parseInt(stats[ptsIdx]) || 0 : 0;
                  const reb = rebIdx >= 0 ? parseInt(stats[rebIdx]) || 0 : 0;
                  const ast = astIdx >= 0 ? parseInt(stats[astIdx]) || 0 : 0;

                  players.push({ team: teamAbbr, name, min, pts, reb, ast });
                }
              }
            }

            if (players.length > 0) {
              playerBoxScores.push({
                event_id: eventId,
                date: dateStr,
                home: homeTeam,
                away: awayTeam,
                home_score: homeScore,
                away_score: awayScore,
                players,
              });
            }

            await sleep(200);
          } catch (e) {
            console.warn(`[PULSE] Could not fetch summary for event ${eventId}:`, e.message);
          }
        }

        fetched++;
        if (fetched % 5 === 0) console.log(`[PULSE] Fetched player data for ${fetched} recent dates...`);
        await sleep(300);
      } catch (e) {
        console.warn(`[PULSE] Could not fetch scoreboard for ${dateStr}:`, e.message);
      }
    }

    if (fetched > 0) {
      console.log(`[PULSE] Augmented player data with ${fetched} recent dates. Total: ${playerBoxScores.length} games`);
    }
  }

  function parsePlayerMins(m) {
    if (typeof m === 'number') return m;
    try {
      return parseInt(m);
    } catch (e) {
      try {
        return parseInt(m.split(':')[0]);
      } catch (e2) {
        return 0;
      }
    }
  }

  function buildPulseHistory() {
    pulseHistoryPicks = [];
    if (!playerBoxScores || playerBoxScores.length === 0) return;

    const pulse = Object.create(PULSEModel);
    pulse.playerHistory = {};

    // Sort by date
    const sorted = [...playerBoxScores].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    // Group by date
    const byDate = {};
    for (const game of sorted) {
      if (!byDate[game.date]) byDate[game.date] = [];
      byDate[game.date].push(game);
    }

    const processedDates = new Set();

    for (const game of sorted) {
      const date = game.date;

      if (!processedDates.has(date)) {
        const dayGames = byDate[date] || [];
        const dayCandidates = [];

        // Generate UNDER picks for all games on this date
        for (const dg of dayGames) {
          const players = (dg.players || []).map(p => ({
            name: p.name,
            team: p.team,
            pts: p.pts,
            reb: p.reb,
            ast: p.ast,
            min: parsePlayerMins(p.min),
          }));

          const picks = pulse.predictGame(players);
          for (const pick of picks) {
            // Check actual result
            const actualPlayer = dg.players.find(pp => pp.name === pick.player);
            if (!actualPlayer) continue;

            const hit = actualPlayer.pts < pick.line;
            dayCandidates.push({
              ...pick,
              actual: actualPlayer.pts,
              hit,
              gameKey: `${dg.away}@${dg.home}`,
              gameDisplay: `${dg.away} @ ${dg.home}`,
            });
          }
        }

        // Select top 2 from different games
        dayCandidates.sort((a, b) => b.strength - a.strength);
        const selected = [];
        const usedGames = new Set();

        for (const c of dayCandidates) {
          if (usedGames.has(c.gameKey)) continue;
          selected.push(c);
          usedGames.add(c.gameKey);
          if (selected.length >= 2) break;
        }

        if (selected.length >= 2) {
          const allHit = selected.every(s => s.hit);
          pulseHistoryPicks.push({
            date,
            leg1Player: selected[0].player,
            leg1Team: selected[0].team,
            leg1Line: selected[0].line,
            leg1Actual: selected[0].actual,
            leg1Hit: selected[0].hit,
            leg1Spike: selected[0].spikePct,
            leg1Tier: selected[0].tier,
            leg1Game: selected[0].gameDisplay,
            leg2Player: selected[1].player,
            leg2Team: selected[1].team,
            leg2Line: selected[1].line,
            leg2Actual: selected[1].actual,
            leg2Hit: selected[1].hit,
            leg2Spike: selected[1].spikePct,
            leg2Tier: selected[1].tier,
            leg2Game: selected[1].gameDisplay,
            won: allHit,
            pnl: allHit ? 251 : -100,
          });
        }

        processedDates.add(date);
      }

      // Update PULSE model with all players from this game
      for (const p of (game.players || [])) {
        const mins = parsePlayerMins(p.min);
        if (mins < 10) continue;
        pulse.updatePlayer(p.name, p.pts, p.reb, p.ast, game.date);
      }
    }

    // Save trained model for today's predictions
    trainedPulseModel = pulse;

    // Build player → team map from all box scores
    for (const game of sorted) {
      for (const p of (game.players || [])) {
        playerTeamMap[p.name] = p.team;
      }
    }

    console.log(`[PULSE] Built history: ${pulseHistoryPicks.length} parlays`);
  }

  // ── VAULT History Builder ──────────────────────────────────────────────────

  function vaultParlayAmerican(numLegs) {
    // At -300 per leg: decimal = 1.333
    const dec = Math.pow(1 + 100 / 300, numLegs);
    return dec >= 2.0 ? Math.round((dec - 1) * 100) : Math.round(-100 / (dec - 1));
  }

  function buildVaultHistory() {
    vaultHistoryPicks = [];
    if (!playerBoxScores || playerBoxScores.length === 0) return;

    const vault = Object.create(VAULTModel);
    vault.playerHistory = {};

    const sorted = [...playerBoxScores].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    const byDate = {};
    for (const game of sorted) {
      if (!byDate[game.date]) byDate[game.date] = [];
      byDate[game.date].push(game);
    }

    const processedDates = new Set();

    for (const game of sorted) {
      const date = game.date;

      if (!processedDates.has(date)) {
        const dayGames = byDate[date] || [];
        const dayCandidates = [];

        for (const dg of dayGames) {
          const gameKey = `${dg.away}@${dg.home}`;
          const players = (dg.players || []).map(p => ({
            name: p.name,
            team: p.team,
            pts: p.pts,
            min: parsePlayerMins(p.min),
          }));

          const picks = vault.predictGame(players);
          for (const pick of picks) {
            const actualPlayer = dg.players.find(pp => pp.name === pick.player);
            if (!actualPlayer) continue;

            const hit = actualPlayer.pts > pick.line;
            dayCandidates.push({
              ...pick,
              actual: actualPlayer.pts,
              hit,
              gameKey,
              gameDisplay: `${dg.away} @ ${dg.home}`,
            });
          }
        }

        // Select top 4-8 legs from different games, sorted by confidence
        dayCandidates.sort((a, b) => b.confidence - a.confidence);
        const selected = [];
        const usedGames = new Set();

        for (const c of dayCandidates) {
          if (usedGames.has(c.gameKey)) continue;
          selected.push(c);
          usedGames.add(c.gameKey);
          if (selected.length >= 8) break;
        }

        if (selected.length >= 4) {
          const allHit = selected.every(s => s.hit);
          const odds = vaultParlayAmerican(selected.length);
          const pnl = allHit ? odds : -100;

          vaultHistoryPicks.push({
            date,
            legs: selected.map(s => ({
              player: s.player,
              team: s.team,
              line: s.line,
              actual: s.actual,
              hit: s.hit,
              confidence: s.confidence,
              l10Avg: s.l10Avg,
              l10Min: s.l10Min,
              gameDisplay: s.gameDisplay,
            })),
            numLegs: selected.length,
            odds,
            won: allHit,
            pnl,
          });
        }

        processedDates.add(date);
      }

      // Update VAULT model with all players from this game
      for (const p of (game.players || [])) {
        const mins = parsePlayerMins(p.min);
        if (mins < 10) continue;
        vault.updatePlayer(p.name, p.pts, mins, game.date);
      }
    }

    // Save trained model for today's predictions
    trainedVaultModel = vault;

    console.log(`[VAULT] Built history: ${vaultHistoryPicks.length} parlays`);
  }

  // ── Generate Today's PULSE + VAULT Picks ─────────────────────────────────

  function generateTodayPulsePicks() {
    todayPulseParlays = [];
    if (!trainedPulseModel || todayGames.length === 0) {
      console.log(`[PULSE] Skipping today's picks: model=${!!trainedPulseModel}, games=${todayGames.length}`);
      return;
    }

    try {
      const dayCandidates = [];

      for (const game of todayGames) {
        const gameKey = `${game.away_team}@${game.home_team}`;
        const teamPlayers = [];
        for (const [name, team] of Object.entries(playerTeamMap)) {
          if (team === game.home_team || team === game.away_team) {
            teamPlayers.push({ name, team, pts: 0, reb: 0, ast: 0, min: 30 });
          }
        }

        const picks = trainedPulseModel.predictGame(teamPlayers);
        for (const pick of picks) {
          dayCandidates.push({
            ...pick,
            gameKey,
            gameDisplay: `${game.away_team} @ ${game.home_team}`,
          });
        }
      }

      // Select top 2 from different games
      dayCandidates.sort((a, b) => b.strength - a.strength);
      const selected = [];
      const usedGames = new Set();
      for (const c of dayCandidates) {
        if (usedGames.has(c.gameKey)) continue;
        selected.push(c);
        usedGames.add(c.gameKey);
        if (selected.length >= 2) break;
      }

      if (selected.length >= 2) {
        todayPulseParlays.push({
          leg1: selected[0],
          leg2: selected[1],
        });
      }

      console.log(`[PULSE] Today's picks: ${todayPulseParlays.length} parlays from ${dayCandidates.length} candidates across ${todayGames.length} games`);
    } catch (e) {
      console.error('[PULSE] Error generating today picks:', e);
    }
  }

  function generateTodayVaultPicks() {
    todayVaultParlays = [];
    if (!trainedVaultModel || todayGames.length === 0) {
      console.log(`[VAULT] Skipping today's picks: model=${!!trainedVaultModel}, games=${todayGames.length}`);
      return;
    }

    try {
      const dayCandidates = [];

      for (const game of todayGames) {
        const gameKey = `${game.away_team}@${game.home_team}`;
        const teamPlayers = [];
        for (const [name, team] of Object.entries(playerTeamMap)) {
          if (team === game.home_team || team === game.away_team) {
            teamPlayers.push({ name, team, pts: 0, min: 30 });
          }
        }

        const picks = trainedVaultModel.predictGame(teamPlayers);
        for (const pick of picks) {
          dayCandidates.push({
            ...pick,
            gameKey,
            gameDisplay: `${game.away_team} @ ${game.home_team}`,
          });
        }
      }

      console.log(`[VAULT] ${dayCandidates.length} qualifying players across ${todayGames.length} games`);

      // Select top legs sorted by confidence, max 1 per game
      dayCandidates.sort((a, b) => b.confidence - a.confidence);
      const selected = [];
      const usedGames = new Set();
      for (const c of dayCandidates) {
        if (usedGames.has(c.gameKey)) continue;
        selected.push(c);
        usedGames.add(c.gameKey);
        if (selected.length >= 8) break;
      }

      // Build parlay if we have enough legs (minimum 4)
      if (selected.length >= 4) {
        const odds = vaultParlayAmerican(selected.length);
        todayVaultParlays.push({
          legs: selected.map(s => ({
            player: s.player,
            team: s.team,
            line: s.line,
            confidence: s.confidence,
            l10Avg: s.l10Avg,
            l10Min: s.l10Min,
            gameDisplay: s.gameDisplay,
          })),
          numLegs: selected.length,
          odds,
        });
      }

      console.log(`[VAULT] Today's picks: ${todayVaultParlays.length} parlays (${selected.length} legs from ${usedGames.size} games)`);
    } catch (e) {
      console.error('[VAULT] Error generating today picks:', e);
    }
  }

  // ── FORTRESS History Builder (Play 12) ────────────────────────────────────

  function fortressParlayAmerican(numLegs) {
    // At -200 per leg: decimal = 1.5
    const dec = Math.pow(1 + 100 / 200, numLegs);
    return dec >= 2.0 ? Math.round((dec - 1) * 100) : Math.round(-100 / (dec - 1));
  }

  function buildFortressHistory() {
    fortressHistoryPicks = [];
    if (!playerBoxScores || playerBoxScores.length === 0) return;

    const fortress = Object.create(FORTRESSModel);
    fortress.playerHistory = {};

    const sorted = [...playerBoxScores].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    const byDate = {};
    for (const game of sorted) {
      if (!byDate[game.date]) byDate[game.date] = [];
      byDate[game.date].push(game);
    }

    const processedDates = new Set();

    for (const game of sorted) {
      const date = game.date;

      if (!processedDates.has(date)) {
        const dayGames = byDate[date] || [];
        const dayCandidates = [];

        for (const dg of dayGames) {
          const gameKey = `${dg.away}@${dg.home}`;
          const players = (dg.players || []).map(p => ({
            name: p.name,
            team: p.team,
            pts: p.pts,
            min: parsePlayerMins(p.min),
          }));

          const picks = fortress.predictGame(players);
          for (const pick of picks) {
            const actualPlayer = dg.players.find(pp => pp.name === pick.player);
            if (!actualPlayer) continue;

            const hit = actualPlayer.pts > pick.line;
            dayCandidates.push({
              ...pick,
              actual: actualPlayer.pts,
              hit,
              gameKey,
              gameDisplay: `${dg.away} @ ${dg.home}`,
            });
          }
        }

        // Select top 3-5 legs from different games, sorted by confidence
        dayCandidates.sort((a, b) => b.confidence - a.confidence);
        const selected = [];
        const usedGames = new Set();

        for (const c of dayCandidates) {
          if (usedGames.has(c.gameKey)) continue;
          selected.push(c);
          usedGames.add(c.gameKey);
          if (selected.length >= 5) break;
        }

        if (selected.length >= 3) {
          const allHit = selected.every(s => s.hit);
          const odds = fortressParlayAmerican(selected.length);
          const pnl = allHit ? odds : -100;

          fortressHistoryPicks.push({
            date,
            legs: selected.map(s => ({
              player: s.player,
              team: s.team,
              line: s.line,
              actual: s.actual,
              hit: s.hit,
              confidence: s.confidence,
              l10Avg: s.l10Avg,
              l10Min: s.l10Min,
              gameDisplay: s.gameDisplay,
            })),
            numLegs: selected.length,
            odds,
            won: allHit,
            pnl,
          });
        }

        processedDates.add(date);
      }

      // Update FORTRESS model with all players from this game
      for (const p of (game.players || [])) {
        const mins = parsePlayerMins(p.min);
        if (mins < 10) continue;
        fortress.updatePlayer(p.name, p.pts, mins, game.date);
      }
    }

    // Save trained model for today's predictions
    trainedFortressModel = fortress;

    console.log(`[FORTRESS] Built history: ${fortressHistoryPicks.length} parlays`);
  }

  // ── Generate Today's FORTRESS Picks ─────────────────────────────────────

  function generateTodayFortressPicks() {
    todayFortressParlays = [];
    if (!trainedFortressModel || todayGames.length === 0) {
      console.log(`[FORTRESS] Skipping today's picks: model=${!!trainedFortressModel}, games=${todayGames.length}`);
      return;
    }

    try {
      const dayCandidates = [];

      for (const game of todayGames) {
        const gameKey = `${game.away_team}@${game.home_team}`;
        const teamPlayers = [];
        for (const [name, team] of Object.entries(playerTeamMap)) {
          if (team === game.home_team || team === game.away_team) {
            teamPlayers.push({ name, team, pts: 0, min: 30 });
          }
        }

        const picks = trainedFortressModel.predictGame(teamPlayers);
        for (const pick of picks) {
          dayCandidates.push({
            ...pick,
            gameKey,
            gameDisplay: `${game.away_team} @ ${game.home_team}`,
          });
        }
      }

      console.log(`[FORTRESS] ${dayCandidates.length} qualifying players across ${todayGames.length} games`);

      // Select top legs sorted by confidence, max 1 per game
      dayCandidates.sort((a, b) => b.confidence - a.confidence);
      const selected = [];
      const usedGames = new Set();
      for (const c of dayCandidates) {
        if (usedGames.has(c.gameKey)) continue;
        selected.push(c);
        usedGames.add(c.gameKey);
        if (selected.length >= 5) break;
      }

      // Build parlay if we have enough legs (minimum 3)
      if (selected.length >= 3) {
        const odds = fortressParlayAmerican(selected.length);
        todayFortressParlays.push({
          legs: selected.map(s => ({
            player: s.player,
            team: s.team,
            line: s.line,
            confidence: s.confidence,
            l10Avg: s.l10Avg,
            l10Min: s.l10Min,
            gameDisplay: s.gameDisplay,
          })),
          numLegs: selected.length,
          odds,
        });
      }

      console.log(`[FORTRESS] Today's picks: ${todayFortressParlays.length} parlays (${selected.length} legs from ${usedGames.size} games)`);
    } catch (e) {
      console.error('[FORTRESS] Error generating today picks:', e);
    }
  }

  // ── SIEGE History Builder (Play 13) ──────────────────────────────────────

  // SIEGE: Calculate parlay odds from mixed per-leg odds
  function siegeCalcParlayOdds(legs) {
    let decimal = 1;
    for (const leg of legs) {
      const odds = leg.liveOdds || leg.estOdds || -200;
      if (odds > 0) {
        decimal *= 1 + odds / 100;
      } else {
        decimal *= 1 + 100 / Math.abs(odds);
      }
    }
    if (decimal >= 2.0) {
      return Math.round((decimal - 1) * 100);
    } else {
      return Math.round(-100 / (decimal - 1));
    }
  }

  // SIEGE: Format American odds for display
  function siegeFormatOdds(odds) {
    return odds > 0 ? `+${odds}` : `${odds}`;
  }

  function buildSiegeHistory() {
    siegeHistoryPicks = [];
    if (!playerBoxScores || playerBoxScores.length === 0) return;

    const siege = Object.create(SIEGEModel);
    siege.playerHistory = {};

    const sorted = [...playerBoxScores].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    const byDate = {};
    for (const game of sorted) {
      if (!byDate[game.date]) byDate[game.date] = [];
      byDate[game.date].push(game);
    }

    const processedDates = new Set();

    for (const game of sorted) {
      const date = game.date;

      if (!processedDates.has(date)) {
        const dayGames = byDate[date] || [];
        const dayCandidates = [];

        for (const dg of dayGames) {
          const gameKey = `${dg.away}@${dg.home}`;
          const players = (dg.players || []).map(p => ({
            name: p.name,
            team: p.team,
            pts: p.pts,
            min: parsePlayerMins(p.min),
          }));

          const picks = siege.predictGame(players);
          for (const pick of picks) {
            const actualPlayer = dg.players.find(pp => pp.name === pick.player);
            if (!actualPlayer) continue;

            const hit = actualPlayer.pts > pick.line;
            dayCandidates.push({
              ...pick,
              actual: actualPlayer.pts,
              hit,
              gameKey,
              gameDisplay: `${dg.away} @ ${dg.home}`,
            });
          }
        }

        // Sort by EV (best expected value first)
        dayCandidates.sort((a, b) => (b.ev || 0) - (a.ev || 0) || b.confidence - a.confidence);
        const selected = [];
        const usedGames = new Set();

        for (const c of dayCandidates) {
          if (usedGames.has(c.gameKey)) continue;
          selected.push(c);
          usedGames.add(c.gameKey);
          if (selected.length >= 3) break;  // Up to 3-leg parlays for optimal EV × payout
        }

        if (selected.length >= 2) {
          const allHit = selected.every(s => s.hit);
          const odds = siegeCalcParlayOdds(selected);
          const parlayDecimal = selected.reduce((d, s) => {
            const o = s.estOdds;
            return d * (o > 0 ? 1 + o/100 : 1 + 100/Math.abs(o));
          }, 1);
          const pnl = allHit ? Math.round((parlayDecimal - 1) * 100) : -100;

          siegeHistoryPicks.push({
            date,
            legs: selected.map(s => ({
              player: s.player,
              team: s.team,
              line: s.line,
              displayLine: s.displayLine,
              actual: s.actual,
              hit: s.hit,
              confidence: s.confidence,
              l10Avg: s.l10Avg,
              l3Avg: s.l3Avg,
              l10Min: s.l10Min,
              estOdds: s.estOdds,
              ratio: s.ratio,
              cv: s.cv,
              momentum: s.momentum,
              hitRate: s.hitRate,
              ev: s.ev,
              gameDisplay: s.gameDisplay,
            })),
            numLegs: selected.length,
            odds,
            won: allHit,
            pnl,
          });
        }

        processedDates.add(date);
      }

      // Update SIEGE model with all players from this game
      for (const p of (game.players || [])) {
        const mins = parsePlayerMins(p.min);
        if (mins < 10) continue;
        siege.updatePlayer(p.name, p.pts, mins, game.date);
      }
    }

    trainedSiegeModel = siege;

    // Validation logging
    const total = siegeHistoryPicks.length;
    const wins = siegeHistoryPicks.filter(p => p.won).length;
    const pnl = siegeHistoryPicks.reduce((s, p) => s + p.pnl, 0);
    let legHits = 0, legTotal = 0;
    for (const p of siegeHistoryPicks) {
      for (const l of p.legs) { legTotal++; if (l.hit) legHits++; }
    }
    const avgOdds = total > 0 ? Math.round(siegeHistoryPicks.reduce((s, p) => s + p.odds, 0) / total) : 0;
    console.log(`[SIEGE] Built history: ${total} parlays, ${wins}-${total-wins} (${total > 0 ? (wins/total*100).toFixed(1) : 0}%), P&L: $${pnl}, ROI: ${total > 0 ? (pnl/(total*100)*100).toFixed(0) : 0}%, Legs: ${legHits}/${legTotal} (${legTotal > 0 ? (legHits/legTotal*100).toFixed(1) : 0}%), Avg odds: ${siegeFormatOdds(avgOdds)}`);
  }

  // SIEGE: Fetch live odds from The Odds API for player_points_alternate
  const ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8';
  const ODDS_API_BASE = 'https://api.the-odds-api.com/v4';

  async function fetchSiegeOddsFromAPI() {
    try {
      // 1. Get today's NBA events
      const eventsUrl = `${ODDS_API_BASE}/sports/basketball_nba/events?apiKey=${ODDS_API_KEY}`;
      const eventsRes = await fetch(eventsUrl);
      if (!eventsRes.ok) {
        console.warn('[SIEGE] Events API returned', eventsRes.status);
        return null;
      }
      const events = await eventsRes.json();
      if (!events || events.length === 0) return null;

      const remaining = eventsRes.headers.get('x-requests-remaining');
      console.log(`[SIEGE] Odds API: ${events.length} events, ${remaining} requests remaining`);

      // 2. Fetch alternate player points odds for each event
      const allOdds = {}; // playerName -> { threshold -> { bestOdds, bestBook, allBooks } }

      for (const event of events) {
        try {
          const oddsUrl = `${ODDS_API_BASE}/sports/basketball_nba/events/${event.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=player_points_alternate&oddsFormat=american`;
          const oddsRes = await fetch(oddsUrl);
          if (!oddsRes.ok) continue;
          const oddsData = await oddsRes.json();

          if (!oddsData || !oddsData.bookmakers) continue;

          // Only use FanDuel odds — most reliable for player availability
          const fdBook = oddsData.bookmakers.find(b => b.key === 'fanduel');
          if (!fdBook) continue;

          const market = fdBook.markets && fdBook.markets.find(m => m.key === 'player_points_alternate');
          if (!market) continue;

          for (const outcome of (market.outcomes || [])) {
            if (outcome.name !== 'Over') continue;
            const player = outcome.description;
            const threshold = outcome.point;
            const odds = outcome.price;

            if (!allOdds[player]) allOdds[player] = {};
            if (!allOdds[player][threshold]) allOdds[player][threshold] = [];
            allOdds[player][threshold].push({
              book: fdBook.title,
              bookKey: fdBook.key,
              odds: odds,
            });
          }
        } catch (e) {
          console.warn('[SIEGE] Error fetching odds for event:', event.id, e);
        }
      }

      // 3. For each player/threshold, use FanDuel odds
      for (const player of Object.keys(allOdds)) {
        for (const threshold of Object.keys(allOdds[player])) {
          const bookOdds = allOdds[player][threshold];
          allOdds[player][threshold] = {
            bestOdds: bookOdds[0].odds,
            bestBook: 'FanDuel',
            numBooks: 1,
            allBooks: bookOdds,
          };
        }
      }

      console.log(`[SIEGE] Fetched FanDuel odds for ${Object.keys(allOdds).length} players`);
      return allOdds;
    } catch (e) {
      console.error('[SIEGE] Error fetching odds from API:', e);
      return null;
    }
  }

  // Find best live odds for a player at a given floor
  function findBestLiveOdds(liveOdds, playerName, floorLine) {
    if (!liveOdds || !liveOdds[playerName]) return null;

    const playerThresholds = liveOdds[playerName];
    let bestMatch = null;

    // Find the highest threshold that is <= our floor line
    // This ensures the actual bet is at least as safe as our model predicts
    for (const threshold of Object.keys(playerThresholds)) {
      const t = parseFloat(threshold);
      if (t <= floorLine) {
        const data = playerThresholds[threshold];
        if (!bestMatch || t > bestMatch.threshold) {
          bestMatch = {
            threshold: t,
            odds: data.bestOdds,
            book: data.bestBook,
            numBooks: data.numBooks,
          };
        }
      }
    }

    return bestMatch;
  }

  async function generateTodaySiegePicks() {
    todaySiegeParlays = [];
    if (!trainedSiegeModel || todayGames.length === 0) {
      console.log(`[SIEGE] Skipping today's picks: model=${!!trainedSiegeModel}, games=${todayGames.length}`);
      return;
    }

    try {
      // Fetch live odds from Odds API
      let liveOdds = null;
      try {
        liveOdds = await fetchSiegeOddsFromAPI();
      } catch (e) {
        console.warn('[SIEGE] Could not fetch live odds, using estimates:', e);
      }

      const dayCandidates = [];

      for (const game of todayGames) {
        const gameKey = `${game.away_team}@${game.home_team}`;
        const teamPlayers = [];
        for (const [name, team] of Object.entries(playerTeamMap)) {
          if (team === game.home_team || team === game.away_team) {
            teamPlayers.push({ name, team, pts: 0, min: 30 });
          }
        }

        const picks = trainedSiegeModel.predictGame(teamPlayers);
        for (const pick of picks) {
          // Try to find live odds from the API
          const live = findBestLiveOdds(liveOdds, pick.player, pick.line);

          // FILTER: Only include players with live FanDuel odds
          // Players who are injured/out won't have props listed on FanDuel
          if (!live) {
            console.log(`[SIEGE] Skipping ${pick.player} — no FanDuel odds available (likely injured/out)`);
            continue;
          }

          dayCandidates.push({
            ...pick,
            gameKey,
            gameDisplay: `${game.away_team} @ ${game.home_team}`,
            liveOdds: live.odds,
            liveThreshold: live.threshold,
            bestBook: live.book,
            numBooks: live.numBooks,
            hasLiveOdds: true,
          });
        }
      }

      console.log(`[SIEGE] ${dayCandidates.length} qualifying players across ${todayGames.length} games`);
      if (liveOdds) {
        const withLive = dayCandidates.filter(c => c.hasLiveOdds).length;
        console.log(`[SIEGE] ${withLive}/${dayCandidates.length} have live odds from API`);
      }

      // Sort by EV (best expected value first)
      dayCandidates.sort((a, b) => (b.ev || 0) - (a.ev || 0) || b.confidence - a.confidence);
      const selected = [];
      const usedGames = new Set();
      for (const c of dayCandidates) {
        if (usedGames.has(c.gameKey)) continue;
        selected.push(c);
        usedGames.add(c.gameKey);
        if (selected.length >= 3) break;  // Up to 3-leg for optimal EV × payout
      }

      if (selected.length >= 2) {
        const odds = siegeCalcParlayOdds(selected);
        const hasAnyLive = selected.some(s => s.hasLiveOdds);

        todaySiegeParlays.push({
          legs: selected.map(s => ({
            player: s.player,
            team: s.team,
            line: s.line,
            displayLine: s.displayLine,
            confidence: s.confidence,
            l10Avg: s.l10Avg,
            l3Avg: s.l3Avg,
            l10Min: s.l10Min,
            estOdds: s.estOdds,
            liveOdds: s.liveOdds,
            liveThreshold: s.liveThreshold,
            bestBook: s.bestBook,
            numBooks: s.numBooks,
            hasLiveOdds: s.hasLiveOdds,
            ratio: s.ratio,
            cv: s.cv,
            momentum: s.momentum,
            gameDisplay: s.gameDisplay,
          })),
          numLegs: selected.length,
          odds,
          hasLiveOdds: hasAnyLive,
        });
      }

      console.log(`[SIEGE] Today's picks: ${todaySiegeParlays.length} parlays (${selected.length} legs from ${usedGames.size} games)`);
    } catch (e) {
      console.error('[SIEGE] Error generating today picks:', e);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // MASTER BET RECOMMENDER INTEGRATION
  // ══════════════════════════════════════════════════════════════════════════

  function runMasterBacktest() {
    if (!seasonData || seasonData.length === 0 || !playerBoxScores) {
      console.log('[MASTER] Skipping backtest: insufficient data');
      return;
    }
    try {
      console.log(`[MASTER] Running walk-forward backtest on ${seasonData.length} games, ${playerBoxScores.length} box scores...`);
      masterBacktestResults = window.MasterBetEngine.runOfflineBacktest(seasonData, playerBoxScores);

      const s = masterBacktestResults.stats;
      console.log(`[MASTER] Backtest complete:`);
      console.log(`  IRONLOCK: ${s.ironlock.wins}/${s.ironlock.total} = ${(s.ironlock.hitRate * 100).toFixed(1)}%, P&L: $${s.ironlock.pnl}, ROI: ${s.ironlock.roi}%`);
      console.log(`  BEDROCK: ${s.bedrock.wins}/${s.bedrock.total} = ${(s.bedrock.hitRate * 100).toFixed(1)}%, P&L: $${s.bedrock.pnl}, ROI: ${s.bedrock.roi}%`);
      console.log(`  APEX: ${s.apex.wins}/${s.apex.total} = ${(s.apex.hitRate * 100).toFixed(1)}%, P&L: $${s.apex.pnl}, ROI: ${s.apex.roi}%`);
      console.log(`  OVERALL: ${s.overall.wins}/${s.overall.total} = ${(s.overall.hitRate * 100).toFixed(1)}%, P&L: $${s.overall.pnl}, ROI: ${s.overall.roi}%`);
    } catch (e) {
      console.error('[MASTER] Backtest error:', e);
    }
  }

  async function generateMasterTodayPicks() {
    masterTodayPicks = [];
    if (todayGames.length === 0) return;

    try {
      // Fetch live FanDuel odds
      let liveOdds = null;
      try {
        liveOdds = await window.MasterBetEngine.fetchLiveOdds();
      } catch (e) {
        console.warn('[MASTER] Could not fetch live odds:', e);
      }

      // Initialize and train the recommender
      const MR = window.MasterBetEngine.MasterRecommender;
      const recommender = Object.create(MR);
      recommender.init();
      masterTodayPicks = await recommender.generateTodayPicks(todayGames, seasonData, playerBoxScores, liveOdds);
      console.log(`[MASTER] Today's picks: ${masterTodayPicks.length} recommendations`);
    } catch (e) {
      console.error('[MASTER] Error generating today picks:', e);
    }
  }

  function renderMasterView() {
    renderMasterDashboard();
    renderMasterTodayPicks();
    renderMasterHistory();
    renderMasterEquityCurve();
  }

  function renderMasterDashboard() {
    if (!masterBacktestResults || !masterBacktestResults.stats) return;
    const s = masterBacktestResults.stats;

    const setEl = (id, text) => {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    };

    // Overall
    setEl('master-overall-accuracy', s.overall.total > 0 ? `${(s.overall.hitRate * 100).toFixed(1)}%` : '—');
    setEl('master-overall-record', s.overall.total > 0 ? `${s.overall.wins}-${s.overall.losses}` : 'No data');

    // Ironlock
    setEl('master-ironlock-accuracy', s.ironlock.total > 0 ? `${(s.ironlock.hitRate * 100).toFixed(1)}%` : '—');
    setEl('master-ironlock-record', s.ironlock.total > 0 ? `${s.ironlock.wins}-${s.ironlock.losses} | ROI ${s.ironlock.roi}%` : 'No data');

    // Bedrock
    setEl('master-bedrock-accuracy', s.bedrock.total > 0 ? `${(s.bedrock.hitRate * 100).toFixed(1)}%` : '—');
    setEl('master-bedrock-record', s.bedrock.total > 0 ? `${s.bedrock.wins}-${s.bedrock.losses} | ROI ${s.bedrock.roi}%` : 'No data');

    // Apex
    setEl('master-apex-accuracy', s.apex.total > 0 ? `${(s.apex.hitRate * 100).toFixed(1)}%` : '—');
    setEl('master-apex-record', s.apex.total > 0 ? `${s.apex.wins}-${s.apex.losses} | ROI ${s.apex.roi}%` : 'No data');

    // P&L
    const pnlColor = s.overall.pnl >= 0 ? 'var(--green)' : 'var(--red)';
    const pnlEl = document.getElementById('master-total-pnl');
    if (pnlEl) {
      pnlEl.textContent = `$${s.overall.pnl >= 0 ? '+' : ''}${s.overall.pnl}`;
      pnlEl.style.color = pnlColor;
    }
    setEl('master-total-roi', `ROI: ${s.overall.roi}%`);

    // Active days
    setEl('master-active-days', `${s.daysWithPicks || 0}`);
    setEl('master-picks-per-day', `of ${s.totalDays || 0} total days`);
  }

  function renderMasterTodayPicks() {
    const loading = document.getElementById('master-today-loading');
    const empty = document.getElementById('master-today-empty');
    const grid = document.getElementById('master-today-picks');
    if (!loading || !empty || !grid) return;

    loading.style.display = 'none';

    if (masterTodayPicks.length === 0) {
      empty.style.display = 'block';
      grid.style.display = 'none';
      return;
    }

    empty.style.display = 'none';
    grid.style.display = '';

    grid.innerHTML = masterTodayPicks.filter(p => p.recommended !== false).map(pick => {
      if (pick.category === 'APEX') {
        return renderMasterApexCard(pick);
      } else if (pick.category === 'IRONLOCK') {
        return renderMasterIronlockCard(pick);
      } else {
        return renderMasterBedrockCard(pick);
      }
    }).join('');
  }

  function renderMasterIronlockCard(pick) {
    return `
      <div class="master-pick-card ironlock-card">
        <div class="master-pick-header">
          <span class="master-pick-type ironlock">IRONLOCK</span>
          <span class="master-pick-confidence">${(pick.confidence * 100).toFixed(0)}% conf</span>
        </div>
        <div class="master-pick-bet">${pick.team} ML ${window.MasterBetEngine.formatOdds(pick.odds)}</div>
        <div class="master-pick-matchup">${pick.gameDisplay || `${pick.opponent} @ ${pick.team}`}</div>
        <div class="master-pick-stats">
          <div class="master-stat-mini">
            <div class="master-stat-mini-label">CDS Score</div>
            <div class="master-stat-mini-value">${pick.cdsScore}</div>
          </div>
          <div class="master-stat-mini">
            <div class="master-stat-mini-label">Model Prob</div>
            <div class="master-stat-mini-value">${(pick.modelProb * 100).toFixed(0)}%</div>
          </div>
          <div class="master-stat-mini">
            <div class="master-stat-mini-label">Edge</div>
            <div class="master-stat-mini-value" style="color: var(--green)">+${(pick.edge * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>`;
  }

  function renderMasterBedrockCard(pick) {
    return `
      <div class="master-pick-card bedrock-card">
        <div class="master-pick-header">
          <span class="master-pick-type bedrock">BEDROCK</span>
          <span class="master-pick-confidence">${(pick.confidence * 100).toFixed(0)}% conf</span>
        </div>
        <div class="master-pick-bet">${pick.player} ${pick.displayLine} ${window.MasterBetEngine.formatOdds(pick.odds)}</div>
        <div class="master-pick-matchup">${pick.gameDisplay || pick.team}</div>
        <div class="master-pick-stats">
          <div class="master-stat-mini">
            <div class="master-stat-mini-label">L10 Avg</div>
            <div class="master-stat-mini-value">${pick.l10Avg}</div>
          </div>
          <div class="master-stat-mini">
            <div class="master-stat-mini-label">L10 Floor</div>
            <div class="master-stat-mini-value">${pick.l10Min}</div>
          </div>
          <div class="master-stat-mini">
            <div class="master-stat-mini-label">Hit Rate</div>
            <div class="master-stat-mini-value" style="color: var(--green)">${(pick.hitRate * 100).toFixed(0)}%</div>
          </div>
        </div>
      </div>`;
  }

  function renderMasterApexCard(pick) {
    const legsHtml = pick.legs.map(leg => {
      const typeClass = leg.legType === 'IRONLOCK' ? 'ironlock' : 'bedrock';
      const info = leg.legType === 'IRONLOCK'
        ? `${leg.team} ML`
        : `${leg.player} ${leg.displayLine}`;
      return `<div class="master-parlay-leg">
        <span class="master-parlay-leg-type ${typeClass}">${leg.legType}</span>
        <span class="master-parlay-leg-info">${info}</span>
        <span class="master-parlay-leg-odds">${window.MasterBetEngine.formatOdds(leg.odds)}</span>
      </div>`;
    }).join('');

    return `
      <div class="master-pick-card apex-card">
        <div class="master-pick-header">
          <span class="master-pick-type apex">APEX ${pick.numLegs}-LEG</span>
          <span class="master-pick-confidence">${(pick.combinedHitRate * 100).toFixed(0)}% combined</span>
        </div>
        <div class="master-pick-bet">${pick.numLegs}-Leg Parlay ${window.MasterBetEngine.formatOdds(pick.odds)}</div>
        <ul class="master-parlay-legs">${legsHtml}</ul>
        <div class="master-pick-stats">
          <div class="master-stat-mini">
            <div class="master-stat-mini-label">Parlay Odds</div>
            <div class="master-stat-mini-value">${window.MasterBetEngine.formatOdds(pick.odds)}</div>
          </div>
          <div class="master-stat-mini">
            <div class="master-stat-mini-label">EV</div>
            <div class="master-stat-mini-value" style="color: ${pick.ev > 0 ? 'var(--green)' : 'var(--red)'}">
              ${pick.ev > 0 ? '+' : ''}${(pick.ev * 100).toFixed(1)}%</div>
          </div>
          <div class="master-stat-mini">
            <div class="master-stat-mini-label"># Legs</div>
            <div class="master-stat-mini-value">${pick.numLegs}</div>
          </div>
        </div>
      </div>`;
  }

  function renderMasterHistory() {
    if (!masterBacktestResults) return;

    const period = currentMasterHistoryPeriod;
    let cutoff = '';
    if (period !== 'all') {
      const days = parseInt(period);
      const latestDate = masterBacktestResults.dates && masterBacktestResults.dates.length > 0
        ? masterBacktestResults.dates[masterBacktestResults.dates.length - 1] : '';
      if (latestDate) {
        const d = parseDateStr(latestDate);
        d.setDate(d.getDate() - days);
        cutoff = d.toISOString().slice(0, 10).replace(/-/g, '');
      }
    }

    // IRONLOCK history
    const ironlockBody = document.getElementById('master-ironlock-body');
    if (ironlockBody) {
      const picks = masterBacktestResults.ironlockPicks.filter(p => !cutoff || p.date >= cutoff);
      if (picks.length === 0) {
        ironlockBody.innerHTML = '<tr><td colspan="9" class="muted">No IRONLOCK picks in this period</td></tr>';
      } else {
        ironlockBody.innerHTML = picks.slice().reverse().map(p => {
          const result = p.won ? '<span style="color:var(--green)">W</span>' : '<span style="color:var(--red)">L</span>';
          const pnlClass = p.pnl >= 0 ? 'style="color:var(--green)"' : 'style="color:var(--red)"';
          return `<tr>
            <td>${formatDate(p.date)}</td>
            <td><strong>${p.team} ML</strong></td>
            <td>${p.awayTeam} @ ${p.homeTeam}</td>
            <td>${p.cdsScore}</td>
            <td>${window.MasterBetEngine.formatOdds(p.odds)}</td>
            <td>+${(p.edge * 100).toFixed(1)}%</td>
            <td>${p.homeScore ? `${p.awayTeam} ${p.awayScore} — ${p.homeTeam} ${p.homeScore}` : '—'}</td>
            <td>${result}</td>
            <td ${pnlClass}>$${p.pnl >= 0 ? '+' : ''}${p.pnl}</td>
          </tr>`;
        }).join('');
      }

      // Summary
      const summary = document.getElementById('master-ironlock-summary');
      if (summary && picks.length > 0) {
        const wins = picks.filter(p => p.won).length;
        const pnl = picks.reduce((s, p) => s + p.pnl, 0);
        summary.innerHTML = `<strong>IRONLOCK:</strong> ${wins}-${picks.length - wins} (${(wins/picks.length*100).toFixed(1)}%) | P&L: $${pnl >= 0 ? '+' : ''}${pnl} | ROI: ${(pnl / (picks.length * 100) * 100).toFixed(1)}%`;
      }
    }

    // BEDROCK history
    const bedrockBody = document.getElementById('master-bedrock-body');
    if (bedrockBody) {
      const picks = masterBacktestResults.bedrockPicks.filter(p => !cutoff || p.date >= cutoff);
      if (picks.length === 0) {
        bedrockBody.innerHTML = '<tr><td colspan="10" class="muted">No BEDROCK picks in this period</td></tr>';
      } else {
        bedrockBody.innerHTML = picks.slice().reverse().map(p => {
          const result = p.won ? '<span style="color:var(--green)">W</span>' : '<span style="color:var(--red)">L</span>';
          const pnlClass = p.pnl >= 0 ? 'style="color:var(--green)"' : 'style="color:var(--red)"';
          return `<tr>
            <td>${formatDate(p.date)}</td>
            <td><strong>${p.player}</strong></td>
            <td>${p.displayLine}</td>
            <td>${window.MasterBetEngine.formatOdds(p.odds)}</td>
            <td>${p.l10Avg}</td>
            <td>${p.l10Min}</td>
            <td>${(p.hitRate * 100).toFixed(0)}%</td>
            <td>${p.actual !== undefined ? p.actual : '—'}</td>
            <td>${result}</td>
            <td ${pnlClass}>$${p.pnl >= 0 ? '+' : ''}${p.pnl}</td>
          </tr>`;
        }).join('');
      }

      const summary = document.getElementById('master-bedrock-summary');
      if (summary && picks.length > 0) {
        const wins = picks.filter(p => p.won).length;
        const pnl = picks.reduce((s, p) => s + p.pnl, 0);
        summary.innerHTML = `<strong>BEDROCK:</strong> ${wins}-${picks.length - wins} (${(wins/picks.length*100).toFixed(1)}%) | P&L: $${pnl >= 0 ? '+' : ''}${pnl} | ROI: ${(pnl / (picks.length * 100) * 100).toFixed(1)}%`;
      }
    }

    // APEX parlay history
    const apexBody = document.getElementById('master-apex-body');
    if (apexBody) {
      const picks = masterBacktestResults.apexParlays.filter(p => !cutoff || p.date >= cutoff);
      if (picks.length === 0) {
        apexBody.innerHTML = '<tr><td colspan="8" class="muted">No APEX parlays in this period</td></tr>';
      } else {
        apexBody.innerHTML = picks.slice().reverse().map(p => {
          const result = p.won ? '<span style="color:var(--green)">W</span>' : '<span style="color:var(--red)">L</span>';
          const pnlClass = p.pnl >= 0 ? 'style="color:var(--green)"' : 'style="color:var(--red)"';
          const legsStr = p.legs.map(l => {
            const wonMark = l.won ? '&#10003;' : '&#10007;';
            const wonColor = l.won ? 'var(--green)' : 'var(--red)';
            if (l.legType === 'IRONLOCK') {
              return `<span style="color:${wonColor}">${wonMark}</span> ${l.team} ML`;
            } else {
              return `<span style="color:${wonColor}">${wonMark}</span> ${l.player} ${l.displayLine}`;
            }
          }).join('<br>');
          return `<tr>
            <td>${formatDate(p.date)}</td>
            <td>${legsStr}</td>
            <td>${p.numLegs}</td>
            <td>${window.MasterBetEngine.formatOdds(p.odds)}</td>
            <td>${(p.combinedHitRate * 100).toFixed(0)}%</td>
            <td>${(p.ev * 100).toFixed(1)}%</td>
            <td>${result}</td>
            <td ${pnlClass}>$${p.pnl >= 0 ? '+' : ''}${p.pnl}</td>
          </tr>`;
        }).join('');
      }

      const summary = document.getElementById('master-apex-summary');
      if (summary && picks.length > 0) {
        const wins = picks.filter(p => p.won).length;
        const pnl = picks.reduce((s, p) => s + p.pnl, 0);
        summary.innerHTML = `<strong>APEX:</strong> ${wins}-${picks.length - wins} (${(wins/picks.length*100).toFixed(1)}%) | P&L: $${pnl >= 0 ? '+' : ''}${pnl} | ROI: ${(pnl / (picks.length * 100) * 100).toFixed(1)}%`;
      }
    }
  }

  function renderMasterEquityCurve() {
    const canvas = document.getElementById('master-equity-canvas');
    if (!canvas || !masterBacktestResults) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 32;
    const h = 300;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    // Build cumulative P&L data
    const allPicks = [
      ...masterBacktestResults.ironlockPicks.map(p => ({ date: p.date, pnl: p.pnl })),
      ...masterBacktestResults.bedrockPicks.map(p => ({ date: p.date, pnl: p.pnl })),
      ...masterBacktestResults.apexParlays.map(p => ({ date: p.date, pnl: p.pnl })),
    ].sort((a, b) => a.date.localeCompare(b.date));

    if (allPicks.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No backtest data available', w / 2, h / 2);
      return;
    }

    let cumPnl = 0;
    const points = [{ x: 0, y: 0 }];
    for (let i = 0; i < allPicks.length; i++) {
      cumPnl += allPicks[i].pnl;
      points.push({ x: i + 1, y: cumPnl });
    }

    const maxX = points.length - 1;
    const minY = Math.min(0, ...points.map(p => p.y));
    const maxY = Math.max(100, ...points.map(p => p.y));
    const rangeY = maxY - minY;

    const pad = { top: 20, right: 20, bottom: 30, left: 60 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const sx = (x) => pad.left + (x / maxX) * plotW;
    const sy = (y) => pad.top + plotH - ((y - minY) / rangeY) * plotH;

    // Background
    ctx.fillStyle = '#1a1f2e';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    const gridSteps = 5;
    for (let i = 0; i <= gridSteps; i++) {
      const yVal = minY + (rangeY / gridSteps) * i;
      const yPos = sy(yVal);
      ctx.beginPath();
      ctx.moveTo(pad.left, yPos);
      ctx.lineTo(w - pad.right, yPos);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '11px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`$${Math.round(yVal)}`, pad.left - 8, yPos + 4);
    }

    // Zero line
    if (minY < 0) {
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.left, sy(0));
      ctx.lineTo(w - pad.right, sy(0));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Equity curve fill
    ctx.beginPath();
    ctx.moveTo(sx(points[0].x), sy(0));
    for (const p of points) {
      ctx.lineTo(sx(p.x), sy(p.y));
    }
    ctx.lineTo(sx(points[points.length - 1].x), sy(0));
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, sy(maxY), 0, sy(minY));
    grad.addColorStop(0, 'rgba(34, 197, 94, 0.15)');
    grad.addColorStop(1, 'rgba(34, 197, 94, 0.02)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Equity curve line
    ctx.beginPath();
    ctx.moveTo(sx(points[0].x), sy(points[0].y));
    for (const p of points) {
      ctx.lineTo(sx(p.x), sy(p.y));
    }
    ctx.strokeStyle = cumPnl >= 0 ? '#22c55e' : '#ef4444';
    ctx.lineWidth = 2;
    ctx.stroke();

    // End point
    const lastP = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(sx(lastP.x), sy(lastP.y), 4, 0, Math.PI * 2);
    ctx.fillStyle = cumPnl >= 0 ? '#22c55e' : '#ef4444';
    ctx.fill();

    // Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${allPicks.length} bets`, w / 2, h - 5);

    ctx.textAlign = 'right';
    ctx.fillStyle = cumPnl >= 0 ? '#22c55e' : '#ef4444';
    ctx.font = 'bold 12px monospace';
    ctx.fillText(`$${cumPnl >= 0 ? '+' : ''}${cumPnl}`, w - pad.right, pad.top - 5);
  }

  // ── Rendering: Today's Picks ───────────────────────────────────────────────

  function renderPicks() {
    const loading = document.getElementById('picks-loading');
    const cdsContainer = document.getElementById('cds-container');
    const empty = document.getElementById('picks-empty');
    const allSection = document.getElementById('all-games-section');

    loading.style.display = 'none';

    const hasPicks = todayCdsPicks.length > 0 || todayPrismPicks.length > 0 || todayNovaPicks.length > 0 || todayFusionParlays.length > 0 || todayPulseParlays.length > 0 || todayVaultParlays.length > 0 || todayFortressParlays.length > 0 || todaySiegeParlays.length > 0;

    if (!hasPicks) {
      if (cdsContainer) cdsContainer.style.display = 'none';
      empty.style.display = 'block';
    } else {
      empty.style.display = 'none';

      // Play 4 CDS picks
      if (cdsContainer && todayCdsPicks.length > 0) {
        cdsContainer.style.display = '';
        cdsContainer.innerHTML = '<h3 class="section-title">Play 4 — CDS Moneyline Picks</h3>' +
          '<div class="picks-grid">' + todayCdsPicks.map(renderCDSCard).join('') + '</div>';
      } else if (cdsContainer) {
        cdsContainer.style.display = 'none';
      }

      // Play 7 — PRISM Convergent Signal
      const prismContainer = document.getElementById('prism-container');
      if (prismContainer && todayPrismPicks.length > 0) {
        prismContainer.style.display = '';
        prismContainer.innerHTML = '<h3 class="section-title">Play 7 — PRISM Convergent Signal <span class="prism-badge">-110</span></h3>' +
          '<div class="picks-grid">' + todayPrismPicks.map(renderPRISMCard).join('') + '</div>';
      } else if (prismContainer) {
        prismContainer.style.display = 'none';
      }

      // Play 8 — NOVA Compound OVER
      const novaContainer = document.getElementById('nova-container');
      if (novaContainer && todayNovaPicks.length > 0) {
        novaContainer.style.display = '';
        novaContainer.innerHTML = '<h3 class="section-title">Play 8 — NOVA Compound OVER <span class="nova-badge">-110</span></h3>' +
          '<div class="picks-grid">' + todayNovaPicks.map(renderNOVACard).join('') + '</div>';
      } else if (novaContainer) {
        novaContainer.style.display = 'none';
      }

      // Play 9 — FUSION Parlay
      const fusionContainer = document.getElementById('fusion-container');
      if (fusionContainer && todayFusionParlays.length > 0) {
        fusionContainer.style.display = '';
        fusionContainer.innerHTML = '<h3 class="section-title">Play 9 — FUSION 2-Leg Parlay <span class="fusion-badge">+264</span></h3>' +
          '<div class="picks-grid">' + todayFusionParlays.map(renderFusionCard).join('') + '</div>';
      } else if (fusionContainer) {
        fusionContainer.style.display = 'none';
      }

      // Play 10 — PULSE Player Prop Parlay
      const pulseContainer = document.getElementById('pulse-container');
      if (pulseContainer && todayPulseParlays.length > 0) {
        pulseContainer.style.display = '';
        pulseContainer.innerHTML = '<h3 class="section-title">Play 10 — PULSE Player Prop Parlay <span class="pulse-badge">+251</span></h3>' +
          '<div class="picks-grid">' + todayPulseParlays.map(renderPulseCard).join('') + '</div>';
      } else if (pulseContainer) {
        pulseContainer.style.display = 'none';
      }

      // Play 11 — VAULT Multi-Leg Floor Parlay
      const vaultContainer = document.getElementById('vault-container');
      if (vaultContainer && todayVaultParlays.length > 0) {
        vaultContainer.style.display = '';
        vaultContainer.innerHTML = '<h3 class="section-title">Play 11 — VAULT Multi-Leg Parlay <span class="vault-badge">+649 avg</span></h3>' +
          '<div class="picks-grid">' + todayVaultParlays.map(renderVaultCard).join('') + '</div>';
      } else if (vaultContainer) {
        vaultContainer.style.display = 'none';
      }

      // Play 12 — FORTRESS Higher-Floor Parlay
      const fortressContainer = document.getElementById('fortress-container');
      if (fortressContainer && todayFortressParlays.length > 0) {
        fortressContainer.style.display = '';
        fortressContainer.innerHTML = '<h3 class="section-title">Play 12 — FORTRESS Higher-Floor Parlay <span class="fortress-badge">+475 avg</span></h3>' +
          '<div class="picks-grid">' + todayFortressParlays.map(renderFortressCard).join('') + '</div>';
      } else if (fortressContainer) {
        fortressContainer.style.display = 'none';
      }

      // Play 13 — SIEGE Adaptive Edge Parlay
      const siegeContainer = document.getElementById('siege-container');
      if (siegeContainer && todaySiegeParlays.length > 0) {
        siegeContainer.style.display = '';
        siegeContainer.innerHTML = '<h3 class="section-title">Play 13 — SIEGE EV-Optimized Parlay <span class="siege-badge">FanDuel Live Odds</span></h3>' +
          '<div class="picks-grid">' + todaySiegeParlays.map(renderSiegeCard).join('') + '</div>';
      } else if (siegeContainer) {
        siegeContainer.style.display = 'none';
      }

    }

    if (todayGames.length > 0) {
      allSection.style.display = '';
    }
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
        ${liveHtml}
      </div>`;
  }

  function renderPRISMCard(pick) {
    const g = pick.game;
    const tierClass = pick.tier === 'ELITE' ? 'conf-prism-elite' : pick.tier === 'HIGH' ? 'conf-prism-high' : 'conf-prism';

    const FACTOR_LABELS = {
      elite_def: 'Elite Defense', good_def: 'Good Defense',
      both_poor_def: 'Both Poor Defense', mediocre_def: 'Mediocre Defense',
      pace_drop_strong: 'Pace Dropping Fast', pace_drop: 'Pace Dropping',
      pace_up_strong: 'Pace Rising Fast', pace_up: 'Pace Rising',
      both_cooling: 'Both Teams Cooling', cooling: 'Cooling Trend',
      both_heating: 'Both Teams Heating', heating: 'Heating Trend',
      low_pace_teams: 'Low-Pace Teams', high_pace_teams: 'High-Pace Teams',
      extreme_luck_gap: 'Extreme Luck Gap', large_luck_gap: 'Large Luck Gap',
      luck_gap: 'Luck Regression', home_declining: 'Home Declining',
      away_improving: 'Away Improving',
    };
    const factorList = pick.factors.map(f => FACTOR_LABELS[f] || f).join(', ');

    let betLine, matchupHtml;
    if (pick.type === 'total') {
      const dirClass = pick.direction === 'OVER' ? 'prism-over' : 'prism-under';
      betLine = `<div class="pick-bet-line ${dirClass}">${pick.direction} ${pick.predTotal} at -110</div>`;
      matchupHtml = `${teamName(g.away_team)} @ ${teamName(g.home_team)}`;
    } else {
      betLine = `<div class="pick-bet-line prism-spread">${pick.betTeam} +${Math.abs(pick.predMargin).toFixed(1)} at -110</div>`;
      matchupHtml = `${teamName(g.away_team)} @ ${teamName(g.home_team)}`;
    }

    let liveHtml = '';
    if (g.status === 'STATUS_FINAL') {
      let won;
      if (pick.type === 'total') {
        const actualTotal = g.home_score + g.away_score;
        won = pick.direction === 'OVER' ? actualTotal > pick.predTotal : actualTotal < pick.predTotal;
        liveHtml = `<div class="pick-live ${won ? 'live-win' : 'live-loss'}">
          <span class="live-label">FINAL</span>
          <span class="live-score">Total: ${g.home_score + g.away_score} (pred: ${pick.predTotal})</span>
          <span class="live-result">${won ? 'W (+$91)' : 'L (-$100)'}</span></div>`;
      } else {
        const actualHM = g.home_score - g.away_score;
        won = actualHM < pick.predMargin;
        liveHtml = `<div class="pick-live ${won ? 'live-win' : 'live-loss'}">
          <span class="live-label">FINAL</span>
          <span class="live-score">${g.away_team} ${g.away_score} @ ${g.home_team} ${g.home_score}</span>
          <span class="live-result">${won ? 'W (+$91)' : 'L (-$100)'}</span></div>`;
      }
    } else if (g.status !== 'STATUS_SCHEDULED') {
      liveHtml = `<div class="pick-live live-active"><span class="live-label">LIVE Q${g.period} ${g.clock}</span></div>`;
    } else {
      liveHtml = `<div class="pick-live live-scheduled"><span class="live-label">${g.time}</span>
        <span class="live-status">Pre-Game — Bet at -110</span></div>`;
    }

    return `
      <div class="pick-card ${tierClass}">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 7</span>
          <span class="pick-conf">${pick.tier} — ${pick.type === 'total' ? 'TOTAL' : 'SPREAD'}</span>
        </div>
        ${betLine}
        <div class="pick-matchup">${matchupHtml}</div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Signal</span>
            <span class="detail-value">${pick.type === 'total' ? pick.direction : 'AWAY COVERS'}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Strength</span>
            <span class="detail-value">${pick.strength}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Factors</span>
            <span class="detail-value">${pick.factors.length}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Odds</span>
            <span class="detail-value">-110</span>
          </div>
        </div>
        <div class="prism-factors">${factorList}</div>
        ${liveHtml}
      </div>`;
  }

  function renderNOVACard(pick) {
    const g = pick.game;
    const tierClass = pick.tier === 'ELITE' ? 'conf-nova-elite' : pick.tier === 'HIGH' ? 'conf-nova-high' : 'conf-nova';

    const FACTOR_LABELS = {
      both_ppg_118_L5: 'Both PPG 118+ (L5)',
      both_ppg_118_L3: 'Both PPG 118+ (L3)',
      both_total_225_L3: 'Both Totals 225+ (L3)',
      porous_defense: 'Porous Defense',
      home_porous_defense: 'Home Porous Def',
      away_porous_defense: 'Away Porous Def',
      home_scoring_rising: 'Home Scoring Rising',
      away_scoring_rising: 'Away Scoring Rising',
      home_high_totals: 'Home High Totals',
      away_high_totals: 'Away High Totals',
      home_high_scoring: 'Home High Scoring',
      away_high_scoring: 'Away High Scoring',
    };
    const factorList = pick.factors.map(f => FACTOR_LABELS[f] || f).join(', ');
    const matchupHtml = `${teamName(g.away_team)} @ ${teamName(g.home_team)}`;

    let liveHtml = '';
    if (g.status === 'STATUS_FINAL') {
      const actualTotal = g.home_score + g.away_score;
      const won = actualTotal > pick.predTotal;
      liveHtml = `<div class="pick-live ${won ? 'live-win' : 'live-loss'}">
        <span class="live-label">FINAL</span>
        <span class="live-score">Total: ${actualTotal} (pred: ${pick.predTotal})</span>
        <span class="live-result">${won ? 'W (+$91)' : 'L (-$100)'}</span></div>`;
    } else if (g.status !== 'STATUS_SCHEDULED') {
      liveHtml = `<div class="pick-live live-active"><span class="live-label">LIVE Q${g.period} ${g.clock}</span></div>`;
    } else {
      liveHtml = `<div class="pick-live live-scheduled"><span class="live-label">${g.time}</span>
        <span class="live-status">Pre-Game — Bet OVER at -110</span></div>`;
    }

    return `
      <div class="pick-card ${tierClass}">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 8</span>
          <span class="pick-conf">${pick.tier} — NOVA Score ${pick.strength}</span>
        </div>
        <div class="pick-bet-line nova-over">OVER ${pick.predTotal} at -110</div>
        <div class="pick-matchup">${matchupHtml}</div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Home PPG (L5)</span>
            <span class="detail-value">${pick.hPpg5}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Away PPG (L5)</span>
            <span class="detail-value">${pick.aPpg5}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Home Avg Total (L3)</span>
            <span class="detail-value">${pick.hAvgTotal3}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Away Avg Total (L3)</span>
            <span class="detail-value">${pick.aAvgTotal3}</span>
          </div>
          <div class="detail">
            <span class="detail-label">NOVA Score</span>
            <span class="detail-value nova-value">${pick.strength}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Odds</span>
            <span class="detail-value">-110</span>
          </div>
        </div>
        <div class="nova-factors">${factorList}</div>
        ${liveHtml}
      </div>`;
  }

  function renderFusionCard(parlay) {
    const l1 = parlay.leg1;
    const l2 = parlay.leg2;
    const g1 = l1.game;
    const g2 = l2.game;

    let leg1Html = `<div class="fusion-leg">
      <span class="fusion-leg-source ${l1.source === 'NOVA' ? 'fusion-nova' : 'fusion-prism'}">${l1.source}</span>
      <span class="fusion-leg-bet">${l1.label} at -110</span>
      <span class="fusion-leg-game">${teamName(g1.away_team)} @ ${teamName(g1.home_team)}</span>
    </div>`;

    let leg2Html = `<div class="fusion-leg">
      <span class="fusion-leg-source ${l2.source === 'NOVA' ? 'fusion-nova' : 'fusion-prism'}">${l2.source}</span>
      <span class="fusion-leg-bet">${l2.label} at -110</span>
      <span class="fusion-leg-game">${teamName(g2.away_team)} @ ${teamName(g2.home_team)}</span>
    </div>`;

    // Check if any game is final
    let liveHtml = '';
    const g1Final = g1.status === 'STATUS_FINAL';
    const g2Final = g2.status === 'STATUS_FINAL';
    if (g1Final && g2Final) {
      let l1Hit, l2Hit;
      if (l1.type === 'total') {
        const at = g1.home_score + g1.away_score;
        l1Hit = l1.direction === 'OVER' ? at > l1.predTotal : at < l1.predTotal;
      } else {
        l1Hit = (g1.home_score - g1.away_score) < l1.predMargin;
      }
      if (l2.type === 'total') {
        const at = g2.home_score + g2.away_score;
        l2Hit = l2.direction === 'OVER' ? at > l2.predTotal : at < l2.predTotal;
      } else {
        l2Hit = (g2.home_score - g2.away_score) < l2.predMargin;
      }
      const won = l1Hit && l2Hit;
      liveHtml = `<div class="pick-live ${won ? 'live-win' : 'live-loss'}">
        <span class="live-label">FINAL</span>
        <span class="live-result">${won ? 'W (+$264)' : 'L (-$100)'}</span></div>`;
    } else {
      liveHtml = `<div class="pick-live live-scheduled">
        <span class="live-status">Pre-Game — Place 2-Leg Parlay</span></div>`;
    }

    return `
      <div class="pick-card conf-fusion">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 9</span>
          <span class="pick-conf">FUSION PARLAY — +264</span>
        </div>
        <div class="pick-bet-line fusion-payout">2-Leg Parlay at +264 ($100 → $364)</div>
        <div class="fusion-legs">
          ${leg1Html}
          <div class="fusion-plus">+</div>
          ${leg2Html}
        </div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Parlay Odds</span>
            <span class="detail-value fusion-value">+264</span>
          </div>
          <div class="detail">
            <span class="detail-label">Each Leg</span>
            <span class="detail-value">-110</span>
          </div>
          <div class="detail">
            <span class="detail-label">Win Payout</span>
            <span class="detail-value fusion-value">+$264</span>
          </div>
          <div class="detail">
            <span class="detail-label">Break-Even</span>
            <span class="detail-value">27.5%</span>
          </div>
        </div>
        ${liveHtml}
      </div>`;
  }

  function renderPulseCard(parlay) {
    const l1 = parlay.leg1;
    const l2 = parlay.leg2;

    return `
      <div class="pick-card conf-pulse">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 10</span>
          <span class="pick-conf">PULSE PARLAY — +251</span>
        </div>
        <div class="pick-bet-line pulse-payout">2-Leg Player Prop Parlay at +251 ($100 → $351)</div>
        <div class="pulse-legs">
          <div class="pulse-leg">
            <span class="pulse-leg-player">${l1.player}</span>
            <span class="pulse-leg-bet">UNDER ${l1.line} PTS at -115</span>
            <span class="pulse-leg-spike">+${l1.spikePct}% spike</span>
          </div>
          <div class="pulse-plus">+</div>
          <div class="pulse-leg">
            <span class="pulse-leg-player">${l2.player}</span>
            <span class="pulse-leg-bet">UNDER ${l2.line} PTS at -115</span>
            <span class="pulse-leg-spike">+${l2.spikePct}% spike</span>
          </div>
        </div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Parlay Odds</span>
            <span class="detail-value pulse-value">+251</span>
          </div>
          <div class="detail">
            <span class="detail-label">Each Leg</span>
            <span class="detail-value">-115</span>
          </div>
          <div class="detail">
            <span class="detail-label">Win Payout</span>
            <span class="detail-value pulse-value">+$251</span>
          </div>
          <div class="detail">
            <span class="detail-label">Break-Even</span>
            <span class="detail-value">28.5%</span>
          </div>
        </div>
        <div class="pick-live live-scheduled">
          <span class="live-status">Pre-Game — Place 2-Leg Player Prop Parlay</span>
        </div>
      </div>`;
  }

  function renderVaultCard(parlay) {
    const legsHtml = parlay.legs.map((l, i) => `
      <div class="vault-leg">
        <span class="vault-leg-player">${l.player}</span>
        <span class="vault-leg-bet">OVER ${l.line} PTS at -300</span>
        <span class="vault-leg-conf">${l.confidence.toFixed(1)}x conf</span>
      </div>
      ${i < parlay.legs.length - 1 ? '<div class="vault-plus">+</div>' : ''}
    `).join('');

    return `
      <div class="pick-card conf-vault">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 11</span>
          <span class="pick-conf">VAULT PARLAY — +${parlay.odds}</span>
        </div>
        <div class="pick-bet-line vault-payout">${parlay.numLegs}-Leg Floor Parlay at +${parlay.odds} ($100 → $${parlay.odds + 100})</div>
        <div class="vault-legs">
          ${legsHtml}
        </div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Parlay Odds</span>
            <span class="detail-value vault-value">+${parlay.odds}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Each Leg</span>
            <span class="detail-value">-300</span>
          </div>
          <div class="detail">
            <span class="detail-label">Win Payout</span>
            <span class="detail-value vault-value">+$${parlay.odds}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Legs</span>
            <span class="detail-value">${parlay.numLegs}</span>
          </div>
        </div>
        <div class="pick-live live-scheduled">
          <span class="live-status">Pre-Game — Place ${parlay.numLegs}-Leg Player Prop Parlay</span>
        </div>
      </div>`;
  }

  function renderFortressCard(parlay) {
    const legsHtml = parlay.legs.map((l, i) => `
      <div class="fortress-leg">
        <span class="fortress-leg-player">${l.player}</span>
        <span class="fortress-leg-bet">OVER ${l.line} PTS at -200</span>
        <span class="fortress-leg-conf">${l.confidence.toFixed(1)}x conf</span>
      </div>
      ${i < parlay.legs.length - 1 ? '<div class="fortress-plus">+</div>' : ''}
    `).join('');

    return `
      <div class="pick-card conf-fortress">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 12</span>
          <span class="pick-conf">FORTRESS PARLAY — +${parlay.odds}</span>
        </div>
        <div class="pick-bet-line fortress-payout">${parlay.numLegs}-Leg Higher-Floor Parlay at +${parlay.odds} ($100 → $${parlay.odds + 100})</div>
        <div class="fortress-legs">
          ${legsHtml}
        </div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Parlay Odds</span>
            <span class="detail-value fortress-value">+${parlay.odds}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Each Leg</span>
            <span class="detail-value">-200</span>
          </div>
          <div class="detail">
            <span class="detail-label">Win Payout</span>
            <span class="detail-value fortress-value">+$${parlay.odds}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Legs</span>
            <span class="detail-value">${parlay.numLegs}</span>
          </div>
        </div>
        <div class="pick-live live-scheduled">
          <span class="live-status">Pre-Game — Place ${parlay.numLegs}-Leg Player Prop Parlay</span>
        </div>
      </div>`;
  }

  function renderSiegeCard(parlay) {
    const legsHtml = parlay.legs.map((l, i) => {
      const oddsDisplay = l.hasLiveOdds
        ? `<span class="siege-live-tag">LIVE</span> ${siegeFormatOdds(l.liveOdds)} (${l.bestBook})`
        : `Est. ${siegeFormatOdds(l.estOdds)}`;
      const bookInfo = '';

      return `
        <div class="siege-leg">
          <span class="siege-leg-player">${l.player} <span class="siege-leg-team">(${l.team})</span></span>
          <span class="siege-leg-bet">${l.displayLine} PTS ${oddsDisplay}</span>
          <span class="siege-leg-odds">L10: ${l.l10Avg} | Min: ${l.l10Min} | Floor: ${l.ratio}% | EV: ${l.ev ? '+' + (l.ev * 100).toFixed(1) + '%' : '—'} ${bookInfo}</span>
        </div>
        ${i < parlay.legs.length - 1 ? '<div class="siege-plus">+</div>' : ''}
      `;
    }).join('');

    const oddsLabel = parlay.odds >= 0 ? `+${parlay.odds}` : `${parlay.odds}`;
    const payoutAmt = parlay.odds >= 0 ? parlay.odds : Math.round(10000 / Math.abs(parlay.odds));
    const oddsSource = parlay.hasLiveOdds ? 'Live Odds API' : 'Estimated';

    return `
      <div class="pick-card conf-siege">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 13</span>
          <span class="pick-conf">SIEGE — ${oddsLabel} ${parlay.hasLiveOdds ? '🔴 LIVE' : ''}</span>
        </div>
        <div class="pick-bet-line siege-payout">${parlay.numLegs}-Leg EV-Optimized Parlay — ${oddsLabel} ($100 → $${payoutAmt + 100})</div>
        <div class="siege-legs">
          ${legsHtml}
        </div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Parlay Odds</span>
            <span class="detail-value siege-value">${oddsLabel}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Source</span>
            <span class="detail-value">${oddsSource}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Win Payout</span>
            <span class="detail-value siege-value">$${payoutAmt + 100}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Legs</span>
            <span class="detail-value">${parlay.numLegs}</span>
          </div>
        </div>
        <div class="pick-live live-scheduled">
          <span class="live-status">Pre-Game — ${parlay.hasLiveOdds ? 'FanDuel Player Points alternate lines' : 'Use Player Points alternate lines on FanDuel'}</span>
        </div>
      </div>`;
  }

  // ── Rendering: All Games Grid ──────────────────────────────────────────────

  function renderAllGames() {
    const grid = document.getElementById('all-games-grid');
    if (!grid || todayGames.length === 0) return;

    grid.innerHTML = todayGames.map(g => {
      const isP4 = todayCdsPicks.some(p => p.game.id === g.id);
      const isP7 = todayPrismPicks.some(p => p.game.id === g.id);
      const isP8 = todayNovaPicks.some(p => p.game.id === g.id);
      const pickClass = (isP4 || isP7 || isP8) ? 'game-picked' : '';

      let statusHtml;
      if (g.status === 'STATUS_FINAL') {
        statusHtml = `<span class="game-status final">Final: ${g.home_score}-${g.away_score}</span>`;
      } else if (g.status === 'STATUS_IN_PROGRESS' || g.status === 'STATUS_HALFTIME') {
        statusHtml = `<span class="game-status live">Q${g.period} ${g.clock}: ${g.home_score}-${g.away_score}</span>`;
      } else {
        statusHtml = `<span class="game-status scheduled">${g.time}</span>`;
      }

      let badge = '';
      const badges = [];
      if (isP4) badges.push('P4');
      if (isP7) badges.push('P7');
      if (isP8) badges.push('P8');
      if (badges.length > 0) badge = '<span class="game-badge">' + badges.join(' + ') + '</span>';

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
    renderCDSHistory();
    renderPRISMHistory();
    renderNOVAHistory();
    renderFusionHistory();
    renderPulseHistory();
    renderVaultHistory();
    renderFortressHistory();
    renderSiegeHistory();
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
          <div class="summary-title">Play 4 — CDS Moneyline</div>
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
      </div>`;
  }

  function renderPRISMHistory() {
    const tbody = document.getElementById('prism-history-body');
    if (!tbody) return;

    let filtered = prismHistoryPicks;
    if (currentPrismHistoryPeriod !== 'all') {
      const cutoff = getCutoffDate(parseInt(currentPrismHistoryPeriod));
      filtered = prismHistoryPicks.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="muted">No PRISM picks in this period</td></tr>';
    } else {
      const FACTOR_SHORT = {
        elite_def: 'Elite Def', good_def: 'Good Def',
        both_poor_def: 'Both Poor Def', mediocre_def: 'Med Def',
        pace_drop_strong: 'Pace Drop+', pace_drop: 'Pace Drop',
        pace_up_strong: 'Pace Up+', pace_up: 'Pace Up',
        both_cooling: 'Cooling', cooling: 'Cool',
        both_heating: 'Heating', heating: 'Heat',
        low_pace_teams: 'Low Pace', high_pace_teams: 'High Pace',
        extreme_luck_gap: 'Luck Gap+', large_luck_gap: 'Luck Gap',
        luck_gap: 'Luck Reg', home_declining: 'Home Down',
        away_improving: 'Away Up',
      };

      tbody.innerHTML = filtered.map(p => {
        const resClass = p.hit ? 'result-win' : 'result-loss';
        const resText = p.hit ? 'W' : 'L';
        const pnlText = p.hit ? '+$91' : '-$100';
        const tierClass = p.tier === 'ELITE' ? 'conf-prism-elite' : p.tier === 'HIGH' ? 'conf-prism-high' : 'conf-prism';
        const dateFormatted = formatDate(p.date);
        const factorStr = p.factors.map(f => FACTOR_SHORT[f] || f).join(', ');

        let betCol;
        if (p.type === 'total') {
          const dirClass = p.direction === 'OVER' ? 'prism-over' : 'prism-under';
          betCol = `<span class="${dirClass}">${p.direction}</span> ${p.predTotal}`;
        } else {
          betCol = `${p.betTeam} +${Math.abs(p.predMargin).toFixed(1)}`;
        }

        let actualCol;
        if (p.type === 'total') {
          actualCol = `${p.actualTotal}`;
        } else {
          actualCol = `${p.actualHomeMargin > 0 ? p.home : p.away} by ${Math.abs(p.actualHomeMargin)}`;
        }

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td>${p.type === 'total' ? 'Total' : 'Spread'}</td>
            <td>${betCol}</td>
            <td>${p.away} @ ${p.home}</td>
            <td><span class="badge ${tierClass}">${p.tier}</span></td>
            <td>${p.strength}</td>
            <td>${actualCol}</td>
            <td><span class="badge ${resClass}">${resText}</span></td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderPRISMSummary(filtered);
  }

  function renderPRISMSummary(filtered) {
    const summary = document.getElementById('prism-history-summary');
    if (!summary) return;

    const hits = filtered.filter(p => p.hit).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';
    const totalWagered = total * 100;
    const roi = totalWagered > 0 ? ((pnl / totalWagered) * 100).toFixed(0) : '0';

    const totals = filtered.filter(p => p.type === 'total');
    const spreads = filtered.filter(p => p.type === 'spread');
    const totalHits = totals.filter(p => p.hit).length;
    const spreadHits = spreads.filter(p => p.hit).length;
    const totalPnl = totals.reduce((s, p) => s + p.pnl, 0);
    const spreadPnl = spreads.reduce((s, p) => s + p.pnl, 0);

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-prism">
          <div class="summary-title">Play 7 — PRISM Combined (at -110)</div>
          <div class="summary-stat">
            <span class="summary-record">${hits}-${total - hits}</span>
            <span class="summary-pct">${hitRate}%</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl} (ROI: ${roi >= 0 ? '+' : ''}${roi}%)
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">Totals / Spread Breakdown</div>
          <div class="summary-stat">
            <span class="summary-record">Totals: ${totalHits}-${totals.length - totalHits}</span>
            <span class="summary-pct">${totals.length > 0 ? (totalHits / totals.length * 100).toFixed(1) + '%' : '—'}</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Spreads: ${spreadHits}-${spreads.length - spreadHits}</span>
            <span class="summary-pct">${spreads.length > 0 ? (spreadHits / spreads.length * 100).toFixed(1) + '%' : '—'}</span>
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">P&amp;L by Type</div>
          <div class="summary-pnl ${totalPnl >= 0 ? 'result-win' : 'result-loss'}">
            Totals: ${totalPnl >= 0 ? '+' : ''}$${totalPnl}
          </div>
          <div class="summary-pnl ${spreadPnl >= 0 ? 'result-win' : 'result-loss'}">
            Spreads: ${spreadPnl >= 0 ? '+' : ''}$${spreadPnl}
          </div>
        </div>
      </div>`;
  }

  function renderNOVAHistory() {
    const tbody = document.getElementById('nova-history-body');
    if (!tbody) return;

    let filtered = novaHistoryPicks;
    if (currentNovaHistoryPeriod !== 'all') {
      const cutoff = getCutoffDate(parseInt(currentNovaHistoryPeriod));
      filtered = novaHistoryPicks.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="muted">No NOVA picks in this period</td></tr>';
    } else {
      tbody.innerHTML = filtered.map(p => {
        const resClass = p.hit ? 'result-win' : 'result-loss';
        const resText = p.hit ? 'W' : 'L';
        const pnlText = p.hit ? '+$91' : '-$100';
        const tierClass = p.tier === 'ELITE' ? 'conf-nova-elite' : p.tier === 'HIGH' ? 'conf-nova-high' : 'conf-nova';
        const dateFormatted = formatDate(p.date);

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td><span class="nova-over">OVER</span> ${p.predTotal}</td>
            <td>${p.away} @ ${p.home}</td>
            <td><span class="badge ${tierClass}">${p.tier}</span></td>
            <td>${p.strength}</td>
            <td>${p.hPpg5} / ${p.aPpg5}</td>
            <td>${p.actualTotal}</td>
            <td><span class="badge ${resClass}">${resText}</span></td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderNOVASummary(filtered);
  }

  function renderNOVASummary(filtered) {
    const summary = document.getElementById('nova-history-summary');
    if (!summary) return;

    const hits = filtered.filter(p => p.hit).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';
    const totalWagered = total * 100;
    const roi = totalWagered > 0 ? ((pnl / totalWagered) * 100).toFixed(0) : '0';

    const elitePicks = filtered.filter(p => p.tier === 'ELITE');
    const eliteHits = elitePicks.filter(p => p.hit).length;
    const elitePnl = elitePicks.reduce((s, p) => s + p.pnl, 0);

    const highPicks = filtered.filter(p => p.tier === 'HIGH');
    const highHits = highPicks.filter(p => p.hit).length;

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-nova">
          <div class="summary-title">Play 8 — NOVA Compound OVER (at -110)</div>
          <div class="summary-stat">
            <span class="summary-record">${hits}-${total - hits}</span>
            <span class="summary-pct">${hitRate}%</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl} (ROI: ${roi >= 0 ? '+' : ''}${roi}%)
          </div>
        </div>
        <div class="summary-card summary-nova-elite">
          <div class="summary-title">ELITE Tier</div>
          <div class="summary-stat">
            <span class="summary-record">${elitePicks.length > 0 ? eliteHits + '-' + (elitePicks.length - eliteHits) : '—'}</span>
            <span class="summary-pct">${elitePicks.length > 0 ? (eliteHits / elitePicks.length * 100).toFixed(1) + '%' : '—'}</span>
          </div>
          <div class="summary-pnl ${elitePnl >= 0 ? 'result-win' : 'result-loss'}">
            ${elitePicks.length > 0 ? (elitePnl >= 0 ? '+' : '') + '$' + elitePnl : '—'}
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">HIGH Tier</div>
          <div class="summary-stat">
            <span class="summary-record">${highPicks.length > 0 ? highHits + '-' + (highPicks.length - highHits) : '—'}</span>
            <span class="summary-pct">${highPicks.length > 0 ? (highHits / highPicks.length * 100).toFixed(1) + '%' : '—'}</span>
          </div>
        </div>
      </div>`;
  }

  function renderFusionHistory() {
    const tbody = document.getElementById('fusion-history-body');
    if (!tbody) return;

    let filtered = fusionHistoryPicks;
    if (currentFusionHistoryPeriod !== 'all') {
      const cutoff = getCutoffDate(parseInt(currentFusionHistoryPeriod));
      filtered = fusionHistoryPicks.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="muted">No FUSION parlays in this period</td></tr>';
    } else {
      tbody.innerHTML = filtered.map(p => {
        const resClass = p.hit ? 'result-win' : 'result-loss';
        const resText = p.hit ? 'W' : 'L';
        const pnlText = p.hit ? '+$264' : '-$100';
        const dateFormatted = formatDate(p.date);
        const l1Class = p.leg1Hit ? 'result-win' : 'result-loss';
        const l2Class = p.leg2Hit ? 'result-win' : 'result-loss';

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td><span class="${p.leg1Source === 'NOVA' ? 'fusion-nova' : 'fusion-prism'}">${p.leg1Source}</span> ${p.leg1Label}</td>
            <td>${p.leg1Game}</td>
            <td><span class="badge ${l1Class}">${p.leg1Hit ? 'W' : 'L'}</span></td>
            <td><span class="${p.leg2Source === 'NOVA' ? 'fusion-nova' : 'fusion-prism'}">${p.leg2Source}</span> ${p.leg2Label}</td>
            <td>${p.leg2Game}</td>
            <td><span class="badge ${resClass}">${resText}</span></td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderFusionSummary(filtered);
  }

  function renderFusionSummary(filtered) {
    const summary = document.getElementById('fusion-history-summary');
    if (!summary) return;

    const hits = filtered.filter(p => p.hit).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';
    const totalWagered = total * 100;
    const roi = totalWagered > 0 ? ((pnl / totalWagered) * 100).toFixed(0) : '0';

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-fusion">
          <div class="summary-title">Play 9 — FUSION 2-Leg Parlay (at +264)</div>
          <div class="summary-stat">
            <span class="summary-record">${hits}-${total - hits}</span>
            <span class="summary-pct">${hitRate}%</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl} (ROI: ${roi >= 0 ? '+' : ''}${roi}%)
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">Parlay Details</div>
          <div class="summary-stat">
            <span class="summary-record">Each leg: -110</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Parlay: +264</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Break-even: 27.5%</span>
          </div>
        </div>
      </div>`;
  }

  function renderPulseHistory() {
    const tbody = document.getElementById('pulse-history-body');
    if (!tbody) return;

    let filtered = pulseHistoryPicks;
    if (currentPulseHistoryPeriod !== 'all') {
      const latestPulseDate = getLatestDateFromPicks(pulseHistoryPicks);
      const cutoff = getCutoffDate(parseInt(currentPulseHistoryPeriod), latestPulseDate);
      filtered = pulseHistoryPicks.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="muted">No PULSE parlays in this period</td></tr>';
    } else {
      tbody.innerHTML = filtered.map(p => {
        const resClass = p.won ? 'result-win' : 'result-loss';
        const resText = p.won ? 'W' : 'L';
        const pnlText = p.won ? '+$251' : '-$100';
        const dateFormatted = formatDate(p.date);
        const l1Class = p.leg1Hit ? 'result-win' : 'result-loss';
        const l2Class = p.leg2Hit ? 'result-win' : 'result-loss';

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td>${p.leg1Player} <span class="pulse-under">U${p.leg1Line}</span> <span class="badge ${l1Class}">${p.leg1Hit ? 'W' : 'L'}</span> (got ${p.leg1Actual})</td>
            <td>${p.leg1Game}</td>
            <td>${p.leg2Player} <span class="pulse-under">U${p.leg2Line}</span> <span class="badge ${l2Class}">${p.leg2Hit ? 'W' : 'L'}</span> (got ${p.leg2Actual})</td>
            <td>${p.leg2Game}</td>
            <td>+251</td>
            <td><span class="badge ${resClass}">${resText}</span></td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderPulseSummary(filtered);
  }

  function renderPulseSummary(filtered) {
    const summary = document.getElementById('pulse-history-summary');
    if (!summary) return;

    const hits = filtered.filter(p => p.won).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';
    const totalWagered = total * 100;
    const roi = totalWagered > 0 ? ((pnl / totalWagered) * 100).toFixed(0) : '0';

    // Individual leg accuracy
    const l1Hits = filtered.filter(p => p.leg1Hit).length;
    const l2Hits = filtered.filter(p => p.leg2Hit).length;
    const legTotal = total * 2;
    const legHits = l1Hits + l2Hits;
    const legPct = legTotal > 0 ? ((legHits / legTotal) * 100).toFixed(1) : '0';

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-pulse">
          <div class="summary-title">Play 10 — PULSE Player Prop Parlay (at +251)</div>
          <div class="summary-stat">
            <span class="summary-record">${hits}-${total - hits}</span>
            <span class="summary-pct">${hitRate}%</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl} (ROI: ${roi >= 0 ? '+' : ''}${roi}%)
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">Leg Details</div>
          <div class="summary-stat">
            <span class="summary-record">Individual legs: ${legHits}/${legTotal} (${legPct}%)</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Each leg: -115 | Parlay: +251</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Break-even: 28.5%</span>
          </div>
        </div>
      </div>`;
  }

  // ── VAULT History ─────────────────────────────────────────────────────────

  function renderVaultHistory() {
    const tbody = document.getElementById('vault-history-body');
    if (!tbody) return;

    let filtered = vaultHistoryPicks;
    if (currentVaultHistoryPeriod !== 'all') {
      const latestVaultDate = getLatestDateFromPicks(vaultHistoryPicks);
      const cutoff = getCutoffDate(parseInt(currentVaultHistoryPeriod), latestVaultDate);
      filtered = vaultHistoryPicks.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="muted">No VAULT parlays in this period</td></tr>';
    } else {
      tbody.innerHTML = filtered.map(p => {
        const resClass = p.won ? 'result-win' : 'result-loss';
        const resText = p.won ? 'W' : 'L';
        const pnlText = p.won ? `+$${p.odds}` : '-$100';
        const dateFormatted = formatDate(p.date);

        const legsStr = p.legs.map(l => {
          const lClass = l.hit ? 'result-win' : 'result-loss';
          return `${l.player} <span class="vault-over">O${l.line}</span> <span class="badge ${lClass}">${l.hit ? 'W' : 'L'}</span> (${l.actual})`;
        }).join(' <span class="vault-sep">|</span> ');

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td class="vault-legs-cell">${legsStr}</td>
            <td>${p.numLegs}</td>
            <td>+${p.odds}</td>
            <td><span class="badge ${resClass}">${resText}</span></td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderVaultSummary(filtered);
  }

  function renderVaultSummary(filtered) {
    const summary = document.getElementById('vault-history-summary');
    if (!summary) return;

    const hits = filtered.filter(p => p.won).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';
    const totalWagered = total * 100;
    const roi = totalWagered > 0 ? ((pnl / totalWagered) * 100).toFixed(0) : '0';

    // Leg accuracy
    let legHits = 0, legTotal = 0;
    for (const p of filtered) {
      for (const l of p.legs) {
        legTotal++;
        if (l.hit) legHits++;
      }
    }
    const legPct = legTotal > 0 ? ((legHits / legTotal) * 100).toFixed(1) : '0';

    const avgLegs = total > 0 ? (filtered.reduce((s, p) => s + p.numLegs, 0) / total).toFixed(1) : '0';
    const avgOdds = total > 0 ? Math.round(filtered.reduce((s, p) => s + p.odds, 0) / total) : 0;

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-vault">
          <div class="summary-title">Play 11 — VAULT Multi-Leg Floor Parlay (avg +${avgOdds})</div>
          <div class="summary-stat">
            <span class="summary-record">${hits}-${total - hits}</span>
            <span class="summary-pct">${hitRate}%</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl} (ROI: ${roi >= 0 ? '+' : ''}${roi}%)
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">Leg Details</div>
          <div class="summary-stat">
            <span class="summary-record">Individual legs: ${legHits}/${legTotal} (${legPct}%)</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Each leg: -300 | Avg parlay: +${avgOdds}</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Avg legs: ${avgLegs} per parlay</span>
          </div>
        </div>
      </div>`;
  }

  // ── FORTRESS History Rendering ──────────────────────────────────────────

  function renderFortressHistory() {
    const tbody = document.getElementById('fortress-history-body');
    if (!tbody) return;

    let filtered = fortressHistoryPicks;
    if (currentFortressHistoryPeriod !== 'all') {
      const latestFortressDate = getLatestDateFromPicks(fortressHistoryPicks);
      const cutoff = getCutoffDate(parseInt(currentFortressHistoryPeriod), latestFortressDate);
      filtered = fortressHistoryPicks.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="muted">No FORTRESS parlays in this period</td></tr>';
    } else {
      tbody.innerHTML = filtered.map(p => {
        const resClass = p.won ? 'result-win' : 'result-loss';
        const resText = p.won ? 'W' : 'L';
        const pnlText = p.won ? `+$${p.odds}` : '-$100';
        const dateFormatted = formatDate(p.date);

        const legsStr = p.legs.map(l => {
          const lClass = l.hit ? 'result-win' : 'result-loss';
          return `${l.player} <span class="fortress-over">O${l.line}</span> <span class="badge ${lClass}">${l.hit ? 'W' : 'L'}</span> (${l.actual})`;
        }).join(' <span class="fortress-sep">|</span> ');

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td class="fortress-legs-cell">${legsStr}</td>
            <td>${p.numLegs}</td>
            <td>+${p.odds}</td>
            <td><span class="badge ${resClass}">${resText}</span></td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderFortressSummary(filtered);
  }

  function renderFortressSummary(filtered) {
    const summary = document.getElementById('fortress-history-summary');
    if (!summary) return;

    const hits = filtered.filter(p => p.won).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';
    const totalWagered = total * 100;
    const roi = totalWagered > 0 ? ((pnl / totalWagered) * 100).toFixed(0) : '0';

    // Leg accuracy
    let legHits = 0, legTotal = 0;
    for (const p of filtered) {
      for (const l of p.legs) {
        legTotal++;
        if (l.hit) legHits++;
      }
    }
    const legPct = legTotal > 0 ? ((legHits / legTotal) * 100).toFixed(1) : '0';

    const avgLegs = total > 0 ? (filtered.reduce((s, p) => s + p.numLegs, 0) / total).toFixed(1) : '0';
    const avgOdds = total > 0 ? Math.round(filtered.reduce((s, p) => s + p.odds, 0) / total) : 0;

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-fortress">
          <div class="summary-title">Play 12 — FORTRESS Higher-Floor Parlay (avg +${avgOdds})</div>
          <div class="summary-stat">
            <span class="summary-record">${hits}-${total - hits}</span>
            <span class="summary-pct">${hitRate}%</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl} (ROI: ${roi >= 0 ? '+' : ''}${roi}%)
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">Leg Details</div>
          <div class="summary-stat">
            <span class="summary-record">Individual legs: ${legHits}/${legTotal} (${legPct}%)</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Each leg: -200 | Avg parlay: +${avgOdds}</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Avg legs: ${avgLegs} per parlay</span>
          </div>
        </div>
      </div>`;
  }

  // ── SIEGE History Rendering ────────────────────────────────────────────

  function renderSiegeHistory() {
    const tbody = document.getElementById('siege-history-body');
    if (!tbody) return;

    let filtered = siegeHistoryPicks;
    if (currentSiegeHistoryPeriod !== 'all') {
      const latestSiegeDate = getLatestDateFromPicks(siegeHistoryPicks);
      const cutoff = getCutoffDate(parseInt(currentSiegeHistoryPeriod), latestSiegeDate);
      filtered = siegeHistoryPicks.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="muted">No SIEGE parlays in this period</td></tr>';
    } else {
      tbody.innerHTML = filtered.map(p => {
        const resClass = p.won ? 'result-win' : 'result-loss';
        const resText = p.won ? 'W' : 'L';
        const pnlText = p.won ? `+$${p.pnl}` : '-$100';
        const dateFormatted = formatDate(p.date);

        const legsStr = p.legs.map(l => {
          const lClass = l.hit ? 'result-win' : 'result-loss';
          return `${l.player} <span class="siege-over">${l.displayLine}</span> <span class="siege-odds-tag">${siegeFormatOdds(l.estOdds)}</span> <span class="badge ${lClass}">${l.hit ? 'W' : 'L'}</span> (${l.actual} pts)`;
        }).join(' <span class="siege-sep">|</span> ');

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td class="siege-legs-cell">${legsStr}</td>
            <td>${p.numLegs}</td>
            <td>${siegeFormatOdds(p.odds)}</td>
            <td><span class="badge ${resClass}">${resText}</span></td>
            <td class="${resClass}">${pnlText}</td>
          </tr>`;
      }).join('');
    }

    renderSiegeSummary(filtered);
  }

  function renderSiegeSummary(filtered) {
    const summary = document.getElementById('siege-history-summary');
    if (!summary) return;

    const hits = filtered.filter(p => p.won).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';
    const totalWagered = total * 100;
    const roi = totalWagered > 0 ? ((pnl / totalWagered) * 100).toFixed(0) : '0';

    let legHits = 0, legTotal = 0;
    for (const p of filtered) {
      for (const l of p.legs) {
        legTotal++;
        if (l.hit) legHits++;
      }
    }
    const legPct = legTotal > 0 ? ((legHits / legTotal) * 100).toFixed(1) : '0';

    const avgLegs = total > 0 ? (filtered.reduce((s, p) => s + p.numLegs, 0) / total).toFixed(1) : '0';
    const avgOdds = total > 0 ? Math.round(filtered.reduce((s, p) => s + p.odds, 0) / total) : 0;

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-siege">
          <div class="summary-title">Play 13 — SIEGE EV-Optimized Parlay (avg ${siegeFormatOdds(avgOdds)})</div>
          <div class="summary-stat">
            <span class="summary-record">${hits}-${total - hits}</span>
            <span class="summary-pct">${hitRate}%</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl} (ROI: ${roi >= 0 ? '+' : ''}${roi}%)
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">Leg Details (EV-Optimized Floors + FanDuel Odds)</div>
          <div class="summary-stat">
            <span class="summary-record">Individual legs: ${legHits}/${legTotal} (${legPct}%)</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Dynamic floors (55–82%) optimized per player for max EV</span>
          </div>
          <div class="summary-stat">
            <span class="summary-record">Avg legs: ${avgLegs} per parlay</span>
          </div>
        </div>
      </div>`;
  }

  // ── Metrics ────────────────────────────────────────────────────────────────

  function updateMetrics() {
    const el = (id) => document.getElementById(id);

    el('metric-picks').textContent = todayCdsPicks.length + todayPrismPicks.length + todayNovaPicks.length + todayFusionParlays.length + todayPulseParlays.length + todayVaultParlays.length + todayFortressParlays.length + todaySiegeParlays.length;
    el('metric-games').textContent = todayGames.length;

    // Play 4 CDS accuracy
    if (cdsHistoryPicks.length > 0) {
      const cdsWins = cdsHistoryPicks.filter(p => p.favWon).length;
      const cdsTotal = cdsHistoryPicks.length;
      const cdsAcc = ((cdsWins / cdsTotal) * 100).toFixed(1);
      const cdsPnl = cdsHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      el('metric-accuracy').textContent = cdsAcc + '%';

      // Play 7 PRISM accuracy
      const prismHits = prismHistoryPicks.filter(p => p.hit).length;
      const prismTotal = prismHistoryPicks.length;
      if (prismTotal > 0) {
        const prismAcc = ((prismHits / prismTotal) * 100).toFixed(1);
        el('metric-cds-accuracy').textContent = prismAcc + '%';
      } else {
        el('metric-cds-accuracy').textContent = '—';
      }

      // Play 8 NOVA accuracy
      const novaHits = novaHistoryPicks.filter(p => p.hit).length;
      const novaTotal = novaHistoryPicks.length;
      const novaMetric = document.getElementById('metric-nova-accuracy');
      if (novaMetric && novaTotal > 0) {
        novaMetric.textContent = ((novaHits / novaTotal) * 100).toFixed(1) + '%';
      }

      // Play 9 FUSION metrics
      const fusionWins = fusionHistoryPicks.filter(p => p.won).length;
      const fusionTotal = fusionHistoryPicks.length;
      const fusionPnl = fusionHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      const fusionMetric = document.getElementById('metric-fusion-accuracy');
      if (fusionMetric && fusionTotal > 0) {
        const fusionRoi = ((fusionPnl / (fusionTotal * 100)) * 100).toFixed(0);
        fusionMetric.textContent = (fusionRoi >= 0 ? '+' : '') + fusionRoi + '%';
      }

      // Play 10 PULSE metrics
      const pulseWins = pulseHistoryPicks.filter(p => p.won).length;
      const pulseTotal = pulseHistoryPicks.length;
      const pulsePnl = pulseHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      const pulseMetric = document.getElementById('metric-pulse-accuracy');
      if (pulseMetric && pulseTotal > 0) {
        const pulseRoi = ((pulsePnl / (pulseTotal * 100)) * 100).toFixed(0);
        pulseMetric.textContent = (pulseRoi >= 0 ? '+' : '') + pulseRoi + '%';
      }

      // Play 11 VAULT metrics
      const vaultWins = vaultHistoryPicks.filter(p => p.won).length;
      const vaultTotal = vaultHistoryPicks.length;
      const vaultPnl = vaultHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      const vaultMetric = document.getElementById('metric-vault-accuracy');
      if (vaultMetric && vaultTotal > 0) {
        const vaultRoi = ((vaultPnl / (vaultTotal * 100)) * 100).toFixed(0);
        vaultMetric.textContent = (vaultRoi >= 0 ? '+' : '') + vaultRoi + '%';
      }

      // Play 12: FORTRESS ROI
      const fortressTotal = fortressHistoryPicks.length;
      const fortressWins = fortressHistoryPicks.filter(p => p.won).length;
      const fortressPnl = fortressHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      const fortressMetric = document.getElementById('metric-fortress-accuracy');
      if (fortressMetric && fortressTotal > 0) {
        const fortressRoi = ((fortressPnl / (fortressTotal * 100)) * 100).toFixed(0);
        fortressMetric.textContent = (fortressRoi >= 0 ? '+' : '') + fortressRoi + '%';
      }

      // Play 13: SIEGE accuracy
      const siegeTotal = siegeHistoryPicks.length;
      const siegeWins = siegeHistoryPicks.filter(p => p.won).length;
      const siegePnl = siegeHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      const siegeMetric = document.getElementById('metric-siege-accuracy');
      if (siegeMetric && siegeTotal > 0) {
        const siegeAcc = ((siegeWins / siegeTotal) * 100).toFixed(1);
        siegeMetric.textContent = siegeAcc + '%';
      }

      // Combined ROI (all models)
      const prismPnl = prismHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      const novaPnl = novaHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      const combinedBets = cdsTotal + prismTotal + novaTotal + fusionTotal + pulseTotal + vaultTotal + fortressTotal + siegeTotal;
      const combinedPnl = cdsPnl + prismPnl + novaPnl + fusionPnl + pulsePnl + vaultPnl + fortressPnl + siegePnl;
      const roi = combinedBets > 0 ? ((combinedPnl / (combinedBets * 100)) * 100).toFixed(0) : '0';
      el('metric-roi').textContent = (roi >= 0 ? '+' : '') + roi + '%';

      // Combined record
      const combinedWins = cdsWins + prismHits + novaHits + fusionWins + pulseWins + vaultWins + fortressWins + siegeWins;
      const combinedTotal = cdsTotal + prismTotal + novaTotal + fusionTotal + pulseTotal + vaultTotal + fortressTotal + siegeTotal;
      el('metric-record').textContent = `${combinedWins}-${combinedTotal - combinedWins}`;
    }
  }

  // ── Live Score Updates ─────────────────────────────────────────────────────

  async function refreshScores() {
    if (!modelReady) return;
    try {
      await fetchTodayGames();
      // Regenerate all today's predictions with fresh game data
      runCDSPredictions();
      runPRISMPredictions();
      runNOVAPredictions();
      buildFusionParlays();
      generateTodayPulsePicks();
      generateTodayVaultPicks();
      generateTodayFortressPicks();
      await generateTodaySiegePicks();
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
    document.querySelectorAll('.cds-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.cds-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentCdsHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.prism-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.prism-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentPrismHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.nova-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.nova-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentNovaHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.fusion-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.fusion-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFusionHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.pulse-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.pulse-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentPulseHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.vault-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.vault-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentVaultHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.fortress-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.fortress-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFortressHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.siege-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.siege-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentSiegeHistoryPeriod = btn.dataset.period;
        renderHistory();
      });
    });

    document.querySelectorAll('.master-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.master-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentMasterHistoryPeriod = btn.dataset.period;
        renderMasterHistory();
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

  function getCutoffDate(days, referenceDate) {
    // Use model-specific latest date if provided, otherwise global latest.
    // This prevents models with older data (PULSE/VAULT) from being filtered
    // against a much newer global latest date (from ESPN-fetched season data).
    const latestDate = referenceDate || getLatestDataDate();
    const d = latestDate ? parseDateStr(latestDate) : new Date();
    d.setDate(d.getDate() - days);
    const cutoff = d.toISOString().slice(0, 10).replace(/-/g, '');
    console.log(`[FILTER] getCutoffDate(${days}): latestDate=${latestDate}, cutoff=${cutoff}`);
    return cutoff;
  }

  function getLatestDateFromPicks(picks) {
    let latest = '';
    for (const p of picks) { if (p.date > latest) latest = p.date; }
    return latest;
  }

  function parseDateStr(s) {
    // '20250224' → Date(2025, 1, 24)
    return new Date(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8));
  }

  function getLatestDataDate() {
    let latest = '';
    for (const p of cdsHistoryPicks) { if (p.date > latest) latest = p.date; }
    for (const p of prismHistoryPicks) { if (p.date > latest) latest = p.date; }
    for (const p of novaHistoryPicks) { if (p.date > latest) latest = p.date; }
    for (const p of fusionHistoryPicks) { if (p.date > latest) latest = p.date; }
    for (const p of pulseHistoryPicks) { if (p.date > latest) latest = p.date; }
    for (const p of vaultHistoryPicks) { if (p.date > latest) latest = p.date; }
    for (const p of fortressHistoryPicks) { if (p.date > latest) latest = p.date; }
    for (const p of siegeHistoryPicks) { if (p.date > latest) latest = p.date; }
    if (playerBoxScores && playerBoxScores.length > 0) {
      for (const g of playerBoxScores) { if (g.date > latest) latest = g.date; }
    }
    if (seasonData && seasonData.length > 0) {
      for (const g of seasonData) { if (g.date > latest) latest = g.date; }
    }
    console.log(`[FILTER] getLatestDataDate=${latest}, historyLens: cds=${cdsHistoryPicks.length} prism=${prismHistoryPicks.length} pulse=${pulseHistoryPicks.length} vault=${vaultHistoryPicks.length} fortress=${fortressHistoryPicks.length} siege=${siegeHistoryPicks.length}`);
    return latest || '';
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
