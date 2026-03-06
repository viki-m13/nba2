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
  let playerBoxScores = [];      // Player box score data for PULSE model
  let seasonData = [];          // Full season game data for model training
  let modelReady = false;
  let currentCdsHistoryPeriod = 'all';
  let currentPrismHistoryPeriod = 'all';
  let currentNovaHistoryPeriod = 'all';
  let currentFusionHistoryPeriod = 'all';
  let currentPulseHistoryPeriod = 'all';
  let useProxy = false;        // True when running on Vercel (CORS proxy available)

  const PRISMModel = window.ParlayEngine.PRISMModel;
  const NOVAModel = window.ParlayEngine.NOVAModel;
  const PULSEModel = window.ParlayEngine.PULSEModel;

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
      buildCDSHistory();
      buildPRISMHistory();
      buildNOVAHistory();
      buildFusionHistory();
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

    console.log(`[PULSE] Built history: ${pulseHistoryPicks.length} parlays`);
  }

  // ── Rendering: Today's Picks ───────────────────────────────────────────────

  function renderPicks() {
    const loading = document.getElementById('picks-loading');
    const cdsContainer = document.getElementById('cds-container');
    const empty = document.getElementById('picks-empty');
    const allSection = document.getElementById('all-games-section');

    loading.style.display = 'none';

    const hasPicks = todayCdsPicks.length > 0 || todayPrismPicks.length > 0 || todayNovaPicks.length > 0 || todayFusionParlays.length > 0 || todayPulseParlays.length > 0;

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
      const cutoff = getCutoffDate(parseInt(currentPulseHistoryPeriod));
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

  // ── Metrics ────────────────────────────────────────────────────────────────

  function updateMetrics() {
    const el = (id) => document.getElementById(id);

    el('metric-picks').textContent = todayCdsPicks.length + todayPrismPicks.length + todayNovaPicks.length + todayFusionParlays.length + todayPulseParlays.length;
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

      // Combined ROI (all models)
      const prismPnl = prismHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      const novaPnl = novaHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      const combinedBets = cdsTotal + prismTotal + novaTotal + fusionTotal + pulseTotal;
      const combinedPnl = cdsPnl + prismPnl + novaPnl + fusionPnl + pulsePnl;
      const roi = combinedBets > 0 ? ((combinedPnl / (combinedBets * 100)) * 100).toFixed(0) : '0';
      el('metric-roi').textContent = (roi >= 0 ? '+' : '') + roi + '%';

      // Combined record
      const combinedWins = cdsWins + prismHits + novaHits + fusionWins + pulseWins;
      const combinedTotal = cdsTotal + prismTotal + novaTotal + fusionTotal + pulseTotal;
      el('metric-record').textContent = `${combinedWins}-${combinedTotal - combinedWins}`;
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
