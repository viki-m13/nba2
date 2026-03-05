// =============================================================================
// MAIN APP CONTROLLER — NBA Dominance System (Play 4 CDS + Play 7 PRISM + Play 8 APEX)
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
  let todayApexPicks = [];       // Games flagged by APEX model (Play 8: adaptive pace)
  let apexHistoryPicks = [];     // Historical Play 8 APEX picks with results
  let seasonData = [];          // Full season game data for model training
  let modelReady = false;
  let currentCdsHistoryPeriod = 'all';
  let currentPrismHistoryPeriod = 'all';
  let currentApexHistoryPeriod = 'all';
  let useProxy = false;        // True when running on Vercel (CORS proxy available)

  const PRISMModel = window.ParlayEngine.PRISMModel;
  const APEXModel = window.ParlayEngine.APEXModel;

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
      runAPEXPredictions();
      buildCDSHistory();
      buildPRISMHistory();
      buildAPEXHistory();
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
    APEXModel.teamHistory = {};
    for (const g of seasonData) {
      CDSModel.updateTeam(g.home_team, g.home_score, g.away_score, g.date);
      CDSModel.updateTeam(g.away_team, g.away_score, g.home_score, g.date);
      PRISMModel.updateTeam(g.home_team, g.home_score, g.away_score, g.date);
      PRISMModel.updateTeam(g.away_team, g.away_score, g.home_score, g.date);
      APEXModel.updateTeam(g.home_team, g.home_score, g.away_score, g.date, true);
      APEXModel.updateTeam(g.away_team, g.away_score, g.home_score, g.date, false);
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

  // ── APEX Predictions (Play 8) ──────────────────────────────────────────

  function runAPEXPredictions() {
    todayApexPicks = [];

    for (const game of todayGames) {
      const picks = APEXModel.predictGame(game.home_team, game.away_team);
      if (picks) {
        for (const pick of picks) {
          todayApexPicks.push({ game, ...pick });
        }
      }
    }

    const tierOrder = { ELITE: 0, HIGH: 1, STRONG: 2 };
    todayApexPicks.sort((a, b) => {
      const to = (tierOrder[a.tier] || 9) - (tierOrder[b.tier] || 9);
      if (to !== 0) return to;
      return b.strength - a.strength;
    });

    console.log(`[APEX] ${todayApexPicks.length} picks from ${todayGames.length} games`);
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

  function buildAPEXHistory() {
    apexHistoryPicks = [];

    const apex = Object.create(APEXModel);
    apex.teamHistory = {};

    const sorted = [...seasonData].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    for (const game of sorted) {
      const picks = apex.predictGame(game.home_team, game.away_team);

      if (picks) {
        const actualTotal = game.home_score + game.away_score;

        for (const pick of picks) {
          // APEX is OVER-only at -110
          const hit = actualTotal > pick.predTotal;

          apexHistoryPicks.push({
            date: game.date,
            home: game.home_team,
            away: game.away_team,
            type: 'over',
            direction: 'OVER',
            predTotal: pick.predTotal,
            strength: pick.strength,
            tier: pick.tier,
            factors: pick.factors,
            actualTotal,
            hit,
            pnl: hit ? 91 : -100,
          });
        }
      }

      apex.updateTeam(game.home_team, game.home_score, game.away_score, game.date, true);
      apex.updateTeam(game.away_team, game.away_score, game.home_score, game.date, false);
    }

    console.log(`[APEX] Built history: ${apexHistoryPicks.length} total picks`);
  }

  // ── Rendering: Today's Picks ───────────────────────────────────────────────

  function renderPicks() {
    const loading = document.getElementById('picks-loading');
    const cdsContainer = document.getElementById('cds-container');
    const empty = document.getElementById('picks-empty');
    const allSection = document.getElementById('all-games-section');

    loading.style.display = 'none';

    const hasPicks = todayCdsPicks.length > 0 || todayPrismPicks.length > 0 || todayApexPicks.length > 0;

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

      // Play 8 — APEX Adaptive Pace Exploiter
      const apexContainer = document.getElementById('apex-container');
      if (apexContainer && todayApexPicks.length > 0) {
        apexContainer.style.display = '';
        apexContainer.innerHTML = '<h3 class="section-title">Play 8 — APEX OVER Totals <span class="apex-badge">-110</span></h3>' +
          '<div class="picks-grid">' + todayApexPicks.map(renderAPEXCard).join('') + '</div>';
      } else if (apexContainer) {
        apexContainer.style.display = 'none';
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

  function renderAPEXCard(pick) {
    const g = pick.game;
    const tierClass = pick.tier === 'ELITE' ? 'conf-apex-elite' : pick.tier === 'HIGH' ? 'conf-apex-high' : 'conf-apex';

    const FACTOR_LABELS = {
      elite_pace: 'Elite Pace', fast_pace: 'Fast Pace',
      both_surging: 'Both Surging', both_heating: 'Both Heating',
      extreme_recent: 'Extreme Recent', recent_high: 'Recent High', recent_elevated: 'Recent Elevated',
      all_windows: 'All Windows High', multi_window: 'Multi-Window',
      extreme_asym: 'Extreme Asymmetry', scoring_asym: 'Scoring Asymmetry',
      terrible_def: 'Terrible Defense', poor_def: 'Poor Defense',
      def_collapsing: 'Def Collapsing', def_softening: 'Def Softening',
      last_game_high: 'Last Game High', consistent_high: 'Consistent High',
      fortress_home: 'Fortress Home', away_roadkill: 'Away Roadkill',
      net_dominance: 'Net Dominance', net_edge: 'Net Edge',
      home_surging: 'Home Surging', home_improving: 'Home Improving',
      away_cratering: 'Away Cratering', away_fading: 'Away Fading',
      off_crush: 'Offense Crush', off_edge: 'Offense Edge',
      huge_wpct_gap: 'Huge Win% Gap', wpct_gap: 'Win% Gap',
      home_hot: 'Home Hot', away_cold: 'Away Cold',
    };
    const factorList = pick.factors.map(f => FACTOR_LABELS[f] || f).join(', ');

    const betLine = `<div class="pick-bet-line apex-over">OVER ${pick.predTotal} at -110</div>`;
    const matchupHtml = `${teamName(g.away_team)} @ ${teamName(g.home_team)}`;

    let liveHtml = '';
    if (g.status === 'STATUS_FINAL') {
      const actualTotal = g.home_score + g.away_score;
      const won = actualTotal > pick.predTotal;
      liveHtml = `<div class="pick-live ${won ? 'live-win' : 'live-loss'}">
        <span class="live-label">FINAL</span>
        <span class="live-score">Total: ${actualTotal} (pred: ${pick.predTotal})</span>
        <span class="live-result">${won ? 'W' : 'L'}</span></div>`;
    } else if (g.status !== 'STATUS_SCHEDULED') {
      liveHtml = `<div class="pick-live live-active"><span class="live-label">LIVE Q${g.period} ${g.clock}</span></div>`;
    } else {
      liveHtml = `<div class="pick-live live-scheduled"><span class="live-label">${g.time}</span>
        <span class="live-status">Pre-Game</span></div>`;
    }

    return `
      <div class="pick-card ${tierClass}">
        <div class="pick-header">
          <span class="pick-verdict">PLAY 8</span>
          <span class="pick-conf">${pick.tier} — OVER TOTAL</span>
        </div>
        ${betLine}
        <div class="pick-matchup">${matchupHtml}</div>
        <div class="pick-details">
          <div class="detail">
            <span class="detail-label">Pred Total</span>
            <span class="detail-value">${pick.predTotal}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Strength</span>
            <span class="detail-value">${pick.strength}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Signals</span>
            <span class="detail-value">${pick.factors.length}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Odds</span>
            <span class="detail-value">-110</span>
          </div>
        </div>
        <div class="apex-factors">${factorList}</div>
        ${liveHtml}
      </div>`;
  }

  // ── Rendering: All Games Grid ──────────────────────────────────────────────

  function renderAllGames() {
    const grid = document.getElementById('all-games-grid');
    if (!grid || todayGames.length === 0) return;

    grid.innerHTML = todayGames.map(g => {
      const isP4 = todayCdsPicks.some(p => p.game.id === g.id);
      const isP7 = todayPrismPicks.some(p => p.game.id === g.id);
      const isP8 = todayApexPicks.some(p => p.game.id === g.id);
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
    renderAPEXHistory();
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

  function renderAPEXHistory() {
    const tbody = document.getElementById('apex-history-body');
    if (!tbody) return;

    let filtered = apexHistoryPicks;
    if (currentApexHistoryPeriod !== 'all') {
      const cutoff = getCutoffDate(parseInt(currentApexHistoryPeriod));
      filtered = apexHistoryPicks.filter(p => p.date >= cutoff);
    }

    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="muted">No APEX picks in this period</td></tr>';
    } else {
      const FACTOR_SHORT = {
        elite_pace: 'Pace+', fast_pace: 'Pace',
        both_surging: 'Surge', both_heating: 'Heat',
        extreme_recent: 'Rcnt+', recent_high: 'Rcnt', recent_elevated: 'Rcnt-',
        all_windows: 'AllW', multi_window: 'MultiW',
        extreme_asym: 'Asym+', scoring_asym: 'Asym',
        terrible_def: 'Def--', poor_def: 'Def-',
        def_collapsing: 'DefCol', def_softening: 'DefSoft',
        last_game_high: 'LastHi', consistent_high: 'ConstHi',
        fortress_home: 'Fort', away_roadkill: 'Road-',
        net_dominance: 'NetDom', net_edge: 'NetEdge',
        home_surging: 'HmSurge', home_improving: 'HmUp',
        away_cratering: 'AwCrater', away_fading: 'AwFade',
        off_crush: 'OffCrush', off_edge: 'OffEdge',
        huge_wpct_gap: 'WPct+', wpct_gap: 'WPct',
        home_hot: 'HmHot', away_cold: 'AwCold',
      };

      tbody.innerHTML = filtered.map(p => {
        const resClass = p.hit ? 'result-win' : 'result-loss';
        const resText = p.hit ? 'W' : 'L';
        const pnlVal = p.pnl;
        const pnlText = pnlVal >= 0 ? `+$${pnlVal}` : `-$${Math.abs(pnlVal)}`;
        const tierClass = p.tier === 'ELITE' ? 'conf-apex-elite' : p.tier === 'HIGH' ? 'conf-apex-high' : 'conf-apex';
        const dateFormatted = formatDate(p.date);
        const factorStr = p.factors.map(f => FACTOR_SHORT[f] || f).join(', ');

        const betCol = `<span class="apex-over">OVER</span> ${p.predTotal}`;
        const actualCol = `${p.actualTotal}`;

        return `
          <tr>
            <td>${dateFormatted}</td>
            <td>${p.type === 'over' ? 'OVER' : 'HOME ML'}</td>
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

    renderAPEXSummary(filtered);
  }

  function renderAPEXSummary(filtered) {
    const summary = document.getElementById('apex-history-summary');
    if (!summary) return;

    const hits = filtered.filter(p => p.hit).length;
    const total = filtered.length;
    const pnl = filtered.reduce((s, p) => s + p.pnl, 0);
    const hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) : '0';
    const totalWagered = total * 100;
    const roi = totalWagered > 0 ? ((pnl / totalWagered) * 100).toFixed(0) : '0';

    // Tier breakdown
    const elites = filtered.filter(p => p.tier === 'ELITE');
    const eliteHits = elites.filter(p => p.hit).length;
    const elitePnl = elites.reduce((s, p) => s + p.pnl, 0);

    summary.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card summary-apex">
          <div class="summary-title">Play 8 — APEX OVER at -110</div>
          <div class="summary-stat">
            <span class="summary-record">${hits}-${total - hits}</span>
            <span class="summary-pct">${hitRate}%</span>
          </div>
          <div class="summary-pnl ${pnl >= 0 ? 'result-win' : 'result-loss'}">
            ${pnl >= 0 ? '+' : ''}$${pnl} (ROI: ${roi >= 0 ? '+' : ''}${roi}%)
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">ELITE Tier Only</div>
          <div class="summary-stat">
            <span class="summary-record">${eliteHits}-${elites.length - eliteHits}</span>
            <span class="summary-pct">${elites.length > 0 ? (eliteHits / elites.length * 100).toFixed(1) + '%' : '—'}</span>
          </div>
          <div class="summary-pnl ${elitePnl >= 0 ? 'result-win' : 'result-loss'}">
            ${elitePnl >= 0 ? '+' : ''}$${elitePnl}
          </div>
        </div>
        <div class="summary-card">
          <div class="summary-title">All OVER Bets at -110</div>
          <div class="summary-pnl result-win">
            Win: +$91
          </div>
          <div class="summary-pnl result-loss">
            Loss: -$100
          </div>
          <div class="summary-pnl" style="color: var(--text-muted); font-size: 12px;">
            Break-even: 52.4%
          </div>
        </div>
      </div>`;
  }

  // ── Metrics ────────────────────────────────────────────────────────────────

  function updateMetrics() {
    const el = (id) => document.getElementById(id);

    el('metric-picks').textContent = todayCdsPicks.length + todayPrismPicks.length + todayApexPicks.length;
    el('metric-games').textContent = todayGames.length;

    // Play 4 CDS accuracy
    if (cdsHistoryPicks.length > 0) {
      const cdsWins = cdsHistoryPicks.filter(p => p.favWon).length;
      const cdsTotal = cdsHistoryPicks.length;
      const cdsAcc = ((cdsWins / cdsTotal) * 100).toFixed(1);
      const cdsPnl = cdsHistoryPicks.reduce((s, p) => s + p.pnl, 0);
      el('metric-accuracy').textContent = cdsAcc + '%';

      // Play 7 PRISM accuracy
      if (prismHistoryPicks.length > 0) {
        const prismHits = prismHistoryPicks.filter(p => p.hit).length;
        const prismTotal = prismHistoryPicks.length;
        const prismAcc = ((prismHits / prismTotal) * 100).toFixed(1);
        const prismPnl = prismHistoryPicks.reduce((s, p) => s + p.pnl, 0);
        el('metric-cds-accuracy').textContent = prismAcc + '%';

        // APEX stats
        const apexHits = apexHistoryPicks.filter(p => p.hit).length;
        const apexTotal = apexHistoryPicks.length;
        const apexPnl = apexHistoryPicks.reduce((s, p) => s + p.pnl, 0);

        // Combined ROI (all plays)
        const combinedBets = cdsTotal + prismTotal + apexTotal;
        const combinedPnl = cdsPnl + prismPnl + apexPnl;
        const roi = combinedBets > 0 ? ((combinedPnl / (combinedBets * 100)) * 100).toFixed(0) : '0';
        el('metric-roi').textContent = (roi >= 0 ? '+' : '') + roi + '%';

        // Combined record
        const combinedWins = cdsWins + prismHits + apexHits;
        const combinedTotal = cdsTotal + prismTotal + apexTotal;
        el('metric-record').textContent = `${combinedWins}-${combinedTotal - combinedWins}`;
      } else {
        el('metric-cds-accuracy').textContent = '—';
        const roi = cdsTotal > 0 ? ((cdsPnl / (cdsTotal * 100)) * 100).toFixed(0) : '0';
        el('metric-roi').textContent = (roi >= 0 ? '+' : '') + roi + '%';
        el('metric-record').textContent = `${cdsWins}-${cdsTotal - cdsWins}`;
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

    document.querySelectorAll('.apex-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.apex-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentApexHistoryPeriod = btn.dataset.period;
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
