// =============================================================================
// MAIN APP CONTROLLER — NBA Dominance ADI Pre-Game System
// =============================================================================

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────────────
  let currentView = 'picks';
  let todayGames = [];          // All games today
  let todayPicks = [];          // Games flagged by ADI model
  let historyPicks = [];        // Historical picks with results
  let seasonData = [];          // Full season game data for model training
  let modelReady = false;
  let currentHistoryPeriod = 7;
  let useProxy = false;        // True when running on Vercel (CORS proxy available)

  const Model = window.ParlayEngine.PreGameModel;

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

  // Normalize ESPN abbreviations to our format
  function norm(abbr) {
    const map = { GSW: 'GS', NYK: 'NY', NOP: 'NO', SAS: 'SA', UTA: 'UTAH', WAS: 'WSH' };
    return map[abbr] || abbr;
  }

  // ── Initialization ─────────────────────────────────────────────────────────

  async function init() {
    console.log('[ADI] Initializing...');
    setupNavigation();
    setupHistoryFilters();
    setStatus('loading', 'Loading model data...');

    // Detect Vercel proxy for CORS
    try {
      const probe = await fetch('/api/nba?endpoint=espn_scoreboard', { method: 'HEAD' });
      useProxy = probe.ok || probe.status === 405 || probe.status === 200;
    } catch (e) { useProxy = false; }
    console.log('[ADI] Proxy available:', useProxy);

    try {
      // Step 1: Load season data to train the ADI model
      await loadSeasonData();

      // Step 2: Fetch today's games
      await fetchTodayGames();

      // Step 3: Run ADI predictions on today's games
      runPredictions();

      // Step 4: Build history from season data
      buildHistory();

      // Step 5: Render everything
      renderPicks();
      renderAllGames();
      renderHistory();
      updateMetrics();

      setStatus('online', 'Model Active');
      modelReady = true;

      // Poll for live score updates every 30s
      setInterval(refreshScores, 30000);

    } catch (err) {
      console.error('[ADI] Init error:', err);
      setStatus('error', 'Failed to load');
    }
  }

  // ── Data Loading ───────────────────────────────────────────────────────────

  async function loadSeasonData() {
    console.log('[ADI] Loading season data...');

    // Try to load the pre-built season file
    try {
      const resp = await fetch('data/espn_full_season_2025.json');
      if (resp.ok) {
        seasonData = await resp.json();
        console.log(`[ADI] Loaded ${seasonData.length} games from season file`);
      }
    } catch (e) {
      console.warn('[ADI] Could not load season file, will fetch from ESPN');
    }

    // Also fetch recent games from ESPN to fill gaps
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
        if (fetched % 5 === 0) {
          console.log(`[ADI] Fetched ${fetched} recent dates...`);
        }
        await sleep(300);
      } catch (e) { /* skip failed dates */ }
    }

    // Feed all data into the PreGameModel
    seasonData.sort((a, b) => (a.date || '').localeCompare(b.date || ''));
    Model.reset();
    for (const g of seasonData) {
      Model.updateTeam(g.home_team, g.home_score, g.away_score, g.date);
      Model.updateTeam(g.away_team, g.away_score, g.home_score, g.date);
    }

    console.log(`[ADI] Model trained on ${seasonData.length} games, ${Object.keys(Model.teamHistory).length} teams`);
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
    console.log('[ADI] Fetching today\'s games...');
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

      console.log(`[ADI] Found ${todayGames.length} games today`);
    } catch (e) {
      console.error('[ADI] Error fetching today:', e);
    }
  }

  // ── ADI Predictions ────────────────────────────────────────────────────────

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
          netGap: pred.netGap,
          favOffRating: pred.favOffRating,
          dogDefRating: pred.dogDefRating,
          bps: pred.bps,
          kellyFraction: pred.betRecommendation.kellyFraction,
          teamTotalBet: pred.teamTotalBet || null,
        });
      }
    }

    // Sort: HIGH first, then STRONG, then by predicted margin
    const confOrder = { HIGH: 0, STRONG: 1, MODERATE: 2 };
    todayPicks.sort((a, b) => {
      const co = (confOrder[a.confidence] || 9) - (confOrder[b.confidence] || 9);
      if (co !== 0) return co;
      return b.predictedMargin - a.predictedMargin;
    });

    console.log(`[ADI] ${todayPicks.length} picks from ${todayGames.length} games`);
  }

  // ── History Builder ────────────────────────────────────────────────────────

  function buildHistory() {
    // Walk through season data chronologically and simulate the model's picks
    historyPicks = [];

    // Create an independent model instance for walk-forward history
    const tempModel = {
      teamHistory: {},
      lookbackWindow: 15,
      updateTeam: ParlayEngine.PreGameModel.updateTeam,
      getMetrics: ParlayEngine.PreGameModel.getMetrics,
      predictGame: ParlayEngine.PreGameModel.predictGame,
    };

    const sorted = [...seasonData].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

    for (const game of sorted) {
      // Predict BEFORE updating
      const pred = tempModel.predictGame(game.home_team, game.away_team);

      if (pred && pred.signals && pred.signals.length > 0) {
        const actualMargin = game.home_score - game.away_score;
        const favWon = (pred.favIsHome && actualMargin > 0) || (!pred.favIsHome && actualMargin < 0);
        const actualMarginAbs = Math.abs(actualMargin);

        const favScore = pred.favIsHome ? game.home_score : game.away_score;
        const teamTotalSignal = pred.teamTotalBet;
        const teamTotalHit = teamTotalSignal ? favScore > teamTotalSignal.line : null;

        historyPicks.push({
          date: game.date,
          favorite: pred.favorite,
          underdog: pred.underdog,
          favIsHome: pred.favIsHome,
          confidence: pred.signals[0].confidence,
          predictedMargin: pred.predictedMargin,
          actualMargin: actualMarginAbs,
          favWon,
          homeScore: game.home_score,
          awayScore: game.away_score,
          favScore,
          teamTotalLine: teamTotalSignal ? teamTotalSignal.line : null,
          teamTotalHit,
          pnl: favWon ? 91 : -100, // at -110 odds: win $91 or lose $100
          teamTotalPnl: teamTotalHit === true ? 91 : teamTotalHit === false ? -100 : 0,
        });
      }

      // Update AFTER prediction
      tempModel.updateTeam(game.home_team, game.home_score, game.away_score, game.date);
      tempModel.updateTeam(game.away_team, game.away_score, game.home_score, game.date);
    }

    console.log(`[ADI] Built history: ${historyPicks.length} picks`);
  }

  // ── Rendering: Today's Picks ───────────────────────────────────────────────

  function renderPicks() {
    const loading = document.getElementById('picks-loading');
    const container = document.getElementById('picks-container');
    const empty = document.getElementById('picks-empty');
    const allSection = document.getElementById('all-games-section');

    loading.style.display = 'none';

    if (todayPicks.length === 0) {
      container.style.display = 'none';
      empty.style.display = 'block';
    } else {
      container.style.display = '';
      empty.style.display = 'none';
      container.innerHTML = todayPicks.map(renderPickCard).join('');
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

    // Live score display
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
      const ahead = favScore > oppScore;
      liveHtml = `
        <div class="pick-live live-active">
          <span class="live-label">LIVE Q${g.period} ${g.clock}</span>
          <span class="live-score">${pick.betTeam} ${favScore} — ${oppTeam} ${oppScore}</span>
          <span class="live-track">${ahead ? 'On Track' : 'Behind'}</span>
        </div>`;
    } else {
      liveHtml = `
        <div class="pick-live live-scheduled">
          <span class="live-label">${g.time}</span>
          <span class="live-status">Pre-Game — Bet Before Tip-Off</span>
        </div>`;
    }

    // Team Total Over bet section
    let teamTotalHtml = '';
    if (pick.teamTotalBet) {
      teamTotalHtml = `
        <div class="pick-team-total">
          <div class="team-total-header">
            <span class="team-total-badge">BET 2</span>
            <span class="team-total-label">TEAM TOTAL</span>
          </div>
          <div class="team-total-line">
            ${pick.betTeam} Team Total OVER ${pick.teamTotalBet.line}
          </div>
          <div class="team-total-stats">
            <span class="tt-stat"><strong>97.6%</strong> hit rate at 110+</span>
            <span class="tt-stat"><strong>88.1%</strong> hit rate at 115+</span>
            <span class="tt-stat">Avg score: <strong>125</strong></span>
          </div>
          <div class="team-total-note">
            Find "${pick.betTeam} Team Total" on your sportsbook &mdash; bet OVER at -110
          </div>
        </div>`;
    }

    return `
      <div class="pick-card ${confClass}">
        <div class="pick-header">
          <span class="pick-verdict">BET</span>
          <span class="pick-conf">${confLabel}</span>
        </div>
        <div class="pick-bet-line">
          ${pick.betTeam} spread at -110
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
            <span class="detail-label">Net Gap</span>
            <span class="detail-value">${pick.netGap.toFixed(1)}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Off Rating</span>
            <span class="detail-value">${pick.favOffRating.toFixed(1)}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Pred. Margin</span>
            <span class="detail-value">${pick.predictedMargin.toFixed(1)}</span>
          </div>
          <div class="detail">
            <span class="detail-label">Kelly</span>
            <span class="detail-value">${(pick.kellyFraction * 100).toFixed(0)}%</span>
          </div>
        </div>
        ${teamTotalHtml}
        ${liveHtml}
      </div>`;
  }

  // ── Rendering: All Games Grid ──────────────────────────────────────────────

  function renderAllGames() {
    const grid = document.getElementById('all-games-grid');
    if (!grid || todayGames.length === 0) return;

    grid.innerHTML = todayGames.map(g => {
      const isPick = todayPicks.some(p => p.game.id === g.id);
      const pickClass = isPick ? 'game-picked' : '';

      let statusHtml;
      if (g.status === 'STATUS_FINAL') {
        statusHtml = `<span class="game-status final">Final: ${g.home_score}-${g.away_score}</span>`;
      } else if (g.status === 'STATUS_IN_PROGRESS' || g.status === 'STATUS_HALFTIME') {
        statusHtml = `<span class="game-status live">Q${g.period} ${g.clock}: ${g.home_score}-${g.away_score}</span>`;
      } else {
        statusHtml = `<span class="game-status scheduled">${g.time}</span>`;
      }

      return `
        <div class="game-card ${pickClass}">
          <div class="game-teams">${g.away_team} @ ${g.home_team}</div>
          ${statusHtml}
          ${isPick ? '<span class="game-badge">ADI PICK</span>' : ''}
        </div>`;
    }).join('');
  }

  // ── Rendering: History ─────────────────────────────────────────────────────

  function renderHistory() {
    const tbody = document.getElementById('history-body');
    if (!tbody) return;

    // Filter by period
    let filtered = historyPicks;
    if (currentHistoryPeriod !== 'all') {
      const cutoff = getCutoffDate(parseInt(currentHistoryPeriod));
      filtered = historyPicks.filter(p => p.date >= cutoff);
    }

    // Reverse chronological
    filtered = [...filtered].reverse();

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="muted">No picks in this period</td></tr>';
      return;
    }

    tbody.innerHTML = filtered.map(p => {
      const resClass = p.favWon ? 'result-win' : 'result-loss';
      const resText = p.favWon ? 'W' : 'L';
      const pnlText = p.favWon ? '+$91' : '-$100';
      const confClass = p.confidence === 'HIGH' ? 'conf-high' : 'conf-strong';
      const dateFormatted = formatDate(p.date);

      // Team total column
      let ttHtml = '<span class="muted">—</span>';
      if (p.teamTotalLine !== null) {
        const ttClass = p.teamTotalHit ? 'result-win' : 'result-loss';
        const ttLabel = p.teamTotalHit ? 'OVER' : 'UNDER';
        ttHtml = `<span class="badge ${ttClass}">${ttLabel}</span> <small>${p.favScore} / ${p.teamTotalLine}</small>`;
      }

      return `
        <tr>
          <td>${dateFormatted}</td>
          <td><strong>${p.favorite}</strong> spread</td>
          <td>${p.favorite} ${p.favIsHome ? 'vs' : '@'} ${p.underdog}</td>
          <td><span class="badge ${confClass}">${p.confidence}</span></td>
          <td>${p.predictedMargin.toFixed(1)}</td>
          <td>${p.actualMargin}</td>
          <td><span class="badge ${resClass}">${resText}</span></td>
          <td class="${resClass}">${pnlText}</td>
          <td>${ttHtml}</td>
        </tr>`;
    }).join('');
  }

  // ── Metrics ────────────────────────────────────────────────────────────────

  function updateMetrics() {
    const el = (id) => document.getElementById(id);

    el('metric-picks').textContent = todayPicks.length;
    el('metric-games').textContent = todayGames.length;

    if (historyPicks.length > 0) {
      const wins = historyPicks.filter(p => p.favWon).length;
      const total = historyPicks.length;
      const accuracy = ((wins / total) * 100).toFixed(1);
      const totalPnl = historyPicks.reduce((s, p) => s + p.pnl, 0);
      const roi = ((totalPnl / (total * 100)) * 100).toFixed(0);

      el('metric-accuracy').textContent = accuracy + '%';
      el('metric-roi').textContent = (roi >= 0 ? '+' : '') + roi + '%';
      el('metric-record').textContent = `${wins}-${total - wins}`;
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
      console.warn('[ADI] Score refresh failed:', e);
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
