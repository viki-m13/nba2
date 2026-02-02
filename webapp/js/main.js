// =============================================================================
// MAIN APP CONTROLLER - Q3 Sniper Prediction System
// =============================================================================

(function () {
  'use strict';

  // ---- State ----
  let currentView = 'dashboard';
  let games = [];
  let liveSignals = [];
  let recentSignals = [];
  let historicalSignals = [];
  let sniperModelLoaded = false;
  let legacyModelLoaded = false;
  let pollInterval = null;
  let recentGamesLoaded = false;
  let currentTimePeriod = 'recent';
  const POLL_MS = 10000;
  const processedGames = new Set();
  const recentProcessedGames = new Set();

  // =========================================================================
  // INITIALIZATION
  // =========================================================================

  async function init() {
    console.log('[Sniper] Initializing...');

    setupNavigation();
    setupTimePeriodFilters();

    await loadModels();
    await loadHistoricalSignals();

    await NbaApi.detectVercelProxy();

    await refreshGames();
    pollInterval = setInterval(refreshGames, POLL_MS);

    setStatus(true);
    console.log('[Sniper] Ready');

    loadRecentGames();
  }

  // =========================================================================
  // MODEL LOADING
  // =========================================================================

  async function loadModels() {
    // Load sniper model (primary)
    const sniperPaths = ['data/sniper_model.json', '../output/sniper_model.json'];
    for (const path of sniperPaths) {
      try {
        const resp = await fetch(path);
        if (resp.ok) {
          const params = await resp.json();
          SniperEngine.loadModel(params);
          sniperModelLoaded = true;
          console.log(`[Sniper] Sniper model loaded from ${path}`);
          break;
        }
      } catch (e) { /* try next */ }
    }

    // Also load legacy model as fallback
    const legacyPaths = ['data/model.json', '../output/q3_terminal_v2_js_model.json'];
    for (const path of legacyPaths) {
      try {
        const resp = await fetch(path);
        if (resp.ok) {
          const params = await resp.json();
          Q3Engine.loadModel(params);
          legacyModelLoaded = true;
          console.log(`[Sniper] Legacy model loaded from ${path}`);
          break;
        }
      } catch (e) { /* try next */ }
    }

    if (!sniperModelLoaded && !legacyModelLoaded) {
      console.warn('[Sniper] No models loaded');
    }
  }

  // =========================================================================
  // HISTORICAL SIGNALS
  // =========================================================================

  async function loadHistoricalSignals() {
    // Try sniper signals first
    const paths = [
      'data/sniper_signals.json', '../output/sniper_signals.json',
      'data/signals.json', '../output/q3_terminal_v2_signals.json'
    ];
    for (const path of paths) {
      try {
        const resp = await fetch(path);
        if (resp.ok) {
          const data = await resp.json();
          historicalSignals = data.signals || [];
          console.log(`[Sniper] Loaded ${historicalSignals.length} historical signals from ${path}`);
          renderSignalsTab();
          renderRegimeTable();
          renderConfidenceTable();
          updatePerformanceStats();
          return;
        }
      } catch (e) { /* try next */ }
    }
    console.warn('[Sniper] No historical signals found');
    historicalSignals = [];
  }

  // =========================================================================
  // NAVIGATION
  // =========================================================================

  function setupNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.addEventListener('click', () => switchView(btn.dataset.view));
    });
  }

  function switchView(view) {
    currentView = view;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    const viewEl = document.getElementById(`view-${view}`);
    if (viewEl) viewEl.classList.add('active');
    const btn = document.querySelector(`.nav-btn[data-view="${view}"]`);
    if (btn) btn.classList.add('active');
  }

  function setupTimePeriodFilters() {
    document.querySelectorAll('.time-period-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.time-period-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentTimePeriod = btn.dataset.period;
        renderSignalsTab();
      });
    });
  }

  // =========================================================================
  // STATUS
  // =========================================================================

  function setStatus(online) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    if (online) {
      dot.classList.remove('offline');
      text.textContent = sniperModelLoaded ? 'Sniper Active' : (legacyModelLoaded ? 'Legacy Model' : 'Connected');
    } else {
      dot.classList.add('offline');
      text.textContent = 'Offline';
    }
  }

  // =========================================================================
  // GAME POLLING
  // =========================================================================

  async function refreshGames() {
    try {
      const result = await NbaApi.fetchAllGames();
      if (result.error) return;

      games = result.games || [];

      const liveGames = games.filter(g => g.status === 'live' || g.status === 'halftime');
      document.getElementById('metric-live').textContent = liveGames.length;
      document.getElementById('metric-live-sub').textContent =
        games.length > 0 ? `${games.length} total today` : 'No games today';
      document.getElementById('games-count').textContent = `${games.length} games`;
      document.getElementById('live-count').textContent = `${liveGames.length} live`;

      renderGamesGrid(games, 'games-grid');
      renderGamesGrid(liveGames.length > 0 ? liveGames : games, 'live-games-grid');

      for (const game of liveGames) {
        checkForSignal(game);
      }

      await processFinishedGames(games);
      renderActiveSignals();

    } catch (e) {
      console.error('[Sniper] Poll error:', e);
    }
  }

  // =========================================================================
  // FINISHED GAME PROCESSING
  // =========================================================================

  async function processFinishedGames(allGames) {
    if (!sniperModelLoaded && !legacyModelLoaded) return;

    const finishedGames = allGames.filter(g => g.status === 'final' && !processedGames.has(g.id));
    if (finishedGames.length === 0) return;

    const batch = finishedGames.slice(0, 5);
    batch.forEach(g => processedGames.add(g.id));

    await Promise.all(batch.map(async (game) => {
      try {
        const possessions = await NbaApi.fetchPlayByPlay(game.id);
        if (possessions.length < 10) return;

        const opts = { openingSpread: 0, openingOU: game.ouLine || 0 };

        // Use sniper engine
        let signals = [];
        if (sniperModelLoaded) {
          signals = SniperEngine.generateSignals(game, possessions, opts);
        } else if (legacyModelLoaded) {
          // Fallback to legacy - only ML_LEADER signals
          const legacy = Q3Engine.generateSignals(game, possessions, opts);
          signals = legacy.filter(s => s.signalType === 'ML_LEADER');
        }

        if (signals.length === 0) return;

        const actualMargin = game.homeScore - game.awayScore;

        for (const sig of signals) {
          sig.gameId = game.id;
          sig.homeTeam = game.homeTeam;
          sig.awayTeam = game.awayTeam;
          sig.homeScore = game.homeScore;
          sig.awayScore = game.awayScore;
          sig.timestamp = new Date().toISOString();
          sig.isFinished = true;
          sig.actualMargin = actualMargin;
          sig.quarter = 'FINAL';

          // Determine result: did the leader win?
          const q3Lead = sig.q3Lead || 0;
          const pickedHome = sig.direction === 'HOME';
          sig.correct = pickedHome ? actualMargin > 0 : actualMargin < 0;

          // Spread cover
          const liveSpread = sig.liveSpread || q3Lead * 0.78;
          sig.leader_covered_spread = pickedHome
            ? actualMargin > liveSpread
            : actualMargin < liveSpread;

          liveSignals.push(sig);
        }

        console.log(`[Sniper] ${signals.length} signals for ${game.awayTeam}@${game.homeTeam}`);
      } catch (e) {
        // skip
      }
    }));
  }

  // =========================================================================
  // RECENT GAMES LOADING
  // =========================================================================

  async function loadRecentGames() {
    if (!sniperModelLoaded && !legacyModelLoaded) {
      const loadingEl = document.getElementById('recent-loading');
      if (loadingEl) loadingEl.style.display = 'none';
      recentGamesLoaded = true;
      renderSignalsTab();
      return;
    }

    console.log('[Sniper] Loading recent games...');
    const loadingEl = document.getElementById('recent-loading');
    if (loadingEl) loadingEl.style.display = '';

    const dateStrings = NbaApi.getRecentDateStrings(21);

    for (let i = 0; i < dateStrings.length; i += 3) {
      const batch = dateStrings.slice(i, i + 3);
      await Promise.all(batch.map(dateStr => processDateForSignals(dateStr)));

      const countEl = document.getElementById('recent-signals-count');
      if (countEl) {
        countEl.textContent = `${recentSignals.length} signals from ${i + batch.length} days`;
      }
    }

    recentGamesLoaded = true;
    console.log(`[Sniper] Loaded ${recentSignals.length} recent signals`);

    if (loadingEl) loadingEl.style.display = 'none';
    renderSignalsTab();
  }

  async function processDateForSignals(dateStr) {
    try {
      const { games: dateGames } = await NbaApi.fetchESPNScoreboardForDate(dateStr);
      const finishedGames = dateGames.filter(g => g.status === 'final');
      if (finishedGames.length === 0) return;

      const batch = finishedGames.slice(0, 8);

      for (let j = 0; j < batch.length; j += 2) {
        const subBatch = batch.slice(j, j + 2);
        await Promise.all(subBatch.map(async (game) => {
          const gameKey = `${game.date}-${game.homeTeam}-${game.awayTeam}`;
          if (recentProcessedGames.has(gameKey)) return;
          recentProcessedGames.add(gameKey);

          try {
            const possessions = await NbaApi.fetchESPNPlayByPlay(game.espnId);
            if (possessions.length < 10) return;

            const opts = { openingSpread: 0, openingOU: game.ouLine || 0 };
            let signals = [];

            if (sniperModelLoaded) {
              signals = SniperEngine.generateSignals(game, possessions, opts);
            } else if (legacyModelLoaded) {
              const legacy = Q3Engine.generateSignals(game, possessions, opts);
              signals = legacy.filter(s => s.signalType === 'ML_LEADER');
            }

            if (signals.length === 0) return;

            const actualMargin = game.homeScore - game.awayScore;

            for (const sig of signals) {
              sig.gameId = game.espnId;
              sig.homeTeam = game.homeTeam;
              sig.awayTeam = game.awayTeam;
              sig.homeScore = game.homeScore;
              sig.awayScore = game.awayScore;
              sig.date = game.date;
              sig.isFinished = true;
              sig.isRecent = true;
              sig.actualMargin = actualMargin;
              sig.quarter = 'FINAL';

              const q3Lead = sig.q3Lead || 0;
              const pickedHome = sig.direction === 'HOME';
              sig.correct = pickedHome ? actualMargin > 0 : actualMargin < 0;

              const liveSpread = sig.liveSpread || q3Lead * 0.78;
              sig.leader_covered_spread = pickedHome
                ? actualMargin > liveSpread
                : actualMargin < liveSpread;

              recentSignals.push(sig);
            }
          } catch (e) { /* skip */ }
        }));
      }
    } catch (e) {
      console.warn(`[Sniper] Error on ${dateStr}:`, e.message);
    }
  }

  // =========================================================================
  // SIGNAL DETECTION (Live)
  // =========================================================================

  function checkForSignal(game) {
    if (!sniperModelLoaded && !legacyModelLoaded) return;
    if (processedGames.has(game.id)) return;

    const q = game.quarter || 0;
    if (q < 4) return;

    const clock = game.gameClock || '12:00';
    const mins = parseInt(clock.split(':')[0]) || 0;
    if (q === 4 && mins < 8) return;

    const possessions = game.possessions || [];
    if (possessions.length < 10) return;

    const opts = { openingSpread: 0, openingOU: game.ouLine || 0 };
    let signals = [];

    if (sniperModelLoaded) {
      signals = SniperEngine.generateSignals(game, possessions, opts);
    } else if (legacyModelLoaded) {
      const legacy = Q3Engine.generateSignals(game, possessions, opts);
      signals = legacy.filter(s => s.signalType === 'ML_LEADER');
    }

    if (signals.length > 0) {
      processedGames.add(game.id);
      for (const sig of signals) {
        sig.gameId = game.id;
        sig.homeTeam = game.homeTeam;
        sig.awayTeam = game.awayTeam;
        sig.homeScore = game.homeScore;
        sig.awayScore = game.awayScore;
        sig.timestamp = new Date().toISOString();
        liveSignals.push(sig);
      }

      if (Notification.permission === 'granted') {
        const s = signals[0];
        new Notification('SNIPER SIGNAL', {
          body: `${s.team || s.betTeam} to WIN | ${(s.confidence * 100).toFixed(1)}% confidence`,
        });
      }

      const card = document.getElementById('live-signals-card');
      if (card) card.style.display = '';
    }
  }

  // =========================================================================
  // RENDERING - GAMES GRID
  // =========================================================================

  function renderGamesGrid(gamesList, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!gamesList || gamesList.length === 0) {
      container.innerHTML = `<div class="empty-state"><h3>No games</h3><p>Check back when NBA games are scheduled.</p></div>`;
      return;
    }

    container.innerHTML = gamesList.map(g => {
      const hasSignal = liveSignals.some(s => s.gameId === g.id);
      const statusClass = g.status === 'live' ? 'live' : g.status === 'final' ? 'final' : 'scheduled';
      let statusText = g.statusText || g.status.toUpperCase();
      if (g.status === 'live') statusText = `Q${g.quarter || '?'} ${g.gameClock || ''}`;
      const atQ3End = g.quarter === 3 && g.gameClock === '0:00';
      if (atQ3End) statusText = 'Q3 END - ANALYZING';

      const diff = (g.homeScore || 0) - (g.awayScore || 0);
      const diffStr = diff > 0 ? `${g.homeTeam} +${diff}` : diff < 0 ? `${g.awayTeam} +${Math.abs(diff)}` : 'Tied';

      return `
        <div class="game-card ${hasSignal ? 'has-signal' : ''}">
          <div class="game-status-bar">
            <span class="game-status ${atQ3End ? 'q3-end' : statusClass}">${statusText}</span>
            <div class="game-status-right">
              ${g.status === 'live' || g.status === 'final' ? `<span class="game-diff">${diffStr}</span>` : ''}
              ${hasSignal ? '<span class="tier-badge gold">SNIPER</span>' : ''}
            </div>
          </div>
          <div class="game-scoreboard">
            <div class="game-team">
              <div class="game-team-name">${g.awayTeam}</div>
              <div class="game-team-sub">Away</div>
            </div>
            <div class="game-score">
              ${g.awayScore || 0}<span class="sep">-</span>${g.homeScore || 0}
            </div>
            <div class="game-team">
              <div class="game-team-name">${g.homeTeam}</div>
              <div class="game-team-sub">Home</div>
            </div>
          </div>
          ${hasSignal ? renderGameSignalSummary(g.id) : ''}
        </div>`;
    }).join('');
  }

  function renderGameSignalSummary(gameId) {
    const sigs = liveSignals.filter(s => s.gameId === gameId);
    if (!sigs.length) return '';
    return sigs.map(s => {
      const team = s.team || s.betTeam || (s.direction === 'HOME' ? s.homeTeam : s.awayTeam);
      const conf = ((s.confidence || 0) * 100).toFixed(0);
      const resultTag = s.correct !== undefined
        ? `<span class="result-tag ${s.correct ? 'win' : 'loss'}">${s.correct ? 'W' : 'L'}</span>`
        : '';
      return `
        <div class="signal-pick game-card-bet">
          <span class="signal-type-badge sniper" style="font-size:8px;padding:2px 5px">SNIPER</span>
          <span class="signal-pick-value">${team} to WIN</span>
          <span class="signal-pick-odds">${conf}% conf</span>
          ${resultTag}
        </div>
      `;
    }).join('');
  }

  // =========================================================================
  // RENDERING - ACTIVE SIGNALS
  // =========================================================================

  function renderActiveSignals() {
    const container = document.getElementById('active-signals-container');
    const countEl = document.getElementById('active-count');

    if (!liveSignals.length) {
      countEl.textContent = '0 signals';
      container.innerHTML = `
        <div class="empty-state">
          <h3>No active signals</h3>
          <p>Sniper signals fire at end of Q3 when the AI identifies a 95%+ confidence winner.</p>
        </div>`;
      return;
    }

    const liveCount = liveSignals.filter(s => !s.isFinished).length;
    const finishedCount = liveSignals.filter(s => s.isFinished).length;

    if (liveCount > 0 && finishedCount > 0) {
      countEl.textContent = `${liveCount} live, ${finishedCount} completed`;
    } else if (liveCount > 0) {
      countEl.textContent = `${liveCount} active`;
    } else {
      countEl.textContent = `${finishedCount} from today`;
    }

    container.innerHTML = liveSignals.map(s => renderSniperCard(s, !s.isFinished)).join('');

    const liveContainer = document.getElementById('live-signals-container');
    if (liveContainer) {
      liveContainer.innerHTML = liveSignals.map(s => renderSniperCard(s, !s.isFinished)).join('');
    }

    const card = document.getElementById('live-signals-card');
    if (card && liveSignals.length > 0) card.style.display = '';
  }

  function renderSniperCard(s, isLive = false) {
    const homeTeam = s.homeTeam || s.home_team || '';
    const awayTeam = s.awayTeam || s.away_team || '';
    const matchup = `${awayTeam} @ ${homeTeam}`;
    const confidence = s.confidence || 0;
    const q3Lead = s.q3Lead || s.q3_lead || 0;
    const regime = s.regime || '';
    const correct = s.correct;
    const team = s.team || s.leader || (s.direction === 'HOME' ? homeTeam : awayTeam);
    const trailer = s.trailer || (s.direction === 'HOME' ? awayTeam : homeTeam);
    const q3LeadAbs = Math.abs(q3Lead);
    const isFinished = s.isFinished || s.final_home;
    const quarter = s.quarter || (isFinished ? 'FINAL' : 'Q4');
    const gameDate = s.date ? formatGameDate(s.date) : '';

    const homeScore = s.homeScore || s.final_home || 0;
    const awayScore = s.awayScore || s.final_away || 0;
    const actualMargin = s.actualMargin || s.actual_margin || (homeScore - awayScore);
    const liveSpread = s.liveSpread || s.live_spread || (q3Lead * 0.78);
    const spreadLine = s.altSpreadLine || (s.direction === 'HOME'
      ? -(Math.round(Math.abs(liveSpread) * 2) / 2)
      : (Math.round(Math.abs(liveSpread) * 2) / 2));

    const tier = s.tier || getSniperTier(confidence);
    const tierClass = tier.toLowerCase();

    const leadQuality = s.leadQuality || s.lead_quality || 0;
    const qualityLabel = leadQuality >= 0.7 ? 'HIGH' : (leadQuality >= 0.5 ? 'GOOD' : 'OK');
    const qualityClass = leadQuality >= 0.7 ? 'positive' : '';

    let resultDetail = '';
    if (!isLive && correct !== undefined) {
      const winner = actualMargin > 0 ? homeTeam : awayTeam;
      const winMargin = Math.abs(actualMargin);
      const spreadCovered = s.leader_covered_spread;

      resultDetail = `
        <div class="signal-result-detail ${correct ? 'win' : 'loss'}">
          <div class="result-badge">${correct ? 'WIN' : 'LOSS'}</div>
          <div class="result-stats">
            <span>Final: <strong>${awayTeam} ${awayScore} - ${homeTeam} ${homeScore}</strong></span>
            <span>${winner} by ${winMargin}</span>
            <span>Spread: ${spreadCovered ? 'Covered' : 'Did not cover'}</span>
          </div>
        </div>`;
    }

    return `
      <div class="signal-card ${tierClass}">
        <div class="signal-header">
          <div class="signal-header-left">
            ${gameDate ? `<span class="signal-date">${gameDate}</span>` : ''}
            <span class="signal-matchup">${matchup}</span>
          </div>
          <div class="signal-header-right">
            <span class="signal-quarter-badge ${isFinished ? 'final' : ''}">${quarter}</span>
            <span class="tier-badge ${tierClass}">${tier}</span>
            <span class="signal-type-badge sniper">SNIPER</span>
          </div>
        </div>

        <div class="signal-scoreboard">
          <div class="signal-team-col">
            <span class="signal-team-abbr ${q3Lead < 0 ? 'leading' : ''}">${awayTeam}</span>
            <span class="signal-team-role">Away</span>
          </div>
          <div class="signal-score-col">
            <div class="signal-score-main">${isFinished ? `${awayScore} - ${homeScore}` : `Q3: ${q3Lead < 0 ? q3LeadAbs : 0} - ${q3Lead > 0 ? q3LeadAbs : 0}`}</div>
            <div class="signal-score-sub">${team} leads by ${q3LeadAbs}</div>
          </div>
          <div class="signal-team-col">
            <span class="signal-team-abbr ${q3Lead > 0 ? 'leading' : ''}">${homeTeam}</span>
            <span class="signal-team-role">Home</span>
          </div>
        </div>

        <div class="bet-slip">
          <div class="bet-slip-header">
            <span class="bet-slip-label">SNIPER BET</span>
            <span class="bet-slip-market">Live Moneyline</span>
          </div>
          <div class="bet-slip-main">
            <div class="bet-slip-pick">${team} to WIN</div>
            <div class="bet-slip-odds">${(confidence * 100).toFixed(1)}%</div>
          </div>
          <div class="bet-slip-detail">Alt: ${team} ${spreadLine > 0 ? '+' : ''}${spreadLine.toFixed(1)} spread @ -110</div>
        </div>

        <div class="signal-body">
          <div class="signal-field">
            <span class="signal-field-label">Confidence</span>
            <span class="signal-field-value positive">${(confidence * 100).toFixed(1)}%</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Lead Quality</span>
            <span class="signal-field-value ${qualityClass}">${qualityLabel} (${(leadQuality * 100).toFixed(0)})</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Q3 Lead</span>
            <span class="signal-field-value">${team} +${q3LeadAbs}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Pred Margin</span>
            <span class="signal-field-value">${formatMargin(s.predictedMargin || s.predicted_margin)}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Regime</span>
            <span class="signal-field-value regime-${regime.toLowerCase()}">${regime}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Tier</span>
            <span class="signal-field-value">${tier}</span>
          </div>
        </div>
        ${resultDetail}
      </div>`;
  }

  function getSniperTier(confidence) {
    if (confidence >= 0.99) return 'DIAMOND';
    if (confidence >= 0.97) return 'PLATINUM';
    if (confidence >= 0.95) return 'GOLD';
    if (confidence >= 0.93) return 'SILVER';
    return 'BRONZE';
  }

  // =========================================================================
  // RENDERING - SIGNALS TAB
  // =========================================================================

  function renderSignalsTab() {
    const tbody = document.getElementById('signals-tbody');
    const summary = document.getElementById('filter-summary');
    const tableTitle = document.getElementById('signals-table-title');
    const tableBadge = document.getElementById('signals-table-badge');
    if (!tbody) return;

    let signals = getSignalsForTimePeriod(currentTimePeriod);

    signals = signals.slice().sort((a, b) => {
      const da = a.date || '0';
      const db = b.date || '0';
      return db.localeCompare(da);
    });

    if (summary) {
      const correct = signals.filter(s => s.correct !== undefined && s.correct).length;
      const total = signals.filter(s => s.correct !== undefined).length;
      const acc = total > 0 ? (correct / total * 100).toFixed(1) : '0.0';
      let pnl = 0;
      signals.forEach(s => {
        if (s.correct === undefined) return;
        const o = s.estimated_odds || s.estimatedOdds || -110;
        if (s.correct) {
          pnl += o < 0 ? 100 / Math.abs(o) : o / 100;
        } else {
          pnl -= 1;
        }
      });
      summary.textContent = `${correct}/${total} (${acc}%) | P&L: ${pnl >= 0 ? '+' : ''}${pnl.toFixed(1)}u`;
    }

    if (tableTitle) {
      const titles = {
        'recent': 'Sniper Signals — Last 3 Weeks',
        'today': 'Today\'s Sniper Signals',
        '3days': 'Sniper Signals — Last 3 Days',
        '7days': 'Sniper Signals — Last 7 Days',
        '14days': 'Sniper Signals — Last 2 Weeks',
        'historical': 'Historical OOS Sniper Signals (2022-23)',
      };
      tableTitle.textContent = titles[currentTimePeriod] || 'Sniper Signals';
    }

    if (tableBadge) {
      tableBadge.textContent = `${signals.length} signals`;
    }

    const display = signals.slice(0, 500);

    if (display.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:40px;color:var(--text-muted)">
        ${currentTimePeriod !== 'historical' && !recentGamesLoaded
          ? '<span class="loading-spinner"></span> Loading recent games...'
          : 'No sniper signals for this period'}
      </td></tr>`;
      return;
    }

    tbody.innerHTML = display.map(s => renderSignalRow(s)).join('');
  }

  function getSignalsForTimePeriod(period) {
    const now = new Date();
    const todayStr = formatDateStr(now);

    if (period === 'historical') {
      return historicalSignals.map(normalizeSignal);
    }

    const allRecent = [
      ...liveSignals.map(s => ({ ...s, isRecent: true, date: s.date || todayStr })),
      ...recentSignals,
    ];

    const seen = new Set();
    const deduped = [];
    for (const s of allRecent) {
      const key = `${s.date || ''}-${s.homeTeam || s.home_team || ''}-${s.awayTeam || s.away_team || ''}`;
      if (!seen.has(key)) {
        seen.add(key);
        deduped.push(s);
      }
    }

    let signals;
    if (period === 'today') {
      signals = deduped.filter(s => (s.date || '') === todayStr);
    } else if (period === '3days') {
      signals = deduped.filter(s => (s.date || '') >= daysAgoStr(3));
    } else if (period === '7days') {
      signals = deduped.filter(s => (s.date || '') >= daysAgoStr(7));
    } else if (period === '14days') {
      signals = deduped.filter(s => (s.date || '') >= daysAgoStr(14));
    } else {
      signals = deduped;
    }

    return signals.map(normalizeSignal);
  }

  function normalizeSignal(s) {
    return {
      date: s.date || '',
      signal_type: s.signal_type || s.signalType || 'SNIPER',
      direction: s.direction || '',
      confidence: s.confidence || 0,
      edge: s.edge || 0,
      q3_lead: s.q3_lead ?? s.q3Lead ?? 0,
      estimated_odds: s.estimated_odds || s.estimatedOdds || -110,
      estimated_ml_odds: s.estimated_ml_odds || s.estimatedOdds || -300,
      home_team: s.home_team || s.homeTeam || '',
      away_team: s.away_team || s.awayTeam || '',
      leader: s.leader || s.team || '',
      final_home: s.final_home || s.homeScore || 0,
      final_away: s.final_away || s.awayScore || 0,
      final_margin: s.final_margin || s.actualMargin || s.actual_margin || 0,
      correct: s.correct,
      leader_won: s.leader_won,
      leader_covered_spread: s.leader_covered_spread,
      live_spread: s.live_spread || s.liveSpread || 0,
      regime: s.regime || '',
      predicted_margin: s.predicted_margin || s.predictedMargin || 0,
      lead_quality: s.lead_quality || s.leadQuality || 0,
      lead_sustainability: s.lead_sustainability || s.leadSustainability || 0,
      model_agree: s.model_agree || s.modelAgree || 0,
      isRecent: s.isRecent || false,
    };
  }

  function renderSignalRow(s) {
    const conf = (s.confidence * 100).toFixed(0);
    const q3Lead = s.q3_lead || 0;
    const homeTeam = s.home_team || '';
    const awayTeam = s.away_team || '';
    const dateStr = s.date ? formatGameDate(s.date) : '';
    const finalHome = s.final_home || 0;
    const finalAway = s.final_away || 0;

    const q3Leader = s.leader || (q3Lead > 0 ? homeTeam : (q3Lead < 0 ? awayTeam : 'TIE'));
    const q3Diff = Math.abs(q3Lead);

    const leadQuality = s.lead_quality || 0;
    const qualityLabel = leadQuality >= 0.7 ? 'HIGH' : (leadQuality >= 0.5 ? 'GOOD' : 'OK');
    const qualityClass = leadQuality >= 0.7 ? 'positive' : '';

    const finalMargin = s.final_margin || (finalHome - finalAway);
    const winner = finalMargin > 0 ? homeTeam : awayTeam;
    const winMargin = Math.abs(finalMargin);
    const resultNote = (finalHome || finalAway) ? `${winner} by ${winMargin}` : '';

    let unitPnl;
    if (s.correct !== undefined) {
      const o = s.estimated_ml_odds || s.estimated_odds || -300;
      if (s.correct) {
        unitPnl = o < 0 ? 100 / Math.abs(o) : o / 100;
      } else {
        unitPnl = -1;
      }
    }

    const resultCell = s.correct !== undefined
      ? `<td class="${s.correct ? 'positive' : 'negative'}">
          <div class="result-cell">
            <span class="result-tag ${s.correct ? 'win' : 'loss'}">${s.correct ? 'W' : 'L'}</span>
            <span class="result-note">${resultNote}</span>
          </div>
        </td>`
      : `<td class="date-cell">-</td>`;

    const pnlCell = unitPnl !== undefined
      ? `<td class="${unitPnl >= 0 ? 'positive' : 'negative'}">${unitPnl >= 0 ? '+' : ''}${unitPnl.toFixed(2)}u</td>`
      : `<td class="date-cell">-</td>`;

    const recentBadge = s.isRecent ? ' <span class="recent-badge">NEW</span>' : '';
    const tier = getSniperTier(s.confidence);

    return `<tr class="${s.correct === false ? 'loss-row' : ''}">
      <td class="date-cell">${dateStr}${recentBadge}</td>
      <td class="matchup-cell">
        <div>${awayTeam} @ ${homeTeam}</div>
        <div class="score-sub">${finalAway}-${finalHome}</div>
      </td>
      <td class="bet-desc-cell"><strong>${q3Leader} ML</strong> <span class="bet-desc-sub">${tier}</span></td>
      <td class="q3-cell">
        <div>${q3Leader} +${q3Diff}</div>
        <div class="regime-sub">${s.regime || ''}</div>
      </td>
      <td>${conf}%</td>
      <td class="${qualityClass}">${qualityLabel}</td>
      ${resultCell}
      ${pnlCell}
    </tr>`;
  }

  // =========================================================================
  // PERFORMANCE TABLES
  // =========================================================================

  function updatePerformanceStats() {
    if (!historicalSignals.length) return;

    const sigs = historicalSignals;
    const mlCorrect = sigs.filter(s => s.correct || s.leader_won).length;
    const total = sigs.length;
    const mlAcc = total > 0 ? (mlCorrect / total * 100).toFixed(1) : '0';

    const spreadCorrect = sigs.filter(s => s.leader_covered_spread).length;
    const spreadAcc = total > 0 ? (spreadCorrect / total * 100).toFixed(1) : '0';

    let mlPnl = 0;
    sigs.forEach(s => {
      const won = s.correct || s.leader_won;
      const odds = s.estimated_ml_odds || s.estimatedOdds || -300;
      if (won) {
        mlPnl += odds < 0 ? 100 / Math.abs(odds) : odds / 100;
      } else {
        mlPnl -= 1;
      }
    });

    const spreadPnl = spreadCorrect * (100 / 110) - (total - spreadCorrect) * 1.0;
    const spreadRoi = total > 0 ? (spreadPnl / total * 100).toFixed(1) : '0';

    setText('perf-ml-record', `${mlCorrect}/${total}`);
    setText('perf-ml-acc', `${mlAcc}%`);
    setText('perf-ml-pnl', `+${mlPnl.toFixed(1)}u`);
    setText('perf-spread-record', `${spreadCorrect}/${total}`);
    setText('perf-spread-acc', `${spreadAcc}%`);
    setText('perf-spread-pnl', `+${spreadPnl.toFixed(1)}u`);
    setText('perf-spread-roi', `+${spreadRoi}%`);

    const mlBar = document.getElementById('perf-ml-bar');
    if (mlBar) mlBar.style.width = `${mlAcc}%`;
    const spreadBar = document.getElementById('perf-spread-bar');
    if (spreadBar) spreadBar.style.width = `${spreadAcc}%`;
  }

  function renderRegimeTable() {
    const tbody = document.getElementById('regime-tbody');
    if (!tbody) return;

    const regimes = [
      { name: 'BLOWOUT', range: '20+' },
      { name: 'COMFORTABLE', range: '12-19' },
      { name: 'COMPETITIVE', range: '6-11' },
      { name: 'TIGHT', range: '0-5' },
    ];

    tbody.innerHTML = regimes.map(r => {
      const sigs = historicalSignals.filter(s => s.regime === r.name);
      const total = sigs.length;
      if (total === 0) {
        return `<tr><td>${r.name}</td><td>${r.range} pts</td><td>0</td><td>-</td><td>-</td><td>-</td></tr>`;
      }

      const mlWins = sigs.filter(s => s.correct || s.leader_won).length;
      const mlAcc = (mlWins / total * 100).toFixed(1);

      const spreadWins = sigs.filter(s => s.leader_covered_spread).length;
      const spreadAcc = (spreadWins / total * 100).toFixed(1);

      const avgConf = (sigs.reduce((a, s) => a + (s.confidence || 0), 0) / total * 100).toFixed(1);

      return `<tr>
        <td>${r.name}</td>
        <td>${r.range} pts</td>
        <td>${total}</td>
        <td class="${mlWins / total >= 0.95 ? 'positive' : ''}">${mlWins}/${total} (${mlAcc}%)</td>
        <td>${spreadWins}/${total} (${spreadAcc}%)</td>
        <td>${avgConf}%</td>
      </tr>`;
    }).join('');
  }

  function renderConfidenceTable() {
    const tbody = document.getElementById('confidence-tbody');
    if (!tbody) return;

    const tiers = [
      { name: 'DIAMOND', lo: 0.99, hi: 1.0 },
      { name: 'PLATINUM', lo: 0.97, hi: 0.99 },
      { name: 'GOLD', lo: 0.95, hi: 0.97 },
      { name: 'SILVER', lo: 0.93, hi: 0.95 },
    ];

    tbody.innerHTML = tiers.map(t => {
      const sigs = historicalSignals.filter(s => s.confidence >= t.lo && s.confidence < t.hi);
      const total = sigs.length;
      if (total === 0) {
        return `<tr><td>${t.name}</td><td>${(t.lo * 100).toFixed(0)}-${(t.hi * 100).toFixed(0)}%</td><td>0</td><td>-</td><td>-</td><td>-</td></tr>`;
      }

      const mlWins = sigs.filter(s => s.correct || s.leader_won).length;
      const mlAcc = (mlWins / total * 100).toFixed(1);

      const spreadWins = sigs.filter(s => s.leader_covered_spread).length;
      const spreadPnl = spreadWins * (100 / 110) - (total - spreadWins) * 1.0;
      const spreadRoi = (spreadPnl / total * 100).toFixed(1);

      return `<tr>
        <td><span class="tier-badge ${t.name.toLowerCase()}">${t.name}</span></td>
        <td>${(t.lo * 100).toFixed(0)}-${(t.hi * 100).toFixed(0)}%</td>
        <td>${total}</td>
        <td class="${mlWins / total >= 0.95 ? 'positive' : ''}">${mlWins}/${total} (${mlAcc}%)</td>
        <td class="${spreadPnl > 0 ? 'positive' : 'negative'}">${spreadPnl >= 0 ? '+' : ''}${spreadPnl.toFixed(1)}u</td>
        <td class="${parseFloat(spreadRoi) > 0 ? 'positive' : 'negative'}">${spreadRoi}%</td>
      </tr>`;
    }).join('');
  }

  // =========================================================================
  // HELPERS
  // =========================================================================

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function formatOdds(odds) {
    if (!odds || odds === 0) return '-';
    const n = Math.round(odds);
    return n > 0 ? `+${n}` : `${n}`;
  }

  function formatMargin(margin) {
    if (margin === undefined || margin === null) return '-';
    const m = parseFloat(margin);
    return m > 0 ? `+${m.toFixed(1)}` : m.toFixed(1);
  }

  function formatDateStr(date) {
    const yyyy = date.getFullYear();
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const dd = String(date.getDate()).padStart(2, '0');
    return `${yyyy}${mm}${dd}`;
  }

  function daysAgoStr(n) {
    const d = new Date();
    d.setDate(d.getDate() - n);
    return formatDateStr(d);
  }

  function formatGameDate(dateStr) {
    if (!dateStr || dateStr.length < 8) return dateStr || '';
    const m = dateStr.substring(4, 6);
    const d = dateStr.substring(6, 8);
    const y = dateStr.substring(0, 4);
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const monthIdx = parseInt(m, 10) - 1;
    const formatted = `${months[monthIdx]} ${parseInt(d, 10)}`;
    const now = new Date();
    const todayStr = formatDateStr(now);
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = formatDateStr(yesterday);
    if (dateStr === todayStr) return `${formatted} (Today)`;
    if (dateStr === yesterdayStr) return `${formatted} (Yest.)`;
    return `${formatted}, ${y}`;
  }

  // =========================================================================
  // START
  // =========================================================================

  function requestNotifications() {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { init(); requestNotifications(); });
  } else {
    init();
    requestNotifications();
  }

})();
