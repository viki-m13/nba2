// =============================================================================
// MAIN APP CONTROLLER - Q3 Terminal Prediction System
// =============================================================================

(function () {
  'use strict';

  // ---- State ----
  let currentView = 'dashboard';
  let games = [];
  let liveSignals = [];    // Signals generated this session
  let historicalSignals = []; // From backtest
  let modelLoaded = false;
  let pollInterval = null;
  const POLL_MS = 10000; // 10 seconds
  const processedGames = new Set(); // Track games we've already signaled on

  // ---- Model parameters (embedded from Python export) ----
  const MODEL_URL = null; // Will embed inline below

  // =========================================================================
  // INITIALIZATION
  // =========================================================================

  async function init() {
    console.log('[Q3Terminal] Initializing...');

    setupNavigation();
    setupFilters();

    // Load ML model parameters
    await loadModel();

    // Load historical signals
    await loadHistoricalSignals();

    // Detect API proxy
    await NbaApi.detectVercelProxy();

    // Start polling
    await refreshGames();
    pollInterval = setInterval(refreshGames, POLL_MS);

    setStatus(true);
    console.log('[Q3Terminal] Ready');
  }

  // =========================================================================
  // MODEL LOADING
  // =========================================================================

  async function loadModel() {
    try {
      // Try to load from file first (when served from same origin)
      const resp = await fetch('../output/q3_terminal_v2_js_model.json');
      if (resp.ok) {
        const params = await resp.json();
        Q3Engine.loadModel(params);
        modelLoaded = true;
        console.log('[Q3Terminal] Model loaded from file');
        return;
      }
    } catch (e) {
      console.warn('[Q3Terminal] Could not load model file, using fallback');
    }

    // Model will work without loading (won't generate live signals)
    modelLoaded = false;
  }

  // =========================================================================
  // HISTORICAL SIGNALS
  // =========================================================================

  async function loadHistoricalSignals() {
    try {
      const resp = await fetch('../output/q3_terminal_v2_signals.json');
      if (resp.ok) {
        const data = await resp.json();
        historicalSignals = data.signals || [];
        console.log(`[Q3Terminal] Loaded ${historicalSignals.length} historical signals`);
        renderHistoricalSignals();
        renderRegimeTable();
        return;
      }
    } catch (e) {
      console.warn('[Q3Terminal] Could not load historical signals');
    }
    historicalSignals = [];
  }

  // =========================================================================
  // NAVIGATION
  // =========================================================================

  function setupNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        switchView(view);
      });
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

  // =========================================================================
  // FILTERS
  // =========================================================================

  function setupFilters() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderHistoricalSignals(btn.dataset.filter);
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
      text.textContent = modelLoaded ? 'Model Active' : 'Connected (no model)';
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
      if (result.error) {
        console.warn('[Q3Terminal] Fetch error:', result.error);
        return;
      }

      games = result.games || [];

      // Update metrics
      const liveGames = games.filter(g => g.status === 'live' || g.status === 'halftime');
      document.getElementById('metric-live').textContent = liveGames.length;
      document.getElementById('metric-live-sub').textContent =
        games.length > 0 ? `${games.length} total today` : 'No games today';
      document.getElementById('games-count').textContent = `${games.length} games`;
      document.getElementById('live-count').textContent = `${liveGames.length} live`;

      // Render games
      renderGamesGrid(games, 'games-grid');
      renderGamesGrid(liveGames.length > 0 ? liveGames : games, 'live-games-grid');

      // Check for Q3-end signals
      for (const game of liveGames) {
        checkForSignal(game);
      }

      // Render active signals
      renderActiveSignals();

    } catch (e) {
      console.error('[Q3Terminal] Poll error:', e);
    }
  }

  // =========================================================================
  // SIGNAL DETECTION
  // =========================================================================

  function checkForSignal(game) {
    if (!modelLoaded) return;
    if (processedGames.has(game.id)) return;

    // Only signal at end of Q3 or beginning of Q4
    const q = game.quarter || 0;
    if (q < 4) return; // Wait until Q4 starts (means Q3 ended)

    // Don't signal late in Q4
    const clock = game.gameClock || '12:00';
    const parts = clock.split(':');
    const mins = parseInt(parts[0]) || 0;
    if (q === 4 && mins < 8) return; // Too late in Q4

    // Generate signals
    const possessions = game.possessions || [];
    if (possessions.length < 10) return;

    const signals = Q3Engine.generateSignals(game, possessions, {
      openingSpread: 0,
      openingOU: game.ouLine || 0,
    });

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

      // Show notification
      if (Notification.permission === 'granted') {
        new Notification('Q3 Terminal Signal', {
          body: `${signals[0].signalType}: ${signals[0].team} (${(signals[0].confidence * 100).toFixed(0)}%)`,
        });
      }

      // Show live signals card
      const card = document.getElementById('live-signals-card');
      if (card) card.style.display = '';

      console.log(`[Q3Terminal] Generated ${signals.length} signals for ${game.awayTeam}@${game.homeTeam}`);
    }
  }

  // =========================================================================
  // RENDERING - GAMES
  // =========================================================================

  function renderGamesGrid(gamesList, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!gamesList || gamesList.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <h3>No games</h3>
          <p>Check back when NBA games are scheduled.</p>
        </div>`;
      return;
    }

    container.innerHTML = gamesList.map(g => {
      const hasSignal = liveSignals.some(s => s.gameId === g.id);
      const statusClass = g.status === 'live' ? 'live' : g.status === 'final' ? 'final' : 'scheduled';

      let statusText = g.statusText || g.status.toUpperCase();
      if (g.status === 'live' && g.quarter >= 3) {
        statusText = `Q${g.quarter} ${g.gameClock}`;
      }

      // Check if at end of Q3
      const atQ3End = g.quarter === 3 && g.gameClock === '0:00';
      if (atQ3End) statusText = 'Q3 END - ANALYZING';

      return `
        <div class="game-card ${hasSignal ? 'has-signal' : ''}">
          <div class="game-status-bar">
            <span class="game-status ${atQ3End ? 'q3-end' : statusClass}">${statusText}</span>
            ${hasSignal ? '<span class="tier-badge gold">SIGNAL</span>' : ''}
          </div>
          <div class="game-scoreboard">
            <div class="game-team">
              <div class="game-team-name">${g.awayTeam}</div>
              <div class="game-team-sub">Away</div>
            </div>
            <div class="game-score">
              ${g.awayScore}<span class="sep">-</span>${g.homeScore}
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

    return sigs.map(s => `
      <div class="signal-pick">
        <span class="signal-pick-label">${s.signalType}</span>
        <span class="signal-pick-value">${s.team || s.direction} ${s.direction === 'OVER' || s.direction === 'UNDER' ? s.direction : ''}</span>
        <span class="signal-pick-odds">${formatOdds(s.estimatedOdds)} | ${(s.confidence * 100).toFixed(0)}%</span>
      </div>
    `).join('');
  }

  // =========================================================================
  // RENDERING - ACTIVE SIGNALS
  // =========================================================================

  function renderActiveSignals() {
    const container = document.getElementById('active-signals-container');
    const countEl = document.getElementById('active-count');

    if (!liveSignals.length) {
      countEl.textContent = '0 active';
      container.innerHTML = `
        <div class="empty-state">
          <h3>No active signals</h3>
          <p>Signals are generated at the end of Q3. Monitor live games below for upcoming opportunities.</p>
        </div>`;
      return;
    }

    countEl.textContent = `${liveSignals.length} active`;

    container.innerHTML = liveSignals.map(s => renderSignalCard(s, true)).join('');

    // Also render in live view
    const liveContainer = document.getElementById('live-signals-container');
    if (liveContainer) {
      liveContainer.innerHTML = liveSignals.map(s => renderSignalCard(s, true)).join('');
    }
  }

  function renderSignalCard(s, isLive = false) {
    const tierClass = (s.tier || 'watch').toLowerCase();
    const typeClass = s.signalType.toLowerCase().replace('_', '-');
    const matchup = `${s.awayTeam || s.away_team || ''} @ ${s.homeTeam || s.home_team || ''}`;

    const confidence = s.confidence || 0;
    const edge = s.edge || 0;
    const q3Lead = s.q3Lead || s.q3_lead || 0;
    const regime = s.regime || '';
    const odds = s.estimatedOdds || s.estimated_odds || -110;
    const correct = s.correct;

    // Determine pick text
    let pickText = '';
    const dir = s.direction;
    const type = s.signalType || s.signal_type || '';

    if (type === 'SPREAD') {
      pickText = `${dir} ${dir === 'HOME' ? (s.homeTeam || s.home_team) : (s.awayTeam || s.away_team)} SPREAD`;
    } else if (type === 'ML_LEADER') {
      pickText = `${s.team || dir} ML`;
    } else if (type === 'ML_TRAILER') {
      pickText = `${s.team || dir} ML (Value)`;
    } else if (type === 'Q4_TOTAL') {
      pickText = `Q4 ${dir}`;
    } else {
      pickText = `${dir}`;
    }

    return `
      <div class="signal-card ${tierClass}">
        <div class="signal-header">
          <span class="signal-matchup">${matchup}</span>
          <div style="display:flex;gap:6px;align-items:center">
            <span class="tier-badge ${tierClass}">${(s.tier || 'WATCH').toUpperCase()}</span>
            <span class="signal-type-badge ${typeClass}">${type.replace('_', ' ')}</span>
          </div>
        </div>
        <div class="signal-body">
          <div class="signal-field">
            <span class="signal-field-label">Confidence</span>
            <span class="signal-field-value">${(confidence * 100).toFixed(1)}%</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Edge</span>
            <span class="signal-field-value positive">${(edge * 100).toFixed(1)}%</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Q3 Lead</span>
            <span class="signal-field-value">${q3Lead > 0 ? '+' : ''}${q3Lead}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Regime</span>
            <span class="signal-field-value">${regime}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Est. Odds</span>
            <span class="signal-field-value">${formatOdds(odds)}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Pred Margin</span>
            <span class="signal-field-value">${formatMargin(s.predictedMargin || s.predicted_margin)}</span>
          </div>
        </div>
        <div class="signal-pick">
          <span class="signal-pick-label">Pick</span>
          <span class="signal-pick-value">${pickText}</span>
          <span class="signal-pick-odds">${formatOdds(odds)}</span>
        </div>
        ${!isLive && correct !== undefined ? `
          <div class="signal-result ${correct ? 'win' : 'loss'}">
            ${correct ? 'WIN' : 'LOSS'} | Actual margin: ${formatMargin(s.actual_margin || s.actualMargin)}
          </div>
        ` : ''}
      </div>`;
  }

  // =========================================================================
  // RENDERING - HISTORICAL SIGNALS TABLE
  // =========================================================================

  function renderHistoricalSignals(filter = 'all') {
    const tbody = document.getElementById('signals-tbody');
    const summary = document.getElementById('filter-summary');
    if (!tbody) return;

    let filtered = historicalSignals;
    if (filter && filter !== 'all') {
      filtered = historicalSignals.filter(s => s.signal_type === filter);
    }

    if (summary) {
      const correct = filtered.filter(s => s.correct).length;
      const total = filtered.length;
      const acc = total > 0 ? (correct / total * 100).toFixed(1) : '0.0';
      summary.textContent = `${correct}/${total} (${acc}%) | ${total} signals`;
    }

    // Show max 200 rows
    const display = filtered.slice(0, 200);

    tbody.innerHTML = display.map(s => {
      const type = s.signal_type || '';
      const typeClass = type.toLowerCase().replace('_', '-');
      const dir = s.direction || '';
      const conf = (s.confidence * 100).toFixed(1);
      const edge = (s.edge * 100).toFixed(1);
      const q3Lead = s.q3_lead || 0;
      const regime = s.regime || '';
      const odds = s.estimated_odds || s.estimatedOdds || -110;

      let pickTeam = dir;
      if (dir === 'HOME') pickTeam = s.home_team;
      else if (dir === 'AWAY') pickTeam = s.away_team;

      let pickText = type === 'Q4_TOTAL' ? dir : pickTeam;

      return `<tr>
        <td>${s.away_team} @ ${s.home_team}</td>
        <td><span class="signal-type-badge ${typeClass}">${type.replace('_', ' ')}</span></td>
        <td>${pickText}</td>
        <td>${conf}%</td>
        <td class="positive">${edge}%</td>
        <td>${q3Lead > 0 ? '+' : ''}${q3Lead}</td>
        <td>${regime}</td>
        <td>${formatOdds(odds)}</td>
        <td class="${s.correct ? 'positive' : 'negative'}">${s.correct ? 'WIN' : 'LOSS'}</td>
      </tr>`;
    }).join('');
  }

  // =========================================================================
  // RENDERING - REGIME TABLE
  // =========================================================================

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
      const spreadSigs = historicalSignals.filter(s => s.signal_type === 'SPREAD' && s.regime === r.name);
      const mlSigs = historicalSignals.filter(s => s.signal_type === 'ML_LEADER' && s.regime === r.name);

      const sW = spreadSigs.filter(s => s.correct).length;
      const sT = spreadSigs.length;
      const sAcc = sT > 0 ? (sW / sT * 100).toFixed(1) + '%' : '-';

      const mW = mlSigs.filter(s => s.correct).length;
      const mT = mlSigs.length;
      const mAcc = mT > 0 ? (mW / mT * 100).toFixed(1) + '%' : '-';

      return `<tr>
        <td>${r.name}</td>
        <td>${r.range} pts</td>
        <td>${sT > 0 ? `${sW}/${sT}` : '-'}</td>
        <td class="${sT > 0 && sW/sT >= 0.65 ? 'positive' : ''}">${sAcc}</td>
        <td>${mT > 0 ? `${mW}/${mT}` : '-'}</td>
        <td class="${mT > 0 && mW/mT >= 0.90 ? 'positive' : ''}">${mAcc}</td>
      </tr>`;
    }).join('');
  }

  // =========================================================================
  // HELPERS
  // =========================================================================

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

  // =========================================================================
  // REQUEST NOTIFICATION PERMISSION
  // =========================================================================

  function requestNotifications() {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }

  // =========================================================================
  // START
  // =========================================================================

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { init(); requestNotifications(); });
  } else {
    init();
    requestNotifications();
  }

})();
