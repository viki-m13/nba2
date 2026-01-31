// =============================================================================
// MAIN APP CONTROLLER - NBA Live Trading Signals
// =============================================================================

(function() {
  'use strict';

  // =========================================================================
  // STATE
  // =========================================================================
  const state = {
    currentView: 'dashboard',
    games: [],
    liveSignals: [],       // Active signals from current live games
    alertHistory: [],      // All alerts (persisted in localStorage)
    selectedGameId: null,
    refreshInterval: 5,    // seconds
    refreshTimer: null,
    countdown: 5,
    countdownTimer: null,
    soundEnabled: true,
    notificationsEnabled: false,
    minTier: 'fade_spread', // minimum tier to show
    seenSignals: new Set(), // track already-alerted signals
    equityData: null,
    signalLog: null,
    historyGames: null,
  };

  const TIER_PRIORITY = { composite: 0, blowout_compress: 1, quant: 2, burst_fade: 3, q3_fade: 4, fade_ml: 5, fade_spread: 6 };

  // =========================================================================
  // INITIALIZATION
  // =========================================================================
  async function init() {
    loadSettings();
    loadAlerts();
    setupNavigation();
    setupEventListeners();
    initHistoricalData();
    navigateTo(window.location.hash.slice(1) || 'dashboard');

    // Auto-detect Vercel proxy
    const hasVercel = await NbaApi.detectVercelProxy();
    if (hasVercel) {
      console.log('[App] Using Vercel proxy for NBA data');
    } else if (!NbaApi.getCorsProxy()) {
      console.log('[App] No CORS proxy configured. Live data may not load. Set one in Settings.');
    }

    startRefreshLoop();

    console.log('[App] NBA Live Trading Signals initialized');
  }

  // =========================================================================
  // SETTINGS
  // =========================================================================
  function loadSettings() {
    try {
      const saved = JSON.parse(localStorage.getItem('nba-signals-settings') || '{}');
      if (saved.corsProxy) NbaApi.setCorsProxy(saved.corsProxy);
      if (saved.refreshInterval) state.refreshInterval = saved.refreshInterval;
      if (saved.soundEnabled !== undefined) state.soundEnabled = saved.soundEnabled;
      if (saved.notificationsEnabled !== undefined) state.notificationsEnabled = saved.notificationsEnabled;
      if (saved.minTier) state.minTier = saved.minTier;
      // Migrate old tier names
      if (['elite', 'strong', 'standard', 'wide'].includes(state.minTier)) state.minTier = 'fade_spread';

      // Update UI
      const proxyInput = document.getElementById('cors-proxy-input');
      const intervalInput = document.getElementById('refresh-interval-input');
      const soundCheck = document.getElementById('sound-enabled');
      const notifCheck = document.getElementById('notification-enabled');
      const tierSelect = document.getElementById('min-tier-select');

      if (proxyInput) proxyInput.value = saved.corsProxy || '';
      if (intervalInput) intervalInput.value = state.refreshInterval;
      if (soundCheck) soundCheck.checked = state.soundEnabled;
      if (notifCheck) notifCheck.checked = state.notificationsEnabled;
      if (tierSelect) tierSelect.value = state.minTier;
    } catch (e) {
      console.warn('[Settings] Error loading:', e);
    }
  }

  function saveSettings() {
    const corsProxy = document.getElementById('cors-proxy-input').value.trim();
    const refreshInterval = parseInt(document.getElementById('refresh-interval-input').value) || 5;
    const soundEnabled = document.getElementById('sound-enabled').checked;
    const notificationsEnabled = document.getElementById('notification-enabled').checked;
    const minTier = document.getElementById('min-tier-select').value;

    NbaApi.setCorsProxy(corsProxy);
    state.refreshInterval = Math.max(3, Math.min(60, refreshInterval));
    state.soundEnabled = soundEnabled;
    state.notificationsEnabled = notificationsEnabled;
    state.minTier = minTier;

    localStorage.setItem('nba-signals-settings', JSON.stringify({
      corsProxy, refreshInterval: state.refreshInterval,
      soundEnabled, notificationsEnabled, minTier,
    }));

    // Request notification permission if enabled
    if (notificationsEnabled && 'Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    // Restart refresh loop
    startRefreshLoop();

    // Close modal
    document.getElementById('settings-modal').classList.add('hidden');
  }

  // =========================================================================
  // ALERTS PERSISTENCE
  // =========================================================================
  function loadAlerts() {
    try {
      const saved = JSON.parse(localStorage.getItem('nba-signals-alerts') || '[]');
      state.alertHistory = saved;
      saved.forEach(a => {
        if (a.signalKey) state.seenSignals.add(a.signalKey);
      });
    } catch (e) {
      state.alertHistory = [];
    }
  }

  function saveAlerts() {
    try {
      // Keep last 200 alerts
      const toSave = state.alertHistory.slice(0, 200);
      localStorage.setItem('nba-signals-alerts', JSON.stringify(toSave));
    } catch (e) {
      console.warn('[Alerts] Error saving:', e);
    }
  }

  function clearAlerts() {
    state.alertHistory = [];
    state.seenSignals.clear();
    saveAlerts();
    renderAlerts();
  }

  // =========================================================================
  // NAVIGATION
  // =========================================================================
  function setupNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const view = link.dataset.view;
        navigateTo(view);
      });
    });
  }

  function navigateTo(view) {
    if (!view || !document.getElementById(`view-${view}`)) view = 'dashboard';

    state.currentView = view;
    window.location.hash = view;

    // Update nav
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    const activeLink = document.querySelector(`.nav-link[data-view="${view}"]`);
    if (activeLink) activeLink.classList.add('active');

    // Show view
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${view}`).classList.add('active');

    // View-specific init
    if (view === 'backtest') renderBacktest();
    if (view === 'history') renderHistory();
    if (view === 'alerts') renderAlerts();
  }

  // =========================================================================
  // EVENT LISTENERS
  // =========================================================================
  function setupEventListeners() {
    // Settings
    document.getElementById('settings-btn').addEventListener('click', () => {
      document.getElementById('settings-modal').classList.remove('hidden');
    });
    document.getElementById('settings-close').addEventListener('click', () => {
      document.getElementById('settings-modal').classList.add('hidden');
    });
    document.querySelector('.modal-overlay')?.addEventListener('click', () => {
      document.getElementById('settings-modal').classList.add('hidden');
    });
    document.getElementById('save-settings-btn').addEventListener('click', saveSettings);

    // Alert banner close
    document.getElementById('alert-banner-close')?.addEventListener('click', () => {
      document.getElementById('signal-alert-banner').classList.add('hidden');
    });

    // Clear alerts
    document.getElementById('clear-alerts-btn')?.addEventListener('click', clearAlerts);

    // Refresh now
    document.getElementById('refresh-now-btn')?.addEventListener('click', fetchAndUpdate);

    // Chart game select
    document.getElementById('chart-game-select')?.addEventListener('change', (e) => {
      const gameId = e.target.value;
      if (gameId) renderGameChart(gameId, 'scoreflow-chart');
    });

    // History game select
    document.getElementById('history-game-select')?.addEventListener('change', (e) => {
      renderHistoryChart(parseInt(e.target.value));
    });

    // Equity mode select
    document.getElementById('equity-mode-select')?.addEventListener('change', (e) => {
      if (state.equityData) {
        Charts.renderEquityCurve('equity-chart', state.equityData, e.target.value);
      }
    });

    // History filter
    document.getElementById('history-filter')?.addEventListener('change', () => {
      renderHistoryTable();
    });

    // Close game detail
    document.getElementById('close-detail-btn')?.addEventListener('click', () => {
      document.getElementById('game-detail-panel').classList.add('hidden');
    });
  }

  // =========================================================================
  // REFRESH LOOP
  // =========================================================================
  function startRefreshLoop() {
    if (state.refreshTimer) clearInterval(state.refreshTimer);
    if (state.countdownTimer) clearInterval(state.countdownTimer);

    state.countdown = state.refreshInterval;

    // Initial fetch
    fetchAndUpdate();

    // Set up recurring fetch
    state.refreshTimer = setInterval(() => {
      fetchAndUpdate();
      state.countdown = state.refreshInterval;
    }, state.refreshInterval * 1000);

    // Countdown display
    state.countdownTimer = setInterval(() => {
      state.countdown = Math.max(0, state.countdown - 1);
      const el = document.getElementById('stat-next-refresh');
      if (el) el.textContent = `${state.countdown}s`;
    }, 1000);
  }

  // =========================================================================
  // FETCH AND UPDATE
  // =========================================================================
  async function fetchAndUpdate() {
    const indicator = document.getElementById('refresh-indicator');
    indicator?.classList.remove('paused');

    try {
      const { games, error } = await NbaApi.fetchAllGames();

      if (error) {
        console.warn('[Fetch] Error:', error);
        indicator?.classList.add('paused');
        document.querySelector('.refresh-text').textContent = 'Error';
        return;
      }

      state.games = games;

      // Process signals
      processSignals(games);

      // Update all UI
      renderDashboard();
      if (state.currentView === 'live') renderLiveGames();
      updateStats();

    } catch (e) {
      console.warn('[Fetch] Exception:', e);
      indicator?.classList.add('paused');
    }
  }

  // =========================================================================
  // PROCESS SIGNALS
  // =========================================================================
  function processSignals(games) {
    state.liveSignals = [];

    for (const game of games) {
      if (game.signal && meetsMinTier(game.signal.tier)) {
        const signalKey = `${game.id}-${game.signal.tier}-Q${game.signal.quarter}`;

        state.liveSignals.push({
          ...game.signal,
          gameId: game.id,
          gameName: `${game.homeTeam} vs ${game.awayTeam}`,
          signalKey,
        });

        // Check if this is a new signal
        if (!state.seenSignals.has(signalKey)) {
          state.seenSignals.add(signalKey);

          const instruction = SignalEngine.getTradeInstruction(game.signal);

          // Add to alert history
          const alert = {
            ...game.signal,
            gameId: game.id,
            gameName: `${game.homeTeam} vs ${game.awayTeam}`,
            signalKey,
            instruction,
            timestamp: Date.now(),
            outcome: 'pending',
          };

          state.alertHistory.unshift(alert);
          saveAlerts();

          // Fire notifications
          fireAlert(alert);
        }
      }
    }
  }

  function meetsMinTier(tier) {
    return TIER_PRIORITY[tier] <= TIER_PRIORITY[state.minTier];
  }

  // =========================================================================
  // FIRE ALERT (sound + notification + banner)
  // =========================================================================
  function fireAlert(alert) {
    const instruction = alert.instruction;

    // Sound alert
    if (state.soundEnabled) {
      playAlertSound(alert.tier);
    }

    // Browser notification
    if (state.notificationsEnabled && 'Notification' in window && Notification.permission === 'granted') {
      new Notification(`${instruction.headline}`, {
        body: `${instruction.spread}\n${instruction.moneyline}\n${instruction.context}`,
        icon: '&#9651;',
        tag: alert.signalKey,
      });
    }

    // Banner
    const banner = document.getElementById('signal-alert-banner');
    const bannerMsg = document.getElementById('alert-banner-msg');
    if (banner && bannerMsg) {
      bannerMsg.textContent = `${instruction.headline} | ${instruction.context}`;
      banner.classList.remove('hidden');

      // Auto-hide after 15 seconds
      setTimeout(() => banner.classList.add('hidden'), 15000);
    }

    // Update alert badge
    updateAlertBadge();
  }

  function playAlertSound(tier) {
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      // Different tones per strategy
      const freqs = { composite: [880, 1100, 880], blowout_compress: [770, 990, 770], quant: [660, 880], burst_fade: [600, 770], q3_fade: [550, 720], fade_ml: [550, 660], fade_spread: [440] };
      const tones = freqs[tier] || [440];

      gain.gain.setValueAtTime(0.3, audioCtx.currentTime);

      let t = audioCtx.currentTime;
      for (const freq of tones) {
        osc.frequency.setValueAtTime(freq, t);
        t += 0.15;
      }

      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
      osc.start(audioCtx.currentTime);
      osc.stop(t + 0.3);
    } catch (e) {
      // Audio not available
    }
  }

  function updateAlertBadge() {
    const badge = document.getElementById('alert-count-badge');
    const pendingCount = state.alertHistory.filter(a => a.outcome === 'pending').length;
    if (badge) {
      badge.textContent = pendingCount;
      badge.classList.toggle('hidden', pendingCount === 0);
    }
  }

  // =========================================================================
  // UPDATE STATS BAR
  // =========================================================================
  function updateStats() {
    const liveGames = state.games.filter(g => g.status === 'live' || g.status === 'halftime');

    document.getElementById('stat-live-games').textContent = liveGames.length;
    document.getElementById('stat-active-signals').textContent = state.liveSignals.length;

    // Session WR
    const decided = state.alertHistory.filter(a => a.outcome === 'win' || a.outcome === 'loss');
    const wins = decided.filter(a => a.outcome === 'win').length;
    if (decided.length > 0) {
      document.getElementById('stat-session-wr').textContent = `${Math.round(wins / decided.length * 100)}%`;
    }

    updateAlertBadge();
  }

  // =========================================================================
  // RENDER DASHBOARD
  // =========================================================================
  function renderDashboard() {
    renderDashboardGames();
    renderDashboardSignals();
    renderUpcomingGames();
    updateChartGameSelect();
  }

  function renderDashboardGames() {
    const container = document.getElementById('dashboard-live-games');
    const countBadge = document.getElementById('live-games-count');
    const liveGames = state.games.filter(g => g.status === 'live' || g.status === 'halftime');

    countBadge.textContent = liveGames.length;

    if (liveGames.length === 0) {
      // Show scheduled games instead
      const scheduled = state.games.filter(g => g.status === 'scheduled');
      if (scheduled.length > 0) {
        container.innerHTML = scheduled.map(g => renderGameCard(g)).join('');
      } else {
        container.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">&#127936;</div>
            <p>No live games right now</p>
            <p class="empty-sub">Games will appear here when they start. Checking every ${state.refreshInterval} seconds.</p>
          </div>`;
      }
      return;
    }

    container.innerHTML = liveGames.map(g => renderGameCard(g)).join('');

    // Add click handlers
    container.querySelectorAll('.game-card').forEach(card => {
      card.addEventListener('click', () => {
        const gameId = card.dataset.gameId;
        renderGameChart(gameId, 'scoreflow-chart');

        const select = document.getElementById('chart-game-select');
        if (select) select.value = gameId;
      });
    });
  }

  function renderGameCard(game) {
    const hasSignal = game.signal != null;
    const diff = game.homeScore - game.awayScore;

    let statusClass = game.status;
    let statusText = game.statusText || game.status.toUpperCase();
    if (game.status === 'live') statusText = `Q${game.quarter} ${game.gameClock}`;
    if (game.status === 'halftime') statusText = 'HALFTIME';
    if (game.status === 'scheduled') {
      // Parse game time
      statusText = game.gameTime ? new Date(game.gameTime).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : 'SCHEDULED';
    }

    const signalBadge = hasSignal ?
      `<span class="signal-tier ${game.signal.tier}">${game.signal.tier.toUpperCase()}</span>` : '';

    return `
      <div class="game-card ${hasSignal ? 'has-signal' : ''}" data-game-id="${game.id}">
        <div class="game-card-header">
          <span class="game-status ${statusClass}">${statusText}</span>
          ${signalBadge}
        </div>
        <div class="game-teams">
          <div class="team-info">
            <span class="team-abbr" style="color: ${game.homeColor || '#e5e7eb'}">${game.homeTeam}</span>
            <span class="team-name">${game.homeName || ''}</span>
          </div>
          <div style="text-align: center;">
            <div class="team-score">${game.homeScore}</div>
          </div>
          <div class="game-vs">VS</div>
          <div style="text-align: center;">
            <div class="team-score">${game.awayScore}</div>
          </div>
          <div class="team-info">
            <span class="team-abbr" style="color: ${game.awayColor || '#e5e7eb'}">${game.awayTeam}</span>
            <span class="team-name">${game.awayName || ''}</span>
          </div>
        </div>
        <div class="game-meta">
          <span class="game-quarter">Diff: ${diff > 0 ? '+' : ''}${diff}</span>
          <span>${game.possessions?.length || 0} plays</span>
        </div>
      </div>`;
  }

  function renderDashboardSignals() {
    const container = document.getElementById('dashboard-signals');
    const countBadge = document.getElementById('signals-count');
    countBadge.textContent = state.liveSignals.length;

    if (state.liveSignals.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">&#128226;</div>
          <p>No active signals</p>
          <p class="empty-sub">Signals fire when quant layers detect mean reversion opportunities (fade the leader).</p>
        </div>`;
      return;
    }

    container.innerHTML = state.liveSignals.map(sig => renderSignalCard(sig)).join('');
  }

  function renderSignalCard(signal) {
    const instruction = SignalEngine.getTradeInstruction(signal);
    if (!instruction) return '';

    const stratName = signal.strategyName || signal.tier.toUpperCase();

    return `
      <div class="signal-card ${signal.tier} new-signal">
        <div class="signal-card-header">
          <span class="signal-tier ${signal.tier}">${stratName}</span>
          <span class="signal-time">Q${signal.quarter} ${signal.quarterTime}</span>
        </div>
        <div class="signal-instruction">
          <div class="signal-action">${instruction.headline}</div>
          <div class="signal-details">
            <div class="detail-row">
              <span class="detail-label">Spread:</span>
              <span class="detail-value">${instruction.spread}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Moneyline:</span>
              <span class="detail-value">${instruction.moneyline}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Validated:</span>
              <span class="detail-value highlight-green">${instruction.combined}</span>
            </div>
          </div>
        </div>
        <div class="signal-details" style="font-size: 0.75rem; margin-top: 0.5rem;">
          <div class="detail-row">
            <span class="detail-label">Lead:</span>
            <span class="detail-value">${signal.lead}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Momentum:</span>
            <span class="detail-value">${signal.momentum}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Min Left:</span>
            <span class="detail-value">${signal.minsRemaining}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Leader Mkt Prob:</span>
            <span class="detail-value">${signal.marketProb}%</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Underdog Odds:</span>
            <span class="detail-value highlight-green">${signal.underdogOdds}</span>
          </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--amber);">
          ${instruction.urgency}
        </div>
      </div>`;
  }

  // =========================================================================
  // UPCOMING GAMES
  // =========================================================================
  function renderUpcomingGames() {
    const container = document.getElementById('upcoming-games');
    const scheduled = state.games.filter(g => g.status === 'scheduled');

    if (scheduled.length === 0) {
      container.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-icon">&#128197;</div>
          <p>No upcoming games found</p>
          <p class="empty-sub">Check back on game day</p>
        </div>`;
      return;
    }

    container.innerHTML = scheduled.map(g => {
      let timeStr = 'TBD';
      if (g.gameTime) {
        try {
          timeStr = new Date(g.gameTime).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
        } catch (e) {
          timeStr = g.statusText || 'TBD';
        }
      }

      return `
        <div class="upcoming-card">
          <div class="game-matchup">
            <span style="color: ${g.awayColor || '#e5e7eb'}">${g.awayTeam}</span>
            <span style="color: var(--text-muted)"> @ </span>
            <span style="color: ${g.homeColor || '#e5e7eb'}">${g.homeTeam}</span>
          </div>
          <div class="game-time">${timeStr}</div>
        </div>`;
    }).join('');
  }

  // =========================================================================
  // CHART GAME SELECT
  // =========================================================================
  function updateChartGameSelect() {
    const select = document.getElementById('chart-game-select');
    if (!select) return;

    const liveGames = state.games.filter(g =>
      (g.status === 'live' || g.status === 'halftime' || g.status === 'final') && g.possessions?.length > 0
    );

    const currentValue = select.value;
    select.innerHTML = '<option value="">Select a game...</option>';
    liveGames.forEach(g => {
      const opt = document.createElement('option');
      opt.value = g.id;
      opt.textContent = `${g.awayTeam} @ ${g.homeTeam} (${g.homeScore}-${g.awayScore})`;
      select.appendChild(opt);
    });

    if (currentValue && liveGames.find(g => g.id === currentValue)) {
      select.value = currentValue;
      renderGameChart(currentValue, 'scoreflow-chart');
    }

    // Hide empty state if we have a selection
    const emptyState = document.getElementById('chart-empty-state');
    if (emptyState) {
      emptyState.style.display = select.value ? 'none' : 'block';
    }
  }

  // =========================================================================
  // RENDER GAME CHART
  // =========================================================================
  function renderGameChart(gameId, canvasId) {
    const game = state.games.find(g => g.id === gameId);
    if (!game || !game.possessions || game.possessions.length === 0) return;

    const signals = game.signal ? [{
      ...game.signal,
      possIdx: game.signal.possessionIndex || game.possessions.length - 1,
      betTeam: game.signal.betTeamAbbr,
    }] : [];

    const title = `${game.awayTeam} @ ${game.homeTeam}`;
    Charts.renderScoreflow(canvasId, game.possessions, signals, title);

    // Hide empty state
    const emptyState = document.getElementById('chart-empty-state');
    if (emptyState) emptyState.style.display = 'none';
  }

  // =========================================================================
  // LIVE GAMES VIEW
  // =========================================================================
  function renderLiveGames() {
    const container = document.getElementById('live-games-full');
    const allGames = state.games;

    if (allGames.length === 0) {
      container.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-icon">&#127936;</div>
          <p>No games found</p>
          <p class="empty-sub">Check back on game day. Data refreshes every ${state.refreshInterval} seconds.</p>
        </div>`;
      return;
    }

    container.innerHTML = allGames.map(g => renderGameCard(g)).join('');

    // Click to show detail
    container.querySelectorAll('.game-card').forEach(card => {
      card.addEventListener('click', () => {
        const gameId = card.dataset.gameId;
        showGameDetail(gameId);
      });
    });
  }

  function showGameDetail(gameId) {
    const game = state.games.find(g => g.id === gameId);
    if (!game) return;

    const panel = document.getElementById('game-detail-panel');
    panel.classList.remove('hidden');

    document.getElementById('detail-game-title').textContent = `${game.awayTeam} @ ${game.homeTeam}`;

    // Render chart
    if (game.possessions && game.possessions.length > 0) {
      const signals = game.signal ? [{
        ...game.signal,
        possIdx: game.signal.possessionIndex || game.possessions.length - 1,
        betTeam: game.signal.betTeamAbbr,
      }] : [];
      Charts.renderScoreflow('detail-scoreflow-chart', game.possessions, signals, '');
    }

    // Render detail info
    const signalsList = document.getElementById('detail-signals-list');
    if (game.signal) {
      signalsList.innerHTML = renderSignalCard(game.signal);
    } else {
      signalsList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.8rem;">No signal detected for this game.</p>';
    }

    const analytics = document.getElementById('detail-analytics');
    analytics.innerHTML = `
      <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem;">
        <h4 style="font-size: 0.8rem; margin-bottom: 0.5rem;">Game Info</h4>
        <div class="signal-details" style="font-size: 0.75rem;">
          <div class="detail-row">
            <span class="detail-label">Score:</span>
            <span class="detail-value">${game.homeScore}-${game.awayScore}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Quarter:</span>
            <span class="detail-value">Q${game.quarter}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Possessions:</span>
            <span class="detail-value">${game.possessions?.length || 0}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Status:</span>
            <span class="detail-value">${game.status.toUpperCase()}</span>
          </div>
        </div>
      </div>`;
  }

  // =========================================================================
  // ALERTS VIEW
  // =========================================================================
  function renderAlerts() {
    const container = document.getElementById('alerts-list');

    if (state.alertHistory.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">&#128276;</div>
          <p>No alerts yet</p>
          <p class="empty-sub">Alerts will appear here when trading signals fire during live games.</p>
        </div>`;
      updateAlertStats();
      return;
    }

    container.innerHTML = state.alertHistory.map(alert => {
      const instruction = alert.instruction || SignalEngine.getTradeInstruction(alert);
      const time = new Date(alert.timestamp).toLocaleString();

      const stratName = alert.strategyName || alert.tier.toUpperCase();

      return `
        <div class="alert-item">
          <div class="alert-item-tier ${alert.tier}">${stratName}</div>
          <div class="alert-item-body">
            <h4>${instruction?.headline || `FADE SIGNAL (${stratName})`}</h4>
            <p>${alert.gameName || ''} | ${time}</p>
            <p style="margin-top: 0.3rem;">${instruction?.spread || ''}</p>
            <p>${instruction?.moneyline || ''}</p>
            <p style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.25rem;">
              Lead: ${alert.lead} | Mom: ${alert.momentum} | ${alert.minsRemaining} min left
            </p>
          </div>
          <div class="alert-item-result">
            <span class="signal-outcome ${alert.outcome || 'pending'}">${(alert.outcome || 'PENDING').toUpperCase()}</span>
          </div>
        </div>`;
    }).join('');

    updateAlertStats();
  }

  function updateAlertStats() {
    const wins = state.alertHistory.filter(a => a.outcome === 'win').length;
    const losses = state.alertHistory.filter(a => a.outcome === 'loss').length;
    const decided = wins + losses;

    document.getElementById('alert-wins').textContent = wins;
    document.getElementById('alert-losses').textContent = losses;
    document.getElementById('alert-wr').textContent = decided > 0 ? `${Math.round(wins / decided * 100)}%` : '--';
  }

  // =========================================================================
  // HISTORICAL VIEW
  // =========================================================================
  function initHistoricalData() {
    state.equityData = HistoricalData.generateEquityCurve();
    state.signalLog = HistoricalData.generateSignalLog();

    // Generate score flows for historical games
    state.historyGames = HistoricalData.GAMES.map(game => ({
      ...game,
      possessions: HistoricalData.generateGameScoreFlow(game),
    }));
  }

  function renderHistory() {
    renderHistoryChart(0);
    renderHistoryTable();
  }

  function renderHistoryChart(gameIndex) {
    const game = state.historyGames[gameIndex];
    if (!game) return;

    const sig = game.signal;
    const signals = [{
      tier: sig.tier,
      possessionIndex: sig.possessionIndex,
      possIdx: sig.possessionIndex,
      betTeam: sig.betTeam,
      betTeamAbbr: sig.betTeam,
    }];

    const title = `${game.homeTeam} vs ${game.awayTeam} - ${game.date}`;
    Charts.renderScoreflow('history-chart', game.possessions, signals, title);
  }

  function renderHistoryTable() {
    const tbody = document.getElementById('history-table-body');
    if (!tbody || !state.signalLog) return;

    const filter = document.getElementById('history-filter')?.value || 'all';

    let filtered = state.signalLog;
    if (filter === 'composite') filtered = filtered.filter(s => s.tier === 'composite');
    else if (filter === 'blowout_compress') filtered = filtered.filter(s => s.tier === 'blowout_compress');
    else if (filter === 'quant') filtered = filtered.filter(s => s.tier === 'quant');
    else if (filter === 'burst_fade') filtered = filtered.filter(s => s.tier === 'burst_fade');
    else if (filter === 'q3_fade') filtered = filtered.filter(s => s.tier === 'q3_fade');
    else if (filter === 'fade_ml') filtered = filtered.filter(s => s.tier === 'fade_ml');
    else if (filter === 'fade_spread') filtered = filtered.filter(s => s.tier === 'fade_spread');
    else if (filter === 'wins') filtered = filtered.filter(s => s.spreadOutcome === 'win');
    else if (filter === 'losses') filtered = filtered.filter(s => s.spreadOutcome === 'loss');

    // Update summary stats
    const totalSignals = filtered.length;
    const spreadWins = filtered.filter(s => s.spreadOutcome === 'win').length;
    const mlWins = filtered.filter(s => s.mlOutcome === 'win').length;
    const totalPnl = filtered.reduce((sum, s) => sum + s.totalPnl, 0);

    document.getElementById('hist-total-signals').textContent = totalSignals;
    document.getElementById('hist-spread-wr').textContent = totalSignals > 0 ? `${(spreadWins / totalSignals * 100).toFixed(1)}%` : '--';
    document.getElementById('hist-ml-wr').textContent = totalSignals > 0 ? `${(mlWins / totalSignals * 100).toFixed(1)}%` : '--';
    document.getElementById('hist-total-pnl').textContent = `${totalPnl >= 0 ? '+' : ''}$${Math.round(totalPnl * 100)}`;

    const stratNames = { composite: 'COMPOSITE', blowout_compress: 'BLOWOUT', quant: 'QUANT', burst_fade: 'BURST', q3_fade: 'Q3 FADE', fade_ml: 'FADE ML', fade_spread: 'FADE SPR' };
    tbody.innerHTML = filtered.slice(0, 100).map(sig => `
      <tr>
        <td>${sig.date}</td>
        <td>${sig.away} @ ${sig.home}</td>
        <td>Fade ${sig.leaderTeam || sig.home}</td>
        <td><span class="tier-badge ${sig.tier} sm">${stratNames[sig.tier] || sig.tier.toUpperCase()}</span></td>
        <td>${sig.lead}</td>
        <td>${sig.momentum}</td>
        <td>${sig.minsRemaining}</td>
        <td>${sig.betTeam} +${sig.lead}</td>
        <td class="${sig.spreadOutcome}">${sig.spreadOutcome.toUpperCase()}</td>
        <td class="${sig.mlOutcome}">${sig.mlOutcome.toUpperCase()}</td>
        <td class="${sig.totalPnl >= 0 ? 'highlight-green' : 'highlight-red'}">
          ${sig.totalPnl >= 0 ? '+' : ''}${sig.totalPnl.toFixed(2)}
        </td>
      </tr>`).join('');
  }

  // =========================================================================
  // BACKTEST VIEW
  // =========================================================================
  function renderBacktest() {
    if (!state.equityData) {
      state.equityData = HistoricalData.generateEquityCurve();
    }

    const mode = document.getElementById('equity-mode-select')?.value || 'combined';
    Charts.renderEquityCurve('equity-chart', state.equityData, mode);
  }

  // =========================================================================
  // STARTUP
  // =========================================================================
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
