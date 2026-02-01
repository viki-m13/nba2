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
    ouSignals: [],         // Q3 Over/Under signals
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
    ouLineOverrides: {},   // Manual O/U line overrides per game
    equityData: null,
    signalLog: null,
    historyGames: null,
  };

  const TIER_PRIORITY = { breakout_ml: -1, composite: 0, blowout_compress: 1, quant: 2, burst_fade: 3, q3_fade: 4, fade_ml: 5, fade_spread: 6 };

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
    if (view === 'analytics') renderAnalytics();
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

    // O/U scan button
    document.getElementById('ou-scan-btn')?.addEventListener('click', () => {
      scanOUSignals();
    });

    // O/U line input - allow Enter key
    document.getElementById('ou-line-input')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') scanOUSignals();
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
      processOUSignals(games);

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
  // Q3 OVER/UNDER SIGNAL PROCESSING
  // =========================================================================
  function processOUSignals(games) {
    state.ouSignals = [];

    const ouLineInput = document.getElementById('ou-line-input');
    const globalOULine = ouLineInput ? parseFloat(ouLineInput.value) : 0;

    for (const game of games) {
      if (game.status !== 'live' && game.status !== 'halftime') continue;
      if (!game.possessions || game.possessions.length < 20) continue;

      // Get O/U line: per-game override > global input > auto-estimate
      const ouLine = state.ouLineOverrides[game.id] || globalOULine || Q3OUEngine.estimateOULine(game.possessions);

      if (!ouLine || ouLine <= 0) continue;

      const signal = Q3OUEngine.evaluateFromPossessions(
        game.possessions, game.homeTeam, game.awayTeam, ouLine
      );

      if (signal) {
        const signalKey = `ou-${game.id}-${signal.tier}-${signal.direction}`;

        state.ouSignals.push({
          ...signal,
          gameId: game.id,
          gameName: `${game.homeTeam} vs ${game.awayTeam}`,
          signalKey,
          gameStatus: `Q${game.quarter} ${game.gameClock}`,
        });

        // Fire alert for new O/U signals
        if (!state.seenSignals.has(signalKey)) {
          state.seenSignals.add(signalKey);

          const instruction = Q3OUEngine.getTradeInstruction(signal);
          const alert = {
            ...signal,
            gameId: game.id,
            gameName: `${game.homeTeam} vs ${game.awayTeam}`,
            signalKey,
            instruction: {
              headline: instruction.headline,
              spread: instruction.bet,
              moneyline: instruction.detail,
              context: instruction.context,
              combined: instruction.urgency,
            },
            timestamp: Date.now(),
            outcome: 'pending',
          };

          state.alertHistory.unshift(alert);
          saveAlerts();
          fireOUAlert(alert);
        }
      }
    }

    renderOUSignals();
  }

  function scanOUSignals() {
    processOUSignals(state.games);
  }

  function fireOUAlert(alert) {
    // Sound
    if (state.soundEnabled) {
      playOUAlertSound(alert.tier);
    }

    // Browser notification
    if (state.notificationsEnabled && 'Notification' in window && Notification.permission === 'granted') {
      new Notification(`Q3 O/U: ${alert.direction} ${alert.ouLine} (${alert.tier})`, {
        body: `${alert.gameName}\n${alert.description}`,
        tag: alert.signalKey,
      });
    }

    // Banner
    const banner = document.getElementById('signal-alert-banner');
    const bannerMsg = document.getElementById('alert-banner-msg');
    if (banner && bannerMsg) {
      bannerMsg.textContent = `Q3 O/U: ${alert.direction} ${alert.ouLine} (${alert.tier}) | ${alert.gameName} | ${alert.accuracyPct}% accuracy`;
      banner.classList.remove('hidden');
      setTimeout(() => banner.classList.add('hidden'), 20000);
    }

    updateAlertBadge();
  }

  function playOUAlertSound(tier) {
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);

      // Ascending tones for O/U signals (distinct from fade signals)
      const tones = {
        PLATINUM: [523, 659, 784, 1047], // C5-E5-G5-C6
        GOLD: [523, 659, 784],           // C5-E5-G5
        SILVER: [523, 659],              // C5-E5
        BRONZE: [523],                   // C5
      };
      const freqs = tones[tier] || [523];

      gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
      let t = audioCtx.currentTime;
      for (const freq of freqs) {
        osc.frequency.setValueAtTime(freq, t);
        t += 0.12;
      }
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
      osc.start(audioCtx.currentTime);
      osc.stop(t + 0.4);
    } catch (e) {}
  }

  function renderOUSignals() {
    const container = document.getElementById('ou-signals-list');
    const countBadge = document.getElementById('ou-signals-count');

    if (!container) return;

    countBadge.textContent = state.ouSignals.length;

    if (state.ouSignals.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">&#9878;</div>
          <p>No O/U signals yet</p>
          <p class="empty-sub">Signals fire at Q3 end when model predicts final total with high margin from O/U line. Enter O/U line above or wait for Q3.</p>
        </div>`;
      return;
    }

    container.innerHTML = state.ouSignals.map(signal => renderOUSignalCard(signal)).join('');
  }

  function renderOUSignalCard(signal) {
    const dirClass = signal.direction.toLowerCase();
    const tierClass = 'ou-' + signal.tier.toLowerCase();
    const tierBadgeClass = signal.tier;

    // Accuracy gauge (map 95-100% to 0-100% width)
    const gaugePct = Math.min(100, Math.max(0, (signal.confidence - 0.90) * 1000));

    return `
      <div class="ou-signal-card ${tierClass}">
        <div class="ou-signal-header">
          <div>
            <span class="ou-direction-badge ${dirClass}">${signal.direction}</span>
            <span style="font-family: var(--font-mono); font-size: 1rem; font-weight: 700; margin-left: 0.5rem;">${signal.ouLine}</span>
          </div>
          <div>
            <span class="ou-tier-badge ${tierBadgeClass}">${signal.tier}</span>
          </div>
        </div>

        <div class="ou-accuracy-gauge">
          <div class="ou-gauge-header">
            <span class="ou-gauge-label">Accuracy (OOS)</span>
            <span class="ou-gauge-value" style="color: ${signal.tierColor};">${signal.accuracyPct}%</span>
          </div>
          <div class="ou-gauge-bar">
            <div class="ou-gauge-fill ${signal.tier.toLowerCase()}" style="width: ${gaugePct}%;"></div>
          </div>
        </div>

        <div class="ou-details-grid">
          <div class="ou-detail-row">
            <span class="ou-detail-label">Q3 Total:</span>
            <span class="ou-detail-value">${signal.q3CumulTotal}</span>
          </div>
          <div class="ou-detail-row">
            <span class="ou-detail-label">Need in Q4:</span>
            <span class="ou-detail-value">${signal.ptsNeeded}</span>
          </div>
          <div class="ou-detail-row">
            <span class="ou-detail-label">Model Q4:</span>
            <span class="ou-detail-value">${signal.predictedQ4}</span>
          </div>
          <div class="ou-detail-row">
            <span class="ou-detail-label">Pred Final:</span>
            <span class="ou-detail-value">${signal.predictedFinal}</span>
          </div>
          <div class="ou-detail-row">
            <span class="ou-detail-label">Margin:</span>
            <span class="ou-detail-value" style="color: ${signal.tierColor};">${signal.margin} pts</span>
          </div>
          <div class="ou-detail-row">
            <span class="ou-detail-label">Pace:</span>
            <span class="ou-detail-value">${signal.paceRatio}x</span>
          </div>
          <div class="ou-detail-row">
            <span class="ou-detail-label">Q3 Lead:</span>
            <span class="ou-detail-value">${signal.q3Lead}</span>
          </div>
          <div class="ou-detail-row">
            <span class="ou-detail-label">Edge/bet:</span>
            <span class="ou-detail-value highlight-gold">+$${signal.edge.toFixed(2)}</span>
          </div>
        </div>

        <div class="ou-action-bar">
          <span class="ou-bet-instruction ${dirClass}">BET ${signal.direction} ${signal.ouLine} @ -110</span>
          <span class="ou-kelly-size">Kelly: ${(signal.kelly * 100).toFixed(1)}%</span>
        </div>

        <div style="font-size: 0.65rem; color: var(--text-muted); margin-top: 0.5rem; text-align: center;">
          ${signal.gameName} | ${signal.gameStatus || ''}
        </div>
      </div>`;
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
    document.getElementById('stat-active-signals').textContent = state.liveSignals.length + state.ouSignals.length;

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
    const mlConf = signal.mlConfidence || 0;

    // ML confidence gauge
    const gaugeColor = mlConf >= 80 ? '#dc2626' : mlConf >= 70 ? '#f59e0b' : mlConf >= 65 ? '#10b981' : '#6b7280';
    const gaugePct = Math.min(100, Math.max(0, (mlConf - 50) * 2)); // 50-100 → 0-100%
    const confidenceGauge = mlConf > 0 ? `
      <div class="ml-confidence-gauge">
        <div class="gauge-header">
          <span class="gauge-label">ML Confidence</span>
          <span class="gauge-value" style="color: ${gaugeColor};">${mlConf}%</span>
        </div>
        <div class="gauge-bar">
          <div class="gauge-fill" style="width: ${gaugePct}%; background: ${gaugeColor};"></div>
          <div class="gauge-markers">
            <span class="gauge-marker" style="left: 30%;" title="65%">|</span>
            <span class="gauge-marker" style="left: 40%;" title="70%">|</span>
            <span class="gauge-marker" style="left: 50%;" title="75%">|</span>
            <span class="gauge-marker" style="left: 60%;" title="80%">|</span>
          </div>
        </div>
        <div class="gauge-ticks">
          <span>50%</span><span>65%</span><span>75%</span><span>85%+</span>
        </div>
      </div>` : '';

    // Breakout indicators
    const breakoutBadges = signal.breakouts ? [
      signal.breakouts.bbSqueeze ? '<span class="breakout-badge bb">BB Squeeze</span>' : '',
      signal.breakouts.donchian ? `<span class="breakout-badge donchian">Donchian ${signal.breakouts.donchian}</span>` : '',
      signal.breakouts.maCrossover ? `<span class="breakout-badge ma">${signal.breakouts.maCrossover === 'golden_cross' ? 'Golden Cross' : 'Death Cross'}</span>` : '',
      signal.breakouts.momentumBreakout ? '<span class="breakout-badge mom">Mom Breakout</span>' : '',
      signal.breakouts.volumeConfirmed ? '<span class="breakout-badge vol">Vol Confirm</span>' : '',
    ].filter(Boolean).join('') : '';

    // Combo signals
    const comboSignals = signal.comboSignals && signal.comboSignals.length > 0
      ? signal.comboSignals.map(c => `<span class="combo-badge" title="${c.wr*100}% WR, +${c.roi}% ROI (${c.n} samples)">${c.name}</span>`).join('')
      : '';

    // Kelly sizing
    const kellyDisplay = signal.kellyPct > 0
      ? `<div class="detail-row"><span class="detail-label">Kelly Size:</span><span class="detail-value highlight-green">${signal.suggestedBetSize}</span></div>`
      : '';

    return `
      <div class="signal-card ${signal.tier} new-signal">
        <div class="signal-card-header">
          <span class="signal-tier ${signal.tier}">${stratName}</span>
          <span class="signal-time">Q${signal.quarter} ${signal.quarterTime}</span>
        </div>
        ${confidenceGauge}
        ${breakoutBadges ? `<div class="breakout-badges-row">${breakoutBadges}</div>` : ''}
        ${comboSignals ? `<div class="combo-badges-row">${comboSignals}</div>` : ''}
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
            ${kellyDisplay}
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
            <span class="detail-label">Breakout Signals:</span>
            <span class="detail-value">${signal.breakoutCount || 0}/6</span>
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
        <div class="signal-urgency">
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
    if (filter === 'breakout_ml') filtered = filtered.filter(s => s.tier === 'breakout_ml');
    else if (filter === 'composite') filtered = filtered.filter(s => s.tier === 'composite');
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

    const stratNames = { breakout_ml: 'ML+BRKOUT', composite: 'COMPOSITE', blowout_compress: 'BLOWOUT', quant: 'QUANT', burst_fade: 'BURST', q3_fade: 'Q3 FADE', fade_ml: 'FADE ML', fade_spread: 'FADE SPR' };
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
  // ANALYTICS VIEW
  // =========================================================================
  function renderAnalytics() {
    // Render confidence tiers cards
    renderConfidenceTiers();

    // Render feature importance chart
    Charts.renderFeatureImportance('feature-importance-chart', SignalEngine.TOP_FEATURES);

    // Render confidence curve chart
    Charts.renderConfidenceCurve('confidence-curve-chart', SignalEngine.CONFIDENCE_TIERS);

    // Render combo signal cards
    renderComboCards();
  }

  function renderConfidenceTiers() {
    const container = document.getElementById('confidence-tiers-grid');
    if (!container) return;

    const tiers = SignalEngine.CONFIDENCE_TIERS;
    container.innerHTML = tiers.map(tier => {
      const isHighlight = tier.threshold === 0.70;
      return `
        <div class="confidence-tier-card ${isHighlight ? 'highlighted' : ''}">
          <div class="tier-threshold">${Math.round(tier.threshold * 100)}%</div>
          <div class="tier-stats">
            <div class="tier-stat">
              <span class="tier-stat-value highlight-green">${tier.wr}%</span>
              <span class="tier-stat-label">Win Rate</span>
            </div>
            <div class="tier-stat">
              <span class="tier-stat-value highlight-green">+${tier.roi}%</span>
              <span class="tier-stat-label">ROI</span>
            </div>
            <div class="tier-stat">
              <span class="tier-stat-value">${tier.signals.toLocaleString()}</span>
              <span class="tier-stat-label">Signals</span>
            </div>
            <div class="tier-stat">
              <span class="tier-stat-value">${(tier.kelly * 100).toFixed(1)}%</span>
              <span class="tier-stat-label">Kelly</span>
            </div>
          </div>
        </div>`;
    }).join('');
  }

  function renderComboCards() {
    const container = document.getElementById('combo-cards-grid');
    if (!container) return;

    const combos = SignalEngine.COMBO_SIGNALS;
    container.innerHTML = combos.map((combo, idx) => {
      const rankColors = ['#dc2626', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'];
      const color = rankColors[idx] || '#6b7280';
      return `
        <div class="combo-card" style="border-top: 3px solid ${color};">
          <div class="combo-card-rank" style="color: ${color};">#${idx + 1}</div>
          <div class="combo-card-name">${combo.name}</div>
          <div class="combo-card-stats">
            <div class="combo-stat">
              <span class="combo-stat-value highlight-green">${(combo.wr * 100).toFixed(1)}%</span>
              <span class="combo-stat-label">Win Rate</span>
            </div>
            <div class="combo-stat">
              <span class="combo-stat-value highlight-green">+${combo.roi.toFixed(1)}%</span>
              <span class="combo-stat-label">ROI</span>
            </div>
            <div class="combo-stat">
              <span class="combo-stat-value">${combo.n.toLocaleString()}</span>
              <span class="combo-stat-label">Samples</span>
            </div>
          </div>
        </div>`;
    }).join('');
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
