// =============================================================================
// MAIN APP CONTROLLER - Q3 Over/Under Signal Engine
// =============================================================================

(function() {
  'use strict';

  // =========================================================================
  // STATE
  // =========================================================================
  const state = {
    currentView: 'dashboard',
    games: [],
    ouSignals: [],         // Active Q3 O/U signals from live games
    alertHistory: [],      // All alerts (persisted in localStorage)
    selectedGameId: null,
    refreshInterval: 5,
    refreshTimer: null,
    countdown: 5,
    countdownTimer: null,
    soundEnabled: true,
    notificationsEnabled: false,
    minTier: 'BRONZE',
    seenSignals: new Set(),
  };

  const TIER_PRIORITY = { PLATINUM: 0, GOLD: 1, SILVER: 2, BRONZE: 3 };
  const WIN_PAYOUT = 100 / 110;

  // =========================================================================
  // INITIALIZATION
  // =========================================================================
  async function init() {
    loadSettings();
    loadAlerts();
    setupNavigation();
    setupEventListeners();
    navigateTo(window.location.hash.slice(1) || 'dashboard');

    // Auto-detect Vercel proxy
    const hasVercel = await NbaApi.detectVercelProxy();
    if (hasVercel) {
      console.log('[App] Using Vercel proxy for NBA data');
    } else if (!NbaApi.getCorsProxy()) {
      console.log('[App] No CORS proxy configured. Live data may not load. Set one in Settings.');
    }

    startRefreshLoop();
    console.log('[App] Q3 O/U Signal Engine initialized');
  }

  // =========================================================================
  // SETTINGS
  // =========================================================================
  function loadSettings() {
    try {
      const saved = JSON.parse(localStorage.getItem('q3ou-settings') || '{}');
      if (saved.corsProxy) NbaApi.setCorsProxy(saved.corsProxy);
      if (saved.refreshInterval) state.refreshInterval = saved.refreshInterval;
      if (saved.soundEnabled !== undefined) state.soundEnabled = saved.soundEnabled;
      if (saved.notificationsEnabled !== undefined) state.notificationsEnabled = saved.notificationsEnabled;
      if (saved.minTier) state.minTier = saved.minTier;

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

    localStorage.setItem('q3ou-settings', JSON.stringify({
      corsProxy, refreshInterval: state.refreshInterval,
      soundEnabled, notificationsEnabled, minTier,
    }));

    if (notificationsEnabled && 'Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    startRefreshLoop();
    document.getElementById('settings-modal').classList.add('hidden');
  }

  // =========================================================================
  // ALERTS PERSISTENCE
  // =========================================================================
  function loadAlerts() {
    try {
      const saved = JSON.parse(localStorage.getItem('q3ou-alerts') || '[]');
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
      const toSave = state.alertHistory.slice(0, 200);
      localStorage.setItem('q3ou-alerts', JSON.stringify(toSave));
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
        navigateTo(link.dataset.view);
      });
    });
  }

  function navigateTo(view) {
    if (!view || !document.getElementById(`view-${view}`)) view = 'dashboard';

    state.currentView = view;
    window.location.hash = view;

    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    const activeLink = document.querySelector(`.nav-link[data-view="${view}"]`);
    if (activeLink) activeLink.classList.add('active');

    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${view}`).classList.add('active');

    if (view === 'backtest') renderBacktest();
    if (view === 'history') renderHistory();
    if (view === 'alerts') renderAlerts();
  }

  // =========================================================================
  // EVENT LISTENERS
  // =========================================================================
  function setupEventListeners() {
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

    document.getElementById('alert-banner-close')?.addEventListener('click', () => {
      document.getElementById('signal-alert-banner').classList.add('hidden');
    });

    document.getElementById('clear-alerts-btn')?.addEventListener('click', clearAlerts);
    document.getElementById('refresh-now-btn')?.addEventListener('click', fetchAndUpdate);

    document.getElementById('chart-game-select')?.addEventListener('change', (e) => {
      if (e.target.value) renderGameChart(e.target.value, 'scoreflow-chart');
    });

    document.getElementById('close-detail-btn')?.addEventListener('click', () => {
      document.getElementById('game-detail-panel').classList.add('hidden');
    });

    document.getElementById('history-filter')?.addEventListener('change', () => renderHistoryTable());
    document.getElementById('history-season-filter')?.addEventListener('change', () => renderHistoryTable());
  }

  // =========================================================================
  // REFRESH LOOP
  // =========================================================================
  function startRefreshLoop() {
    if (state.refreshTimer) clearInterval(state.refreshTimer);
    if (state.countdownTimer) clearInterval(state.countdownTimer);

    state.countdown = state.refreshInterval;
    fetchAndUpdate();

    state.refreshTimer = setInterval(() => {
      fetchAndUpdate();
      state.countdown = state.refreshInterval;
    }, state.refreshInterval * 1000);

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
      processOUSignals(games);
      renderDashboard();
      if (state.currentView === 'live') renderLiveGames();
      updateStats();

    } catch (e) {
      console.warn('[Fetch] Exception:', e);
      indicator?.classList.add('paused');
    }
  }

  // =========================================================================
  // Q3 O/U SIGNAL PROCESSING
  // =========================================================================
  function processOUSignals(games) {
    state.ouSignals = [];

    for (const game of games) {
      if (game.status !== 'live' && game.status !== 'halftime') continue;

      if (!game.possessions || game.possessions.length < 20) {
        console.log(`[Signal] ${game.awayTeam} @ ${game.homeTeam}: skipped - only ${game.possessions?.length || 0} possessions`);
        continue;
      }

      // O/U line must come from ESPN data - NO fallback estimation
      const ouLine = game.ouLine || 0;
      if (!ouLine || ouLine <= 0) {
        console.warn(`[Signal] ${game.awayTeam} @ ${game.homeTeam}: skipped - no O/U line (ESPN odds unavailable)`);
        continue;
      }

      console.log(`[Signal] ${game.awayTeam} @ ${game.homeTeam}: evaluating with O/U ${ouLine}, ${game.possessions.length} possessions`);

      const signal = Q3OUEngine.evaluateFromPossessions(
        game.possessions, game.homeTeam, game.awayTeam, ouLine
      );

      if (!signal) {
        const lastPos = game.possessions[game.possessions.length - 1];
        console.log(`[Signal] ${game.awayTeam} @ ${game.homeTeam}: no signal (Q${lastPos?.quarter} ${lastPos?.quarterTime} - need Q3 <=3:00 or Q4+, or margin < 10)`);
      }

      if (signal && !meetsMinTier(signal.tier)) {
        console.log(`[Signal] ${game.awayTeam} @ ${game.homeTeam}: ${signal.tier} ${signal.direction} filtered (min tier: ${state.minTier})`);
      }

      if (signal && meetsMinTier(signal.tier)) {
        console.log(`[Signal] ${game.awayTeam} @ ${game.homeTeam}: ${signal.tier} ${signal.direction} O/U ${ouLine} | margin ${signal.margin} | pred ${signal.predictedFinal}`);
        const signalKey = `ou-${game.id}-${signal.tier}-${signal.direction}`;

        state.ouSignals.push({
          ...signal,
          gameId: game.id,
          gameName: `${game.homeTeam} vs ${game.awayTeam}`,
          signalKey,
          gameStatus: `Q${game.quarter} ${game.gameClock}`,
        });

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
              bet: instruction.bet,
              detail: instruction.detail,
              context: instruction.context,
              urgency: instruction.urgency,
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

  function meetsMinTier(tier) {
    return TIER_PRIORITY[tier] <= TIER_PRIORITY[state.minTier];
  }

  // =========================================================================
  // ALERT NOTIFICATIONS
  // =========================================================================
  function fireOUAlert(alert) {
    if (state.soundEnabled) playOUAlertSound(alert.tier);

    if (state.notificationsEnabled && 'Notification' in window && Notification.permission === 'granted') {
      new Notification(`Q3 O/U: ${alert.direction} ${alert.ouLine} (${alert.tier})`, {
        body: `${alert.gameName}\n${alert.description}`,
        tag: alert.signalKey,
      });
    }

    const banner = document.getElementById('signal-alert-banner');
    const bannerMsg = document.getElementById('alert-banner-msg');
    if (banner && bannerMsg) {
      bannerMsg.textContent = `${alert.direction} ${alert.ouLine} (${alert.tier}) | ${alert.gameName} | ${alert.accuracyPct}% accuracy`;
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

      const tones = {
        PLATINUM: [523, 659, 784, 1047],
        GOLD: [523, 659, 784],
        SILVER: [523, 659],
        BRONZE: [523],
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
    document.getElementById('stat-active-signals').textContent = state.ouSignals.length;
    updateAlertBadge();
  }

  // =========================================================================
  // RENDER DASHBOARD
  // =========================================================================
  function renderDashboard() {
    renderDashboardGames();
    renderUpcomingGames();
    updateChartGameSelect();
  }

  function renderDashboardGames() {
    const container = document.getElementById('dashboard-live-games');
    const countBadge = document.getElementById('live-games-count');
    const liveGames = state.games.filter(g => g.status === 'live' || g.status === 'halftime');

    countBadge.textContent = liveGames.length;

    if (liveGames.length === 0) {
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
    const hasOUSignal = state.ouSignals.some(s => s.gameId === game.id);
    const diff = game.homeScore - game.awayScore;

    let statusText = game.statusText || game.status.toUpperCase();
    if (game.status === 'live') statusText = `Q${game.quarter} ${game.gameClock}`;
    if (game.status === 'halftime') statusText = 'HALFTIME';
    if (game.status === 'scheduled') {
      statusText = game.gameTime ? new Date(game.gameTime).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : 'SCHEDULED';
    }

    const ouBadge = hasOUSignal ? (() => {
      const sig = state.ouSignals.find(s => s.gameId === game.id);
      return `<span class="ou-tier-badge ${sig.tier}">${sig.tier} ${sig.direction}</span>`;
    })() : '';

    return `
      <div class="game-card ${hasOUSignal ? 'has-signal' : ''}" data-game-id="${game.id}">
        <div class="game-card-header">
          <span class="game-status ${game.status}">${statusText}</span>
          ${ouBadge}
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
          ${game.ouLine ? `<span>O/U: ${game.ouLine}</span>` : '<span style="color:#ef4444;">No O/U line</span>'}
          <span>${game.possessions?.length || 0} plays</span>
        </div>
      </div>`;
  }

  // =========================================================================
  // RENDER O/U SIGNALS
  // =========================================================================
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
          <p class="empty-sub">Signals fire at Q3 end when model predicts final total with high margin from pregame O/U line (ESPN). Waiting for Q3 games.</p>
        </div>`;
      return;
    }

    container.innerHTML = state.ouSignals.map(signal => renderOUSignalCard(signal)).join('');
  }

  function renderOUSignalCard(signal) {
    const dirClass = signal.direction.toLowerCase();
    const tierClass = 'ou-' + signal.tier.toLowerCase();
    const gaugePct = Math.min(100, Math.max(0, (signal.confidence - 0.90) * 1000));

    return `
      <div class="ou-signal-card ${tierClass}">
        <div class="ou-signal-header">
          <div>
            <span class="ou-direction-badge ${dirClass}">${signal.direction}</span>
            <span style="font-family: var(--font-mono); font-size: 1rem; font-weight: 700; margin-left: 0.5rem;">${signal.ouLine}</span>
          </div>
          <div>
            <span class="ou-tier-badge ${signal.tier}">${signal.tier}</span>
          </div>
        </div>

        <div class="ou-accuracy-gauge">
          <div class="ou-gauge-header">
            <span class="ou-gauge-label">Accuracy (validated)</span>
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

    const emptyState = document.getElementById('chart-empty-state');
    if (emptyState) emptyState.style.display = select.value ? 'none' : 'block';
  }

  function renderGameChart(gameId, canvasId) {
    const game = state.games.find(g => g.id === gameId);
    if (!game || !game.possessions || game.possessions.length === 0) return;

    const title = `${game.awayTeam} @ ${game.homeTeam}`;
    Charts.renderScoreflow(canvasId, game.possessions, [], title);

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

    container.querySelectorAll('.game-card').forEach(card => {
      card.addEventListener('click', () => showGameDetail(card.dataset.gameId));
    });
  }

  function showGameDetail(gameId) {
    const game = state.games.find(g => g.id === gameId);
    if (!game) return;

    const panel = document.getElementById('game-detail-panel');
    panel.classList.remove('hidden');

    document.getElementById('detail-game-title').textContent = `${game.awayTeam} @ ${game.homeTeam}`;

    if (game.possessions && game.possessions.length > 0) {
      Charts.renderScoreflow('detail-scoreflow-chart', game.possessions, [], '');
    }

    const signalsList = document.getElementById('detail-signals-list');
    const ouSig = state.ouSignals.find(s => s.gameId === gameId);
    if (ouSig) {
      signalsList.innerHTML = renderOUSignalCard(ouSig);
    } else {
      signalsList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.8rem;">No Q3 O/U signal for this game. Signals fire at Q3 end when margin from O/U line is >= 10 points.</p>';
    }

    const analytics = document.getElementById('detail-analytics');
    analytics.innerHTML = `
      <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem;">
        <h4 style="font-size: 0.8rem; margin-bottom: 0.5rem;">Game Info</h4>
        <div class="signal-details" style="font-size: 0.75rem;">
          <div class="detail-row"><span class="detail-label">Score:</span><span class="detail-value">${game.homeScore}-${game.awayScore}</span></div>
          <div class="detail-row"><span class="detail-label">Quarter:</span><span class="detail-value">Q${game.quarter}</span></div>
          <div class="detail-row"><span class="detail-label">Plays:</span><span class="detail-value">${game.possessions?.length || 0}</span></div>
          <div class="detail-row"><span class="detail-label">Status:</span><span class="detail-value">${game.status.toUpperCase()}</span></div>
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
          <p class="empty-sub">Alerts appear when Q3 O/U signals fire during live games.</p>
        </div>`;
      updateAlertStats();
      return;
    }

    container.innerHTML = state.alertHistory.map(alert => {
      const instruction = alert.instruction;
      const time = new Date(alert.timestamp).toLocaleString();

      return `
        <div class="alert-item">
          <div class="alert-item-tier ou-${(alert.tier || 'bronze').toLowerCase()}">${alert.tier || 'O/U'}</div>
          <div class="alert-item-body">
            <h4>${instruction?.headline || `${alert.direction} ${alert.ouLine} (${alert.tier})`}</h4>
            <p>${alert.gameName || ''} | ${time}</p>
            <p style="margin-top: 0.3rem;">${instruction?.bet || ''}</p>
            <p>${instruction?.detail || ''}</p>
            <p style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.25rem;">
              ${instruction?.context || ''}
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
  function renderHistory() {
    renderHistoryTable();
    renderEquityCurve();
  }

  function renderHistoryTable() {
    const tbody = document.getElementById('history-table-body');
    if (!tbody || !window.HistoricalData) return;

    const signals = window.HistoricalData.signals || [];
    const tierFilter = document.getElementById('history-filter')?.value || 'all';
    const seasonFilter = document.getElementById('history-season-filter')?.value || 'all';

    let filtered = signals;

    if (tierFilter === 'PLATINUM' || tierFilter === 'GOLD' || tierFilter === 'SILVER' || tierFilter === 'BRONZE') {
      filtered = filtered.filter(s => s.tier === tierFilter);
    } else if (tierFilter === 'OVER' || tierFilter === 'UNDER') {
      filtered = filtered.filter(s => s.direction === tierFilter);
    } else if (tierFilter === 'correct') {
      filtered = filtered.filter(s => s.openingCorrect);
    } else if (tierFilter === 'incorrect') {
      filtered = filtered.filter(s => !s.openingCorrect && !s.isPushOpening);
    }

    if (seasonFilter !== 'all') {
      filtered = filtered.filter(s => s.season === seasonFilter);
    }

    // Update summary stats
    const total = filtered.length;
    const correct = filtered.filter(s => s.openingCorrect).length;
    const incorrect = filtered.filter(s => !s.openingCorrect && !s.isPushOpening).length;
    const pnl = correct * WIN_PAYOUT - incorrect;
    const roi = total > 0 ? (pnl / total * 100) : 0;

    document.getElementById('hist-total-signals').textContent = total.toLocaleString();
    document.getElementById('hist-accuracy').textContent = total > 0 ? `${(correct / total * 100).toFixed(1)}%` : '--';
    document.getElementById('hist-total-pnl').textContent = `${pnl >= 0 ? '+' : ''}${pnl.toFixed(1)}u`;
    document.getElementById('hist-roi').textContent = `${roi >= 0 ? '+' : ''}${roi.toFixed(1)}%`;

    // Pred Final vs Actual Final directional accuracy (responds to filters)
    renderPredAccuracy(filtered);

    const showingBadge = document.getElementById('hist-showing-count');
    if (showingBadge) showingBadge.textContent = Math.min(filtered.length, 200);

    tbody.innerHTML = filtered.slice(0, 200).map(sig => {
      const pnlVal = sig.openingCorrect ? WIN_PAYOUT : (sig.isPushOpening ? 0 : -1);
      const resultClass = sig.openingCorrect ? 'highlight-green' : (sig.isPushOpening ? '' : 'highlight-red');
      const resultText = sig.openingCorrect ? 'HIT' : (sig.isPushOpening ? 'PUSH' : 'MISS');

      return `
        <tr>
          <td>${sig.date || ''}</td>
          <td>${sig.awayTeam} @ ${sig.homeTeam}</td>
          <td><span class="ou-direction-badge ${sig.direction.toLowerCase()}">${sig.direction}</span></td>
          <td><span class="ou-tier-badge ${sig.tier}">${sig.tier}</span></td>
          <td>${sig.ouLine}</td>
          <td>${sig.q3CumulTotal}</td>
          <td>${sig.predictedFinal}</td>
          <td>${sig.finalTotal}</td>
          <td>${sig.openingMargin}</td>
          <td class="${resultClass}">${resultText}</td>
          <td class="${pnlVal >= 0 ? 'highlight-green' : 'highlight-red'}">${pnlVal >= 0 ? '+' : ''}${pnlVal.toFixed(2)}</td>
        </tr>`;
    }).join('');
  }

  function renderEquityCurve() {
    if (!window.HistoricalData || !window.HistoricalData.equityCurve) return;

    const curve = window.HistoricalData.equityCurve;
    const data = curve.map((pt, i) => ({ x: i + 1, y: pt.cumPnl }));

    Charts.renderEquityCurve('equity-chart', { combined: data }, 'combined');
  }

  // =========================================================================
  // BACKTEST VIEW
  // =========================================================================
  function renderBacktest() {
    if (!window.HistoricalData || !window.HistoricalData.summary) return;

    const summary = window.HistoricalData.summary;

    // Update tier cards from real data
    for (const tier of ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']) {
      const tierData = summary.byTier[tier];
      if (!tierData) continue;

      const prefix = 'bt-' + tier.toLowerCase();
      const accEl = document.getElementById(`${prefix}-acc`);
      const nEl = document.getElementById(`${prefix}-n`);
      const wlEl = document.getElementById(`${prefix}-wl`);
      const roiEl = document.getElementById(`${prefix}-roi`);
      const pnlEl = document.getElementById(`${prefix}-pnl`);

      if (accEl) accEl.textContent = `${(tierData.accuracy * 100).toFixed(1)}%`;
      if (nEl) nEl.textContent = tierData.signals;
      if (wlEl) wlEl.textContent = `${tierData.wins}-${tierData.losses}`;
      if (roiEl) roiEl.textContent = `+${tierData.roi}%`;
      if (pnlEl) pnlEl.textContent = `+${tierData.pnl.toFixed(1)}u`;
    }

    // Render season breakdown table
    const tbody = document.getElementById('backtest-breakdown-body');
    if (tbody) {
      const signals = window.HistoricalData.signals || [];
      const rows = [];

      for (const season of ['2021-22', '2022-23']) {
        for (const tier of ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']) {
          const tierSignals = signals.filter(s => s.season === season && s.tier === tier);
          if (tierSignals.length === 0) continue;

          const wins = tierSignals.filter(s => s.openingCorrect).length;
          const losses = tierSignals.filter(s => !s.openingCorrect && !s.isPushOpening).length;
          const acc = wins / tierSignals.length;
          const pnl = wins * WIN_PAYOUT - losses;
          const roi = pnl / tierSignals.length * 100;

          rows.push(`
            <tr>
              <td>${season}</td>
              <td><span class="ou-tier-badge ${tier}">${tier}</span></td>
              <td>${tierSignals.length}</td>
              <td>${wins}</td>
              <td class="${acc >= 0.95 ? 'highlight-gold' : 'highlight-green'}">${(acc * 100).toFixed(1)}%</td>
              <td class="highlight-green">+${pnl.toFixed(1)}u</td>
              <td class="highlight-green">+${roi.toFixed(1)}%</td>
            </tr>`);
        }
      }

      tbody.innerHTML = rows.join('');
    }
  }

  // =========================================================================
  // PRED FINAL vs ACTUAL FINAL DIRECTIONAL ACCURACY
  // =========================================================================
  function renderPredAccuracy(filtered) {
    const el = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };

    if (!filtered || filtered.length === 0) {
      el('pred-acc-directional', '--');
      el('pred-acc-record', '--');
      el('pred-mae', '--');
      el('pred-avg-diff', '--');
      const tbody = document.getElementById('pred-accuracy-breakdown-body');
      if (tbody) tbody.innerHTML = '';
      return;
    }

    // Helper: compute directional accuracy stats for a set of signals
    function computeStats(sigs) {
      const total = sigs.length;
      if (total === 0) return null;

      let correct = 0;
      let totalAbsError = 0;
      let totalDiff = 0;

      for (const s of sigs) {
        // Directional accuracy: did predictedFinal correctly call the
        // direction (OVER/UNDER) relative to the O/U line, matching
        // the actual final total's direction?
        if (s.openingCorrect) correct++;
        totalAbsError += Math.abs(s.predictedFinal - s.finalTotal);
        totalDiff += (s.predictedFinal - s.finalTotal);
      }

      return {
        total,
        correct,
        accuracy: correct / total,
        mae: totalAbsError / total,
        avgDiff: totalDiff / total,
        avgPred: sigs.reduce((sum, s) => sum + s.predictedFinal, 0) / total,
        avgActual: sigs.reduce((sum, s) => sum + s.finalTotal, 0) / total,
      };
    }

    // Overall stats for the current filtered set
    const overall = computeStats(filtered);

    // Populate summary cards
    el('pred-acc-directional', `${(overall.accuracy * 100).toFixed(1)}%`);
    el('pred-acc-record', `${overall.correct} / ${overall.total}`);
    el('pred-mae', `${overall.mae.toFixed(1)}`);
    const diff = overall.avgDiff;
    el('pred-avg-diff', `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}`);

    // Breakdown table by tier (within current filter)
    const tbody = document.getElementById('pred-accuracy-breakdown-body');
    if (!tbody) return;

    const rows = [];
    for (const tier of ['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']) {
      const tierSigs = filtered.filter(s => s.tier === tier);
      const stats = computeStats(tierSigs);
      if (!stats) continue;

      const accClass = stats.accuracy >= 0.95 ? 'highlight-gold' : (stats.accuracy >= 0.85 ? 'highlight-green' : '');

      rows.push(`
        <tr>
          <td><span class="ou-tier-badge ${tier}">${tier}</span></td>
          <td>${stats.total}</td>
          <td>${stats.correct}</td>
          <td class="${accClass}">${(stats.accuracy * 100).toFixed(1)}%</td>
          <td>${stats.avgPred.toFixed(1)}</td>
          <td>${stats.avgActual.toFixed(1)}</td>
          <td>${stats.mae.toFixed(1)}</td>
        </tr>`);
    }

    tbody.innerHTML = rows.join('');
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
