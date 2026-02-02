// =============================================================================
// MAIN APP CONTROLLER - Q3 Terminal Prediction System
// =============================================================================

(function () {
  'use strict';

  // ---- State ----
  let currentView = 'dashboard';
  let games = [];
  let liveSignals = [];    // Signals generated this session
  let recentSignals = [];  // Signals from recent days (ESPN)
  let historicalSignals = []; // From backtest
  let modelLoaded = false;
  let pollInterval = null;
  let recentGamesLoaded = false;
  let currentTimePeriod = 'recent'; // 'recent', 'today', '3days', '7days', '14days', 'historical'
  const POLL_MS = 10000; // 10 seconds
  const processedGames = new Set(); // Track games we've already signaled on
  const recentProcessedGames = new Set(); // Track recent games we've processed

  // ---- Model parameters (embedded from Python export) ----
  const MODEL_URL = null; // Will embed inline below

  // =========================================================================
  // INITIALIZATION
  // =========================================================================

  async function init() {
    console.log('[Q3Terminal] Initializing...');

    setupNavigation();
    setupFilters();
    setupTimePeriodFilters();

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

    // Load recent games in the background (non-blocking)
    loadRecentGames();
  }

  // =========================================================================
  // MODEL LOADING
  // =========================================================================

  async function loadModel() {
    const paths = ['data/model.json', '../output/q3_terminal_v2_js_model.json'];
    for (const path of paths) {
      try {
        const resp = await fetch(path);
        if (resp.ok) {
          const params = await resp.json();
          Q3Engine.loadModel(params);
          modelLoaded = true;
          console.log(`[Q3Terminal] Model loaded from ${path}`);
          return;
        }
      } catch (e) {
        // Try next path
      }
    }
    console.warn('[Q3Terminal] Could not load model from any path');
    modelLoaded = false;
  }

  // =========================================================================
  // HISTORICAL SIGNALS
  // =========================================================================

  async function loadHistoricalSignals() {
    const paths = ['data/signals.json', '../output/q3_terminal_v2_signals.json'];
    for (const path of paths) {
      try {
        const resp = await fetch(path);
        if (resp.ok) {
          const data = await resp.json();
          historicalSignals = data.signals || [];
          console.log(`[Q3Terminal] Loaded ${historicalSignals.length} historical signals from ${path}`);
          renderSignalsTab();
          renderRegimeTable();
          return;
        }
      } catch (e) {
        // Try next path
      }
    }
    console.warn('[Q3Terminal] Could not load historical signals from any path');
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
        renderSignalsTab(btn.dataset.filter);
      });
    });
  }

  function setupTimePeriodFilters() {
    document.querySelectorAll('.time-period-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.time-period-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentTimePeriod = btn.dataset.period;
        // Get current signal type filter
        const activeFilter = document.querySelector('.filter-btn.active');
        const filter = activeFilter ? activeFilter.dataset.filter : 'all';
        renderSignalsTab(filter);
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

      // Check for Q3-end signals on live games
      for (const game of liveGames) {
        checkForSignal(game);
      }

      // Retroactively analyze finished games for signals
      await processFinishedGames(games);

      // Render active signals (includes both live and finished)
      renderActiveSignals();

    } catch (e) {
      console.error('[Q3Terminal] Poll error:', e);
    }
  }

  // =========================================================================
  // FINISHED GAME SIGNAL PROCESSING
  // =========================================================================

  async function processFinishedGames(allGames) {
    if (!modelLoaded) return;

    const finishedGames = allGames.filter(g => g.status === 'final' && !processedGames.has(g.id));
    if (finishedGames.length === 0) return;

    // Process up to 5 finished games per poll cycle
    const batch = finishedGames.slice(0, 5);

    // Mark as processed immediately to prevent duplicate processing from concurrent polls
    batch.forEach(g => processedGames.add(g.id));

    await Promise.all(batch.map(async (game) => {
      try {
        const possessions = await NbaApi.fetchPlayByPlay(game.id);

        if (possessions.length < 10) return;

        const signals = Q3Engine.generateSignals(game, possessions, {
          openingSpread: 0,
          openingOU: game.ouLine || 0,
        });

        if (signals.length === 0) return;

        // Determine Q3 end state for result calculation
        const q3Poss = possessions.filter(p => p.quarter <= 3);
        const q3End = q3Poss.length > 0 ? q3Poss[q3Poss.length - 1] : null;
        const q3Total = q3End ? q3End.homeScore + q3End.awayScore : 0;
        const finalTotal = game.homeScore + game.awayScore;
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

          // Q3 end scores
          if (q3End) {
            sig.q3HomeScore = q3End.homeScore;
            sig.q3AwayScore = q3End.awayScore;
          }

          // Determine if signal was correct
          if (sig.signalType === 'SPREAD') {
            const pickedHome = sig.direction === 'HOME';
            sig.correct = pickedHome
              ? actualMargin > sig.liveSpread
              : actualMargin < sig.liveSpread;
            // Cover margin
            if (pickedHome) {
              sig.cover_margin = actualMargin - sig.liveSpread;
            } else {
              sig.cover_margin = sig.liveSpread - actualMargin;
            }
          } else if (sig.signalType === 'ML_LEADER' || sig.signalType === 'ML_TRAILER') {
            const pickedHome = sig.direction === 'HOME';
            sig.correct = pickedHome ? actualMargin > 0 : actualMargin < 0;
          } else if (sig.signalType === 'Q4_TOTAL' && q3End) {
            const actualQ4 = finalTotal - q3Total;
            sig.actualQ4 = actualQ4;
            sig.correct = sig.direction === 'OVER'
              ? actualQ4 > sig.liveQ4OU
              : actualQ4 < sig.liveQ4OU;
          }

          liveSignals.push(sig);
        }

        console.log(`[Q3Terminal] Generated ${signals.length} signals for finished game ${game.awayTeam}@${game.homeTeam}`);
      } catch (e) {
        console.warn(`[Q3Terminal] Error processing finished game ${game.id}:`, e.message);
      }
    }));
  }

  // =========================================================================
  // RECENT GAMES LOADING (Past days from ESPN)
  // =========================================================================

  async function loadRecentGames() {
    if (!modelLoaded) {
      console.warn('[Q3Terminal] Model not loaded, skipping recent games');
      const loadingEl = document.getElementById('recent-loading');
      if (loadingEl) loadingEl.style.display = 'none';
      recentGamesLoaded = true;
      renderSignalsTab();
      return;
    }

    console.log('[Q3Terminal] Loading recent games...');
    const loadingEl = document.getElementById('recent-loading');
    if (loadingEl) loadingEl.style.display = '';

    const dateStrings = NbaApi.getRecentDateStrings(21); // Last 3 weeks

    // Process dates in batches of 3 to avoid overwhelming the API
    for (let i = 0; i < dateStrings.length; i += 3) {
      const batch = dateStrings.slice(i, i + 3);
      const results = await Promise.all(batch.map(dateStr => processDateForSignals(dateStr)));

      // Update count display after each batch
      const countEl = document.getElementById('recent-signals-count');
      if (countEl) {
        countEl.textContent = `${recentSignals.length} signals from ${i + batch.length} days`;
      }
    }

    recentGamesLoaded = true;
    console.log(`[Q3Terminal] Loaded ${recentSignals.length} recent signals`);

    if (loadingEl) loadingEl.style.display = 'none';

    // Re-render the signals tab
    const activeFilter = document.querySelector('.filter-btn.active');
    const filter = activeFilter ? activeFilter.dataset.filter : 'all';
    renderSignalsTab(filter);
  }

  async function processDateForSignals(dateStr) {
    try {
      const { games: dateGames, eventIds } = await NbaApi.fetchESPNScoreboardForDate(dateStr);
      const finishedGames = dateGames.filter(g => g.status === 'final');

      if (finishedGames.length === 0) return;

      // Process up to 8 games per date (avoid too many requests)
      const batch = finishedGames.slice(0, 8);

      // Fetch PBP for each finished game (2 at a time to be gentle on API)
      for (let j = 0; j < batch.length; j += 2) {
        const subBatch = batch.slice(j, j + 2);
        await Promise.all(subBatch.map(async (game) => {
          const gameKey = `${game.date}-${game.homeTeam}-${game.awayTeam}`;
          if (recentProcessedGames.has(gameKey)) return;
          recentProcessedGames.add(gameKey);

          try {
            const possessions = await NbaApi.fetchESPNPlayByPlay(game.espnId);
            if (possessions.length < 10) return;

            const signals = Q3Engine.generateSignals(game, possessions, {
              openingSpread: 0,
              openingOU: game.ouLine || 0,
            });

            if (signals.length === 0) return;

            // Determine Q3 end state
            const q3Poss = possessions.filter(p => p.quarter <= 3);
            const q3End = q3Poss.length > 0 ? q3Poss[q3Poss.length - 1] : null;
            const q3Total = q3End ? q3End.homeScore + q3End.awayScore : 0;
            const finalTotal = game.homeScore + game.awayScore;
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

              if (q3End) {
                sig.q3HomeScore = q3End.homeScore;
                sig.q3AwayScore = q3End.awayScore;
              }

              // Determine if signal was correct
              if (sig.signalType === 'SPREAD') {
                const pickedHome = sig.direction === 'HOME';
                sig.correct = pickedHome
                  ? actualMargin > sig.liveSpread
                  : actualMargin < sig.liveSpread;
                sig.cover_margin = pickedHome
                  ? actualMargin - sig.liveSpread
                  : sig.liveSpread - actualMargin;
              } else if (sig.signalType === 'ML_LEADER' || sig.signalType === 'ML_TRAILER') {
                const pickedHome = sig.direction === 'HOME';
                sig.correct = pickedHome ? actualMargin > 0 : actualMargin < 0;
              } else if (sig.signalType === 'Q4_TOTAL' && q3End) {
                const actualQ4 = finalTotal - q3Total;
                sig.actualQ4 = actualQ4;
                sig.correct = sig.direction === 'OVER'
                  ? actualQ4 > sig.liveQ4OU
                  : actualQ4 < sig.liveQ4OU;
              }

              recentSignals.push(sig);
            }
          } catch (e) {
            // Skip games that fail
          }
        }));
      }
    } catch (e) {
      console.warn(`[Q3Terminal] Error processing date ${dateStr}:`, e.message);
    }
  }

  // =========================================================================
  // SIGNAL DETECTION (Live Games)
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

      // Show notification with explicit bet instruction
      if (Notification.permission === 'granted') {
        const firstBet = buildBetInstruction(signals[0]);
        new Notification('Q3 Terminal - BET NOW', {
          body: `${firstBet.market}: ${firstBet.pick} @ ${firstBet.oddsDisplay}`,
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
      if (g.status === 'live') {
        statusText = `Q${g.quarter || '?'} ${g.gameClock || ''}`;
      }

      // Check if at end of Q3
      const atQ3End = g.quarter === 3 && g.gameClock === '0:00';
      if (atQ3End) statusText = 'Q3 END - ANALYZING';

      // Point differential
      const diff = (g.homeScore || 0) - (g.awayScore || 0);
      const diffStr = diff > 0 ? `${g.homeTeam} +${diff}` : diff < 0 ? `${g.awayTeam} +${Math.abs(diff)}` : 'Tied';

      return `
        <div class="game-card ${hasSignal ? 'has-signal' : ''}">
          <div class="game-status-bar">
            <span class="game-status ${atQ3End ? 'q3-end' : statusClass}">${statusText}</span>
            <div class="game-status-right">
              ${g.status === 'live' || g.status === 'final' ? `<span class="game-diff">${diffStr}</span>` : ''}
              ${hasSignal ? '<span class="tier-badge gold">SIGNAL</span>' : ''}
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
      const betInfo = buildBetInstruction(s);
      const type = s.signalType || '';
      const typeClass = type.toLowerCase().replace('_', '-');
      const resultTag = s.correct !== undefined
        ? `<span class="result-tag ${s.correct ? 'win' : 'loss'}">${s.correct ? 'W' : 'L'}</span>`
        : '';
      return `
        <div class="signal-pick game-card-bet">
          <span class="signal-type-badge ${typeClass}" style="font-size:8px;padding:2px 5px">${type.replace('_', ' ')}</span>
          <span class="signal-pick-value">${betInfo.pick}</span>
          <span class="signal-pick-odds">${betInfo.oddsDisplay} | ${(s.confidence * 100).toFixed(0)}%</span>
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
          <h3>No signals yet</h3>
          <p>Signals are generated at the end of Q3 for live games and retroactively analyzed for today's finished games.</p>
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

    // For finished games, pass isLive=false so WIN/LOSS results are shown
    container.innerHTML = liveSignals.map(s => renderSignalCard(s, !s.isFinished)).join('');

    // Also render in live view
    const liveContainer = document.getElementById('live-signals-container');
    if (liveContainer) {
      liveContainer.innerHTML = liveSignals.map(s => renderSignalCard(s, !s.isFinished)).join('');
    }

    // Show live signals card if we have any signals
    const card = document.getElementById('live-signals-card');
    if (card && liveSignals.length > 0) card.style.display = '';
  }

  function renderSignalCard(s, isLive = false) {
    const tierClass = (s.tier || 'watch').toLowerCase();
    const typeClass = (s.signalType || s.signal_type || '').toLowerCase().replace('_', '-');
    const homeTeam = s.homeTeam || s.home_team || '';
    const awayTeam = s.awayTeam || s.away_team || '';
    const matchup = `${awayTeam} @ ${homeTeam}`;

    const confidence = s.confidence || 0;
    const edge = s.edge || 0;
    const q3Lead = s.q3Lead || s.q3_lead || 0;
    const regime = s.regime || '';
    const odds = s.estimatedOdds || s.estimated_odds || -110;
    const correct = s.correct;
    const type = s.signalType || s.signal_type || '';
    const dir = s.direction;

    // Game scores
    const homeScore = s.homeScore || s.final_home || 0;
    const awayScore = s.awayScore || s.final_away || 0;
    const finalMargin = s.actualMargin || s.actual_margin || s.final_margin || (homeScore - awayScore);
    const gameDate = s.date ? formatGameDate(s.date) : '';
    const isFinished = s.isFinished || s.final_home;
    const quarter = s.quarter || (isFinished ? 'FINAL' : 'Q4');

    // Q3 score estimation
    const q3Home = q3Lead > 0 ? Math.round((homeScore + awayScore) * 0.75 / 2 + q3Lead / 2) : Math.round((homeScore + awayScore) * 0.75 / 2 + q3Lead / 2);
    const q3Away = q3Home - q3Lead;
    const pointDiff = Math.abs(q3Lead);

    // Build explicit bet instruction
    const betInfo = buildBetInstruction(s);

    // Determine leader/trailer at Q3
    const q3Leader = q3Lead > 0 ? homeTeam : (q3Lead < 0 ? awayTeam : 'TIE');
    const q3Trailer = q3Lead > 0 ? awayTeam : (q3Lead < 0 ? homeTeam : 'TIE');

    // Build result detail for finished games
    let resultDetail = '';
    if (!isLive && correct !== undefined) {
      if (type === 'Q4_TOTAL') {
        const actualQ4 = s.actual_q4 || s.actualQ4 || 0;
        const line = s.live_q4_ou || s.liveQ4OU || 0;
        const q4Diff = (actualQ4 - line).toFixed(1);
        resultDetail = `
          <div class="signal-result-detail ${correct ? 'win' : 'loss'}">
            <div class="result-badge">${correct ? 'WIN' : 'LOSS'}</div>
            <div class="result-stats">
              <span>Q4 Actual: <strong>${actualQ4.toFixed(0)} pts</strong></span>
              <span>Line: ${line.toFixed(1)}</span>
              <span>Margin: <strong>${q4Diff > 0 ? '+' : ''}${q4Diff}</strong></span>
              <span>Final: ${awayTeam} ${awayScore} - ${homeTeam} ${homeScore}</span>
            </div>
          </div>`;
      } else {
        const spreadLine = s.liveSpread || s.live_spread || 0;
        const coverMargin = s.cover_margin !== undefined ? s.cover_margin : 0;
        const winner = finalMargin > 0 ? homeTeam : awayTeam;
        const winMargin = Math.abs(finalMargin);
        resultDetail = `
          <div class="signal-result-detail ${correct ? 'win' : 'loss'}">
            <div class="result-badge">${correct ? 'WIN' : 'LOSS'}</div>
            <div class="result-stats">
              <span>Final: <strong>${awayTeam} ${awayScore} - ${homeTeam} ${homeScore}</strong></span>
              <span>${winner} wins by ${winMargin}</span>
              ${type === 'SPREAD' ? `<span>Cover: ${coverMargin > 0 ? '+' : ''}${coverMargin.toFixed(1)} pts</span>` : ''}
              ${type === 'ML_LEADER' || type === 'ML_TRAILER' ? `<span>Picked: ${dir === 'HOME' ? homeTeam : awayTeam} ${correct ? '(correct)' : '(wrong)'}</span>` : ''}
            </div>
          </div>`;
      }
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
            <span class="tier-badge ${tierClass}">${(s.tier || 'WATCH').toUpperCase()}</span>
            <span class="signal-type-badge ${typeClass}">${type.replace('_', ' ')}</span>
          </div>
        </div>

        <div class="signal-scoreboard">
          <div class="signal-team-col">
            <span class="signal-team-abbr ${q3Lead < 0 ? 'leading' : ''}">${awayTeam}</span>
            <span class="signal-team-role">Away</span>
          </div>
          <div class="signal-score-col">
            <div class="signal-score-main">${isFinished ? `${awayScore} - ${homeScore}` : `Q3: ${q3Away} - ${q3Home}`}</div>
            <div class="signal-score-sub">${isFinished ? `Q3 End: ${q3Away} - ${q3Home}` : `Diff: ${pointDiff} pts`}</div>
          </div>
          <div class="signal-team-col">
            <span class="signal-team-abbr ${q3Lead > 0 ? 'leading' : ''}">${homeTeam}</span>
            <span class="signal-team-role">Home</span>
          </div>
        </div>

        <div class="bet-slip">
          <div class="bet-slip-header">
            <span class="bet-slip-label">BET</span>
            <span class="bet-slip-market">${betInfo.market}</span>
          </div>
          <div class="bet-slip-main">
            <div class="bet-slip-pick">${betInfo.pick}</div>
            <div class="bet-slip-odds">${betInfo.oddsDisplay}</div>
          </div>
          ${betInfo.lineDetail ? `<div class="bet-slip-detail">${betInfo.lineDetail}</div>` : ''}
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
            <span class="signal-field-value">${q3Leader} +${pointDiff}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Model Margin</span>
            <span class="signal-field-value">${formatMargin(s.predictedMargin || s.predicted_margin)}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Mkt Spread</span>
            <span class="signal-field-value">${formatMargin(s.liveSpread || s.live_spread)}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Regime</span>
            <span class="signal-field-value regime-${regime.toLowerCase()}">${regime}</span>
          </div>
        </div>
        ${resultDetail}
      </div>`;
  }

  /**
   * Build explicit bet instruction from signal data.
   * Returns { market, pick, oddsDisplay, lineDetail, oddsNote }
   */
  function buildBetInstruction(s) {
    const type = s.signalType || s.signal_type || '';
    const dir = s.direction || '';
    const homeTeam = s.homeTeam || s.home_team || 'HOME';
    const awayTeam = s.awayTeam || s.away_team || 'AWAY';
    const odds = s.estimatedOdds || s.estimated_odds || -110;

    if (type === 'SPREAD') {
      const team = dir === 'HOME' ? homeTeam : awayTeam;
      const liveSpread = s.liveSpread || s.live_spread || 0;
      let betLineVal;
      if (dir === 'HOME') {
        betLineVal = -(Math.round(Math.abs(liveSpread) * 2) / 2);
      } else {
        betLineVal = Math.round(Math.abs(liveSpread) * 2) / 2;
      }
      const betLineStr = betLineVal > 0 ? `+${betLineVal.toFixed(1)}` : betLineVal.toFixed(1);

      return {
        market: 'Live Point Spread',
        pick: `${team} ${betLineStr}`,
        oddsDisplay: '-110',
        lineDetail: '',
        oddsNote: '',
      };
    }

    if (type === 'ML_LEADER') {
      const team = s.team || (dir === 'HOME' ? homeTeam : awayTeam);
      const oddsNum = Math.round(odds);
      return {
        market: 'Live Moneyline',
        pick: `${team} to WIN`,
        oddsDisplay: formatOdds(oddsNum),
        lineDetail: '',
        oddsNote: '',
      };
    }

    if (type === 'ML_TRAILER') {
      const team = s.team || (dir === 'HOME' ? homeTeam : awayTeam);
      const oddsNum = Math.round(odds);
      return {
        market: 'Live Moneyline (Underdog)',
        pick: `${team} to WIN`,
        oddsDisplay: `+${Math.abs(oddsNum)}`,
        lineDetail: '',
        oddsNote: '',
      };
    }

    if (type === 'Q4_TOTAL') {
      const q4OU = s.liveQ4OU || s.live_q4_ou || 0;
      const q4Line = Math.round(q4OU * 2) / 2;
      const predQ4 = s.predictedQ4 || s.predicted_q4 || 0;
      return {
        market: 'Live Q4 Total',
        pick: `Q4 ${dir} ${q4Line.toFixed(1)}`,
        oddsDisplay: '-110',
        lineDetail: `Model: ${predQ4.toFixed(1)} pts`,
        oddsNote: '',
      };
    }

    return {
      market: type,
      pick: dir,
      oddsDisplay: formatOdds(odds),
      lineDetail: '',
      oddsNote: '',
    };
  }

  // =========================================================================
  // RENDERING - SIGNALS TAB (Recent + Historical)
  // =========================================================================

  function renderSignalsTab(filter = 'all') {
    const tbody = document.getElementById('signals-tbody');
    const summary = document.getElementById('filter-summary');
    const tableTitle = document.getElementById('signals-table-title');
    const tableBadge = document.getElementById('signals-table-badge');
    if (!tbody) return;

    // Build combined signal list based on time period
    let signals = getSignalsForTimePeriod(currentTimePeriod, filter);

    // Sort by date descending (most recent first)
    signals = signals.slice().sort((a, b) => {
      const da = a.date || '0';
      const db = b.date || '0';
      return db.localeCompare(da);
    });

    // Update summary stats
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

    // Update table title based on period
    if (tableTitle) {
      const titles = {
        'recent': 'Recent Signals — Last 3 Weeks',
        'today': 'Today\'s Signals',
        '3days': 'Signals — Last 3 Days',
        '7days': 'Signals — Last 7 Days',
        '14days': 'Signals — Last 2 Weeks',
        'historical': 'Trade Log — Historical OOS Signals (2022-23)',
      };
      tableTitle.textContent = titles[currentTimePeriod] || 'Signals';
    }

    if (tableBadge) {
      if (currentTimePeriod === 'historical') {
        tableBadge.textContent = 'Walk-Forward Validated';
      } else {
        tableBadge.textContent = `${signals.length} signals`;
      }
    }

    // Show max 500 rows
    const display = signals.slice(0, 500);

    if (display.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:40px;color:var(--text-muted)">
        ${currentTimePeriod !== 'historical' && !recentGamesLoaded
          ? '<span class="loading-spinner"></span> Loading recent games...'
          : 'No signals found for this time period'}
      </td></tr>`;
      return;
    }

    tbody.innerHTML = display.map(s => renderSignalRow(s)).join('');
  }

  function getSignalsForTimePeriod(period, filter) {
    const now = new Date();
    const todayStr = formatDateStr(now);

    let signals = [];

    if (period === 'historical') {
      // Only backtest historical signals
      signals = historicalSignals.map(normalizeSignal);
    } else {
      // Combine live (today) + recent signals
      const allRecent = [
        ...liveSignals.map(s => ({ ...s, isRecent: true, date: s.date || todayStr })),
        ...recentSignals,
      ];

      // Deduplicate by game + signal type
      const seen = new Set();
      const deduped = [];
      for (const s of allRecent) {
        const key = `${s.date || ''}-${s.homeTeam || s.home_team || ''}-${s.awayTeam || s.away_team || ''}-${s.signalType || s.signal_type || ''}`;
        if (!seen.has(key)) {
          seen.add(key);
          deduped.push(s);
        }
      }

      if (period === 'today') {
        signals = deduped.filter(s => (s.date || '') === todayStr);
      } else if (period === '3days') {
        const cutoff = daysAgoStr(3);
        signals = deduped.filter(s => (s.date || '') >= cutoff);
      } else if (period === '7days') {
        const cutoff = daysAgoStr(7);
        signals = deduped.filter(s => (s.date || '') >= cutoff);
      } else if (period === '14days') {
        const cutoff = daysAgoStr(14);
        signals = deduped.filter(s => (s.date || '') >= cutoff);
      } else {
        // 'recent' = all recent + today
        signals = deduped;
      }

      signals = signals.map(normalizeSignal);
    }

    // Apply signal type filter
    if (filter && filter !== 'all') {
      signals = signals.filter(s => (s.signal_type || s.signalType) === filter);
    }

    return signals;
  }

  /** Normalize signal objects to a consistent format for table rendering */
  function normalizeSignal(s) {
    return {
      date: s.date || '',
      signal_type: s.signal_type || s.signalType || '',
      direction: s.direction || '',
      confidence: s.confidence || 0,
      edge: s.edge || 0,
      q3_lead: s.q3_lead ?? s.q3Lead ?? 0,
      estimated_odds: s.estimated_odds || s.estimatedOdds || -110,
      home_team: s.home_team || s.homeTeam || '',
      away_team: s.away_team || s.awayTeam || '',
      final_home: s.final_home || s.homeScore || 0,
      final_away: s.final_away || s.awayScore || 0,
      final_margin: s.final_margin || s.actualMargin || ((s.final_home || s.homeScore || 0) - (s.final_away || s.awayScore || 0)),
      correct: s.correct,
      cover_margin: s.cover_margin,
      actual_q4: s.actual_q4 || s.actualQ4 || 0,
      live_q4_ou: s.live_q4_ou || s.liveQ4OU || 0,
      live_spread: s.live_spread || s.liveSpread || 0,
      regime: s.regime || '',
      predicted_margin: s.predicted_margin || s.predictedMargin || 0,
      isRecent: s.isRecent || false,
    };
  }

  function renderSignalRow(s) {
    const type = s.signal_type || '';
    const typeClass = type.toLowerCase().replace('_', '-');
    const conf = (s.confidence * 100).toFixed(0);
    const edge = (s.edge * 100).toFixed(1);
    const q3Lead = s.q3_lead || 0;
    const odds = s.estimated_odds || -110;
    const homeTeam = s.home_team || '';
    const awayTeam = s.away_team || '';

    const dateStr = s.date ? formatGameDate(s.date) : '';
    const finalHome = s.final_home || 0;
    const finalAway = s.final_away || 0;
    const finalMargin = s.final_margin || (finalHome - finalAway);
    const winner = finalMargin > 0 ? homeTeam : awayTeam;
    const winMargin = Math.abs(finalMargin);

    const q3Leader = q3Lead > 0 ? homeTeam : (q3Lead < 0 ? awayTeam : 'TIE');
    const q3Diff = Math.abs(q3Lead);

    const betDesc = buildHistoricalBetDescription(s);

    let resultNote = '';
    if (type === 'Q4_TOTAL') {
      const actualQ4 = s.actual_q4 || 0;
      const line = s.live_q4_ou || 0;
      resultNote = actualQ4 ? `Q4: ${actualQ4.toFixed ? actualQ4.toFixed(0) : actualQ4} (line ${line.toFixed ? line.toFixed(1) : line})` : '';
    } else if (type === 'SPREAD') {
      const cover = s.cover_margin;
      resultNote = cover !== undefined ? `Cover: ${cover > 0 ? '+' : ''}${cover.toFixed ? cover.toFixed(1) : cover}` : '';
    } else {
      resultNote = finalHome || finalAway ? `${winner} by ${winMargin}` : '';
    }

    let unitPnl;
    if (s.correct !== undefined) {
      if (s.correct) {
        unitPnl = odds < 0 ? 100 / Math.abs(odds) : odds / 100;
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

    return `<tr class="${s.correct === false ? 'loss-row' : ''}">
      <td class="date-cell">${dateStr}${recentBadge}</td>
      <td class="matchup-cell">
        <div>${awayTeam} @ ${homeTeam}</div>
        <div class="score-sub">${finalAway}-${finalHome}</div>
      </td>
      <td><span class="signal-type-badge ${typeClass}">${type.replace('_', ' ')}</span></td>
      <td class="bet-desc-cell">${betDesc}</td>
      <td class="q3-cell">
        <div>${q3Leader} +${q3Diff}</div>
        <div class="regime-sub">${s.regime || ''}</div>
      </td>
      <td>${conf}%</td>
      <td class="positive">${edge}%</td>
      ${resultCell}
      ${pnlCell}
    </tr>`;
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

  /** Build a short explicit bet description for historical table rows */
  function buildHistoricalBetDescription(s) {
    const type = s.signal_type || '';
    const dir = s.direction || '';
    const homeTeam = s.home_team || '';
    const awayTeam = s.away_team || '';
    const liveSpread = s.live_spread || 0;
    const odds = s.estimated_odds || -110;

    if (type === 'SPREAD') {
      const team = dir === 'HOME' ? homeTeam : awayTeam;
      let betLine;
      if (dir === 'HOME') {
        betLine = -(Math.round(Math.abs(liveSpread) * 2) / 2);
      } else {
        betLine = Math.round(Math.abs(liveSpread) * 2) / 2;
      }
      const lineStr = betLine > 0 ? `+${betLine.toFixed(1)}` : betLine.toFixed(1);
      return `<strong>${team} ${lineStr}</strong> <span class="bet-desc-sub">spread @ -110</span>`;
    }

    if (type === 'ML_LEADER') {
      const team = dir === 'HOME' ? homeTeam : awayTeam;
      return `<strong>${team} ML</strong> <span class="bet-desc-sub">@ ${formatOdds(Math.round(odds))}</span>`;
    }

    if (type === 'ML_TRAILER') {
      const team = dir === 'HOME' ? homeTeam : awayTeam;
      return `<strong>${team} ML</strong> <span class="bet-desc-sub">dog @ +${Math.abs(Math.round(odds))}</span>`;
    }

    if (type === 'Q4_TOTAL') {
      const q4OU = s.live_q4_ou || 0;
      const q4Line = Math.round(q4OU * 2) / 2;
      return `<strong>Q4 ${dir} ${q4Line.toFixed(1)}</strong> <span class="bet-desc-sub">@ -110</span>`;
    }

    return dir;
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

  function formatGameDate(dateStr) {
    if (!dateStr || dateStr.length < 8) return dateStr || '';
    const y = dateStr.substring(0, 4);
    const m = dateStr.substring(4, 6);
    const d = dateStr.substring(6, 8);
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const monthIdx = parseInt(m, 10) - 1;
    const formatted = `${months[monthIdx]} ${parseInt(d, 10)}`;

    // Add relative label for recent dates
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
