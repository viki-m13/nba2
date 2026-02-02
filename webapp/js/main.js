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
          renderHistoricalSignals();
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

          // Determine if signal was correct
          if (sig.signalType === 'SPREAD') {
            const pickedHome = sig.direction === 'HOME';
            sig.correct = pickedHome
              ? actualMargin > sig.liveSpread
              : actualMargin < sig.liveSpread;
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

    return sigs.map(s => {
      const betInfo = buildBetInstruction(s);
      return `
        <div class="signal-pick game-card-bet">
          <span class="signal-pick-label">${betInfo.market}</span>
          <span class="signal-pick-value">${betInfo.pick}</span>
          <span class="signal-pick-odds">${betInfo.oddsDisplay} | ${(s.confidence * 100).toFixed(0)}% conf</span>
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
    const matchup = `${s.awayTeam || s.away_team || ''} @ ${s.homeTeam || s.home_team || ''}`;

    const confidence = s.confidence || 0;
    const edge = s.edge || 0;
    const q3Lead = s.q3Lead || s.q3_lead || 0;
    const regime = s.regime || '';
    const odds = s.estimatedOdds || s.estimated_odds || -110;
    const correct = s.correct;
    const type = s.signalType || s.signal_type || '';
    const dir = s.direction;

    // Build explicit bet instruction
    const betInfo = buildBetInstruction(s);

    return `
      <div class="signal-card ${tierClass}">
        <div class="signal-header">
          <span class="signal-matchup">${matchup}</span>
          <div style="display:flex;gap:6px;align-items:center">
            <span class="tier-badge ${tierClass}">${(s.tier || 'WATCH').toUpperCase()}</span>
            <span class="signal-type-badge ${typeClass}">${type.replace('_', ' ')}</span>
          </div>
        </div>

        <div class="bet-slip">
          <div class="bet-slip-header">
            <span class="bet-slip-label">PLACE THIS BET</span>
            <span class="bet-slip-market">${betInfo.market}</span>
          </div>
          <div class="bet-slip-main">
            <div class="bet-slip-pick">${betInfo.pick}</div>
            <div class="bet-slip-odds">${betInfo.oddsDisplay}</div>
          </div>
          ${betInfo.lineDetail ? `<div class="bet-slip-detail">${betInfo.lineDetail}</div>` : ''}
          <div class="bet-slip-note">${betInfo.oddsNote}</div>
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
            <span class="signal-field-label">Q3 Score</span>
            <span class="signal-field-value">${q3Lead > 0 ? 'Home +' + q3Lead : q3Lead < 0 ? 'Away +' + Math.abs(q3Lead) : 'Tied'}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Model Margin</span>
            <span class="signal-field-value">${formatMargin(s.predictedMargin || s.predicted_margin)}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Live Spread Est.</span>
            <span class="signal-field-value">${formatMargin(s.liveSpread || s.live_spread)}</span>
          </div>
          <div class="signal-field">
            <span class="signal-field-label">Regime</span>
            <span class="signal-field-value">${regime}</span>
          </div>
        </div>
        ${!isLive && correct !== undefined ? `
          <div class="signal-result ${correct ? 'win' : 'loss'}">
            ${correct ? 'WIN' : 'LOSS'} | ${type === 'Q4_TOTAL'
              ? `Q4 total: ${(s.actual_q4 || s.actualQ4 || 0).toFixed(0)} pts (line: ${((s.live_q4_ou || s.liveQ4OU || 0) * 2 / 2).toFixed(1)})`
              : `Final margin: ${formatMargin(s.actual_margin || s.actualMargin)}`}
          </div>
        ` : ''}
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
      // The bet line: if betting HOME and home leads, they get a negative spread
      // liveSpread is home-relative (positive = home leading)
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
        oddsNote: '-110 is standard juice for spread bets at all major sportsbooks. If your book shows different juice (e.g. -115), the edge is slightly reduced.',
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
        oddsNote: `Estimated market odds: ${formatOdds(oddsNum)}. Actual odds vary by sportsbook. Take the best ML price available on ${team}.`,
      };
    }

    if (type === 'ML_TRAILER') {
      const team = s.team || (dir === 'HOME' ? homeTeam : awayTeam);
      const oddsNum = Math.round(odds);
      return {
        market: 'Live Moneyline (Underdog Value)',
        pick: `${team} to WIN (underdog)`,
        oddsDisplay: `+${Math.abs(oddsNum)}`,
        lineDetail: '',
        oddsNote: `Estimated odds: +${Math.abs(oddsNum)}. Shop for the best plus-odds price. Only bet if you can get plus odds (+). The value is in the payout, not win rate.`,
      };
    }

    if (type === 'Q4_TOTAL') {
      const q4OU = s.liveQ4OU || s.live_q4_ou || 0;
      const q4Line = Math.round(q4OU * 2) / 2;
      const predQ4 = s.predictedQ4 || s.predicted_q4 || 0;
      return {
        market: 'Live Q4 Total Points',
        pick: `Q4 ${dir} ${q4Line.toFixed(1)}`,
        oddsDisplay: '-110',
        lineDetail: `Model predicts ${predQ4.toFixed(1)} combined pts in Q4.`,
        oddsNote: '-110 is standard juice for totals bets at all major sportsbooks. If your book shows different juice (e.g. -115), the edge is slightly reduced.',
      };
    }

    return {
      market: type,
      pick: dir,
      oddsDisplay: formatOdds(odds),
      lineDetail: '',
      oddsNote: 'Check your sportsbook for current odds.',
    };
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
      const odds = s.estimated_odds || s.estimatedOdds || -110;

      // Build explicit bet description for historical table
      const betDesc = buildHistoricalBetDescription(s);

      return `<tr>
        <td>${s.away_team} @ ${s.home_team}</td>
        <td><span class="signal-type-badge ${typeClass}">${type.replace('_', ' ')}</span></td>
        <td class="bet-desc-cell">${betDesc}</td>
        <td>${conf}%</td>
        <td class="positive">${edge}%</td>
        <td>${formatOdds(odds)}</td>
        <td class="${s.correct ? 'positive' : 'negative'}">${s.correct ? 'WIN' : 'LOSS'}</td>
      </tr>`;
    }).join('');
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
      return `<strong>${team} ML</strong> <span class="bet-desc-sub">moneyline @ ${formatOdds(Math.round(odds))}</span>`;
    }

    if (type === 'ML_TRAILER') {
      const team = dir === 'HOME' ? homeTeam : awayTeam;
      return `<strong>${team} ML</strong> <span class="bet-desc-sub">underdog @ +${Math.abs(Math.round(odds))}</span>`;
    }

    if (type === 'Q4_TOTAL') {
      const q4OU = s.live_q4_ou || 0;
      const q4Line = Math.round(q4OU * 2) / 2;
      return `<strong>Q4 ${dir} ${q4Line.toFixed(1)}</strong> <span class="bet-desc-sub">total @ -110</span>`;
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
