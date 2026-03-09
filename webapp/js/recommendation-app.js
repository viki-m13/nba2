/**
 * NBA Betting Recommendation App Controller
 * ==========================================
 * Connects the RecommendationEngine to the webapp UI.
 * Handles data loading, backtest execution, today's picks generation,
 * and history display with daily updates.
 */

(function () {
  'use strict';

  const ENGINE = window.RecommendationEngine;
  const ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8';
  const ODDS_API_BASE = 'https://api.the-odds-api.com/v4';

  const TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GS', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NO',
    'New York Knicks': 'NY', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SA',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTAH',
    'Washington Wizards': 'WSH',
  };

  function teamAbbr(name) { return TEAM_MAP[name] || name; }

  let playerBoxScores = [];
  let historicalOdds = [];
  let backtestResults = null;
  let allSignals = [];
  let activeFilter = 'all';

  // ========================================================================
  // INITIALIZATION
  // ========================================================================

  async function init() {
    setupNav();
    setupFilters();
    await loadData();
    runBacktestAndDisplay();
    await generateTonightPicks();
  }

  // ========================================================================
  // DATA LOADING
  // ========================================================================

  async function loadData() {
    try {
      const [boxRes, oddsRes, signalsRes] = await Promise.all([
        fetch('data/player_boxscores.json'),
        fetch('data/historical_odds.json'),
        fetch('data/recommendation_signals.json').catch(() => null),
      ]);

      playerBoxScores = await boxRes.json();
      historicalOdds = await oddsRes.json();

      if (signalsRes && signalsRes.ok) {
        allSignals = await signalsRes.json();
      }

      // Also try loading 2026 data
      try {
        const [box2026Res, odds2026Res] = await Promise.all([
          fetch('data/player_boxscores_2026.json').catch(() => null),
          fetch('data/historical_odds_2026.json').catch(() => null),
        ]);
        if (box2026Res && box2026Res.ok) {
          const box2026 = await box2026Res.json();
          playerBoxScores = [...playerBoxScores, ...box2026];
        }
        if (odds2026Res && odds2026Res.ok) {
          const odds2026 = await odds2026Res.json();
          historicalOdds = [...historicalOdds, ...odds2026];
        }
      } catch (e) { /* optional data */ }

      console.log(`[REC-APP] Loaded: ${playerBoxScores.length} box scores, ${historicalOdds.length} odds records`);
    } catch (e) {
      console.error('[REC-APP] Error loading data:', e);
    }
  }

  // ========================================================================
  // BACKTEST
  // ========================================================================

  function runBacktestAndDisplay() {
    if (!playerBoxScores.length || !historicalOdds.length) return;

    console.log('[REC-APP] Running walk-forward backtest...');
    backtestResults = ENGINE.runBacktest(playerBoxScores, historicalOdds);

    if (backtestResults) {
      console.log('[REC-APP] Backtest results:', backtestResults.stats);
      allSignals = backtestResults.signals;
    }

    renderDashboard();
    renderTierCards();
    renderHistory();
  }

  // ========================================================================
  // TONIGHT'S GAMES & PICKS
  // ========================================================================

  async function generateTonightPicks() {
    const gamesStatus = document.getElementById('tonight-games-status');
    const picksStatus = document.getElementById('tonight-picks-status');

    try {
      // Fetch today's games from Odds API
      gamesStatus.textContent = 'Fetching tonight\'s games...';
      let events = [];
      let liveOdds = { events: [], playerProps: {} };

      try {
        const eventsUrl = `${ODDS_API_BASE}/sports/basketball_nba/events?apiKey=${ODDS_API_KEY}`;
        const eventsRes = await fetch(eventsUrl);
        if (eventsRes.ok) {
          events = await eventsRes.json();
          liveOdds.events = events;
        }
      } catch (e) {
        console.warn('[REC-APP] Could not fetch events:', e);
      }

      // Render games
      renderTonightGames(events);

      if (events.length === 0) {
        gamesStatus.textContent = 'No games scheduled tonight.';
        picksStatus.textContent = 'No games available for analysis.';
        return;
      }
      gamesStatus.style.display = 'none';

      // Fetch player props for each game
      picksStatus.textContent = `Analyzing ${events.length} games...`;

      for (const event of events) {
        try {
          const markets = 'player_points_alternate,player_rebounds_alternate,player_assists_alternate';
          const propsUrl = `${ODDS_API_BASE}/sports/basketball_nba/events/${event.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=${markets}&oddsFormat=american&bookmakers=fanduel`;
          const propsRes = await fetch(propsUrl);
          if (!propsRes.ok) continue;

          const propsData = await propsRes.json();
          const homeAbbr = teamAbbr(event.home_team);
          const awayAbbr = teamAbbr(event.away_team);
          const gameKey = `${awayAbbr}@${homeAbbr}`;

          const fd = propsData.bookmakers && propsData.bookmakers.find(b => b.key === 'fanduel');
          if (!fd) continue;

          const gameProps = { lines: {}, rebLines: {}, astLines: {} };
          const marketMap = {
            'player_points_alternate': 'lines',
            'player_rebounds_alternate': 'rebLines',
            'player_assists_alternate': 'astLines',
          };

          for (const mkt of (fd.markets || [])) {
            const st = marketMap[mkt.key];
            if (!st) continue;
            for (const outcome of (mkt.outcomes || [])) {
              const player = outcome.description;
              const threshold = outcome.point;
              if (!gameProps[st][player]) gameProps[st][player] = {};
              if (!gameProps[st][player][threshold]) gameProps[st][player][threshold] = {};
              if (outcome.name === 'Over') {
                gameProps[st][player][threshold].overOdds = outcome.price;
              }
            }
          }

          liveOdds.playerProps[gameKey] = gameProps;
        } catch (e) {
          console.warn('[REC-APP] Error fetching props:', e);
        }
      }

      // Generate recommendations
      if (Object.keys(liveOdds.playerProps).length === 0) {
        picksStatus.textContent = 'No FanDuel player props available. Check back closer to game time.';
        return;
      }

      const recommendation = ENGINE.generateTodayPicks(liveOdds);
      if (!recommendation || (recommendation.singles.length === 0 && recommendation.parlays.length === 0)) {
        picksStatus.textContent = 'No bets meet our strict quality thresholds tonight. The engine only recommends when confluence is high.';
        return;
      }

      picksStatus.style.display = 'none';
      renderTonightPicks(recommendation);

      // Save signals for history
      const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      const newSignals = ENGINE.formatSignalForStorage(recommendation, today);
      // Merge with existing (avoid duplicates for today)
      allSignals = allSignals.filter(s => s.date !== today);
      allSignals.push(...newSignals);

    } catch (e) {
      console.error('[REC-APP] Error generating tonight picks:', e);
      picksStatus.textContent = 'Error loading picks. See console for details.';
    }
  }

  // ========================================================================
  // RENDER: Tonight's Games
  // ========================================================================

  function renderTonightGames(events) {
    const container = document.getElementById('tonight-games');
    if (!container) return;

    const dateStr = new Date().toLocaleDateString('en-US', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
    });

    if (!events || events.length === 0) {
      container.innerHTML = '';
      return;
    }

    const gamesHtml = events.map(e => {
      const away = teamAbbr(e.away_team);
      const home = teamAbbr(e.home_team);
      let timeStr = 'TBD';
      if (e.commence_time) {
        try {
          const d = new Date(e.commence_time);
          timeStr = d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
        } catch (err) { timeStr = 'TBD'; }
      }

      return `
        <div class="game-card">
          <div class="game-teams">
            <span class="game-away">${away}</span>
            <span class="game-at">@</span>
            <span class="game-home">${home}</span>
          </div>
          <div class="game-time">${timeStr}</div>
          <div class="game-date">${dateStr}</div>
        </div>
      `;
    }).join('');

    container.innerHTML = gamesHtml;
  }

  // ========================================================================
  // RENDER: Tonight's Picks
  // ========================================================================

  function renderTonightPicks(recommendation) {
    const betTypeBadge = document.getElementById('tonight-bet-type');
    const singlesContainer = document.getElementById('tonight-singles');
    const parlaysContainer = document.getElementById('tonight-parlays');

    // Show bet type badge
    const typeLabels = {
      'single': 'SINGLE BET',
      'multi_single': 'MULTIPLE SINGLES',
      'parlay': 'PARLAY',
    };
    const typeColors = {
      'single': '#22c55e',
      'multi_single': '#3b82f6',
      'parlay': '#a78bfa',
    };

    betTypeBadge.style.display = 'flex';
    betTypeBadge.innerHTML = `
      <span class="bet-type-label" style="background: ${typeColors[recommendation.betType]}20; color: ${typeColors[recommendation.betType]}; border: 1px solid ${typeColors[recommendation.betType]}40;">
        ${typeLabels[recommendation.betType] || 'RECOMMENDATION'}
      </span>
      <span class="bet-type-reason">${recommendation.reasoning}</span>
    `;

    // Render singles
    if (recommendation.singles.length > 0) {
      singlesContainer.style.display = 'grid';
      singlesContainer.innerHTML = recommendation.singles.map(s => renderSingleCard(s)).join('');
    }

    // Render parlays
    if (recommendation.parlays.length > 0) {
      parlaysContainer.style.display = 'grid';
      parlaysContainer.innerHTML = recommendation.parlays.map(p => renderParlayCard(p)).join('');
    }
  }

  function renderSingleCard(pick) {
    const oddsStr = ENGINE.formatOdds(pick.odds);
    const resultClass = pick.hit === true ? 'result-win' : pick.hit === false ? 'result-loss' : 'result-pending';
    const resultText = pick.hit === true ? 'WIN' : pick.hit === false ? 'LOSS' : 'PENDING';

    return `
      <div class="pick-card single-pick">
        <div class="pick-header">
          <span class="pick-type single-type">SINGLE</span>
          <span class="pick-odds">${oddsStr}</span>
        </div>
        <div class="pick-player">${pick.player}</div>
        <div class="pick-line">Over ${pick.line} ${pick.statLabel}</div>
        <div class="pick-stats">
          <div class="pick-stat">
            <div class="pick-stat-label">CCS</div>
            <div class="pick-stat-value">${(pick.cascadeScore * 100).toFixed(0)}%</div>
          </div>
          <div class="pick-stat">
            <div class="pick-stat-label">Hit Rate</div>
            <div class="pick-stat-value">${(pick.hitRate * 100).toFixed(0)}%</div>
          </div>
          <div class="pick-stat">
            <div class="pick-stat-label">Edge</div>
            <div class="pick-stat-value edge-value">${(pick.edge * 100).toFixed(1)}%</div>
          </div>
        </div>
        <div class="pick-innovations">
          <span class="innovation-chip" title="Temporal Convergence Zone Detection">TCZD: ${(pick.tczd * 100).toFixed(0)}%</span>
          <span class="innovation-chip" title="Volatility-Adjusted Confidence">VACS: ${(pick.vacs * 100).toFixed(0)}%</span>
          <span class="innovation-chip" title="Margin-of-Safety Depth">MSDS: ${(pick.msds * 100).toFixed(0)}%</span>
          <span class="innovation-chip" title="Regime-Aware Filter">RAF: ${(pick.raf * 100).toFixed(0)}%</span>
        </div>
        <div class="pick-detail-row">
          <span>Avg: ${pick.avg.toFixed(1)} | Floor: ${pick.floor}</span>
          <span>EV: ${(pick.ev * 100).toFixed(1)}%</span>
        </div>
        <div class="pick-result">
          <span class="${resultClass}">${resultText}</span>
          ${pick.actual != null ? `<span class="pick-actual">Actual: ${pick.actual}</span>` : ''}
        </div>
      </div>
    `;
  }

  function renderParlayCard(parlay) {
    const oddsStr = ENGINE.formatOdds(parlay.odds);
    const parlayHit = parlay.legs.every(l => l.hit === true);
    const hasPending = parlay.legs.some(l => l.hit === null || l.hit === undefined);
    const resultClass = hasPending ? 'result-pending' : (parlayHit ? 'result-win' : 'result-loss');
    const resultText = hasPending ? 'PENDING' : (parlayHit ? 'WIN' : 'LOSS');

    const legsHtml = parlay.legs.map(l => {
      const legOdds = ENGINE.formatOdds(l.odds);
      const legResult = l.hit === true ? 'result-win' : l.hit === false ? 'result-loss' : '';
      const actualStr = l.actual != null ? ` (${l.actual})` : '';

      return `
        <div class="parlay-leg">
          <div class="parlay-leg-info">
            <span class="parlay-leg-player ${legResult}">${l.player}</span>
            <span class="parlay-leg-game">${l.team} | CCS: ${(l.cascadeScore * 100).toFixed(0)}%</span>
          </div>
          <div class="parlay-leg-right">
            <span class="parlay-leg-line">O${l.line} ${l.statLabel}${actualStr}</span>
            <span class="parlay-leg-odds">${legOdds}</span>
          </div>
        </div>
      `;
    }).join('');

    return `
      <div class="pick-card parlay-pick">
        <div class="pick-header">
          <span class="pick-type parlay-type">PARLAY ${parlay.numLegs}-LEG</span>
          <span class="pick-odds">${oddsStr}</span>
        </div>
        <div class="pick-stats">
          <div class="pick-stat">
            <div class="pick-stat-label">Combined Prob</div>
            <div class="pick-stat-value">${(parlay.combinedHitRate * 100).toFixed(1)}%</div>
          </div>
          <div class="pick-stat">
            <div class="pick-stat-label">Avg CCS</div>
            <div class="pick-stat-value">${(parlay.avgCascade * 100).toFixed(0)}%</div>
          </div>
          <div class="pick-stat">
            <div class="pick-stat-label">EV</div>
            <div class="pick-stat-value edge-value">${(parlay.ev * 100).toFixed(1)}%</div>
          </div>
        </div>
        <div class="parlay-legs">${legsHtml}</div>
        <div class="parlay-summary">
          <span class="${resultClass}">${resultText}</span>
          <span class="parlay-payout">$100 to win $${Math.round((parlay.decimalOdds - 1) * 100)}</span>
        </div>
      </div>
    `;
  }

  // ========================================================================
  // RENDER: Dashboard
  // ========================================================================

  function renderDashboard() {
    const container = document.getElementById('dashboard');
    if (!container || !backtestResults) return;

    const s = backtestResults.stats;
    const singles = s.singles;
    const parlays = s.parlays;
    const overall = s.overall;

    container.innerHTML = `
      <div class="stat-card highlight">
        <div class="stat-value">${(singles.accuracy * 100).toFixed(1)}%</div>
        <div class="stat-label">Singles Accuracy</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${singles.wins}-${singles.total - singles.wins}</div>
        <div class="stat-label">Singles Record</div>
      </div>
      <div class="stat-card" style="border-color: ${singles.roi >= 0 ? '#22c55e' : '#ef4444'};">
        <div class="stat-value" style="color: ${singles.roi >= 0 ? '#22c55e' : '#ef4444'};">${singles.roi >= 0 ? '+' : ''}${(singles.roi * 100).toFixed(1)}%</div>
        <div class="stat-label">Singles ROI</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${(parlays.accuracy * 100).toFixed(1)}%</div>
        <div class="stat-label">Parlay Win Rate</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${parlays.wins}-${parlays.total - parlays.wins}</div>
        <div class="stat-label">Parlay Record</div>
      </div>
      <div class="stat-card" style="border-color: ${overall.pnl >= 0 ? '#22c55e' : '#ef4444'};">
        <div class="stat-value" style="color: ${overall.pnl >= 0 ? '#22c55e' : '#ef4444'};">$${overall.pnl >= 0 ? '+' : ''}${overall.pnl}</div>
        <div class="stat-label">Total P&L ($100/bet)</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${(parlays.legAccuracy * 100).toFixed(1)}%</div>
        <div class="stat-label">Parlay Leg Rate</div>
      </div>
    `;
  }

  // ========================================================================
  // RENDER: Tier Cards
  // ========================================================================

  function renderTierCards() {
    const container = document.getElementById('tier-cards');
    if (!container || !backtestResults) return;

    const s = backtestResults.stats;

    container.innerHTML = `
      <div class="tier-card" style="border-left: 3px solid #22c55e; background: rgba(34, 197, 94, 0.08);">
        <div class="tier-header">
          <span class="tier-icon">&#9889;</span>
          <span class="tier-name" style="color: #22c55e;">SINGLES</span>
        </div>
        <div class="tier-desc">Ultra-selective single bets — CCS >= ${(ENGINE.CONFIG.SINGLE_BET_MIN_CONFIDENCE * 100).toFixed(0)}%</div>
        <div class="tier-stats">
          <span>Record: <strong>${s.singles.wins}/${s.singles.total}</strong> (${(s.singles.accuracy * 100).toFixed(1)}%)</span>
          <span>ROI: <strong style="color: ${s.singles.roi >= 0 ? '#22c55e' : '#ef4444'};">${s.singles.roi >= 0 ? '+' : ''}${(s.singles.roi * 100).toFixed(1)}%</strong></span>
          <span>P&L: <strong style="color: ${s.singles.pnl >= 0 ? '#22c55e' : '#ef4444'};">$${s.singles.pnl >= 0 ? '+' : ''}${s.singles.pnl}</strong></span>
        </div>
      </div>

      <div class="tier-card" style="border-left: 3px solid #a78bfa; background: rgba(167, 139, 250, 0.08);">
        <div class="tier-header">
          <span class="tier-icon">&#9883;</span>
          <span class="tier-name" style="color: #a78bfa;">PARLAYS</span>
        </div>
        <div class="tier-desc">Multi-leg parlays — independence-verified with CSIV</div>
        <div class="tier-stats">
          <span>Record: <strong>${s.parlays.wins}/${s.parlays.total}</strong> (${(s.parlays.accuracy * 100).toFixed(1)}%)</span>
          <span>Leg Rate: <strong>${(s.parlays.legAccuracy * 100).toFixed(1)}%</strong> (${s.parlays.hitLegs}/${s.parlays.totalLegs})</span>
          <span>P&L: <strong style="color: ${s.parlays.pnl >= 0 ? '#22c55e' : '#ef4444'};">$${s.parlays.pnl >= 0 ? '+' : ''}${s.parlays.pnl}</strong></span>
        </div>
      </div>

      <div class="tier-card" style="border-left: 3px solid #3b82f6; background: rgba(59, 130, 246, 0.08);">
        <div class="tier-header">
          <span class="tier-icon">&#10022;</span>
          <span class="tier-name" style="color: #3b82f6;">OVERALL</span>
        </div>
        <div class="tier-desc">Combined performance across all bet types</div>
        <div class="tier-stats">
          <span>Record: <strong>${s.overall.wins}/${s.overall.total}</strong> (${(s.overall.accuracy * 100).toFixed(1)}%)</span>
          <span>ROI: <strong style="color: ${s.overall.roi >= 0 ? '#22c55e' : '#ef4444'};">${s.overall.roi >= 0 ? '+' : ''}${(s.overall.roi * 100).toFixed(1)}%</strong></span>
          <span>P&L: <strong style="color: ${s.overall.pnl >= 0 ? '#22c55e' : '#ef4444'};">$${s.overall.pnl >= 0 ? '+' : ''}${s.overall.pnl}</strong></span>
        </div>
      </div>
    `;
  }

  // ========================================================================
  // RENDER: History
  // ========================================================================

  function renderHistory() {
    renderHistoryOverallStats();
    renderHistoryCategoryStats();
    renderHistoryTable();
  }

  function renderHistoryOverallStats() {
    const container = document.getElementById('history-overall-stats');
    if (!container) return;

    const resolved = allSignals.filter(s => s.hit !== null && s.hit !== undefined);
    const wins = resolved.filter(s => {
      if (s.betType === 'parlay') {
        return s.legs ? s.legs.every(l => l.hit) : s.hit;
      }
      return s.hit;
    }).length;
    const total = resolved.length;
    const acc = total > 0 ? (wins / total * 100).toFixed(1) : '--';

    let pnl = 0;
    resolved.forEach(s => { pnl += s.pnl || 0; });
    const roi = total > 0 ? (pnl / (total * ENGINE.CONFIG.UNIT_SIZE) * 100).toFixed(1) : '--';

    const activeDays = new Set(allSignals.map(s => s.date)).size;
    const pendingCount = allSignals.filter(s => s.hit === null || s.hit === undefined).length;

    container.innerHTML = `
      <div class="history-stat-row">
        <div class="history-stat">
          <div class="history-stat-value">${acc}%</div>
          <div class="history-stat-label">Overall Accuracy</div>
        </div>
        <div class="history-stat">
          <div class="history-stat-value">${wins}-${total - wins}</div>
          <div class="history-stat-label">Record</div>
        </div>
        <div class="history-stat">
          <div class="history-stat-value" style="color: ${pnl >= 0 ? '#22c55e' : '#ef4444'};">$${pnl >= 0 ? '+' : ''}${pnl}</div>
          <div class="history-stat-label">Total P&L</div>
        </div>
        <div class="history-stat">
          <div class="history-stat-value" style="color: ${parseFloat(roi) >= 0 ? '#22c55e' : '#ef4444'};">${roi >= 0 ? '+' : ''}${roi}%</div>
          <div class="history-stat-label">ROI</div>
        </div>
        <div class="history-stat">
          <div class="history-stat-value">${activeDays}</div>
          <div class="history-stat-label">Active Days</div>
        </div>
        ${pendingCount > 0 ? `
        <div class="history-stat">
          <div class="history-stat-value" style="color: var(--text-dim);">${pendingCount}</div>
          <div class="history-stat-label">Pending</div>
        </div>` : ''}
      </div>
    `;
  }

  function renderHistoryCategoryStats() {
    const container = document.getElementById('history-category-stats');
    if (!container) return;

    const singles = allSignals.filter(s => s.betType === 'single' && s.hit !== null && s.hit !== undefined);
    const parlays = allSignals.filter(s => s.betType === 'parlay' && s.hit !== null && s.hit !== undefined);

    const sWins = singles.filter(s => s.hit).length;
    const pWins = parlays.filter(s => s.legs ? s.legs.every(l => l.hit) : s.hit).length;

    const sPnl = singles.reduce((s, p) => s + (p.pnl || 0), 0);
    const pPnl = parlays.reduce((s, p) => s + (p.pnl || 0), 0);

    container.innerHTML = `
      <div class="category-stat-card">
        <div class="category-label" style="color: #22c55e;">Singles</div>
        <div class="category-detail">${sWins}/${singles.length} (${singles.length > 0 ? (sWins/singles.length*100).toFixed(0) : '--'}%) | P&L: $${sPnl >= 0 ? '+' : ''}${sPnl}</div>
      </div>
      <div class="category-stat-card">
        <div class="category-label" style="color: #a78bfa;">Parlays</div>
        <div class="category-detail">${pWins}/${parlays.length} (${parlays.length > 0 ? (pWins/parlays.length*100).toFixed(0) : '--'}%) | P&L: $${pPnl >= 0 ? '+' : ''}${pPnl}</div>
      </div>
    `;
  }

  function renderHistoryTable() {
    const tbody = document.getElementById('history-body');
    if (!tbody) return;

    let filtered = [...allSignals];
    if (activeFilter === 'single') {
      filtered = filtered.filter(s => s.betType === 'single');
    } else if (activeFilter === 'parlay') {
      filtered = filtered.filter(s => s.betType === 'parlay');
    }

    // Sort by date descending
    filtered.sort((a, b) => (b.date || '').localeCompare(a.date || ''));

    let html = '';
    for (const s of filtered) {
      if (s.betType === 'single') {
        html += renderSingleHistoryRow(s);
      } else if (s.betType === 'parlay') {
        html += renderParlayHistoryRow(s);
      }
    }

    tbody.innerHTML = html || '<tr><td colspan="7" class="status-msg">No history data yet. Run backtest or wait for tonight\'s picks to resolve.</td></tr>';
  }

  function renderSingleHistoryRow(s) {
    const resultClass = s.hit === true ? 'result-win' : s.hit === false ? 'result-loss' : 'result-pending';
    const resultText = s.hit === true ? 'WIN' : s.hit === false ? 'LOSS' : 'PENDING';
    const pnlText = s.pnl != null ? `${s.pnl >= 0 ? '+' : ''}$${s.pnl}` : '--';
    const pnlClass = s.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    const actualStr = s.actual != null ? ` (${s.actual})` : '';

    return `
      <tr>
        <td>${formatDate(s.date)}</td>
        <td><span class="type-badge single-badge">SINGLE</span></td>
        <td>${s.player} O${s.line} ${s.statLabel || ''}${actualStr}</td>
        <td>${ENGINE.formatOdds(s.odds)}</td>
        <td>${s.cascadeScore ? (s.cascadeScore * 100).toFixed(0) + '%' : '--'}</td>
        <td><span class="${resultClass}">${resultText}</span></td>
        <td class="${pnlClass}">${pnlText}</td>
      </tr>
    `;
  }

  function renderParlayHistoryRow(s) {
    const parlayHit = s.legs ? s.legs.every(l => l.hit === true) : s.hit === true;
    const hasPending = s.legs ? s.legs.some(l => l.hit === null || l.hit === undefined) : s.hit === null;
    const resultClass = hasPending ? 'result-pending' : (parlayHit ? 'result-win' : 'result-loss');
    const resultText = hasPending ? 'PENDING' : (parlayHit ? 'WIN' : 'LOSS');
    const pnlText = s.pnl != null ? `${s.pnl >= 0 ? '+' : ''}$${s.pnl}` : '--';
    const pnlClass = s.pnl != null ? (s.pnl >= 0 ? 'pnl-pos' : 'pnl-neg') : '';

    const oddsStr = s.parlay_american ? ENGINE.formatOdds(s.parlay_american) : ENGINE.formatOdds(s.odds || 0);
    const ccsStr = s.avgCascade ? (s.avgCascade * 100).toFixed(0) + '%' : '--';

    // Legs detail
    const legsHtml = (s.legs || []).map(l => {
      const icon = l.hit === true ? '&#10003;' : l.hit === false ? '&#10007;' : '&#8226;';
      const cls = l.hit === true ? 'leg-hit' : l.hit === false ? 'leg-miss' : 'leg-pending';
      const actualStr = l.actual != null ? ` (${l.actual})` : '';
      return `<span class="leg-chip ${cls}">${icon} ${l.player} O${l.line} ${l.statLabel || ''}${actualStr}</span>`;
    }).join('');

    return `
      <tr>
        <td>${formatDate(s.date)}</td>
        <td><span class="type-badge parlay-badge">${s.n_legs || (s.legs || []).length}-LEG</span></td>
        <td><div class="legs-container">${legsHtml}</div></td>
        <td>${oddsStr}</td>
        <td>${ccsStr}</td>
        <td><span class="${resultClass}">${resultText}</span></td>
        <td class="${pnlClass}">${pnlText}</td>
      </tr>
    `;
  }

  // ========================================================================
  // RENDER: Strategy Method
  // ========================================================================

  function renderMethod() {
    const container = document.getElementById('method-content');
    if (!container) return;

    container.innerHTML = `
      <div class="method-card" style="grid-column: 1 / -1;">
        <h3>NBA Betting Recommendation Engine v1.0</h3>
        <p>Patent-pending autonomous nightly recommendation system developed using
        <strong><a href="https://github.com/karpathy/autoresearch" style="color: #3b82f6;">Karpathy's AutoResearch</a></strong>
        methodology (monotonic ratchet, keep improvements, discard regressions) combined with
        <strong><a href="https://github.com/affaan-m/everything-claude-code" style="color: #3b82f6;">everything-claude-code</a></strong>
        quality gates and verification loops.</p>
        <p>All results validated against <strong>real FanDuel odds</strong> from The Odds API
        and <strong>real outcomes</strong> from ESPN box scores. Walk-forward validated with
        no future data leakage.</p>
      </div>

      <div class="method-card">
        <h3>1. Temporal Convergence Zone Detection (TCZD)</h3>
        <p>Identifies maximum-predictability windows by detecting when L5, L10, and L15
        performance floors converge within a tight band above the betting line. When all
        three floors agree, the prediction is maximally reliable.</p>
      </div>

      <div class="method-card">
        <h3>2. Volatility-Adjusted Confidence Scoring (VACS)</h3>
        <p>Computes betting confidence using coefficient of variation analysis relative to the
        line. Low CV combined with high floor produces geometrically higher confidence than
        either metric alone.</p>
      </div>

      <div class="method-card">
        <h3>3. Margin-of-Safety Depth Score (MSDS)</h3>
        <p>Quantifies bet robustness by measuring minimum, median, and mean distances above
        the line across multiple time windows. Ensures the player clears the line by
        comfortable margins, not razor-thin edges.</p>
      </div>

      <div class="method-card">
        <h3>4. Adaptive Bet Type Selector (ABTS)</h3>
        <p>Automatically selects optimal bet structure (single, multiple singles, or parlay)
        based on the quality distribution of available signals. Only recommends parlays when
        multiple independent high-confidence legs are available.</p>
      </div>

      <div class="method-card">
        <h3>5. Cross-Stat Independence Verification (CSIV)</h3>
        <p>Verifies statistical independence between parlay legs by computing cross-correlation
        matrices. Rejects leg combinations with correlation above threshold to ensure true
        multiplicative probability.</p>
      </div>

      <div class="method-card">
        <h3>6. Regime-Aware Filtering (RAF)</h3>
        <p>Detects player performance regime transitions using change-point detection on rolling
        statistics. Filters out players in transition periods to avoid betting on unstable
        performance patterns.</p>
      </div>

      <div class="method-card" style="grid-column: 1 / -1; border-color: #22c55e;">
        <h3 style="color: #f59e0b;">7. Confluence Cascade Scoring (CCS)</h3>
        <p>The core scoring innovation: produces a final recommendation score by cascading
        all independent signals (TCZD, VACS, MSDS, RAF) through a weighted geometric mean.
        <strong>Every signal must independently exceed its minimum threshold</strong> for
        the bet to qualify. This ensures no single strong signal can compensate for a weak one.</p>
        <ul>
          <li>TCZD weight: 30% — temporal floor convergence</li>
          <li>VACS weight: 25% — volatility-adjusted confidence</li>
          <li>MSDS weight: 25% — margin-of-safety depth</li>
          <li>RAF weight: 20% — regime stability</li>
        </ul>
      </div>

      <div class="method-card">
        <h3>Bet Types</h3>
        <ul>
          <li><strong style="color: #22c55e;">Single Bet</strong> — One elite-confidence pick
          (CCS >= ${(ENGINE.CONFIG.SINGLE_BET_MIN_CONFIDENCE * 100).toFixed(0)}%). Recommended when one leg stands far above the rest.</li>
          <li><strong style="color: #3b82f6;">Multiple Singles</strong> — 2-5 independent high-confidence picks.
          Each bet stands alone — no correlation risk.</li>
          <li><strong style="color: #a78bfa;">Parlay</strong> — 2-4 legs combined for higher payout.
          Only built from independence-verified legs (CSIV). Max 1 leg per game.</li>
        </ul>
      </div>

      <div class="method-card">
        <h3>Anti-Overfitting Measures</h3>
        <ul>
          <li><strong>Walk-forward only</strong> — signals generated before outcomes are known</li>
          <li><strong>Multi-window rolling profiles</strong> — L5, L10, L15, L20 windows prevent recency bias</li>
          <li><strong>Real odds validation</strong> — all backtests use actual FanDuel odds from The Odds API</li>
          <li><strong>Minimum 15-game sample</strong> — players need substantial history before eligibility</li>
          <li><strong>No curve fitting</strong> — thresholds set from data distribution, not outcome optimization</li>
          <li><strong>Regime filtering</strong> — avoids betting during unstable performance transitions</li>
          <li><strong>Geometric cascade</strong> — prevents gaming by requiring ALL signals to pass independently</li>
        </ul>
      </div>

      <div class="method-card">
        <h3>AutoResearch Methodology</h3>
        <ul>
          <li><strong>Autonomous experiment loop</strong> — hypothesize, test, keep/discard, repeat</li>
          <li><strong>Monotonic ratchet</strong> — each innovation only retained if it improves performance</li>
          <li><strong>Git-as-experiment-tracker</strong> — every experiment is a traceable diff</li>
          <li><strong>Fixed evaluation budget</strong> — consistent dataset and metrics for fair comparison</li>
          <li><strong>Simplicity principle</strong> — favor fewer, stronger signals over many weak ones</li>
        </ul>
      </div>
    `;
  }

  // ========================================================================
  // NAVIGATION & FILTERS
  // ========================================================================

  function setupNav() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        btn.classList.add('active');
        const view = btn.dataset.view;
        const el = document.getElementById(`view-${view}`);
        if (el) el.classList.add('active');
      });
    });
  }

  function setupFilters() {
    document.querySelectorAll('#history-filters .filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#history-filters .filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilter = btn.dataset.filter;
        renderHistoryTable();
      });
    });
  }

  // ========================================================================
  // UTILITIES
  // ========================================================================

  function formatDate(dateStr) {
    if (!dateStr || dateStr.length < 8) return dateStr || '--';
    return `${dateStr.slice(4, 6)}/${dateStr.slice(6, 8)}/${dateStr.slice(0, 4)}`;
  }

  // ========================================================================
  // BOOT
  // ========================================================================

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Also render method on load
  setTimeout(renderMethod, 100);
})();
