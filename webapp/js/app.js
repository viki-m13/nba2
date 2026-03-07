// =============================================================================
// NBA Prop Picks — App Controller (Parlays Only)
// =============================================================================

(function () {
  'use strict';

  let backtestResults = null;
  let todaySingles = [];
  let todayParlays = [];
  let currentPeriod = 'all';
  let seasonData = [];
  let playerBoxScores = [];
  let historicalOdds = [];

  // --- Initialization ---

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    setupNavigation();
    setupFilters();
    await loadData();
    await loadIncrementalOdds();
    runBacktest();
    renderDashboard();
    renderHistory();
    await generateTodayPicks();
  }

  // --- Data Loading ---

  async function loadData() {
    try {
      const [seasonRes, boxRes, oddsRes] = await Promise.all([
        fetch('data/espn_full_season_2025.json'),
        fetch('data/player_boxscores.json'),
        fetch('data/historical_odds.json'),
      ]);

      seasonData = await seasonRes.json();
      playerBoxScores = await boxRes.json();
      historicalOdds = await oddsRes.json();

      console.log(`Loaded: ${seasonData.length} games, ${playerBoxScores.length} box scores, ${historicalOdds.length} historical odds`);
    } catch (e) {
      console.error('Error loading data:', e);
    }
  }

  // --- Incremental Odds Loading ---

  async function loadIncrementalOdds() {
    try {
      const newOdds = await window.BettingEngine.fetchMissingDailyOdds(historicalOdds, 10);
      if (newOdds.length > 0) {
        historicalOdds = [...historicalOdds, ...newOdds];
        historicalOdds.sort((a, b) => a.date.localeCompare(b.date) || a.gameKey.localeCompare(b.gameKey));
        console.log(`Merged ${newOdds.length} new odds records. Total: ${historicalOdds.length}`);
      }
    } catch (e) {
      console.warn('Error loading incremental odds:', e);
    }
  }

  // --- Backtest ---

  function runBacktest() {
    if (!seasonData.length || !playerBoxScores.length || !historicalOdds.length) {
      console.warn('Insufficient data for backtest');
      return;
    }

    backtestResults = window.BettingEngine.runBacktest(seasonData, playerBoxScores, historicalOdds);
    console.log('Backtest results:', backtestResults.stats);
  }

  // --- Today's Picks ---

  async function generateTodayPicks() {
    const statusEl = document.getElementById('today-status');

    try {
      // Fetch today's games from ESPN
      let todayGames = [];
      try {
        todayGames = await window.NbaApi.fetchScoreboard();
      } catch (e) {
        console.warn('Could not fetch scoreboard:', e);
      }

      if (!todayGames || todayGames.length === 0) {
        statusEl.textContent = 'No games scheduled today.';
        return;
      }

      statusEl.textContent = `Found ${todayGames.length} games. Fetching FanDuel odds...`;

      // Train the model on all historical data
      const model = Object.create(window.BettingEngine.PlayerModel);
      model.history = {};
      const sorted = [...playerBoxScores].sort((a, b) => a.date.localeCompare(b.date));
      for (const g of sorted) {
        for (const p of (g.players || [])) {
          const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
          if (mins < 10) continue;
          model.update(p.name, p.pts, mins, g.date, p.team);
        }
      }

      // Fetch live FanDuel odds
      let liveOdds = null;
      try {
        liveOdds = await window.BettingEngine.fetchLiveOdds();
      } catch (e) {
        console.warn('Could not fetch live odds:', e);
        statusEl.textContent = 'Could not fetch FanDuel odds. Showing historical data only.';
        return;
      }

      if (!liveOdds || Object.keys(liveOdds.playerProps).length === 0) {
        statusEl.textContent = 'No FanDuel player props available right now. Check back closer to game time.';
        return;
      }

      // Evaluate props for each game
      todaySingles = [];
      for (const [gameKey, gameData] of Object.entries(liveOdds.playerProps)) {
        for (const [playerName, lines] of Object.entries(gameData.lines)) {
          const prop = model.findBestProp(playerName, lines);
          if (prop) {
            todaySingles.push({
              ...prop,
              gameKey,
              gameDisplay: `${gameData.away} @ ${gameData.home}`,
            });
          }
        }
      }

      // Sort by confidence
      todaySingles.sort((a, b) => b.confidence - a.confidence);

      // Build parlays
      todayParlays = window.BettingEngine.buildParlays(todaySingles);

      // Add gameDisplay to parlay legs
      for (const parlay of todayParlays) {
        parlay.legs = parlay.legs.map(leg => {
          const matchingSingle = todaySingles.find(s => s.player === leg.player && s.line === leg.line);
          return {
            ...leg,
            gameKey: matchingSingle ? matchingSingle.gameKey : '',
            gameDisplay: matchingSingle ? matchingSingle.gameDisplay : '',
          };
        });
      }

      // Save odds to local history for future reference
      const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      window.BettingEngine.saveOddsToHistory(todaySingles, today);

      // Render
      if (todayParlays.length === 0) {
        statusEl.textContent = 'No qualifying parlays for today\'s games. Our filters are strict to maintain high accuracy.';
      } else {
        statusEl.style.display = 'none';
        renderTodayPicks();
      }
    } catch (e) {
      console.error('Error generating picks:', e);
      statusEl.textContent = 'Error loading picks. See console for details.';
    }
  }

  function renderTodayPicks() {
    const parlaysEl = document.getElementById('today-parlays');

    if (todayParlays.length > 0) {
      parlaysEl.innerHTML = todayParlays.map(renderParlayCard).join('');
    }
  }

  function formatGameKey(gameKey) {
    if (!gameKey) return '';
    return gameKey.replace('@', ' @ ');
  }

  function renderParlayCard(parlay) {
    const legsHtml = parlay.legs.map(leg => `
      <div class="parlay-leg">
        <div class="parlay-leg-info">
          <span class="parlay-leg-player">${leg.player} (${leg.team})</span>
          <span class="parlay-leg-game">${leg.gameDisplay || formatGameKey(leg.gameKey)}</span>
        </div>
        <div class="parlay-leg-right">
          <span class="parlay-leg-line">OVER ${leg.line}</span>
          <span class="parlay-leg-odds">${window.BettingEngine.formatOdds(leg.odds)}</span>
        </div>
      </div>`).join('');

    const today = new Date();
    const dateStr = `${String(today.getMonth()+1).padStart(2,'0')}/${String(today.getDate()).padStart(2,'0')}/${today.getFullYear()}`;
    const isMega = parlay.numLegs >= 6;
    const parlayLabel = isMega ? `${parlay.numLegs}-Leg Mega Parlay` : `${parlay.numLegs}-Leg Parlay`;

    return `
      <div class="pick-card parlay${isMega ? ' mega-parlay' : ''}">
        <div class="pick-header">
          <span class="pick-date">${dateStr}</span>
          <span class="pick-type">${parlayLabel}</span>
          <span class="pick-odds">${window.BettingEngine.formatOdds(parlay.odds)}</span>
        </div>
        <div class="parlay-legs">${legsHtml}</div>
        <div class="parlay-summary">
          <span>Combined Hit Rate: ${(parlay.combinedHitRate * 100).toFixed(1)}%</span>
          <span class="parlay-ev">EV: ${parlay.ev > 0 ? '+' : ''}${(parlay.ev * 100).toFixed(1)}%</span>
          <span class="parlay-payout">$100 wins $${Math.round((parlay.decimalOdds - 1) * 100)}</span>
        </div>
      </div>`;
  }

  // --- Dashboard ---

  function renderDashboard() {
    if (!backtestResults) return;
    const s = backtestResults.stats;

    document.getElementById('stat-accuracy').textContent = `${(s.parlays.hitRate * 100).toFixed(1)}%`;
    document.getElementById('stat-record').textContent = `${s.parlays.wins}-${s.parlays.losses}`;
    document.getElementById('stat-pnl').textContent = `$${s.parlays.pnl >= 0 ? '+' : ''}${s.parlays.pnl}`;
    document.getElementById('stat-roi').textContent = `${s.parlays.roi >= 0 ? '+' : ''}${s.parlays.roi}%`;
    document.getElementById('stat-parlay-record').textContent = `${s.singles.wins}-${s.singles.losses}`;
    document.getElementById('stat-days').textContent = `${s.daysWithPicks}/${s.totalDays}`;
  }

  // --- History ---

  function renderHistory() {
    if (!backtestResults) return;

    const { parlays } = getFilteredResults();

    // Walk-forward accuracy stats
    const statsEl = document.getElementById('history-stats');
    const wins = parlays.filter(p => p.won).length;
    const total = parlays.length;
    const hitRate = total > 0 ? (wins / total * 100).toFixed(1) : '0.0';
    const pnl = parlays.reduce((s, p) => s + p.pnl, 0);
    const roi = total > 0 ? (pnl / (total * 100) * 100).toFixed(1) : '0.0';

    // Breakdown by leg count
    const byLegs = {};
    for (const p of parlays) {
      const n = p.numLegs || p.legs.length;
      if (!byLegs[n]) byLegs[n] = { wins: 0, total: 0 };
      byLegs[n].total++;
      if (p.won) byLegs[n].wins++;
    }
    const breakdownHtml = Object.keys(byLegs).sort((a, b) => a - b).map(n => {
      const g = byLegs[n];
      const rate = (g.wins / g.total * 100).toFixed(0);
      return `<span class="history-stat-breakdown">${n}-Leg: ${g.wins}-${g.total - g.wins} (${rate}%)</span>`;
    }).join('');

    statsEl.innerHTML = `
      <div class="history-stat-row">
        <div class="history-stat">
          <span class="history-stat-value">${hitRate}%</span>
          <span class="history-stat-label">Walk-Forward Accuracy</span>
        </div>
        <div class="history-stat">
          <span class="history-stat-value">${wins}-${total - wins}</span>
          <span class="history-stat-label">Record</span>
        </div>
        <div class="history-stat">
          <span class="history-stat-value ${pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">$${pnl >= 0 ? '+' : ''}${pnl}</span>
          <span class="history-stat-label">P&L ($100/bet)</span>
        </div>
        <div class="history-stat">
          <span class="history-stat-value">${roi >= 0 ? '+' : ''}${roi}%</span>
          <span class="history-stat-label">ROI</span>
        </div>
      </div>
      <div class="history-stat-breakdowns">${breakdownHtml}</div>`;

    // Parlays table
    const parlaysBody = document.querySelector('#parlays-history tbody');
    parlaysBody.innerHTML = parlays.slice().reverse().map(p => {
      const numLegs = p.numLegs || p.legs.length;
      const legsHtml = p.legs.map(l => {
        const game = formatGameKey(l.gameKey);
        const resultClass = l.won ? 'result-win' : 'result-loss';
        return `<div class="history-leg">
          <span class="history-leg-player">${l.player} (${l.team})</span>
          <span class="history-leg-line">OVER ${l.line}</span>
          <span class="history-leg-odds">${window.BettingEngine.formatOdds(l.odds)}</span>
          <span class="history-leg-game">${game}</span>
          <span class="history-leg-actual ${resultClass}">${l.actual} pts</span>
        </div>`;
      }).join('');

      return `
        <tr>
          <td>${formatDate(p.date)}</td>
          <td>${numLegs}-Leg</td>
          <td class="legs-cell">${legsHtml}</td>
          <td>${window.BettingEngine.formatOdds(p.odds)}</td>
          <td class="${p.won ? 'result-win' : 'result-loss'}">${p.won ? 'WIN' : 'LOSS'}</td>
          <td class="${p.pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${p.pnl >= 0 ? '+' : ''}$${p.pnl}</td>
        </tr>`;
    }).join('');
  }

  function getFilteredResults() {
    if (!backtestResults) return { parlays: [] };

    let parlays = backtestResults.parlays;

    if (currentPeriod !== 'all') {
      const days = parseInt(currentPeriod);
      const allDates = [...new Set(parlays.map(p => p.date))].sort();
      const cutoff = allDates[Math.max(0, allDates.length - days)] || '';
      parlays = parlays.filter(p => p.date >= cutoff);
    }

    return { parlays };
  }

  // --- Navigation ---

  function setupNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById(`view-${btn.dataset.view}`).classList.add('active');
      });
    });
  }

  function setupFilters() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentPeriod = btn.dataset.period;
        renderHistory();
      });
    });
  }

  // --- Helpers ---

  function formatDate(dateStr) {
    // YYYYMMDD -> MM/DD
    return `${dateStr.slice(4, 6)}/${dateStr.slice(6, 8)}`;
  }
})();
