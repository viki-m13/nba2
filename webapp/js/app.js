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
    renderEquityCurve();
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

    // Show parlay stats prominently since that's all we care about
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

    // Parlays table
    const parlaysBody = document.querySelector('#parlays-history tbody');
    parlaysBody.innerHTML = parlays.slice().reverse().map(p => {
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

  // --- Equity Curve ---

  function renderEquityCurve() {
    if (!backtestResults) return;

    const canvas = document.getElementById('equity-chart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = 300 * dpr;
    ctx.scale(dpr, dpr);

    const w = canvas.offsetWidth;
    const h = 300;
    const pad = { top: 20, right: 20, bottom: 30, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Build cumulative P&L from parlays only
    const allPicks = [...backtestResults.parlays]
      .sort((a, b) => a.date.localeCompare(b.date));

    let cumPnl = 0;
    const points = [{ x: 0, y: 0 }];
    for (let i = 0; i < allPicks.length; i++) {
      cumPnl += allPicks[i].pnl;
      points.push({ x: i + 1, y: cumPnl });
    }

    const maxX = points.length - 1;
    const minY = Math.min(0, ...points.map(p => p.y));
    const maxY = Math.max(100, ...points.map(p => p.y));
    const rangeY = maxY - minY;

    const scaleX = (x) => pad.left + (x / maxX) * plotW;
    const scaleY = (y) => pad.top + plotH - ((y - minY) / rangeY) * plotH;

    // Background
    ctx.fillStyle = '#1a1d27';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#2d3140';
    ctx.lineWidth = 1;
    const ySteps = 5;
    for (let i = 0; i <= ySteps; i++) {
      const yVal = minY + (rangeY * i / ySteps);
      const y = scaleY(yVal);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();

      ctx.fillStyle = '#8b8fa3';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('$' + Math.round(yVal), pad.left - 8, y + 4);
    }

    // Zero line
    const zeroY = scaleY(0);
    ctx.strokeStyle = '#4a4e5c';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, zeroY);
    ctx.lineTo(w - pad.right, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // P&L line
    ctx.strokeStyle = '#a78bfa';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const x = scaleX(points[i].x);
      const y = scaleY(points[i].y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill below line
    ctx.lineTo(scaleX(maxX), scaleY(0));
    ctx.lineTo(scaleX(0), scaleY(0));
    ctx.closePath();
    ctx.fillStyle = 'rgba(167, 139, 250, 0.08)';
    ctx.fill();

    // Endpoint dot
    const lastPt = points[points.length - 1];
    ctx.fillStyle = '#a78bfa';
    ctx.beginPath();
    ctx.arc(scaleX(lastPt.x), scaleY(lastPt.y), 4, 0, Math.PI * 2);
    ctx.fill();

    // Label
    ctx.fillStyle = '#a78bfa';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`$${cumPnl}`, scaleX(lastPt.x) + 8, scaleY(lastPt.y) + 4);
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
