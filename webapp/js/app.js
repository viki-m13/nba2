// =============================================================================
// NBA Prop Picks — App Controller v4.0
// =============================================================================

(function () {
  'use strict';

  let backtestResults = null;
  let todayParlays = [];
  let currentPeriod = 'all';
  let seasonData = [];
  let playerBoxScores = [];
  let historicalOdds = [];
  let rebAstProps = [];
  let livePicksData = null;
  let altBacktestResults = null;

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    setupNavigation();
    setupFilters();
    await loadData();
    runBacktest();

    // Run alt-line backtest using existing historical data
    runAltBacktest();

    // Seed live history for recent days if needed
    await seedLiveHistory(5);

    // Resolve any unresolved live picks
    try {
      livePicksData = await window.BettingEngine.resolveLivePicks();
    } catch (e) {
      console.warn('Could not resolve live picks:', e);
      livePicksData = window.BettingEngine.loadLivePicks();
    }

    renderDashboard();
    renderHistory();
    await generateTodayPicks();
  }

  function runAltBacktest() {
    if (!window.NbaAltBacktest || !playerBoxScores.length) return;
    console.log('[APP] Running alt-line walk-forward backtest...');
    try {
      altBacktestResults = window.NbaAltBacktest.runBacktest(playerBoxScores, historicalOdds, rebAstProps);
      if (altBacktestResults && altBacktestResults.stats) {
        const s = altBacktestResults.stats;
        console.log('[ALT-BACKTEST] Results:');
        console.log(`  Legs: ${s.overall.legs.total} total, ${s.overall.legs.hitRate * 100}% hit rate, ROI ${s.overall.legs.roi}%`);
        console.log(`  Parlays: ${s.overall.parlays.total} total, ${s.overall.parlays.hitRate * 100}% hit rate, ROI ${s.overall.parlays.roi}%`);
        console.log('  By market:', s.byMarket);
        console.log('  By builder:', s.byBuilder);
        console.log('  By role:', s.byRole);
        console.log('  By edge bucket:', s.byEdge);
      }
    } catch (e) {
      console.warn('[APP] Alt backtest error:', e);
    }
  }

  // --- Data Loading ---

  async function loadData() {
    try {
      const [seasonRes, boxRes, oddsRes, rebAstRes, livePicksRes] = await Promise.all([
        fetch('data/espn_full_season_2025.json'),
        fetch('data/player_boxscores.json'),
        fetch('data/historical_odds.json'),
        fetch('data/historical_reb_ast_props.json').catch(() => null),
        fetch('data/live_picks_2026.json').catch(() => null),
      ]);

      seasonData = await seasonRes.json();
      playerBoxScores = await boxRes.json();
      historicalOdds = await oddsRes.json();
      if (rebAstRes && rebAstRes.ok) rebAstProps = await rebAstRes.json();
      if (livePicksRes && livePicksRes.ok) {
        window.BettingEngine.setBasePicksData(await livePicksRes.json());
      }

      console.log(`Loaded: ${seasonData.length} games, ${playerBoxScores.length} box scores, ${historicalOdds.length} odds`);
    } catch (e) {
      console.error('Error loading data:', e);
    }
  }

  // --- Backtest ---

  function runBacktest() {
    if (!seasonData.length || !playerBoxScores.length || !historicalOdds.length) return;
    backtestResults = window.BettingEngine.runBacktest(seasonData, playerBoxScores, historicalOdds, rebAstProps);
    console.log('Backtest results:', backtestResults.stats);
  }

  // --- Seed Live History ---

  async function seedLiveHistory(numDays) {
    const existing = window.BettingEngine.loadLivePicks();
    const existingDates = new Set(existing.parlays.map(p => p.date));

    const today = new Date();
    const datesToSeed = [];
    for (let i = 1; i <= numDays; i++) {
      const d = new Date(today);
      d.setDate(d.getDate() - i);
      const ds = `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}`;
      if (!existingDates.has(ds)) datesToSeed.push(ds);
    }

    if (datesToSeed.length === 0) return;

    // Build model from all box score data
    const sorted = [...playerBoxScores].sort((a, b) => a.date.localeCompare(b.date));
    const model = Object.create(window.BettingEngine.PlayerModel);
    model.history = {};
    for (const g of sorted) {
      for (const p of (g.players || [])) {
        const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
        if (mins < 10) continue;
        model.update(p.name, {
          pts: p.pts,
          reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
          ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
          min: mins,
        }, g.date, p.team);
      }
    }

    for (const date of datesToSeed) {
      try {
        const dayOdds = await window.BettingEngine.fetchHistoricalOddsForDate(date);
        if (!dayOdds || dayOdds.length === 0) continue;

        const daySingles = [];
        for (const record of dayOdds) {
          const gameKey = record.gameKey;
          const gameDisplay = `${record.awayTeam} @ ${record.homeTeam}`;

          const addPick = (prop) => {
            if (!prop) return;
            if (!daySingles.find(s => s.player === prop.player && s.statType === prop.statType && s.line === prop.line)) {
              daySingles.push({ ...prop, gameKey, gameDisplay });
            }
          };

          for (const [name, lines] of Object.entries(record.playerProps || {})) addPick(model.findBestProp(name, 'points', lines));
          for (const [name, lines] of Object.entries(record.playerRebProps || {})) addPick(model.findBestProp(name, 'rebounds', lines));
          for (const [name, lines] of Object.entries(record.playerAstProps || {})) addPick(model.findBestProp(name, 'assists', lines));
        }

        const dayParlays = window.BettingEngine.buildParlays(daySingles);
        for (const parlay of dayParlays) {
          parlay.legs = parlay.legs.map(leg => {
            const match = daySingles.find(s => s.player === leg.player && s.line === leg.line && s.statType === leg.statType);
            return { ...leg, gameKey: match ? match.gameKey : '', gameDisplay: match ? match.gameDisplay : '' };
          });
        }

        if (dayParlays.length > 0) {
          window.BettingEngine.saveTodayParlays(dayParlays, [], date);
        }

        await new Promise(r => setTimeout(r, 500));
      } catch (e) {
        console.warn(`[SEED] Error seeding ${date}:`, e);
      }
    }
  }

  // --- Today's Picks ---

  async function generateTodayPicks() {
    const statusEl = document.getElementById('today-status');

    try {
      let todayGames = [];
      try {
        const sbResult = await window.NbaApi.fetchScoreboard();
        todayGames = (sbResult && sbResult.games) || sbResult || [];
        if (!Array.isArray(todayGames)) todayGames = [];
      } catch (e) {
        console.warn('Could not fetch scoreboard:', e);
      }

      if (todayGames.length === 0) {
        statusEl.textContent = 'No games scheduled today.';
        renderTodayGames([]);
        return;
      }

      statusEl.textContent = `Found ${todayGames.length} games. Fetching FanDuel odds...`;
      renderTodayGames(todayGames);

      // Build model from historical data
      const sorted = [...playerBoxScores].sort((a, b) => a.date.localeCompare(b.date));
      const model = Object.create(window.BettingEngine.PlayerModel);
      model.history = {};
      for (const g of sorted) {
        for (const p of (g.players || [])) {
          const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
          if (mins < 10) continue;
          model.update(p.name, {
            pts: p.pts,
            reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
            ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
            min: mins,
          }, g.date, p.team);
        }
      }

      // Fetch live odds
      let liveOdds = null;
      try {
        liveOdds = await window.BettingEngine.fetchLiveOdds();
      } catch (e) {
        statusEl.textContent = 'Could not fetch FanDuel odds.';
        return;
      }

      if (liveOdds && liveOdds.quotaExhausted) {
        statusEl.textContent = 'Odds API quota exhausted. Will resume when quota resets.';
        statusEl.style.color = 'var(--gold)';
        return;
      }

      if (!liveOdds || Object.keys(liveOdds.playerProps).length === 0) {
        statusEl.textContent = 'No FanDuel player props available. Check back closer to game time.';
        return;
      }

      // Evaluate all props
      const todaySingles = [];
      for (const [gameKey, gameData] of Object.entries(liveOdds.playerProps)) {
        const gameDisplay = `${gameData.away} @ ${gameData.home}`;

        const addPick = (prop) => {
          if (!prop) return;
          if (!todaySingles.find(s => s.player === prop.player && s.statType === prop.statType && s.line === prop.line)) {
            todaySingles.push({ ...prop, gameKey, gameDisplay });
          }
        };

        for (const [name, lines] of Object.entries(gameData.lines || {})) addPick(model.findBestProp(name, 'points', lines));
        for (const [name, lines] of Object.entries(gameData.rebLines || {})) addPick(model.findBestProp(name, 'rebounds', lines));
        for (const [name, lines] of Object.entries(gameData.astLines || {})) addPick(model.findBestProp(name, 'assists', lines));
      }

      // Build legacy parlays
      todayParlays = window.BettingEngine.buildParlays(todaySingles);
      for (const parlay of todayParlays) {
        parlay.legs = parlay.legs.map(leg => {
          const match = todaySingles.find(s => s.player === leg.player && s.line === leg.line && s.statType === leg.statType);
          return { ...leg, gameKey: match ? match.gameKey : '', gameDisplay: match ? match.gameDisplay : '' };
        });
      }

      // --- Alt-Line Pipeline ---
      // Share the model with the engine so the feature table can access history
      window.BettingEngine.PlayerModel.history = model.history;

      // Build game-level odds map for environment context
      const gameOddsMap = {};
      // Try to get spread/total from ESPN
      try {
        const espnData = await window.NbaApi.fetchESPNOdds();
        for (const game of todayGames) {
          const gk = `${game.awayTeam}@${game.homeTeam}`;
          if (game.ouLine || espnData.odds[`${game.homeTeam}-${game.awayTeam}`]) {
            gameOddsMap[gk] = {
              total: game.ouLine || espnData.odds[`${game.homeTeam}-${game.awayTeam}`] || 220,
              spread_point: 0,
            };
          }
        }
      } catch (e) { console.warn('Could not build game odds map:', e); }

      // Generate feature table and slips
      const featureTable = window.BettingEngine.generateFeatureTable(liveOdds, gameOddsMap);
      const slips = window.BettingEngine.generateSlips(featureTable);

      // Render alt-line slips
      renderAltSlips(slips, featureTable);

      // Save all parlays (legacy + new) to live tracking
      const allParlays = [
        ...todayParlays,
        ...(slips.core || []),
        ...(slips.screenshot || []),
        ...(slips.nuke || []),
      ];
      const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      window.BettingEngine.saveTodayParlays(allParlays, [], today);
      livePicksData = window.BettingEngine.loadLivePicks();
      renderLiveHistory();

      const totalSlips = (slips.core || []).length + (slips.screenshot || []).length + (slips.nuke || []).length;
      if (totalSlips === 0 && todayParlays.length === 0) {
        statusEl.textContent = 'No qualifying parlays today. Filters are strict for safety.';
      } else {
        statusEl.style.display = 'none';
        renderTodayPicks();
      }
    } catch (e) {
      console.error('Error generating picks:', e);
      statusEl.textContent = 'Error loading picks. See console.';
    }
  }

  // --- Rendering ---

  function renderTodayPicks() {
    const parlaysEl = document.getElementById('today-parlays');
    parlaysEl.innerHTML = todayParlays.map(renderParlayCard).join('');
  }

  function renderParlayCard(parlay) {
    const legsHtml = parlay.legs.map(leg => `
      <div class="parlay-leg">
        <div class="parlay-leg-info">
          <span class="parlay-leg-player">${leg.player} (${leg.team})</span>
          <span class="parlay-leg-game">${leg.gameDisplay || leg.gameKey.replace('@', ' @ ')}</span>
        </div>
        <div class="parlay-leg-right">
          <span class="parlay-leg-stat">${leg.statLabel || 'PTS'}</span>
          <span class="parlay-leg-line">OVER ${leg.line}</span>
          <span class="parlay-leg-odds">${window.BettingEngine.formatOdds(leg.odds)}</span>
        </div>
      </div>`).join('');

    const today = new Date();
    const dateStr = `${String(today.getMonth()+1).padStart(2,'0')}/${String(today.getDate()).padStart(2,'0')}/${today.getFullYear()}`;
    const statTypes = new Set(parlay.legs.map(l => l.statType || 'points'));
    const label = statTypes.size > 1 ? `${parlay.numLegs}-Leg Multi-Stat Parlay` : `${parlay.numLegs}-Leg Parlay`;

    return `
      <div class="pick-card parlay${statTypes.size > 1 ? ' multi-stat' : ''}">
        <div class="pick-header">
          <span class="pick-date">${dateStr}</span>
          <span class="pick-type">${label}</span>
          <span class="pick-odds">${window.BettingEngine.formatOdds(parlay.odds)}</span>
        </div>
        <div class="parlay-legs">${legsHtml}</div>
        <div class="parlay-summary">
          <span>Hit Rate: ${(parlay.combinedHitRate * 100).toFixed(1)}%</span>
          <span class="parlay-ev">EV: ${parlay.ev > 0 ? '+' : ''}${(parlay.ev * 100).toFixed(1)}%</span>
          <span class="parlay-payout">$100 wins $${Math.round((parlay.decimalOdds - 1) * 100)}</span>
        </div>
      </div>`;
  }

  function renderTodayGames(games) {
    const container = document.getElementById('today-games');
    if (!container) return;

    const dateStr = new Date().toLocaleDateString('en-US', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
    });

    if (!games || games.length === 0) {
      container.innerHTML = `<div class="today-games-header"><h3>Today's Games</h3><span class="today-games-date">${dateStr}</span></div><p class="status-msg">No games scheduled.</p>`;
      return;
    }

    const gamesHtml = games.map(g => {
      const away = g.awayTeam || g.away_team || '???';
      const home = g.homeTeam || g.home_team || '???';
      let timeInfo = '';
      if (g.status === 'final') timeInfo = `Final: ${g.awayScore}-${g.homeScore}`;
      else if (g.status === 'live' || g.status === 'halftime') timeInfo = g.statusText || 'Live';
      else if (g.gameTime) { try { timeInfo = new Date(g.gameTime).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' }); } catch (e) { timeInfo = 'TBD'; } }
      else timeInfo = g.statusText || 'TBD';

      return `<div class="today-game-card"><span class="today-game-teams">${away} @ ${home}</span><span class="today-game-time">${timeInfo}</span></div>`;
    }).join('');

    container.innerHTML = `<div class="today-games-header"><h3>Today's Games</h3><span class="today-games-date">${dateStr}</span></div><div class="today-games-grid">${gamesHtml}</div>`;
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

    if (s.recent30) {
      const el = document.getElementById('stat-recent');
      if (el) el.textContent = `${(s.recent30.hitRate * 100).toFixed(1)}%`;
    }
  }

  // --- History ---

  function renderHistory() {
    if (!backtestResults) return;

    const parlays = getFilteredParlays();
    const wins = parlays.filter(p => p.won).length;
    const total = parlays.length;
    const pnl = parlays.reduce((s, p) => s + p.pnl, 0);
    const roi = total > 0 ? (pnl / (total * 100) * 100).toFixed(1) : '0.0';

    const allDates = [...new Set(parlays.map(p => p.date))].sort();
    const r14Cutoff = allDates[Math.max(0, allDates.length - 14)] || '';
    const recent14 = parlays.filter(p => p.date >= r14Cutoff);
    const r14Wins = recent14.filter(p => p.won).length;
    const r14Rate = recent14.length > 0 ? (r14Wins / recent14.length * 100).toFixed(1) : '--';

    const byLegs = {};
    for (const p of parlays) {
      const n = p.numLegs || p.legs.length;
      if (!byLegs[n]) byLegs[n] = { wins: 0, total: 0 };
      byLegs[n].total++;
      if (p.won) byLegs[n].wins++;
    }
    const breakdownHtml = Object.keys(byLegs).sort((a, b) => a - b).map(n => {
      const g = byLegs[n];
      return `<span class="history-stat-breakdown">${n}-Leg: ${g.wins}-${g.total - g.wins} (${(g.wins/g.total*100).toFixed(0)}%)</span>`;
    }).join('');

    document.getElementById('history-stats').innerHTML = `
      <div class="history-stat-row">
        <div class="history-stat"><span class="history-stat-value">${total > 0 ? (wins/total*100).toFixed(1) : '--'}%</span><span class="history-stat-label">Walk-Forward Accuracy</span></div>
        <div class="history-stat"><span class="history-stat-value highlight-recent">${r14Rate}%</span><span class="history-stat-label">Last 14 Days</span></div>
        <div class="history-stat"><span class="history-stat-value">${wins}-${total-wins}</span><span class="history-stat-label">Record</span></div>
        <div class="history-stat"><span class="history-stat-value ${pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">$${pnl >= 0 ? '+' : ''}${pnl}</span><span class="history-stat-label">P&L ($100/bet)</span></div>
        <div class="history-stat"><span class="history-stat-value">${roi >= 0 ? '+' : ''}${roi}%</span><span class="history-stat-label">ROI</span></div>
      </div>
      <div class="history-stat-breakdowns">${breakdownHtml}</div>`;

    document.querySelector('#parlays-history tbody').innerHTML = renderParlayRows(
      parlays.sort((a, b) => b.date.localeCompare(a.date))
    );

    renderLiveHistory();
  }

  function renderParlayRows(list) {
    return list.map(p => {
      const numLegs = p.numLegs || p.legs.length;
      const legsHtml = p.legs.map(l => {
        const lbl = l.statLabel || window.BettingEngine.statLabel(l.statType || 'points');
        const hasResult = l.actual !== null && l.actual !== undefined;
        const cls = hasResult ? (l.won ? 'result-win' : 'result-loss') : '';
        const txt = hasResult ? `${l.actual} ${lbl.toLowerCase()}` : 'Pending';
        return `<div class="history-leg">
          <span class="history-leg-player">${l.player} (${l.team})</span>
          <span class="history-leg-stat">${lbl}</span>
          <span class="history-leg-line">OVER ${l.line}</span>
          <span class="history-leg-odds">${window.BettingEngine.formatOdds(l.odds)}</span>
          <span class="history-leg-game">${l.gameDisplay || (l.gameKey || '').replace('@', ' @ ')}</span>
          <span class="history-leg-actual ${cls}">${txt}</span>
        </div>`;
      }).join('');

      const statTypes = new Set(p.legs.map(l => l.statType || 'points'));
      const typeLabel = statTypes.size > 1 ? `${numLegs}-Leg Multi` : `${numLegs}-Leg`;
      const isResolved = p.resolved !== undefined ? p.resolved : true;
      const resultText = isResolved ? (p.won ? 'WIN' : 'LOSS') : 'PENDING';
      const resultClass = isResolved ? (p.won ? 'result-win' : 'result-loss') : 'result-pending';
      const pnlText = isResolved ? `${p.pnl >= 0 ? '+' : ''}$${p.pnl}` : '--';
      const pnlClass = isResolved ? (p.pnl >= 0 ? 'pnl-pos' : 'pnl-neg') : '';

      return `<tr>
        <td>${formatDate(p.date)}</td>
        <td>${typeLabel}</td>
        <td class="legs-cell">${legsHtml}</td>
        <td>${window.BettingEngine.formatOdds(p.odds)}</td>
        <td class="${resultClass}">${resultText}</td>
        <td class="${pnlClass}">${pnlText}</td>
      </tr>`;
    }).join('');
  }

  function renderLiveHistory() {
    if (!livePicksData) return;
    const liveParlays = livePicksData.parlays || [];
    if (liveParlays.length === 0) return;

    const resolved = liveParlays.filter(p => p.resolved);
    const wins = resolved.filter(p => p.won).length;
    const pnl = resolved.reduce((s, p) => s + (p.pnl || 0), 0);
    const pending = liveParlays.filter(p => !p.resolved).length;

    const statsHtml = resolved.length > 0 ? `
      <div class="history-stat-row" style="margin-top: 0.5rem;">
        <div class="history-stat"><span class="history-stat-value">${(wins/resolved.length*100).toFixed(1)}%</span><span class="history-stat-label">Live Accuracy</span></div>
        <div class="history-stat"><span class="history-stat-value">${wins}-${resolved.length-wins}</span><span class="history-stat-label">Record</span></div>
        <div class="history-stat"><span class="history-stat-value ${pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">$${pnl >= 0 ? '+' : ''}${pnl}</span><span class="history-stat-label">P&L</span></div>
        ${pending > 0 ? `<div class="history-stat"><span class="history-stat-value" style="color: var(--text-dim);">${pending}</span><span class="history-stat-label">Pending</span></div>` : ''}
      </div>` : '<p class="status-msg" style="margin: 0.5rem 0;">No resolved picks yet.</p>';

    const section = document.createElement('div');
    section.className = 'history-section live-tracking-section';
    section.innerHTML = `
      <h3 style="margin-top: 1.5rem; border-top: 2px solid var(--accent); padding-top: 1rem;">
        2025-26 Live Tracking <span style="font-size: 0.75rem; color: var(--text-dim); font-weight: 400;">Auto-resolves daily</span>
      </h3>
      <div class="history-stats">${statsHtml}</div>
      <table class="history-table"><thead><tr><th>Date</th><th>Type</th><th>Legs</th><th>Parlay Odds</th><th>Result</th><th>P&L</th></tr></thead>
        <tbody>${renderParlayRows(liveParlays.sort((a, b) => b.date.localeCompare(a.date)))}</tbody>
      </table>`;

    const historyView = document.getElementById('view-history');
    const old = historyView.querySelector('.live-tracking-section');
    if (old) old.remove();
    historyView.appendChild(section);
  }

  function getFilteredParlays() {
    if (!backtestResults) return [];
    let parlays = backtestResults.parlays;
    if (currentPeriod !== 'all') {
      const days = parseInt(currentPeriod);
      const allDates = [...new Set(parlays.map(p => p.date))].sort();
      const cutoff = allDates[Math.max(0, allDates.length - days)] || '';
      parlays = parlays.filter(p => p.date >= cutoff);
    }
    return parlays;
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

  function formatDate(dateStr) {
    return `${dateStr.slice(4, 6)}/${dateStr.slice(6, 8)}/${dateStr.slice(0, 4)}`;
  }

  // =========================================================================
  // Alt-Line Slip Rendering
  // =========================================================================

  function renderAltSlips(slips, featureTable) {
    renderSlipSection('alt-core-slips', slips.core || [], 'core');
    renderSlipSection('alt-screenshot-slips', slips.screenshot || [], 'screenshot');
    renderSlipSection('alt-nuke-slips', slips.nuke || [], 'nuke');
    renderFeatureTable(featureTable || []);
  }

  function renderSlipSection(containerId, parlays, type) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const grid = container.querySelector('.picks-grid');
    if (!grid) return;

    if (parlays.length === 0) {
      grid.innerHTML = `<p class="status-msg" style="margin: 0.5rem 0; font-size: 0.85rem;">No ${type} parlays qualify today.</p>`;
      return;
    }

    grid.innerHTML = parlays.map(p => renderAltParlayCard(p)).join('');
  }

  function renderAltParlayCard(parlay) {
    const legsHtml = parlay.legs.map(leg => {
      const roleClass = leg.roleTag === 'primary_creator' ? 'role-creator' :
                         leg.roleTag === 'high_usage_scorer' ? 'role-scorer' :
                         leg.roleTag === 'rebound_heavy_big' ? 'role-big' : '';
      return `
        <div class="parlay-leg ${roleClass}">
          <div class="parlay-leg-info">
            <span class="parlay-leg-player">${leg.player} <span style="opacity: 0.6;">(${leg.team})</span></span>
            <span class="parlay-leg-game">${leg.gameDisplay || ''}</span>
          </div>
          <div class="parlay-leg-right">
            <span class="parlay-leg-stat">${leg.marketLabel || leg.statLabel || 'ALT PTS'}</span>
            <span class="parlay-leg-line">OVER ${leg.line}</span>
            <span class="parlay-leg-odds">${window.BettingEngine.formatOdds(leg.bookOdds || leg.odds)}</span>
          </div>
          <div class="parlay-leg-meta" style="font-size: 0.7rem; opacity: 0.5; margin-top: 2px;">
            ${leg.roleTag || ''} | prob ${((leg.modelProb || 0) * 100).toFixed(0)}% | edge ${((leg.edge || 0) * 100).toFixed(1)}% | avg ${leg.avg || '--'}
          </div>
        </div>`;
    }).join('');

    const builder = parlay.builder || 'PARLAY';
    const builderClass = builder === 'CORE' ? 'builder-core' :
                         builder === 'SCREENSHOT' ? 'builder-screenshot' :
                         builder === 'NUKE' ? 'builder-nuke' :
                         builder === 'SGP' ? 'builder-sgp' : '';

    return `
      <div class="pick-card parlay multi-stat ${builderClass}">
        <div class="pick-header">
          <span class="pick-type">${builder} ${parlay.numLegs}-Leg${parlay.hasSGP ? ' (SGP)' : ''}</span>
          <span class="pick-odds">${window.BettingEngine.formatOdds(parlay.odds)}</span>
        </div>
        <div class="parlay-legs">${legsHtml}</div>
        <div class="parlay-summary">
          <span>Combined Prob: ${(parlay.combinedHitRate * 100).toFixed(2)}%</span>
          <span class="parlay-ev">EV: ${parlay.ev > 0 ? '+' : ''}${(parlay.ev * 100).toFixed(1)}%</span>
          <span class="parlay-payout">$100 wins $${Math.round((parlay.decimalOdds - 1) * 100)}</span>
        </div>
      </div>`;
  }

  function renderFeatureTable(featureTable) {
    const container = document.getElementById('feature-table-container');
    if (!container) return;

    if (featureTable.length === 0) {
      container.innerHTML = '<p class="status-msg">No candidate legs available.</p>';
      return;
    }

    const headers = ['Player', 'Team', 'Opp', 'Market', 'Line', 'Odds', 'Role', 'Min Cert', 'L10 Hit', 'Model Prob', 'Edge', 'Fragility', 'Avg', 'Floor', 'EV'];
    const headerHtml = headers.map(h => `<th style="padding: 4px 6px; text-align: left; font-size: 0.7rem; white-space: nowrap;">${h}</th>`).join('');

    const rowsHtml = featureTable.slice(0, 100).map(r => {
      const edgeClass = r.edge >= 0.05 ? 'pnl-pos' : r.edge >= 0.02 ? '' : 'pnl-neg';
      return `<tr style="font-size: 0.7rem;">
        <td style="padding: 3px 6px; white-space: nowrap;">${r.player}</td>
        <td style="padding: 3px 6px;">${r.team}</td>
        <td style="padding: 3px 6px;">${r.opponent}</td>
        <td style="padding: 3px 6px;">${r.marketLabel}</td>
        <td style="padding: 3px 6px;">${r.line}</td>
        <td style="padding: 3px 6px;">${window.BettingEngine.formatOdds(r.bookOdds)}</td>
        <td style="padding: 3px 6px; font-size: 0.65rem;">${r.roleTag}</td>
        <td style="padding: 3px 6px;">${((r.minutesCertainty || 0) * 100).toFixed(0)}%</td>
        <td style="padding: 3px 6px;">${((r.l10Hit || 0) * 100).toFixed(0)}%</td>
        <td style="padding: 3px 6px;">${((r.modelProb || 0) * 100).toFixed(1)}%</td>
        <td style="padding: 3px 6px;" class="${edgeClass}">${((r.edge || 0) * 100).toFixed(1)}%</td>
        <td style="padding: 3px 6px;">${((r.fragility || 0) * 100).toFixed(0)}%</td>
        <td style="padding: 3px 6px;">${r.avg || '--'}</td>
        <td style="padding: 3px 6px;">${r.floor || '--'}</td>
        <td style="padding: 3px 6px;">${((r.ev || 0) * 100).toFixed(1)}%</td>
      </tr>`;
    }).join('');

    container.innerHTML = `
      <table style="border-collapse: collapse; width: 100%; background: var(--card-bg, #1a1a2e); border-radius: 8px;">
        <thead><tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">${headerHtml}</tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
      ${featureTable.length > 100 ? `<p style="font-size: 0.7rem; opacity: 0.5; margin-top: 4px;">Showing top 100 of ${featureTable.length} candidate legs</p>` : ''}`;
  }
})();
