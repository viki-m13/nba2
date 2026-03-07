// =============================================================================
// NBA Alt-Line Engine — App Controller v5.0
// =============================================================================

(function () {
  'use strict';

  let currentPeriod = 'all';
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
    runAltBacktest();
    renderDashboard();
    renderHistory();

    // Resolve any unresolved live picks
    try {
      livePicksData = await window.BettingEngine.resolveLivePicks();
    } catch (e) {
      console.warn('Could not resolve live picks:', e);
      livePicksData = window.BettingEngine.loadLivePicks();
    }
    renderLiveHistory();

    await generateTodayPicks();
  }

  // --- Data Loading ---

  async function loadData() {
    try {
      const [boxRes, oddsRes, rebAstRes, livePicksRes] = await Promise.all([
        fetch('data/player_boxscores.json'),
        fetch('data/historical_odds.json'),
        fetch('data/historical_reb_ast_props.json').catch(() => null),
        fetch('data/live_picks_2026.json').catch(() => null),
      ]);

      playerBoxScores = await boxRes.json();
      historicalOdds = await oddsRes.json();
      if (rebAstRes && rebAstRes.ok) rebAstProps = await rebAstRes.json();
      if (livePicksRes && livePicksRes.ok) {
        window.BettingEngine.setBasePicksData(await livePicksRes.json());
      }

      console.log(`Loaded: ${playerBoxScores.length} box scores, ${historicalOdds.length} odds`);
    } catch (e) {
      console.error('Error loading data:', e);
    }
  }

  // --- Alt-Line Backtest ---

  function runAltBacktest() {
    if (!window.NbaAltBacktest || !playerBoxScores.length) return;
    console.log('[APP] Running alt-line walk-forward backtest...');
    try {
      altBacktestResults = window.NbaAltBacktest.runBacktest(playerBoxScores, historicalOdds, rebAstProps);
      if (altBacktestResults && altBacktestResults.stats) {
        const s = altBacktestResults.stats;
        console.log(`[ALT-BACKTEST] Legs: ${s.overall.legs.total}, ${(s.overall.legs.hitRate*100).toFixed(1)}% hit, ROI ${s.overall.legs.roi}%`);
        console.log(`[ALT-BACKTEST] Parlays: ${s.overall.parlays.total}, ${(s.overall.parlays.hitRate*100).toFixed(1)}% win, ROI ${s.overall.parlays.roi}%`);
      }
    } catch (e) {
      console.warn('[APP] Alt backtest error:', e);
    }
  }

  // --- Today's Picks ---

  async function generateTodayPicks() {
    const statusEl = document.getElementById('today-status');

    try {
      statusEl.textContent = 'Fetching FanDuel odds...';

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

      // Fetch live odds from Odds API — primary game source
      let liveOdds = null;
      try {
        liveOdds = await window.BettingEngine.fetchLiveOdds();
      } catch (e) {
        console.warn('Could not fetch FanDuel odds:', e);
      }

      if (liveOdds && liveOdds.quotaExhausted) {
        statusEl.textContent = 'Odds API quota exhausted. Will resume when quota resets.';
        statusEl.style.color = 'var(--gold)';
        return;
      }

      // Build today's games from Odds API events
      let todayGames = [];
      if (liveOdds && liveOdds.events && liveOdds.events.length > 0) {
        todayGames = liveOdds.events.map(e => ({
          homeTeam: window.BettingEngine.teamAbbr(e.home_team),
          awayTeam: window.BettingEngine.teamAbbr(e.away_team),
          gameTime: e.commence_time,
          status: 'scheduled',
          statusText: '',
        }));
      }

      // Try scoreboard for live scores (optional, non-blocking)
      try {
        const sbResult = await window.NbaApi.fetchScoreboard();
        const sbGames = (sbResult && sbResult.games) || [];
        for (const sb of sbGames) {
          const match = todayGames.find(g => g.homeTeam === sb.homeTeam && g.awayTeam === sb.awayTeam);
          if (match) {
            match.homeScore = sb.homeScore;
            match.awayScore = sb.awayScore;
            match.status = sb.status;
            match.statusText = sb.statusText;
            match.ouLine = sb.ouLine;
          }
        }
        if (todayGames.length === 0 && sbGames.length > 0) todayGames = sbGames;
      } catch (e) {
        console.warn('Scoreboard fetch failed (non-critical):', e);
      }

      if (todayGames.length === 0) {
        statusEl.textContent = 'No games scheduled today.';
        renderTodayGames([]);
        return;
      }

      statusEl.textContent = `Found ${todayGames.length} games. Analyzing player props...`;
      renderTodayGames(todayGames);

      if (!liveOdds || Object.keys(liveOdds.playerProps).length === 0) {
        statusEl.textContent = 'No FanDuel player props available. Check back closer to game time.';
        return;
      }

      // --- Alt-Line Pipeline ---
      window.BettingEngine.PlayerModel.history = model.history;

      const gameOddsMap = {};
      try {
        const espnData = await window.NbaApi.fetchESPNOdds();
        for (const game of todayGames) {
          const gk = `${game.awayTeam}@${game.homeTeam}`;
          const espnKey = `${game.homeTeam}-${game.awayTeam}`;
          if (game.ouLine || (espnData && espnData.odds && espnData.odds[espnKey])) {
            gameOddsMap[gk] = {
              total: game.ouLine || espnData.odds[espnKey] || 220,
              spread_point: 0,
            };
          }
        }
      } catch (e) { console.warn('Could not build game odds map:', e); }

      const featureTable = window.BettingEngine.generateFeatureTable(liveOdds, gameOddsMap);
      const slips = window.BettingEngine.generateSlips(featureTable);

      renderAltSlips(slips, featureTable);

      // Save to live tracking
      const allParlays = [
        ...(slips.core || []),
        ...(slips.screenshot || []),
        ...(slips.nuke || []),
      ];
      const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      window.BettingEngine.saveTodayParlays(allParlays, [], today);
      livePicksData = window.BettingEngine.loadLivePicks();
      renderLiveHistory();

      const totalSlips = (slips.core || []).length + (slips.screenshot || []).length + (slips.nuke || []).length;
      if (totalSlips === 0) {
        statusEl.textContent = `Analyzed ${featureTable.length} candidate legs across ${todayGames.length} games. No parlays pass all filters today.`;
      } else {
        statusEl.style.display = 'none';
      }
    } catch (e) {
      console.error('Error generating picks:', e);
      statusEl.textContent = 'Error loading picks. See console.';
    }
  }

  // --- Rendering ---

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
      const away = g.awayTeam || '???';
      const home = g.homeTeam || '???';
      let timeInfo = '';
      if (g.status === 'final') timeInfo = `Final: ${g.awayScore}-${g.homeScore}`;
      else if (g.status === 'live' || g.status === 'halftime') timeInfo = g.statusText || 'Live';
      else if (g.gameTime) { try { timeInfo = new Date(g.gameTime).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' }); } catch (e) { timeInfo = 'TBD'; } }
      else timeInfo = g.statusText || 'TBD';

      return `<div class="today-game-card"><span class="today-game-teams">${away} @ ${home}</span><span class="today-game-time">${timeInfo}</span></div>`;
    }).join('');

    container.innerHTML = `<div class="today-games-header"><h3>Today's Games</h3><span class="today-games-date">${dateStr}</span></div><div class="today-games-grid">${gamesHtml}</div>`;
  }

  // --- Dashboard (Alt-Line Stats) ---

  function renderDashboard() {
    if (!altBacktestResults || !altBacktestResults.stats) return;
    const s = altBacktestResults.stats;
    const legs = s.overall.legs;
    const parlays = s.overall.parlays;

    document.getElementById('stat-accuracy').textContent = `${parlays.total > 0 ? (parlays.hitRate * 100).toFixed(1) : '--'}%`;
    document.getElementById('stat-record').textContent = `${parlays.wins}-${parlays.losses}`;
    document.getElementById('stat-pnl').textContent = `$${parlays.pnl >= 0 ? '+' : ''}${parlays.pnl}`;
    document.getElementById('stat-roi').textContent = `${parlays.roi >= 0 ? '+' : ''}${parlays.roi}%`;
    document.getElementById('stat-leg-record').textContent = `${legs.wins}-${legs.losses}`;
    document.getElementById('stat-leg-hit').textContent = `${legs.total > 0 ? (legs.hitRate * 100).toFixed(1) : '--'}%`;
    document.getElementById('stat-days').textContent = `${altBacktestResults.dates.length}`;
  }

  // --- History (Alt-Line Only) ---

  function renderHistory() {
    renderAltBacktestHistory();
  }

  function renderAltBacktestHistory() {
    const section = document.getElementById('alt-backtest-section');
    if (!section) return;

    if (!altBacktestResults || !altBacktestResults.stats) {
      section.style.display = 'none';
      return;
    }
    section.style.display = '';

    const s = altBacktestResults.stats;
    const legs = s.overall.legs;
    const parlays = s.overall.parlays;

    document.getElementById('alt-backtest-stats').innerHTML = `
      <div class="history-stat-row">
        <div class="history-stat"><span class="history-stat-value">${legs.total > 0 ? (legs.hitRate * 100).toFixed(1) : '--'}%</span><span class="history-stat-label">Leg Hit Rate (${legs.total} legs)</span></div>
        <div class="history-stat"><span class="history-stat-value">${parlays.total > 0 ? (parlays.hitRate * 100).toFixed(1) : '--'}%</span><span class="history-stat-label">Parlay Win Rate</span></div>
        <div class="history-stat"><span class="history-stat-value">${parlays.wins}-${parlays.losses}</span><span class="history-stat-label">Parlay Record</span></div>
        <div class="history-stat"><span class="history-stat-value ${parlays.pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">$${parlays.pnl >= 0 ? '+' : ''}${parlays.pnl}</span><span class="history-stat-label">P&L ($100/bet)</span></div>
        <div class="history-stat"><span class="history-stat-value">${parlays.roi >= 0 ? '+' : ''}${parlays.roi}%</span><span class="history-stat-label">ROI</span></div>
      </div>`;

    // Breakdown by builder, market, role
    let breakdownHtml = '<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; font-size: 0.8rem;">';

    breakdownHtml += '<div><strong style="color: var(--text-dim);">By Builder</strong>';
    for (const [k, v] of Object.entries(s.byBuilder)) {
      if (v.total === 0) continue;
      const cls = v.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      breakdownHtml += `<div>${k}: ${v.wins}-${v.losses} (${(v.hitRate*100).toFixed(0)}%) <span class="${cls}">$${v.pnl >= 0 ? '+' : ''}${v.pnl}</span></div>`;
    }
    breakdownHtml += '</div>';

    breakdownHtml += '<div><strong style="color: var(--text-dim);">By Market</strong>';
    for (const [k, v] of Object.entries(s.byMarket)) {
      if (v.total === 0) continue;
      const label = k.replace('player_', '').replace('_alternate', '').toUpperCase();
      breakdownHtml += `<div>${label}: ${v.wins}-${v.losses} (${(v.hitRate*100).toFixed(0)}%)</div>`;
    }
    breakdownHtml += '</div>';

    breakdownHtml += '<div><strong style="color: var(--text-dim);">By Role</strong>';
    for (const [k, v] of Object.entries(s.byRole)) {
      if (v.total === 0) continue;
      const cls = v.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      breakdownHtml += `<div>${k.replace(/_/g, ' ')}: ${(v.hitRate*100).toFixed(0)}% <span class="${cls}">${v.roi >= 0 ? '+' : ''}${v.roi}%</span></div>`;
    }
    breakdownHtml += '</div>';

    breakdownHtml += '</div>';
    document.getElementById('alt-backtest-breakdown').innerHTML = breakdownHtml;

    // Parlay rows (filtered)
    const altParlays = altBacktestResults.parlays || [];
    let filtered = altParlays;
    if (currentPeriod !== 'all') {
      const days = parseInt(currentPeriod);
      const allDates = [...new Set(filtered.map(p => p.date))].sort();
      const cutoff = allDates[Math.max(0, allDates.length - days)] || '';
      filtered = filtered.filter(p => p.date >= cutoff);
    }
    document.querySelector('#alt-parlays-history tbody').innerHTML = renderAltParlayRows(
      filtered.sort((a, b) => b.date.localeCompare(a.date))
    );
  }

  function renderAltParlayRows(list) {
    return list.map(p => {
      const numLegs = p.numLegs || p.legs.length;
      const legsHtml = p.legs.map(l => {
        const lbl = l.marketLabel || window.BettingEngine.statLabel(l.statType || l.marketType || 'points');
        const hasResult = l.actual !== null && l.actual !== undefined;
        const cls = hasResult ? (l.won ? 'result-win' : 'result-loss') : '';
        const txt = hasResult ? `${l.actual} ${lbl.toLowerCase()}` : 'Pending';
        const roleTag = l.roleTag ? `<span style="font-size: 0.65rem; opacity: 0.5;"> [${l.roleTag.replace(/_/g, ' ')}]</span>` : '';
        const probEdge = l.modelProb ? `<span style="font-size: 0.65rem; opacity: 0.5;"> prob ${(l.modelProb*100).toFixed(0)}% edge ${((l.edge||0)*100).toFixed(1)}%</span>` : '';
        return `<div class="history-leg">
          <span class="history-leg-player">${l.player} (${l.team})${roleTag}</span>
          <span class="history-leg-stat">${lbl}</span>
          <span class="history-leg-line">OVER ${l.line}</span>
          <span class="history-leg-odds">${window.BettingEngine.formatOdds(l.bookOdds || l.odds)}</span>
          <span class="history-leg-game">${l.gameDisplay || (l.gameKey || '').replace('@', ' @ ')}</span>
          <span class="history-leg-actual ${cls}">${txt}${probEdge}</span>
        </div>`;
      }).join('');

      const builder = p.builder || 'PARLAY';
      const builderClass = builder === 'CORE' ? 'builder-core' :
                           builder === 'SCREENSHOT' ? 'builder-screenshot' :
                           builder === 'NUKE' ? 'builder-nuke' : '';
      const isResolved = p.resolved !== undefined ? p.resolved : true;
      const resultText = isResolved ? (p.won ? 'WIN' : 'LOSS') : 'PENDING';
      const resultClass = isResolved ? (p.won ? 'result-win' : 'result-loss') : 'result-pending';
      const pnlText = isResolved ? `${p.pnl >= 0 ? '+' : ''}$${p.pnl}` : '--';
      const pnlClass = isResolved ? (p.pnl >= 0 ? 'pnl-pos' : 'pnl-neg') : '';

      return `<tr>
        <td>${formatDate(p.date)}</td>
        <td><span class="${builderClass}" style="font-weight: 600; font-size: 0.8rem;">${builder}</span> ${numLegs}-Leg</td>
        <td class="legs-cell">${legsHtml}</td>
        <td>${window.BettingEngine.formatOdds(p.odds)}</td>
        <td class="${resultClass}">${resultText}</td>
        <td class="${pnlClass}">${pnlText}</td>
      </tr>`;
    }).join('');
  }

  // --- Live Tracking History ---

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
      <table class="history-table"><thead><tr><th>Date</th><th>Builder</th><th>Legs</th><th>Parlay Odds</th><th>Result</th><th>P&L</th></tr></thead>
        <tbody>${renderAltParlayRows(liveParlays.sort((a, b) => b.date.localeCompare(a.date)))}</tbody>
      </table>`;

    const historyView = document.getElementById('view-history');
    const old = historyView.querySelector('.live-tracking-section');
    if (old) old.remove();
    historyView.appendChild(section);
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
  // Alt-Line Slip Rendering (Today's Picks)
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
