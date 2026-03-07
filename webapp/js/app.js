// =============================================================================
// NBA Prop Picks — App Controller (Multi-Stat Parlays)
// =============================================================================

(function () {
  'use strict';

  let backtestResults = null;
  let spBacktestResults = null; // Super Payout backtest
  let todaySingles = [];
  let todayParlays = [];
  let todaySPParlays = []; // Super Payout today's picks
  let currentPeriod = 'all';
  let seasonData = [];
  let playerBoxScores = [];
  let historicalOdds = [];
  let rebAstProps = [];
  let livePicksData = null; // 2025-26 live tracked picks

  // --- Initialization ---

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    setupNavigation();
    setupFilters();
    await loadData();
    await loadIncrementalOdds();
    runBacktest();

    // Resolve any unresolved live picks from past days
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

  // --- Data Loading ---

  async function loadData() {
    try {
      const [seasonRes, boxRes, oddsRes, rebAstRes] = await Promise.all([
        fetch('data/espn_full_season_2025.json'),
        fetch('data/player_boxscores.json'),
        fetch('data/historical_odds.json'),
        fetch('data/historical_reb_ast_props.json').catch(() => null),
      ]);

      seasonData = await seasonRes.json();
      playerBoxScores = await boxRes.json();
      historicalOdds = await oddsRes.json();
      if (rebAstRes && rebAstRes.ok) {
        rebAstProps = await rebAstRes.json();
      }

      console.log(`Loaded: ${seasonData.length} games, ${playerBoxScores.length} box scores, ${historicalOdds.length} historical odds, ${rebAstProps.length} reb/ast props`);
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

    backtestResults = window.BettingEngine.runBacktest(seasonData, playerBoxScores, historicalOdds, rebAstProps);
    console.log('Backtest results:', backtestResults.stats);

    // Run Super Payout backtest (points-only, separate engine)
    if (window.SuperPayoutEngine) {
      spBacktestResults = window.SuperPayoutEngine.runBacktest(seasonData, playerBoxScores, historicalOdds);
      console.log('Super Payout backtest:', spBacktestResults.stats);
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
        renderTodayGames([]); // Show empty state
        return;
      }

      statusEl.textContent = `Found ${todayGames.length} games. Fetching FanDuel odds (points, rebounds, assists)...`;
      renderTodayGames(todayGames);

      // Check data freshness — warn if box scores are from a different season
      const sorted = [...playerBoxScores].sort((a, b) => a.date.localeCompare(b.date));
      const latestBoxDate = sorted.length > 0 ? sorted[sorted.length - 1].date : '';
      const todayStr = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      const daysSinceLastBox = latestBoxDate ? Math.floor(
        (new Date(todayStr.slice(0,4) + '-' + todayStr.slice(4,6) + '-' + todayStr.slice(6,8))
        - new Date(latestBoxDate.slice(0,4) + '-' + latestBoxDate.slice(4,6) + '-' + latestBoxDate.slice(6,8)))
        / (1000 * 60 * 60 * 24)
      ) : 999;

      if (daysSinceLastBox > 60) {
        console.warn(`[APP] Box score data is ${daysSinceLastBox} days old (last: ${latestBoxDate}). Model uses prior season stats.`);
        const warnEl = document.createElement('div');
        warnEl.className = 'status-msg';
        warnEl.style.color = 'var(--gold)';
        warnEl.textContent = `Note: Player stats are from the ${latestBoxDate.slice(0,4)}-${parseInt(latestBoxDate.slice(0,4))+1} season. Picks use live FanDuel odds but evaluate against last season's performance data.`;
        const parlaysEl = document.getElementById('today-parlays');
        parlaysEl.parentNode.insertBefore(warnEl, parlaysEl);
      }

      // Train the model on all historical data
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

      // Fetch live FanDuel odds (now includes rebounds + assists)
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

      // Evaluate props for each game across all stat types and both tiers
      todaySingles = [];
      for (const [gameKey, gameData] of Object.entries(liveOdds.playerProps)) {
        const gameDisplay = `${gameData.away} @ ${gameData.home}`;

        const addPick = (prop) => {
          if (!prop) return;
          // Avoid duplicate player+stat+line
          if (!todaySingles.find(s => s.player === prop.player && s.statType === prop.statType && s.line === prop.line)) {
            todaySingles.push({ ...prop, gameKey, gameDisplay });
          }
        };

        // Points (Tier 1 + Tier 2)
        for (const [playerName, lines] of Object.entries(gameData.lines || {})) {
          addPick(model.findBestStatProp(playerName, 'points', lines));
          addPick(model.findBestT2Prop(playerName, 'points', lines));
        }

        // Rebounds (Tier 1 + Tier 2)
        for (const [playerName, lines] of Object.entries(gameData.rebLines || {})) {
          addPick(model.findBestStatProp(playerName, 'rebounds', lines));
          addPick(model.findBestT2Prop(playerName, 'rebounds', lines));
        }

        // Assists (Tier 1 + Tier 2)
        for (const [playerName, lines] of Object.entries(gameData.astLines || {})) {
          addPick(model.findBestStatProp(playerName, 'assists', lines));
          addPick(model.findBestT2Prop(playerName, 'assists', lines));
        }
      }

      // Sort by confidence
      todaySingles.sort((a, b) => b.confidence - a.confidence);

      // Build parlays (now with multi-stat diversity)
      todayParlays = window.BettingEngine.buildParlays(todaySingles);

      // Add gameDisplay to parlay legs
      for (const parlay of todayParlays) {
        parlay.legs = parlay.legs.map(leg => {
          const matchingSingle = todaySingles.find(s =>
            s.player === leg.player && s.line === leg.line && s.statType === leg.statType
          );
          return {
            ...leg,
            gameKey: matchingSingle ? matchingSingle.gameKey : '',
            gameDisplay: matchingSingle ? matchingSingle.gameDisplay : '',
          };
        });
      }

      // --- Super Payout picks (points-only, separate model) ---
      todaySPParlays = [];
      if (window.SuperPayoutEngine) {
        const spModel = Object.create(window.SuperPayoutEngine.PlayerModel);
        spModel.history = {};
        for (const g of sorted) {
          for (const p of (g.players || [])) {
            const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
            if (mins < 10) continue;
            spModel.update(p.name, p.pts, mins, g.date, p.team);
          }
        }

        const spSingles = [];
        for (const [gameKey, gameData] of Object.entries(liveOdds.playerProps)) {
          const gameDisplay = `${gameData.away} @ ${gameData.home}`;
          for (const [playerName, lines] of Object.entries(gameData.lines || {})) {
            const prop = spModel.findBestPayoutProp(playerName, lines);
            if (prop) {
              spSingles.push({ ...prop, gameKey, gameDisplay });
            }
          }
        }

        if (spSingles.length >= 2) {
          todaySPParlays = window.SuperPayoutEngine.buildSuperParlays(spSingles);
          for (const parlay of todaySPParlays) {
            parlay.legs = parlay.legs.map(leg => {
              const match = spSingles.find(s => s.player === leg.player && s.line === leg.line);
              return {
                ...leg,
                gameKey: match ? match.gameKey : '',
                gameDisplay: match ? match.gameDisplay : '',
              };
            });
          }
        }
      }

      // Save parlays to live tracking (2025-26 season)
      const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      window.BettingEngine.saveTodayParlays(todayParlays, todaySPParlays, today);
      livePicksData = window.BettingEngine.loadLivePicks();
      renderLiveHistory();

      // Render
      if (todayParlays.length === 0 && todaySPParlays.length === 0) {
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
    let html = '';

    if (todayParlays.length > 0) {
      html += todayParlays.map(renderParlayCard).join('');
    }

    if (todaySPParlays.length > 0) {
      html += `<div class="sp-section-header" style="grid-column: 1 / -1;">
        <h3>SUPER PAYOUT Parlays</h3>
        <span class="sp-badge">Points-Only High Odds</span>
      </div>`;
      html += todaySPParlays.map(renderSuperPayoutCard).join('');
    }

    parlaysEl.innerHTML = html;
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
          <span class="parlay-leg-stat">${leg.statLabel || 'PTS'}</span>
          <span class="parlay-leg-line">OVER ${leg.line}</span>
          <span class="parlay-leg-odds">${window.BettingEngine.formatOdds(leg.odds)}</span>
        </div>
      </div>`).join('');

    const today = new Date();
    const dateStr = `${String(today.getMonth()+1).padStart(2,'0')}/${String(today.getDate()).padStart(2,'0')}/${today.getFullYear()}`;

    // Count stat types in this parlay
    const statTypes = new Set(parlay.legs.map(l => l.statType || 'points'));
    const isMultiStat = statTypes.size > 1;
    const parlayLabel = isMultiStat
      ? `${parlay.numLegs}-Leg Multi-Stat Parlay`
      : `${parlay.numLegs}-Leg Parlay`;

    return `
      <div class="pick-card parlay${isMultiStat ? ' multi-stat' : ''}">
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

  function renderSuperPayoutCard(parlay) {
    const legsHtml = parlay.legs.map(leg => `
      <div class="parlay-leg">
        <div class="parlay-leg-info">
          <span class="parlay-leg-player">${leg.player} (${leg.team})</span>
          <span class="parlay-leg-game">${leg.gameDisplay || formatGameKey(leg.gameKey)}</span>
        </div>
        <div class="parlay-leg-right">
          <span class="parlay-leg-stat">PTS</span>
          <span class="parlay-leg-line">OVER ${leg.line}</span>
          <span class="parlay-leg-odds">${window.SuperPayoutEngine.formatOdds(leg.odds)}</span>
        </div>
      </div>`).join('');

    const today = new Date();
    const dateStr = `${String(today.getMonth()+1).padStart(2,'0')}/${String(today.getDate()).padStart(2,'0')}/${today.getFullYear()}`;

    return `
      <div class="pick-card super-payout">
        <div class="pick-header">
          <span class="pick-date">${dateStr}</span>
          <span class="pick-type">${parlay.numLegs}-Leg Super Payout <span class="sp-badge">SP</span></span>
          <span class="pick-odds">${window.SuperPayoutEngine.formatOdds(parlay.odds)}</span>
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

    const today = new Date();
    const dateStr = today.toLocaleDateString('en-US', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
    });

    if (!games || games.length === 0) {
      container.innerHTML = `
        <div class="today-games-header">
          <h3>Today's Games</h3>
          <span class="today-games-date">${dateStr}</span>
        </div>
        <p class="status-msg">No games scheduled.</p>`;
      return;
    }

    const gamesHtml = games.map(g => {
      const away = g.awayTeam || g.away_team || '???';
      const home = g.homeTeam || g.home_team || '???';
      let timeInfo = '';
      if (g.status === 'final') {
        timeInfo = `Final: ${g.awayScore}-${g.homeScore}`;
      } else if (g.status === 'live' || g.status === 'halftime') {
        timeInfo = g.statusText || 'Live';
      } else if (g.gameTime) {
        // Parse ET time
        try {
          const gt = new Date(g.gameTime);
          timeInfo = gt.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
        } catch (e) {
          timeInfo = g.statusText || 'TBD';
        }
      } else {
        timeInfo = g.statusText || 'TBD';
      }

      return `<div class="today-game-card">
        <span class="today-game-teams">${away} @ ${home}</span>
        <span class="today-game-time">${timeInfo}</span>
      </div>`;
    }).join('');

    container.innerHTML = `
      <div class="today-games-header">
        <h3>Today's Games</h3>
        <span class="today-games-date">${dateStr}</span>
      </div>
      <div class="today-games-grid">${gamesHtml}</div>`;
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

    // Recent accuracy
    if (s.recent30) {
      const recentEl = document.getElementById('stat-recent');
      if (recentEl) {
        recentEl.textContent = `${(s.recent30.hitRate * 100).toFixed(1)}%`;
      }
    }

    // Super Payout stats
    if (spBacktestResults) {
      const sp = spBacktestResults.stats.parlays;
      const spRecordEl = document.getElementById('stat-sp-record');
      const spPnlEl = document.getElementById('stat-sp-pnl');
      if (spRecordEl) spRecordEl.textContent = `${sp.wins}-${sp.losses}`;
      if (spPnlEl) spPnlEl.textContent = `$${sp.pnl >= 0 ? '+' : ''}${sp.pnl}`;
    }
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

    // Rolling recent accuracy (last 14 days of filtered data)
    const allDates = [...new Set(parlays.map(p => p.date))].sort();
    const recent14Cutoff = allDates[Math.max(0, allDates.length - 14)] || '';
    const recent14 = parlays.filter(p => p.date >= recent14Cutoff);
    const r14Wins = recent14.filter(p => p.won).length;
    const r14Rate = recent14.length > 0 ? (r14Wins / recent14.length * 100).toFixed(1) : '--';

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
          <span class="history-stat-value highlight-recent">${r14Rate}%</span>
          <span class="history-stat-label">Last 14 Days</span>
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

    // Super Payout stats
    const { spParlays } = getFilteredResults();
    if (spParlays.length > 0) {
      const spWins = spParlays.filter(p => p.won).length;
      const spTotal = spParlays.length;
      const spPnl = spParlays.reduce((s, p) => s + p.pnl, 0);
      const spRate = (spWins / spTotal * 100).toFixed(1);
      const spAvgOdds = Math.round(spParlays.reduce((s, p) => s + p.odds, 0) / spTotal);
      statsEl.innerHTML += `
        <div class="history-stat-row" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border);">
          <div class="history-stat">
            <span class="history-stat-value" style="color: var(--gold);">${spRate}%</span>
            <span class="history-stat-label">Super Payout Accuracy</span>
          </div>
          <div class="history-stat">
            <span class="history-stat-value" style="color: var(--gold);">${spWins}-${spTotal - spWins}</span>
            <span class="history-stat-label">SP Record</span>
          </div>
          <div class="history-stat">
            <span class="history-stat-value ${spPnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">$${spPnl >= 0 ? '+' : ''}${spPnl}</span>
            <span class="history-stat-label">SP P&L</span>
          </div>
          <div class="history-stat">
            <span class="history-stat-value" style="color: var(--gold);">${spAvgOdds > 0 ? '+' : ''}${spAvgOdds}</span>
            <span class="history-stat-label">SP Avg Odds</span>
          </div>
        </div>`;
    }

    // Parlays table
    const parlaysBody = document.querySelector('#parlays-history tbody');
    const allHistoryParlays = [
      ...parlays.map(p => ({ ...p, source: 'standard' })),
      ...spParlays.map(p => ({ ...p, source: 'sp' })),
    ].sort((a, b) => b.date.localeCompare(a.date));

    parlaysBody.innerHTML = renderParlayRows(allHistoryParlays);

    // --- Live Tracked Picks (2025-26 Season) ---
    renderLiveHistory();
  }

  function renderParlayRows(parlayList) {
    return parlayList.map(p => {
      const numLegs = p.numLegs || p.legs.length;
      const legsHtml = p.legs.map(l => {
        const game = l.gameDisplay || formatGameKey(l.gameKey);
        const statLbl = l.statLabel || window.BettingEngine.statLabel(l.statType || 'points');
        const hasResult = l.actual !== null && l.actual !== undefined;
        const resultClass = hasResult ? (l.won ? 'result-win' : 'result-loss') : '';
        const actualText = hasResult ? `${l.actual} ${statLbl.toLowerCase()}` : 'Pending';
        return `<div class="history-leg">
          <span class="history-leg-player">${l.player} (${l.team})</span>
          <span class="history-leg-stat">${statLbl}</span>
          <span class="history-leg-line">OVER ${l.line}</span>
          <span class="history-leg-odds">${window.BettingEngine.formatOdds(l.odds)}</span>
          <span class="history-leg-game">${game}</span>
          <span class="history-leg-actual ${resultClass}">${actualText}</span>
        </div>`;
      }).join('');

      const statTypes = new Set(p.legs.map(l => l.statType || 'points'));
      const isSP = p.source === 'sp' || p.isSuperPayout;
      const typeLabel = isSP
        ? `${numLegs}-Leg SP`
        : (statTypes.size > 1 ? `${numLegs}-Leg Multi` : `${numLegs}-Leg`);

      const isResolved = p.resolved !== undefined ? p.resolved : true;
      const resultText = isResolved ? (p.won ? 'WIN' : 'LOSS') : 'PENDING';
      const resultClass = isResolved ? (p.won ? 'result-win' : 'result-loss') : 'result-pending';
      const pnlText = isResolved ? `${p.pnl >= 0 ? '+' : ''}$${p.pnl}` : '--';
      const pnlClass = isResolved ? (p.pnl >= 0 ? 'pnl-pos' : 'pnl-neg') : '';

      return `
        <tr${isSP ? ' style="background: rgba(245, 158, 11, 0.04);"' : ''}>
          <td>${formatDate(p.date)}</td>
          <td>${typeLabel}${isSP ? ' <span class="sp-badge">SP</span>' : ''}</td>
          <td class="legs-cell">${legsHtml}</td>
          <td>${window.BettingEngine.formatOdds(p.odds)}</td>
          <td class="${resultClass}">${resultText}</td>
          <td class="${pnlClass}">${pnlText}</td>
        </tr>`;
    }).join('');
  }

  function renderLiveHistory() {
    if (!livePicksData) return;

    const liveParlays = [
      ...livePicksData.parlays.map(p => ({ ...p, source: 'standard' })),
      ...livePicksData.spParlays.map(p => ({ ...p, source: 'sp' })),
    ];

    if (liveParlays.length === 0) return;

    // Stats for resolved picks only
    const resolved = liveParlays.filter(p => p.resolved);
    const liveWins = resolved.filter(p => p.won).length;
    const liveTotal = resolved.length;
    const livePnl = resolved.reduce((s, p) => s + (p.pnl || 0), 0);
    const pending = liveParlays.filter(p => !p.resolved).length;

    const liveStatsHtml = liveTotal > 0 ? `
      <div class="history-stat-row" style="margin-top: 0.5rem;">
        <div class="history-stat">
          <span class="history-stat-value">${liveTotal > 0 ? ((liveWins / liveTotal * 100).toFixed(1)) : '--'}%</span>
          <span class="history-stat-label">Live Accuracy</span>
        </div>
        <div class="history-stat">
          <span class="history-stat-value">${liveWins}-${liveTotal - liveWins}</span>
          <span class="history-stat-label">Record</span>
        </div>
        <div class="history-stat">
          <span class="history-stat-value ${livePnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">$${livePnl >= 0 ? '+' : ''}${livePnl}</span>
          <span class="history-stat-label">P&L</span>
        </div>
        ${pending > 0 ? `<div class="history-stat">
          <span class="history-stat-value" style="color: var(--text-dim);">${pending}</span>
          <span class="history-stat-label">Pending</span>
        </div>` : ''}
      </div>` : `<p class="status-msg" style="margin: 0.5rem 0;">No resolved picks yet. Results update automatically after games finish.</p>`;

    // Build the live section HTML
    const liveSection = document.createElement('div');
    liveSection.className = 'history-section';
    liveSection.innerHTML = `
      <h3 style="margin-top: 1.5rem; border-top: 2px solid var(--accent); padding-top: 1rem;">
        2025-26 Live Tracking
        <span style="font-size: 0.75rem; color: var(--text-dim); font-weight: 400;">Auto-resolves daily</span>
      </h3>
      <div class="history-stats">${liveStatsHtml}</div>
      <table class="history-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Type</th>
            <th>Legs</th>
            <th>Parlay Odds</th>
            <th>Result</th>
            <th>P&L</th>
          </tr>
        </thead>
        <tbody>${renderParlayRows(liveParlays.sort((a, b) => b.date.localeCompare(a.date)))}</tbody>
      </table>`;

    // Insert after the backtest history section
    const historyView = document.getElementById('view-history');
    // Remove old live section if re-rendering
    const oldLive = historyView.querySelector('.live-tracking-section');
    if (oldLive) oldLive.remove();

    liveSection.classList.add('live-tracking-section');
    historyView.appendChild(liveSection);
  }

  function getFilteredResults() {
    if (!backtestResults) return { parlays: [], spParlays: [] };

    let parlays = backtestResults.parlays;
    let spParlays = spBacktestResults ? spBacktestResults.parlays : [];

    if (currentPeriod !== 'all') {
      const days = parseInt(currentPeriod);
      const allDates = [...new Set([...parlays, ...spParlays].map(p => p.date))].sort();
      const cutoff = allDates[Math.max(0, allDates.length - days)] || '';
      parlays = parlays.filter(p => p.date >= cutoff);
      spParlays = spParlays.filter(p => p.date >= cutoff);
    }

    return { parlays, spParlays };
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
    return `${dateStr.slice(4, 6)}/${dateStr.slice(6, 8)}/${dateStr.slice(0, 4)}`;
  }
})();
