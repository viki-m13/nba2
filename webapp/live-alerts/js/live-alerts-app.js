/**
 * Live Alerts App — Webapp Controller with Live KDT Monitoring
 */
(function () {
  'use strict';

  const DATA_BASE = 'live-alerts/data';

  let laData = { signals: [], stats: {}, tiers: { tiers: [] } };
  let laView = 'live';
  let laFilter = 'all';
  let laInitialized = false;
  let liveState = null;
  let liveError = null;
  let engineReady = false;

  // ── Data Loading ─────────────────────────────────────────────
  async function loadLAData() {
    try {
      const [signals, stats, tiers] = await Promise.all([
        fetch(`${DATA_BASE}/mlb_live_signals.json`).then(r => r.ok ? r.json() : []),
        fetch(`${DATA_BASE}/mlb_live_stats.json`).then(r => r.ok ? r.json() : {}),
        fetch(`${DATA_BASE}/mlb_live_tiers.json`).then(r => r.ok ? r.json() : { tiers: [] }),
      ]);
      laData = { signals, stats, tiers };
    } catch (e) {
      console.warn('[LiveAlerts] Failed to load data:', e);
    }
  }

  // ── Main Render ──────────────────────────────────────────────
  function render() {
    const section = document.getElementById('live-alerts-section');
    if (!section) return;

    let html = '';

    // Nav tabs
    html += `<div style="display:flex;gap:0.5rem;justify-content:center;margin-bottom:1.5rem">
      <button class="nav-btn la-view-btn ${laView === 'live' ? 'active' : ''}" data-la-view="live">Live Monitor</button>
      <button class="nav-btn la-view-btn ${laView === 'tonight' ? 'active' : ''}" data-la-view="tonight">Backtest Tiers</button>
      <button class="nav-btn la-view-btn ${laView === 'history' ? 'active' : ''}" data-la-view="history">Full History</button>
      <button class="nav-btn la-view-btn ${laView === 'strategy' ? 'active' : ''}" data-la-view="strategy">Strategy</button>
    </div>`;

    if (laView === 'live') {
      html += renderLive();
    } else if (laView === 'tonight') {
      html += renderTiers();
    } else if (laView === 'history') {
      html += renderHistory();
    } else {
      html += renderStrategy();
    }

    section.innerHTML = html;
    bindLAEvents(section);
  }

  // ── LIVE MONITOR VIEW ────────────────────────────────────────
  function renderLive() {
    if (!engineReady) {
      return '<div style="text-align:center;color:var(--text-dim);padding:3rem">Starting KDT Live Engine...</div>';
    }
    if (liveError) {
      return `<div style="text-align:center;color:var(--red);padding:3rem">${liveError}</div>`;
    }
    if (!liveState) {
      return '<div style="text-align:center;color:var(--text-dim);padding:3rem">Polling ESPN for live games...</div>';
    }

    let html = '';
    const now = liveState.lastPoll ? new Date(liveState.lastPoll).toLocaleTimeString() : '...';

    // Status bar
    html += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;padding:0.5rem 1rem;background:var(--surface);border:1px solid var(--border);border-radius:8px">
      <div style="display:flex;align-items:center;gap:0.5rem">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;animation:pulse 2s infinite"></span>
        <span style="font-size:0.85rem;color:var(--text)">KDT Engine Active</span>
      </div>
      <div style="font-size:0.8rem;color:var(--text-dim)">Last poll: ${now} | Polling every 30s</div>
      <button class="nav-btn la-refresh-btn" style="font-size:0.75rem;padding:4px 10px">Refresh Now</button>
    </div>`;

    // Pulse animation
    html += `<style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}</style>`;

    // Active alerts section
    const alerts = liveState.activeAlerts || [];
    if (alerts.length > 0) {
      html += `<div style="margin-bottom:1.5rem">
        <h3 style="color:#f59e0b;margin:0 0 0.75rem;font-size:1rem">ACTIVE ALERTS (${alerts.length})</h3>`;
      for (const a of alerts) {
        html += renderAlertCard(a);
      }
      html += '</div>';
    }

    // Session alerts history
    const hist = liveState.alertHistory || [];
    const resolved = liveState.resolvedBets || [];
    if (hist.length > 0 || resolved.length > 0) {
      html += renderSessionStats(hist, resolved);
    }

    // Live scoreboard
    const live = liveState.live || [];
    const pre = liveState.pre || [];
    const post = liveState.post || [];

    if (live.length > 0) {
      html += `<h3 style="color:var(--text);margin:1.5rem 0 0.75rem;font-size:1rem">LIVE GAMES (${live.length})</h3>`;
      html += '<div style="display:grid;gap:0.75rem">';
      for (const g of live) { html += renderGameCard(g); }
      html += '</div>';
    }

    if (pre.length > 0) {
      html += `<h3 style="color:var(--text-dim);margin:1.5rem 0 0.75rem;font-size:0.9rem">UPCOMING (${pre.length})</h3>`;
      html += '<div style="display:grid;gap:0.5rem">';
      for (const g of pre) { html += renderPreGame(g); }
      html += '</div>';
    }

    if (post.length > 0) {
      html += `<h3 style="color:var(--text-dim);margin:1.5rem 0 0.75rem;font-size:0.9rem">FINAL (${post.length})</h3>`;
      html += '<div style="display:grid;gap:0.5rem">';
      for (const g of post) { html += renderPostGame(g); }
      html += '</div>';
    }

    if (live.length === 0 && pre.length === 0 && post.length === 0) {
      html += '<div style="text-align:center;color:var(--text-dim);padding:3rem">No MLB games on the schedule today.</div>';
    }

    return html;
  }

  function renderAlertCard(a) {
    const tier = a.tier || {};
    const color = tier.color || '#f59e0b';
    return `<div style="background:var(--surface);border:1px solid ${color};border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.5rem;border-left:4px solid ${color}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
        <div>
          <span style="color:${color};font-weight:700;font-size:0.9rem">${tier.name || tier.id}</span>
          <span style="color:var(--text-dim);font-size:0.8rem;margin-left:0.5rem">${a.score}</span>
        </div>
        <span style="font-size:0.75rem;color:var(--text-dim)">${new Date(a.timestamp).toLocaleTimeString()}</span>
      </div>
      <div style="display:flex;gap:1.5rem;font-size:0.9rem">
        <div><span style="color:var(--text-dim)">BUY:</span> <strong style="color:var(--text)">${a.team} ML</strong></div>
        <div><span style="color:var(--text-dim)">Price:</span> <strong style="color:var(--gold)">$${a.buyPrice.toFixed(3)}</strong></div>
        <div><span style="color:var(--text-dim)">KD:</span> <strong style="color:#22c55e">${(a.peakKd * 100).toFixed(1)}%</strong></div>
        <div><span style="color:var(--text-dim)">SWP:</span> <strong>${(a.swp * 100).toFixed(1)}%</strong></div>
        <div><span style="color:var(--text-dim)">Inn ${a.inning}, +${a.lead}</span></div>
      </div>
    </div>`;
  }

  function renderSessionStats(hist, resolved) {
    const wins = resolved.filter(r => r.won).length;
    const losses = resolved.filter(r => r.won === false).length;
    const pnl = resolved.reduce((s, r) => s + (r.pnl || 0), 0);
    const pending = hist.length - resolved.length;
    return `<div style="display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap">
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;text-align:center;flex:1;min-width:80px">
        <div style="font-size:1.1rem;font-weight:700;color:var(--gold)">${hist.length}</div>
        <div style="font-size:0.7rem;color:var(--text-dim)">Alerts Fired</div>
      </div>
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;text-align:center;flex:1;min-width:80px">
        <div style="font-size:1.1rem;font-weight:700;color:var(--accent)">${wins}W</div>
        <div style="font-size:0.7rem;color:var(--text-dim)">Won</div>
      </div>
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;text-align:center;flex:1;min-width:80px">
        <div style="font-size:1.1rem;font-weight:700;color:var(--red)">${losses}L</div>
        <div style="font-size:0.7rem;color:var(--text-dim)">Lost</div>
      </div>
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;text-align:center;flex:1;min-width:80px">
        <div style="font-size:1.1rem;font-weight:700;color:var(--blue)">${pending}</div>
        <div style="font-size:0.7rem;color:var(--text-dim)">Pending</div>
      </div>
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;text-align:center;flex:1;min-width:80px">
        <div style="font-size:1.1rem;font-weight:700;color:${pnl >= 0 ? 'var(--accent)' : 'var(--red)'}">$${pnl >= 0 ? '+' : ''}${Math.round(pnl)}</div>
        <div style="font-size:0.7rem;color:var(--text-dim)">Session P&L</div>
      </div>
    </div>`;
  }

  function renderGameCard(g) {
    const homeLead = g.homeScore - g.awayScore;
    const awayLead = g.awayScore - g.homeScore;
    const homeKd = g.homeKd ? g.homeKd.kd : 0;
    const awayKd = g.awayKd ? g.awayKd.kd : 0;
    const homeSwp = g.homeKd ? g.homeKd.swp : null;
    const awaySwp = g.awayKd ? g.awayKd.swp : null;

    const hasAlert = (g.newAlerts || []).length > 0;
    const borderColor = hasAlert ? '#f59e0b' : 'var(--border)';

    let kdBar = '';
    if (homeLead > 0 && homeKd > 0) {
      const pct = Math.min(homeKd * 100 / 20, 100);
      const kdColor = homeKd >= 0.10 ? '#22c55e' : '#8b8fa3';
      kdBar = `<div style="margin-top:0.5rem">
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:var(--text-dim)">
          <span>KD: ${(homeKd * 100).toFixed(1)}% (${g.homeTeam})</span>
          <span>SWP: ${homeSwp ? (homeSwp * 100).toFixed(1) + '%' : '-'} vs ESPN: ${g.homeWP ? (g.homeWP * 100).toFixed(1) + '%' : '-'}</span>
        </div>
        <div style="height:4px;background:var(--bg);border-radius:2px;margin-top:3px">
          <div style="height:100%;width:${pct.toFixed(0)}%;background:${kdColor};border-radius:2px;transition:width 0.5s"></div>
        </div>
      </div>`;
    } else if (awayLead > 0 && awayKd > 0) {
      const pct = Math.min(awayKd * 100 / 20, 100);
      const kdColor = awayKd >= 0.10 ? '#22c55e' : '#8b8fa3';
      kdBar = `<div style="margin-top:0.5rem">
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:var(--text-dim)">
          <span>KD: ${(awayKd * 100).toFixed(1)}% (${g.awayTeam})</span>
          <span>SWP: ${awaySwp ? (awaySwp * 100).toFixed(1) + '%' : '-'} vs ESPN: ${g.awayWP ? (g.awayWP * 100).toFixed(1) + '%' : '-'}</span>
        </div>
        <div style="height:4px;background:var(--bg);border-radius:2px;margin-top:3px">
          <div style="height:100%;width:${pct.toFixed(0)}%;background:${kdColor};border-radius:2px;transition:width 0.5s"></div>
        </div>
      </div>`;
    }

    return `<div style="background:var(--surface);border:1px solid ${borderColor};border-radius:10px;padding:0.75rem 1rem${hasAlert ? ';box-shadow:0 0 12px rgba(245,158,11,0.15)' : ''}">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div style="display:flex;align-items:center;gap:1rem;font-size:0.95rem">
          <span style="font-weight:600;color:var(--text);min-width:40px;text-align:right">${g.awayTeam}</span>
          <span style="color:var(--text);font-weight:700;font-size:1.1rem">${g.awayScore}</span>
          <span style="color:var(--text-dim);font-size:0.8rem">@</span>
          <span style="color:var(--text);font-weight:700;font-size:1.1rem">${g.homeScore}</span>
          <span style="font-weight:600;color:var(--text);min-width:40px">${g.homeTeam}</span>
        </div>
        <div style="text-align:right">
          <div style="font-size:0.85rem;color:var(--gold);font-weight:600">${g.statusDetail || 'Inn ' + g.inning}</div>
          ${g.homeWP != null ? `<div style="font-size:0.75rem;color:var(--text-dim)">WP: ${g.awayTeam} ${(g.awayWP * 100).toFixed(0)}% / ${g.homeTeam} ${(g.homeWP * 100).toFixed(0)}%</div>` : ''}
        </div>
      </div>
      ${kdBar}
    </div>`;
  }

  function renderPreGame(g) {
    return `<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;display:flex;justify-content:space-between;align-items:center">
      <span style="font-size:0.9rem;color:var(--text)">${g.awayTeam} @ ${g.homeTeam}</span>
      <span style="font-size:0.8rem;color:var(--text-dim)">${g.statusDetail || 'Scheduled'}</span>
    </div>`;
  }

  function renderPostGame(g) {
    return `<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;display:flex;justify-content:space-between;align-items:center;opacity:0.7">
      <div style="font-size:0.9rem;color:var(--text)">${g.awayTeam} <strong>${g.awayScore}</strong> @ <strong>${g.homeScore}</strong> ${g.homeTeam}</div>
      <span style="font-size:0.8rem;color:var(--text-dim)">Final</span>
    </div>`;
  }

  // ── BACKTEST TIERS VIEW ──────────────────────────────────────
  function renderTiers() {
    const tiers = laData.tiers.tiers || [];
    if (!tiers.length) return '<div style="text-align:center;color:var(--text-dim);padding:3rem">No tier data available.</div>';

    // Dashboard stats
    const overall = laData.stats.overall || {};
    const daily = laData.stats.daily || {};
    const wins = overall.wins || 0;
    const total = overall.total || 0;
    const acc = total > 0 ? (wins / total * 100).toFixed(1) : '0.0';
    const roi = ((overall.roi || 0) * 100).toFixed(1);
    const pnl = overall.pnl || 0;

    let html = `<div class="dashboard" style="margin-bottom:1.5rem">
      <div class="stat-card"><div class="stat-value">${acc}<span class="stat-unit">%</span></div><div class="stat-label">Accuracy (${wins}W / ${total - wins}L)</div></div>
      <div class="stat-card"><div class="stat-value" style="color:var(--blue)">${roi}<span class="stat-unit">%</span></div><div class="stat-label">ROI</div></div>
      <div class="stat-card"><div class="stat-value" style="color:${pnl >= 0 ? 'var(--accent)' : 'var(--red)'}">$${pnl >= 0 ? '+' : ''}${Math.round(pnl).toLocaleString()}</div><div class="stat-label">Season Profit ($100/bet)</div></div>
      <div class="stat-card"><div class="stat-value" style="color:var(--gold)">${(daily.bets_per_day || 0).toFixed(1)}</div><div class="stat-label">Alerts/Day</div></div>
    </div>`;

    html += '<div style="display:grid;gap:1rem">';
    for (const tier of tiers) {
      const s = tier.stats || {};
      const a2 = s.accuracy ? (s.accuracy * 100).toFixed(1) : '0.0';
      const r2 = s.roi ? (s.roi * 100).toFixed(1) : '0.0';
      const p2 = s.pnl || 0;
      const color = tier.color || '#8b8fa3';
      html += `<div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.25rem 1.5rem;border-left:4px solid ${color}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem">
          <div><h3 style="font-size:1.05rem;color:${color};margin:0">${tier.name || tier.id}</h3>
            <div style="font-size:0.8rem;color:var(--text-dim);margin-top:2px">${s.rule || ''}</div></div>
          <div style="text-align:right"><div style="font-size:1.3rem;font-weight:700;color:${p2 >= 0 ? 'var(--accent)' : 'var(--red)'}">$${p2 >= 0 ? '+' : ''}${Math.round(p2).toLocaleString()}</div>
            <div style="font-size:0.75rem;color:var(--text-dim)">Season P&L</div></div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.75rem;text-align:center">
          <div><div style="font-size:1.1rem;font-weight:700">${a2}%</div><div style="font-size:0.7rem;color:var(--text-dim)">Accuracy</div></div>
          <div><div style="font-size:1.1rem;font-weight:700;color:var(--blue)">${r2}%</div><div style="font-size:0.7rem;color:var(--text-dim)">ROI</div></div>
          <div><div style="font-size:1.1rem;font-weight:700">${s.total || 0}</div><div style="font-size:0.7rem;color:var(--text-dim)">${s.wins || 0}W / ${s.losses || 0}L</div></div>
          <div><div style="font-size:1.1rem;font-weight:700;color:var(--gold)">${s.avg_buy ? '$' + s.avg_buy.toFixed(3) : '-'}</div><div style="font-size:0.7rem;color:var(--text-dim)">Avg Buy</div></div>
          <div><div style="font-size:1.1rem;font-weight:700;color:${(s.avg_profit_per_bet || 0) >= 0 ? 'var(--accent)' : 'var(--red)'}">${s.avg_profit_per_bet ? '$' + s.avg_profit_per_bet.toFixed(2) : '-'}</div><div style="font-size:0.7rem;color:var(--text-dim)">Avg P&L/Bet</div></div>
        </div>
      </div>`;
    }
    html += '</div>';
    return html;
  }

  // ── HISTORY VIEW ─────────────────────────────────────────────
  function renderHistory() {
    const signals = laData.signals || [];
    if (!signals.length) return '<div style="text-align:center;color:var(--text-dim);padding:3rem">No historical data.</div>';

    const tiers = laData.tiers.tiers || [];
    let html = `<div style="display:flex;gap:0.5rem;flex-wrap:wrap;justify-content:center;margin-bottom:1rem">
      <button class="filter-btn la-filter-btn ${laFilter === 'all' ? 'active' : ''}" data-la-filter="all">All (${signals.length})</button>`;
    for (const t of tiers) {
      html += `<button class="filter-btn la-filter-btn ${laFilter === t.id ? 'active' : ''}" data-la-filter="${t.id}" style="border-color:${t.color}">${t.name} (${(t.stats || {}).total || 0})</button>`;
    }
    html += '</div>';

    const filtered = laFilter === 'all' ? signals : signals.filter(s => s.tierId === laFilter);
    const wins = filtered.filter(s => s.hit).length;
    const pnl = filtered.reduce((sum, s) => sum + (s.pnl || 0), 0);
    const acc = filtered.length > 0 ? (wins / filtered.length * 100).toFixed(1) : '0.0';

    html += `<div style="display:flex;gap:1.5rem;justify-content:center;margin-bottom:1rem;font-size:0.85rem;color:var(--text-dim)">
      <span>${filtered.length} bets</span><span style="color:var(--accent)">${wins}W</span>
      <span style="color:var(--red)">${filtered.length - wins}L</span><span>${acc}%</span>
      <span style="color:${pnl >= 0 ? 'var(--accent)' : 'var(--red)'}">$${pnl >= 0 ? '+' : ''}${Math.round(pnl).toLocaleString()}</span>
    </div>`;

    const sorted = [...filtered].sort((a, b) => (b.date || '').localeCompare(a.date || ''));
    html += '<table class="history-table"><thead><tr><th>Date</th><th>Tier</th><th>Pick</th><th>Buy$</th><th>Result</th><th>P&L</th></tr></thead><tbody>';
    const limit = Math.min(sorted.length, 200);
    for (let i = 0; i < limit; i++) {
      const s = sorted[i];
      const hitLabel = s.hit ? 'WIN' : 'LOSS';
      const hitBg = s.hit ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)';
      const hitColor = s.hit ? 'var(--accent)' : 'var(--red)';
      const tc = (tiers.find(t => t.id === s.tierId) || {}).color || '#8b8fa3';
      html += `<tr>
        <td>${fmtDate(s.date)}</td>
        <td><span style="color:${tc};font-weight:600;font-size:0.8rem">${s.tierId}</span></td>
        <td>${s.team} ML (Inn ${s.inning}, +${s.lead})</td>
        <td style="color:var(--gold)">$${(s.buyPrice || 0).toFixed(2)}</td>
        <td><span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;background:${hitBg};color:${hitColor}">${hitLabel}</span></td>
        <td style="font-weight:600;color:${(s.pnl || 0) >= 0 ? 'var(--accent)' : 'var(--red)'}">$${s.pnl >= 0 ? '+' : ''}${(s.pnl || 0).toFixed(0)}</td>
      </tr>`;
    }
    if (sorted.length > limit) html += `<tr><td colspan="6" style="text-align:center;color:var(--text-dim);padding:1rem">Showing ${limit} of ${sorted.length}</td></tr>`;
    html += '</tbody></table>';
    return html;
  }

  // ── STRATEGY VIEW ────────────────────────────────────────────
  function renderStrategy() {
    const tiers = laData.tiers.tiers || [];
    let tierRules = '';
    for (const t of tiers) {
      const s = t.stats || {};
      tierRules += `<li style="margin-bottom:0.5rem"><strong style="color:${t.color || '#8b8fa3'}">${t.name}</strong>: ${s.rule || ''} <span style="color:var(--text-dim)"> — ${s.accuracy ? (s.accuracy * 100).toFixed(1) : '?'}% accuracy</span></li>`;
    }
    return `<div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.5rem 2rem;max-width:800px;margin:0 auto">
      <h3 style="color:#f59e0b;margin-bottom:1rem">Kinetic Divergence Theory (KDT)</h3>
      <p style="color:var(--text-dim);line-height:1.7;margin-bottom:1rem"><strong style="color:var(--text)">Core Concept:</strong> Baseball win probability has two layers — Structural Win Probability (SWP) based only on inning + lead (empirical from 2,364 real games), and Kinetic Win Probability (KWP) from ESPN/markets that reacts to every micro-event. When KD = SWP - KWP > 10%, the market underprices the leader.</p>
      <p style="color:var(--text-dim);line-height:1.7;margin-bottom:1rem"><strong style="color:var(--text)">Entry Method — IDS:</strong> When an opponent rally causes KWP to drop (KD spikes), but the defense survives the inning, buy at the start of the next inning. The rally failed but the market hasn't repriced — this is the optimal entry point.</p>
      <p style="color:var(--text-dim);line-height:1.7;margin-bottom:0.5rem"><strong style="color:var(--text)">Alert Tiers:</strong></p>
      <ul style="color:var(--text-dim);line-height:1.8;padding-left:1.5rem;margin-bottom:1rem">${tierRules}</ul>
      <p style="color:var(--text-dim);line-height:1.7"><strong style="color:var(--text)">Live Monitor:</strong> The Live Monitor tab polls ESPN every 30 seconds, computes KD for every live game, detects IDS entries, and fires real-time alerts — all in your browser.</p>
    </div>`;
  }

  // ── Event Binding ────────────────────────────────────────────
  function bindLAEvents(section) {
    section.querySelectorAll('.la-view-btn').forEach(btn => {
      btn.addEventListener('click', () => { laView = btn.dataset.laView; render(); });
    });
    section.querySelectorAll('.la-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => { laFilter = btn.dataset.laFilter; render(); });
    });
    section.querySelectorAll('.la-refresh-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (window.KDTLiveEngine) window.KDTLiveEngine.poll();
      });
    });
  }

  function fmtDate(s) {
    if (!s || s.length < 8) return s || '';
    return `${s.slice(4, 6)}/${s.slice(6, 8)}/${s.slice(0, 4)}`;
  }

  // ── Public Init ──────────────────────────────────────────────
  window.initLiveAlerts = async function () {
    if (laInitialized) return;
    laInitialized = true;

    // Load backtest data (non-blocking)
    loadLAData().then(() => { if (laView !== 'live') render(); });

    // Initialize KDT Live Engine
    if (window.KDTLiveEngine) {
      const ok = await window.KDTLiveEngine.init();
      engineReady = ok;
      render();

      if (ok) {
        window.KDTLiveEngine.start(
          function onUpdate(state) {
            liveState = state;
            if (laView === 'live') render();
          },
          function onAlert(alert) {
            console.log('[KDT] NEW ALERT:', alert.tier.name, alert.team, alert.score);
          }
        );
      } else {
        liveError = 'Failed to load SWP table';
        render();
      }
    } else {
      liveError = 'KDT engine not loaded';
      render();
    }
  };
})();
