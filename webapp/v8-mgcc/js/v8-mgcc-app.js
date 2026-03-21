/**
 * V8 CSPE — Convergence Score Parlay Engine Webapp Controller
 * =========================================================
 * Self-contained controller for the V8 CSPE parlay strategy.
 * Completely independent — does NOT touch original or positive-odds code.
 *
 * Loads data from webapp/v8-mgcc/data/ and renders into #v8-mgcc-section.
 */
(function () {
  'use strict';

  const DATA_BASE = 'v8-mgcc/data';

  let v8Data = {
    nba: { signals: [], stats: {}, recommendations: {} },
    mlb: { signals: [], stats: {}, recommendations: {} },
  };
  let v8Sport = 'nba';
  let v8View = 'tonight';
  let v8Initialized = false;

  // ── Data Loading ─────────────────────────────────────────────
  async function loadV8Data() {
    for (const sport of ['nba', 'mlb']) {
      try {
        const [signals, stats, recs] = await Promise.all([
          fetch(`${DATA_BASE}/${sport}_v8_signals.json`).then(r => r.ok ? r.json() : []),
          fetch(`${DATA_BASE}/${sport}_v8_stats.json`).then(r => r.ok ? r.json() : {}),
          fetch(`${DATA_BASE}/${sport}_v8_recommendations.json`).then(r => r.ok ? r.json() : {}),
        ]);
        v8Data[sport] = { signals, stats, recommendations: recs };
      } catch (e) {
        console.warn(`[V8] Failed to load ${sport}:`, e);
      }
    }
  }

  // ── Rendering ────────────────────────────────────────────────
  function render() {
    const section = document.getElementById('v8-mgcc-section');
    if (!section) return;

    const d = v8Data[v8Sport];
    // Stats may be nested under d.stats.parlays (legacy) or flat at d.stats (current)
    const parlays = d.stats.parlays || d.stats || {};
    const byLegs = d.stats.by_legs || {};
    const wins = parlays.wins || 0;
    const total = parlays.total || 0;
    const acc = total > 0 ? (wins / total * 100).toFixed(1) : '0.0';
    const roi = parlays.roi != null ? (parlays.roi * 100).toFixed(1) : '0.0';
    const pnl = parlays.pnl || 0;
    const legAcc = parlays.leg_accuracy ? (parlays.leg_accuracy * 100).toFixed(1) : '—';

    let html = '';

    // Dashboard — V8 specific metrics
    html += `
      <div class="dashboard" style="margin-bottom:1.5rem">
        <div class="stat-card">
          <div class="stat-value">${acc}<span class="stat-unit">%</span></div>
          <div class="stat-label">Parlay Accuracy</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:var(--blue)">${legAcc}<span class="stat-unit">%</span></div>
          <div class="stat-label">Leg Accuracy</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:${pnl >= 0 ? 'var(--accent)' : 'var(--red)'}">$${pnl >= 0 ? '+' : ''}${pnl.toLocaleString()}</div>
          <div class="stat-label">Total P&L</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:var(--blue)">${roi}<span class="stat-unit">%</span></div>
          <div class="stat-label">ROI</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:var(--gold)">${total}</div>
          <div class="stat-label">Parlays (${wins}W / ${total - wins}L)</div>
        </div>
      </div>`;

    // Leg-count breakdown cards
    if (Object.keys(byLegs).length > 0) {
      html += '<div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin-bottom:1.5rem">';
      for (const [n, s] of Object.entries(byLegs)) {
        const lAcc = s.leg_total > 0 ? (s.leg_hits / s.leg_total * 100).toFixed(1) : '—';
        const pAcc = s.total > 0 ? (s.wins / s.total * 100).toFixed(1) : '—';
        const lRoi = s.wagered > 0 ? (s.pnl / s.wagered * 100).toFixed(1) : '—';
        html += `
          <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:0.75rem 1rem;min-width:140px;flex:1">
            <div style="font-size:1.1rem;font-weight:700;color:var(--accent);margin-bottom:0.3rem">${n}-Leg Parlays</div>
            <div style="font-size:0.8rem;color:var(--text-dim);line-height:1.6">
              ${s.total} parlays | ${s.wins} wins<br>
              Parlay Acc: <strong style="color:var(--text)">${pAcc}%</strong><br>
              Leg Acc: <strong style="color:var(--text)">${lAcc}%</strong><br>
              P&L: <strong style="color:${s.pnl >= 0 ? 'var(--accent)' : 'var(--red)'}">$${s.pnl >= 0 ? '+' : ''}${s.pnl}</strong> (${lRoi}% ROI)
            </div>
          </div>`;
      }
      html += '</div>';
    }

    // Sport + view nav
    html += `
      <div class="v8-sport-tabs" style="display:flex;gap:0;background:var(--bg);border-radius:8px;padding:3px;border:1px solid var(--border);width:fit-content;margin:0 auto 1rem">
        <button class="v8-sport-btn ${v8Sport === 'nba' ? 'active' : ''}" data-v8-sport="nba"
          style="background:${v8Sport === 'nba' ? '#22c55e' : 'none'};color:${v8Sport === 'nba' ? '#0f1117' : '#8b8fa3'};border:none;padding:0.5rem 1.25rem;font-size:0.85rem;font-weight:700;letter-spacing:0.05em;cursor:pointer;border-radius:6px;transition:all 0.2s">NBA</button>
        <button class="v8-sport-btn ${v8Sport === 'mlb' ? 'active' : ''}" data-v8-sport="mlb"
          style="background:${v8Sport === 'mlb' ? '#22c55e' : 'none'};color:${v8Sport === 'mlb' ? '#0f1117' : '#8b8fa3'};border:none;padding:0.5rem 1.25rem;font-size:0.85rem;font-weight:700;letter-spacing:0.05em;cursor:pointer;border-radius:6px;transition:all 0.2s">MLB</button>
      </div>
      <div style="display:flex;gap:0.5rem;justify-content:center;margin-bottom:1.5rem">
        <button class="nav-btn v8-view-btn ${v8View === 'tonight' ? 'active' : ''}" data-v8-view="tonight">Today's Picks</button>
        <button class="nav-btn v8-view-btn ${v8View === 'history' ? 'active' : ''}" data-v8-view="history">History</button>
        <button class="nav-btn v8-view-btn ${v8View === 'strategy' ? 'active' : ''}" data-v8-view="strategy">Strategy</button>
      </div>`;

    // Content
    if (v8View === 'tonight') {
      html += renderTonight(d);
    } else if (v8View === 'history') {
      html += renderHistory(d);
    } else {
      html += renderStrategy();
    }

    section.innerHTML = html;
    bindV8Events(section);
  }

  // ── Tonight's Picks ──────────────────────────────────────────
  function renderTonight(d) {
    const recs = d.recommendations;
    if (!recs || !recs.picks || recs.picks.length === 0) {
      const dateStr = recs && recs.date ? fmtDate(recs.date) : 'today';
      return `<div style="text-align:center;color:var(--text-dim);padding:3rem;font-size:1rem">
        <div style="margin-bottom:0.5rem;font-size:1.1rem;color:var(--text)">${v8Sport.toUpperCase()} V8 CSPE Parlays — ${dateStr}</div>
        No picks for today. Check back after the daily cron runs (~12:45 PM ET).
      </div>`;
    }

    let html = `<div class="section-block"><h2>${v8Sport.toUpperCase()} V8 CSPE Parlays — ${fmtDate(recs.date)}</h2>`;
    html += '<div style="display:grid;gap:0.75rem;margin-top:1rem">';

    for (const p of recs.picks) {
      html += renderParlayCard(p);
    }

    html += '</div></div>';
    return html;
  }

  function renderParlayCard(p) {
    const hitCls = p.hit === true ? 'hit' : (p.hit === false ? 'miss' : 'pending');
    const hitLabel = p.hit === true ? 'WIN' : (p.hit === false ? 'LOSS' : 'PENDING');
    const pnlStr = p.pnl != null ? `$${p.pnl >= 0 ? '+' : ''}${p.pnl}` : '';
    const oddsStr = p.odds >= 0 ? `+${p.odds}` : `${p.odds}`;
    const nLegs = p.n_legs || (p.legs ? p.legs.length : '?');
    const probStr = p.combined_prob ? `${(p.combined_prob * 100).toFixed(0)}%` : '';

    let legsHtml = '';
    if (p.legs) {
      for (const leg of p.legs) {
        const lh = leg.hit === true ? '✓' : (leg.hit === false ? '✗' : '•');
        const lhColor = leg.hit === true ? 'var(--accent)' : (leg.hit === false ? 'var(--red)' : 'var(--text-dim)');
        const lo = leg.odds >= 0 ? `+${leg.odds}` : leg.odds;
        const actualStr = leg.actual != null ? ` → ${leg.actual}` : '';
        const edgeStr = leg.edge ? ` [${(leg.edge * 100).toFixed(0)}% edge]` : (leg.gates ? ` [${leg.gates}/7 gates]` : '');
        const tierStr = leg.tier ? ` (${leg.tier})` : '';
        legsHtml += `<div style="color:var(--text-dim);font-size:0.85rem">
          <span style="color:${lhColor};font-weight:700">${lh}</span>
          ${leg.player} O${leg.line} ${(leg.statLabel || leg.stat || '').toUpperCase()} (${lo})${actualStr}${edgeStr}${tierStr}
        </div>`;
      }
    }

    return `
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem 1.25rem">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.5rem">
          <div>
            <strong style="color:var(--text);font-size:1rem">${nLegs}-Leg CSPE Parlay</strong>
            ${probStr ? `<span style="color:var(--text-dim);font-size:0.8rem;margin-left:0.5rem">Est. ${probStr} prob</span>` : ''}
          </div>
          <div style="text-align:right">
            <div style="color:var(--gold);font-weight:700;font-size:1.1rem">${oddsStr}</div>
            <span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:700;
              background:${hitCls === 'hit' ? 'rgba(34,197,94,0.15)' : hitCls === 'miss' ? 'rgba(239,68,68,0.15)' : 'rgba(245,158,11,0.15)'};
              color:${hitCls === 'hit' ? 'var(--accent)' : hitCls === 'miss' ? 'var(--red)' : 'var(--gold)'}">${hitLabel}</span>
            <div style="font-weight:700;margin-top:2px;color:${(p.pnl || 0) >= 0 ? 'var(--accent)' : 'var(--red)'};font-size:0.9rem">${pnlStr}</div>
          </div>
        </div>
        <div style="border-top:1px solid var(--border);padding-top:0.5rem">${legsHtml}</div>
      </div>`;
  }

  // ── History ──────────────────────────────────────────────────
  function renderHistory(d) {
    const signals = d.signals || [];
    if (!signals.length) {
      return '<div style="text-align:center;color:var(--text-dim);padding:3rem">No historical parlays available.</div>';
    }

    const sorted = [...signals].sort((a, b) => (b.date || '').localeCompare(a.date || ''));

    let html = `<table class="history-table"><thead><tr>
      <th>Date</th><th>Type</th><th>Legs</th><th>Odds</th><th>Result</th><th>P&L</th><th>Details</th>
    </tr></thead><tbody>`;

    for (const s of sorted) {
      const hitCls = s.hit === true ? 'hit' : (s.hit === false ? 'miss' : 'pending');
      const hitLabel = s.hit === true ? 'WIN' : (s.hit === false ? 'LOSS' : '?');
      const pnl = s.pnl != null ? `$${s.pnl >= 0 ? '+' : ''}${s.pnl}` : '';
      const odds = s.odds >= 0 ? `+${s.odds}` : `${s.odds}`;
      const nLegs = s.n_legs || (s.legs ? s.legs.length : '?');
      const engine = (s.engine || '').replace('v8_cspe_nba', '').replace('v8_cspe_mlb', '').replace('v8_mgcc_nba_', '').replace('v8_mgcc_mlb_', '');

      // Leg summary
      let legSummary = '';
      if (s.legs) {
        const legHits = s.legs.filter(l => l.hit).length;
        legSummary = `${legHits}/${s.legs.length} legs hit`;
        if (s.legs.length <= 3) {
          legSummary += ': ' + s.legs.map(l => {
            const icon = l.hit ? '✓' : '✗';
            return `${icon}${l.player ? l.player.split(' ').pop() : '?'}`;
          }).join(', ');
        }
      }

      html += `<tr>
        <td>${fmtDate(s.date)}</td>
        <td>${nLegs}L ${engine}</td>
        <td style="font-size:0.82rem">${legSummary}</td>
        <td style="color:var(--gold)">${odds}</td>
        <td><span style="padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;
          background:${hitCls === 'hit' ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)'};
          color:${hitCls === 'hit' ? 'var(--accent)' : 'var(--red)'}">${hitLabel}</span></td>
        <td style="font-weight:600;color:${(s.pnl || 0) >= 0 ? 'var(--accent)' : 'var(--red)'}">${pnl}</td>
        <td><button class="v8-expand-btn" data-idx="${sorted.indexOf(s)}" style="background:none;border:1px solid var(--border);color:var(--text-dim);padding:2px 8px;border-radius:4px;cursor:pointer;font-size:0.75rem">▶</button></td>
      </tr>`;

      // Expandable leg details (hidden by default)
      if (s.legs) {
        html += `<tr class="v8-leg-detail" data-detail-idx="${sorted.indexOf(s)}" style="display:none">
          <td colspan="7" style="padding:0.5rem 1rem;background:var(--bg)">
            <div style="font-size:0.85rem;color:var(--text-dim)">`;
        for (const l of s.legs) {
          const icon = l.hit === true ? '✓' : (l.hit === false ? '✗' : '•');
          const iconColor = l.hit === true ? 'var(--accent)' : (l.hit === false ? 'var(--red)' : 'var(--text-dim)');
          const lo = l.odds >= 0 ? `+${l.odds}` : l.odds;
          const act = l.actual != null ? ` → actual: ${l.actual}` : '';
          const gates = l.gates ? ` | ${l.gates}/7 gates` : '';
          html += `<div><span style="color:${iconColor};font-weight:700">${icon}</span> ${l.player || '?'} O${l.line} ${(l.statLabel || l.stat || '').toUpperCase()} (${lo})${act}${gates}</div>`;
        }
        html += '</div></td></tr>';
      }
    }

    html += '</tbody></table>';
    return html;
  }

  // ── Strategy ─────────────────────────────────────────────────
  function renderStrategy() {
    return `
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.5rem 2rem;max-width:800px;margin:0 auto">
        <h3 style="color:var(--accent);margin-bottom:1rem">Convergence Score Parlay Engine (CSPE)</h3>
        <p style="color:var(--text-dim);line-height:1.7;margin-bottom:1rem">
          <strong style="color:var(--text)">Objective:</strong> Maximize ROI through edge-centric dual-tier parlay construction. Instead of ranking by raw accuracy (which always selects heavy favorites with tiny payouts), CSPE ranks by EDGE — the gap between true probability and market-implied probability — enabling high-payout parlays with genuine positive expected value.
        </p>
        <p style="color:var(--text-dim);line-height:1.7;margin-bottom:0.5rem">
          <strong style="color:var(--text)">Patentable Innovations:</strong>
        </p>
        <ol style="color:var(--text-dim);line-height:1.9;padding-left:1.5rem;margin-bottom:1rem">
          <li><strong style="color:var(--blue)">Edge-Centric Convergence Scoring (ECCS)</strong> — Continuous multi-dimensional scoring combining Z-score floor analysis, Bayesian credible bounds, entropy-calibrated confidence, and market edge into a single composite signal</li>
          <li><strong style="color:var(--blue)">Dual-Tier Architecture (DTA)</strong> — Core tier (2L heavy favorites for consistency) + Amplifier tier (4-5L edge-ranked for explosive ROI)</li>
          <li><strong style="color:var(--blue)">Anchor-Amplifier Fusion (AAF)</strong> — Hybrid parlays combining ultra-safe anchor legs with high-edge amplifier legs for massive combined odds with acceptable hit rates</li>
          <li><strong style="color:var(--blue)">Z-Score Floor Analysis (ZFA)</strong> — Measures how many standard deviations the player's worst-case performance exceeds the line</li>
          <li><strong style="color:var(--blue)">Entropy-Calibrated Confidence (ECC)</strong> — Shannon entropy as reliability multiplier: predictable distributions amplify confidence</li>
        </ol>
        <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:1rem;margin-bottom:1rem">
          <strong style="color:var(--gold);font-size:0.9rem">NBA Performance (Walk-Forward Backtested)</strong>
          <div style="color:var(--text-dim);font-size:0.85rem;margin-top:0.3rem;line-height:1.7">
            2L Parlays: 19 parlays | <strong style="color:var(--accent)">94.7% accuracy</strong> | 97.4% leg accuracy | +47.6% ROI<br>
            3L Parlays: 5 parlays | 60.0% accuracy | 86.7% leg accuracy | +30.8% ROI<br>
            <strong style="color:var(--accent)">Combined: 24 parlays | $+1,994 P&L | +45.3% ROI | 19 active days</strong><br>
            Temporal: H1 100% | H2 75% | All months profitable
          </div>
        </div>
        <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:1rem;margin-bottom:1rem">
          <strong style="color:var(--gold);font-size:0.9rem">MLB Performance (Walk-Forward Backtested)</strong>
          <div style="color:var(--text-dim);font-size:0.85rem;margin-top:0.3rem;line-height:1.7">
            Singles: 47 picks | <strong style="color:var(--accent)">91.5% accuracy</strong> | +34.6% ROI<br>
            2L Parlays: 9 parlays | 88.9% accuracy | 94.4% leg accuracy | +84.4% ROI<br>
            3L Parlays: 1 parlay | 100% accuracy | 100% leg accuracy | +300% ROI<br>
            <strong style="color:var(--accent)">Combined: 57 picks | 91.2% accuracy | 92.6% leg accuracy | $+4,625 P&L | +50.4% ROI</strong><br>
            Temporal: H1 92.9% | H2 89.7% | 3.2pp gap (STABLE)
          </div>
        </div>
        <p style="color:var(--text-dim);line-height:1.7">
          <strong style="color:var(--text)">Validation:</strong> Walk-forward only, real Odds API odds, Beta(1,1) Bayesian prior, no parameter optimization on test data, temporal stability validated across H1/H2 splits.
        </p>
      </div>`;
  }

  // ── Events ───────────────────────────────────────────────────
  function bindV8Events(section) {
    section.querySelectorAll('.v8-sport-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        v8Sport = btn.dataset.v8Sport;
        render();
      });
    });
    section.querySelectorAll('.v8-view-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        v8View = btn.dataset.v8View;
        render();
      });
    });
    // Expand/collapse leg details in history
    section.querySelectorAll('.v8-expand-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = btn.dataset.idx;
        const row = section.querySelector(`.v8-leg-detail[data-detail-idx="${idx}"]`);
        if (row) {
          const visible = row.style.display !== 'none';
          row.style.display = visible ? 'none' : '';
          btn.textContent = visible ? '▶' : '▼';
        }
      });
    });
  }

  // ── Helpers ──────────────────────────────────────────────────
  function fmtDate(s) {
    if (!s || s.length < 8) return s || '';
    return `${s.slice(4, 6)}/${s.slice(6, 8)}/${s.slice(0, 4)}`;
  }

  // ── Public API ─────────────────────────────────────────────
  window.initV8MGCC = async function () {
    if (v8Initialized) return;
    v8Initialized = true;
    await loadV8Data();
    render();
  };
})();
