/**
 * Positive Odds Models — Webapp Controller
 * =========================================
 * Self-contained controller for the positive-odds HECE models.
 * Loads data from experiments/webapp/data/ and renders into the
 * #positive-odds-section container.
 *
 * Does NOT touch or depend on any existing recommendation-engine
 * or recommendation-app code.
 */
(function () {
  'use strict';

  const DATA_BASE = 'positive-odds/data';

  let poData = {
    nba: { signals: [], stats: {}, recommendations: {} },
    mlb: { signals: [], stats: {}, recommendations: {} },
  };
  let poSport = 'nba';
  let poView = 'tonight';
  let poInitialized = false;

  // ── Data Loading ─────────────────────────────────────────────
  async function loadPOData() {
    for (const sport of ['nba', 'mlb']) {
      try {
        const [signals, stats, recs] = await Promise.all([
          fetch(`${DATA_BASE}/${sport}_signals.json`).then(r => r.ok ? r.json() : []),
          fetch(`${DATA_BASE}/${sport}_stats.json`).then(r => r.ok ? r.json() : {}),
          fetch(`${DATA_BASE}/${sport}_recommendations.json`).then(r => r.ok ? r.json() : {}),
        ]);
        poData[sport] = { signals, stats, recommendations: recs };
      } catch (e) {
        console.warn(`[PO] Failed to load ${sport}:`, e);
      }
    }
  }

  // ── Rendering ────────────────────────────────────────────────
  function render() {
    const section = document.getElementById('positive-odds-section');
    if (!section) return;

    const d = poData[poSport];
    const overall = d.stats.overall || {};
    const wins = overall.wins || 0;
    const total = overall.total || 0;
    const acc = total > 0 ? (wins / total * 100).toFixed(1) : '0.0';
    const roi = ((overall.roi || 0) * 100).toFixed(1);
    const pnl = overall.pnl || 0;
    const sportLabel = poSport.toUpperCase();

    let html = '';

    // Dashboard
    html += `
      <div class="dashboard" style="margin-bottom:1.5rem">
        <div class="stat-card">
          <div class="stat-value">${acc}<span class="stat-unit">%</span></div>
          <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:var(--blue)">${roi}<span class="stat-unit">%</span></div>
          <div class="stat-label">ROI</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:${pnl >= 0 ? 'var(--accent)' : 'var(--red)'}">$${pnl >= 0 ? '+' : ''}${pnl.toLocaleString()}</div>
          <div class="stat-label">Total P&L</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:var(--gold)">${total}</div>
          <div class="stat-label">Picks (${wins}W / ${total - wins}L)</div>
        </div>
      </div>`;

    // Sport nav tabs
    html += `
      <div class="po-sport-tabs" style="display:flex;gap:0;background:var(--bg);border-radius:8px;padding:3px;border:1px solid var(--border);width:fit-content;margin:0 auto 1rem">
        <button class="po-sport-btn ${poSport === 'nba' ? 'active' : ''}" data-po-sport="nba">NBA</button>
        <button class="po-sport-btn ${poSport === 'mlb' ? 'active' : ''}" data-po-sport="mlb">MLB</button>
      </div>
      <div style="display:flex;gap:0.5rem;justify-content:center;margin-bottom:1.5rem">
        <button class="nav-btn po-view-btn ${poView === 'tonight' ? 'active' : ''}" data-po-view="tonight">Today's Picks</button>
        <button class="nav-btn po-view-btn ${poView === 'history' ? 'active' : ''}" data-po-view="history">History</button>
        <button class="nav-btn po-view-btn ${poView === 'strategy' ? 'active' : ''}" data-po-view="strategy">Strategy</button>
      </div>`;

    // Content
    if (poView === 'tonight') {
      html += renderTonight(d);
    } else if (poView === 'history') {
      html += renderHistory(d);
    } else {
      html += renderStrategy();
    }

    section.innerHTML = html;
    bindPOEvents(section);
  }

  function renderTonight(d) {
    const recs = d.recommendations;
    if (!recs || !recs.picks || recs.picks.length === 0) {
      const dateStr = recs && recs.date ? fmtDate(recs.date) : 'today';
      return `<div style="text-align:center;color:var(--text-dim);padding:3rem;font-size:1rem">
        <div style="margin-bottom:0.5rem;font-size:1.1rem;color:var(--text)">${poSport.toUpperCase()} Positive Odds Picks — ${dateStr}</div>
        No picks for today. Props typically open 1-2 hours before game time.
      </div>`;
    }

    let html = `<div class="section-block"><h2>${poSport.toUpperCase()} Positive Odds Picks — ${fmtDate(recs.date)}</h2>`;
    html += '<div class="picks-grid" style="display:grid;gap:0.75rem;margin-top:1rem">';

    for (const p of recs.picks) {
      html += renderPickCard(p);
    }

    html += '</div></div>';
    return html;
  }

  function renderPickCard(p) {
    const isParlay = p.betType === 'parlay';
    const hitCls = p.hit === true ? 'hit' : (p.hit === false ? 'miss' : 'pending');
    const hitLabel = p.hit === true ? 'WIN' : (p.hit === false ? 'LOSS' : 'PENDING');
    const pnlStr = p.pnl != null ? `$${p.pnl >= 0 ? '+' : ''}${p.pnl}` : '';
    const oddsStr = p.odds >= 0 ? `+${p.odds}` : `${p.odds}`;

    let inner = '';
    if (isParlay) {
      inner = `<strong>${p.bet || (p.n_legs + '-Leg Parlay')}</strong>`;
      if (p.legs) {
        inner += '<div style="margin-top:0.4rem;font-size:0.85rem;color:var(--text-dim)">';
        for (const leg of p.legs) {
          const lh = leg.hit === true ? '✓' : (leg.hit === false ? '✗' : '?');
          const lo = leg.odds >= 0 ? `+${leg.odds}` : leg.odds;
          inner += `<div>${lh} ${leg.player} O${leg.line} ${(leg.statLabel || leg.stat || '').toUpperCase()} (${lo})</div>`;
        }
        inner += '</div>';
      }
    } else {
      const edgeStr = p.edge ? ` | Edge ${(p.edge * 100).toFixed(1)}%` : '';
      const hrStr = p.hitRate ? ` | HR ${(p.hitRate * 100).toFixed(0)}%` : '';
      const actualStr = p.actual != null ? ` → ${p.actual}` : '';
      inner = `
        <strong>${p.player}</strong>
        <div style="font-size:0.85rem;color:var(--text-dim);margin-top:2px">
          O${p.line} ${(p.statLabel || p.stat || '').toUpperCase()}${actualStr}${edgeStr}${hrStr}
        </div>`;
    }

    return `
      <div style="display:flex;justify-content:space-between;align-items:center;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem 1.25rem">
        <div style="flex:1">${inner}</div>
        <div style="min-width:60px;text-align:center;color:var(--gold);font-weight:700;font-size:1.05rem">${oddsStr}</div>
        <div style="min-width:90px;text-align:right">
          <span class="result-badge result-${hitCls}" style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:0.8rem;font-weight:700;
            background:${hitCls === 'hit' ? 'rgba(34,197,94,0.15)' : hitCls === 'miss' ? 'rgba(239,68,68,0.15)' : 'rgba(245,158,11,0.15)'};
            color:${hitCls === 'hit' ? 'var(--accent)' : hitCls === 'miss' ? 'var(--red)' : 'var(--gold)'}">
            ${hitLabel}
          </span>
          <div style="font-weight:700;margin-top:4px;color:${(p.pnl || 0) >= 0 ? 'var(--accent)' : 'var(--red)'}">${pnlStr}</div>
        </div>
      </div>`;
  }

  function renderHistory(d) {
    const signals = d.signals || [];
    if (!signals.length) {
      return '<div style="text-align:center;color:var(--text-dim);padding:3rem">No historical picks available.</div>';
    }

    const sorted = [...signals].sort((a, b) => (b.date || '').localeCompare(a.date || ''));

    let html = `<table class="history-table"><thead><tr>
      <th>Date</th><th>Type</th><th>Pick</th><th>Odds</th><th>Edge</th><th>Result</th><th>P&L</th>
    </tr></thead><tbody>`;

    for (const s of sorted) {
      const hitCls = s.hit === true ? 'hit' : (s.hit === false ? 'miss' : 'pending');
      const hitLabel = s.hit === true ? 'WIN' : (s.hit === false ? 'LOSS' : '?');
      const pnl = s.pnl != null ? `$${s.pnl >= 0 ? '+' : ''}${s.pnl}` : '';
      const edge = s.edge ? `${(s.edge * 100).toFixed(1)}%` : '-';
      const odds = s.odds >= 0 ? `+${s.odds}` : `${s.odds}`;
      const bet = s.betType === 'parlay'
        ? `${s.n_legs || s.legs?.length || '?'}-Leg Parlay`
        : `${s.player} O${s.line} ${(s.statLabel || s.stat || '').toUpperCase()}`;
      const actual = s.betType !== 'parlay' && s.actual != null ? ` (${s.actual})` : '';

      html += `<tr>
        <td>${fmtDate(s.date)}</td>
        <td>${s.betType === 'parlay' ? 'Parlay' : 'Single'}</td>
        <td>${bet}${actual}</td>
        <td style="color:var(--gold)">${odds}</td>
        <td>${edge}</td>
        <td><span class="result-badge result-${hitCls}" style="padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;
          background:${hitCls === 'hit' ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)'};
          color:${hitCls === 'hit' ? 'var(--accent)' : 'var(--red)'}">${hitLabel}</span></td>
        <td style="font-weight:600;color:${(s.pnl || 0) >= 0 ? 'var(--accent)' : 'var(--red)'}">${pnl}</td>
      </tr>`;
    }

    html += '</tbody></table>';
    return html;
  }

  function renderStrategy() {
    return `
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.5rem 2rem;max-width:800px;margin:0 auto">
        <h3 style="color:var(--accent);margin-bottom:1rem">Hybrid Ensemble Certainty Engine (HECE)</h3>
        <p style="color:var(--text-dim);line-height:1.7;margin-bottom:1rem">
          <strong style="color:var(--text)">Objective:</strong> Maximize ROI with 90%+ accuracy using exclusively plus-money odds (+100 and above).
        </p>
        <p style="color:var(--text-dim);line-height:1.7;margin-bottom:0.5rem">
          <strong style="color:var(--text)">Signal Architecture:</strong>
        </p>
        <ul style="color:var(--text-dim);line-height:1.8;padding-left:1.5rem;margin-bottom:1rem">
          <li><strong style="color:var(--blue)">Gravitational Floor Theory (GFT)</strong> — Multi-window weighted floor detection with exponential decay</li>
          <li><strong style="color:var(--blue)">Bayesian Edge Quantification (BEQ)</strong> — Credible interval lower bound vs market implied probability</li>
          <li><strong style="color:var(--blue)">Entropy Stability Index (ESI)</strong> — Shannon entropy of stat distribution for predictability scoring</li>
          <li><strong style="color:var(--blue)">5 Proprietary Boosters</strong> — Pace-adjusted opportunity, minutes stability, rest-day delta, opponent stat allowance, hot hand momentum</li>
          <li><strong style="color:var(--gold)">High Hit-Rate Gate (≥77%)</strong> — Only bets with proven historical hit rate</li>
          <li><strong style="color:var(--gold)">Positive EV Filter (≥10%)</strong> — Minimum expected value threshold</li>
        </ul>
        <p style="color:var(--text-dim);line-height:1.7;margin-bottom:1rem">
          <strong style="color:var(--text)">Key Innovation:</strong> Multi-signal unanimity — ALL signals must independently agree. Combined score uses geometric mean of GFT, BEQ edge, and ESI stability, plus additive proprietary boosters.
        </p>
        <p style="color:var(--text-dim);line-height:1.7;margin-bottom:1rem">
          <strong style="color:var(--text)">Validation:</strong> Walk-forward only (no future data), real historical odds from The Odds API, leakage detection passed, 500-iteration monotonic ratchet optimization with temporal consistency regularization.
        </p>
        <p style="color:var(--text-dim);line-height:1.7">
          <strong style="color:var(--text)">Odds Range:</strong> +100 to +400 (plus-money only)
        </p>
      </div>`;
  }

  // ── Event Binding ────────────────────────────────────────────
  function bindPOEvents(section) {
    section.querySelectorAll('.po-sport-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        poSport = btn.dataset.poSport;
        render();
      });
    });
    section.querySelectorAll('.po-view-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        poView = btn.dataset.poView;
        render();
      });
    });
  }

  // ── Helpers ──────────────────────────────────────────────────
  function fmtDate(s) {
    if (!s || s.length < 8) return s || '';
    return `${s.slice(4, 6)}/${s.slice(6, 8)}/${s.slice(0, 4)}`;
  }

  // ── Public API ───────────────────────────────────────────────
  window.initPositiveOdds = async function () {
    if (poInitialized) return;
    poInitialized = true;
    await loadPOData();
    render();
  };
})();
