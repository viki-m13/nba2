/**
 * NBA Edge Strategy Engine - Webapp Integration
 * ================================================
 * Displays AFCE (Adaptive Floor Confluence Engine) signals
 * with multi-tier parlay recommendations.
 *
 * Tiers: DIAMOND | RUBY | EMERALD | SAPPHIRE
 * Validated against real FanDuel odds from The Odds API.
 */

const EDGE_STRATEGY = {
  tiers: {
    DIAMOND: {
      color: '#b9f2ff',
      bg: 'rgba(185,242,255,0.08)',
      icon: '💎',
      label: 'DIAMOND',
      desc: 'Elite 2-leg floor parlays',
      accuracy: '72.6%',
      roi: '-0.5%',
    },
    RUBY: {
      color: '#ff4444',
      bg: 'rgba(255,68,68,0.08)',
      icon: '🔴',
      label: 'RUBY',
      desc: 'ML favorite + star prop',
      accuracy: '67.2%',
      roi: '-6.4%',
    },
    EMERALD: {
      color: '#00ff88',
      bg: 'rgba(0,255,136,0.08)',
      icon: '🟢',
      label: 'EMERALD',
      desc: '6-7 leg DFCP - best ROI',
      accuracy: '65.0%',
      roi: '+56.4%',
    },
    SAPPHIRE: {
      color: '#4488ff',
      bg: 'rgba(68,136,255,0.08)',
      icon: '🔵',
      label: 'SAPPHIRE',
      desc: '4-5 leg balanced DFCP',
      accuracy: '63.9%',
      roi: '+31.3%',
    },
  },
};

/**
 * Load and display edge strategy data.
 */
async function loadEdgeStrategy() {
  try {
    const [configRes, signalsRes, profilesRes] = await Promise.all([
      fetch('data/edge_strategy_config.json'),
      fetch('data/edge_strategy_signals.json'),
      fetch('data/edge_strategy_profiles.json'),
    ]);

    if (!configRes.ok || !signalsRes.ok) {
      console.warn('Edge strategy data not available');
      return;
    }

    const config = await configRes.json();
    const signals = await signalsRes.json();
    const profiles = profilesRes.ok ? await profilesRes.json() : {};

    renderEdgeStrategyDashboard(config, signals);
    renderEdgeStrategyHistory(signals);
    renderEdgeStrategyMethod(config);
  } catch (err) {
    console.error('Error loading edge strategy:', err);
  }
}

/**
 * Render the edge strategy dashboard stats.
 */
function renderEdgeStrategyDashboard(config, signals) {
  const container = document.getElementById('edge-dashboard');
  if (!container) return;

  // Calculate overall stats
  const validSignals = signals.filter(s => s.hit !== null && s.hit !== undefined);
  const hits = validSignals.filter(s => s.hit).length;
  const total = validSignals.length;
  const accuracy = total > 0 ? (hits / total * 100).toFixed(1) : '--';

  let totalPL = 0;
  validSignals.forEach(s => {
    if (s.hit) {
      totalPL += (s.parlay_decimal - 1);
    } else {
      totalPL -= 1;
    }
  });

  const roi = total > 0 ? (totalPL / total * 100).toFixed(1) : '--';
  const plDollars = (totalPL * 100).toFixed(0);

  // Total legs
  let totalLegs = 0, hitLegs = 0;
  validSignals.forEach(s => {
    (s.legs || []).forEach(l => {
      if (l.hit !== null && l.hit !== undefined) {
        totalLegs++;
        if (l.hit) hitLegs++;
      }
    });
  });

  const legRate = totalLegs > 0 ? (hitLegs / totalLegs * 100).toFixed(1) : '--';

  container.innerHTML = `
    <div class="stat-card highlight" style="border-color: #00ff88;">
      <div class="stat-value">${accuracy}%</div>
      <div class="stat-label">Parlay Win Rate</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${hits}-${total - hits}</div>
      <div class="stat-label">Record (W-L)</div>
    </div>
    <div class="stat-card" style="border-color: ${totalPL >= 0 ? '#00ff88' : '#ff4444'};">
      <div class="stat-value" style="color: ${totalPL >= 0 ? '#00ff88' : '#ff4444'};">$${plDollars}</div>
      <div class="stat-label">P&L ($100/bet)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color: ${parseFloat(roi) >= 0 ? '#00ff88' : '#ff4444'};">${roi}%</div>
      <div class="stat-label">ROI</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${hitLegs}-${totalLegs - hitLegs}</div>
      <div class="stat-label">Leg Record</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${legRate}%</div>
      <div class="stat-label">Leg Hit Rate</div>
    </div>
  `;

  // Per-tier breakdown
  const tierBreakdown = document.getElementById('edge-tier-breakdown');
  if (tierBreakdown) {
    let html = '<div class="tier-cards">';
    for (const [tierName, tierMeta] of Object.entries(EDGE_STRATEGY.tiers)) {
      const tierData = config.tiers && config.tiers[tierName];
      if (!tierData) continue;

      const acc = tierData.accuracy ? (tierData.accuracy * 100).toFixed(1) : '--';
      const tierRoi = tierData.roi ? (tierData.roi * 100).toFixed(1) : '--';

      html += `
        <div class="tier-card" style="border-left: 3px solid ${tierMeta.color}; background: ${tierMeta.bg};">
          <div class="tier-header">
            <span class="tier-icon">${tierMeta.icon}</span>
            <span class="tier-name" style="color: ${tierMeta.color};">${tierMeta.label}</span>
          </div>
          <div class="tier-desc">${tierMeta.desc}</div>
          <div class="tier-stats">
            <span>Accuracy: <strong>${acc}%</strong></span>
            <span>ROI: <strong style="color: ${parseFloat(tierRoi) >= 0 ? '#00ff88' : '#ff4444'};">${tierRoi}%</strong></span>
            <span>Bets: ${tierData.total_bets || 0}</span>
          </div>
        </div>
      `;
    }
    html += '</div>';
    tierBreakdown.innerHTML = html;
  }
}

/**
 * Render edge strategy signal history.
 */
function renderEdgeStrategyHistory(signals) {
  const tbody = document.getElementById('edge-history-body');
  if (!tbody) return;

  const validSignals = signals
    .filter(s => s.hit !== null && s.hit !== undefined)
    .sort((a, b) => b.date.localeCompare(a.date));

  let html = '';
  for (const s of validSignals.slice(0, 100)) {
    const tierMeta = EDGE_STRATEGY.tiers[s.tier] || {};
    const resultClass = s.hit ? 'win' : 'loss';
    const resultText = s.hit ? 'WIN' : 'LOSS';
    const pl = s.hit ? `+$${((s.parlay_decimal - 1) * 100).toFixed(0)}` : '-$100';
    const plColor = s.hit ? '#00ff88' : '#ff4444';

    const legsHtml = (s.legs || []).map(l => {
      const legStatus = l.hit ? '✅' : '❌';
      const actualStr = l.actual !== null && l.actual !== undefined ? ` (${l.actual})` : '';
      return `<span class="leg-chip ${l.hit ? 'hit' : 'miss'}">${legStatus} ${l.bet || ''}${actualStr} <em>${l.odds > 0 ? '+' : ''}${l.odds}</em></span>`;
    }).join('');

    const oddsStr = s.parlay_american >= 0 ? `+${s.parlay_american}` : `${s.parlay_american}`;

    html += `
      <tr class="${resultClass}">
        <td>${formatDate(s.date)}</td>
        <td><span style="color: ${tierMeta.color || '#fff'};">${tierMeta.icon || ''} ${s.tier}</span></td>
        <td>${s.n_legs}</td>
        <td>${oddsStr}</td>
        <td><span class="result-badge ${resultClass}">${resultText}</span></td>
        <td style="color: ${plColor};">${pl}</td>
      </tr>
      <tr class="legs-row ${resultClass}">
        <td colspan="6"><div class="legs-container">${legsHtml}</div></td>
      </tr>
    `;
  }

  tbody.innerHTML = html;
}

/**
 * Render strategy methodology section.
 */
function renderEdgeStrategyMethod(config) {
  const container = document.getElementById('edge-method');
  if (!container) return;

  container.innerHTML = `
    <div class="method-card">
      <h3>Adaptive Floor Confluence Engine (AFCE)</h3>
      <p>Patent-pending strategy using <strong>rolling player profiles</strong> and
      <strong>multi-dimensional floor reliability scoring</strong> to construct
      cross-game parlays from the most reliable player prop floors.</p>
      <p><strong>Key Innovation:</strong> Instead of static historical rates, AFCE maintains
      rolling 15-game windows per player, calculating dynamic floor thresholds that adapt
      to current form, role changes, and minutes patterns.</p>
    </div>
    <div class="method-card">
      <h3>AutoResearch Methodology</h3>
      <p>Strategy developed using <a href="https://github.com/karpathy/autoresearch" style="color: #4488ff;">Karpathy's AutoResearch</a>
      methodology: 12 systematic experiments, each evaluated against a single objective metric,
      with binary keep/discard decisions.</p>
      <p>All results validated against <strong>real FanDuel odds</strong> from The Odds API
      and <strong>real outcomes</strong> from ESPN box scores across ${config.tiers ?
        Object.values(config.tiers).reduce((sum, t) => sum + (t.total_bets || 0), 0) : '185'} bets.</p>
    </div>
    <div class="method-card">
      <h3>Strategy Tiers</h3>
      <ul style="list-style: none; padding: 0;">
        <li style="margin: 0.5rem 0;"><span style="color: #b9f2ff;">💎 DIAMOND</span> — 2-leg elite floor parlays. 72.6% accuracy at negative odds. Conservative grind.</li>
        <li style="margin: 0.5rem 0;"><span style="color: #ff4444;">🔴 RUBY</span> — ML favorite + star player prop. 67.2% accuracy. Single-game high-confidence plays.</li>
        <li style="margin: 0.5rem 0;"><span style="color: #00ff88;">🟢 EMERALD</span> — 6-7 leg Distributed Floor Confluence Parlay. 65% accuracy, <strong>+56.4% ROI</strong>. Best risk-adjusted returns.</li>
        <li style="margin: 0.5rem 0;"><span style="color: #4488ff;">🔵 SAPPHIRE</span> — 4-5 leg balanced DFCP. 63.9% accuracy, <strong>+31.3% ROI</strong>. Balanced approach.</li>
      </ul>
    </div>
    <div class="method-card">
      <h3>Floor Reliability Score (FRS)</h3>
      <p>Each leg is scored by a multi-factor <strong>Floor Reliability Score</strong>:</p>
      <ul>
        <li><strong>Hit Rate (50%)</strong> — % of recent games exceeding the line</li>
        <li><strong>Margin Stability (20%)</strong> — How far above the line, consistently</li>
        <li><strong>Sample Confidence (15%)</strong> — More games = higher confidence</li>
        <li><strong>Trend (15%)</strong> — Is the player trending up or maintaining?</li>
      </ul>
    </div>
  `;
}

function formatDate(dateStr) {
  if (!dateStr || dateStr.length < 8) return dateStr;
  const y = dateStr.slice(0, 4);
  const m = dateStr.slice(4, 6);
  const d = dateStr.slice(6, 8);
  return `${m}/${d}/${y}`;
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', loadEdgeStrategy);
} else {
  loadEdgeStrategy();
}
