// =============================================================================
// CHARTS - Chart.js based visualizations
// =============================================================================

window.Charts = (function() {

  let scoreflowChart = null;
  let detailChart = null;
  let historyChart = null;
  let equityChart = null;

  const CHART_COLORS = {
    differential: '#10b981',
    differentialNeg: '#ef4444',
    grid: 'rgba(255, 255, 255, 0.06)',
    text: '#6b7280',
    zero: 'rgba(255, 255, 255, 0.15)',
    signal: {
      composite: '#f59e0b',
      quant: '#3b82f6',
      fade_ml: '#a855f7',
      fade_spread: '#10b981',
    },
  };

  // =========================================================================
  // COMMON CHART OPTIONS
  // =========================================================================
  function getBaseOptions(title) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        title: {
          display: !!title,
          text: title || '',
          color: '#e5e7eb',
          font: { size: 13, weight: 600, family: 'Inter' },
          padding: { bottom: 12 },
        },
        tooltip: {
          backgroundColor: '#1c1c28',
          titleColor: '#e5e7eb',
          bodyColor: '#9ca3af',
          borderColor: '#333346',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 10,
          titleFont: { weight: 600 },
          bodyFont: { family: 'JetBrains Mono', size: 12 },
        },
      },
      scales: {
        x: {
          grid: { color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.text, font: { size: 10 } },
        },
        y: {
          grid: { color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.text, font: { family: 'JetBrains Mono', size: 11 } },
        },
      },
    };
  }

  // =========================================================================
  // SCORE FLOW CHART
  // =========================================================================
  function renderScoreflow(canvasId, possessions, signals, gameTitle) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Destroy existing chart
    const existingChart = canvasId === 'scoreflow-chart' ? scoreflowChart :
                          canvasId === 'detail-scoreflow-chart' ? detailChart :
                          canvasId === 'history-chart' ? historyChart : null;
    if (existingChart) existingChart.destroy();

    const ctx = canvas.getContext('2d');
    const labels = possessions.map((_, i) => i);
    const diffs = possessions.map(p => p.differential);

    // Build signal annotations
    const annotations = {};
    if (signals && signals.length > 0) {
      signals.forEach((sig, idx) => {
        const posIdx = sig.possessionIndex || sig.possIdx || 0;
        annotations[`signal-${idx}`] = {
          type: 'line',
          xMin: posIdx,
          xMax: posIdx,
          borderColor: CHART_COLORS.signal[sig.tier] || '#f59e0b',
          borderWidth: 2,
          borderDash: [4, 4],
          label: {
            display: true,
            content: `${sig.tier.toUpperCase()} ${sig.betTeam || sig.betTeamAbbr || ''}`,
            position: 'start',
            backgroundColor: CHART_COLORS.signal[sig.tier] || '#f59e0b',
            color: '#000',
            font: { size: 10, weight: 700, family: 'Inter' },
            padding: { top: 2, bottom: 2, left: 6, right: 6 },
            borderRadius: 4,
          },
        };
      });
    }

    // Zero line
    annotations['zero-line'] = {
      type: 'line',
      yMin: 0,
      yMax: 0,
      borderColor: CHART_COLORS.zero,
      borderWidth: 1,
    };

    // Quarter lines
    const quarterBounds = [];
    let lastQ = 0;
    possessions.forEach((p, i) => {
      if (p.quarter !== lastQ) {
        quarterBounds.push(i);
        lastQ = p.quarter;
      }
    });
    quarterBounds.forEach((pos, idx) => {
      if (idx > 0) {
        annotations[`q-${idx}`] = {
          type: 'line',
          xMin: pos,
          xMax: pos,
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          borderDash: [3, 3],
          label: {
            display: true,
            content: `Q${idx + 1}`,
            position: 'end',
            backgroundColor: 'transparent',
            color: '#6b7280',
            font: { size: 9 },
          },
        };
      }
    });

    const options = getBaseOptions(gameTitle || '');
    options.plugins.annotation = { annotations };
    options.scales.x.title = { display: true, text: 'Possession', color: '#6b7280', font: { size: 10 } };
    options.scales.y.title = { display: true, text: 'Score Differential (Home - Away)', color: '#6b7280', font: { size: 10 } };

    options.plugins.tooltip.callbacks = {
      label: function(context) {
        const idx = context.dataIndex;
        const p = possessions[idx];
        if (!p) return '';
        return [
          `Diff: ${p.differential > 0 ? '+' : ''}${p.differential}`,
          `Score: ${p.homeScore}-${p.awayScore}`,
          `Q${p.quarter} ${p.quarterTime}`,
        ];
      },
    };

    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, 'rgba(16, 185, 129, 0.3)');
    gradient.addColorStop(0.5, 'rgba(16, 185, 129, 0.02)');
    gradient.addColorStop(1, 'rgba(239, 68, 68, 0.2)');

    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          data: diffs,
          borderColor: function(context) {
            const value = context.raw;
            if (value === undefined) return CHART_COLORS.differential;
            return value >= 0 ? CHART_COLORS.differential : CHART_COLORS.differentialNeg;
          },
          borderWidth: 2,
          backgroundColor: gradient,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: '#e5e7eb',
          tension: 0.3,
          segment: {
            borderColor: function(ctx) {
              const val = ctx.p1.raw;
              return val >= 0 ? CHART_COLORS.differential : CHART_COLORS.differentialNeg;
            },
          },
        }],
      },
      options,
    });

    // Store reference
    if (canvasId === 'scoreflow-chart') scoreflowChart = chart;
    else if (canvasId === 'detail-scoreflow-chart') detailChart = chart;
    else if (canvasId === 'history-chart') historyChart = chart;

    return chart;
  }

  // =========================================================================
  // EQUITY CURVE CHART
  // =========================================================================
  function renderEquityCurve(canvasId, equityData, mode) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    if (equityChart) equityChart.destroy();

    const ctx = canvas.getContext('2d');

    const datasets = [];

    if (mode === 'combined' || mode === 'all') {
      const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, 'rgba(16, 185, 129, 0.2)');
      gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');

      datasets.push({
        label: 'Combined (Spread + ML)',
        data: equityData.combined,
        borderColor: '#10b981',
        backgroundColor: gradient,
        fill: true,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.3,
      });
    }

    if (mode === 'spread' || mode === 'all') {
      datasets.push({
        label: 'Spread Only',
        data: equityData.spreadOnly,
        borderColor: '#3b82f6',
        backgroundColor: 'transparent',
        fill: false,
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 3,
        tension: 0.3,
        borderDash: mode === 'all' ? [4, 4] : [],
      });
    }

    if (mode === 'ml' || mode === 'all') {
      datasets.push({
        label: 'Moneyline Only',
        data: equityData.mlOnly,
        borderColor: '#a855f7',
        backgroundColor: 'transparent',
        fill: false,
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 3,
        tension: 0.3,
        borderDash: mode === 'all' ? [2, 2] : [],
      });
    }

    const options = getBaseOptions('');
    options.plugins.legend = {
      display: datasets.length > 1,
      position: 'top',
      labels: {
        color: '#9ca3af',
        font: { size: 11 },
        padding: 16,
        usePointStyle: true,
        pointStyleWidth: 12,
      },
    };
    options.scales.x.title = { display: true, text: 'Signal #', color: '#6b7280', font: { size: 10 } };
    options.scales.y.title = { display: true, text: 'Cumulative P&L (units)', color: '#6b7280', font: { size: 10 } };

    // Zero line
    options.plugins.annotation = {
      annotations: {
        'zero': {
          type: 'line',
          yMin: 0,
          yMax: 0,
          borderColor: CHART_COLORS.zero,
          borderWidth: 1,
        },
      },
    };

    options.plugins.tooltip.callbacks = {
      label: function(context) {
        return `${context.dataset.label}: ${context.raw.y > 0 ? '+' : ''}${context.raw.y.toFixed(2)} units`;
      },
    };

    equityChart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options,
    });

    return equityChart;
  }

  // =========================================================================
  // DESTROY ALL CHARTS
  // =========================================================================
  function destroyAll() {
    [scoreflowChart, detailChart, historyChart, equityChart].forEach(c => {
      if (c) c.destroy();
    });
    scoreflowChart = detailChart = historyChart = equityChart = null;
  }

  // =========================================================================
  // PUBLIC API
  // =========================================================================
  return {
    renderScoreflow,
    renderEquityCurve,
    destroyAll,
  };

})();
