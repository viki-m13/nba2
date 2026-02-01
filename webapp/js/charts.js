// =============================================================================
// CHARTS - Chart.js visualizations for Q3 O/U Signal Engine
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

    const existingChart = canvasId === 'scoreflow-chart' ? scoreflowChart :
                          canvasId === 'detail-scoreflow-chart' ? detailChart :
                          canvasId === 'history-chart' ? historyChart : null;
    if (existingChart) existingChart.destroy();

    const ctx = canvas.getContext('2d');
    const labels = possessions.map((_, i) => i);
    const diffs = possessions.map(p => p.differential);

    const annotations = {};

    // Zero line
    annotations['zero-line'] = {
      type: 'line',
      yMin: 0,
      yMax: 0,
      borderColor: CHART_COLORS.zero,
      borderWidth: 1,
    };

    // Quarter lines
    let lastQ = 0;
    possessions.forEach((p, i) => {
      if (p.quarter !== lastQ && lastQ > 0) {
        annotations[`q-${lastQ}`] = {
          type: 'line',
          xMin: i,
          xMax: i,
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          borderDash: [3, 3],
          label: {
            display: true,
            content: `Q${p.quarter}`,
            position: 'end',
            backgroundColor: 'transparent',
            color: '#6b7280',
            font: { size: 9 },
          },
        };
      }
      lastQ = p.quarter;
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

    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, 'rgba(16, 185, 129, 0.2)');
    gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');

    const datasets = [{
      label: 'Cumulative P&L (units)',
      data: equityData.combined,
      borderColor: '#10b981',
      backgroundColor: gradient,
      fill: true,
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.3,
    }];

    const options = getBaseOptions('');
    options.scales.x.title = { display: true, text: 'Signal #', color: '#6b7280', font: { size: 10 } };
    options.scales.y.title = { display: true, text: 'Cumulative P&L (units)', color: '#6b7280', font: { size: 10 } };

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
        return `P&L: ${context.raw.y > 0 ? '+' : ''}${context.raw.y.toFixed(2)} units`;
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
