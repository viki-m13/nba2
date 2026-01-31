// =============================================================================
// CHARTS - Chart.js based visualizations
// =============================================================================

window.Charts = (function() {

  let scoreflowChart = null;
  let detailChart = null;
  let historyChart = null;
  let equityChart = null;
  let featureImportanceChart = null;
  let confidenceCurveChart = null;

  const CHART_COLORS = {
    differential: '#10b981',
    differentialNeg: '#ef4444',
    grid: 'rgba(255, 255, 255, 0.06)',
    text: '#6b7280',
    zero: 'rgba(255, 255, 255, 0.15)',
    signal: {
      breakout_ml: '#dc2626',
      composite: '#f59e0b',
      blowout_compress: '#ec4899',
      quant: '#3b82f6',
      burst_fade: '#f97316',
      q3_fade: '#06b6d4',
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
  // FEATURE IMPORTANCE CHART (horizontal bar)
  // =========================================================================
  function renderFeatureImportance(canvasId, features) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    if (featureImportanceChart) featureImportanceChart.destroy();

    const ctx = canvas.getContext('2d');

    // Sort by importance (highest first)
    const sorted = [...features].sort((a, b) => b.importance - a.importance);
    const labels = sorted.map(f => f.name);
    const values = sorted.map(f => f.importance);
    const maxVal = Math.max(...values);

    // Color gradient from green (high) to blue (low)
    const colors = values.map((v, i) => {
      const ratio = v / maxVal;
      if (ratio > 0.8) return '#10b981';
      if (ratio > 0.6) return '#34d399';
      if (ratio > 0.4) return '#3b82f6';
      if (ratio > 0.2) return '#6366f1';
      return '#8b5cf6';
    });

    const options = getBaseOptions('');
    options.indexAxis = 'y';
    options.scales.x.title = { display: true, text: 'Information Gain', color: '#6b7280', font: { size: 10 } };
    options.scales.y.ticks = { color: '#e5e7eb', font: { size: 11, family: 'Inter' } };
    options.scales.x.ticks = { color: '#6b7280', font: { size: 10, family: 'JetBrains Mono' } };
    options.plugins.tooltip.callbacks = {
      label: function(context) {
        return `Importance: ${(context.raw * 100).toFixed(2)}%`;
      },
    };

    featureImportanceChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: colors,
          borderColor: colors.map(c => c + '80'),
          borderWidth: 1,
          borderRadius: 4,
          barPercentage: 0.7,
        }],
      },
      options,
    });

    return featureImportanceChart;
  }

  // =========================================================================
  // CONFIDENCE THRESHOLD CHART (dual axis: WR + ROI)
  // =========================================================================
  function renderConfidenceCurve(canvasId, tiers) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    if (confidenceCurveChart) confidenceCurveChart.destroy();

    const ctx = canvas.getContext('2d');
    const labels = tiers.map(t => `${Math.round(t.threshold * 100)}%`);
    const wrData = tiers.map(t => t.wr);
    const roiData = tiers.map(t => t.roi);
    const signalData = tiers.map(t => t.signals);

    const options = getBaseOptions('');
    options.plugins.legend = {
      display: true,
      position: 'top',
      labels: {
        color: '#9ca3af',
        font: { size: 11 },
        padding: 16,
        usePointStyle: true,
        pointStyleWidth: 12,
      },
    };
    options.scales = {
      x: {
        grid: { color: CHART_COLORS.grid },
        ticks: { color: CHART_COLORS.text, font: { size: 11 } },
        title: { display: true, text: 'Confidence Threshold', color: '#6b7280', font: { size: 11 } },
      },
      y: {
        type: 'linear',
        position: 'left',
        grid: { color: CHART_COLORS.grid },
        ticks: { color: '#10b981', font: { family: 'JetBrains Mono', size: 11 }, callback: v => v + '%' },
        title: { display: true, text: 'Win Rate %', color: '#10b981', font: { size: 11 } },
      },
      y2: {
        type: 'linear',
        position: 'right',
        grid: { display: false },
        ticks: { color: '#3b82f6', font: { family: 'JetBrains Mono', size: 11 }, callback: v => '+' + v + '%' },
        title: { display: true, text: 'ROI %', color: '#3b82f6', font: { size: 11 } },
      },
    };

    options.plugins.tooltip.callbacks = {
      label: function(context) {
        const idx = context.dataIndex;
        const tier = tiers[idx];
        if (context.dataset.label === 'Win Rate') return `Win Rate: ${tier.wr}%`;
        if (context.dataset.label === 'ROI') return `ROI: +${tier.roi}%`;
        return `Signals: ${tier.signals}`;
      },
      afterBody: function(context) {
        const idx = context[0]?.dataIndex;
        if (idx === undefined) return [];
        const tier = tiers[idx];
        return [`Signals: ${tier.signals}`, `Sharpe: ${tier.sharpe.toFixed(2)}`, `Kelly: ${(tier.kelly * 100).toFixed(1)}%`];
      },
    };

    // Break-even line annotation
    options.plugins.annotation = {
      annotations: {
        'breakeven': {
          type: 'line',
          yMin: 52.4,
          yMax: 52.4,
          borderColor: 'rgba(239, 68, 68, 0.4)',
          borderWidth: 1,
          borderDash: [4, 4],
          label: {
            display: true,
            content: 'Break-even (52.4%)',
            position: 'start',
            backgroundColor: 'rgba(239, 68, 68, 0.2)',
            color: '#ef4444',
            font: { size: 9 },
            padding: { top: 2, bottom: 2, left: 4, right: 4 },
          },
        },
      },
    };

    confidenceCurveChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Win Rate',
            data: wrData,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            fill: true,
            borderWidth: 2.5,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: '#10b981',
            pointBorderColor: '#fff',
            pointBorderWidth: 1.5,
            tension: 0.3,
            yAxisID: 'y',
          },
          {
            label: 'ROI',
            data: roiData,
            borderColor: '#3b82f6',
            backgroundColor: 'transparent',
            fill: false,
            borderWidth: 2.5,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: '#3b82f6',
            pointBorderColor: '#fff',
            pointBorderWidth: 1.5,
            tension: 0.3,
            yAxisID: 'y2',
          },
        ],
      },
      options,
    });

    return confidenceCurveChart;
  }

  // =========================================================================
  // DESTROY ALL CHARTS
  // =========================================================================
  function destroyAll() {
    [scoreflowChart, detailChart, historyChart, equityChart, featureImportanceChart, confidenceCurveChart].forEach(c => {
      if (c) c.destroy();
    });
    scoreflowChart = detailChart = historyChart = equityChart = featureImportanceChart = confidenceCurveChart = null;
  }

  // =========================================================================
  // PUBLIC API
  // =========================================================================
  return {
    renderScoreflow,
    renderEquityCurve,
    renderFeatureImportance,
    renderConfidenceCurve,
    destroyAll,
  };

})();
