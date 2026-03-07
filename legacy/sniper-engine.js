// =============================================================================
// Sniper Engine - Ultra-High Confidence NBA Prediction (JavaScript)
// =============================================================================
// Ports the Python Sniper Model to JavaScript for real-time live game inference.
//
// Target: 99%+ win rate on leader moneyline picks.
// Only fires when ALL sniper gates pass:
//   1. Calibrated confidence >= threshold
//   2. Lead quality score >= threshold
//   3. Not a tied game
//   4. Lead within valid range
//
// Uses Logistic Regression (winner probability) + Ridge (margin) with
// pre-trained coefficients + calibration lookup table.
// =============================================================================

window.SniperEngine = (function () {
  'use strict';

  let MODEL = null;

  const Q4_SIGMA = 9.5;

  // =========================================================================
  // MODEL LOADING
  // =========================================================================

  function loadModel(params) {
    MODEL = params;
    console.log(`[Sniper] Model loaded: ${MODEL.features.length} features, v=${MODEL.version}`);
  }

  function isModelLoaded() {
    return MODEL !== null;
  }

  // =========================================================================
  // CALIBRATION
  // =========================================================================

  /** Interpolate calibrated probability from lookup table */
  function calibrate(rawProb) {
    if (!MODEL || !MODEL.calibration_x || !MODEL.calibration_y) return rawProb;
    const xs = MODEL.calibration_x;
    const ys = MODEL.calibration_y;
    if (rawProb <= xs[0]) return ys[0];
    if (rawProb >= xs[xs.length - 1]) return ys[ys.length - 1];
    // Linear interpolation
    for (let i = 1; i < xs.length; i++) {
      if (rawProb <= xs[i]) {
        const t = (rawProb - xs[i - 1]) / (xs[i] - xs[i - 1]);
        return ys[i - 1] + t * (ys[i] - ys[i - 1]);
      }
    }
    return rawProb;
  }

  // =========================================================================
  // FEATURE EXTRACTION
  // =========================================================================

  function extractFeatures(game, possessions, opts = {}) {
    if (!possessions || possessions.length === 0) return null;

    const currentQ = game.quarter || 0;
    if (currentQ < 3) return null;

    const q3Possessions = possessions.filter(p => p.quarter <= 3);
    if (q3Possessions.length === 0) return null;

    // Quarter end scores
    const qEnds = {};
    for (const p of possessions) {
      if (p.quarter <= 4) {
        qEnds[p.quarter] = { home: p.homeScore, away: p.awayScore };
      }
    }
    if (!qEnds[3]) return null;

    const q3Home = qEnds[3].home;
    const q3Away = qEnds[3].away;
    const q1Home = qEnds[1] ? qEnds[1].home : 0;
    const q1Away = qEnds[1] ? qEnds[1].away : 0;
    const q2Home = qEnds[2] ? qEnds[2].home - q1Home : 0;
    const q2Away = qEnds[2] ? qEnds[2].away - q1Away : 0;
    const q3h = q3Home - (qEnds[2] ? qEnds[2].home : 0);
    const q3a = q3Away - (qEnds[2] ? qEnds[2].away : 0);

    const openingSpread = opts.openingSpread || 0;
    const openingOU = opts.openingOU || 0;

    const leadTS = buildLeadTimeSeries(q3Possessions, 30);
    const scoringEvents = buildScoringEvents(q3Possessions, possessions);

    const f = {};

    // ---- Core score ----
    const q3Lead = q3Home - q3Away;
    f.q3_lead = q3Lead;
    f.q3_lead_abs = Math.abs(q3Lead);
    f.q3_total = q3Home + q3Away;
    f.is_home_leading = q3Lead > 0 ? 1 : (q3Lead < 0 ? 0 : 0.5);

    f.q1_margin = q1Home - q1Away;
    f.q2_margin = q2Home - q2Away;
    f.q3_margin = q3h - q3a;
    f.h1_margin = (q1Home + q2Home) - (q1Away + q2Away);

    f.q1_total = q1Home + q1Away;
    f.q2_total = q2Home + q2Away;
    f.q3_qtotal = q3h + q3a;
    f.h1_total = f.q1_total + f.q2_total;

    const qtotals = [f.q1_total, f.q2_total, f.q3_qtotal];
    f.avg_q_pace = mean(qtotals);
    f.pace_var = std(qtotals);
    f.pace_trend = f.q3_qtotal - f.q1_total;
    f.q3_pace_vs_avg = f.q3_qtotal - f.avg_q_pace;

    const homeQs = [q1Home, q2Home, q3h];
    const awayQs = [q1Away, q2Away, q3a];
    f.home_q_std = std(homeQs);
    f.away_q_std = std(awayQs);
    f.home_q_trend = q3h - q1Home;
    f.away_q_trend = q3a - q1Away;

    const qMargins = [f.q1_margin, f.q2_margin, f.q3_margin];
    f.margin_trend = qMargins[2] - qMargins[0];
    f.margin_consistency = std(qMargins);
    f.all_qs_same_direction = (qMargins.every(m => m > 0) || qMargins.every(m => m < 0)) ? 1 : 0;

    const cumMargins = [qMargins[0], qMargins[0] + qMargins[1], qMargins[0] + qMargins[1] + qMargins[2]];
    f.lead_built_gradually = (
      Math.abs(cumMargins[0]) <= Math.abs(cumMargins[1]) &&
      Math.abs(cumMargins[1]) <= Math.abs(cumMargins[2]) &&
      Math.abs(cumMargins[2]) > 5
    ) ? 1 : 0;

    // Pregame lines
    f.opening_spread = openingSpread;
    f.opening_ou = openingOU;
    f.pregame_home_ml = 0;
    f.pregame_away_ml = 0;
    const expectedQ3Margin = openingSpread * 0.75;
    f.spread_surprise = q3Lead - expectedQ3Margin;
    f.spread_surprise_abs = Math.abs(f.spread_surprise);
    const projFinalTotal = f.q3_total + f.avg_q_pace;
    f.total_pace_vs_ou = openingOU > 0 ? projFinalTotal - openingOU : 0;

    // Lead dynamics
    Object.assign(f, leadDynamics(leadTS, q3Lead));
    Object.assign(f, dnaFeatures(leadTS));
    Object.assign(f, taFeatures(leadTS));

    // Scoring patterns
    Object.assign(f, scoringPatterns(scoringEvents, q3Home, q3Away));

    // Fouls/turnovers/timeouts (not available in NBA.com PBP)
    f.foul_differential = 0;
    f.q3_home_fouls = 0;
    f.q3_away_fouls = 0;
    f.turnover_differential = 0;
    f.home_timeouts_used = 0;
    f.away_timeouts_used = 0;
    f.timeout_advantage = 0;

    // Interaction features
    f.lead_x_momentum = f.q3_lead * (f.lead_momentum || 0);
    f.lead_x_volatility = f.q3_lead_abs * (f.lead_vol || 0);
    f.lead_x_pace = f.q3_lead_abs * f.avg_q_pace;
    f.lead_x_surprise = f.q3_lead * f.spread_surprise;
    f.momentum_x_fouls = (f.lead_momentum || 0) * f.foul_differential;
    f.pace_x_surprise = f.avg_q_pace * f.spread_surprise_abs;
    f.consistency_x_lead = f.margin_consistency * f.q3_lead_abs;
    f.gradual_x_lead = f.lead_built_gradually * f.q3_lead_abs;

    // ====== SNIPER-SPECIFIC FEATURES ======
    const leadDir = q3Lead > 0 ? 1 : (q3Lead < 0 ? -1 : 0);

    // Lead sustainability composite
    const marginNorm = Math.min(f.margin_consistency / 10.0, 1.0);
    f.lead_sustainability = (
      (f.lead_built_gradually || 0) * 0.25 +
      (f.all_qs_same_direction || 0) * 0.25 +
      (f.lead_stability || 0) * 0.25 +
      (1 - marginNorm) * 0.25
    );

    // Theoretical win probability
    f.theo_win_prob = f.q3_lead_abs > 0 ? normalCDF(f.q3_lead_abs / Q4_SIGMA) : 0.5;

    // Momentum alignment
    const momentum = f.lead_momentum || 0;
    f.momentum_aligned = ((q3Lead > 0 && momentum > 0) || (q3Lead < 0 && momentum < 0)) ? 1.0 : 0.0;
    f.momentum_magnitude_aligned = Math.abs(momentum) * (f.momentum_aligned ? 1 : -1);

    // 3PT dependency
    if (q3Lead > 0) {
      f.leader_3pt_dependency = f.sp_home_3pt_rate || 0;
      f.trailer_3pt_rate = f.sp_away_3pt_rate || 0;
    } else if (q3Lead < 0) {
      f.leader_3pt_dependency = f.sp_away_3pt_rate || 0;
      f.trailer_3pt_rate = f.sp_home_3pt_rate || 0;
    } else {
      f.leader_3pt_dependency = 0;
      f.trailer_3pt_rate = 0;
    }

    // Favorable quarters
    const favorableQs = leadDir !== 0 ? qMargins.filter(m => m * leadDir > 0).length : 0;
    f.favorable_quarters = favorableQs;
    f.dominated_all_qs = favorableQs === 3 ? 1.0 : 0.0;

    // Late Q3 strength
    f.q3_late_strength = (f.sp_late_q3_margin || 0) * leadDir;

    // Closest approach
    if (leadTS.length > 0 && leadDir !== 0) {
      if (leadDir > 0) {
        f.closest_approach = Math.min(...leadTS);
      } else {
        f.closest_approach = -Math.max(...leadTS);
      }
    } else {
      f.closest_approach = 0;
    }
    f.min_lead_buffer = leadDir !== 0 ? Math.max(0, f.closest_approach) : 0;
    f.lead_never_lost = (leadDir !== 0 && f.closest_approach > 0) ? 1.0 : 0.0;

    // Trailer scoring indicators
    if (leadDir > 0) {
      f.trailer_dry_spell = f.sp_away_dry_spell || 0;
      f.trailer_largest_run = f.sp_away_largest_run || 0;
    } else if (leadDir < 0) {
      f.trailer_dry_spell = f.sp_home_dry_spell || 0;
      f.trailer_largest_run = f.sp_home_largest_run || 0;
    } else {
      f.trailer_dry_spell = 0;
      f.trailer_largest_run = 0;
    }

    // Lead velocity/acceleration
    const earlyLead = (f.lead_early || 0) * leadDir;
    const midLead = (f.lead_mid || 0) * leadDir;
    const lateLead = (f.lead_late || 0) * leadDir;
    f.lead_velocity_late = lateLead - midLead;
    f.lead_acceleration = (lateLead - midLead) - (midLead - earlyLead);

    // Pregame alignment
    if (openingSpread !== 0) {
      const pregameFavHome = openingSpread < 0;
      const leaderIsFav = (q3Lead > 0 && pregameFavHome) || (q3Lead < 0 && !pregameFavHome);
      f.leader_is_pregame_fav = leaderIsFav ? 1.0 : 0.0;
    } else {
      f.leader_is_pregame_fav = 0.5;
    }

    // Scoring balance
    const leaderQs = leadDir > 0 ? homeQs : (leadDir < 0 ? awayQs : [0, 0, 0]);
    const lqMean = mean(leaderQs);
    f.leader_scoring_balance = 1 - (std(leaderQs) / (lqMean + 1e-10));
    f.leader_q3_scoring = leaderQs.length > 2 ? leaderQs[2] : 0;

    // Sniper heuristic
    f.sniper_heuristic = (
      f.lead_sustainability * 0.20 +
      f.theo_win_prob * 0.25 +
      f.momentum_aligned * 0.15 +
      f.dominated_all_qs * 0.15 +
      f.lead_never_lost * 0.15 +
      Math.min(f.leader_scoring_balance, 1.0) * 0.10
    );

    // Enhanced interactions
    f.sustainability_x_lead = f.lead_sustainability * f.q3_lead_abs;
    f.momentum_aligned_x_lead = f.momentum_magnitude_aligned * f.q3_lead_abs;
    f.theo_x_actual_lead = f.theo_win_prob * f.q3_lead_abs;
    f.quality_x_lead = f.sniper_heuristic * f.q3_lead_abs;
    f.fav_alignment_x_lead = f.leader_is_pregame_fav * f.q3_lead_abs;
    f.dominated_x_lead = f.dominated_all_qs * f.q3_lead_abs;

    return {
      features: f,
      gameState: {
        q3Home, q3Away, q3Lead,
        q1Home, q1Away, q2Home, q2Away, q3h, q3a,
        homeTeam: game.homeTeam,
        awayTeam: game.awayTeam,
        avgPace: f.avg_q_pace,
      },
    };
  }

  // =========================================================================
  // MODEL INFERENCE
  // =========================================================================

  function predict(features) {
    if (!MODEL) throw new Error('Model not loaded');

    const featureNames = MODEL.features;
    const means = MODEL.scaler_mean;
    const stds = MODEL.scaler_std;
    const lrCoef = MODEL.lr_coef;
    const lrIntercept = MODEL.lr_intercept;
    const ridgeCoef = MODEL.ridge_coef;
    const ridgeIntercept = MODEL.ridge_intercept;

    // Scale features
    const scaled = featureNames.map((name, i) => {
      const val = features[name] || 0;
      const s = stds[i] || 1;
      return s !== 0 ? (val - means[i]) / s : 0;
    });

    // Logistic regression: raw P(leader wins)
    let z = lrIntercept;
    for (let i = 0; i < scaled.length; i++) {
      z += lrCoef[i] * scaled[i];
    }
    const rawProb = 1 / (1 + Math.exp(-z));

    // Calibrated probability
    const calProb = calibrate(rawProb);

    // Ridge regression: predicted margin
    let margin = ridgeIntercept;
    for (let i = 0; i < scaled.length; i++) {
      margin += ridgeCoef[i] * scaled[i];
    }

    return { rawProb, calProb, predictedMargin: margin };
  }

  // =========================================================================
  // SIGNAL GENERATION (Sniper Mode)
  // =========================================================================

  function generateSignals(game, possessions, opts = {}) {
    const result = extractFeatures(game, possessions, opts);
    if (!result) return [];

    const { features, gameState } = result;
    const { rawProb, calProb, predictedMargin } = predict(features);

    const q3Lead = gameState.q3Lead;
    const q3LeadAbs = Math.abs(q3Lead);

    // Skip tied games
    if (q3LeadAbs < 1) return [];

    // Skip extreme blowouts (worthless odds)
    if (q3LeadAbs > 30) return [];

    const thresholds = MODEL.sniper_thresholds || {};
    const minConf = thresholds.min_confidence || 0.93;
    const minQuality = thresholds.min_lead_quality || 0.40;
    const regimeFloors = thresholds.regime_floors || {};

    // Determine regime
    let regime;
    if (q3LeadAbs >= 20) regime = 'BLOWOUT';
    else if (q3LeadAbs >= 12) regime = 'COMFORTABLE';
    else if (q3LeadAbs >= 6) regime = 'COMPETITIVE';
    else regime = 'TIGHT';

    // Leader probability
    const leaderProb = calProb;

    // Regime-specific confidence floor
    const confFloor = regimeFloors[regime] || minConf;

    // ---- SNIPER GATES ----

    // Gate 1: Confidence
    if (leaderProb < confFloor) return [];

    // Gate 2: Lead quality
    const leadQuality = features.sniper_heuristic || 0;
    if (leadQuality < minQuality) return [];

    // All gates passed - generate signal
    const leader = q3Lead > 0 ? gameState.homeTeam : gameState.awayTeam;
    const trailer = q3Lead > 0 ? gameState.awayTeam : gameState.homeTeam;

    const mktMLOdds = estimateMLOdds(q3LeadAbs);
    const liveSpread = q3Lead * 0.78;

    const signals = [];

    // Primary signal: Leader to WIN
    signals.push({
      signalType: 'SNIPER',
      direction: q3Lead > 0 ? 'HOME' : 'AWAY',
      team: leader,
      trailer: trailer,
      confidence: leaderProb,
      predictedMargin: predictedMargin,
      liveSpread: liveSpread,
      estimatedOdds: mktMLOdds,
      leadQuality: leadQuality,
      leadSustainability: features.lead_sustainability || 0,
      momentumAligned: features.momentum_aligned || 0,
      dominatedAllQs: features.dominated_all_qs || 0,
      regime: regime,
      q3Lead: q3Lead,
      tier: getSniperTier(leaderProb),
      // Bet instruction
      betMarket: 'Live Moneyline',
      betTeam: leader,
      betInstruction: `${leader} to WIN`,
      betOddsNote: `Model: ${(leaderProb * 100).toFixed(1)}% win probability. Estimated odds ~${Math.round(mktMLOdds)}. Bet leader ML.`,
      betMinOdds: Math.round(mktMLOdds),
      // Spread alternative
      altSpreadTeam: leader,
      altSpreadLine: q3Lead > 0
        ? -(Math.round(Math.abs(liveSpread) * 2) / 2)
        : (Math.round(Math.abs(liveSpread) * 2) / 2),
    });

    return signals;
  }

  // =========================================================================
  // HELPERS
  // =========================================================================

  function getSniperTier(confidence) {
    if (confidence >= 0.99) return 'DIAMOND';
    if (confidence >= 0.97) return 'PLATINUM';
    if (confidence >= 0.95) return 'GOLD';
    if (confidence >= 0.93) return 'SILVER';
    return 'BRONZE';
  }

  function estimateMLOdds(leadAbs) {
    if (leadAbs === 0) return -110;
    const winProb = normalCDF(leadAbs / Q4_SIGMA);
    if (winProb >= 0.99) return -10000;
    if (winProb <= 0.01) return 10000;
    if (winProb >= 0.5) return Math.round(-(winProb / (1 - winProb)) * 100);
    return Math.round(((1 - winProb) / winProb) * 100);
  }

  function probToAmerican(p) {
    if (p <= 0.01) return 10000;
    if (p >= 0.99) return -10000;
    if (p >= 0.5) return Math.round(-(p / (1 - p)) * 100);
    return Math.round(((1 - p) / p) * 100);
  }

  function normalCDF(x) {
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
    const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.sqrt(2);
    const t = 1 / (1 + p * x);
    const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return 0.5 * (1 + sign * y);
  }

  // ---- Lead time series ----
  function buildLeadTimeSeries(possessions, interval) {
    if (!possessions.length) return [0];
    const maxTime = 2160;
    const n = Math.floor(maxTime / interval) + 1;
    const ts = new Array(n).fill(0);
    let idx = 0, currentLead = 0;
    for (let i = 0; i < n; i++) {
      const t = i * interval;
      while (idx < possessions.length && possessions[idx].timestamp <= t) {
        currentLead = possessions[idx].differential;
        idx++;
      }
      ts[i] = currentLead;
    }
    return ts;
  }

  function buildScoringEvents(q3Possessions, allPossessions) {
    return q3Possessions
      .filter(p => p.event === 'score')
      .map(p => {
        const prevIdx = allPossessions.indexOf(p) - 1;
        const prev = prevIdx >= 0 ? allPossessions[prevIdx] : { homeScore: 0, awayScore: 0 };
        return {
          period: p.quarter,
          secs: p.timestamp,
          pts: Math.abs((p.homeScore - prev.homeScore) + (p.awayScore - prev.awayScore)) || 2,
          lead: p.homeScore - p.awayScore,
          homeScore: p.homeScore,
          awayScore: p.awayScore,
          isHome: p.team === 'home',
        };
      });
  }

  // ---- Statistical helpers ----
  function mean(arr) {
    if (!arr.length) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  function std(arr) {
    if (arr.length < 2) return 0;
    const m = mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
  }

  function linreg(x, y) {
    const n = x.length;
    if (n < 2) return { slope: 0, r2: 0 };
    const mx = mean(x), my = mean(y);
    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - mx, dy = y[i] - my;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }
    const slope = denX > 0 ? num / denX : 0;
    const r = (denX > 0 && denY > 0) ? num / Math.sqrt(denX * denY) : 0;
    return { slope, r2: r * r };
  }

  // ---- Lead dynamics ----
  function leadDynamics(ts, q3Lead) {
    const f = {};
    const n = ts.length;
    if (n < 4) {
      return {
        lead_vol: 0, lead_range: 0, max_home_lead: 0, max_away_lead: 0,
        lead_changes: 0, ties: 0, lead_slope: 0, lead_r2: 0,
        lead_early: 0, lead_mid: 0, lead_late: 0,
        lead_momentum: 0, lead_accel: 0,
        halftime_lead: 0, q3_lead_growth: 0, lead_stability: 0, lead_mean_abs: 0,
      };
    }
    f.lead_vol = std(ts);
    f.lead_range = Math.max(...ts) - Math.min(...ts);
    f.max_home_lead = Math.max(...ts);
    f.max_away_lead = -Math.min(...ts);
    let changes = 0;
    for (let i = 1; i < n; i++) {
      if (Math.sign(ts[i]) !== Math.sign(ts[i - 1]) && ts[i] !== 0 && ts[i - 1] !== 0) changes++;
    }
    f.lead_changes = Math.floor(changes / 2);
    f.ties = ts.filter(v => v === 0).length;
    const x = Array.from({ length: n }, (_, i) => i);
    const reg = linreg(x, ts);
    f.lead_slope = reg.slope;
    f.lead_r2 = reg.r2;
    const third = Math.max(1, Math.floor(n / 3));
    f.lead_early = mean(ts.slice(0, third));
    f.lead_mid = mean(ts.slice(third, 2 * third));
    f.lead_late = mean(ts.slice(2 * third));
    f.lead_momentum = f.lead_late - f.lead_mid;
    f.lead_accel = (f.lead_late - f.lead_mid) - (f.lead_mid - f.lead_early);
    const halfIdx = Math.min(n - 1, Math.floor(n * 2 / 3));
    f.halftime_lead = ts[halfIdx];
    f.q3_lead_growth = q3Lead - f.halftime_lead;
    if (q3Lead > 0) f.lead_stability = ts.filter(v => v >= q3Lead * 0.5).length / n;
    else if (q3Lead < 0) f.lead_stability = ts.filter(v => v <= q3Lead * 0.5).length / n;
    else f.lead_stability = 0.5;
    f.lead_mean_abs = mean(ts.map(Math.abs));
    return f;
  }

  // ---- DNA features ----
  function dnaFeatures(ts) {
    const f = {};
    const n = ts.length;
    if (n < 8) {
      return {
        dna_hurst: 0.5, dna_ac1: 0, dna_ac3: 0, dna_ac5: 0,
        dna_fft_dom: 0, dna_fft_lo: 0, dna_fft_hi: 0,
        dna_entropy: 0, dna_xing_rate: 0, dna_ou_theta: 0,
        dna_max_run: 0, dna_avg_run: 0, dna_skew: 0, dna_kurt: 0,
      };
    }
    f.dna_hurst = hurst(ts);
    const m = mean(ts);
    const c = ts.map(v => v - m);
    const v = c.reduce((s, x) => s + x * x, 0) / n + 1e-10;
    for (const lag of [1, 3, 5]) {
      if (lag < n) {
        let ac = 0;
        for (let i = 0; i < n - lag; i++) ac += c[i] * c[i + lag];
        f[`dna_ac${lag}`] = ac / ((n - lag) * v);
      } else {
        f[`dna_ac${lag}`] = 0;
      }
    }
    try {
      const centered = ts.map(x => x - m);
      const half = Math.floor(n / 2);
      const fftMag = [];
      for (let k = 0; k < half; k++) {
        let re = 0, im = 0;
        for (let j = 0; j < n; j++) {
          const angle = -2 * Math.PI * k * j / n;
          re += centered[j] * Math.cos(angle);
          im += centered[j] * Math.sin(angle);
        }
        fftMag.push(Math.sqrt(re * re + im * im));
      }
      if (fftMag.length > 1) {
        const total = fftMag.slice(1).reduce((a, b) => a + b, 0) + 1e-10;
        let domIdx = 1;
        for (let k = 2; k < fftMag.length; k++) {
          if (fftMag[k] > fftMag[domIdx]) domIdx = k;
        }
        f.dna_fft_dom = domIdx / n;
        const mid = Math.max(1, Math.floor(fftMag.length / 2));
        f.dna_fft_lo = fftMag.slice(1, mid).reduce((a, b) => a + b, 0) / total;
        f.dna_fft_hi = fftMag.slice(mid).reduce((a, b) => a + b, 0) / total;
      } else {
        f.dna_fft_dom = 0; f.dna_fft_lo = 0; f.dna_fft_hi = 0;
      }
    } catch {
      f.dna_fft_dom = 0; f.dna_fft_lo = 0; f.dna_fft_hi = 0;
    }
    const diffs = [];
    for (let i = 1; i < n; i++) diffs.push(ts[i] - ts[i - 1]);
    if (diffs.length > 0) {
      const bins = [-Infinity, -5, -2, 0, 2, 5, Infinity];
      const hist = new Array(bins.length - 1).fill(0);
      for (const d of diffs) {
        for (let b = 0; b < bins.length - 1; b++) {
          if (d >= bins[b] && d < bins[b + 1]) { hist[b]++; break; }
        }
      }
      const total = diffs.length;
      let entropy = 0;
      for (const h of hist) {
        if (h > 0) { const p = h / total; entropy -= p * Math.log2(p); }
      }
      f.dna_entropy = entropy;
    } else {
      f.dna_entropy = 0;
    }
    let crossings = 0;
    for (let i = 1; i < n; i++) {
      if (Math.sign(ts[i]) !== Math.sign(ts[i - 1]) && ts[i] !== 0) crossings++;
    }
    f.dna_xing_rate = crossings / Math.max(1, n - 1);
    if (diffs.length >= 4) {
      const xArr = ts.slice(0, -1);
      const reg = linreg(xArr, diffs);
      f.dna_ou_theta = -reg.slope;
    } else {
      f.dna_ou_theta = 0;
    }
    const runs = [];
    let run = 1;
    for (let i = 1; i < diffs.length; i++) {
      if (Math.sign(diffs[i]) === Math.sign(diffs[i - 1]) && diffs[i] !== 0) run++;
      else { if (run > 0) runs.push(run); run = 1; }
    }
    if (run > 0) runs.push(run);
    f.dna_max_run = runs.length ? Math.max(...runs) : 0;
    f.dna_avg_run = runs.length ? mean(runs) : 0;
    const s = std(ts);
    if (s > 0) {
      const z = ts.map(x => (x - mean(ts)) / s);
      f.dna_skew = mean(z.map(x => x ** 3));
      f.dna_kurt = mean(z.map(x => x ** 4)) - 3;
    } else {
      f.dna_skew = 0;
      f.dna_kurt = 0;
    }
    return f;
  }

  function hurst(ts) {
    const n = ts.length;
    const maxLag = Math.min(Math.floor(n / 2), 20);
    if (maxLag < 3) return 0.5;
    const lags = [], rsVals = [];
    for (let lag = 2; lag < maxLag; lag++) {
      const sub = ts.slice(0, lag);
      const m = mean(sub);
      const dev = [];
      let cum = 0;
      for (const x of sub) { cum += x - m; dev.push(cum); }
      const R = Math.max(...dev) - Math.min(...dev);
      const S = std(sub) || 1e-10;
      lags.push(Math.log(lag));
      rsVals.push(Math.log(R / S + 1e-10));
    }
    if (lags.length < 2) return 0.5;
    const reg = linreg(lags, rsVals);
    return Math.max(0, Math.min(1, reg.slope));
  }

  // ---- TA features ----
  function taFeatures(ts) {
    const f = {};
    const n = ts.length;
    if (n < 10) {
      return {
        ta_rsi: 50, ta_bb_pos: 0.5, ta_bb_width: 0, ta_macd: 0,
        ta_roc5: 0, ta_roc15: 0, ta_mom_div: 0, ta_willr: -50,
      };
    }
    const diffs = [];
    for (let i = 1; i < n; i++) diffs.push(ts[i] - ts[i - 1]);
    const w = Math.min(14, diffs.length);
    const recent = diffs.slice(-w);
    const gains = recent.filter(d => d > 0).reduce((a, b) => a + b, 0);
    const losses = -recent.filter(d => d < 0).reduce((a, b) => a + b, 0);
    f.ta_rsi = losses > 0 ? 100 - 100 / (1 + gains / losses) : (gains > 0 ? 100 : 50);
    const bw = Math.min(20, n);
    const bRecent = ts.slice(-bw);
    const bMu = mean(bRecent);
    const bSd = std(bRecent) + 1e-10;
    f.ta_bb_pos = (ts[n - 1] - (bMu - 2 * bSd)) / (4 * bSd + 1e-10);
    f.ta_bb_width = 4 * bSd / (Math.abs(bMu) + 1e-10);
    if (n >= 12) {
      f.ta_macd = emaVal(ts, Math.min(5, Math.floor(n / 2))) - emaVal(ts, Math.min(12, n - 1));
    } else {
      f.ta_macd = 0;
    }
    f.ta_roc5 = n > 5 ? ts[n - 1] - ts[n - 6] : 0;
    f.ta_roc15 = n > 15 ? ts[n - 1] - ts[n - 16] : 0;
    const half = Math.floor(n / 2);
    if (half > 1) {
      const firstMax = Math.max(...ts.slice(0, half).map(Math.abs));
      const secondMax = Math.max(...ts.slice(half).map(Math.abs));
      const d1 = [];
      for (let i = 1; i < half; i++) d1.push(Math.abs(ts[i] - ts[i - 1]));
      const d2 = [];
      for (let i = half + 1; i < n; i++) d2.push(Math.abs(ts[i] - ts[i - 1]));
      f.ta_mom_div = (secondMax - firstMax) - (mean(d2) - mean(d1)) * 5;
    } else {
      f.ta_mom_div = 0;
    }
    const wr = Math.min(20, n);
    const wrSlice = ts.slice(-wr);
    const hi = Math.max(...wrSlice);
    const lo = Math.min(...wrSlice);
    f.ta_willr = hi !== lo ? (hi - ts[n - 1]) / (hi - lo) * -100 : -50;
    return f;
  }

  function emaVal(ts, window) {
    const alpha = 2 / (window + 1);
    let val = ts[0];
    for (let i = 1; i < ts.length; i++) val = alpha * ts[i] + (1 - alpha) * val;
    return val;
  }

  // ---- Scoring patterns ----
  function scoringPatterns(events, q3Home, q3Away) {
    const f = {};
    if (!events || events.length === 0) {
      return {
        sp_late_q3_margin: 0, sp_late_q3_pts: 0, sp_q3_last5_pts: 0,
        sp_home_3pt_rate: 0, sp_away_3pt_rate: 0, sp_ft_rate: 0,
        sp_home_largest_run: 0, sp_away_largest_run: 0,
        sp_home_dry_spell: 0, sp_away_dry_spell: 0, sp_lead_at_each_break: 0,
      };
    }
    const late = events.filter(e => e.period === 3 && e.secs >= 1980);
    if (late.length >= 2) {
      f.sp_late_q3_margin = late[late.length - 1].lead - late[0].lead;
      f.sp_late_q3_pts = late.reduce((s, e) => s + e.pts, 0);
    } else {
      f.sp_late_q3_margin = 0; f.sp_late_q3_pts = 0;
    }
    const l5 = events.filter(e => e.period === 3 && e.secs >= 1860);
    f.sp_q3_last5_pts = l5.reduce((s, e) => s + e.pts, 0);
    const home3 = events.filter(e => e.isHome && e.pts === 3).length;
    const away3 = events.filter(e => !e.isHome && e.pts === 3).length;
    const homeFG = events.filter(e => e.isHome && e.pts >= 2).length;
    const awayFG = events.filter(e => !e.isHome && e.pts >= 2).length;
    f.sp_home_3pt_rate = homeFG > 0 ? home3 / homeFG : 0;
    f.sp_away_3pt_rate = awayFG > 0 ? away3 / awayFG : 0;
    const fts = events.filter(e => e.pts === 1).length;
    f.sp_ft_rate = events.length > 0 ? fts / events.length : 0;
    f.sp_home_largest_run = largestRun(events, true);
    f.sp_away_largest_run = largestRun(events, false);
    f.sp_home_dry_spell = longestDry(events, true);
    f.sp_away_dry_spell = longestDry(events, false);
    const breaks = [];
    for (const q of [1, 2, 3]) {
      const qEvents = events.filter(e => e.period === q);
      if (qEvents.length) breaks.push(qEvents[qEvents.length - 1].lead);
    }
    f.sp_lead_at_each_break = breaks.length ? std(breaks) : 0;
    return f;
  }

  function largestRun(events, isHome) {
    let maxRun = 0, run = 0;
    for (const e of events) {
      if (e.isHome === isHome) run += e.pts;
      else { maxRun = Math.max(maxRun, run); run = 0; }
    }
    return Math.max(maxRun, run);
  }

  function longestDry(events, isHome) {
    const team = events.filter(e => e.isHome === isHome);
    if (team.length < 2) return 0;
    let maxGap = 0;
    for (let i = 1; i < team.length; i++) maxGap = Math.max(maxGap, team[i].secs - team[i - 1].secs);
    return maxGap;
  }

  // =========================================================================
  // PUBLIC API
  // =========================================================================
  return {
    loadModel,
    isModelLoaded,
    extractFeatures,
    predict,
    generateSignals,
    getSniperTier,
    probToAmerican,
    normalCDF,
    estimateMLOdds,
    calibrate,
  };

})();
