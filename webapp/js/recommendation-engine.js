/**
 * NBA Ultra Betting Engine v1.0 — JavaScript Port
 * =================================================
 * Direct port of the Python Ultra Engine (src/ultra_engine.py) to JavaScript
 * for live tonight's pick generation.
 *
 * Uses the SAME logic as the backtest:
 * - Gravitational Floor Theory (GFT)
 * - Bayesian Edge Quantification (BEQ)
 * - Entropic Stability Index (ESI)
 * - Inverse Market Asymmetry Detection (IMAD)
 * - Edge-Maximized Parlay Construction (EMPC)
 * - 6-Gate Quality Cascade
 *
 * Config loaded from output/ultra_engine_config.json (AutoResearch-optimized).
 */

window.RecommendationEngine = (function () {
  'use strict';

  // =========================================================================
  // CONFIGURATION — Matches optimized Ultra Engine config
  // Loaded from output/ultra_engine_config.json at runtime if available
  // =========================================================================

  const CONFIG = {
    // Player eligibility
    MIN_GAMES: 17,
    MIN_MINUTES: 21,
    WARM_UP_GAMES: 20,

    // Gravitational Floor Theory (GFT)
    GFT_WINDOWS: [5, 10, 15],
    GFT_DECAY_RATE: 0.9154,
    GFT_GRAVITY_STRENGTH: 0.3593,
    GFT_MIN_CLEARANCE: 0.5,
    GFT_CONVERGENCE_MAX_SPREAD: 5.6659,

    // Bayesian Edge Quantification (BEQ)
    BEQ_PRIOR_ALPHA: 1.0,
    BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.8709,
    BEQ_MIN_EDGE: 0.05,

    // Entropic Stability Index (ESI)
    ESI_BINS: 6,
    ESI_MAX_ENTROPY: 0.8274,
    ESI_TREND_WEIGHT: 0.3007,

    // Inverse Market Asymmetry Detection (IMAD)
    IMAD_MIN_ASYMMETRY: 0.1,
    IMAD_VOLUME_DISCOUNT: 0.02,

    // Quality Gates
    GATE_MIN_GFT_SCORE: 0.4,
    GATE_MIN_BEQ_EDGE: 0.05,
    GATE_MIN_ESI_STABILITY: 0.15,
    GATE_MIN_IMAD_SCORE: 0.02,
    GATE_MIN_HIT_RATE: 0.8101,
    GATE_MIN_COMBINED: 0.55,

    // Bet Type Selection
    SINGLE_MIN_SCORE: 0.75,
    MULTI_SINGLE_MIN_SCORE: 0.6685,
    PARLAY_LEG_MIN_SCORE: 0.6,

    // Parlay Construction
    PARLAY_MIN_LEGS: 2,
    PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.3285,
    PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.1,

    // Odds Filters
    MIN_ODDS: -324,
    MAX_ODDS: -110,
    PREFERRED_ODDS_RANGE: [-500, -150],

    // Bankroll
    UNIT_SIZE: 100,
    MAX_DAILY_UNITS: 5,
    KELLY_FRACTION: 0.25,
  };

  // =========================================================================
  // ODDS UTILITIES
  // =========================================================================

  function americanToDecimal(odds) {
    return odds > 0 ? 1 + odds / 100 : 1 + 100 / Math.abs(odds);
  }

  function decimalToAmerican(decimal) {
    if (decimal >= 2.0) return Math.round((decimal - 1) * 100);
    return Math.round(-100 / (decimal - 1));
  }

  function formatOdds(odds) {
    return odds > 0 ? `+${odds}` : `${odds}`;
  }

  function impliedProbability(odds) {
    return 1 / americanToDecimal(odds);
  }

  const STAT_MAP = { points: 'pts', rebounds: 'reb', assists: 'ast', pra: 'pra' };
  const STAT_LABELS = { points: 'PTS', rebounds: 'REB', assists: 'AST', pts: 'PTS', reb: 'REB', ast: 'AST', pra: 'PRA' };

  // =========================================================================
  // MATH UTILITIES
  // =========================================================================

  /**
   * Approximate inverse CDF of Beta distribution (normal approximation).
   * Matches Python beta_ppf().
   */
  function betaPPF(alpha, betaParam, p) {
    if (alpha <= 0 || betaParam <= 0) return 0.5;
    if (p <= 0 || p >= 1) return alpha / (alpha + betaParam);

    const mean = alpha / (alpha + betaParam);
    const variance = (alpha * betaParam) / ((alpha + betaParam) ** 2 * (alpha + betaParam + 1));
    const std = Math.sqrt(variance > 0 ? variance : 0.001);

    // Abramowitz and Stegun rational approximation for inverse normal
    let z;
    if (p < 0.5) {
      const t = Math.sqrt(-2 * Math.log(p));
      z = -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
            (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t));
    } else {
      const t = Math.sqrt(-2 * Math.log(1 - p));
      z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
          (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
    }

    return Math.max(0, Math.min(1, mean + z * std));
  }

  /**
   * Normalized Shannon entropy. Matches Python shannon_entropy().
   */
  function shannonEntropy(values, nBins) {
    if (values.length < 3) return 1.0;

    const minV = Math.min(...values);
    const maxV = Math.max(...values);
    if (maxV === minV) return 0.0;

    const binWidth = (maxV - minV) / nBins;
    const counts = new Array(nBins).fill(0);
    for (const v of values) {
      const idx = Math.min(Math.floor((v - minV) / binWidth), nBins - 1);
      counts[idx]++;
    }

    const n = values.length;
    let entropy = 0;
    for (const c of counts) {
      if (c > 0) {
        const p = c / n;
        entropy -= p * Math.log2(p);
      }
    }

    const maxEntropy = Math.log2(nBins);
    return maxEntropy > 0 ? entropy / maxEntropy : 1.0;
  }

  // =========================================================================
  // PLAYER MODEL — with minutes filtering (matches Python UltraPlayerModel)
  // =========================================================================

  const PlayerModel = {
    profiles: {},

    reset() { this.profiles = {}; },

    update(name, stats, date, team, opponent) {
      if (!this.profiles[name]) this.profiles[name] = { games: [], team: '' };
      const enriched = { ...stats, date, team, opponent };
      if (enriched.pra === undefined) {
        enriched.pra = (enriched.pts || 0) + (enriched.reb || 0) + (enriched.ast || 0);
      }
      this.profiles[name].games.push(enriched);
      this.profiles[name].team = team;
      if (this.profiles[name].games.length > 50) {
        this.profiles[name].games = this.profiles[name].games.slice(-50);
      }
    },

    getProfile(name) {
      return this.profiles[name] || null;
    },

    /**
     * Get recent values, filtering low-minute games (matches Python get_values).
     */
    getValues(name, statKey, window, minMinutes) {
      const profile = this.profiles[name];
      if (!profile) return null;

      let games = profile.games;
      // Filter low-minute games when minMinutes is specified
      if (minMinutes && minMinutes > 0) {
        games = games.filter(g => (g.min || 0) >= minMinutes);
      }

      if (window) {
        games = games.slice(-window);
      }
      if (!games.length) return null;
      return games.map(g => g[statKey] || 0);
    },

    getTeam(name) {
      const p = this.profiles[name];
      return p ? p.team : null;
    },

    gameCount(name) {
      const p = this.profiles[name];
      return p ? p.games.length : 0;
    },

    isWarm(name, minGames) {
      return this.gameCount(name) >= minGames;
    },

    getRecentMinutes(name, window) {
      const vals = this.getValues(name, 'min', window || 5);
      if (!vals || !vals.length) return 0;
      return vals.reduce((s, v) => s + v, 0) / vals.length;
    },
  };

  // =========================================================================
  // GRAVITATIONAL FLOOR THEORY (GFT) — Matches Python compute_gft()
  // =========================================================================

  function computeGFT(name, statKey, line) {
    const windows = CONFIG.GFT_WINDOWS;
    const decay = CONFIG.GFT_DECAY_RATE;
    const gravity = CONFIG.GFT_GRAVITY_STRENGTH;
    const minClearance = CONFIG.GFT_MIN_CLEARANCE;
    const maxSpread = CONFIG.GFT_CONVERGENCE_MAX_SPREAD;

    const floors = [];
    const clearances = [];

    for (const w of windows) {
      const values = PlayerModel.getValues(name, statKey, w, CONFIG.MIN_MINUTES);
      if (!values || values.length < Math.min(w, 5)) return null;

      const n = values.length;
      const weights = [];
      for (let i = 0; i < n; i++) {
        weights.push(Math.pow(decay, n - 1 - i));
      }
      const totalWeight = weights.reduce((s, w) => s + w, 0);

      // Weighted mean (talent estimate)
      let weightedMean = 0;
      for (let i = 0; i < n; i++) {
        weightedMean += values[i] * weights[i];
      }
      weightedMean /= totalWeight;

      // Sort by value with weights for weighted percentile
      const sortedWeighted = values.map((v, i) => ({ val: v, wt: weights[i] }))
        .sort((a, b) => a.val - b.val);

      // Weighted 10th percentile
      let cumWeight = 0;
      let p10Value = sortedWeighted[0].val;
      for (const { val, wt } of sortedWeighted) {
        cumWeight += wt;
        if (cumWeight / totalWeight >= 0.10) {
          p10Value = val;
          break;
        }
      }

      // Gravitational pull: floor is pulled toward the mean
      const rawFloor = p10Value;
      const gravFloor = rawFloor + gravity * (weightedMean - rawFloor);

      const clearance = gravFloor - line;
      if (clearance < minClearance) return null;

      floors.push(gravFloor);
      clearances.push(clearance);
    }

    // Check convergence across windows
    const floorSpread = Math.max(...floors) - Math.min(...floors);
    if (floorSpread > maxSpread) return null;

    const convergence = Math.max(0, 1 - floorSpread / maxSpread);
    const avgClearance = clearances.reduce((s, c) => s + c, 0) / clearances.length;
    const depth = Math.min(1.0, avgClearance / 8.0);

    const score = convergence * 0.5 + depth * 0.5;

    return {
      score,
      convergence,
      depth,
      floors,
      clearances,
      floorSpread,
      avgClearance,
    };
  }

  // =========================================================================
  // BAYESIAN EDGE QUANTIFICATION (BEQ) — Matches Python compute_beq()
  // =========================================================================

  function computeBEQ(name, statKey, line, marketOdds) {
    const values = PlayerModel.getValues(name, statKey, 20, CONFIG.MIN_MINUTES);
    if (!values || values.length < 10) return null;

    const hits = values.filter(v => v > line).length;
    const misses = values.length - hits;

    const alpha = CONFIG.BEQ_PRIOR_ALPHA + hits;
    const betaParam = CONFIG.BEQ_PRIOR_BETA + misses;

    const meanProb = alpha / (alpha + betaParam);
    const ciLevel = 1 - CONFIG.BEQ_CREDIBLE_LEVEL;
    const lowerBound = betaPPF(alpha, betaParam, ciLevel);

    const mktProb = impliedProbability(marketOdds);
    const edge = lowerBound - mktProb;

    // Extended window for stability check
    const valuesExtended = PlayerModel.getValues(name, statKey, 30, CONFIG.MIN_MINUTES);
    let extendedHitRate = null;
    if (valuesExtended && valuesExtended.length >= 20) {
      extendedHitRate = valuesExtended.filter(v => v > line).length / valuesExtended.length;
    }

    return {
      meanProb,
      lowerBound,
      marketProb: mktProb,
      edge,
      alpha,
      beta: betaParam,
      hits,
      total: values.length,
      hitRate: hits / values.length,
      extendedHitRate,
    };
  }

  // =========================================================================
  // ENTROPIC STABILITY INDEX (ESI) — Matches Python compute_esi()
  // =========================================================================

  function computeESI(name, statKey) {
    const values = PlayerModel.getValues(name, statKey, 20, CONFIG.MIN_MINUTES);
    if (!values || values.length < 10) return null;

    const nBins = CONFIG.ESI_BINS;
    const trendWeight = CONFIG.ESI_TREND_WEIGHT;

    // 1. Distributional entropy
    const distEntropy = shannonEntropy(values, nBins);

    // 2. Trend stability
    const mid = Math.floor(values.length / 2);
    const firstHalf = values.slice(0, mid);
    const secondHalf = values.slice(mid);

    const meanFirst = firstHalf.reduce((s, v) => s + v, 0) / firstHalf.length;
    const meanSecond = secondHalf.reduce((s, v) => s + v, 0) / secondHalf.length;
    const overallMean = values.reduce((s, v) => s + v, 0) / values.length;

    if (overallMean === 0) return null;

    const trendShift = Math.abs(meanSecond - meanFirst) / overallMean;
    const trendStability = Math.max(0, 1 - trendShift / 0.3);

    // 3. Tail risk
    const sortedVals = [...values].sort((a, b) => a - b);
    const p15 = sortedVals[Math.max(0, Math.floor(sortedVals.length * 0.15))];
    const recent = values.slice(-5);
    const tailEvents = recent.filter(v => v <= p15).length;
    const tailRisk = 1 - tailEvents / recent.length;

    // Combined ESI
    const entropyScore = Math.max(0, 1 - distEntropy / CONFIG.ESI_MAX_ENTROPY);
    let stability = (1 - trendWeight) * entropyScore + trendWeight * trendStability;
    stability = stability * tailRisk;

    return {
      stability,
      entropy: distEntropy,
      trendStability,
      tailRisk,
      mean: overallMean,
      trendShift,
    };
  }

  // =========================================================================
  // INVERSE MARKET ASYMMETRY DETECTION (IMAD) — Matches Python compute_imad()
  // =========================================================================

  function computeIMAD(beqResult, esiResult, gftResult) {
    if (!beqResult || !esiResult || !gftResult) return null;

    const bayesianEdge = beqResult.edge;
    const stabilityPremium = esiResult.stability * 0.05;
    const depthPremium = gftResult.depth * 0.05;

    const totalAsymmetry = bayesianEdge + stabilityPremium + depthPremium;

    let strongSignals = 0;
    if (beqResult.edge > CONFIG.BEQ_MIN_EDGE) strongSignals++;
    if (esiResult.stability > CONFIG.GATE_MIN_ESI_STABILITY) strongSignals++;
    if (gftResult.score > CONFIG.GATE_MIN_GFT_SCORE) strongSignals++;

    const agreementFactor = strongSignals / 3.0;
    const score = totalAsymmetry * agreementFactor;

    return {
      score,
      totalAsymmetry,
      bayesianEdge,
      stabilityPremium,
      depthPremium,
      strongSignals,
      agreementFactor,
    };
  }

  // =========================================================================
  // ULTRA SIGNAL — 6-Gate Quality Cascade (matches Python compute_ultra_signal)
  // =========================================================================

  function computeUltraSignal(playerName, statKey, line, marketOdds) {
    // Gate 0: Data requirements
    if (!PlayerModel.isWarm(playerName, CONFIG.MIN_GAMES)) return null;
    const recentMins = PlayerModel.getRecentMinutes(playerName, 5);
    if (recentMins < CONFIG.MIN_MINUTES) return null;

    // Gate 1: Gravitational Floor Theory
    const gft = computeGFT(playerName, statKey, line);
    if (!gft || gft.score < CONFIG.GATE_MIN_GFT_SCORE) return null;

    // Gate 2: Bayesian Edge Quantification
    const beq = computeBEQ(playerName, statKey, line, marketOdds);
    if (!beq || beq.edge < CONFIG.GATE_MIN_BEQ_EDGE) return null;

    // Gate 3: Entropic Stability Index
    const esi = computeESI(playerName, statKey);
    if (!esi || esi.stability < CONFIG.GATE_MIN_ESI_STABILITY) return null;

    // Gate 4: Inverse Market Asymmetry Detection
    const imad = computeIMAD(beq, esi, gft);
    if (!imad || imad.score < CONFIG.GATE_MIN_IMAD_SCORE) return null;

    // Gate 5: Raw hit rate sanity check
    if (beq.hitRate < CONFIG.GATE_MIN_HIT_RATE) return null;

    // Gate 6: Extended window consistency (anti-overfitting)
    if (beq.extendedHitRate !== null) {
      if (beq.extendedHitRate < beq.hitRate - 0.10) return null;
      if (beq.extendedHitRate < CONFIG.GATE_MIN_HIT_RATE - 0.05) return null;
    }

    // Combined score: geometric mean of all signal components
    const components = [
      gft.score,
      Math.min(1.0, beq.edge / 0.20 + 0.5),
      esi.stability,
      Math.min(1.0, imad.score / 0.15 + 0.5),
    ];

    const logSum = components.reduce((s, c) => s + Math.log(Math.max(0.001, c)), 0);
    let combined = Math.exp(logSum / components.length);
    combined = Math.max(0, Math.min(1.0, combined));

    if (combined < CONFIG.GATE_MIN_COMBINED) return null;

    // EV calculation (conservative — uses lower bound)
    const decimal = americanToDecimal(marketOdds);
    const ev = beq.lowerBound * decimal - 1;

    // Kelly fraction
    const kellyFull = (beq.lowerBound * decimal - 1) / (decimal - 1);
    const kelly = Math.max(0, kellyFull * CONFIG.KELLY_FRACTION);

    return {
      player: playerName,
      stat: statKey,
      line,
      odds: marketOdds,
      combinedScore: combined,
      gft,
      beq,
      esi,
      imad,
      ev,
      kelly,
      hitRate: beq.hitRate,
      bayesianProb: beq.meanProb,
      lowerBoundProb: beq.lowerBound,
      marketImplied: beq.marketProb,
      edge: beq.edge,
    };
  }

  // =========================================================================
  // EVALUATE A PROP — Uses Ultra Signal (replaces legacy evaluateProp)
  // =========================================================================

  function evaluateProp(playerName, statType, line, odds) {
    const statKey = STAT_MAP[statType] || statType;

    // Odds filter
    if (odds < CONFIG.MIN_ODDS || odds > CONFIG.MAX_ODDS) return null;

    const signal = computeUltraSignal(playerName, statKey, line, odds);
    if (!signal) return null;

    const profile = PlayerModel.getProfile(playerName);
    const decimal = americanToDecimal(odds);

    // Map to the format expected by the rest of the app
    return {
      player: playerName,
      team: profile ? profile.team : '',
      statType,
      statLabel: STAT_LABELS[statType] || statType.toUpperCase(),
      line,
      odds,
      decimalOdds: decimal,
      engine: 'ultra',

      // Ultra Engine signal scores
      gft: signal.gft.score,
      beq: signal.beq.edge,
      esi: signal.esi.stability,
      imad: signal.imad.score,

      cascadeScore: signal.combinedScore,

      // Betting metrics
      impliedProb: signal.marketImplied,
      hitRate: signal.hitRate,
      edge: signal.edge,
      ev: signal.ev,
      avg: signal.esi.mean,
      floor: signal.gft.floors[signal.gft.floors.length - 1],
    };
  }

  // =========================================================================
  // FIND BEST PROP FOR A PLAYER FROM AVAILABLE LINES
  // =========================================================================

  function findBestProp(playerName, statType, fdLines) {
    if (!fdLines || typeof fdLines !== 'object') return null;

    let best = null;
    for (const [threshold, data] of Object.entries(fdLines)) {
      const line = parseFloat(threshold);
      const odds = data.overOdds;
      if (!odds) continue;

      const result = evaluateProp(playerName, statType, line, odds);
      if (!result) continue;

      if (!best || result.cascadeScore > best.cascadeScore) {
        best = result;
      }
    }
    return best;
  }

  // =========================================================================
  // BET TYPE SELECTOR — Matches Ultra Engine thresholds
  // =========================================================================

  function selectBetType(candidates) {
    const sorted = [...candidates].sort((a, b) => b.cascadeScore - a.cascadeScore);

    const elite = sorted.filter(c => c.cascadeScore >= CONFIG.SINGLE_MIN_SCORE);
    const strong = sorted.filter(c => c.cascadeScore >= CONFIG.MULTI_SINGLE_MIN_SCORE);
    const moderate = sorted.filter(c => c.cascadeScore >= CONFIG.PARLAY_LEG_MIN_SCORE);

    const recommendations = {
      singles: [],
      parlays: [],
      betType: 'none',
      reasoning: '',
    };

    // Tier 1: Elite singles
    if (elite.length >= 1) {
      recommendations.singles = elite.slice(0, CONFIG.MAX_DAILY_UNITS);
      recommendations.betType = elite.length === 1 ? 'single' : 'multi_single';
      recommendations.reasoning = `${elite.length} elite-confidence pick(s) found (score >= ${CONFIG.SINGLE_MIN_SCORE})`;
    }

    // Tier 2: Build parlays from strong candidates
    if (moderate.length >= CONFIG.PARLAY_MIN_LEGS) {
      const parlay = buildOptimalParlay(moderate);
      if (parlay) {
        recommendations.parlays.push(parlay);
        if (recommendations.betType === 'none') {
          recommendations.betType = 'parlay';
          recommendations.reasoning = `${moderate.length} strong candidates for EMPC parlay construction`;
        }
      }
    }

    // Include strong singles even if we have parlays
    if (recommendations.singles.length === 0 && strong.length > 0) {
      recommendations.singles = strong.slice(0, 3);
      if (recommendations.betType === 'none') {
        recommendations.betType = strong.length === 1 ? 'single' : 'multi_single';
        recommendations.reasoning = `${strong.length} strong-confidence pick(s) found`;
      }
    }

    return recommendations;
  }

  // =========================================================================
  // PARLAY CONSTRUCTION — EMPC (matches Python construct_optimal_parlays)
  // =========================================================================

  function computeCorrelation(values1, values2) {
    if (!values1 || !values2) return 1;
    const n = Math.min(values1.length, values2.length);
    if (n < 5) return 1;

    const v1 = values1.slice(-n);
    const v2 = values2.slice(-n);

    const mean1 = v1.reduce((s, v) => s + v, 0) / n;
    const mean2 = v2.reduce((s, v) => s + v, 0) / n;

    let cov = 0, var1 = 0, var2 = 0;
    for (let i = 0; i < n; i++) {
      const d1 = v1[i] - mean1;
      const d2 = v2[i] - mean2;
      cov += d1 * d2;
      var1 += d1 * d1;
      var2 += d2 * d2;
    }

    const denom = Math.sqrt(var1 * var2);
    return denom === 0 ? 1 : Math.abs(cov / denom);
  }

  function buildOptimalParlay(candidates) {
    // Sort by edge (highest first) — matches EMPC
    const sorted = [...candidates].sort((a, b) => b.edge - a.edge);

    const selected = [];
    const usedTeams = new Set();
    const usedPlayers = new Set();

    for (const leg of sorted) {
      if (selected.length >= CONFIG.PARLAY_MAX_LEGS) break;
      if (usedPlayers.has(leg.player)) continue;

      // Team diversification (no same-game legs)
      const team = leg.team;
      if (!CONFIG.PARLAY_SAME_GAME_ALLOWED && usedTeams.has(team)) continue;

      // Correlation check with existing legs
      let maxCorr = 0;
      for (const existing of selected) {
        const statKey1 = STAT_MAP[existing.statType] || existing.statType;
        const statKey2 = STAT_MAP[leg.statType] || leg.statType;
        const v1 = PlayerModel.getValues(existing.player, statKey1, 20, CONFIG.MIN_MINUTES);
        const v2 = PlayerModel.getValues(leg.player, statKey2, 20, CONFIG.MIN_MINUTES);
        const corr = computeCorrelation(v1, v2);
        maxCorr = Math.max(maxCorr, corr);
      }

      if (maxCorr > CONFIG.PARLAY_MAX_CORRELATION) continue;

      selected.push(leg);
      usedPlayers.add(leg.player);
      if (team) usedTeams.add(team);
    }

    if (selected.length < CONFIG.PARLAY_MIN_LEGS) return null;

    // Compute parlay metrics
    const decimal = selected.reduce((d, l) => d * l.decimalOdds, 1);
    const american = decimalToAmerican(decimal);
    const combinedHitRate = selected.reduce((p, l) => p * l.hitRate, 1);
    const totalEdge = selected.reduce((s, l) => s + l.edge, 0);
    const ev = combinedHitRate * decimal - 1;

    if (ev <= 0 || totalEdge < CONFIG.PARLAY_MIN_COMBINED_EDGE) return null;

    return {
      legs: selected,
      numLegs: selected.length,
      odds: american,
      decimalOdds: Math.round(decimal * 100) / 100,
      combinedHitRate: Math.round(combinedHitRate * 10000) / 10000,
      ev: Math.round(ev * 1000) / 1000,
      avgCascade: selected.reduce((s, l) => s + l.cascadeScore, 0) / selected.length,
      totalEdge,
    };
  }

  // =========================================================================
  // WALK-FORWARD BACKTEST WITH REAL ODDS
  // =========================================================================

  function runBacktest(boxScores, historicalOdds) {
    PlayerModel.reset();

    const sortedGames = [...boxScores].sort((a, b) => a.date.localeCompare(b.date));

    const oddsByDate = {};
    for (const od of (historicalOdds || [])) {
      if (!oddsByDate[od.date]) oddsByDate[od.date] = {};
      const key = od.gameKey || `${od.awayTeam}@${od.homeTeam}`;
      oddsByDate[od.date][key] = od;
    }

    const boxByDate = {};
    for (const g of boxScores) {
      (boxByDate[g.date] || (boxByDate[g.date] = [])).push(g);
    }

    const allSignals = [];
    const processedDates = new Set();
    const updatedDates = new Set();

    for (const game of sortedGames) {
      const date = game.date;

      if (!processedDates.has(date)) {
        processedDates.add(date);

        const dayOdds = oddsByDate[date] || {};
        const dayBoxes = boxByDate[date] || [];
        const candidates = [];

        for (const bg of dayBoxes) {
          const gameKey = `${bg.away}@${bg.home}`;
          const og = dayOdds[gameKey];
          if (!og || !og.playerProps) continue;

          for (const player of (bg.players || [])) {
            // Note: we do NOT filter on actual minutes played here to avoid
            // survivorship bias. The player model's gate checks (MIN_GAMES,
            // MIN_MINUTES on historical avg) handle eligibility using only
            // pre-game data. Filtering on actual minutes would exclude players
            // who got injured mid-game — bets a real bettor would have lost.
            const ptsLines = og.playerProps[player.name];
            if (ptsLines) {
              const prop = findBestProp(player.name, 'points', ptsLines);
              if (prop) {
                prop.actual = player.pts;
                prop.hit = player.pts > prop.line;
                prop.gameKey = gameKey;
                prop.date = date;
                candidates.push(prop);
              }
            }

            // PRA from dedicated PRA odds (if available in historical data)
            const praLines = og.praProps && og.praProps[player.name];
            if (praLines) {
              const praProp = findBestProp(player.name, 'pra', praLines);
              if (praProp) {
                const actualPra = (parseInt(player.pts) || 0) + (parseInt(player.reb) || 0) + (parseInt(player.ast) || 0);
                praProp.actual = actualPra;
                praProp.hit = actualPra > praProp.line;
                praProp.gameKey = gameKey;
                praProp.date = date;
                candidates.push(praProp);
              }
            }

            // Also check reb/ast lines from historical data
            const rebLines = og.rebProps && og.rebProps[player.name];
            if (rebLines) {
              const prop = findBestProp(player.name, 'rebounds', rebLines);
              if (prop) {
                prop.actual = parseInt(player.reb) || 0;
                prop.hit = prop.actual > prop.line;
                prop.gameKey = gameKey;
                prop.date = date;
                candidates.push(prop);
              }
            }
            const astLines = og.astProps && og.astProps[player.name];
            if (astLines) {
              const prop = findBestProp(player.name, 'assists', astLines);
              if (prop) {
                prop.actual = parseInt(player.ast) || 0;
                prop.hit = prop.actual > prop.line;
                prop.gameKey = gameKey;
                prop.date = date;
                candidates.push(prop);
              }
            }
          }
        }

        // Deduplicate: keep best cascade score per player per day
        candidates.sort((a, b) => b.cascadeScore - a.cascadeScore);
        const seenPlayers = new Set();
        const dedupedCandidates = [];
        for (const c of candidates) {
          if (!seenPlayers.has(c.player)) {
            seenPlayers.add(c.player);
            dedupedCandidates.push(c);
          }
        }

        if (dedupedCandidates.length > 0) {
          const recommendation = selectBetType(dedupedCandidates);

          for (const single of recommendation.singles) {
            allSignals.push({
              ...single,
              betType: 'single',
              date,
              pnl: single.hit
                ? Math.round((single.decimalOdds - 1) * CONFIG.UNIT_SIZE)
                : -CONFIG.UNIT_SIZE,
            });
          }

          for (const parlay of recommendation.parlays) {
            const parlayHit = parlay.legs.every(l => l.hit);
            allSignals.push({
              betType: 'parlay',
              date,
              tier: 'FORTRESS',
              n_legs: parlay.numLegs,
              parlay_decimal: parlay.decimalOdds,
              parlay_american: parlay.odds,
              hit: parlayHit,
              avgCascade: parlay.avgCascade,
              combinedHitRate: parlay.combinedHitRate,
              ev: parlay.ev,
              legs: parlay.legs.map(l => ({
                player: l.player,
                team: l.team,
                stat: STAT_MAP[l.statType] || l.statType,
                statLabel: l.statLabel,
                line: l.line,
                odds: l.odds,
                cascadeScore: l.cascadeScore,
                gft: l.gft,
                beq: l.beq,
                esi: l.esi,
                imad: l.imad,
                hitRate: l.hitRate,
                actual: l.actual,
                hit: l.hit,
                bet: `${l.player} O${l.line} ${l.statLabel}`,
              })),
              pnl: parlayHit
                ? Math.round((parlay.decimalOdds - 1) * CONFIG.UNIT_SIZE)
                : -CONFIG.UNIT_SIZE,
            });
          }
        }
      }

      // Walk-forward: update model AFTER generating signals (once per date)
      if (!updatedDates.has(game.date)) {
        updatedDates.add(game.date);
        const dateBoxes = boxByDate[game.date] || [];
        for (const bg of dateBoxes) {
          for (const p of (bg.players || [])) {
            const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
            if (mins < 5) continue;
            PlayerModel.update(p.name, {
              pts: p.pts,
              reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
              ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
              min: mins,
            }, game.date, p.team, bg.home === p.team ? bg.away : bg.home);
          }
        }
      }
    }

    // Compute stats
    const singles = allSignals.filter(s => s.betType === 'single');
    const parlays = allSignals.filter(s => s.betType === 'parlay');

    const singleWins = singles.filter(s => s.hit).length;
    const singlePnl = singles.reduce((s, p) => s + p.pnl, 0);

    const parlayWins = parlays.filter(s => s.hit).length;
    const parlayPnl = parlays.reduce((s, p) => s + p.pnl, 0);

    let totalLegs = 0, hitLegs = 0;
    parlays.forEach(p => {
      (p.legs || []).forEach(l => {
        totalLegs++;
        if (l.hit) hitLegs++;
      });
    });

    return {
      signals: allSignals,
      singles,
      parlays,
      stats: {
        singles: {
          total: singles.length,
          wins: singleWins,
          accuracy: singles.length > 0 ? singleWins / singles.length : 0,
          pnl: singlePnl,
          roi: singles.length > 0 ? singlePnl / (singles.length * CONFIG.UNIT_SIZE) : 0,
        },
        parlays: {
          total: parlays.length,
          wins: parlayWins,
          accuracy: parlays.length > 0 ? parlayWins / parlays.length : 0,
          pnl: parlayPnl,
          roi: parlays.length > 0 ? parlayPnl / (parlays.length * CONFIG.UNIT_SIZE) : 0,
          totalLegs,
          hitLegs,
          legAccuracy: totalLegs > 0 ? hitLegs / totalLegs : 0,
        },
        overall: {
          total: allSignals.length,
          wins: singleWins + parlayWins,
          accuracy: allSignals.length > 0 ? (singleWins + parlayWins) / allSignals.length : 0,
          pnl: singlePnl + parlayPnl,
          roi: allSignals.length > 0 ? (singlePnl + parlayPnl) / (allSignals.length * CONFIG.UNIT_SIZE) : 0,
        },
      },
    };
  }

  // =========================================================================
  // GENERATE TODAY'S RECOMMENDATIONS
  // =========================================================================

  function generateTodayPicks(liveOdds) {
    if (!liveOdds || !liveOdds.playerProps) return null;

    const candidates = [];

    for (const [gameKey, gameData] of Object.entries(liveOdds.playerProps)) {
      // Points
      if (gameData.lines) {
        for (const [playerName, lines] of Object.entries(gameData.lines)) {
          const prop = findBestProp(playerName, 'points', lines);
          if (prop) {
            prop.gameKey = gameKey;
            prop.gameDisplay = gameKey.replace('@', ' @ ');
            candidates.push(prop);
          }
        }
      }

      // Rebounds
      if (gameData.rebLines) {
        for (const [playerName, lines] of Object.entries(gameData.rebLines)) {
          const prop = findBestProp(playerName, 'rebounds', lines);
          if (prop) {
            prop.gameKey = gameKey;
            prop.gameDisplay = gameKey.replace('@', ' @ ');
            candidates.push(prop);
          }
        }
      }

      // Assists
      if (gameData.astLines) {
        for (const [playerName, lines] of Object.entries(gameData.astLines)) {
          const prop = findBestProp(playerName, 'assists', lines);
          if (prop) {
            prop.gameKey = gameKey;
            prop.gameDisplay = gameKey.replace('@', ' @ ');
            candidates.push(prop);
          }
        }
      }

      // PRA (Points + Rebounds + Assists) — from dedicated PRA market
      if (gameData.praLines) {
        for (const [playerName, lines] of Object.entries(gameData.praLines)) {
          const prop = findBestProp(playerName, 'pra', lines);
          if (prop) {
            prop.gameKey = gameKey;
            prop.gameDisplay = gameKey.replace('@', ' @ ');
            candidates.push(prop);
          }
        }
      }
    }

    // Deduplicate: keep best cascade score per player (matches Python backtest)
    candidates.sort((a, b) => b.cascadeScore - a.cascadeScore);
    const seen = new Set();
    const deduped = [];
    for (const c of candidates) {
      if (!seen.has(c.player)) {
        seen.add(c.player);
        deduped.push(c);
      }
    }

    console.log(`[ULTRA-JS] Evaluated ${Object.keys(liveOdds.playerProps).length} games, found ${deduped.length} candidates passing all 6 gates (${candidates.length} before dedup)`);

    return selectBetType(deduped);
  }

  // =========================================================================
  // SIGNAL FORMAT FOR PERSISTENCE
  // =========================================================================

  function formatSignalForStorage(recommendation, date) {
    const signals = [];

    for (const single of (recommendation.singles || [])) {
      signals.push({
        date,
        betType: 'single',
        tier: 'FORTRESS',
        engine: 'ultra',
        player: single.player,
        team: single.team,
        stat: STAT_MAP[single.statType] || single.statType,
        statLabel: single.statLabel,
        line: single.line,
        odds: single.odds,
        cascadeScore: single.cascadeScore,
        gft: single.gft,
        beq: single.beq,
        esi: single.esi,
        imad: single.imad,
        hitRate: single.hitRate,
        edge: single.edge,
        ev: single.ev,
        avg: single.avg,
        floor: single.floor,
        bet: `${single.player} O${single.line} ${single.statLabel}`,
        hit: null,
        actual: null,
      });
    }

    for (const parlay of (recommendation.parlays || [])) {
      signals.push({
        date,
        betType: 'parlay',
        tier: 'FORTRESS',
        engine: 'ultra',
        n_legs: parlay.numLegs,
        parlay_decimal: parlay.decimalOdds,
        parlay_american: parlay.odds,
        positive_odds: parlay.odds > 0,
        avgCascade: parlay.avgCascade,
        combinedHitRate: parlay.combinedHitRate,
        ev: parlay.ev,
        hit: null,
        legs: parlay.legs.map(l => ({
          player: l.player,
          team: l.team,
          stat: STAT_MAP[l.statType] || l.statType,
          statLabel: l.statLabel,
          line: l.line,
          odds: l.odds,
          cascadeScore: l.cascadeScore,
          gft: l.gft,
          beq: l.beq,
          esi: l.esi,
          imad: l.imad,
          hitRate: l.hitRate,
          edge: l.edge,
          actual: null,
          hit: null,
          bet: `${l.player} O${l.line} ${l.statLabel}`,
        })),
      });
    }

    return signals;
  }

  // =========================================================================
  // CONFIG LOADER — Load optimized config from file if available
  // =========================================================================

  function loadConfig(configObj) {
    if (!configObj || !configObj.config) return;
    const c = configObj.config;
    for (const key of Object.keys(c)) {
      if (CONFIG.hasOwnProperty(key)) {
        CONFIG[key] = c[key];
      }
    }
    console.log('[ULTRA-JS] Loaded optimized config:', configObj.score, 'score,', configObj.improvements, 'improvements');
  }

  // =========================================================================
  // PUBLIC API
  // =========================================================================

  return {
    PlayerModel,
    CONFIG,
    loadConfig,
    runBacktest,
    generateTodayPicks,
    formatSignalForStorage,
    evaluateProp,
    findBestProp,
    selectBetType,
    buildOptimalParlay,
    americanToDecimal,
    decimalToAmerican,
    formatOdds,
    STAT_LABELS,
    computeGFT,
    computeBEQ,
    computeESI,
    computeIMAD,
    computeUltraSignal,
  };
})();
