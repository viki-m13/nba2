/**
 * MLB Ultra Betting Engine v2.0 — Patent-Pending
 * =================================================
 * 10 Novel Innovations for MLB Batter Prop Betting:
 *
 * 1.  Gravitational Floor Theory (GFT) — Recency-weighted performance floors
 * 2.  Bayesian Edge Quantification (BEQ) — Beta posterior with credible intervals
 * 3.  Entropic Stability Index (ESI) — Shannon entropy consistency scoring
 * 4.  Inverse Market Asymmetry Detection (IMAD) — Multi-signal mispricing fusion
 * 5.  Consecutive Game Streak Momentum (CGSM) — Hit streak tracking
 * 6.  At-Bat Volume Confidence (ABVC) — Opportunity-weighted confidence
 * 7.  Poisson-Bayesian Fusion (PBF) — Discrete outcome probability modeling
 * 8.  Opponent-Adjusted Lambda (OAL) — Team pitching quality adjustment
 * 9.  Venue Performance Differential (VPD) — Home/away split analysis
 * 10. Edge-Maximized Parlay Construction (EMPC) — Correlation-verified parlays
 *
 * 9-Gate Quality Cascade — All gates must pass simultaneously
 * AutoResearch-Optimized parameters via monotonic ratchet methodology
 */

window.MLBRecommendationEngine = (function () {
  'use strict';

  // =========================================================================
  // CONFIGURATION — AutoResearch-Optimized for MLB v2.0
  // =========================================================================

  const CONFIG = {
    // Player eligibility
    MIN_GAMES: 8,
    MIN_AB: 2,
    WARM_UP_GAMES: 10,

    // Gravitational Floor Theory (GFT) — tuned for MLB stat scale (0-4 hits)
    GFT_WINDOWS: [3, 5, 10],
    GFT_DECAY_RATE: 0.90,
    GFT_GRAVITY_STRENGTH: 0.40,
    GFT_MIN_CLEARANCE: 0.05,
    GFT_CONVERGENCE_MAX_SPREAD: 3.0,

    // Bayesian Edge Quantification (BEQ) — calibrated like NBA (87% credible)
    BEQ_PRIOR_ALPHA: 1.0,
    BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.87,
    BEQ_MIN_EDGE: 0.03,

    // Entropic Stability Index (ESI)
    ESI_BINS: 5,
    ESI_MAX_ENTROPY: 0.85,
    ESI_TREND_WEIGHT: 0.25,

    // Inverse Market Asymmetry Detection (IMAD)
    IMAD_MIN_ASYMMETRY: 0.05,
    IMAD_VOLUME_DISCOUNT: 0.02,

    // Consecutive Game Streak Momentum (CGSM)
    CGSM_MIN_STREAK: 2,
    CGSM_LOOKBACK: 10,
    CGSM_WEIGHT: 0.08,

    // At-Bat Volume Confidence (ABVC) — fixed division-by-zero
    ABVC_MIN_AVG_AB: 3.0,
    ABVC_ELITE_AB: 5.0,
    ABVC_WEIGHT: 0.07,

    // Poisson-Bayesian Fusion (PBF) — NEW v2.0
    PBF_DECAY_RATE: 0.92,
    PBF_MIN_LAMBDA: 0.3,
    PBF_WEIGHT: 0.10,
    PBF_MIN_PROB_MARGIN: 0.05,

    // Opponent-Adjusted Lambda (OAL) — NEW v2.0
    OAL_ENABLED: true,
    OAL_WEIGHT: 0.05,
    OAL_MAX_ADJUSTMENT: 0.25,

    // Venue Performance Differential (VPD) — NEW v2.0
    VPD_ENABLED: true,
    VPD_MIN_GAMES: 5,
    VPD_WEIGHT: 0.05,
    VPD_MAX_BOOST: 0.10,

    // Quality Gates — 9-Gate Cascade
    GATE_MIN_GFT_SCORE: 0.40,
    GATE_MIN_BEQ_EDGE: 0.05,
    GATE_MIN_ESI_STABILITY: 0.15,
    GATE_MIN_IMAD_SCORE: 0.02,
    GATE_MIN_HIT_RATE: 0.80,
    GATE_MIN_COMBINED: 0.55,
    GATE_MIN_STREAK: 2,
    GATE_MIN_ABVC: 0.30,
    GATE_MIN_PBF_MARGIN: 0.03,

    // Bet Type Selection
    SINGLE_MIN_SCORE: 0.72,
    MULTI_SINGLE_MIN_SCORE: 0.65,
    PARLAY_LEG_MIN_SCORE: 0.58,

    // Parlay Construction
    PARLAY_MIN_LEGS: 2,
    PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.40,
    PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.08,

    // Odds Filters — widened to capture MLB prop range
    MIN_ODDS: -450,
    MAX_ODDS: -115,
    PREFERRED_ODDS_RANGE: [-500, -110],

    // Bankroll
    UNIT_SIZE: 100,
    MAX_DAILY_UNITS: 5,
    KELLY_FRACTION: 0.25,

    // Contextual adjustments
    HOME_BOOST: 0.015,
    BACK_TO_BACK_PENALTY: 0.00,
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

  const STAT_MAP = {
    hits: 'h', total_bases: 'tb', rbis: 'rbi', runs: 'r', home_runs: 'hr',
    h: 'h', tb: 'tb', rbi: 'rbi', r: 'r', hr: 'hr',
    hrtb: 'hrtb'
  };
  const STAT_LABELS = {
    hits: 'H', total_bases: 'TB', rbis: 'RBI', runs: 'R', home_runs: 'HR',
    h: 'H', tb: 'TB', rbi: 'RBI', r: 'R', hr: 'HR', hrtb: 'H+R+TB'
  };

  // =========================================================================
  // MATH UTILITIES
  // =========================================================================

  function betaPPF(alpha, betaParam, p) {
    if (alpha <= 0 || betaParam <= 0) return 0.5;
    if (p <= 0 || p >= 1) return alpha / (alpha + betaParam);

    const mean = alpha / (alpha + betaParam);
    const variance = (alpha * betaParam) / ((alpha + betaParam) ** 2 * (alpha + betaParam + 1));
    const std = Math.sqrt(variance > 0 ? variance : 0.001);

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

  /**
   * PATENT-PENDING: Poisson CDF computation for discrete outcome modeling
   * P(X <= k) where X ~ Poisson(lambda)
   */
  function poissonCDF(lambda, k) {
    if (lambda <= 0) return 1.0;
    let sum = 0;
    let term = Math.exp(-lambda);
    for (let i = 0; i <= k; i++) {
      sum += term;
      term *= lambda / (i + 1);
    }
    return Math.min(1.0, sum);
  }

  /**
   * P(X > line) where X ~ Poisson(lambda)
   * For fractional lines like 0.5, we need P(X >= 1) = 1 - P(X = 0)
   */
  function poissonOverProb(lambda, line) {
    const k = Math.floor(line);
    return 1 - poissonCDF(lambda, k);
  }

  // =========================================================================
  // PLAYER MODEL — Enhanced with venue/opponent tracking
  // =========================================================================

  const PlayerModel = {
    profiles: {},

    reset() { this.profiles = {}; },

    update(name, stats, date, team, opponent, isHome) {
      if (!this.profiles[name]) this.profiles[name] = { games: [], team: '' };
      const enriched = { ...stats, date, team, opponent, isHome: !!isHome };
      if (enriched.hrtb === undefined) {
        enriched.hrtb = (enriched.h || 0) + (enriched.r || 0) + (enriched.tb || 0);
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

    getValues(name, statKey, window, minAB) {
      const profile = this.profiles[name];
      if (!profile) return null;

      let games = profile.games;
      if (minAB && minAB > 0) {
        games = games.filter(g => (g.ab || 0) >= minAB);
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

    getRecentAB(name, window) {
      const vals = this.getValues(name, 'ab', window || 5);
      if (!vals || !vals.length) return 0;
      return vals.reduce((s, v) => s + v, 0) / vals.length;
    },

    getStreak(name, statKey, line, minAB) {
      const profile = this.profiles[name];
      if (!profile) return 0;

      let games = profile.games;
      if (minAB && minAB > 0) {
        games = games.filter(g => (g.ab || 0) >= minAB);
      }

      let streak = 0;
      for (let i = games.length - 1; i >= 0; i--) {
        if ((games[i][statKey] || 0) > line) {
          streak++;
        } else {
          break;
        }
      }
      return streak;
    },

    getStreakInWindow(name, statKey, line, lookback, minAB) {
      const profile = this.profiles[name];
      if (!profile) return { streak: 0, hitsInWindow: 0, windowSize: 0 };

      let games = profile.games;
      if (minAB && minAB > 0) {
        games = games.filter(g => (g.ab || 0) >= minAB);
      }

      const window = games.slice(-lookback);
      let streak = 0;
      let hitsInWindow = 0;

      for (let i = window.length - 1; i >= 0; i--) {
        if ((window[i][statKey] || 0) > line) {
          hitsInWindow++;
          if (streak === window.length - 1 - i) streak++;
        }
      }

      return { streak, hitsInWindow, windowSize: window.length };
    },

    getAvgAB(name, window) {
      const profile = this.profiles[name];
      if (!profile) return 0;
      const games = profile.games.slice(-(window || 10));
      if (games.length === 0) return 0;
      return games.reduce((s, g) => s + (g.ab || 0), 0) / games.length;
    },

    /**
     * PATENT-PENDING: Venue Performance Differential (VPD)
     * Returns { homeValues, awayValues } for a stat
     */
    getVenueSplit(name, statKey, minAB) {
      const profile = this.profiles[name];
      if (!profile) return null;

      let games = profile.games;
      if (minAB && minAB > 0) {
        games = games.filter(g => (g.ab || 0) >= minAB);
      }

      const homeGames = games.filter(g => g.isHome);
      const awayGames = games.filter(g => !g.isHome);

      return {
        homeValues: homeGames.map(g => g[statKey] || 0),
        awayValues: awayGames.map(g => g[statKey] || 0),
        homeCount: homeGames.length,
        awayCount: awayGames.length,
      };
    },

    /**
     * Get games filtered by opponent for OAL computation
     */
    getGamesVsTeam(name, opponent, statKey) {
      const profile = this.profiles[name];
      if (!profile) return [];
      return profile.games
        .filter(g => g.opponent === opponent)
        .map(g => g[statKey] || 0);
    },
  };

  // =========================================================================
  // OPPONENT QUALITY TRACKER — For OAL computation
  // =========================================================================

  const OpponentTracker = {
    teamStats: {},

    reset() { this.teamStats = {}; },

    update(team, statsAllowed) {
      if (!this.teamStats[team]) this.teamStats[team] = { gamesAsDefense: [] };
      this.teamStats[team].gamesAsDefense.push(statsAllowed);
      if (this.teamStats[team].gamesAsDefense.length > 30) {
        this.teamStats[team].gamesAsDefense =
          this.teamStats[team].gamesAsDefense.slice(-30);
      }
    },

    /**
     * PATENT-PENDING: Opponent-Adjusted Lambda (OAL)
     * Returns a multiplier for how much easier/harder this opponent is to hit against
     */
    getOpponentFactor(team, statKey) {
      if (!this.teamStats[team]) return 1.0;
      const games = this.teamStats[team].gamesAsDefense;
      if (games.length < 5) return 1.0;

      // Compute average stat allowed per game by this team's defense
      const avgAllowed = games.reduce((sum, g) => sum + (g[statKey] || 0), 0) / games.length;

      // League average baseline (approximate — will be refined by optimizer)
      const leagueAvg = this._getLeagueAvg(statKey);
      if (leagueAvg <= 0) return 1.0;

      // Factor: >1.0 means team allows more than average (easier opponent)
      const raw = avgAllowed / leagueAvg;
      // Clamp adjustment to prevent extreme swings
      return Math.max(1 - CONFIG.OAL_MAX_ADJUSTMENT, Math.min(1 + CONFIG.OAL_MAX_ADJUSTMENT, raw));
    },

    _getLeagueAvg(statKey) {
      const teams = Object.values(this.teamStats);
      if (teams.length < 5) return 1.0;

      let totalSum = 0, totalGames = 0;
      for (const t of teams) {
        for (const g of t.gamesAsDefense) {
          totalSum += g[statKey] || 0;
          totalGames++;
        }
      }
      return totalGames > 0 ? totalSum / totalGames : 1.0;
    },
  };

  // =========================================================================
  // GRAVITATIONAL FLOOR THEORY (GFT) — Calibrated for MLB stat scale
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
      const values = PlayerModel.getValues(name, statKey, w, CONFIG.MIN_AB);
      if (!values || values.length < Math.min(w, 3)) return null;

      const n = values.length;
      const weights = [];
      for (let i = 0; i < n; i++) {
        weights.push(Math.pow(decay, n - 1 - i));
      }
      const totalWeight = weights.reduce((s, w) => s + w, 0);

      let weightedMean = 0;
      for (let i = 0; i < n; i++) {
        weightedMean += values[i] * weights[i];
      }
      weightedMean /= totalWeight;

      const sortedWeighted = values.map((v, i) => ({ val: v, wt: weights[i] }))
        .sort((a, b) => a.val - b.val);

      let cumWeight = 0;
      let p10Value = sortedWeighted[0].val;
      for (const { val, wt } of sortedWeighted) {
        cumWeight += wt;
        if (cumWeight / totalWeight >= 0.10) {
          p10Value = val;
          break;
        }
      }

      const rawFloor = p10Value;
      const gravFloor = rawFloor + gravity * (weightedMean - rawFloor);

      const clearance = gravFloor - line;
      if (clearance < minClearance) return null;

      floors.push(gravFloor);
      clearances.push(clearance);
    }

    const floorSpread = Math.max(...floors) - Math.min(...floors);
    if (floorSpread > maxSpread) return null;

    const convergence = Math.max(0, 1 - floorSpread / maxSpread);
    const avgClearance = clearances.reduce((s, c) => s + c, 0) / clearances.length;
    const depth = Math.min(1.0, avgClearance / 4.0);

    const score = convergence * 0.5 + depth * 0.5;

    return {
      score, convergence, depth, floors, clearances, floorSpread, avgClearance,
    };
  }

  // =========================================================================
  // BAYESIAN EDGE QUANTIFICATION (BEQ) — Calibrated like NBA success
  // =========================================================================

  function computeBEQ(name, statKey, line, marketOdds) {
    const values = PlayerModel.getValues(name, statKey, 20, CONFIG.MIN_AB);
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

    const valuesExtended = PlayerModel.getValues(name, statKey, 30, CONFIG.MIN_AB);
    let extendedHitRate = null;
    if (valuesExtended && valuesExtended.length >= 20) {
      extendedHitRate = valuesExtended.filter(v => v > line).length / valuesExtended.length;
    }

    return {
      meanProb, lowerBound, marketProb: mktProb, edge,
      alpha, beta: betaParam, hits, total: values.length,
      hitRate: hits / values.length, extendedHitRate,
    };
  }

  // =========================================================================
  // ENTROPIC STABILITY INDEX (ESI)
  // =========================================================================

  function computeESI(name, statKey) {
    const values = PlayerModel.getValues(name, statKey, 20, CONFIG.MIN_AB);
    if (!values || values.length < 10) return null;

    const nBins = CONFIG.ESI_BINS;
    const trendWeight = CONFIG.ESI_TREND_WEIGHT;

    const distEntropy = shannonEntropy(values, nBins);

    const mid = Math.floor(values.length / 2);
    const firstHalf = values.slice(0, mid);
    const secondHalf = values.slice(mid);

    const meanFirst = firstHalf.reduce((s, v) => s + v, 0) / firstHalf.length;
    const meanSecond = secondHalf.reduce((s, v) => s + v, 0) / secondHalf.length;
    const overallMean = values.reduce((s, v) => s + v, 0) / values.length;

    if (overallMean === 0) return null;

    const trendShift = Math.abs(meanSecond - meanFirst) / overallMean;
    const trendStability = Math.max(0, 1 - trendShift / 0.3);

    const sortedVals = [...values].sort((a, b) => a - b);
    const p15 = sortedVals[Math.max(0, Math.floor(sortedVals.length * 0.15))];
    const recent = values.slice(-5);
    const tailEvents = recent.filter(v => v <= p15).length;
    const tailRisk = 1 - tailEvents / recent.length;

    const entropyScore = Math.max(0, 1 - distEntropy / CONFIG.ESI_MAX_ENTROPY);
    let stability = (1 - trendWeight) * entropyScore + trendWeight * trendStability;
    stability = stability * tailRisk;

    return {
      stability, entropy: distEntropy, trendStability, tailRisk,
      mean: overallMean, trendShift,
    };
  }

  // =========================================================================
  // INVERSE MARKET ASYMMETRY DETECTION (IMAD)
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
      score, totalAsymmetry, bayesianEdge, stabilityPremium,
      depthPremium, strongSignals, agreementFactor,
    };
  }

  // =========================================================================
  // CONSECUTIVE GAME STREAK MOMENTUM (CGSM)
  // Patent-Pending — Tracks consecutive game hit streaks
  // =========================================================================

  function computeCGSM(name, statKey, line) {
    const streakInfo = PlayerModel.getStreakInWindow(
      name, statKey, line, CONFIG.CGSM_LOOKBACK, CONFIG.MIN_AB
    );

    if (streakInfo.windowSize < 5) return null;

    const streak = streakInfo.streak;
    const windowHitRate = streakInfo.hitsInWindow / streakInfo.windowSize;

    const streakScore = Math.min(1.0, streak / CONFIG.CGSM_LOOKBACK);
    const consistencyScore = windowHitRate;

    const score = streakScore * 0.6 + consistencyScore * 0.4;

    return {
      score,
      streak,
      windowHitRate,
      hitsInWindow: streakInfo.hitsInWindow,
      windowSize: streakInfo.windowSize,
    };
  }

  // =========================================================================
  // AT-BAT VOLUME CONFIDENCE (ABVC) — Fixed division-by-zero bug
  // Patent-Pending — Weights confidence by at-bat opportunity volume
  // =========================================================================

  function computeABVC(name) {
    const avgAB = PlayerModel.getAvgAB(name, 10);
    if (avgAB < CONFIG.ABVC_MIN_AVG_AB) return null;

    // Fixed: prevent division by zero when MIN == ELITE
    const abRange = CONFIG.ABVC_ELITE_AB - CONFIG.ABVC_MIN_AVG_AB;
    const volumeScore = abRange > 0
      ? Math.min(1.0, (avgAB - CONFIG.ABVC_MIN_AVG_AB) / abRange)
      : (avgAB >= CONFIG.ABVC_ELITE_AB ? 1.0 : 0.5);

    const abValues = PlayerModel.getValues(name, 'ab', 10, 1);
    if (!abValues || abValues.length < 5) return null;

    const meanAB = abValues.reduce((s, v) => s + v, 0) / abValues.length;
    const variance = abValues.reduce((s, v) => s + (v - meanAB) ** 2, 0) / abValues.length;
    const cv = Math.sqrt(variance) / (meanAB || 1);
    const consistencyScore = Math.max(0, 1 - cv);

    const score = volumeScore * 0.6 + consistencyScore * 0.4;

    return {
      score,
      avgAB,
      volumeScore,
      consistencyScore,
      cv,
    };
  }

  // =========================================================================
  // PATENT-PENDING: POISSON-BAYESIAN FUSION (PBF)
  // Models discrete baseball outcomes as Poisson process with Bayesian
  // recency-weighted lambda estimation. Fundamentally different from
  // continuous-stat NBA approach — accounts for the discrete nature of
  // hits (0,1,2,3,4) rather than treating them as continuous variables.
  //
  // Key innovation: Fuses Poisson probability P(X > line | lambda) with
  // Bayesian credible interval to produce calibrated discrete probability
  // that accounts for both the generating process and parameter uncertainty.
  // =========================================================================

  function computePBF(name, statKey, line, marketOdds) {
    const values = PlayerModel.getValues(name, statKey, 20, CONFIG.MIN_AB);
    if (!values || values.length < 10) return null;

    // Compute recency-weighted Poisson lambda
    const decay = CONFIG.PBF_DECAY_RATE;
    const n = values.length;
    let weightedSum = 0;
    let totalWeight = 0;
    for (let i = 0; i < n; i++) {
      const w = Math.pow(decay, n - 1 - i);
      weightedSum += values[i] * w;
      totalWeight += w;
    }
    const lambda = weightedSum / totalWeight;

    if (lambda < CONFIG.PBF_MIN_LAMBDA) return null;

    // Poisson probability of exceeding line
    const poissonProb = poissonOverProb(lambda, line);

    // Market implied probability
    const mktProb = impliedProbability(marketOdds);

    // Margin of safety: how much Poisson prob exceeds market
    const margin = poissonProb - mktProb;

    // Confidence based on sample size and lambda stability
    const extValues = PlayerModel.getValues(name, statKey, 30, CONFIG.MIN_AB);
    let lambdaStability = 1.0;
    if (extValues && extValues.length >= 15) {
      const extLambda = extValues.reduce((s, v) => s + v, 0) / extValues.length;
      const lambdaDrift = Math.abs(lambda - extLambda) / (extLambda || 1);
      lambdaStability = Math.max(0, 1 - lambdaDrift / 0.3);
    }

    const score = Math.min(1.0, (margin / 0.20 + 0.5)) * lambdaStability;

    return {
      score,
      lambda,
      poissonProb,
      marketProb: mktProb,
      margin,
      lambdaStability,
    };
  }

  // =========================================================================
  // PATENT-PENDING: OPPONENT-ADJUSTED LAMBDA (OAL)
  // Adjusts player's expected production rate based on opposing team's
  // defensive quality. Uses rolling team-level stats to identify
  // plus/minus matchups that the market may not fully price in.
  //
  // Key innovation: The adjustment is asymmetric — it boosts confidence
  // against weak pitching but doesn't penalize against average pitching,
  // only against elite pitching. This creates a directional edge.
  // =========================================================================

  function computeOAL(name, statKey, opponent) {
    if (!CONFIG.OAL_ENABLED) return null;

    const factor = OpponentTracker.getOpponentFactor(opponent, statKey);
    if (factor === 1.0) return null;

    // Asymmetric: boost for easy opponents, slight penalty for tough ones
    let adjustment = 0;
    if (factor > 1.05) {
      // Opponent allows more than average — favorable matchup
      adjustment = Math.min(CONFIG.OAL_MAX_ADJUSTMENT, (factor - 1.0) * 0.5);
    } else if (factor < 0.90) {
      // Tough opponent — slight confidence reduction
      adjustment = Math.max(-CONFIG.OAL_MAX_ADJUSTMENT * 0.5, (factor - 1.0) * 0.3);
    }

    const score = 0.5 + adjustment;

    return {
      score: Math.max(0, Math.min(1.0, score)),
      factor,
      adjustment,
      opponent,
    };
  }

  // =========================================================================
  // PATENT-PENDING: VENUE PERFORMANCE DIFFERENTIAL (VPD)
  // Analyzes player's home vs away performance splits to identify
  // venue-dependent edge. Baseball has significant park factors
  // (Coors Field, Yankee Stadium, etc.) that affect hitting stats.
  //
  // Key innovation: Only applies boost when the venue direction
  // (home/away) aligns with the player's historical strength,
  // creating a one-directional confidence enhancer.
  // =========================================================================

  function computeVPD(name, statKey, isHome) {
    if (!CONFIG.VPD_ENABLED) return null;

    const split = PlayerModel.getVenueSplit(name, statKey, CONFIG.MIN_AB);
    if (!split) return null;
    if (split.homeCount < CONFIG.VPD_MIN_GAMES || split.awayCount < CONFIG.VPD_MIN_GAMES) {
      return null;
    }

    const homeMean = split.homeValues.reduce((s, v) => s + v, 0) / split.homeCount;
    const awayMean = split.awayValues.reduce((s, v) => s + v, 0) / split.awayCount;
    const overallMean = (homeMean * split.homeCount + awayMean * split.awayCount) /
      (split.homeCount + split.awayCount);

    if (overallMean === 0) return null;

    // Compute venue differential
    const currentVenueMean = isHome ? homeMean : awayMean;
    const venueDiff = (currentVenueMean - overallMean) / overallMean;

    // Only apply boost when player performs BETTER at current venue
    let boost = 0;
    if (venueDiff > 0.05) {
      boost = Math.min(CONFIG.VPD_MAX_BOOST, venueDiff * 0.5);
    }

    const score = 0.5 + boost;

    return {
      score: Math.max(0, Math.min(1.0, score)),
      homeMean,
      awayMean,
      venueDiff,
      boost,
      isHome,
    };
  }

  // =========================================================================
  // ULTRA SIGNAL — 9-Gate Quality Cascade (v2.0 Enhanced)
  // =========================================================================

  function computeUltraSignal(playerName, statKey, line, marketOdds, opponent, isHome) {
    // Gate 0: Player eligibility
    if (!PlayerModel.isWarm(playerName, CONFIG.MIN_GAMES)) return null;
    const recentAB = PlayerModel.getRecentAB(playerName, 5);
    if (recentAB < CONFIG.MIN_AB) return null;

    // Gate 1: Gravitational Floor Theory
    const gft = computeGFT(playerName, statKey, line);
    if (!gft || gft.score < CONFIG.GATE_MIN_GFT_SCORE) return null;

    // Gate 2: Bayesian Edge Quantification
    const beq = computeBEQ(playerName, statKey, line, marketOdds);
    if (!beq || beq.edge < CONFIG.GATE_MIN_BEQ_EDGE) return null;

    // Gate 3: Entropic Stability Index
    const esi = computeESI(playerName, statKey);
    if (!esi || esi.stability < CONFIG.GATE_MIN_ESI_STABILITY) return null;

    // Gate 4: Inverse Market Asymmetry
    const imad = computeIMAD(beq, esi, gft);
    if (!imad || imad.score < CONFIG.GATE_MIN_IMAD_SCORE) return null;

    // Gate 5: Raw Hit Rate Sanity Check
    if (beq.hitRate < CONFIG.GATE_MIN_HIT_RATE) return null;

    // Gate 5b: Extended Window Anti-Overfitting
    if (beq.extendedHitRate !== null) {
      if (beq.extendedHitRate < beq.hitRate - 0.10) return null;
      if (beq.extendedHitRate < CONFIG.GATE_MIN_HIT_RATE - 0.05) return null;
    }

    // Gate 6: Consecutive Game Streak Momentum
    const cgsm = computeCGSM(playerName, statKey, line);
    if (!cgsm || cgsm.streak < CONFIG.GATE_MIN_STREAK) return null;

    // Gate 7: At-Bat Volume Confidence
    const abvc = computeABVC(playerName);
    if (!abvc || abvc.score < CONFIG.GATE_MIN_ABVC) return null;

    // Gate 8: Poisson-Bayesian Fusion (NEW v2.0)
    const pbf = computePBF(playerName, statKey, line, marketOdds);
    if (!pbf || pbf.margin < CONFIG.GATE_MIN_PBF_MARGIN) return null;

    // Compute optional enhancers (don't gate on these — they boost score)
    const oal = computeOAL(playerName, statKey, opponent);
    const vpd = computeVPD(playerName, statKey, isHome);

    // Combined score: geometric mean of core 4 + weighted add of MLB-specific signals
    const coreComponents = [
      gft.score,
      Math.min(1.0, beq.edge / 0.20 + 0.5),
      esi.stability,
      Math.min(1.0, imad.score / 0.15 + 0.5),
    ];

    const logSum = coreComponents.reduce((s, c) => s + Math.log(Math.max(0.001, c)), 0);
    let combined = Math.exp(logSum / coreComponents.length);

    // Weight core signals vs MLB-specific innovations
    const coreWeight = 1.0 - CONFIG.CGSM_WEIGHT - CONFIG.ABVC_WEIGHT - CONFIG.PBF_WEIGHT;
    combined = combined * coreWeight
             + cgsm.score * CONFIG.CGSM_WEIGHT
             + abvc.score * CONFIG.ABVC_WEIGHT
             + pbf.score * CONFIG.PBF_WEIGHT;

    // Apply optional boosts from OAL and VPD
    if (oal && oal.adjustment > 0) {
      combined += oal.adjustment * CONFIG.OAL_WEIGHT;
    }
    if (vpd && vpd.boost > 0) {
      combined += vpd.boost * CONFIG.VPD_WEIGHT;
    }

    // Contextual: home boost
    if (isHome) {
      combined += CONFIG.HOME_BOOST;
    }

    combined = Math.max(0, Math.min(1.0, combined));

    if (combined < CONFIG.GATE_MIN_COMBINED) return null;

    const decimal = americanToDecimal(marketOdds);
    const ev = beq.lowerBound * decimal - 1;

    const kellyFull = (beq.lowerBound * decimal - 1) / (decimal - 1);
    const kelly = Math.max(0, kellyFull * CONFIG.KELLY_FRACTION);

    return {
      player: playerName, stat: statKey, line, odds: marketOdds,
      combinedScore: combined, gft, beq, esi, imad, cgsm, abvc, pbf, oal, vpd,
      ev, kelly, hitRate: beq.hitRate, bayesianProb: beq.meanProb,
      lowerBoundProb: beq.lowerBound, marketImplied: beq.marketProb, edge: beq.edge,
      poissonProb: pbf.poissonProb, poissonLambda: pbf.lambda,
    };
  }

  // =========================================================================
  // EVALUATE A PROP
  // =========================================================================

  function evaluateProp(playerName, statType, line, odds, opponent, isHome) {
    const statKey = STAT_MAP[statType] || statType;

    if (odds < CONFIG.MIN_ODDS || odds > CONFIG.MAX_ODDS) return null;

    const signal = computeUltraSignal(playerName, statKey, line, odds, opponent, isHome);
    if (!signal) return null;

    const profile = PlayerModel.getProfile(playerName);
    const decimal = americanToDecimal(odds);

    return {
      player: playerName,
      team: profile ? profile.team : '',
      statType,
      statLabel: STAT_LABELS[statType] || statType.toUpperCase(),
      line, odds, decimalOdds: decimal,
      engine: 'ultra',
      gft: signal.gft.score,
      beq: signal.beq.edge,
      esi: signal.esi.stability,
      imad: signal.imad.score,
      cgsm: signal.cgsm.score,
      abvc: signal.abvc.score,
      pbf: signal.pbf.score,
      pbfLambda: signal.pbf.lambda,
      pbfProb: signal.pbf.poissonProb,
      oal: signal.oal ? signal.oal.score : null,
      vpd: signal.vpd ? signal.vpd.score : null,
      cascadeScore: signal.combinedScore,
      impliedProb: signal.marketImplied,
      hitRate: signal.hitRate,
      edge: signal.edge,
      ev: signal.ev,
      avg: signal.esi.mean,
      floor: signal.gft.floors[signal.gft.floors.length - 1],
      streak: signal.cgsm.streak,
      avgAB: signal.abvc.avgAB,
    };
  }

  // =========================================================================
  // FIND BEST PROP — Evaluates all available lines for a player/stat
  // =========================================================================

  function findBestProp(playerName, statType, fdLines, opponent, isHome) {
    if (!fdLines || typeof fdLines !== 'object') return null;

    let best = null;
    for (const [threshold, data] of Object.entries(fdLines)) {
      const line = parseFloat(threshold);
      const odds = data.overOdds;
      if (!odds) continue;

      const result = evaluateProp(playerName, statType, line, odds, opponent, isHome);
      if (!result) continue;

      if (!best || result.cascadeScore > best.cascadeScore) {
        best = result;
      }
    }
    return best;
  }

  // =========================================================================
  // BET TYPE SELECTOR — Same structure as NBA model
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

    if (elite.length >= 1) {
      recommendations.singles = elite.slice(0, 3);
      recommendations.betType = elite.length === 1 ? 'single' : 'multi_single';
      recommendations.reasoning = `${elite.length} elite-confidence pick(s) found (score >= ${CONFIG.SINGLE_MIN_SCORE})`;
    }

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
  // PARLAY CONSTRUCTION — EMPC (Edge-Maximized Parlay Construction)
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
    const sorted = [...candidates].sort((a, b) => b.edge - a.edge);

    const selected = [];
    const usedTeams = new Set();
    const usedPlayers = new Set();

    for (const leg of sorted) {
      if (selected.length >= CONFIG.PARLAY_MAX_LEGS) break;
      if (usedPlayers.has(leg.player)) continue;

      const team = leg.team;
      if (!CONFIG.PARLAY_SAME_GAME_ALLOWED && usedTeams.has(team)) continue;

      let maxCorr = 0;
      for (const existing of selected) {
        const statKey1 = STAT_MAP[existing.statType] || existing.statType;
        const statKey2 = STAT_MAP[leg.statType] || leg.statType;
        const v1 = PlayerModel.getValues(existing.player, statKey1, 20, CONFIG.MIN_AB);
        const v2 = PlayerModel.getValues(leg.player, statKey2, 20, CONFIG.MIN_AB);
        const corr = computeCorrelation(v1, v2);
        maxCorr = Math.max(maxCorr, corr);
      }

      if (maxCorr > CONFIG.PARLAY_MAX_CORRELATION) continue;

      selected.push(leg);
      usedPlayers.add(leg.player);
      if (team) usedTeams.add(team);
    }

    if (selected.length < CONFIG.PARLAY_MIN_LEGS) return null;

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
  // WALK-FORWARD BACKTEST WITH REAL ODDS — Enhanced with OAL/VPD tracking
  // =========================================================================

  function runBacktest(boxScores, historicalOdds) {
    PlayerModel.reset();
    OpponentTracker.reset();

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
          if (!og) continue;

          for (const player of (bg.players || [])) {
            const ab = player.ab || 0;
            if (ab < CONFIG.MIN_AB) continue;

            const isHome = player.team === bg.home;
            const opponent = isHome ? bg.away : bg.home;

            // Check hits props
            if (og.hitsProps && og.hitsProps[player.name]) {
              const prop = findBestProp(player.name, 'hits', og.hitsProps[player.name], opponent, isHome);
              if (prop) {
                prop.actual = player.h;
                prop.hit = player.h > prop.line;
                prop.gameKey = gameKey;
                prop.date = date;
                candidates.push(prop);
              }
            }

            // Check total bases props
            if (og.tbProps && og.tbProps[player.name]) {
              const prop = findBestProp(player.name, 'total_bases', og.tbProps[player.name], opponent, isHome);
              if (prop) {
                prop.actual = player.tb;
                prop.hit = player.tb > prop.line;
                prop.gameKey = gameKey;
                prop.date = date;
                candidates.push(prop);
              }
            }

            // Check RBI props
            if (og.rbiProps && og.rbiProps[player.name]) {
              const prop = findBestProp(player.name, 'rbis', og.rbiProps[player.name], opponent, isHome);
              if (prop) {
                prop.actual = player.rbi;
                prop.hit = player.rbi > prop.line;
                prop.gameKey = gameKey;
                prop.date = date;
                candidates.push(prop);
              }
            }

            // Check runs props
            if (og.runsProps && og.runsProps[player.name]) {
              const prop = findBestProp(player.name, 'runs', og.runsProps[player.name], opponent, isHome);
              if (prop) {
                prop.actual = player.r;
                prop.hit = player.r > prop.line;
                prop.gameKey = gameKey;
                prop.date = date;
                candidates.push(prop);
              }
            }
          }
        }

        if (candidates.length > 0) {
          const recommendation = selectBetType(candidates);

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

      // Walk-forward: update models AFTER generating signals
      const dateBoxes = boxByDate[game.date] || [];
      for (const bg of dateBoxes) {
        // Track opponent quality stats for OAL
        const homePlayerStats = { h: 0, tb: 0, rbi: 0, r: 0, hr: 0, count: 0 };
        const awayPlayerStats = { h: 0, tb: 0, rbi: 0, r: 0, hr: 0, count: 0 };

        for (const p of (bg.players || [])) {
          const ab = p.ab || 0;
          if (ab < 1) continue;

          const isHome = p.team === bg.home;
          const opponent = isHome ? bg.away : bg.home;

          PlayerModel.update(p.name, {
            h: p.h || 0,
            tb: p.tb || 0,
            rbi: p.rbi || 0,
            r: p.r || 0,
            hr: p.hr || 0,
            ab: ab,
            bb: p.bb || 0,
            so: p.so || 0,
            sb: p.sb || 0,
          }, game.date, p.team, opponent, isHome);

          // Accumulate team hitting stats for opponent tracking
          if (isHome) {
            homePlayerStats.h += p.h || 0;
            homePlayerStats.tb += p.tb || 0;
            homePlayerStats.rbi += p.rbi || 0;
            homePlayerStats.r += p.r || 0;
            homePlayerStats.count++;
          } else {
            awayPlayerStats.h += p.h || 0;
            awayPlayerStats.tb += p.tb || 0;
            awayPlayerStats.rbi += p.rbi || 0;
            awayPlayerStats.r += p.r || 0;
            awayPlayerStats.count++;
          }
        }

        // Update opponent tracker: home team's pitching allowed away team's stats
        if (awayPlayerStats.count > 0) {
          OpponentTracker.update(bg.home, {
            h: awayPlayerStats.h / awayPlayerStats.count,
            tb: awayPlayerStats.tb / awayPlayerStats.count,
            rbi: awayPlayerStats.rbi / awayPlayerStats.count,
            r: awayPlayerStats.r / awayPlayerStats.count,
          });
        }
        if (homePlayerStats.count > 0) {
          OpponentTracker.update(bg.away, {
            h: homePlayerStats.h / homePlayerStats.count,
            tb: homePlayerStats.tb / homePlayerStats.count,
            rbi: homePlayerStats.rbi / homePlayerStats.count,
            r: homePlayerStats.r / homePlayerStats.count,
          });
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
      const [away, home] = gameKey.split('@');

      // Hits
      if (gameData.hitsLines) {
        for (const [playerName, lines] of Object.entries(gameData.hitsLines)) {
          const playerTeam = PlayerModel.getTeam(playerName);
          const isHome = playerTeam === home;
          const opponent = isHome ? away : home;
          const prop = findBestProp(playerName, 'hits', lines, opponent, isHome);
          if (prop) {
            prop.gameKey = gameKey;
            prop.gameDisplay = gameKey.replace('@', ' @ ');
            candidates.push(prop);
          }
        }
      }

      // Total Bases
      if (gameData.tbLines) {
        for (const [playerName, lines] of Object.entries(gameData.tbLines)) {
          const playerTeam = PlayerModel.getTeam(playerName);
          const isHome = playerTeam === home;
          const opponent = isHome ? away : home;
          const prop = findBestProp(playerName, 'total_bases', lines, opponent, isHome);
          if (prop) {
            prop.gameKey = gameKey;
            prop.gameDisplay = gameKey.replace('@', ' @ ');
            candidates.push(prop);
          }
        }
      }

      // RBIs
      if (gameData.rbiLines) {
        for (const [playerName, lines] of Object.entries(gameData.rbiLines)) {
          const playerTeam = PlayerModel.getTeam(playerName);
          const isHome = playerTeam === home;
          const opponent = isHome ? away : home;
          const prop = findBestProp(playerName, 'rbis', lines, opponent, isHome);
          if (prop) {
            prop.gameKey = gameKey;
            prop.gameDisplay = gameKey.replace('@', ' @ ');
            candidates.push(prop);
          }
        }
      }

      // Runs
      if (gameData.runsLines) {
        for (const [playerName, lines] of Object.entries(gameData.runsLines)) {
          const playerTeam = PlayerModel.getTeam(playerName);
          const isHome = playerTeam === home;
          const opponent = isHome ? away : home;
          const prop = findBestProp(playerName, 'runs', lines, opponent, isHome);
          if (prop) {
            prop.gameKey = gameKey;
            prop.gameDisplay = gameKey.replace('@', ' @ ');
            candidates.push(prop);
          }
        }
      }
    }

    console.log(`[ULTRA-MLB v2] Evaluated ${Object.keys(liveOdds.playerProps).length} games, found ${candidates.length} candidates passing all 9 gates`);

    return selectBetType(candidates);
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
        cgsm: single.cgsm,
        abvc: single.abvc,
        pbf: single.pbf,
        oal: single.oal,
        vpd: single.vpd,
        hitRate: single.hitRate,
        edge: single.edge,
        ev: single.ev,
        avg: single.avg,
        floor: single.floor,
        streak: single.streak,
        avgAB: single.avgAB,
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
  // CONFIG LOADER
  // =========================================================================

  function loadConfig(configObj) {
    if (!configObj || !configObj.config) return;
    const c = configObj.config;
    for (const key of Object.keys(c)) {
      if (CONFIG.hasOwnProperty(key)) {
        CONFIG[key] = c[key];
      }
    }
    console.log('[ULTRA-MLB v2] Loaded optimized config:', configObj.score, 'score,', configObj.improvements, 'improvements');
  }

  // =========================================================================
  // PUBLIC API
  // =========================================================================

  return {
    PlayerModel,
    OpponentTracker,
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
    computeCGSM,
    computeABVC,
    computePBF,
    computeOAL,
    computeVPD,
    computeUltraSignal,
  };
})();
