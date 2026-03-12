/**
 * MLB Ultra Betting Engine v3.0 — Patent-Pending
 * =================================================
 * 16 Novel Innovations for MLB Batter Prop Betting:
 *
 * CORE (from v2.0):
 * 1.  Gravitational Floor Theory (GFT) — Recency-weighted performance floors
 * 2.  Bayesian Edge Quantification (BEQ) — Beta posterior with credible intervals
 * 3.  Entropic Stability Index (ESI) — Shannon entropy consistency scoring
 * 4.  Inverse Market Asymmetry Detection (IMAD) — Multi-signal mispricing fusion
 * 5.  Consecutive Game Streak Momentum (CGSM) — Hit streak tracking
 * 6.  At-Bat Volume Confidence (ABVC) — Opportunity-weighted confidence
 * 7.  Poisson-Bayesian Fusion (PBF) — Discrete outcome probability modeling
 * 8.  Opponent-Adjusted Lambda (OAL) — Team pitching quality adjustment
 * 9.  Venue Performance Differential (VPD) — Home/away split analysis
 *
 * NEW v3.0 INNOVATIONS:
 * 10. Sliding-Window Temporal Attention (SWTA) — Multi-resolution temporal analysis
 * 11. Adversarial Signal Consensus (ASC) — Three-agent adversarial validation
 * 12. Discrete Markov Chain Transition (DMCT) — Sequential outcome modeling
 * 13. Temporal Regime Detection (TRD) — Hot/cold streak regime identification
 * 14. Correlation-Aware Multi-Prop Diversification (CAMPD) — Cross-stat diversification
 * 15. Pitcher Strength Index (PSI) — Opposing pitching quality factor
 * 16. Edge-Maximized Parlay Construction V3 (EMPC-V3) — Enhanced correlation-verified parlays
 *
 * 12-Gate Quality Cascade — Calibrated like NBA's successful model
 * Geometric mean combined scoring prevents weak-signal compensation
 * Walk-forward CV with purge buffer for overfitting protection
 */

window.MLBRecommendationEngineV3 = (function () {
  'use strict';

  // =========================================================================
  // CONFIGURATION — Calibrated like NBA's successful defaults
  // =========================================================================

  const CONFIG = {
    // Player eligibility
    MIN_GAMES: 8,
    MIN_AB: 2,
    WARM_UP_GAMES: 10,

    // Gravitational Floor Theory (GFT) — looser like NBA
    GFT_WINDOWS: [3, 5, 10],
    GFT_DECAY_RATE: 0.91,
    GFT_GRAVITY_STRENGTH: 0.36,
    GFT_MIN_CLEARANCE: 0.05,
    GFT_CONVERGENCE_MAX_SPREAD: 4.0,

    // Bayesian Edge Quantification (BEQ)
    BEQ_PRIOR_ALPHA: 1.0,
    BEQ_PRIOR_BETA: 1.0,
    BEQ_CREDIBLE_LEVEL: 0.87,
    BEQ_MIN_EDGE: 0.03,

    // Entropic Stability Index (ESI)
    ESI_BINS: 6,
    ESI_MAX_ENTROPY: 0.83,
    ESI_TREND_WEIGHT: 0.30,

    // Inverse Market Asymmetry Detection (IMAD)
    IMAD_MIN_ASYMMETRY: 0.05,
    IMAD_VOLUME_DISCOUNT: 0.02,

    // Consecutive Game Streak Momentum (CGSM)
    CGSM_MIN_STREAK: 1,
    CGSM_LOOKBACK: 8,
    CGSM_WEIGHT: 0.06,

    // At-Bat Volume Confidence (ABVC) — fixed division-by-zero
    ABVC_MIN_AVG_AB: 2.5,
    ABVC_ELITE_AB: 5.0,
    ABVC_WEIGHT: 0.05,

    // Poisson-Bayesian Fusion (PBF)
    PBF_DECAY_RATE: 0.92,
    PBF_MIN_LAMBDA: 0.3,
    PBF_WEIGHT: 0.10,
    PBF_MIN_PROB_MARGIN: 0.03,

    // Opponent-Adjusted Lambda (OAL)
    OAL_ENABLED: true,
    OAL_WEIGHT: 0.03,
    OAL_MAX_ADJUSTMENT: 0.20,

    // Venue Performance Differential (VPD)
    VPD_ENABLED: true,
    VPD_MIN_GAMES: 3,
    VPD_WEIGHT: 0.04,
    VPD_MAX_BOOST: 0.12,

    // NEW v3.0: Sliding-Window Temporal Attention (SWTA)
    SWTA_ENABLED: true,
    SWTA_SHORT_WINDOW: 3,
    SWTA_SHORT_DECAY: 0.95,
    SWTA_MED_WINDOW: 7,
    SWTA_MED_DECAY: 0.90,
    SWTA_LONG_WINDOW: 15,
    SWTA_LONG_DECAY: 0.85,
    SWTA_SHORT_WEIGHT: 0.50,
    SWTA_MED_WEIGHT: 0.30,
    SWTA_LONG_WEIGHT: 0.20,
    SWTA_WEIGHT: 0.06,

    // NEW v3.0: Adversarial Signal Consensus (ASC)
    ASC_ENABLED: true,
    ASC_BULL_THRESHOLD: 0.60,
    ASC_BEAR_MAX_RISK: 0.40,
    ASC_ARBITER_MIN_CONSENSUS: 0.65,
    ASC_WEIGHT: 0.04,

    // NEW v3.0: Discrete Markov Chain Transition (DMCT)
    DMCT_ENABLED: true,
    DMCT_ORDER: 2,
    DMCT_MIN_TRANSITIONS: 8,
    DMCT_WEIGHT: 0.05,
    DMCT_MIN_PROB: 0.60,

    // NEW v3.0: Temporal Regime Detection (TRD)
    TRD_ENABLED: true,
    TRD_HOT_THRESHOLD: 1.15,
    TRD_COLD_THRESHOLD: 0.85,
    TRD_WINDOW: 10,
    TRD_BOOST: 0.03,
    TRD_PENALTY: 0.05,

    // NEW v3.0: Correlation-Aware Multi-Prop Diversification (CAMPD)
    CAMPD_MAX_SAME_STAT: 2,
    CAMPD_CROSS_STAT_BONUS: 0.02,

    // NEW v3.0: Pitcher Strength Index (PSI)
    PSI_ENABLED: true,
    PSI_WEIGHT: 0.03,
    PSI_MIN_GAMES: 5,
    PSI_ELITE_THRESHOLD: 0.85,
    PSI_WEAK_THRESHOLD: 1.15,

    // Quality Gates — LOOSER like NBA to generate more bets
    GATE_MIN_GFT_SCORE: 0.35,
    GATE_MIN_BEQ_EDGE: 0.04,
    GATE_MIN_ESI_STABILITY: 0.12,
    GATE_MIN_IMAD_SCORE: 0.015,
    GATE_MIN_HIT_RATE: 0.78,
    GATE_MIN_COMBINED: 0.52,
    GATE_MIN_STREAK: 1,
    GATE_MIN_ABVC: 0.25,
    GATE_MIN_PBF_MARGIN: 0.02,

    // Bet Type Selection
    SINGLE_MIN_SCORE: 0.70,
    MULTI_SINGLE_MIN_SCORE: 0.60,
    PARLAY_LEG_MIN_SCORE: 0.52,

    // Parlay Construction — CRITICAL for ROI
    PARLAY_MIN_LEGS: 2,
    PARLAY_MAX_LEGS: 4,
    PARLAY_MAX_CORRELATION: 0.35,
    PARLAY_SAME_GAME_ALLOWED: false,
    PARLAY_MIN_COMBINED_EDGE: 0.05,
    PARLAY_MAX_PER_DAY: 2,

    // Odds Filters — WIDER like NBA
    MIN_ODDS: -500,
    MAX_ODDS: -110,
    PREFERRED_ODDS_RANGE: [-500, -110],

    // Bankroll & Kelly-Criterion Proportional Sizing (Patent-Pending)
    UNIT_SIZE: 100,
    MAX_DAILY_UNITS: 8,
    KELLY_FRACTION: 0.25,
    KELLY_SIZING: true,             // Enable Kelly-proportional bet sizing
    KELLY_MIN_UNITS: 0.5,           // Minimum bet size (0.5 units)
    KELLY_MAX_UNITS: 5.0,           // Maximum bet size (5 units)
    KELLY_SIZING_MULTIPLIER: 3.0,   // Scales Kelly fraction to unit multiplier
    PARLAY_UNIT_BONUS: 1.5,         // Parlays get additional unit multiplier (EV justified)

    // Contextual
    HOME_BOOST: 0.02,
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
    return odds > 0 ? '+' + odds : '' + odds;
  }

  function impliedProbability(odds) {
    return 1 / americanToDecimal(odds);
  }

  const STAT_MAP = {
    hits: 'h', total_bases: 'tb', rbis: 'rbi', runs: 'r', home_runs: 'hr',
    h: 'h', tb: 'tb', rbi: 'rbi', r: 'r', hr: 'hr',
    hrtb: 'hrtb',
  };

  const STAT_LABELS = {
    hits: 'H', total_bases: 'TB', rbis: 'RBI', runs: 'R', home_runs: 'HR',
    h: 'H', tb: 'TB', rbi: 'RBI', r: 'R', hr: 'HR', hrtb: 'H+R+TB',
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
   * Poisson CDF: P(X <= k) where X ~ Poisson(lambda)
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
   */
  function poissonOverProb(lambda, line) {
    const k = Math.floor(line);
    return 1 - poissonCDF(lambda, k);
  }


  // =========================================================================
  // PLAYER MODEL — Enhanced with 60-game window, transition matrix, regime
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
      // v3.0: store up to 60 games (was 50 in v2.0)
      if (this.profiles[name].games.length > 60) {
        this.profiles[name].games = this.profiles[name].games.slice(-60);
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
     * Venue Performance Differential split data
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

    /**
     * NEW v3.0: Get transition matrix for DMCT
     * Models the sequence of hit/miss outcomes as a Markov chain.
     * With order=2, tracks P(hit | last 2 outcomes).
     * Returns { matrix, currentState, nTransitions }
     */
    getTransitionMatrix(name, statKey, line, order) {
      const profile = this.profiles[name];
      if (!profile) return null;

      const games = profile.games.filter(g => (g.ab || 0) >= CONFIG.MIN_AB);
      if (games.length < order + 2) return null;

      // Build binary sequence: 1 = hit (over line), 0 = miss
      const sequence = games.map(g => ((g[statKey] || 0) > line) ? 1 : 0);

      // Build transition counts
      // For order=2, states are '00', '01', '10', '11'
      const transitions = {};
      let totalTransitions = 0;

      for (let i = order; i < sequence.length; i++) {
        const state = sequence.slice(i - order, i).join('');
        const next = sequence[i];
        if (!transitions[state]) transitions[state] = { hit: 0, miss: 0, total: 0 };
        if (next === 1) {
          transitions[state].hit++;
        } else {
          transitions[state].miss++;
        }
        transitions[state].total++;
        totalTransitions++;
      }

      // Current state: last `order` outcomes
      const currentState = sequence.slice(-order).join('');

      return {
        transitions,
        currentState,
        nTransitions: totalTransitions,
        sequenceLength: sequence.length,
      };
    },

    /**
     * NEW v3.0: Get regime data for TRD
     * Compares recent window mean to season mean
     */
    getRegime(name, statKey, window) {
      const profile = this.profiles[name];
      if (!profile) return null;

      const allGames = profile.games.filter(g => (g.ab || 0) >= CONFIG.MIN_AB);
      if (allGames.length < window + 3) return null;

      const allValues = allGames.map(g => g[statKey] || 0);
      const recentValues = allValues.slice(-window);

      const seasonMean = allValues.reduce((s, v) => s + v, 0) / allValues.length;
      const recentMean = recentValues.reduce((s, v) => s + v, 0) / recentValues.length;

      if (seasonMean === 0) return null;

      const ratio = recentMean / seasonMean;

      return {
        seasonMean,
        recentMean,
        ratio,
        sampleSize: allValues.length,
        windowSize: recentValues.length,
      };
    },
  };

  // =========================================================================
  // OPPONENT QUALITY TRACKER — For OAL and PSI computation
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
     * Returns a multiplier for opponent defensive quality
     */
    getOpponentFactor(team, statKey) {
      if (!this.teamStats[team]) return 1.0;
      const games = this.teamStats[team].gamesAsDefense;
      if (games.length < 5) return 1.0;

      const avgAllowed = games.reduce((sum, g) => sum + (g[statKey] || 0), 0) / games.length;
      const leagueAvg = this._getLeagueAvg(statKey);
      if (leagueAvg <= 0) return 1.0;

      const raw = avgAllowed / leagueAvg;
      return Math.max(1 - CONFIG.OAL_MAX_ADJUSTMENT, Math.min(1 + CONFIG.OAL_MAX_ADJUSTMENT, raw));
    },

    /**
     * NEW v3.0: Get overall pitching quality factor for PSI
     * Averages across all stat types to get overall pitching strength
     */
    getPitchingQuality(team) {
      if (!this.teamStats[team]) return null;
      const games = this.teamStats[team].gamesAsDefense;
      if (games.length < CONFIG.PSI_MIN_GAMES) return null;

      const statKeys = ['h', 'tb', 'rbi', 'r'];
      let totalRatio = 0;
      let validStats = 0;

      for (const statKey of statKeys) {
        const leagueAvg = this._getLeagueAvg(statKey);
        if (leagueAvg <= 0) continue;
        const teamAvg = games.reduce((sum, g) => sum + (g[statKey] || 0), 0) / games.length;
        totalRatio += teamAvg / leagueAvg;
        validStats++;
      }

      if (validStats === 0) return null;
      return totalRatio / validStats;
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
  // AT-BAT VOLUME CONFIDENCE (ABVC) — Fixed division-by-zero
  // =========================================================================

  function computeABVC(name) {
    const avgAB = PlayerModel.getAvgAB(name, 10);
    if (avgAB < CONFIG.ABVC_MIN_AVG_AB) return null;

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
  // POISSON-BAYESIAN FUSION (PBF)
  // =========================================================================

  function computePBF(name, statKey, line, marketOdds) {
    const values = PlayerModel.getValues(name, statKey, 20, CONFIG.MIN_AB);
    if (!values || values.length < 10) return null;

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

    const poissonProb = poissonOverProb(lambda, line);
    const mktProb = impliedProbability(marketOdds);
    const margin = poissonProb - mktProb;

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
  // OPPONENT-ADJUSTED LAMBDA (OAL)
  // =========================================================================

  function computeOAL(name, statKey, opponent) {
    if (!CONFIG.OAL_ENABLED) return null;

    const factor = OpponentTracker.getOpponentFactor(opponent, statKey);
    if (factor === 1.0) return null;

    let adjustment = 0;
    if (factor > 1.05) {
      adjustment = Math.min(CONFIG.OAL_MAX_ADJUSTMENT, (factor - 1.0) * 0.5);
    } else if (factor < 0.90) {
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
  // VENUE PERFORMANCE DIFFERENTIAL (VPD)
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

    const currentVenueMean = isHome ? homeMean : awayMean;
    const venueDiff = (currentVenueMean - overallMean) / overallMean;

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
  // NEW v3.0: SLIDING-WINDOW TEMPORAL ATTENTION (SWTA)
  // Patent-Pending — Multi-resolution temporal analysis at different
  // granularities. Inspired by autoresearch SSSL sliding window attention.
  //
  // Uses three different time windows with different decay rates:
  // - Short (3 games, 0.95 decay): captures immediate form
  // - Medium (7 games, 0.90 decay): captures recent trends
  // - Long (15 games, 0.85 decay): captures baseline consistency
  //
  // Computes weighted floor for each window, then blends using
  // configurable weights. Returns score based on how well all three
  // windows agree the player will exceed the line.
  // =========================================================================

  function computeSWTA(name, statKey, line) {
    if (!CONFIG.SWTA_ENABLED) return null;

    const windowConfigs = [
      { size: CONFIG.SWTA_SHORT_WINDOW, decay: CONFIG.SWTA_SHORT_DECAY, weight: CONFIG.SWTA_SHORT_WEIGHT },
      { size: CONFIG.SWTA_MED_WINDOW, decay: CONFIG.SWTA_MED_DECAY, weight: CONFIG.SWTA_MED_WEIGHT },
      { size: CONFIG.SWTA_LONG_WINDOW, decay: CONFIG.SWTA_LONG_DECAY, weight: CONFIG.SWTA_LONG_WEIGHT },
    ];

    const windowResults = [];
    let totalWeight = 0;
    let weightedScoreSum = 0;
    let allAboveLine = true;

    for (const wc of windowConfigs) {
      const values = PlayerModel.getValues(name, statKey, wc.size, CONFIG.MIN_AB);
      if (!values || values.length < Math.min(wc.size, 3)) return null;

      const n = values.length;

      // Compute decay-weighted mean for this window
      let wSum = 0;
      let wTotal = 0;
      for (let i = 0; i < n; i++) {
        const w = Math.pow(wc.decay, n - 1 - i);
        wSum += values[i] * w;
        wTotal += w;
      }
      const weightedMean = wSum / wTotal;

      // Compute decay-weighted floor (10th percentile)
      const weightedPairs = values.map((v, i) => ({
        val: v,
        wt: Math.pow(wc.decay, n - 1 - i),
      })).sort((a, b) => a.val - b.val);

      let cumW = 0;
      let floorVal = weightedPairs[0].val;
      for (const { val, wt } of weightedPairs) {
        cumW += wt;
        if (cumW / wTotal >= 0.10) {
          floorVal = val;
          break;
        }
      }

      // Window clearance: how far the floor is above the line
      const clearance = floorVal - line;
      const hitRate = values.filter(v => v > line).length / n;

      // Window score: blend of clearance signal and hit rate
      const clearanceScore = clearance > 0 ? Math.min(1.0, clearance / 2.0) : 0;
      const windowScore = clearanceScore * 0.4 + hitRate * 0.6;

      if (floorVal <= line) allAboveLine = false;

      windowResults.push({
        size: wc.size,
        weightedMean,
        floor: floorVal,
        clearance,
        hitRate,
        score: windowScore,
      });

      weightedScoreSum += windowScore * wc.weight;
      totalWeight += wc.weight;
    }

    // Blend scores from all windows
    const blendedScore = totalWeight > 0 ? weightedScoreSum / totalWeight : 0;

    // Agreement bonus: if all windows have positive clearance
    const agreementBonus = allAboveLine ? 0.05 : 0;

    // Convergence: measure how close window scores are to each other
    const scores = windowResults.map(w => w.score);
    const scoreSpread = Math.max(...scores) - Math.min(...scores);
    const convergenceBonus = Math.max(0, (1 - scoreSpread) * 0.03);

    const finalScore = Math.min(1.0, blendedScore + agreementBonus + convergenceBonus);

    return {
      score: finalScore,
      windows: windowResults,
      blendedScore,
      agreementBonus,
      convergenceBonus,
      allAboveLine,
    };
  }

  // =========================================================================
  // NEW v3.0: ADVERSARIAL SIGNAL CONSENSUS (ASC)
  // Patent-Pending — Three-agent adversarial validation inspired by
  // ECC's three-agent adversarial approach.
  //
  // Three evaluation paths:
  // - Bull: Focuses on edge signals (BEQ edge, PBF margin, IMAD score)
  // - Bear: Focuses on risk signals (ESI instability, trend shifts, tail risk)
  // - Arbiter: Takes geometric mean of Bull and Bear. Must exceed threshold.
  // =========================================================================

  function computeASC(allSignals) {
    if (!CONFIG.ASC_ENABLED) return null;

    const { beq, esi, gft, imad, pbf, cgsm, swta } = allSignals;

    // BULL PATH: optimistic assessment focusing on edge signals
    const bullComponents = [];

    // BEQ edge contribution (normalized to 0-1)
    if (beq && beq.edge > 0) {
      bullComponents.push(Math.min(1.0, beq.edge / 0.15 + 0.3));
    }

    // PBF margin contribution
    if (pbf && pbf.margin > 0) {
      bullComponents.push(Math.min(1.0, pbf.margin / 0.15 + 0.3));
    }

    // IMAD score contribution
    if (imad && imad.score > 0) {
      bullComponents.push(Math.min(1.0, imad.score / 0.10 + 0.3));
    }

    // GFT depth contribution
    if (gft) {
      bullComponents.push(Math.min(1.0, gft.depth * 1.2 + 0.2));
    }

    // CGSM streak contribution
    if (cgsm && cgsm.streak > 0) {
      bullComponents.push(Math.min(1.0, cgsm.streak / 5 + 0.3));
    }

    if (bullComponents.length === 0) return null;

    const bullScore = bullComponents.reduce((s, v) => s + v, 0) / bullComponents.length;

    // BEAR PATH: pessimistic assessment focusing on risk signals
    const bearRisks = [];

    // ESI instability risk
    if (esi) {
      const instabilityRisk = 1 - esi.stability;
      bearRisks.push(instabilityRisk);

      // Trend shift risk
      if (esi.trendShift > 0.15) {
        bearRisks.push(Math.min(1.0, esi.trendShift / 0.4));
      }

      // Tail risk from ESI
      const tailRiskFactor = 1 - esi.tailRisk;
      if (tailRiskFactor > 0.2) {
        bearRisks.push(tailRiskFactor);
      }
    }

    // Streak fragility: if streak is 0 or 1, higher risk
    if (cgsm) {
      if (cgsm.streak <= 1) {
        bearRisks.push(0.4);
      }
      // Low window hit rate is risky
      if (cgsm.windowHitRate < 0.7) {
        bearRisks.push(Math.min(1.0, (0.7 - cgsm.windowHitRate) * 3));
      }
    }

    // PBF lambda stability risk
    if (pbf && pbf.lambdaStability < 0.8) {
      bearRisks.push(1 - pbf.lambdaStability);
    }

    // SWTA disagreement risk
    if (swta && !swta.allAboveLine) {
      bearRisks.push(0.3);
    }

    // Bear score: average risk (lower = less risky = better)
    const avgRisk = bearRisks.length > 0
      ? bearRisks.reduce((s, v) => s + v, 0) / bearRisks.length
      : 0.5;

    // Bear score: inverse of risk (higher = safer)
    const bearScore = Math.max(0, 1 - avgRisk);

    // ARBITER: geometric mean of bull and bear scores
    const arbiterScore = Math.sqrt(Math.max(0.001, bullScore) * Math.max(0.001, bearScore));

    const consensus = arbiterScore >= CONFIG.ASC_ARBITER_MIN_CONSENSUS;

    return {
      bullScore: Math.round(bullScore * 1000) / 1000,
      bearScore: Math.round(bearScore * 1000) / 1000,
      arbiterScore: Math.round(arbiterScore * 1000) / 1000,
      consensus,
      avgRisk: Math.round(avgRisk * 1000) / 1000,
    };
  }

  // =========================================================================
  // NEW v3.0: DISCRETE MARKOV CHAIN TRANSITION (DMCT)
  // Patent-Pending — Models hit/miss sequences as a Markov chain.
  // With order=2, tracks P(hit | last 2 outcomes).
  // Captures momentum and mean-reversion patterns unique to baseball.
  // =========================================================================

  function computeDMCT(name, statKey, line) {
    if (!CONFIG.DMCT_ENABLED) return null;

    const matrixData = PlayerModel.getTransitionMatrix(
      name, statKey, line, CONFIG.DMCT_ORDER
    );
    if (!matrixData) return null;

    const { transitions, currentState, nTransitions } = matrixData;

    if (nTransitions < CONFIG.DMCT_MIN_TRANSITIONS) return null;

    // Get transition probability from current state
    const stateData = transitions[currentState];
    if (!stateData || stateData.total < 2) {
      // Fallback: use overall hit rate across all states
      let totalHits = 0;
      let totalObs = 0;
      for (const state of Object.values(transitions)) {
        totalHits += state.hit;
        totalObs += state.total;
      }
      if (totalObs === 0) return null;

      const overallProb = totalHits / totalObs;
      return {
        score: Math.min(1.0, overallProb),
        transitionProb: overallProb,
        currentState,
        nTransitions,
        confidence: 0.5,
        source: 'fallback',
      };
    }

    const transitionProb = stateData.hit / stateData.total;

    // Confidence based on number of observations for this state
    const stateConfidence = Math.min(1.0, stateData.total / 10);

    // Blend transition probability with overall rate for smoothing
    let totalHits = 0;
    let totalObs = 0;
    for (const state of Object.values(transitions)) {
      totalHits += state.hit;
      totalObs += state.total;
    }
    const overallRate = totalObs > 0 ? totalHits / totalObs : 0.5;

    // Bayesian-style smoothing: weight state-specific vs overall
    const smoothedProb = stateConfidence * transitionProb + (1 - stateConfidence) * overallRate;

    // Score normalized
    const score = Math.min(1.0, smoothedProb);

    return {
      score,
      transitionProb,
      smoothedProb,
      currentState,
      nTransitions,
      stateTotal: stateData.total,
      confidence: stateConfidence,
      overallRate,
      source: 'state',
    };
  }

  // =========================================================================
  // NEW v3.0: TEMPORAL REGIME DETECTION (TRD)
  // Patent-Pending — Detects hot/cold/normal performance regimes.
  // Compares recent window mean to season mean.
  // Hot regime: boost confidence. Cold regime: penalize.
  // =========================================================================

  function computeTRD(name, statKey) {
    if (!CONFIG.TRD_ENABLED) return null;

    const regimeData = PlayerModel.getRegime(name, statKey, CONFIG.TRD_WINDOW);
    if (!regimeData) return null;

    const { ratio, seasonMean, recentMean } = regimeData;

    let regime = 'normal';
    let adjustment = 0;

    if (ratio >= CONFIG.TRD_HOT_THRESHOLD) {
      regime = 'hot';
      adjustment = CONFIG.TRD_BOOST;
    } else if (ratio <= CONFIG.TRD_COLD_THRESHOLD) {
      regime = 'cold';
      adjustment = -CONFIG.TRD_PENALTY;
    }

    return {
      regime,
      ratio: Math.round(ratio * 1000) / 1000,
      adjustment,
      seasonMean: Math.round(seasonMean * 1000) / 1000,
      recentMean: Math.round(recentMean * 1000) / 1000,
    };
  }

  // =========================================================================
  // NEW v3.0: PITCHER STRENGTH INDEX (PSI)
  // Patent-Pending — Uses OpponentTracker data to assess pitching quality.
  // Elite pitching (allows < 85% of league avg): penalty.
  // Weak pitching (allows > 115% of league avg): boost.
  // =========================================================================

  function computePSI(opponent) {
    if (!CONFIG.PSI_ENABLED) return null;

    const quality = OpponentTracker.getPitchingQuality(opponent);
    if (quality === null) return null;

    let factor = 0;
    let assessment = 'average';

    if (quality <= CONFIG.PSI_ELITE_THRESHOLD) {
      // Elite pitching staff — penalize hitter confidence
      factor = -Math.min(0.10, (CONFIG.PSI_ELITE_THRESHOLD - quality) * 0.5);
      assessment = 'elite';
    } else if (quality >= CONFIG.PSI_WEAK_THRESHOLD) {
      // Weak pitching staff — boost hitter confidence
      factor = Math.min(0.10, (quality - CONFIG.PSI_WEAK_THRESHOLD) * 0.5);
      assessment = 'weak';
    } else {
      factor = 0;
      assessment = 'average';
    }

    const score = 0.5 + factor;

    return {
      score: Math.max(0, Math.min(1.0, score)),
      factor,
      assessment,
      quality: Math.round(quality * 1000) / 1000,
      opponent,
    };
  }

  // =========================================================================
  // NEW v3.0: CORRELATION-AWARE MULTI-PROP DIVERSIFICATION (CAMPD)
  // Patent-Pending — Post-processing step that limits same-stat picks
  // per day and adds bonus for cross-stat diversity.
  // =========================================================================

  function applyCAMPD(candidates) {
    if (!candidates || candidates.length === 0) return candidates;

    // Count picks by stat type
    const statCounts = {};
    const result = [];

    // Sort by score descending so we keep the best ones
    const sorted = [...candidates].sort((a, b) => b.cascadeScore - a.cascadeScore);

    for (const c of sorted) {
      const stat = STAT_MAP[c.statType] || c.statType;
      statCounts[stat] = (statCounts[stat] || 0);

      if (statCounts[stat] >= CONFIG.CAMPD_MAX_SAME_STAT) {
        continue; // Skip — too many of same stat type
      }

      statCounts[stat]++;
      result.push(c);
    }

    // Apply cross-stat diversity bonus
    const uniqueStats = Object.keys(statCounts).filter(k => statCounts[k] > 0);
    if (uniqueStats.length >= 2) {
      for (const c of result) {
        c.campdBonus = CONFIG.CAMPD_CROSS_STAT_BONUS;
        c.cascadeScore = Math.min(1.0, c.cascadeScore + c.campdBonus);
      }
    }

    return result;
  }


  // =========================================================================
  // ULTRA SIGNAL V3 — 12-Gate Quality Cascade
  // Uses geometric mean of core 4 + weighted contributions from all signals
  // =========================================================================

  function computeUltraSignalV3(playerName, statKey, line, marketOdds, opponent, isHome) {
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

    // Gate 8: Poisson-Bayesian Fusion
    const pbf = computePBF(playerName, statKey, line, marketOdds);
    if (!pbf || pbf.margin < CONFIG.GATE_MIN_PBF_MARGIN) return null;

    // Compute v3.0 signals (non-gating first pass)
    const swta = computeSWTA(playerName, statKey, line);
    const dmct = computeDMCT(playerName, statKey, line);
    const trd = computeTRD(playerName, statKey);
    const psi = computePSI(opponent);

    // Gate 9: DMCT transition probability (soft gate — only if enough data)
    if (CONFIG.DMCT_ENABLED && dmct && dmct.confidence >= 0.5) {
      if (dmct.smoothedProb < CONFIG.DMCT_MIN_PROB) {
        // Soft gate: don't reject outright, but mark as lower confidence
        // Only reject if probability is very low
        if (dmct.smoothedProb < CONFIG.DMCT_MIN_PROB - 0.10) return null;
      }
    }

    // Gate 10: TRD cold regime hard rejection
    if (CONFIG.TRD_ENABLED && trd && trd.regime === 'cold') {
      return null;
    }

    // Compute optional enhancers
    const oal = computeOAL(playerName, statKey, opponent);
    const vpd = computeVPD(playerName, statKey, isHome);

    // Compute ASC with all signals collected
    const asc = computeASC({ beq, esi, gft, imad, pbf, cgsm, swta });

    // Combined score: geometric mean of core 4 (GFT, normalized BEQ, ESI, normalized IMAD)
    const coreComponents = [
      gft.score,
      Math.min(1.0, beq.edge / 0.20 + 0.5),
      esi.stability,
      Math.min(1.0, imad.score / 0.15 + 0.5),
    ];

    const logSum = coreComponents.reduce((s, c) => s + Math.log(Math.max(0.001, c)), 0);
    let combined = Math.exp(logSum / coreComponents.length);

    // Weight allocation: core + all signal contributions
    const totalAddonWeight = CONFIG.CGSM_WEIGHT + CONFIG.ABVC_WEIGHT + CONFIG.PBF_WEIGHT +
      CONFIG.SWTA_WEIGHT + CONFIG.ASC_WEIGHT + CONFIG.DMCT_WEIGHT +
      CONFIG.OAL_WEIGHT + CONFIG.VPD_WEIGHT + CONFIG.PSI_WEIGHT;
    const coreWeight = 1.0 - totalAddonWeight;

    combined = combined * coreWeight;

    // Add weighted contributions from all signals
    combined += cgsm.score * CONFIG.CGSM_WEIGHT;
    combined += abvc.score * CONFIG.ABVC_WEIGHT;
    combined += pbf.score * CONFIG.PBF_WEIGHT;

    // SWTA contribution
    if (swta) {
      combined += swta.score * CONFIG.SWTA_WEIGHT;
    }

    // ASC contribution (only if consensus)
    if (asc && asc.consensus) {
      combined += asc.arbiterScore * CONFIG.ASC_WEIGHT;
    }

    // DMCT contribution
    if (dmct) {
      combined += dmct.score * CONFIG.DMCT_WEIGHT;
    }

    // OAL contribution
    if (oal && oal.adjustment > 0) {
      combined += oal.adjustment * CONFIG.OAL_WEIGHT;
    }

    // VPD contribution
    if (vpd && vpd.boost > 0) {
      combined += vpd.boost * CONFIG.VPD_WEIGHT;
    }

    // PSI contribution
    if (psi) {
      combined += psi.factor * CONFIG.PSI_WEIGHT;
    }

    // TRD adjustment
    if (trd) {
      combined += trd.adjustment;
    }

    // Contextual: home boost
    if (isHome) {
      combined += CONFIG.HOME_BOOST;
    }

    combined = Math.max(0, Math.min(1.0, combined));

    // Gate 11: Combined score gate
    if (combined < CONFIG.GATE_MIN_COMBINED) return null;

    const decimal = americanToDecimal(marketOdds);
    const ev = beq.lowerBound * decimal - 1;

    const kellyFull = (beq.lowerBound * decimal - 1) / (decimal - 1);
    const kelly = Math.max(0, kellyFull * CONFIG.KELLY_FRACTION);

    return {
      player: playerName, stat: statKey, line, odds: marketOdds,
      combinedScore: combined, gft, beq, esi, imad, cgsm, abvc, pbf,
      oal, vpd, swta, asc, dmct, trd, psi,
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

    const signal = computeUltraSignalV3(playerName, statKey, line, odds, opponent, isHome);
    if (!signal) return null;

    const profile = PlayerModel.getProfile(playerName);
    const decimal = americanToDecimal(odds);

    return {
      player: playerName,
      team: profile ? profile.team : '',
      statType,
      statLabel: STAT_LABELS[statType] || statType.toUpperCase(),
      line, odds, decimalOdds: decimal,
      engine: 'ultra-v3',
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
      swta: signal.swta ? signal.swta.score : null,
      asc: signal.asc ? signal.asc.arbiterScore : null,
      ascConsensus: signal.asc ? signal.asc.consensus : null,
      dmct: signal.dmct ? signal.dmct.score : null,
      dmctProb: signal.dmct ? signal.dmct.transitionProb : null,
      trd: signal.trd ? signal.trd.regime : null,
      trdAdjustment: signal.trd ? signal.trd.adjustment : null,
      psi: signal.psi ? signal.psi.score : null,
      psiAssessment: signal.psi ? signal.psi.assessment : null,
      cascadeScore: signal.combinedScore,
      impliedProb: signal.marketImplied,
      hitRate: signal.hitRate,
      edge: signal.edge,
      ev: signal.ev,
      avg: signal.esi.mean,
      floor: signal.gft.floors[signal.gft.floors.length - 1],
      streak: signal.cgsm.streak,
      avgAB: signal.abvc.avgAB,
      kelly: signal.kelly || 0,
      campdBonus: 0,
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
  // BET TYPE SELECTOR — Enhanced for v3.0 with parlay generation
  // =========================================================================

  function selectBetType(candidates) {
    // Apply CAMPD diversification
    const diversified = applyCAMPD(candidates);

    const sorted = [...diversified].sort((a, b) => b.cascadeScore - a.cascadeScore);

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
      recommendations.singles = elite.slice(0, 5);
      recommendations.betType = elite.length === 1 ? 'single' : 'multi_single';
      recommendations.reasoning = elite.length + ' elite-confidence pick(s) found (score >= ' + CONFIG.SINGLE_MIN_SCORE + ')';
    }

    // Build parlays from moderate candidates — up to PARLAY_MAX_PER_DAY
    if (moderate.length >= CONFIG.PARLAY_MIN_LEGS) {
      const parlays = buildOptimalParlayV3(moderate);
      for (const parlay of parlays) {
        recommendations.parlays.push(parlay);
      }
      if (recommendations.betType === 'none' && parlays.length > 0) {
        recommendations.betType = 'parlay';
        recommendations.reasoning = moderate.length + ' strong candidates for EMPC-V3 parlay construction';
      }
    }

    if (recommendations.singles.length === 0 && strong.length > 0) {
      recommendations.singles = strong.slice(0, 5);
      if (recommendations.betType === 'none') {
        recommendations.betType = strong.length === 1 ? 'single' : 'multi_single';
        recommendations.reasoning = strong.length + ' strong-confidence pick(s) found';
      }
    }

    return recommendations;
  }


  // =========================================================================
  // PARLAY CONSTRUCTION V3 — Enhanced with CAMPD, cross-stat diversity
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

  function buildOptimalParlayV3(candidates) {
    const parlays = [];
    const sorted = [...candidates].sort((a, b) => b.edge - a.edge);

    // Build up to PARLAY_MAX_PER_DAY parlays
    const usedInParlays = new Set();

    for (let parlayIdx = 0; parlayIdx < CONFIG.PARLAY_MAX_PER_DAY; parlayIdx++) {
      const selected = [];
      const usedTeams = new Set();
      const usedPlayers = new Set();
      const usedStats = {};

      for (const leg of sorted) {
        if (selected.length >= CONFIG.PARLAY_MAX_LEGS) break;
        if (usedPlayers.has(leg.player)) continue;

        // For second+ parlay, try to use different legs
        if (parlayIdx > 0 && usedInParlays.has(leg.player + '|' + leg.statType)) continue;

        const team = leg.team;
        if (!CONFIG.PARLAY_SAME_GAME_ALLOWED && usedTeams.has(team)) continue;

        // CAMPD: prefer cross-stat diversity in parlays
        const stat = STAT_MAP[leg.statType] || leg.statType;
        if ((usedStats[stat] || 0) >= CONFIG.CAMPD_MAX_SAME_STAT) continue;

        // Check correlation with existing legs
        // PATENT-PENDING: Independence-Aware Correlation Check (IACC)
        // Players from different games are statistically independent by definition
        // Only check correlation for same-game matchups (same gameKey)
        let maxCorr = 0;
        for (const existing of selected) {
          // If different games, outcomes are independent — correlation is 0
          if (leg.gameKey !== existing.gameKey) continue;

          // Same game: compute actual statistical correlation
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
        usedStats[stat] = (usedStats[stat] || 0) + 1;
      }

      if (selected.length < CONFIG.PARLAY_MIN_LEGS) break;

      const decimal = selected.reduce((d, l) => d * l.decimalOdds, 1);
      const american = decimalToAmerican(decimal);
      const combinedHitRate = selected.reduce((p, l) => p * l.hitRate, 1);
      const totalEdge = selected.reduce((s, l) => s + l.edge, 0);
      const ev = combinedHitRate * decimal - 1;

      if (ev <= 0 || totalEdge < CONFIG.PARLAY_MIN_COMBINED_EDGE) continue;

      // CAMPD cross-stat bonus for parlay
      const parlayStats = new Set(selected.map(l => STAT_MAP[l.statType] || l.statType));
      const crossStatBonus = parlayStats.size >= 2 ? CONFIG.CAMPD_CROSS_STAT_BONUS : 0;

      parlays.push({
        legs: selected,
        numLegs: selected.length,
        odds: american,
        decimalOdds: Math.round(decimal * 100) / 100,
        combinedHitRate: Math.round(combinedHitRate * 10000) / 10000,
        ev: Math.round(ev * 1000) / 1000,
        avgCascade: selected.reduce((s, l) => s + l.cascadeScore, 0) / selected.length,
        totalEdge,
        crossStatBonus,
        uniqueStats: parlayStats.size,
      });

      // Mark legs as used for this parlay round
      for (const leg of selected) {
        usedInParlays.add(leg.player + '|' + leg.statType);
      }
    }

    return parlays;
  }

  // =========================================================================
  // WALK-FORWARD BACKTEST — Enhanced with parlay tracking and OAL/VPD/PSI
  // =========================================================================

  function runBacktest(boxScores, historicalOdds) {
    PlayerModel.reset();
    OpponentTracker.reset();

    const sortedGames = [...boxScores].sort((a, b) => a.date.localeCompare(b.date));

    const oddsByDate = {};
    for (const od of (historicalOdds || [])) {
      if (!oddsByDate[od.date]) oddsByDate[od.date] = {};
      const key = od.gameKey || (od.awayTeam + '@' + od.homeTeam);
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
          const gameKey = bg.away + '@' + bg.home;
          const og = dayOdds[gameKey];
          if (!og) continue;

          for (const player of (bg.players || [])) {
            // Note: we do NOT filter on actual at-bats here to avoid
            // survivorship bias. The player model's gate checks handle
            // eligibility using only pre-game data. Filtering on actual ABs
            // would exclude players who were scratched or pulled early —
            // bets a real bettor would have lost.
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

            // Check home runs props
            if (og.hrProps && og.hrProps[player.name]) {
              const prop = findBestProp(player.name, 'home_runs', og.hrProps[player.name], opponent, isHome);
              if (prop) {
                prop.actual = player.hr;
                prop.hit = player.hr > prop.line;
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
            // PATENT-PENDING: Kelly-Criterion Proportional Sizing (KCPS)
            // Bet size scales with edge magnitude — higher edge = more units
            // Mathematically optimal for geometric bankroll growth (Kelly 1956)
            let unitMult = 1.0;
            if (CONFIG.KELLY_SIZING && single.kelly > 0) {
              unitMult = Math.max(CONFIG.KELLY_MIN_UNITS,
                Math.min(CONFIG.KELLY_MAX_UNITS,
                  1.0 + single.kelly * CONFIG.KELLY_SIZING_MULTIPLIER));
            }
            const betSize = Math.round(CONFIG.UNIT_SIZE * unitMult);

            allSignals.push({
              ...single,
              betType: 'single',
              date,
              unitMultiplier: unitMult,
              betSize,
              pnl: single.hit
                ? Math.round((single.decimalOdds - 1) * betSize)
                : -betSize,
              wagered: betSize,
            });
          }

          for (const parlay of recommendation.parlays) {
            const parlayHit = parlay.legs.every(l => l.hit);
            // Kelly sizing for parlays: use average leg kelly + parlay bonus
            let unitMult = 1.0;
            if (CONFIG.KELLY_SIZING) {
              const avgKelly = parlay.legs.reduce((s, l) => s + (l.kelly || 0), 0) / parlay.legs.length;
              unitMult = Math.max(CONFIG.KELLY_MIN_UNITS,
                Math.min(CONFIG.KELLY_MAX_UNITS,
                  CONFIG.PARLAY_UNIT_BONUS + avgKelly * CONFIG.KELLY_SIZING_MULTIPLIER));
            }
            const betSize = Math.round(CONFIG.UNIT_SIZE * unitMult);

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
              crossStatBonus: parlay.crossStatBonus,
              uniqueStats: parlay.uniqueStats,
              unitMultiplier: unitMult,
              betSize,
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
                kelly: l.kelly || 0,
                bet: l.player + ' O' + l.line + ' ' + l.statLabel,
              })),
              pnl: parlayHit
                ? Math.round((parlay.decimalOdds - 1) * betSize)
                : -betSize,
              wagered: betSize,
            });
          }
        }
      }

      // Walk-forward: update models AFTER generating signals (once per date)
      if (updatedDates.has(game.date)) continue;
      updatedDates.add(game.date);
      const dateBoxes = boxByDate[game.date] || [];
      for (const bg of dateBoxes) {
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

          if (isHome) {
            homePlayerStats.h += p.h || 0;
            homePlayerStats.tb += p.tb || 0;
            homePlayerStats.rbi += p.rbi || 0;
            homePlayerStats.r += p.r || 0;
            homePlayerStats.hr += p.hr || 0;
            homePlayerStats.count++;
          } else {
            awayPlayerStats.h += p.h || 0;
            awayPlayerStats.tb += p.tb || 0;
            awayPlayerStats.rbi += p.rbi || 0;
            awayPlayerStats.r += p.r || 0;
            awayPlayerStats.hr += p.hr || 0;
            awayPlayerStats.count++;
          }
        }

        if (awayPlayerStats.count > 0) {
          OpponentTracker.update(bg.home, {
            h: awayPlayerStats.h / awayPlayerStats.count,
            tb: awayPlayerStats.tb / awayPlayerStats.count,
            rbi: awayPlayerStats.rbi / awayPlayerStats.count,
            r: awayPlayerStats.r / awayPlayerStats.count,
            hr: awayPlayerStats.hr / awayPlayerStats.count,
          });
        }
        if (homePlayerStats.count > 0) {
          OpponentTracker.update(bg.away, {
            h: homePlayerStats.h / homePlayerStats.count,
            tb: homePlayerStats.tb / homePlayerStats.count,
            rbi: homePlayerStats.rbi / homePlayerStats.count,
            r: homePlayerStats.r / homePlayerStats.count,
            hr: homePlayerStats.hr / homePlayerStats.count,
          });
        }
      }
    }

    // Compute stats with Kelly-proportional wagered amounts
    const singles = allSignals.filter(s => s.betType === 'single');
    const parlays = allSignals.filter(s => s.betType === 'parlay');

    const singleWins = singles.filter(s => s.hit).length;
    const singlePnl = singles.reduce((s, p) => s + p.pnl, 0);
    const singleWagered = singles.reduce((s, p) => s + (p.wagered || CONFIG.UNIT_SIZE), 0);

    const parlayWins = parlays.filter(s => s.hit).length;
    const parlayPnl = parlays.reduce((s, p) => s + p.pnl, 0);
    const parlayWagered = parlays.reduce((s, p) => s + (p.wagered || CONFIG.UNIT_SIZE), 0);

    let totalLegs = 0, hitLegs = 0;
    parlays.forEach(p => {
      (p.legs || []).forEach(l => {
        totalLegs++;
        if (l.hit) hitLegs++;
      });
    });

    const totalWagered = singleWagered + parlayWagered;

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
          wagered: singleWagered,
          roi: singleWagered > 0 ? singlePnl / singleWagered : 0,
        },
        parlays: {
          total: parlays.length,
          wins: parlayWins,
          accuracy: parlays.length > 0 ? parlayWins / parlays.length : 0,
          pnl: parlayPnl,
          wagered: parlayWagered,
          roi: parlayWagered > 0 ? parlayPnl / parlayWagered : 0,
          totalLegs,
          hitLegs,
          legAccuracy: totalLegs > 0 ? hitLegs / totalLegs : 0,
        },
        overall: {
          total: allSignals.length,
          wins: singleWins + parlayWins,
          accuracy: allSignals.length > 0 ? (singleWins + parlayWins) / allSignals.length : 0,
          pnl: singlePnl + parlayPnl,
          wagered: totalWagered,
          roi: totalWagered > 0 ? (singlePnl + parlayPnl) / totalWagered : 0,
        },
      },
    };
  }


  // =========================================================================
  // PURGED WALK-FORWARD CROSS-VALIDATION
  // Splits data into nFolds temporal folds with purge buffer between
  // train/test to prevent information leakage. Validates against overfitting.
  // =========================================================================

  function runCrossValidation(boxScores, historicalOdds, nFolds) {
    if (!nFolds || nFolds < 2) nFolds = 5;

    const sortedGames = [...boxScores].sort((a, b) => a.date.localeCompare(b.date));
    const allDates = [...new Set(sortedGames.map(g => g.date))].sort();

    if (allDates.length < nFolds * 5) {
      return { error: 'Insufficient data for cross-validation', folds: [] };
    }

    const foldSize = Math.floor(allDates.length / nFolds);
    const purgeBuffer = 3; // 3-day purge between train and test
    const foldResults = [];

    for (let fold = 0; fold < nFolds; fold++) {
      const testStart = fold * foldSize;
      const testEnd = Math.min(testStart + foldSize, allDates.length);
      const testDates = new Set(allDates.slice(testStart, testEnd));

      // Training dates: everything before test minus purge buffer
      const trainEndIdx = Math.max(0, testStart - purgeBuffer);
      const trainDates = new Set(allDates.slice(0, trainEndIdx));

      if (trainDates.size < CONFIG.WARM_UP_GAMES) {
        // Not enough training data for this fold
        continue;
      }

      // Filter box scores and odds for this fold
      const trainBoxScores = sortedGames.filter(g => trainDates.has(g.date));
      const testBoxScores = sortedGames.filter(g => testDates.has(g.date));
      const trainOdds = (historicalOdds || []).filter(o => trainDates.has(o.date));
      const testOdds = (historicalOdds || []).filter(o => testDates.has(o.date));

      // Run backtest: first train on training data to build models
      PlayerModel.reset();
      OpponentTracker.reset();

      // Build models from training data
      const trainSorted = [...trainBoxScores].sort((a, b) => a.date.localeCompare(b.date));
      for (const game of trainSorted) {
        const dateBoxes = trainBoxScores.filter(g => g.date === game.date);
        for (const bg of dateBoxes) {
          const homePlayerStats = { h: 0, tb: 0, rbi: 0, r: 0, hr: 0, count: 0 };
          const awayPlayerStats = { h: 0, tb: 0, rbi: 0, r: 0, hr: 0, count: 0 };

          for (const p of (bg.players || [])) {
            const ab = p.ab || 0;
            if (ab < 1) continue;

            const isHome = p.team === bg.home;
            const opponent = isHome ? bg.away : bg.home;

            PlayerModel.update(p.name, {
              h: p.h || 0, tb: p.tb || 0, rbi: p.rbi || 0,
              r: p.r || 0, hr: p.hr || 0, ab: ab,
              bb: p.bb || 0, so: p.so || 0, sb: p.sb || 0,
            }, game.date, p.team, opponent, isHome);

            if (isHome) {
              homePlayerStats.h += p.h || 0;
              homePlayerStats.tb += p.tb || 0;
              homePlayerStats.rbi += p.rbi || 0;
              homePlayerStats.r += p.r || 0;
              homePlayerStats.hr += p.hr || 0;
              homePlayerStats.count++;
            } else {
              awayPlayerStats.h += p.h || 0;
              awayPlayerStats.tb += p.tb || 0;
              awayPlayerStats.rbi += p.rbi || 0;
              awayPlayerStats.r += p.r || 0;
              awayPlayerStats.hr += p.hr || 0;
              awayPlayerStats.count++;
            }
          }

          if (awayPlayerStats.count > 0) {
            OpponentTracker.update(bg.home, {
              h: awayPlayerStats.h / awayPlayerStats.count,
              tb: awayPlayerStats.tb / awayPlayerStats.count,
              rbi: awayPlayerStats.rbi / awayPlayerStats.count,
              r: awayPlayerStats.r / awayPlayerStats.count,
              hr: awayPlayerStats.hr / awayPlayerStats.count,
            });
          }
          if (homePlayerStats.count > 0) {
            OpponentTracker.update(bg.away, {
              h: homePlayerStats.h / homePlayerStats.count,
              tb: homePlayerStats.tb / homePlayerStats.count,
              rbi: homePlayerStats.rbi / homePlayerStats.count,
              r: homePlayerStats.r / homePlayerStats.count,
              hr: homePlayerStats.hr / homePlayerStats.count,
            });
          }
        }
      }

      // Now generate predictions on test data (walk-forward within test)
      const oddsByDate = {};
      for (const od of testOdds) {
        if (!oddsByDate[od.date]) oddsByDate[od.date] = {};
        const key = od.gameKey || (od.awayTeam + '@' + od.homeTeam);
        oddsByDate[od.date][key] = od;
      }

      const boxByDate = {};
      for (const g of testBoxScores) {
        (boxByDate[g.date] || (boxByDate[g.date] = [])).push(g);
      }

      const foldSignals = [];
      const processedDates = new Set();

      for (const game of testBoxScores.sort((a, b) => a.date.localeCompare(b.date))) {
        const date = game.date;

        if (!processedDates.has(date)) {
          processedDates.add(date);

          const dayOdds = oddsByDate[date] || {};
          const dayBoxes = boxByDate[date] || [];
          const candidates = [];

          for (const bg of dayBoxes) {
            const gameKey = bg.away + '@' + bg.home;
            const og = dayOdds[gameKey];
            if (!og) continue;

            for (const player of (bg.players || [])) {
              const ab = player.ab || 0;
              if (ab < CONFIG.MIN_AB) continue;

              const isHome = player.team === bg.home;
              const opponent = isHome ? bg.away : bg.home;

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
              let unitMult = 1.0;
              if (CONFIG.KELLY_SIZING && single.kelly > 0) {
                unitMult = Math.max(CONFIG.KELLY_MIN_UNITS,
                  Math.min(CONFIG.KELLY_MAX_UNITS,
                    1.0 + single.kelly * CONFIG.KELLY_SIZING_MULTIPLIER));
              }
              const betSize = Math.round(CONFIG.UNIT_SIZE * unitMult);
              foldSignals.push({
                ...single,
                betType: 'single',
                date,
                hit: single.hit,
                wagered: betSize,
                pnl: single.hit
                  ? Math.round((single.decimalOdds - 1) * betSize)
                  : -betSize,
              });
            }

            for (const parlay of recommendation.parlays) {
              const parlayHit = parlay.legs.every(l => l.hit);
              let unitMult = 1.0;
              if (CONFIG.KELLY_SIZING) {
                const avgKelly = parlay.legs.reduce((s, l) => s + (l.kelly || 0), 0) / parlay.legs.length;
                unitMult = Math.max(CONFIG.KELLY_MIN_UNITS,
                  Math.min(CONFIG.KELLY_MAX_UNITS,
                    CONFIG.PARLAY_UNIT_BONUS + avgKelly * CONFIG.KELLY_SIZING_MULTIPLIER));
              }
              const betSize = Math.round(CONFIG.UNIT_SIZE * unitMult);
              foldSignals.push({
                betType: 'parlay',
                date,
                hit: parlayHit,
                n_legs: parlay.numLegs,
                wagered: betSize,
                pnl: parlayHit
                  ? Math.round((parlay.decimalOdds - 1) * betSize)
                  : -betSize,
              });
            }
          }
        }

        // Walk-forward: update models with test data after prediction
        const dateBoxes = boxByDate[game.date] || [];
        for (const bg of dateBoxes) {
          const homePlayerStats = { h: 0, tb: 0, rbi: 0, r: 0, hr: 0, count: 0 };
          const awayPlayerStats = { h: 0, tb: 0, rbi: 0, r: 0, hr: 0, count: 0 };

          for (const p of (bg.players || [])) {
            const ab = p.ab || 0;
            if (ab < 1) continue;
            const isHome = p.team === bg.home;
            const opponent = isHome ? bg.away : bg.home;

            PlayerModel.update(p.name, {
              h: p.h || 0, tb: p.tb || 0, rbi: p.rbi || 0,
              r: p.r || 0, hr: p.hr || 0, ab: ab,
              bb: p.bb || 0, so: p.so || 0, sb: p.sb || 0,
            }, game.date, p.team, opponent, isHome);

            if (isHome) {
              homePlayerStats.h += p.h || 0;
              homePlayerStats.tb += p.tb || 0;
              homePlayerStats.rbi += p.rbi || 0;
              homePlayerStats.r += p.r || 0;
              homePlayerStats.hr += p.hr || 0;
              homePlayerStats.count++;
            } else {
              awayPlayerStats.h += p.h || 0;
              awayPlayerStats.tb += p.tb || 0;
              awayPlayerStats.rbi += p.rbi || 0;
              awayPlayerStats.r += p.r || 0;
              awayPlayerStats.hr += p.hr || 0;
              awayPlayerStats.count++;
            }
          }

          if (awayPlayerStats.count > 0) {
            OpponentTracker.update(bg.home, {
              h: awayPlayerStats.h / awayPlayerStats.count,
              tb: awayPlayerStats.tb / awayPlayerStats.count,
              rbi: awayPlayerStats.rbi / awayPlayerStats.count,
              r: awayPlayerStats.r / awayPlayerStats.count,
              hr: awayPlayerStats.hr / awayPlayerStats.count,
            });
          }
          if (homePlayerStats.count > 0) {
            OpponentTracker.update(bg.away, {
              h: homePlayerStats.h / homePlayerStats.count,
              tb: homePlayerStats.tb / homePlayerStats.count,
              rbi: homePlayerStats.rbi / homePlayerStats.count,
              r: homePlayerStats.r / homePlayerStats.count,
              hr: homePlayerStats.hr / homePlayerStats.count,
            });
          }
        }
      }

      // Fold stats
      const foldSingles = foldSignals.filter(s => s.betType === 'single');
      const foldParlays = foldSignals.filter(s => s.betType === 'parlay');
      const foldWins = foldSignals.filter(s => s.hit).length;
      const foldPnl = foldSignals.reduce((s, p) => s + p.pnl, 0);
      const foldWagered = foldSignals.reduce((s, p) => s + (p.wagered || CONFIG.UNIT_SIZE), 0);

      foldResults.push({
        fold: fold + 1,
        trainDays: trainDates.size,
        testDays: testDates.size,
        totalSignals: foldSignals.length,
        singles: foldSingles.length,
        parlays: foldParlays.length,
        wins: foldWins,
        accuracy: foldSignals.length > 0 ? foldWins / foldSignals.length : 0,
        pnl: foldPnl,
        wagered: foldWagered,
        roi: foldWagered > 0 ? foldPnl / foldWagered : 0,
      });
    }

    // Aggregate stats across folds
    const totalSignals = foldResults.reduce((s, f) => s + f.totalSignals, 0);
    const totalWins = foldResults.reduce((s, f) => s + f.wins, 0);
    const totalPnl = foldResults.reduce((s, f) => s + f.pnl, 0);
    const accuracies = foldResults.filter(f => f.totalSignals > 0).map(f => f.accuracy);
    const rois = foldResults.filter(f => f.totalSignals > 0).map(f => f.roi);

    return {
      nFolds: foldResults.length,
      purgeBuffer,
      folds: foldResults,
      aggregate: {
        totalSignals,
        totalWins,
        overallAccuracy: totalSignals > 0 ? totalWins / totalSignals : 0,
        overallPnl: totalPnl,
        overallWagered: foldResults.reduce((s, f) => s + (f.wagered || f.totalSignals * CONFIG.UNIT_SIZE), 0),
        overallROI: foldResults.reduce((s, f) => s + (f.wagered || f.totalSignals * CONFIG.UNIT_SIZE), 0) > 0
          ? totalPnl / foldResults.reduce((s, f) => s + (f.wagered || f.totalSignals * CONFIG.UNIT_SIZE), 0) : 0,
        meanAccuracy: accuracies.length > 0 ? accuracies.reduce((s, v) => s + v, 0) / accuracies.length : 0,
        stdAccuracy: accuracies.length > 1 ? Math.sqrt(
          accuracies.reduce((s, v) => s + (v - accuracies.reduce((ss, vv) => ss + vv, 0) / accuracies.length) ** 2, 0) / (accuracies.length - 1)
        ) : 0,
        meanROI: rois.length > 0 ? rois.reduce((s, v) => s + v, 0) / rois.length : 0,
        stdROI: rois.length > 1 ? Math.sqrt(
          rois.reduce((s, v) => s + (v - rois.reduce((ss, vv) => ss + vv, 0) / rois.length) ** 2, 0) / (rois.length - 1)
        ) : 0,
      },
    };
  }


  // =========================================================================
  // PARAMETER SENSITIVITY ANALYSIS
  // For each key parameter, perturb by +/-5% and measure score degradation.
  // Low sensitivity = robust model. High sensitivity = fragile parameter.
  // =========================================================================

  function runSensitivityAnalysis(boxScores, historicalOdds) {
    const keyParams = [
      'GFT_DECAY_RATE', 'GFT_GRAVITY_STRENGTH', 'GFT_MIN_CLEARANCE',
      'GFT_CONVERGENCE_MAX_SPREAD',
      'BEQ_CREDIBLE_LEVEL', 'BEQ_MIN_EDGE',
      'ESI_MAX_ENTROPY', 'ESI_TREND_WEIGHT',
      'PBF_DECAY_RATE', 'PBF_MIN_LAMBDA', 'PBF_MIN_PROB_MARGIN',
      'GATE_MIN_GFT_SCORE', 'GATE_MIN_BEQ_EDGE', 'GATE_MIN_ESI_STABILITY',
      'GATE_MIN_IMAD_SCORE', 'GATE_MIN_HIT_RATE', 'GATE_MIN_COMBINED',
      'GATE_MIN_STREAK', 'GATE_MIN_ABVC', 'GATE_MIN_PBF_MARGIN',
      'SWTA_SHORT_DECAY', 'SWTA_MED_DECAY', 'SWTA_LONG_DECAY',
      'ASC_ARBITER_MIN_CONSENSUS',
      'DMCT_MIN_PROB', 'TRD_HOT_THRESHOLD', 'TRD_COLD_THRESHOLD',
    ];

    // Baseline run
    const baselineResult = runBacktest(boxScores, historicalOdds);
    const baselineAccuracy = baselineResult.stats.overall.accuracy;
    const baselineROI = baselineResult.stats.overall.roi;
    const baselineTotal = baselineResult.stats.overall.total;

    const sensitivities = {};

    for (const param of keyParams) {
      const originalValue = CONFIG[param];
      if (originalValue === undefined || originalValue === null) continue;

      // Perturb up by 5%
      CONFIG[param] = originalValue * 1.05;
      const upResult = runBacktest(boxScores, historicalOdds);
      const upAccuracy = upResult.stats.overall.accuracy;
      const upROI = upResult.stats.overall.roi;
      const upTotal = upResult.stats.overall.total;

      // Perturb down by 5%
      CONFIG[param] = originalValue * 0.95;
      const downResult = runBacktest(boxScores, historicalOdds);
      const downAccuracy = downResult.stats.overall.accuracy;
      const downROI = downResult.stats.overall.roi;
      const downTotal = downResult.stats.overall.total;

      // Restore
      CONFIG[param] = originalValue;

      // Sensitivity scores
      const accuracySensitivity = baselineAccuracy > 0
        ? (Math.abs(upAccuracy - baselineAccuracy) + Math.abs(downAccuracy - baselineAccuracy)) / (2 * baselineAccuracy)
        : 0;

      const roiSensitivity = baselineROI !== 0
        ? (Math.abs(upROI - baselineROI) + Math.abs(downROI - baselineROI)) / (2 * Math.abs(baselineROI))
        : 0;

      const volumeSensitivity = baselineTotal > 0
        ? (Math.abs(upTotal - baselineTotal) + Math.abs(downTotal - baselineTotal)) / (2 * baselineTotal)
        : 0;

      sensitivities[param] = {
        originalValue,
        accuracySensitivity: Math.round(accuracySensitivity * 10000) / 10000,
        roiSensitivity: Math.round(roiSensitivity * 10000) / 10000,
        volumeSensitivity: Math.round(volumeSensitivity * 10000) / 10000,
        overallSensitivity: Math.round((accuracySensitivity + roiSensitivity + volumeSensitivity) / 3 * 10000) / 10000,
        upAccuracy: Math.round(upAccuracy * 10000) / 10000,
        downAccuracy: Math.round(downAccuracy * 10000) / 10000,
        upROI: Math.round(upROI * 10000) / 10000,
        downROI: Math.round(downROI * 10000) / 10000,
        upTotal,
        downTotal,
      };
    }

    // Sort by overall sensitivity (most sensitive first)
    const sorted = Object.entries(sensitivities)
      .sort((a, b) => b[1].overallSensitivity - a[1].overallSensitivity);

    return {
      baseline: {
        accuracy: Math.round(baselineAccuracy * 10000) / 10000,
        roi: Math.round(baselineROI * 10000) / 10000,
        total: baselineTotal,
      },
      sensitivities: Object.fromEntries(sorted),
      mostSensitive: sorted.slice(0, 5).map(([k, v]) => ({
        param: k,
        sensitivity: v.overallSensitivity,
      })),
      leastSensitive: sorted.slice(-5).map(([k, v]) => ({
        param: k,
        sensitivity: v.overallSensitivity,
      })),
    };
  }

  // =========================================================================
  // GENERATE TODAY'S RECOMMENDATIONS
  // =========================================================================

  function generateTodayPicks(liveOdds) {
    if (!liveOdds || !liveOdds.playerProps) return null;

    const candidates = [];

    for (const [gameKey, gameData] of Object.entries(liveOdds.playerProps)) {
      const parts = gameKey.split('@');
      const away = parts[0];
      const home = parts[1];

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

      // Home Runs
      if (gameData.hrLines) {
        for (const [playerName, lines] of Object.entries(gameData.hrLines)) {
          const playerTeam = PlayerModel.getTeam(playerName);
          const isHome = playerTeam === home;
          const opponent = isHome ? away : home;
          const prop = findBestProp(playerName, 'home_runs', lines, opponent, isHome);
          if (prop) {
            prop.gameKey = gameKey;
            prop.gameDisplay = gameKey.replace('@', ' @ ');
            candidates.push(prop);
          }
        }
      }
    }

    console.log('[ULTRA-MLB v3] Evaluated ' + Object.keys(liveOdds.playerProps).length + ' games, found ' + candidates.length + ' candidates passing all 12 gates');

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
        engine: 'ultra-v3',
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
        swta: single.swta,
        asc: single.asc,
        dmct: single.dmct,
        trd: single.trd,
        psi: single.psi,
        hitRate: single.hitRate,
        edge: single.edge,
        ev: single.ev,
        avg: single.avg,
        floor: single.floor,
        streak: single.streak,
        avgAB: single.avgAB,
        bet: single.player + ' O' + single.line + ' ' + single.statLabel,
        hit: null,
        actual: null,
      });
    }

    for (const parlay of (recommendation.parlays || [])) {
      signals.push({
        date,
        betType: 'parlay',
        tier: 'FORTRESS',
        engine: 'ultra-v3',
        n_legs: parlay.numLegs,
        parlay_decimal: parlay.decimalOdds,
        parlay_american: parlay.odds,
        positive_odds: parlay.odds > 0,
        avgCascade: parlay.avgCascade,
        combinedHitRate: parlay.combinedHitRate,
        ev: parlay.ev,
        crossStatBonus: parlay.crossStatBonus || 0,
        uniqueStats: parlay.uniqueStats || 0,
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
          bet: l.player + ' O' + l.line + ' ' + l.statLabel,
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
    console.log('[ULTRA-MLB v3] Loaded optimized config:', configObj.score, 'score,', configObj.improvements, 'improvements');
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
    runCrossValidation,
    runSensitivityAnalysis,
    generateTodayPicks,
    formatSignalForStorage,
    evaluateProp,
    findBestProp,
    selectBetType,
    buildOptimalParlayV3,
    applyCAMPD,
    americanToDecimal,
    decimalToAmerican,
    formatOdds,
    impliedProbability,
    STAT_MAP,
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
    computeSWTA,
    computeASC,
    computeDMCT,
    computeTRD,
    computePSI,
    computeUltraSignalV3,
  };
})();
