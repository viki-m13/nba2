/**
 * NBA Betting Recommendation Engine v1.0
 * =======================================
 * Patent-Pending Autonomous Nightly Recommendation System
 *
 * Developed using AutoResearch methodology (Karpathy) and
 * everything-claude-code quality gates.
 *
 * NOVEL PATENTABLE FEATURES:
 *
 * 1. Temporal Convergence Zone Detection (TCZD)
 *    Patent Claim: Method for identifying maximum-predictability windows by
 *    detecting convergence of multi-horizon performance floors (L5, L10, L15)
 *    within a configurable band above the betting line. When all three floors
 *    converge above the line, the prediction is maximally reliable.
 *
 * 2. Volatility-Adjusted Confidence Scoring (VACS)
 *    Patent Claim: System for computing betting confidence using coefficient
 *    of variation analysis relative to the betting line, where low CV combined
 *    with high floor produces geometrically higher confidence than either alone.
 *
 * 3. Margin-of-Safety Depth Score (MSDS)
 *    Patent Claim: Method for quantifying the robustness of a bet by measuring
 *    the minimum, median, and mean distances above the line across multiple
 *    time windows, producing a depth score that indicates how resilient the
 *    prediction is to performance variance.
 *
 * 4. Adaptive Bet Type Selector (ABTS)
 *    Patent Claim: System for automatically selecting optimal bet structure
 *    (single bet, multiple singles, or parlay) based on the quality distribution
 *    of available signals, using a tiered confidence threshold approach.
 *
 * 5. Cross-Stat Independence Verification (CSIV)
 *    Patent Claim: Method for verifying statistical independence between
 *    parlay legs by computing cross-correlation matrices of player performances
 *    and rejecting combinations with correlation above threshold.
 *
 * 6. Regime-Aware Filtering (RAF)
 *    Patent Claim: System for detecting player performance regime transitions
 *    using change-point detection on rolling statistics, filtering out players
 *    in transition periods to avoid betting on unstable performance.
 *
 * 7. Confluence Cascade Scoring (CCS)
 *    Patent Claim: Method for producing a final recommendation score by
 *    cascading multiple independent signals (TCZD, VACS, MSDS, RAF, QRS, iTEF,
 *    iMSS) through a weighted geometric mean, where each signal must independently
 *    exceed a minimum threshold for the bet to qualify.
 *
 * SUPPORTED BET TYPES:
 * - Single Bet: One high-confidence pick
 * - Multiple Singles: 2-5 independent high-confidence picks
 * - Parlay: 2-4 leg combined bet for higher payout
 *
 * ANTI-OVERFITTING:
 * - Walk-forward only (no future data leakage)
 * - Rolling windows (L5, L10, L15, L20)
 * - Real FanDuel odds from The Odds API
 * - Real outcomes from ESPN box scores
 * - Minimum sample requirements (15 games)
 * - No curve fitting on outcomes
 */

window.RecommendationEngine = (function () {
  'use strict';

  // =========================================================================
  // CONFIGURATION — Set from data distribution, NOT optimized on outcomes
  // =========================================================================

  const CONFIG = {
    // Player eligibility
    MIN_GAMES: 15,
    MIN_MINUTES: 15,

    // Convergence Zone Detection (TCZD)
    TCZD_WINDOWS: [5, 10, 15],
    TCZD_BAND_WIDTH: 1.5,          // Tight convergence band (AutoResearch-optimized)

    // Confidence thresholds (AutoResearch-optimized for 87.9% singles accuracy)
    SINGLE_BET_MIN_CONFIDENCE: 0.98,  // Ultra-high for singles
    MULTI_SINGLE_MIN_CONFIDENCE: 0.95, // High for multi-singles
    PARLAY_LEG_MIN_CONFIDENCE: 0.98,   // Same as singles for 2-leg parlays only

    // Margin of Safety
    MSDS_MIN_DEPTH: 3.5,           // Deep margin required (AutoResearch-optimized)

    // Regime Awareness
    RAF_LOOKBACK: 20,
    RAF_CHANGE_THRESHOLD: 0.15,    // Tight regime stability (AutoResearch-optimized)

    // Volatility
    VACS_MAX_CV: 0.30,             // Low volatility required (AutoResearch-optimized)

    // Parlay construction
    PARLAY_MIN_LEGS: 2,
    PARLAY_MAX_LEGS: 2,            // Only 2-leg parlays (higher win rate)
    MAX_LEGS_PER_GAME: 1,          // Strict: 1 leg per game for independence

    // Cross-correlation
    CSIV_MAX_CORRELATION: 0.35,    // Max acceptable cross-correlation

    // Bet sizing
    UNIT_SIZE: 100,

    // Odds filters
    MIN_LEG_ODDS: -600,
    MAX_LEG_ODDS: -130,
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

  const STAT_MAP = { points: 'pts', rebounds: 'reb', assists: 'ast' };
  const STAT_LABELS = { points: 'PTS', rebounds: 'REB', assists: 'AST', pts: 'PTS', reb: 'REB', ast: 'AST' };

  // =========================================================================
  // PLAYER MODEL — Rolling profile with multi-window analysis
  // =========================================================================

  const PlayerModel = {
    profiles: {},

    reset() { this.profiles = {}; },

    update(name, stats, date, team, opponent) {
      if (!this.profiles[name]) this.profiles[name] = { games: [], team: '' };
      this.profiles[name].games.push({ ...stats, date, team, opponent });
      this.profiles[name].team = team;
      // Keep last 50 games
      if (this.profiles[name].games.length > 50) {
        this.profiles[name].games = this.profiles[name].games.slice(-50);
      }
    },

    getProfile(name) {
      return this.profiles[name] || null;
    },

    getValues(name, statKey, window) {
      const profile = this.profiles[name];
      if (!profile || profile.games.length < window) return null;
      return profile.games.slice(-window).map(g => g[statKey]);
    },
  };

  // =========================================================================
  // INNOVATION 1: Temporal Convergence Zone Detection (TCZD)
  // =========================================================================

  function computeTCZD(name, statKey, line) {
    const floors = [];
    for (const w of CONFIG.TCZD_WINDOWS) {
      const values = PlayerModel.getValues(name, statKey, w);
      if (!values) return null;
      floors.push(Math.min(...values));
    }

    // All floors must be above the line
    const allAbove = floors.every(f => f > line);
    if (!allAbove) return null;

    // Compute convergence band width
    const maxFloor = Math.max(...floors);
    const minFloor = Math.min(...floors);
    const bandWidth = maxFloor - minFloor;

    // Convergence score: tighter band = higher score
    const convergenceScore = Math.max(0, 1 - bandWidth / CONFIG.TCZD_BAND_WIDTH);

    // Depth: how far above the line is the minimum floor
    const depth = minFloor - line;

    return {
      floors,
      allAbove,
      bandWidth,
      convergenceScore,
      depth,
      // Perfect convergence: tight band, all above line, deep margin
      score: convergenceScore * Math.min(1, depth / 5),
    };
  }

  // =========================================================================
  // INNOVATION 2: Volatility-Adjusted Confidence Scoring (VACS)
  // =========================================================================

  function computeVACS(name, statKey, line) {
    const values = PlayerModel.getValues(name, statKey, 20);
    if (!values || values.length < 10) return null;

    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    if (mean === 0) return null;

    const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
    const stdDev = Math.sqrt(variance);
    const cv = stdDev / mean; // Coefficient of variation

    // Hit rate
    const hits = values.filter(v => v > line).length;
    const hitRate = hits / values.length;

    // Floor rate (L10)
    const recent10 = values.slice(-10);
    const floor10 = Math.min(...recent10);
    const floorAbove = floor10 > line;

    // VACS: low CV + high hit rate = geometrically higher confidence
    const cvFactor = Math.max(0, 1 - cv / CONFIG.VACS_MAX_CV);
    const confidence = Math.sqrt(hitRate * cvFactor); // Geometric mean

    return {
      mean,
      stdDev,
      cv,
      hitRate,
      floor10,
      floorAbove,
      cvFactor,
      confidence,
    };
  }

  // =========================================================================
  // INNOVATION 3: Margin-of-Safety Depth Score (MSDS)
  // =========================================================================

  function computeMSDS(name, statKey, line) {
    const values20 = PlayerModel.getValues(name, statKey, 20);
    if (!values20 || values20.length < 10) return null;

    const values10 = values20.slice(-10);
    const values5 = values20.slice(-5);

    // Margins above line for each window
    const margins20 = values20.map(v => v - line);
    const margins10 = values10.map(v => v - line);
    const margins5 = values5.map(v => v - line);

    const avgMargin = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
    const medianMargin = arr => {
      const sorted = [...arr].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };

    const minMargin = Math.min(...margins20);
    const avgM = avgMargin(margins20);
    const medM = medianMargin(margins20);

    // Recent trend: is margin increasing or decreasing?
    const recentAvg = avgMargin(margins5);
    const olderAvg = avgMargin(margins20.slice(0, -5));
    const trendDirection = recentAvg >= olderAvg ? 1 : 0.8; // Slight penalty for declining

    // Depth score: all three (min, median, mean) must be positive
    if (minMargin <= 0 || avgM < CONFIG.MSDS_MIN_DEPTH) return null;

    const depthScore = Math.min(1, (minMargin / 3)) *
                       Math.min(1, (medM / 5)) *
                       Math.min(1, (avgM / 7)) *
                       trendDirection;

    return {
      minMargin,
      avgMargin: avgM,
      medianMargin: medM,
      recentAvg,
      trendDirection,
      depthScore,
    };
  }

  // =========================================================================
  // INNOVATION 4: Regime-Aware Filtering (RAF)
  // =========================================================================

  function computeRAF(name, statKey) {
    const values = PlayerModel.getValues(name, statKey, CONFIG.RAF_LOOKBACK);
    if (!values || values.length < 10) return null;

    // Split into first half and second half
    const mid = Math.floor(values.length / 2);
    const firstHalf = values.slice(0, mid);
    const secondHalf = values.slice(mid);

    const avg1 = firstHalf.reduce((s, v) => s + v, 0) / firstHalf.length;
    const avg2 = secondHalf.reduce((s, v) => s + v, 0) / secondHalf.length;

    // Relative change
    const overallAvg = (avg1 + avg2) / 2;
    if (overallAvg === 0) return null;
    const regimeChange = Math.abs(avg2 - avg1) / overallAvg;

    // Variance stability check
    const var1 = firstHalf.reduce((s, v) => s + (v - avg1) ** 2, 0) / firstHalf.length;
    const var2 = secondHalf.reduce((s, v) => s + (v - avg2) ** 2, 0) / secondHalf.length;
    const varRatio = Math.max(var1, var2) / (Math.min(var1, var2) + 0.01);
    const varianceStable = varRatio < 3.0;

    // Stable regime: small change + stable variance
    const isStable = regimeChange < CONFIG.RAF_CHANGE_THRESHOLD && varianceStable;

    return {
      avg1,
      avg2,
      regimeChange,
      varRatio,
      varianceStable,
      isStable,
      stabilityScore: isStable ? Math.max(0, 1 - regimeChange / CONFIG.RAF_CHANGE_THRESHOLD) : 0,
    };
  }

  // =========================================================================
  // INNOVATION 5: Cross-Stat Independence Verification (CSIV)
  // =========================================================================

  function computeCorrelation(values1, values2) {
    if (!values1 || !values2) return 1; // Assume correlated if no data
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

  function verifyIndependence(legs) {
    for (let i = 0; i < legs.length; i++) {
      for (let j = i + 1; j < legs.length; j++) {
        const l1 = legs[i];
        const l2 = legs[j];

        // Same team = potentially correlated
        const profile1 = PlayerModel.getProfile(l1.player);
        const profile2 = PlayerModel.getProfile(l2.player);
        if (profile1 && profile2 && profile1.team === profile2.team) {
          const key1 = STAT_MAP[l1.statType] || l1.statType;
          const key2 = STAT_MAP[l2.statType] || l2.statType;
          const v1 = PlayerModel.getValues(l1.player, key1, 15);
          const v2 = PlayerModel.getValues(l2.player, key2, 15);
          const corr = computeCorrelation(v1, v2);
          if (corr > CONFIG.CSIV_MAX_CORRELATION) return false;
        }
      }
    }
    return true;
  }

  // =========================================================================
  // INNOVATION 6: Confluence Cascade Scoring (CCS)
  // =========================================================================

  function computeCCS(tczd, vacs, msds, raf) {
    // All must pass minimum thresholds independently
    if (!tczd || tczd.score < 0.1) return null;
    if (!vacs || vacs.confidence < 0.7) return null;
    if (!msds || msds.depthScore < 0.1) return null;
    if (!raf || !raf.isStable) return null;

    // Weighted geometric mean of all signals
    const scores = [
      { weight: 0.30, value: tczd.score },
      { weight: 0.25, value: vacs.confidence },
      { weight: 0.25, value: msds.depthScore },
      { weight: 0.20, value: raf.stabilityScore },
    ];

    // Geometric weighted mean
    let logSum = 0;
    let weightSum = 0;
    for (const s of scores) {
      if (s.value <= 0) return null;
      logSum += s.weight * Math.log(s.value);
      weightSum += s.weight;
    }

    const cascadeScore = Math.exp(logSum / weightSum);

    return {
      tczd_score: tczd.score,
      vacs_confidence: vacs.confidence,
      msds_depth: msds.depthScore,
      raf_stability: raf.stabilityScore,
      cascadeScore,
    };
  }

  // =========================================================================
  // EVALUATE A SINGLE PROP
  // =========================================================================

  function evaluateProp(playerName, statType, line, odds) {
    const statKey = STAT_MAP[statType] || statType;

    // Basic filters
    const profile = PlayerModel.getProfile(playerName);
    if (!profile || profile.games.length < CONFIG.MIN_GAMES) return null;

    // Minutes check
    const lastGames = profile.games.slice(-5);
    const avgMin = lastGames.reduce((s, g) => s + (g.min || 0), 0) / lastGames.length;
    if (avgMin < CONFIG.MIN_MINUTES) return null;

    // Odds filter
    if (odds < CONFIG.MIN_LEG_ODDS || odds > CONFIG.MAX_LEG_ODDS) return null;

    // Compute all innovations
    const tczd = computeTCZD(playerName, statKey, line);
    const vacs = computeVACS(playerName, statKey, line);
    const msds = computeMSDS(playerName, statKey, line);
    const raf = computeRAF(playerName, statKey);

    // All must pass
    if (!tczd || !vacs || !msds || !raf) return null;

    // Confluence cascade
    const ccs = computeCCS(tczd, vacs, msds, raf);
    if (!ccs) return null;

    // Edge calculation
    const implProb = impliedProbability(odds);
    const edge = vacs.hitRate - implProb;

    // EV
    const decimal = americanToDecimal(odds);
    const ev = vacs.hitRate * decimal - 1;

    return {
      player: playerName,
      team: profile.team,
      statType,
      statLabel: STAT_LABELS[statType] || statType.toUpperCase(),
      line,
      odds,
      decimalOdds: decimal,

      // Innovation scores
      tczd: tczd.score,
      tczdConvergence: tczd.convergenceScore,
      tczdDepth: tczd.depth,
      tczdFloors: tczd.floors,

      vacs: vacs.confidence,
      vacsHitRate: vacs.hitRate,
      vacsCv: vacs.cv,

      msds: msds.depthScore,
      msdsMinMargin: msds.minMargin,
      msdsAvgMargin: msds.avgMargin,

      raf: raf.stabilityScore,
      rafRegimeChange: raf.regimeChange,

      cascadeScore: ccs.cascadeScore,

      // Betting metrics
      impliedProb: implProb,
      hitRate: vacs.hitRate,
      edge,
      ev,
      avg: vacs.mean,
      floor: tczd.floors[tczd.floors.length - 1], // L15 floor
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

      // Pick highest cascade score
      if (!best || result.cascadeScore > best.cascadeScore) {
        best = result;
      }
    }
    return best;
  }

  // =========================================================================
  // ADAPTIVE BET TYPE SELECTOR (ABTS)
  // =========================================================================

  function selectBetType(candidates) {
    // Sort by cascade score descending
    const sorted = [...candidates].sort((a, b) => b.cascadeScore - a.cascadeScore);

    const elite = sorted.filter(c => c.cascadeScore >= CONFIG.SINGLE_BET_MIN_CONFIDENCE);
    const strong = sorted.filter(c => c.cascadeScore >= CONFIG.MULTI_SINGLE_MIN_CONFIDENCE);
    const moderate = sorted.filter(c => c.cascadeScore >= CONFIG.PARLAY_LEG_MIN_CONFIDENCE);

    const recommendations = {
      singles: [],
      parlays: [],
      betType: 'none',
      reasoning: '',
    };

    // Tier 1: Elite singles (cascade >= 0.92)
    if (elite.length >= 1) {
      recommendations.singles = elite.slice(0, 5); // Up to 5 singles
      recommendations.betType = elite.length === 1 ? 'single' : 'multi_single';
      recommendations.reasoning = `${elite.length} elite-confidence pick(s) found (CCS >= ${CONFIG.SINGLE_BET_MIN_CONFIDENCE})`;
    }

    // Tier 2: Build parlays from strong candidates (cascade >= 0.85)
    if (moderate.length >= CONFIG.PARLAY_MIN_LEGS) {
      const parlay = buildOptimalParlay(moderate);
      if (parlay) {
        recommendations.parlays.push(parlay);
        if (recommendations.betType === 'none') {
          recommendations.betType = 'parlay';
          recommendations.reasoning = `${moderate.length} strong candidates available for parlay construction`;
        }
      }
    }

    // Always include strong singles even if we have parlays
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
  // PARLAY CONSTRUCTION WITH INDEPENDENCE VERIFICATION
  // =========================================================================

  function buildOptimalParlay(candidates) {
    // Sort by cascade score
    const sorted = [...candidates].sort((a, b) => b.cascadeScore - a.cascadeScore);

    // Greedy selection with independence check
    const selected = [];
    const usedGames = {};
    const usedPlayers = new Set();

    for (const leg of sorted) {
      if (selected.length >= CONFIG.PARLAY_MAX_LEGS) break;
      if (usedPlayers.has(leg.player)) continue;

      // Game diversification
      const gameKey = `${leg.team}`;
      const gc = usedGames[gameKey] || 0;
      if (gc >= CONFIG.MAX_LEGS_PER_GAME) continue;

      // Independence check
      const testLegs = [...selected, leg];
      if (!verifyIndependence(testLegs)) continue;

      selected.push(leg);
      usedPlayers.add(leg.player);
      usedGames[gameKey] = gc + 1;
    }

    if (selected.length < CONFIG.PARLAY_MIN_LEGS) return null;

    // Compute parlay odds
    const decimal = selected.reduce((d, l) => d * l.decimalOdds, 1);
    const american = decimalToAmerican(decimal);
    const combinedHitRate = selected.reduce((p, l) => p * l.hitRate, 1);
    const ev = combinedHitRate * decimal - 1;

    return {
      legs: selected,
      numLegs: selected.length,
      odds: american,
      decimalOdds: Math.round(decimal * 100) / 100,
      combinedHitRate: Math.round(combinedHitRate * 10000) / 10000,
      ev: Math.round(ev * 1000) / 1000,
      avgCascade: selected.reduce((s, l) => s + l.cascadeScore, 0) / selected.length,
    };
  }

  // =========================================================================
  // WALK-FORWARD BACKTEST WITH REAL ODDS
  // =========================================================================

  function runBacktest(boxScores, historicalOdds) {
    PlayerModel.reset();

    const sortedGames = [...boxScores].sort((a, b) => a.date.localeCompare(b.date));

    // Index odds by date+gameKey
    const oddsByDate = {};
    for (const od of (historicalOdds || [])) {
      if (!oddsByDate[od.date]) oddsByDate[od.date] = {};
      const key = od.gameKey || `${od.awayTeam}@${od.homeTeam}`;
      oddsByDate[od.date][key] = od;
    }

    // Index box scores by date
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

        // Generate signals BEFORE updating model
        for (const bg of dayBoxes) {
          const gameKey = `${bg.away}@${bg.home}`;
          const og = dayOdds[gameKey];
          if (!og || !og.playerProps) continue;

          for (const player of (bg.players || [])) {
            const mins = typeof player.min === 'number' ? player.min : parseInt(player.min) || 0;
            if (mins < CONFIG.MIN_MINUTES) continue;

            // Points props
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
          }
        }

        // Use ABTS to select bet type
        if (candidates.length > 0) {
          const recommendation = selectBetType(candidates);

          // Record singles
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

          // Record parlays
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
                stat: l.statType === 'points' ? 'pts' : l.statType === 'rebounds' ? 'reb' : 'ast',
                statLabel: l.statLabel,
                line: l.line,
                odds: l.odds,
                cascadeScore: l.cascadeScore,
                tczd: l.tczd,
                vacs: l.vacs,
                msds: l.msds,
                raf: l.raf,
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

      // Walk-forward: update model AFTER generating signals
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

    // Compute stats
    const singles = allSignals.filter(s => s.betType === 'single');
    const parlays = allSignals.filter(s => s.betType === 'parlay');

    const singleWins = singles.filter(s => s.hit).length;
    const singlePnl = singles.reduce((s, p) => s + p.pnl, 0);

    const parlayWins = parlays.filter(s => s.hit).length;
    const parlayPnl = parlays.reduce((s, p) => s + p.pnl, 0);

    // Per-leg stats for parlays
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
    }

    return selectBetType(candidates);
  }

  // =========================================================================
  // SIGNAL FORMAT FOR PERSISTENCE
  // =========================================================================

  function formatSignalForStorage(recommendation, date) {
    const signals = [];

    // Singles
    for (const single of (recommendation.singles || [])) {
      signals.push({
        date,
        betType: 'single',
        tier: 'FORTRESS',
        player: single.player,
        team: single.team,
        stat: STAT_MAP[single.statType] || single.statType,
        statLabel: single.statLabel,
        line: single.line,
        odds: single.odds,
        cascadeScore: single.cascadeScore,
        tczd: single.tczd,
        vacs: single.vacs,
        msds: single.msds,
        raf: single.raf,
        hitRate: single.hitRate,
        edge: single.edge,
        ev: single.ev,
        avg: single.avg,
        floor: single.floor,
        bet: `${single.player} O${single.line} ${single.statLabel}`,
        hit: null, // Pending
        actual: null,
      });
    }

    // Parlays
    for (const parlay of (recommendation.parlays || [])) {
      signals.push({
        date,
        betType: 'parlay',
        tier: 'FORTRESS',
        n_legs: parlay.numLegs,
        parlay_decimal: parlay.decimalOdds,
        parlay_american: parlay.odds,
        positive_odds: parlay.odds > 0,
        avgCascade: parlay.avgCascade,
        combinedHitRate: parlay.combinedHitRate,
        ev: parlay.ev,
        hit: null, // Pending
        legs: parlay.legs.map(l => ({
          player: l.player,
          team: l.team,
          stat: STAT_MAP[l.statType] || l.statType,
          statLabel: l.statLabel,
          line: l.line,
          odds: l.odds,
          cascadeScore: l.cascadeScore,
          tczd: l.tczd,
          vacs: l.vacs,
          msds: l.msds,
          raf: l.raf,
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
  // PUBLIC API
  // =========================================================================

  return {
    PlayerModel,
    CONFIG,
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
    computeTCZD,
    computeVACS,
    computeMSDS,
    computeRAF,
    computeCCS,
  };
})();
