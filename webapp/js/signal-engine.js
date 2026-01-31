// =============================================================================
// SIGNAL ENGINE - NBA Live Trading Signal Detection
// =============================================================================
// Combines the best validated strategies:
// - 4-tier momentum strategy (Elite/Strong/Standard/Wide)
// - Lead + aligned momentum + time window
// - Spread -5 + Moneyline per signal
// - Validated on 156 real ESPN NBA games
// =============================================================================

window.SignalEngine = (function() {

  // =========================================================================
  // STRATEGY TIERS WITH VALIDATED STATS
  // =========================================================================
  const STRATEGY_TIERS = {
    elite:    { minMom: 14, spreadWR: 97.6, mlWR: 97.7, spreadEV: 26, mlEV: 15, combinedEV: 41, color: '#f59e0b' },
    strong:   { minMom: 12, spreadWR: 88.0, mlWR: 92.0, spreadEV: 16, mlEV: 8,  combinedEV: 24, color: '#3b82f6' },
    standard: { minMom: 10, spreadWR: 82.4, mlWR: 94.1, spreadEV: 8,  mlEV: 11, combinedEV: 19, color: '#a855f7' },
    wide:     { minMom: 8,  spreadWR: 78.0, mlWR: 90.7, spreadEV: 3,  mlEV: 7,  combinedEV: 10, color: '#6b7280' },
  };

  const TIER_ORDER = ['elite', 'strong', 'standard', 'wide'];

  // =========================================================================
  // MARKET PROBABILITY MODEL (calibrated to NBA empirical benchmarks)
  // =========================================================================
  function normalCdf(x) {
    if (x > 6) return 0.9999;
    if (x < -6) return 0.0001;
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
    const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const sign = x >= 0 ? 1 : -1;
    const absX = Math.abs(x);
    const t = 1.0 / (1.0 + p * absX);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX / 2);
    return 0.5 * (1.0 + sign * y);
  }

  function estimateMarketWinProb(lead, minsRemaining) {
    if (lead <= 0 || minsRemaining <= 0) return 50;
    const mins = Math.max(minsRemaining, 0.5);
    const SIGMA = 2.6; // Calibrated volatility
    const z = lead / (SIGMA * Math.sqrt(mins));
    const prob = normalCdf(z);
    return Math.max(51, Math.min(99.8, prob * 100));
  }

  function estimateLiveOdds(lead, minsRemaining, vigPct = 4.5) {
    const prob = estimateMarketWinProb(lead, minsRemaining) / 100;
    const marketProb = Math.min(0.995, prob + (vigPct / 100) * prob);
    if (marketProb >= 0.5) {
      return Math.round(-(marketProb / (1 - marketProb)) * 100);
    }
    return Math.round(((1 - marketProb) / marketProb) * 100);
  }

  function payoutFromOdds(odds) {
    if (odds < 0) return 100 / Math.abs(odds);
    return odds / 100;
  }

  // =========================================================================
  // CALCULATE MINUTES REMAINING
  // =========================================================================
  function calculateMinsRemaining(quarter, quarterTime) {
    const parts = quarterTime.split(':');
    const minutes = parseInt(parts[0]) || 0;
    const seconds = parseInt(parts[1]) || 0;
    const timeInQuarter = minutes + seconds / 60;
    const quartersRemaining = 4 - quarter;
    return quartersRemaining * 12 + timeInQuarter;
  }

  // =========================================================================
  // CALCULATE 5-MIN MOMENTUM
  // =========================================================================
  function calculateMomentum5Min(possessions, currentIndex) {
    const currentPos = possessions[currentIndex];
    const currentTimestamp = currentPos.timestamp;
    const fiveMinutesAgo = currentTimestamp - 300;

    let startIdx = 0;
    for (let i = 0; i < currentIndex; i++) {
      if (possessions[i].timestamp <= fiveMinutesAgo) {
        startIdx = i;
      } else {
        break;
      }
    }

    if (startIdx >= currentIndex) return { homePts: 0, awayPts: 0, momentum: 0 };

    const startPos = possessions[startIdx];
    const endPos = possessions[currentIndex];

    const homePts = endPos.homeScore - startPos.homeScore;
    const awayPts = endPos.awayScore - startPos.awayScore;

    return { homePts, awayPts, momentum: homePts - awayPts };
  }

  // =========================================================================
  // DETECT SIGNAL AT CURRENT STATE
  // =========================================================================
  function detectSignal(possessions, currentIndex) {
    if (!possessions || possessions.length < 25 || currentIndex < 25) return null;

    const pos = possessions[currentIndex];
    const { quarter, quarterTime, homeScore, awayScore } = pos;
    const minsRemaining = calculateMinsRemaining(quarter, quarterTime);

    // Time window: 12-24 minutes remaining
    if (minsRemaining < 12 || minsRemaining > 24) return null;

    const scoreDiff = homeScore - awayScore;
    const lead = Math.abs(scoreDiff);

    // Minimum lead of 10
    if (lead < 10) return null;

    // Determine leading team
    let leadingTeam;
    if (scoreDiff > 0) {
      leadingTeam = 'home';
    } else if (scoreDiff < 0) {
      leadingTeam = 'away';
    } else {
      return null; // Tie
    }

    // Calculate 5-min momentum
    const { homePts, awayPts, momentum } = calculateMomentum5Min(possessions, currentIndex);
    const mom = Math.abs(momentum);

    // Momentum must align with lead
    if (scoreDiff > 0 && momentum <= 0) return null; // Home leads but away has momentum
    if (scoreDiff < 0 && momentum >= 0) return null; // Away leads but home has momentum

    // Minimum momentum of 8
    if (mom < 8) return null;

    // Determine tier (highest matching)
    let tier = null;
    for (const t of TIER_ORDER) {
      if (mom >= STRATEGY_TIERS[t].minMom) {
        tier = t;
        break;
      }
    }

    if (!tier) return null;

    const stats = STRATEGY_TIERS[tier];
    const marketProb = estimateMarketWinProb(lead, minsRemaining);
    const liveOdds = estimateLiveOdds(lead, minsRemaining);
    const mlPayout = payoutFromOdds(liveOdds);

    // Get team abbreviations
    const homeAbbr = pos.homeTeam || 'HOME';
    const awayAbbr = pos.awayTeam || 'AWAY';
    const betTeamAbbr = leadingTeam === 'home' ? homeAbbr : awayAbbr;

    return {
      tier,
      leadingTeam,
      betTeamAbbr,
      homeAbbr,
      awayAbbr,
      lead,
      momentum: mom,
      minsRemaining: Math.round(minsRemaining * 10) / 10,
      quarter,
      quarterTime,
      homeScore,
      awayScore,
      spreadBet: -5,
      marketProb: Math.round(marketProb * 10) / 10,
      modelProb: stats.mlWR,
      edge: Math.round((stats.mlWR - marketProb) * 10) / 10,
      liveOdds,
      mlPayout: Math.round(mlPayout * 1000) / 1000,
      spreadWR: stats.spreadWR,
      mlWR: stats.mlWR,
      spreadEV: stats.spreadEV,
      mlEV: stats.mlEV,
      combinedEV: stats.combinedEV,
      color: stats.color,
      timestamp: Date.now(),
      possessionIndex: currentIndex,
    };
  }

  // =========================================================================
  // RUN SIGNAL DETECTION ON FULL GAME (for backtest/historical)
  // =========================================================================
  function runOnGame(possessions, homeAbbr, awayAbbr, finalHomeScore, finalAwayScore) {
    const signals = [];
    const signalledIndices = new Set();

    if (!possessions || possessions.length < 25) return signals;

    // Add team info to possessions for display
    possessions.forEach(p => {
      p.homeTeam = homeAbbr;
      p.awayTeam = awayAbbr;
    });

    for (let i = 25; i < possessions.length; i++) {
      // Avoid signals too close together (min 10 possessions apart)
      let tooClose = false;
      for (const idx of signalledIndices) {
        if (Math.abs(i - idx) < 10) {
          tooClose = true;
          break;
        }
      }
      if (tooClose) continue;

      const signal = detectSignal(possessions, i);
      if (signal) {
        // Grade the signal
        const finalDiff = finalHomeScore - finalAwayScore;

        // ML outcome: did the team win?
        let mlOutcome;
        if (signal.leadingTeam === 'home') {
          mlOutcome = finalDiff > 0 ? 'win' : 'loss';
        } else {
          mlOutcome = finalDiff < 0 ? 'win' : 'loss';
        }

        // Spread outcome: did the team win by more than 5?
        const finalMargin = signal.leadingTeam === 'home' ? finalDiff : -finalDiff;
        let spreadOutcome;
        if (finalMargin > 5) spreadOutcome = 'win';
        else if (finalMargin === 5) spreadOutcome = 'push';
        else spreadOutcome = 'loss';

        signal.mlOutcome = mlOutcome;
        signal.spreadOutcome = spreadOutcome;
        signal.finalHomeScore = finalHomeScore;
        signal.finalAwayScore = finalAwayScore;

        // Calculate P&L
        const spreadPnl = spreadOutcome === 'win' ? (100/110) : spreadOutcome === 'push' ? 0 : -1;
        const mlPnl = mlOutcome === 'win' ? signal.mlPayout : -1;
        signal.spreadPnl = Math.round(spreadPnl * 100) / 100;
        signal.mlPnl = Math.round(mlPnl * 100) / 100;
        signal.totalPnl = Math.round((spreadPnl + mlPnl) * 100) / 100;

        signals.push(signal);
        signalledIndices.add(i);

        // Only one signal per game for live trading
        break;
      }
    }

    return signals;
  }

  // =========================================================================
  // LIVE SCAN: Check current game state for signal
  // =========================================================================
  function scanLiveGame(possessions, homeAbbr, awayAbbr) {
    if (!possessions || possessions.length < 25) return null;

    // Add team info
    possessions.forEach(p => {
      p.homeTeam = homeAbbr;
      p.awayTeam = awayAbbr;
    });

    // Check the most recent possession
    const signal = detectSignal(possessions, possessions.length - 1);
    return signal;
  }

  // =========================================================================
  // GENERATE TRADE INSTRUCTION
  // =========================================================================
  function getTradeInstruction(signal) {
    if (!signal) return null;

    const team = signal.betTeamAbbr;
    const tier = signal.tier.toUpperCase();

    return {
      headline: `${tier} SIGNAL: Bet ${team}`,
      spread: `Take ${team} ${signal.spreadBet} SPREAD (${signal.spreadWR}% WR, +$${signal.spreadEV} EV)`,
      moneyline: `Take ${team} MONEYLINE at ~${signal.liveOdds} (${signal.mlWR}% WR, +$${signal.mlEV} EV)`,
      combined: `Combined EV: +$${signal.combinedEV} per $100 each bet`,
      context: `Lead: ${signal.lead} | Mom: ${signal.momentum} | ${signal.minsRemaining} min left | Q${signal.quarter} ${signal.quarterTime}`,
      urgency: signal.tier === 'elite' ? 'ACT NOW - Elite signal, highest confidence' :
               signal.tier === 'strong' ? 'High confidence - Strong signal detected' :
               signal.tier === 'standard' ? 'Good setup - Standard signal' :
               'Moderate setup - Consider position size',
    };
  }

  // =========================================================================
  // PUBLIC API
  // =========================================================================
  return {
    STRATEGY_TIERS,
    TIER_ORDER,
    detectSignal,
    runOnGame,
    scanLiveGame,
    getTradeInstruction,
    calculateMinsRemaining,
    calculateMomentum5Min,
    estimateMarketWinProb,
    estimateLiveOdds,
    normalCdf,
  };

})();
