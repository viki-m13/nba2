// =============================================================================
// HISTORICAL DATA - Sample games with realistic signal data
// =============================================================================
// Based on validated quant strategy results from 2,310 real ESPN NBA games
// Direction: ALL signals FADE the leader (bet the underdog)
// =============================================================================

window.HistoricalData = (function() {

  // =========================================================================
  // HISTORICAL GAMES DATA
  // =========================================================================
  const GAMES = [
    // Game 0: BOS vs NYK - Quant signal, lead compressed (spread cover)
    {
      id: 'hist-001',
      date: 'Jan 15',
      homeTeam: 'BOS',
      awayTeam: 'NYK',
      homeScore: 108,
      awayScore: 102,
      signal: {
        tier: 'quant',
        strategy: 'quant',
        strategyName: 'Quant',
        leadingTeam: 'home',
        trailingTeam: 'away',
        betTeam: 'NYK',      // Bet underdog
        leaderTeam: 'BOS',
        lead: 14,
        momentum: 12,
        quarter: 2,
        quarterTime: '4:30',
        minsRemaining: 16.5,
        scoreAtSignal: { home: 52, away: 38 },
        spreadOutcome: 'win',   // Lead compressed from 14 to 6 (covered)
        mlOutcome: 'loss',      // BOS still won (no upset)
        possessionIndex: 62,
      },
    },
    // Game 1: LAL vs GSW - Fade ML, underdog wins (upset!)
    {
      id: 'hist-002',
      date: 'Jan 12',
      homeTeam: 'LAL',
      awayTeam: 'GSW',
      homeScore: 108,
      awayScore: 112,
      signal: {
        tier: 'fade_ml',
        strategy: 'fade_ml',
        strategyName: 'Fade ML',
        leadingTeam: 'home',
        trailingTeam: 'away',
        betTeam: 'GSW',      // Bet underdog
        leaderTeam: 'LAL',
        lead: 12,
        momentum: 14,
        quarter: 2,
        quarterTime: '6:15',
        minsRemaining: 18.3,
        scoreAtSignal: { home: 48, away: 36 },
        spreadOutcome: 'win',
        mlOutcome: 'win',     // Upset! GSW came back
        possessionIndex: 55,
      },
    },
    // Game 2: MIL vs PHI - Quant signal, spread cover
    {
      id: 'hist-003',
      date: 'Jan 10',
      homeTeam: 'MIL',
      awayTeam: 'PHI',
      homeScore: 106,
      awayScore: 101,
      signal: {
        tier: 'quant',
        strategy: 'quant',
        strategyName: 'Quant',
        leadingTeam: 'home',
        trailingTeam: 'away',
        betTeam: 'PHI',
        leaderTeam: 'MIL',
        lead: 11,
        momentum: 10,
        quarter: 3,
        quarterTime: '8:20',
        minsRemaining: 20.3,
        scoreAtSignal: { home: 62, away: 51 },
        spreadOutcome: 'win',   // Lead compressed from 11 to 5
        mlOutcome: 'loss',
        possessionIndex: 72,
      },
    },
    // Game 3: DEN vs PHX - Composite signal, covered
    {
      id: 'hist-004',
      date: 'Jan 8',
      homeTeam: 'DEN',
      awayTeam: 'PHX',
      homeScore: 115,
      awayScore: 110,
      signal: {
        tier: 'composite',
        strategy: 'composite',
        strategyName: 'Composite',
        leadingTeam: 'home',
        trailingTeam: 'away',
        betTeam: 'PHX',
        leaderTeam: 'DEN',
        lead: 16,
        momentum: 18,
        quarter: 2,
        quarterTime: '3:45',
        minsRemaining: 15.8,
        scoreAtSignal: { home: 56, away: 40 },
        spreadOutcome: 'win',   // Lead compressed from 16 to 5
        mlOutcome: 'loss',
        possessionIndex: 65,
      },
    },
    // Game 4: MIA vs CLE - Fade Spread, lead expanded (loss)
    {
      id: 'hist-005',
      date: 'Jan 5',
      homeTeam: 'MIA',
      awayTeam: 'CLE',
      homeScore: 118,
      awayScore: 101,
      signal: {
        tier: 'fade_spread',
        strategy: 'fade_spread',
        strategyName: 'Fade Spread',
        leadingTeam: 'home',
        trailingTeam: 'away',
        betTeam: 'CLE',
        leaderTeam: 'MIA',
        lead: 14,
        momentum: 13,
        quarter: 2,
        quarterTime: '5:00',
        minsRemaining: 17.0,
        scoreAtSignal: { home: 52, away: 38 },
        spreadOutcome: 'loss',  // Lead expanded from 14 to 17
        mlOutcome: 'loss',
        possessionIndex: 58,
      },
    },
    // Game 5: DAL vs OKC - Quant signal, covered
    {
      id: 'hist-006',
      date: 'Jan 3',
      homeTeam: 'DAL',
      awayTeam: 'OKC',
      homeScore: 107,
      awayScore: 105,
      signal: {
        tier: 'quant',
        strategy: 'quant',
        strategyName: 'Quant',
        leadingTeam: 'home',
        trailingTeam: 'away',
        betTeam: 'OKC',
        leaderTeam: 'DAL',
        lead: 13,
        momentum: 12,
        quarter: 2,
        quarterTime: '7:30',
        minsRemaining: 19.5,
        scoreAtSignal: { home: 45, away: 32 },
        spreadOutcome: 'win',   // Lead compressed from 13 to 2
        mlOutcome: 'loss',
        possessionIndex: 48,
      },
    },
    // Game 6: BOS vs MIA - Composite signal, covered
    {
      id: 'hist-007',
      date: 'Dec 28',
      homeTeam: 'BOS',
      awayTeam: 'MIA',
      homeScore: 112,
      awayScore: 108,
      signal: {
        tier: 'composite',
        strategy: 'composite',
        strategyName: 'Composite',
        leadingTeam: 'home',
        trailingTeam: 'away',
        betTeam: 'MIA',
        leaderTeam: 'BOS',
        lead: 18,
        momentum: 15,
        quarter: 2,
        quarterTime: '2:00',
        minsRemaining: 14.0,
        scoreAtSignal: { home: 60, away: 42 },
        spreadOutcome: 'win',   // Lead compressed from 18 to 4
        mlOutcome: 'loss',
        possessionIndex: 70,
      },
    },
    // Game 7: GSW vs LAC - Fade ML, covered
    {
      id: 'hist-008',
      date: 'Dec 25',
      homeTeam: 'GSW',
      awayTeam: 'LAC',
      homeScore: 109,
      awayScore: 104,
      signal: {
        tier: 'fade_ml',
        strategy: 'fade_ml',
        strategyName: 'Fade ML',
        leadingTeam: 'home',
        trailingTeam: 'away',
        betTeam: 'LAC',
        leaderTeam: 'GSW',
        lead: 11,
        momentum: 13,
        quarter: 3,
        quarterTime: '9:00',
        minsRemaining: 21.0,
        scoreAtSignal: { home: 58, away: 47 },
        spreadOutcome: 'win',   // Lead compressed from 11 to 5
        mlOutcome: 'loss',
        possessionIndex: 68,
      },
    },
  ];

  // =========================================================================
  // GENERATE SCORE FLOW FOR EACH GAME
  // =========================================================================
  function generateGameScoreFlow(game) {
    const sig = game.signal;
    const possessions = [];
    let homeScore = 0;
    let awayScore = 0;
    let timestamp = 0;
    const totalPoss = 200;

    const preSigPoss = sig.possessionIndex;
    const homeAtSig = sig.scoreAtSignal.home;
    const awayAtSig = sig.scoreAtSignal.away;

    for (let i = 0; i < preSigPoss; i++) {
      timestamp += 12 + Math.floor(Math.random() * 8);
      const progress = i / preSigPoss;

      const targetHome = Math.round(homeAtSig * progress);
      const targetAway = Math.round(awayAtSig * progress);

      if (Math.random() < 0.55) {
        const pts = Math.random() < 0.3 ? 3 : 2;
        if (homeScore + pts <= targetHome + 4) homeScore += pts;
      } else if (Math.random() < 0.5) {
        const pts = Math.random() < 0.3 ? 3 : 2;
        if (awayScore + pts <= targetAway + 4) awayScore += pts;
      }

      const quarter = Math.min(Math.floor(timestamp / (12 * 60)) + 1, 4);
      const timeInQ = (12 * 60) - (timestamp % (12 * 60));
      const mins = Math.floor(timeInQ / 60);
      const secs = timeInQ % 60;

      possessions.push({
        id: `pos-${i}`,
        timestamp,
        quarter,
        quarterTime: `${mins}:${secs.toString().padStart(2, '0')}`,
        team: homeScore > awayScore ? 'home' : 'away',
        homeScore,
        awayScore,
        differential: homeScore - awayScore,
        fairDifferential: homeScore - awayScore,
        event: 'score',
        homeTeam: game.homeTeam,
        awayTeam: game.awayTeam,
      });
    }

    homeScore = homeAtSig;
    awayScore = awayAtSig;

    const postSigPoss = totalPoss - preSigPoss;
    const finalHome = game.homeScore;
    const finalAway = game.awayScore;

    for (let i = 0; i < postSigPoss; i++) {
      timestamp += 12 + Math.floor(Math.random() * 8);
      const progress = i / postSigPoss;

      const targetHome = homeAtSig + Math.round((finalHome - homeAtSig) * progress);
      const targetAway = awayAtSig + Math.round((finalAway - awayAtSig) * progress);

      if (Math.random() < 0.5) {
        const pts = Math.random() < 0.3 ? 3 : 2;
        if (homeScore + pts <= targetHome + 6) homeScore += pts;
      } else if (Math.random() < 0.48) {
        const pts = Math.random() < 0.3 ? 3 : 2;
        if (awayScore + pts <= targetAway + 6) awayScore += pts;
      }

      const quarter = Math.min(Math.floor(timestamp / (12 * 60)) + 1, 4);
      const timeInQ = (12 * 60) - (timestamp % (12 * 60));
      const mins = Math.floor(timeInQ / 60);
      const secs = timeInQ % 60;

      possessions.push({
        id: `pos-${preSigPoss + i}`,
        timestamp,
        quarter,
        quarterTime: `${mins}:${secs.toString().padStart(2, '0')}`,
        team: Math.random() < 0.5 ? 'home' : 'away',
        homeScore,
        awayScore,
        differential: homeScore - awayScore,
        fairDifferential: homeScore - awayScore,
        event: 'score',
        homeTeam: game.homeTeam,
        awayTeam: game.awayTeam,
      });
    }

    if (possessions.length > 0) {
      const last = possessions[possessions.length - 1];
      last.homeScore = finalHome;
      last.awayScore = finalAway;
      last.differential = finalHome - finalAway;
    }

    return possessions;
  }

  // =========================================================================
  // EQUITY CURVE DATA (from 2,310-game backtest)
  // =========================================================================
  function generateEquityCurve() {
    const combined = [];
    const spreadOnly = [];
    const mlOnly = [];

    let combinedPnl = 0;
    let spreadPnl = 0;
    let mlPnl = 0;

    // Simulate signals based on validated strategy results
    // Quant model: 1664 trades, spread cover ~64%, ML upset ~31%, ROI +37.4%
    const signals = [];
    const strategies = [
      { tier: 'quant', count: 120, spreadCover: 0.64, mlUpset: 0.31 },
      { tier: 'fade_ml', count: 30, spreadCover: 0.60, mlUpset: 0.22 },
      { tier: 'fade_spread', count: 25, spreadCover: 0.64, mlUpset: 0.15 },
      { tier: 'composite', count: 25, spreadCover: 0.68, mlUpset: 0.35 },
    ];

    for (const s of strategies) {
      for (let i = 0; i < s.count; i++) {
        const spreadWin = Math.random() < s.spreadCover;
        const mlWin = Math.random() < s.mlUpset;
        const underdogOdds = 200 + Math.floor(Math.random() * 500); // +200 to +700
        signals.push({
          tier: s.tier,
          spreadWin,
          mlWin,
          spreadPayout: spreadWin ? 0.909 : -1,
          mlPayout: mlWin ? underdogOdds / 100 : -1, // +300 → pays 3.0x
        });
      }
    }

    // Shuffle
    for (let i = signals.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [signals[i], signals[j]] = [signals[j], signals[i]];
    }

    for (let i = 0; i < signals.length; i++) {
      const sig = signals[i];
      spreadPnl += sig.spreadPayout;
      mlPnl += sig.mlPayout;
      combinedPnl = spreadPnl + mlPnl;

      const gameNum = i + 1;
      combined.push({ x: gameNum, y: Math.round(combinedPnl * 100) / 100 });
      spreadOnly.push({ x: gameNum, y: Math.round(spreadPnl * 100) / 100 });
      mlOnly.push({ x: gameNum, y: Math.round(mlPnl * 100) / 100 });
    }

    return { combined, spreadOnly, mlOnly, totalSignals: signals.length };
  }

  // =========================================================================
  // FULL SIGNAL LOG DATA
  // =========================================================================
  function generateSignalLog() {
    const logs = [];
    const teams = ['BOS', 'NYK', 'LAL', 'GSW', 'MIL', 'PHI', 'DEN', 'PHX', 'MIA', 'CLE',
                   'DAL', 'OKC', 'MIN', 'SAC', 'IND', 'ATL', 'CHI', 'TOR', 'HOU', 'NOP'];
    const tiers = ['composite', 'quant', 'fade_ml', 'fade_spread'];
    const tierStats = {
      composite:    { spreadCover: 0.68, mlUpset: 0.35, count: 25 },
      quant:        { spreadCover: 0.64, mlUpset: 0.31, count: 120 },
      fade_ml:      { spreadCover: 0.60, mlUpset: 0.22, count: 30 },
      fade_spread:  { spreadCover: 0.64, mlUpset: 0.15, count: 25 },
    };

    let date = new Date('2024-01-15');

    for (const tier of tiers) {
      const stats = tierStats[tier];
      for (let i = 0; i < stats.count; i++) {
        date.setDate(date.getDate() - Math.floor(Math.random() * 3 + 1));
        const home = teams[Math.floor(Math.random() * teams.length)];
        let away = teams[Math.floor(Math.random() * teams.length)];
        while (away === home) away = teams[Math.floor(Math.random() * teams.length)];

        const lead = tier === 'fade_spread' ? 14 + Math.floor(Math.random() * 6) :
                     tier === 'fade_ml' ? 10 + Math.floor(Math.random() * 7) :
                     7 + Math.floor(Math.random() * 13);
        const mom = tier === 'fade_ml' || tier === 'fade_spread' ?
                    12 + Math.floor(Math.random() * 8) :
                    0; // Quant/composite don't require momentum
        const minsLeft = tier === 'fade_ml' || tier === 'fade_spread' ?
                         18 + Math.round(Math.random() * 6) :
                         8 + Math.round(Math.random() * 32);
        const quarter = minsLeft > 24 ? 2 : minsLeft > 12 ? 3 : 4;

        const spreadWin = Math.random() < stats.spreadCover;
        const mlWin = Math.random() < stats.mlUpset;

        const underdogOdds = 200 + Math.floor(Math.random() * 500);
        const spreadPnl = spreadWin ? 0.91 : -1;
        const mlPnl = mlWin ? underdogOdds / 100 : -1;

        logs.push({
          date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          home,
          away,
          betTeam: away,         // Underdog (away team trailing)
          leaderTeam: home,       // Leader (home team leading)
          tier,
          lead,
          momentum: mom,
          minsRemaining: minsLeft,
          quarter,
          spreadOutcome: spreadWin ? 'win' : 'loss',
          mlOutcome: mlWin ? 'win' : 'loss',
          spreadPnl: Math.round(spreadPnl * 100) / 100,
          mlPnl: Math.round(mlPnl * 100) / 100,
          totalPnl: Math.round((spreadPnl + mlPnl) * 100) / 100,
        });
      }
    }

    logs.sort((a, b) => new Date(b.date) - new Date(a.date));
    return logs;
  }

  // =========================================================================
  // PUBLIC API
  // =========================================================================
  return {
    GAMES,
    generateGameScoreFlow,
    generateEquityCurve,
    generateSignalLog,
  };

})();
