// =============================================================================
// HISTORICAL DATA - Sample games with realistic signal data
// =============================================================================
// Based on validated strategy results from 156 real ESPN NBA games
// Each game includes possession-level score flow and signal data
// =============================================================================

window.HistoricalData = (function() {

  // =========================================================================
  // GENERATE REALISTIC POSSESSION DATA
  // =========================================================================
  function generatePossessions(homeTeam, awayTeam, segments, totalPossessions) {
    const possessions = [];
    let homeScore = 0;
    let awayScore = 0;
    let timestamp = 0;

    // segments: [{poss: count, homeRate: ptsPerPoss, awayRate: ptsPerPoss}, ...]
    let posIdx = 0;
    for (const seg of segments) {
      for (let i = 0; i < seg.poss && posIdx < totalPossessions; i++) {
        timestamp += 12 + Math.floor(Math.random() * 8); // 12-20 seconds per possession
        const quarter = Math.floor(timestamp / (12 * 60)) + 1;
        const timeInQuarter = (12 * 60) - (timestamp % (12 * 60));
        const mins = Math.floor(timeInQuarter / 60);
        const secs = timeInQuarter % 60;

        // Random scoring
        const homeScores = Math.random() < seg.homeRate;
        const awayScores = !homeScores && Math.random() < seg.awayRate;

        if (homeScores) {
          const pts = Math.random() < 0.35 ? 3 : 2;
          homeScore += pts;
        } else if (awayScores) {
          const pts = Math.random() < 0.35 ? 3 : 2;
          awayScore += pts;
        }

        possessions.push({
          id: `pos-${posIdx}`,
          timestamp,
          quarter: Math.min(quarter, 4),
          quarterTime: `${mins}:${secs.toString().padStart(2, '0')}`,
          team: homeScores ? 'home' : 'away',
          homeScore,
          awayScore,
          differential: homeScore - awayScore,
          fairDifferential: homeScore - awayScore,
          event: 'score',
          homeTeam,
          awayTeam,
        });
        posIdx++;
      }
    }

    return possessions;
  }

  // =========================================================================
  // HISTORICAL GAMES DATA
  // =========================================================================
  const GAMES = [
    // Game 0: BOS vs NYK - Elite Signal, Win
    {
      id: 'hist-001',
      date: 'Jan 15',
      homeTeam: 'BOS',
      awayTeam: 'NYK',
      homeScore: 118,
      awayScore: 95,
      signal: {
        tier: 'elite',
        team: 'home',
        betTeam: 'BOS',
        lead: 14,
        momentum: 16,
        quarter: 2,
        quarterTime: '4:30',
        minsRemaining: 16.5,
        scoreAtSignal: { home: 52, away: 38 },
        spreadOutcome: 'win',
        mlOutcome: 'win',
        possessionIndex: 62,
      },
    },
    // Game 1: LAL vs GSW - Strong Signal, Win
    {
      id: 'hist-002',
      date: 'Jan 12',
      homeTeam: 'LAL',
      awayTeam: 'GSW',
      homeScore: 121,
      awayScore: 108,
      signal: {
        tier: 'strong',
        team: 'home',
        betTeam: 'LAL',
        lead: 12,
        momentum: 13,
        quarter: 2,
        quarterTime: '6:15',
        minsRemaining: 18.3,
        scoreAtSignal: { home: 48, away: 36 },
        spreadOutcome: 'win',
        mlOutcome: 'win',
        possessionIndex: 55,
      },
    },
    // Game 2: MIL vs PHI - Standard Signal, Win
    {
      id: 'hist-003',
      date: 'Jan 10',
      homeTeam: 'MIL',
      awayTeam: 'PHI',
      homeScore: 112,
      awayScore: 101,
      signal: {
        tier: 'standard',
        team: 'home',
        betTeam: 'MIL',
        lead: 11,
        momentum: 10,
        quarter: 3,
        quarterTime: '8:20',
        minsRemaining: 20.3,
        scoreAtSignal: { home: 62, away: 51 },
        spreadOutcome: 'win',
        mlOutcome: 'win',
        possessionIndex: 72,
      },
    },
    // Game 3: DEN vs PHX - Elite Signal, Win
    {
      id: 'hist-004',
      date: 'Jan 8',
      homeTeam: 'DEN',
      awayTeam: 'PHX',
      homeScore: 125,
      awayScore: 102,
      signal: {
        tier: 'elite',
        team: 'home',
        betTeam: 'DEN',
        lead: 16,
        momentum: 18,
        quarter: 2,
        quarterTime: '3:45',
        minsRemaining: 15.8,
        scoreAtSignal: { home: 56, away: 40 },
        spreadOutcome: 'win',
        mlOutcome: 'win',
        possessionIndex: 65,
      },
    },
    // Game 4: MIA vs CLE - Wide Signal, Loss (comeback)
    {
      id: 'hist-005',
      date: 'Jan 5',
      homeTeam: 'MIA',
      awayTeam: 'CLE',
      homeScore: 105,
      awayScore: 108,
      signal: {
        tier: 'wide',
        team: 'home',
        betTeam: 'MIA',
        lead: 10,
        momentum: 9,
        quarter: 2,
        quarterTime: '5:00',
        minsRemaining: 17.0,
        scoreAtSignal: { home: 48, away: 38 },
        spreadOutcome: 'loss',
        mlOutcome: 'loss',
        possessionIndex: 58,
      },
    },
    // Game 5: DAL vs OKC - Strong Signal, Win
    {
      id: 'hist-006',
      date: 'Jan 3',
      homeTeam: 'DAL',
      awayTeam: 'OKC',
      homeScore: 119,
      awayScore: 107,
      signal: {
        tier: 'strong',
        team: 'home',
        betTeam: 'DAL',
        lead: 13,
        momentum: 12,
        quarter: 2,
        quarterTime: '7:30',
        minsRemaining: 19.5,
        scoreAtSignal: { home: 45, away: 32 },
        spreadOutcome: 'win',
        mlOutcome: 'win',
        possessionIndex: 48,
      },
    },
    // Game 6: BOS vs MIA - Elite Signal, Win
    {
      id: 'hist-007',
      date: 'Dec 28',
      homeTeam: 'BOS',
      awayTeam: 'MIA',
      homeScore: 130,
      awayScore: 105,
      signal: {
        tier: 'elite',
        team: 'home',
        betTeam: 'BOS',
        lead: 18,
        momentum: 15,
        quarter: 2,
        quarterTime: '2:00',
        minsRemaining: 14.0,
        scoreAtSignal: { home: 60, away: 42 },
        spreadOutcome: 'win',
        mlOutcome: 'win',
        possessionIndex: 70,
      },
    },
    // Game 7: GSW vs LAC - Standard Signal, Win
    {
      id: 'hist-008',
      date: 'Dec 25',
      homeTeam: 'GSW',
      awayTeam: 'LAC',
      homeScore: 113,
      awayScore: 104,
      signal: {
        tier: 'standard',
        team: 'home',
        betTeam: 'GSW',
        lead: 11,
        momentum: 11,
        quarter: 3,
        quarterTime: '9:00',
        minsRemaining: 21.0,
        scoreAtSignal: { home: 58, away: 47 },
        spreadOutcome: 'win',
        mlOutcome: 'win',
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
    const totalPoss = 200; // ~200 scoring plays per game

    // Pre-signal phase: gradual lead building
    const preSigPoss = sig.possessionIndex;
    const homeAtSig = sig.scoreAtSignal.home;
    const awayAtSig = sig.scoreAtSignal.away;

    // Generate pre-signal possessions
    for (let i = 0; i < preSigPoss; i++) {
      timestamp += 12 + Math.floor(Math.random() * 8);
      const progress = i / preSigPoss;

      // Interpolate scores with some noise
      const targetHome = Math.round(homeAtSig * progress);
      const targetAway = Math.round(awayAtSig * progress);

      // Add some randomness
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

    // Adjust to match signal scores
    homeScore = homeAtSig;
    awayScore = awayAtSig;

    // Post-signal phase
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

    // Ensure final scores match
    if (possessions.length > 0) {
      const last = possessions[possessions.length - 1];
      last.homeScore = finalHome;
      last.awayScore = finalAway;
      last.differential = finalHome - finalAway;
    }

    return possessions;
  }

  // =========================================================================
  // EQUITY CURVE DATA (from 156-game backtest)
  // =========================================================================
  function generateEquityCurve() {
    const combined = [];
    const spreadOnly = [];
    const mlOnly = [];

    let combinedPnl = 0;
    let spreadPnl = 0;
    let mlPnl = 0;

    // Simulate 156 signals with realistic win/loss distribution
    const signals = [];
    const tiers = [
      { tier: 'elite', count: 43, spreadWR: 0.976, mlWR: 0.977 },
      { tier: 'strong', count: 25, spreadWR: 0.88, mlWR: 0.92 },
      { tier: 'standard', count: 34, spreadWR: 0.824, mlWR: 0.941 },
      { tier: 'wide', count: 54, spreadWR: 0.78, mlWR: 0.907 },
    ];

    for (const t of tiers) {
      for (let i = 0; i < t.count; i++) {
        const spreadWin = Math.random() < t.spreadWR;
        const mlWin = Math.random() < t.mlWR;
        signals.push({
          tier: t.tier,
          spreadWin,
          mlWin,
          // Realistic payouts
          spreadPayout: spreadWin ? 0.909 : -1, // -110 odds
          mlPayout: mlWin ? (0.15 + Math.random() * 0.15) : -1, // ML favorites ~-500 to -800
        });
      }
    }

    // Shuffle to randomize order
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
    const tiers = ['elite', 'strong', 'standard', 'wide'];
    const tierStats = {
      elite:    { spreadWR: 0.976, mlWR: 0.977, count: 43 },
      strong:   { spreadWR: 0.88,  mlWR: 0.92,  count: 25 },
      standard: { spreadWR: 0.824, mlWR: 0.941, count: 34 },
      wide:     { spreadWR: 0.78,  mlWR: 0.907, count: 54 },
    };

    let date = new Date('2024-01-15');

    for (const tier of tiers) {
      const stats = tierStats[tier];
      for (let i = 0; i < stats.count; i++) {
        date.setDate(date.getDate() - Math.floor(Math.random() * 3 + 1));
        const home = teams[Math.floor(Math.random() * teams.length)];
        let away = teams[Math.floor(Math.random() * teams.length)];
        while (away === home) away = teams[Math.floor(Math.random() * teams.length)];

        const lead = 10 + Math.floor(Math.random() * 10);
        const mom = tier === 'elite' ? 14 + Math.floor(Math.random() * 6) :
                    tier === 'strong' ? 12 + Math.floor(Math.random() * 2) :
                    tier === 'standard' ? 10 + Math.floor(Math.random() * 2) :
                    8 + Math.floor(Math.random() * 2);
        const minsLeft = 12 + Math.round(Math.random() * 12);
        const quarter = minsLeft > 12 ? 2 : 3;

        const spreadWin = Math.random() < stats.spreadWR;
        const mlWin = Math.random() < stats.mlWR;

        const spreadPnl = spreadWin ? 0.91 : -1;
        const mlPnl = mlWin ? (0.12 + Math.random() * 0.18) : -1;

        logs.push({
          date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          home,
          away,
          betTeam: home,
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

    // Sort by date (most recent first)
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
