#!/usr/bin/env node
// =============================================================================
// Seed Live Picks — MLB ULTRA BETTING ENGINE v3.0
// Fetches live odds, generates picks using the Ultra Engine v3,
// resolves results via ESPN, outputs to mlb/webapp/data/
// =============================================================================

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// --- curl-based fetch ---
function curlFetch(url) {
  try {
    const result = execSync(`curl -s "${url}"`, { maxBuffer: 10 * 1024 * 1024, timeout: 30000 });
    return JSON.parse(result.toString());
  } catch (e) {
    return null;
  }
}

global.fetch = async (url) => {
  const data = curlFetch(url);
  if (!data || data.error_code) {
    return { ok: false, status: data ? 400 : 500, headers: { get: () => null }, json: async () => data };
  }
  return { ok: true, status: 200, headers: { get: () => null }, json: async () => data };
};

global.window = global;
global.document = { readyState: 'complete', addEventListener: () => {}, querySelectorAll: () => [] };
global.localStorage = {
  _data: {},
  getItem(k) { return this._data[k] || null; },
  setItem(k, v) { this._data[k] = v; },
};

// Load MLB Ultra Engine v3
eval(fs.readFileSync(path.join(__dirname, '../webapp/js/recommendation-engine-mlb-v3.js'), 'utf8'));

const ENGINE = global.window.MLBRecommendationEngineV3;
const DATA_DIR = path.join(__dirname, '../webapp/data');
const OUTPUT_DIR = path.join(__dirname, '../output');
const SIGNALS_FILE = path.join(DATA_DIR, 'mlb_ultra_signals_v3.json');
const STATS_FILE = path.join(DATA_DIR, 'mlb_ultra_backtest_stats_v3.json');
const RECS_FILE = path.join(DATA_DIR, 'mlb_ultra_recommendations.json');

const ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8';
const ODDS_API_BASE = 'https://api.the-odds-api.com/v4';

const TEAM_MAP = {
  'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
  'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
  'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
  'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
  'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
  'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
  'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
  'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
  'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
  'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
  'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
  'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
  'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL',
  'Tampa Bay Rays': 'TB', 'Texas Rangers': 'TEX',
  'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH',
};

function teamAbbr(name) { return TEAM_MAP[name] || name; }

function loadJSON(filename) {
  try { return JSON.parse(fs.readFileSync(path.join(DATA_DIR, filename), 'utf8')); }
  catch (e) { return []; }
}

function getDateStr(daysAgo) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}`;
}

function sleep(ms) { execSync(`sleep ${ms / 1000}`); }

// =============================================================================
// Resolve pending signals against ESPN box scores
// =============================================================================
async function resolveResults() {
  let signals = [];
  try {
    signals = JSON.parse(fs.readFileSync(SIGNALS_FILE, 'utf8'));
  } catch (e) { return; }

  const pending = signals.filter(s => s.hit === null || s.hit === undefined);
  if (pending.length === 0) {
    console.log('No pending signals to resolve.');
    return;
  }

  const pendingDates = [...new Set(pending.map(s => s.date))].sort();
  console.log(`Resolving ${pending.length} pending signals from ${pendingDates.length} dates...`);

  let resolved = 0;

  for (const date of pendingDates) {
    try {
      const scoreboardUrl = `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=${date}`;
      const scoreboardRes = await fetch(scoreboardUrl);
      if (!scoreboardRes.ok) continue;
      const scoreboardData = await scoreboardRes.json();

      if (!scoreboardData.events || scoreboardData.events.length === 0) continue;

      // Fetch box scores for each completed game
      const playerStats = {}; // player -> { h, tb, rbi, r, hr, ... }

      for (const event of scoreboardData.events) {
        if (event.status && event.status.type && event.status.type.completed !== true) continue;

        try {
          const boxUrl = `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event=${event.id}`;
          const boxRes = await fetch(boxUrl);
          if (!boxRes.ok) continue;
          const boxData = await boxRes.json();

          // Extract batter stats from box score
          const rosters = boxData.rosters || boxData.boxscore?.players || [];
          for (const team of rosters) {
            for (const category of (team.statistics || [])) {
              if (category.type !== 'batting' && category.name !== 'batting') continue;
              const labels = (category.labels || []).map(l => l.toLowerCase());
              for (const athlete of (category.athletes || [])) {
                const name = athlete.athlete?.displayName;
                if (!name) continue;
                const stats = athlete.stats || [];
                const statObj = {};
                labels.forEach((label, i) => {
                  statObj[label] = stats[i];
                });
                const h = parseInt(statObj.h) || 0;
                const doubles = parseInt(statObj['2b']) || 0;
                const triples = parseInt(statObj['3b']) || 0;
                const hr = parseInt(statObj.hr) || 0;
                const rbi = parseInt(statObj.rbi) || 0;
                const r = parseInt(statObj.r) || 0;
                const tb = h + doubles + triples * 2 + hr * 3;

                playerStats[name] = { h, tb, rbi, r, hr };
              }
            }
          }
        } catch (e) { /* skip game */ }
        sleep(100);
      }

      // Resolve signals for this date
      for (const signal of signals) {
        if (signal.date !== date) continue;
        if (signal.hit !== null && signal.hit !== undefined) continue;

        if (signal.betType === 'single') {
          const ps = playerStats[signal.player];
          if (!ps) continue;
          const statKey = signal.stat || 'h';
          const actual = ps[statKey] || 0;
          signal.actual = actual;
          signal.hit = actual > signal.line;
          const decimal = signal.odds > 0 ? (signal.odds / 100) + 1 : (100 / Math.abs(signal.odds)) + 1;
          signal.pnl = signal.hit ? Math.round((decimal - 1) * 100) : -100;
          resolved++;
        } else if (signal.betType === 'parlay' && signal.legs) {
          let allHit = true;
          let anyResolved = false;
          for (const leg of signal.legs) {
            if (leg.hit !== null && leg.hit !== undefined) { if (!leg.hit) allHit = false; continue; }
            const ps = playerStats[leg.player];
            if (!ps) { allHit = false; continue; }
            const statKey = leg.stat || 'h';
            const actual = ps[statKey] || 0;
            leg.actual = actual;
            leg.hit = actual > leg.line;
            if (!leg.hit) allHit = false;
            anyResolved = true;
          }
          if (anyResolved && signal.legs.every(l => l.hit !== null && l.hit !== undefined)) {
            signal.hit = allHit;
            const parlayDecimal = signal.parlay_decimal || signal.legs.reduce((d, l) => {
              const ld = l.odds > 0 ? (l.odds / 100) + 1 : (100 / Math.abs(l.odds)) + 1;
              return d * ld;
            }, 1);
            signal.pnl = allHit ? Math.round((parlayDecimal - 1) * 100) : -100;
            resolved++;
          }
        }
      }
    } catch (e) {
      console.warn(`Error resolving date ${date}:`, e.message);
    }
    sleep(200);
  }

  if (resolved > 0) {
    fs.writeFileSync(SIGNALS_FILE, JSON.stringify(signals, null, 2));
    console.log(`Resolved ${resolved} signals.`);

    // Update backtest stats
    const singles = signals.filter(s => s.betType === 'single' && s.hit !== null && s.hit !== undefined);
    const parlays = signals.filter(s => s.betType === 'parlay' && s.hit !== null && s.hit !== undefined);
    const singleWins = singles.filter(s => s.hit).length;
    const parlayWins = parlays.filter(s => s.hit).length;
    const singlePnl = singles.reduce((s, p) => s + (p.pnl || 0), 0);
    const parlayPnl = parlays.reduce((s, p) => s + (p.pnl || 0), 0);
    const singleWagered = singles.length * 100;
    const parlayWagered = parlays.length * 100;
    const totalLegs = parlays.reduce((s, p) => s + (p.legs ? p.legs.length : 0), 0);
    const hitLegs = parlays.reduce((s, p) => s + (p.legs ? p.legs.filter(l => l.hit).length : 0), 0);

    const stats = {
      singles: {
        total: singles.length, wins: singleWins,
        accuracy: singles.length > 0 ? singleWins / singles.length : 0,
        pnl: singlePnl, wagered: singleWagered,
        roi: singleWagered > 0 ? singlePnl / singleWagered : 0,
      },
      parlays: {
        total: parlays.length, wins: parlayWins,
        accuracy: parlays.length > 0 ? parlayWins / parlays.length : 0,
        pnl: parlayPnl, wagered: parlayWagered,
        roi: parlayWagered > 0 ? parlayPnl / parlayWagered : 0,
        totalLegs, hitLegs,
        legAccuracy: totalLegs > 0 ? hitLegs / totalLegs : 0,
      },
      overall: {
        total: singles.length + parlays.length,
        wins: singleWins + parlayWins,
        accuracy: (singles.length + parlays.length) > 0 ? (singleWins + parlayWins) / (singles.length + parlays.length) : 0,
        pnl: singlePnl + parlayPnl,
        wagered: singleWagered + parlayWagered,
        roi: (singleWagered + parlayWagered) > 0 ? (singlePnl + parlayPnl) / (singleWagered + parlayWagered) : 0,
      },
    };
    fs.writeFileSync(STATS_FILE, JSON.stringify(stats, null, 2));
    console.log('Updated backtest stats.');
  } else {
    console.log('No signals could be resolved (games may not be complete yet).');
  }
}

async function main() {
  console.log('\n' + '='.repeat(60));
  console.log('MLB ULTRA BETTING ENGINE v3.0 — Seed Live Picks');
  console.log('='.repeat(60));

  // =========================================================================
  // STEP 0: Resolve pending picks from previous days
  // =========================================================================
  await resolveResults();

  // Load optimized config if available
  try {
    const configPath = path.join(OUTPUT_DIR, 'mlb_ultra_engine_v3_config.json');
    if (fs.existsSync(configPath)) {
      const configData = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      ENGINE.loadConfig(configData);
    }
  } catch (e) { /* use defaults */ }

  // Build player model from all box scores
  const boxScores = loadJSON('mlb_player_boxscores.json');
  console.log(`Loaded ${boxScores.length} box score records`);

  ENGINE.PlayerModel.reset();
  const sortedGames = [...boxScores].sort((a, b) => (a.date || '').localeCompare(b.date || ''));
  for (const game of sortedGames) {
    for (const p of (game.players || [])) {
      const ab = p.ab || 0;
      if (ab < 1) continue;
      ENGINE.PlayerModel.update(p.name, {
        h: p.h || 0,
        tb: p.tb || 0,
        rbi: p.rbi || 0,
        r: p.r || 0,
        hr: p.hr || 0,
        ab: ab,
        bb: p.bb || 0,
        so: p.so || 0,
        sb: p.sb || 0,
      }, game.date, p.team, game.home === p.team ? game.away : game.home);
    }
  }
  console.log(`Player model built with ${Object.keys(ENGINE.PlayerModel.profiles).length} players`);

  // Fetch tonight's games (regular season + preseason/spring training)
  console.log('\nFetching tonight\'s games from Odds API...');
  let events = [];
  try {
    const [regRes, preRes] = await Promise.all([
      fetch(`${ODDS_API_BASE}/sports/baseball_mlb/events?apiKey=${ODDS_API_KEY}`).catch(() => null),
      fetch(`${ODDS_API_BASE}/sports/baseball_mlb_preseason/events?apiKey=${ODDS_API_KEY}`).catch(() => null),
    ]);
    if (regRes && regRes.ok) {
      const regEvents = await regRes.json();
      events.push(...regEvents);
    }
    if (preRes && preRes.ok) {
      const preEvents = await preRes.json();
      // Tag preseason events so we use the right sport key for odds
      for (const e of preEvents) e._sportKey = 'baseball_mlb_preseason';
      events.push(...preEvents);
    }
  } catch (e) {
    console.warn('Could not fetch events:', e.message);
  }

  if (!events || events.length === 0) {
    console.log('No games scheduled tonight.');
    const today = getDateStr(0);
    fs.writeFileSync(RECS_FILE, JSON.stringify({
      generated: new Date().toISOString(), date: today,
      engine: 'MLB Ultra Engine v3.0', config_version: 'optimized',
      recommendation: { betType: 'none', reasoning: 'No games scheduled tonight', singles: [], parlays: [] },
    }, null, 2));
    return;
  }

  console.log(`Found ${events.length} games tonight`);

  // Fetch batter props for each game
  const liveOdds = { events, playerProps: {} };

  for (const event of events) {
    try {
      const markets = 'batter_hits_alternate,batter_total_bases_alternate,batter_rbis_alternate,batter_runs_scored_alternate';
      const sportKey = event._sportKey || 'baseball_mlb';
      const propsUrl = `${ODDS_API_BASE}/sports/${sportKey}/events/${event.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=${markets}&oddsFormat=american&bookmakers=fanduel`;
      const propsRes = await fetch(propsUrl);
      if (!propsRes.ok) continue;

      const propsData = await propsRes.json();
      const homeAbbr = teamAbbr(event.home_team);
      const awayAbbr = teamAbbr(event.away_team);
      const gameKey = `${awayAbbr}@${homeAbbr}`;

      const fd = propsData.bookmakers && propsData.bookmakers.find(b => b.key === 'fanduel');
      if (!fd) continue;

      const gameProps = { hitsLines: {}, tbLines: {}, rbiLines: {}, runsLines: {} };
      const marketMap = {
        'batter_hits_alternate': 'hitsLines',
        'batter_total_bases_alternate': 'tbLines',
        'batter_rbis_alternate': 'rbiLines',
        'batter_runs_scored_alternate': 'runsLines',
      };

      for (const mkt of (fd.markets || [])) {
        const st = marketMap[mkt.key];
        if (!st) continue;
        for (const outcome of (mkt.outcomes || [])) {
          const player = outcome.description;
          const threshold = outcome.point;
          if (!gameProps[st][player]) gameProps[st][player] = {};
          if (!gameProps[st][player][threshold]) gameProps[st][player][threshold] = {};
          if (outcome.name === 'Over') {
            gameProps[st][player][threshold].overOdds = outcome.price;
          }
        }
      }

      liveOdds.playerProps[gameKey] = gameProps;
      console.log(`  ${gameKey}: ${Object.keys(gameProps.hitsLines).length} H, ${Object.keys(gameProps.tbLines).length} TB, ${Object.keys(gameProps.rbiLines).length} RBI, ${Object.keys(gameProps.runsLines).length} R props`);
    } catch (e) {
      console.warn(`  Error fetching props for ${event.id}:`, e.message);
    }
    sleep(200);
  }

  if (Object.keys(liveOdds.playerProps).length === 0) {
    console.log('\nNo FanDuel batter props available.');
    const today = getDateStr(0);
    fs.writeFileSync(RECS_FILE, JSON.stringify({
      generated: new Date().toISOString(), date: today,
      engine: 'MLB Ultra Engine v3.0', config_version: 'optimized',
      recommendation: { betType: 'none', reasoning: 'No FanDuel batter props available yet', singles: [], parlays: [] },
    }, null, 2));
    return;
  }

  // Generate recommendations
  console.log('\nRunning Ultra Engine signal analysis...');
  const recommendation = ENGINE.generateTodayPicks(liveOdds);

  if (!recommendation || (recommendation.singles.length === 0 && recommendation.parlays.length === 0)) {
    console.log('No bets meet Ultra Engine quality thresholds tonight.');
    const today = getDateStr(0);
    fs.writeFileSync(RECS_FILE, JSON.stringify({
      generated: new Date().toISOString(), date: today,
      engine: 'MLB Ultra Engine v3.0', config_version: 'optimized',
      recommendation: { betType: 'none', reasoning: 'No bets meet quality thresholds tonight', singles: [], parlays: [] },
    }, null, 2));
    return;
  }

  // Format and save signals
  const today = getDateStr(0);
  const newSignals = ENGINE.formatSignalForStorage(recommendation, today);

  let existingSignals = [];
  try {
    existingSignals = JSON.parse(fs.readFileSync(SIGNALS_FILE, 'utf8'));
  } catch (e) { /* no existing signals */ }

  // Remove today's existing live signals (avoid duplicates)
  existingSignals = existingSignals.filter(s => !(s.date === today && s.source === 'live'));
  // Tag live signals so they're never overwritten by backtest exports
  for (const sig of newSignals) sig.source = 'live';
  existingSignals.push(...newSignals);

  // Remove backtest signals for any date that has live picks (backtest signals
  // for live dates are retroactive and were never shown to the user)
  const liveDates = new Set(existingSignals.filter(s => s.source === 'live').map(s => s.date));
  existingSignals = existingSignals.filter(s => s.source === 'live' || !liveDates.has(s.date));

  fs.writeFileSync(SIGNALS_FILE, JSON.stringify(existingSignals, null, 2));

  // Save recommendations — store the full recommendation object so the webapp
  // can render directly without re-fetching live odds (ensures consistency)
  const recsOutput = {
    generated: new Date().toISOString(),
    date: today,
    engine: 'MLB Ultra Engine v3.0',
    config_version: 'optimized',
    recommendation: {
      betType: recommendation.betType,
      reasoning: recommendation.reasoning,
      singles: recommendation.singles.map(s => ({
        player: s.player,
        team: s.team,
        statType: s.statType,
        statLabel: s.statLabel,
        line: s.line,
        odds: s.odds,
        cascadeScore: s.cascadeScore,
        gft: s.gft,
        beq: s.beq,
        esi: s.esi,
        imad: s.imad,
        hitRate: s.hitRate,
        edge: s.edge,
        ev: s.ev,
        avg: s.avg,
        floor: s.floor,
        betSubType: s.cascadeScore >= ENGINE.CONFIG.SINGLE_MIN_SCORE ? 'single' : 'multi_single',
        hit: null,
        actual: null,
      })),
      parlays: recommendation.parlays.map(p => ({
        numLegs: p.numLegs,
        odds: p.odds,
        decimalOdds: p.decimalOdds,
        avgCascade: p.avgCascade,
        combinedHitRate: p.combinedHitRate,
        ev: p.ev,
        legs: p.legs.map(l => ({
          player: l.player,
          team: l.team,
          statType: l.statType,
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
          hit: null,
          actual: null,
        })),
      })),
    },
  };

  fs.writeFileSync(RECS_FILE, JSON.stringify(recsOutput, null, 2));

  // Print summary
  console.log('\n' + '='.repeat(60));
  console.log('TONIGHT\'S MLB ULTRA ENGINE PICKS:');
  console.log('='.repeat(60));

  for (const s of recommendation.singles) {
    const typeLabel = s.cascadeScore >= ENGINE.CONFIG.SINGLE_MIN_SCORE ? 'SINGLE' : 'MULTI_SINGLE';
    console.log(`  ${typeLabel}: ${s.player} O${s.line} ${(s.statLabel || 'H').toUpperCase()} | Odds: ${ENGINE.formatOdds(s.odds)} | Score: ${(s.cascadeScore * 100).toFixed(0)}%`);
  }
  for (const p of recommendation.parlays) {
    const legs = p.legs.map(l => `${l.player} O${l.line} ${(l.statLabel || 'H').toUpperCase()}`).join(' + ');
    console.log(`  PARLAY: ${legs} | Odds: ${ENGINE.formatOdds(p.odds)}`);
  }

  console.log(`\nSingles: ${recommendation.singles.length}`);
  console.log(`Parlays: ${recommendation.parlays.length}`);
  console.log(`\nSaved to ${SIGNALS_FILE}`);
}

main().catch(console.error);
