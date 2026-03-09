#!/usr/bin/env node
// =============================================================================
// Seed Live Picks — MLB ULTRA BETTING ENGINE v1.0
// Fetches live odds, generates picks using the Ultra Engine,
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

// Load MLB Ultra Engine
eval(fs.readFileSync(path.join(__dirname, '../webapp/js/recommendation-engine-mlb.js'), 'utf8'));

const ENGINE = global.window.MLBRecommendationEngine;
const DATA_DIR = path.join(__dirname, '../webapp/data');
const OUTPUT_DIR = path.join(__dirname, '../output');
const SIGNALS_FILE = path.join(DATA_DIR, 'mlb_ultra_signals.json');
const STATS_FILE = path.join(DATA_DIR, 'mlb_ultra_backtest_stats.json');
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

async function main() {
  console.log('\n' + '='.repeat(60));
  console.log('MLB ULTRA BETTING ENGINE v1.0 — Seed Live Picks');
  console.log('='.repeat(60));

  // Load optimized config if available
  try {
    const configPath = path.join(OUTPUT_DIR, 'mlb_ultra_engine_config.json');
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

  // Fetch tonight's games
  console.log('\nFetching tonight\'s games from Odds API...');
  let events = [];
  try {
    const eventsUrl = `${ODDS_API_BASE}/sports/baseball_mlb/events?apiKey=${ODDS_API_KEY}`;
    const eventsRes = await fetch(eventsUrl);
    if (eventsRes.ok) {
      events = await eventsRes.json();
    }
  } catch (e) {
    console.warn('Could not fetch events:', e.message);
  }

  if (!events || events.length === 0) {
    console.log('No games scheduled tonight.');
    return;
  }

  console.log(`Found ${events.length} games tonight`);

  // Fetch batter props for each game
  const liveOdds = { events, playerProps: {} };

  for (const event of events) {
    try {
      const markets = 'batter_hits,batter_total_bases,batter_rbis,batter_runs_scored';
      const propsUrl = `${ODDS_API_BASE}/sports/baseball_mlb/events/${event.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=${markets}&oddsFormat=american&bookmakers=fanduel`;
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
        'batter_hits': 'hitsLines',
        'batter_total_bases': 'tbLines',
        'batter_rbis': 'rbiLines',
        'batter_runs_scored': 'runsLines',
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
    return;
  }

  // Generate recommendations
  console.log('\nRunning Ultra Engine signal analysis...');
  const recommendation = ENGINE.generateTodayPicks(liveOdds);

  if (!recommendation || (recommendation.singles.length === 0 && recommendation.parlays.length === 0)) {
    console.log('No bets meet Ultra Engine quality thresholds tonight.');
    return;
  }

  // Format and save signals
  const today = getDateStr(0);
  const newSignals = ENGINE.formatSignalForStorage(recommendation, today);

  let existingSignals = [];
  try {
    existingSignals = JSON.parse(fs.readFileSync(SIGNALS_FILE, 'utf8'));
  } catch (e) { /* no existing signals */ }

  existingSignals = existingSignals.filter(s => s.date !== today);
  existingSignals.push(...newSignals);

  fs.writeFileSync(SIGNALS_FILE, JSON.stringify(existingSignals, null, 2));

  // Save recommendations
  const recsOutput = {
    generated: new Date().toISOString(),
    engine: 'MLB Ultra Engine v1.0',
    config_version: 'optimized',
    tonight_picks: [],
  };

  for (const s of recommendation.singles) {
    recsOutput.tonight_picks.push({
      bet_type: s.cascadeScore >= ENGINE.CONFIG.SINGLE_MIN_SCORE ? 'single' : 'multi_single',
      player: s.player, stat: s.statType, line: s.line, odds: s.odds,
      combined_score: s.cascadeScore, edge: s.edge, hit_rate: s.hitRate,
      bayesian_prob: s.impliedProb + s.edge, suggested_wager: 50,
    });
  }

  for (const p of recommendation.parlays) {
    recsOutput.tonight_picks.push({
      bet_type: 'parlay',
      player: null, stat: null, line: null, odds: null,
      combined_score: null, edge: null, hit_rate: null,
      bayesian_prob: null, suggested_wager: 100,
      legs: p.legs.map(l => ({
        player: l.player, stat: l.statType,
        line: l.line, odds: l.odds,
        combined_score: l.cascadeScore, edge: l.edge,
      })),
      parlay_odds: p.odds,
      parlay_ev: p.ev,
    });
  }

  fs.writeFileSync(RECS_FILE, JSON.stringify(recsOutput, null, 2));

  // Print summary
  console.log('\n' + '='.repeat(60));
  console.log('TONIGHT\'S MLB ULTRA ENGINE PICKS:');
  console.log('='.repeat(60));

  for (const pick of recsOutput.tonight_picks) {
    if (pick.bet_type === 'parlay') {
      const legs = pick.legs.map(l => `${l.player} O${l.line} ${(l.stat || 'h').toUpperCase()}`).join(' + ');
      console.log(`  PARLAY: ${legs} | Odds: ${ENGINE.formatOdds(pick.parlay_odds)}`);
    } else {
      console.log(`  ${pick.bet_type.toUpperCase()}: ${pick.player} O${pick.line} ${(pick.stat || 'h').toUpperCase()} | Odds: ${ENGINE.formatOdds(pick.odds)} | Score: ${(pick.combined_score * 100).toFixed(0)}%`);
    }
  }

  console.log(`\nSingles: ${recommendation.singles.length}`);
  console.log(`Parlays: ${recommendation.parlays.length}`);
  console.log(`\nSaved to ${SIGNALS_FILE}`);
}

main().catch(console.error);
