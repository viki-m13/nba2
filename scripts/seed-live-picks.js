#!/usr/bin/env node
// =============================================================================
// Seed Live Picks — ULTRA BETTING ENGINE v1.0
// Fetches historical odds, generates picks using the Ultra Engine,
// resolves results via ESPN, outputs to webapp/data/live_picks_2026.json
// =============================================================================

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// --- curl-based fetch (bypasses DNS issues in sandboxed Node) ---
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

// --- Shim browser globals for the Ultra Engine JS ---
global.window = global;
global.document = { readyState: 'complete', addEventListener: () => {}, querySelectorAll: () => [] };
global.localStorage = {
  _data: {},
  getItem(k) { return this._data[k] || null; },
  setItem(k, v) { this._data[k] = v; },
};

// Load Ultra Engine (recommendation-engine.js)
eval(fs.readFileSync(path.join(__dirname, '../webapp/js/recommendation-engine.js'), 'utf8'));

const ENGINE = global.window.RecommendationEngine;
const DATA_DIR = path.join(__dirname, '../webapp/data');
const OUTPUT_DIR = path.join(__dirname, '../output');
const SIGNALS_FILE = path.join(DATA_DIR, 'ultra_signals.json');
const STATS_FILE = path.join(DATA_DIR, 'ultra_backtest_stats.json');
const RECS_FILE = path.join(DATA_DIR, 'ultra_recommendations.json');

const ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8';
const ODDS_API_BASE = 'https://api.the-odds-api.com/v4';

const TEAM_MAP = {
  'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
  'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
  'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
  'Golden State Warriors': 'GS', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
  'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',
  'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
  'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
  'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NO',
  'New York Knicks': 'NY', 'Oklahoma City Thunder': 'OKC',
  'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
  'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
  'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SA',
  'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTAH',
  'Washington Wizards': 'WSH',
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
  console.log('ULTRA BETTING ENGINE v1.0 — Seed Live Picks');
  console.log('='.repeat(60));

  // Load optimized config if available
  try {
    const configPath = path.join(OUTPUT_DIR, 'ultra_engine_config.json');
    if (fs.existsSync(configPath)) {
      const configData = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      ENGINE.loadConfig(configData);
    }
  } catch (e) { /* use defaults */ }

  // Build player model from all box scores
  const boxScores = [...loadJSON('player_boxscores.json'), ...loadJSON('player_boxscores_2026.json')];
  console.log(`Loaded ${boxScores.length} box score records`);

  ENGINE.PlayerModel.reset();
  const sortedGames = [...boxScores].sort((a, b) => (a.date || '').localeCompare(b.date || ''));
  for (const game of sortedGames) {
    for (const p of (game.players || [])) {
      const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
      if (mins < 5) continue;
      ENGINE.PlayerModel.update(p.name, {
        pts: p.pts || 0,
        reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
        ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
        min: mins,
      }, game.date, p.team, game.home === p.team ? game.away : game.home);
    }
  }
  console.log(`Player model built with ${Object.keys(ENGINE.PlayerModel.profiles).length} players`);

  // Fetch tonight's games and props from Odds API
  console.log('\nFetching tonight\'s games from Odds API...');
  let events = [];
  try {
    const eventsUrl = `${ODDS_API_BASE}/sports/basketball_nba/events?apiKey=${ODDS_API_KEY}`;
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

  // Fetch player props for each game
  const liveOdds = { events, playerProps: {} };

  for (const event of events) {
    try {
      const markets = 'player_points_alternate,player_rebounds_alternate,player_assists_alternate,player_points_rebounds_assists_alternate';
      const propsUrl = `${ODDS_API_BASE}/sports/basketball_nba/events/${event.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=${markets}&oddsFormat=american&bookmakers=fanduel`;
      const propsRes = await fetch(propsUrl);
      if (!propsRes.ok) continue;

      const propsData = await propsRes.json();
      const homeAbbr = teamAbbr(event.home_team);
      const awayAbbr = teamAbbr(event.away_team);
      const gameKey = `${awayAbbr}@${homeAbbr}`;

      const fd = propsData.bookmakers && propsData.bookmakers.find(b => b.key === 'fanduel');
      if (!fd) continue;

      const gameProps = { lines: {}, rebLines: {}, astLines: {}, praLines: {} };
      const marketMap = {
        'player_points_alternate': 'lines',
        'player_rebounds_alternate': 'rebLines',
        'player_assists_alternate': 'astLines',
        'player_points_rebounds_assists_alternate': 'praLines',
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
      console.log(`  ${gameKey}: ${Object.keys(gameProps.lines).length} PTS, ${Object.keys(gameProps.rebLines).length} REB, ${Object.keys(gameProps.astLines).length} AST, ${Object.keys(gameProps.praLines).length} PRA props`);
    } catch (e) {
      console.warn(`  Error fetching props for ${event.id}:`, e.message);
    }
    sleep(200); // Rate limit
  }

  if (Object.keys(liveOdds.playerProps).length === 0) {
    console.log('\nNo FanDuel player props available. Props typically open 1-2 hours before game time.');
    return;
  }

  // Generate recommendations using Ultra Engine
  console.log('\nRunning Ultra Engine signal analysis...');
  const recommendation = ENGINE.generateTodayPicks(liveOdds);

  if (!recommendation || (recommendation.singles.length === 0 && recommendation.parlays.length === 0)) {
    console.log('No bets meet Ultra Engine quality thresholds tonight.');
    return;
  }

  // Format and save signals
  const today = getDateStr(0);
  const newSignals = ENGINE.formatSignalForStorage(recommendation, today);

  // Load existing signals and merge
  let existingSignals = [];
  try {
    existingSignals = JSON.parse(fs.readFileSync(SIGNALS_FILE, 'utf8'));
  } catch (e) { /* no existing signals */ }

  // Remove today's existing signals (avoid duplicates)
  existingSignals = existingSignals.filter(s => s.date !== today);
  existingSignals.push(...newSignals);

  fs.writeFileSync(SIGNALS_FILE, JSON.stringify(existingSignals, null, 2));

  // Save recommendations — store the full recommendation object so the webapp
  // can render directly without re-fetching live odds (ensures consistency)
  const recsOutput = {
    generated: new Date().toISOString(),
    date: today,
    engine: 'Ultra Engine v1.0',
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
  console.log('TONIGHT\'S ULTRA ENGINE PICKS:');
  console.log('='.repeat(60));

  for (const s of recommendation.singles) {
    const typeLabel = s.cascadeScore >= ENGINE.CONFIG.SINGLE_MIN_SCORE ? 'SINGLE' : 'MULTI_SINGLE';
    console.log(`  ${typeLabel}: ${s.player} O${s.line} ${(s.statLabel || 'PTS').toUpperCase()} | Odds: ${ENGINE.formatOdds(s.odds)} | Score: ${(s.cascadeScore * 100).toFixed(0)}%`);
  }
  for (const p of recommendation.parlays) {
    const legs = p.legs.map(l => `${l.player} O${l.line} ${(l.statLabel || 'PTS').toUpperCase()}`).join(' + ');
    console.log(`  PARLAY: ${legs} | Odds: ${ENGINE.formatOdds(p.odds)}`);
  }

  console.log(`\nSingles: ${recommendation.singles.length}`);
  console.log(`Parlays: ${recommendation.parlays.length}`);
  console.log(`\nSaved to ${SIGNALS_FILE}`);

  // Check API quota
  try {
    const quotaResult = execSync(`curl -s -I "${ODDS_API_BASE}/sports/basketball_nba/events?apiKey=${ODDS_API_KEY}" 2>/dev/null | grep -i x-requests-remaining`, { timeout: 10000 });
    console.log(`\nOdds API: ${quotaResult.toString().trim()}`);
  } catch (e) { /* skip */ }
}

main().catch(console.error);
