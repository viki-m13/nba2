/**
 * Vercel Cron Handler — ULTRA BETTING ENGINE v1.0 (NBA)
 * Adapted from scripts/seed-live-picks.js for Vercel serverless.
 * Uses native fetch() instead of curl, writes to /tmp, commits to GitHub.
 */

const fs = require('fs');
const path = require('path');

const sleep = ms => new Promise(r => setTimeout(r, ms));

function findProjectRoot() {
  let dir = __dirname;
  for (let i = 0; i < 10; i++) {
    if (fs.existsSync(path.join(dir, 'experiments'))) return dir;
    dir = path.dirname(dir);
  }
  return process.cwd();
}

async function commitToGithub(files, message) {
  const token = process.env.GITHUB_TOKEN;
  const repo = process.env.GITHUB_REPO || 'viki-m13/nba2';
  const branch = process.env.GITHUB_BRANCH || 'main';
  if (!token) return null;

  const baseUrl = `https://api.github.com/repos/${repo}`;
  const hdrs = {
    'Authorization': `token ${token}`,
    'Accept': 'application/vnd.github.v3+json',
    'Content-Type': 'application/json',
    'User-Agent': 'vercel-cron',
  };

  async function api(ep, data, method = 'GET') {
    const res = await fetch(`${baseUrl}${ep}`, {
      method, headers: hdrs,
      body: data ? JSON.stringify(data) : undefined,
    });
    return res.json();
  }

  const ref = await api(`/git/ref/heads/${branch}`);
  const headSha = ref.object.sha;
  const commitData = await api(`/git/commits/${headSha}`);

  const treeItems = [];
  for (const [repoPath, content] of files) {
    const blob = await api('/git/blobs', { content, encoding: 'utf-8' }, 'POST');
    treeItems.push({ path: repoPath, mode: '100644', type: 'blob', sha: blob.sha });
  }

  const newTree = await api('/git/trees', {
    base_tree: commitData.tree.sha, tree: treeItems,
  }, 'POST');
  const newCommit = await api('/git/commits', {
    message, tree: newTree.sha, parents: [headSha],
  }, 'POST');
  await api(`/git/refs/heads/${branch}`, { sha: newCommit.sha }, 'PATCH');
  return newCommit.sha;
}

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

function getDateStr(daysAgo = 0) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
}

module.exports = async function handler(req, res) {
  const auth = req.headers['authorization'];
  if (process.env.CRON_SECRET && auth !== `Bearer ${process.env.CRON_SECRET}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  try {
    const result = await runCron();
    return res.status(200).json(result);
  } catch (err) {
    return res.status(500).json({ error: err.message, stack: err.stack });
  }
};

async function runCron() {
  const PROJECT_ROOT = findProjectRoot();
  const DEPLOYED_DATA = path.join(PROJECT_ROOT, 'webapp', 'data');
  const TMP_DATA = '/tmp/ultra-nba';
  fs.mkdirSync(TMP_DATA, { recursive: true });

  // Copy existing data to /tmp
  for (const f of ['ultra_signals.json', 'ultra_backtest_stats.json', 'ultra_recommendations.json']) {
    const src = path.join(DEPLOYED_DATA, f);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(TMP_DATA, f));
  }

  const SIGNALS_FILE = path.join(TMP_DATA, 'ultra_signals.json');
  const STATS_FILE = path.join(TMP_DATA, 'ultra_backtest_stats.json');
  const RECS_FILE = path.join(TMP_DATA, 'ultra_recommendations.json');

  // Setup browser shims for Ultra Engine
  global.window = global;
  global.document = { readyState: 'complete', addEventListener: () => {}, querySelectorAll: () => [] };
  global.localStorage = { _data: {}, getItem(k) { return this._data[k] || null; }, setItem(k, v) { this._data[k] = v; } };

  // Load Ultra Engine
  const enginePath = path.join(PROJECT_ROOT, 'webapp', 'js', 'recommendation-engine.js');
  eval(fs.readFileSync(enginePath, 'utf8'));
  const ENGINE = global.window.RecommendationEngine;

  const ODDS_API_KEY = process.env.ODDS_API_KEY || '3879c3373a31421d8ef7d428b8758cd8';
  const ODDS_API_BASE = 'https://api.the-odds-api.com/v4';

  // --- STEP 0: Resolve pending picks ---
  let signals = [];
  try { signals = JSON.parse(fs.readFileSync(SIGNALS_FILE, 'utf8')); } catch (e) {}

  const pending = signals.filter(s => s.hit === null || s.hit === undefined);
  let resolved = 0;

  if (pending.length > 0) {
    const pendingDates = [...new Set(pending.map(s => s.date))].sort();
    for (const date of pendingDates) {
      try {
        const sbRes = await fetch(`https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=${date}`);
        if (!sbRes.ok) continue;
        const sbData = await sbRes.json();
        if (!sbData.events?.length) continue;

        const playerStats = {};
        for (const event of sbData.events) {
          if (!event.status?.type?.completed) continue;
          try {
            const boxRes = await fetch(`https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=${event.id}`);
            if (!boxRes.ok) continue;
            const boxData = await boxRes.json();
            for (const team of (boxData.boxscore?.players || [])) {
              for (const sg of (team.statistics || [])) {
                const headers = (sg.labels || []).map(l => l.toLowerCase());
                const pi = headers.indexOf('pts'), ri = headers.indexOf('reb'), ai = headers.indexOf('ast');
                for (const ath of (sg.athletes || [])) {
                  const name = ath.athlete?.displayName;
                  if (!name) continue;
                  const s = ath.stats || [];
                  const pts = parseInt(s[pi]) || 0, reb = parseInt(s[ri]) || 0, ast = parseInt(s[ai]) || 0;
                  playerStats[name] = { pts, reb, ast, pra: pts + reb + ast };
                }
              }
            }
          } catch (e) {}
          await sleep(100);
        }

        for (const signal of signals) {
          if (signal.date !== date || signal.hit != null) continue;
          if (signal.betType === 'single') {
            const ps = playerStats[signal.player];
            if (!ps) continue;
            const actual = ps[signal.stat || 'pts'] || 0;
            signal.actual = actual;
            signal.hit = actual > signal.line;
            const dec = signal.odds > 0 ? signal.odds / 100 + 1 : 100 / Math.abs(signal.odds) + 1;
            signal.pnl = signal.hit ? Math.round((dec - 1) * 100) : -100;
            resolved++;
          } else if (signal.betType === 'parlay' && signal.legs) {
            let allHit = true, anyRes = false;
            for (const leg of signal.legs) {
              if (leg.hit != null) { if (!leg.hit) allHit = false; continue; }
              const ps = playerStats[leg.player];
              if (!ps) { allHit = false; continue; }
              leg.actual = ps[leg.stat || 'pts'] || 0;
              leg.hit = leg.actual > leg.line;
              if (!leg.hit) allHit = false;
              anyRes = true;
            }
            if (anyRes && signal.legs.every(l => l.hit != null)) {
              signal.hit = allHit;
              const pd = signal.parlay_decimal || signal.legs.reduce((d, l) => {
                return d * (l.odds > 0 ? l.odds / 100 + 1 : 100 / Math.abs(l.odds) + 1);
              }, 1);
              signal.pnl = allHit ? Math.round((pd - 1) * 100) : -100;
              resolved++;
            }
          }
        }
      } catch (e) {}
      await sleep(200);
    }

    if (resolved > 0) {
      fs.writeFileSync(SIGNALS_FILE, JSON.stringify(signals, null, 2));
      const sgl = signals.filter(s => s.betType === 'single' && s.hit != null);
      const par = signals.filter(s => s.betType === 'parlay' && s.hit != null);
      const sw = sgl.filter(s => s.hit).length, pw = par.filter(s => s.hit).length;
      const sp = sgl.reduce((a, s) => a + (s.pnl || 0), 0), pp = par.reduce((a, s) => a + (s.pnl || 0), 0);
      const swg = sgl.length * 100, pwg = par.length * 100;
      const tl = par.reduce((a, s) => a + (s.legs?.length || 0), 0);
      const hl = par.reduce((a, s) => a + (s.legs?.filter(l => l.hit).length || 0), 0);
      fs.writeFileSync(STATS_FILE, JSON.stringify({
        singles: { total: sgl.length, wins: sw, accuracy: sgl.length > 0 ? sw / sgl.length : 0, pnl: sp, wagered: swg, roi: swg > 0 ? sp / swg : 0 },
        parlays: { total: par.length, wins: pw, accuracy: par.length > 0 ? pw / par.length : 0, pnl: pp, wagered: pwg, roi: pwg > 0 ? pp / pwg : 0, totalLegs: tl, hitLegs: hl, legAccuracy: tl > 0 ? hl / tl : 0 },
        overall: { total: sgl.length + par.length, wins: sw + pw, accuracy: (sgl.length + par.length) > 0 ? (sw + pw) / (sgl.length + par.length) : 0, pnl: sp + pp, wagered: swg + pwg, roi: (swg + pwg) > 0 ? (sp + pp) / (swg + pwg) : 0 },
      }, null, 2));
    }
  }

  // --- STEP 1: Load config ---
  try {
    const configPath = path.join(PROJECT_ROOT, 'output', 'ultra_engine_config.json');
    if (fs.existsSync(configPath)) ENGINE.loadConfig(JSON.parse(fs.readFileSync(configPath, 'utf8')));
  } catch (e) {}

  // --- STEP 2: Build player model ---
  function loadJSON(filename) {
    try { return JSON.parse(fs.readFileSync(path.join(DEPLOYED_DATA, filename), 'utf8')); }
    catch (e) { return []; }
  }

  const boxScores = [...loadJSON('player_boxscores.json'), ...loadJSON('player_boxscores_2026.json')];
  ENGINE.PlayerModel.reset();
  const sorted = [...boxScores].sort((a, b) => (a.date || '').localeCompare(b.date || ''));
  for (const game of sorted) {
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

  // --- STEP 3: Fetch live odds ---
  let events = [];
  try {
    const evRes = await fetch(`${ODDS_API_BASE}/sports/basketball_nba/events?apiKey=${ODDS_API_KEY}`);
    if (evRes.ok) events = await evRes.json();
  } catch (e) {}

  const today = getDateStr();

  if (!events.length) {
    writeRecsIfEmpty(RECS_FILE, today, 'No games scheduled tonight');
    return { status: 'ok', message: 'No games tonight' };
  }

  const now = new Date();
  const upcoming = events.filter(e => !e.commence_time || new Date(e.commence_time) > now);

  if (!upcoming.length) {
    writeRecsIfEmpty(RECS_FILE, today, 'All games have already started');
    return { status: 'ok', message: 'All games already started' };
  }

  const liveOdds = { events: upcoming, playerProps: {} };
  for (const event of upcoming) {
    try {
      const markets = 'player_points,player_rebounds,player_assists,player_points_rebounds_assists,player_points_alternate,player_rebounds_alternate,player_assists_alternate,player_points_rebounds_assists_alternate';
      const propsRes = await fetch(`${ODDS_API_BASE}/sports/basketball_nba/events/${event.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=${markets}&oddsFormat=american&bookmakers=fanduel`);
      if (!propsRes.ok) continue;
      const propsData = await propsRes.json();
      const gameKey = `${teamAbbr(event.away_team)}@${teamAbbr(event.home_team)}`;
      const fd = propsData.bookmakers?.find(b => b.key === 'fanduel');
      if (!fd) continue;

      const gameProps = { lines: {}, rebLines: {}, astLines: {}, praLines: {} };
      const mm = {
        player_points: 'lines', player_points_alternate: 'lines',
        player_rebounds: 'rebLines', player_rebounds_alternate: 'rebLines',
        player_assists: 'astLines', player_assists_alternate: 'astLines',
        player_points_rebounds_assists: 'praLines', player_points_rebounds_assists_alternate: 'praLines',
      };
      for (const mkt of (fd.markets || [])) {
        const st = mm[mkt.key];
        if (!st) continue;
        for (const o of (mkt.outcomes || [])) {
          if (o.name !== 'Over') continue;
          if (!gameProps[st][o.description]) gameProps[st][o.description] = {};
          if (!gameProps[st][o.description][o.point]) gameProps[st][o.description][o.point] = {};
          gameProps[st][o.description][o.point].overOdds = o.price;
        }
      }
      liveOdds.playerProps[gameKey] = gameProps;
    } catch (e) {}
    await sleep(200);
  }

  if (!Object.keys(liveOdds.playerProps).length) {
    writeRecsIfEmpty(RECS_FILE, today, 'No FanDuel player props available yet');
    return { status: 'ok', message: 'No FanDuel props available' };
  }

  // --- STEP 4: Generate picks ---
  const recommendation = ENGINE.generateTodayPicks(liveOdds);
  if (!recommendation || (!recommendation.singles.length && !recommendation.parlays.length)) {
    writeRecsIfEmpty(RECS_FILE, today, 'No bets meet quality thresholds tonight');
    return { status: 'ok', message: 'No picks meet thresholds' };
  }

  const newSignals = ENGINE.formatSignalForStorage(recommendation, today);
  for (const sig of newSignals) { sig.source = 'live'; sig.generated_at = new Date().toISOString(); }

  // Merge signals (accumulate, dedup)
  let existingSignals = [];
  try { existingSignals = JSON.parse(fs.readFileSync(SIGNALS_FILE, 'utf8')); } catch (e) {}

  const todayLive = existingSignals.filter(s => s.date === today && s.source === 'live');
  const existKeys = new Set();
  for (const s of todayLive) {
    if (s.betType === 'single') existKeys.add(`${s.player}|${s.line}|${s.stat || 'pts'}|single`);
    else if (s.betType === 'parlay' && s.legs) existKeys.add(`parlay|${s.legs.map(l => `${l.player}|${l.line}|${l.stat || 'pts'}`).sort().join('~')}`);
  }

  const unique = newSignals.filter(s => {
    let k;
    if (s.betType === 'single') k = `${s.player}|${s.line}|${s.stat || 'pts'}|single`;
    else if (s.betType === 'parlay' && s.legs) k = `parlay|${s.legs.map(l => `${l.player}|${l.line}|${l.stat || 'pts'}`).sort().join('~')}`;
    return !existKeys.has(k);
  });
  existingSignals.push(...unique);

  const liveDates = new Set(existingSignals.filter(s => s.source === 'live').map(s => s.date));
  existingSignals = existingSignals.filter(s => s.source === 'live' || !liveDates.has(s.date));
  fs.writeFileSync(SIGNALS_FILE, JSON.stringify(existingSignals, null, 2));

  // Save recommendations (accumulate across runs)
  let existingRecs = { recommendation: { singles: [], parlays: [] } };
  try {
    const prev = JSON.parse(fs.readFileSync(RECS_FILE, 'utf8'));
    if (prev.date === today) existingRecs = prev;
  } catch (e) {}

  const prevSingles = existingRecs.recommendation.singles || [];
  const prevParlays = existingRecs.recommendation.parlays || [];
  const recKeys = new Set();
  for (const s of prevSingles) recKeys.add(`${s.player}|${s.line}|${s.statType || 'pts'}|single`);
  for (const p of prevParlays) recKeys.add(`parlay|${(p.legs || []).map(l => `${l.player}|${l.line}|${l.statType || 'pts'}`).sort().join('~')}`);

  const allSingles = [...prevSingles, ...recommendation.singles.filter(s => {
    return !recKeys.has(`${s.player}|${s.line}|${s.statType || 'pts'}|single`);
  }).map(s => ({
    player: s.player, team: s.team, statType: s.statType, statLabel: s.statLabel,
    line: s.line, odds: s.odds, cascadeScore: s.cascadeScore,
    gft: s.gft, beq: s.beq, esi: s.esi, imad: s.imad,
    hitRate: s.hitRate, edge: s.edge, ev: s.ev, avg: s.avg, floor: s.floor,
    betSubType: s.cascadeScore >= ENGINE.CONFIG.SINGLE_MIN_SCORE ? 'single' : 'multi_single',
    hit: null, actual: null,
  }))];

  const allParlays = [...prevParlays, ...recommendation.parlays.filter(p => {
    const lk = (p.legs || []).map(l => `${l.player}|${l.line}|${l.statType || 'pts'}`).sort().join('~');
    return !recKeys.has(`parlay|${lk}`);
  }).map(p => ({
    numLegs: p.numLegs, odds: p.odds, decimalOdds: p.decimalOdds,
    avgCascade: p.avgCascade, combinedHitRate: p.combinedHitRate, ev: p.ev,
    legs: p.legs.map(l => ({
      player: l.player, team: l.team, statType: l.statType, statLabel: l.statLabel,
      line: l.line, odds: l.odds, cascadeScore: l.cascadeScore,
      gft: l.gft, beq: l.beq, esi: l.esi, imad: l.imad,
      hitRate: l.hitRate, edge: l.edge, hit: null, actual: null,
    })),
  }))];

  fs.writeFileSync(RECS_FILE, JSON.stringify({
    generated: new Date().toISOString(), date: today,
    engine: 'Ultra Engine v1.0', config_version: 'optimized',
    recommendation: {
      betType: recommendation.betType, reasoning: recommendation.reasoning,
      singles: allSingles, parlays: allParlays,
    },
  }, null, 2));

  // --- STEP 5: Commit to GitHub ---
  const outputFiles = [];
  for (const f of ['ultra_signals.json', 'ultra_backtest_stats.json', 'ultra_recommendations.json']) {
    const fp = path.join(TMP_DATA, f);
    if (fs.existsSync(fp)) outputFiles.push([`webapp/data/${f}`, fs.readFileSync(fp, 'utf8')]);
  }

  const commitSha = outputFiles.length > 0
    ? await commitToGithub(outputFiles, `Ultra NBA picks: ${new Date().toISOString()}`)
    : null;

  return {
    status: 'ok',
    singles: recommendation.singles.length,
    parlays: recommendation.parlays.length,
    resolved,
    commit: commitSha,
  };
}

function writeRecsIfEmpty(recsFile, today, reasoning) {
  let existing = null;
  try { existing = JSON.parse(fs.readFileSync(recsFile, 'utf8')); } catch (e) {}
  if (existing && existing.date === today &&
      ((existing.recommendation.singles || []).length > 0 ||
       (existing.recommendation.parlays || []).length > 0)) {
    return; // Keep existing picks from earlier runs
  }
  fs.writeFileSync(recsFile, JSON.stringify({
    generated: new Date().toISOString(), date: today,
    engine: 'Ultra Engine v1.0', config_version: 'optimized',
    recommendation: { betType: 'none', reasoning, singles: [], parlays: [] },
  }, null, 2));
}
