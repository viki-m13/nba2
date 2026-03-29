/**
 * Vercel Cron Handler — MLB ULTRA BETTING ENGINE v3.0
 * Adapted from mlb/scripts/seed-live-picks-mlb.js for Vercel serverless.
 * Uses native fetch() instead of curl, writes to /tmp, commits to GitHub.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
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

function getDateStr(daysAgo = 0) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
}

export default async function handler(req, res) {
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

  // MLB data lives in mlb/webapp/data/ (used by the seed script)
  const MLB_DATA_DIR = path.join(PROJECT_ROOT, 'mlb', 'webapp', 'data');
  // Signals/recs go to webapp/data/ (served by the webapp)
  const DEPLOYED_DATA = path.join(PROJECT_ROOT, 'webapp', 'data');
  const TMP_DATA = '/tmp/ultra-mlb';
  fs.mkdirSync(TMP_DATA, { recursive: true });

  // Copy existing signal files to /tmp
  for (const f of ['mlb_ultra_signals_v3.json', 'mlb_ultra_backtest_stats_v3.json', 'mlb_ultra_recommendations.json']) {
    const src = path.join(DEPLOYED_DATA, f);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(TMP_DATA, f));
  }

  const SIGNALS_FILE = path.join(TMP_DATA, 'mlb_ultra_signals_v3.json');
  const STATS_FILE = path.join(TMP_DATA, 'mlb_ultra_backtest_stats_v3.json');
  const RECS_FILE = path.join(TMP_DATA, 'mlb_ultra_recommendations.json');

  // Setup browser shims
  global.window = global;
  global.document = { readyState: 'complete', addEventListener: () => {}, querySelectorAll: () => [] };
  global.localStorage = { _data: {}, getItem(k) { return this._data[k] || null; }, setItem(k, v) { this._data[k] = v; } };

  // Load MLB Ultra Engine v3
  const enginePath = path.join(PROJECT_ROOT, 'mlb', 'webapp', 'js', 'recommendation-engine-mlb-v3.js');
  eval(fs.readFileSync(enginePath, 'utf8'));
  const ENGINE = global.window.MLBRecommendationEngineV3;

  const ODDS_API_KEY = process.env.ODDS_API_KEY || '3879c3373a31421d8ef7d428b8758cd8';
  const ODDS_API_BASE = 'https://api.the-odds-api.com/v4';

  function loadJSON(filename, dir) {
    try { return JSON.parse(fs.readFileSync(path.join(dir || MLB_DATA_DIR, filename), 'utf8')); }
    catch (e) { return []; }
  }

  // --- STEP 0: Resolve pending picks ---
  let signals = [];
  try { signals = JSON.parse(fs.readFileSync(SIGNALS_FILE, 'utf8')); } catch (e) {}

  const pending = signals.filter(s => s.hit === null || s.hit === undefined);
  let resolvedCount = 0;

  if (pending.length > 0) {
    const pendingDates = [...new Set(pending.map(s => s.date))].sort();
    for (const date of pendingDates) {
      try {
        const sbRes = await fetch(`https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=${date}`);
        if (!sbRes.ok) continue;
        const sbData = await sbRes.json();
        if (!sbData.events?.length) continue;

        const playerStats = {};
        for (const event of sbData.events) {
          if (!event.status?.type?.completed) continue;
          try {
            const boxRes = await fetch(`https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event=${event.id}`);
            if (!boxRes.ok) continue;
            const boxData = await boxRes.json();
            const rosters = boxData.rosters || boxData.boxscore?.players || [];
            for (const team of rosters) {
              for (const cat of (team.statistics || [])) {
                if (cat.type !== 'batting' && cat.name !== 'batting') continue;
                const labels = (cat.labels || []).map(l => l.toLowerCase());
                for (const ath of (cat.athletes || [])) {
                  const name = ath.athlete?.displayName;
                  if (!name) continue;
                  const s = ath.stats || [];
                  const so = {};
                  labels.forEach((l, i) => { so[l] = s[i]; });
                  const h = parseInt(so.h) || 0;
                  const doubles = parseInt(so['2b']) || 0, triples = parseInt(so['3b']) || 0;
                  const hr = parseInt(so.hr) || 0, rbi = parseInt(so.rbi) || 0, r = parseInt(so.r) || 0;
                  playerStats[name] = { h, tb: h + doubles + triples * 2 + hr * 3, rbi, r, hr };
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
            const actual = ps[signal.stat || 'h'] || 0;
            signal.actual = actual;
            signal.hit = actual > signal.line;
            const dec = signal.odds > 0 ? signal.odds / 100 + 1 : 100 / Math.abs(signal.odds) + 1;
            signal.pnl = signal.hit ? Math.round((dec - 1) * 100) : -100;
            resolvedCount++;
          } else if (signal.betType === 'parlay' && signal.legs) {
            let allHit = true, anyRes = false;
            for (const leg of signal.legs) {
              if (leg.hit != null) { if (!leg.hit) allHit = false; continue; }
              const ps = playerStats[leg.player];
              if (!ps) { allHit = false; continue; }
              leg.actual = ps[leg.stat || 'h'] || 0;
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
              resolvedCount++;
            }
          }
        }
      } catch (e) {}
      await sleep(200);
    }

    if (resolvedCount > 0) {
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
    const configPath = path.join(PROJECT_ROOT, 'mlb', 'output', 'mlb_ultra_engine_v3_config.json');
    if (fs.existsSync(configPath)) ENGINE.loadConfig(JSON.parse(fs.readFileSync(configPath, 'utf8')));
  } catch (e) {}

  // --- STEP 2: Build player model ---
  const boxScores = loadJSON('mlb_player_boxscores.json');
  ENGINE.PlayerModel.reset();
  const sorted = [...boxScores].sort((a, b) => (a.date || '').localeCompare(b.date || ''));
  for (const game of sorted) {
    for (const p of (game.players || [])) {
      if ((p.ab || 0) < 1) continue;
      ENGINE.PlayerModel.update(p.name, {
        h: p.h || 0, tb: p.tb || 0, rbi: p.rbi || 0, r: p.r || 0,
        hr: p.hr || 0, ab: p.ab || 0, bb: p.bb || 0, so: p.so || 0, sb: p.sb || 0,
      }, game.date, p.team, game.home === p.team ? game.away : game.home);
    }
  }

  // --- STEP 3: Fetch live odds (regular + preseason) ---
  let events = [];
  try {
    const [regRes, preRes] = await Promise.all([
      fetch(`${ODDS_API_BASE}/sports/baseball_mlb/events?apiKey=${ODDS_API_KEY}`).catch(() => ({ ok: false })),
      fetch(`${ODDS_API_BASE}/sports/baseball_mlb_preseason/events?apiKey=${ODDS_API_KEY}`).catch(() => ({ ok: false })),
    ]);
    if (regRes.ok) events.push(...(await regRes.json()));
    if (preRes.ok) {
      const pre = await preRes.json();
      for (const e of pre) e._sportKey = 'baseball_mlb_preseason';
      events.push(...pre);
    }
  } catch (e) {}

  const today = getDateStr();

  if (!events.length) {
    writeEmptyRecs(RECS_FILE, today, 'No games scheduled tonight');
    return { status: 'ok', message: 'No games tonight' };
  }

  const liveOdds = { events, playerProps: {} };
  for (const event of events) {
    try {
      const sportKey = event._sportKey || 'baseball_mlb';
      const markets = 'batter_hits_alternate,batter_total_bases_alternate,batter_rbis_alternate,batter_runs_scored_alternate';
      const propsRes = await fetch(`${ODDS_API_BASE}/sports/${sportKey}/events/${event.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=${markets}&oddsFormat=american&bookmakers=fanduel`);
      if (!propsRes.ok) continue;
      const propsData = await propsRes.json();
      const gameKey = `${teamAbbr(event.away_team)}@${teamAbbr(event.home_team)}`;
      const fd = propsData.bookmakers?.find(b => b.key === 'fanduel');
      if (!fd) continue;

      const gameProps = { hitsLines: {}, tbLines: {}, rbiLines: {}, runsLines: {} };
      const mm = {
        batter_hits_alternate: 'hitsLines',
        batter_total_bases_alternate: 'tbLines',
        batter_rbis_alternate: 'rbiLines',
        batter_runs_scored_alternate: 'runsLines',
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
    writeEmptyRecs(RECS_FILE, today, 'No FanDuel batter props available yet');
    return { status: 'ok', message: 'No FanDuel props available' };
  }

  // --- STEP 4: Generate picks ---
  const recommendation = ENGINE.generateTodayPicks(liveOdds);
  if (!recommendation || (!recommendation.singles.length && !recommendation.parlays.length)) {
    writeEmptyRecs(RECS_FILE, today, 'No bets meet quality thresholds tonight');
    return { status: 'ok', message: 'No picks meet thresholds' };
  }

  const newSignals = ENGINE.formatSignalForStorage(recommendation, today);
  for (const sig of newSignals) sig.source = 'live';

  // Merge signals
  let existingSignals = [];
  try { existingSignals = JSON.parse(fs.readFileSync(SIGNALS_FILE, 'utf8')); } catch (e) {}

  existingSignals = existingSignals.filter(s => !(s.date === today && s.source === 'live'));
  existingSignals.push(...newSignals);
  const liveDates = new Set(existingSignals.filter(s => s.source === 'live').map(s => s.date));
  existingSignals = existingSignals.filter(s => s.source === 'live' || !liveDates.has(s.date));
  fs.writeFileSync(SIGNALS_FILE, JSON.stringify(existingSignals, null, 2));

  // Save recommendations
  fs.writeFileSync(RECS_FILE, JSON.stringify({
    generated: new Date().toISOString(), date: today,
    engine: 'MLB Ultra Engine v3.0', config_version: 'optimized',
    recommendation: {
      betType: recommendation.betType, reasoning: recommendation.reasoning,
      singles: recommendation.singles.map(s => ({
        player: s.player, team: s.team, statType: s.statType, statLabel: s.statLabel,
        line: s.line, odds: s.odds, cascadeScore: s.cascadeScore,
        gft: s.gft, beq: s.beq, esi: s.esi, imad: s.imad,
        hitRate: s.hitRate, edge: s.edge, ev: s.ev, avg: s.avg, floor: s.floor,
        betSubType: s.cascadeScore >= ENGINE.CONFIG.SINGLE_MIN_SCORE ? 'single' : 'multi_single',
        hit: null, actual: null,
      })),
      parlays: recommendation.parlays.map(p => ({
        numLegs: p.numLegs, odds: p.odds, decimalOdds: p.decimalOdds,
        avgCascade: p.avgCascade, combinedHitRate: p.combinedHitRate, ev: p.ev,
        legs: p.legs.map(l => ({
          player: l.player, team: l.team, statType: l.statType, statLabel: l.statLabel,
          line: l.line, odds: l.odds, cascadeScore: l.cascadeScore,
          gft: l.gft, beq: l.beq, esi: l.esi, imad: l.imad,
          hitRate: l.hitRate, edge: l.edge, hit: null, actual: null,
        })),
      })),
    },
  }, null, 2));

  // --- STEP 5: Commit to GitHub ---
  const outputFiles = [];
  for (const f of ['mlb_ultra_signals_v3.json', 'mlb_ultra_backtest_stats_v3.json', 'mlb_ultra_recommendations.json']) {
    const fp = path.join(TMP_DATA, f);
    if (fs.existsSync(fp)) outputFiles.push([`webapp/data/${f}`, fs.readFileSync(fp, 'utf8')]);
  }

  const commitSha = outputFiles.length > 0
    ? await commitToGithub(outputFiles, `Ultra MLB picks: ${new Date().toISOString()}`)
    : null;

  return {
    status: 'ok',
    singles: recommendation.singles.length,
    parlays: recommendation.parlays.length,
    resolved: resolvedCount,
    commit: commitSha,
  };
}

function writeEmptyRecs(recsFile, today, reasoning) {
  fs.writeFileSync(recsFile, JSON.stringify({
    generated: new Date().toISOString(), date: today,
    engine: 'MLB Ultra Engine v3.0', config_version: 'optimized',
    recommendation: { betType: 'none', reasoning, singles: [], parlays: [] },
  }, null, 2));
}
