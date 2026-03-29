/**
 * Vercel Cron Handler — V8 CSPE & Positive Odds Picks
 * Runs Python pick scripts via child_process, then commits results to GitHub.
 *
 * Handles 4 cron jobs via query params:
 *   /api/cron/picks?strategy=v8&sport=nba
 *   /api/cron/picks?strategy=v8&sport=mlb
 *   /api/cron/picks?strategy=positive-odds&sport=nba
 *   /api/cron/picks?strategy=positive-odds&sport=mlb
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const CONFIGS = {
  'v8|nba': {
    script: 'seed_live_v8_picks',
    dataSubdir: 'v8-mgcc/data',
    files: { SIGNALS_FILE: 'nba_v8_signals.json', STATS_FILE: 'nba_v8_stats.json', RECS_FILE: 'nba_v8_recommendations.json' },
    commitPrefix: 'V8 CSPE NBA',
  },
  'v8|mlb': {
    script: 'seed_live_v8_picks_mlb',
    dataSubdir: 'v8-mgcc/data',
    files: { SIGNALS_FILE: 'mlb_v8_signals.json', STATS_FILE: 'mlb_v8_stats.json', RECS_FILE: 'mlb_v8_recommendations.json' },
    commitPrefix: 'V8 CSPE MLB',
  },
  'positive-odds|nba': {
    script: 'seed_live_picks',
    dataSubdir: 'positive-odds/data',
    files: { SIGNALS_FILE: 'nba_signals.json', STATS_FILE: 'nba_stats.json', RECS_FILE: 'nba_recommendations.json' },
    commitPrefix: 'Positive Odds NBA',
  },
  'positive-odds|mlb': {
    script: 'seed_live_picks_mlb',
    dataSubdir: 'positive-odds/data',
    files: { SIGNALS_FILE: 'mlb_signals.json', STATS_FILE: 'mlb_stats.json', RECS_FILE: 'mlb_recommendations.json' },
    commitPrefix: 'Positive Odds MLB',
  },
};

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

export default async function handler(req, res) {
  const auth = req.headers['authorization'];
  if (process.env.CRON_SECRET && auth !== `Bearer ${process.env.CRON_SECRET}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const { strategy, sport } = req.query;
  if (!strategy || !sport) {
    return res.status(400).json({ error: 'strategy and sport query params required' });
  }

  const config = CONFIGS[`${strategy}|${sport}`];
  if (!config) {
    return res.status(400).json({ error: `Unknown strategy/sport: ${strategy}/${sport}` });
  }

  try {
    const result = await runCron(config);
    return res.status(200).json(result);
  } catch (err) {
    return res.status(500).json({ error: err.message, stderr: err.stderr || '', stack: err.stack });
  }
};

async function runCron(config) {
  const projectRoot = findProjectRoot();
  const runnerPath = path.join(projectRoot, 'experiments', 'scripts', '_vercel_runner.py');
  const tmpData = `/tmp/cron-${config.script}`;

  // Find Python binary
  let pythonBin = 'python3';
  for (const p of ['/var/lang/bin/python3', '/usr/bin/python3', '/usr/local/bin/python3']) {
    if (fs.existsSync(p)) { pythonBin = p; break; }
  }

  // Run the Python runner script
  const env = {
    ...process.env,
    TZ: 'America/New_York',
    CRON_SCRIPT: config.script,
    CRON_DATA_SUBDIR: config.dataSubdir,
    CRON_FILES: JSON.stringify(config.files),
    CRON_OUTPUT_DIR: tmpData,
  };

  let stdout;
  try {
    stdout = execSync(`${pythonBin} "${runnerPath}"`, {
      env,
      timeout: 250000,
      maxBuffer: 50 * 1024 * 1024,
      cwd: projectRoot,
    }).toString();
  } catch (err) {
    throw new Error(`Python script failed: ${err.stderr?.toString() || err.message}`);
  }

  // Collect output files and commit to GitHub
  const outputFiles = [];
  for (const [attr, fname] of Object.entries(config.files)) {
    const fpath = path.join(tmpData, fname);
    if (fs.existsSync(fpath)) {
      outputFiles.push([`webapp/${config.dataSubdir}/${fname}`, fs.readFileSync(fpath, 'utf8')]);
    }
  }

  let commitSha = null;
  if (outputFiles.length > 0) {
    const now = new Date().toLocaleString('en-US', { timeZone: 'America/New_York' });
    commitSha = await commitToGithub(outputFiles, `${config.commitPrefix} picks: ${now}`);
  }

  return {
    status: 'ok',
    script: config.script,
    files: outputFiles.length,
    commit: commitSha,
    output: stdout.substring(0, 2000),
  };
}
