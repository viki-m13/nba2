#!/usr/bin/env node
// =============================================================================
// Seed Live Picks — Fetches historical odds, generates picks, resolves via ESPN
// Outputs to webapp/data/live_picks_2026.json
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

// --- Shim browser globals ---
global.window = global;
global.localStorage = {
  _data: {},
  getItem(k) { return this._data[k] || null; },
  setItem(k, v) { this._data[k] = v; },
};

// Load engine files
eval(fs.readFileSync(path.join(__dirname, '../webapp/js/engine.js'), 'utf8'));
eval(fs.readFileSync(path.join(__dirname, '../webapp/js/nba-api.js'), 'utf8'));

const DATA_DIR = path.join(__dirname, '../webapp/data');
const OUTPUT_FILE = path.join(DATA_DIR, 'live_picks_2026.json');

function loadJSON(filename) {
  try { return JSON.parse(fs.readFileSync(path.join(DATA_DIR, filename), 'utf8')); }
  catch (e) { return []; }
}

function loadExistingPicks() {
  try { return JSON.parse(fs.readFileSync(OUTPUT_FILE, 'utf8')); }
  catch (e) { return { parlays: [] }; }
}

function getDateStr(daysAgo) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}`;
}

async function main() {
  const numDays = parseInt(process.argv[2]) || 5;
  console.log(`\n=== Seeding live picks for last ${numDays} days ===\n`);

  // Build model from box scores
  const playerBoxScores = loadJSON('player_boxscores.json');
  console.log(`Loaded ${playerBoxScores.length} box score records`);

  const sorted = [...playerBoxScores].sort((a, b) => a.date.localeCompare(b.date));
  const model = Object.create(window.BettingEngine.PlayerModel);
  model.history = {};
  for (const g of sorted) {
    for (const p of (g.players || [])) {
      const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
      if (mins < 10) continue;
      model.update(p.name, {
        pts: p.pts,
        reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
        ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
        min: mins,
      }, g.date, p.team);
    }
  }
  console.log(`Model built with ${Object.keys(model.history).length} players`);

  // Load existing picks
  const existing = loadExistingPicks();
  const existingDates = new Set((existing.parlays || []).map(p => p.date));

  // Determine dates to seed
  const datesToSeed = [];
  for (let i = 1; i <= numDays; i++) {
    const ds = getDateStr(i);
    if (!existingDates.has(ds)) datesToSeed.push(ds);
  }

  if (datesToSeed.length === 0) {
    console.log('All dates already seeded.\n');
  } else {
    console.log(`Dates to seed: ${datesToSeed.join(', ')}\n`);
  }

  for (const date of datesToSeed) {
    try {
      console.log(`[${date}] Fetching historical odds...`);
      const dayOdds = await window.BettingEngine.fetchHistoricalOddsForDate(date);
      if (!dayOdds || dayOdds.length === 0) { console.log(`[${date}] No odds, skipping`); continue; }

      console.log(`[${date}] ${dayOdds.length} games with odds`);

      const daySingles = [];
      for (const record of dayOdds) {
        const gameKey = record.gameKey;
        const gameDisplay = `${record.awayTeam} @ ${record.homeTeam}`;

        const addPick = (prop) => {
          if (!prop) return;
          if (!daySingles.find(s => s.player === prop.player && s.statType === prop.statType && s.line === prop.line)) {
            daySingles.push({ ...prop, gameKey, gameDisplay });
          }
        };

        for (const [name, lines] of Object.entries(record.playerProps || {})) addPick(model.findBestProp(name, 'points', lines));
        for (const [name, lines] of Object.entries(record.playerRebProps || {})) addPick(model.findBestProp(name, 'rebounds', lines));
        for (const [name, lines] of Object.entries(record.playerAstProps || {})) addPick(model.findBestProp(name, 'assists', lines));
      }

      console.log(`[${date}] ${daySingles.length} qualifying singles`);

      const dayParlays = window.BettingEngine.buildParlays(daySingles);
      for (const parlay of dayParlays) {
        parlay.legs = parlay.legs.map(leg => {
          const match = daySingles.find(s => s.player === leg.player && s.line === leg.line && s.statType === leg.statType);
          return { ...leg, gameKey: match ? match.gameKey : '', gameDisplay: match ? match.gameDisplay : '' };
        });
      }

      for (const p of dayParlays) {
        existing.parlays.push({
          date, numLegs: p.numLegs || p.legs.length,
          odds: p.odds, decimalOdds: p.decimalOdds,
          combinedHitRate: p.combinedHitRate, ev: p.ev,
          legs: p.legs.map(l => ({
            player: l.player, team: l.team,
            statType: l.statType || 'points',
            statLabel: l.statLabel || window.BettingEngine.statLabel(l.statType || 'points'),
            line: l.line, odds: l.odds,
            gameKey: l.gameKey || '', gameDisplay: l.gameDisplay || '',
            actual: null, won: null,
          })),
          resolved: false, won: null, pnl: null,
          savedAt: new Date().toISOString(),
        });
      }

      console.log(`[${date}] Generated ${dayParlays.length} parlays`);
      await new Promise(r => setTimeout(r, 300));
    } catch (e) {
      console.error(`[${date}] Error:`, e.message);
    }
  }

  // --- Resolve all unresolved picks ---
  console.log('\n=== Resolving picks via ESPN box scores ===\n');
  const today = getDateStr(0);
  const unresolvedDates = [...new Set(
    existing.parlays.filter(p => !p.resolved).map(p => p.date)
  )].filter(d => d < today).sort();

  for (const date of unresolvedDates) {
    try {
      console.log(`[${date}] Fetching ESPN scoreboard...`);
      const { games, eventIds } = await window.NbaApi.fetchESPNScoreboardForDate(date);
      if (!games.length) { console.log(`[${date}] No games`); continue; }

      const playerStats = {};
      let finalCount = 0;
      for (const [teamKey, eventId] of Object.entries(eventIds)) {
        const game = games.find(g => g.id === eventId || g.espnId === eventId);
        if (game && game.status !== 'final') continue;
        finalCount++;
        const boxScore = await window.NbaApi.fetchESPNBoxScore(eventId);
        if (!boxScore) continue;
        for (const p of boxScore) playerStats[p.name.toLowerCase()] = p;
        await new Promise(r => setTimeout(r, 200));
      }

      const allFinal = finalCount === Object.keys(eventIds).length;
      console.log(`[${date}] ${Object.keys(playerStats).length} players (${finalCount}/${Object.keys(eventIds).length} games final)`);

      const findPlayer = (name) => {
        const lower = name.toLowerCase();
        if (playerStats[lower]) return playerStats[lower];
        const parts = lower.split(' ');
        if (parts.length >= 2) {
          const found = Object.entries(playerStats).find(([n]) => n.includes(parts[0]) && n.includes(parts[parts.length - 1]));
          if (found) return found[1];
        }
        return null;
      };

      let resolved = 0;
      let wins = 0;
      for (const parlay of existing.parlays) {
        if (parlay.date !== date || parlay.resolved) continue;
        for (const leg of parlay.legs) {
          if (leg.actual !== null && leg.actual !== undefined) continue;
          const pStats = findPlayer(leg.player);
          if (pStats) {
            const key = leg.statType === 'rebounds' ? 'reb' : leg.statType === 'assists' ? 'ast' : 'pts';
            leg.actual = pStats[key];
            leg.won = leg.actual > leg.line;
          } else if (allFinal) {
            leg.actual = 0;
            leg.won = false;
          }
        }
        if (parlay.legs.every(l => l.actual !== null && l.actual !== undefined)) {
          parlay.resolved = true;
          parlay.won = parlay.legs.every(l => l.won);
          parlay.pnl = parlay.won ? Math.round((parlay.decimalOdds - 1) * 100) : -100;
          resolved++;
          if (parlay.won) wins++;
        }
      }

      console.log(`[${date}] Resolved: ${wins}/${resolved} won`);
    } catch (e) {
      console.error(`[${date}] Error:`, e.message);
    }
  }

  // Write output
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(existing, null, 2));
  console.log(`\n=== Saved to ${OUTPUT_FILE} ===`);

  const resolvedParlays = existing.parlays.filter(p => p.resolved);
  const totalWins = resolvedParlays.filter(p => p.won).length;
  const pnl = resolvedParlays.reduce((s, p) => s + (p.pnl || 0), 0);
  console.log(`\nSummary:`);
  console.log(`  Parlays: ${existing.parlays.length} (${resolvedParlays.length} resolved)`);
  console.log(`  Record: ${totalWins}-${resolvedParlays.length - totalWins}`);
  console.log(`  P&L: $${pnl >= 0 ? '+' : ''}${pnl}`);
  if (resolvedParlays.length > 0) console.log(`  Accuracy: ${(totalWins / resolvedParlays.length * 100).toFixed(1)}%`);

  try {
    const quotaResult = execSync('curl -s -I "https://api.the-odds-api.com/v4/sports/basketball_nba/events?apiKey=3879c3373a31421d8ef7d428b8758cd8" 2>/dev/null | grep -i x-requests-remaining', { timeout: 10000 });
    console.log(`\nOdds API: ${quotaResult.toString().trim()}`);
  } catch (e) { /* skip */ }
}

main().catch(console.error);
