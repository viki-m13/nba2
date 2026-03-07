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

// Override global fetch with curl-based version
global.fetch = async (url) => {
  const data = curlFetch(url);
  if (!data || data.error_code) {
    return {
      ok: false,
      status: data ? 400 : 500,
      headers: { get: () => null },
      json: async () => data,
    };
  }
  return {
    ok: true,
    status: 200,
    headers: { get: () => null },
    json: async () => data,
  };
};

// --- Shim browser globals ---
global.window = global;
global.localStorage = {
  _data: {},
  getItem(k) { return this._data[k] || null; },
  setItem(k, v) { this._data[k] = v; },
};

// Load engine files
const engineCode = fs.readFileSync(path.join(__dirname, '../webapp/js/engine.js'), 'utf8');
const spEngineCode = fs.readFileSync(path.join(__dirname, '../webapp/js/engine-superpayout.js'), 'utf8');
const nbaApiCode = fs.readFileSync(path.join(__dirname, '../webapp/js/nba-api.js'), 'utf8');

eval(engineCode);
eval(spEngineCode);
eval(nbaApiCode);

const DATA_DIR = path.join(__dirname, '../webapp/data');
const OUTPUT_FILE = path.join(DATA_DIR, 'live_picks_2026.json');

// --- Load existing data ---
function loadJSON(filename) {
  try {
    return JSON.parse(fs.readFileSync(path.join(DATA_DIR, filename), 'utf8'));
  } catch (e) {
    return [];
  }
}

function loadExistingPicks() {
  try {
    return JSON.parse(fs.readFileSync(OUTPUT_FILE, 'utf8'));
  } catch (e) {
    return { parlays: [], spParlays: [] };
  }
}

function getDateStr(daysAgo) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}`;
}

async function main() {
  const numDays = parseInt(process.argv[2]) || 5;
  console.log(`\n=== Seeding live picks for last ${numDays} days ===\n`);

  // Load box scores and build model
  const playerBoxScores = loadJSON('player_boxscores.json');
  console.log(`Loaded ${playerBoxScores.length} box score records`);

  const sorted = [...playerBoxScores].sort((a, b) => a.date.localeCompare(b.date));

  // Build multi-stat model
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

  // Build SP model
  let spModel = null;
  if (window.SuperPayoutEngine) {
    spModel = Object.create(window.SuperPayoutEngine.PlayerModel);
    spModel.history = {};
    for (const g of sorted) {
      for (const p of (g.players || [])) {
        const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
        if (mins < 10) continue;
        spModel.update(p.name, p.pts, mins, g.date, p.team);
      }
    }
  }

  // Load existing picks
  const existing = loadExistingPicks();
  const existingDates = new Set([
    ...existing.parlays.map(p => p.date),
    ...existing.spParlays.map(p => p.date),
  ]);

  // Determine dates to seed
  const datesToSeed = [];
  for (let i = 1; i <= numDays; i++) {
    const ds = getDateStr(i);
    if (!existingDates.has(ds)) datesToSeed.push(ds);
  }

  if (datesToSeed.length === 0) {
    console.log('All dates already seeded. Skipping to resolution.\n');
  } else {
    console.log(`Dates to seed: ${datesToSeed.join(', ')}\n`);
  }

  // Fetch odds and generate picks for each date
  for (const date of datesToSeed) {
    try {
      console.log(`[${date}] Fetching historical odds...`);
      const dayOdds = await window.BettingEngine.fetchHistoricalOddsForDate(date);

      if (!dayOdds || dayOdds.length === 0) {
        console.log(`[${date}] No odds available, skipping`);
        continue;
      }

      console.log(`[${date}] ${dayOdds.length} games with odds`);

      // Build singles
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

        // Points
        for (const [playerName, lines] of Object.entries(record.playerProps || {})) {
          addPick(model.findBestStatProp(playerName, 'points', lines));
          addPick(model.findBestT2Prop(playerName, 'points', lines));
        }
        // Rebounds
        for (const [playerName, lines] of Object.entries(record.playerRebProps || {})) {
          addPick(model.findBestStatProp(playerName, 'rebounds', lines));
          addPick(model.findBestT2Prop(playerName, 'rebounds', lines));
        }
        // Assists
        for (const [playerName, lines] of Object.entries(record.playerAstProps || {})) {
          addPick(model.findBestStatProp(playerName, 'assists', lines));
          addPick(model.findBestT2Prop(playerName, 'assists', lines));
        }
      }

      daySingles.sort((a, b) => b.confidence - a.confidence);
      console.log(`[${date}] ${daySingles.length} qualifying singles (pts/reb/ast)`);

      // Build parlays
      const dayParlays = window.BettingEngine.buildParlays(daySingles);
      for (const parlay of dayParlays) {
        parlay.legs = parlay.legs.map(leg => {
          const match = daySingles.find(s =>
            s.player === leg.player && s.line === leg.line && s.statType === leg.statType
          );
          return { ...leg, gameKey: match ? match.gameKey : '', gameDisplay: match ? match.gameDisplay : '' };
        });
      }

      // SP parlays
      let daySPParlays = [];
      if (spModel && window.SuperPayoutEngine) {
        const spSingles = [];
        for (const record of dayOdds) {
          const gameKey = record.gameKey;
          const gameDisplay = `${record.awayTeam} @ ${record.homeTeam}`;
          for (const [playerName, lines] of Object.entries(record.playerProps || {})) {
            const prop = spModel.findBestPayoutProp(playerName, lines);
            if (prop) spSingles.push({ ...prop, gameKey, gameDisplay });
          }
        }
        if (spSingles.length >= 2) {
          daySPParlays = window.SuperPayoutEngine.buildSuperParlays(spSingles);
          for (const parlay of daySPParlays) {
            parlay.legs = parlay.legs.map(leg => {
              const match = spSingles.find(s => s.player === leg.player && s.line === leg.line);
              return { ...leg, gameKey: match ? match.gameKey : '', gameDisplay: match ? match.gameDisplay : '' };
            });
          }
        }
      }

      // Save parlays
      for (const p of dayParlays) {
        existing.parlays.push({
          date,
          numLegs: p.numLegs || p.legs.length,
          odds: p.odds,
          decimalOdds: p.decimalOdds,
          combinedHitRate: p.combinedHitRate,
          ev: p.ev,
          legs: p.legs.map(l => ({
            player: l.player, team: l.team, statType: l.statType || 'points',
            statLabel: l.statLabel || window.BettingEngine.statLabel(l.statType || 'points'),
            line: l.line, odds: l.odds, gameKey: l.gameKey || '', gameDisplay: l.gameDisplay || '',
            actual: null, won: null,
          })),
          resolved: false, won: null, pnl: null,
          savedAt: new Date().toISOString(),
        });
      }

      for (const p of daySPParlays) {
        existing.spParlays.push({
          date,
          numLegs: p.numLegs || p.legs.length,
          odds: p.odds,
          decimalOdds: p.decimalOdds,
          combinedHitRate: p.combinedHitRate,
          ev: p.ev,
          isSuperPayout: true,
          legs: p.legs.map(l => ({
            player: l.player, team: l.team, statType: l.statType || 'points',
            statLabel: l.statLabel || 'PTS',
            line: l.line, odds: l.odds, gameKey: l.gameKey || '', gameDisplay: l.gameDisplay || '',
            actual: null, won: null,
          })),
          resolved: false, won: null, pnl: null,
          savedAt: new Date().toISOString(),
        });
      }

      console.log(`[${date}] Generated ${dayParlays.length} parlays + ${daySPParlays.length} SP parlays`);

      // Rate limit
      await new Promise(r => setTimeout(r, 300));
    } catch (e) {
      console.error(`[${date}] Error:`, e.message);
    }
  }

  // --- Resolve all unresolved picks ---
  console.log('\n=== Resolving picks via ESPN box scores ===\n');

  const today = getDateStr(0);
  const allParlays = [...existing.parlays, ...existing.spParlays];
  const unresolvedDates = [...new Set(
    allParlays.filter(p => !p.resolved).map(p => p.date)
  )].filter(d => d < today).sort();

  for (const date of unresolvedDates) {
    try {
      console.log(`[${date}] Fetching ESPN scoreboard...`);
      const { games, eventIds } = await window.NbaApi.fetchESPNScoreboardForDate(date);

      if (!games.length) {
        console.log(`[${date}] No games found`);
        continue;
      }

      // Fetch box scores for final games
      const playerStats = {};
      let finalCount = 0;
      for (const [teamKey, eventId] of Object.entries(eventIds)) {
        const game = games.find(g => g.id === eventId || g.espnId === eventId);
        if (game && game.status !== 'final') continue;
        finalCount++;
        const boxScore = await window.NbaApi.fetchESPNBoxScore(eventId);
        if (!boxScore) continue;
        for (const p of boxScore) {
          playerStats[p.name.toLowerCase()] = p;
        }
        await new Promise(r => setTimeout(r, 200));
      }

      const allFinal = finalCount === Object.keys(eventIds).length;
      console.log(`[${date}] ${Object.keys(playerStats).length} players in box scores (${finalCount}/${Object.keys(eventIds).length} games final)`);

      // Helper: find player stats by name
      const findPlayer = (playerName) => {
        const lower = playerName.toLowerCase();
        if (playerStats[lower]) return playerStats[lower];
        const parts = lower.split(' ');
        if (parts.length >= 2) {
          const found = Object.entries(playerStats).find(([name]) =>
            name.includes(parts[0]) && name.includes(parts[parts.length - 1])
          );
          if (found) return found[1];
        }
        return null;
      };

      // Resolve
      const resolveParlay = (parlay) => {
        if (parlay.date !== date || parlay.resolved) return;

        for (const leg of parlay.legs) {
          if (leg.actual !== null && leg.actual !== undefined) continue;
          const pStats = findPlayer(leg.player);
          if (pStats) {
            const statKey = leg.statType === 'rebounds' ? 'reb' : leg.statType === 'assists' ? 'ast' : 'pts';
            leg.actual = pStats[statKey];
            leg.won = leg.actual > leg.line;
          } else if (allFinal) {
            // DNP: player not in any box score, all games done → loss
            leg.actual = 0;
            leg.won = false;
          }
        }

        const allLegsResolved = parlay.legs.every(l => l.actual !== null && l.actual !== undefined);
        if (allLegsResolved) {
          parlay.resolved = true;
          parlay.won = parlay.legs.every(l => l.won);
          parlay.pnl = parlay.won ? Math.round((parlay.decimalOdds - 1) * 100) : -100;
        }
      };

      existing.parlays.forEach(resolveParlay);
      existing.spParlays.forEach(resolveParlay);

      const dayParlays = allParlays.filter(p => p.date === date && p.resolved);
      const wins = dayParlays.filter(p => p.won).length;
      console.log(`[${date}] Resolved: ${wins}/${dayParlays.length} won`);

    } catch (e) {
      console.error(`[${date}] Resolve error:`, e.message);
    }
  }

  // --- Write output ---
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(existing, null, 2));
  console.log(`\n=== Saved to ${OUTPUT_FILE} ===`);

  // Summary
  const resolvedParlays = existing.parlays.filter(p => p.resolved);
  const resolvedSP = existing.spParlays.filter(p => p.resolved);
  const allResolved = [...resolvedParlays, ...resolvedSP];
  const wins = allResolved.filter(p => p.won).length;
  const pnl = allResolved.reduce((s, p) => s + (p.pnl || 0), 0);

  console.log(`\nSummary:`);
  console.log(`  Standard parlays: ${existing.parlays.length} (${resolvedParlays.length} resolved)`);
  console.log(`  SP parlays: ${existing.spParlays.length} (${resolvedSP.length} resolved)`);
  console.log(`  Record: ${wins}-${allResolved.length - wins}`);
  console.log(`  P&L: $${pnl >= 0 ? '+' : ''}${pnl}`);
  if (allResolved.length > 0) {
    console.log(`  Accuracy: ${(wins / allResolved.length * 100).toFixed(1)}%`);
  }

  // Check quota
  try {
    const quotaResult = execSync('curl -s -I "https://api.the-odds-api.com/v4/sports/basketball_nba/events?apiKey=3879c3373a31421d8ef7d428b8758cd8" 2>/dev/null | grep -i x-requests-remaining', { timeout: 10000 });
    console.log(`\nOdds API: ${quotaResult.toString().trim()}`);
  } catch (e) { /* skip */ }
}

main().catch(console.error);
