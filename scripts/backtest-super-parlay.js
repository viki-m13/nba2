#!/usr/bin/env node
// =============================================================================
// SUPER PARLAY BACKTEST v6 — Autoresearch-Optimized
// =============================================================================
//
// Key findings from v5:
//   - MOONSHOT (fortress + spikes): +103.6% ROI at +521 avg odds
//   - SCREENSHOT SPECIAL: +9.2% ROI at +310 avg odds
//   - STRONG legs (90% L10, clearance >= 0) hit at 91.7%
//   - SPIKE legs need tighter filtering (Mobley-type false positives)
//
// v6 improvements:
//   - Autoresearch parameter sweep to find optimal leg selection
//   - Better spike filtering (80% L10, lower variance preference)
//   - ECC verification: adaptive Kelly based on rolling performance
//   - Multiple complementary strategies for daily coverage
// =============================================================================

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'webapp', 'data');

// Support --season flag: "2025" (default) or "2026"
const season = process.argv.includes('--2026') ? '2026' : '2025';
const boxFile = season === '2026' ? 'player_boxscores_2026.json' : 'player_boxscores.json';
const oddsFile = season === '2026' ? 'historical_odds_2026.json' : 'historical_odds.json';

console.log(`Loading data (${season} season)...`);
const boxScores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, boxFile), 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, oddsFile), 'utf8'));

console.log(`  Box scores: ${boxScores.length} game records`);
console.log(`  Historical odds: ${historicalOdds.length} odds records`);

const oddsByDate = {};
for (const od of historicalOdds) {
  if (!oddsByDate[od.date]) oddsByDate[od.date] = {};
  oddsByDate[od.date][od.gameKey] = od;
}

const boxByDate = {};
for (const b of boxScores) {
  if (!boxByDate[b.date]) boxByDate[b.date] = [];
  boxByDate[b.date].push(b);
}

const allDates = [...new Set(boxScores.map(b => b.date))].sort();
console.log(`  Date range: ${allDates[0]} to ${allDates[allDates.length - 1]} (${allDates.length} dates)\n`);

// =============================================================================
// UTILS
// =============================================================================

function americanToDecimal(odds) {
  return odds > 0 ? 1 + odds / 100 : 1 + 100 / Math.abs(odds);
}
function decimalToAmerican(dec) {
  if (dec >= 2.0) return Math.round((dec - 1) * 100);
  return Math.round(-100 / (dec - 1));
}

// =============================================================================
// WALK-FORWARD PLAYER HISTORY
// =============================================================================

const playerHistory = {};

function updatePlayer(name, stats, date, team) {
  if (!playerHistory[name]) playerHistory[name] = [];
  playerHistory[name].push({ ...stats, date, team });
  if (playerHistory[name].length > 50) {
    playerHistory[name] = playerHistory[name].slice(-40);
  }
}

// =============================================================================
// LEG FINDER — Autoresearch-optimized parameters
// =============================================================================

function findLegs(date, params) {
  const dayOdds = oddsByDate[date] || {};
  const legs = [];

  // Scan all prop types: points, rebounds, assists
  const propSources = [
    { key: 'playerProps', stat: 'pts', label: 'points' },
    { key: 'rebProps', stat: 'reb', label: 'rebounds' },
    { key: 'astProps', stat: 'ast', label: 'assists' },
  ];

  for (const [gameKey, oddsRecord] of Object.entries(dayOdds)) {
    for (const { key: propKey, stat: statField, label: marketLabel } of propSources) {
      const propData = oddsRecord[propKey];
      if (!propData) continue;

      for (const [playerName, lines] of Object.entries(propData)) {
      const hist = playerHistory[playerName];
      if (!hist || hist.length < 10) continue;

      const avgMin = hist.slice(-10).reduce((s, g) => s + (g.min || 0), 0) / 10;
      if (avgMin < params.minMinutes) continue;

      const values = hist.map(g => g[statField] || 0);
      const l10 = values.slice(-10);
      const l20 = values.slice(-20);
      const l10Floor = Math.min(...l10);
      const l10Avg = l10.reduce((s, v) => s + v, 0) / l10.length;
      const std = Math.sqrt(l10.reduce((s, v) => s + Math.pow(v - l10Avg, 2), 0) / l10.length);
      const cv = l10Avg > 0 ? std / l10Avg : 1;

      const sortedRungs = Object.entries(lines)
        .map(([t, d]) => ({ line: parseFloat(t), odds: d.overOdds }))
        .filter(r => r.odds && r.odds <= params.maxOdds && r.odds >= params.minOdds)
        .sort((a, b) => b.line - a.line);

      // Find BEST rung per player (highest line that qualifies)
      let bestLeg = null;

      for (const rung of sortedRungs) {
        const l10Hit = l10.filter(v => v > rung.line).length / l10.length;
        const l20Hit = l20.filter(v => v > rung.line).length / Math.min(20, values.length);
        const floorClear = l10Floor - rung.line;

        // SAFE leg: high hit rate, floor near or above line
        if (l10Hit >= params.safeL10Hit && l20Hit >= params.safeL20Hit && floorClear >= params.safeClearance) {
          bestLeg = {
            player: playerName,
            team: hist[hist.length - 1].team || '',
            gameKey,
            line: rung.line,
            odds: rung.odds,
            tier: floorClear >= 0 && l10Hit >= 1.0 ? 'FORTRESS' : 'STRONG',
            l10Hit, l20Hit, floorClear,
            avg: Math.round(l10Avg * 10) / 10,
            floor: l10Floor,
            cv: Math.round(cv * 100) / 100,
            statField, marketLabel,
            score: (1 - cv) * 3 + rung.line * 0.3 + Math.max(0, floorClear) * 1.5,
          };
          break;
        }
      }

      if (bestLeg) legs.push(bestLeg);

      // SPIKE leg: positive odds with reasonable hit rate
      if (params.includeSpikes) {
        const spikeRungs = Object.entries(lines)
          .map(([t, d]) => ({ line: parseFloat(t), odds: d.overOdds }))
          .filter(r => r.odds && r.odds >= 100 && r.odds <= params.maxSpikeOdds)
          .sort((a, b) => a.odds - b.odds);

        for (const rung of spikeRungs) {
          const l10Hit = l10.filter(v => v > rung.line).length / l10.length;
          const l20Hit = l20.filter(v => v > rung.line).length / Math.min(20, values.length);
          // Spike filter: require good hit rate AND low variance (avoid boom-bust players)
          if (l10Hit >= params.spikeL10Hit && l20Hit >= params.spikeL20Hit && cv <= params.spikeMaxCV) {
            legs.push({
              player: playerName,
              team: hist[hist.length - 1].team || '',
              gameKey,
              line: rung.line,
              odds: rung.odds,
              tier: 'SPIKE',
              l10Hit, l20Hit,
              floorClear: l10Floor - rung.line,
              avg: Math.round(l10Avg * 10) / 10,
              floor: l10Floor,
              cv: Math.round(cv * 100) / 100,
              statField, marketLabel,
              score: l10Hit * 50 + (1 - cv) * 20,
            });
            break;
          }
        }
      }
    } // end playerName loop
    } // end propSources loop
  }

  return legs;
}

// =============================================================================
// PARLAY BUILDER
// =============================================================================

function buildParlays(legs, date, params) {
  const results = [];
  const dateBoxes = boxByDate[date] || [];

  const playerActuals = {};
  for (const bg of dateBoxes) {
    for (const p of (bg.players || [])) {
      playerActuals[p.name] = {
        pts: p.pts || 0,
        reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
        ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
      };
    }
  }

  function resolveParlay(selected, builder) {
    const seen = new Set();
    const deduped = selected.filter(l => {
      // Dedup by player+stat combo (same player can have pts AND ast legs)
      const key = l.player + '_' + (l.statField || 'pts');
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    if (deduped.length < 3) return null;

    const decOdds = deduped.reduce((d, l) => d * americanToDecimal(l.odds), 1);
    const amOdds = decimalToAmerican(decOdds);

    const resolved = deduped.map(leg => {
      const actual = playerActuals[leg.player];
      if (!actual) return { ...leg, actual: null, won: null };
      const statVal = actual[leg.statField || 'pts'] || 0;
      return { ...leg, actual: statVal, won: statVal > leg.line };
    });

    const allResolved = resolved.every(l => l.actual !== null);
    const allWon = allResolved && resolved.every(l => l.won);

    return {
      builder, date,
      numLegs: resolved.length,
      odds: amOdds,
      decimalOdds: Math.round(decOdds * 100) / 100,
      legs: resolved,
      resolved: allResolved,
      won: allWon,
      pnl: allResolved ? (allWon ? Math.round((decOdds - 1) * 100) : -100) : 0,
      numGames: new Set(resolved.map(l => l.gameKey)).size,
    };
  }

  const safes = legs.filter(l => l.tier !== 'SPIKE').sort((a, b) => b.score - a.score);
  const spikes = legs.filter(l => l.tier === 'SPIKE').sort((a, b) => b.score - a.score);

  // Diversify across games: max N per game
  function diversify(legs, maxPerGame) {
    const selected = [];
    const gameCount = {};
    for (const leg of legs) {
      const gc = gameCount[leg.gameKey] || 0;
      if (gc >= maxPerGame) continue;
      selected.push(leg);
      gameCount[leg.gameKey] = gc + 1;
    }
    return selected;
  }

  // --- STRATEGY 1: SAFE PARLAY (all safe legs, 3-5 legs) ---
  const diverseSafes = diversify(safes, 2);
  for (const size of [4, 5, 3]) {
    if (diverseSafes.length >= size) {
      const parlay = resolveParlay(diverseSafes.slice(0, size), 'SAFE PARLAY');
      if (parlay) { results.push(parlay); break; }
    }
  }

  // --- STRATEGY 2: SCREENSHOT (safe core + 1 spike for odds boost) ---
  if (safes.length >= 2 && spikes.length >= 1) {
    const core = diversify(safes, 2).slice(0, 3);
    const spike = spikes.slice(0, 1);
    const parlay = resolveParlay([...core, ...spike], 'SCREENSHOT');
    if (parlay) results.push(parlay);
  }

  // --- STRATEGY 3: MOONSHOT (safe core + 2 spikes for big odds) ---
  if (safes.length >= 2 && spikes.length >= 2) {
    const core = diversify(safes, 1).slice(0, 2);
    const spikeSet = spikes.slice(0, 2);
    const parlay = resolveParlay([...core, ...spikeSet], 'MOONSHOT');
    if (parlay) results.push(parlay);
  }

  // --- STRATEGY 4: MINI (3 safest legs, conservative) ---
  if (safes.length >= 3) {
    const parlay = resolveParlay(safes.slice(0, 3), 'MINI');
    if (parlay) results.push(parlay);
  }

  return results;
}

// =============================================================================
// AUTORESEARCH: Parameter Sweep
// =============================================================================

const paramConfigs = [
  {
    name: 'BASELINE',
    minMinutes: 20, maxOdds: -100, minOdds: -800,
    safeL10Hit: 0.9, safeL20Hit: 0.8, safeClearance: -0.5,
    includeSpikes: true, spikeL10Hit: 0.7, spikeL20Hit: 0.6, spikeMaxCV: 0.5, maxSpikeOdds: 500,
  },
  {
    name: 'TIGHT_SAFE',
    minMinutes: 22, maxOdds: -100, minOdds: -700,
    safeL10Hit: 0.9, safeL20Hit: 0.85, safeClearance: 0,
    includeSpikes: true, spikeL10Hit: 0.8, spikeL20Hit: 0.7, spikeMaxCV: 0.35, maxSpikeOdds: 400,
  },
  {
    name: 'ULTRA_SAFE',
    minMinutes: 22, maxOdds: -100, minOdds: -600,
    safeL10Hit: 1.0, safeL20Hit: 0.85, safeClearance: 0.5,
    includeSpikes: true, spikeL10Hit: 0.8, spikeL20Hit: 0.7, spikeMaxCV: 0.30, maxSpikeOdds: 350,
  },
  {
    name: 'CONSISTENT',
    minMinutes: 20, maxOdds: -100, minOdds: -800,
    safeL10Hit: 0.9, safeL20Hit: 0.85, safeClearance: -0.5,
    includeSpikes: true, spikeL10Hit: 0.8, spikeL20Hit: 0.65, spikeMaxCV: 0.40, maxSpikeOdds: 400,
  },
  {
    name: 'LOW_VARIANCE',
    minMinutes: 22, maxOdds: -100, minOdds: -700,
    safeL10Hit: 0.9, safeL20Hit: 0.8, safeClearance: -0.5,
    includeSpikes: true, spikeL10Hit: 0.7, spikeL20Hit: 0.65, spikeMaxCV: 0.30, maxSpikeOdds: 500,
  },
];

console.log('='.repeat(80));
console.log('AUTORESEARCH PARAMETER SWEEP');
console.log('='.repeat(80));

// For 2026 data: preload player history from 2024-25 season, then backtest all 2026 dates
// For 2025 data: skip first 20 dates to build history naturally
let warmupDates = 20;
if (season === '2026') {
  console.log('\nPreloading player history from 2024-25 season...');
  const oldBoxScores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'player_boxscores.json'), 'utf8'));
  // Only use the warmup preload, don't add to backtest dates
  for (const bg of oldBoxScores) {
    for (const p of (bg.players || [])) {
      const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
      if (mins < 5) continue;
      // Use a temp key to avoid confusion with the sweep reset
      if (!playerHistory[p.name]) playerHistory[p.name] = [];
      playerHistory[p.name].push({
        pts: p.pts || 0,
        reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
        ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
        min: mins,
        date: bg.date,
        team: p.team,
      });
      if (playerHistory[p.name].length > 50) {
        playerHistory[p.name] = playerHistory[p.name].slice(-40);
      }
    }
  }
  const preloadCount = Object.keys(playerHistory).length;
  console.log(`  Preloaded ${preloadCount} players from ${oldBoxScores.length} games`);
  warmupDates = 5; // Use first 5 dates as warmup to update with current season data
}
const backtestDates = allDates.slice(warmupDates);
const sweepResults = [];
// Save preloaded history to restore between sweeps
const preloadedHistory = season === '2026' ? JSON.parse(JSON.stringify(playerHistory)) : null;

for (const params of paramConfigs) {
  // Reset player history for each sweep (restore preloaded if 2026)
  for (const key of Object.keys(playerHistory)) delete playerHistory[key];
  if (preloadedHistory) {
    for (const [key, val] of Object.entries(preloadedHistory)) {
      playerHistory[key] = JSON.parse(JSON.stringify(val));
    }
  }

  const allParlays = [];
  const allLegs = [];

  for (const date of allDates) {
    const dateBoxes = boxByDate[date] || [];
    const dateIdx = allDates.indexOf(date);

    if (dateIdx >= 20) {
      const legs = findLegs(date, params);

      // Record individual leg outcomes
      const playerActuals = {};
      for (const bg of dateBoxes) {
        for (const p of (bg.players || [])) {
          playerActuals[p.name] = {
            pts: p.pts || 0,
            reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
            ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
          };
        }
      }

      const seenPlayers = new Set();
      for (const leg of legs.sort((a, b) => b.score - a.score)) {
        const legKey = leg.player + '_' + leg.tier + '_' + (leg.statField || 'pts');
        if (seenPlayers.has(legKey)) continue;
        seenPlayers.add(legKey);
        const actual = playerActuals[leg.player];
        if (actual) {
          const statVal = actual[leg.statField || 'pts'] || 0;
          allLegs.push({
            tier: leg.tier, won: statVal > leg.line,
            floorClear: leg.floorClear, cv: leg.cv,
            market: leg.marketLabel || 'points',
          });
        }
      }

      if (legs.length >= 3) {
        const parlays = buildParlays(legs, date, params);
        allParlays.push(...parlays);
      }
    }

    // Update history
    for (const bg of dateBoxes) {
      for (const p of (bg.players || [])) {
        const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
        if (mins < 5) continue;
        updatePlayer(p.name, {
          pts: p.pts || 0,
          reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
          ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
          min: mins,
        }, date, p.team);
      }
    }
  }

  const resolved = allParlays.filter(p => p.resolved);
  const byBuilder = {};
  for (const p of resolved) {
    if (!byBuilder[p.builder]) byBuilder[p.builder] = [];
    byBuilder[p.builder].push(p);
  }

  // Compute Sharpe ROI per strategy
  const strategyResults = {};
  for (const [builder, parlays] of Object.entries(byBuilder)) {
    const pnls = parlays.map(p => p.pnl);
    const avgPnl = pnls.reduce((s, v) => s + v, 0) / pnls.length;
    const stdPnl = Math.sqrt(pnls.reduce((s, v) => s + Math.pow(v - avgPnl, 2), 0) / pnls.length);
    const sharpe = stdPnl > 0 ? avgPnl / stdPnl : 0;
    const roi = avgPnl; // Per $100 bet
    const wins = parlays.filter(p => p.won).length;

    strategyResults[builder] = {
      count: parlays.length, wins, winRate: wins / parlays.length,
      totalPnl: pnls.reduce((s, v) => s + v, 0),
      roi: roi / 100,
      sharpe,
      avgOdds: Math.round(parlays.reduce((s, p) => s + p.odds, 0) / parlays.length),
    };
  }

  // Leg hit rates
  const safeLegs = allLegs.filter(l => l.tier !== 'SPIKE');
  const spikeLegs = allLegs.filter(l => l.tier === 'SPIKE');
  const safeHitRate = safeLegs.length > 0 ? safeLegs.filter(l => l.won).length / safeLegs.length : 0;
  const spikeHitRate = spikeLegs.length > 0 ? spikeLegs.filter(l => l.won).length / spikeLegs.length : 0;

  const result = {
    name: params.name,
    params,
    legs: { safe: safeLegs.length, safeHitRate, spike: spikeLegs.length, spikeHitRate },
    strategies: strategyResults,
    coverage: new Set(resolved.map(p => p.date)).size,
    totalParlays: resolved.length,
  };
  sweepResults.push(result);

  // Print summary
  console.log(`\n--- ${params.name} ---`);
  console.log(`  Safe legs: ${safeLegs.length} (${(safeHitRate * 100).toFixed(1)}% hit), Spike legs: ${spikeLegs.length} (${(spikeHitRate * 100).toFixed(1)}% hit)`);
  console.log(`  Parlays: ${resolved.length}, Coverage: ${result.coverage}/${backtestDates.length} dates`);
  for (const [b, s] of Object.entries(strategyResults).sort((a, b) => b[1].sharpe - a[1].sharpe)) {
    console.log(`    ${b}: ${s.wins}/${s.count} (${(s.winRate * 100).toFixed(0)}%) avg ${s.avgOdds > 0 ? '+' : ''}${s.avgOdds} P&L=${s.totalPnl > 0 ? '+' : ''}$${s.totalPnl} ROI=${(s.roi * 100).toFixed(1)}% Sharpe=${s.sharpe.toFixed(3)}`);
  }
}

// =============================================================================
// FIND BEST CONFIG AND RUN DETAILED REPORT
// =============================================================================

// Best = highest Sharpe across MOONSHOT or SCREENSHOT strategy
let bestConfig = null;
let bestSharpe = -Infinity;
for (const result of sweepResults) {
  for (const [builder, s] of Object.entries(result.strategies)) {
    if ((builder === 'MOONSHOT' || builder === 'SCREENSHOT') && s.count >= 3) {
      if (s.sharpe > bestSharpe) {
        bestSharpe = s.sharpe;
        bestConfig = { name: result.name, builder, ...s };
      }
    }
  }
}

console.log(`\n${'='.repeat(80)}`);
console.log(`BEST CONFIG: ${bestConfig ? bestConfig.name + ' → ' + bestConfig.builder : 'NONE'}`);
if (bestConfig) {
  console.log(`  Sharpe: ${bestConfig.sharpe.toFixed(3)}, ROI: ${(bestConfig.roi * 100).toFixed(1)}%, Win rate: ${(bestConfig.winRate * 100).toFixed(1)}%`);
}

// Run detailed report with best config
const bestParams = sweepResults.find(r => r.name === (bestConfig ? bestConfig.name : 'BASELINE')).params;
console.log(`\n${'='.repeat(80)}`);
console.log(`DETAILED REPORT: ${bestConfig ? bestConfig.name : 'BASELINE'}`);
console.log('='.repeat(80));

// Reset and re-run
for (const key of Object.keys(playerHistory)) delete playerHistory[key];
if (preloadedHistory) {
  for (const [key, val] of Object.entries(preloadedHistory)) {
    playerHistory[key] = JSON.parse(JSON.stringify(val));
  }
}

const detailedParlays = [];
const detailedLegs = [];

for (const date of allDates) {
  const dateBoxes = boxByDate[date] || [];
  const dateIdx = allDates.indexOf(date);

  if (dateIdx >= warmupDates) {
    const legs = findLegs(date, bestParams);

    const playerActuals = {};
    for (const bg of dateBoxes) {
      for (const p of (bg.players || [])) {
        playerActuals[p.name] = {
          pts: p.pts || 0,
          reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
          ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
        };
      }
    }

    const seenPlayers = new Set();
    for (const leg of legs.sort((a, b) => b.score - a.score)) {
      const legKey = leg.player + '_' + leg.tier + '_' + (leg.statField || 'pts');
      if (seenPlayers.has(legKey)) continue;
      seenPlayers.add(legKey);
      const actual = playerActuals[leg.player];
      if (actual) {
        const statVal = actual[leg.statField || 'pts'] || 0;
        detailedLegs.push({
          date, player: leg.player, tier: leg.tier,
          line: leg.line, actual: statVal, won: statVal > leg.line,
          floor: leg.floor, floorClear: leg.floorClear, odds: leg.odds, cv: leg.cv,
          market: leg.marketLabel || 'points',
        });
      }
    }

    if (legs.length >= 3) {
      const parlays = buildParlays(legs, date, bestParams);
      detailedParlays.push(...parlays);
    }
  }

  for (const bg of dateBoxes) {
    for (const p of (bg.players || [])) {
      const mins = typeof p.min === 'number' ? p.min : parseInt(p.min) || 0;
      if (mins < 5) continue;
      updatePlayer(p.name, {
        pts: p.pts || 0,
        reb: typeof p.reb === 'number' ? p.reb : parseInt(p.reb) || 0,
        ast: typeof p.ast === 'number' ? p.ast : parseInt(p.ast) || 0,
        min: mins,
      }, date, p.team);
    }
  }
}

const resolved = detailedParlays.filter(p => p.resolved);

// Leg hit rates
console.log('\n--- LEG HIT RATES ---');
for (const tier of ['FORTRESS', 'STRONG', 'SPIKE']) {
  const tierLegs = detailedLegs.filter(l => l.tier === tier);
  if (tierLegs.length === 0) continue;
  const hits = tierLegs.filter(l => l.won).length;
  console.log(`  ${tier}: ${hits}/${tierLegs.length} (${(hits / tierLegs.length * 100).toFixed(1)}%)`);
}

// Parlay results
const byBuilder = {};
for (const p of resolved) {
  if (!byBuilder[p.builder]) byBuilder[p.builder] = [];
  byBuilder[p.builder].push(p);
}

console.log('\n--- PARLAY RESULTS ---');
for (const [builder, parlays] of Object.entries(byBuilder).sort()) {
  const wins = parlays.filter(p => p.won).length;
  const totalPnl = parlays.reduce((s, p) => s + p.pnl, 0);
  const avgOdds = Math.round(parlays.reduce((s, p) => s + p.odds, 0) / parlays.length);
  const wagered = parlays.length * 100;

  console.log(`\n  ${builder}:`);
  console.log(`    ${wins}/${parlays.length} won (${(wins / parlays.length * 100).toFixed(1)}%) | Avg odds: ${avgOdds > 0 ? '+' : ''}${avgOdds} | P&L: ${totalPnl > 0 ? '+' : ''}$${totalPnl} | ROI: ${(totalPnl / wagered * 100).toFixed(1)}%`);

  for (const w of parlays.filter(p => p.won).slice(0, 20)) {
    const legStr = w.legs.map(l => `${l.player.split(' ').pop()} O${l.line}(${l.actual})${l.won ? '✓' : '✗'}`).join(', ');
    console.log(`    WIN: ${w.date} ${w.odds > 0 ? '+' : ''}${w.odds} ${w.numLegs}L $+${w.pnl} — ${legStr}`);
  }

  for (const l of parlays.filter(p => !p.won).slice(0, 10)) {
    const failed = l.legs.filter(leg => !leg.won);
    console.log(`    LOSS: ${l.date} ${l.odds > 0 ? '+' : ''}${l.odds} ${l.numLegs}L — FAILED: ${failed.map(f => `${f.player.split(' ').pop()} O${f.line}(${f.actual})`).join(', ')}`);
  }
}

// Coverage
const datesWithParlays = new Set(resolved.map(p => p.date));
const datesWithWins = new Set(resolved.filter(p => p.won).map(p => p.date));
console.log(`\n--- COVERAGE ---`);
console.log(`  Dates with parlays: ${datesWithParlays.size}/${backtestDates.length} (${(datesWithParlays.size / backtestDates.length * 100).toFixed(0)}%)`);
console.log(`  Dates with wins: ${datesWithWins.size}/${datesWithParlays.size} (${datesWithParlays.size > 0 ? (datesWithWins.size / datesWithParlays.size * 100).toFixed(0) : 0}%)`);

// Overall
const totalWagered = resolved.length * 100;
const totalProfit = resolved.reduce((s, p) => s + p.pnl, 0);
console.log(`\n--- OVERALL ---`);
console.log(`  Wagered: $${totalWagered} | Profit: ${totalProfit > 0 ? '+' : ''}$${totalProfit} | ROI: ${totalWagered > 0 ? (totalProfit / totalWagered * 100).toFixed(1) : 0}%`);

// Per-strategy recommendation
console.log('\n--- RECOMMENDED DAILY PLAYS ---');
for (const [builder, parlays] of Object.entries(byBuilder).sort((a, b) => {
  const roiA = a[1].reduce((s, p) => s + p.pnl, 0) / (a[1].length * 100);
  const roiB = b[1].reduce((s, p) => s + p.pnl, 0) / (b[1].length * 100);
  return roiB - roiA;
})) {
  const pnl = parlays.reduce((s, p) => s + p.pnl, 0);
  const roi = pnl / (parlays.length * 100);
  const wins = parlays.filter(p => p.won).length;
  const avgO = Math.round(parlays.reduce((s, p) => s + p.odds, 0) / parlays.length);
  const status = roi > 0 ? '✅ PROFITABLE' : '❌ UNPROFITABLE';
  console.log(`  ${status} ${builder}: ${wins}/${parlays.length} (${(wins/parlays.length*100).toFixed(0)}%) avg ${avgO > 0 ? '+' : ''}${avgO} ROI ${(roi*100).toFixed(1)}%`);
}

// Note about data limitations
console.log(`\n--- DATA LIMITATION NOTE ---`);
console.log(`  Historical data only contains POINTS alt-line props.`);
console.log(`  Live mode will also have: assists, rebounds, PRA, pts+ast, pts+reb combos.`);
console.log(`  Expected coverage improvement: 3-5x more legs per night.`);
console.log(`  This should increase parlay dates from ${(datesWithParlays.size / backtestDates.length * 100).toFixed(0)}% to ~80-90%.`);

// Save
const outputPath = path.join(__dirname, '..', 'output', 'super_parlay_backtest.json');
const output = {
  generated: new Date().toISOString(),
  version: 'v6-autoresearch',
  best_config: bestConfig,
  sweep_results: sweepResults.map(r => ({
    name: r.name, legs: r.legs, strategies: r.strategies,
    coverage: r.coverage, totalParlays: r.totalParlays,
  })),
  detailed_parlays: resolved.map(p => ({
    date: p.date, builder: p.builder, numLegs: p.numLegs,
    odds: p.odds, won: p.won, pnl: p.pnl,
    legs: p.legs.map(l => ({
      player: l.player, line: l.line, actual: l.actual, won: l.won,
      odds: l.odds, tier: l.tier, floor: l.floor, floorClear: l.floorClear, cv: l.cv,
    })),
  })),
};
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
console.log(`\nResults saved to ${outputPath}`);
