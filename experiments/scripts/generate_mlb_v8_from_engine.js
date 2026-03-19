#!/usr/bin/env node
/**
 * Generate MLB V8 data using the proven V3 engine with optimized config.
 * Runs backtest on expanded (full 2025 season) data and saves as V8 format.
 * Does NOT modify the original or positive-odds models.
 */

const fs = require('fs');
const path = require('path');

// Set up globals for the engine (it expects browser context)
global.window = global;
global.document = { readyState: 'complete', addEventListener: () => {}, querySelectorAll: () => [] };
global.localStorage = { _data: {}, getItem(k) { return this._data[k] || null; }, setItem(k, v) { this._data[k] = v; } };

// Load V3 engine
const enginePath = path.join(__dirname, '../../mlb/webapp/js/recommendation-engine-mlb-v3.js');
eval(fs.readFileSync(enginePath, 'utf8'));
const ENGINE = global.window.MLBRecommendationEngineV3;

// Load expanded data (full 2025 regular season)
const DATA_DIR = path.join(__dirname, '../../mlb/webapp/data');
const boxScores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_player_boxscores_expanded.json'), 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_historical_odds.json'), 'utf8'));

console.log('=== MLB V8 CSPE Data Generator ===');
console.log(`Box scores: ${boxScores.length} games`);
console.log(`Historical odds: ${historicalOdds.length} records`);

// Load optimized V3 config
const configPath = path.join(DATA_DIR, 'mlb_ultra_engine_v3_config.json');
let optimizedConfig = {};
if (fs.existsSync(configPath)) {
  const cfgData = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  optimizedConfig = cfgData.config || {};
  console.log(`Loaded optimized config (score: ${cfgData.score || '?'})`);
}

// Apply optimized config
for (const [k, v] of Object.entries(optimizedConfig)) {
  if (ENGINE.CONFIG.hasOwnProperty(k)) {
    ENGINE.CONFIG[k] = v;
  }
}

// Run backtest
console.log('\nRunning backtest...');
const results = ENGINE.runBacktest(boxScores, historicalOdds);

if (!results || !results.stats) {
  console.error('Backtest failed!');
  process.exit(1);
}

const stats = results.stats;
console.log(`\nResults:`);
console.log(`  Overall: ${stats.overall.total} picks, ${stats.overall.wins} wins, ${(stats.overall.accuracy * 100).toFixed(1)}% acc, $${stats.overall.pnl}, ${(stats.overall.roi * 100).toFixed(1)}% ROI`);
console.log(`  Singles: ${stats.singles.total} picks, ${stats.singles.wins} wins, ${(stats.singles.accuracy * 100).toFixed(1)}% acc`);
console.log(`  Parlays: ${stats.parlays.total} picks, ${stats.parlays.wins} wins, ${(stats.parlays.accuracy * 100).toFixed(1)}% acc, leg acc: ${(stats.parlays.legAccuracy * 100).toFixed(1)}%`);

// Temporal validation
const signals = results.signals || [];
const allDates = [...new Set(signals.map(s => s.date))].sort();
const mid = Math.floor(allDates.length / 2);
const h1Dates = new Set(allDates.slice(0, mid));
const h2Dates = new Set(allDates.slice(mid));
const h1 = signals.filter(s => h1Dates.has(s.date));
const h2 = signals.filter(s => h2Dates.has(s.date));

for (const [label, half] of [['H1', h1], ['H2', h2]]) {
  if (half.length > 0) {
    const hWins = half.filter(s => s.hit).length;
    const hAcc = hWins / half.length;
    const hPnl = half.reduce((s, x) => s + (x.pnl || 0), 0);
    console.log(`  ${label}: ${half.length} picks, ${(hAcc * 100).toFixed(1)}% acc, $${hPnl}`);
  }
}

if (h1.length > 0 && h2.length > 0) {
  const gap = Math.abs(h1.filter(s => s.hit).length / h1.length - h2.filter(s => s.hit).length / h2.length) * 100;
  console.log(`  Gap: ${gap.toFixed(1)}pp (${gap <= 15 ? 'STABLE' : 'UNSTABLE'})`);
}

// Convert to V8 format
const v8Signals = [];
const byLegs = {};

for (const sig of signals) {
  if (sig.betType === 'parlay') {
    const nLegs = sig.n_legs || (sig.legs ? sig.legs.length : 0);
    const v8Sig = {
      date: sig.date,
      betType: 'parlay',
      n_legs: nLegs,
      legs: (sig.legs || []).map(l => ({
        player: l.player || '',
        team: l.team || '',
        line: l.line || 0,
        odds: l.odds || 0,
        stat: l.stat || l.statLabel || '',
        statLabel: l.statLabel || (l.stat || '').toUpperCase(),
        cascadeScore: l.cascadeScore || 0,
        hit: l.hit,
        actual: l.actual,
        gates: 0,
        clearance: 0,
        hr_20: l.hitRate || 0,
        edge: l.edge || 0,
        tier: '',
      })),
      odds: sig.parlay_american || sig.odds || 0,
      hit: sig.hit,
      pnl: sig.pnl || 0,
      wager: sig.wagered || sig.betSize || 100,
      bet: `V8 CSPE ${nLegs}-Leg [mlb]`,
      engine: 'v8_cspe_mlb',
      source: 'backtest',
      combined_prob: sig.combinedHitRate || 0,
      parlay_ev: sig.ev || 0,
      tier: sig.tier || '',
    };
    v8Signals.push(v8Sig);

    // Aggregate by legs
    if (!byLegs[nLegs]) {
      byLegs[nLegs] = { wins: 0, total: 0, pnl: 0, wagered: 0, legs_total: 0, legs_hits: 0 };
    }
    byLegs[nLegs].total++;
    if (sig.hit) byLegs[nLegs].wins++;
    byLegs[nLegs].pnl += sig.pnl || 0;
    byLegs[nLegs].wagered += sig.wagered || sig.betSize || 100;
    for (const l of (sig.legs || [])) {
      byLegs[nLegs].legs_total++;
      if (l.hit) byLegs[nLegs].legs_hits++;
    }
  } else {
    // Singles - convert to single-leg parlay format for V8 display
    const v8Sig = {
      date: sig.date,
      betType: 'parlay',
      n_legs: 1,
      legs: [{
        player: sig.player || '',
        team: sig.team || '',
        line: sig.line || 0,
        odds: sig.odds || 0,
        stat: sig.statLabel || sig.statType || '',
        statLabel: sig.statLabel || '',
        cascadeScore: sig.cascadeScore || 0,
        hit: sig.hit,
        actual: sig.actual,
        gates: 0,
        clearance: 0,
        hr_20: sig.hitRate || 0,
        edge: sig.edge || 0,
        tier: '',
      }],
      odds: sig.odds || 0,
      hit: sig.hit,
      pnl: sig.pnl || 0,
      wager: sig.wagered || sig.betSize || 100,
      bet: `V8 CSPE Single [mlb]`,
      engine: 'v8_cspe_mlb',
      source: 'backtest',
      combined_prob: 0,
      parlay_ev: sig.ev || 0,
      tier: '',
    };
    v8Signals.push(v8Sig);

    if (!byLegs[1]) {
      byLegs[1] = { wins: 0, total: 0, pnl: 0, wagered: 0, legs_total: 0, legs_hits: 0 };
    }
    byLegs[1].total++;
    if (sig.hit) byLegs[1].wins++;
    byLegs[1].pnl += sig.pnl || 0;
    byLegs[1].wagered += sig.wagered || sig.betSize || 100;
    byLegs[1].legs_total++;
    if (sig.hit) byLegs[1].legs_hits++;
  }
}

// Compute by_legs stats
for (const [n, s] of Object.entries(byLegs)) {
  s.accuracy = s.total > 0 ? s.wins / s.total : 0;
  s.roi = s.wagered > 0 ? s.pnl / s.wagered : 0;
  s.leg_accuracy = s.legs_total > 0 ? s.legs_hits / s.legs_total : 0;
  // Rename for webapp compat
  s.leg_total = s.legs_total;
  s.leg_hits = s.legs_hits;
}

// Compute overall parlay stats (all signals including singles treated as 1-leg parlays)
const resolved = v8Signals.filter(s => s.hit !== undefined && s.hit !== null);
const totalWins = resolved.filter(s => s.hit).length;
const totalPnl = resolved.reduce((s, x) => s + (x.pnl || 0), 0);
const totalWager = resolved.reduce((s, x) => s + (x.wager || 100), 0);
const allLegs = [];
for (const s of resolved) {
  for (const l of (s.legs || [])) {
    allLegs.push(l);
  }
}
const legHits = allLegs.filter(l => l.hit).length;

const v8Stats = {
  parlays: {
    total: resolved.length,
    wins: totalWins,
    accuracy: resolved.length > 0 ? totalWins / resolved.length : 0,
    pnl: totalPnl,
    roi: totalWager > 0 ? totalPnl / totalWager : 0,
    leg_total: allLegs.length,
    leg_hits: legHits,
    leg_accuracy: allLegs.length > 0 ? legHits / allLegs.length : 0,
  },
  by_legs: byLegs,
  model: 'MLB V8 CSPE (Poisson-Bayesian Convergence Engine)',
  innovation: '16-signal gate cascade with Poisson-Bayesian fusion for discrete MLB stats',
  generated: new Date().toISOString(),
};

// Save to V8 data directory
const V8_DIR = path.join(__dirname, '../../webapp/v8-mgcc/data');
if (!fs.existsSync(V8_DIR)) fs.mkdirSync(V8_DIR, { recursive: true });

fs.writeFileSync(path.join(V8_DIR, 'mlb_v8_signals.json'), JSON.stringify(v8Signals, null, 2));
fs.writeFileSync(path.join(V8_DIR, 'mlb_v8_stats.json'), JSON.stringify(v8Stats, null, 2));

// Empty recommendations (no live MLB games right now)
const recs = {
  generated: new Date().toISOString(),
  date: new Date().toISOString().slice(0, 10).replace(/-/g, ''),
  engine: 'MLB V8 CSPE',
  picks: [],
};
fs.writeFileSync(path.join(V8_DIR, 'mlb_v8_recommendations.json'), JSON.stringify(recs, null, 2));

console.log(`\n=== V8 Data Saved ===`);
console.log(`Signals: ${v8Signals.length}`);
console.log(`Stats: ${JSON.stringify(v8Stats.parlays, null, 2)}`);
console.log(`Saved to ${V8_DIR}/`);
