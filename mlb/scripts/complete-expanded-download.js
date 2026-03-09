#!/usr/bin/env node
// =============================================================================
// Complete the expanded MLB data download:
// 1. Download game-level odds + props for dates with NO odds at all
// 2. Fix props for dates that have game-level odds but missing props
// Uses CORRECT market keys: batter_hits (NOT batter_hits_alternate)
// Saves incrementally to avoid data loss
// =============================================================================

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const DATA_DIR = path.join(__dirname, '..', 'webapp', 'data');
const ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8';

const MLB_TEAM_MAP = {
  'ARI': 'ARI', 'Arizona Diamondbacks': 'ARI',
  'ATL': 'ATL', 'Atlanta Braves': 'ATL',
  'BAL': 'BAL', 'Baltimore Orioles': 'BAL',
  'BOS': 'BOS', 'Boston Red Sox': 'BOS',
  'CHC': 'CHC', 'Chicago Cubs': 'CHC',
  'CWS': 'CWS', 'Chicago White Sox': 'CWS', 'CHW': 'CWS',
  'CIN': 'CIN', 'Cincinnati Reds': 'CIN',
  'CLE': 'CLE', 'Cleveland Guardians': 'CLE',
  'COL': 'COL', 'Colorado Rockies': 'COL',
  'DET': 'DET', 'Detroit Tigers': 'DET',
  'HOU': 'HOU', 'Houston Astros': 'HOU',
  'KC': 'KC', 'Kansas City Royals': 'KC', 'KCR': 'KC',
  'LAA': 'LAA', 'Los Angeles Angels': 'LAA',
  'LAD': 'LAD', 'Los Angeles Dodgers': 'LAD',
  'MIA': 'MIA', 'Miami Marlins': 'MIA',
  'MIL': 'MIL', 'Milwaukee Brewers': 'MIL',
  'MIN': 'MIN', 'Minnesota Twins': 'MIN',
  'NYM': 'NYM', 'New York Mets': 'NYM',
  'NYY': 'NYY', 'New York Yankees': 'NYY',
  'OAK': 'OAK', 'Oakland Athletics': 'OAK',
  'PHI': 'PHI', 'Philadelphia Phillies': 'PHI',
  'PIT': 'PIT', 'Pittsburgh Pirates': 'PIT',
  'SD': 'SD', 'San Diego Padres': 'SD', 'SDP': 'SD',
  'SF': 'SF', 'San Francisco Giants': 'SF', 'SFG': 'SF',
  'SEA': 'SEA', 'Seattle Mariners': 'SEA',
  'STL': 'STL', 'St. Louis Cardinals': 'STL',
  'TB': 'TB', 'Tampa Bay Rays': 'TB', 'TBR': 'TB',
  'TEX': 'TEX', 'Texas Rangers': 'TEX',
  'TOR': 'TOR', 'Toronto Blue Jays': 'TOR',
  'WSH': 'WSH', 'Washington Nationals': 'WSH', 'WAS': 'WSH',
};

function tri(abbr) { return MLB_TEAM_MAP[abbr] || abbr; }

function curlJSON(url) {
  try {
    const result = execSync(`curl -s --max-time 30 "${url}"`, { maxBuffer: 10 * 1024 * 1024 });
    return JSON.parse(result.toString());
  } catch (e) {
    return null;
  }
}

function sleep(ms) { execSync(`sleep ${ms / 1000}`); }

function fetchProps(eventId, isoDate) {
  const propsUrl = `https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/${eventId}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=batter_hits,batter_total_bases,batter_rbis,batter_runs_scored&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
  const propsData = curlJSON(propsUrl);

  const result = {};
  if (propsData?.data) {
    const pfb = (propsData.data.bookmakers || []).find(b => b.key === 'fanduel');
    if (pfb) {
      const hitsProps = {}, tbProps = {}, rbiProps = {}, runsProps = {};
      for (const mkt of (pfb.markets || [])) {
        for (const o of (mkt.outcomes || [])) {
          if (o.name === 'Over' && o.description && o.point !== undefined) {
            const target = mkt.key.includes('hits') ? hitsProps
              : mkt.key.includes('total_bases') ? tbProps
              : mkt.key.includes('rbis') ? rbiProps
              : mkt.key.includes('runs') ? runsProps : null;
            if (target) {
              if (!target[o.description]) target[o.description] = {};
              target[o.description][o.point] = { overOdds: o.price };
            }
          }
        }
      }
      if (Object.keys(hitsProps).length > 0) result.hitsProps = hitsProps;
      if (Object.keys(tbProps).length > 0) result.tbProps = tbProps;
      if (Object.keys(rbiProps).length > 0) result.rbiProps = rbiProps;
      if (Object.keys(runsProps).length > 0) result.runsProps = runsProps;
    }
  }
  return result;
}

function saveCheckpoint(allOdds) {
  allOdds.sort((a, b) => a.date.localeCompare(b.date));
  fs.writeFileSync(oddsPath, JSON.stringify(allOdds, null, 2));
  const withHits = allOdds.filter(r => r.hitsProps).length;
  console.log(`    [checkpoint: ${allOdds.length} records, ${withHits} with props]`);
}

// Load data
const boxscores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_player_boxscores_expanded.json'), 'utf8'));
const oddsPath = path.join(DATA_DIR, 'mlb_historical_odds_expanded.json');
let allOdds = JSON.parse(fs.readFileSync(oddsPath, 'utf8'));

const boxDates = [...new Set(boxscores.map(b => b.date))].sort();
const oddsDatesSet = new Set(allOdds.map(o => o.date));

// PHASE 1: Download game-level odds + props for dates with NO odds
const missingDates = boxDates.filter(d => !oddsDatesSet.has(d));
console.log('=== PHASE 1: Download odds for missing dates ===');
console.log(`Missing dates: ${missingDates.length}\n`);

let apiCalls = 0;

for (let i = 0; i < missingDates.length; i++) {
  const dateStr = missingDates[i];
  const y = dateStr.substring(0, 4);
  const m = dateStr.substring(4, 6);
  const dd = dateStr.substring(6, 8);
  const isoDate = `${y}-${m}-${dd}T17:00:00Z`;

  process.stdout.write(`  ${dateStr}...`);

  const gamesUrl = `https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds/?apiKey=${ODDS_API_KEY}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
  const gamesData = curlJSON(gamesUrl);
  apiCalls++;
  sleep(500);

  if (!gamesData || !gamesData.data || gamesData.data.length === 0) {
    process.stdout.write(' no odds\n');
    continue;
  }

  process.stdout.write(` ${gamesData.data.length} games`);
  let dateProps = 0;

  for (const game of gamesData.data) {
    const homeTeam = tri(game.home_team);
    const awayTeam = tri(game.away_team);
    const gameKey = `${awayTeam}@${homeTeam}`;

    const record = {
      date: dateStr, gameKey, eventId: game.id,
      homeTeam, awayTeam, commenceTime: game.commence_time,
    };

    const fb = (game.bookmakers || []).find(b => b.key === 'fanduel');
    if (fb) {
      for (const mkt of (fb.markets || [])) {
        if (mkt.key === 'h2h') {
          for (const o of (mkt.outcomes || [])) {
            if (o.name === game.home_team) record.home_ml = o.price;
            else record.away_ml = o.price;
          }
        } else if (mkt.key === 'spreads') {
          for (const o of (mkt.outcomes || [])) {
            if (o.name === game.home_team) { record.spread_home = o.price; record.spread_point = o.point; }
            else record.spread_away = o.price;
          }
        } else if (mkt.key === 'totals') {
          for (const o of (mkt.outcomes || [])) {
            if (o.name === 'Over') { record.total = o.point; record.total_over = o.price; }
            else record.total_under = o.price;
          }
        }
      }
    }

    // Fetch batter props with CORRECT market keys
    sleep(500);
    const props = fetchProps(game.id, isoDate);
    apiCalls++;
    Object.assign(record, props);
    if (props.hitsProps) dateProps++;

    allOdds.push(record);
  }

  process.stdout.write(` (${dateProps} with props)\n`);

  if (i % 5 === 4) saveCheckpoint(allOdds);
  sleep(300);
}

saveCheckpoint(allOdds);
console.log(`\nPhase 1 done. API calls: ${apiCalls}\n`);

// PHASE 2: Fix props for dates that have game-level odds but no props
const needsPropsRecords = allOdds.filter(r => !r.hitsProps);
const alreadyHasProps = allOdds.filter(r => r.hitsProps).length;

// Group by date
const dateGroups = {};
for (const r of needsPropsRecords) {
  if (!dateGroups[r.date]) dateGroups[r.date] = [];
  dateGroups[r.date].push(r);
}
const fixDates = Object.keys(dateGroups).sort();

console.log('=== PHASE 2: Fix props for records with game-level odds but no props ===');
console.log(`Records needing props: ${needsPropsRecords.length} across ${fixDates.length} dates`);
console.log(`Already have props: ${alreadyHasProps}\n`);

let fixedCount = 0;

for (let i = 0; i < fixDates.length; i++) {
  const dateStr = fixDates[i];
  const records = dateGroups[dateStr];
  const y = dateStr.substring(0, 4);
  const m = dateStr.substring(4, 6);
  const dd = dateStr.substring(6, 8);
  const isoDate = `${y}-${m}-${dd}T17:00:00Z`;

  process.stdout.write(`  ${dateStr}: ${records.length} games`);
  let dateFixed = 0;

  for (const record of records) {
    sleep(500);
    const props = fetchProps(record.eventId, isoDate);
    apiCalls++;
    Object.assign(record, props);
    if (props.hitsProps) { dateFixed++; fixedCount++; }
  }

  process.stdout.write(` -> ${dateFixed} fixed\n`);

  if (i % 5 === 4) saveCheckpoint(allOdds);
  sleep(300);
}

// Final save
saveCheckpoint(allOdds);

const finalDates = [...new Set(allOdds.map(o => o.date))].sort();
const finalWithHits = allOdds.filter(r => r.hitsProps).length;
const finalWithTB = allOdds.filter(r => r.tbProps).length;

console.log(`\n=== FINAL SUMMARY ===`);
console.log(`  Total odds records: ${allOdds.length}`);
console.log(`  Total odds dates: ${finalDates.length}`);
console.log(`  Range: ${finalDates[0]} - ${finalDates[finalDates.length - 1]}`);
console.log(`  With hits props: ${finalWithHits}`);
console.log(`  With TB props: ${finalWithTB}`);
console.log(`  Props fixed in Phase 2: ${fixedCount}`);
console.log(`  Total API calls: ${apiCalls}`);
console.log(`  Saved to: ${oddsPath}`);
