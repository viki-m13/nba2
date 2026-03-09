#!/usr/bin/env node
// =============================================================================
// Download EXPANDED MLB historical data for thorough backtesting validation
// - Downloads ESPN boxscores (FREE) for full 2025 season (Apr 1 - Sep 28)
// - Downloads Odds API data ONLY for dates not already in existing data
// - Saves incrementally to avoid data loss on interruption
// - Merges with existing data at the end
// =============================================================================

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const DATA_DIR = path.join(__dirname, '..', 'webapp', 'data');
const ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8';

// Full 2025 MLB regular season range
const START_DATE = '2025-04-01';
const END_DATE = '2025-09-28';

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

function formatDate(d) {
  return d.getFullYear().toString() +
    String(d.getMonth() + 1).padStart(2, '0') +
    String(d.getDate()).padStart(2, '0');
}

// Load existing data to avoid re-downloading
function loadExisting() {
  let existingBox = [];
  let existingOdds = [];
  try {
    existingBox = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_player_boxscores.json'), 'utf8'));
  } catch (e) {}
  try {
    existingOdds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_historical_odds.json'), 'utf8'));
  } catch (e) {}
  return { existingBox, existingOdds };
}

// Generate all dates in range
const dates = [];
for (let d = new Date(START_DATE); d <= new Date(END_DATE); d.setDate(d.getDate() + 1)) {
  dates.push(formatDate(new Date(d)));
}

const { existingBox, existingOdds } = loadExisting();
const existingBoxDates = new Set(existingBox.map(b => b.date));
const existingOddsDates = new Set(existingOdds.map(o => o.date));
const existingBoxEventIds = new Set(existingBox.map(b => b.event_id));

console.log('=== MLB Expanded Data Download ===');
console.log(`Target range: ${START_DATE} to ${END_DATE} (${dates.length} dates)`);
console.log(`Existing boxscores: ${existingBox.length} games across ${existingBoxDates.size} dates`);
console.log(`Existing odds: ${existingOdds.length} records across ${existingOddsDates.size} dates`);

// =============================================================================
// STEP 1: ESPN boxscores (FREE) — download ALL missing dates
// =============================================================================
console.log('\n=== STEP 1: Downloading ESPN boxscores (FREE) ===\n');

const newBoxscores = [];
let espnSkipped = 0;
let espnDownloaded = 0;

for (const dateStr of dates) {
  // Skip dates we already have (but allow partial dates to be re-checked)
  if (existingBoxDates.has(dateStr)) {
    espnSkipped++;
    continue;
  }

  const url = `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=${dateStr}`;
  const data = curlJSON(url);

  if (!data || !data.events || data.events.length === 0) continue;

  const finalGames = data.events.filter(e => e.status?.type?.name === 'STATUS_FINAL');
  if (finalGames.length === 0) continue;

  process.stdout.write(`  ${dateStr}: ${finalGames.length} games`);
  let dateGames = 0;

  for (const event of finalGames) {
    // Skip games we already have
    if (existingBoxEventIds.has(event.id)) continue;

    sleep(200);

    const summaryUrl = `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event=${event.id}`;
    const summary = curlJSON(summaryUrl);
    if (!summary) continue;

    const comp = event.competitions?.[0];
    if (!comp) continue;

    let homeTeam = '', awayTeam = '', homeScore = 0, awayScore = 0;
    for (const c of (comp.competitors || [])) {
      if (c.homeAway === 'home') {
        homeTeam = tri(c.team?.abbreviation || '');
        homeScore = parseInt(c.score || '0');
      } else {
        awayTeam = tri(c.team?.abbreviation || '');
        awayScore = parseInt(c.score || '0');
      }
    }

    const players = [];
    if (summary.boxscore?.players) {
      for (const teamBox of summary.boxscore.players) {
        const teamAbbr = tri(teamBox.team?.abbreviation || '');
        for (const statGroup of (teamBox.statistics || [])) {
          const statName = (statGroup.type || statGroup.name || '').toLowerCase();
          if (!statName.includes('bat')) continue;

          const keys = statGroup.keys || statGroup.labels || [];
          for (const athlete of (statGroup.athletes || [])) {
            const name = athlete.athlete?.displayName || '';
            const vals = athlete.stats || [];
            const statMap = {};
            for (let k = 0; k < keys.length; k++) {
              statMap[keys[k].toUpperCase()] = vals[k] || '0';
            }

            const ab = parseInt(statMap['AB'] || statMap['ATBATS'] || '0');
            if (ab < 1) continue;

            players.push({
              team: teamAbbr,
              name,
              pid: athlete.athlete?.id || '',
              ab,
              h: parseInt(statMap['H'] || statMap['HITS'] || '0'),
              r: parseInt(statMap['R'] || statMap['RUNS'] || '0'),
              rbi: parseInt(statMap['RBI'] || statMap['RBIS'] || '0'),
              hr: parseInt(statMap['HR'] || statMap['HOMERUNS'] || '0'),
              bb: parseInt(statMap['BB'] || statMap['WALKS'] || '0'),
              so: parseInt(statMap['K'] || statMap['SO'] || statMap['STRIKEOUTS'] || '0'),
              sb: parseInt(statMap['SB'] || statMap['STOLENBASES'] || '0'),
              tb: parseInt(statMap['TB'] || statMap['TOTALBASES'] || '0'),
            });
          }
        }
      }
    }

    if (players.length > 0) {
      newBoxscores.push({
        event_id: event.id,
        date: dateStr,
        home: homeTeam,
        away: awayTeam,
        home_score: homeScore,
        away_score: awayScore,
        players,
      });
      dateGames++;
    }
  }

  espnDownloaded++;
  process.stdout.write(` -> ${dateGames} new\n`);
  sleep(250);

  // Save incrementally every 10 dates
  if (espnDownloaded % 10 === 0) {
    const mergedBox = [...existingBox, ...newBoxscores].sort((a, b) => a.date.localeCompare(b.date));
    fs.writeFileSync(path.join(DATA_DIR, 'mlb_player_boxscores_expanded.json'), JSON.stringify(mergedBox, null, 2));
    process.stdout.write(`    [saved checkpoint: ${mergedBox.length} total games]\n`);
  }
}

// Merge and save final boxscores
const allBoxscores = [...existingBox, ...newBoxscores].sort((a, b) => a.date.localeCompare(b.date));
const boxPath = path.join(DATA_DIR, 'mlb_player_boxscores_expanded.json');
fs.writeFileSync(boxPath, JSON.stringify(allBoxscores, null, 2));

const allBoxDates = [...new Set(allBoxscores.map(b => b.date))].sort();
console.log(`\n  ESPN Summary:`);
console.log(`  Skipped dates (already had): ${espnSkipped}`);
console.log(`  Downloaded dates: ${espnDownloaded}`);
console.log(`  New games: ${newBoxscores.length}`);
console.log(`  Total games: ${allBoxscores.length}`);
console.log(`  Total dates: ${allBoxDates.length}`);
console.log(`  Range: ${allBoxDates[0]} - ${allBoxDates[allBoxDates.length - 1]}`);
console.log(`  Saved to: ${boxPath}\n`);

// =============================================================================
// STEP 2: Odds API (COSTS MONEY) — only fetch dates NOT already in existing data
// =============================================================================
// Only fetch odds for dates where we have boxscores but NOT odds
const newOddsDates = allBoxDates.filter(d => !existingOddsDates.has(d));
console.log(`=== STEP 2: Downloading odds for ${newOddsDates.length} NEW dates (skipping ${existingOddsDates.size} existing) ===\n`);

const newOdds = [];
let oddsApiCalls = 0;

for (const dateStr of newOddsDates) {
  const y = dateStr.substring(0, 4);
  const m = dateStr.substring(4, 6);
  const dd = dateStr.substring(6, 8);
  const isoDate = `${y}-${m}-${dd}T17:00:00Z`;

  process.stdout.write(`  ${dateStr}...`);

  const gamesUrl = `https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds/?apiKey=${ODDS_API_KEY}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
  const gamesData = curlJSON(gamesUrl);
  oddsApiCalls++;
  sleep(500);

  if (!gamesData || !gamesData.data || gamesData.data.length === 0) {
    process.stdout.write(' no odds\n');
    continue;
  }

  // Log remaining API usage if available
  if (gamesData.remaining_requests !== undefined) {
    process.stdout.write(` [API remaining: ${gamesData.remaining_requests}]`);
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

    // Fetch batter props — use correct market keys (NOT _alternate suffix)
    sleep(500);
    const propsUrl = `https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/${game.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=batter_hits,batter_total_bases,batter_rbis,batter_runs_scored&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
    const propsData = curlJSON(propsUrl);
    oddsApiCalls++;

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
        if (Object.keys(hitsProps).length > 0) { record.hitsProps = hitsProps; dateProps++; }
        if (Object.keys(tbProps).length > 0) record.tbProps = tbProps;
        if (Object.keys(rbiProps).length > 0) record.rbiProps = rbiProps;
        if (Object.keys(runsProps).length > 0) record.runsProps = runsProps;
      }
    }

    newOdds.push(record);
  }

  process.stdout.write(` (${dateProps} with props) -> ${newOdds.length} total new\n`);

  // Save incrementally every 5 dates to protect against interruption
  if (newOddsDates.indexOf(dateStr) % 5 === 4) {
    const mergedOdds = [...existingOdds, ...newOdds].sort((a, b) => a.date.localeCompare(b.date));
    fs.writeFileSync(path.join(DATA_DIR, 'mlb_historical_odds_expanded.json'), JSON.stringify(mergedOdds, null, 2));
    process.stdout.write(`    [saved checkpoint: ${mergedOdds.length} total odds records, ${oddsApiCalls} API calls]\n`);
  }

  sleep(500);
}

// Final merge and save odds
const allOdds = [...existingOdds, ...newOdds].sort((a, b) => a.date.localeCompare(b.date));
const oddsPath = path.join(DATA_DIR, 'mlb_historical_odds_expanded.json');
fs.writeFileSync(oddsPath, JSON.stringify(allOdds, null, 2));

const allOddsDates = [...new Set(allOdds.map(o => o.date))].sort();
const withHits = allOdds.filter(o => o.hitsProps && Object.keys(o.hitsProps).length > 0).length;
const withTB = allOdds.filter(o => o.tbProps && Object.keys(o.tbProps).length > 0).length;
const withRBI = allOdds.filter(o => o.rbiProps && Object.keys(o.rbiProps).length > 0).length;

console.log(`\n=== FINAL SUMMARY ===`);
console.log(`  Total games (boxscores): ${allBoxscores.length}`);
console.log(`  Total boxscore dates: ${allBoxDates.length}`);
console.log(`  Boxscore range: ${allBoxDates[0]} - ${allBoxDates[allBoxDates.length - 1]}`);
console.log(`  Total odds records: ${allOdds.length}`);
console.log(`  Total odds dates: ${allOddsDates.length}`);
console.log(`  Odds range: ${allOddsDates[0]} - ${allOddsDates[allOddsDates.length - 1]}`);
console.log(`  With batter hits props: ${withHits}`);
console.log(`  With total bases props: ${withTB}`);
console.log(`  With RBI props: ${withRBI}`);
console.log(`  Odds API calls used: ${oddsApiCalls}`);
console.log(`\n  Boxscores saved to: ${boxPath}`);
console.log(`  Odds saved to: ${oddsPath}`);
console.log(`\n  NOTE: These are saved as *_expanded.json files.`);
console.log(`  After verifying, copy them over the originals to use in backtesting.`);
