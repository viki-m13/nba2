#!/usr/bin/env node
// =============================================================================
// Quick MLB data download — focused on 30 days with correct ESPN parsing
// =============================================================================

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const DATA_DIR = path.join(__dirname, '..', 'webapp', 'data');
const WEBAPP_DATA_DIR = path.join(__dirname, '..', '..', 'webapp', 'mlb', 'data');
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
    const result = execSync(`curl -s --max-time 20 "${url}"`, { maxBuffer: 10 * 1024 * 1024 });
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

// Fetch Aug 15 - Sep 15 2025 (peak regular season, 30 days)
const dates = [];
const start = new Date('2025-08-15');
const end = new Date('2025-09-15');
for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
  dates.push(formatDate(new Date(d)));
}

console.log(`=== Downloading MLB Data (${dates[0]} to ${dates[dates.length-1]}) ===\n`);

// STEP 1: ESPN boxscores
const allBoxscores = [];
for (const dateStr of dates) {
  const url = `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=${dateStr}`;
  const data = curlJSON(url);
  if (!data || !data.events) continue;

  const finalGames = data.events.filter(e => e.status?.type?.name === 'STATUS_FINAL');
  if (finalGames.length === 0) continue;

  console.log(`  ${dateStr}: ${finalGames.length} games`);

  for (const event of finalGames) {
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
          const stype = (statGroup.type || statGroup.name || '').toLowerCase();
          if (!stype.includes('bat')) continue;

          // ESPN labels: ['hits-atBats', 'atBats', 'runs', 'hits', 'RBIs', 'homeRuns', 'walks', 'strikeouts', ...]
          const keys = statGroup.keys || statGroup.labels || [];
          const abIdx = keys.indexOf('atBats');
          const rIdx = keys.indexOf('runs');
          const hIdx = keys.indexOf('hits');
          const rbiIdx = keys.indexOf('RBIs');
          const hrIdx = keys.indexOf('homeRuns');
          const bbIdx = keys.indexOf('walks');
          const soIdx = keys.indexOf('strikeouts');

          for (const athlete of (statGroup.athletes || [])) {
            const name = athlete.athlete?.displayName || '';
            const vals = athlete.stats || [];
            if (!name || vals.length === 0) continue;

            const ab = parseInt(vals[abIdx] || '0');
            if (ab < 1) continue;

            const h = parseInt(vals[hIdx] || '0');
            const hr = parseInt(vals[hrIdx] || '0');

            // Calculate total bases (estimate: H + 2B + 2*3B + 3*HR; approximate as H + HR*3 for singles-heavy)
            // Better: TB = H + doubles + 2*triples + 3*HR. Without breakdown, use H + HR*3 as lower bound
            // Actually we can get it from slugging * AB if available, but simpler: use hits + extra bases
            const tb = h + hr * 3; // conservative estimate (all non-HR hits = singles)

            players.push({
              team: teamAbbr,
              name,
              pid: athlete.athlete?.id || '',
              ab,
              h,
              r: parseInt(vals[rIdx] || '0'),
              rbi: parseInt(vals[rbiIdx] || '0'),
              hr,
              bb: parseInt(vals[bbIdx] || '0'),
              so: parseInt(vals[soIdx] || '0'),
              sb: 0,
              tb,
            });
          }
        }
      }
    }

    if (players.length > 0) {
      allBoxscores.push({
        event_id: event.id,
        date: dateStr,
        home: homeTeam,
        away: awayTeam,
        home_score: homeScore,
        away_score: awayScore,
        players,
      });
    }
  }

  sleep(300);
}

allBoxscores.sort((a, b) => a.date.localeCompare(b.date));

// Save to both locations
const boxPath1 = path.join(DATA_DIR, 'mlb_player_boxscores.json');
const boxPath2 = path.join(WEBAPP_DATA_DIR, 'mlb_player_boxscores.json');
fs.writeFileSync(boxPath1, JSON.stringify(allBoxscores, null, 2));
fs.writeFileSync(boxPath2, JSON.stringify(allBoxscores, null, 2));

const boxDates = [...new Set(allBoxscores.map(b => b.date))].sort();
console.log(`\n  Total games: ${allBoxscores.length}, dates: ${boxDates.length}`);

// STEP 2: Historical odds
console.log(`\nSTEP 2: Fetching odds for ${boxDates.length} dates...\n`);

const allOdds = [];
for (const dateStr of boxDates) {
  const y = dateStr.substring(0, 4);
  const m = dateStr.substring(4, 6);
  const dd = dateStr.substring(6, 8);
  const isoDate = `${y}-${m}-${dd}T17:00:00Z`;

  const gamesUrl = `https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds/?apiKey=${ODDS_API_KEY}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
  const gamesData = curlJSON(gamesUrl);
  sleep(500);

  if (!gamesData || !gamesData.data || gamesData.data.length === 0) {
    console.log(`  ${dateStr}: no odds`);
    continue;
  }

  console.log(`  ${dateStr}: ${gamesData.data.length} games with odds`);

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
        }
      }
    }

    // Fetch batter props
    sleep(400);
    const propsUrl = `https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/${game.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=batter_hits_alternate,batter_total_bases_alternate,batter_rbis_alternate,batter_runs_scored_alternate&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
    const propsData = curlJSON(propsUrl);

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
        if (Object.keys(hitsProps).length > 0) record.hitsProps = hitsProps;
        if (Object.keys(tbProps).length > 0) record.tbProps = tbProps;
        if (Object.keys(rbiProps).length > 0) record.rbiProps = rbiProps;
        if (Object.keys(runsProps).length > 0) record.runsProps = runsProps;
      }
    }

    allOdds.push(record);
  }
  sleep(400);
}

allOdds.sort((a, b) => a.date.localeCompare(b.date));
const oddsPath1 = path.join(DATA_DIR, 'mlb_historical_odds.json');
const oddsPath2 = path.join(WEBAPP_DATA_DIR, 'mlb_historical_odds.json');
fs.writeFileSync(oddsPath1, JSON.stringify(allOdds, null, 2));
fs.writeFileSync(oddsPath2, JSON.stringify(allOdds, null, 2));

const withHits = allOdds.filter(o => o.hitsProps).length;
const withTB = allOdds.filter(o => o.tbProps).length;

console.log(`\n=== SUMMARY ===`);
console.log(`  Games: ${allBoxscores.length}`);
console.log(`  Odds records: ${allOdds.length}`);
console.log(`  With hits props: ${withHits}`);
console.log(`  With TB props: ${withTB}`);
console.log(`  Saved to ${boxPath1} & ${oddsPath1}`);
