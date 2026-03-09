#!/usr/bin/env node
// =============================================================================
// Fetch MLB 2025 season data for backtesting
// Downloads boxscores from ESPN and odds from The Odds API using curl
// Saves data in same format as NBA files for the Ultra Engine
// =============================================================================

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const DATA_DIR = path.join(__dirname, '..', 'webapp', 'data');
const ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8';
const DAYS_BACK = 60;

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

// =============================================================================
// STEP 1: Fetch ESPN boxscores
// =============================================================================

console.log('=== Fetching MLB Season Data ===\n');
console.log('STEP 1: Fetching ESPN boxscores...');

const allBoxscores = [];
const today = new Date();

for (let i = 1; i <= DAYS_BACK; i++) {
  const d = new Date(today);
  d.setDate(d.getDate() - i);
  const dateStr = formatDate(d);

  const url = `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=${dateStr}`;
  const data = curlJSON(url);

  if (!data || !data.events || data.events.length === 0) {
    continue;
  }

  const finalGames = data.events.filter(e => e.status?.type?.name === 'STATUS_FINAL');
  if (finalGames.length === 0) continue;

  console.log(`  ${dateStr}: ${finalGames.length} games`);

  for (const event of finalGames) {
    sleep(250);

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

    // Extract batter stats from box score
    const players = [];
    if (summary.boxscore?.players) {
      for (const teamBox of summary.boxscore.players) {
        const teamAbbr = tri(teamBox.team?.abbreviation || '');
        for (const statGroup of (teamBox.statistics || [])) {
          // Only get batting stats
          if (statGroup.type !== 'batting' && statGroup.name !== 'batting') continue;
          const keys = statGroup.keys || statGroup.labels || [];
          for (const athlete of (statGroup.athletes || [])) {
            const name = athlete.athlete?.displayName || '';
            const vals = athlete.stats || [];
            const statMap = {};
            for (let k = 0; k < keys.length; k++) {
              statMap[keys[k]] = vals[k] || '0';
            }

            // Parse AB — skip players who didn't bat
            const ab = parseInt(statMap['AB'] || statMap['atBats'] || '0');
            if (ab < 1) continue;

            players.push({
              team: teamAbbr,
              name,
              pid: athlete.athlete?.id || '',
              ab,
              h: parseInt(statMap['H'] || statMap['hits'] || '0'),
              r: parseInt(statMap['R'] || statMap['runs'] || '0'),
              rbi: parseInt(statMap['RBI'] || statMap['RBIs'] || '0'),
              hr: parseInt(statMap['HR'] || statMap['homeRuns'] || '0'),
              bb: parseInt(statMap['BB'] || statMap['walks'] || '0'),
              so: parseInt(statMap['K'] || statMap['SO'] || statMap['strikeouts'] || '0'),
              sb: parseInt(statMap['SB'] || statMap['stolenBases'] || '0'),
              tb: parseInt(statMap['TB'] || statMap['totalBases'] || '0'),
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
const boxPath = path.join(DATA_DIR, 'mlb_player_boxscores.json');
fs.writeFileSync(boxPath, JSON.stringify(allBoxscores, null, 2));

const dates = [...new Set(allBoxscores.map(b => b.date))].sort();
console.log(`\n  Total games: ${allBoxscores.length}`);
console.log(`  Date range: ${dates[0] || 'N/A'} to ${dates[dates.length - 1] || 'N/A'} (${dates.length} dates)`);
console.log(`  Saved to ${boxPath}`);

if (allBoxscores.length === 0) {
  console.log('\nNo boxscores found. MLB season may not be active. Creating empty files.');
  fs.writeFileSync(path.join(DATA_DIR, 'mlb_historical_odds.json'), '[]');
  process.exit(0);
}

// =============================================================================
// STEP 2: Fetch historical odds
// =============================================================================

console.log(`\nSTEP 2: Fetching historical odds for ${dates.length} dates...`);

const allOdds = [];

for (const dateStr of dates) {
  const y = dateStr.substring(0, 4);
  const m = dateStr.substring(4, 6);
  const dd = dateStr.substring(6, 8);
  const isoDate = `${y}-${m}-${dd}T17:00:00Z`;

  console.log(`  ${dateStr}...`);

  const gamesUrl = `https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds/?apiKey=${ODDS_API_KEY}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
  const gamesData = curlJSON(gamesUrl);
  sleep(500);

  if (!gamesData || !gamesData.data || gamesData.data.length === 0) {
    console.log(`    No odds data`);
    continue;
  }

  console.log(`    ${gamesData.data.length} games with odds`);

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

    // Fetch player props (batter props)
    sleep(500);
    const propsUrl = `https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/${game.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=batter_hits_alternate,batter_total_bases_alternate,batter_rbis_alternate,batter_runs_scored_alternate,batter_home_runs&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
    const propsData = curlJSON(propsUrl);

    if (propsData?.data) {
      const pfb = (propsData.data.bookmakers || []).find(b => b.key === 'fanduel');
      if (pfb) {
        const hitsProps = {}, tbProps = {}, rbiProps = {}, runsProps = {}, hrProps = {};
        for (const mkt of (pfb.markets || [])) {
          for (const o of (mkt.outcomes || [])) {
            if (o.name === 'Over' && o.description && o.point !== undefined) {
              const target = mkt.key.includes('hits') ? hitsProps
                : mkt.key.includes('total_bases') ? tbProps
                : mkt.key.includes('rbis') ? rbiProps
                : mkt.key.includes('runs') ? runsProps
                : mkt.key.includes('home_runs') ? hrProps : null;
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
        if (Object.keys(hrProps).length > 0) record.hrProps = hrProps;
      }
    }

    allOdds.push(record);
  }

  sleep(500);
}

allOdds.sort((a, b) => a.date.localeCompare(b.date));
const oddsPath = path.join(DATA_DIR, 'mlb_historical_odds.json');
fs.writeFileSync(oddsPath, JSON.stringify(allOdds, null, 2));

const withHits = allOdds.filter(o => o.hitsProps && Object.keys(o.hitsProps).length > 0).length;
const withTB = allOdds.filter(o => o.tbProps && Object.keys(o.tbProps).length > 0).length;
const withRBI = allOdds.filter(o => o.rbiProps && Object.keys(o.rbiProps).length > 0).length;

console.log(`\n=== SUMMARY ===`);
console.log(`  Games: ${allBoxscores.length}`);
console.log(`  Dates: ${dates.length}`);
console.log(`  Odds records: ${allOdds.length}`);
console.log(`  With batter hits props: ${withHits}`);
console.log(`  With total bases props: ${withTB}`);
console.log(`  With RBI props: ${withRBI}`);
console.log(`\n  Boxscores saved to: ${boxPath}`);
console.log(`  Odds saved to: ${oddsPath}`);
