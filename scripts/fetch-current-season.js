#!/usr/bin/env node
// =============================================================================
// Fetch recent 2025-26 NBA season data for backtesting
// Downloads boxscores from ESPN and odds from The Odds API using curl
// Saves data in same format as existing files for future reuse
// =============================================================================

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const DATA_DIR = path.join(__dirname, '..', 'webapp', 'data');
const ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8';
const DAYS_BACK = 30;

const ESPN_TO_TRICODE = {
  'ATL': 'ATL', 'BOS': 'BOS', 'BKN': 'BKN', 'CHA': 'CHA', 'CHI': 'CHI',
  'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GS': 'GS',
  'GSW': 'GS', 'HOU': 'HOU', 'IND': 'IND', 'LAC': 'LAC', 'LAL': 'LAL',
  'MEM': 'MEM', 'MIA': 'MIA', 'MIL': 'MIL', 'MIN': 'MIN', 'NO': 'NO',
  'NOP': 'NO', 'NY': 'NY', 'NYK': 'NY', 'OKC': 'OKC', 'ORL': 'ORL',
  'PHI': 'PHI', 'PHX': 'PHX', 'POR': 'POR', 'SA': 'SA', 'SAS': 'SA',
  'SAC': 'SAC', 'TOR': 'TOR', 'UTA': 'UTA', 'UTAH': 'UTA', 'WAS': 'WAS',
};

const TEAM_NAME_MAP = {
  'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
  'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
  'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
  'Golden State Warriors': 'GS', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
  'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
  'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
  'New Orleans Pelicans': 'NO', 'New York Knicks': 'NY', 'Oklahoma City Thunder': 'OKC',
  'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
  'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SA',
  'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
};

function tri(abbr) { return ESPN_TO_TRICODE[abbr] || TEAM_NAME_MAP[abbr] || abbr; }

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

console.log('=== Fetching 2025-26 NBA Season Data ===\n');
console.log('STEP 1: Fetching ESPN boxscores...');

const allBoxscores = [];
const today = new Date();

for (let i = 1; i <= DAYS_BACK; i++) {
  const d = new Date(today);
  d.setDate(d.getDate() - i);
  const dateStr = formatDate(d);

  const url = `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=${dateStr}`;
  const data = curlJSON(url);

  if (!data || !data.events || data.events.length === 0) {
    continue;
  }

  const finalGames = data.events.filter(e => e.status?.type?.name === 'STATUS_FINAL');
  if (finalGames.length === 0) continue;

  console.log(`  ${dateStr}: ${finalGames.length} games`);

  for (const event of finalGames) {
    sleep(250);

    const summaryUrl = `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=${event.id}`;
    const summary = curlJSON(summaryUrl);
    if (!summary) continue;

    // Extract teams and scores
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

    // Extract player stats
    // ESPN keys: minutes, points, rebounds, assists, turnovers, steals, blocks,
    //   fieldGoalsMade-fieldGoalsAttempted, threePointFieldGoalsMade-..., freeThrowsMade-...
    const players = [];
    if (summary.boxscore?.players) {
      for (const teamBox of summary.boxscore.players) {
        const teamAbbr = tri(teamBox.team?.abbreviation || '');
        for (const statGroup of (teamBox.statistics || [])) {
          const keys = statGroup.keys || [];
          for (const athlete of (statGroup.athletes || [])) {
            const name = athlete.athlete?.displayName || '';
            const vals = athlete.stats || [];
            const statMap = {};
            for (let k = 0; k < keys.length; k++) {
              statMap[keys[k]] = vals[k] || '0';
            }

            const mins = parseInt(statMap['minutes'] || '0');
            if (mins < 1) continue;

            players.push({
              team: teamAbbr,
              name,
              pid: athlete.athlete?.id || '',
              min: statMap['minutes'] || '0',
              pts: parseInt(statMap['points'] || '0'),
              fg: statMap['fieldGoalsMade-fieldGoalsAttempted'] || '0-0',
              three: statMap['threePointFieldGoalsMade-threePointFieldGoalsAttempted'] || '0-0',
              ft: statMap['freeThrowsMade-freeThrowsAttempted'] || '0-0',
              reb: parseInt(statMap['rebounds'] || '0'),
              ast: parseInt(statMap['assists'] || '0'),
              to: parseInt(statMap['turnovers'] || '0'),
              stl: parseInt(statMap['steals'] || '0'),
              blk: parseInt(statMap['blocks'] || '0'),
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
const boxPath = path.join(DATA_DIR, 'player_boxscores_2026.json');
fs.writeFileSync(boxPath, JSON.stringify(allBoxscores, null, 2));

const dates = [...new Set(allBoxscores.map(b => b.date))].sort();
console.log(`\n  Total games: ${allBoxscores.length}`);
console.log(`  Date range: ${dates[0] || 'N/A'} to ${dates[dates.length - 1] || 'N/A'} (${dates.length} dates)`);
console.log(`  Saved to ${boxPath}`);

if (allBoxscores.length === 0) {
  console.log('\nNo boxscores found. Exiting.');
  process.exit(1);
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

  const gamesUrl = `https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds/?apiKey=${ODDS_API_KEY}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
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

    // Parse game-level odds
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

    // Fetch player props
    sleep(500);
    const propsUrl = `https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events/${game.id}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=player_points_alternate,player_rebounds_alternate,player_assists_alternate,player_points_rebounds_assists,player_points_rebounds_assists_alternate&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
    const propsData = curlJSON(propsUrl);

    if (propsData?.data) {
      const pfb = (propsData.data.bookmakers || []).find(b => b.key === 'fanduel');
      if (pfb) {
        const playerProps = {}, rebProps = {}, astProps = {}, praProps = {};
        for (const mkt of (pfb.markets || [])) {
          for (const o of (mkt.outcomes || [])) {
            if (o.name === 'Over' && o.description && o.point !== undefined) {
              const target = mkt.key === 'player_points_alternate' ? playerProps
                : mkt.key === 'player_rebounds_alternate' ? rebProps
                : mkt.key === 'player_assists_alternate' ? astProps
                : (mkt.key === 'player_points_rebounds_assists' || mkt.key === 'player_points_rebounds_assists_alternate') ? praProps
                : null;
              if (target) {
                if (!target[o.description]) target[o.description] = {};
                target[o.description][o.point] = { overOdds: o.price };
              }
            }
          }
        }
        if (Object.keys(playerProps).length > 0) record.playerProps = playerProps;
        if (Object.keys(rebProps).length > 0) record.rebProps = rebProps;
        if (Object.keys(astProps).length > 0) record.astProps = astProps;
        if (Object.keys(praProps).length > 0) record.praProps = praProps;
      }
    }

    allOdds.push(record);
  }

  sleep(500);
}

allOdds.sort((a, b) => a.date.localeCompare(b.date));
const oddsPath = path.join(DATA_DIR, 'historical_odds_2026.json');
fs.writeFileSync(oddsPath, JSON.stringify(allOdds, null, 2));

const withPts = allOdds.filter(o => o.playerProps && Object.keys(o.playerProps).length > 0).length;
const withReb = allOdds.filter(o => o.rebProps && Object.keys(o.rebProps).length > 0).length;
const withAst = allOdds.filter(o => o.astProps && Object.keys(o.astProps).length > 0).length;

console.log(`\n=== SUMMARY ===`);
console.log(`  Games: ${allBoxscores.length}`);
console.log(`  Dates: ${dates.length}`);
console.log(`  Odds records: ${allOdds.length}`);
console.log(`  With player point props: ${withPts}`);
console.log(`  With rebound props: ${withReb}`);
console.log(`  With assist props: ${withAst}`);
console.log(`\n  Boxscores saved to: ${boxPath}`);
console.log(`  Odds saved to: ${oddsPath}`);
