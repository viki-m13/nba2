// Consolidate all raw historical odds files into a single structured JSON
const fs = require('fs');
const path = require('path');

const RAW_DIR = 'webapp/data/historical_odds_raw';
const OUTPUT = 'webapp/data/historical_odds.json';

const TEAM_MAP = {
  'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
  'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
  'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
  'Golden State Warriors': 'GS', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
  'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',
  'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
  'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
  'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NO',
  'New York Knicks': 'NY', 'Oklahoma City Thunder': 'OKC',
  'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
  'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
  'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SA',
  'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTAH',
  'Washington Wizards': 'WSH',
};

function teamAbbr(name) { return TEAM_MAP[name] || name; }

const files = fs.readdirSync(RAW_DIR).sort();
const gameOddsFiles = files.filter(f => f.startsWith('game_odds_'));
const propsFiles = files.filter(f => f.startsWith('props_'));

console.log(`Game odds files: ${gameOddsFiles.length}`);
console.log(`Player prop files: ${propsFiles.length}`);

// Build props lookup: date_eventId -> player props data
const propsMap = {};
for (const pf of propsFiles) {
  const match = pf.match(/props_(\d{8})_(.+)\.json/);
  if (!match) continue;
  const [, date, eventId] = match;
  try {
    const raw = JSON.parse(fs.readFileSync(path.join(RAW_DIR, pf), 'utf8'));
    const eventData = raw.data || raw;
    const fdBook = (eventData.bookmakers || []).find(b => b.key === 'fanduel');
    if (!fdBook) continue;
    const mkt = (fdBook.markets || []).find(m => m.key === 'player_points_alternate');
    if (!mkt) continue;

    const playerLines = {};
    for (const outcome of (mkt.outcomes || [])) {
      const player = outcome.description;
      const threshold = outcome.point;
      if (!playerLines[player]) playerLines[player] = {};
      if (!playerLines[player][threshold]) playerLines[player][threshold] = {};
      if (outcome.name === 'Over') {
        playerLines[player][threshold].overOdds = outcome.price;
      } else {
        playerLines[player][threshold].underOdds = outcome.price;
      }
    }
    propsMap[`${date}_${eventId}`] = playerLines;
  } catch (e) { /* skip bad files */ }
}

console.log(`Parsed player props for ${Object.keys(propsMap).length} events`);

// Build consolidated historical odds
const allOdds = [];

for (const gf of gameOddsFiles) {
  const match = gf.match(/game_odds_(\d{8})\.json/);
  if (!match) continue;
  const date = match[1];

  try {
    const raw = JSON.parse(fs.readFileSync(path.join(RAW_DIR, gf), 'utf8'));
    const events = raw.data || [];

    for (const event of events) {
      const homeAbbr = teamAbbr(event.home_team);
      const awayAbbr = teamAbbr(event.away_team);
      const gameKey = `${awayAbbr}@${homeAbbr}`;

      const fd = (event.bookmakers || []).find(b => b.key === 'fanduel');
      if (!fd) continue;

      const record = {
        date,
        gameKey,
        eventId: event.id,
        homeTeam: homeAbbr,
        awayTeam: awayAbbr,
        commenceTime: event.commence_time,
      };

      for (const mkt of (fd.markets || [])) {
        if (mkt.key === 'h2h') {
          const homeO = mkt.outcomes.find(o => o.name === event.home_team);
          const awayO = mkt.outcomes.find(o => o.name === event.away_team);
          if (homeO) record.home_ml = homeO.price;
          if (awayO) record.away_ml = awayO.price;
        } else if (mkt.key === 'spreads') {
          const homeO = mkt.outcomes.find(o => o.name === event.home_team);
          const awayO = mkt.outcomes.find(o => o.name === event.away_team);
          if (homeO) { record.spread_home = homeO.price; record.spread_point = homeO.point; }
          if (awayO) { record.spread_away = awayO.price; }
        } else if (mkt.key === 'totals') {
          const overO = mkt.outcomes.find(o => o.name === 'Over');
          const underO = mkt.outcomes.find(o => o.name === 'Under');
          if (overO) { record.total = overO.point; record.total_over = overO.price; }
          if (underO) { record.total_under = underO.price; }
        }
      }

      // Attach player props if we have them
      const propsKey = `${date}_${event.id}`;
      if (propsMap[propsKey]) {
        record.playerProps = propsMap[propsKey];
      }

      allOdds.push(record);
    }
  } catch (e) {
    console.error(`Error processing ${gf}: ${e.message}`);
  }
}

// Sort by date
allOdds.sort((a, b) => a.date.localeCompare(b.date) || a.gameKey.localeCompare(b.gameKey));

fs.writeFileSync(OUTPUT, JSON.stringify(allOdds, null, 2));

// Stats
const withProps = allOdds.filter(g => g.playerProps);
const withML = allOdds.filter(g => g.home_ml);
const dates = [...new Set(allOdds.map(g => g.date))];
console.log(`\nConsolidated: ${allOdds.length} games across ${dates.length} dates`);
console.log(`  With ML odds: ${withML.length}`);
console.log(`  With player props: ${withProps.length}`);
console.log(`  Date range: ${dates[0]} to ${dates[dates.length-1]}`);
console.log(`  File size: ${(fs.statSync(OUTPUT).size / 1024 / 1024).toFixed(1)} MB`);
