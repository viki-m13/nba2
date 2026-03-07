// Fetch historical FanDuel odds for the 2024-25 season to power the Master Bet Recommender
// This downloads h2h, spreads, totals for each game date and saves to a JSON file.
// Uses The Odds API historical endpoints (requires paid plan).

const fs = require('fs');
const https = require('https');

const API_KEY = '3879c3373a31421d8ef7d428b8758cd8';
const API_BASE = 'https://api.the-odds-api.com/v4';

// Team name mapping from Odds API to our abbreviations
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

function fetchJSON(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(data), headers: res.headers });
        } catch (e) { reject(new Error('JSON parse error: ' + data.slice(0, 200))); }
      });
    }).on('error', reject);
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
  // Load season data to get game dates
  const seasonData = JSON.parse(fs.readFileSync('data/espn_full_season_2025.json', 'utf8'));
  const dates = [...new Set(seasonData.map(g => g.date))].sort();
  console.log(`Found ${dates.length} game dates from ${dates[0]} to ${dates[dates.length - 1]}`);

  // Load existing historical odds if any
  let existingOdds = [];
  const OUTPUT_FILE = 'webapp/data/historical_odds.json';
  try {
    existingOdds = JSON.parse(fs.readFileSync(OUTPUT_FILE, 'utf8'));
    console.log(`Loaded ${existingOdds.length} existing historical odds records`);
  } catch (e) { /* no existing file */ }

  const existingDates = new Set(existingOdds.map(o => o.date));
  const allOdds = [...existingOdds];

  // Process each date, skip already fetched
  let fetched = 0;
  let apiCalls = 0;

  for (const dateStr of dates) {
    if (existingDates.has(dateStr)) {
      continue;
    }

    const isoDate = `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}T12:00:00Z`;

    try {
      // Fetch game-level odds (h2h, spreads, totals) - 1 API call
      const oddsUrl = `${API_BASE}/historical/sports/basketball_nba/odds?apiKey=${API_KEY}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&date=${isoDate}&bookmakers=fanduel`;
      const oddsResult = await fetchJSON(oddsUrl);
      apiCalls++;

      if (oddsResult.status !== 200) {
        console.warn(`  ${dateStr}: API returned ${oddsResult.status}`);
        await sleep(1000);
        continue;
      }

      const remaining = oddsResult.headers['x-requests-remaining'];
      const events = oddsResult.data.data || [];

      for (const event of events) {
        const homeAbbr = teamAbbr(event.home_team);
        const awayAbbr = teamAbbr(event.away_team);
        const gameKey = `${awayAbbr}@${homeAbbr}`;

        const fd = (event.bookmakers || []).find(b => b.key === 'fanduel');
        if (!fd) continue;

        const record = {
          date: dateStr,
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

        allOdds.push(record);
      }

      fetched++;
      console.log(`  ${dateStr}: ${events.length} games (${remaining} API calls remaining, ${apiCalls} used)`);

      // Rate limiting
      await sleep(500);

      // Safety: check remaining requests
      if (parseInt(remaining) < 100) {
        console.log('WARNING: Low API quota, stopping early to preserve requests');
        break;
      }

    } catch (e) {
      console.error(`  ${dateStr}: Error: ${e.message}`);
      await sleep(2000);
    }
  }

  // Save results
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allOdds, null, 2));
  console.log(`\nSaved ${allOdds.length} historical odds records to ${OUTPUT_FILE}`);
  console.log(`Fetched ${fetched} new dates using ${apiCalls} API calls`);
}

main().catch(console.error);
