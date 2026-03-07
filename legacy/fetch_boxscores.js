// Fetch player box score data from ESPN for prop analysis
const fs = require('fs');
const https = require('https');

const data = JSON.parse(fs.readFileSync('./webapp/data/espn_full_season_2025.json', 'utf8'));
const sorted = [...data].sort((a, b) => (a.date || '').localeCompare(b.date || ''));

// Only fetch recent games (Dec-Feb) to stay within reasonable limits
const recent = sorted.filter(g => g.date >= '20241201' && g.event_id);
console.log(`${recent.length} games to fetch box scores for (Dec 2024 - Feb 2025)`);

function fetchJSON(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } }, res => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(e); }
      });
    }).on('error', reject);
  });
}

async function fetchBoxScore(eventId) {
  const url = `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=${eventId}`;
  const data = await fetchJSON(url);

  const players = [];
  const boxscore = data.boxscore;
  if (!boxscore || !boxscore.players) return null;

  for (const team of boxscore.players) {
    const teamAbbr = team.team.abbreviation;
    for (const stat of (team.statistics || [])) {
      for (const athlete of (stat.athletes || [])) {
        const stats = athlete.stats;
        if (!stats || stats.length < 13) continue;
        // ESPN box score stat order: MIN, FG, 3PT, FT, OREB, DREB, REB, AST, STL, BLK, TO, PF, PTS
        const mins = parseInt(stats[0]) || 0;
        if (mins < 10) continue; // skip low-minute players

        players.push({
          name: athlete.athlete.displayName,
          id: athlete.athlete.id,
          team: teamAbbr,
          mins,
          pts: parseInt(stats[12]) || 0,
          reb: parseInt(stats[6]) || 0,
          ast: parseInt(stats[7]) || 0,
          stl: parseInt(stats[8]) || 0,
          blk: parseInt(stats[9]) || 0,
          fg: stats[1], // e.g. "10-18"
          threePt: stats[2],
          ft: stats[3],
        });
      }
    }
  }

  return players;
}

async function main() {
  const allBoxScores = [];
  let fetched = 0;
  let errors = 0;

  for (const game of recent) {
    try {
      const players = await fetchBoxScore(game.event_id);
      if (players && players.length > 0) {
        allBoxScores.push({
          date: game.date,
          home: game.home_team,
          away: game.away_team,
          eventId: game.event_id,
          players,
        });
      }
      fetched++;
      if (fetched % 20 === 0) console.log(`Fetched ${fetched}/${recent.length} (${errors} errors)...`);

      // Rate limit: 200ms between requests
      await new Promise(r => setTimeout(r, 200));
    } catch (e) {
      errors++;
      if (errors < 5) console.log(`Error fetching ${game.event_id}: ${e.message}`);
    }
  }

  console.log(`\nDone: ${allBoxScores.length} games with box scores, ${errors} errors`);

  // Save to file
  fs.writeFileSync('./webapp/data/player_boxscores.json', JSON.stringify(allBoxScores, null, 2));
  console.log('Saved to webapp/data/player_boxscores.json');

  // Quick stats
  const allPlayers = {};
  for (const game of allBoxScores) {
    for (const p of game.players) {
      if (!allPlayers[p.id]) allPlayers[p.id] = { name: p.name, games: [] };
      allPlayers[p.id].games.push({ date: game.date, pts: p.pts, reb: p.reb, ast: p.ast, mins: p.mins });
    }
  }

  console.log(`\n${Object.keys(allPlayers).length} unique players tracked`);

  // Show top scorers
  const topScorers = Object.values(allPlayers)
    .filter(p => p.games.length >= 20)
    .map(p => ({ name: p.name, games: p.games.length, avgPts: (p.games.reduce((s, g) => s + g.pts, 0) / p.games.length).toFixed(1) }))
    .sort((a, b) => b.avgPts - a.avgPts)
    .slice(0, 15);

  console.log('\nTop 15 Scorers (20+ games):');
  topScorers.forEach(p => console.log(`  ${p.name}: ${p.avgPts} PPG (${p.games} games)`));
}

main().catch(console.error);
