// Explore what data is available from ESPN/NBA endpoints
const https = require('https');

function fetch(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); } catch(e) { resolve(data); }
      });
    }).on('error', reject);
  });
}

async function main() {
  console.log('═══════════════════════════════════════════════');
  console.log('EXPLORING ESPN & NBA DATA ENDPOINTS');
  console.log('═══════════════════════════════════════════════\n');

  // 1. ESPN Scoreboard - what fields are available?
  console.log('--- ESPN SCOREBOARD (sample game) ---');
  try {
    const sb = await fetch('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20250220');
    if (sb.events && sb.events.length > 0) {
      const event = sb.events[0];
      console.log('Event keys:', Object.keys(event));
      const comp = event.competitions[0];
      console.log('Competition keys:', Object.keys(comp));
      console.log('Status keys:', Object.keys(comp.status || {}));
      console.log('Status type:', comp.status?.type);

      // Check for odds/betting data
      if (comp.odds) {
        console.log('\n*** ODDS DATA FOUND ***');
        console.log(JSON.stringify(comp.odds, null, 2).slice(0, 2000));
      }

      // Check competitor details
      const team = comp.competitors[0];
      console.log('\nCompetitor keys:', Object.keys(team));
      if (team.statistics) console.log('Stats:', JSON.stringify(team.statistics).slice(0, 500));
      if (team.records) console.log('Records:', JSON.stringify(team.records).slice(0, 500));
      if (team.leaders) console.log('Leaders:', JSON.stringify(team.leaders).slice(0, 300));
    }
  } catch(e) { console.log('Error:', e.message); }

  // 2. ESPN Team stats endpoint
  console.log('\n--- ESPN TEAM STATS ---');
  try {
    const ts = await fetch('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/BOS/statistics');
    console.log('Team stats keys:', Object.keys(ts));
    if (ts.results) console.log('Results:', JSON.stringify(ts.results).slice(0, 500));
    if (ts.stats) console.log('Stats:', JSON.stringify(ts.stats).slice(0, 1000));
  } catch(e) { console.log('Error:', e.message); }

  // 3. ESPN standings
  console.log('\n--- ESPN STANDINGS ---');
  try {
    const st = await fetch('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/standings');
    if (st.children && st.children.length > 0) {
      const conf = st.children[0];
      console.log('Conference keys:', Object.keys(conf));
      if (conf.standings && conf.standings.entries && conf.standings.entries.length > 0) {
        const entry = conf.standings.entries[0];
        console.log('Standing entry keys:', Object.keys(entry));
        if (entry.stats) {
          console.log('Available stat names:', entry.stats.map(s => s.name || s.type).join(', '));
          console.log('Sample stats:', JSON.stringify(entry.stats.slice(0, 10), null, 2));
        }
      }
    }
  } catch(e) { console.log('Error:', e.message); }

  // 4. Check for ESPN game summary (box score level)
  console.log('\n--- ESPN GAME SUMMARY ---');
  try {
    const gs = await fetch('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=401704627');
    console.log('Summary keys:', Object.keys(gs));
    if (gs.boxscore) {
      console.log('Boxscore keys:', Object.keys(gs.boxscore));
      if (gs.boxscore.teams) {
        console.log('Team boxscore count:', gs.boxscore.teams.length);
        const t = gs.boxscore.teams[0];
        console.log('Team boxscore keys:', Object.keys(t));
        if (t.statistics) {
          console.log('Team stats:', JSON.stringify(t.statistics).slice(0, 1000));
        }
      }
    }
    if (gs.predictor) {
      console.log('\n*** PREDICTOR DATA ***');
      console.log(JSON.stringify(gs.predictor, null, 2).slice(0, 500));
    }
    if (gs.odds) {
      console.log('\n*** ODDS DATA ***');
      console.log(JSON.stringify(gs.odds, null, 2).slice(0, 1000));
    }
    if (gs.winprobability) console.log('Has win probability data!');
    if (gs.leaders) console.log('Has leaders data');
    if (gs.againstTheSpread) {
      console.log('\n*** ATS DATA ***');
      console.log(JSON.stringify(gs.againstTheSpread, null, 2).slice(0, 500));
    }
  } catch(e) { console.log('Error:', e.message); }

  // 5. Check NBA.com endpoints
  console.log('\n--- NBA.COM ADVANCED STATS ---');
  try {
    const headers = {
      'User-Agent': 'Mozilla/5.0',
      'Referer': 'https://www.nba.com/'
    };
    // This may be blocked but worth trying
    const nba = await fetch('https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=');
    if (nba.resultSets) {
      console.log('NBA API has resultSets:', nba.resultSets.length);
      console.log('Headers:', nba.resultSets[0].headers);
    } else {
      console.log('NBA API response type:', typeof nba, String(nba).slice(0, 200));
    }
  } catch(e) { console.log('NBA API error:', e.message); }

  // 6. Check ESPN schedule for back-to-back info
  console.log('\n--- ESPN SCHEDULE (for B2B detection) ---');
  try {
    const sched = await fetch('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20250219');
    const teams19 = new Set();
    for (const ev of (sched.events || [])) {
      const comp = ev.competitions[0];
      for (const c of comp.competitors) {
        teams19.add(c.team.abbreviation);
      }
    }
    const sched2 = await fetch('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20250220');
    const teams20 = new Set();
    for (const ev of (sched2.events || [])) {
      const comp = ev.competitions[0];
      for (const c of comp.competitors) {
        teams20.add(c.team.abbreviation);
      }
    }
    const b2b = [...teams20].filter(t => teams19.has(t));
    console.log('Teams playing 2/19:', [...teams19].join(', '));
    console.log('Teams playing 2/20:', [...teams20].join(', '));
    console.log('Back-to-back teams:', b2b.join(', '));
  } catch(e) { console.log('Error:', e.message); }
}

main().catch(console.error);
