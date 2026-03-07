#!/usr/bin/env node
// Fetch recent historical odds and box scores using curl (Node fetch blocked)
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const API_KEY = '3879c3373a31421d8ef7d428b8758cd8';
const BASE = 'https://api.the-odds-api.com/v4';
const ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba';
const RAW_DIR = 'webapp/data/historical_odds_raw';
const ODDS_FILE = 'webapp/data/historical_odds.json';
const BOX_FILE = 'webapp/data/player_boxscores.json';
const SEASON_FILE = 'webapp/data/espn_full_season_2025.json';

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

const ESPN_TO_OURS = {
  'GS': 'GS', 'NY': 'NY', 'NO': 'NO', 'SA': 'SA', 'WSH': 'WSH', 'UTAH': 'UTAH',
  'GSW': 'GS', 'NYK': 'NY', 'NOP': 'NO', 'SAS': 'SA', 'WAS': 'WSH', 'UTA': 'UTAH',
};

function teamAbbr(name) { return TEAM_MAP[name] || name; }
function espnAbbr(a) { return ESPN_TO_OURS[a] || a; }

function curlJSON(url) {
  try {
    const out = execSync(`curl -s "${url}"`, { timeout: 15000, maxBuffer: 10 * 1024 * 1024 }).toString();
    return JSON.parse(out);
  } catch (e) {
    return null;
  }
}

function sleep(ms) { execSync(`sleep ${ms / 1000}`); }

// Get dates with NBA games from Feb 25 to Mar 6 2025
function getMissingDates(existingDates) {
  const dates = [];
  const start = new Date('2025-02-25');
  const end = new Date('2025-03-07');

  for (let d = new Date(start); d < end; d.setDate(d.getDate() + 1)) {
    const ds = `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}`;
    if (!existingDates.has(ds)) dates.push(ds);
  }
  return dates;
}

function fetchOddsForDate(dateStr) {
  const isoDate = `${dateStr.slice(0,4)}-${dateStr.slice(4,6)}-${dateStr.slice(6,8)}T12:00:00Z`;
  const results = [];

  const gamesUrl = `${BASE}/historical/sports/basketball_nba/odds?apiKey=${API_KEY}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&date=${isoDate}&bookmakers=fanduel`;
  const gamesData = curlJSON(gamesUrl);
  if (!gamesData) return results;

  const events = gamesData.data || [];

  // Save raw file
  const rawFile = path.join(RAW_DIR, `game_odds_${dateStr}.json`);
  if (!fs.existsSync(rawFile)) {
    fs.writeFileSync(rawFile, JSON.stringify(gamesData, null, 2));
  }

  for (const event of events) {
    const homeA = teamAbbr(event.home_team);
    const awayA = teamAbbr(event.away_team);
    const gameKey = `${awayA}@${homeA}`;

    const fd = (event.bookmakers || []).find(b => b.key === 'fanduel');
    if (!fd) continue;

    const record = {
      date: dateStr, gameKey, eventId: event.id,
      homeTeam: homeA, awayTeam: awayA, commenceTime: event.commence_time,
    };

    for (const mkt of (fd.markets || [])) {
      if (mkt.key === 'h2h') {
        const ho = mkt.outcomes.find(o => o.name === event.home_team);
        const ao = mkt.outcomes.find(o => o.name === event.away_team);
        if (ho) record.home_ml = ho.price;
        if (ao) record.away_ml = ao.price;
      } else if (mkt.key === 'spreads') {
        const ho = mkt.outcomes.find(o => o.name === event.home_team);
        const ao = mkt.outcomes.find(o => o.name === event.away_team);
        if (ho) { record.spread_home = ho.price; record.spread_point = ho.point; }
        if (ao) record.spread_away = ao.price;
      } else if (mkt.key === 'totals') {
        const ov = mkt.outcomes.find(o => o.name === 'Over');
        const un = mkt.outcomes.find(o => o.name === 'Under');
        if (ov) { record.total = ov.point; record.total_over = ov.price; }
        if (un) record.total_under = un.price;
      }
    }

    // Fetch player props
    sleep(200);
    const propsUrl = `${BASE}/historical/sports/basketball_nba/events/${event.id}/odds?apiKey=${API_KEY}&regions=us&markets=player_points_alternate&oddsFormat=american&date=${isoDate}&bookmakers=fanduel`;
    const propsData = curlJSON(propsUrl);
    if (propsData) {
      const ed = propsData.data || propsData;
      const fdb = (ed.bookmakers || []).find(b => b.key === 'fanduel');
      if (fdb) {
        const mkt = (fdb.markets || []).find(m => m.key === 'player_points_alternate');
        if (mkt) {
          const playerLines = {};
          for (const o of (mkt.outcomes || [])) {
            if (!playerLines[o.description]) playerLines[o.description] = {};
            if (!playerLines[o.description][o.point]) playerLines[o.description][o.point] = {};
            if (o.name === 'Over') playerLines[o.description][o.point].overOdds = o.price;
            else playerLines[o.description][o.point].underOdds = o.price;
          }
          record.playerProps = playerLines;
          const propsFile = path.join(RAW_DIR, `props_${dateStr}_${event.id}.json`);
          if (!fs.existsSync(propsFile)) fs.writeFileSync(propsFile, JSON.stringify(propsData, null, 2));
        }
      }
    }

    results.push(record);
  }

  return results;
}

function fetchBoxScoresForDate(dateStr) {
  const results = [];

  const url = `${ESPN_BASE}/scoreboard?dates=${dateStr}`;
  const data = curlJSON(url);
  if (!data) return results;

  for (const event of (data.events || [])) {
    const comp = event.competitions?.[0];
    if (!comp) continue;

    let homeTeam = '', awayTeam = '', homeScore = 0, awayScore = 0;
    for (const c of (comp.competitors || [])) {
      const tri = espnAbbr(c.team?.abbreviation || '');
      const score = parseInt(c.score) || 0;
      if (c.homeAway === 'home') { homeTeam = tri; homeScore = score; }
      else { awayTeam = tri; awayScore = score; }
    }
    if (!homeTeam || !awayTeam) continue;

    const statusType = comp.status?.type?.name || '';
    if (statusType !== 'STATUS_FINAL') continue;

    sleep(200);
    const summaryUrl = `${ESPN_BASE}/summary?event=${event.id}`;
    const summary = curlJSON(summaryUrl);
    if (!summary) continue;

    const players = [];
    for (const team of (summary.boxscore?.players || [])) {
      const ta = espnAbbr(team.team?.abbreviation || '');
      for (const sg of (team.statistics || [])) {
        for (const ath of (sg.athletes || [])) {
          const stats = {};
          const labels = sg.labels || [];
          const values = ath.stats || [];
          for (let i = 0; i < labels.length; i++) stats[labels[i]] = values[i];

          players.push({
            team: ta,
            name: ath.athlete?.displayName || '',
            pid: ath.athlete?.id || '',
            min: String(parseInt(stats['MIN']) || 0),
            pts: parseInt(stats['PTS']) || 0,
            fg: stats['FG'] || '0-0',
            three: stats['3PT'] || '0-0',
            ft: stats['FT'] || '0-0',
            reb: parseInt(stats['REB']) || 0,
            ast: parseInt(stats['AST']) || 0,
            stl: parseInt(stats['STL']) || 0,
            blk: parseInt(stats['BLK']) || 0,
            to: parseInt(stats['TO']) || 0,
          });
        }
      }
    }

    results.push({
      event_id: event.id,
      date: dateStr,
      home: homeTeam,
      away: awayTeam,
      home_score: homeScore,
      away_score: awayScore,
      players,
    });
  }

  return results;
}

// Main
const existingOdds = JSON.parse(fs.readFileSync(ODDS_FILE, 'utf8'));
const existingBox = JSON.parse(fs.readFileSync(BOX_FILE, 'utf8'));
const existingSeason = JSON.parse(fs.readFileSync(SEASON_FILE, 'utf8'));

const existingOddsDates = new Set(existingOdds.map(o => o.date));
const existingBoxDates = new Set(existingBox.map(b => b.date));

const dates = getMissingDates(existingOddsDates);
console.log(`Fetching data for ${dates.length} missing dates`);

let newOdds = [];
let newBoxScores = [];
let newSeasonGames = [];

for (const date of dates) {
  process.stdout.write(`${date}: `);

  if (!existingOddsDates.has(date)) {
    const odds = fetchOddsForDate(date);
    process.stdout.write(`${odds.length} odds (${odds.filter(o=>o.playerProps).length} w/props) `);
    newOdds.push(...odds);
    sleep(300);
  }

  if (!existingBoxDates.has(date)) {
    const boxes = fetchBoxScoresForDate(date);
    process.stdout.write(`${boxes.length} box scores`);
    newBoxScores.push(...boxes);
    for (const box of boxes) {
      newSeasonGames.push({
        date: box.date, home_team: box.home, away_team: box.away,
        home_score: box.home_score, away_score: box.away_score,
        margin: box.home_score - box.away_score, event_id: box.event_id,
      });
    }
    sleep(300);
  }

  console.log('');
}

if (newOdds.length > 0) {
  const all = [...existingOdds, ...newOdds].sort((a,b) => a.date.localeCompare(b.date) || a.gameKey.localeCompare(b.gameKey));
  fs.writeFileSync(ODDS_FILE, JSON.stringify(all, null, 2));
  console.log(`\nSaved ${all.length} total odds (+${newOdds.length})`);
}

if (newBoxScores.length > 0) {
  const all = [...existingBox, ...newBoxScores].sort((a,b) => a.date.localeCompare(b.date));
  fs.writeFileSync(BOX_FILE, JSON.stringify(all, null, 2));
  console.log(`Saved ${all.length} total box scores (+${newBoxScores.length})`);
}

if (newSeasonGames.length > 0) {
  const all = [...existingSeason, ...newSeasonGames].sort((a,b) => a.date.localeCompare(b.date));
  fs.writeFileSync(SEASON_FILE, JSON.stringify(all, null, 2));
  console.log(`Saved ${all.length} total season games (+${newSeasonGames.length})`);
}

console.log('Done!');
