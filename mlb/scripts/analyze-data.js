#!/usr/bin/env node
// Quick analysis of MLB data characteristics to guide optimization
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '../webapp/data');
const boxScores = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_player_boxscores.json'), 'utf8'));
const historicalOdds = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'mlb_historical_odds.json'), 'utf8'));

console.log('=== MLB DATA ANALYSIS ===');
console.log(`Box scores: ${boxScores.length} games`);
console.log(`Historical odds: ${historicalOdds.length} records`);

// Analyze prop line distributions
let hitsLines = [], tbLines = [], rbiLines = [], runsLines = [];
let hitsResults = [], tbResults = [], rbiResults = [], runsResults = [];

for (const od of historicalOdds) {
  const date = od.date;
  const gameKey = od.gameKey || `${od.awayTeam}@${od.homeTeam}`;

  // Find matching box score
  const box = boxScores.find(b => b.date === date && `${b.away}@${b.home}` === gameKey);
  if (!box) continue;

  // Analyze hits props
  if (od.hitsProps) {
    for (const [player, lines] of Object.entries(od.hitsProps)) {
      for (const [threshold, data] of Object.entries(lines)) {
        const line = parseFloat(threshold);
        const odds = data.overOdds;
        if (!odds) continue;
        hitsLines.push({ line, odds });

        const playerData = (box.players || []).find(p => p.name === player);
        if (playerData) {
          hitsResults.push({ player, line, odds, actual: playerData.h || 0, hit: (playerData.h || 0) > line });
        }
      }
    }
  }

  // Analyze TB props
  if (od.tbProps) {
    for (const [player, lines] of Object.entries(od.tbProps)) {
      for (const [threshold, data] of Object.entries(lines)) {
        const line = parseFloat(threshold);
        const odds = data.overOdds;
        if (!odds) continue;
        tbLines.push({ line, odds });

        const playerData = (box.players || []).find(p => p.name === player);
        if (playerData) {
          tbResults.push({ player, line, odds, actual: playerData.tb || 0, hit: (playerData.tb || 0) > line });
        }
      }
    }
  }

  // Analyze RBI props
  if (od.rbiProps) {
    for (const [player, lines] of Object.entries(od.rbiProps)) {
      for (const [threshold, data] of Object.entries(lines)) {
        const line = parseFloat(threshold);
        const odds = data.overOdds;
        if (!odds) continue;
        rbiLines.push({ line, odds });

        const playerData = (box.players || []).find(p => p.name === player);
        if (playerData) {
          rbiResults.push({ player, line, odds, actual: playerData.rbi || 0, hit: (playerData.rbi || 0) > line });
        }
      }
    }
  }

  // Analyze runs props
  if (od.runsProps) {
    for (const [player, lines] of Object.entries(od.runsProps)) {
      for (const [threshold, data] of Object.entries(lines)) {
        const line = parseFloat(threshold);
        const odds = data.overOdds;
        if (!odds) continue;
        runsLines.push({ line, odds });

        const playerData = (box.players || []).find(p => p.name === player);
        if (playerData) {
          runsResults.push({ player, line, odds, actual: playerData.r || 0, hit: (playerData.r || 0) > line });
        }
      }
    }
  }
}

function analyzeMarket(name, results) {
  if (results.length === 0) { console.log(`\n${name}: NO DATA`); return; }

  const lines = [...new Set(results.map(r => r.line))].sort((a,b) => a-b);
  console.log(`\n${name}: ${results.length} total props`);
  console.log(`  Lines: ${lines.join(', ')}`);

  for (const line of lines) {
    const forLine = results.filter(r => r.line === line);
    const hits = forLine.filter(r => r.hit).length;
    const rate = hits / forLine.length;
    const avgOdds = forLine.reduce((s, r) => s + r.odds, 0) / forLine.length;
    console.log(`  Line ${line}: ${forLine.length} props, ${hits} hits, ${(rate*100).toFixed(1)}% hit rate, avg odds ${avgOdds.toFixed(0)}`);

    // By odds range
    const favorable = forLine.filter(r => r.odds >= -200 && r.odds <= -110);
    if (favorable.length > 0) {
      const favHits = favorable.filter(r => r.hit).length;
      console.log(`    Odds [-200,-110]: ${favorable.length} props, ${(favHits/favorable.length*100).toFixed(1)}% hit rate`);
    }
    const heavy = forLine.filter(r => r.odds < -200);
    if (heavy.length > 0) {
      const heavyHits = heavy.filter(r => r.hit).length;
      console.log(`    Odds [<-200]: ${heavy.length} props, ${(heavyHits/heavy.length*100).toFixed(1)}% hit rate`);
    }
  }
}

analyzeMarket('HITS', hitsResults);
analyzeMarket('TOTAL BASES', tbResults);
analyzeMarket('RBI', rbiResults);
analyzeMarket('RUNS', runsResults);

// Analyze player game counts
const playerGames = {};
for (const game of boxScores) {
  for (const p of (game.players || [])) {
    if ((p.ab || 0) >= 2) {
      playerGames[p.name] = (playerGames[p.name] || 0) + 1;
    }
  }
}

const gameCounts = Object.values(playerGames);
console.log(`\n=== PLAYER GAME COUNTS ===`);
console.log(`Total players with 2+ AB: ${gameCounts.length}`);
console.log(`Players with 15+ games: ${gameCounts.filter(g => g >= 15).length}`);
console.log(`Players with 10+ games: ${gameCounts.filter(g => g >= 10).length}`);
console.log(`Players with 20+ games: ${gameCounts.filter(g => g >= 20).length}`);
console.log(`Max games: ${Math.max(...gameCounts)}`);
