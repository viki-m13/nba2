#!/usr/bin/env node
// =============================================================================
// Download MLB batter props from Odds API historical endpoint
// Uses correct market names: batter_hits, batter_total_bases, batter_rbis, batter_runs_scored
// =============================================================================

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const DATA_DIR = path.join(__dirname, '..', 'webapp', 'data');
const WEBAPP_DATA_DIR = path.join(__dirname, '..', '..', 'webapp', 'mlb', 'data');
const ODDS_API_KEY = '3879c3373a31421d8ef7d428b8758cd8';

function curlJSON(url) {
  try {
    const result = execSync(`curl -s --max-time 30 "${url}"`, { maxBuffer: 10 * 1024 * 1024 });
    return JSON.parse(result.toString());
  } catch (e) {
    return null;
  }
}

function sleep(ms) { execSync(`sleep ${ms / 1000}`); }

// Load existing odds data
const oddsPath = path.join(DATA_DIR, 'mlb_historical_odds.json');
let allOdds = JSON.parse(fs.readFileSync(oddsPath, 'utf8'));
console.log(`Loaded ${allOdds.length} existing odds records`);

// Find records without props
const needProps = allOdds.filter(o => !o.hitsProps && !o.tbProps);
console.log(`${needProps.length} records need props\n`);

let propsFound = 0;
let processed = 0;

for (const record of needProps) {
  processed++;
  const dateStr = record.date;
  const y = dateStr.substring(0, 4);
  const m = dateStr.substring(4, 6);
  const dd = dateStr.substring(6, 8);
  const isoDate = `${y}-${m}-${dd}T17:00:00Z`;

  // Correct market names for MLB (NO _alternate suffix)
  const propsUrl = `https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/${record.eventId}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=batter_hits,batter_total_bases,batter_rbis,batter_runs_scored&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
  const propsData = curlJSON(propsUrl);

  if (propsData?.data) {
    const pfb = (propsData.data.bookmakers || []).find(b => b.key === 'fanduel');
    if (pfb) {
      const hitsProps = {}, tbProps = {}, rbiProps = {}, runsProps = {};
      for (const mkt of (pfb.markets || [])) {
        for (const o of (mkt.outcomes || [])) {
          if (o.name === 'Over' && o.description && o.point !== undefined) {
            const target = mkt.key === 'batter_hits' ? hitsProps
              : mkt.key === 'batter_total_bases' ? tbProps
              : mkt.key === 'batter_rbis' ? rbiProps
              : mkt.key === 'batter_runs_scored' ? runsProps : null;
            if (target) {
              if (!target[o.description]) target[o.description] = {};
              target[o.description][o.point] = { overOdds: o.price };
            }
          }
        }
      }
      if (Object.keys(hitsProps).length > 0) { record.hitsProps = hitsProps; propsFound++; }
      if (Object.keys(tbProps).length > 0) record.tbProps = tbProps;
      if (Object.keys(rbiProps).length > 0) record.rbiProps = rbiProps;
      if (Object.keys(runsProps).length > 0) record.runsProps = runsProps;
    }
  }

  if (processed % 10 === 0) {
    console.log(`  Processed ${processed}/${needProps.length} (${propsFound} with props)`);
  }

  sleep(500);
}

console.log(`\nDone: ${propsFound}/${needProps.length} records got props`);

// Save updated odds data
fs.writeFileSync(oddsPath, JSON.stringify(allOdds, null, 2));
fs.writeFileSync(path.join(WEBAPP_DATA_DIR, 'mlb_historical_odds.json'), JSON.stringify(allOdds, null, 2));

const withHits = allOdds.filter(o => o.hitsProps && Object.keys(o.hitsProps).length > 0).length;
const withTB = allOdds.filter(o => o.tbProps && Object.keys(o.tbProps).length > 0).length;
const withRBI = allOdds.filter(o => o.rbiProps && Object.keys(o.rbiProps).length > 0).length;

console.log(`\n=== FINAL STATS ===`);
console.log(`  Total odds records: ${allOdds.length}`);
console.log(`  With hits props: ${withHits}`);
console.log(`  With TB props: ${withTB}`);
console.log(`  With RBI props: ${withRBI}`);
console.log(`  Saved to ${oddsPath}`);
