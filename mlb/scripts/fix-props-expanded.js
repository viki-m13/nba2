#!/usr/bin/env node
// =============================================================================
// Fix props in expanded odds data — re-fetch using correct market keys
// The game-level odds (h2h, spreads, totals) are already correct.
// Only re-fetches batter props for records missing them.
// =============================================================================

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const DATA_DIR = path.join(__dirname, '..', 'webapp', 'data');
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

// Load expanded odds
const oddsPath = path.join(DATA_DIR, 'mlb_historical_odds_expanded.json');
const allOdds = JSON.parse(fs.readFileSync(oddsPath, 'utf8'));

// Find records missing props
const needsProps = allOdds.filter(r => !r.hitsProps);
const withProps = allOdds.filter(r => r.hitsProps);

console.log('=== Fix Props in Expanded Odds ===');
console.log(`Total records: ${allOdds.length}`);
console.log(`Already have props: ${withProps.length}`);
console.log(`Need props: ${needsProps.length}`);

// Group by date for progress tracking
const dateGroups = {};
for (const r of needsProps) {
  if (!dateGroups[r.date]) dateGroups[r.date] = [];
  dateGroups[r.date].push(r);
}
const datesToFix = Object.keys(dateGroups).sort();
console.log(`Dates to fix: ${datesToFix.length}\n`);

let fixed = 0;
let apiCalls = 0;

for (const dateStr of datesToFix) {
  const records = dateGroups[dateStr];
  const y = dateStr.substring(0, 4);
  const m = dateStr.substring(4, 6);
  const dd = dateStr.substring(6, 8);
  const isoDate = `${y}-${m}-${dd}T17:00:00Z`;

  process.stdout.write(`  ${dateStr}: ${records.length} games`);
  let dateProps = 0;

  for (const record of records) {
    sleep(500);

    // Use CORRECT market keys (no _alternate suffix)
    const propsUrl = `https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/${record.eventId}/odds?apiKey=${ODDS_API_KEY}&regions=us&markets=batter_hits,batter_total_bases,batter_rbis,batter_runs_scored&oddsFormat=american&bookmakers=fanduel&date=${isoDate}`;
    const propsData = curlJSON(propsUrl);
    apiCalls++;

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
        if (Object.keys(hitsProps).length > 0) { record.hitsProps = hitsProps; dateProps++; fixed++; }
        if (Object.keys(tbProps).length > 0) record.tbProps = tbProps;
        if (Object.keys(rbiProps).length > 0) record.rbiProps = rbiProps;
        if (Object.keys(runsProps).length > 0) record.runsProps = runsProps;
      }
    }
  }

  process.stdout.write(` -> ${dateProps} with props (${apiCalls} API calls total)\n`);

  // Save checkpoint every 5 dates
  if (datesToFix.indexOf(dateStr) % 5 === 4) {
    allOdds.sort((a, b) => a.date.localeCompare(b.date));
    fs.writeFileSync(oddsPath, JSON.stringify(allOdds, null, 2));
    process.stdout.write(`    [checkpoint saved: ${allOdds.filter(r => r.hitsProps).length} total with props]\n`);
  }
}

// Final save
allOdds.sort((a, b) => a.date.localeCompare(b.date));
fs.writeFileSync(oddsPath, JSON.stringify(allOdds, null, 2));

const finalWithHits = allOdds.filter(r => r.hitsProps).length;
const finalWithTB = allOdds.filter(r => r.tbProps).length;

console.log(`\n=== PROPS FIX SUMMARY ===`);
console.log(`  Records fixed: ${fixed}`);
console.log(`  Total with hits props: ${finalWithHits}`);
console.log(`  Total with TB props: ${finalWithTB}`);
console.log(`  API calls used: ${apiCalls}`);
console.log(`  Saved to: ${oddsPath}`);
