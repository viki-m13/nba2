const sigs = JSON.parse(require("fs").readFileSync("mlb/webapp/data/mlb_ultra_signals_v3.json", "utf8"));
const singles = sigs.filter(s => s.betType === "single");
const byDate = {};
for (const s of singles) {
  if (byDate[s.date] === undefined) byDate[s.date] = [];
  byDate[s.date].push(s);
}
const counts = Object.entries(byDate).map(([d,v]) => ({date:d, count:v.length, wins:v.filter(x=>x.hit).length})).sort((a,b)=>b.count-a.count);
console.log("=== DAILY PICK DISTRIBUTION ===");
console.log("Total unique dates:", counts.length);
console.log("Dates with 2+ picks:", counts.filter(c=>c.count>=2).length);
console.log("Dates with 3+ picks:", counts.filter(c=>c.count>=3).length);
console.log("\nTop dates:");
for (const c of counts.slice(0,15)) {
  console.log("  " + c.date + ": " + c.count + " picks, " + c.wins + " wins (" + (c.wins/c.count*100).toFixed(0) + "%)");
}
const statTypes = {};
for (const s of singles) {
  const k = s.statLabel || "?";
  if (statTypes[k] === undefined) statTypes[k] = {total:0,wins:0};
  statTypes[k].total++;
  if (s.hit) statTypes[k].wins++;
}
console.log("\n=== STAT TYPE BREAKDOWN ===");
for (const [k,v] of Object.entries(statTypes).sort((a,b)=>b[1].total-a[1].total)) {
  console.log(k + ": " + v.total + " bets, " + v.wins + " wins (" + (v.wins/v.total*100).toFixed(1) + "%)");
}
