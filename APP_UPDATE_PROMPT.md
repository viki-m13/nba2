# Comprehensive App Update: Add Moneyline Bets + Fix All Win Rates & EV

## OVERVIEW

Our NBA in-game trading app currently only supports spread bets and displays INCORRECT win rates. The original validation had a bug: `spread_covered` checked if the team maintained its FULL lead (`final_margin >= actual_lead`), not whether the REDUCED spread (-7 or -5) was covered. The ~94-100% numbers were essentially moneyline win rates, not spread coverage rates.

We re-validated on the same 156 ESPN historical games using the CORRECT methodology:
- Reduced spread coverage: did the team win by MORE than the spread (-7 or -5)?
- Moneyline win: did the team win the game?
- One signal per game (first valid), priority ordering
- 12-24 minutes remaining window
- Push = final margin equals spread exactly (excluded from win rate)

## CORRECT VALIDATED NUMBERS

### SPREAD BETS (validated on 156 real NBA games from ESPN):

| Strategy | Signals | W-P-L | Win Rate | Market Prob | Edge | EV per $100 |
|----------|---------|-------|----------|-------------|------|-------------|
| Sweet Spot | 41 | 33-2-6 | 84.6% | 74% | +10.6% | +$14 |
| Moderate | 33 | 27-1-5 | 84.4% | 76% | +8.4% | +$11 |
| Mid-Range | 19 | 17-1-1 | 94.4% | 79% | +15.4% | +$20 |
| Safe | 27 | 24-2-1 | 96.0% | 84% | +12.0% | +$14 |

Average Spread Win Rate: ~90%

### MONEYLINE BETS (validated on same 156 games):

| Strategy | Signals | W-L | Win Rate | Market Prob | Edge | EV per $100 |
|----------|---------|-----|----------|-------------|------|-------------|
| Sweet Spot | 41 | 37-4 | 90.2% | 82% | +8.2% | +$10 |
| Moderate | 33 | 30-3 | 90.9% | 85% | +5.9% | +$7 |
| Mid-Range | 19 | 19-0 | 100% | 89% | +11.0% | +$12 |
| Safe | 27 | 26-1 | 96.3% | 92% | +4.3% | +$5 |

Average Moneyline Win Rate: ~94%

### COMBINED EV (betting both spread AND moneyline per signal):

| Strategy | Spread EV | ML EV | Combined EV per $100 each |
|----------|-----------|-------|---------------------------|
| Sweet Spot | +$14 | +$10 | +$24 |
| Moderate | +$11 | +$7 | +$18 |
| Mid-Range | +$20 | +$12 | +$32 |
| Safe | +$14 | +$5 | +$19 |

---

## FILES TO UPDATE

The following files need changes. The strategy signal CONDITIONS (lead ranges, momentum thresholds, time window, priority ordering) stay THE SAME. Only the displayed win rates, edge values, EV values, and bet types change.

---

### 1. `src/lib/types.ts` — Add moneyline to betType

**Change the Alert interface betType field:**

OLD:
```typescript
betType: 'spread' | 'total';
```

NEW:
```typescript
betType: 'spread' | 'moneyline' | 'spread_and_ml';
```

---

### 2. `src/lib/nba-api.ts` — Update calculateTrueEdge + getEntrySignal

#### 2a. Update `calculateTrueEdge` function (~line 1266)

The function needs to return values for BOTH spread and moneyline. Change the EdgeCalculation interface and function:

OLD EdgeCalculation interface (~line 1256):
```typescript
interface EdgeCalculation {
  modelProb: number;
  marketProb: number;
  edge: number;
  expectedValue: number;
  confidence: 'high' | 'medium' | 'low';
}
```

NEW:
```typescript
interface EdgeCalculation {
  // Spread bet metrics
  spreadModelProb: number;    // Spread win rate (%)
  spreadMarketProb: number;   // Spread market probability (%)
  spreadEdge: number;         // Spread edge (%)
  spreadEV: number;           // Spread expected value per unit
  // Moneyline bet metrics
  mlModelProb: number;        // ML win rate (%)
  mlMarketProb: number;       // ML market probability (%)
  mlEdge: number;             // ML edge (%)
  mlEV: number;               // ML expected value per unit
  // Legacy fields for backward compat
  modelProb: number;          // Use spread WR for display
  marketProb: number;         // Use spread market prob for display
  edge: number;               // Use spread edge for display
  expectedValue: number;      // Use combined EV for display
  confidence: 'high' | 'medium' | 'low';
}
```

OLD calculateTrueEdge switch block:
```typescript
switch (strategy) {
  case 'sweet_spot':
    modelProb = 94.9;
    marketProb = 74;
    break;
  case 'moderate':
    modelProb = 94.5;
    marketProb = 76;
    break;
  case 'mid_range':
    modelProb = 96.4;
    marketProb = 79;
    break;
  case 'safe':
    modelProb = 100;
    marketProb = 84;
    break;
  default:
    modelProb = 52.38;
    marketProb = 50;
}
```

NEW calculateTrueEdge function body:
```typescript
function calculateTrueEdge(
  differential: number,
  quarter: number,
  rsi: number,
  volatility: number,
  recentRun: ScoringRun | null,
  nearSR: boolean,
  strategy: StrategyType,
  lead?: number
): EdgeCalculation {
  const confidence: 'high' | 'medium' | 'low' = 'high';

  // VALIDATED on 156 real ESPN NBA games (2022-24 seasons)
  // Spread = reduced spread (-7 or -5), Moneyline = team wins game
  let spreadModelProb: number;
  let spreadMarketProb: number;
  let mlModelProb: number;
  let mlMarketProb: number;

  switch (strategy) {
    case 'sweet_spot':
      spreadModelProb = 84.6;
      spreadMarketProb = 74;
      mlModelProb = 90.2;
      mlMarketProb = 82;
      break;
    case 'moderate':
      spreadModelProb = 84.4;
      spreadMarketProb = 76;
      mlModelProb = 90.9;
      mlMarketProb = 85;
      break;
    case 'mid_range':
      spreadModelProb = 94.4;
      spreadMarketProb = 79;
      mlModelProb = 100;
      mlMarketProb = 89;
      break;
    case 'safe':
      spreadModelProb = 96.0;
      spreadMarketProb = 84;
      mlModelProb = 96.3;
      mlMarketProb = 92;
      break;
    default:
      spreadModelProb = 52;
      spreadMarketProb = 50;
      mlModelProb = 52;
      mlMarketProb = 50;
  }

  const spreadEdge = spreadModelProb - spreadMarketProb;
  const mlEdge = mlModelProb - mlMarketProb;

  // EV = P(win) * payoutOnWin - P(loss) * 1
  // payoutOnWin = (100 - marketProb) / marketProb
  const spreadPayout = (100 - spreadMarketProb) / spreadMarketProb;
  const spreadEV = (spreadModelProb / 100) * spreadPayout - ((100 - spreadModelProb) / 100);

  const mlPayout = (100 - mlMarketProb) / mlMarketProb;
  const mlEV = (mlModelProb / 100) * mlPayout - ((100 - mlModelProb) / 100);

  const combinedEV = spreadEV + mlEV;

  return {
    spreadModelProb: Math.round(spreadModelProb * 10) / 10,
    spreadMarketProb: Math.round(spreadMarketProb * 10) / 10,
    spreadEdge: Math.round(spreadEdge * 10) / 10,
    spreadEV: Math.round(spreadEV * 100) / 100,
    mlModelProb: Math.round(mlModelProb * 10) / 10,
    mlMarketProb: Math.round(mlMarketProb * 10) / 10,
    mlEdge: Math.round(mlEdge * 10) / 10,
    mlEV: Math.round(mlEV * 100) / 100,
    // Legacy fields - use spread for primary display
    modelProb: Math.round(spreadModelProb * 10) / 10,
    marketProb: Math.round(spreadMarketProb * 10) / 10,
    edge: Math.round(spreadEdge * 10) / 10,
    expectedValue: Math.round(combinedEV * 100) / 100,
    confidence,
  };
}
```

#### 2b. Update `getEntrySignal` comments (~line 1488)

OLD comments on each strategy:
```
// STRATEGY 1: SWEET SPOT (94.9% WR, +$28 EV, 20.9% edge)
// STRATEGY 2: MODERATE (94.5% WR, +$24 EV, 18.5% edge)
// STRATEGY 3: MID-RANGE (96.4% WR, +$22 EV, 17.4% edge)
// STRATEGY 4: SAFE (100% WR, +$19 EV, 16.0% edge)
```

NEW:
```
// STRATEGY 1: SWEET SPOT (Spread: 84.6% WR, +$14 EV | ML: 90.2% WR, +$10 EV)
// STRATEGY 2: MODERATE (Spread: 84.4% WR, +$11 EV | ML: 90.9% WR, +$7 EV)
// STRATEGY 3: MID-RANGE (Spread: 94.4% WR, +$20 EV | ML: 100% WR, +$12 EV)
// STRATEGY 4: SAFE (Spread: 96.0% WR, +$14 EV | ML: 96.3% WR, +$5 EV)
```

#### 2c. Update signal creation to include both bet types

Where signals are created (around where betType, recommendation, betInstruction, expectedOutcome are set), update to:

- `betType`: Change from `'spread'` to `'spread_and_ml'`
- `recommendation`: Change from e.g. `'BOS -7 SPREAD'` to `'BOS -7 SPREAD + ML'`
- `betInstruction`: Change from `'BET: Take BOS -7 SPREAD'` to `'BET: Take BOS -7 SPREAD & BOS MONEYLINE'`
- `expectedOutcome`: Change from `'EXPECT: Celtics to win by more than 7 points'` to `'SPREAD: Win by 8+ pts (84.6% WR) | ML: Win game (90.2% WR)'`

The edge display should use the SPREAD edge (since that's the primary bet with higher payout).

---

### 3. `src/lib/backtest-signals.ts` — Update all values

#### 3a. Update header comments (lines 1-31)

Change ALL win rate, EV, and edge values:

OLD:
```
// STRATEGY 1: SWEET SPOT (94.9% WR, +$27 EV, 20.9% edge)
// STRATEGY 2: MODERATE (94.5% WR, +$22 EV, 18.5% edge)
// STRATEGY 3: MID-RANGE (96.4% WR, +$19 EV, 17.4% edge)
// STRATEGY 4: SAFE (100% WR, +$16 EV, 16.0% edge)
```

NEW:
```
// STRATEGY 1: SWEET SPOT
//   Spread -7: 84.6% WR, +$14 EV, 10.6% edge
//   Moneyline: 90.2% WR, +$10 EV, 8.2% edge
//
// STRATEGY 2: MODERATE
//   Spread -7: 84.4% WR, +$11 EV, 8.4% edge
//   Moneyline: 90.9% WR, +$7 EV, 5.9% edge
//
// STRATEGY 3: MID-RANGE
//   Spread -7: 94.4% WR, +$20 EV, 15.4% edge
//   Moneyline: 100% WR, +$12 EV, 11.0% edge
//
// STRATEGY 4: SAFE
//   Spread -5: 96.0% WR, +$14 EV, 12.0% edge
//   Moneyline: 96.3% WR, +$5 EV, 4.3% edge
```

#### 3b. Update `getSpreadSignal` return values (lines 144-200)

For each strategy, update `expectedWR`, `expectedEV`, and `edge`:

```typescript
// SWEET SPOT
return {
  side, signal: 'sweet_spot', lead, momentum: mom, spreadBet: -7,
  expectedWR: 84.6, expectedEV: 14, edge: 10.6,
};

// MODERATE
return {
  side, signal: 'moderate', lead, momentum: mom, spreadBet: -7,
  expectedWR: 84.4, expectedEV: 11, edge: 8.4,
};

// MID-RANGE
return {
  side, signal: 'mid_range', lead, momentum: mom, spreadBet: -7,
  expectedWR: 94.4, expectedEV: 20, edge: 15.4,
};

// SAFE
return {
  side, signal: 'safe', lead, momentum: mom, spreadBet: -5,
  expectedWR: 96.0, expectedEV: 14, edge: 12.0,
};
```

#### 3c. Update `getSpreadSignal` JSDoc comments (lines 87-103)

Same pattern — update all WR/EV/edge values in the docstring to match the spread numbers above.

#### 3d. Add moneyline grading function

After the existing `gradeSpreadSignal` function, add:

```typescript
/**
 * Grade a moneyline signal against final score.
 * Simply checks if the team won the game.
 */
function gradeMoneylineSignal(
  betTeam: 'home' | 'away',
  finalHomeScore: number,
  finalAwayScore: number
): 'win' | 'loss' {
  const finalDiff = finalHomeScore - finalAwayScore;
  if (betTeam === 'home') {
    return finalDiff > 0 ? 'win' : 'loss';
  } else {
    return finalDiff < 0 ? 'win' : 'loss';
  }
}
```

#### 3e. Update BacktestSignal interface to include ML outcome

Add to the BacktestSignal interface:
```typescript
mlOutcome: 'win' | 'loss';
mlModelProbability: number;
mlMarketProbability: number;
mlEdge: number;
mlPayoutPerUnit: number;
```

And update `runSpreadStrategy` to populate these fields and the `BacktestResult` to include ML stats.

---

### 4. `src/app/paywall.tsx` — Update all display values

#### 4a. Update strategy section header (~line 210)

OLD: `4 Proven Strategies (96.5% Avg Win Rate)`
NEW: `4 Proven Strategies — Spread + Moneyline`

#### 4b. Update Sweet Spot card (~lines 214-230)

OLD:
```
<Text className="text-emerald-400 text-xs font-bold">94.9%</Text>
...
<Text className="text-gray-400 text-xs">Bet -7 spread | +$28 EV | 20.9% edge</Text>
```

NEW:
```
<Text className="text-emerald-400 text-xs font-bold">84.6% / 90.2%</Text>
...
<Text className="text-gray-400 text-xs">Spread -7: 84.6% WR, +$14 EV | ML: 90.2% WR, +$10 EV</Text>
```

#### 4c. Update Moderate card (~lines 233-249)

OLD:
```
<Text className="text-blue-400 text-xs font-bold">94.5%</Text>
...
<Text className="text-gray-400 text-xs">Bet -7 spread | +$24 EV | 18.5% edge</Text>
```

NEW:
```
<Text className="text-blue-400 text-xs font-bold">84.4% / 90.9%</Text>
...
<Text className="text-gray-400 text-xs">Spread -7: 84.4% WR, +$11 EV | ML: 90.9% WR, +$7 EV</Text>
```

#### 4d. Update Mid-Range card (~lines 252-268)

OLD:
```
<Text className="text-purple-400 text-xs font-bold">96.4%</Text>
...
<Text className="text-gray-400 text-xs">Bet -7 spread | +$22 EV | 17.4% edge</Text>
```

NEW:
```
<Text className="text-purple-400 text-xs font-bold">94.4% / 100%</Text>
...
<Text className="text-gray-400 text-xs">Spread -7: 94.4% WR, +$20 EV | ML: 100% WR, +$12 EV</Text>
```

#### 4e. Update Safe card (~lines 271-287)

OLD:
```
<Text className="text-amber-400 text-xs font-bold">100%</Text>
...
<Text className="text-gray-400 text-xs">Bet -5 spread | +$19 EV | 16.0% edge</Text>
```

NEW:
```
<Text className="text-amber-400 text-xs font-bold">96.0% / 96.3%</Text>
...
<Text className="text-gray-400 text-xs">Spread -5: 96.0% WR, +$14 EV | ML: 96.3% WR, +$5 EV</Text>
```

---

### 5. `src/app/how-it-works.tsx` — Update all values + add ML explanation

#### 5a. Update "Our Philosophy" section (~lines 80-115)

Update each strategy bullet:

OLD:
```
Sweet Spot - 94.9% WR, +$27 EV, 20.9% edge
Moderate - 94.5% WR, +$22 EV, 18.5% edge
Mid-Range - 96.4% WR, +$19 EV, 17.4% edge
Safe - 100% WR, +$16 EV, 16.0% edge
```

NEW:
```
Sweet Spot - Spread 84.6% / ML 90.2% WR, +$24 combined EV
Moderate - Spread 84.4% / ML 90.9% WR, +$18 combined EV
Mid-Range - Spread 94.4% / ML 100% WR, +$32 combined EV
Safe - Spread 96.0% / ML 96.3% WR, +$19 combined EV
```

Also update the intro text to mention BOTH spread and moneyline betting.

#### 5b. Update "The 4 Reduced Spread Strategies" section (~lines 147-238)

For each strategy card, update the win rate badge and EV text. Also add a note about moneyline.

Example for Sweet Spot:

OLD:
```
<Text className="text-emerald-400 text-xs font-bold">94.9% WR</Text>
...
<Text className="text-gray-500 text-xs mt-1">Best EV: +$27 | 20.9% edge</Text>
```

NEW:
```
<Text className="text-emerald-400 text-xs font-bold">84.6% Spread / 90.2% ML</Text>
...
<Text className="text-gray-500 text-xs mt-1">Spread +$14 EV | ML +$10 EV | Combined +$24</Text>
```

Apply same pattern to Moderate, Mid-Range, Safe with their respective numbers.

#### 5c. Update "Key Insight" box (~line 233-237)

OLD:
```
By betting REDUCED spreads instead of full lead, win rates jump from ~60% to 94-100%.
```

NEW:
```
Each signal recommends TWO bets: a reduced spread (-7 or -5) AND a moneyline bet. Spread bets have higher payouts per win but lower win rates (84-96% WR). Moneyline bets have higher win rates (90-100% WR) but lower payouts per win. Combined EV is +$18-$32 per signal.
```

#### 5d. Update Strategy Conditions Table (~lines 240-287)

Update the Edge column to show both:

| Strategy | Lead | Mom | Bet | Spread Edge | ML Edge |
|----------|------|-----|-----|-------------|---------|
| Sweet | 10-14 | 10+ | -7 + ML | 10.6% | 8.2% |
| Moderate | 12-16 | 12+ | -7 + ML | 8.4% | 5.9% |
| Mid | 14-18 | 14+ | -7 + ML | 15.4% | 11.0% |
| Safe | 16-20 | 12+ | -5 + ML | 12.0% | 4.3% |

#### 5e. Update "Reduced Spread Bet Grading" section (~lines 382-430)

Add a MONEYLINE subsection explaining:
- Moneyline = bet the leading team simply wins the game
- Win = team wins (any margin), Loss = team loses
- No push possible on moneyline
- Higher win rate but lower payout per win

Update the validated rates:
OLD: `Sweet Spot: 94.9% • Moderate: 94.5% • Mid-Range: 96.4% • Safe: 100%`
NEW: `Spread: 84.6% / 84.4% / 94.4% / 96.0% • ML: 90.2% / 90.9% / 100% / 96.3%`

#### 5f. Update "Market Probability & Edge" section (~lines 432-489)

Update all displayed values:
- Sweet Spot market prob: 74% (spread) / 82% (ML)
- Edge values: 10.6% (spread) / 8.2% (ML)
- Add explanation that ML has higher market prob (lower payout) but higher win rate

OLD:
```
Our model achieves 94.5-100% actual win rates vs 74-84% market expectations.
This gap is our edge (16-21%).
```

NEW:
```
Spread bets: 84-96% actual WR vs 74-84% market (edge: 8-15%).
Moneyline bets: 90-100% actual WR vs 82-92% market (edge: 4-11%).
Both bet types are positive EV. Combined edge per signal: +$18-$32.
```

---

### 6. `src/app/backtest.tsx` — Update sample data and strategy cards

#### 6a. Update header comments (~lines 23-43)

Update all strategy comments with correct numbers (same pattern as backtest-signals.ts).

#### 6b. Update SAMPLE_SIGNAL_DATA (~lines 47-59)

Update the win/loss ratios to match real spread win rates. Currently shows ~94% WR. Fix to match real spread WR:

```typescript
const SAMPLE_SIGNAL_DATA = [
  // Sweet Spot - 84.6% Spread WR
  ...Array(22).fill({ impliedProb: 74, outcome: 'win', strategy: 'sweet_spot' }),
  ...Array(4).fill({ impliedProb: 74, outcome: 'loss', strategy: 'sweet_spot' }),
  // Moderate - 84.4% Spread WR
  ...Array(27).fill({ impliedProb: 76, outcome: 'win', strategy: 'moderate' }),
  ...Array(5).fill({ impliedProb: 76, outcome: 'loss', strategy: 'moderate' }),
  // Mid-Range - 94.4% Spread WR
  ...Array(17).fill({ impliedProb: 79, outcome: 'win', strategy: 'mid_range' }),
  ...Array(1).fill({ impliedProb: 79, outcome: 'loss', strategy: 'mid_range' }),
  // Safe - 96.0% Spread WR
  ...Array(24).fill({ impliedProb: 84, outcome: 'win', strategy: 'safe' }),
  ...Array(1).fill({ impliedProb: 84, outcome: 'loss', strategy: 'safe' }),
];
```

#### 6c. Update strategy cards section (~lines 351-428)

Update each strategy card's win rate badge and EV text to show both spread and ML values.

Example for Sweet Spot:
OLD: `94.9% WR` badge, `Lead 10-14 | Mom 10+ | Bet: -7 | +$27 EV`
NEW: `84.6% / 90.2%` badge, `Lead 10-14 | Mom 10+ | -7 Spread + ML | +$24 Combined EV`

Apply same to Moderate (84.4%/90.9%, +$18), Mid-Range (94.4%/100%, +$32), Safe (96.0%/96.3%, +$19).

#### 6d. Update "4 Reduced Spread Strategies" banner text (~line 216)

OLD: `Validated on 300+ real NBA games with 96.5% average win rate.`
NEW: `Validated on 156 real NBA games. ~90% spread / ~94% ML average win rate.`

#### 6e. Update disclosures section (~lines 526-533)

OLD:
```
• Sweet Spot: 94.9% WR, Moderate: 94.5% WR, Mid-Range: 96.4% WR, Safe: 100% WR
```

NEW:
```
• Spread WR: Sweet Spot 84.6%, Moderate 84.4%, Mid-Range 94.4%, Safe 96.0%
• ML WR: Sweet Spot 90.2%, Moderate 90.9%, Mid-Range 100%, Safe 96.3%
• Both spread and moneyline bets recommended per signal
```

---

### 7. `src/app/alerts.tsx` — Update sample alerts

#### 7a. Update ALL sample alert objects (~lines 17-246)

For each sample alert, update:

- `betType`: Change from `'spread'` to `'spread_and_ml'`
- `recommendation`: e.g., `'BOS -7 SPREAD + ML'`
- `edge`: Use spread edge (10.6 for sweet_spot, 8.4 for moderate, 15.4 for mid_range, 12.0 for safe)
- `modelProbability`: Use spread WR (84.6, 84.4, 94.4, 96.0)
- `impliedProbability`: Keep same (74, 76, 79, 84)
- `reasonCodes`: Update WR values, e.g., `'Sweet Spot Strategy (84.6% Spread / 90.2% ML)'`
- `betInstruction`: e.g., `'BET: Take BOS -7 SPREAD & BOS MONEYLINE'`
- `expectedOutcome`: e.g., `'SPREAD: Win by 8+ pts (84.6%) | ML: Win game (90.2%)'`

Example for Sweet Spot alert:
```typescript
{
  id: 'sample-sweet-1',
  gameId: 'sample-game-1',
  timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
  gameTime: 'Q2 9:30',
  betType: 'spread_and_ml',
  team: 'home',
  recommendation: 'BOS -7 SPREAD + ML',
  edge: 10.6,
  modelProbability: 84.6,
  impliedProbability: 74,
  confidence: 'high',
  reasonCodes: [
    'Sweet Spot Strategy (84.6% Spread / 90.2% ML)',
    '12-pt lead with 11-pt momentum',
    '12-24 min window: 21 min remaining',
  ],
  riskLevel: 'low',
  outcome: 'win',
  isHighConviction: false,
  betInstruction: 'BET: Take BOS -7 SPREAD & BOS MONEYLINE',
  expectedOutcome: 'SPREAD: Win by 8+ (84.6% WR) | ML: Win game (90.2% WR)',
  scoreAtSignal: { home: 48, away: 36, differential: 12 },
  spreadBet: -7,
  indicatorsAtSignal: [],
  _gameHomeTeam: 'BOS',
  _gameAwayTeam: 'NYK',
  _gameDate: 'Jan 4',
  _isSample: true,
  _strategy: 'sweet_spot',
},
```

Apply same pattern to ALL 8 sample alerts, using the correct values for each strategy.

#### 7b. Update "96.5%" in alerts header (~line 402)

OLD: `<Text className="text-emerald-400 text-lg font-bold">96.5%</Text>`
NEW: `<Text className="text-emerald-400 text-lg font-bold">~90%</Text>` (or show both: `90% / 94%`)

Also update label from "Avg Win Rate" to "Avg Spread WR" or "Spread / ML".

#### 7c. Update backtest track record defaults (~lines 502-504)

OLD: `'96.5%'` (win rate), `290` (wins), `10` (losses), `300` (total)
NEW: Use the real validated numbers (independent, with overlap across strategies):
- wins: `101`, losses: `13`, pushes: `6`, total: `120`, win rate: `88.6%`
- Or rounded: wins `101`, losses `13`, total `114` (decided), win rate `89%`
- Update label from "Avg Win Rate" to "Spread Win Rate" or show both spread/ML

---

### 8. `src/components/AlertCard.tsx` — Update bet type display

#### 8a. Update betType display (~line 103)

The card shows `alert.betType.toUpperCase()`. For the new `'spread_and_ml'` value:

Add a display mapping:
```typescript
const betTypeDisplay = alert.betType === 'spread_and_ml'
  ? 'SPREAD + MONEYLINE'
  : alert.betType.toUpperCase();
```

Then use `{betTypeDisplay}` instead of `{alert.betType.toUpperCase()}`.

Apply the same to `AlertCardCompact` at line 250.

---

### 9. Summary of ALL value changes

Quick reference — every place these OLD values appear must change:

| Old Value | Context | New Value |
|-----------|---------|-----------|
| 94.9% | Sweet Spot WR | 84.6% (spread) / 90.2% (ML) |
| 94.5% | Moderate WR | 84.4% (spread) / 90.9% (ML) |
| 96.4% | Mid-Range WR | 94.4% (spread) / 100% (ML) |
| 100% | Safe WR | 96.0% (spread) / 96.3% (ML) |
| 96.5% | Average WR | ~90% spread / ~94% ML |
| 20.9% | Sweet Spot edge | 10.6% spread / 8.2% ML |
| 18.5% | Moderate edge | 8.4% spread / 5.9% ML |
| 17.4% | Mid-Range edge | 15.4% spread / 11.0% ML |
| 16.0% | Safe edge | 12.0% spread / 4.3% ML |
| +$27/$28 | Sweet Spot EV | +$14 spread / +$10 ML / +$24 combined |
| +$22/$24 | Moderate EV | +$11 spread / +$7 ML / +$18 combined |
| +$19/$22 | Mid-Range EV | +$20 spread / +$12 ML / +$32 combined |
| +$16/$19 | Safe EV | +$14 spread / +$5 ML / +$19 combined |
| 300+ games | Validation size | 156 real ESPN games |
| "spread" betType | Bet type | "spread_and_ml" |

### 10. DO NOT CHANGE

- Signal CONDITIONS stay exactly the same (lead ranges, momentum thresholds, time window 12-24 min)
- Priority ordering stays the same (sweet_spot → moderate → mid_range → safe)
- 5-minute momentum calculation stays the same
- Market probability values for spread (74, 76, 79, 84) stay the same
- Scoring-play-only signal detection stays the same
- One signal per game behavior stays the same
