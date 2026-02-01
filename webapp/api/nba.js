// Vercel Serverless API Proxy for NBA.com endpoints
// Solves CORS issues when deploying to Vercel

export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const { endpoint, gameId } = req.query;

  let url;

  switch (endpoint) {
    case 'scoreboard':
      url = 'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json';
      break;
    case 'playbyplay':
      if (!gameId) {
        return res.status(400).json({ error: 'gameId required for playbyplay' });
      }
      url = `https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_${gameId}.json`;
      break;
    case 'espn_scoreboard':
      url = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard';
      break;
    default:
      return res.status(400).json({ error: 'Invalid endpoint. Use: scoreboard or playbyplay' });
  }

  try {
    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0',
      },
    });

    if (!response.ok) {
      return res.status(response.status).json({
        error: `NBA API returned ${response.status}`,
      });
    }

    const data = await response.json();

    // Cache for 3 seconds (live data)
    res.setHeader('Cache-Control', 's-maxage=3, stale-while-revalidate=1');

    return res.status(200).json(data);
  } catch (error) {
    return res.status(500).json({
      error: 'Failed to fetch from NBA API',
      message: error.message,
    });
  }
}
