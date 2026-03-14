// Vercel Serverless API Proxy for MLB ESPN endpoints
// Solves CORS issues when deploying to Vercel

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const { endpoint, eventId, dates } = req.query;

  let url;

  switch (endpoint) {
    case 'scoreboard':
      url = 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard';
      if (dates) {
        url += `?dates=${dates}`;
      }
      break;
    case 'summary':
      if (!eventId) {
        return res.status(400).json({ error: 'eventId required for summary' });
      }
      url = `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event=${eventId}`;
      break;
    default:
      return res.status(400).json({ error: 'Invalid endpoint. Use: scoreboard or summary' });
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
        error: `ESPN API returned ${response.status}`,
      });
    }

    const data = await response.json();

    const isHistorical = endpoint === 'scoreboard' && dates;
    res.setHeader('Cache-Control', isHistorical
      ? 's-maxage=300, stale-while-revalidate=60'
      : 's-maxage=30, stale-while-revalidate=10');

    return res.status(200).json(data);
  } catch (error) {
    return res.status(500).json({
      error: 'Failed to fetch from ESPN API',
      message: error.message,
    });
  }
}
