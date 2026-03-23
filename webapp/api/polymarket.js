// Vercel Serverless API Proxy for Polymarket endpoints
// Fetches real-time MLB game prices from Polymarket CLOB + Gamma APIs

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const { endpoint, query, token_id, token_ids } = req.query;

  let url;

  switch (endpoint) {
    case 'markets':
      // Search Polymarket markets — filter for MLB games
      // Gamma API: https://gamma-api.polymarket.com/markets
      url = 'https://gamma-api.polymarket.com/markets';
      const params = new URLSearchParams();
      if (query) params.set('tag', query);
      params.set('active', 'true');
      params.set('closed', 'false');
      params.set('limit', '100');
      const qs = params.toString();
      if (qs) url += `?${qs}`;
      break;

    case 'search':
      // Full-text search for markets by team name / game
      url = `https://gamma-api.polymarket.com/markets?_q=${encodeURIComponent(query || 'MLB')}&active=true&closed=false&limit=50`;
      break;

    case 'price':
      // Get real-time price for a token
      if (!token_id) {
        return res.status(400).json({ error: 'token_id required' });
      }
      url = `https://clob.polymarket.com/price?token_id=${token_id}&side=buy`;
      break;

    case 'prices':
      // Get prices for multiple tokens
      if (!token_ids) {
        return res.status(400).json({ error: 'token_ids required (comma-separated)' });
      }
      // Fetch prices for each token in parallel
      try {
        const ids = token_ids.split(',').slice(0, 20); // max 20
        const priceResults = await Promise.allSettled(
          ids.map(async (tid) => {
            const r = await fetch(
              `https://clob.polymarket.com/price?token_id=${tid.trim()}&side=buy`,
              { headers: { 'Accept': 'application/json' }, signal: AbortSignal.timeout(8000) }
            );
            if (!r.ok) return { token_id: tid.trim(), price: null, error: r.status };
            const d = await r.json();
            return { token_id: tid.trim(), price: parseFloat(d.price || '0') };
          })
        );
        const prices = priceResults.map((r, i) =>
          r.status === 'fulfilled' ? r.value : { token_id: ids[i].trim(), price: null, error: 'failed' }
        );
        res.setHeader('Cache-Control', 's-maxage=10, stale-while-revalidate=5');
        return res.status(200).json({ prices });
      } catch (error) {
        return res.status(500).json({ error: 'Failed to fetch prices', message: error.message });
      }

    case 'book':
      // Get order book for a token
      if (!token_id) {
        return res.status(400).json({ error: 'token_id required' });
      }
      url = `https://clob.polymarket.com/book?token_id=${token_id}`;
      break;

    case 'events':
      // Get events (grouped markets) — sports events
      url = `https://gamma-api.polymarket.com/events?tag=${encodeURIComponent(query || 'mlb')}&active=true&closed=false&limit=50`;
      break;

    default:
      return res.status(400).json({
        error: 'Invalid endpoint. Use: markets, search, price, prices, book, events',
      });
  }

  try {
    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0',
      },
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      return res.status(response.status).json({
        error: `Polymarket API returned ${response.status}`,
        detail: text.slice(0, 500),
      });
    }

    const data = await response.json();
    res.setHeader('Cache-Control', 's-maxage=15, stale-while-revalidate=5');
    return res.status(200).json(data);
  } catch (error) {
    return res.status(500).json({
      error: 'Failed to fetch from Polymarket API',
      message: error.message,
    });
  }
}
