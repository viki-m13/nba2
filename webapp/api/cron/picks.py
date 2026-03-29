"""
Vercel Cron Handler — V8 CSPE & Positive Odds Picks
=====================================================
Handles 4 cron jobs via query params:
  /api/cron/picks?strategy=v8&sport=nba
  /api/cron/picks?strategy=v8&sport=mlb
  /api/cron/picks?strategy=positive-odds&sport=nba
  /api/cron/picks?strategy=positive-odds&sport=mlb
"""

from http.server import BaseHTTPRequestHandler
import sys
import os
import json
import shutil
import importlib.util
import urllib.request
import traceback
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# Config for each (strategy, sport) pair
CONFIGS = {
    ('v8', 'nba'): {
        'script': 'seed_live_v8_picks',
        'data_subdir': 'v8-mgcc/data',
        'files': {
            'SIGNALS_FILE': 'nba_v8_signals.json',
            'STATS_FILE': 'nba_v8_stats.json',
            'RECS_FILE': 'nba_v8_recommendations.json',
        },
        'commit_prefix': 'V8 CSPE NBA',
    },
    ('v8', 'mlb'): {
        'script': 'seed_live_v8_picks_mlb',
        'data_subdir': 'v8-mgcc/data',
        'files': {
            'SIGNALS_FILE': 'mlb_v8_signals.json',
            'STATS_FILE': 'mlb_v8_stats.json',
            'RECS_FILE': 'mlb_v8_recommendations.json',
        },
        'commit_prefix': 'V8 CSPE MLB',
    },
    ('positive-odds', 'nba'): {
        'script': 'seed_live_picks',
        'data_subdir': 'positive-odds/data',
        'files': {
            'SIGNALS_FILE': 'nba_signals.json',
            'STATS_FILE': 'nba_stats.json',
            'RECS_FILE': 'nba_recommendations.json',
        },
        'commit_prefix': 'Positive Odds NBA',
    },
    ('positive-odds', 'mlb'): {
        'script': 'seed_live_picks_mlb',
        'data_subdir': 'positive-odds/data',
        'files': {
            'SIGNALS_FILE': 'mlb_signals.json',
            'STATS_FILE': 'mlb_stats.json',
            'RECS_FILE': 'mlb_recommendations.json',
        },
        'commit_prefix': 'Positive Odds MLB',
    },
}


def find_project_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.exists(os.path.join(d, 'experiments')):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


def http_fetch(url):
    """Drop-in replacement for curl_fetch using urllib."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def commit_to_github(files, message):
    """Create a single commit with multiple file changes via GitHub Git Data API."""
    token = os.environ.get('GITHUB_TOKEN', '')
    repo = os.environ.get('GITHUB_REPO', 'viki-m13/nba2')
    branch = os.environ.get('GITHUB_BRANCH', 'main')

    if not token:
        print('GITHUB_TOKEN not set, skipping commit')
        return None

    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json',
        'User-Agent': 'vercel-cron',
    }
    base_url = f'https://api.github.com/repos/{repo}'

    def api(endpoint, data=None, method='GET'):
        url = f'{base_url}{endpoint}'
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers)
        req.method = method
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    # Get current branch HEAD
    ref = api(f'/git/ref/heads/{branch}')
    head_sha = ref['object']['sha']
    commit_data = api(f'/git/commits/{head_sha}')
    tree_sha = commit_data['tree']['sha']

    # Create blobs and tree entries
    tree_items = []
    for repo_path, content in files:
        blob = api('/git/blobs', {'content': content, 'encoding': 'utf-8'}, 'POST')
        tree_items.append({
            'path': repo_path, 'mode': '100644', 'type': 'blob', 'sha': blob['sha'],
        })

    # Create tree, commit, update ref
    new_tree = api('/git/trees', {'base_tree': tree_sha, 'tree': tree_items}, 'POST')
    new_commit = api('/git/commits', {
        'message': message, 'tree': new_tree['sha'], 'parents': [head_sha],
    }, 'POST')
    api(f'/git/refs/heads/{branch}', {'sha': new_commit['sha']}, 'PATCH')

    return new_commit['sha']


def run_cron(strategy, sport):
    os.environ['TZ'] = 'America/New_York'

    config = CONFIGS.get((strategy, sport))
    if not config:
        raise ValueError(f'Unknown strategy/sport: {strategy}/{sport}')

    project_root = find_project_root()
    experiments_dir = os.path.join(project_root, 'experiments')

    # Setup /tmp output directory
    tmp_data = f'/tmp/cron-{strategy}-{sport}'
    os.makedirs(tmp_data, exist_ok=True)

    # Copy existing data files to /tmp (for read-modify-write pattern)
    src_data = os.path.join(project_root, 'webapp', config['data_subdir'])
    if os.path.isdir(src_data):
        for attr, fname in config['files'].items():
            src = os.path.join(src_data, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp_data, fname))

    # Add experiments dir to sys.path for strategy imports
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)

    # Load the script module via importlib
    script_path = os.path.join(experiments_dir, 'scripts', f'{config["script"]}.py')
    spec = importlib.util.spec_from_file_location(
        f'_cron_{config["script"]}', script_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Override output paths to writable /tmp
    mod.WEBAPP_DATA = tmp_data
    for attr, fname in config['files'].items():
        setattr(mod, attr, os.path.join(tmp_data, fname))

    # Replace curl_fetch with urllib (curl may not be available on Vercel)
    if hasattr(mod, 'curl_fetch'):
        mod.curl_fetch = http_fetch

    # Run the main function
    mod.main()

    # Collect output files and commit to GitHub
    output_files = []
    for attr, fname in config['files'].items():
        fpath = os.path.join(tmp_data, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                content = f.read()
            output_files.append((f'webapp/{config["data_subdir"]}/{fname}', content))

    commit_sha = None
    if output_files:
        now = datetime.now().strftime('%Y-%m-%d %I:%M %p')
        commit_sha = commit_to_github(
            output_files, f'{config["commit_prefix"]} picks: {now}'
        )

    return {
        'status': 'ok',
        'strategy': strategy,
        'sport': sport,
        'files': len(output_files),
        'commit': commit_sha,
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Verify cron secret
        auth = self.headers.get('Authorization', '')
        cron_secret = os.environ.get('CRON_SECRET', '')
        if cron_secret and auth != f'Bearer {cron_secret}':
            self.send_response(401)
            self.end_headers()
            return

        # Parse query params
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        strategy = params.get('strategy', [''])[0]
        sport = params.get('sport', [''])[0]

        if not strategy or not sport:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': 'strategy and sport query params required'
            }).encode())
            return

        try:
            result = run_cron(strategy, sport)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc(),
            }).encode())
