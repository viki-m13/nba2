"""
Vercel Cron Runner — executes Python pick scripts with /tmp output.
Called by the Node.js cron handler via child_process.
Env vars: CRON_SCRIPT, CRON_DATA_SUBDIR, CRON_FILES (JSON), CRON_OUTPUT_DIR
"""
import sys
import os
import json
import shutil
import importlib.util
import urllib.request


def http_fetch(url):
    """Drop-in replacement for curl_fetch using urllib."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def main():
    script_name = os.environ['CRON_SCRIPT']
    data_subdir = os.environ['CRON_DATA_SUBDIR']
    files_map = json.loads(os.environ['CRON_FILES'])
    tmp_data = os.environ['CRON_OUTPUT_DIR']

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    experiments_dir = os.path.join(project_root, 'experiments')

    os.makedirs(tmp_data, exist_ok=True)

    # Copy existing data files to /tmp for read-modify-write
    src_data = os.path.join(project_root, 'webapp', data_subdir)
    if os.path.isdir(src_data):
        for attr, fname in files_map.items():
            src = os.path.join(src_data, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp_data, fname))

    # Add experiments dir to sys.path for strategy imports
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)

    # Load the script module via importlib
    script_path = os.path.join(experiments_dir, 'scripts', f'{script_name}.py')
    spec = importlib.util.spec_from_file_location(f'_cron_{script_name}', script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Override output paths to writable /tmp
    mod.WEBAPP_DATA = tmp_data
    for attr, fname in files_map.items():
        setattr(mod, attr, os.path.join(tmp_data, fname))

    # Replace curl_fetch with urllib (curl may not be available on Vercel)
    if hasattr(mod, 'curl_fetch'):
        mod.curl_fetch = http_fetch

    # Run the main function
    mod.main()

    # Output which files were created (read by the JS handler)
    output = {}
    for attr, fname in files_map.items():
        fpath = os.path.join(tmp_data, fname)
        if os.path.exists(fpath):
            output[fname] = fpath
    # Print JSON to stdout for the JS handler to read
    print('__CRON_OUTPUT__' + json.dumps(output))


if __name__ == '__main__':
    main()
