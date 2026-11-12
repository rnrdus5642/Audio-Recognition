"""Publish the approved inference model/package using the existing GitHub Release flow.

Uses the configured Git credential helper in memory; never writes or prints its
credential. Local passing Unity evidence is required before creating a release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import requests

REPO = 'rnrdus5642/Audio-Recognition'
TAG = 'model-v4-child-jamo-epoch3-sentis1'


def publish(directory):
    assert directory.drive.upper() == 'C:'
    for name in ('sentis1-gpu-validation.json', 'installed-package-validation.json'):
        assert json.loads((directory / name).read_text(encoding='utf-8'))['success'], name
    tests = ET.parse(directory / 'policy-tests.xml').getroot()
    assert tests.get('result') == 'Passed' and int(tests.get('passed')) == 22
    release_dir = directory / 'model-release'
    manifest = json.loads((release_dir / 'model-manifest.json').read_text(encoding='utf-8'))
    assert manifest['tag'] == TAG and manifest['repository'] == REPO
    package = directory / manifest['package']['file']
    assert hashlib.sha256(package.read_bytes()).hexdigest() == manifest['package']['sha256']
    paths = [release_dir / part['name'] for part in manifest['parts']]
    for path, part in zip(paths, manifest['parts']):
        assert path.stat().st_size == part['bytes']
        with path.open('rb') as source:
            assert hashlib.file_digest(source, 'sha256').hexdigest() == part['sha256']
    paths += [release_dir / 'model-manifest.json', release_dir / 'RELEASE_NOTES.md', package]
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    credential = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                                text=True, capture_output=True, env=env)
    if credential.returncode:
        raise RuntimeError('Configured GitHub credential is unavailable')
    fields = dict(line.split('=', 1) for line in credential.stdout.splitlines() if '=' in line)
    token = fields.get('password')
    if not token:
        raise RuntimeError('Configured GitHub credential is unavailable')
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                            'User-Agent': 'AudioAnswer-ModelRelease', 'X-GitHub-Api-Version': '2022-11-28'})
    api = 'https://api.github.com/repos/' + REPO
    response = session.get(api + '/releases/tags/' + TAG, timeout=30)
    if response.status_code == 404:
        commit = subprocess.check_output(['git', 'rev-parse', 'origin/main'], text=True).strip()
        response = session.post(api + '/releases', json={
            'tag_name': TAG, 'target_commitish': commit, 'name': 'Final 3-epoch child HuBERT + Unity word context v5',
            'body': (release_dir / 'RELEASE_NOTES.md').read_text(encoding='utf-8'), 'draft': True, 'prerelease': True}, timeout=30)
    response.raise_for_status()
    release = response.json()
    assets = {asset['name']: asset for asset in release['assets']}
    upload = release['upload_url'].split('{')[0]
    for path in paths:
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if path.name in assets:
            existing = assets[path.name]
            if existing['size'] != path.stat().st_size or existing.get('digest') != 'sha256:' + digest:
                raise RuntimeError('Existing release asset differs or lacks a digest: ' + path.name)
            print('Verified existing asset: ' + path.name, flush=True)
            continue
        if not release['draft']:
            raise RuntimeError('Refusing to modify a published release with missing assets')
        print('Uploading ' + path.name + ' (' + str(path.stat().st_size) + ' bytes)', flush=True)
        with path.open('rb') as source:
            response = session.post(upload, params={'name': path.name}, data=source,
                                    headers={'Content-Type': 'application/octet-stream'}, timeout=(30, 1800))
        response.raise_for_status()
        asset = response.json()
        if asset['size'] != path.stat().st_size or asset.get('digest') != 'sha256:' + digest:
            raise RuntimeError('Uploaded asset digest mismatch: ' + path.name)
        print('Uploaded and verified ' + path.name, flush=True)
    if release['draft']:
        response = session.patch(api + '/releases/' + str(release['id']), json={'draft': False}, timeout=30)
        response.raise_for_status()
        release = response.json()
    else:
        response = session.get(api + '/releases/' + str(release['id']), timeout=30)
        response.raise_for_status()
        release = response.json()
    record = {'url': release['html_url'], 'tag': TAG, 'draft': release['draft'], 'prerelease': release['prerelease'],
              'assets': [{'name': a['name'], 'bytes': a['size'], 'digest': a.get('digest'),
                          'url': a['browser_download_url']} for a in release['assets']]}
    (directory / 'published-release.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('Published ' + record['url'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    publish(args.directory.resolve())
