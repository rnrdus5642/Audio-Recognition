"""Download public checkpoints at pinned revisions with size/hash verification."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time
import traceback
import urllib.request
from common import WORK, MODELS, digest, dump, load


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'AudioRecognition-encoder-screen/1.0'}), timeout=60)


def one(key, spec):
    out = WORK / 'models' / key
    out.mkdir(parents=True, exist_ok=True)
    verified = out / 'verified.json'
    if verified.exists():
        rec = load(verified)
        assert digest(out / rec['weight']) == rec['sha256']
        print('REUSED', key, flush=True)
        return rec
    rec = dict(key=key, **spec)
    if 'repo' in spec:
        api_path = out / 'api.json'
        if api_path.exists():
            api = load(api_path)
        else:
            with get('https://huggingface.co/api/models/' + spec['repo'] + '?blobs=true') as f:
                api = json.load(f)
            dump(api_path, api)
        rec['revision'] = api['sha']
        rec['license'] = api.get('cardData', {}).get('license')
        by_name = {r['rfilename']: r for r in api['siblings']}
        filename = spec.get('weight') or ('model.safetensors' if 'model.safetensors' in by_name else 'pytorch_model.bin')
        detail = by_name[filename]
        expected = detail.get('lfs', {}).get('sha256')
        expected_size = detail.get('size')
        prefix = f"https://huggingface.co/{spec['repo']}/resolve/{api['sha']}/"
        for name in ['config.json', 'preprocessor_config.json', 'README.md', 'meta.yaml', 'model/config.yaml']:
            if name in by_name:
                file = out / name
                if not file.exists():
                    file.parent.mkdir(parents=True, exist_ok=True)
                    with get(prefix + name) as f:
                        file.write_bytes(f.read())
        url = prefix + filename
    else:
        filename = 'checkpoint.pt'
        url = spec['url']
        expected = expected_size = None
        rec['license'] = 'apache-2.0'
    file = out / filename
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(file.name + '.partial')
    if not file.exists():
        for attempt in range(3):
            try:
                with get(url) as response, tmp.open('wb') as f:
                    length = int(response.headers.get('Content-Length', 0))
                    size = 0
                    last = time.monotonic()
                    print('DOWNLOAD', key, round(length / 1e9, 2), 'GB', flush=True)
                    while part := response.read(8 * 1024 * 1024):
                        f.write(part)
                        size += len(part)
                        if time.monotonic() - last > 25:
                            print('PROGRESS', key, round(size / 1e9, 2), 'GB', flush=True)
                            last = time.monotonic()
                if expected_size: assert size == expected_size
                if length: assert size == length
                if expected: assert digest(tmp) == expected
                tmp.replace(file)
                break
            except Exception as exc:
                print('RETRY', key, attempt, type(exc).__name__, str(exc)[:120], flush=True)
                if attempt == 2: raise
    sha = digest(file)
    if expected: assert sha == expected
    if expected_size: assert file.stat().st_size == expected_size
    rec.update(weight=filename, bytes=file.stat().st_size, sha256=sha,
               publisher_hash_available=bool(expected), source_url=url)
    dump(verified, rec)
    print('VERIFIED', key, flush=True)
    return rec


if __name__ == '__main__':
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(one, k, s): k for k, s in MODELS.items()}
        for f in as_completed(futures):
            try: results.append(f.result())
            except Exception:
                failure = {'key': futures[f], 'traceback': traceback.format_exc()}
                failures.append(failure)
                print('FAILED', futures[f], failure['traceback'], flush=True)
            dump(WORK / 'downloads.json', {'completed': results, 'failed': failures})
    print('DOWNLOADS_FINISHED', len(results), 'failed', len(failures), flush=True)
