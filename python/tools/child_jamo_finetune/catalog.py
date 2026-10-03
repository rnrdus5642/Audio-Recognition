"""Where every source WAV starts inside its archive.

Two hundred hours of audio is about a hundred gigabytes, and copying it
out would eat most of the free disk for no benefit: the archives are
plain, uncompressed tars, so a WAV inside one is just a byte range.
Recording (archive, offset, size) once lets training open the archive
like any other file, seek, and read exactly that range. The archives are
opened read-only and never modified.

Building the catalogue does not read the payloads. `tarfile` seeks over
them and only the 512-byte headers are read, so a 97 GiB archive is
scanned in minutes.

    python python/tools/child_jamo_finetune/catalog.py
"""
from pathlib import Path
import csv
import re
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, WORK, digest, dump, load, log
from workspace_paths import historical_path, physical_path

COLUMNS = ['wav', 'archive', 'offset', 'size']
OUT = WORK / 'catalog'


def archives():
    found = sorted(p for p in RAW.rglob('*.tar')
                   if re.match(r'[TV]S_', p.name, re.IGNORECASE))
    if not found:
        raise SystemExit(f'no audio archives under {RAW}')
    return found


def identity(path):
    """What makes a catalogue valid for one archive file."""
    path = physical_path(path)
    stat = path.stat()
    return {'path': str(historical_path(path)), 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns}


def scan(path):
    """Write one archive's WAV offsets; returns the rows written.

    A finished catalogue is only reused when the archive it describes is
    still the same file: same resolved path, size and mtime, and the same
    catalogue bytes. Reusing on filename alone would silently survive a
    re-downloaded or truncated archive, and every byte offset in it would
    then point at the wrong audio.
    """
    target = OUT / (path.name + '.csv')
    sidecar = OUT / (path.name + '.json')
    if target.exists() and sidecar.exists():
        record = load(sidecar)
        if (record['archive'] == identity(path)
                and record['catalog_sha256'] == digest(target)):
            return record['wav_members']
        log('catalog.log', f'CATALOG_STALE {path.name}; rescanning')
    temp = target.with_suffix('.csv.tmp')
    n, t0 = 0, time.monotonic()
    with temp.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        with tarfile.open(path, 'r:') as tar:
            for m in tar:
                if not m.isfile() or not m.name.lower().endswith('.wav'):
                    continue
                w.writerow([Path(m.name).name, path.name, m.offset_data, m.size])
                n += 1
                if n % 100000 == 0:
                    log('catalog.log', f'CATALOG {path.name} {n:,} '
                        f'{time.monotonic() - t0:.0f}s')
    temp.replace(target)
    dump(sidecar, {'archive': identity(path), 'wav_members': n,
                   'catalog_sha256': digest(target)})
    log('catalog.log', f'CATALOG_DONE {path.name} {n:,} members '
        f'{time.monotonic() - t0:.0f}s')
    return n


def read():
    """{wav basename: (archive, offset, size)} plus the ambiguous names."""
    entries, duplicates = {}, set()
    for path in sorted(OUT.glob('*.tar.csv')):
        with path.open(encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                name = r['wav']
                value = (r['archive'], int(r['offset']), int(r['size']))
                if name in entries and entries[name] != value:
                    duplicates.add(name)
                entries[name] = value
    return entries, duplicates


def bindings():
    """The recorded identity of every catalogued archive."""
    return {p.stem: load(p) for p in sorted(OUT.glob('*.tar.json'))}


def verify():
    """Re-check every archive against its catalogue; raise on a mismatch.

    Called at training start. The manifests hold byte offsets, so an
    archive that changed underneath them is not a warning - it means the
    run would train on whatever bytes now sit at those offsets.
    """
    recorded = bindings()
    if not recorded:
        raise RuntimeError(f'no catalogue under {OUT}; run catalog.py')
    problems = []
    for name, record in sorted(recorded.items()):
        path = physical_path(record['archive']['path'])
        if not path.exists():
            problems.append(f'{name}: archive missing at {path}')
            continue
        if identity(path) != record['archive']:
            problems.append(f'{name}: archive changed since it was catalogued')
        table = OUT / (name + '.csv')
        if not table.exists() or digest(table) != record['catalog_sha256']:
            problems.append(f'{name}: catalogue file changed since it was written')
    if problems:
        raise RuntimeError('source archives failed verification:\n  '
                           + '\n  '.join(problems))
    # The full binding, not just the size: a rebuilt catalogue for a
    # different copy of an archive could match on byte count alone.
    return {name: {'archive': dict(r['archive']),
                   'catalog_sha256': r['catalog_sha256'],
                   'wav_members': r['wav_members']}
            for name, r in sorted(recorded.items())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = archives()
    log('catalog.log', f'CATALOG_START {len(paths)} archives')
    total = sum(scan(p) for p in paths)
    entries, duplicates = read()
    summary = {
        'archives': [{'name': p.name, **identity(p)} for p in paths],
        'bindings': verify(),
        'wav_members': total,
        'distinct_basenames': len(entries),
        'ambiguous_basenames': sorted(duplicates)[:50],
        'ambiguous_basename_count': len(duplicates),
    }
    dump(WORK / 'catalog-summary.json', summary)
    log('catalog.log', f'CATALOG_COMPLETE members={total:,} '
        f'distinct={len(entries):,} ambiguous={len(duplicates):,}')


if __name__ == '__main__':
    raise SystemExit(main())
