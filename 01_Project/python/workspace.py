"""Physical workspace paths, with read support for frozen pre-move records.

Only ``historical_path`` returns an old spelling, for provenance comparison.
File I/O uses ``physical_path``; no legacy directory or junction is needed.
"""
from functools import lru_cache
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DATASETS = WORKSPACE / '02_Datasets'
RECORDINGS = WORKSPACE / '03_Recordings'
CONFIG = PROJECT / '.workspace-location.json'


@lru_cache(maxsize=2)
def _relocations(include_aliases=False):
    if not CONFIG.is_file():
        return ()
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    if config.get('version') not in (1, 2):
        raise RuntimeError('Unsupported workspace location record')
    original_workspace = Path(config['physical_workspace'])
    entries = []
    mappings = config['directories']
    if include_aliases:
        mappings = config.get('path_aliases', []) + mappings
    for entry in mappings:
        old = Path(entry['source'])
        saved_destination = Path(entry['destination'])
        relative = saved_destination.relative_to(original_workspace)
        destination = (WORKSPACE / relative).resolve()
        if not old.is_absolute() or not destination.is_relative_to(WORKSPACE.resolve()):
            raise RuntimeError('Invalid workspace relocation')
        entries.append((old, saved_destination, destination))
    return tuple(entries)


def physical_path(path):
    """Translate a saved absolute path without requiring its old location."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        for old, saved_destination, destination in _relocations(True):
            for previous in (old, saved_destination):
                try:
                    relative = candidate.relative_to(previous)
                except ValueError:
                    continue
                result = (destination / relative).resolve()
                if not result.is_relative_to(destination):
                    raise ValueError('Saved path escapes its relocated directory')
                return result
    return candidate


def historical_path(path):
    """Frozen identity only; never pass this result to a filesystem reader.

    Archive size/mtime, catalogue hashes, encoder hashes and manifest hashes
    remain checked by the callers. Relocation changes where bytes are read,
    not the recorded identity of the experiment that produced them.
    """
    actual = physical_path(path).resolve()
    for old, _, destination in _relocations():
        try:
            return old / actual.relative_to(destination)
        except ValueError:
            continue
    return actual
