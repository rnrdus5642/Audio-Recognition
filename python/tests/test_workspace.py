"""Relocation must work when old directories do not exist."""
import json
from pathlib import Path

import pytest

from python import workspace


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    old = tmp_path / 'removed' / 'old-project'
    current = tmp_path / 'workspace'
    target = current / '01_Project'
    target.mkdir(parents=True)
    config = target / '.workspace-location.json'
    config.write_text(json.dumps({
        'version': 2, 'physical_workspace': str(current),
        'directories': [{'source': str(old), 'destination': str(target)}],
    }), encoding='utf-8')
    monkeypatch.setattr(workspace, 'CONFIG', config)
    monkeypatch.setattr(workspace, 'WORKSPACE', current)
    workspace._relocations.cache_clear()
    yield old, target
    workspace._relocations.cache_clear()


def test_old_manifest_reads_current_bytes_without_links(relocated):
    old, target = relocated
    (target / 'recording.wav').write_bytes(b'unchanged audio')
    assert not old.exists()
    translated = workspace.physical_path(old / 'recording.wav')
    assert translated.read_bytes() == b'unchanged audio'
    assert workspace.historical_path(translated) == old / 'recording.wav'


def test_unknown_or_similarly_named_directory_is_not_rewritten(relocated):
    old, _ = relocated
    sibling = old.with_name(old.name + '-other') / 'file'
    assert workspace.physical_path(sibling) == sibling
    assert workspace.physical_path('relative/file') == Path('relative/file')


def test_saved_path_cannot_escape_mapped_directory(relocated):
    old, _ = relocated
    with pytest.raises(ValueError, match='escapes'):
        workspace.physical_path(old / '..' / 'another-project' / 'file')


def test_missing_file_does_not_fall_back_to_old_copy(relocated):
    old, _ = relocated
    old.mkdir(parents=True)
    (old / 'stale.wav').write_bytes(b'stale copy')
    with pytest.raises(FileNotFoundError):
        workspace.physical_path(old / 'stale.wav').read_bytes()


def test_more_specific_audio_alias_precedes_parent_directory(relocated):
    old, target = relocated
    config = json.loads(workspace.CONFIG.read_text(encoding='utf-8'))
    actual_audio = target.parent / '02_Datasets' / 'audio'
    config['path_aliases'] = [{'source': str(old / 'audio'), 'destination': str(actual_audio)}]
    workspace.CONFIG.write_text(json.dumps(config), encoding='utf-8')
    assert workspace.physical_path(old / 'audio' / 'clip.wav') == actual_audio / 'clip.wav'
    assert workspace.physical_path(old / 'index.csv') == target / 'index.csv'
