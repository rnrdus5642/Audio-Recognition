"""The audit snapshot is the single record that vouches for the data.

Each manifest is bound to the audit that produced it, and that audit
freezes the label contract, the selection it drew from, and the archive
bindings the byte offsets refer to. The failure these guard against is
quiet: two records describing different generations while each check
passes on its own - for instance `select` re-run without `audit`, so the
manifests are new but the audit still describes the old ones.

Every case here restores the file it touched.

    .venv-gpu/Scripts/python.exe -m pytest \
        python/tools/child_jamo_finetune/test_provenance.py -q
"""
from pathlib import Path
import contextlib
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog
import integrity
from common import WORK

SPLITS = {'train': 'train', 'valid': 'valid', 'test': 'test'}


@contextlib.contextmanager
def edited(path, mutate):
    """Apply a change to a JSON file, then put it back byte for byte."""
    path = Path(path)
    original = path.read_bytes()
    try:
        document = json.loads(original.decode('utf-8'))
        mutate(document)
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
        yield
    finally:
        path.write_bytes(original)


def test_the_current_data_passes():
    contract = integrity.contract(SPLITS)
    assert contract['splits']['train']['vouched_by'] == 'audit-summary.json'
    assert len(contract['source_archives']) == 8


def test_reselecting_without_reauditing_is_refused():
    """The exact gap: new selection metadata, stale audit, both self-consistent."""
    def touch(document):
        document['seed'] = document['seed'] + '-rerun'

    with edited(WORK / 'selection-summary.json', touch):
        with pytest.raises(RuntimeError, match='selection-summary.json has changed'):
            integrity.contract(SPLITS)


def test_a_changed_archive_binding_is_refused():
    """Offsets are meaningless against an archive that is not the audited one."""
    name = sorted(catalog.bindings())[0]

    def touch(document):
        document['archive']['mtime_ns'] += 1

    with edited(catalog.OUT / f'{name}.json', touch):
        with pytest.raises(RuntimeError) as raised:
            integrity.contract(SPLITS)
        # catalog.verify() catches it first; either way the run stops.
        assert name in str(raised.value)


def test_a_changed_label_contract_is_refused():
    def touch(document):
        document['provenance']['labels_contract_sha256'] = '0' * 64

    with edited(WORK / 'audit-summary.json', touch):
        with pytest.raises(RuntimeError, match='label contract'):
            integrity.contract(SPLITS)


def test_a_manifest_the_audit_does_not_know_is_refused():
    def touch(document):
        document['manifest_sha256']['train'] = '0' * 64

    with edited(WORK / 'audit-summary.json', touch):
        with pytest.raises(RuntimeError, match='not produced by the recorded'):
            integrity.contract(SPLITS)


def test_an_audit_without_provenance_is_refused():
    """An audit from before this binding existed must not be trusted."""
    def touch(document):
        document.pop('provenance')

    with edited(WORK / 'audit-summary.json', touch):
        with pytest.raises(RuntimeError, match='predates provenance recording'):
            integrity.contract(SPLITS)


def test_smoke_views_carry_their_own_binding():
    contract = integrity.contract({'train': 'smoketrain', 'valid': 'smokevalid',
                                   'test': 'smoketest'})
    assert all(s['vouched_by'] == 'smoke-view.json'
               for s in contract['splits'].values())
