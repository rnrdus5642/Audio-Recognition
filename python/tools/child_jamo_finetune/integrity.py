"""The immutable identity of everything that decides what a run trains on.

Hashing the current files against each other is not enough. Editing
`labels.py` and keeping yesterday's manifest would produce a run whose
config records the new label contract while the targets on disk came from
the old one, and every number afterwards would be attributed to the wrong
implementation. So each manifest is checked against the audit that
actually produced it, and that audit records which label contract was in
force at the time.

The same applies to the audio. Manifests address recordings by byte range
inside an archive, so the archives are bound by resolved path, size and
mtime together with the catalogue's own hash - not by size alone, which a
rebuild elsewhere could reproduce by coincidence.

The encoder is bound by its weights *and* its `config.json` and
`preprocessor_config.json`. The preprocessor is what says this checkpoint
takes raw unnormalised audio; swapping it changes the input contract
without touching a single weight.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog
import labels
import quality
import reviewed
from common import ENCODER, ROOT, WORK, digest, load
from workspace_paths import historical_path

#: Splits built by manifest.py, provenance recorded in audit-summary.json.
AUDITED = ('train', 'valid', 'test')


def encoder_identity():
    files = ['model.safetensors', 'config.json', 'preprocessor_config.json']
    return {'path': str(historical_path(ENCODER)),
            **{f: digest(ENCODER / f) for f in files}}


def _vouched(name):
    """The one record that vouches for a split, and what it froze.

    For the audited splits everything comes from `audit-summary.json`:
    the manifest hash, the label contract in force when the targets were
    written, the selection metadata they were selected by, and the
    archive bindings they were read from. Taking any of those from a
    separate, mutable file would let two records describe different
    generations while each check passed on its own.
    """
    if name in AUDITED:
        audit = load(WORK / 'audit-summary.json')
        if 'provenance' not in audit:
            raise RuntimeError(
                'audit-summary.json predates provenance recording; re-run '
                '"manifest.py audit" to bind these manifests to their '
                'selection, label contract and archives.')
        return ('audit-summary.json', audit['manifest_sha256'].get(name),
                audit['provenance'])
    repair_path = WORK / f'audit-{name}.json'
    if repair_path.exists():
        audit = load(repair_path)
        if (audit.get('kind') != 'reviewed_training_repair' or
                not audit.get('complete') or audit.get('split') != name):
            raise RuntimeError(f'Invalid repair audit for split "{name}"')
        return (repair_path.name, audit['manifest_sha256'].get(name), audit['provenance'])
    view = load(WORK / 'smoke-view.json')
    return ('smoke-view.json', view['views'].get(name, {}).get('sha256'),
            {'labels_version': view['labels_version'],
             'labels_contract_sha256': view['labels_contract_sha256']})


def _provenance(name):
    """Check one split against the record that vouches for it."""
    path = WORK / f'manifest-{name}.jsonl'
    if not path.exists():
        raise FileNotFoundError(f'missing manifest for split "{name}": {path}')
    current = digest(path)
    source, recorded, frozen = _vouched(name)
    if recorded != current:
        raise RuntimeError(
            f'split "{name}" was not produced by the recorded {source}: '
            f'manifest is {current[:16]}, {source} says {recorded and recorded[:16]}. '
            f'Rebuild the manifests before training.')
    if frozen['labels_contract_sha256'] != labels.contract_sha256():
        raise RuntimeError(
            f'split "{name}" was built with label contract '
            f'{frozen["labels_contract_sha256"][:16]} but labels.py now hashes '
            f'to {labels.contract_sha256()[:16]}. The targets on disk are not '
            f'the ones this code would produce; rebuild the manifests.')
    out = {'manifest_sha256': current, 'vouched_by': source,
           'labels_contract_sha256': frozen['labels_contract_sha256']}
    if name in AUDITED:
        # The selection that produced these rows, and the archives they
        # were read from, are part of the same snapshot.
        live = digest(WORK / 'selection-summary.json')
        if frozen['selection_summary_sha256'] != live:
            raise RuntimeError(
                f'selection-summary.json has changed since the audit that '
                f'vouches for split "{name}" (audit recorded '
                f'{frozen["selection_summary_sha256"][:16]}, file is '
                f'{live[:16]}). Selection was re-run without re-auditing; '
                f're-run "manifest.py audit".')
        bound = catalog.verify()
        if frozen['source_archives'] != bound:
            changed = sorted(k for k in set(frozen['source_archives']) | set(bound)
                             if frozen['source_archives'].get(k) != bound.get(k))
            raise RuntimeError(
                f'the source archives no longer match the audit that vouches '
                f'for split "{name}": {", ".join(changed)}. The manifests '
                f'address audio by byte offset, so re-run "manifest.py audit" '
                f'against the current archives.')
        out['selection_summary_sha256'] = live
    if 'base_split' in frozen:
        if frozen['base_split'] != 'train':
            raise RuntimeError('A repair must derive from the original training split')
        parent = _provenance('train')
        if (parent['manifest_sha256'] != frozen['base_manifest_sha256'] or
                digest(WORK / 'audit-summary.json') != frozen['base_audit_sha256']):
            raise RuntimeError('Baseline provenance changed since the repair audit')
        evidence = frozen['listening_evidence']
        if digest(ROOT / evidence['path']) != evidence['sha256']:
            raise RuntimeError('Listening evidence changed since the repair audit')
        out['repair_audit_sha256'] = digest(WORK / source)
    if 'reviewed_repairs_sha256' in frozen:
        if reviewed.contract_sha256() != frozen['reviewed_repairs_sha256']:
            raise RuntimeError('Listening decisions changed; rebuild the repaired data')
        if quality.contract_sha256() != frozen['quality_contract_sha256']:
            raise RuntimeError('Quality policy changed; re-audit the data')
        review_file = frozen.get('quality_review_file', 'quality-review.json')
        if digest(WORK / review_file) != frozen['quality_review_sha256']:
            raise RuntimeError('Quality review report changed since the audit')
        out['reviewed_repairs_sha256'] = frozen['reviewed_repairs_sha256']
        out['quality_review_sha256'] = frozen['quality_review_sha256']
    return out


def contract(splits):
    """Everything frozen for the life of a run. Raises rather than warns."""
    return {
        'source_archives': catalog.verify(),
        'audited_provenance_sha256': digest(WORK / 'audit-summary.json'),
        'encoder': encoder_identity(),
        'labels_version': labels.LABELS_VERSION,
        'labels_contract_sha256': labels.contract_sha256(),
        'splits': {role: {'name': name, **_provenance(name)}
                   for role, name in splits.items()},
    }


def differences(saved, fresh, path=()):
    """Every leaf where two contracts disagree, as dotted paths."""
    out = {}
    if isinstance(saved, dict) and isinstance(fresh, dict):
        for key in sorted(set(saved) | set(fresh)):
            out.update(differences(saved.get(key), fresh.get(key),
                                   path + (str(key),)))
    elif saved != fresh:
        out['.'.join(path)] = {'checkpoint': saved, 'now': fresh}
    return out
