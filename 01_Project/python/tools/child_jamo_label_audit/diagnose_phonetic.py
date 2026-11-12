"""Compare frozen listening answers with the unchanged model, after review.

Capture the logits of the serving Engine's actual forward pass. References
are used only afterwards for edit counts and CTC likelihood diagnostics.
Neither answers nor training labels are changed by this tool.
"""
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

from analyze_phonetic import QUALIFICATIONS, assess, read_qualifications, read_snapshot
from inventory_phonetic import DEST, ROOT
from common import digest, dump, load
import labels

sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_web'))
from engine import Engine, pcm_sha256
import compose
sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_diagnose'))
from ctc import log_softmax, sequence_logp


def main():
    snapshot = read_snapshot()
    evidence = load(DEST / 'evidence.json')
    qualifications = read_qualifications()['qualifications']
    qualification_sha = digest(QUALIFICATIONS)
    assessment = assess(evidence, snapshot, qualifications)
    if assessment['pending_questions']:
        raise RuntimeError('Finish the frozen listening set before this analysis.')
    preparation = load(DEST / 'preparation.json')
    for name, field in [('selection.json', 'selection_sha256'),
                        ('review.json', 'review_sha256'),
                        ('evidence.json', 'evidence_sha256'),
                        ('comparison-plan.json', 'comparison_plan_sha256')]:
        assert digest(DEST / name) == preparation[field], name
    selection = load(DEST / 'selection.json')
    assert labels.contract_sha256() == selection['labels_contract_sha256']
    answers = {r['number']: r for r in assessment['items']}
    revision = snapshot['latest_revision']
    suffix = f'revision-{revision:03d}-qualified-{qualification_sha[:12]}'
    output = DEST / f'baseline-diagnosis-{suffix}.json'
    if output.exists():
        raise RuntimeError(f'Keep the existing diagnosis: {output}')
    traces = DEST / f'baseline-logits-{suffix}.npz'
    if traces.exists():
        raise RuntimeError(f'Keep the existing frame evidence: {traces}')

    torch.set_num_threads(4)
    model = Engine()
    print('Loading the unchanged baseline for 15 reviewed clips.', flush=True)
    model.load()
    assert model.identity['checkpoint_sha256'] == selection['baseline_checkpoint_sha256']
    rows, arrays = [], {}
    for entry in evidence:
        number = entry['number']
        answer = answers[number]
        assert answer['binding'] == entry['binding']
        source = DEST / 'audio' / entry['served_train']['file']
        assert digest(source) == entry['served_train']['sha256']
        waveform, rate = sf.read(source, dtype='float32')
        assert rate == 16000 and len(waveform) == entry['row']['samples']
        assert pcm_sha256(waveform) == entry['row']['pcm_sha256']

        captured = []
        handle = model.head.register_forward_hook(
            lambda _module, _inputs, result: captured.append(result.detach().cpu()))
        try:
            prediction = model.transcribe(waveform, use_cache=False)
        finally:
            handle.remove()
        assert len(captured) == 1
        logits = captured[0][0, :prediction['frames']].numpy().copy()
        argmax = logits.argmax(-1).tolist()
        assert compose.collapse(argmax) == prediction['token_ids']
        assert compose.compose(prediction['token_ids'])['text'] == prediction['text']
        arrays[f'question_{number:02d}'] = logits

        # These references never enter the forward pass or greedy decoding.
        original = answer['original_ids']
        confirmed = answer['confirmed_ids']
        logp = log_softmax(logits)
        different = ([] if confirmed is None else
                     [op for op in compose.alignment(confirmed, prediction['token_ids'])
                      if op['op'] != 'equal'])
        affected = {token for op in different for token in op['reference'] + op['hypothesis']}
        token_evidence = []
        for token in sorted(affected, key=labels.VOCAB.__getitem__):
            token_id = labels.VOCAB[token]
            peak = int(logp[:, token_id].argmax())
            token_evidence.append({
                'token': token, 'argmax_frames': argmax.count(token_id),
                'peak_frame': peak,
                'peak_frame_softmax': float(np.exp(logp[peak, token_id])),
                'winner_at_peak': labels.TOKENS[argmax[peak]],
            })
        runs = []
        for frame, token_id in enumerate(argmax):
            if runs and runs[-1]['token_id'] == token_id:
                runs[-1]['end_frame_exclusive'] = frame + 1
            else:
                runs.append({'token_id': token_id, 'token': labels.TOKENS[token_id],
                             'start_frame': frame, 'end_frame_exclusive': frame + 1})
        rows.append({
            'number': number, 'case': entry['case'], 'role': entry['role'],
            'binding': entry['binding'], 'pcm_sha256': entry['row']['pcm_sha256'],
            'answer_revision': answer['answer_revision'], 'answer_status': answer['status'],
            'selected_surface': answer['selected_surface'],
            'qualification_ids': answer['qualification_ids'],
            'original_surface': answer['original_surface'],
            'confirmed_surface': answer['confirmed_surface'], 'prediction': prediction,
            'original_score': compose.compare(original, prediction['token_ids']),
            'confirmed_score': None if confirmed is None else compose.compare(confirmed, prediction['token_ids']),
            'differences_from_confirmed': different,
            'original_sequence_logp': sequence_logp(logp, original),
            'confirmed_sequence_logp': None if confirmed is None else sequence_logp(logp, confirmed),
            'affected_token_evidence': token_evidence, 'frame_runs': runs,
        })
        print(f"{number:02d}: {prediction['text']} / confirmed={answer['confirmed_surface']}", flush=True)

    assert read_snapshot() == snapshot, 'Answers changed during analysis; do not publish stale results.'
    assert digest(QUALIFICATIONS) == qualification_sha
    summary = {}
    for role in ('all', 'relabel_train', 'matched_development'):
        group = [r for r in rows if role == 'all' or r['role'] == role]
        confirmed = [r for r in group if r['confirmed_score'] is not None]
        summary[role] = {
            'recordings': len(group), 'confirmed_recordings': len(confirmed),
            'unlabelled_questions': [r['number'] for r in group if r['confirmed_score'] is None],
            'exact_confirmed': sum(r['confirmed_score']['exact'] for r in confirmed),
            'confirmed_errors': sum(r['confirmed_score']['errors'] for r in confirmed),
            'confirmed_reference_jamo': sum(r['confirmed_score']['reference_jamo'] for r in confirmed),
        }
    np.savez_compressed(traces, **arrays)
    report = {
        'analyzed_at_utc': datetime.now(timezone.utc).isoformat(),
        'answer_revision': revision, 'review_sha256': snapshot['review_sha256'],
        'qualifications_sha256': qualification_sha, 'qualifications': qualifications,
        'evidence_sha256': digest(DEST / 'evidence.json'),
        'script_sha256': digest(Path(__file__)), 'identity': model.identity,
        'software': {'torch': torch.__version__, 'numpy': np.__version__},
        'logits_file': traces.name, 'logits_sha256': digest(traces),
        'summary': summary, 'items': rows,
        'limits': [
            'Selected development recordings and previously trained examples, not a population accuracy estimate.',
            'Confirmed fields mean eligible single-listener selections, not independently verified sound labels. Later uncertain cases are excluded from scoring.',
            'Other-form answers have no scoring target; their notes are not converted into labels.',
            'CTC sequence likelihoods are post-hoc diagnostics, never input to recognition.',
            'Frame indices are encoder positions, not verified phonetic boundaries; softmax is not human pronunciation probability.',
            'A weak token score does not establish why training produced that score.',
        ],
        'training_started': False, 'model_replaced': False,
    }
    dump(output, report)
    date = datetime.now().astimezone().strftime('%Y%m%d')
    dump(ROOT / f'docs/data/child-jamo-phonetic-diagnosis-{date}-qualified-{qualification_sha[:12]}.json', report)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
