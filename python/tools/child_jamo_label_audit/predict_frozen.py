"""Attach the unchanged baseline's outputs AFTER selection. Never serve these to listeners."""
import hashlib
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_web'))
from engine import Engine

DEST = ROOT / '.venv/child-jamo-web-20260916/dataset-audit-20260928'
EXPECTED = '5a330f1bea3a17d1d063fdfd71a0cc25394695b0b797de01b293f6f688257aa7'


def main():
    evidence = json.loads((DEST / 'evidence.json').read_text(encoding='utf-8'))
    selection = (DEST / 'selection.json').read_bytes()
    print('Loading the unchanged baseline for 20 frozen clips; no training.', flush=True)
    engine = Engine()
    engine.load()
    assert engine.identity['checkpoint_sha256'] == EXPECTED
    predictions = []
    for item in evidence:
        wave, rate = sf.read(DEST / 'audio' / item['served_train']['file'], dtype='float32')
        assert rate == 16000
        result = engine.transcribe(wave, pcm_sha256=item['row']['pcm_sha256'])
        predictions.append({'number': item['number'], 'binding': item['binding'], 'group': item['group'],
                            'prediction': result, 'training_target_agreement': result['token_ids'] == item['row']['label_ids']})
        print(f"Predicted {item['number']}/20", flush=True)
    assert (DEST / 'selection.json').read_bytes() == selection
    report = {'checkpoint_sha256': EXPECTED, 'selection_sha256': hashlib.sha256(selection).hexdigest(),
              'identity': engine.identity, 'selection_modified': False, 'predictions': predictions,
              'target_agreements': sum(p['training_target_agreement'] for p in predictions),
              'target_disagreements': sum(not p['training_target_agreement'] for p in predictions),
              'scope': 'After-selection analysis only. Model/target agreement is not acoustic correctness; predictions are not shown to the listener.'}
    (DEST / 'baseline-predictions.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('target_agreements', 'target_disagreements', 'selection_modified')}), flush=True)


if __name__ == '__main__':
    main()
