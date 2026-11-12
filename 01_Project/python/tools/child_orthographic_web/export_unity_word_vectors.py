"""Export current web v4 replay evidence and action families for the Unity port."""
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import uuid

from verify_postprocessing import Replay
from word_game import WordGames, StartWordGame

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / '01_Project/unity/Packages/com.domicube.phoneme-matching'
SOURCE = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test/length-window-20261030-054803/inputs.jsonl.gz'
OUTPUT = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/releases/Unity-web-v4-20261030'


def meta(path):
    destination = Path(str(path) + '.meta')
    if destination.exists():
        return
    importer = 'MonoImporter:\n  externalObjects: {}\n  serializedVersion: 2\n  defaultReferences: []\n  executionOrder: 0\n  icon: {instanceID: 0}' if path.suffix == '.cs' else 'DefaultImporter:\n  externalObjects: {}'
    destination.write_text(f'fileFormatVersion: 2\nguid: {uuid.uuid4().hex}\n{importer}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n', encoding='utf-8')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    family_path = Path(__file__).with_name('answer_families.json')
    families = json.loads(family_path.read_text(encoding='utf-8'))['families']
    lines = ['// Generated from web answer_families.json by export_unity_word_vectors.py.',
             '// Source SHA256: ' + hashlib.sha256(family_path.read_bytes()).hexdigest(),
             'namespace DomiCube.PhonemeMatching', '{', '    internal static class OrthographicActionFamilies',
             '    {', '        internal static readonly string[][] Words =', '        {']
    for family, words in families.items():
        lines.append('            new[] { ' + ', '.join(json.dumps(w, ensure_ascii=False) for w in [family, *words]) + ' },')
    lines += ['        };', '    }', '}']
    dest = PACKAGE / 'Runtime/OrthographicActionFamilies.cs'
    dest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    replay = Replay(WordGames, StartWordGame)
    cases, frames = [], 0
    scalar_fields = ['candidate', 'similarity', 'passed', 'ambiguous', 'matched_text', 'comparison_text',
                     'normalization', 'blocked_by', 'decision_reason', 'confirmation_waiting', 'streak', 'candidate_scores']
    with gzip.open(SOURCE, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            result = replay.run(row['targets'], row['frames'])
            checks = []
            for index, trace in enumerate(result['trace']):
                checks.append({'text': trace['text'], 'seconds': trace['elapsed_seconds'],
                               'voice_fraction': trace['voice_fraction'],
                               'expected': {k: trace[k] for k in scalar_fields} |
                               {'confirmed': result['accepted'] and index == len(result['trace']) - 1}})
            cases.append({'id': row['id'], 'targets': row['targets'], 'checks': checks})
            frames += len(checks)
            replay.games.sessions.clear()
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ['word_game.py', 'answer_policy.py', 'answer_families.json']}
    # Store compressed data; tests read it directly from the installed package.
    destination = PACKAGE / 'Tests/Runtime/orthographic_word_vectors.json.gz'
    payload = {'policy': 'orthographic-word-preview-v4', 'cases': cases, 'frames': frames,
               'source_hashes': source_hashes,
               'matrix_sha256': hashlib.sha256((ROOT / '01_Project/shared/confusion_matrices/ko_child_v2.json').read_bytes()).hexdigest()}
    with destination.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
        zipped.write(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    for path in [dest, destination, PACKAGE / 'Runtime/OrthographicWordPolicy.cs',
                 PACKAGE / 'Tests/Runtime/OrthographicWordPolicyTests.cs']:
        if path.exists(): meta(path)
    receipt = {'created_at': datetime.now().astimezone().isoformat(), 'cases': len(cases), 'frames': frames,
               'families': len(families), 'action_spellings': sum(map(len, families.values())),
               'source_hashes': source_hashes, 'fixture_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
               'fixture_bytes': destination.stat().st_size,
               'input_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    (OUTPUT / 'parity-fixtures.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
