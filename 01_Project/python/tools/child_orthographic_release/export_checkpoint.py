"""Export an immutable inference bundle without changing an active training run."""
import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export(run, destination, label):
    # The training writer owns its lock/index/pins. Only read committed receipts.
    run = Path(run).resolve()
    destination = Path(destination).resolve()
    if run.drive.upper() != 'C:' or destination.drive.upper() != 'C:':
        raise ValueError('Only C-drive paths are supported for this project')
    if destination.is_relative_to(run) or run.is_relative_to(destination):
        raise ValueError('Inference releases must be outside the active run')
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    receipt = read(run / 'checkpoints/committed.json')[0]
    name = receipt['file']
    if Path(name).name != name or not name.startswith('ckpt-') or not name.endswith('.pt'):
        raise ValueError('Invalid committed checkpoint filename')
    source = run / 'checkpoints' / name
    snapshot = destination / 'source-checkpoint.snapshot.pt'
    # A separate NTFS link protects the immutable file against rolling retention.
    # Never write through this link or change the trainer's pinned generations.
    try:
        os.link(source, snapshot)
    except OSError:
        shutil.copyfile(source, snapshot)
    print(json.dumps({'event': 'snapshot_acquired', 'receipt': receipt}), flush=True)
    if snapshot.stat().st_size != receipt['bytes'] or digest(snapshot) != receipt['sha256']:
        raise ValueError('Source checkpoint size/SHA256 mismatch; snapshot retained for inspection')

    import torch
    torch.set_num_threads(2)
    # This is a trusted locally produced training checkpoint; it includes RNG state.
    payload = torch.load(snapshot, map_location='cpu', weights_only=False)
    state = payload['state']
    if state['total_updates'] != receipt['update'] or state['weights_id'] != receipt['weights_id']:
        raise ValueError('Checkpoint state and committed receipt differ')
    model, head = payload['model'], payload['head']
    if tuple(head['1.weight'].shape) != (69, 1280) or tuple(head['1.bias'].shape) != (69,):
        raise ValueError('Expected the current 69-token orthographic head')
    contract = payload['contract']
    del payload
    gc.collect()
    temporary = destination / 'model.pt.tmp'
    with temporary.open('wb') as stream:
        torch.save({'model': model, 'head': head}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination / 'model.pt')
    del model, head
    gc.collect()

    encoder = ROOT / '01_Project/.venv/encoder-screen-0914/models/hubertko1b'
    prep = ROOT / '02_Datasets/학습준비/자모_원문라벨_3-6세'
    config = read(encoder / 'config.json')
    config.update(layerdrop=0, apply_spec_augment=False)
    preprocessing = read(encoder / 'preprocessor_config.json')
    if preprocessing['do_normalize'] is not False:
        raise ValueError('Unexpected waveform normalization policy')
    vocab = read(prep / 'vocab.json')
    if len(vocab) != 69 or sorted(vocab.values()) != list(range(69)) or vocab.get('|') != 1:
        raise ValueError('Unexpected orthographic vocabulary')
    write(destination / 'config.json', config)
    write(destination / 'preprocessor_config.json', preprocessing)
    write(destination / 'vocab.json', vocab)
    write(destination / 'training-contract.json', contract)
    progress = read(run / 'progress.json')
    equivalents = state['joint_audio_seconds'] / (progress['train_hours'] * 3600)
    manifest = {
        'format': 'orthographic-hubert-ctc-v1',
        'label': label,
        'provisional': True,
        'created_at': datetime.now().astimezone().isoformat(),
        'source_receipt': receipt,
        'epoch_equivalents': equivalents,
        'training_lineage': state['lineage'],
        'sample_rate': 16000,
        'channels': 1,
        'normalize_waveform': False,
        'decoder': 'ctc_greedy_collapse_then_unicode_nfc',
        'blank_id': 0,
        'space_id': 1,
        'vocab_size': 69,
        'hidden_size': 1280,
        'weights_file': 'model.pt',
        'files': {},
        'validation': {'status': 'not_yet_smoke_tested', 'source_checkpoint_monitor_cer': None},
    }
    for filename in ('model.pt', 'config.json', 'preprocessor_config.json', 'vocab.json', 'training-contract.json'):
        path = destination / filename
        manifest['files'][filename] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    write(destination / 'release.json', manifest)
    # Remove only our temporary link, after the independent bundle is complete.
    snapshot.unlink()
    print(json.dumps({'event': 'export_complete', 'destination': str(destination),
                      'epoch_equivalents': equivalents, 'weights_bytes': manifest['files']['model.pt']['bytes']},
                     ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=EXPERIMENT / 'main')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='epoch2-preview')
    args = parser.parse_args()
    export(args.run, args.output, args.label)


if __name__ == '__main__':
    main()
