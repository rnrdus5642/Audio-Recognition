"""Merge the preparation artifacts into the single preflight report.

Everything here is read back from files the pipeline wrote, so the report
cannot drift from what actually ran.

    python python/tools/child_jamo_finetune/report.py
"""
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels
from common import ENCODER, WORK, digest, dump, load

CHOSEN = {'offload_layers': 26, 'batch_budget': 640000, 'accumulate': 8,
          'max_batch': 32, 'eval_budget': 1280000}


def main():
    report = load(WORK / 'claude-preflight-report.json')
    selection = load(WORK / 'selection-summary.json')
    audit = load(WORK / 'audit-summary.json')
    catalog = load(WORK / 'catalog-summary.json')

    report['dataset'] = {
        'source_index_sha256': selection['index_sha256'],
        'source_splits_sha256': selection['splits_sha256'],
        'previous_dev_csv_sha256': selection['old_dev_sha256'],
        'selection_seed': selection['seed'],
        'bounds_seconds': selection['bounds_seconds'],
        'age_allocation': selection['age_allocation'],
        'selection_exclusions': selection['exclusions'],
        'audit_drops': audit['drops'],
        'final': audit['counts'],
        'manifest_sha256': audit['manifest_sha256'],
        'manifests': {s: str(WORK / f'manifest-{s}.jsonl')
                      for s in ['train', 'valid', 'test']},
        'target_expression_coverage': audit['target_expression_coverage'],
    }
    report['labels'] = {
        'version': labels.LABELS_VERSION,
        'contract_sha256': labels.contract_sha256(),
        'outputs': len(labels.TOKENS),
        'tokens': labels.TOKENS,
        'derivation': 'NFC, drop sentence punctuation, require Hangul and '
                      'spaces only, KoreanG2P.apply_rules, decompose to '
                      'compatibility jamo, "|" between words',
        'unsupported_policy': 'transcripts with digits, Latin, parentheses or '
                              'brackets are excluded and counted, never '
                              'silently stripped',
    }
    report['source_integrity'] = {
        'archives': catalog['archives'],
        'wav_members_catalogued': catalog['wav_members'],
        'distinct_basenames': catalog['distinct_basenames'],
        'ambiguous_basenames_excluded': catalog['ambiguous_basename_count'],
        'archives_modified': False,
        'audio_extracted_to_disk': False,
        'method': 'byte-offset references into the unmodified plain tars; '
                  'payloads are read on demand and hashed during the audit',
    }
    report['encoder'] = {
        **load(ENCODER / 'verified.json'),
        'local_safetensors_sha256': digest(ENCODER / 'model.safetensors'),
        'head': 'fresh 42-output linear CTC head; nothing carried over from '
                'the 400-update pilot',
    }
    report['recommended_config'] = CHOSEN
    free = shutil.disk_usage(WORK.anchor).free
    report['disk'] = {
        'free_bytes': free, 'free_gib': free / 2 ** 30,
        'resumable_checkpoint_bytes': report['resume']['checkpoint_bytes'],
        'peak_checkpoint_bytes': 2 * report['resume']['checkpoint_bytes']
                                 + 2 * 3850521519,
        'note': 'one last.pt and one best.pt are kept; the peak includes the '
                'temporary file each atomic write creates',
    }
    report['smoke_test'] = load(WORK / 'smoke-progress-final.json')
    report['long_run'] = {
        'started': False,
        'reason': 'held for main-agent review, as instructed',
        'command': 'powershell -File python\\tools\\child_jamo_finetune\\'
                   'launch.ps1',
        'resume': 'powershell -File python\\tools\\child_jamo_finetune\\'
                  'launch.ps1 -Resume',
        'progress': str(WORK / 'main/progress.json'),
        'log': str(WORK / 'run/main.log'),
    }
    report['limitations'] = [
        'Targets are transcript-derived pronunciation forms produced by the '
        'project rules. They are not annotations of what a child actually '
        'pronounced; the corpus ErrorTagged fields are N/A.',
        'JER therefore measures recovery of the intended pronunciation form. '
        'It is not a phoneme or IPA error rate and does not bound the '
        "product's false-accept rate.",
        'The valid and test speakers are unseen by this run, but they were '
        'used to evaluate a previous model, so they are not pristine '
        'never-touched subjects.',
        'Unity compatibility for this encoder was established under Unity 6 '
        'GPUCompute with Inference Engine 2.6.1. Legacy Sentis 1.2 did not '
        'pass for XLarge and no claim is made there.',
        'The preflight measures memory, throughput and correctness of the '
        'training path. It says nothing about convergence or final accuracy.',
        'Four target expressions (실로폰채, 북채, 내려놔, 연주해) are absent or '
        'near-absent from the corpus, so this data cannot teach them.',
    ]
    dump(WORK / 'claude-preflight-report.json', report)
    print('report written:', WORK / 'claude-preflight-report.json')


if __name__ == '__main__':
    main()
