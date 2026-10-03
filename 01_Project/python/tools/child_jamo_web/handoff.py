"""Assemble the handoff record from the artifacts this task produced.

    python python/tools/child_jamo_web/handoff.py
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import urllib.request

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(WEB.parents[0] / 'child_jamo_finetune'))
from common import ROOT, dump, load

WORK = ROOT / '.venv/child-jamo-web-20260916'


def main():
    # PowerShell's Set-Content -Encoding utf8 writes a BOM.
    pid_file = json.loads((WORK / 'server.pid.json').read_text(
        encoding='utf-8-sig'))
    status = json.load(urllib.request.urlopen(
        pid_file['url'] + 'api/status', timeout=30))
    spread = load(WORK / 'spread-validation.json')
    qa = load(WORK / 'qa-browser.json')

    start_status = (WORK / 'task-start-status.txt').read_text(
        encoding='utf-8', errors='replace').splitlines()
    now_status = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                                capture_output=True, text=True).stdout.splitlines()
    added = sorted(set(now_status) - set(start_status))

    report = {
        'what': '직접 미세조정한 한국어 자모 HuBERT 모델의 인식 결과를 '
                '브라우저에서 듣고 확인하는 로컬 페이지',
        'status': 'RUNNING - 서버가 백그라운드에서 동작 중입니다',
        'url': pid_file['url'],
        'server': {
            'server_pid': pid_file['pid'],
            'instance_token': pid_file['token'],
            'port': pid_file['port'],
            'bind': '127.0.0.1 전용',
            'started': pid_file['started'],
            'log': str(WORK / 'server.log'),
            'pid_file': str(WORK / 'server.pid.json'),
            'restart': 'powershell -File start_child_jamo_web.ps1',
            'stop': 'powershell -File start_child_jamo_web.ps1 -Stop',
        },
        'model': {
            'checkpoint': status['model']['checkpoint'],
            'checkpoint_bytes': status['model']['checkpoint_bytes'],
            'selected_by_validation': status['model']['selected_by_validation'],
            'encoder': status['model']['encoder'],
            'checkpoint_sha256': status['model']['checkpoint_sha256'],
            'encoder_files_sha256': status['model']['encoder_files_sha256'],
            'historical_provenance': status['model']['historical_provenance'],
            'matmul_allow_tf32': status['model']['matmul_allow_tf32'],
            'labels_version': status['model']['labels_version'],
            'labels_contract_sha256': status['model']['labels_contract_sha256'],
            'parameters': status['model']['parameters'],
            'precision': status['model']['precision'],
            'attention': status['model']['attention'],
            'device': status['model']['device'],
            'load_seconds': status['model']['load_seconds'],
            'historical_test': status['model']['historical_test'],
            'historical_note': status['model']['historical_note'],
        },
        'data': {
            'splits': status['splits'],
            'default': 'test (모델이 학습 중 보지 않은 2,999개)',
            'train_note': '15만 개는 처음 선택할 때 뒤에서 읽어 옵니다.',
            'access': '매니페스트가 기록한 archive+offset+size 로 원본 TAR 을 '
                      '읽기 전용으로 읽습니다. 경로를 받는 엔드포인트는 없습니다.',
        },
        'validation': {
            'unit_tests': {
                'files': ['python/tools/child_jamo_web/test_compose.py',
                          'python/tools/child_jamo_web/test_routes.py'],
                'passed': 35,
                'covers': ['CTC 붕괴(반복 자모와 blank)', '자모 -> 한글 합성',
                           'ㅇ 초성 정책', '합쳐지지 않는 자모 보존',
                           '종성 불가 자음', '학습과 같은 오류율 계산',
                           '100% 초과 비클램프', '추론이 정답을 읽지 않음',
                           '미세조정 체크포인트 신원/캐시 키', 'TF32 정책',
                           '감사된 PCM 해시 불일치 거부', '알 수 없는 split/샘플 404',
                           '묶음 범위가 선택 기준', '전부 실패 시 JER=null',
                           '업로드 재생본 = 추론 입력', '44.1kHz 스테레오 처리',
                           '헤더 기반 길이 거부', '업로드 토큰 경로 주입 방지'],
            },
            'spread': {
                'file': str(WORK / 'spread-validation.json'),
                'selection_rule': spread['selection_rule'],
                'recordings': spread['recordings'],
                'selection_jer': spread['selection_jer'],
                'selection_exact': spread['selection_exact'],
                'latency_seconds': spread['latency_seconds'],
                'decoder_agreement_with_training_batched_path':
                    spread['decoder_agreement_all_identical'],
                'note': '나이 x 길이 칸마다 매니페스트 순서대로 고른 고정 표본. '
                        '성공 사례만 고르지 않았습니다. 기록된 최종 테스트 '
                        '점수는 다시 계산하거나 덮어쓰지 않았습니다.',
            },
            'browser': {
                'file': str(WORK / 'qa-browser.json'),
                'browser': qa['browser'],
                'passed': qa['passed'],
                'checks': qa['checks'],
                'screenshots': [str(WORK / 'screenshots' / name)
                                for name in qa['screenshots']],
                'expected_console_errors': qa.get('expected_console_errors'),
                'unexpected_console_errors': qa.get('unexpected_console_errors'),
            },
        },
        'evidence_samples': spread['samples'][:6],
        'files_added': {
            'package': 'python/tools/child_jamo_web/',
            'launcher': 'start_child_jamo_web.ps1',
            'work_dir': str(WORK),
            'git_status_new_entries': added,
        },
        'unchanged': [
            'best.pt / last.pt / result.json / progress.json (읽기만 함)',
            'manifest-{test,valid,train}.jsonl',
            'labels.py 및 g2p (해시 고정, 확인만 함)',
            'python/tools/child_jamo_finetune/ 학습 코드',
            '기존 web_test.py, 제품 매칭/Unity 코드',
            '원본 음성 아카이브 (읽기 전용)',
            '.venv-gpu 학습 환경 (QA 패키지는 별도 가상환경에 설치)',
        ],
        'limitations': [
            '자모 오류율은 전사에서 유도한 예상 발음 형태와 비교한 값입니다. '
            '아이가 실제로 낸 소리에 대한 전문가 표기가 아니므로, 발음 오류 '
            '진단이나 단어 합격 판정으로 읽으면 안 됩니다.',
            '화면 아래 전체 테스트 성적은 고정된 체크포인트의 과거 집계이며, '
            '지금 보고 있는 개별 결과와 다릅니다.',
            '모델이 낼 수 있는 기호는 자모 40개뿐이라, 겹받침처럼 어휘에 없는 '
            '형태는 원리상 나올 수 없습니다.',
            '검증 음성은 체크포인트를 고르는 데 쓰였으므로 테스트만큼 '
            '"본 적 없는" 음성이 아닙니다. 기본값은 테스트입니다.',
            '업로드는 WAV/FLAC/OGG 만 받습니다. ffmpeg 가 필요한 형식은 '
            '지원하지 않습니다.',
        ],
        'at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    dump(WORK / 'claude-web-handoff.json', report)
    print('written', WORK / 'claude-web-handoff.json')
    print('url', report['url'], '| server pid',
          report['server']['server_pid'])
    print('spread JER', round(report['validation']['spread']['selection_jer'], 4),
          '| browser passed', report['validation']['browser']['passed'])


if __name__ == '__main__':
    main()
