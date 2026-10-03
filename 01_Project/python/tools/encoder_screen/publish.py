"""Generate reviewable aggregate tables; never publish audio or clip metadata."""
from pathlib import Path
import argparse
import copy
from common import ROOT,WORK,MODELS,load,dump,digest
from collect import collect,maybe

NAMES={'xlsr300':'XLS-R 300M','omni300':'OmniASR W2V 300M','w2vbert':'W2v-BERT 2.0',
       'hubertko':'한국어 HuBERT Large','wavlm':'WavLM Large','huberten':'영어 HuBERT Large',
       'xlsr1b':'XLS-R 1B','omni1b':'OmniASR W2V 1B','hubertko1b':'한국어 HuBERT XLarge',
       'mms1b':'MMS 1B','xeus':'XEUS'}


def player_summary(r):
    if not r:return None
    keep=['key','unity','in_editor','platform','graphics_device','graphics_api','backend','actual_backend',
          'success','utc','cpu_fallback_layers','total_model_layers','error','model_load_ms','worker_create_ms',
          'cold_inference_ms','warm_median_ms','warm_p95_ms','warm_times_ms','native_memory_before',
          'native_memory_after','all_finite','all_ctc_sequences_equal','mismatch_frames','total_frames','within_500ms_hop']
    out={k:r[k] for k in keep if k in r}
    cases=r.get('cases',[])
    if cases:
        out.update(cases=len(cases),ctc_equal_cases=sum(x.get('decoded_equal',False) for x in cases),
                   max_abs=max(x['max_abs'] for x in cases),worst_case_rmse=max(x['rmse'] for x in cases),
                   within_original_0_02_absolute_tolerance=all(x['max_abs']<.02 for x in cases))
    return out


def evidence(path):
    return {'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':digest(path)}


def run(draft=False):
    full=collect()
    if not draft:
        assert all(full['completed_phases'].values()),full['completed_phases']
        assert all(r['selected_probe']['all_four_conditions_complete'] for r in full['candidates'].values())
    audit={k:copy.deepcopy(full[k]) for k in ['date','environment','contract','manifest_sha256',
            'train_jamo_counts','notes','label_audit','normalization_decision','completed_phases','second_seed_selection']}
    audit['draft']=draft;audit['candidates']={};audit['evidence']=[]
    audit['trained_pairwise']=full['trained_pairwise']
    for key,r in full['candidates'].items():
        keep=['spec','download','selected_probe','finetunes']
        result={k:copy.deepcopy(r[k]) for k in keep}
        result['name']=NAMES[key]
        result['parameters']=r['features']['parameters']
        for mode in ['frozen_probe','normalized_probe']:
            if r.get(mode):
                result[mode]={k:r[mode][k] for k in ['method','selected','test','head_sha256'] if k in r[mode]}
        result['unity6']=player_summary(r['unity6_player'])
        result['unity2022']=player_summary(r['unity2022_player'])
        if r.get('export'):
            e=r['export'];checks=e['torch_onnx_parity']
            result['onnx']={k:e[k] for k in ['opset','waveform_samples','preprocessing_embedded',
                          'external_weight_bytes','weight_tensor_count','onnx_sha256'] if k in e}
            result['onnx']['all_ort_absolute_tolerance_passed']=all(x['max_abs']<.02 for x in checks)
            result['onnx']['all_ort_ctc_equal']=e.get('all_ort_ctc_equal',all(x.get('ctc_equal',x['equal_frames']==x['frames']) for x in checks))
            result['onnx'].update(max_abs=max(x['max_abs'] for x in checks),
                                  max_relative_l2=max((x['relative_l2'] for x in checks if 'relative_l2' in x),default=None))
        for name in ['conversion-verification.json','adapter-verification.json']:
            f=WORK/'models'/key/name
            if f.exists():result[name]=load(f)
        audit['candidates'][key]=result
        for f in [WORK/'probes'/key/'result.json',WORK/'probes-normalized'/key/'result.json',
                  WORK/'exports'/key/'export.json',WORK/'exports'/key/'unity-player.json']:
            if f.exists():audit['evidence'].append(evidence(f))
        for f in (WORK/'finetunes'/key).glob('seed*/result.json'):
            audit['evidence'].append(evidence(f))
    audit['trained_deployment']=[]
    for r in full.get('trained_deployment') or []:
        value={k:r[k] for k in ['key','variant','error'] if k in r}
        value['runs']=[player_summary(x.get('result')) for x in r.get('runs',[])]
        if r.get('variant'):
            d=WORK/'exports'/r['variant'];export=maybe(d/'export.json')
            if export:
                value['checkpoint_sha256']=export.get('deployed_checkpoint_sha256',export['finetune']['checkpoint_sha256'])
                value['onnx_sha256']=export['onnx_sha256']
                value['all_ort_absolute_tolerance_passed']=export['all_ort_absolute_tolerance_passed']
                value['python_wrapper_all_ctc_equal']=export.get('python_wrapper_all_ctc_equal')
                value['python_wrapper_max_abs']=export.get('python_wrapper_max_abs')
            for f in d.glob('unity-player-repeat*.json'):audit['evidence'].append(evidence(f))
        audit['trained_deployment'].append(value)
    audit['final_compatibility']=[]
    for r in full.get('final_compatibility') or []:
        value={k:r[k] for k in ['key','version','error'] if k in r}
        value['player']=player_summary(r.get('player',{}).get('result'))
        audit['final_compatibility'].append(value)
    audit['deployed_words']={}
    for key in MODELS:
        path=WORK/'exports'/(key+'-pilot917')/'owned-windows/result.json'
        if path.exists():
            result=load(path)
            audit['deployed_words'][key]={k:v for k,v in result.items() if k!='records'}
            audit['evidence'].append(evidence(path))
    audit['harness']=[evidence(f) for f in sorted(Path(__file__).parent.rglob('*'))
                      if f.is_file() and f.suffix in ['.py','.txt','.md','.json']]
    audit['label_implementation']=[evidence(ROOT/p) for p in ['python/build/g2p/ko/g2p.py','python/build/g2p/ko/jamo_ipa.py']]
    audit['adamw_offload_verification']=maybe(WORK/'adamw-offload-verification.json')
    audit['bert_diagnostics']={}
    native=maybe(WORK/'exports/w2vbert/native-reference-parity.json')
    if native:audit['bert_diagnostics']['original_preprocessing_reference']={k:v for k,v in native.items() if k!='cases'}
    for name in ['w2vbert-fbank-debug','w2vbert-shape-folded']:
        path=WORK/'exports'/name
        audit['bert_diagnostics'][name]={'unity6':player_summary(maybe(path/'unity-player.json')),
                                        'unity2022':player_summary(maybe(path/'unity2022-player.json'))}
        if (path/'folding.json').exists():
            folding=load(path/'folding.json')
            audit['bert_diagnostics'][name]['folding']={k:v for k,v in folding.items() if k!='ort_vs_pytorch'}
        for f in path.glob('*.json'):
            if f.name not in ['job.json'] and not f.name.startswith('job-'):
                audit['evidence'].append(evidence(f))
    folder=WORK/'report-draft' if draft else ROOT/'docs/data'
    folder.mkdir(parents=True,exist_ok=True)
    dump(folder/'pretrained-encoder-windows-unity-2026-09-14.json',audit)
    lines=[]
    def add(text=''):lines.append(text)
    def percent(x):return f'{x*100:.2f}%'
    def exact(r):return f"{r['exact']}/{r['recordings']}"
    add('| 모델 | 인코더 규모 | 고정 본체 JER | 400 update 후 JER | Unity 6 p95 | GPU local 사용량 | 원본 CTC 일치 |')
    add('|---|---:|---:|---:|---:|---:|---|')
    for key,r in audit['candidates'].items():
        probe=r['selected_probe'];fine=r['finetunes'].get('seed917');gpu=r['unity6']
        n=gpu.get('native_memory_after',{}).get('gpu_local_usage_bytes',0)/2**30
        agree=f"{gpu.get('ctc_equal_cases',0)}/{gpu.get('cases',0)}"
        add(f"| {NAMES[key]} | {r['parameters']/1e6:.1f}M | {percent(probe['test']['jer'])} | {percent(fine['test']['jer']) if fine else '미실시'} | {gpu.get('warm_p95_ms',0):.2f}ms | {n:.2f}GiB | {agree} |")
    add();add('추가 학습의 검증 점수와 학습 자원:');add()
    add('| 모델 | 선택 step | valid JER | test JER | 본체 학습 전 대비 test 변화 | PyTorch 최대 GPU 할당 | 실행 시간* |')
    add('|---|---:|---:|---:|---:|---:|---:|')
    for key,r in audit['candidates'].items():
        for seed,f in r['finetunes'].items():
            d=f['test']['jer']-r['selected_probe']['test']['jer']
            add(f"| {NAMES[key]} ({seed}) | {f['selected']['step']} | {percent(f['selected']['valid']['jer'])} | {percent(f['test']['jer'])} | {d*100:+.2f}%p | {f['cuda_peak_allocated_bytes']/2**30:.2f}GiB | {f['elapsed_seconds']/60:.1f}분 |")
    add();add('* 검증·파일 저장·추가 단어 평가와 일부 병행한 출력층/기능 검사의 영향을 포함한 경과 시간이다. 순수 학습 처리량 비교에 쓰지 않는다.');add()
    add('보유 녹음의 엄격한 자모열 완전 일치(53개 전체):');add()
    add('| 모델 | 원본 | 앞뒤 무음 각각 1초 | 백색잡음 20dB | 백색잡음 10dB |')
    add('|---|---:|---:|---:|---:|')
    for key,r in audit['candidates'].items():
        f=r['finetunes'].get('seed917')
        if not f:continue
        groups={x['condition']:x for x in f['domain_final']['combined']}
        add('| '+NAMES[key]+' | '+' | '.join(exact(groups[c]) for c in ['clean','pad_1s','noise_20db','noise_10db'])+' |')
    selection=maybe(WORK/'trained-deployment-selection.json')
    if selection:
        keys=[key for _,key in selection['selected']]
        add();add('최종 비교 후보의 단어별 원본 완전 일치:');add()
        add('| 단어 | '+' | '.join(NAMES[k] for k in keys)+' |')
        add('|---|'+'---:|'*len(keys))
        order=['실로폰채','들어','잡아','쳐','때려','연주해','놓아','놔','내려놔','흔들어','북채','두드려','작아']
        for word in order:
            values=[]
            for key in keys:
                rows=audit['candidates'][key]['finetunes']['seed917']['domain_final']['by_word']
                row=next(r for r in rows if r['condition']=='clean' and r['word']==word)
                values.append(exact(row))
            add('| '+word+' | '+' | '.join(values)+' |')
        add();add('`작아` 두 개는 유사어 구별용이다. 이 표는 발음 오류 진단이나 서비스의 정답 통과율을 나타내지 않는다.')
    if audit['trained_deployment']:
        add();add('학습한 가중치를 다시 내보낸 Unity 6 반복 검사:');add()
        add('| 모델 | 반복 | p95 | CTC 일치 입력 | 최대 절대 오차 | GPU local |')
        add('|---|---:|---:|---:|---:|---:|')
        for item in audit['trained_deployment']:
            for i,r in enumerate(item['runs'],1):
                if not r:continue
                memory=r.get('native_memory_after',{}).get('gpu_local_usage_bytes',0)
                add(f"| {NAMES[item['key']]} | {i} | {r.get('warm_p95_ms',0):.2f}ms | {r.get('ctc_equal_cases',0)}/{r.get('cases',0)} | {r.get('max_abs',0):.6f} | {memory/2**30:.2f}GiB |")
    trained_legacy=[r for r in audit['final_compatibility']
                    if r.get('key','').endswith('-pilot917') and r.get('version')=='unity2022']
    if trained_legacy:
        add();add('학습한 가중치의 Unity 2022 / Sentis 1.2 검사:');add()
        add('| 모델 | 판정 | CTC 일치 입력 | 불일치 frame | 최대 절대 오차 | p95 |')
        add('|---|---|---:|---:|---:|---:|')
        for item in trained_legacy:
            key=item['key'].removesuffix('-pilot917');r=item['player']
            verdict='통과' if r.get('success') else '출력 대조 실패'
            add(f"| {NAMES[key]} | {verdict} | {r.get('ctc_equal_cases',0)}/{r.get('cases',0)} | {r.get('mismatch_frames',0)}/{r.get('total_frames',0)} | {r.get('max_abs',0):.6f} | {r.get('warm_p95_ms',0):.2f}ms |")
        add();add('출력 대조에 실패한 실행의 지연을 유효한 인식 성능으로 해석하지 않는다.')
    if audit['deployed_words']:
        add();add('2.5초 고정 입력에서 보유 단어를 실제 Unity GPU로 검사한 결과:');add()
        add('| 모델 | 발화 위치 | Unity 완전 일치 | Unity JER | Python/Unity CTC 일치 |')
        add('|---|---|---:|---:|---:|')
        for key,result in audit['deployed_words'].items():
            for row in result['overall']:
                position='가운데' if row['position']=='center' else '끝쪽, 최대 0.1초 여백'
                add(f"| {NAMES[key]} | {position} | {row['unity_exact']}/{row['recordings']} | {percent(row['unity_jer'])} | {result['all_ctc_equal']} ({result['received_results']}개 입력) |")
    add();add('Unity 2022 / Sentis 1.2 원본 그래프 검사:');add()
    add('| 모델 | CTC 일치 | p95 | 기록 |');add('|---|---:|---:|---|')
    for key,r in audit['candidates'].items():
        p=r['unity2022']
        if not p:continue
        if p.get('error'):
            add(f"| {NAMES[key]} | 실행 실패 | — | {p['error'].splitlines()[0]} |")
        else:
            add(f"| {NAMES[key]} | {p.get('ctc_equal_cases',0)}/{p.get('cases',0)} | {p.get('warm_p95_ms',0):.2f}ms | CPU fallback {len(p.get('cpu_fallback_layers',[]))}개 |")
    add();add('출력 크기 보정에 대한 민감도:');add()
    add('| 모델 | 원래 출력 test JER | 단일 스칼라 test JER | valid로 선택한 조건 |')
    add('|---|---:|---:|---|')
    for key,r in audit['candidates'].items():
        norm=r.get('normalized_probe')
        add(f"| {NAMES[key]} | {percent(r['frozen_probe']['test']['jer'])} | {percent(norm['test']['jer']) if norm else '진행 중'} | {r['selected_probe']['kind']} |")
    add();add('학습 완료 후보와 XLS-R 300M의 test JER 차이:');add()
    add('| 모델 | 차이 | 탐색적 화자 bootstrap 95% 구간 |')
    add('|---|---:|---:|')
    for key,r in audit['candidates'].items():
        f=r['finetunes'].get('seed917')
        if not f or 'vs_xlsr300_seed917' not in f:continue
        b=f['vs_xlsr300_seed917'];lo,hi=b['percentile_95']
        add(f"| {NAMES[key]} | {b['candidate_minus_baseline_jer']*100:+.2f}%p | [{lo*100:+.2f}, {hi*100:+.2f}]%p |")
    add();add('이 구간은 같은 149명의 평가 화자를 복원 추출한 탐색적 결과다. 다중 모델 선택·사전학습 자료 중복·학습 seed 전체의 불확실성을 포함하지 않는다.')
    if audit['trained_pairwise']:
        add();add('한국어 HuBERT Large와의 추가 비교:');add()
        add('| 모델 | test JER 차이 | 탐색적 95% 구간 |');add('|---|---:|---:|')
        for pair,b in audit['trained_pairwise'].items():
            key=pair.split('_minus_')[0];lo,hi=b['percentile_95']
            add(f"| {NAMES[key]} | {b['candidate_minus_baseline_jer']*100:+.2f}%p | [{lo*100:+.2f}, {hi*100:+.2f}]%p |")
    (folder/'pretrained-encoder-tables-2026-09-14.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('PUBLISHED',str(folder),'draft',draft,flush=True)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--draft',action='store_true');args=p.parse_args()
    run(args.draft)
