"""One-shot compact status for the serialized experiments."""
from pathlib import Path
from common import WORK,MODELS,load

if __name__=='__main__':
    active=load(WORK/'active-job.json')
    print('ACTIVE', {k:v for k,v in active.items() if k in ['key','pid','finished']})
    for key in MODELS:
        p=WORK/'probes'/key/'result.json'
        if p.exists():
            r=load(p);print('PROBE',key,round(r['test']['jer']*100,2))
    for f in ['cpu-finished.json','gpu-probes-finished.json','gpu-players-finished.json','parallel-heads-finished.json']+[f'phase{i}-finished.json' for i in range(2,8)]:
        print(f,(WORK/f).exists())
    for name in ['encoder-phase2-resume-0914.log','encoder-parallel-heads-0914.log','encoder-phase3-resume-0914.log','encoder-phase4-0914.log','encoder-phase5-0914.log','encoder-phase6-0914.log']:
        p=WORK.parent/name
        if p.exists():
            data=p.read_bytes();encoding='utf16' if data.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf8'
            print(name,data.decode(encoding,errors='replace').splitlines()[-3:])
    for key in MODELS:
        p=WORK/'probes-normalized'/key/'result.json'
        if p.exists():
            r=load(p);print('SCALAR_PROBE',key,round(r['test']['jer']*100,2))
    for p in sorted((WORK/'finetunes').glob('*/seed*/progress.json')):
        r=load(p);print('TRAIN',p.parent.parent.name,p.parent.name,r['history'][-1]['step'],r['best'])
