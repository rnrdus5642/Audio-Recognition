"""Korean 1B adaptation and actual Unity checks of the selected trained pilots."""
from pathlib import Path
import sys
import time
import traceback
from common import WORK,dump,load
from deploy import command,convert,player

if __name__=='__main__':
    while not (WORK/'phase3-finished.json').exists():time.sleep(10)
    print('KOREAN_1B_ADAPTATION',flush=True)
    rc=command([sys.executable,'-X','utf8','-u',Path(__file__).with_name('finetune.py'),'hubertko1b',
                '--cpu-offload-layers','24'],WORK/'logs/finetune-hubertko1b.log',timeout=7200)
    dump(WORK/'finetunes/hubertko1b/job-status.json',{'key':'hubertko1b','exit_code':rc})
    print('KOREAN_1B_RESULT',rc,flush=True)
    scores=[]
    for key in ['hubertko','xlsr300','omni300','w2vbert','wavlm','hubertko1b']:
        p=WORK/'finetunes'/key/'seed917/result.json'
        if p.exists():scores.append((load(p)['selected']['valid']['jer'],key))
    selected=sorted(scores)[:2]
    dump(WORK/'trained-deployment-selection.json',{'rule':'Two smallest valid JER scores, seed917 pilots','selected':selected,'all_scores':scores})
    results=[]
    for _,key in selected:
        variant=f'{key}-pilot917'
        try:
            print('TRAINED_EXPORT',key,flush=True)
            rc=command([sys.executable,'-X','utf8','-u',Path(__file__).with_name('export.py'),key,'--finetune-seed','917'],
                        WORK/'logs'/f'export-{variant}.log')
            if rc:raise RuntimeError(f'Export exited {rc}')
            imported=convert(variant)
            if not imported.get('success'):raise RuntimeError(str(imported))
            runs=[player(variant,suffix=f'-repeat{repeat}') for repeat in range(3)]
            result={'key':key,'variant':variant,'import':imported,'runs':runs}
        except Exception:result={'key':key,'error':traceback.format_exc()}
        results.append(result);dump(WORK/'trained-deployment-results.json',results)
        print('TRAINED_DEPLOYMENT_RESULT',key,'error' not in result,flush=True)
    dump(WORK/'phase4-finished.json',{'finished':True})
