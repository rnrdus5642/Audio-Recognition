"""Complete the standardized frozen comparison for every candidate."""
from pathlib import Path
import sys
import time
from common import WORK,MODELS,dump
from deploy import command

if __name__=='__main__':
    while not (WORK/'phase4-finished.json').exists():time.sleep(10)
    results=[]
    for key in MODELS:
        if not (WORK/'features'/key/'metadata.json').exists():continue
        if (WORK/'probes-normalized'/key/'result.json').exists():continue
        print('NORMALIZED_REMAINING',key,flush=True)
        rc=command([sys.executable,'-X','utf8','-u',Path(__file__).with_name('normalized_probe.py'),key],
                    WORK/'logs'/f'normalized-{key}.log',timeout=1800)
        results.append({'key':key,'exit_code':rc});dump(WORK/'normalized-remaining-results.json',results)
        print('NORMALIZED_REMAINING_RESULT',key,rc,flush=True)
    dump(WORK/'phase5-finished.json',{'finished':True})
