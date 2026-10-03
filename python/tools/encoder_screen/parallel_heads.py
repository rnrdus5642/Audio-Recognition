"""Small linear-head jobs alongside adaptation, with a barrier before timings."""
from pathlib import Path
import sys
from common import WORK,MODELS,load,dump
from deploy import command

if __name__=='__main__':
    dump(WORK/'parallel-heads-active.json',{'note':'Head-only GPU jobs may overlap adaptation; phase3 waits before standalone timings'})
    results=[]
    # Prioritize the next full-adaptation jobs to avoid duplicate work/waiting.
    for key in ['omni300','w2vbert','wavlm','hubertko1b','huberten','xlsr1b','omni1b','mms1b','xeus']:
        if (WORK/'probes-normalized'/key/'result.json').exists():continue
        print('PARALLEL_HEAD',key,flush=True)
        rc=command([sys.executable,'-X','utf8','-u',Path(__file__).with_name('normalized_probe.py'),key],
                    WORK/'logs'/f'normalized-{key}.log',timeout=7200)
        results.append({'key':key,'exit_code':rc});dump(WORK/'parallel-head-results.json',results)
        print('PARALLEL_HEAD_RESULT',key,rc,flush=True)
    dump(WORK/'parallel-heads-finished.json',{'finished':True,'results':results})
