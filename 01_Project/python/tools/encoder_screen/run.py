"""Serialize GPU experiments, preserve logs and every failure, resume completed jobs."""
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, WORK, MODELS, dump, load

if __name__=='__main__':
    while not (WORK/'manifest.json').exists():
        print('WAIT_DATA',flush=True);time.sleep(10)
    selected = sys.argv[1:] or [k for k,v in MODELS.items() if v['family']!='xeus']
    logs=WORK/'logs';logs.mkdir(exist_ok=True)
    results=[]
    for key in selected:
        while not (WORK/'models'/key/'verified.json').exists():
            print('WAIT_MODEL',key,flush=True);time.sleep(10)
        print('START',key,flush=True)
        with (logs/(key+'.log')).open('w',encoding='utf-8') as log:
            cmd=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('probe.py')),key]
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            dump(WORK/'active-job.json',{'key':key,'pid':p.pid,'started_unix':time.time(),'command':cmd})
            rc=p.wait()
        rec={'key':key,'exit_code':rc,'log':str(logs/(key+'.log'))}
        if (WORK/'probes'/key/'result.json').exists():rec['result']=load(WORK/'probes'/key/'result.json')
        results.append(rec);dump(WORK/'runner-results.json',results)
        print('FINISHED',key,'exit',rc,flush=True)
    dump(WORK/'active-job.json',{'finished':True,'jobs':len(results)})
    print('ALL_PROBES_FINISHED',len(results),flush=True)
