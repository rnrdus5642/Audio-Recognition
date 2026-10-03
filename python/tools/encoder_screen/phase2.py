"""GPU work queued after the original controller, avoiding benchmark contention."""
from pathlib import Path
import sys
import time
import traceback
from common import WORK, dump, load
from deploy import command, player


if __name__=='__main__':
    while not load(WORK/'active-job.json').get('finished'):
        time.sleep(10)
    for result in load(WORK/'runner-results.json'):
        if result['exit_code'] and result['key'] != 'xlsr300':
            dump(WORK/'failures'/f"{result['key']}-final.json",result)
    for key in ([] if (WORK/'gpu-probes-finished.json').exists() else ['xlsr300','xeus']):
        print('LATE_PROBE',key,flush=True)
        rc = command([sys.executable,'-X','utf8','-u',Path(__file__).with_name('probe.py'),key],
                     WORK/'logs'/f'{key}-retry.log',timeout=3600)
        if rc:
            dump(WORK/'failures'/f'{key}-final.json',{'key':key,'exit_code':rc})
        print('LATE_PROBE_RESULT',key,rc,flush=True)
    dump(WORK/'gpu-probes-finished.json',{'finished':True})
    while not (WORK/'cpu-finished.json').exists(): time.sleep(10)
    completed_players=(WORK/'gpu-players-finished.json').exists()
    results=load(WORK/'player-results.json') if completed_players else []
    for item in ([] if completed_players else load(WORK/'deploy-results.json')):
        key=item['key']
        if not item.get('import',{}).get('success'): continue
        print('UNITY_PLAYER',key,flush=True)
        try:
            result=player(key)
        except Exception:
            result={'key':key,'error':traceback.format_exc()}
        results.append(result);dump(WORK/'player-results.json',results)
        print('UNITY_PLAYER_RESULT',key,result.get('exit_code'),flush=True)
    dump(WORK/'gpu-players-finished.json',{'finished':True})
    for key in ['hubertko','xlsr300','omni300','w2vbert']:
        print('PILOT_TRAIN',key,flush=True)
        rc=command([sys.executable,'-X','utf8','-u',Path(__file__).with_name('finetune.py'),key],
                   WORK/'logs'/f'finetune-{key}.log',timeout=7200)
        dump(WORK/'finetunes'/key/'job-status.json',{'key':key,'exit_code':rc})
        print('PILOT_TRAIN_RESULT',key,rc,flush=True)
    dump(WORK/'phase2-finished.json',{'finished':True})
