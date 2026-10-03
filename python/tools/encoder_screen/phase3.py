"""Finish legacy player checks, WavLM adaptation, and a second training seed."""
from pathlib import Path
import sys
import time
import traceback
from common import WORK,dump,load
from deploy import command
from legacy import player as legacy_player


if __name__=='__main__':
    while not (WORK/'phase2-finished.json').exists():time.sleep(10)
    while (WORK/'parallel-heads-active.json').exists() and not (WORK/'parallel-heads-finished.json').exists():time.sleep(10)
    while not (WORK/'legacy-import-finished.json').exists():time.sleep(10)
    results=[]
    for item in load(WORK/'legacy-import-results.json'):
        if not item.get('result',{}).get('success'):continue
        key=item['key'];print('LEGACY_PLAYER',key,flush=True)
        try: result=legacy_player(key)
        except Exception:result={'key':key,'error':traceback.format_exc()}
        results.append(result);dump(WORK/'legacy-player-results.json',results)
        print('LEGACY_PLAYER_RESULT',key,result.get('exit_code'),flush=True)
    print('PILOT_TRAIN wavlm',flush=True)
    rc=command([sys.executable,'-X','utf8','-u',Path(__file__).with_name('finetune.py'),'wavlm'],
               WORK/'logs/finetune-wavlm.log',timeout=7200)
    dump(WORK/'finetunes/wavlm/job-status.json',{'key':'wavlm','exit_code':rc})
    print('PILOT_TRAIN_RESULT wavlm',rc,flush=True)
    scores=[]
    for key in ['hubertko','xlsr300','omni300','w2vbert','wavlm']:
        path=WORK/'finetunes'/key/'seed917/result.json'
        if path.exists(): scores.append((load(path)['selected']['valid']['jer'],key))
    if scores:
        _,key=min(scores)
        dump(WORK/'second-seed-selection.json',{'key':key,'rule':'Smallest valid JER among completed seed917 pilots, not test or owned words',
                                               'valid_scores':scores,'seed':918})
        print('SECOND_SEED',key,flush=True)
        rc=command([sys.executable,'-X','utf8','-u',Path(__file__).with_name('finetune.py'),key,'--seed','918'],
                   WORK/'logs'/f'finetune-{key}-seed918.log',timeout=7200)
        dump(WORK/'second-seed-status.json',{'key':key,'exit_code':rc})
        print('SECOND_SEED_RESULT',key,rc,flush=True)
    dump(WORK/'phase3-finished.json',{'finished':True})
