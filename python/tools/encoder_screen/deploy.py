"""Isolated real Unity importer/player automation; never targets a user project."""
from pathlib import Path
import argparse
import subprocess
import sys
import time
import traceback
from common import ROOT, WORK, MODELS, dump, load

UNITY = Path('C:/Program Files/Unity/Hub/Editor/6000.3.18f1/Editor/Unity.exe')
PROJECT = WORK/'unity6'


def command(args, log, timeout=1800):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf8') as f:
        p = subprocess.Popen([str(x) for x in args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        try: return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(); raise


def convert(key):
    out = WORK/'exports'/key
    record = out/'model.sentis.import.json'
    if record.exists() and load(record).get('success') and (out/'model.sentis').exists():
        return load(record)
    engine_log = WORK/f'unity6-import-{key}.log'
    rc = command([UNITY,'-batchmode','-quit','-projectPath',PROJECT,'-force-d3d11',
                  '-executeMethod','EncoderScreenEditor.Convert','-encoderOnnx',out/'model.onnx',
                  '-encoderSentis',out/'model.sentis','-logFile',engine_log], WORK/f'unity6-launch-{key}.log')
    result = load(record) if record.exists() else {'success': False, 'error':'No importer result was produced'}
    return {**result,'exit_code':rc,'log':str(engine_log)}


def player(key, suffix=''):
    out = WORK/'exports'/key
    job = load(out/'job.json')
    output = out/f'unity-player{suffix}.json'
    job['output'] = str(output.resolve())
    job_path = out/f'job-player{suffix}.json'; dump(job_path,job)
    rc = command([WORK/'player6/EncoderScreen.exe','-batchmode','-force-d3d11',
                  '-encoderJob',job_path,'-logFile',out/f'unity-player{suffix}.log'],
                 out/f'unity-player-launch{suffix}.log',timeout=600)
    return {'key':key,'exit_code':rc,'result':load(output) if output.exists() else None}


def cpu_all():
    results = []
    for key in ['omni300','w2vbert','hubertko','wavlm','huberten','xlsr1b','omni1b','hubertko1b','mms1b','xlsr300','xeus']:
        while not (WORK/'probes'/key/'result.json').exists():
            failed = WORK/'failures'/f'{key}-final.json'
            if failed.exists(): break
            time.sleep(10)
        out = WORK/'exports'/key
        try:
            if not (WORK/'probes'/key/'result.json').exists():
                raise RuntimeError('Encoder probe did not finish: '+str(failed))
            print('DEPLOY_EXPORT',key,flush=True)
            if not (out/'export.json').exists():
                rc = command([sys.executable,'-X','utf8','-u',Path(__file__).with_name('export.py'),key],
                             WORK/'logs'/f'export-{key}.log')
                if rc != 0: raise RuntimeError(f'ONNX export exited {rc}; see logs/export-{key}.log')
            print('DEPLOY_IMPORT',key,flush=True)
            result = convert(key)
            results.append({'key':key,'import':result})
            print('DEPLOY_IMPORT_RESULT',key,result['success'],flush=True)
        except Exception:
            result = {'key':key,'error':traceback.format_exc()}
            results.append(result); print('DEPLOY_FAILED',key,result['error'],flush=True)
        dump(WORK/'deploy-results.json',results)
    dump(WORK/'cpu-finished.json',{'finished':True,'results':results})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cpu-all','convert','player']);p.add_argument('key',nargs='?')
    args=p.parse_args()
    if args.action=='cpu-all':cpu_all()
    elif args.action=='convert':print(convert(args.key))
    else:print(player(args.key))
