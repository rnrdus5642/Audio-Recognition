"""Representative import and standalone checks in Unity 2022 / Sentis 1.2."""
from pathlib import Path
import argparse
import time
import traceback
from common import WORK, dump, load
from deploy import command

KEYS=['hubertko','omni300','w2vbert','wavlm','xlsr300']
UNITY=Path('C:/Program Files/Unity/Hub/Editor/2022.3.62f2/Editor/Unity.exe')


def convert(key):
    out=WORK/'exports'/key;target=out/'model-sentis1.sentis'
    record=Path(str(target)+'.import.json')
    if record.exists() and load(record).get('success') and target.exists():return load(record)
    rc=command([UNITY,'-batchmode','-quit','-projectPath',WORK/'unity2022','-force-d3d11',
                '-executeMethod','EncoderScreenEditor.Convert','-encoderOnnx',out/'model.onnx',
                '-encoderSentis',target,'-logFile',WORK/f'unity2022-import-{key}.log'],
               WORK/f'unity2022-launch-{key}.log')
    return {**(load(record) if record.exists() else {'success':False,'error':'No import record'}),'exit_code':rc}


def imports():
    results=[]
    for key in KEYS:
        out=WORK/'exports'/key
        while not (out/'export.json').exists():
            if (WORK/'cpu-finished.json').exists(): break
            time.sleep(10)
        target=out/'model-sentis1.sentis'
        record=Path(str(target)+'.import.json')
        try:
            if not (out/'export.json').exists():raise RuntimeError('No completed ONNX export')
            print('LEGACY_IMPORT',key,flush=True)
            rc=command([UNITY,'-batchmode','-quit','-projectPath',WORK/'unity2022','-force-d3d11',
                        '-executeMethod','EncoderScreenEditor.Convert','-encoderOnnx',out/'model.onnx',
                        '-encoderSentis',target,'-logFile',WORK/f'unity2022-import-{key}.log'],
                       WORK/f'unity2022-launch-{key}.log')
            result={'key':key,'exit_code':rc,'result':load(record) if record.exists() else None}
        except Exception:
            result={'key':key,'error':traceback.format_exc()}
        results.append(result); dump(WORK/'legacy-import-results.json',results)
        print('LEGACY_IMPORT_RESULT',key,result.get('exit_code'),flush=True)
    dump(WORK/'legacy-import-finished.json',{'finished':True})


def player(key,suffix=''):
    out=WORK/'exports'/key
    job=load(out/'job.json');job['model']=str((out/'model-sentis1.sentis').resolve())
    job['output']=str((out/f'unity2022-player{suffix}.json').resolve())
    path=out/f'job2022{suffix}.json';dump(path,job)
    rc=command([WORK/'player2022/EncoderScreen.exe','-batchmode','-force-d3d11','-encoderJob',path,
                '-logFile',out/f'unity2022-player{suffix}.log'],out/f'unity2022-player-launch{suffix}.log',timeout=600)
    return {'key':key,'exit_code':rc,'result':load(job['output']) if Path(job['output']).exists() else None}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['imports','player']);p.add_argument('key',nargs='?')
    args=p.parse_args()
    if args.action=='imports':imports()
    else:print(player(args.key,suffix='-functional-concurrent'))
