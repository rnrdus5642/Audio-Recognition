"""Read-only metadata/energy audit; thresholds do not establish speech boundaries."""
from collections import Counter, defaultdict
import gzip
import json
import math
import random
import wave

import numpy as np

from common import ROOT, PREP, WORK, read, dump


def main():
    catalogs=read(PREP/'dataset-plan.json')['active_catalogs']
    stats={}; samples=[]; rng=random.Random(917)
    for name,directory in catalogs.items():
        count=Counter(); selected=[]
        with gzip.open(PREP/directory/'train.jsonl.gz','rt',encoding='utf8') as stream:
            for i,line in enumerate(stream):
                row=json.loads(line); count['records']+=1
                count['metadata_seconds']+=row['metadata_seconds']
                if len(selected)<512:selected.append(row)
                else:
                    k=rng.randrange(i+1)
                    if k<512:selected[k]=row
                try:
                    a=float(row['speech_start_metadata']); b=float(row['speech_end_metadata'])
                    n=float(row['metadata_seconds'])
                    if not (math.isfinite(a) and math.isfinite(b) and 0<=a<b<=n+.01):
                        raise ValueError()
                    cut=max(0.,a-.25)+max(0.,n-b-.25)
                    count['valid_spans']+=1
                    count['potential_removed_seconds_250ms_padding']+=cut
                    count['records_shortened_250ms_padding']+=int(cut>0)
                except (KeyError,TypeError,ValueError):
                    count['missing_or_invalid_spans']+=1
        stats[name]=dict(count); samples+=selected
        print(json.dumps({name:stats[name]},ensure_ascii=False),flush=True)
    energy=defaultdict(list); annotations=[]
    for row in samples:
        with wave.open(str(ROOT/row['audio_path']),'rb') as wav:
            x=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float64)/32768
        n=len(x)//320
        if not n:continue
        rms=np.sqrt(np.mean(x[:n*320].reshape(n,320)**2,axis=1))
        for db in (-70,-60,-50):
            active=np.flatnonzero(rms>10**(db/20))
            cut=0. if not len(active) else float(max(0.,active[0]*.02-.25)+max(0.,len(x)/16000-(active[-1]+1)*.02-.25))
            energy[f'{row["dataset_id"]}/{db}'].append((cut,len(x)/16000))
        if row['dataset_id']==540:
            a=float(row['speech_start_metadata']); b=float(row['speech_end_metadata'])
            lo=max(0,int((a-.25)*16000)); hi=min(len(x),int((b+.25)*16000))
            removed=np.concatenate((x[:lo],x[hi:]))
            annotations.append({'record_id':row['record_id'],'removed_seconds':len(removed)/16000,
                                'removed_peak':float(np.max(np.abs(removed))) if len(removed) else 0.})
    report={'status':'audit_only_no_audio_modified','padding_seconds':.25,'catalog_train_stats':stats,
        'sample_count_per_dataset':512,'energy_thresholds_are_not_speech_labels':True,
        'energy_sample':{k:{'potential_removed_seconds':float(sum(v[0] for v in vs)),
                            'sample_audio_seconds':sum(v[1] for v in vs),
                            'records_shortened':int(sum(v[0]>0 for v in vs)),
                            'max_removed_seconds':float(max(v[0] for v in vs))} for k,vs in energy.items()},
        'annotation_sample_540':annotations,
        'limitations':['Metadata durations precede 25 long-file training exclusions.',
                       'Low energy is not proof of silence; annotations require boundary checks.',
                       'No VAD or forced alignment performed; labels and original WAVs are unchanged.']}
    dump(WORK/'silence-audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='annotation_sample_540'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
