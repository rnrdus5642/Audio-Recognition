"""Whole owned words in fixed Unity windows, with no word/threshold forcing."""
from collections import defaultdict
import argparse
import numpy as np
import soundfile as sf
import torch
from common import ROOT,WORK,load,dump,digest
from probe import encoder,decode,edit_distance
from prepare import labels
from export import EndToEnd
from normalized_probe import StandardHead
from deploy import command


@torch.inference_mode()
def run(key):
    torch.set_num_threads(2)
    variant=key+'-pilot917';directory=WORK/'exports'/variant
    out=directory/'owned-windows';out.mkdir(parents=True,exist_ok=True)
    model,feature,_,_=encoder(key,'cpu')
    try:model.set_attn_implementation('eager')
    except ValueError:pass
    weights=torch.load(WORK/'finetunes'/key/'seed917/best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(weights['encoder'],strict=True)
    head=StandardHead(weights['head']['mean'],weights['head']['scale'],len(load(WORK/'vocab.json')))
    head.load_state_dict(weights['head'],strict=True);del weights
    wrapper=EndToEnd(model,feature,key,head).eval()
    source=ROOT/'.venv/adult-model-comparison-0913/inputs.json';clips=load(source)['clips']
    cases=[]
    for n,(identifier,row) in enumerate(clips.items()):
        assert digest(row['path'])==row['sha256']
        wave,sr=sf.read(row['path'],dtype='float32')
        assert sr==16000 and wave.ndim==1 and len(wave)<=40000
        remaining=40000-len(wave)
        starts={'center':remaining//2,'end_margin_up_to100ms':max(0,remaining-1600)}
        for position,left in starts.items():
            audio=np.pad(wave,(left,remaining-left)).astype('<f4')
            case_id=f'{n:03d}-{position}'
            input_file=out/(case_id+'-input.f32');audio.tofile(input_file)
            logits=wrapper(torch.from_numpy(audio).unsqueeze(0)).numpy().astype('<f4')
            assert np.isfinite(logits).all()
            logits_file=out/(case_id+'-reference.f32');logits.tofile(logits_file)
            _,reference=labels(row['label'])
            cases.append({'id':case_id,'input_file':str(input_file.resolve()),'input_shape':[1,40000],
                          'logits_file':str(logits_file.resolve()),'output_shape':list(logits.shape),
                          'word':row['label'],'position':position,'reference_ids':reference,
                          'source_sha256':row['sha256'],'source_samples':len(wave),'left_padding_samples':left})
    job={'key':variant+'-owned-windows','model':str((directory/'model.sentis').resolve()),
         'output':str((out/'unity-player.json').resolve()),'cases':cases,'repeats':4}
    dump(out/'job.json',job)
    rc=command([WORK/'player6/EncoderScreen.exe','-batchmode','-force-d3d11','-encoderJob',out/'job.json',
                '-logFile',out/'unity-player.log'],out/'launch.log',timeout=600)
    result=load(out/'unity-player.json')
    by_id={r['id']:r for r in result.get('cases',[])}
    records=[]
    for case in cases:
        actual=by_id.get(case['id'])
        if not actual:continue
        ref=[i for i in case['reference_ids'] if i!=1]
        native=[i for i in actual['expected_ctc_ids'] if i!=1]
        unity=[i for i in actual['actual_ctc_ids'] if i!=1]
        records.append({'id':case['id'],'word':case['word'],'position':case['position'],
                        'reference_jamo':len(ref),'python_errors':edit_distance(ref,native),
                        'unity_errors':edit_distance(ref,unity),'max_abs':actual['max_abs'],
                        'ctc_equal':actual['decoded_equal'],'source_sha256':case['source_sha256']})
    groups=defaultdict(list)
    for row in records:groups[(row['position'],row['word'])].append(row)
    by_word=[{'position':p,'word':w,'recordings':len(rows),
              'python_exact':sum(r['python_errors']==0 for r in rows),
              'unity_exact':sum(r['unity_errors']==0 for r in rows)}
             for (p,w),rows in sorted(groups.items())]
    overall=[]
    for p in ['center','end_margin_up_to100ms']:
        rows=[r for r in records if r['position']==p]
        if rows:
            total=sum(r['reference_jamo'] for r in rows)
            overall.append({'position':p,'recordings':len(rows),
                            'python_exact':sum(r['python_errors']==0 for r in rows),
                            'unity_exact':sum(r['unity_errors']==0 for r in rows),
                            'python_jer':sum(r['python_errors'] for r in rows)/total,
                            'unity_jer':sum(r['unity_errors'] for r in rows)/total})
    report={'key':key,'exit_code':rc,'source_manifest_sha256':digest(source),
            'unique_recordings':len(clips),'derived_inputs':len(cases),'received_results':len(records),
            'all_ctc_equal':len(records)==len(cases) and all(r['ctc_equal'] for r in records),
            'max_abs':max((r['max_abs'] for r in records),default=None),'by_word':by_word,'overall':overall,
            'notes':['Same 53 recordings in two zero-padding positions, not 106 independent voices',
                     'All words preserved; no cropping, aliases, thresholds or target forcing',
                     'Fixed-window greedy transcription agreement, not complete microphone/VAD/product acceptance validation'],
            'records':records}
    dump(out/'result.json',report)
    print('DEPLOYED_WORDS',key,report['all_ctc_equal'],overall,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('key');args=p.parse_args();run(args.key)
