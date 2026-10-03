"""Isolate the W2v-BERT frontend from its encoder for real GPU numeric checks."""
import numpy as np
import torch
from transformers import AutoFeatureExtractor
from common import WORK,dump
from frontend import BertFbank
from export import cases
from deploy import convert,player

if __name__=='__main__':
    torch.set_num_threads(4)
    out=WORK/'exports/w2vbert-fbank-debug';out.mkdir(parents=True,exist_ok=True)
    feature=AutoFeatureExtractor.from_pretrained(WORK/'models/w2vbert',local_files_only=True)
    front=BertFbank(feature).eval();prepared=[]
    with torch.inference_mode():
        for case in cases():
            wave=torch.from_numpy(np.fromfile(case['input_file'],dtype='<f4')).unsqueeze(0)
            expected=front(wave).numpy().astype('<f4')
            target=out/(case['id']+'.f32');expected.tofile(target)
            prepared.append({**case,'logits_file':str(target.resolve()),'output_shape':list(expected.shape)})
    torch.onnx.export(front,(wave,),str(out/'model.onnx'),input_names=['waveform'],output_names=['logits'],
                      opset_version=17,dynamo=False)
    dump(out/'job.json',{'key':'w2vbert-fbank-debug','model':str((out/'model.sentis').resolve()),
                        'output':str((out/'unity-player.json').resolve()),'cases':prepared,'repeats':4})
    print(convert('w2vbert-fbank-debug'),flush=True)
    r=player('w2vbert-fbank-debug')
    for case in (r.get('result') or {}).get('cases',[]):print(case,flush=True)
