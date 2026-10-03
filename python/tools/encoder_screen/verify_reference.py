"""Compare exported Python reference with the library's original preprocessing."""
import argparse
import numpy as np
import torch
from common import WORK,load,dump
from probe import encoder,input_values,decode


@torch.inference_mode()
def run(key):
    torch.set_num_threads(2)
    model,feature,info,verified=encoder(key,'cpu')
    original_attention=getattr(model.config,'_attn_implementation',None)
    head=torch.nn.Linear(model.config.hidden_size,len(load(WORK/'vocab.json')))
    head.load_state_dict(torch.load(WORK/'probes'/key/'head.pt',map_location='cpu',weights_only=True))
    head.eval();rows=[]
    for case in load(WORK/'exports'/key/'job.json')['cases']:
        wave=np.fromfile(case['input_file'],dtype='<f4')
        values=input_values(feature,wave,'cpu')
        actual=head(model(**values).last_hidden_state).numpy()
        target=np.fromfile(case['logits_file'],dtype='<f4').reshape(case['output_shape'])
        assert actual.shape==target.shape and np.isfinite(actual).all()
        rows.append({'id':case['id'],'max_abs':float(np.abs(actual-target).max()),
            'relative_l2':float(np.linalg.norm(actual-target)/max(np.linalg.norm(target),1e-12)),
            'ctc_equal':decode(actual.argmax(-1)[0].tolist())==decode(target.argmax(-1)[0].tolist())})
    result={'native_attention_implementation':original_attention,'cases':rows,
            'all_ctc_equal':all(r['ctc_equal'] for r in rows),'max_abs':max(r['max_abs'] for r in rows),
            'note':'Original feature extractor/attention-mask/native CPU model vs stored export-wrapper reference, same raw head'}
    dump(WORK/'exports'/key/'native-reference-parity.json',result)
    print('NATIVE_REFERENCE',key,result['all_ctc_equal'],result['max_abs'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('key');args=p.parse_args();run(args.key)
