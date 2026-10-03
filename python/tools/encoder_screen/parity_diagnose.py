"""Check whether CPU ORT graph optimizations account for absolute-error outliers."""
import argparse
import gc
import numpy as np
import onnxruntime as ort
from common import WORK,load,dump
from probe import decode


def run(key):
    out=WORK/'exports'/key
    initial=load(out/'export.json');job=load(out/'job.json')
    ids={r['id'] for r in initial['torch_onnx_parity'] if r['max_abs']>=.02}
    results=[]
    for mode in ['ORT_DISABLE_ALL','ORT_ENABLE_BASIC']:
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        options.graph_optimization_level=getattr(ort.GraphOptimizationLevel,mode)
        session=ort.InferenceSession(str(out/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        for case in job['cases']:
            if case['id'] not in ids:continue
            x=np.fromfile(case['input_file'],dtype='<f4').reshape(case['input_shape'])
            want=np.fromfile(case['logits_file'],dtype='<f4').reshape(case['output_shape'])
            got=session.run(None,{'waveform':x})[0]
            rec={'mode':mode,'id':case['id'],'max_abs':float(np.abs(want-got).max()),
                 'relative_l2':float(np.linalg.norm(want-got)/np.linalg.norm(want)),
                 'ctc_equal':decode(want.argmax(-1)[0].tolist())==decode(got.argmax(-1)[0].tolist())}
            results.append(rec);print(key,rec,flush=True)
        del session;gc.collect()
    dump(out/'ort-optimization-diagnostic.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('keys',nargs='+');args=p.parse_args()
    for key in args.keys:run(key)
