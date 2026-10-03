"""Export wave->jamo logits with embedded preprocessing and numeric references."""
from pathlib import Path
from collections import Counter
import argparse
import gc
import json
import sys
import time
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import soundfile as sf
import torch
from common import ROOT,WORK,MODELS,dump,load,digest
from probe import encoder,input_values
from frontend import WaveNormalize,BertFbank


class EndToEnd(torch.nn.Module):
    def __init__(self,model,feature,key,head):
        super().__init__();self.encoder=model;self.head=head
        self.bert=MODELS[key]['family']=='wav2vec2-bert'
        self.frontend=BertFbank(feature) if self.bert else WaveNormalize(getattr(feature,'do_normalize',True),getattr(feature,'normalize_epsilon',1e-7))

    def forward(self,waveform):
        x=self.frontend(waveform)
        hidden=self.encoder(x,return_dict=False)[0]
        return self.head(hidden)


def cases():
    directory=WORK/'export-cases';directory.mkdir(exist_ok=True)
    file=directory/'cases.json'
    if file.exists():return load(file)
    m=load(WORK/'manifest.json')
    short=sorted((r for r in m['rows'] if r['split']=='test'),key=lambda r:(r['samples'],r['id']))[:8]
    assert len(short)==8
    result=[]
    for row in short:
        x,sr=sf.read(WORK/'audio'/row['wav'],dtype='float32');assert sr==16000
        start=max(0,(len(x)-40000)//2)
        x=x[start:start+40000]
        x=np.pad(x,(0,40000-len(x)))
        path=directory/(row['id']+'.f32');x.astype('<f4').tofile(path)
        result.append({'id':row['id'],'input_file':str(path.resolve()),'input_shape':[1,40000],
                       'source':'child test audio, center window or zero padding to 2.5 seconds; numerical parity only',
                       'source_samples':row['samples'],'window_start_sample':start,'source_text':row['text']})
    rng=np.random.default_rng(917)
    base=np.fromfile(result[0]['input_file'],dtype='<f4')
    extras={'gain_minus12db':base*.25118864,'noise_20db':base+rng.normal(0,max(float(np.std(base)),.001)*.1,40000).astype(np.float32),
            'silence':np.zeros(40000,dtype=np.float32),'noise_only':rng.normal(0,.01,40000).astype(np.float32)}
    for name,x in extras.items():
        path=directory/(name+'.f32');x.astype('<f4').tofile(path)
        result.append({'id':name,'input_file':str(path.resolve()),'input_shape':[1,40000],'source':'controlled parity input'})
    dump(file,result)
    return result


def export(key, finetune_seed=None):
    variant=key if finetune_seed is None else f'{key}-pilot{finetune_seed}'
    out=WORK/'exports'/variant;out.mkdir(parents=True,exist_ok=True)
    model,feature,info,verified=encoder(key,device='cpu')
    try:model.set_attn_implementation('eager')
    except ValueError:pass
    vocab=load(WORK/'vocab.json')
    head=torch.nn.Linear(model.config.hidden_size,len(vocab))
    probe_file=WORK/'probes'/key/'head.pt'
    finetune=None
    if finetune_seed is None:
        assert probe_file.exists(),probe_file
        head.load_state_dict(torch.load(probe_file,map_location='cpu',weights_only=True))
    else:
        directory=WORK/'finetunes'/key/f'seed{finetune_seed}'
        finetune=load(directory/'result.json')
        weights=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(weights['encoder'],strict=True)
        if 'linear.weight' in weights['head']:
            from normalized_probe import StandardHead
            head=StandardHead(weights['head']['mean'],weights['head']['scale'],len(vocab))
        head.load_state_dict(weights['head'],strict=True)
        del weights
    wrapper=EndToEnd(model,feature,key,head).eval()
    sample_cases=cases()
    prepared=[];preprocess=[];wrapper_checks=[]
    with torch.inference_mode():
        for item in sample_cases:
            wave=np.fromfile(item['input_file'],dtype='<f4')
            raw=torch.from_numpy(wave).unsqueeze(0)
            before=input_values(feature,wave,'cpu')
            actual=wrapper.frontend(raw)
            expected=before.get('input_values',before.get('input_features'))
            diff=(expected-actual).abs()
            assert actual.shape==expected.shape,(actual.shape,expected.shape)
            pp={'id':item['id'],'max_abs':diff.max().item(),'mean_abs':diff.mean().item()}
            preprocess.append(pp)
            assert pp['max_abs']<.005,pp
            logits=wrapper(raw).numpy().astype('<f4')
            assert np.isfinite(logits).all()
            reference=head(model(**before).last_hidden_state).numpy()
            from probe import decode
            wrapper_checks.append({'id':item['id'],'max_abs':float(np.abs(reference-logits).max()),
                'relative_l2':float(np.linalg.norm(reference-logits)/max(np.linalg.norm(reference),1e-12)),
                'ctc_equal':decode(reference.argmax(-1)[0].tolist())==decode(logits.argmax(-1)[0].tolist())})
            target=out/(item['id']+'-torch.f32');logits.tofile(target)
            prepared.append({**item,'logits_file':str(target.resolve()),'output_shape':list(logits.shape)})
    dump(out/'preprocessing-parity.json',preprocess)
    dump(out/'python-wrapper-parity.json',wrapper_checks)
    dummy=torch.from_numpy(np.fromfile(sample_cases[0]['input_file'],dtype='<f4')).unsqueeze(0)
    graph_path=out/'graph-unbound.onnx'
    start=time.monotonic()
    # No embedded parameters here: this avoids protobuf's 2GB serialization cap.
    # Bind exact named tensors below into separate, modest-sized external files.
    torch.onnx.export(wrapper,(dummy,),str(graph_path),input_names=['waveform'],output_names=['logits'],
        export_params=False,keep_initializers_as_inputs=True,do_constant_folding=False,
        opset_version=17,dynamo=False)
    graph=onnx.load(graph_path,load_external_data=False)
    state=wrapper.state_dict()
    initializers=[];kept=[];weight_records=[]
    for item in list(graph.graph.input):
        if item.name=='waveform':kept.append(item);continue
        assert item.name in state,('unmapped exported parameter',item.name)
        arr=state[item.name].detach().cpu().numpy()
        tensor=numpy_helper.from_array(arr,name=item.name)
        filename=f'w{len(initializers):04d}.bin'
        weight_file=out/filename
        weight_file.write_bytes(tensor.raw_data)
        onnx.external_data_helper.set_external_data(tensor,location=filename,offset=0,length=len(tensor.raw_data))
        tensor.ClearField('raw_data')
        initializers.append(tensor)
        weight_records.append({'name':item.name,'file':filename,'shape':list(arr.shape),'sha256':digest(weight_file)})
    graph.graph.ClearField('input');graph.graph.input.extend(kept)
    graph.graph.initializer.extend(initializers)
    onnx_path=out/'model.onnx';onnx.save_model(graph,str(onnx_path))
    onnx.checker.check_model(str(onnx_path))
    ops=dict(Counter(n.op_type for n in graph.graph.node))
    del wrapper,model,head,state,graph
    gc.collect()
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=['CPUExecutionProvider'])
    checks=[]
    for item in prepared:
        x=np.fromfile(item['input_file'],dtype='<f4').reshape(item['input_shape'])
        expected=np.fromfile(item['logits_file'],dtype='<f4').reshape(item['output_shape'])
        actual=session.run(None,{'waveform':x})[0]
        d=np.abs(expected-actual)
        same=int(np.sum(expected.argmax(-1)==actual.argmax(-1)))
        from probe import decode
        record={'id':item['id'],'max_abs':float(d.max()),'mean_abs':float(d.mean()),
                'relative_l2':float(np.linalg.norm(expected-actual)/max(np.linalg.norm(expected),1e-12)),
                'equal_frames':same,'frames':actual.shape[1],
                'ctc_equal':decode(expected.argmax(-1)[0].tolist())==decode(actual.argmax(-1)[0].tolist()),
                'within_initial_absolute_tolerance':bool(d.max()<.02)}
        checks.append(record)
        assert np.isfinite(actual).all(),record
    job={'key':variant,'model':str((out/'model.sentis').resolve()),'output':str((out/'unity-player.json').resolve()),'cases':prepared,'repeats':40}
    dump(out/'job.json',job)
    dump(out/'export.json',{'key':variant,'base_key':key,'finetune':finetune,'opset':17,'waveform_samples':40000,'preprocessing_embedded':True,
                          'weights':weight_records,'ops':ops,'torch_onnx_parity':checks,'source':verified,
                          'raw_probe_head_sha256':digest(probe_file),
                          'deployed_weights_source':'raw frozen probe' if finetune is None else 'finetune checkpoint: encoder and head',
                          'deployed_checkpoint_sha256':None if finetune is None else finetune['checkpoint_sha256'],
                          'export_seconds':time.monotonic()-start,
                          'onnx_sha256':digest(onnx_path),
                          'python_wrapper_all_ctc_equal':all(r['ctc_equal'] for r in wrapper_checks),
                          'python_wrapper_max_abs':max(r['max_abs'] for r in wrapper_checks),
                          'all_ort_absolute_tolerance_passed':all(r['within_initial_absolute_tolerance'] for r in checks),
                          'all_ort_ctc_equal':all(r['ctc_equal'] for r in checks)})
    print('EXPORT_COMPLETE',variant,len(initializers),'tensors',len(ops),'operator types',flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('key');p.add_argument('--finetune-seed',type=int);args=p.parse_args()
    export(args.key,args.finetune_seed)
