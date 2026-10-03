"""Bounded fixed-shape constant folding diagnostic for the BERT Unity graph.

Only inferred fixed Shape/Size and small all-constant nodes are evaluated. No
runtime input is observed or replaced. The unmodified graph/results are kept.
"""
from collections import Counter
import gc
import shutil
import numpy as np
import onnx
from onnx import helper,numpy_helper
from onnx.reference import ReferenceEvaluator
import onnxruntime as ort
from common import WORK,load,dump,digest
from probe import decode


def run():
    source=WORK/'exports/w2vbert'
    out=WORK/'exports/w2vbert-shape-folded';out.mkdir(parents=True,exist_ok=True)
    # ONNX 1.22 intentionally rejects multiply-linked external tensor files.
    # Repair only this diagnostic's links from its first failed attempt.
    for target in out.glob('w*.bin'):
        src=(source/target.name).resolve();target=target.resolve()
        assert src.is_relative_to(source.resolve()) and target.is_relative_to(out.resolve())
        if src.exists() and target.samefile(src):target.unlink()
    graph=onnx.load(source/'model.onnx',load_external_data=False)
    inferred=onnx.shape_inference.infer_shapes(graph,strict_mode=True,data_prop=True)
    shapes={}
    for value in list(inferred.graph.input)+list(inferred.graph.value_info)+list(inferred.graph.output):
        dims=value.type.tensor_type.shape.dim
        if dims and all(d.HasField('dim_value') for d in dims):
            shapes[value.name]=tuple(d.dim_value for d in dims)
    constants={};initializers={p.name:p for p in graph.graph.initializer}
    for name,p in initializers.items():
        shapes[name]=tuple(p.dims)
        if np.prod(p.dims,dtype=np.int64)<=65536:
            copy=onnx.TensorProto();copy.CopyFrom(p)
            constants[name]=numpy_helper.to_array(copy,base_dir=str(source))
    allowed={'Constant','Identity','Add','Sub','Mul','Div','Neg','Pow','Sqrt','Cast',
             'Gather','Unsqueeze','Squeeze','Concat','Reshape','Transpose','Expand',
             'Range','Slice','ConstantOfShape','Equal','Less','Greater','Where',
             'ReduceProd','ReduceSum','ReduceMean','Clip'}
    nodes=[];folded=Counter();generated=[]
    for node in graph.graph.node:
        result=None
        if node.op_type in ['Shape','Size'] and node.input[0] in shapes:
            shape=np.array(shapes[node.input[0]],np.int64)
            if node.op_type=='Size':result=[np.array(np.prod(shape),np.int64)]
            else:
                attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
                result=[shape[attrs.get('start',0):attrs.get('end',len(shape))]]
        elif node.op_type in allowed and all(x in constants for x in node.input if x):
            # Avoid accidentally materializing an enormous Expand/Range.
            if node.op_type in ['Expand','ConstantOfShape']:
                shape=constants[node.input[1 if node.op_type=='Expand' else 0]]
                if np.prod(shape,dtype=np.int64)>65536:nodes.append(node);continue
            if node.op_type=='Range':
                span=(constants[node.input[1]]-constants[node.input[0]])/constants[node.input[2]]
                if span>65536:nodes.append(node);continue
            try:
                evaluator=ReferenceEvaluator(node,opsets={'':17})
                result=evaluator.run(None,{x:constants[x] for x in node.input if x})
            except (NotImplementedError,TypeError,ValueError,KeyError):result=None
        if result is not None and all(np.asarray(x).size<=65536 for x in result):
            assert len(result)==len(node.output)
            for name,array in zip(node.output,result):
                array=np.asarray(array);constants[name]=array;shapes[name]=array.shape
                generated.append(numpy_helper.from_array(array,name=name))
            folded[node.op_type]+=1
        else:nodes.append(node)
    # Remove the now-unused computations, without changing graph outputs.
    needed={o.name for o in graph.graph.output};kept=[]
    for node in reversed(nodes):
        if any(o in needed for o in node.output):
            kept.append(node);needed.update(x for x in node.input if x)
    graph.graph.ClearField('node');graph.graph.node.extend(reversed(kept))
    params=[p for p in list(graph.graph.initializer)+generated if p.name in needed]
    graph.graph.ClearField('initializer');graph.graph.initializer.extend(params)
    graph.graph.ClearField('value_info')
    for p in graph.graph.initializer:
        if p.data_location==onnx.TensorProto.EXTERNAL:
            relative=next(v.value for v in p.external_data if v.key=='location')
            src=(source/relative).resolve();target=(out/relative).resolve()
            assert src.is_relative_to(source.resolve()) and target.is_relative_to(out.resolve())
            if target.exists() and target.samefile(src):target.unlink()
            if not target.exists():shutil.copyfile(src,target)
    onnx.save_model(graph,str(out/'model.onnx'));onnx.checker.check_model(str(out/'model.onnx'))
    before=len(inferred.graph.node);after=len(graph.graph.node)
    del graph,inferred,constants,initializers,generated;gc.collect()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(out/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    job=load(source/'job.json');results=[]
    for case in job['cases']:
        x=np.fromfile(case['input_file'],dtype='<f4').reshape(case['input_shape'])
        target=np.fromfile(case['logits_file'],dtype='<f4').reshape(case['output_shape'])
        actual=session.run(None,{'waveform':x})[0]
        assert np.isfinite(actual).all()
        results.append({'id':case['id'],'max_abs':float(np.abs(actual-target).max()),
                        'ctc_equal':decode(actual.argmax(-1)[0].tolist())==decode(target.argmax(-1)[0].tolist())})
    job.update(key='w2vbert-shape-folded',model=str((out/'model.sentis').resolve()),
               output=str((out/'unity-player.json').resolve()))
    dump(out/'job.json',job)
    result={'source_onnx_sha256':digest(source/'model.onnx'),'onnx_sha256':digest(out/'model.onnx'),
            'fixed_input_shape':[1,40000],'nodes_before':before,'nodes_after':after,
            'folded':dict(folded),'ort_vs_pytorch':results,
            'note':'Diagnostic export; original BERT export and failed Unity evidence retained'}
    dump(out/'folding.json',result)
    print('SHAPE_FOLD_COMPLETE',before,after,dict(folded),'ctc_all',all(x['ctc_equal'] for x in results),flush=True)


if __name__=='__main__':run()
