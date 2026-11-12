"""Export the fixed 2.5-second Unity input contract, with CPU numeric checks."""
import argparse
from collections import Counter
import gc
import gzip
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import soundfile as sf
import torch

from inference import Engine, decode, read, sha

ROOT = Path(__file__).resolve().parents[4]
SAMPLES = 40000


class UnityModel(torch.nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, waveform):
        # Raw PCM floats; the HuBERT checkpoint was trained without gain normalization.
        return self.head(self.encoder(waveform, return_dict=False)[0])


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def export(bundle, output):
    output.mkdir(parents=True, exist_ok=False)
    engine = Engine(bundle, device='cpu', threads=2)
    engine.load()
    with gzip.open(ROOT / '02_Datasets/학습준비/자모_원문라벨_3-6세/manifests-20261027/monitor.jsonl.gz', 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    cases, waves = [], []
    for age in (3, 4, 5, 6):
        row = next(row for row in rows if row['age'] == age and 1.3 < row['seconds'] < 2.4)
        audio_path = (ROOT / row['audio_path']).resolve()
        assert audio_path.drive.upper() == 'C:' and audio_path.is_relative_to(ROOT)
        wave, rate = sf.read(audio_path, dtype='float32')
        assert rate == 16000 and wave.ndim == 1
        waves.append(np.pad(wave, (SAMPLES-len(wave), 0)))
        cases.append({'id': f'age-{age}', 'reference': row['text'], 'source_samples': len(wave)})
    waves += [np.zeros(SAMPLES, dtype=np.float32), np.random.default_rng(917).normal(0, .005, SAMPLES).astype(np.float32)]
    cases += [{'id': 'silence', 'reference': ''}, {'id': 'noise-only', 'reference': ''}]
    sdpa = []
    for case, wave in zip(cases, waves):
        result = engine.transcribe(wave)
        sdpa.append(result['text'])
        print(json.dumps({'event':'reference','id':case['id'],'text':result['text']},ensure_ascii=False),flush=True)
    engine.model.set_attn_implementation('eager')
    wrapper = UnityModel(engine.model, engine.head).eval()
    references = []
    with torch.inference_mode():
        for case, wave, expected_text in zip(cases, waves, sdpa):
            raw = torch.from_numpy(wave[None,:])
            logits = wrapper(raw).numpy().astype('<f4')
            text = decode(logits.argmax(-1)[0], engine.tokens)[0]
            assert text == expected_text, (case['id'], text, expected_text)
            input_file, logits_file = output / f'{case["id"]}-input.f32', output / f'{case["id"]}-torch.f32'
            wave.astype('<f4').tofile(input_file)
            logits.tofile(logits_file)
            references.append(logits)
            case.update(input_file=str(input_file.resolve()), input_shape=[1,SAMPLES],
                        logits_file=str(logits_file.resolve()), output_shape=list(logits.shape), text=text)
    graph_path = output / 'unbound.onnx'
    torch.onnx.export(wrapper, (torch.zeros(1,SAMPLES),), str(graph_path),
                      input_names=['waveform'], output_names=['logits'], export_params=False,
                      keep_initializers_as_inputs=True, do_constant_folding=False, opset_version=17, dynamo=False)
    graph = onnx.load(graph_path, load_external_data=False)
    state = wrapper.state_dict()
    kept, initializers, weights = [], [], []
    for item in graph.graph.input:
        if item.name == 'waveform':
            kept.append(item)
            continue
        if item.name not in state:
            raise ValueError(f'Unmapped parameter: {item.name}')
        value = state[item.name].detach().cpu().numpy()
        tensor = numpy_helper.from_array(value, name=item.name)
        filename = f'w{len(initializers):04d}.bin'
        target = output / filename
        target.write_bytes(tensor.raw_data)
        onnx.external_data_helper.set_external_data(tensor, location=filename, offset=0, length=len(tensor.raw_data))
        tensor.ClearField('raw_data')
        initializers.append(tensor)
        weights.append({'file':filename,'bytes':target.stat().st_size,'sha256':sha(target)})
    graph.graph.ClearField('input'); graph.graph.input.extend(kept)
    graph.graph.initializer.extend(initializers)
    model_path = output / 'hubert_ko_child_jamo.onnx'
    onnx.save_model(graph, str(model_path))
    onnx.checker.check_model(str(model_path))
    operators = dict(Counter(node.op_type for node in graph.graph.node))
    tokens = engine.tokens
    model_id = engine.status()['model_id']
    del wrapper, engine, state, graph, initializers, value, tensor
    gc.collect()
    print(json.dumps({'event':'onnx_written','weights':len(weights),'bytes':sum(w['bytes'] for w in weights)}),flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads=2; options.inter_op_num_threads=1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    checks = []
    for case, wave, expected in zip(cases, waves, references):
        started = time.perf_counter()
        actual = session.run(None, {'waveform':wave[None,:]})[0]
        elapsed = time.perf_counter()-started
        text = decode(actual.argmax(-1)[0],tokens)[0]
        difference = np.abs(expected-actual)
        record = {'id':case['id'],'text':text,'ctc_equal':text==case['text'],
                  'max_abs':float(difference.max()),'relative_l2':float(np.linalg.norm(expected-actual)/max(np.linalg.norm(expected),1e-12)),
                  'inference_seconds':elapsed}
        checks.append(record)
        assert np.isfinite(actual).all() and record['ctc_equal'] and record['relative_l2'] < .001, record
        print(json.dumps({'event':'onnx_parity',**record},ensure_ascii=False),flush=True)
    vocab = {'_comment':'Orthographic Unicode NFD tokens; CTC collapse then NFC composition. Raw audio, no normalization.',
             'model':'hubert_ko_child_jamo','vocab_size':69,'blank_id':0,'unk_id':-1,'word_delimiter_id':1,
             'normalize':False,'compose_hangul_jamo':True,'normalize_epsilon':1e-7,'sampling_rate':16000,'tokens':[tokens[i] for i in range(69)]}
    write(output/'hubert_ko_child_jamo_vocab.json',vocab)
    write(output/'cases.json',cases)
    write(output/'export.json',{'status':'complete','source_model_id':model_id,'onnx':model_path.name,
                               'onnx_sha256':sha(model_path),'input_samples':SAMPLES,'sample_rate':16000,
                               'normalize':False,'opset':17,'operators':operators,'weights':weights,'cpu_checks':checks})
    print(json.dumps({'event':'export_complete','directory':str(output)},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export(args.bundle.resolve(),args.output.resolve())
