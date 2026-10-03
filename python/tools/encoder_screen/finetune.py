"""Small controlled encoder fine-tuning pilot, separate from production training.

Same split, sample order, update budget, optimizer and jamo head type. The CNN
frontend is frozen for waveform encoders; transformer/Conformer blocks and CTC
head are updated. This measures early adaptation, not convergence or a final
hyperparameter search. Test scores are produced only after validation selection.
"""
import argparse
import gc
import math
import random
import time
import numpy as np
import soundfile as sf
import torch
from common import WORK, dump, load, digest
from probe import encoder, input_values, decode, edit_distance


@torch.inference_mode()
def evaluate(model, head, values, rows):
    model.eval(); head.eval()
    errors = total = exact = 0
    predictions = []
    for row in rows:
        x = {k: v.cuda() for k, v in values[row['id']].items()}
        logits = head(model(**x).last_hidden_state)
        assert torch.isfinite(logits).all(), row['id']
        hyp = decode(logits.argmax(-1)[0].cpu().tolist())
        ref = [i for i in row['label_ids'] if i != 1]
        err = edit_distance(ref, [i for i in hyp if i != 1])
        errors += err; total += len(ref); exact += err == 0
        predictions.append({'id': row['id'], 'hyp_ids': hyp, 'errors': err, 'reference_jamo': len(ref)})
    return {'jer': errors/total, 'errors': errors, 'reference_jamo': total,
            'exact': exact, 'recordings': len(rows)}, predictions


def run(key, steps=400, seed=917, cpu_offload_layers=0):
    out = WORK/'finetunes'/key/f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        prior=load(out/'result.json')
        assert prior['config']['steps']==steps, 'Existing run uses a different update budget'
        assert prior['config']['manifest_sha256']==digest(WORK/'manifest.json'), 'Dataset changed'
        assert prior['config']['cpu_offload_first_encoder_layers']==cpu_offload_layers, 'Existing run uses different optimizer placement'
        return prior
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from normalized_probe import run as normalized_probe, StandardHead
    normalized_probe(key)
    # The optional head job must not change the requested adaptation RNG state.
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    manifest = load(WORK/'manifest.json')
    rows = {s: [r for r in manifest['rows'] if r['split'] == s] for s in ['train','valid','test']}
    model, feature, info, verified = encoder(key)
    for name in ['mask_time_prob', 'mask_feature_prob', 'layerdrop']:
        if hasattr(model.config, name): setattr(model.config, name, 0.)
    if hasattr(model.config, 'apply_spec_augment'): model.config.apply_spec_augment = False
    if hasattr(model, 'feature_extractor'):
        for p in model.feature_extractor.parameters(): p.requires_grad_(False)
        if hasattr(model.feature_extractor, '_requires_grad'):
            model.feature_extractor._requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    stats=load(WORK/'probes-normalized'/key/'statistics.json')
    raw=load(WORK/'probes'/key/'result.json')
    scaled=load(WORK/'probes-normalized'/key/'result.json')
    use_scaled=scaled['selected']['valid']['jer']<raw['selected']['valid']['jer']
    if use_scaled:
        head = StandardHead(stats['mean'],stats['scale'],len(manifest['vocab'])).cuda()
        head_file = WORK/'probes-normalized'/key/'head.pt'
        head.load_state_dict(torch.load(head_file, map_location='cpu', weights_only=True))
    else:
        head = StandardHead(np.zeros(model.config.hidden_size),np.ones(model.config.hidden_size),len(manifest['vocab'])).cuda()
        head_file = WORK/'probes'/key/'head.pt'
        head.linear.load_state_dict(torch.load(head_file,map_location='cpu',weights_only=True))
    values = {}
    for row in manifest['rows']:
        audio, sr = sf.read(WORK/'audio'/row['wav'], dtype='float32')
        assert sr == 16000
        values[row['id']] = input_values(feature, audio, 'cpu')
    config = {'key': key, 'seed': seed, 'steps': steps, 'effective_batch': 8, 'micro_batch': 1,
              'encoder_lr': 2e-5, 'head_lr': 1e-4, 'warmup_steps': 40, 'weight_decay': .01,
              'amp': 'bfloat16 training; float32 evaluation and weights', 'gradient_clip': 1.,
              'cpu_offload_first_encoder_layers':cpu_offload_layers,
              'head_standardization':'train-only single scalar' if use_scaled else 'identity (raw features)',
              'head_selection':'Minimum valid JER over raw/scalar-normalized x two learning rates; no test/owned-word selection',
              'head_statistics_sha256':digest(WORK/'probes-normalized'/key/'statistics.json'),
              'frozen': 'convolutional waveform feature_extractor if present; transformer/Conformer trainable',
              'mask_time_prob': 0., 'mask_feature_prob': 0., 'layerdrop': 0.,
              'sampling': 'same seed shuffled train order, batch of 8 with gradient accumulation',
              'manifest_sha256': digest(WORK/'manifest.json'), 'initial_head_sha256': digest(head_file),
              'source': verified, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'limitations': ['Short adaptation pilot, not converged training', 'One learning-rate setting',
                              'Transcript-derived labels do not validate pronunciation-error diagnosis']}
    dump(out/'config.json', config)
    initial, _ = evaluate(model, head, values, rows['valid'])
    print('FINETUNE_INITIAL', key, initial, flush=True)
    from domain import score as domain_score
    domain_score(model,head,feature,out/'domain-initial.json')
    if cpu_offload_layers:
        from offload import PartialCPUAdamW
        opt=PartialCPUAdamW(model,head,cpu_offload_layers)
        config['cpu_offloaded_parameters']=opt.offloaded_parameters
        dump(out/'config.json',config)
    else:
        opt = torch.optim.AdamW([{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 2e-5},
                                 {'params': head.parameters(), 'lr': 1e-4}], weight_decay=.01, foreach=False)
    rng = random.Random(seed)
    order = list(range(len(rows['train']))); rng.shuffle(order); cursor = 0
    history = []; best = None; start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps+1):
        model.train(); head.train()
        ratio = min(step/40, 1.) * max(0., (steps-step)/(steps-40)) if step > 40 else step/40
        opt.param_groups[0]['lr'] = 2e-5 * ratio
        opt.param_groups[1]['lr'] = 1e-4 * ratio
        opt.zero_grad(set_to_none=True)
        losses = []
        for micro in range(8):
            if cursor == len(order): rng.shuffle(order); cursor = 0
            row = rows['train'][order[cursor]]; cursor += 1
            x = {k: v.cuda() for k, v in values[row['id']].items()}
            target = torch.tensor(row['label_ids'], dtype=torch.long, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = head(model(**x).last_hidden_state)
            logp = logits.float().log_softmax(-1).transpose(0,1)
            loss = torch.nn.functional.ctc_loss(logp, target, [logp.shape[0]], [len(target)],
                                                 blank=0, reduction='mean', zero_infinity=False)
            assert torch.isfinite(loss), (key, step, row['id'])
            (loss/8).backward(); losses.append(loss.item())
        norm = torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()), 1., error_if_nonfinite=True)
        opt.step()
        if step % 25 == 0:
            print('FINETUNE', key, step, 'loss', round(float(np.mean(losses)),4),
                  'seconds', round(time.monotonic()-start,1), flush=True)
        if step % 100 == 0 or step == steps:
            score, _ = evaluate(model, head, values, rows['valid'])
            history.append({'step':step, 'valid':score, 'seconds':time.monotonic()-start})
            if best is None or score['jer'] < best['valid']['jer']:
                best = {'step':step, 'valid':score}
                torch.save({'encoder':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                            'head':{k:v.detach().cpu() for k,v in head.state_dict().items()}}, out/'best.pt')
            dump(out/'progress.json', {'history':history,'best':best})
            print('FINETUNE_VALID',key,step,score,flush=True)
    del opt; gc.collect(); torch.cuda.empty_cache()
    saved = torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['encoder'],strict=True); head.load_state_dict(saved['head'],strict=True)
    del saved
    test, predictions = evaluate(model, head, values, rows['test'])
    dump(out/'test-predictions.json', predictions)
    domain_score(model,head,feature,out/'domain-final.json')
    result = {'config':config,'initial_valid':initial,'selected':best,'history':history,'test':test,
              'elapsed_seconds':time.monotonic()-start,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(), 'checkpoint_sha256':digest(out/'best.pt')}
    dump(out/'result.json',result)
    print('FINETUNE_COMPLETE',key,test,flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('key'); p.add_argument('--steps',type=int,default=400)
    p.add_argument('--seed',type=int,default=917)
    p.add_argument('--cpu-offload-layers',type=int,default=0); args=p.parse_args()
    run(args.key,args.steps,args.seed,args.cpu_offload_layers)
