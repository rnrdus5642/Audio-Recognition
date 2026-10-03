"""Explicit fairseq2 W2V encoder -> Transformers mapping, with independent math check.

The reference equations follow the publisher's wav2vec2 frontend and PRE-norm
Transformer configuration. This is not a run of the fairseq2 native library.
"""
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model
from common import WORK, dump, load, digest


class OmniFeature:
    do_normalize = True
    normalize_epsilon = 1e-5

    def __call__(self, audio, sampling_rate=16000, return_tensors='pt'):
        assert sampling_rate == 16000
        x = torch.as_tensor(np.array(audio, dtype=np.float32, copy=True))
        return {'input_values':F.layer_norm(x,x.shape).unsqueeze(0)}


def map_name(k):
    if k.startswith('encoder_frontend.feature_extractor.layers.'):
        return k.replace('encoder_frontend.feature_extractor.layers.', 'feature_extractor.conv_layers.')
    for old,new in [
        ('encoder_frontend.post_extract_layer_norm.', 'feature_projection.layer_norm.'),
        ('encoder_frontend.model_dim_proj.', 'feature_projection.projection.'),
        ('encoder_frontend.pos_encoder.conv.weight_g', 'encoder.pos_conv_embed.conv.parametrizations.weight.original0'),
        ('encoder_frontend.pos_encoder.conv.weight_v', 'encoder.pos_conv_embed.conv.parametrizations.weight.original1'),
        ('encoder_frontend.pos_encoder.conv.', 'encoder.pos_conv_embed.conv.'),
        ('masker.temporal_mask_embed', 'masked_spec_embed'),
    ]:
        if k.startswith(old): return k.replace(old,new,1)
    if k.startswith('encoder.'):
        for old,new in [('.self_attn_layer_norm.','.layer_norm.'),('.ffn_layer_norm.','.final_layer_norm.'),
                        ('.self_attn.output_proj.','.attention.out_proj.'),('.self_attn.','.attention.'),
                        ('.ffn.inner_proj.','.feed_forward.intermediate_dense.'),('.ffn.output_proj.','.feed_forward.output_dense.')]:
            k=k.replace(old,new)
        return k
    return None


def functional_reference(s,x,layers,heads=16):
    def norm(x,p):return F.layer_norm(x,(x.shape[-1],),s[p+'.weight'],s[p+'.bias'],1e-5)
    def linear(x,p):return F.linear(x,s[p+'.weight'],s[p+'.bias'])
    z=x.unsqueeze(1)
    for i,stride in enumerate([5,2,2,2,2,2,2]):
        p=f'encoder_frontend.feature_extractor.layers.{i}'
        z=F.conv1d(z,s[p+'.conv.weight'],s[p+'.conv.bias'],stride=stride)
        z=F.gelu(norm(z.transpose(1,2),p+'.layer_norm')).transpose(1,2)
    z=linear(norm(z.transpose(1,2),'encoder_frontend.post_extract_layer_norm'),'encoder_frontend.model_dim_proj')
    p='encoder_frontend.pos_encoder.conv'
    v=s[p+'.weight_v'];g=s[p+'.weight_g']
    weight=v*g/torch.sqrt((v*v).sum((0,1),keepdim=True))
    pos=F.conv1d(z.transpose(1,2),weight,s[p+'.bias'],padding=64,groups=16)[:,:,:-1]
    z=z+F.gelu(pos).transpose(1,2)
    b,t,d=z.shape
    for i in range(layers):
        p=f'encoder.layers.{i}'
        h=norm(z,p+'.self_attn_layer_norm')
        q,k,v=[linear(h,p+'.self_attn.'+n+'_proj').reshape(b,t,heads,d//heads).transpose(1,2) for n in ['q','k','v']]
        a=torch.softmax((q@k.transpose(-1,-2))*(d//heads)**-.5,dim=-1)@v
        z=z+linear(a.transpose(1,2).reshape(b,t,d),p+'.self_attn.output_proj')
        z=z+linear(F.gelu(linear(norm(z,p+'.ffn_layer_norm'),p+'.ffn.inner_proj')),p+'.ffn.output_proj')
    return norm(z,'encoder.layer_norm')


def converted_encoder(key):
    directory=WORK/'models'/key
    source=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)['model']
    config=Wav2Vec2Config.from_pretrained(WORK/'models/xlsr300',local_files_only=True)
    layers=48 if key=='omni1b' else 24
    config.num_hidden_layers=layers
    config.hidden_size=1280 if layers==48 else 1024
    config.intermediate_size=5120 if layers==48 else 4096
    config.hidden_dropout=0. if layers==48 else .1
    config.attention_dropout=0.; config.layerdrop=0.
    config._attn_implementation='eager'
    mapped={}; ignored=[]
    for name,value in source.items():
        target=map_name(name)
        if target is None:
            assert name.startswith(('quantizer.','final_proj.','final_target_proj.')),name
            ignored.append(name)
        else:
            assert target not in mapped,target
            mapped[target]=value
    with torch.device('meta'):model=Wav2Vec2Model(config)
    result=model.load_state_dict(mapped,strict=True,assign=True)
    model.eval()
    info={'missing_keys':result.missing_keys,'unexpected_keys':result.unexpected_keys,
          'ignored_pretraining_only_keys':ignored,'mapped_encoder_tensors':len(mapped)}
    record=directory/'conversion-verification.json'
    if not record.exists():
        model.cuda()
        src_gpu={k:v.cuda() for k,v in source.items()}
        metrics=[]
        for n in [16000,40000]:
            x=torch.randn(1,n,device='cuda',generator=torch.Generator(device='cuda').manual_seed(917+n))
            with torch.inference_mode():
                native=functional_reference(src_gpu,x,layers)
                actual=model(x).last_hidden_state
            diff=(native-actual).abs()
            rec={'samples':n,'max_abs':diff.max().item(),'mean_abs':diff.mean().item(),
                 'relative_l2':((native-actual).norm()/native.norm()).item()}
            metrics.append(rec)
            assert rec['relative_l2']<1e-4,rec
        dump(record,{'source_sha256':digest(directory/'checkpoint.pt'), **info,'reference_comparison':metrics,
                     'reference':'Independent PyTorch functional equations following public fairseq2 frontend/PRE-norm configuration; fairseq2 native runtime not available on this Windows environment.',
                     'normalize_epsilon':1e-5})
        del src_gpu,native,actual
        torch.cuda.empty_cache()
    else:
        assert load(record)['source_sha256']==digest(directory/'checkpoint.pt')
    return model,OmniFeature(),info
