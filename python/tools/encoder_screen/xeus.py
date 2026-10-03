"""Strict XEUS encoder adapter for the isolated Windows screening environment.

Uses ESPnet 202412 E-Branchformer and publisher CNN/positional modules from the
pinned ESPnet revision. No training-only masking or prediction heads are loaded.
The representation is the final encoder block output, following the publisher's
use_final_output=False example, rather than a learned weighted sum of layers.
"""
import importlib.util
import sys
from types import SimpleNamespace
import torch
import yaml
from common import WORK, dump, load, digest
from convert_omni import OmniFeature


def module(name, filename):
    path = WORK / 'vendor' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class XeusEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        cnn = module('espnet2.asr.frontend.cnn', 'main_espnet2_asr_frontend_cnn.py')
        embedding = module('encoder_screen_xeus_embedding', 'main_espnet2_legacy_nets_pytorch_backend_transformer_embedding.py')
        from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
        from espnet2.asr.preencoder.linear import LinearProjection
        args = dict(config['frontend_conf'])
        # Normalization is performed in the common waveform frontend exactly once.
        args['normalize_audio'] = False
        self.frontend = cnn.CNNFrontend(**args)
        self.preencoder = LinearProjection(input_size=512, **config['preencoder_conf'])
        args = dict(config['encoder_conf'])
        args['pos_enc_layer_type'] = 'abs_pos'
        self.encoder = EBranchformerEncoder(input_size=1024, **args)
        # 202412's encoder has the same branch equations, but its distribution
        # predates the convolutional positional module present in this checkpoint.
        self.encoder.embed = torch.nn.Sequential(
            embedding.ConvolutionalPositionalEmbedding(1024, .1, weight_norm='legacy'))
        self.config = SimpleNamespace(hidden_size=1024)

    def set_attn_implementation(self, name):
        if name != 'eager':
            raise ValueError(name)

    def forward(self, input_values, attention_mask=None, return_dict=True):
        lengths = torch.full((input_values.shape[0],), input_values.shape[1],
                             dtype=torch.long, device=input_values.device)
        if attention_mask is not None:
            lengths = attention_mask.sum(-1)
        hidden, lengths = self.frontend(input_values, lengths)
        hidden, lengths = self.preencoder(hidden, lengths)
        mask = (torch.arange(hidden.shape[1],device=hidden.device)[None,:] < lengths[:,None])[:,None,:]
        hidden = self.encoder.embed(hidden)
        for block in self.encoder.encoders:
            hidden, mask = block(hidden, mask)
        # The published SSL example requests the last intermediate block. The
        # extra ASR after_norm is not part of that representation.
        return SimpleNamespace(last_hidden_state=hidden) if return_dict else (hidden,)


def xeus_encoder():
    from vendor import ensure
    ensure()
    directory = WORK / 'models' / 'xeus'
    config = yaml.safe_load((directory / 'model/config.yaml').read_text(encoding='utf8'))
    model = XeusEncoder(config)
    source = torch.load(directory / 'model/xeus_checkpoint_new.pth', map_location='cpu', weights_only=True)
    wanted, ignored = {}, []
    for name, tensor in source.items():
        if name.startswith(('frontend.', 'preencoder.', 'encoder.')):
            wanted[name] = tensor
        else:
            assert name.startswith(('losses.', 'util_modules.')) or name == 'global_step', name
            ignored.append(name)
    result = model.load_state_dict(wanted, strict=True, assign=True)
    info = {'missing_keys': result.missing_keys, 'unexpected_keys': result.unexpected_keys,
            'ignored_pretraining_only_keys': ignored, 'mapped_encoder_tensors': len(wanted),
            'espnet_distribution': '202412', 'vendor_revision': load(WORK/'vendor/espnet-main-revision.json'),
            'representation': 'last encoder block, before extra ASR after_norm; publisher use_final_output=False',
            'normalize_epsilon': 1e-5}
    dump(directory/'adapter-verification.json', info)
    return model.eval(), OmniFeature(), info


if __name__ == '__main__':
    torch.set_num_threads(4)
    model, _, info = xeus_encoder()
    with torch.inference_mode():
        y = model(torch.randn(1, 16000)).last_hidden_state
    print(info)
    print('XEUS_CPU', sum(p.numel() for p in model.parameters()), y.shape, torch.isfinite(y).all().item())
