"""ONNX-friendly, fixed-window wave normalization and W2V-BERT filterbank.

The filterbank is checked against the checkpoint's actual HF feature extractor.
It uses real-valued convolutions/matrix products, so no runtime Python or FFT
operator is required in Unity.
"""
import numpy as np
import torch
import torch.nn.functional as F


class WaveNormalize(torch.nn.Module):
    def __init__(self, enabled=True, eps=1e-7):
        super().__init__();self.enabled=enabled;self.eps=eps

    def forward(self,waveform):
        if not self.enabled:return waveform
        x=waveform-waveform.mean(-1,keepdim=True)
        return x/torch.sqrt((x*x).mean(-1,keepdim=True)+self.eps)


class BertFbank(torch.nn.Module):
    def __init__(self,feature,samples=40000):
        super().__init__()
        self.samples=samples
        self.frames=(samples-400)//160+1
        # Frame extraction with a fixed gather table is equivalent to snip_edges.
        index=np.arange(400)[None,:]+np.arange(self.frames)[:,None]*160
        self.register_buffer('frame_index',torch.tensor(index,dtype=torch.int64))
        self.register_buffer('window',torch.tensor(feature.window,dtype=torch.float32))
        angles=2*np.pi*np.arange(257)[:,None]*np.arange(400)[None,:]/512
        self.register_buffer('cosine',torch.tensor(np.cos(angles).T,dtype=torch.float32))
        self.register_buffer('sine',torch.tensor(-np.sin(angles).T,dtype=torch.float32))
        self.register_buffer('mel',torch.tensor(feature.mel_filters,dtype=torch.float32))

    def forward(self,waveform):
        frames=waveform[:,self.frame_index]*32768.
        frames=frames-frames.mean(-1,keepdim=True)
        first=frames[:,:,:1]*(1.-.97)
        rest=frames[:,:,1:]-.97*frames[:,:,:-1]
        frames=torch.cat([first,rest],-1)*self.window
        real=frames@self.cosine;imag=frames@self.sine
        power=real*real+imag*imag
        bank=torch.log(torch.clamp(power@self.mel,min=1.192092955078125e-7))
        centered=bank-bank.mean(1,keepdim=True)
        # Checkpoint frontend uses sample variance (ddof=1), not population variance.
        variance=(centered*centered).sum(1,keepdim=True)/(self.frames-1)
        normalized=centered/torch.sqrt(variance+1e-7)
        if self.frames%2:normalized=F.pad(normalized,(0,0,0,1),value=1.)
        return normalized.reshape(waveform.shape[0],(self.frames+1)//2,160)
