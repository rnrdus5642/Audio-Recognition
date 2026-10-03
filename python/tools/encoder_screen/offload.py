"""FP32 AdamW with selected layers' optimizer state and updates on the CPU.

This preserves AdamW's equations, master-weight precision, learning rate and
global gradient clipping; it changes placement and transfer cost. It is an
isolated pilot option for the 1B Korean encoder on this 16 GB GPU.
"""
import re
import torch


class PartialCPUAdamW:
    def __init__(self, model, head, layers, encoder_lr=2e-5, head_lr=1e-4):
        self.pairs=[];gpu=[]
        for name,p in model.named_parameters():
            if not p.requires_grad:continue
            match=re.match(r'encoder\.layers\.(\d+)\.',name)
            if match and int(match.group(1))<layers:
                cpu=torch.nn.Parameter(p.detach().cpu().clone())
                self.pairs.append((p,cpu))
            else:gpu.append(p)
        assert self.pairs, 'No encoder layers matched the offload rule'
        self.gpu=torch.optim.AdamW([{'params':gpu,'lr':encoder_lr},
                                   {'params':head.parameters(),'lr':head_lr}],weight_decay=.01,foreach=False)
        self.cpu=torch.optim.AdamW([p for _,p in self.pairs],lr=encoder_lr,weight_decay=.01,foreach=False)
        self.offloaded_parameters=sum(p.numel() for _,p in self.pairs)

    @property
    def param_groups(self):return self.gpu.param_groups

    def zero_grad(self,set_to_none=True):
        self.gpu.zero_grad(set_to_none=set_to_none)
        self.cpu.zero_grad(set_to_none=set_to_none)
        for gpu,_ in self.pairs:gpu.grad=None

    @torch.no_grad()
    def step(self):
        self.cpu.param_groups[0]['lr']=self.gpu.param_groups[0]['lr']
        for gpu,cpu in self.pairs:
            cpu.grad=None if gpu.grad is None else gpu.grad.detach().cpu()
            gpu.grad=None
        self.gpu.step();self.cpu.step()
        for gpu,cpu in self.pairs:
            gpu.copy_(cpu)
            cpu.grad=None
