"""AdamW for a 962M encoder on a 16 GB card, and resumable.

Weights and gradients alone take 7.7 GB in fp32; AdamW's two moments
take another 7.7 GB, which does not fit alongside activations. The
moments for the lowest `offload` transformer layers, and the update that
uses them, move to the CPU. AdamW's equations, learning rates, weight
decay and the global gradient clip are untouched - only where the
arithmetic happens changes.

This is the resumable version of the pilot's optimizer. Restarting a
multi-day run from a checkpoint has to reproduce the next update
exactly, so both moment sets are saved and restored. The CPU master
copies are not saved: `step` ends by copying them onto the GPU
parameters, so after any completed update the two are identical and the
masters can be rebuilt from the model's own weights.
"""
import re

import torch


class PartialCPUAdamW:
    ENCODER_GROUP, HEAD_GROUP = 0, 1

    def __init__(self, model, head, offload, encoder_lr, head_lr,
                 weight_decay=0.01, foreach=True):
        self.pairs, resident = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            match = re.match(r'encoder\.layers\.(\d+)\.', name)
            if match and int(match.group(1)) < offload:
                self.pairs.append((p, torch.nn.Parameter(p.detach().cpu().clone())))
            else:
                resident.append(p)
        self.gpu = torch.optim.AdamW(
            [{'params': resident, 'lr': encoder_lr},
             {'params': list(head.parameters()), 'lr': head_lr}],
            weight_decay=weight_decay, foreach=foreach)
        self.cpu = (torch.optim.AdamW([p for _, p in self.pairs], lr=encoder_lr,
                                      weight_decay=weight_decay, foreach=foreach)
                    if self.pairs else None)
        self.offloaded_parameters = sum(p.numel() for _, p in self.pairs)
        self.resident_parameters = sum(p.numel() for p in resident)

    def set_lr(self, encoder_lr, head_lr):
        self.gpu.param_groups[self.ENCODER_GROUP]['lr'] = encoder_lr
        self.gpu.param_groups[self.HEAD_GROUP]['lr'] = head_lr
        if self.cpu is not None:
            self.cpu.param_groups[0]['lr'] = encoder_lr

    def zero_grad(self):
        self.gpu.zero_grad(set_to_none=True)
        for gpu, _ in self.pairs:
            gpu.grad = None

    @torch.no_grad()
    def step(self):
        # During the frozen-head phase no offloaded parameter has a
        # gradient, and AdamW would be a no-op for all of them. Skipping
        # the whole CPU leg avoids moving gigabytes of unchanged master
        # weights back to the GPU on every update of that phase.
        moving = any(gpu.grad is not None for gpu, _ in self.pairs)
        if moving:
            for gpu, cpu in self.pairs:
                cpu.grad = None if gpu.grad is None else gpu.grad.detach().to(
                    'cpu', non_blocking=False)
                gpu.grad = None
        self.gpu.step()
        if self.cpu is not None and moving:
            self.cpu.step()
            for gpu, cpu in self.pairs:
                gpu.copy_(cpu)
                cpu.grad = None

    def state_dict(self):
        return {'gpu': self.gpu.state_dict(),
                'cpu': self.cpu.state_dict() if self.cpu else None}

    def load_state_dict(self, saved):
        self.gpu.load_state_dict(saved['gpu'])
        if self.cpu is not None:
            self.cpu.load_state_dict(saved['cpu'])
            # The masters were equal to the GPU weights when this was
            # written; the loaded weights are the same, so re-sync them.
            for gpu, cpu in self.pairs:
                cpu.data.copy_(gpu.detach().cpu())
