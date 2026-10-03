"""Compare partitioned AdamW against ordinary AdamW, including clipping/state."""
import copy
import torch
from common import WORK,dump
from offload import PartialCPUAdamW

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.Module()
        self.encoder.layers=torch.nn.ModuleList([torch.nn.Linear(7,7),torch.nn.Linear(7,7)])
    def forward(self,x):
        for layer in self.encoder.layers:x=torch.nn.functional.gelu(layer(x))
        return x

def run():
    torch.manual_seed(917)
    model=Toy();head=torch.nn.Linear(7,3)
    reference=copy.deepcopy(model);ref_head=copy.deepcopy(head)
    tested=PartialCPUAdamW(model,head,1)
    expected=torch.optim.AdamW([{'params':reference.parameters(),'lr':2e-5},
                               {'params':ref_head.parameters(),'lr':1e-4}],weight_decay=.01,foreach=False)
    records=[]
    for step in range(12):
        x=torch.randn(4,7);target=torch.randn(4,3)
        for m,h,opt in [(model,head,tested),(reference,ref_head,expected)]:
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(h(m(x)),target).backward()
            torch.nn.utils.clip_grad_norm_(list(m.parameters())+list(h.parameters()),.1)
            opt.param_groups[0]['lr']=2e-5*(step+1)/12
            opt.param_groups[1]['lr']=1e-4*(step+1)/12
            opt.step()
        a=list(model.parameters())+list(head.parameters());b=list(reference.parameters())+list(ref_head.parameters())
        diff=max((p-q).abs().max().item() for p,q in zip(a,b))
        assert diff<1e-7,(step,diff)
        records.append({'step':step+1,'max_parameter_difference':diff})
    dump(WORK/'adamw-offload-verification.json',{'reference':'ordinary torch.optim.AdamW foreach=False',
          'placement':'CPU reference verifies partition/update/state logic; real CUDA/CPU placement is measured by the pilot',
          'steps':records})
    print('OFFLOAD_ADAMW_VERIFIED',max(x['max_parameter_difference'] for x in records))

if __name__=='__main__':run()
