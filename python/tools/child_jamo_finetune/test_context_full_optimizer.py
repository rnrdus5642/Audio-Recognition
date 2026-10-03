"""The all-layer CPU-state configuration must preserve AdamW and resume."""
import copy

import torch

from optim import PartialCPUAdamW


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])

    def forward(self, x):
        x = self.projection(x)
        for layer in self.encoder.layers:
            x = torch.tanh(layer(x))
        return x


def update(model, head, optimizer, x, y, partial):
    optimizer.zero_grad() if partial else optimizer.zero_grad(set_to_none=True)
    (head(model(x)) - y).square().mean().backward()
    optimizer.step()


def test_all_layer_offload_matches_adamw_and_resumes_with_both_moment_sets():
    torch.manual_seed(92801)
    model, head = TinyEncoder(), torch.nn.Linear(4, 3)
    reference_model, reference_head = copy.deepcopy(model), copy.deepcopy(head)
    optimizer = PartialCPUAdamW(model, head, 48, 1e-3, 2e-3, foreach=True)
    reference = torch.optim.AdamW([{'params': reference_model.parameters(), 'lr': 1e-3},
                                  {'params': reference_head.parameters(), 'lr': 2e-3}],
                                 weight_decay=.01, foreach=True)
    x, y = torch.randn(7, 4), torch.randn(7, 3)
    for _ in range(3):
        update(model, head, optimizer, x, y, True)
        update(reference_model, reference_head, reference, x, y, False)
    saved = copy.deepcopy(optimizer.state_dict())
    resumed_model, resumed_head = copy.deepcopy(model), copy.deepcopy(head)
    resumed = PartialCPUAdamW(resumed_model, resumed_head, 48, 1e-3, 2e-3, foreach=True)
    resumed.load_state_dict(saved)
    for _ in range(2):
        update(model, head, optimizer, x, y, True)
        update(resumed_model, resumed_head, resumed, x, y, True)
        update(reference_model, reference_head, reference, x, y, False)
    for parameters in zip(list(model.parameters()) + list(head.parameters()),
                          list(resumed_model.parameters()) + list(resumed_head.parameters()),
                          list(reference_model.parameters()) + list(reference_head.parameters()), strict=True):
        torch.testing.assert_close(parameters[0], parameters[1], rtol=0, atol=0)
        torch.testing.assert_close(parameters[0], parameters[2], rtol=1e-6, atol=1e-7)
