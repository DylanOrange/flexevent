import torch
from torch import nn


class MoEConv(nn.Module):

    def __init__(self, d, M=4, K=1, noisy_gating=True):
        super().__init__()
        del noisy_gating
        self.M = M
        self.k = K
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.softmax = nn.Softmax(1)
        assert self.k <= self.M

    def forward(self, features):
        pooled = self.gap(features).view(features.shape[0], -1)
        logits = pooled @ self.w_gate
        top_logits, top_indices = logits.topk(self.k, dim=1)
        top_gates = self.softmax(top_logits)
        gates = torch.zeros_like(logits).float().scatter(
            1, top_indices, top_gates
        )
        return gates.to(logits.dtype)
