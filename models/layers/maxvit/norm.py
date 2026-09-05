import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            tensor, self.normalized_shape, self.weight, self.bias, self.eps)
