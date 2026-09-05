
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
            self,
            num_classes=80,
            strides=(8, 16, 32),
            in_channels=(256, 512, 1024),
            act="silu",
            depthwise=False,
            compile_cfg: Optional[Dict] = None,
            ignore_label: Optional[int] = None,
    ):
        super().__init__()
        del ignore_label
        self.num_classes = num_classes
        self.decode_in_inference = True
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        conv = DWConv if depthwise else BaseConv

        self.output_strides = None
        self.output_grids = None
        hidden_dim = int(256 * in_channels[-1] / 1024)

        for in_channel in in_channels:
            self.stems.append(BaseConv(in_channel, hidden_dim, 1, 1, act=act))
            self.cls_convs.append(nn.Sequential(
                conv(hidden_dim, hidden_dim, 3, 1, act=act),
                conv(hidden_dim, hidden_dim, 3, 1, act=act),
            ))
            self.reg_convs.append(nn.Sequential(
                conv(hidden_dim, hidden_dim, 3, 1, act=act),
                conv(hidden_dim, hidden_dim, 3, 1, act=act),
            ))
            self.cls_preds.append(nn.Conv2d(hidden_dim, self.num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(hidden_dim, 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(hidden_dim, 1, 1, 1, 0))

        self.strides = strides
        self.initialize_biases(prior_prob=0.01)

        if compile_cfg is not None:
            compile_model = compile_cfg["enable"]
            if compile_model and th_compile is not None:
                self.forward = th_compile(self.forward, **compile_cfg["args"])
            elif compile_model:
                print("Could not compile YOLOXHead because torch.compile is unavailable")

    def initialize_biases(self, prior_prob: float) -> None:
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for conv in self.cls_preds:
            nn.init.constant_(conv.bias, bias_value)
        for conv in self.obj_preds:
            nn.init.constant_(conv.bias, bias_value)

    def forward(self, features):
        outputs = []
        for index, (cls_conv, reg_conv, feature) in enumerate(
                zip(self.cls_convs, self.reg_convs, features)):
            feature = self.stems[index](feature)
            cls_output = self.cls_preds[index](cls_conv(feature)).sigmoid()
            reg_feature = reg_conv(feature)
            reg_output = self.reg_preds[index](reg_feature)
            obj_output = self.obj_preds[index](reg_feature).sigmoid()
            outputs.append(torch.cat([reg_output, obj_output, cls_output], dim=1))

        self.hw = [output.shape[-2:] for output in outputs]
        output = torch.cat(
            [value.flatten(start_dim=2) for value in outputs], dim=2
        ).permute(0, 2, 1)
        if self.decode_in_inference:
            output = self.decode_outputs(output)
        return output

    def decode_outputs(self, outputs):
        if self.output_grids is None:
            assert self.output_strides is None
            dtype = outputs.dtype
            device = outputs.device
            grids = []
            strides = []
            for (height, width), stride in zip(self.hw, self.strides):
                y_grid, x_grid = torch.meshgrid([
                    torch.arange(height, device=device, dtype=dtype),
                    torch.arange(width, device=device, dtype=dtype),
                ], indexing="ij")
                grid = torch.stack((x_grid, y_grid), 2).view(1, -1, 2)
                grids.append(grid)
                strides.append(torch.full(
                    (*grid.shape[:2], 1), stride, device=device, dtype=dtype))
            self.output_grids = torch.cat(grids, dim=1)
            self.output_strides = torch.cat(strides, dim=1)
        return torch.cat([
            (outputs[..., :2] + self.output_grids) * self.output_strides,
            torch.exp(outputs[..., 2:4]) * self.output_strides,
            outputs[..., 4:],
        ], dim=-1)
