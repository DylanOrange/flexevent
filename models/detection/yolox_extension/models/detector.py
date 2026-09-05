

from typing import Optional

import torch as th
from omegaconf import DictConfig

from data.utils.types import BackboneFeatures, LstmStates
from utils.timers import TimerDummy as CudaTimer
from ...recurrent_backbone import build_recurrent_backbone
from .build import build_yolox_fpn, build_yolox_head


class YoloXDetector(th.nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        rvt_cfg = model_cfg.rvt_block
        fpn_cfg = model_cfg.rvt_fpn
        rgb_cfg = model_cfg.rgb_backbone
        head_cfg = model_cfg.yolox_head

        self.fpn_position = str(fpn_cfg.get("position", "post_fusion"))
        if self.fpn_position not in {"post_fusion", "pre_fusion"}:
            raise ValueError(f"Unsupported model.rvt_fpn.position={self.fpn_position!r}")

        backbone_impl = build_recurrent_backbone(rvt_cfg, rgb_cfg)
        in_channels = backbone_impl.get_stage_dims(fpn_cfg.in_stages)
        strides = backbone_impl.get_strides(fpn_cfg.in_stages)
        self.rvt_block, self.rgb_backbone = backbone_impl.take_model_components()
        self.rvt_fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)
        self.yolox_head = build_yolox_head(
            head_cfg, in_channels=in_channels, strides=strides)
        object.__setattr__(self, '_backbone_impl', backbone_impl)

    def forward_backbone(
            self,
            x: th.Tensor,
            image: th.Tensor,
            previous_states: Optional[LstmStates] = None,
            token_mask: Optional[th.Tensor] = None,
            x_b: Optional[th.Tensor] = None):
        with CudaTimer(device=x.device, timer_name="Backbone"):
            event_fpn = self.rvt_fpn if self.fpn_position == "pre_fusion" else None
            return self._backbone_impl(
                x, image, previous_states, token_mask, x_b,
                event_fpn=event_fpn,
                rvt_block=self.rvt_block,
                rgb_backbone=self.rgb_backbone)

    def forward_detect(
            self,
            backbone_features: BackboneFeatures) -> th.Tensor:
        device = next(iter(backbone_features.values())).device
        if self.fpn_position == "post_fusion":
            with CudaTimer(device=device, timer_name="FPN"):
                features = self.rvt_fpn(backbone_features)
        else:
            features = tuple(
                backbone_features[stage] for stage in self.rvt_fpn.in_features)
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs = self.yolox_head(features)
        return outputs
