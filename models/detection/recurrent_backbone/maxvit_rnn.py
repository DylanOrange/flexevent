from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from detectron2.modeling import build_backbone
from detectron2.structures import ImageList

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates
from models.layers.rnn import DWSConvLSTM2d
from models.layers.maxvit.maxvit import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    get_downsample_layer_Cf2Cl,
    PartitionType)

from models.layers.maxvit.moe_inference import MoEConv
from .base import BaseDetector


class RNNDetector(BaseDetector):
    def __init__(self, mdl_config: DictConfig, rgb_config: DictConfig):
        super().__init__()

        ###### Config ######
        in_channels = mdl_config.input_channels#20
        embed_dim = mdl_config.embed_dim#64
        self.use_image_branch = bool(mdl_config.get('use_image_branch', True))
        self.dual_frequency_fusion = str(
            mdl_config.get('dual_frequency_fusion', 'mean')
        )
        assert self.dual_frequency_fusion in {'sum', 'mean'}, \
            self.dual_frequency_fusion
        dim_multiplier_per_stage = tuple(mdl_config.dim_multiplier)
        num_blocks_per_stage = tuple(mdl_config.num_blocks)#[1, 1, 1, 1]
        T_max_chrono_init_per_stage = tuple(mdl_config.T_max_chrono_init)#[4, 8, 16, 32]
        enable_masking = mdl_config.enable_masking

        num_stages = len(num_blocks_per_stage)
        assert num_stages == 4

        assert isinstance(embed_dim, int)
        assert num_stages == len(dim_multiplier_per_stage)
        assert num_stages == len(num_blocks_per_stage)
        assert num_stages == len(T_max_chrono_init_per_stage)

        ###### Compile if requested ######
        compile_cfg = mdl_config.get('compile', None)
        if compile_cfg is not None:
            compile_mdl = compile_cfg.enable
            if compile_mdl and th_compile is not None:
                compile_args = OmegaConf.to_container(compile_cfg.args, resolve=True, throw_on_missing=True)
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print('Could not compile backbone because torch.compile is not available')
        ##################################

        if self.use_image_branch:
            # Image backbone for Event+RGB FlexFuse.
            self.image_input_format = str(
                rgb_config.get('input_format', 'legacy_bgr'))
            assert self.image_input_format in {'legacy_bgr', 'rgb'}, \
                f'Unsupported image input format: {self.image_input_format}'
            self.image_backbone = build_backbone(rgb_config)
            assert not rgb_config.get('load_pretrained_weights', False)
            self.in_features = rgb_config.MODEL.ROI_HEADS.IN_FEATURES
            self.size_divisibility = self.image_backbone.size_divisibility

            pixel_mean = th.Tensor(rgb_config.MODEL.PIXEL_MEAN).view(3, 1, 1)
            pixel_std = th.Tensor(rgb_config.MODEL.PIXEL_STD).view(3, 1, 1)
            self.normalizer = lambda x: (x - pixel_mean.to(x.device)) / pixel_std.to(x.device)

        else:
            self.image_backbone = None
            self.in_features = ()
            self.size_divisibility = 1

        input_dim = in_channels
        patch_size = mdl_config.stem.patch_size#4
        stride = 1
        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]#64,128,256,512
        
        image_fusion = [self.use_image_branch] * num_stages
        self.stages = nn.ModuleList()
        self.strides = []
        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
                enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            spatial_downsample_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]
            enable_masking_in_stage = enable_masking and stage_idx == 0
            stage = RNNDetectorStage(dim_in=input_dim,
                                     stage_dim=stage_dim,
                                     spatial_downsample_factor=spatial_downsample_factor,
                                     num_blocks=num_blocks,
                                     enable_token_masking=enable_masking_in_stage,
                                     T_max_chrono_init=T_max_chrono_init_stage,
                                     stage_cfg=mdl_config.stage,
                                     image_fusion = image_fusion[stage_idx])
            stride = stride * spatial_downsample_factor
            self.strides.append(stride)

            input_dim = stage_dim
            self.stages.append(stage)

        self.num_stages = num_stages

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < self.num_stages, stage_indices
        return tuple(self.stage_dims[stage_idx] for stage_idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < self.num_stages, stage_indices
        return tuple(self.strides[stage_idx] for stage_idx in stage_indices)

    def take_model_components(self) -> Tuple[nn.ModuleList, Optional[nn.Module]]:
        rvt_block = self._modules.pop('stages')
        rgb_backbone = self._modules.pop('image_backbone', None)
        return rvt_block, rgb_backbone

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        if self.image_input_format == 'rgb':
            batched_inputs = [x[[2, 1, 0], ...] for x in batched_inputs]
        images = [self.normalizer(x) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)# image tensor 2,3,224,320

        return images
    
    def forward(self, x: th.Tensor, image: th.Tensor, prev_states: Optional[LstmStates] = None,
                token_mask: Optional[th.Tensor] = None, x_b: Optional[th.Tensor] = None,
                event_fpn: Optional[nn.Module] = None,
                rvt_block: Optional[nn.ModuleList] = None,
                rgb_backbone: Optional[nn.Module] = None) \
            -> Tuple[BackboneFeatures, LstmStates]:
        assert rvt_block is not None
        dual_frequency = x_b is not None
        pre_fusion_fpn = event_fpn is not None
        if dual_frequency:
            assert x.shape == x_b.shape, (x.shape, x_b.shape)
            if prev_states is None:
                prev_states_a, prev_states_b = None, None
            else:
                assert isinstance(prev_states, tuple) and len(prev_states) == 2
                prev_states_a, prev_states_b = prev_states
            if prev_states_a is None:
                prev_states_a = [None] * self.num_stages
            if prev_states_b is None:
                prev_states_b = [None] * self.num_stages
            assert len(prev_states_a) == len(prev_states_b) == self.num_stages
            states_a: LstmStates = list()
            states_b: LstmStates = list()
        else:
            if prev_states is None:
                prev_states = [None] * self.num_stages
            assert len(prev_states) == self.num_stages
            states: LstmStates = list()

        output: Dict[int, FeatureMap] = {}
        event_output_a: Dict[int, FeatureMap] = {}
        event_output_b: Dict[int, FeatureMap] = {}
        if self.use_image_branch:
            # Process image and extract FPN features.
            images = self.preprocess_image(image)
            assert rgb_backbone is not None
            src = rgb_backbone(images.tensor)
            features = [src[f] for f in self.in_features]
        else:
            features = [None] * self.num_stages

        for stage_idx, stage in enumerate(rvt_block):
            stage_token_mask = token_mask if stage_idx == 0 else None
            fuse_in_stage = not pre_fusion_fpn or stage_idx == 0
            if dual_frequency:
                (x, moe_x_a), state_a = stage(
                    x, prev_states_a[stage_idx], stage_token_mask,
                    features[stage_idx], fuse_image=fuse_in_stage)
                (x_b, moe_x_b), state_b = stage(
                    x_b, prev_states_b[stage_idx], stage_token_mask,
                    features[stage_idx], fuse_image=fuse_in_stage)
                states_a.append(state_a)
                states_b.append(state_b)
                event_output_a[stage_idx + 1] = x
                event_output_b[stage_idx + 1] = x_b
                moe_x = moe_x_a + moe_x_b
                if self.dual_frequency_fusion == 'mean':
                    moe_x = moe_x / 2
            else:
                (x, moe_x), state = stage(
                    x, prev_states[stage_idx], stage_token_mask,
                    features[stage_idx], fuse_image=fuse_in_stage)
                states.append(state)
                event_output_a[stage_idx + 1] = x
            stage_number = stage_idx + 1
            output[stage_number] = moe_x

        if pre_fusion_fpn:
            fpn_stages = tuple(event_fpn.in_features)
            assert fpn_stages == (2, 3, 4), fpn_stages
            event_pyramid_a = event_fpn(event_output_a)
            event_pyramid_b = event_fpn(event_output_b) if dual_frequency else None
            assert len(event_pyramid_a) == len(fpn_stages)
            for pyramid_idx, (stage_number, event_feature_a) in enumerate(
                    zip(fpn_stages, event_pyramid_a)):
                stage = rvt_block[stage_number - 1]
                rgb_feature = features[stage_number - 1]
                fused_a = stage.fuse_features(
                    event_feature_a, rgb_feature)
                if dual_frequency:
                    fused_b = stage.fuse_features(
                        event_pyramid_b[pyramid_idx], rgb_feature)
                    fused = fused_a + fused_b
                    if self.dual_frequency_fusion == 'mean':
                        fused = fused / 2
                else:
                    fused = fused_a
                output[stage_number] = fused
        return output, (states_a, states_b) if dual_frequency else states


class MaxVitAttentionPairCl(nn.Module):
    def __init__(self,
                 dim: int,
                 skip_first_norm: bool,
                 attention_cfg: DictConfig):
        super().__init__()

        self.att_window = PartitionAttentionCl(dim=dim,
                                               partition_type=PartitionType.WINDOW,
                                               attention_cfg=attention_cfg,
                                               skip_first_norm=skip_first_norm)
        self.att_grid = PartitionAttentionCl(dim=dim,
                                             partition_type=PartitionType.GRID,
                                             attention_cfg=attention_cfg,
                                             skip_first_norm=False)

    def forward(self, x):
        x = self.att_window(x)
        x = self.att_grid(x)
        return x


class RNNDetectorStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self,
                 dim_in: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 enable_token_masking: bool,
                 T_max_chrono_init: Optional[int],
                 stage_cfg: DictConfig,
                 image_fusion):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        downsample_cfg = stage_cfg.downsample
        lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention
        self.image_fusion = image_fusion
        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in,
                                                           dim_out=stage_dim,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg)
        blocks = [MaxVitAttentionPairCl(dim=stage_dim,
                                        skip_first_norm=i == 0 and self.downsample_cf2cl.output_is_normed(),
                                        attention_cfg=attention_cfg) for i in range(num_blocks)]
        self.att_blocks = nn.ModuleList(blocks)
        self.lstm = DWSConvLSTM2d(dim=stage_dim,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))

        ###### Mask Token ################
        self.mask_token = nn.Parameter(th.zeros(1, 1, 1, stage_dim),
                                       requires_grad=True) if enable_token_masking else None
        if self.mask_token is not None:
            th.nn.init.normal_(self.mask_token, std=.02)
        ##################################
        if self.image_fusion:
            self.moe_conv_layer = MoEConv(M=2, d=2*stage_dim, K=2)
            self.embedder = nn.Conv2d(in_channels=256, out_channels=stage_dim, kernel_size=1, stride=1, padding=0)

    def fuse_features(self, x: th.Tensor, roi_features: th.Tensor):
        if not self.image_fusion:
            return x
        assert roi_features is not None
        roi_features = self.embedder(roi_features)
        assert roi_features.shape == x.shape, (roi_features.shape, x.shape)
        shared_feature = th.cat([roi_features, x], dim=1)
        gates = self.moe_conv_layer(shared_feature)
        gates = gates.view(-1, 2, 1, 1, 1)
        moe_x = gates[:, 0] * roi_features + gates[:, 1] * x
        return moe_x

    def forward(self, x: th.Tensor,
                h_and_c_previous: Optional[LstmState] = None,
                token_mask: Optional[th.Tensor] = None,
                roi_features=None,
                fuse_image: bool = True) \
            -> Tuple[FeatureMap, LstmState]:
        x = self.downsample_cf2cl(x)
        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            x[token_mask] = self.mask_token

        for blk in self.att_blocks:
            x = blk(x)
        x = nhwC_2_nChw(x) 

        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]

        if fuse_image:
            moe_x = self.fuse_features(x, roi_features)
        else:
            moe_x = x
        return (x, moe_x), h_c_tuple
