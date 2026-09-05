from omegaconf import DictConfig

from .maxvit_rnn import RNNDetector as MaxViTRNNDetector


def build_recurrent_backbone(rvt_cfg: DictConfig, rgb_cfg: DictConfig):
    name = rvt_cfg.name
    if name == 'MaxViTRNN':
        return MaxViTRNNDetector(rvt_cfg, rgb_cfg)
    else:
        raise NotImplementedError
