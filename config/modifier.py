import math
from typing import Tuple

from omegaconf import DictConfig, open_dict

from data.dsec_utils.class_config import get_dsec_class_spec
from data.utils.spatial import get_dataloading_hw


def dynamically_modify_inference_config(config: DictConfig) -> None:
    """Resolve input geometry and the selected DSEC class count."""
    with open_dict(config):
        dataset_hw = get_dataloading_hw(dataset_config=config.dataset)
        split = int(config.model.rvt_block.partition_split_32)
        multiple = 32 * split
        model_hw = _round_up_hw(dataset_hw, multiple)
        config.model.rvt_block.in_res_hw = model_hw
        config.model.rvt_block.stage.attention.partition_size = tuple(
            value // multiple for value in model_hw
        )
        config.model.yolox_head.num_classes = get_dsec_class_spec(
            str(config.dataset.class_mode)
        ).num_classes


def _round_up_hw(hw: Tuple[int, int], multiple: int) -> Tuple[int, int]:
    return tuple(math.ceil(value / multiple) * multiple for value in hw)
