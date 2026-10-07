from functools import partialmethod
from typing import List

from omegaconf import DictConfig
from torchdata.datapipes.map import MapDataPipe

from data.genx_utils.augmentation import RandAugmentIterDataPipe
from data.utils.stream_concat_datapipe import ConcatStreamingDataPipe
from data.utils.stream_sharded_datapipe import ShardedStreamingDataPipe


def build_streaming_evaluation_dataset(
    datapipes: List[MapDataPipe], batch_size: int
) -> ShardedStreamingDataPipe:
    if not datapipes:
        raise ValueError("At least one sequence datapipe is required")
    return ShardedStreamingDataPipe(
        datapipe_list=datapipes,
        batch_size=batch_size,
        fill_value=datapipes[0].get_fully_padded_sample(),
    )


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return NewCls


def build_streaming_train_dataset(datapipes: List[MapDataPipe], dataset_config: DictConfig,
                                  batch_size: int, num_workers: int) -> ConcatStreamingDataPipe:
    assert len(datapipes) > 0
    augmentation_datapipe_type = partialclass(RandAugmentIterDataPipe, dataset_config=dataset_config)
    return ConcatStreamingDataPipe(datapipe_list=datapipes,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   augmentation_pipeline=augmentation_datapipe_type,
                                   print_seed_debug=False)
