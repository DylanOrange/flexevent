from typing import List

from torchdata.datapipes.map import MapDataPipe

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
