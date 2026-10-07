from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch as th

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.utils.types import BackboneFeatures, DatasetSamplingMode, LstmStates


class Mode(Enum):
    TRAIN = auto()
    VAL = auto()
    VAL_HIGH = auto()
    TEST = auto()


mode_2_string = {Mode.TRAIN: "train", Mode.VAL: "val", Mode.VAL_HIGH: "val180", Mode.TEST: "test"}


class BackboneFeatureSelector:
    def __init__(self):
        self.features = {}

    def add_backbone_features(
            self,
            backbone_features: BackboneFeatures,
            selected_indices: Optional[List[int]] = None) -> None:
        if selected_indices is not None:
            assert len(selected_indices) > 0
        for key, value in backbone_features.items():
            selected = value[selected_indices] if selected_indices is not None else value
            self.features.setdefault(key, []).append(selected)

    def get_batched_backbone_features(self) -> Optional[BackboneFeatures]:
        if not self.features:
            return None
        return {key: th.cat(value, dim=0) for key, value in self.features.items()}


class RNNStates:
    def __init__(self):
        self.states = {}

    @classmethod
    def recursive_detach(cls, value: Union[th.Tensor, List, Tuple, Dict]):
        if isinstance(value, th.Tensor):
            return value.detach()
        if isinstance(value, list):
            return [cls.recursive_detach(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.recursive_detach(item) for item in value)
        if isinstance(value, dict):
            return {key: cls.recursive_detach(item) for key, item in value.items()}
        raise NotImplementedError

    @classmethod
    def recursive_reset(
            cls,
            value: Union[th.Tensor, List, Tuple, Dict],
            indices_or_bool_tensor: Optional[Union[List[int], torch.Tensor]] = None):
        if isinstance(value, th.Tensor):
            assert not value.requires_grad
            if indices_or_bool_tensor is None:
                value[:] = 0
            else:
                value[indices_or_bool_tensor] = 0
            return value
        if isinstance(value, list):
            return [cls.recursive_reset(item, indices_or_bool_tensor) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.recursive_reset(item, indices_or_bool_tensor) for item in value)
        if isinstance(value, dict):
            return {
                key: cls.recursive_reset(item, indices_or_bool_tensor)
                for key, item in value.items()
            }
        raise NotImplementedError

    def save_states_and_detach(self, worker_id: int, states: LstmStates) -> None:
        self.states[worker_id] = self.recursive_detach(states)

    def get_states(self, worker_id: int) -> Optional[LstmStates]:
        return self.states.get(worker_id)

    def reset(
            self,
            worker_id: int,
            indices_or_bool_tensor: Optional[Union[List[int], torch.Tensor]] = None) -> None:
        if worker_id in self.states:
            self.states[worker_id] = self.recursive_reset(
                self.states[worker_id], indices_or_bool_tensor)


def mixed_collate_fn(x1: Union[th.Tensor, List[th.Tensor]], x2: Union[th.Tensor, List[th.Tensor]]):
    if isinstance(x1, th.Tensor):
        assert isinstance(x2, th.Tensor)
        return th.cat((x1, x2))
    if isinstance(x1, SparselyBatchedObjectLabels):
        assert isinstance(x2, SparselyBatchedObjectLabels)
        return x1 + x2
    if isinstance(x1, list):
        assert isinstance(x2, list)
        assert len(x1) == len(x2)
        return [mixed_collate_fn(x1=el_1, x2=el_2) for el_1, el_2 in zip(x1, x2)]
    if isinstance(x1, tuple):
        assert isinstance(x2, tuple)
        assert len(x1) == len(x2)
        return tuple(mixed_collate_fn(x1=el_1, x2=el_2) for el_1, el_2 in zip(x1, x2))
    if isinstance(x1, str):
        assert isinstance(x2, str)
        return x1, x2
    raise NotImplementedError


def merge_mixed_batches(batch: Dict[str, Any]):
    """Concatenate the streaming and the random batch of mixed sampling."""
    if 'data' in batch:
        return batch
    rnd_data = batch[DatasetSamplingMode.RANDOM]['data']
    stream_batch = batch[DatasetSamplingMode.STREAM]
    stream_data = stream_batch['data']
    assert rnd_data.keys() == stream_data.keys(), (rnd_data.keys(), stream_data.keys())
    return {
        'worker_id': stream_batch['worker_id'],
        'data': {key: mixed_collate_fn(stream_data[key], rnd_data[key]) for key in rnd_data},
    }
