from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch as th

from data.utils.types import BackboneFeatures, LstmStates


class Mode(Enum):
    TEST = auto()


mode_2_string = {Mode.TEST: "test"}


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
