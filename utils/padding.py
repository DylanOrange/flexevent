from typing import Tuple

import torch as th
import torch.nn.functional as F


class InputPadderFromShape:
    def __init__(self, desired_hw: Tuple[int, int]):
        assert isinstance(desired_hw, tuple)
        assert len(desired_hw) == 2
        self.desired_hw = desired_hw

    def pad_tensor_ev_repr(self, ev_repr: th.Tensor) -> th.Tensor:
        height, width = ev_repr.shape[-2:]
        desired_height, desired_width = self.desired_hw
        assert height <= desired_height and width <= desired_width
        return F.pad(
            ev_repr,
            (0, desired_width - width, 0, desired_height - height),
            mode="constant",
            value=0,
        )
