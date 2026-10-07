from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch


class DataType(Enum):
    EV_REPR_A = auto()
    EV_REPR_B = auto()
    EV_WINDOW_A = auto()
    EV_WINDOW_B = auto()
    IMAGE = auto()
    OBJLABELS_SEQ = auto()
    IS_PADDED_MASK = auto()
    IS_FIRST_SAMPLE = auto()
    TIMESTAMP = auto()
    SEQUENCE_NAME = auto()
    NMS = auto()
    IS_PSEUDO_FRAME_SEQ = auto()
    TOKEN_MASK = auto()


class DatasetSamplingMode(str, Enum):
    RANDOM = "random"
    STREAM = "stream"
    MIXED = "mixed"


class ObjDetOutput(Enum):
    LABELS_PROPH = auto()
    PRED_PROPH = auto()
    SKIP_VIZ = auto()


LstmState = Optional[Tuple[torch.Tensor, torch.Tensor]]
LstmStates = List[LstmState]
FeatureMap = torch.Tensor
BackboneFeatures = Dict[int, torch.Tensor]

LoaderDataDictGenX = Dict[DataType, object]
