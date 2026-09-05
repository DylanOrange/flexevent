from dataclasses import dataclass
from typing import Dict, Tuple


DSEC_NATIVE_CLASSES: Tuple[str, ...] = (
    'pedestrian',
    'rider',
    'car',
    'bus',
    'truck',
    'bicycle',
    'motorcycle',
    'train',
)


@dataclass(frozen=True)
class DSECClassSpec:
    model_class_names: Tuple[str, ...]
    raw_to_model_name: Dict[str, str]
    # One entry per model class. A negative value means "ignore during AP".
    model_to_eval_id: Tuple[int, ...]
    eval_class_names: Tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.model_class_names)


_CLASS_SPECS = {
    # Two-output protocol with DAGR-compatible (car, pedestrian) ordering.
    '2class': DSECClassSpec(
        model_class_names=('car', 'pedestrian'),
        raw_to_model_name={
            'pedestrian': 'pedestrian',
            'car': 'car',
            'bus': 'car',
            'truck': 'car',
        },
        model_to_eval_id=(0, 1),
        eval_class_names=('car', 'pedestrian'),
    ),
    # DSEC defines eight categories, but ``train`` has no annotated boxes.
    '8class': DSECClassSpec(
        model_class_names=DSEC_NATIVE_CLASSES[:-1],
        raw_to_model_name={name: name for name in DSEC_NATIVE_CLASSES[:-1]},
        model_to_eval_id=tuple(range(len(DSEC_NATIVE_CLASSES) - 1)),
        eval_class_names=DSEC_NATIVE_CLASSES[:-1],
    ),
}


def get_dsec_class_spec(class_mode: str) -> DSECClassSpec:
    try:
        return _CLASS_SPECS[str(class_mode)]
    except KeyError as exc:
        choices = ', '.join(sorted(_CLASS_SPECS))
        raise ValueError(
            f'Unknown DSEC class_mode={class_mode!r}; expected one of: {choices}'
        ) from exc
