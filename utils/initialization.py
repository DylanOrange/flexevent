from pathlib import Path

import torch

RVT_PREFIXES = {
    "mdl.backbone.stages.": "mdl.rvt_block.",
    "mdl.fpn.": "mdl.rvt_fpn.",
    "mdl.yolox_head.": "mdl.yolox_head.",
}
NEW_BLOCK_LAYERS = (".moe_conv_layer.", ".embedder.")


def _state_dict(path):
    # Lightning checkpoints are pickled, so weights_only=True fails on torch>=2.6.
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def _load_merged(module, partial):
    state = module.state_dict()
    state.update(partial)
    module.load_state_dict(state)


def load_weights(module, path):
    module.load_state_dict(_state_dict(path))


def initialize_rgb(module, path):
    from detectron2.checkpoint import DetectionCheckpointer

    incompatible = DetectionCheckpointer(module.mdl.rgb_backbone).load(str(Path(path)))
    missing = [key for key in getattr(incompatible, "missing_keys", ())
               if not key.startswith(("fpn_lateral", "fpn_output"))]
    assert not missing, f"RGB weights missing from {path}: {missing}"


def initialize_event(module, path):
    target = module.state_dict()
    imported = {}
    for key, value in _state_dict(path).items():
        for old, new in RVT_PREFIXES.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        else:
            continue
        if ".cls_preds." in key:
            continue
        assert key in target, f"Unexpected RVT tensor: {key}"
        assert value.shape == target[key].shape, f"Shape mismatch: {key}"
        imported[key] = value

    expected = {
        key for key in target
        if key.startswith(tuple(RVT_PREFIXES.values()))
        and ".cls_preds." not in key
        and not any(n in key for n in NEW_BLOCK_LAYERS)
    }
    missing = sorted(expected - set(imported))
    assert not missing, f"RVT weights missing from {path}: {missing}"
    _load_merged(module, imported)
