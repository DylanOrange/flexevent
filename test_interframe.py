import os
import sys
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.backends import cuda, cudnn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.modifier import dynamically_modify_inference_config
from modules.utils.fetch import fetch_data_module, fetch_model_module

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy("file_system")

def _load_weights(module: pl.LightningModule, checkpoint: Path) -> None:
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    module.load_state_dict(state_dict)


@hydra.main(config_path="config", config_name="val", version_base="1.2")
def main(config: DictConfig) -> None:
    dynamically_modify_inference_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    assert bool(config.use_test_set), "Set use_test_set=true"
    assert bool(config.validation.interframe), "set validation.interframe=true"
    gpu = config.hardware.gpus
    assert isinstance(gpu, int), "hardware.gpus must be one GPU index"

    module = fetch_model_module(config=config)
    _load_weights(module=module, checkpoint=Path(config.checkpoint))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[gpu],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=None,
        precision=config.precision,
        move_metrics_to_cpu=False,
        limit_test_batches=config.validation.get("limit_test_batches", 1.0),
    )
    with torch.inference_mode():
        configured_offsets = config.validation.get("interframe_num_us", None)
        offsets_us = (
            np.linspace(0, 50_000, 10)
            if configured_offsets is None
            else configured_offsets
        )
        for offset_us in offsets_us:
            print(f"Interframe offset: {float(offset_us):.6f} us")
            config.dataset.num_us = float(offset_us)
            data_module = fetch_data_module(config=config)
            trainer.test(model=module, datamodule=data_module)


def _disable_hydra_outputs() -> None:
    defaults = (
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    )
    missing = []
    for value in defaults:
        key = value.split("=", 1)[0]
        if not any(arg.split("=", 1)[0] == key for arg in sys.argv[1:]):
            missing.append(value)
    first_option = next(
        (index for index, arg in enumerate(sys.argv[1:], start=1) if arg.startswith("--")),
        len(sys.argv),
    )
    sys.argv[first_option:first_option] = missing


if __name__ == "__main__":
    _disable_hydra_outputs()
    main()
