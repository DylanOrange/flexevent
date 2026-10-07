import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from config.modifier import dynamically_modify_inference_config
from modules.data.training import TrainingDataModule
from modules.detection import Module
from utils.initialization import initialize_event, initialize_rgb, load_weights


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(config: DictConfig):
    pl.seed_everything(int(config.seed), workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy("file_system")
    dynamically_modify_inference_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    module = Module(config)
    if not config.resume:
        if config.training.teacher_checkpoint:
            load_weights(module, config.training.teacher_checkpoint)
        else:
            initialize_rgb(module, config.training.rgb_weights)
            initialize_event(module, config.training.event_weights)

    data = TrainingDataModule(
        config.dataset, config.hardware.num_workers.train,
        config.hardware.num_workers.eval, config.batch_size.train, config.batch_size.eval)
    output = Path(config.output_dir)
    # Global AP alone, or its mean with 180 Hz AP when interframe validation is on.
    monitor = "val/score" if config.dataset.eval.interframe else "val/AP"
    callbacks = [
        ModelCheckpoint(
            dirpath=output / "checkpoints", monitor=monitor, mode="max",
            save_top_k=config.checkpointing.save_top_k, save_last=True,
            filename="step={step}", auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = CSVLogger(str(output), name="metrics")
    if config.wandb.enable:
        logger = [logger, WandbLogger(project=config.wandb.project, save_dir=str(output))]
    devices = config.hardware.gpus
    devices = [int(devices)] if isinstance(devices, int) else list(devices)
    trainer = pl.Trainer(
        accelerator="gpu", devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True) if len(devices) > 1 else None,
        precision=config.precision, max_steps=config.training.max_steps,
        max_epochs=-1, gradient_clip_val=config.training.gradient_clip_val,
        callbacks=callbacks, logger=logger, default_root_dir=str(output),
        val_check_interval=config.validation.every_n_steps,
        check_val_every_n_epoch=None, num_sanity_val_steps=0,
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.validation.limit_val_batches,
        log_every_n_steps=10, replace_sampler_ddp=False,
    )
    trainer.fit(module, datamodule=data, ckpt_path=config.resume)


if __name__ == "__main__":
    main()
