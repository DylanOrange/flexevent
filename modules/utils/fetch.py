import pytorch_lightning as pl
from omegaconf import DictConfig

from modules.data.dsec import DSECDataModule
from modules.detection import Module


def fetch_model_module(config: DictConfig) -> pl.LightningModule:
    if config.model.name == "rnndet":
        return Module(config)
    raise NotImplementedError(config.model.name)


def fetch_data_module(config: DictConfig) -> pl.LightningDataModule:
    if config.dataset.name != "dsec":
        raise NotImplementedError(config.dataset.name)
    return DSECDataModule(
        config.dataset,
        num_workers_train=config.hardware.num_workers.eval,
        num_workers_eval=config.hardware.num_workers.eval,
        batch_size_train=config.batch_size.eval,
        batch_size_eval=config.batch_size.eval,
    )
