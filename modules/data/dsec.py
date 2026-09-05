from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dsec_utils.dsec_data import DSEC
from data.dsec_utils.dsec_det.dataset import DSECDet
from data.dsec_utils.dsec_det.io import yaml_file_to_dict
from data.genx_utils.collate import custom_collate_streaming
from data.genx_utils.dataset_streaming import build_streaming_evaluation_dataset


class DSECDataModule(pl.LightningDataModule):
    """DSEC data module."""

    def __init__(self, config, num_workers_train, num_workers_eval,
                 batch_size_train, batch_size_eval):
        super().__init__()
        del num_workers_train, batch_size_train
        self.dataset_config = config
        self.root = Path(config.path).expanduser()
        self.sequence_length = int(config.sequence_length)
        self.batch_size_eval = int(batch_size_eval)
        self.num_workers_eval = int(num_workers_eval)
        self.min_bbox_diag = float(config.min_bbox_diag)
        self.min_bbox_height = float(config.min_bbox_height)
        self.num_us = float(config.get("num_us", -1))
        self.class_mode = str(config.class_mode)

        split_file = Path(__file__).resolve().parents[2] / "data/dsec_utils/dsec_split.yaml"
        self.split_config = yaml_file_to_dict(split_file)

        dual = config.flexfuse.dual_frequency
        self.dual_frequency = bool(dual.enable)
        self.high_window_ratio = float(dual.high_window_ratio)
        self.high_window_crop = str(dual.inference_crop)
        self.high_window_seed = int(dual.inference_seed)
        if self.high_window_crop not in {"trailing", "deterministic_random"}:
            raise ValueError(
                "high_window_crop must be 'trailing' or "
                "'deterministic_random'"
            )

        self.validation_dataset = None
        self.test_dataset = None

    def _build_dataset(self, split: str):
        base = DSECDet(
            root=self.root,
            split=split,
            sync="back",
            split_config=self.split_config,
        )
        datapipes = []
        stream_length = 0
        for sequence in tqdm(
            self.split_config[split],
            desc=f"creating streaming {split} datasets",
        ):
            datapipe = DSEC(
                base,
                sequence_name=sequence,
                sequence_length=self.sequence_length,
                min_bbox_diag=self.min_bbox_diag,
                min_bbox_height=self.min_bbox_height,
                num_us=self.num_us,
                dual_frequency=self.dual_frequency,
                high_window_ratio=self.high_window_ratio,
                high_window_crop=self.high_window_crop,
                high_window_seed=self.high_window_seed,
                class_mode=self.class_mode,
            )
            datapipes.append(datapipe)
            stream_length += len(datapipe)
        print(f"DSEC {split}: {stream_length} streaming samples")
        return build_streaming_evaluation_dataset(
            datapipes=datapipes,
            batch_size=self.batch_size_eval,
        )

    def setup(self, stage=None):
        if stage in (None, "validate"):
            self.validation_dataset = self._build_dataset("val")
        if stage in (None, "test"):
            self.test_dataset = self._build_dataset("test")

    def val_dataloader(self):
        if self.validation_dataset is None:
            raise RuntimeError("Call setup('validate') before val_dataloader()")
        return DataLoader(
            self.validation_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers_eval,
            pin_memory=False,
            drop_last=False,
            collate_fn=custom_collate_streaming,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers_eval,
            pin_memory=False,
            drop_last=False,
            collate_fn=custom_collate_streaming,
        )
