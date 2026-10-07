import math

from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from data.dsec_utils.dsec_det.dataset import DSECDet
from data.dsec_utils.training import DSECTraining
from data.genx_utils.collate import custom_collate_rnd, custom_collate_streaming
from data.genx_utils.dataset_streaming import build_streaming_train_dataset
from data.utils.augmentor import RandomSpatialAugmentorGenX
from modules.data.dsec import DSECDataModule


class TrainingDataModule(DSECDataModule):
    """Mixed random/streaming training data plus Global and 180 Hz validation."""

    def __init__(self, config, num_workers_train, num_workers_eval,
                 batch_size_train, batch_size_eval):
        super().__init__(config, num_workers_train, num_workers_eval,
                         batch_size_train, batch_size_eval)
        self.batch_size_train = int(batch_size_train)
        self.num_workers_train = int(num_workers_train)
        self.supervision = str(config.train.supervision)
        self.pseudo_labels_root = None
        if self.supervision == "pseudo":
            if not config.train.pseudo_labels_root:
                raise ValueError("supervision=pseudo requires dataset.train.pseudo_labels_root")
            self.pseudo_labels_root = config.train.pseudo_labels_root
        self.spatial_augmentor = RandomSpatialAugmentorGenX(
            dataset_hw=tuple(config.resolution_hw),
            automatic_randomization=True,
            augm_config=config.data_augmentation.random,
        )
        self.train_batch_size = {}
        self.train_workers = {}

    def _split_mixed_batch(self):
        """Distribute batch size and workers between random and streaming sampling."""
        assert self.batch_size_train >= 2, "mixed sampling needs a batch size of at least 2"
        assert self.num_workers_train >= 2, "mixed sampling needs at least 2 workers"
        weight_random = self.dataset_config.train.mixed.w_random
        weight_stream = self.dataset_config.train.mixed.w_stream
        assert weight_random > 0 and weight_stream > 0
        batch_random = min(
            round(self.batch_size_train * weight_random / (weight_stream + weight_random)),
            self.batch_size_train - 1)
        workers_random = min(
            math.ceil(self.num_workers_train * batch_random / self.batch_size_train),
            self.num_workers_train - 1)
        self.train_batch_size = {
            "random": batch_random, "stream": self.batch_size_train - batch_random}
        self.train_workers = {
            "random": workers_random, "stream": self.num_workers_train - workers_random}

    def setup(self, stage=None):
        if stage != "fit":
            super().setup(stage)
            return
        self._split_mixed_batch()
        base = DSECDet(root=self.root, split="train", sync="back",
                       split_config=self.split_config,
                       pseudo_labels_root=self.pseudo_labels_root)
        options = dict(
            supervision=self.supervision,
            sequence_length=self.sequence_length,
            min_bbox_diag=self.min_bbox_diag,
            min_bbox_height=self.min_bbox_height,
            high_window_ratio=self.high_window_ratio,
            high_window_crop="random",
            class_mode=self.class_mode,
        )
        stream_datapipes, random_datapipes = [], []
        sequences = self.dataset_config.train.sequences or self.split_config["train"]
        for sequence in tqdm(sequences, desc="creating train datasets"):
            stream_datapipes.append(DSECTraining(base, sequence, mode="stream", **options))
            random_datapipes.append(DSECTraining(
                base, sequence, mode="random", transform=self.spatial_augmentor, **options))
        self.train_stream_dataset = build_streaming_train_dataset(
            datapipes=stream_datapipes,
            dataset_config=self.dataset_config,
            batch_size=self.train_batch_size["stream"],
            num_workers=self.train_workers["stream"],
        )
        self.train_random_dataset = ConcatDataset(random_datapipes)

        split = str(self.dataset_config.eval.fit_split)
        self.validation_dataset = self._build_dataset(split)
        if self.dataset_config.eval.interframe:
            self.validation_high_dataset = self._build_dataset(split, num_us=1_000_000 / 180)

    def val_dataloader(self):
        global_loader = super().val_dataloader()
        if not self.dataset_config.eval.interframe:
            return global_loader
        high_loader = DataLoader(
            self.validation_high_dataset, batch_size=None, shuffle=False,
            num_workers=self.num_workers_eval, pin_memory=False,
            drop_last=False, collate_fn=custom_collate_streaming)
        return [global_loader, high_loader]

    def train_dataloader(self):
        return {
            "stream": DataLoader(
                self.train_stream_dataset, batch_size=None, shuffle=False,
                num_workers=self.train_workers["stream"], pin_memory=False,
                drop_last=False, collate_fn=custom_collate_streaming),
            "random": DataLoader(
                self.train_random_dataset, batch_size=self.train_batch_size["random"],
                shuffle=True, num_workers=self.train_workers["random"], pin_memory=False,
                drop_last=True, collate_fn=custom_collate_rnd),
        }
