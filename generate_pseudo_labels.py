"""Run a FlexTune Teacher over the training sequences in one temporal direction.

Writes OUTPUT_DIR/<direction>/<sequence>.npy with the raw detections at every
timestamp of the 2x grid. Run it for both directions, then build the
pseudo-labels with scripts/build_pseudo_labels.py.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from config.modifier import dynamically_modify_inference_config
from data.dsec_utils.dsec_det.dataset import DSECDet
from data.dsec_utils.dsec_det.io import yaml_file_to_dict
from data.dsec_utils.training import DSECTraining
from data.genx_utils.collate import custom_collate_streaming
from data.genx_utils.dataset_streaming import build_streaming_evaluation_dataset
from data.utils.types import DataType
from models.detection.yolox.postprocess import postprocess
from modules.detection import Module
from modules.utils.detection import BackboneFeatureSelector, Mode
from utils.initialization import load_weights
from utils.pseudo_labels import predictions_to_array


class GenerationDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        dataset_cfg = self.config.dataset
        split_config = yaml_file_to_dict(
            Path(__file__).resolve().parent / 'data/dsec_utils/dsec_split.yaml')
        sequences = list(split_config['train'])
        if self.config.sequences is not None:
            requested = [str(name) for name in self.config.sequences]
            unknown = sorted(set(requested) - set(sequences))
            assert not unknown, f'Not training sequences: {unknown}'
            sequences = requested
        base = DSECDet(root=Path(dataset_cfg.path), split='train', sync='back',
                       split_config=split_config)
        dual = dataset_cfg.flexfuse.dual_frequency
        datapipes = [
            DSECTraining(
                base,
                sequence,
                sequence_length=dataset_cfg.sequence_length,
                min_bbox_diag=dataset_cfg.min_bbox_diag,
                min_bbox_height=dataset_cfg.min_bbox_height,
                high_window_ratio=dual.high_window_ratio,
                high_window_crop='deterministic_random',
                high_window_seed=dual.inference_seed,
                class_mode=dataset_cfg.class_mode,
                generate=True,
                reverse_time=self.config.direction == 'reverse',
            )
            for sequence in sequences
        ]
        self.test_dataset = build_streaming_evaluation_dataset(
            datapipes=datapipes, batch_size=self.config.batch_size.eval)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=None, shuffle=False,
                          num_workers=self.config.hardware.num_workers.eval,
                          pin_memory=False, drop_last=False,
                          collate_fn=custom_collate_streaming)


class Teacher(Module):
    def __init__(self, full_config: DictConfig):
        super().__init__(full_config)
        self.predictions = defaultdict(list)

    def test_step(self, batch, batch_idx):
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)
        mode = Mode.TEST
        ev_sequence = data[DataType.EV_REPR_A]
        padded = data[DataType.IS_PADDED_MASK]

        self.mode_2_rnn_states[mode].reset(
            worker_id=worker_id, indices_or_bool_tensor=data[DataType.IS_FIRST_SAMPLE])
        states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        selector = BackboneFeatureSelector()
        timestamps, sequences = [], []
        for tidx, ev_tensors in enumerate(ev_sequence):
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors.to(dtype=self.dtype))
            ev_tensors_b = self.input_padder.pad_tensor_ev_repr(
                data[DataType.EV_REPR_B][tidx].to(dtype=self.dtype))
            features, states = self.mdl.forward_backbone(
                x=ev_tensors, image=data[DataType.IMAGE][tidx],
                previous_states=states, x_b=ev_tensors_b)
            valid = [i for i, is_padded in enumerate(padded[tidx]) if not bool(is_padded)]
            if valid:
                selector.add_backbone_features(backbone_features=features, selected_indices=valid)
                timestamps.extend(data[DataType.TIMESTAMP][tidx][i] for i in valid)
                sequences.extend(str(data[DataType.SEQUENCE_NAME][tidx][i]) for i in valid)
        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=states)
        if not sequences:
            return

        predictions = self.mdl.forward_detect(
            backbone_features=selector.get_batched_backbone_features())
        predictions = postprocess(
            prediction=predictions,
            num_classes=self.mdl_config.yolox_head.num_classes,
            confidence_threshold=self.mdl_config.postprocess.confidence_threshold,
            nms_threshold=self.mdl_config.postprocess.nms_threshold,
        )
        for sequence, boxes in zip(sequences, predictions_to_array(predictions, timestamps)):
            self.predictions[sequence].append(boxes)

    def on_test_epoch_end(self):
        output_dir = Path(self.full_config.output_dir) / self.full_config.direction
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            'checkpoint': str(Path(self.full_config.checkpoint).resolve()),
            'direction': self.full_config.direction,
            'class_mode': self.full_config.dataset.class_mode,
            'sequences': sorted(self.predictions),
        }
        for sequence in sorted(self.predictions):
            boxes = np.concatenate(self.predictions[sequence])
            boxes.sort(order='t')
            np.save(output_dir / f'{sequence}.npy', boxes)
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


@hydra.main(config_path='config', config_name='generate', version_base='1.2')
def main(config: DictConfig):
    pl.seed_everything(int(config.seed), workers=True)
    assert config.direction in ('forward', 'reverse'), config.direction
    dynamically_modify_inference_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    module = Teacher(config)
    load_weights(module, config.checkpoint)
    trainer = pl.Trainer(accelerator='gpu', devices=[int(config.hardware.gpus)],
                         logger=False, enable_checkpointing=False,
                         precision=config.precision)
    with torch.inference_mode():
        trainer.test(model=module, datamodule=GenerationDataModule(config))


if __name__ == '__main__':
    main()
