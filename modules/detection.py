from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.dsec_utils.class_config import get_dsec_class_spec
from data.utils.types import DataType, DatasetSamplingMode
from models.detection.yolox.postprocess import postprocess
from models.detection.yolox_extension.models.detector import YoloXDetector
from modules.utils.detection import BackboneFeatureSelector, Mode, RNNStates, mode_2_string
from utils.evaluation.dsec import DSECEvaluator, to_prophesee
from utils.padding import InputPadderFromShape


class Module(pl.LightningModule):

    def __init__(self, full_config: DictConfig):
        super().__init__()
        self.full_config = full_config
        self.mdl_config = full_config.model
        self.input_padder = InputPadderFromShape(
            desired_hw=tuple(self.mdl_config.rvt_block.in_res_hw))
        self.mdl = YoloXDetector(self.mdl_config)
        self.mode_2_rnn_states = {Mode.TEST: RNNStates()}

    def setup(self, stage: Optional[str] = None) -> None:
        class_spec = get_dsec_class_spec(
            self.full_config.dataset.get("class_mode", "legacy_3"))
        self.mode_2_psee_evaluator = {
            Mode.TEST: DSECEvaluator(
                downsample_by_2=self.full_config.dataset.downsample_by_factor_2,
                classes=class_spec.eval_class_names,
                class_id_mapping=class_spec.model_to_eval_id,
            )
        }
        self.mode_2_sampling_mode = {
            Mode.TEST: self.full_config.dataset.eval.sampling,
        }
        assert self.mode_2_sampling_mode[Mode.TEST] in (
            DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        self.mode_2_hw = {Mode.TEST: None}
        self.mode_2_batch_size = {Mode.TEST: None}
        self.mode_2_rnn_states[Mode.TEST] = RNNStates()

    @staticmethod
    def get_worker_id_from_batch(batch: Any) -> int:
        return batch["worker_id"]

    @staticmethod
    def get_data_from_batch(batch: Any):
        return batch["data"]

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        dual_frequency = DataType.EV_REPR_B in data
        ev_tensor_sequence = (
            data[DataType.EV_REPR_A] if dual_frequency else data[DataType.EV_REPR])
        ev_tensor_b_sequence = data[DataType.EV_REPR_B] if dual_frequency else None
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        image_tensor_sequence = data[DataType.IMAGE]

        self.mode_2_rnn_states[mode].reset(
            worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        previous_states = self.mode_2_rnn_states[mode].get_states(worker_id)
        feature_selector = BackboneFeatureSelector()
        object_labels = []

        for time_index in range(sequence_len):
            collect_predictions = (
                time_index == sequence_len - 1
                or self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM
            )
            event_tensor = self.input_padder.pad_tensor_ev_repr(
                ev_tensor_sequence[time_index].to(dtype=self.dtype))
            event_tensor_b = None
            if dual_frequency:
                event_tensor_b = self.input_padder.pad_tensor_ev_repr(
                    ev_tensor_b_sequence[time_index].to(dtype=self.dtype))
                assert event_tensor.shape == event_tensor_b.shape

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(event_tensor.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == tuple(event_tensor.shape[-2:])

            backbone_features, previous_states = self.mdl.forward_backbone(
                x=event_tensor,
                image=image_tensor_sequence[time_index],
                previous_states=previous_states,
                x_b=event_tensor_b,
            )
            if collect_predictions:
                labels, valid_indices = sparse_obj_labels[
                    time_index].get_valid_labels_and_batch_indices()
                if labels:
                    feature_selector.add_backbone_features(
                        backbone_features, selected_indices=valid_indices)
                    object_labels.extend(labels)

        self.mode_2_rnn_states[mode].save_states_and_detach(
            worker_id=worker_id, states=previous_states)
        if not object_labels:
            return None

        predictions = self.mdl.forward_detect(
            backbone_features=feature_selector.get_batched_backbone_features())
        processed = postprocess(
            prediction=predictions,
            num_classes=self.mdl_config.yolox_head.num_classes,
            confidence_threshold=self.mdl_config.postprocess.confidence_threshold,
            nms_threshold=self.mdl_config.postprocess.nms_threshold,
        )
        labels_prophesee, predictions_prophesee = to_prophesee(
            object_labels, processed)
        evaluator = self.mode_2_psee_evaluator[mode]
        evaluator.add_labels(labels_prophesee)
        evaluator.add_predictions(predictions_prophesee)

        return None

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        evaluator = self.mode_2_psee_evaluator[mode]
        assert evaluator.has_data()
        height, width = self.mode_2_hw[mode]
        metrics = evaluator.evaluate_buffer(img_height=height, img_width=width)
        assert metrics is not None
        self._log_evaluator_metrics(mode=mode, metrics=metrics)
        evaluator.reset_buffer()

    def _log_evaluator_metrics(self, mode: Mode, metrics: Dict[str, float]) -> None:
        prefix = f"{mode_2_string[mode]}/"
        log_dict = {}
        for name, metric in metrics.items():
            value = torch.as_tensor(metric, device=self.device)
            assert value.ndim == 0
            log_dict[f"{prefix}{name}"] = value
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            batch_size=self.mode_2_batch_size[mode],
            sync_dist=True,
        )
