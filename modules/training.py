from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType
from modules.utils.detection import BackboneFeatureSelector, Mode, merge_mixed_batches


class TrainingMixin:
    """Training step and optimizer of the detection module."""

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)
        mode = Mode.TRAIN

        ev_tensor_sequence = data[DataType.EV_REPR_A]
        ev_tensor_b_sequence = data[DataType.EV_REPR_B]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        image_tensor_sequence = data[DataType.IMAGE]
        # Only present when training with pseudo-labels.
        is_pseudo_sequence = data.get(DataType.IS_PSEUDO_FRAME_SEQ)

        self.mode_2_rnn_states[mode].reset(
            worker_id=worker_id, indices_or_bool_tensor=data[DataType.IS_FIRST_SAMPLE])

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        obj_labels = []
        is_pseudo = []
        fusion_losses = []
        for tidx in range(sequence_len):
            ev_tensors = self.input_padder.pad_tensor_ev_repr(
                ev_tensor_sequence[tidx].to(dtype=self.dtype))
            ev_tensors_b = self.input_padder.pad_tensor_ev_repr(
                ev_tensor_b_sequence[tidx].to(dtype=self.dtype))
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, fusion_loss, prev_states = self.mdl.forward_backbone(
                x=ev_tensors,
                image=image_tensor_sequence[tidx],
                previous_states=prev_states,
                x_b=ev_tensors_b,
                return_losses=True,
            )
            current_labels, valid_batch_indices = \
                sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            if len(current_labels) == 0:
                continue
            for labels in current_labels:
                labels.to(device=ev_tensors.device)
            backbone_feature_selector.add_backbone_features(
                backbone_features=backbone_features, selected_indices=valid_batch_indices)
            fusion_losses.append(sum(fusion_loss.values()) / len(fusion_loss))
            obj_labels.extend(current_labels)
            if is_pseudo_sequence is not None:
                flags = torch.as_tensor(is_pseudo_sequence[tidx], dtype=torch.bool)
                # A pseudo-labelled frame without boxes is treated like a GT frame.
                is_pseudo.extend(
                    bool(flag) and len(labels) > 0
                    for flag, labels in zip(flags[valid_batch_indices], current_labels))

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        assert len(obj_labels) > 0
        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(
            obj_label_list=obj_labels, format_='yolox').to(dtype=self.dtype)

        loss_options = {}
        if is_pseudo_sequence is not None:
            pseudo_cfg = self.train_config.pseudo_loss
            loss_options = dict(
                is_pseudo=torch.tensor(is_pseudo, device=labels_yolox.device, dtype=torch.bool),
                pseudo_box_weight=float(pseudo_cfg.box_weight),
                pseudo_negative_weight=float(pseudo_cfg.negative_weight),
                pseudo_loss_weight=float(pseudo_cfg.loss_weight),
            )
        _, losses = self.mdl.forward_detect(
            backbone_features=backbone_feature_selector.get_batched_backbone_features(),
            targets=labels_yolox,
            **loss_options,
        )
        fusion_loss = torch.stack(fusion_losses).mean()
        self.log_dict({f'train/{key}': value for key, value in losses.items()},
                      on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('train/fusion_loss', fusion_loss, on_step=True, on_epoch=True,
                 batch_size=batch_size, sync_dist=True)
        return losses['loss'] + self.train_config.fusion_loss_weight * fusion_loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.mdl.parameters(), lr=self.train_config.learning_rate,
                                      weight_decay=self.train_config.weight_decay)
        scheduler_cfg = self.train_config.lr_scheduler
        if not scheduler_cfg.use:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.train_config.learning_rate,
            total_steps=int(scheduler_cfg.total_steps),
            pct_start=scheduler_cfg.pct_start,
            div_factor=scheduler_cfg.div_factor,
            final_div_factor=scheduler_cfg.final_div_factor / scheduler_cfg.div_factor,
            cycle_momentum=False,
            anneal_strategy='linear',
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
