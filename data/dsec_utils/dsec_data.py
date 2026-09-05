import hashlib

import cv2
import numpy as np
import torch
from einops import rearrange
from torchdata.datapipes.map import MapDataPipe

from data.dsec_utils.class_config import get_dsec_class_spec
from data.dsec_utils.dsec_det.io import extract_from_h5_by_timewindow
from data.dsec_utils.utils import (
    compute_class_mapping,
    crop_tracks,
    filter_tracks,
    interpolate_timestamps,
    map_classes,
    rescale_tracks,
)
from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels
from data.utils.representations import StackedHistogram
from data.utils.types import DataType


def interpolate_tracks(first, second, timestamp):
    if len(first) != len(second):
        raise ValueError("Track sets must have equal length for interpolation")
    if len(first) == 0:
        return second
    first = first[first["track_id"].argsort()]
    second = second[second["track_id"].argsort()]
    t0, t1 = first["t"][0], second["t"][0]
    if not t0 < t1:
        raise ValueError("Track timestamps must increase")
    ratio = (timestamp - t0) / (t1 - t0)
    output = first.copy()
    for key in "xywh":
        output[key] = first[key] * (1 - ratio) + second[key] * ratio
    return output


class DSEC(MapDataPipe):
    """One recurrent DSEC stream."""

    def __init__(self, dataset, sequence_name: str, sequence_length=5,
                 min_bbox_diag=0, min_bbox_height=0, cropped_height=430,
                 num_us=-1, dual_frequency=False, high_window_ratio=0.5,
                 high_window_crop="trailing", high_window_seed=42,
                 class_mode="legacy_3"):
        self.dataset = dataset
        self.sequence_name = sequence_name
        self.sequence_length = int(sequence_length)
        self.num_us = float(num_us)
        self.dual_frequency = bool(dual_frequency)
        self.high_window_ratio = float(high_window_ratio)
        self.high_window_crop = str(high_window_crop)
        self.high_window_seed = int(high_window_seed)
        if not 0 < self.high_window_ratio <= 1:
            raise ValueError("high_window_ratio must be in (0, 1]")
        if self.high_window_crop not in {"trailing", "deterministic_random"}:
            raise ValueError(f"Unsupported high_window_crop: {self.high_window_crop}")

        class_spec = get_dsec_class_spec(class_mode)
        self.width = dataset.width
        self.crop_height = int(cropped_height)
        self.scale = 2
        self.event_representation = StackedHistogram(
            bins=10,
            height=self.crop_height // self.scale,
            width=self.width // self.scale,
            count_cutoff=10,
        )
        self.class_remapping = compute_class_mapping(
            class_spec.model_class_names,
            dataset.classes,
            class_spec.raw_to_model_name,
        )
        _, self.image_timestamps = interpolate_timestamps(
            dataset, sequence_name
        )
        self.image_index_pairs, self.gt_mask = filter_tracks(
            dataset=dataset,
            sequence_name=sequence_name,
            image_width=self.width,
            scale=self.scale,
            crop_height=self.crop_height,
            class_remapping=self.class_remapping,
            min_bbox_height=min_bbox_height,
            min_bbox_diag=min_bbox_diag,
            timestamps=self.image_timestamps,
            only_perfect_tracks=self.num_us != -1,
        )
        sample_count = len(self.image_index_pairs)
        self.start_indices = list(
            range(0, sample_count, self.sequence_length)
        )
        self.stop_indices = self.start_indices[1:] + [sample_count]
        self.padding_representation = torch.zeros(
            (20, self.crop_height // self.scale, self.width // self.scale),
            dtype=torch.uint8,
        )
        self.image_padding_representation = torch.zeros(
            (3, self.crop_height // self.scale, self.width // self.scale),
            dtype=torch.uint8,
        )
        self.padding_window = (-1, -1)

    def __len__(self):
        return len(self.start_indices)

    def _preprocess_detections(self, detections):
        detections = crop_tracks(
            rescale_tracks(detections, self.scale),
            self.width // self.scale,
            self.crop_height // self.scale,
        )
        detections["class_id"], keep = map_classes(
            detections["class_id"], self.class_remapping
        )
        return detections[keep]

    def _preprocess_events(self, events):
        keep = events["y"] < self.crop_height // self.scale
        events = {key: value[keep] for key, value in events.items()}
        events["p"] = torch.clamp(events["p"], min=0)
        return self.event_representation.construct(
            x=events["x"],
            y=events["y"],
            pol=events["p"],
            time=events["t"],
        )

    def _preprocess_image(self, image):
        image = image[:self.crop_height]
        image = cv2.resize(
            image,
            (self.width // self.scale, self.crop_height // self.scale),
            interpolation=cv2.INTER_CUBIC,
        )
        return torch.from_numpy(image).permute(2, 0, 1)

    def _to_sparse_label(self, labels):
        arrays = [
            labels[key].astype("float32")
            for key in ObjectLabels._str2idx
        ]
        tensor = torch.from_numpy(rearrange(arrays, "fields L -> L fields"))
        return ObjectLabels(
            tensor,
            input_size_hw=(
                self.crop_height // self.scale,
                self.width // self.scale,
            ),
        )

    @staticmethod
    def _time_crop(events, start, end):
        keep = (events["t"] >= start) & (events["t"] < end)
        return {key: value[keep] for key, value in events.items()}

    def _high_window_offset(self, start, end, maximum):
        if maximum <= 0:
            return 0
        key = (
            f"{self.high_window_seed}\0{self.sequence_name}\0"
            f"{int(start)}\0{int(end)}"
        ).encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") % (maximum + 1)

    def _dual_frequency_events(self, events, low_start, low_end):
        low_start, low_end = int(low_start), int(low_end)
        event_a = self._time_crop(events, low_start, low_end)
        duration = low_end - low_start
        high_duration = min(
            int(round(duration * self.high_window_ratio)), duration
        )
        max_offset = duration - high_duration
        if self.high_window_crop == "deterministic_random":
            high_start = low_start + self._high_window_offset(
                low_start, low_end, max_offset
            )
        else:
            high_start = low_end - high_duration
        high_end = high_start + high_duration
        event_b = self._time_crop(event_a, high_start, high_end)
        return (
            self._preprocess_events(event_a),
            self._preprocess_events(event_b),
            (low_start, low_end),
            (high_start, high_end),
        )

    def __getitem__(self, index):
        start, stop = self.start_indices[index], self.stop_indices[index]
        directory = self.dataset.directories[self.sequence_name]
        timestamp_pairs = self.image_timestamps[self.image_index_pairs]
        starts = timestamp_pairs[:, 0][start:stop]
        ends = timestamp_pairs[:, 1][start:stop]
        event_batches = extract_from_h5_by_timewindow(
            directory.events.event_file, starts, ends
        )

        event_list, event_a_list, event_b_list = [], [], []
        window_a_list, window_b_list = [], []
        label_list, image_list = [], []
        padded, timestamps, sequences, offsets = [], [], [], []
        for start_ts, end_ts, events in zip(starts, ends, event_batches):
            image_index = (
                np.searchsorted(
                    self.image_timestamps, end_ts - 8000, side="left"
                )
                - 1
            )
            image_list.append(
                self._preprocess_image(
                    self.dataset.get_image(
                        image_index, directory_name=directory.root.name
                    )
                )
            )
            detections = self._preprocess_detections(
                self.dataset.get_tracks(
                    image_index + 1,
                    mask=self.gt_mask,
                    directory_name=directory.root.name,
                )
            )
            event_end = self.image_timestamps[image_index + 1]
            if self.num_us >= 0:
                previous = self._preprocess_detections(
                    self.dataset.get_tracks(
                        image_index,
                        mask=self.gt_mask,
                        directory_name=directory.root.name,
                    )
                )
                event_end = self.image_timestamps[image_index] + self.num_us
                events = {
                    key: value[events["t"] < event_end]
                    for key, value in events.items()
                }
                detections = interpolate_tracks(
                    previous, detections, event_end
                )

            label_list.append(self._to_sparse_label(detections))
            if self.dual_frequency:
                event_a, event_b, window_a, window_b = (
                    self._dual_frequency_events(events, start_ts, event_end)
                )
                event_a_list.append(event_a)
                event_b_list.append(event_b)
                window_a_list.append(window_a)
                window_b_list.append(window_b)
            else:
                event_list.append(self._preprocess_events(events))
            padded.append(False)
            timestamps.append(event_end)
            sequences.append(self.sequence_name)
            offsets.append(self.num_us)

        padding = self.sequence_length - len(label_list)
        if padding:
            padded.extend([True] * padding)
            if self.dual_frequency:
                event_a_list.extend([self.padding_representation] * padding)
                event_b_list.extend([self.padding_representation] * padding)
                window_a_list.extend([self.padding_window] * padding)
                window_b_list.extend([self.padding_window] * padding)
            else:
                event_list.extend([self.padding_representation] * padding)
            image_list.extend([self.image_padding_representation] * padding)
            label_list.extend([None] * padding)
            timestamps.extend([-1] * padding)
            sequences.extend([""] * padding)
            offsets.extend([-1] * padding)

        output = {
            DataType.OBJLABELS_SEQ: SparselyBatchedObjectLabels(label_list),
            DataType.IS_FIRST_SAMPLE: index == 0,
            DataType.IS_PADDED_MASK: padded,
            DataType.IMAGE: image_list,
            DataType.TIMESTAMP: timestamps,
            DataType.SEQUENCE_NAME: sequences,
            DataType.NMS: offsets,
        }
        if self.dual_frequency:
            output.update({
                DataType.EV_REPR_A: event_a_list,
                DataType.EV_REPR_B: event_b_list,
                DataType.EV_WINDOW_A: window_a_list,
                DataType.EV_WINDOW_B: window_b_list,
            })
        else:
            output[DataType.EV_REPR] = event_list
        return output

    def get_fully_padded_sample(self):
        labels = [None] * self.sequence_length
        events = [self.padding_representation] * self.sequence_length
        output = {
            DataType.OBJLABELS_SEQ: SparselyBatchedObjectLabels(labels),
            DataType.IS_FIRST_SAMPLE: False,
            DataType.IS_PADDED_MASK: [True] * self.sequence_length,
            DataType.IMAGE: [self.image_padding_representation]
                            * self.sequence_length,
            DataType.TIMESTAMP: [-1] * self.sequence_length,
            DataType.SEQUENCE_NAME: [""] * self.sequence_length,
            DataType.NMS: [-1] * self.sequence_length,
        }
        if self.dual_frequency:
            output.update({
                DataType.EV_REPR_A: events,
                DataType.EV_REPR_B: list(events),
                DataType.EV_WINDOW_A: [self.padding_window]
                                      * self.sequence_length,
                DataType.EV_WINDOW_B: [self.padding_window]
                                      * self.sequence_length,
            })
        else:
            output[DataType.EV_REPR] = events
        return output
