"""DSEC streams for training and for FlexTune pseudo-label generation."""
import numpy as np

from data.dsec_utils.dsec_data import DSEC
from data.dsec_utils.dsec_det.io import extract_from_h5_by_timewindow
from data.dsec_utils.utils import (
    construct_pairs,
    crop_tracks,
    filter_small_bboxes,
    interpolate_timestamps,
)
from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.utils.types import DataType


def sparse_gt_pairs(timestamps, gt_timestamps):
    """Intervals [midpoint, annotated frame] on the 2x timestamp grid."""
    is_gt = np.any(np.abs(timestamps[:, None] - gt_timestamps[None, :]) <= 1000, axis=1)
    ends = np.flatnonzero(is_gt)
    ends = ends[ends > 0]
    assert np.all(ends % 2 == 0), "annotated frames must lie on even indices of the 2x grid"
    return np.stack((ends - 1, ends), axis=1)


def pseudo_label_pairs(timestamps, gt_timestamps, pseudo_timestamps):
    """Consecutive 2x-grid intervals ending at an annotated or a pseudo-labelled timestamp."""
    keyframes = np.flatnonzero(
        np.any(np.abs(timestamps[:, None] - gt_timestamps) <= 1000, axis=1))
    midpoints = timestamps[1::2]
    labelled = np.flatnonzero(
        np.any(np.abs(midpoints[:, None] - pseudo_timestamps) <= 1000, axis=1))
    return construct_pairs(np.concatenate([keyframes, 2 * labelled + 1]), 2)


class DSECTraining(DSEC):
    """DSEC stream with random sampling and the supervision modes used for training.

    supervision:
        gt         full intervals between annotated frames (Global models)
        sparse_gt  second half of every annotated interval, i.e. [midpoint, GT]
                   (FlexTune Teacher)
        pseudo     every 2x-grid interval that ends at an annotated or a
                   pseudo-labelled timestamp (FlexTune Student)

    With ``generate=True`` the stream visits every 2x-grid interval without
    labels, which is used to run the Teacher; ``reverse_time`` runs it backwards.
    """

    def __init__(self, dataset, sequence_name, mode="stream", supervision="gt",
                 sequence_length=5, transform=None, min_bbox_diag=0,
                 min_bbox_height=0, high_window_ratio=0.5,
                 high_window_crop="random", high_window_seed=42, class_mode="2class",
                 generate=False, reverse_time=False):
        super().__init__(
            dataset,
            sequence_name,
            sequence_length=sequence_length,
            min_bbox_diag=min_bbox_diag,
            min_bbox_height=min_bbox_height,
            high_window_ratio=high_window_ratio,
            high_window_crop=high_window_crop,
            high_window_seed=high_window_seed,
            class_mode=class_mode,
        )
        if mode not in ("stream", "random"):
            raise ValueError(f"Unsupported mode: {mode}")
        if supervision not in ("gt", "sparse_gt", "pseudo"):
            raise ValueError(f"Unsupported supervision: {supervision}")
        if reverse_time and not generate:
            raise ValueError("reverse_time is only used for pseudo-label generation")
        self.supervision = supervision
        self.transform = transform
        self.generate = generate
        self.reverse_time = reverse_time
        self.highres_timestamps, _ = interpolate_timestamps(dataset, sequence_name)
        self.pseudo_mask = None

        # The parent already built self.image_index_pairs for the "gt" mode.
        self.input_timestamps = self.image_timestamps
        if generate:
            num_samples = len(self.highres_timestamps) - 1
        else:
            tracks = dataset.directories[sequence_name].tracks
            gt_timestamps = tracks.tracks["t"][self.gt_mask].astype(np.int64)
            if supervision == "sparse_gt":
                self.input_timestamps = self.highres_timestamps
                self.image_index_pairs = sparse_gt_pairs(
                    self.highres_timestamps, gt_timestamps)
            elif supervision == "pseudo":
                self.input_timestamps = self.highres_timestamps
                pseudo_labels = crop_tracks(
                    tracks.pseudo_labels,
                    self.width // self.scale,
                    self.crop_height // self.scale,
                )
                self.pseudo_mask = filter_small_bboxes(
                    pseudo_labels["w"], pseudo_labels["h"],
                    min_bbox_height, min_bbox_diag,
                )
                self.image_index_pairs = pseudo_label_pairs(
                    self.highres_timestamps,
                    gt_timestamps,
                    pseudo_labels[self.pseudo_mask]["t"].astype(np.int64),
                )
            num_samples = len(self.image_index_pairs)

        if mode == "stream":
            self.start_indices = list(range(0, num_samples, self.sequence_length))
            self.stop_indices = self.start_indices[1:] + [num_samples]
        else:
            self.start_indices = list(range(num_samples - self.sequence_length))
            self.stop_indices = [x + self.sequence_length for x in self.start_indices]

    def __getitem__(self, index):
        if self.generate:
            frames = self._generation_frames(index)
        else:
            frames = self._training_frames(index)
        sample = self._collect(frames, is_first_sample=index == 0)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _training_frames(self, index):
        start, stop = self.start_indices[index], self.stop_indices[index]
        directory = self.dataset.directories[self.sequence_name]
        timestamp_pairs = self.input_timestamps[self.image_index_pairs]
        starts = timestamp_pairs[:, 0][start:stop]
        ends = timestamp_pairs[:, 1][start:stop]
        event_batches = extract_from_h5_by_timewindow(
            directory.events.event_file, starts, ends)

        frames = []
        for start_ts, end_ts, events in zip(starts, ends, event_batches):
            image_index = np.searchsorted(
                self.image_timestamps, end_ts - 8000, side="left") - 1
            image = self._preprocess_image(
                self.dataset.get_image(image_index, directory_name=directory.root.name))
            is_gt_frame = bool(np.any(np.abs(end_ts - self.image_timestamps) <= 8000))
            if is_gt_frame:
                detections = self._preprocess_detections(self.dataset.get_tracks(
                    image_index + 1, mask=self.gt_mask, directory_name=directory.root.name))
            else:
                detections = crop_tracks(
                    self.dataset.get_pseudo_labels(
                        end_ts, mask=self.pseudo_mask, directory_name=directory.root.name),
                    self.width // self.scale,
                    self.crop_height // self.scale,
                )
            # The window is closed at the next image timestamp. For a midpoint
            # the extracted events already stop at the midpoint itself.
            window_end = self.image_timestamps[image_index + 1]
            frames.append(dict(
                events=events,
                window=(start_ts, window_end),
                image=image,
                label=self._to_sparse_label(detections),
                timestamp=window_end,
                is_pseudo=not is_gt_frame,
            ))
        return frames

    def _generation_frames(self, index):
        start, stop = self.start_indices[index], self.stop_indices[index]
        directory = self.dataset.directories[self.sequence_name]
        intervals = np.arange(len(self.highres_timestamps) - 1)
        if self.reverse_time:
            intervals = intervals[::-1]
        intervals = intervals[start:stop]
        starts = self.highres_timestamps[intervals]
        ends = self.highres_timestamps[intervals + 1]
        event_batches = extract_from_h5_by_timewindow(
            directory.events.event_file, starts, ends)

        frames = []
        for start_ts, end_ts, events in zip(starts, ends, event_batches):
            # Predict at the end of the interval in the direction of travel.
            target_ts = start_ts if self.reverse_time else end_ts
            image_index = np.searchsorted(
                self.image_timestamps, target_ts - 8000, side="left") - 1
            image_index = max(0, image_index + (0 if self.reverse_time else 1))
            frames.append(dict(
                events=events,
                window=(start_ts, end_ts),
                image=self._preprocess_image(self.dataset.get_image(
                    image_index, directory_name=directory.root.name)),
                label=None,
                timestamp=target_ts,
                is_pseudo=False,
            ))
        return frames

    def _event_representations(self, frame):
        event_a, event_b, window_a, window_b = self._dual_frequency_events(
            frame["events"], *frame["window"])
        if self.reverse_time:
            # Flipping the channel axis reverses the time bins and swaps the
            # polarities, which is what running time backwards does.
            event_a = event_a.flip(0)
            event_b = event_b.flip(0)
        return event_a, event_b, window_a, window_b

    def _collect(self, frames, is_first_sample):
        padding = self.sequence_length - len(frames)
        sample = {
            DataType.OBJLABELS_SEQ: SparselyBatchedObjectLabels(
                [frame["label"] for frame in frames] + [None] * padding),
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: [False] * len(frames) + [True] * padding,
            DataType.IMAGE: [frame["image"] for frame in frames]
                            + [self.image_padding_representation] * padding,
            DataType.TIMESTAMP: [frame["timestamp"] for frame in frames] + [-1] * padding,
            DataType.SEQUENCE_NAME: [self.sequence_name] * len(frames) + [""] * padding,
            DataType.NMS: [-1] * self.sequence_length,
        }

        events = [self._event_representations(frame) for frame in frames]
        event_padding = [self.padding_representation] * padding
        window_padding = [self.padding_window] * padding
        sample.update({
            DataType.EV_REPR_A: [e[0] for e in events] + event_padding,
            DataType.EV_REPR_B: [e[1] for e in events] + event_padding,
            DataType.EV_WINDOW_A: [e[2] for e in events] + window_padding,
            DataType.EV_WINDOW_B: [e[3] for e in events] + window_padding,
        })

        if self.supervision == "pseudo" and not self.generate:
            sample[DataType.IS_PSEUDO_FRAME_SEQ] = (
                [frame["is_pseudo"] for frame in frames] + [False] * padding)
        return sample

    def get_fully_padded_sample(self):
        sample = super().get_fully_padded_sample()
        if self.supervision == "pseudo" and not self.generate:
            sample[DataType.IS_PSEUDO_FRAME_SEQ] = [False] * self.sequence_length
        return sample
