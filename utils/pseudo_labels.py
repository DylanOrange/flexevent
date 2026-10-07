"""FlexTune pseudo-label construction from teacher predictions."""
import numpy as np
import torch
import torchvision.ops as ops

from data.dsec_utils.utils import crop_tracks, map_classes, rescale_tracks
from modules.tracking.linear import LinearTracker

# Detections are stored at half resolution and cropped to the DSEC-Det height.
WIDTH, HEIGHT = 320, 215

PREDICTION_DTYPE = np.dtype([
    ('t', '<i8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'),
    ('class_id', '<u4'), ('track_id', '<u4'), ('class_confidence', '<f4'),
    ('object_confidence', '<f4'), ('detection_confidence', '<f4'),
])

LABEL_DTYPE = np.dtype({
    'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'class_confidence', 'track_id'],
    'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<f4', '<u4'],
    'offsets': [0, 8, 12, 16, 20, 24, 28, 32],
    'itemsize': 40,
})


def predictions_to_array(predictions, timestamps):
    """Convert postprocessed YOLOX outputs (x1, y1, x2, y2, obj, cls, class_id)."""
    arrays = []
    for prediction, timestamp in zip(predictions, timestamps):
        count = 0 if prediction is None else len(prediction)
        array = np.zeros((count,), dtype=PREDICTION_DTYPE)
        if count:
            values = prediction.detach().cpu().numpy()
            array['t'] = int(timestamp)
            array['x'] = values[:, 0]
            array['y'] = values[:, 1]
            array['w'] = values[:, 2] - values[:, 0]
            array['h'] = values[:, 3] - values[:, 1]
            array['class_id'] = values[:, 6]
            array['object_confidence'] = values[:, 4]
            array['class_confidence'] = values[:, 5]
            array['detection_confidence'] = values[:, 4] * values[:, 5]
        arrays.append(array)
    return arrays


def to_label_dtype(boxes):
    converted = np.empty(boxes.shape, dtype=LABEL_DTYPE)
    for field in boxes.dtype.names:
        if field in converted.dtype.names:
            converted[field] = boxes[field]
    return converted


def preprocess_gt(tracks, class_mapping):
    """Bring official DSEC tracks to the resolution and classes of the detector."""
    tracks = crop_tracks(rescale_tracks(tracks, 2), WIDTH, HEIGHT)
    tracks['class_id'], keep = map_classes(tracks['class_id'], class_mapping)
    return tracks[keep]


def filter_predictions(boxes, num_classes, threshold, class_thresholds=None):
    thresholds = np.full(len(boxes), threshold, dtype=np.float32)
    for class_id, class_threshold in (class_thresholds or {}).items():
        thresholds[boxes['class_id'] == class_id] = class_threshold
    keep = (boxes['class_id'] < num_classes) & (boxes['class_confidence'] > thresholds)
    return boxes[keep]


def valid_box_mask(boxes):
    diagonal = np.sqrt(boxes['w'] ** 2 + boxes['h'] ** 2)
    finite = np.isfinite(boxes['x']) & np.isfinite(boxes['y']) \
        & np.isfinite(boxes['w']) & np.isfinite(boxes['h'])
    return finite & (boxes['w'] > 10) & (boxes['h'] > 10) & (diagonal > 20)


def merge_directions(forward, reverse, nms_threshold):
    """Class-wise NMS over the union of forward and reverse predictions per timestamp."""
    merged = np.concatenate((forward, reverse))
    kept = []
    for timestamp in np.unique(merged['t']):
        frame_indices = np.flatnonzero(merged['t'] == timestamp)
        frame = merged[frame_indices]
        for class_id in np.unique(frame['class_id']):
            class_indices = np.flatnonzero(frame['class_id'] == class_id)
            boxes = frame[class_indices]
            xyxy = np.stack((boxes['x'], boxes['y'],
                             boxes['x'] + boxes['w'], boxes['y'] + boxes['h']), axis=1)
            keep = ops.batched_nms(
                torch.tensor(xyxy, dtype=torch.float32),
                torch.tensor(boxes['class_confidence'], dtype=torch.float32),
                torch.tensor(boxes['class_id'].astype(np.int32), dtype=torch.int32),
                nms_threshold).numpy()
            kept.extend(frame_indices[class_indices[keep]].tolist())
    merged = merged[np.asarray(kept, dtype=np.int64)]
    return merged[np.argsort(merged, order='t', kind='stable')]


def _closest_timestamp(timestamp, candidates, max_diff=8000):
    diffs = np.abs(candidates - timestamp)
    index = np.argmin(diffs)
    return candidates[index] if diffs[index] <= max_diff else None


def _tracker_input(boxes):
    """[x, y, w, h] boxes to [cx, cy, w, h, class_id]."""
    return np.stack([boxes['x'] + 0.5 * boxes['w'], boxes['y'] + 0.5 * boxes['h'],
                     boxes['w'], boxes['h'], boxes['class_id']], axis=-1)


def track_filter(timestamps, gt, pseudo, min_hits=6):
    """Track GT and pseudo boxes jointly and drop pseudo boxes on short tracks.

    A pseudo box is kept if its track contains a GT box, has at least
    ``min_hits`` hits, or is still alive at the end of the sequence.
    """
    tracker = LinearTracker(img_hw=(HEIGHT, WIDTH))
    gt_ts = gt['t'].astype(np.int64)
    pseudo_ts = pseudo['t'].astype(np.int64)
    frames, frame_is_pseudo = [], []
    for frame_idx, timestamp in enumerate(timestamps):
        closest_gt = _closest_timestamp(timestamp, gt_ts)
        if closest_gt is not None:
            boxes = gt[gt_ts == closest_gt]
            tracker.update(frame_idx=frame_idx, dets=_tracker_input(boxes),
                           is_gt=np.ones(len(boxes), dtype=bool))
            frames.append(boxes)
            frame_is_pseudo.append(False)
        if timestamp in pseudo_ts:
            boxes = crop_tracks(pseudo[pseudo_ts == timestamp], WIDTH, HEIGHT)
            tracker.update(frame_idx=frame_idx, dets=_tracker_input(boxes),
                           is_gt=np.zeros(len(boxes), dtype=bool))
            frames.append(boxes)
            frame_is_pseudo.append(True)
        elif closest_gt is None:
            tracker.update(frame_idx)
    tracker.finish()

    kept_frames = []
    box_idx = 0
    for boxes, is_pseudo in zip(frames, frame_is_pseudo):
        keep = np.ones(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            track = tracker.get_bbox_tracker(box_idx)
            box_idx += 1
            if is_pseudo and not (track.is_gt or track.hits >= min_hits or not track.done):
                keep[i] = False
        kept_frames.append(boxes[keep])
    return kept_frames


def build_label_stream(timestamps, gt, pseudo, num_classes, min_hits=6):
    """Return GT and filtered pseudo boxes on the 2x timestamp grid, sorted by time."""
    frames = track_filter(timestamps, gt, pseudo, min_hits)
    if not frames:
        return np.empty(0, dtype=LABEL_DTYPE)
    labels = to_label_dtype(np.concatenate(frames))
    index = np.searchsorted(timestamps, labels['t'])
    left = timestamps[np.clip(index - 1, 0, len(timestamps) - 1)]
    right = timestamps[np.clip(index, 0, len(timestamps) - 1)]
    on_grid = np.minimum(np.abs(labels['t'] - left), np.abs(labels['t'] - right)) <= 1000
    labels = labels[on_grid & valid_box_mask(labels) & (labels['class_id'] < num_classes)]
    return labels[np.argsort(labels, order='t', kind='stable')]
