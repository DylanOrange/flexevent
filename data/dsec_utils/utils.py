import numpy as np


def construct_pairs(indices, n=2):
    indices = np.sort(indices)
    rows = [indices[i:i + 1 - n] for i in range(n - 1)] + [indices[n - 1:]]
    stacked = np.stack(rows)
    mask = np.ones_like(stacked[0], dtype=bool)
    for offset, row in enumerate(stacked):
        mask &= stacked[0] + offset == row
    return stacked[..., mask].T


def rescale_tracks(tracks, scale):
    tracks = tracks.copy()
    for key in "xywh":
        tracks[key] /= scale
    return tracks


def crop_tracks(tracks, width, height):
    tracks = tracks.copy()
    x1, y1 = tracks["x"], tracks["y"]
    x2, y2 = x1 + tracks["w"], y1 + tracks["h"]
    x1, x2 = np.clip(x1, 0, width - 1), np.clip(x2, 0, width - 1)
    y1, y2 = np.clip(y1, 0, height - 1), np.clip(y2, 0, height - 1)
    tracks["x"], tracks["y"] = x1, y1
    tracks["w"], tracks["h"] = x2 - x1, y2 - y1
    return tracks


def map_classes(class_ids, old_to_new_mapping):
    new_ids = old_to_new_mapping[class_ids]
    return new_ids, new_ids > -1


def filter_small_bboxes(width, height, bbox_height=20, bbox_diag=30):
    diagonal = np.sqrt(height ** 2 + width ** 2)
    return (
        (diagonal > bbox_diag)
        & (width > bbox_height)
        & (height > bbox_height)
    )


def filter_tracks(dataset, sequence_name, image_width, scale, crop_height,
                  class_remapping, min_bbox_height=0, min_bbox_diag=0,
                  timestamps=None, only_perfect_tracks=False):
    tracks = dataset.directories[sequence_name].tracks.tracks
    scaled = crop_tracks(
        rescale_tracks(tracks, scale), image_width // scale, crop_height // scale
    )
    _, class_mask = map_classes(scaled["class_id"], class_remapping)
    size_mask = filter_small_bboxes(
        scaled["w"], scaled["h"], min_bbox_height, min_bbox_diag
    )
    track_mask = size_mask & class_mask
    label_timestamps = scaled[track_mask]["t"].astype(np.int64)
    correspondence = np.sort(
        np.unique(np.nonzero(np.isin(timestamps, label_timestamps))[0])
    )
    pairs = construct_pairs(correspondence, 2)
    if only_perfect_tracks:
        brackets = timestamps[pairs]
        mapping = compute_img_idx_to_track_idx(tracks["t"], brackets)
        valid = filter_by_only_perfect_tracks(scaled, mapping, track_mask)
        pairs = pairs[valid]
    return pairs, track_mask


def interpolate_timestamps(dataset, sequence_name):
    image_timestamps = dataset.directories[sequence_name].images.timestamps
    count = (len(image_timestamps) - 1) * 2 + 1
    return (
        np.linspace(image_timestamps[0], image_timestamps[-1], count).astype(int),
        image_timestamps,
    )


def filter_by_only_perfect_tracks(tracks, mapping, tracks_mask=None):
    starts, stops = mapping
    valid = np.ones_like(starts[0], dtype=bool)
    for index in range(starts.shape[1]):
        samples = [
            tracks[starts[row][index]:stops[row][index]]
            for row in range(len(starts))
        ]
        if tracks_mask is not None:
            masks = [
                tracks_mask[starts[row][index]:stops[row][index]]
                for row in range(len(starts))
            ]
            samples = [sample[mask] for sample, mask in zip(samples, masks)]
        valid[index] = not _is_invalid_track(samples)
    return valid


def _is_invalid_track(samples):
    samples = [sample[sample["track_id"].argsort()] for sample in samples]
    initial = samples[0]
    for current in samples[1:]:
        if len(current) != len(initial):
            return True
        if not np.array_equal(current["track_id"], initial["track_id"]):
            return True
        if np.min(_compute_iou(initial, current)) < 0.10:
            return True
    return False


def _compute_iou(first, second):
    x1, x2 = first["x"], first["x"] + first["w"]
    y1, y2 = first["y"], first["y"] + first["h"]
    gx1, gx2 = second["x"], second["x"] + second["w"]
    gy1, gy2 = second["y"], second["y"] + second["h"]
    ix1, iy1 = np.maximum(x1, gx1), np.maximum(y1, gy1)
    ix2, iy2 = np.minimum(x2, gx2), np.minimum(y2, gy2)
    intersection = np.zeros_like(x1)
    mask = (iy2 > iy1) & (ix2 > ix1)
    intersection[mask] = (ix2[mask] - ix1[mask]) * (iy2[mask] - iy1[mask])
    union = (
        (x2 - x1) * (y2 - y1)
        + (gx2 - gx1) * (gy2 - gy1)
        - intersection
        + 1e-9
    )
    return intersection / union


def _contiguous_indices(values):
    _, counts = np.unique(values, return_counts=True)
    indices = np.concatenate((np.array([0]), counts)).cumsum()
    return np.stack((indices[:-1], indices[1:]), axis=-1)


def _mapping_for_query(timestamps, query):
    mapping = _contiguous_indices(timestamps)
    return mapping[np.isin(np.unique(timestamps), query)].T


def compute_img_idx_to_track_idx(timestamps, queries):
    return np.stack([
        _mapping_for_query(timestamps, query) for query in queries.T
    ])


def compute_class_mapping(classes, all_classes, mapping):
    output = []
    for raw_name in all_classes:
        mapped = mapping.get(raw_name)
        output.append(classes.index(mapped) if mapped in classes else -1)
    return np.asarray(output)
