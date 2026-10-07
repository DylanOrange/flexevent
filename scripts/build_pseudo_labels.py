"""Build FlexTune pseudo-labels from forward and reverse teacher predictions.

Reads PREDICTIONS_ROOT/{forward,reverse}/<sequence>.npy written by
generate_pseudo_labels.py and writes OUTPUT_ROOT/<sequence>.npy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dsec_utils.class_config import DSEC_NATIVE_CLASSES, get_dsec_class_spec
from data.dsec_utils.dsec_det.io import yaml_file_to_dict
from data.dsec_utils.utils import compute_class_mapping, crop_tracks
from utils.pseudo_labels import (
    HEIGHT, WIDTH, build_label_stream, filter_predictions, merge_directions,
    preprocess_gt, to_label_dtype, valid_box_mask,
)

# Per-class confidence thresholds used with --classwise for the 8-class model.
CLASSWISE_THRESHOLDS = {
    'pedestrian': 0.70, 'rider': 0.75, 'car': 0.75, 'bus': 0.75,
    'truck': 0.70, 'bicycle': 0.75, 'motorcycle': 0.70,
}


def main():
    dataset_config = yaml_file_to_dict(ROOT / 'config/dataset/dsec.yaml')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=dataset_config['path'])
    parser.add_argument('--predictions-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--class-mode', choices=('2class', '8class'), required=True)
    parser.add_argument('--classwise', action='store_true',
                        help='use per-class confidence thresholds')
    parser.add_argument('--confidence-threshold', type=float, default=0.6)
    parser.add_argument('--nms-threshold', type=float, default=0.45)
    parser.add_argument('--min-track-hits', type=int, default=6)
    parser.add_argument('--sequences', nargs='+')
    args = parser.parse_args()

    spec = get_dsec_class_spec(args.class_mode)
    sequences = args.sequences or yaml_file_to_dict(ROOT / 'data/dsec_utils/dsec_split.yaml')['train']
    class_thresholds = None
    if args.classwise:
        class_thresholds = {i: CLASSWISE_THRESHOLDS[name]
                            for i, name in enumerate(spec.model_class_names)}
    class_mapping = compute_class_mapping(
        spec.model_class_names, DSEC_NATIVE_CLASSES, spec.raw_to_model_name)

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        'class_mode': args.class_mode,
        'confidence_threshold': args.confidence_threshold,
        'class_thresholds': class_thresholds,
        'nms_threshold': args.nms_threshold,
        'min_track_hits': args.min_track_hits,
        'sequences': list(sequences),
    }
    for sequence in sequences:
        predictions = []
        for direction in ('forward', 'reverse'):
            boxes = np.load(args.predictions_root / direction / f'{sequence}.npy')
            boxes = filter_predictions(boxes, spec.num_classes,
                                       args.confidence_threshold, class_thresholds)
            boxes = crop_tracks(boxes, WIDTH, HEIGHT)
            predictions.append(boxes[valid_box_mask(boxes)])
        pseudo = to_label_dtype(merge_directions(*predictions, args.nms_threshold))

        sequence_root = args.data_root / 'train' / sequence
        image_timestamps = np.genfromtxt(sequence_root / 'images/timestamps.txt', dtype='int64')
        timestamps = np.linspace(image_timestamps[0], image_timestamps[-1],
                                 2 * len(image_timestamps) - 1).astype(np.int64)
        gt = preprocess_gt(np.load(sequence_root / 'object_detections/left/tracks.npy'),
                           class_mapping)

        labels = build_label_stream(timestamps, gt, pseudo, spec.num_classes,
                                    args.min_track_hits)
        np.save(args.output_root / f'{sequence}.npy', labels)

    (args.output_root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
