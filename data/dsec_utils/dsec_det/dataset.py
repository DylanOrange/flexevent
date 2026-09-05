from pathlib import Path

import cv2
import numpy as np

from data.dsec_utils.class_config import DSEC_NATIVE_CLASSES
from data.dsec_utils.dsec_det.directory import DSECDirectory
from data.dsec_utils.dsec_det.preprocessing import compute_img_idx_to_track_idx


class DSECDet:
    """Read DSEC images and official detection labels."""

    def __init__(self, root: Path, split: str, sync: str = "back",
                 split_config=None):
        root = Path(root)
        if split_config is None or split not in split_config:
            raise KeyError(split)
        if sync not in {"front", "back"}:
            raise ValueError("sync must be 'front' or 'back'")
        if not root.is_dir():
            raise FileNotFoundError(root)

        self.classes = DSEC_NATIVE_CLASSES
        self.sync = sync
        self.height = 480
        self.width = 640
        train_root = root / "train"
        test_root = root / "test"

        requested = list(split_config[split])
        available = {
            path.name: path
            for base in (train_root, test_root)
            if base.is_dir()
            for path in base.iterdir()
            if path.is_dir()
        }
        missing = [name for name in requested if name not in available]
        if missing:
            raise FileNotFoundError(
                f"Missing DSEC {split} sequences: {', '.join(missing)}"
            )
        self.subsequence_directories = sorted(
            (available[name] for name in requested),
            key=self._first_timestamp,
        )
        self.directories = {}
        self.img_idx_track_idxs = {}
        for path in self.subsequence_directories:
            directory = DSECDirectory(path)
            self.directories[path.name] = directory
            self.img_idx_track_idxs[path.name] = compute_img_idx_to_track_idx(
                directory.tracks.tracks["t"], directory.images.timestamps
            )

    @staticmethod
    def _first_timestamp(sequence_path: Path):
        return np.genfromtxt(
            sequence_path / "images/timestamps.txt", dtype="int64"
        )[0]

    @staticmethod
    def _index_window(index, num_indices, sync):
        if sync == "front":
            if not 0 < index < num_indices:
                raise IndexError(index)
            return index - 1, index
        if not 0 <= index < num_indices - 1:
            raise IndexError(index)
        return index, index + 1

    def get_tracks(self, index, mask=None, directory_name=None):
        index, mapping, directory = self._relative_index(index, directory_name)
        if self.sync == "front":
            if not 0 < index < len(mapping):
                raise IndexError(index)
            start = index - 1
        else:
            if not 0 <= index < len(mapping):
                raise IndexError(index)
            start = index
        row_start, row_end = mapping[start]
        tracks = directory.tracks.tracks[row_start:row_end]
        if mask is not None:
            tracks = tracks[mask[row_start:row_end]]
        return tracks

    def get_image(self, index, directory_name=None):
        index, _, directory = self._relative_index(index, directory_name)
        image = cv2.imread(str(directory.images.image_files_distorted[index]))
        if image is None:
            raise FileNotFoundError(directory.images.image_files_distorted[index])
        return image

    def _relative_index(self, index, directory_name=None):
        if directory_name is not None:
            return (
                index,
                self.img_idx_track_idxs[directory_name],
                self.directories[directory_name],
            )
        for path in self.subsequence_directories:
            mapping = self.img_idx_track_idxs[path.name]
            sequence_len = len(mapping) - 1
            if index < sequence_len:
                return index, mapping, self.directories[path.name]
            index -= sequence_len
        raise IndexError(index)
