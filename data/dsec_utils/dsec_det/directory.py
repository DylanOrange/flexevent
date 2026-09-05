from functools import lru_cache

import numpy as np


class BaseDirectory:
    def __init__(self, root):
        self.root = root


class DSECDirectory:
    def __init__(self, root):
        self.root = root
        self.images = ImageDirectory(root / "images")
        self.events = EventDirectory(root / "events")
        self.tracks = TracksDirectory(root / "object_detections")


class ImageDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def timestamps(self):
        return np.genfromtxt(self.root / "timestamps.txt", dtype="int64")

    @property
    @lru_cache(maxsize=1)
    def image_files_distorted(self):
        return sorted((self.root / "left/distorted").glob("*.png"))


class EventDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def event_file(self):
        downsampled = self.root / "left/events_2x.h5"
        if not downsampled.is_file():
            raise FileNotFoundError(downsampled)
        return downsampled


class TracksDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def tracks(self):
        return np.load(self.root / "left/tracks.npy", allow_pickle=False)
