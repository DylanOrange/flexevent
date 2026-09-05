
import argparse
import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401: register Blosc filters
import numba
import numpy as np
from tqdm import tqdm


H5_BLOSC_COMPRESSION_FLAGS = {
    "compression": 32001,
    "compression_opts": (0, 0, 0, 0, 1, 2, 5),
    "chunks": True,
}


@numba.jit(nopython=True, cache=True)
def _filter_events_resize(x, y, p, mask, change_map, fx, fy):
    for index in range(len(x)):
        x_low = x[index] // fx
        y_low = y[index] // fy
        change_map[y_low, x_low] += p[index] * 1.0 / (fx * fy)
        if abs(change_map[y_low, x_low]) >= 1:
            mask[index] = True
            change_map[y_low, x_low] -= p[index]
    return mask, change_map


def downsample_events(events, change_map, input_height=480, input_width=640,
                      output_height=240, output_width=320):
    if change_map is None:
        change_map = np.zeros((output_height, output_width), dtype="float32")
    fx = int(input_width / output_width)
    fy = int(input_height / output_height)
    mask = np.zeros(len(events["t"]), dtype="bool")
    mask, change_map = _filter_events_resize(
        events["x"], events["y"], events["p"], mask, change_map, fx, fy
    )
    selected = {key: value[mask] for key, value in events.items()}
    selected["x"] = (selected["x"] / fx).astype("uint16")
    selected["y"] = (selected["y"] / fy).astype("uint16")
    return selected, change_map


def create_ms_to_idx(t_us):
    t_ms = t_us // 1000
    milliseconds, counts = np.unique(t_ms, return_counts=True)
    output = np.zeros(shape=(t_ms[-1] + 2,), dtype="uint64")
    output[milliseconds + 1] = counts
    return output[:-1].cumsum()


def read_events(handle, start, stop):
    event_group = handle["events"]
    return {
        "x": event_group["x"][start:stop],
        "y": event_group["y"][start:stop],
        "p": event_group["p"][start:stop],
        "t": event_group["t"][start:stop].astype("int64")
        + handle["t_offset"][()],
    }


class H5Writer:
    def __init__(self, path):
        self.handle = h5py.File(path, "w")
        self.t_offset = None
        self.num_events = 0
        for name, dtype in (("x", "u2"), ("y", "u2"),
                            ("p", "u1"), ("t", "u4")):
            self.handle.create_dataset(
                f"events/{name}", shape=(2**16,), maxshape=(None,),
                dtype=dtype, **H5_BLOSC_COMPRESSION_FLAGS
            )

    def add_data(self, events):
        if not len(events["t"]):
            return
        if self.t_offset is None:
            self.t_offset = events["t"][0]
            self.handle.create_dataset("t_offset", data=self.t_offset, dtype="i8")
        relative_t = events["t"] - self.t_offset
        size = len(relative_t)
        old_size = self.num_events
        self.num_events += size
        values = dict(events)
        values["t"] = relative_t
        for name in ("x", "y", "p", "t"):
            dataset = self.handle[f"events/{name}"]
            dataset.resize(self.num_events, axis=0)
            dataset[old_size:self.num_events] = values[name]

    def finish(self):
        if self.num_events == 0:
            raise RuntimeError("Downsampling produced no events")
        timestamps = self.handle["events/t"][()]
        self.handle.create_dataset(
            "ms_to_idx", data=create_ms_to_idx(timestamps), dtype="u8",
            **H5_BLOSC_COMPRESSION_FLAGS
        )
        self.handle.close()


def main(input_path, output_path, chunk_size=100_000):
    input_path = Path(input_path)
    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() or temporary_path.exists():
        raise FileExistsError(output_path if output_path.exists() else temporary_path)

    with h5py.File(input_path, "r") as source:
        total = len(source["events/t"])
        writer = H5Writer(temporary_path)
        change_map = None
        with tqdm(total=(total + chunk_size - 1) // chunk_size,
                  desc=input_path.parents[2].name) as progress:
            for start in range(0, total, chunk_size):
                events = read_events(source, start, min(start + chunk_size, total))
                events["p"] = 2 * events["p"].astype("int8") - 1
                selected, change_map = downsample_events(events, change_map)
                selected["p"] = ((selected["p"] + 1) // 2).astype("uint8")
                writer.add_data(selected)
                progress.update(1)
        writer.finish()
    os.replace(temporary_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    arguments = parser.parse_args()
    main(arguments.input_path, arguments.output_path, arguments.chunk_size)
