from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import numpy as np
import torch
import yaml


def _extract_from_h5_by_index(filehandle, starts, ends):
    events = filehandle["events"]
    output = []
    for start, end in zip(np.atleast_1d(starts), np.atleast_1d(ends)):
        x = events["x"][start:end].astype("int64")
        y = events["y"][start:end].astype("int64")
        output.append({
            "p": torch.from_numpy(events["p"][start:end]),
            "t": torch.from_numpy(
                events["t"][start:end].astype("int64")
                + filehandle["t_offset"][()]
            ),
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
        })
    return output


def extract_from_h5_by_timewindow(h5file, t_min_us, t_max_us):
    with h5py.File(str(h5file), "r") as handle:
        ms_to_idx = np.asarray(handle["ms_to_idx"], dtype="int64")
        offset = handle["t_offset"][()]
        start_ms = np.floor_divide(np.asarray(t_min_us) - offset, 1000)
        end_ms = np.floor_divide(np.asarray(t_max_us) - offset, 1000)
        return _extract_from_h5_by_index(
            handle, ms_to_idx[start_ms], ms_to_idx[end_ms]
        )


def yaml_file_to_dict(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
