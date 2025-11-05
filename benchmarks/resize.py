# Benchmarks from https://github.com/deshaw/versioned-hdf5/issues/155

import h5py
import numpy as np

from versioned_hdf5 import VersionedHDF5File

dt = np.dtype("double")


def time_resize():
    with h5py.File("foo.h5", "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("0") as sv:
            sv.create_dataset(
                "bar",
                (2, 15220, 2),
                chunks=(300, 100, 2),
                dtype=dt,
                data=np.full((2, 15220, 2), 0, dtype=dt),
            )

    with h5py.File("foo.h5", "r+") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("1") as sv:
            bar = sv["bar"]
            bar.resize((3, 15222, 2))


time_resize.timeout = 1200  # type: ignore[attr-defined]


# Pure HDF5 for comparison
def time_resize_hdf5():
    with h5py.File("foo.h5", "w") as f:
        f.create_dataset(
            "bar",
            (2, 15220, 2),
            chunks=(300, 100, 2),
            dtype=dt,
            data=np.full((2, 15220, 2), 0, dtype=dt),
            maxshape=(None, None, None),
        )

    with h5py.File("foo.h5", "r+") as f:
        bar = f["bar"]
        bar.resize((3, 15222, 2))


def time_resize_and_write():
    with h5py.File("foo.h5", "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("0") as sv:
            sv.create_dataset(
                "bar",
                (1, 10, 2),
                chunks=(600, 2, 4),
                dtype=dt,
                data=np.full((1, 10, 2), 0, dtype=dt),
            )

    for i in range(1, 100):
        with h5py.File("foo.h5", "r+") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(str(i)) as sv:
                bar = sv["bar"]
                bar.resize((1, (i + 1) * 10, 2))
                bar[:, -10:, :] = np.full((1, 10, 2), i, dtype=dt)


time_resize_and_write.timeout = 1200  # type: ignore[attr-defined]


def time_resize_and_write_hdf5_no_copy():
    with h5py.File("foo.h5", "w") as f:
        f.create_dataset(
            "bar",
            (1, 10, 2),
            chunks=(600, 2, 4),
            dtype=dt,
            data=np.full((1, 10, 2), 0, dtype=dt),
            maxshape=(None, None, None),
        )

    for i in range(1, 100):
        with h5py.File("foo.h5", "r+") as f:
            bar = f["bar"]
            bar.resize((1, (i + 1) * 10, 2))
            bar[:, -10:, :] = np.full((1, 10, 2), i, dtype=dt)


def time_resize_and_write_hdf5():
    with h5py.File("foo.h5", "w") as f:
        f.create_dataset(
            "bar0",
            (1, 10, 2),
            chunks=(600, 2, 4),
            dtype=dt,
            data=np.full((1, 10, 2), 0, dtype=dt),
            maxshape=(None, None, None),
        )

    for i in range(1, 100):
        with h5py.File("foo.h5", "r+") as f:
            bar = f.create_dataset(
                f"bar{i}",
                chunks=(600, 2, 4),
                dtype=dt,
                data=f[f"bar{i - 1}"],
                maxshape=(None, None, None),
            )
            bar.resize((1, (i + 1) * 10, 2))
            bar[:, -10:, :] = np.full((1, 10, 2), i, dtype=dt)
