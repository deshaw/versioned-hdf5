# Benchmarks from https://github.com/deshaw/versioned-hdf5/issues/167

import h5py
import numpy as np

from versioned_hdf5 import VersionedHDF5File

dt = np.dtype("double")


def time_many_chunks():
    d0 = 2
    d1 = 15220
    d2 = 2
    shape = (d0, d1, d2)
    chunks = (600, 2, 4)
    with h5py.File("foo.h5", "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("0") as sv:
            sv.create_dataset(
                "bar",
                shape=shape,
                maxshape=(None, None, None),
                chunks=chunks,
                dtype=dt,
                data=np.full(shape, 0, dtype=dt),
            )

    i = 1
    with h5py.File("foo.h5", "r+") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version(str(i)) as sv:
            sv["bar"][:] = np.full(shape, i, dtype=dt)


def time_many_chunks_integer_index():
    d0 = 2
    d1 = 15220
    d2 = 2
    shape = (d0, d1, d2)
    chunks = (600, 2, 4)
    with h5py.File("foo.h5", "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("0") as sv:
            sv.create_dataset(
                "bar",
                shape=shape,
                maxshape=(None, None, None),
                chunks=chunks,
                dtype=dt,
                data=np.full(shape, 0, dtype=dt),
            )

    i = 1
    with h5py.File("foo.h5", "r+") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version(str(i)) as sv:
            i2 = np.random.choice(d1, 30, replace=False)
            i2 = np.sort(i2)
            sv["bar"][:, i2, :] = np.full((d0, len(i2), d2), i, dtype=dt)


def time_many_chunks_arange():
    d0 = 2
    d1 = 15220
    d2 = 2
    shape = (d0, d1, d2)
    chunks = (600, 2, 4)
    with h5py.File("foo.h5", "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("0") as sv:
            sv.create_dataset(
                "bar",
                shape=shape,
                maxshape=(None, None, None),
                chunks=chunks,
                dtype=dt,
                data=np.arange(np.prod(shape), dtype=dt).reshape(shape),
            )
