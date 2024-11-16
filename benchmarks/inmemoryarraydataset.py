import os

import h5py

from versioned_hdf5 import VersionedHDF5File
from versioned_hdf5.wrappers import InMemoryArrayDataset

try:
    from versioned_hdf5.wrappers import DatasetWrapper
except ImportError:

    class DatasetWrapper:
        pass


import numpy as np


class TimeInMemoryArrayDataset:
    timeout = 1000

    def teardown(self):
        os.remove("bench.hdf5")

    def time_getattr(self):
        with h5py.File("bench.hdf5", "w") as f:
            versioned_file = VersionedHDF5File(f)
            with versioned_file.stage_version("version1") as g:
                dataset = g.create_dataset(
                    "data",
                    data=np.arange(10000).reshape((100, 10, 10)),
                    chunks=(3, 3, 3),
                )
                assert (
                    isinstance(dataset, InMemoryArrayDataset)
                    or isinstance(dataset, DatasetWrapper)
                    and isinstance(dataset.dataset, InMemoryArrayDataset)
                )
                dataset[:, 0, 0:6]

    def time_setattr(self):
        with h5py.File("bench.hdf5", "w") as f:
            versioned_file = VersionedHDF5File(f)
            with versioned_file.stage_version("version1") as g:
                dataset = g.create_dataset(
                    "data",
                    data=np.arange(10000).reshape((100, 10, 10)),
                    chunks=(3, 3, 3),
                )
                assert (
                    isinstance(dataset, InMemoryArrayDataset)
                    or isinstance(dataset, DatasetWrapper)
                    and isinstance(dataset.dataset, InMemoryArrayDataset)
                )
                dataset[:, 0, 0:6] = -1

    def time_resize_bigger(self):
        with h5py.File("bench.hdf5", "w") as f:
            versioned_file = VersionedHDF5File(f)
            with versioned_file.stage_version("version1") as g:
                dataset = g.create_dataset(
                    "data",
                    data=np.arange(10000).reshape((100, 10, 10)),
                    chunks=(3, 3, 3),
                )
                assert (
                    isinstance(dataset, InMemoryArrayDataset)
                    or isinstance(dataset, DatasetWrapper)
                    and isinstance(dataset.dataset, InMemoryArrayDataset)
                )
                dataset.resize((100, 100, 100))

    def time_resize_smaller(self):
        with h5py.File("bench.hdf5", "w") as f:
            versioned_file = VersionedHDF5File(f)
            with versioned_file.stage_version("version1") as g:
                dataset = g.create_dataset(
                    "data",
                    data=np.arange(10000).reshape((100, 10, 10)),
                    chunks=(3, 3, 3),
                )
                assert (
                    isinstance(dataset, InMemoryArrayDataset)
                    or isinstance(dataset, DatasetWrapper)
                    and isinstance(dataset.dataset, InMemoryArrayDataset)
                )
                dataset.resize((10, 10, 10))
