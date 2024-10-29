import os

import h5py

from versioned_hdf5 import VersionedHDF5File
from versioned_hdf5.wrappers import InMemoryDataset

try:
    from versioned_hdf5.wrappers import DatasetWrapper
except ImportError:

    class DatasetWrapper:
        pass


import numpy as np


class TimeInMemoryDataset:
    timeout = 1000

    def setup(self):
        if hasattr(self, "file"):
            self.file.close()
        if os.path.exists("bench.hdf5"):
            os.remove("bench.hdf5")

        with h5py.File("bench.hdf5", "w") as f:
            versioned_file = VersionedHDF5File(f)

            with versioned_file.stage_version("version1") as g:
                g.create_dataset(
                    "data",
                    data=np.arange(10000).reshape((100, 10, 10)),
                    chunks=(3, 3, 3),
                )

        self.file = h5py.File("bench.hdf5", "a")
        self.versioned_file = VersionedHDF5File(self.file)

    # def teardown(self):
    #     self.file.close()
    #     os.remove('bench.hdf5')

    def time_getitem(self):
        dataset = self.versioned_file["version1"]["data"]
        assert (
            isinstance(dataset, InMemoryDataset)
            or isinstance(dataset, DatasetWrapper)
            and isinstance(dataset.dataset, InMemoryDataset)
        )
        dataset[:, 0, 0:6]

    def time_setitem(self):
        # https://github.com/airspeed-velocity/asv/issues/966
        self.setup()
        with self.versioned_file.stage_version("version2") as g:
            dataset = g["data"]
            assert (
                isinstance(dataset, InMemoryDataset)
                or isinstance(dataset, DatasetWrapper)
                and isinstance(dataset.dataset, InMemoryDataset)
            )
            dataset[:, 0, 0:6] = -1

    def time_resize_bigger(self):
        # https://github.com/airspeed-velocity/asv/issues/966
        self.setup()
        with self.versioned_file.stage_version("version2") as g:
            dataset = g["data"]
            assert (
                isinstance(dataset, InMemoryDataset)
                or isinstance(dataset, DatasetWrapper)
                and isinstance(dataset.dataset, InMemoryDataset)
            )
            dataset.resize((100, 100, 100))

    def time_resize_smaller(self):
        # https://github.com/airspeed-velocity/asv/issues/966
        self.setup()
        with self.versioned_file.stage_version("version2") as g:
            dataset = g["data"]
            assert (
                isinstance(dataset, InMemoryDataset)
                or isinstance(dataset, DatasetWrapper)
                and isinstance(dataset.dataset, InMemoryDataset)
            )
            dataset.resize((10, 10, 10))
