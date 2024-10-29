# Pure hdf5 version of TimeInMemoryDataset and TimeInMemoryArraDataset
import os

import h5py
import numpy as np


class TimePureHDF5:
    def setup(self):
        self.file = h5py.File("bench.hdf5", "w")
        self.file.create_dataset(
            "data",
            data=np.arange(10000).reshape((100, 10, 10)),
            chunks=(3, 3, 3),
            maxshape=(None, None, None),
        )

    def teardown(self):
        self.file.close()
        os.remove("bench.hdf5")

    def time_getattr(self):
        dataset = self.file["data"]
        dataset[:, 0, 0:6]

    def time_setattr(self):
        dataset = self.file["data"]
        dataset[:, 0, 0:6] = -1

    def time_resize_bigger(self):
        dataset = self.file["data"]
        dataset.resize((100, 100, 100))

    def time_resize_smaller(self):
        dataset = self.file["data"]
        dataset.resize((10, 10, 10))
