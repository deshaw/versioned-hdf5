import h5py
import numpy as np

from versioned_hdf5 import VersionedHDF5File


class TimeDatetimeAccess:
    def setup(self):
        with h5py.File("foo.h5", "w") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version("0") as sv:
                sv.create_dataset("bar", data=np.random.rand(10))

            for i in range(1, 100):
                with vf.stage_version(str(i)) as sv:
                    sv["bar"][:] = np.random.rand(10)
            self.dt = np.datetime64(vf[str(50)].attrs["timestamp"])

    def time_version_by_datetime(self):
        # Based on https://github.com/deshaw/versioned-hdf5/issues/170
        with h5py.File("foo.h5", "r") as f:
            vf = VersionedHDF5File(f)
            for _ in range(100):
                _ = vf[self.dt]["bar"][:]
