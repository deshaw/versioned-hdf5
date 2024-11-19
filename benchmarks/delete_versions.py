import os
import shutil
import tempfile

import h5py
import numpy

from versioned_hdf5 import VersionedHDF5File, delete_versions

filename = "delete_versions_bench.h5"


class TimeDeleting:
    params = [10, 30, 50]
    timeout = 1000

    def setup(self, n):
        if not os.path.exists(filename):
            with h5py.File(filename, "w") as f:
                vf = VersionedHDF5File(f)
                with vf.stage_version("init") as sv:
                    sv.create_dataset(
                        "values",
                        shape=(0, 0),
                        dtype="float",
                        fillvalue=numpy.nan,
                        chunks=(22, 100),
                        maxshape=(None, None),
                        compression="lzf",
                    )

            # generate some test data with around 1000 versions
            v = 1
            with h5py.File(filename, "r+") as f:
                vf = VersionedHDF5File(f)
                for d in range(3):
                    with vf.stage_version(str(v)) as sv:
                        values_ds = sv["values"]
                        values_ds.resize(
                            (values_ds.shape[0] + 1, values_ds.shape[1] + 5000)
                        )
                        values_ds[-1, -5000] = numpy.random.rand()
                        v += 1
                    for c in range(n):
                        with vf.stage_version(str(v)) as sv:
                            values_ds = sv["values"]
                            idxs = numpy.random.choice(
                                values_ds.shape[1], 50, replace=False
                            )
                            values_ds[-1, idxs] = numpy.random.rand(50)
                            v += 1

    def teardown(self, n):
        os.remove(filename)

    def time_delete(self, n):
        tmp_name = tempfile.mktemp(".h5")
        shutil.copy2(filename, tmp_name)
        try:
            # want to keep only every 10th version
            versions_to_delete = []
            with h5py.File(tmp_name, "r") as f:
                vf = VersionedHDF5File(f)
                versions = sorted(
                    [(v, vf._versions[v].attrs["timestamp"]) for v in vf._versions],
                    key=lambda t: t[1],
                )
                for i, v in enumerate(versions):
                    if i % 10 != 0:
                        versions_to_delete.append(v[0])

            with h5py.File(tmp_name, "r+") as f:
                delete_versions(f, versions_to_delete)
        finally:
            os.remove(tmp_name)
