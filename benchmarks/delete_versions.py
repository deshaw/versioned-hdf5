import numpy

from versioned_hdf5 import delete_versions

from .common import Benchmark


class TimeDeleteVersions(Benchmark):
    params = [10, 30, 50]
    param_names = ["n"]

    def setup(self, n):
        super().setup()

        with self.vfile.stage_version("init") as sv:
            sv.create_dataset(
                "values",
                shape=(0, 0),
                dtype="float",
                fillvalue=numpy.nan,
                chunks=(22, 100),
                maxshape=(None, None),
                compression="lzf",
            )

        # generate some test data with 30~150 versions
        v = 1
        for _ in range(3):
            with self.vfile.stage_version(str(v)) as sv:
                values_ds = sv["values"]
                values_ds.resize((values_ds.shape[0] + 1, values_ds.shape[1] + 5000))
                values_ds[-1, -5000] = self.rng.random()
                v += 1
            for _ in range(n):
                with self.vfile.stage_version(str(v)) as sv:
                    values_ds = sv["values"]
                    idxs = self.rng.choice(values_ds.shape[1], 50, replace=False)
                    values_ds[-1, idxs] = self.rng.random(50)
                    v += 1

        # Keep only every 10th version
        self.versions_to_delete = [str(i) for i in range(1, v) if i % 10 != 0]
        self.reopen()

    def time_delete_versions(self, n):
        self.assert_clean_setup()
        delete_versions(self.vfile, self.versions_to_delete)
