import warnings

import numpy as np

from .common import Benchmark


class TimeDatetimeAccess(Benchmark):
    def setup(self):
        super().setup()
        with self.vfile.stage_version("0") as sv:
            sv.create_dataset("data", data=self.rng.random(10))

        for i in range(1, 100):
            with self.vfile.stage_version(str(i)) as sv:
                sv["data"][:] = self.rng.random(10)
        ts = self.vfile[str(50)].attrs["timestamp"]
        # Suppress numpy timezone warning when converting to datetime64
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="no explicit representation of timezones.*",
            )
            self.dt = np.datetime64(ts)
        self.reopen()

    def time_version_by_datetime(self):
        # Based on https://github.com/deshaw/versioned-hdf5/issues/170
        for _ in range(100):
            _ = self.vfile[self.dt]["data"][:]
