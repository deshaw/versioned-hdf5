import os

import h5py
import numpy as np

from versioned_hdf5 import VersionedHDF5File


class Benchmark:
    """Common setup and teardown for all versioned-hdf5 benchmarks."""

    def setup(self, *args, **kwargs):
        self.rng = np.random.default_rng(42)
        # cwd is a temporary directory created by asv
        self.file = h5py.File("bench.hdf5", "w")
        self.vfile = VersionedHDF5File(self.file)
        self._clean_setup = True

    def reopen(self):
        """Close the dataset, thus ensuring everything has been
        flushed to disk, and reopen it.
        """
        self.file.close()
        self.file = h5py.File("bench.hdf5", "r+")
        self.vfile = VersionedHDF5File(self.file)

    def assert_clean_setup(self):
        """Assert that the setup was executed just before the test invoking this method.

        This is currently not the case for reruns, which means that you must add the
        --quick parameter in most cases:
        https://github.com/airspeed-velocity/asv/issues/966

        All tests that modify the state should call this on their first line.
        """
        if not self._clean_setup:
            raise RuntimeError(
                "This test modifies the state and can only run with the --quick flag "
                "(asv#966)"
            )
        self._clean_setup = False

    def teardown(self, *args, **kwargs):
        self.file.close()
        os.remove("bench.hdf5")
