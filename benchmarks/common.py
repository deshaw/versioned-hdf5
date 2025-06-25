import os

import h5py
import numpy as np
from numpy.typing import DTypeLike

from versioned_hdf5 import VersionedHDF5File

try:
    from versioned_hdf5.h5py_compat import HAS_NPYSTRINGS
except ImportError:  # versioned_hdf5 < 2.1
    HAS_NPYSTRINGS = False


def require_npystrings():
    """Skip if StringDType is not supported. To be called by setup() for dtype='T'."""
    if not HAS_NPYSTRINGS:
        raise NotImplementedError(
            "NpyStrings require numpy>=2.0, h5py >=3.14, versioned-dhf5 >=2.1"
        )


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

    def rand_strings(
        self, shape: tuple[int, ...], min_nchars: int, max_nchars: int, dtype: DTypeLike
    ) -> np.ndarray:
        """Generate a ndarray of random strings"""
        assert 0 <= min_nchars <= max_nchars
        rand_chars = (
            self.rng.integers(
                ord("0"), ord("z"), (np.prod(shape), max_nchars), dtype=np.uint8
            )
            .view("S1")
            .astype("U1")
            .tolist()
        )
        if min_nchars < max_nchars:
            res = [
                "".join(row)[: self.rng.integers(min_nchars, max_nchars)]
                for row in rand_chars
            ]
        else:
            res = ["".join(row) for row in rand_chars]

        return np.asarray(res, dtype=dtype).reshape(shape)
