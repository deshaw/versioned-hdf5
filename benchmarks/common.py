import gc
import os
import tempfile
import tracemalloc
from functools import wraps

import h5py
import numpy as np
from asv_runner.benchmarks.mark import skip_benchmark_if
from numpy.typing import DTypeLike

from versioned_hdf5 import VersionedHDF5File

try:
    import memray
except ImportError:  # Not available on Windows
    memray = None

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


def peak_memory(func):
    """Decorator for a benchmark function to report peak memory usage.

    Unlike asv's ``peakmem_*`` benchmarks, which publish the high-water mark of the
    resident set size of the whole benchmark process (``getrusage()``), this exclusively
    measures what the benchmark function itself allocates.

    Expose it under a ``track_*`` name, e.g.::

        def time_something(self, ...):
            ...

        track_something = peak_memory(time_something)

    asv's ``track_*`` benchmarks publish whatever the benchmark function returns, so the
    decorated function returns the peak amount of memory that Python allocated while it
    ran, in bytes.

    Do not use a ``mem_*`` name: that prefix belongs to asv's pympler-based benchmark
    type, which reports the size of the object that the function returns; a scalar
    measurement like this one always comes out as 0 there.

    Notes
    -----
    On Windows, where memray is not available, this does not measure memory allocated by
    libhdf5.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Collect before starting, so that garbage left behind by setup() is not freed
        # - and therefore untracked - midway through the measurement.
        gc.collect()

        if memray:
            # Linux / MacOSX
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "memray.bin")
                with memray.Tracker(path, trace_python_allocators=True):
                    func(*args, **kwargs)
                return memray.FileReader(path).metadata.peak_memory
        else:
            # Windows. tracemalloc() only tracks memory allocated by Python and NumPy,
            # but not by libhdf5.
            tracemalloc.start()
            try:
                func(*args, **kwargs)
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

    # Read by asv_runner's TrackBenchmark, which otherwise reports the value in
    # asv's default unit of "unit".
    wrapper.unit = "bytes"
    return wrapper


skip_slow = os.getenv("ASV_RUNSLOW") != "1"


def slow(test):
    """Skip tests marked as @slow by default"""
    return skip_benchmark_if(skip_slow)(test)


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
