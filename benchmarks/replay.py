"""Benchmarks for replay.py: delete_versions, modify_metadata, recreate_dataset"""

from __future__ import annotations

import gc

import numpy as np

from versioned_hdf5 import delete_versions, modify_metadata
from versioned_hdf5.replay import recreate_dataset, tmp_group

from .common import Benchmark

#: Shape of the dataset in every version, and its chunk size:
#: 128 MiB of float64 in 256 KiB chunks, i.e. 256 chunks
SHAPE = (8192, 2048)
CHUNK_SIZE = (32, 1024)
DATASET_BYTES = np.prod(SHAPE) * 8
CHUNK_BYTES = np.prod(CHUNK_SIZE) * 8

#: Name of the dataset within each version group
NAME = "values"

#: Callbacks for recreate_dataset()
RECREATE_CASES = [
    # Rewrite every version as it is
    "no_callback",
    # Drop an intermediate version
    "drop_version",
]

#: Keyword arguments for modify_metadata()
MODIFY_METADATA_CASES = {
    # Rewrite every version without altering any metadata
    "noop": {},
    # Halve the number of chunks
    "rechunk": {"chunks": (CHUNK_SIZE[0] * 2, CHUNK_SIZE[1])},
    # Replace every point equal to the old fillvalue with the new one
    "fillvalue": {"fillvalue": 1.5},
    # Change dtype from float64 to float32
    "dtype": {"dtype": "f4"},
    # Compress an uncompressed dataset of incompressible random data
    "compress": {"compression": "lzf"},
}


class _ReplayBenchmark(Benchmark):
    """Common setup for the benchmarks in this module."""

    # Every benchmark in this module modifies the file it runs on, so setup()
    # must run again before every single test (asv#966)
    number = 1
    warmup_time = 0

    def setup(self, case):
        super().setup(case)
        with self.vfile.stage_version("v0") as sv:
            sv.create_dataset(NAME, data=self.rng.random(SHAPE), chunks=CHUNK_SIZE)
        with self.vfile.stage_version("v1") as sv:
            sv[NAME][0, 0] = -1.0
        with self.vfile.stage_version("v2") as sv:
            sv[NAME][32, 0] = -1.0
        gc.collect()


class Baseline(_ReplayBenchmark):
    params = ["baseline"]
    param_names = ["case"]

    def peakmem_baseline(self, case):
        """Measure RAM usage when doing nothing"""


class TimeDeleteVersions(_ReplayBenchmark):
    params = ["v0", "v1", "v2"]
    param_names = ["case"]

    def setup(self, case):
        super().setup(case)

    def time_delete_versions(self, case):
        self.assert_clean_setup()
        delete_versions(self.file, case)

    peakmem_delete_versions = time_delete_versions


class TimeRecreateDataset(_ReplayBenchmark):
    params = [RECREATE_CASES]
    param_names = ["case"]

    def setup(self, case):
        super().setup(case)
        self.newf = tmp_group(self.file)

        if case == "no_callback":
            self.callback = None
        elif case == "drop_version":

            def drop_version(dataset, version_name):
                return None if version_name == "v1" else dataset

            self.callback = drop_version
        else:
            raise AssertionError("unreachable")

    def time_recreate_dataset(self, case):
        self.assert_clean_setup()
        recreate_dataset(self.file, NAME, self.newf, callback=self.callback)

    peakmem_recreate_dataset = time_recreate_dataset


class TimeModifyMetadata(_ReplayBenchmark):
    params = [list(MODIFY_METADATA_CASES)]
    param_names = ["case"]

    def setup(self, case):
        super().setup(case)
        self.kwargs = MODIFY_METADATA_CASES[case]

    def time_modify_metadata(self, case):
        self.assert_clean_setup()
        modify_metadata(self.file, NAME, **self.kwargs)

    peakmem_modify_metadata = time_modify_metadata
