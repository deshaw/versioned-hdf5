"""Benchmarks for replay.py: delete_versions, modify_metadata, recreate_dataset"""

from __future__ import annotations

import h5py
import numpy as np

from versioned_hdf5 import delete_versions, modify_metadata
from versioned_hdf5.replay import recreate_dataset, tmp_group

from .common import Benchmark, peak_memory, require_npystrings, slow

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
    # Change dtype from float64 to int8, which is 8 times narrower
    "dtype_i1": {"dtype": "i1"},
    # Compress an uncompressed dataset of incompressible random data
    "compress": {"compression": "lzf"},
}


class _ReplayBenchmark(Benchmark):
    """Common setup for the benchmarks in this module."""

    # Every benchmark in this module modifies the file it runs on, so setup()
    # must run again before every single test (asv#966)
    number = 1
    warmup_time = 0

    def setup(self, *args, **kwargs):
        super().setup()
        with self.vfile.stage_version("v0") as sv:
            sv.create_dataset(NAME, data=self.rng.random(SHAPE), chunks=CHUNK_SIZE)
        with self.vfile.stage_version("v1") as sv:
            sv[NAME][0, 0] = -1.0
        with self.vfile.stage_version("v2") as sv:
            sv[NAME][32, 0] = -1.0


class TimeDeleteVersions(_ReplayBenchmark):
    params = ["v0", "v1", "v2"]
    param_names = ["case"]

    def time_delete_versions(self, case):
        self.assert_clean_setup()
        delete_versions(self.file, case)

    track_peakmem_delete_versions = peak_memory(time_delete_versions)


class TimeRecreateDataset(_ReplayBenchmark):
    params = [RECREATE_CASES]
    param_names = ["case"]

    def setup(self, case):
        super().setup()
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

    track_peakmem_recreate_dataset = peak_memory(time_recreate_dataset)


class TimeRecreateDatasetBlocked(Benchmark):
    """Trigger dynamically-sized block copy (replay::_rewrite_block_bytes)

    `recreate_dataset()` rewrites every version into a brand new `raw_data`, so the
    on-disk hash table that `_rewrite_block_bytes()` sizes the block from starts empty
    and grows by one version's chunks at a time. A single-version file therefore only
    ever gets the floor block size; a multi-version file exercises the whole range.
    """

    number = 1
    warmup_time = 0
    # A sample takes ~20 s to time and 5x as long under memray; asv's default is 60 s.
    timeout = 1200

    #: Versions in the file, each with unique data, so that the hash table that the
    #: next version is rewritten against grows at every step. The first version always
    #: gets the 64 MiB floor, just like the fixed-size baseline, so a handful of them
    #: are needed before the block outgrows the floor and the difference is visible.
    n_versions = 6

    # 1 GiB / 16 kiB = 64k chunks per version, so the table grows from empty to
    # 384k chunks and the block from the 64 MiB floor to the 512 MiB cap:
    # 64, 128, 256, 384, 512, 512 MiB.
    shape = (1024 * 1024 * 1024 // 8,)
    chunks = (16 * 1024 // 8,)

    def setup(self):
        super().setup()
        for i in range(self.n_versions):
            with self.vfile.stage_version(f"v{i}") as sv:
                sv.create_dataset(
                    NAME, data=self.rng.random(self.shape), chunks=self.chunks
                )
        self.reopen()
        self.newf = tmp_group(self.file)

    @slow
    def time_recreate_dataset(self):
        self.assert_clean_setup()
        recreate_dataset(self.file, NAME, self.newf)

    # At 16 kiB chunks the benchmark peak memory is dominated by what libhdf5 allocates
    # for the virtual mappings of the source and destination datasets (~45 kiB per
    # chunk), not by the copy block. This has been measured as 4.2~4.8 GB worth of
    # libhdf5 metadata, 10x the max scratch area size and very noisy.
    track_peakmem_recreate_dataset = peak_memory(time_recreate_dataset)  # noqa: ERA001


class TimeRecreateDatasetBlockedStrings(Benchmark):
    """Same as TimeRecreateDatasetBlocked, but with variable-length strings."""

    number = 1
    warmup_time = 0
    timeout = 1200

    n_versions = 5
    shape = (2**24,)
    chunks = (1024,)

    params = ["O", "T"]
    param_names = ["dtype"]

    def setup(self, dtype):
        if dtype == "T":
            require_npystrings()

        super().setup()
        for i in range(self.n_versions):
            # Benchmark.rand_strings() is too slow for 16.8 million strings
            data = (
                self.rng.integers(ord("a"), ord("z") + 1, (*self.shape, 10), np.uint8)
                .view("S10")[:, 0]
                .astype(dtype)
            )
            with self.vfile.stage_version(f"v{i}") as sv:
                sv.create_dataset(
                    NAME,
                    data=data,
                    dtype="T" if dtype == "T" else h5py.string_dtype(),
                    chunks=self.chunks,
                )
        self.reopen()
        self.newf = tmp_group(self.file)

    @slow
    def time_recreate_dataset(self, *args, **kwargs):
        self.assert_clean_setup()
        recreate_dataset(self.file, NAME, self.newf)

    track_peakmem_recreate_dataset = peak_memory(time_recreate_dataset)


class TimeModifyMetadata(Benchmark):
    params = [
        ["dense", "sparse"],
        ["inmem", "disk"],
        list(MODIFY_METADATA_CASES),
    ]
    param_names = ["density", "where", "case"]

    # 512 MiB of float64's, i.e. 2,048 chunks in 8 blocks
    SHAPE = (2**15, 2048)

    def setup(self, density, where, case):
        self.kwargs = MODIFY_METADATA_CASES[case]
        super().setup()

        if density == "dense":
            with self.vfile.stage_version("v0") as sv:
                sv.create_dataset(
                    NAME,
                    data=self.rng.random(self.SHAPE),
                    chunks=CHUNK_SIZE,
                )
        else:
            assert density == "sparse"
            with self.vfile.stage_version("v0") as sv:
                sv.create_dataset(
                    NAME,
                    shape=self.SHAPE,
                    chunks=CHUNK_SIZE,
                )
                sv[NAME][0, 0] = 1
                sv[NAME][32, 1024] = 2

        if where == "inmem":
            # Keep the staging InMemoryGroup wrappers alive
            self._hold = sv
        else:
            assert where == "disk"
            del sv
            self.reopen()

    def time_modify_metadata(self, *args, **kwargs):
        self.assert_clean_setup()
        modify_metadata(self.file, NAME, **self.kwargs)

    track_peakmem_modify_metadata = peak_memory(time_modify_metadata)


class TimeManyVersions(Benchmark):
    number = 1
    warmup_time = 0

    # 16 MiB of float64 in 8 kiB chunks, i.e. 2,048 chunks per version
    shape = (2 * 1024 * 1024,)
    chunks = (1024,)
    n_versions = 4

    def setup(self):
        super().setup()
        with self.vfile.stage_version("v0") as sv:
            sv.create_dataset(
                NAME, data=self.rng.random(self.shape), chunks=self.chunks
            )
        for i in range(1, self.n_versions):
            with self.vfile.stage_version(f"v{i}") as sv:
                sv[NAME][i] = -1.0
        self.newf = tmp_group(self.file)

    def time_recreate_dataset_many_versions(self):
        self.assert_clean_setup()
        recreate_dataset(self.file, NAME, self.newf)

    def time_modify_metadata_many_versions(self):
        self.assert_clean_setup()
        modify_metadata(self.file, NAME, fillvalue=1.5)

    track_peakmem_recreate_dataset_many_versions = peak_memory(
        time_recreate_dataset_many_versions
    )
    track_peakmem_modify_metadata_many_versions = peak_memory(
        time_modify_metadata_many_versions
    )
