"""Benchmarks for chunk-size guessing when creating 1D datasets (PR 547).

``sv[name] = arr`` leaves chunks=None, so the chunk size is guessed from the data
shape (h5py heuristic, which grows with the dataset). ``sv.create_dataset(data=arr)``
instead hard-codes ``DEFAULT_CHUNK_SIZE`` (4096). The PR 547 commit path briefly
guessed from an empty staging array instead, collapsing ``sv[name] = arr`` to
1024-element chunks; with 1e7 float64 elements that is ~10x more chunks and makes
every subsequent commit (e.g. a one-element update) much slower.
"""

from __future__ import annotations

import numpy as np

from .common import Benchmark

#: 80 MiB of float64; h5py guesses (9766,) chunks for it, vs. (1024,) when guessing
#: from an empty array and (4096,) for create_dataset(data=arr).
N = 10**7


class TimeSetitemVsCreateDataset(Benchmark):
    """One-element update + commit of a 1e7 float64 dataset, by creation method."""

    params = [["setitem", "create_dataset"]]
    param_names = ["how"]

    # The benchmark mutates the staged version, so setup() must run again before
    # every single sample (asv#966)
    number = 1
    warmup_time = 0

    def setup(self, how):
        super().setup(how)
        arr = np.arange(N, dtype="f8")
        with self.vfile.stage_version("v0") as sv:
            if how == "setitem":
                sv["d"] = arr
            else:
                sv.create_dataset("d", data=arr)
        self.ctx = self.vfile.stage_version("v1")
        version = self.ctx.__enter__()
        version["d"][5] = -1.0

    def time_one_element_commit(self, how):
        self.assert_clean_setup()
        self.ctx.__exit__(None, None, None)
        del self.ctx

    def teardown(self, how):
        if hasattr(self, "ctx"):
            self.ctx.__exit__(None, None, None)
            del self.ctx
        super().teardown(how)
