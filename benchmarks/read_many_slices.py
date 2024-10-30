import numpy as np
from versioned_hdf5.slicetools import read_many_slices

from .common import Benchmark

# 1 KiB, 64 KiB, 1 MiB of float64 per chunk
CHUNK_SIZES = [1024 // 8, 64 * 1024 // 8, 1024 * 1024 // 8]
# Total bytes transferred per benchmark call ~16 MiB
TOTAL_ELEMENTS = 16 * 1024 * 1024 // 8


class TimeReadManySlicesNumPy:
    """Benchmark read_many_slices with NumPy src and NumPy dst."""

    params = [CHUNK_SIZES]
    param_names = ["chunk_size"]

    def setup(self, chunk_size):
        rng = np.random.default_rng(42)
        n_slices = TOTAL_ELEMENTS // chunk_size
        # src is twice as large as dst so we can pick non-contiguous src offsets
        self.src = rng.random(2 * TOTAL_ELEMENTS, dtype=np.float64)
        self.dst = np.zeros(TOTAL_ELEMENTS, dtype=np.float64)
        # Contiguous, chunk-aligned offsets in both src and dst
        starts = (np.arange(n_slices) * chunk_size).reshape(-1, 1).astype(np.uint64)
        self.src_start = starts
        self.dst_start = starts
        self.count = np.full((n_slices, 1), chunk_size, dtype=np.uint64)

    def time_read_many_slices(self, chunk_size):
        read_many_slices(self.src, self.dst, self.src_start, self.dst_start, self.count)


class _TimeReadManySlicesH5Base(Benchmark):
    """Common setup for h5py read/write benchmarks."""

    params = [CHUNK_SIZES, [True, None, False]]
    param_names = ["chunk_size", "fast"]

    direction = ""  # "h5_to_np" or "np_to_h5"; overridden by subclasses

    def setup(self, chunk_size, fast):
        super().setup(chunk_size, fast)
        rng = np.random.default_rng(42)
        n_slices = TOTAL_ELEMENTS // chunk_size

        self.np_arr = rng.random(TOTAL_ELEMENTS, dtype=np.float64)
        # h5py dataset twice as large so source offsets can be non-contiguous if needed
        self.h5_dset = self.file.create_dataset(
            "data",
            # Initialize with data so reads return something realistic
            data=rng.random(2 * TOTAL_ELEMENTS, dtype=np.float64),
            chunks=(chunk_size,),
        )

        starts = (np.arange(n_slices) * chunk_size).reshape(-1, 1).astype(np.uint64)
        self.src_start = starts
        self.dst_start = starts
        self.count = np.full((n_slices, 1), chunk_size, dtype=np.uint64)


class TimeReadManySlicesH5ToNp(_TimeReadManySlicesH5Base):
    """Benchmark read_many_slices with h5py src and NumPy dst."""

    def time_read_many_slices(self, chunk_size, fast):
        read_many_slices(
            self.h5_dset,
            self.np_arr,
            self.src_start,
            self.dst_start,
            self.count,
            fast=fast,
        )


class TimeReadManySlicesNpToH5(_TimeReadManySlicesH5Base):
    """Benchmark read_many_slices with numpy src and h5py dst."""

    def time_read_many_slices(self, chunk_size, fast):
        read_many_slices(
            self.np_arr,
            self.h5_dset,
            self.src_start,
            self.dst_start,
            self.count,
            fast=fast,
        )
