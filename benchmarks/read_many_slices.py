import numpy as np
from versioned_hdf5.slicetools import read_many_slices

from .common import Benchmark

# 1 KiB, 64 KiB, 1 MiB of float64 per chunk
CHUNK_SIZES = [1024 // 8, 64 * 1024 // 8, 1024 * 1024 // 8]
# Total bytes transferred per benchmark call ~16 MiB
TOTAL_ELEMENTS = 16 * 1024 * 1024 // 8

# Memory layouts of the NumPy side of the transfer
LAYOUTS = [
    # C-contiguous buffer
    "contiguous",
    # The innermost axis is C-contiguous, but the last point of a row is not adjacent in
    # memory to the first point of the next row
    "sliced",
    # 1D array in memory, with replicated rows; each row is a contiguous buffer
    "broadcast_outer",
    # 1D array in memory, with replicated columns
    "broadcast_inner",
    # The innermost axis is C-contiguous, but rows are copied in reverse order
    "reverse_outer",
    # The innermost axis is copied in reverse order
    "reverse_inner",
    # Copy every other row. The innermost axis is C-contiguous.
    "strided_outer",
    # Copy every other column.
    "strided_inner",
    # Outermost NumPy axis is the innermost one in memory (Fortran order)
    "transposed",
]

NCOLS = 128


def make_array(
    shape: tuple[int, int], layout: str, rng: np.random.Generator
) -> np.ndarray:
    """Return a NumPy array of with the given shape and layout,
    full of random float64 data
    """
    if layout == "contiguous":
        return rng.random(shape, np.float64)
    if layout == "sliced":
        return rng.random((shape[0], shape[1] + 8), np.float64)[:, 4:-4]
    if layout == "broadcast_outer":
        return np.broadcast_to(rng.random((1, shape[1]), np.float64), shape)
    if layout == "broadcast_inner":
        return np.broadcast_to(rng.random((shape[0], 1), np.float64), shape)
    if layout == "reverse_outer":
        return rng.random(shape, np.float64)[::-1, :]
    if layout == "reverse_inner":
        return rng.random(shape, np.float64)[:, ::-1]
    if layout == "strided_outer":
        return rng.random((shape[0] * 2, shape[1]), np.float64)[::2, :]
    if layout == "strided_inner":
        return rng.random((shape[0], shape[1] * 2), np.float64)[:, ::2]
    if layout == "transposed":
        return rng.random(shape[::-1], np.float64).T
    raise AssertionError(f"unknown layout: {layout}")


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


class TimeReadManySlices(Benchmark):
    """Benchmark read_many_slices between h5py src and a contiguous NumPy array"""

    params = [CHUNK_SIZES, [True, None, False]]
    param_names = ["chunk_size", "fast"]

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

    def time_read_many_slices_h5_to_np(self, chunk_size, fast):
        read_many_slices(
            self.h5_dset,
            self.np_arr,
            self.src_start,
            self.dst_start,
            self.count,
            fast=fast,
        )

    def time_read_many_slices_np_to_h5(self, chunk_size, fast):
        read_many_slices(
            self.np_arr,
            self.h5_dset,
            self.src_start,
            self.dst_start,
            self.count,
            fast=fast,
        )


class TimeReadManySlicesNonContiguous(Benchmark):
    """Benchmark read_many_slices between h5py src and a non-contiguous NumPy array"""

    params = [CHUNK_SIZES, LAYOUTS, [None, False]]
    param_names = ["chunk_size", "layout", "fast"]

    def setup(self, chunk_size, layout, fast):
        super().setup(chunk_size, layout, fast)
        rows_per_slice = max(1, chunk_size // NCOLS)
        nrows = TOTAL_ELEMENTS // NCOLS
        n_slices = nrows // rows_per_slice

        self.np_arr = make_array((nrows, NCOLS), layout, self.rng)
        self.h5_dset = self.file.create_dataset(
            "data",
            # Initialize with data so reads return something realistic
            data=self.rng.random((nrows, NCOLS)),
            chunks=(rows_per_slice, NCOLS),
        )

        # Contiguous, chunk-aligned offsets in both src and dst
        starts = np.zeros((n_slices, 2), dtype=np.uint64)
        starts[:, 0] = np.arange(n_slices) * rows_per_slice
        self.src_start = starts
        self.dst_start = starts
        self.count = np.array([rows_per_slice, NCOLS], dtype=np.uint64)

    def time_read_many_slices_h5_to_np(self, chunk_size, layout, fast):
        if "broadcast" in layout:
            from asv_runner.benchmarks.mark import SkipNotImplemented

            raise SkipNotImplemented("read-only array")

        read_many_slices(
            self.h5_dset,
            self.np_arr,
            self.src_start,
            self.dst_start,
            self.count,
            fast=fast,
        )

    def time_read_many_slices_np_to_h5(self, chunk_size, layout, fast):
        read_many_slices(
            self.np_arr,
            self.h5_dset,
            self.src_start,
            self.dst_start,
            self.count,
            fast=fast,
        )
