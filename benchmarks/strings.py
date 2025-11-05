import h5py

from .common import Benchmark, require_npystrings


class TimeStrings(Benchmark):
    """Benchmark for string dtypes"""

    params = [
        ["h5py", "versioned_hdf5"],
        ["S", "O", "T"],
        [1, 8, 64, 256],
    ]
    param_names = ["library", "dtype", "max_nchars"]

    SHAPE = (200, 200)
    CHUNKS = (100, 100)

    def setup(self, library, dtype, max_nchars):
        super().setup()
        if dtype == "S":
            dtype = f"S{max_nchars}"
        elif dtype == "O":  # object strings
            dtype = h5py.string_dtype()
        elif dtype == "T":  # NpyStrings a.k.a. StringDType
            require_npystrings()

        self.data1 = self.rand_strings(self.SHAPE, 0, max_nchars, dtype)
        self.data2 = self.rand_strings(self.SHAPE, 0, max_nchars, dtype)

        if library == "h5py":
            self.file.create_dataset("x", data=self.data1, chunks=self.CHUNKS)
            self.reopen()
            self.ds = self.file["x"]
        else:
            with self.vfile.stage_version("v0") as sv:
                sv.create_dataset("x", data=self.data1, chunks=self.CHUNKS)
            self.reopen()
            self.ctx = self.vfile.stage_version("v1")
            sv = self.ctx.__enter__()
            self.ds = sv["x"]

    def time_getitem(self, library, dtype, max_nchars):
        if dtype == "T":
            _ = self.ds.astype("T")[:]
        elif dtype == "S":
            _ = self.ds[:].astype("U")
        else:
            assert dtype == "O"
            _ = self.ds.asstr()[:]

    def time_setitem(self, library, dtype, max_nchars):
        self.assert_clean_setup()
        self.ds[:] = self.data2

    def time_update_identical(self, library, dtype, max_nchars):
        self.assert_clean_setup()
        self.ds[:] = self.data1
        if library == "versioned_hdf5":
            self.ctx.__exit__(None, None, None)
            del self.ctx

    def time_update_different(self, library, dtype, max_nchars):
        self.assert_clean_setup()
        self.ds[:] = self.data2
        if library == "versioned_hdf5":
            self.ctx.__exit__(None, None, None)
            del self.ctx

    def time_read_write_loop_same_chunk(self, library, dtype, max_nchars):
        self.assert_clean_setup()
        view = self.ds.astype("T") if dtype == "T" else self.ds
        for i in range(0, self.ds.shape[0], 2):
            self.ds[i + 1, :] = view[i, :][::-1]

    def time_read_write_loop_different_chunk(self, library, dtype, max_nchars):
        self.assert_clean_setup()
        view = self.ds.astype("T") if dtype == "T" else self.ds
        for i in range(0, self.ds.shape[0] // 2):
            self.ds[-i - 1, :] = view[i, :][::-1]

    def time_read_write_loop_fast_astype(self, library, dtype, max_nchars):
        self.assert_clean_setup()
        for i in range(0, self.ds.shape[0], 2):
            view = self.ds.astype("T") if dtype == "T" else self.ds
            self.ds[i + 1, :] = view[i, :][::-1]

    def teardown(self, library, dtype, max_nchars):
        if hasattr(self, "ctx"):
            self.ctx.__exit__(None, None, None)
        super().teardown()
