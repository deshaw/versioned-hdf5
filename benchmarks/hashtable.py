import h5py

from versioned_hdf5.hashtable import Hashtable

from .common import Benchmark, require_npystrings


class TimeHashtable(Benchmark):
    params = [
        ["f", "S", "O", "T"],
        [1, 4, 8, 64, 256],
    ]
    param_names = ["dtype", "nbytes"]

    CHUNKS = (100, 100)

    def setup(self, dtype, nbytes):
        super().setup()
        if dtype == "f" and nbytes not in (4, 8):
            raise NotImplementedError()
        if dtype in ("f", "S"):
            dtype = f"{dtype}{nbytes}"
        if dtype == "O":  # object strings
            dtype = h5py.string_dtype()
        if dtype == "T":  # NpyStrings a.k.a. StringDType
            require_npystrings()

        # Not needed to benchmark hash() but required to initialize the
        # Hashtable instance.
        with self.vfile.stage_version("init") as sv:
            # Create an initial empty hashtable
            sv.create_dataset(
                "x",
                shape=self.CHUNKS,
                chunks=self.CHUNKS,
                dtype=dtype,
                fillvalue=0 if dtype in ("f4", "f8") else "",
            )
        self.hashtable = Hashtable(self.file, "x")

        if dtype in ("f4", "f8"):
            self.chunk = self.rng.standard_normal(self.CHUNKS).astype(dtype)
        else:
            self.chunk = self.rand_strings(self.CHUNKS, 0, nbytes, dtype)

    def time_hash(self, dtype, nbytes):
        self.hashtable.hash(self.chunk)
