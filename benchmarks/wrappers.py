import numpy as np

from .common import Benchmark


class TimeWrappers(Benchmark):
    params = [
        "h5py.Dataset",
        "InMemoryArrayDataset",
        "InMemoryDataset",
        "InMemorySparseDataset",
    ]
    param_names = ["kind"]

    def setup(self, kind):
        super().setup()
        if kind == "h5py.Dataset":
            self.file.create_dataset(
                "data",
                data=np.arange(10000).reshape((100, 10, 10)),
                chunks=(3, 3, 3),
                maxshape=(None, None, None),
            )
            self.reopen()
            self.ds = self.file["data"]
        elif kind == "InMemoryArrayDataset":
            self.ctx = self.vfile.stage_version("v0")
            version = self.ctx.__enter__()
            self.ds = version.create_dataset(
                "data",
                data=np.arange(10000).reshape((100, 10, 10)),
                chunks=(3, 3, 3),
            )
        elif kind == "InMemoryDataset":
            with self.vfile.stage_version("v0") as ctx:
                self.ds = ctx.create_dataset(
                    "data",
                    data=np.arange(10000).reshape((100, 10, 10)),
                    chunks=(3, 3, 3),
                    maxshape=(None, None, None),
                )
            self.reopen()
            self.ctx = self.vfile.stage_version("v1")
            version = self.ctx.__enter__()
            self.ds = version["data"]
        elif kind == "InMemorySparseDataset":
            self.ctx = self.vfile.stage_version("v0")
            version = self.ctx.__enter__()
            self.ds = version.create_dataset(
                "data",
                shape=(100, 10, 10),
                chunks=(3, 3, 3),
                maxshape=(None, None, None),
            )

    def time_getattr(self, kind):
        self.ds[:, 0, 0:6]

    def time_setattr(self, kind):
        self.assert_clean_setup()
        self.ds[:, 0, 0:6] = -1
        if kind != "h5py.Dataset":
            # Include commit in the benchmark.
            # Otherwise, it makes no sense to compare versioned_hdf5 vs. h5py.
            self.ctx.__exit__(None, None, None)  # commit
            del self.ctx

    def time_resize_bigger(self, kind):
        self.assert_clean_setup()
        self.ds.resize((100, 100, 100))
        if kind != "h5py.Dataset":
            self.ctx.__exit__(None, None, None)  # commit
            del self.ctx

    def time_resize_smaller(self, kind):
        self.assert_clean_setup()
        self.ds.resize((10, 10, 10))
        if kind != "h5py.Dataset":
            self.ctx.__exit__(None, None, None)  # commit
            del self.ctx

    def teardown(self, kind):
        if hasattr(self, "ctx"):
            self.ctx.__exit__(None, None, None)  # commit
            del self.ctx
        super().teardown()


class TimeCreateDataset(Benchmark):
    params = [["dense", "sparse"], ["h5py", "versioned_hdf5"]]
    param_names = ["density", "library"]

    def setup(self, density, library):
        super().setup()
        self.kwargs = {"chunks": (3, 3, 3)}
        if density == "dense":
            self.kwargs["data"] = np.arange(10000).reshape((100, 10, 10))
        else:
            self.kwargs["shape"] = (100, 10, 10)

    def time_create_dataset(self, density, library):
        self.assert_clean_setup()
        if library == "h5py":
            self.file.create_dataset("data", **self.kwargs)
        else:
            with self.vfile.stage_version("v0") as sv:
                sv.create_dataset("data", **self.kwargs)
        self.file.close()


class TimeCommit(Benchmark):
    params = [
        [
            "v1_dense",
            "v1_sparse",
            "v2_no_changes",
            "v2_modified_no_changes",
            "v2_modified_all_changes",
        ],
        [(10, 10), (50, 50), (250, 250)],
    ]
    param_names = ["kind", "chunks"]

    def setup(self, kind, chunks):
        shape = (500, 500)
        data = np.arange(500 * 500).reshape(shape)

        super().setup()
        if kind.startswith("v1_"):
            self.ctx = self.vfile.stage_version("v1")
            self.version = self.ctx.__enter__()
            if kind == "v1_dense":
                self.version.create_dataset("data", data=data, chunks=chunks)
            elif kind == "v1_sparse":
                self.version.create_dataset("data", shape=shape, chunks=chunks)
            else:
                raise AssertionError("unreachable")

        elif kind.startswith("v2_"):
            with self.vfile.stage_version("v1") as version:
                version.create_dataset("data", data=data, chunks=chunks)
            self.ctx = self.vfile.stage_version("v2")
            self.version = self.ctx.__enter__()
            ds = self.version["data"]
            if kind == "v2_no_changes":
                pass
            elif kind == "v2_modified_no_changes":
                ds[:] = ds[:]
            elif kind == "v2_modified_all_changes":
                ds[:] = ds[:] + 123
            else:
                raise AssertionError("unreachable")

        else:
            raise AssertionError("unreachable")

    def time_commit(self, kind, chunks):
        self.assert_clean_setup()
        self.ctx.__exit__(None, None, None)
