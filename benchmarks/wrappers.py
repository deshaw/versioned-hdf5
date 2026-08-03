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
            # Create new dataset with data=
            "v1_dense",
            # Create new dataset with shape=
            "v1_sparse_empty",
            # Create new dataset with shape=; then fill all chunks
            "v1_sparse_full",
            # Create new dataset with data=np.full(shape, fillvalue)
            # Then set a single chunk to non-fillvalue
            "v1_dense_fillvalue",
            # As above, but sparse
            "v1_sparse_fillvalue",
            # New version of existing dataset; no __setitem__ calls
            "v2_no_changes",
            # A single point changes
            "v2_one_change",
            # New version; [:] = ... hot-swaps InMemoryDataset -> InMemoryArrayDataset
            # (identical contents)
            "v2_hotswap_no_changes",
            # As above, but chunk contents change
            "v2_hotswap_all_changes",
            # Points are not updated all at once; doesn't trigger the hotswap
            "v2_modified_no_changes",
            "v2_modified_all_changes",
            # v2 replaces all chunks of v1, but v3 is identical to v1
            # [:] triggers a InMemoryDataset -> InMemoryArrayDataset hotswap
            "v3_restore_obsolete",
            # As above, but points are not update at once which prevents the hotswap
            "v3_restore_obsolete_hotswap",
        ],
        [(25, 25), (50, 50), (250, 250)],
    ]
    param_names = ["kind", "chunks"]

    def setup(self, kind, chunks):
        shape = (1000, 1000)
        if kind.endswith("_fillvalue"):
            data = np.zeros(shape, dtype=np.float64)
            data[0, 0] = 1
        else:
            rng = np.random.default_rng(0)
            data = rng.random(size=shape, dtype=np.float64)

        super().setup()
        if kind.startswith("v1_"):
            self.ctx = self.vfile.stage_version("v1")
            self.version = self.ctx.__enter__()
            if kind in ("v1_dense", "v1_dense_fillvalue"):
                self.version.create_dataset("data", data=data, chunks=chunks)
            elif kind == "v1_sparse_empty":
                self.version.create_dataset("data", shape=shape, chunks=chunks)
            elif kind in ("v1_sparse_full", "v1_sparse_fillvalue"):
                ds = self.version.create_dataset("data", shape=shape, chunks=chunks)
                # Don't write all at once, or it would trigger a hotswap
                # InMemorySparseDataset -> InMemoryArrayDataset
                ds[0] = data[0]
                ds[1:] = data[1:]
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
            elif kind == "v2_one_change":
                ds[0, 0] = data[0, 0] + 123
            elif kind == "v2_hotswap_no_changes":
                ds[:] = data
            elif kind == "v2_hotswap_all_changes":
                ds[:] = data + 123
            elif kind == "v2_modified_no_changes":
                # Don't write all at once, or it would trigger a hotswap
                # InMemoryDataset -> InMemoryArrayDataset
                ds[0] = data[0]
                ds[1:] = data[1:]
            elif kind == "v2_modified_all_changes":
                ds[0] = data[0] + 123
                ds[1:] = data[1:] + 123
            else:
                raise AssertionError("unreachable")

        elif kind.startswith("v3_"):
            with self.vfile.stage_version("v1") as version:
                version.create_dataset("data", data=data, chunks=chunks)
            with self.vfile.stage_version("v2") as version:
                version["data"] = data + 123
            self.ctx = self.vfile.stage_version("v3")
            self.version = self.ctx.__enter__()
            ds = self.version["data"]
            if kind == "v3_restore_obsolete":
                # Don't write all at once, or it would trigger a hotswap
                # InMemoryDataset -> InMemoryArrayDataset
                ds[0] = data[0]
                ds[1:] = data[1:]
            elif kind == "v3_restore_obsolete_hotswap":
                ds[:] = data
            else:
                raise AssertionError("unreachable")

        else:
            raise AssertionError("unreachable")

    def time_commit(self, kind, chunks):
        self.assert_clean_setup()
        self.ctx.__exit__(None, None, None)
