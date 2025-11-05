import datetime
import logging
import random
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager

import h5py
import numpy as np

sys.path.append("..")

from generate_data_base import TestDatasetPerformanceBase

from versioned_hdf5.api import VersionedHDF5File


@contextmanager
def temp_dir_ctx():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)


class TestVersionedDatasetPerformance(TestDatasetPerformanceBase):
    @classmethod
    def _write_transactions_sparse(
        cls,
        name,
        chunk_size,
        compression,
        versions,
        print_transactions,
        num_rows_initial,
        num_transactions,
        num_rows_per_append,
        pct_changes,
        num_changes,
        pct_deletes,
        num_deletes,
        pct_inserts,
        num_inserts,
    ):
        logger = logging.getLogger(__name__)

        filename = f"{name}.h5"
        tts = []
        f = h5py.File(filename, "w")
        told = time.time()
        t0 = told
        times = []
        try:
            if versions:
                file = VersionedHDF5File(f)
                with file.stage_version("initial_version") as group:
                    key0_ds = group.create_dataset(
                        name + "/key0",
                        data=np.random.rand(num_rows_initial),
                        dtype=(np.dtype("int64")),
                        chunks=chunk_size,
                        compression=compression,
                    )
                    key1_ds = group.create_dataset(
                        name + "/key1",
                        data=np.random.rand(num_rows_initial),
                        dtype=(np.dtype("int64")),
                        chunks=chunk_size,
                        compression=compression,
                    )
                    val_ds = group.create_dataset(
                        name + "/val",
                        data=np.random.rand(num_rows_initial),
                        dtype=(np.dtype("float64")),
                        chunks=chunk_size,
                        compression=compression,
                    )
            else:
                key0_ds = f.create_dataset(
                    name + "/key0",
                    data=np.random.rand(num_rows_initial),
                    dtype=(np.dtype("int64")),
                    maxshape=(None,),
                    chunks=(chunk_size,),
                    compression=compression,
                )
                key1_ds = f.create_dataset(
                    name + "/key1",
                    data=np.random.rand(num_rows_initial),
                    dtype=(np.dtype("int64")),
                    maxshape=(None,),
                    chunks=(chunk_size,),
                    compression=compression,
                )
                val_ds = f.create_dataset(
                    name + "/val",
                    data=np.random.rand(num_rows_initial),
                    dtype=(np.dtype("float64")),
                    maxshape=(None,),
                    chunks=(chunk_size,),
                    compression=compression,
                )

            for a in range(num_transactions):
                if print_transactions:
                    print("Transaction", a)
                tt = datetime.datetime.utcnow()
                if versions:
                    with file.stage_version(str(tt)) as group:
                        key0_ds = group[name + "/key0"]
                        key1_ds = group[name + "/key1"]
                        val_ds = group[name + "/val"]
                        cls._modify_dss_sparse(
                            key0_ds,
                            key1_ds,
                            val_ds,
                            num_rows_per_append,
                            pct_changes if a > 0 else 0.0,
                            num_changes,
                            pct_deletes if a > 0 else 0.0,
                            num_deletes,
                            pct_inserts if a > 0 else 0.0,
                            num_inserts,
                        )
                else:
                    cls._modify_dss_sparse(
                        key0_ds,
                        key1_ds,
                        val_ds,
                        num_rows_per_append,
                        pct_changes if a > 0 else 0.0,
                        num_changes,
                        pct_deletes if a > 0 else 0.0,
                        num_deletes,
                        pct_inserts if a > 0 else 0.0,
                        num_inserts,
                    )
                t = time.time()
                times.append(t - told)
                told = t
                tts.append(tt)
                logger.info("Wrote transaction %d at transaction time %s", a, tt)
                f.flush()
            times.append(t - t0)
        finally:
            f.close()
        return times

    @classmethod
    def _write_transactions_dense(
        cls,
        name,
        chunk_size,
        compression,
        versions,
        print_transactions,
        num_rows_initial_0,
        num_rows_initial_1,
        num_transactions,
        num_rows_per_append_0,
        pct_changes,
        num_changes,
        pct_deletes,
        num_deletes_0,
        num_deletes_1,
        pct_inserts,
        num_inserts_0,
        num_inserts_1,
    ):
        logger = logging.getLogger(__name__)

        # with temp_dir_ctx() as tmp_dir:
        # with f"/{name}.h5" as filename:
        filename = f"{name}.h5"
        tts = []
        f = h5py.File(filename, "w")
        told = time.time()
        t0 = told
        times = []
        try:
            if versions:
                file = VersionedHDF5File(f)
                with file.stage_version("initial_version") as group:
                    key0_ds = group.create_dataset(
                        name + "/key0",
                        data=np.random.rand(num_rows_initial_0),
                        dtype=(np.dtype("int64")),
                        chunks=chunk_size,
                        compression=compression,
                    )
                    key1_ds = group.create_dataset(
                        name + "/key1",
                        data=np.random.rand(num_rows_initial_1),
                        dtype=(np.dtype("int64")),
                        chunks=chunk_size,
                        compression=compression,
                    )
                    # two dimensional value array
                    val_ds = group.create_dataset(
                        name + "/val",
                        data=np.random.rand(num_rows_initial_0, num_rows_initial_1),
                        dtype=np.dtype("float64"),
                        chunks=(chunk_size, chunk_size),
                        compression=compression,
                    )
            else:
                key0_ds = f.create_dataset(
                    name + "/key0",
                    data=np.random.rand(num_rows_initial_0),
                    dtype=np.dtype("int64"),
                    maxshape=(None,),
                    chunks=(chunk_size,),
                    compression=compression,
                )
                key1_ds = f.create_dataset(
                    name + "/key1",
                    data=np.random.rand(num_rows_initial_0),
                    dtype=np.dtype("int64"),
                    maxshape=(None,),
                    chunks=(chunk_size,),
                    compression=compression,
                )
                val_ds = f.create_dataset(
                    name + "/val",
                    data=np.random.rand(num_rows_initial_0, num_rows_initial_1),
                    dtype=np.dtype("float64"),
                    maxshape=(None, None),
                    chunks=(chunk_size, chunk_size),
                    compression=compression,
                )

            for a in range(num_transactions):
                if print_transactions:
                    print(f"Transaction {a} of {num_transactions}")
                tt = datetime.datetime.utcnow()
                if versions:
                    with file.stage_version(str(tt)) as group:
                        key0_ds = group[name + "/key0"]
                        key1_ds = group[name + "/key1"]
                        val_ds = group[name + "/val"]
                        cls._modify_dss_dense(
                            key0_ds,
                            key1_ds,
                            val_ds,
                            num_rows_per_append_0,
                            pct_changes if a > 0 else 0.0,
                            num_changes,
                            pct_deletes if a > 0 else 0.0,
                            num_deletes_0,
                            num_deletes_1,
                            pct_inserts if a > 0 else 0.0,
                            num_inserts_0,
                            num_inserts_1,
                        )
                else:
                    cls._modify_dss_dense(
                        key0_ds,
                        key1_ds,
                        val_ds,
                        num_rows_per_append_0,
                        pct_changes if a > 0 else 0.0,
                        num_changes,
                        pct_deletes if a > 0 else 0.0,
                        num_deletes_0,
                        num_deletes_1,
                        pct_inserts if a > 0 else 0.0,
                        num_inserts_0,
                        num_inserts_1,
                    )

                t = time.time()
                times.append(t - told)
                told = t
                tts.append(tt)
                logger.info("Wrote transaction %d at transaction time %s", a, tt)
                f.flush()
            times.append(t - t0)
        finally:
            f.close()
        return times

    @classmethod
    def _write_transactions_dense_old(
        cls,
        name,
        num_rows_initial_0,
        num_rows_initial_1,
        num_transactions,
        num_rows_per_append_0,
        pct_changes,
        num_changes,
        pct_deletes,
        num_deletes_0,
        num_deletes_1,
        pct_inserts,
        num_inserts_0,
        num_inserts_1,
    ):
        logger = logging.getLogger(__name__)

        with temp_dir_ctx() as tmp_dir:
            filename = tmp_dir + f"/{name}.h5"
            tts = []
            f = h5py.File(filename, "w")
            file = VersionedHDF5File(f)
            try:
                with file.stage_version("initial_version") as group:
                    key0_ds = group.create_dataset(
                        name + "/key0",
                        data=np.random.rand(num_rows_initial_0),
                        dtype=(np.dtype("int64")),
                    )
                    key1_ds = group.create_dataset(
                        name + "/key1",
                        data=np.random.rand(num_rows_initial_1),
                        dtype=(np.dtype("int64")),
                    )
                    val_ds = group.create_dataset(
                        name + "/val",
                        data=np.random.rand(num_rows_initial_0 * num_rows_initial_1),
                        dtype=(np.dtype("float64")),
                    )
                for a in range(num_transactions):
                    tt = datetime.datetime.utcnow()
                    with file.stage_version(str(tt)) as group:
                        key0_ds = group[name + "/key0"]
                        key1_ds = group[name + "/key1"]
                        val_ds = group[name + "/val"]
                        cls._modify_dss_dense_old(
                            key0_ds,
                            key1_ds,
                            val_ds,
                            num_rows_per_append_0,
                            pct_changes if a > 0 else 0.0,
                            num_changes,
                            pct_deletes if a > 0 else 0.0,
                            num_deletes_0,
                            num_deletes_1,
                            pct_inserts if a > 0 else 0.0,
                            num_inserts_0,
                            num_inserts_1,
                        )

                    tts.append(tt)
                    logger.info("Wrote transaction %d at transaction time %s", a, tt)
            finally:
                f.close()

    @classmethod
    def _modify_dss_dense_old(
        cls,
        key0_ds,
        key1_ds,
        val_ds,
        num_rows_per_append_0,
        pct_changes,
        num_changes,
        pct_deletes,
        num_deletes_0,
        num_deletes_1,
        pct_inserts,
        num_inserts_0,
        num_inserts_1,
    ):
        n_key0 = len(key0_ds)
        n_key1 = len(key1_ds)
        n_val = len(val_ds)
        assert n_val == n_key0 * n_key1
        # change values
        if random.randrange(0, 100) <= pct_changes:
            r_num_chgs = int(np.random.randn() + num_changes)
            for _ in range(r_num_chgs):
                r = random.randrange(0, n_val)
                val_ds[r] = np.random.rand()
        # delete rows
        if random.randrange(0, 100) <= pct_deletes:
            # delete from values in two steps

            # 1. delete from key0 and associated vals
            r_num_dels_0 = max(int(np.random.randn() + num_deletes_0), 1)
            rs_0 = [random.randrange(0, n_key0) for _ in range(r_num_dels_0)]

            rs_val = [r0 * n_key1 + r1 for r0 in rs_0 for r1 in range(n_key1)]
            n_val -= len(rs_val)
            arr_val = val_ds[:]
            arr_val = np.delete(arr_val, rs_val)

            n_key0 -= r_num_dels_0
            arr_key0 = key0_ds[:]
            arr_key0 = np.delete(arr_key0, rs_0)
            key0_ds.resize((n_key0,), refcheck=False)
            key0_ds[:] = arr_key0

            # 2. delete from key1 and associated vals
            r_num_dels_1 = max(int(np.random.randn() + num_deletes_1), 1)
            rs_1 = [random.randrange(0, n_key1) for _ in range(r_num_dels_1)]

            rs_val = [r0 * n_key1 + r1 for r0 in range(n_key0) for r1 in rs_1]
            n_val -= len(rs_val)
            arr_val = np.delete(arr_val, rs_val)
            val_ds.resize((n_val,), refcheck=False)
            val_ds[:] = arr_val

            n_key1 -= r_num_dels_1
            arr_key1 = key1_ds[:]
            arr_key1 = np.delete(arr_key1, rs_1)
            key1_ds.resize((n_key1,), refcheck=False)
            key1_ds[:] = arr_key1
        # insert rows
        if random.randrange(0, 100) <= pct_inserts:
            # insert into values in two steps
            arr_val = val_ds[:]

            # 1. insert into key0 and associated vals
            rand_num_inss_0 = max(int(np.random.randn() + num_inserts_0), 1)
            rs_0 = [random.randrange(0, n_key0) for _ in range(rand_num_inss_0)]

            rs_val = [r0 * n_key1 + r1 for r0 in rs_0 for r1 in range(n_key1)]
            n_val += len(rs_val)
            arr_val = np.insert(arr_val, rs_val, [np.random.rand() for _ in rs_val])

            arr_key0 = key0_ds[:]
            arr_key0 = np.insert(
                arr_key0, rs_0, np.random.randint(0, int(1e6), size=len(rs_0))
            )
            n_key0 += rand_num_inss_0
            key0_ds.resize((n_key0,), refcheck=False)
            key0_ds[:] = arr_key0

            # 2. insert into key1 and associated vals
            rand_num_inss_1 = max(int(np.random.randn() + num_inserts_1), 1)
            rs_1 = [random.randrange(0, n_key1) for _ in range(rand_num_inss_1)]

            rs_val = [r0 * n_key1 + r1 for r0 in range(n_key0) for r1 in rs_1]
            n_val += len(rs_val)
            arr_val = np.insert(arr_val, rs_val, np.random.rand(len(rs_val)))
            val_ds.resize((n_val,), refcheck=False)
            val_ds[:] = arr_val

            arr_key1 = key1_ds[:]
            arr_key1 = np.insert(
                arr_key1, rs_1, np.random.randint(0, int(1e6), size=len(rs_1))
            )
            n_key1 += rand_num_inss_1
            key1_ds.resize((n_key1,), refcheck=False)
            key1_ds[:] = arr_key1
        # append
        rand_num_apps_0 = int(np.random.randn() + num_rows_per_append_0)
        if rand_num_apps_0 > 0:
            # append to key0 and associated vals
            n_key0 += rand_num_apps_0
            key0_ds.resize((n_key0,), refcheck=False)
            key0_ds[-rand_num_apps_0:] = np.random.randint(
                0, int(1e6), size=rand_num_apps_0
            )

            num_val_apps = n_key1 * rand_num_apps_0
            n_val += num_val_apps
            val_ds.resize((n_val,), refcheck=False)
            val_ds[-num_val_apps:] = np.random.rand(num_val_apps)

    def test_mostly_appends_dense_old(
        self,
        num_transactions=250,
        filename="test_mostly_appends_dense_old",
        chunk_size=None,
        compression=None,
        versions=True,
        print_transactions=False,
        deterministic=False,
    ):
        num_rows_initial_0 = 30
        num_rows_initial_1 = 30
        num_rows_per_append_0 = 1

        if deterministic:
            pct_inserts = 0
            pct_deletes = 0
            pct_changes = 0
        else:
            pct_inserts = 5
            pct_deletes = 1
            pct_changes = 5

        num_inserts_0 = 1
        num_inserts_1 = 10
        num_deletes_0 = 1
        num_deletes_1 = 1
        num_changes = 10

        return self._write_transactions_dense_old(
            filename,
            chunk_size,
            compression,
            versions,
            print_transactions,
            num_rows_initial_0,
            num_rows_initial_1,
            num_transactions,
            num_rows_per_append_0,
            pct_changes,
            num_changes,
            pct_deletes,
            num_deletes_0,
            num_deletes_1,
            pct_inserts,
            num_inserts_0,
            num_inserts_1,
        )


if __name__ == "__main__":
    num_transactions = [500]
    for t in num_transactions:
        times = TestVersionedDatasetPerformance().test_large_fraction_changes_sparse(
            t, print_transactions=True
        )
