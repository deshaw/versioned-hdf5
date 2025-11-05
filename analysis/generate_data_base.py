import abc
import random
from unittest import TestCase

import numpy as np
import scipy.stats


class TestDatasetPerformanceBase(TestCase, metaclass=abc.ABCMeta):
    """
    Test cases for the most common use cases where we encounter when we write data to
    HDF5.

    In general all data has multiple columns which are divided into "keys" and "values".
    The keys determine the identity of the row (this stock, this time) and the values
    are the associated values (the price, ...).

    We have two different implementation methods:
    - "sparse": key and value columns are stored as arrays of equal length and to get
      the i-th "row" you read key0[i], key1[i], ..., val0[i], val1[i], ...
    - "dense": key columns are the labels of the axes of the data and the length of the
      value column is the product of the length of the key columns:
      len(val0) == len(key0) * len(key1) * ...
      To get the i-th row you retrieve
        key0[i // len(key1) // len(key2) // ...],
        key1[(i // len(key2) // len(key3) // ...) % len(key1)],
        key2[(i // len(key3) // len(key4) // ...) % len(key2)],
        ...,
        val0[i], val1[i], ...
      TODO: check the math!
    """

    # models
    RECENCTNESS_POWERLAW_SHAPE = 20.0

    def test_mostly_appends_sparse(
        self,
        num_transactions=250,
        filename="test_mostly_appends_sparse",
        chunk_size=None,
        compression=None,
        versions=True,
        print_transactions=False,
        deterministic=False,
    ):
        num_rows_initial = 1000
        num_rows_per_append = 1000

        if deterministic:
            pct_inserts = 0
            pct_deletes = 0
            pct_changes = 0
        else:
            pct_inserts = 5
            pct_deletes = 1
            pct_changes = 5

        num_inserts = 10
        num_deletes = 10
        num_changes = 10

        return self._write_transactions_sparse(
            filename,
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
        )

    @classmethod
    @abc.abstractmethod
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
        pass

    @classmethod
    def _get_rand_fn(cls, dtype):
        if dtype == np.dtype("int64"):
            return lambda size=None: np.random.randint(0, int(1e6), size=size)
        elif dtype == np.dtype("float64"):
            return np.random.rand
        else:
            raise ValueError("implement other dtypes")

    @classmethod
    def _modify_dss_sparse(
        cls,
        key0_ds,
        key1_ds,
        val_ds,
        num_rows_per_append,
        pct_changes,
        num_changes,
        pct_deletes,
        num_deletes,
        pct_inserts,
        num_inserts,
    ):
        ns = set([len(ds) for ds in [key0_ds, key1_ds, val_ds]])
        assert len(ns) == 1
        n = next(iter(ns))
        # change values
        if random.randrange(0, 100) <= pct_changes:
            r_num_chgs = int(np.random.randn() + num_changes)
            rand_fn = cls._get_rand_fn(val_ds.dtype)
            for _ in range(r_num_chgs):
                r = random.randrange(0, n)
                val_ds[r] = rand_fn()
        # delete rows
        if random.randrange(0, 100) <= pct_deletes:
            r_num_dels = max(int(np.random.randn() + num_deletes), 1)
            pdf = scipy.stats.powerlaw.rvs(
                TestDatasetPerformanceBase.RECENCTNESS_POWERLAW_SHAPE, size=r_num_dels
            )
            rs = np.unique((pdf * n).astype("int64"))
            minr = min(rs)
            n -= len(rs)
            for ds in [key0_ds, key1_ds, val_ds]:
                arr = ds[minr:]
                arr = np.delete(arr, rs - minr)
                ds.resize((n,))
                ds[minr:] = arr
        # insert rows
        if random.randrange(0, 100) <= pct_inserts:
            rand_num_inss = max(int(np.random.randn() + num_inserts), 1)
            pdf = scipy.stats.powerlaw.rvs(
                TestDatasetPerformanceBase.RECENCTNESS_POWERLAW_SHAPE,
                size=rand_num_inss,
            )
            rs = np.unique((pdf * n).astype("int64"))
            minr = min(rs)
            n += len(rs)
            for ds in [key0_ds, key1_ds, val_ds]:
                rand_fn = cls._get_rand_fn(ds.dtype)
                arr = ds[minr:]
                arr = np.insert(arr, rs - minr, [rand_fn() for _ in rs])
                ds.resize((n,))
                ds[minr:] = arr
        # append
        rand_num_apps = int(10 * np.random.randn() + num_rows_per_append)
        if rand_num_apps > 0:
            n += rand_num_apps
            for ds in [key0_ds, key1_ds, val_ds]:
                rand_fn = cls._get_rand_fn(ds.dtype)
                ds.resize((n,))
                ds[-rand_num_apps:] = rand_fn(rand_num_apps)

    def test_large_fraction_changes_sparse(
        self,
        num_transactions=250,
        filename="test_large_fraction_changes_sparse",
        chunk_size=None,
        compression=None,
        versions=True,
        print_transactions=False,
        deterministic=False,
    ):
        num_rows_initial = 5000
        num_rows_per_append = 10

        if deterministic:
            pct_inserts = 0
            pct_deletes = 0
            pct_changes = 0
        else:
            pct_inserts = 1
            pct_deletes = 1
            pct_changes = 90

        num_inserts = 10
        num_deletes = 10
        num_changes = 1000

        return self._write_transactions_sparse(
            filename,
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
        )

    def test_small_fraction_changes_sparse(
        self,
        num_transactions=250,
        filename="test_small_fraction_changes_sparse",
        chunk_size=None,
        compression=None,
        versions=True,
        print_transactions=False,
        deterministic=False,
    ):
        num_rows_initial = 5000
        num_rows_per_append = 10

        if deterministic:
            pct_inserts = 0
            pct_deletes = 0
            pct_changes = 0
        else:
            pct_inserts = 1
            pct_deletes = 1
            pct_changes = 90

        num_inserts = 10
        num_deletes = 10
        num_changes = 10

        return self._write_transactions_sparse(
            filename,
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
        )

    def test_large_fraction_constant_sparse(
        self,
        num_transactions=250,
        filename="test_large_fraction_constant_sparse",
        chunk_size=None,
        compression=None,
        versions=True,
        print_transactions=False,
        deterministic=False,  # noqa: ARG002
    ):
        num_rows_initial = 5000
        num_rows_per_append = 0  # triggers the constant size test (FIXME)

        pct_inserts = 0
        pct_deletes = 0
        pct_changes = 0

        num_inserts = 10
        num_deletes = 10
        num_changes = 1000

        return self._write_transactions_sparse(
            filename,
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
        )

    def test_mostly_appends_dense(
        self,
        num_transactions=250,
        filename="test_mostly_appends_dense",
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

        return self._write_transactions_dense(
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

    @classmethod
    @abc.abstractmethod
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
        pass

    @classmethod
    def _modify_dss_dense(
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
        val_shape = val_ds.shape
        assert val_shape == (n_key0, n_key1)
        # change values
        if random.randrange(0, 100) <= pct_changes:
            r_num_chgs = int(np.random.randn() + num_changes)
            for _ in range(r_num_chgs):
                r = (random.randrange(0, n_key0), random.randrange(0, n_key1))
                val_ds[r] = np.random.rand()
        # delete rows
        if random.randrange(0, 100) <= pct_deletes:
            # delete from values in two steps

            # 1. delete from key0 and associated vals
            r_num_dels_0 = max(int(np.random.randn() + num_deletes_0), 1)
            pdf = scipy.stats.powerlaw.rvs(
                TestDatasetPerformanceBase.RECENCTNESS_POWERLAW_SHAPE, size=r_num_dels_0
            )
            rs_0 = np.unique((pdf * n_key0).astype("int64"))
            minr_0 = min(rs_0)

            n_key0 -= len(rs_0)
            arr_key0 = key0_ds[minr_0:]
            arr_key0 = np.delete(arr_key0, rs_0 - minr_0)
            key0_ds.resize((n_key0,))
            key0_ds[minr_0:] = arr_key0

            arr_val = val_ds[minr_0:, :]
            val_shape = (val_shape[0] - len(rs_0), val_shape[1])
            val_ds.resize(val_shape)
            arr_val = np.delete(arr_val, rs_0 - minr_0, axis=0)
            val_ds[minr_0:, :] = arr_val

            # 2. delete from key1 and associated vals
            r_num_dels_1 = max(int(np.random.randn() + num_deletes_1), 1)
            pdf = scipy.stats.powerlaw.rvs(
                TestDatasetPerformanceBase.RECENCTNESS_POWERLAW_SHAPE, size=r_num_dels_1
            )
            rs_1 = np.unique((pdf * n_key1).astype("int64"))
            minr_1 = min(rs_1)

            n_key1 -= len(rs_1)
            arr_key1 = key1_ds[minr_1:]
            arr_key1 = np.delete(arr_key1, rs_1 - minr_1)
            key1_ds.resize((n_key1,))
            key1_ds[minr_1:] = arr_key1

            arr_val = val_ds[:, minr_1:]
            val_shape = (val_shape[0], val_shape[1] - len(rs_1))
            val_ds.resize(val_shape)
            arr_val = np.delete(arr_val, rs_1 - minr_1, axis=1)
            val_ds[:, minr_1:] = arr_val
        # insert rows
        if random.randrange(0, 100) <= pct_inserts:
            # insert into values in two steps

            # 1. insert into key0 and associated vals
            rand_num_inss_0 = max(int(np.random.randn() + num_inserts_0), 1)
            pdf = scipy.stats.powerlaw.rvs(
                TestDatasetPerformanceBase.RECENCTNESS_POWERLAW_SHAPE,
                size=rand_num_inss_0,
            )
            rs_0 = np.unique((pdf * n_key0).astype("int64"))
            minr_0 = min(rs_0)

            arr_key0 = key0_ds[minr_0:]
            arr_key0 = np.insert(
                arr_key0, rs_0 - minr_0, np.random.randint(0, int(1e6), size=len(rs_0))
            )
            n_key0 += len(rs_0)
            key0_ds.resize((n_key0,))
            key0_ds[minr_0:] = arr_key0

            arr_val = val_ds[minr_0:, :]
            val_shape = (val_shape[0] + len(rs_0), val_shape[1])
            val_ds.resize(val_shape)
            arr_val = np.insert(
                arr_val, rs_0 - minr_0, np.random.rand(len(rs_0), n_key1), axis=0
            )
            val_ds[minr_0:, :] = arr_val

            # 2. insert into key1 and associated vals
            rand_num_inss_1 = max(int(np.random.randn() + num_inserts_1), 1)
            pdf = scipy.stats.powerlaw.rvs(
                TestDatasetPerformanceBase.RECENCTNESS_POWERLAW_SHAPE,
                size=rand_num_inss_1,
            )
            rs_1 = np.unique((pdf * n_key1).astype("int64"))
            minr_1 = min(rs_1)

            arr_key1 = key1_ds[minr_1:]
            arr_key1 = np.insert(
                arr_key1, rs_1 - minr_1, np.random.randint(0, int(1e6), size=len(rs_1))
            )
            n_key1 += len(rs_1)
            key1_ds.resize((n_key1,))
            key1_ds[minr_1:] = arr_key1

            arr_val = val_ds[:, minr_1:]
            val_shape = (val_shape[0], val_shape[1] + len(rs_1))
            val_ds.resize(val_shape)
            arr_val = np.insert(
                arr_val, rs_1 - minr_1, np.random.rand(n_key0, len(rs_1)), axis=1
            )
            val_ds[:, minr_1:] = arr_val
        # append
        rand_num_apps_0 = int(np.random.randn() + num_rows_per_append_0)
        if rand_num_apps_0 > 0:
            # append to key0 and associated vals
            n_key0 += rand_num_apps_0
            key0_ds.resize((n_key0,))
            key0_ds[-rand_num_apps_0:] = np.random.randint(
                0, int(1e6), size=rand_num_apps_0
            )

            val_shape = (n_key0, n_key1)
            val_ds.resize(val_shape)
            val_ds[-rand_num_apps_0:, :] = np.random.rand(rand_num_apps_0, n_key1)
