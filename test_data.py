from __future__ import (absolute_import, division, print_function, with_statement)


import datetime
import logging
import random
from unittest import TestCase

import h5py
import numpy as np

# from .utils import temp_dir_ctx


class TestVersionedDatasetPerformance(TestCase):
    """
    Test cases for the most common use cases where we encounter when we write data to HDF5.

    In general all data has multiple columns which are divided into "keys" and "values". The keys determine
    the identity of the row (this stock, this time) and the values are the associated values (the price, ...).


    We have two different implementation methods:
    - "sparse": key and value columns are stored as arrays of equal length and to get the i-th "row" you
      read key0[i], key1[i], ..., val0[i], val1[i], ...
    - "dense": key columns are the labels of the axes of the data and the length of the value column is the product
      of the length of the key columns: len(val0) == len(key0) * len(key1) * ...
      To get the i-th row you retrieve
        key0[i // len(key1) // len(key2) // ...],
        key1[(i // len(key2) // len(key3) // ...) % len(key1)],
        key2[(i // len(key3) // len(key4) // ...) % len(key2)],
        ...,
        val0[i], val1[i], ...
      TODO: check the math!
    """
    def test_mostly_appends_sparse(self):
        num_transactions = 250 * 10

        num_rows_initial = 1000

        num_rows_per_append = 1000

        pct_inserts = 5
        num_inserts = 10

        pct_deletes = 1
        num_deletes = 10

        pct_changes = 5
        num_changes = 10

        name = 'test_mostly_appends'

        self._write_transactions_sparse(name, num_rows_initial, num_transactions, num_rows_per_append, pct_changes,
                                        num_changes, pct_deletes, num_deletes, pct_inserts, num_inserts)


    @classmethod
    def _write_transactions_sparse(cls, name, num_rows_initial, num_transactions, num_rows_per_append,
                                   pct_changes, num_changes,
                                   pct_deletes, num_deletes,
                                   pct_inserts, num_inserts):
        logger = logging.getLogger(__name__)

        tmp_dir = '.'
        filename = tmp_dir + f'/{name}.h5'
        tts = []
        f = h5py.File(filename, 'w')
        try:
            key0_ds = f.create_dataset(name + '/key0', data=np.random.rand(num_rows_initial),
                                       dtype=(np.dtype('int64')), maxshape=(None,), chunks=(int(1e4),))
            key1_ds = f.create_dataset(name + '/key1', data=np.random.rand(num_rows_initial),
                                       dtype=(np.dtype('int64')), maxshape=(None,), chunks=(int(1e4),))
            val_ds = f.create_dataset(name + '/val', data=np.random.rand(num_rows_initial),
                                      dtype=(np.dtype('float64')), maxshape=(None,), chunks=(int(1e4),))
            for a in range(num_transactions):
                tt = cls._modify_dss_sparse(key0_ds, key1_ds, val_ds, num_rows_per_append,
                                            pct_changes if a > 0 else 0.0, num_changes,
                                            pct_deletes if a > 0 else 0.0, num_deletes,
                                            pct_inserts if a > 0 else 0.0, num_inserts)
                tts.append(tt)
                logger.info('Wrote transaction %d at transaction time %s', a, tt)
        finally:
            f.close()


    @classmethod
    def _get_rand_fn(cls, dtype):
        if dtype == np.dtype('int64'):
            return lambda size=None: np.random.randint(0, int(1e6), size=size)
        elif dtype == np.dtype('float64'):
            return np.random.rand
        else:
            raise ValueError('implement other dtypes')


    @classmethod
    def _modify_dss_sparse(cls, key0_ds, key1_ds, val_ds, num_rows_per_append,
                          pct_changes, num_changes,
                           pct_deletes, num_deletes,
                           pct_inserts, num_inserts):
        tt = datetime.datetime.utcnow()
        ns = set([len(ds) for ds in [key0_ds, key1_ds, val_ds]])
        assert len(ns) == 1
        n = next(iter(ns))
        # change values
        if random.randrange(0, 100) <= pct_changes:
            r_num_chgs = int(np.random.randn() + num_changes)
            rand_fn = cls._get_rand_fn(val_ds.dtype)
            for b in range(r_num_chgs):
                r = random.randrange(0, n)
                val_ds[r] = rand_fn()
        # delete rows
        if random.randrange(0, 100) <= pct_deletes:
            r_num_dels = max(int(np.random.randn() + num_deletes), 1)
            rs = [random.randrange(0, n) for _ in range(r_num_dels)]
            n -= r_num_dels
            for ds in [key0_ds, key1_ds, val_ds]:
                arr = ds[:]
                arr = np.delete(arr, rs)
                ds.resize((n,))
                ds[:] = arr
        # insert rows
        if random.randrange(0, 100) <= pct_inserts:
            rand_num_inss = max(int(np.random.randn() + num_inserts), 1)
            rs = [random.randrange(0, n) for _ in range(rand_num_inss)]
            n += rand_num_inss
            for ds in [key0_ds, key1_ds, val_ds]:
                rand_fn = cls._get_rand_fn(ds.dtype)
                arr = ds[:]
                arr = np.insert(arr, rs, [rand_fn() for _ in rs])
                ds.resize((n,))
                ds[:] = arr
        # append
        rand_num_apps = int(10 * np.random.randn() + num_rows_per_append)
        if rand_num_apps > 0:
            n += rand_num_apps
            for ds in [key0_ds, key1_ds, val_ds]:
                rand_fn = cls._get_rand_fn(ds.dtype)
                ds.resize((n,))
                ds[-rand_num_apps:] = rand_fn(rand_num_apps)
        return tt


    def test_large_fraction_changes_sparse(self):
        num_transactions = 250 * 10

        num_rows_initial = 5000

        num_rows_per_append = 10

        pct_inserts = 1
        num_inserts = 10

        pct_deletes = 1
        num_deletes = 10

        pct_changes = 90
        num_changes = 1000

        name = 'test_large_fraction_changes'

        self._write_transactions_sparse(name, num_rows_initial, num_transactions, num_rows_per_append, pct_changes,
                                        num_changes, pct_deletes, num_deletes, pct_inserts, num_inserts)


    def test_small_fraction_changes_sparse(self):
        num_transactions = 250 * 10

        num_rows_initial = 5000

        num_rows_per_append = 10

        pct_inserts = 1
        num_inserts = 10

        pct_deletes = 1
        num_deletes = 10

        pct_changes = 90
        num_changes = 10

        name = 'test_small_fraction_changes'

        self._write_transactions_sparse(name, num_rows_initial, num_transactions, num_rows_per_append, pct_changes,
                                        num_changes, pct_deletes, num_deletes, pct_inserts, num_inserts)


    def test_mostly_appends_dense(self):
        num_transactions = 250 * 10

        num_rows_initial_0 = 30
        num_rows_initial_1 = 30

        num_rows_per_append_0 = 1

        pct_inserts = 5
        num_inserts_0 = 1
        num_inserts_1 = 10

        pct_deletes = 1
        num_deletes_0 = 1
        num_deletes_1 = 1

        pct_changes = 5
        num_changes = 10

        name = 'test_mostly_appends'

        self._write_transactions_dense(name, num_rows_initial_0, num_rows_initial_1,
                                       num_transactions,
                                       num_rows_per_append_0,
                                       pct_changes, num_changes,
                                       pct_deletes, num_deletes_0, num_deletes_1,
                                       pct_inserts, num_inserts_0, num_inserts_1)


    @classmethod
    def _write_transactions_dense(cls, name, num_rows_initial_0, num_rows_initial_1,
                                  num_transactions,
                                  num_rows_per_append_0,
                                  pct_changes, num_changes,
                                  pct_deletes, num_deletes_0, num_deletes_1,
                                  pct_inserts, num_inserts_0, num_inserts_1):
        logger = logging.getLogger(__name__)

        tmp_dir = '.'
        filename = tmp_dir + f'/{name}.h5'
        tts = []
        f = h5py.File(filename, 'w')
        try:
            key0_ds = f.create_dataset(name + '/key0', data=np.random.rand(num_rows_initial_0),
                                       dtype=(np.dtype('int64')), maxshape=(None,), chunks=(int(1e4),))
            key1_ds = f.create_dataset(name + '/key1', data=np.random.rand(num_rows_initial_1),
                                       dtype=(np.dtype('int64')), maxshape=(None,), chunks=(int(1e4),))
            val_ds = f.create_dataset(name + '/val', data=np.random.rand(num_rows_initial_0 * num_rows_initial_1),
                                      dtype=(np.dtype('float64')), maxshape=(None,), chunks=(int(1e4),))
            for a in range(num_transactions):
                tt = cls._modify_dss_dense(key0_ds, key1_ds, val_ds,
                                           num_rows_per_append_0,
                                           pct_changes if a > 0 else 0.0, num_changes,
                                           pct_deletes if a > 0 else 0.0, num_deletes_0, num_deletes_1,
                                           pct_inserts if a > 0 else 0.0, num_inserts_0, num_inserts_1)
                tts.append(tt)
                logger.info('Wrote transaction %d at transaction time %s', a, tt)
        finally:
            f.close()


    @classmethod
    def _modify_dss_dense(cls, key0_ds, key1_ds, val_ds,
                          num_rows_per_append_0,
                          pct_changes, num_changes,
                          pct_deletes, num_deletes_0, num_deletes_1,
                          pct_inserts, num_inserts_0, num_inserts_1):
        tt = datetime.datetime.utcnow()
        n_key0 = len(key0_ds)
        n_key1 = len(key1_ds)
        n_val = len(val_ds)
        assert n_val == n_key0 * n_key1
        # change values
        if random.randrange(0, 100) <= pct_changes:
            r_num_chgs = int(np.random.randn() + num_changes)
            for b in range(r_num_chgs):
                r = random.randrange(0, n_val)
                val_ds[r] = np.random.rand()
        # delete rows
        if random.randrange(0, 100) <= pct_deletes:
            # delete from values in two steps
            arr_val = val_ds[:]

            # 1. delete from key0 and associated vals
            r_num_dels_0 = max(int(np.random.randn() + num_deletes_0), 1)
            rs_0 = [random.randrange(0, n_key0) for _ in range(r_num_dels_0)]

            rs_val = [r0 * n_key1 + r1 for r0 in rs_0 for r1 in range(n_key1)]
            n_val -= len(rs_val)
            arr_val = np.delete(arr_val, rs_val)

            n_key0 -= r_num_dels_0
            arr_key0 = key0_ds[:]
            arr_key0 = np.delete(arr_key0, rs_0)
            key0_ds.resize((n_key0,))
            key0_ds[:] = arr_key0

            # 2. delete from key1 and associated vals
            r_num_dels_1 = max(int(np.random.randn() + num_deletes_1), 1)
            rs_1 = [random.randrange(0, n_key1) for _ in range(r_num_dels_1)]

            rs_val = [r0 * n_key1 + r1 for r0 in range(n_key0) for r1 in rs_1]
            n_val -= len(rs_val)
            arr_val = np.delete(arr_val, rs_val)
            val_ds.resize((n_val,))
            val_ds[:] = arr_val

            n_key1 -= r_num_dels_1
            arr_key1 = key1_ds[:]
            arr_key1 = np.delete(arr_key1, rs_1)
            key1_ds.resize((n_key1,))
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
            arr_key0 = np.insert(arr_key0, rs_0, np.random.randint(0, int(1e6), size=len(rs_0)))
            n_key0 += rand_num_inss_0
            key0_ds.resize((n_key0,))
            key0_ds[:] = arr_key0

            # 2. insert into key1 and associated vals
            rand_num_inss_1 = max(int(np.random.randn() + num_inserts_1), 1)
            rs_1 = [random.randrange(0, n_key1) for _ in range(rand_num_inss_1)]

            rs_val = [r0 * n_key1 + r1 for r0 in range(n_key0) for r1 in rs_1]
            n_val += len(rs_val)
            arr_val = np.insert(arr_val, rs_val, np.random.rand(len(rs_val)))
            val_ds.resize((n_val,))
            val_ds[:] = arr_val

            arr_key1 = key1_ds[:]
            arr_key1 = np.insert(arr_key1, rs_1, np.random.randint(0, int(1e6), size=len(rs_1)))
            n_key1 += rand_num_inss_1
            key1_ds.resize((n_key1,))
            key1_ds[:] = arr_key1
        # append
        rand_num_apps_0 = int(np.random.randn() + num_rows_per_append_0)
        if rand_num_apps_0 > 0:
            # append to key0 and associated vals
            n_key0 += rand_num_apps_0
            key0_ds.resize((n_key0,))
            key0_ds[-rand_num_apps_0:] = np.random.randint(0, int(1e6), size=rand_num_apps_0)

            num_val_apps = n_key1 * rand_num_apps_0
            n_val += num_val_apps
            val_ds.resize((n_val,))
            val_ds[-num_val_apps:] = np.random.rand(num_val_apps)
        return tt
