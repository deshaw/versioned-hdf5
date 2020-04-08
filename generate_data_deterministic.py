from __future__ import (absolute_import, division, print_function, with_statement)


import datetime
import logging
import random
from unittest import TestCase

import h5py
import numpy as np

from versioned_hdf5.api import VersionedHDF5File

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
    def test_mostly_appends_sparse(self,
                                   num_transactions=250,
                                   filename="test_mostly_appends_sparse",
                                   chunk_size=None,
                                   compression=None,
                                   print_transactions=False):

        num_rows_initial = 1000

        num_rows_per_append = 1000

        pct_inserts = 5
        num_inserts = 10

        pct_deletes = 1
        num_deletes = 10

        pct_changes = 5
        num_changes = 10

        #name = f"test_mostly_appends_sparse_{num_transactions}"

        self._write_transactions_sparse(filename,
                                        print_transactions,
                                        chunk_size,
                                        compression,
                                        num_rows_initial,
                                        num_transactions,
                                        num_rows_per_append,
                                        pct_changes, num_changes,
                                        pct_deletes, num_deletes,
                                        pct_inserts, num_inserts)
        

    @classmethod
    def _write_transactions_sparse(cls, name, print_transactions,
                                   chunk_size,
                                   compression,
                                   num_rows_initial,
                                   num_transactions,
                                   num_rows_per_append,
                                   pct_changes, num_changes,
                                   pct_deletes, num_deletes,
                                   pct_inserts, num_inserts):
        logger = logging.getLogger(__name__)

        tmp_dir = '.'
        filename = tmp_dir + f'/{name}.h5'
        f = h5py.File(filename, 'w')
        file = VersionedHDF5File(f)
        try:
            with file.stage_version("initial_version") as group:
                key0_ds = group.create_dataset(name + '/key0',
                                               data=np.random.rand(num_rows_initial),
                                               dtype=(np.dtype('int64')),
                                               chunks=chunk_size,
                                               compression=compression)
                key1_ds = group.create_dataset(name + '/key1',
                                               data=np.random.rand(num_rows_initial),
                                               dtype=(np.dtype('int64')),
                                               chunks=chunk_size,
                                               compression=compression)
                val_ds = group.create_dataset(name + '/val',
                                              data=np.random.rand(num_rows_initial),
                                              dtype=(np.dtype('float64')),
                                              chunks=chunk_size,
                                              compression=compression)
            for a in range(num_transactions):
                if print_transactions:
                    print("Transaction", a)
                tt = datetime.datetime.utcnow()
                with file.stage_version(str(tt)) as group:
                    key0_ds = group[name + '/key0']
                    key1_ds = group[name + '/key1']
                    val_ds = group[name + '/val']
                    cls._modify_dss_sparse_deterministic(key0_ds,
                                                         key1_ds,
                                                         val_ds,
                                                         num_rows_per_append,
                                                         pct_changes if a > 0 else 0.0,
                                                         num_changes,
                                                         pct_deletes if a > 0 else 0.0,
                                                         num_deletes,
                                                         pct_inserts if a > 0 else 0.0,
                                                         num_inserts)

                #tts.append(tt)
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
    def _modify_dss_sparse_deterministic(cls,
                                         key0_ds, key1_ds, val_ds,
                                         num_rows_per_append,
                                         pct_changes, num_changes,
                                         pct_deletes, num_deletes,
                                         pct_inserts, num_inserts):

        # Making sure key0_ds, key1_ds and val_ds all have the same size
        ns = set([len(ds) for ds in [key0_ds, key1_ds, val_ds]])
        assert len(ns) == 1

        n = next(iter(ns))
        
        # change values
        rand_fn = cls._get_rand_fn(val_ds.dtype)
        for b in range(num_changes):
            r = random.randrange(0, n)
            val_ds[r] = rand_fn()

        # delete rows
        rs = [random.randrange(0, n) for _ in range(num_deletes)]
        while len(set(rs)) < num_deletes:
            rs.append(random.randrange(0,n))
        n -= num_deletes
        for ds in [key0_ds, key1_ds, val_ds]:
            arr = ds[:]
            arr = np.delete(arr, rs)
            ds.resize((n,))
            ds[:] = arr

        # insert rows
        rs = [random.randrange(0, n) for _ in range(num_inserts)]
        n += num_inserts
        for ds in [key0_ds, key1_ds, val_ds]:
            rand_fn = cls._get_rand_fn(ds.dtype)
            arr = ds[:]
            arr = np.insert(arr, rs, [rand_fn() for _ in rs])
            ds.resize((n,))
            ds[:] = arr

        # append
        n += num_rows_per_append
        for ds in [key0_ds, key1_ds, val_ds]:
            rand_fn = cls._get_rand_fn(ds.dtype)
            ds.resize((n,))
            ds[-num_rows_per_append:] = rand_fn(num_rows_per_append)


    def test_large_fraction_changes_sparse(self,
                                           num_transactions=250,
                                           filename="test_large_fraction_changes_sparse",
                                           chunk_size=None,
                                           compression=None,
                                           print_transactions=False):

        num_rows_initial = 5000

        num_rows_per_append = 10

        pct_inserts = 1
        num_inserts = 10

        pct_deletes = 1
        num_deletes = 10

        pct_changes = 90

        num_changes = 1000
        
        #name = f"test_large_fraction_changes_sparse_{num_transactions}"

        self._write_transactions_sparse(filename, print_transactions,
                                        chunk_size,
                                        compression,
                                        num_rows_initial,
                                        num_transactions,
                                        num_rows_per_append,
                                        pct_changes, num_changes,
                                        pct_deletes, num_deletes,
                                        pct_inserts, num_inserts)


    def test_small_fraction_changes_sparse(self,
                                           num_transactions=250,
                                           filename="test_small_fraction_changes_sparse",
                                           chunk_size=None,
                                           print_transactions=False):

        num_rows_initial = 5000

        num_rows_per_append = 10

        pct_inserts = 1
        num_inserts = 10

        pct_deletes = 1
        num_deletes = 10

        pct_changes = 90
        num_changes = 10

        #name = f"test_small_fraction_changes_sparse_{num_transactions}"

        self._write_transactions_sparse(filename, print_transactions,
                                        chunk_size,
                                        compression,
                                        num_rows_initial,
                                        num_transactions,
                                        num_rows_per_append,
                                        pct_changes, num_changes,
                                        pct_deletes, num_deletes,
                                        pct_inserts, num_inserts)


    def test_mostly_appends_dense(self,
                                  num_transactions=250,
                                  filename="test_mostly_appends_dense",
                                  chunk_size=None,
                                  print_transactions=False):

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

        #name = f"test_mostly_appends_dense_{num_transactions}"

        self._write_transactions_dense(filename,
                                       chunk_size,
                                       num_rows_initial_0,
                                       num_rows_initial_1,
                                       num_transactions,
                                       num_rows_per_append_0,
                                       pct_changes, num_changes,
                                       pct_deletes, num_deletes_0, num_deletes_1,
                                       pct_inserts, num_inserts_0, num_inserts_1)
        

    @classmethod
    def _write_transactions_dense(cls, name, chunk_size,
                                  num_rows_initial_0, num_rows_initial_1,
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
        file = VersionedHDF5File(f)
        try:
            with file.stage_version("initial_version") as group:
                key0_ds = group.create_dataset(name + '/key0', data=np.random.rand(num_rows_initial_0),
                                               dtype=(np.dtype('int64')), chunks=chunk_size)
                key1_ds = group.create_dataset(name + '/key1', data=np.random.rand(num_rows_initial_1),
                                               dtype=(np.dtype('int64')), chunks=chunk_size)
                val_ds = group.create_dataset(name + '/val', data=np.random.rand(num_rows_initial_0 * num_rows_initial_1),
                                              dtype=(np.dtype('float64')), chunks=chunk_size)
            for a in range(num_transactions):
                #print(f"Transaction {a} of {num_transactions}")
                tt = datetime.datetime.utcnow()
                with file.stage_version(str(tt)) as group:
                    key0_ds = group[name + '/key0']
                    key1_ds = group[name + '/key1']
                    val_ds = group[name + '/val']
                    cls._modify_dss_dense_deterministic(key0_ds, key1_ds, val_ds,
                                          num_rows_per_append_0,
                                          pct_changes if a > 0 else 0.0, num_changes,
                                          pct_deletes if a > 0 else 0.0, num_deletes_0, num_deletes_1,
                                          pct_inserts if a > 0 else 0.0, num_inserts_0, num_inserts_1)

                #tts.append(tt)
                logger.info('Wrote transaction %d at transaction time %s', a, tt)
        finally:
            f.close()


    @classmethod
    def _modify_dss_dense_deterministic(cls,
                                        key0_ds, key1_ds, val_ds,
                                        num_rows_per_append_0,
                                        pct_changes, num_changes,
                                        pct_deletes, num_deletes_0, num_deletes_1,
                                        pct_inserts, num_inserts_0, num_inserts_1):
        n_key0 = len(key0_ds)
        n_key1 = len(key1_ds)
        n_val = len(val_ds)
        assert n_val == n_key0 * n_key1
        
        # change values
        for b in range(num_changes):
            r = random.randrange(0, n_val)
            val_ds[r] = np.random.rand()

        # delete rows ============================
        # delete from values in two steps
        arr_val = val_ds[:]

        # 1. delete from key0 and associated vals
        rs_0 = [random.randrange(0, n_key0) for _ in range(num_deletes_0)]

        rs_val = [r0 * n_key1 + r1 for r0 in rs_0 for r1 in range(n_key1)]
        n_val -= len(rs_val)
        arr_val = np.delete(arr_val, rs_val)

        n_key0 -= num_deletes_0
        arr_key0 = key0_ds[:]
        arr_key0 = np.delete(arr_key0, rs_0)
        key0_ds.resize((n_key0,))
        key0_ds[:] = arr_key0

        # 2. delete from key1 and associated vals
        rs_1 = [random.randrange(0, n_key1) for _ in range(num_deletes_1)]

        rs_val = [r0 * n_key1 + r1 for r0 in range(n_key0) for r1 in rs_1]
        n_val -= len(rs_val)
        arr_val = np.delete(arr_val, rs_val)
        val_ds.resize((n_val,))
        val_ds[:] = arr_val

        n_key1 -= num_deletes_1
        arr_key1 = key1_ds[:]
        arr_key1 = np.delete(arr_key1, rs_1)
        key1_ds.resize((n_key1,))
        key1_ds[:] = arr_key1

        # insert rows =====================
        # insert into values in two steps
        arr_val = val_ds[:]

        # 1. insert into key0 and associated vals
        rs_0 = [random.randrange(0, n_key0) for _ in range(num_inserts_0)]

        rs_val = [r0 * n_key1 + r1 for r0 in rs_0 for r1 in range(n_key1)]
        n_val += len(rs_val)
        arr_val = np.insert(arr_val, rs_val, [np.random.rand() for _ in rs_val])

        arr_key0 = key0_ds[:]
        arr_key0 = np.insert(arr_key0, rs_0, np.random.randint(0, int(1e6), size=len(rs_0)))
        n_key0 += num_inserts_0
        key0_ds.resize((n_key0,))
        key0_ds[:] = arr_key0

        # 2. insert into key1 and associated vals
        rs_1 = [random.randrange(0, n_key1) for _ in range(num_inserts_1)]

        rs_val = [r0 * n_key1 + r1 for r0 in range(n_key0) for r1 in rs_1]
        n_val += len(rs_val)
        arr_val = np.insert(arr_val, rs_val, np.random.rand(len(rs_val)))
        val_ds.resize((n_val,))
        val_ds[:] = arr_val

        arr_key1 = key1_ds[:]
        arr_key1 = np.insert(arr_key1, rs_1, np.random.randint(0, int(1e6), size=len(rs_1)))
        n_key1 += num_inserts_1
        key1_ds.resize((n_key1,))
        key1_ds[:] = arr_key1

        # append ======================
        # append to key0 and associated vals
        n_key0 += num_rows_per_append_0
        key0_ds.resize((n_key0,))
        key0_ds[-num_rows_per_append_0:] = np.random.randint(0, int(1e6), size=num_rows_per_append_0)
        
        num_val_apps = n_key1 * num_rows_per_append_0
        n_val += num_val_apps
        val_ds.resize((n_val,))
        val_ds[-num_val_apps:] = np.random.rand(num_val_apps)

if __name__ == '__main__':

    #num_transactions = [50, 100, 500, 1000, 2000]#, 5000, 10000]
    num_transactions = [50]
    for t in num_transactions:
        TestVersionedDatasetPerformance().test_large_fraction_changes_sparse(t)
