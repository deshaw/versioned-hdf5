from __future__ import (absolute_import, division, print_function, with_statement)

import datetime
import logging
import random
import time
from unittest import TestCase

import h5py
import numpy as np
import scipy.stats

from versioned_hdf5.api import VersionedHDF5File


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

    # models
    RECENCTNESS_POWERLAW_SHAPE = 20.0

    @classmethod
    def _write_transactions_sparse(cls, name,
                                   chunk_size,
                                   compression,
                                   versions,
                                   print_transactions,
                                   num_rows_initial,
                                   num_transactions,
                                   num_rows_per_append,
                                   num_changes,
                                   num_deletes,
                                   num_inserts):
        logger = logging.getLogger(__name__)

        tmp_dir = '.'
        filename = tmp_dir + f'/{name}.h5'
        f = h5py.File(filename, 'w')
        told = time.time()
        t0 = told
        times = []
        try:
            if versions:
                file = VersionedHDF5File(f)
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
            else:
                key0_ds = f.create_dataset(name + '/key0',
                                           data=np.random.rand(num_rows_initial),
                                           dtype=(np.dtype('int64')),
                                           maxshape=(None,),
                                           chunks=(chunk_size,),
                                           compression=compression)
                key1_ds = f.create_dataset(name + '/key1',
                                           data=np.random.rand(num_rows_initial),
                                           dtype=(np.dtype('int64')),
                                           maxshape=(None,),
                                           chunks=(chunk_size,),
                                           compression=compression)
                val_ds = f.create_dataset(name + '/val',
                                          data=np.random.rand(num_rows_initial),
                                          dtype=(np.dtype('float64')),
                                          maxshape=(None,),
                                          chunks=(chunk_size,),
                                          compression=compression)

            for a in range(num_transactions):
                if print_transactions:
                    print("Transaction", a)
                tt = datetime.datetime.utcnow()
                if versions:
                    with file.stage_version(str(tt)) as group:
                        key0_ds = group[name + '/key0']
                        key1_ds = group[name + '/key1']
                        val_ds = group[name + '/val']
                        cls._modify_dss_sparse_deterministic(key0_ds,
                                                             key1_ds,
                                                             val_ds,
                                                             num_rows_per_append,
                                                             num_changes,
                                                             num_deletes,
                                                             num_inserts)
                else:
                    cls._modify_dss_sparse_deterministic(key0_ds, key1_ds, val_ds,
                                                         num_rows_per_append,
                                                         num_changes,
                                                         num_deletes,
                                                         num_inserts)

                t = time.time()
                times.append(t-told)
                told = t
                # tts.append(tt)
                logger.info('Wrote transaction %d at transaction time %s', a, tt)
            times.append(t-t0)
        finally:
            f.close()
        return times

    def test_mostly_appends_sparse(self,
                                   num_transactions=250,
                                   filename="test_mostly_appends_sparse",
                                   chunk_size=None,
                                   compression=None,
                                   versions=True,
                                   print_transactions=False):

        num_rows_initial = 1000
        num_rows_per_append = 1000
        num_inserts = 10
        num_deletes = 10
        num_changes = 10

        # name = f"test_mostly_appends_sparse_{num_transactions}"

        times = self._write_transactions_sparse(filename,
                                                chunk_size,
                                                compression,
                                                versions,
                                                print_transactions,
                                                num_rows_initial,
                                                num_transactions,
                                                num_rows_per_append,
                                                num_changes,
                                                num_deletes,
                                                num_inserts)
        return times



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
                                         num_changes,
                                         num_deletes,
                                         num_inserts):

        # Making sure key0_ds, key1_ds and val_ds all have the same size
        ns = set([len(ds) for ds in [key0_ds, key1_ds, val_ds]])
        assert len(ns) == 1

        n = next(iter(ns))

        # change values
        rand_fn = cls._get_rand_fn(val_ds.dtype)
        for b in range(num_changes):
            r = random.randrange(0, n)
            val_ds[r] = rand_fn()

        # insert rows
        if num_inserts > 0:
            pdf = scipy.stats.powerlaw.rvs(TestVersionedDatasetPerformance.RECENCTNESS_POWERLAW_SHAPE, size=num_inserts)
            rs = np.unique((pdf * n).astype('int64'))
            minr = min(rs)
            n += len(rs)
            for ds in [key0_ds, key1_ds, val_ds]:
                rand_fn = cls._get_rand_fn(ds.dtype)
                arr = ds[minr:]
                arr = np.insert(arr, rs - minr, [rand_fn() for _ in rs])
                ds.resize((n,))
                ds[minr:] = arr

        # delete rows
        if num_deletes > 0:
            if num_rows_per_append != 0:
                pdf = scipy.stats.powerlaw.rvs(TestVersionedDatasetPerformance.RECENCTNESS_POWERLAW_SHAPE, size=num_deletes)
                rs = np.unique((pdf * n).astype('int64'))
                minr = min(rs)
            n -= len(rs)
            for ds in [key0_ds, key1_ds, val_ds]:
                arr = ds[minr:]
                arr = np.delete(arr, rs - minr)
                ds.resize((n,))
                ds[minr:] = arr


        # append
        if num_rows_per_append > 0:
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
                                           versions=True,
                                           print_transactions=False):

        num_rows_initial = 5000
        num_rows_per_append = 10
        num_inserts = 10
        num_deletes = 10
        num_changes = 1000

        # name = f"test_large_fraction_changes_sparse_{num_transactions}"

        times = self._write_transactions_sparse(filename,
                                                chunk_size,
                                                compression,
                                                versions,
                                                print_transactions,
                                                num_rows_initial,
                                                num_transactions,
                                                num_rows_per_append,
                                                num_changes,
                                                num_deletes,
                                                num_inserts)
        return times

    def test_small_fraction_changes_sparse(self,
                                           num_transactions=250,
                                           filename="test_small_fraction_changes_sparse",
                                           chunk_size=None,
                                           compression=None,
                                           versions=True,
                                           print_transactions=False):

        num_rows_initial = 5000
        num_rows_per_append = 10
        num_inserts = 10
        num_deletes = 10
        num_changes = 10

        # name = f"test_small_fraction_changes_sparse_{num_transactions}"

        times = self._write_transactions_sparse(filename,
                                                chunk_size,
                                                compression,
                                                versions,
                                                print_transactions,
                                                num_rows_initial,
                                                num_transactions,
                                                num_rows_per_append,
                                                num_changes,
                                                num_deletes,
                                                num_inserts)
        return times

    def test_mostly_appends_dense(self,
                                  num_transactions=250,
                                  filename="test_mostly_appends_dense",
                                  chunk_size=None,
                                  compression=None,
                                  versions=True,
                                  print_transactions=False):

        num_rows_initial_0 = 30
        num_rows_initial_1 = 30
        num_rows_per_append_0 = 1
        num_inserts_0 = 1
        num_inserts_1 = 10
        num_deletes_0 = 1
        num_deletes_1 = 1
        num_changes = 10

        # name = f"test_mostly_appends_dense_{num_transactions}"

        times = self._write_transactions_dense(filename,
                                               chunk_size,
                                               compression,
                                               versions,
                                               print_transactions,
                                               num_rows_initial_0,
                                               num_rows_initial_1,
                                               num_transactions,
                                               num_rows_per_append_0,
                                               num_changes,
                                               num_deletes_0, num_deletes_1,
                                               num_inserts_0, num_inserts_1)
        return times

    @classmethod
    def _write_transactions_dense(cls, name,
                                  chunk_size,
                                  compression,
                                  versions,
                                  print_transactions,
                                  num_rows_initial_0, num_rows_initial_1,
                                  num_transactions,
                                  num_rows_per_append_0,
                                  num_changes,
                                  num_deletes_0, num_deletes_1,
                                  num_inserts_0, num_inserts_1):

        logger = logging.getLogger(__name__)

        tmp_dir = '.'
        filename = tmp_dir + f'/{name}.h5'
        #tts = []
        f = h5py.File(filename, 'w')
        told = time.time()
        t0 = told
        times = []
        try:
            if versions:
                file = VersionedHDF5File(f)
                with file.stage_version("initial_version") as group:
                    key0_ds = group.create_dataset(name + '/key0',
                                                   data=np.random.rand(num_rows_initial_0),
                                                   dtype=(np.dtype('int64')),
                                                   chunks=chunk_size,
                                                   compression=compression)
                    key1_ds = group.create_dataset(name + '/key1',
                                                   data=np.random.rand(num_rows_initial_1),
                                                   dtype=(np.dtype('int64')),
                                                   chunks=chunk_size,
                                                   compression=compression)
                    val_ds = group.create_dataset(name + '/val',
                                                  data=np.random.rand(num_rows_initial_0 * num_rows_initial_1),
                                                  dtype=(np.dtype('float64')),
                                                  chunks=chunk_size,
                                                  compression=compression)
            else:
                key0_ds = f.create_dataset(name + '/key0',
                                           data=np.random.rand(num_rows_initial_0),
                                           dtype=(np.dtype('int64')),
                                           maxshape=(None,),
                                           chunks=(chunk_size,),
                                           compression=compression)
                key1_ds = f.create_dataset(name + '/key1',
                                           data=np.random.rand(num_rows_initial_1),
                                           dtype=(np.dtype('int64')),
                                           maxshape=(None,),
                                           chunks=(chunk_size,),
                                           compression=compression)
                val_ds = f.create_dataset(name + '/val',
                                          data=np.random.rand(num_rows_initial_0*num_rows_initial_1),
                                          dtype=(np.dtype('float64')),
                                          maxshape=(None,),
                                          chunks=(chunk_size,),
                                          compression=compression)

            for a in range(num_transactions):
                if print_transactions:
                    print(f"Transaction {a} of {num_transactions}")
                tt = datetime.datetime.utcnow()
                if versions:
                    with file.stage_version(str(tt)) as group:
                        key0_ds = group[name + '/key0']
                        key1_ds = group[name + '/key1']
                        val_ds = group[name + '/val']
                        cls._modify_dss_dense_deterministic(key0_ds, key1_ds, val_ds,
                                                            num_rows_per_append_0,
                                                            num_changes,
                                                            num_deletes_0, num_deletes_1,
                                                            num_inserts_0, num_inserts_1)
                else:
                    cls._modify_dss_dense_deterministic(key0_ds, key1_ds, val_ds,
                                                        num_rows_per_append_0,
                                                        num_changes,
                                                        num_deletes_0,
                                                        num_deletes_1,
                                                        num_inserts_0,
                                                        num_inserts_1)
                t = time.time()
                times.append(t-told)
                told=t
                #tts.append(tt)
                logger.info('Wrote transaction %d at transaction time %s', a, tt)
            times.append(t-t0)
        finally:
            f.close()
        return times

    @classmethod
    def _modify_dss_dense_deterministic(cls,
                                        key0_ds, key1_ds, val_ds,
                                        num_rows_per_append_0,
                                        num_changes,
                                        num_deletes_0, num_deletes_1,
                                        num_inserts_0, num_inserts_1):
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

        # 1. delete from key0 and associated vals
        pdf = scipy.stats.powerlaw.rvs(TestVersionedDatasetPerformance.RECENCTNESS_POWERLAW_SHAPE, size=num_deletes_0)
        rs_0 = np.unique((pdf * n_key0).astype('int64'))
        minr_0 = min(rs_0)

        arr_key0 = key0_ds[minr_0:]
        arr_key0 = np.delete(arr_key0, rs_0 - minr_0)
        n_key0 -= len(rs_0)
        key0_ds.resize((n_key0,))
        key0_ds[minr_0:] = arr_key0

        rs_val = [r0 * n_key1 + r1 for r0 in rs_0 for r1 in range(n_key1)]
        n_val -= len(rs_val)
        arr_val = val_ds[minr_0:]
        arr_val = np.delete(arr_val, rs_val - minr_0)

        val_ds.resize((n_val,))
        val_ds[minr_0:] = arr_val

        # 2. delete from key1 and associated vals
        pdf = scipy.stats.powerlaw.rvs(TestVersionedDatasetPerformance.RECENCTNESS_POWERLAW_SHAPE, size=num_deletes_1)
        rs_1 = np.unique((pdf * n_key1).astype('int64'))
        minr_1 = min(rs_1)

        arr_key1 = key1_ds[minr_1:]
        arr_key1 = np.delete(arr_key1, rs_1 - minr_1)
        n_key1 -= len(rs_1)
        key1_ds.resize((n_key1,))
        key1_ds[minr_1:] = arr_key1

        rs_val = [r0 * n_key1 + r1 for r0 in range(n_key0) for r1 in rs_1]
        n_val -= len(rs_val)
        arr_val = val_ds[minr_1:]
        arr_val = np.delete(arr_val, rs_val - minr_1)

        val_ds.resize((n_val,))
        val_ds[minr_1:] = arr_val

        # insert rows =====================
        # insert into values in two steps

        # 1. insert into key0 and associated vals
        pdf = scipy.stats.powerlaw.rvs(TestVersionedDatasetPerformance.RECENCTNESS_POWERLAW_SHAPE, size=num_inserts_0)
        rs_0 = np.unique((pdf * n_key0).astype('int64'))
        minr_0 = min(rs_0)

        arr_key0 = key0_ds[minr_0:]
        arr_key0 = np.insert(arr_key0, rs_0 - minr_0, np.random.randint(0, int(1e6), size=len(rs_0)))
        n_key0 += len(rs_0)
        key0_ds.resize((n_key0,))
        key0_ds[minr_0:] = arr_key0

        arr_val = val_ds[minr_0:]
        rs_val = [r0 * n_key0 + r1 for r0 in rs_0-minr_0 for r1 in range(n_key1)]
        n_val += len(rs_val)
        arr_val = np.insert(arr_val, rs_val, np.random.rand(len(rs_val)))
        val_ds.resize((n_val,))
        val_ds[minr_0:] = arr_val

        # 2. insert into key1 and associated vals
        pdf = scipy.stats.powerlaw.rvs(TestVersionedDatasetPerformance.RECENCTNESS_POWERLAW_SHAPE, size=num_inserts_1)
        rs_1 = np.unique((pdf * n_key1).astype('int64'))
        minr_1 = min(rs_1)

        arr_key1 = key1_ds[minr_1:]
        arr_key1 = np.insert(arr_key1, rs_1 - minr_1, np.random.randint(0, int(1e6), size=len(rs_1)))
        n_key1 += len(rs_1)
        key1_ds.resize((n_key1,))
        key1_ds[minr_1:] = arr_key1

        arr_val = val_ds[minr_1:]
        rs_val = [r1 * n_key1 + r0 for r0 in range(n_key0) for r1 in rs_1-minr_1]
        n_val += len(rs_val)
        val_ds.resize((n_val,))
        arr_val = np.insert(arr_val, rs_val, np.random.rand(len(rs_val)))
        val_ds[minr_1:] = arr_val

        # append ======================
        # append to key0 and associated vals
        n_key0 += num_rows_per_append_0
        key0_ds.resize((n_key0,))
        key0_ds[-num_rows_per_append_0:] = np.random.randint(0, int(1e6), size=num_rows_per_append_0)

        num_val_apps = n_key1 * num_rows_per_append_0
        n_val += num_val_apps
        val_ds.resize((n_val,))
        val_ds[-num_val_apps:] = np.random.rand(num_val_apps)

    def test_large_fraction_constant_sparse(self,
                                            num_transactions=250,
                                            filename="test_large_fraction_constant_sparse",
                                            chunk_size=None,
                                            compression=None,
                                            versions=True,
                                            print_transactions=False):

        num_rows_initial = 5000
        num_rows_per_append = 0 # triggers the constant size test (FIXME)
        num_inserts = 10        
        num_deletes = 10        
        num_changes = 1000

        times = self._write_transactions_sparse(filename,
                                                chunk_size,
                                                compression,
                                                versions,
                                                print_transactions,
                                                num_rows_initial,
                                                num_transactions,
                                                num_rows_per_append,
                                                num_changes,
                                                num_deletes,
                                                num_inserts)
        return times


if __name__ == '__main__':

    #num_transactions = [50, 100, 500, 1000, 2000]#, 5000, 10000]
    num_transactions = [500]
    for t in num_transactions:
        times = TestVersionedDatasetPerformance().test_large_fraction_constant_sparse(t)
