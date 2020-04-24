from __future__ import (absolute_import, division, print_function, with_statement)

import logging

import h5py
import numpy as np

from utils import temp_dir_ctx
from analysis.generate_data_base import TestDatasetPerformanceBase


class TestDatasetPerformance(TestDatasetPerformanceBase):
    @classmethod
    def _write_transactions_sparse(cls, name, num_rows_initial, num_transactions, num_rows_per_append,
                                   pct_changes, num_changes,
                                   pct_deletes, num_deletes,
                                   pct_inserts, num_inserts):
        logger = logging.getLogger(__name__)

        with temp_dir_ctx() as tmp_dir:
            filename = tmp_dir + f'/{name}_sparse_no_versions.h5'
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
                    print("Transaction", a, "of", num_transactions)
                    tt = cls._modify_dss_sparse(key0_ds, key1_ds, val_ds, num_rows_per_append,
                                                pct_changes if a > 0 else 0.0, num_changes,
                                                pct_deletes if a > 0 else 0.0, num_deletes,
                                                pct_inserts if a > 0 else 0.0, num_inserts)
                    tts.append(tt)
                    logger.info('Wrote transaction %d at transaction time %s', a, tt)
            finally:
                f.close()


    @classmethod
    def _write_transactions_dense_old(cls, name, num_rows_initial_0, num_rows_initial_1,
                                      num_transactions,
                                      num_rows_per_append_0,
                                      pct_changes, num_changes,
                                      pct_deletes, num_deletes_0, num_deletes_1,
                                      pct_inserts, num_inserts_0, num_inserts_1):
        logger = logging.getLogger(__name__)

        with temp_dir_ctx() as tmp_dir:
            filename = tmp_dir + f'/{name}_dense_no_versions.h5'
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
                    print("Transaction", a, "of", num_transactions)
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
    def _write_transactions_dense(cls, name, num_rows_initial_0, num_rows_initial_1,
                                  num_transactions,
                                  num_rows_per_append_0,
                                  pct_changes, num_changes,
                                  pct_deletes, num_deletes_0, num_deletes_1,
                                  pct_inserts, num_inserts_0, num_inserts_1):
        logger = logging.getLogger(__name__)

        with temp_dir_ctx() as tmp_dir:
            filename = tmp_dir + f'/{name}_dense.h5'
            tts = []
            f = h5py.File(filename, 'w')
            try:
                key0_ds = f.create_dataset(name + '/key0', data=np.random.rand(num_rows_initial_0),
                                           dtype=np.dtype('int64'), maxshape=(None,), chunks=(int(1e4),))
                key1_ds = f.create_dataset(name + '/key1', data=np.random.rand(num_rows_initial_1),
                                           dtype=np.dtype('int64'), maxshape=(None,), chunks=(int(1e4),))
                # two dimensional value array
                val_ds = f.create_dataset(name + '/val', data=np.random.rand(num_rows_initial_0, num_rows_initial_1),
                                          dtype=np.dtype('float64'), maxshape=(None, None), chunks=(int(1e4), int(1e4)))
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


if __name__ == '__main__':
    TestDatasetPerformance().test_mostly_appends_dense()
