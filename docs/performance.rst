.. _performance:

Performance
===========

This document lists a set of performance tests that have been considered for
Versioned HDF5 files. This is not an exhaustive set of tests, and more analysis
can be added in the future.

The main goals of these tests are:

- To evaluate the performance of reading/writing Versioned HDF5 files and compare it with non-versioned files (that is, files where only the last version of the datasets is stored);
- To evaluate the performance when reading/writing data to a Versioned HDF5 file in a set of different use cases;
- To evaluate the performance when different parameters options are considered for chunking and compression on Versioned HDF5 files.

When different versions of a dataset are stored in a Versioned HDF5 file, modified copies of the data are stored as new versions. This means that there may be duplication of data between versions, which might impact the performance of reading, writing or manipulating these files.

In order to analyze this, we will consider test files created with a variable number of versions (or transactions) of a dataset consisting of three ndarrays of variable length. One test includes a two-dimensional ndarray as part of this dataset, but all other test cases consist of three one-dimensional ndarrays per dataset.

With each new version a certain number of rows is added, removed, and modified. For
these tests, all changes are made to elements chosen according to a `power law
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlaw.html>`__
which biases the modifications towards the end of the array, simulating a possible use case of modifying more recent results in a given timeseries.

The tests are as follows:

1. A large fraction of changes is made to the dataset with each new version: The dataset initially has three arrays with 5000 rows, and 1000 positions are chosen at random and changed, and a small number (at most 10) rows are added or deleted with each new version. We will refer to this test as ``test_large_fraction_changes_sparse``.

2. A small fraction of changes is made to the dataset with each new version: The dataset initially has three arrays with 5000, but only 10 positions are chosen at random and changed, and a small number (at most 10) rows are added or deleted with each new version. We will refer to this test as ``test_small_fraction_changes_sparse``.

3. A large fraction of changes is made to the dataset with each version, with the same three arrays of 5000 rows defined initially, 1000 positions are chosen at random and changed, but the size of the final array remains constant (no new rows are added and no rows are deleted). We will refer to this test as ``test_large_fraction_constant_sparse``.

4. The number of modifications is dominated by the number of appended rows. This is divided into two tests:

   - In the first case, the dataset contains three one-dimensional arrays with 1000 rows initially, and 1000 rows are added with each new version. A small number (at most 10) values are chosen at random, following the power law described above, and changed or deleted. We call this test ``test_mostly_appends_sparse``.
   - In the second case, the dataset contains one two-dimensional array with shape ``(30, 30)`` and two one-dimensional arrays acting as indices to the 2d array. In this case, rows are only appended in the first axis of the two-dimensional array, and a small number of positions (at most 10) is chosen at random and changed. We call this test ``test_mostly_appends_dense``.

To test the performance of VersionedHDF5 files, we have chosen to compare a few different chunk sizes and compression algorithms. These values have been chosen heuristically, and optimal values depend on different use cases and nature of the datasets stored in the file.

- **File sizes**: As the number of versions in a file grows, its size on disk is also expected to grow. However, it is reasonable to expect that the overhead of storing metadata for versioned files doesn't cause the file sizes to explode as the number of versions increases. To see the tests and analysis, go to :ref:`performance_filesizes`.
- **Read and write speeds**: To see a brief outline of how VersionedHDF5Files behave in terms of creation time, read and write speeds and a comparison with unversioned files (plain HDF5 files generated using h5py), see :ref:`performance_io`.

.. toctree::
   :hidden:

   performance_filesizes.rst
   performance_io.rst
