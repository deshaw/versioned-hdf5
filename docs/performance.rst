Performance
===========

This document lists a set of performance tests that have been considered for
Versioned HDF5 files. This is not an exhaustive set of tests, and more analysis
can be added in the future.

The main goals of these tests are:

- To evaluate the performance of Versioned HDF5 files versus non-versioned
  files;
- To evaluate the performance when reading/writing data to a Versioned HDF5
  file in a set of different use cases;
- Evaluate the performance when different parameters options are considered for
  chunking and compression on Versioned HDF5 files.

When different versions of a dataset are stored in a Versioned HDF5 file,
modified copies of the data are stored as new versions. In order to analyze
the performance of the versioned files, we'll create different datasets. These
datasets are all one-dimensional ndarrays, and with each new version a certain
number of rows is added, removed, and modified. For these tests, all changes
are made to elements chosen according to a `power law
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlaw.html>`__
which biases the modifications towards the end of the array.

The tests are as follows:

1. A large fraction of changes is made to the dataset with each new version; 
2. A small fraction of changes is made to the dataset with each new version;
3. A large fraction of changes is made to the dataset with each version, but
   the size of the final array remains constant (no new rows are added and no
   rows are deleted);
4. The number of modifications is dominated by the number of appended rows.

Read and write speeds
---------------------

First, we'll compare the times required to create each file. For each of the
datasets detailed above, we'll consider different chunk sizes and different
compression algorithms as well.

.. code::

   >>> plot_creation


we'll compare the times required to read data from all versions in a
file, sequentially.

.. code::

   >>> plot_sequentially

Next, we'll compute the times required to read a specific version from the
Versioned HDF5 file. In this particular case, we choose to read the version
which is approximately at half of the number of versions in this file.

.. note::

   Although possible, it is not recommended to read versions using integer
   indexing as the performance of reading versions from their name it far
   superior.

.. code::

   >>> plot_middle

In the next plot, we compare the times necessary to read the first version and
the latest version on each file.

.. code::

   >>> plot_latest_v_first

Reading the latest version means reading the last version to be stored in the
file. For comparison, we will also measure reading times for files generated
without versioning (i.e. with no use of Versioned HDF5).

File sizes
----------

As the number of versions in a file grows, its size on disk is also expected to
grow. However, it is reasonable to expect that the overhead of storing metadata
for versioned files doesn't cause the file sizes to explode as the number of
versions increases. We'll see below that this is indeed the case.

.. code::

   >>> plot_file_sizes
