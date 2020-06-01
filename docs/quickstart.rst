Quickstart Guide
================

Let's say you have an HDF5 file with contents that might change over time. You
may add or remove datasets, change the contents of the data or the metadata, and
would like to keep a record of which changes occurred when, and a way to recover
previous versions of this file. Versioned HDF5 allows you to do that by building
a versioning API on top of h5py.

First, you must open an ``.h5`` file and create a `h5py File Object
<http://docs.h5py.org/en/stable/high/file.html>`__ in write mode::

  >>> import h5py
  >>> fileobject = h5py.File('filename.h5', 'w')

Now, you can use the :any:`VersionedHDF5File` constructor on this file object to
create a versioned HDF5 file object::

  >>> from versioned_hdf5 import VersionedHDF5File
  >>> versioned_file = VersionedHDF5File(fileobject)

You can see that this ``versioned_file`` object has the following attributes:

  - ``f``: the original ``h5py`` File Object;
  - ``current_version``: at this point, it should return ``__first_version__``,
    as we haven't created any additional versions.

To create a new version, use the :any:`stage_version` function. For example, if
we do

.. code::

  >>> with versioned_file.stage_version('version2') as group:
  ...     group['mydataset'] = np.ones(10000)

The context manager returns a h5py *group* object, which should be modified
in-place to build the new version. When the context manager exits, the version
will be written to the file. This has two effects. First, the h5py file object
``fileobject`` now has metadata associated with versions::

  >>> fileobject.keys()
  <KeysViewHDF5 ['_version_data']>

All the data from the versioned HDF5 file is stored in the ``_version_data``
group on the file, but this should not be accessed directly: any interaction
with the versioning should happen through the API. ``versioned_file`` can now be
used to expose versioned data by version name::

  >>> v2 = versioned_file['version2']
  >>> v2
  <InMemoryGroup "/_version_data/versions/version2" (1 members)>
  >>> v2['mydataset']
  <HDF5 dataset "mydataset": shape (10000,), type "<f8">

To access the actual data stored in version ``version2``, we use the same syntax
as ``h5py``::

  >>> dataset = v2['mydataset']
  >>> dataset[()]
  array([1., 1., 1., ..., 1., 1., 1.])

.. note::

   Versioned HDF5 files have a special structure and should not be modified
   directly. Also note that once a version is created in the file, it should be
   treated as read-only. Some protections are in place to prevent accidental
   modification, but it is not possible in the HDF5 layer to make a dataset or
   group read-only, so modifications made outside of this library could result
   in breaking things.

When you are done manipulating data, use ``fileobject.close()`` to make sure the
HDF5 file is written properly to disk. Note that the ``VersionedHDF5File``
object does not need to be closed.