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
  <Committed InMemoryGroup "/_version_data/versions/version2">
  >>> v2['mydataset']
  <InMemoryArrayDataset "mydataset": shape (10000,), type "<f8">

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

When you are done manipulating data, both the ``h5py`` and ``VersionedHDF5File``
objects must be closed to make sure the HDF5 file is written properly to disk
(including data about versions.) This can be achieved by

.. code::

  >>> fileobject.close()
  >>> versioned_file.close()


Other Options
-------------

When a version is committed to a VersionedHDF5File, a timestamp is automatically
added to it. The timestamp for each version can be retrieved via the version's
``attrs``::

  >>> versioned_file['version1'].attrs['timestamp']

Since the HDF5 specification does not currently support writing
``datetime.datetime`` or ``numpy.datetime`` objects to HDF5 files, these timestamps
are stored as strings, using the following format::

  ``"%Y-%m-%d %H:%M:%S.%f%z"``

The timestamps are registered in UTC. For more details on the format string
above, see the ``datetime.datetime.strftime`` function documentation.

The timestamp can also be used as an index to retrieve a chosen version from the
file. In this case, either a ``datetime.datetime`` or a ``numpy.datetime64`` object
must be used as a key. For example, if

.. code::

  >>> t = datetime.datetime.now(datetime.timezone.utc)

then using

.. code::

  >>> versioned_file[t]

returns the version with timestamp equal to ``t`` (converted to a string according
to the format mentioned above).

It is also possible to assign a timestamp manually to a file. Again, this
requires using either a ``datetime.datetime`` or a ``numpy.datetime64`` object as
the timestamp::

  >>> ts = datetime.datetime(2020, 6, 29, 23, 58, 21, 116470, tzinfo=datetime.timezone.utc)
  >>> with versioned_file.stage_version('version1', timestamp=ts) as group:
  >>>    group['mydataset'] = data

Now::

  >>> versioned_file[ts]

returns the same as ``versioned_file['version1']``.
