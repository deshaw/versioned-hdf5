Installation
============

With conda (easy)
-----------------

You can install Versioned HDF5 binaries via conda on Linux, MacOSX and Windows with

.. code-block:: bash

   $ conda install -c conda-forge versioned-hdf5


With pip (harder)
-----------------

You *cannot* naively install Versioned HDF5 binaries with pip, as there are no pre-built
wheels on PyPi due to complexity of sharing the libhdf5 C library with h5py.

In order to install from sources, you will need libhdf5 headers and shared libraries to
be installed, either locally with conda (``conda install -c conda-forge hdf5``) or
system wide.
When compiling versioned-hdf5 from sources, you *must* compile h5py too to guarantee
that both packages are compiled and linked against the same libhdf5 C library:

.. code-block:: bash

   $ pip install h5py versioned-hdf5 --no-binary h5py,versioned-hdf5

.. warning::

   Compiling versioned-hdf5 from sources while h5py is installed from pre-built
   pypi wheels will cause runtime failures.


Dependencies
------------

Currently, Versioned HDF5 has the following runtime dependencies:

- ``numpy``
- ``h5py``
- ``ndindex``

Refer to ``pyproject.toml`` for minimum supported versions.
