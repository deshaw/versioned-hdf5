Installation
============

You can install Versioned HDF5 binaries via conda on Linux, MacOSX and Windows with

.. code-block:: bash

   $ conda install -c conda-forge versioned-hdf5

or via pip on Linux and MacOSX with

.. code-block:: bash

   $ pip install versioned-hdf5

In order to install from sources, you will need libhdf5 headers and shared libraries
to be installed, either locally with conda (``conda install -c conda-forge hdf5``) or
system wide.
When compiling versioned-hdf5 from sources, it is advised to compile h5py too:

.. code-block:: bash

   $ pip install h5py versioned-hdf5 --no-binary h5py,versioned-hdf5


Dependencies
------------

Currently, Versioned HDF5 has the following runtime dependencies:

- ``numpy``
- ``h5py``
- ``ndindex``

Refer to ``pyproject.toml`` for minimum supported versions.
