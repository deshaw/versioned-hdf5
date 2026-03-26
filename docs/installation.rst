Installation
============

You can install Versioned HDF5 binaries via conda on Linux, MacOSX and Windows with

.. code-block:: bash

   $ conda install -c conda-forge versioned-hdf5

or via pip on Linux and MacOSX with

.. code-block:: bash

   $ pip install versioned-hdf5

.. note::

   Wheels on PyPi depend on an exact h5py version. This is due to limitations of the
   wheel format. If you need a different h5py version, please either install via conda
   or build from sources.

In order to install from sources, you will need libhdf5 headers and shared libraries
to be installed, either locally with conda (``conda install -c conda-forge hdf5``) or
system wide.
When compiling versioned-hdf5 from sources, it is advised to compile h5py too:

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
