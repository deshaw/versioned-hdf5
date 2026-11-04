Installation
============

You can install Versioned HDF5 binaries via conda on Linux, macOS and Windows with

.. code-block:: bash

   $ conda install -c conda-forge versioned-hdf5

or via pip with

.. code-block:: bash

   $ pip install versioned-hdf5

.. note::

   Wheels on PyPI depend on an exact h5py version. This is due to limitations of the
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
   PyPI wheels will cause runtime failures.


OpenSSL
-------

Versioned HDF5 neither links against nor bundles OpenSSL. ``versioned_hdf5.hash``
computes the SHA256 digests with OpenSSL's EVP API, and resolves the entry points it
needs at import time from the ``libcrypto`` library that the process has already loaded,
preferring the one CPython's ``_hashlib`` loaded so that there is only one copy of
OpenSSL in the process. A runtime OpenSSL 1.1.x or 3.x must therefore be available. The
fallback is a system ``libcrypto`` that can be opened by its well-known name
(``libcrypto.so.3``, ``libcrypto.so.1.1``, ``libcrypto.3.dylib``,
``libcrypto-3-x64.dll``, ...). If neither is found, ``import versioned_hdf5`` raises an
``ImportError`` describing what was tried.

Building from source only requires the libhdf5 headers and shared libraries described
above, not the OpenSSL development headers.

``versioned_hdf5.hash.LIBCRYPTO_PATH`` and ``versioned_hdf5.hash.LIBCRYPTO_VERSION``
report which library was bound to; see :doc:`reference`.


Dependencies
------------

Currently, Versioned HDF5 has the following runtime dependencies:

- ``numpy``
- ``h5py``
- ``ndindex``

Refer to ``pyproject.toml`` for minimum supported versions.
