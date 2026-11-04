API Documentation
=================

VersionedHDF5File
-----------------

.. autoclass:: versioned_hdf5.VersionedHDF5File
   :members:
   :undoc-members:


Version Replaying
-----------------

The functions in this module allow replaying versions in a file in-place, in
order to globally modify metadata across all versions that otherwise cannot be
changed across versions, such as the dtype of a dataset. This also allows
editing data in old versions, and deleting datasets or versions.

.. autofunction:: versioned_hdf5.replay.delete_version
.. autofunction:: versioned_hdf5.replay.delete_versions
.. autofunction:: versioned_hdf5.replay.modify_metadata
.. autofunction:: versioned_hdf5.replay.recreate_dataset
.. autofunction:: versioned_hdf5.replay.swap
.. autofunction:: versioned_hdf5.replay.tmp_group


Hashing
-------

``versioned_hdf5.hash`` computes the SHA256 digests of the chunks; it resolves OpenSSL's
EVP entry points at runtime, as described in :doc:`installation`. These constants report
what it bound to.

``versioned_hdf5.hash.LIBCRYPTO_PATH``
    Path of the ``libcrypto`` shared library the entry points were resolved from, made
    absolute by ``os.path.realpath()``. It is the path of the interpreter executable
    when OpenSSL is linked statically into the interpreter.

``versioned_hdf5.hash.LIBCRYPTO_VERSION``
    ``OpenSSL_version(OPENSSL_VERSION)`` of that library, e.g.
    ``"OpenSSL 3.5.1 1 Jul 2025"``. Empty when the library does not export the symbol.
