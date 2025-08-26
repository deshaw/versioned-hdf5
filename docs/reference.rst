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
