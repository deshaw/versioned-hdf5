"""The single lock used by versioned-hdf5.

h5py serializes *all* access to libhdf5 with its interpreter-wide, reentrant
``phil`` lock. libhdf5 itself is not thread safe, and versioned-hdf5 additionally
keeps shared mutable Python state (the ``VersionedHDF5File`` instance registry and
version caches, the ``InMemoryGroup`` wrapper caches, lazily created
``StagedChangesArray`` objects, and the on-disk hash tables).

versioned-hdf5 reuses ``phil`` as its only lock. Each public API call, wrapper
method, backend operation, version setup/commit step and cache mutation acquires
``phil`` for the duration of that individual call. ``stage_version()`` deliberately
does not hold ``phil`` across user-controlled code: setup and commit are separate
locked critical sections, and mutations made through the returned group are
individually locked. Direct h5py I/O in another thread therefore cannot deadlock
behind a versioned-hdf5 context, while version groups and shared Python state stay
consistent.

``phil`` is reentrant (``threading.RLock`` on free-threaded builds, h5py's
``FastRLock`` otherwise), so nested public calls and h5py's own
``with_phil``-decorated methods keep working. There is exactly one lock in the
system and no lock-order inversion is possible.

Pure computation that is already ``nogil`` in the Cython modules is left alone; it
runs under ``phil`` too, but only for the duration of the logical operation that
owns it. This means that free-threaded builds do not make HDF5 work parallel: the
GIL is off, but ``phil`` still serializes it, exactly as h5py requires.
"""

from h5py._hl.base import phil, with_phil

__all__ = ["phil", "with_phil"]
