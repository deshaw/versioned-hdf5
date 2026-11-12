"""The locking model of versioned-hdf5.

h5py serializes *all* access to libhdf5 with its interpreter-wide, reentrant
``phil`` lock. libhdf5 itself is not thread safe, and versioned-hdf5 additionally
keeps shared mutable Python state (the ``VersionedHDF5File`` instance registry and
version caches, the ``InMemoryGroup`` wrapper caches, lazily created
``StagedChangesArray`` objects, and the on-disk hash tables).

versioned-hdf5 reuses ``phil`` for everything that interleaves HDF5 calls with
shared state. Each compound public API call, wrapper method, backend operation,
version setup/commit step and cache mutation acquires ``phil`` for the duration of
that individual call. ``stage_version()`` deliberately does not hold ``phil``
across user-controlled code: setup and commit are separate
locked critical sections, and mutations made through the returned group are
individually locked. Direct h5py I/O in another thread therefore cannot deadlock
behind a versioned-hdf5 context, while version groups and shared Python state stay
consistent.

``phil`` is reentrant (``threading.RLock`` on free-threaded builds, h5py's
``FastRLock`` otherwise), so nested public calls and h5py's own
``with_phil``-decorated methods keep working.

What does *not* take ``phil``
-----------------------------
Every individual libhdf5 call is already serialized by h5py, and by
``slicetools.pyx`` where it calls libhdf5 directly. An extra ``phil`` in this
package is therefore only needed to make a *sequence* of steps atomic. Functions
that need no such thing are left undecorated:

- pure functions, e.g. ``Hashtable.hash``;
- a single read or write of one attribute, e.g. ``InMemoryDataset.shape``;
- a single h5py call, e.g. ``VersionedHDF5File.current_version``;
- pure delegation to a function that locks, e.g. ``DatasetWrapper.__getitem__``;
- constructors of objects that no other thread can see yet.

Lighter locks
-------------
State that is never touched together with HDF5 may use its own lock instead, as
``Hashtable`` does for its in-memory table. Such a lock must be a *leaf*: nothing
that can acquire ``phil`` (any h5py call, any function decorated above) may run
while it is held. ``phil`` then the leaf lock is the only permitted order, so no
lock-order inversion is possible.

This is also why the wrappers cannot use per-object locks. ``InMemoryGroup`` and
the ``InMemoryDataset`` family read HDF5 in the middle of updating their own state
(lazy wrapper creation, ``StagedChangesArray`` loading chunks from ``raw_data``),
and ``commit_version`` walks them while holding ``phil``; a per-object lock would
be taken both before and after ``phil``. The wrapper cache
``InMemoryGroup._caches`` hashes ``GroupID`` keys, which calls into h5py, and is
pruned from weakref callbacks that can fire at any time, so it needs a reentrant
lock that is ordered with ``phil`` - that is, ``phil`` itself.

Pure computation that is already ``nogil`` in the Cython modules is left alone; it
runs under ``phil`` too, but only for the duration of the logical operation that
owns it. This means that free-threaded builds do not make HDF5 work parallel: the
GIL is off, but ``phil`` still serializes it, exactly as h5py requires.
"""

from h5py._hl.base import phil, with_phil

__all__ = ["phil", "with_phil"]
