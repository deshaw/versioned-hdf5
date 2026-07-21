Staging changes in memory
=========================
This section explains the low level design of ``InMemoryDataset``.

When a ``InMemoryDataset`` is modified in any way, all changed chunks are held in memory
until they are ready to be committed to disk. To do so, ``InMemoryDataset`` wraps around
a ``StagedChangesArray`` object, which is a numpy.ndarray-like object.

The ``StagedChangesArray`` holds its data in *slabs*, which is a list of array-like
objects built as the concatenation of chunks along axis 0, typically with shape
``(n*chunk_size[0], *chunk_size[1:])``, where n is the number of chunks on the slab.
This is the same layout as the ``raw_data`` dataset on disk.

The list of slabs is made of three parts:

- The *full slab* is a special slab that is always present and contains exactly one
  read-only chunk, a broadcasted numpy array full of the fill_value.
  It's always at index 0.
- The *base slabs* are array-likes that are treated as read-only. At the moment of
  writing, when the ``InMemoryDataset`` creates the ``StagedChangesArray``, it passes to
  it only one base slab, that is the ``raw_data`` ``h5py.Dataset`` - which,
  conveniently, is a numpy-like object. This slab is typically found at index 1, but may
  be missing for a dataset that's completely empty.
- The *staged slabs* are writeable numpy arrays that are created automatically by
  the ``StagedChangesArray`` whenever there is need to modify a chunk that lies on
  either the full slab or a base slab.

Two numpy arrays of metadata are used to keep track of the chunks:

- ``slab_indices`` is an array of integers, with the same dimensionality as the virtual
  array represented by the ``StagedChangesArray`` and one point per chunk, which
  contains the index of the slab in the ``slabs`` list that contains the chunk.
- ``slab_offsets`` is an array of identical shape to ``slab_indices`` that contains the
  offset of the chunk within the slab along axis 0.

e.g.::

  chunk_size[0] = 10

  slab_indices      slab_offsets  slabs[1]              slabs[2]
  (0 = fill_value)

  1 1 0             30 10  0       0:10 (unreferenced)   0:10 (unreferenced)
  0 2 0              0 20  0      10:20                 10:20
  0 2 0              0 10  0      20:30 (unreferenced)  20:30
                                  30:40

  virtual array (* = chunk completely covered in fill_value)

  slabs[1][30:40]  slabs[1][10:20]  slabs[0][0:10]*
  slabs[0][ 0:10]* slabs[2][20:30]  slabs[0][0:10]*
  slabs[0][ 0:10]* slabs[2][10:20]  slabs[0][0:10]*

In addition to the slabs, the ``StagedChangesArray`` holds a ``hash_tables`` list which
is kept exactly parallel to ``slabs`` (same length; ``None`` wherever the matching slab
is ``None``). ``hash_tables[i]`` is a NumPy array with dtype ``uint64`` and shape ``(n,
4)`` (256 bits rows), where n is the number of chunks in ``slabs[i]`` along axis 0; each
row holds one SHA256 digest, so ``hash_tables[i][j, :]`` is the hash of the chunk at
``slabs[i][j * chunk_size[0]:(j + 1) * chunk_size[0]]``.

These hashes are used by
:ref:`commit <staged_changes_commit>` to deduplicate chunks.
The hashes of the staged slabs are computed on demand at commit time (so if a staged
chunk is created and then deleted, it is not unnecessarily hashed). The hashes of the
base slab are loaded from disk.

In order to be performant, operations are chained together in a way that minimizes
Python interaction and is heavily optimized with Cython. To this extent, each user
request (``__getitem__``, ``__setitem__``, ``resize()``, or ``load()``) is broken down
into a series of slice pairs, each covering at most one chunk, that are encoded
in one numpy array per pair of slabs (source and destination) involved in the transfer.
Each slice pair consists of

- an n-dimensional *source start* index, e.g. the coordinates of the top-left corner to
  read from the source array-like;
- an n-dimensional *destination start* index, e.g. the coordinates of the top-left
  corner to write to in the destination array;
- an n-dimensional *count*, e.g. the number of points to read from the source array-like
  and write to the destination array (they always match, so there's only one count);
- an n-dimensional *source stride*, a.k.a. step, with 1 meaning contiguous (not to be
  confused with numpy's strides, which is the number of bytes between points along an
  axis);
- an n-dimensional *destination stride*.

Those familiar with the HDF5 C API may have recognized that this is a direct mapping to
the ``start``, ``stride``, and ``count`` parameters of the two `H5Sselect_hyperslab`_
function calls - one for the source and one for the destination spaces - that the
library needs to prepend to a `H5Dread`_ call.

The ``StagedChangesArray`` is completely agnostic of the underlying storage - anything
will work as long as it's got a basic numpy-like API. Once the data is ready to be
transferred between two slabs, the ``StagedChangesArray`` calls the
``read_many_slices()`` function, which identifies if the source slab is a
``h5py.Dataset`` or a numpy array and calls two different implementations to execute the
transfer - in the case of ``h5py.Dataset``, a for loop in C, directly to the underlying
HDF5 C library, of `H5Sselect_hyperslab`_ (source dataset) ➞
`H5Sselect_hyperslab`_ (destination numpy array) ➞ `H5Dread`_.

The source array-like can be either:

- a base slab (the ``raw_data`` ``h5py.Dataset``); all source start indices along
  axis 0 need to be shifted by the value indicated in ``slab_offsets``;
- a staged slab (a numpy array in memory), again shifted by ``slab_offsets``;
- the full slab (a broadcasted numpy array);
- the value parameter of the ``__setitem__`` method.

The destination array can be either:

- a staged slab (a numpy array in memory), shifted by ``slab_offsets``;
- the return value of the ``__getitem__`` method, which is created empty at the
  beginning of the method call and then progressively filled slice by slice;
- a new base slab created by ``commit()``, which is a view to an extension of the
  ``raw_data`` ``h5py.Dataset``.


Plans
-----
To encapsulate the complex decision-making logic of the ``StagedChangesArray`` methods,
the actual methods of the class are designed as fairly dumb wrappers which

1. create a ``*Plan`` class with all the information needed to execute the operation
   (``GetItemPlan`` for ``__getitem__()``, ``SetItemPlan`` for ``__setitem__()``, etc.);
2. consume the plan, implementing its decisions - chiefly by calling the
   ``read_many_slices()`` function for each pair of slabs involved in the transfer;
3. discard the plan.

The ``*Plan`` classes are agnostic to data types, never access the actual data (slabs,
``__getitem__`` return value, or ``__setitem__`` value parameter) and exclusively deal
in shapes, chunks, and indices.

For debugging purposes, these classes can be generated without executing the method that
consumes them by calling the ``StagedChangesArray._*_plan()`` methods; this allows
pretty-printing the list of their instructions e.g. in a Jupyter notebook.

For example, in order to debug what will happen when you call ``dset[2:5, 3:6] = 42``,
where ``dset`` is a staged versioned_hdf5 dataset, you can run::

    >>> dset.dataset.staged_changes._setitem_plan((slice(2, 5), slice(3, 6)))

    SetItemPlan<value_shape=(3, 3), value_view=[:, :], append 2 empty slabs,
    7 slice transfers among 3 slab pairs, drop 0 slabs>
      slabs.append(empty((6, 2)))  # slabs[2]
      slabs.append(empty((2, 2)))  # slabs[3]
      # 3 transfers from slabs[1] to slabs[2]
      slabs[2][0:2, 0:2] = slabs[1][10:12, 0:2]
      slabs[2][2:4, 0:2] = slabs[1][18:20, 0:2]
      slabs[2][4:6, 0:2] = slabs[1][20:22, 0:2]
      # 1 transfers from value to slabs[3]
      slabs[3][0:2, 0:2] = value[0:2, 1:3]
      # 3 transfers from value to slabs[2]
      slabs[2][0:2, 1:2] = value[0:2, 0:1]
      slabs[2][2:3, 1:2] = value[2:3, 0:1]
      slabs[2][4:5, 0:2] = value[2:3, 1:3]
    slab_indices:
    [[1 1 1 1]
     [1 2 3 1]
     [1 2 2 1]
     [1 1 1 1]]
    slab_offsets:
    [[ 0  2  4  6]
     [ 8  0  0 14]
     [16  2  4 22]
     [24 26 28 30]]


General plans algorithm
-----------------------
All plans share a similar workflow:

1. Preprocess the index, passed by the user as a parameter to ``__getitem__`` and
   ``__setitem__``, into a list of ``IndexChunkMapper`` objects (one per axis).

2. Query the ``IndexChunkMapper``'s to convert the index of points provided by the user
   to an index of chunks along each axis, then use the indices of chunks to slice the
   ``slab_indices`` and ``slab_offsets`` arrays to obtain the metadata of only the
   chunks that are impacted by the selection.

3. Further refine the above selection on a chunk-by-chunk basis using a mask, depending
   on the value of the matching point of ``slab_indices``. Different masking functions,
   which depend on the specific use case, select/deselect the full slab, the base slabs,
   or the staged slabs. For example, the ``load()`` method - which ensures that
   everything is loaded into memory - will only select the chunks that lie on the base
   slabs.

4. You now have three aligned flattened lists:

   - n-dimensional chunk indices that were selected both at step 2 and 3;
   - the corresponding point of ``slab_indices``, and
   - the corresponding point of ``slab_offsets``.

5. Sort by ``slab_indices`` and partition along them. This is to break the rest of the
   algorithm into separate calls to ``read_many_slices()``, one per pair of source and
   destination slabs. Note that a transfer operation is always from N slabs to 1 slab
   or to the ``__getitem__`` return value, or from 1 slab or the ``__setitem__`` value
   parameter to N slabs, and that the slab index can mean either source or destination
   depending on context.

6. For each *(chunk index, slab index, slab offset)* triplet from the above lists, query
   the ``IndexChunkMapper``'s again, independently for each axis, to convert the global
   n-dimensional index of points that was originally provided by the user to a local
   index that only impacts the chunk. For each axis, this will return:

   - exactly one 1-dimensional slice pair, in case of basic indices (scalars or slices);
   - one or more 1-dimensional slice pairs, in case of advanced indices (arrays of
     indices or arrays of bools).

7. Put the list of 1-dimensional slices in pseudo-cartesian product to produce a list of
   n-dimensional slices, one for each point impacted by the selection.
   It is pseudo-cartesian because at step 3 we have been cherry-picking points in the
   hyperspace of chunks; if we hadn't done that, only limiting ourselves to the
   selection along each axis at step 2, it would be a true cartesian product.

8. If the destination array is a new slab, update ``slab_indices`` and ``slab_offsets``
   to reflect the new position of the chunks.

9. Feed the list of n-dimensional slices to the ``read_many_slices()`` function, which
   will actually transfer the data.

10. Go back to step 6 and repeat for the next pair of source and destination
    arrays/slabs.


``__getitem__`` algorithm
-------------------------
``GetItemPlan`` is one of the simplest plans once you have encapsulated the general
algorithm described at the previous paragraphs.
It makes no distinction between full, base, or staged slabs and there is no per-chunk
masking. It figures out the shape of the return value, creates it with ``numpy.empty``,
and then transfers from each slab into it.

**There is no cache on read**: calling the same index twice will result in two separate
reads to the base slabs, which typically translates to two calls to
``h5py.Dataset.__getitem__`` and two disk accesses. However, note that the HDF5 C
library features its own caching, configurable via ``rdcc_nbytes`` and ``rdcc_nslots``.

For this reason, this method never modifies the state of the ``StagedChangesArray``.


``__setitem__`` algorithm
-------------------------
``SetItemPlan`` is substantially more complex than ``GetItemPlan`` because it needs to
handle the following use cases:

1. The index *completely* covers a chunk that lies either on the full slab or on a base
   slab. The chunk must be replaced with a brand new one in a new staged slab, which is
   filled with a copy of the contents of the ``__setitem__`` value parameter.
   ``slab_indices`` and ``slab_offsets`` are updated to reflect the new position of the
   chunk on a staged slab. The original full or base slab is never accessed.
2. The index *partially* covers a chunk that lies on the full slab or on a base slab.
   The chunk is first copied over from the full or base slab to a brand new staged slab,
   which is then updated with the contents of the ``__setitem__`` value parameter.
   ``slab_indices`` and ``slab_offsets`` are updated to reflect the new position of the
   chunk on a staged slab.
3. The index covers a chunk that is already lying on a staged slab. The slab is
   updated in place; ``slab_indices`` and ``slab_offsets`` are not modified.

To help handle the first two use cases, the ``IndexChunkMapper``'s have the concept of
*selected chunks*, which are chunks that contain at least one point of the index along
one axis, and *whole chunks*, which are chunks where *all* points of the chunk are
covered by the index along one axis.

Moving to the n-dimensional space,

- a chunk is selected when it's caught by the intersection of the selected chunk indices
  along all axes;
- a chunk is *wholly* selected when it's caught by the intersection of the whole chunk
  indices along all axes;
- a chunk is *partially* selected if it's selected along all axes, but not wholly
  selected along at least one axis.

**Example**

>>> arr.shape
(30, 50)
>> arr.chunk_size
(10, 10)
>>> arr[5:20, 30:] = 42

The above example partially selects chunks (0, 3) and (0, 4) and wholly selects chunks
(1, 3) and (1, 4)::

    01234
  0 ...pp
  1 ...ww
  2 .....

The ``SetItemPlan`` thus runs the general algorithm twice:

1. With a mask that picks the chunks that lie either on full of base slabs, intersected
   with the mask of partially selected chunks. These chunks are moved to the staged
   slabs.
2. Without any mask, as now all chunks either lie on staged slabs or are wholly selected
   by the update; in the latter case ``__setitem__`` creates a new slab with ``numpy.empty``
   and appends it to ``StagedChangesArray.slabs``.
   The updated surfaces are then copied from the ``__setitem__`` value parameter.


``resize()`` algorithm
----------------------

``ResizePlan`` iterates along all axes and resizes the array independently for each axis
that changed shape. This typically causes the ``slab_indices`` and ``slab_offsets``
arrays to change shape too.

Special attention needs to be paid to *edge chunks*, that is the last row or column of
chunks along one axis, which may not be exactly divisible by the ``chunk_size`` before
and/or after the resize operation.

Shrinking is trivial: if an edge chunk needs to be reduced in size along one or more
axes, it doesn't need to be actually modified on the slabs. The ``StagedChangesArray``
simply knows that, from this moment on, everything beyond the edge of the chunk on the
slab is to be treated as uninitialised memory.

Creating brand new chunks when enlarging is also trivial, as they are simply filled with
0 on both ``slab_indices`` and ``slab_offsets`` to represent that they lie on the full
slab. They won't exist on the staged slabs until someone writes to them with
``__setitem__``.

Enlarging edge chunks that don't lie on the full slab is more involved, as they need to
be physically filled with the fill_value:

1. If a chunk lies on a base slab, it first needs to be transferred over to a staged
   slab, which is created brand new for the occasion;
2. then, there is a transfer from the full slab to the staged slab for the extra area
   that needs to be filled with the fill_value.


``load()`` algorithm
--------------------

``LoadPlan`` ensures that all chunks are either on the full slab or on a staged slab. It
selects all chunks in that lie on a base slab and transfers them to a brand new staged
slab.


.. _staged_changes_commit:

``commit()`` algorithm
----------------------
To *commit* is to take all the chunks that lie on staged slabs and move them to a single
brand new base slab (backed by HDF5), deduplicating them along the way. It is the
inverse of ``load()``: where ``load()`` copies the base slabs into a new staged slab,
``commit()`` copies the (surviving) staged slabs into a new base slab and then drops all
the staged slabs.

Before commit
^^^^^^^^^^^^^
The ``StagedChangesArray.slabs`` list is accompanied by a paired ``hash_tables`` list.
``hash_tables[1]``, which matches ``slabs[1]`` which is the h5py ``raw_data`` dataset,
is loaded from h5py into memory. When ``__setitem__`` etc. append a new staged slab to
``slabs``, they also append a ``None`` placeholder to ``hash_tables``: hashes are
computed lazily at commit time, not eagerly at ``__setitem__`` time.

Commit process
^^^^^^^^^^^^^^
Committing is broken down into the following stages:

1. Define a ``HashPlan`` to discover which chunks don't have a known hash. This
   typically translates to all staged chunks that were not deleted by later
   mutations, plus the full slab. The output is a table of hashing instructions (slab,
   selection of the slab to hash, target location on the ``hash_tables``) which is
   similar in design to the table produced by the various plans that result in copying
   data between slabs.
2. Execute the ``HashPlan`` to generate the hashes. This is performed by Cython without
   ever involving the Python API, so that it remains performant for small chunks.
3. Define a ``CommitPlan``, which:

   a. reads the hashes of the staged chunks that were just generated, plus the hashes of
      all the chunks on the base slabs from HDF5;
   b. drops duplicate staged chunks (any that are identical to the full chunk, a base
      chunk on HDF5, or another staged chunk);
   c. plans to copy the unique staged chunks to a new base slab;
   d. updates ``slab_indices`` and ``slab_offsets`` so that all chunks that were
      previously pointing to a staged slab now point to the full slab, an old base slab,
      or the new base slab.

4. Allocate a new base slab with a `np.empty`-like callback that was provided by the
   wrapper. Under the hood, the function extends the h5py `raw_data` dataset and returns
   a view to the new surface. `StagedChangesArray` doesn't know anything about this -
   from it's point of view it's just a writeable NumPy-like array.
5. Execute the ``CommitPlan``. The low-level Cython function that performs the transfer
   drops the generic Numpy-like abstraction and takes a specialised code path to copy
   from dense NumPy to HDF5. Again the iteration does not involve any Python API.
6. Dereference the staged slabs (remove them from the ``staged_slabs`` list entirely;
   don't just replace them with None's)
7. Return control to the wrapper, which just needs to append the new ``hash_table`` to
   ``raw_data``'s hash table on h5py.

Nuances and caveats
^^^^^^^^^^^^^^^^^^^
Because ``hash_tables[i][j, :]`` always contains the hash of ``slabs[i][j *
chunk_size[0]:(j+1)*chunk_size[0]``, one needs to be careful when writing to disk a slab
that ends with an edge chunk: it must not be trimmed at the bottom.

``CommitPlan`` scans *all* available hashes, including those of chunks that were in
previous versions, which are still on ``slabs[1]`` (a.k.a. ``raw_data``) have since been
deleted. This can resuscitate an old chunk after its deletion.

When ``CommitPlan`` finds a duplicate chunk, the slab with the lowest index wins the
tie:

1. the full slab always wins, which means that if the user calls ``__setitem__`` to
   completely fill a chunk with the ``fill_value``, nothing will be written on disk;
2. then the base slabs win, meaning if the user calls ``__setitem__`` to write to a
   chunk values that are identical to any chunk already on disk, nothing gets written;
3. finally the lowest (arbitrary) staged chunk wins, meaning that two identical stage
   chunks result in only one chunk being written.

For sake of simplicity, ``CommitPlan`` is blind to the geometry of edge chunks. To
compensate, staged slabs are initialised to the ``fill_value``, so that the cells of an
edge chunk that lie beyond the array boundary hold the ``fill_value`` rather than
uninitialised memory; this keeps the layout deterministic when ``commit()`` later copies
whole chunks (padding included) to HDF5.

A new base slab is appended *only if at least one staged chunk survives deduplication*.
If every staged chunk turns out to be a duplicate of a base or full chunk (or there are
no staged chunks at all), nothing is written: no new base slab is created and
``n_base_slabs`` is left unchanged; the staged chunks are simply repointed to their
duplicates before the staged slabs are dropped. Crucially, this may mean you end up with
a new *version* without any new *data* (the new version is a remix of pre-existing
chunks).

Reclaiming memory
-----------------
Each plan that mutates the state - ``SetItemPlan``, ``ResizePlan``, and ``LoadPlan`` -
has a chance of not needing a chunk anymore on a particular slab, either because that
chunk does not exist anymore (``resize()`` to shrink the shape) or because it's been
moved from a base slab to a staged slab (``__setitem__``, ``resize()``, or ``load()``).

When a chunk leaves a slab, it leaves an empty area in the old slab. This is normal and
fine when the slab is disk-backed (the ``raw_data`` ``h5py.Datataset`` that serves as a
base slab), but results in memory fragmentation and potentially a perceived "memory
leak" from the final user when the slab is in memory (a staged slab). For the sake of
simplicity, the surface is never reused; later operations just create new slabs.

In practice, fragmentation should not be a problem, as it only happens if someone
updates a chunk with ``__setitem__`` and later drops that very same chunk with
``resize()`` - which is obviously wasteful so it should not be part of a typical
workflow. Additionally, slabs are cleaned up as soon as the staged version is committed.

If a slab is completely empty, however - in other words, it no longer appears in
``slab_indices`` - it *may* be dropped. This is guaranteed to happen for staged slabs
and *may* happen for base slabs too (if computationally cheap to determine). Note that
nothing particular happens today when the ``raw_data`` base slab, which is a hdf5
dataset,is deferenced by the ``StagedChangesArray``.

When a slab is dropped, it is replaced by None in the ``slabs`` list, which dereferences
it. This allows not to change all the following slab indices after the operation.
The full slab is never dropped, as it may be needed later by ``resize()`` to create new
chunks or partially fill existing edge chunks.


Copy-on-Write (CoW) mechanics
-----------------------------
Several methods (``copy()``, ``astype()``, ``refill()``) perform a functional deep
copy of the array. What is actually happening however is that ``slab_indices``,
``slab_offsets``, and chunks are replaced on both the source and destination with
read-only views. Upon the first write access (which may never happen!) on either side,
these views are lazily replaced with a writeable deep copy.

Additionally, ``astype()`` doesn't actually change the dtype of staged slabs until the
first write *or read* access. Upon first access, the full staged slab is converted to
the new dtype, not just the selection requested.


Hot-swapping the base slabs
---------------------------
When calling ``astype()``, base slabs are normally fully loaded into memory and then
converted with NumPy to the new dtype. This may not be desirable:
`h5py.Dataset.astype()`_ returns a lazy ``AsTypeView`` object, which directly performs
dtype conversions in libhdf5, only on the areas selected with ``__getitem__``. To
leverage this, ``StagedChangesArray.astype()`` has the option of hot-swapping the base
slabs with any other array-like objects with the new dtype and the same shapes as the
original slabs.


API interaction
---------------
.. graphviz::

  digraph {
      node [shape=box];

      user [shape=ellipse; label="Final user"]

      subgraph cluster_0 {
          label="wrappers.py";

          DatasetWrapper -> InMemoryDataset;
          DatasetWrapper -> InMemorySparseDataset;
          DatasetWrapper -> InMemoryArrayDataset;
      }

      subgraph cluster_1 {
          label="staged_changes.py";

          StagedChangesArray;
          getitem [label="__getitem__()"]
          setitem [label="__setitem__()"]
          resize [label="resize()"]
          load [label="load()"]
          changes [label="changes()"]
          commit [label="commit()"]
          all_plans [label="All \*Plan classes", style="dashed"]

          StagedChangesArray -> getitem -> GetItemPlan -> TransferPlan;
          StagedChangesArray -> setitem -> SetItemPlan -> TransferPlan;
          StagedChangesArray -> resize -> ResizePlan -> TransferPlan;
          StagedChangesArray -> load -> LoadPlan -> TransferPlan;
          StagedChangesArray -> changes -> ChangesPlan;
          StagedChangesArray -> commit -> CommitPlan -> TransferPlan;
          commit -> HashPlan -> HashSlabPlan;

          # No actual dependency; just a positioning hack
          HashPlan -> all_plans [style="invis"]
      }

      subgraph cluster_2 {
          label="subchunk_map.py";

          read_many_slices_param_nd -> IndexChunkMapper;
      }

      subgraph cluster_3 {
          label="slicetools.pyx";

          build_slab_indices_and_offsets;
          read_many_slices;
          hdf5_c [label="libhdf5 C API (via Cython)"];
          build_slab_indices_and_offsets -> hdf5_c;
          read_many_slices -> hdf5_c;
      }

      subgraph cluster_4 {
          label="hash.pyx";
          hash_slab;
          libcrypto_api [label="libcrypto C API (via Cython)"]
      }

      libcrypto [label="libcrypto (OpenSSL)"];
      hash_slab -> libcrypto_api -> libcrypto;

      user -> DatasetWrapper;

      InMemoryDataset -> StagedChangesArray;
      InMemoryDataset -> build_slab_indices_and_offsets;
      InMemorySparseDataset -> StagedChangesArray;
      TransferPlan -> read_many_slices;
      TransferPlan -> read_many_slices_param_nd;
      HashSlabPlan -> hash_slab;

      all_plans -> IndexChunkMapper;

      InMemorySparseDataset -> commit_version;
      InMemoryArrayDataset -> commit_version;
      InMemoryDataset -> commit_version;

      h5py;
      commit_version -> h5py;
      commit_version -> changes;
      commit_version -> commit;
      hdf5_file [label="HDF5 file"; shape=cylinder];
      h5py -> hdf5_file;
      hdf5_c -> hdf5_file;
  }

**Notes**

- ``read_many_slices_param_nd`` has API awareness of ``read_many_slices`` to craft its
  input parameters, but no API integration.
- Likewise, ``build_slab_indices_and_offsets`` knows about the format of the
  ``slab_indices`` and ``slab_offsets`` of ``StagedChangesArray``, but does not directly
  interact with it.


.. _H5Sselect_hyperslab: https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d
.. _H5Dread: https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c
.. _h5py.Dataset.astype(): https://docs.h5py.org/en/stable/high/dataset.html#h5py.Dataset.astype
