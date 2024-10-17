Staging changes in memory
=========================
This section explains the low level design of ``InMemoryDataset``.

When a ``InMemoryDataset`` is modified in any way, all changed chunks are held in memory
until they are ready to be committed to disk.
To do so, ``InMemoryDataset`` wraps around a ``StagedChangesArray`` object, which is a
numpy.ndarray-like with a basic and opaque API:

.. class:: StagedChangesArray(base_getitem, shape, chunk_size, dtype=None, fill_value=0)

    ``base_getitem: Callable[[Any], np.ndarray``
        The ``__getitem__`` method of the underlying ndarray-like
        (which may or may not be a ``h5py.Dataset``).

        .. note::
            Returned arrays can be written back to unless they are views of another
            array. Set the writeable flag to False before returning them to prevent
            this (in most cases, you don't need to worry about this).

    ``shape: tuple[int, ...]``
        The current shape of the array. If ``resize()`` has been called, it may diff
        from the one of the underlying array.
    ``chunk_size: tuple[int, ...]``
        The division of the array into chunks which, when wrapping around a
        h5py.Dataset, must match the one of the underlying array.

    .. property:: chunk_states

        ``NDArray[np.intp]``

        Array with same number of dimesions as base, containing 1 point per chunk, with
        the following values:

        0
            Chunk is not modified and must be read with ``base_getitem()``
        \-1
            Chunk is full of ``fill_value``
        1+
            Chunk has been modified and is stored in ``chunk_values`` at the matching
            index.

    .. property:: chunk_values

        ``list[NDArray[T] | None]``

        Modified chunks, as indexed by ``chunk_states``. Chunk 0 is always None.
        Other chunks can be None if they've been deleted after a resize() operation
        shrunk the array.

    .. property:: has_changes

        ``bool``

        True if there are any changes from the base array; False otherwise.
        In other words, if the ``changes()`` method will yield anything at all.

    .. method:: changes(load_base=False, full_chunks=True)

        Yield all the changed chunks so far, as tuples of

        - slice index in the base array
        - chunk value, or None if the chunk has been removed after a ``resize()``

        ``load_base``: bool, optional
            If True, load all chunks from the base array, even if they haven't been
            modified.
        ``full_chunks``: bool, optional
            If True, yield all chunks, even if they are full of the fill_value. Full
            chunks will be created on the fly.
            If False, only yield chunks that are at least partially filled with data.

    .. method:: __getitem__(idx)

        Get a slice of data from the array. This reads from the staged chunks
        in memory when available and from the base array otherwise.

    .. method:: __setitem__(idx)

        Break the given value into chunks and store it in memory.
        Do not modify the base array.

        .. note::

            This method may preserve views of the value array and assumes it is OK to
            write back to it on later calls to __setitem__. If this is not desirable,
            you need to set the writeable flag to False on the value array before
            passing it to __setitem__.

    .. method:: resize(shape: tuple[int, ...])

        Change the array shape and fill new elements with ``fill_value``.
        Note that this works like ``h5py.Dataset.resize()``,
        and not like ``numpy.ndarray.resize()`` - meaning elements are not reflowed.

    .. method:: copy()

        Return a new ``StagedChangesArray``. This is a Copy-on-Write (CoW).

    .. method:: astype(dtype, casting="unsafe")

        Return a new ``StagedChangesArray`` with a different dtype.

    .. method:: refill(fill_value)

        Return a new ``StagedChangesArray`` with a different ``fill_value``.

    .. staticmethod:: full(shape, chunk_size, dtype=None, fill_value=0)

        Create a new ``StagedChangesArray`` with all chunks already in memory and
        full of ``fill_value``.
        It won't consume any significant amounts of memory until it's modified.


The key methods of ``StagedChangesArray`` are explained below. Note that there will be
references to internal methods implemented in the package that are not documented here
for brevity.


``*Plan`` classes
-----------------

All these methods encapsulate their complex decision-making logic into classes with no
access to the actual data: ``GetItemPlan`` for ``__getitem__()``, ``SetItemPlan`` for
``__setitem__()`` and so on. The methods of ``StagedChangesArray`` are themselves dumb,
as they just generate the Plan class and then execute its instructions.

This design is meant to facilitate debugging. To know what
``StagedChangesArray.__getitem__(idx)`` will do in any given situation, one just needs
to call ``StagedChangesArray._getitem_plan(idx)``, and so on for the other methods.
The ``_*_plan()`` methods trivially extract the relevant bits of state from
``StagedChangesArray`` and use them to construct the matching ``*Plan`` class.
From there a developer can inspect the output e.g. in a Jupyter notebook.


``__getitem__`` algorithm
-------------------------

1. Preprocess the index with ``index_chunk_mappers()`` to generate an instance of a
   ``IndexChunkMapper`` subclass for each axis, matching the axes on the index
   (e.g. ``SliceMapper`` for slices), plus padding with ``EverythingMapper``'s for the
   axes not in the index.
2. Create an empty output array using ``ndindex.ndindex(idx).newshape()``.
3. Query ``IndexChunkMapper.chunks_indexer()`` on each axis and use it to slice
   ``StagedChangesArray.chunk_states`` to obtain only the chunks that are involved by
   the index.
4. From the slice of ``StagedChangesArray.chunk_states``, get the chunks that are
   non-zero - e.g either modified and stored in ``StageChangesArray.chunk_values``
   (``chunk_states > 0``) or full of ``fill_value`` (``chunk_states == -1``).
5. For each modified chunk, for each axis, call
   ``out, sub = IndexChunkMapper.chunk_submap(i, i + 1, shift=True)``, where i is the
   index of the chunk along the axis, to figure out the ``sub`` portion of the original
   index that selects the data within the chunk and the ``out`` portion of the index to
   deposit it in the output array and use them to populate the final output.

   .. code-block:: python

      for i in range(ndim):
          out[i], sub[i] = mappers[i].chunk_submap(
              chunk_idx[i], chunk_idx[i] + 1, shift=True
          )
      output[out] = chunk[sub]

6. Repeat for the full chunks (you won't need ``sub``):

   .. code-block:: python

      for i in range(ndim):
          out[i], _ = mappers[i].chunk_submap(chunk_idx[i], chunk_idx[i] + 1, shift=True)
      output[out] = fill_value

7. Feed the indices of the modified and full chunks to ``fill_hyperspace()`` to
   generate a list of hyperrectangles of unmodified chunks.
8. For each hyperrectangle, call
   ``out, sub = IndexChunkMapper.chunk_submap(a, b, shift=False)``
   where [a, b[ are the edges of the hyperrectangle along each axis.

9. Read directly from the base array into the output:

   .. code-block:: python

      for rect in fill_hyperspace(obstacles, chunk_states.shape):
          rect_start, rect_stop = rect[:ndim], rect[ndim:]
          for i in range(ndim):
              out[i], sub[i] = mappers[i].chunk_submap(
                  rect_start[i], rect_stop[i], shift=False
              )
          output[out] = StagedChangesArray.base_getitem(sub)


``__setitem__`` algorithm
-------------------------

1. Exactly like in ``__getitem__()``, find the chunks in
   ``StagedChangesArray.chunk_states`` that are impacted by the index

2. For each chunk that is already modified and stored in
   ``StagedChangesArray.chunk_values`` (``chunk_states > 0``), update its contents,
   like in ``__getitem__()`` but in reverse:

   .. code-block:: python

      for i in range(ndim):
          out[i], sub[i] = mappers[i].chunk_submap(
              chunk_idx[i], chunk_idx[i] + 1, shift=True
          )
      chunk[sub] = value[out]

3. As a special case of the above, if the ``sub`` index selects the whole chunk, then
   replace the chunk instead. This in most cases can happen trivially, creating a view
   in ``chunk_values`` of the ``__setitem__`` value,

   .. code-block:: python

      chunk = value[out]

   but in more complex cases you will need to create an empty chunk and then fill it;
   e.g. consider ``a[[0, 2, 1]] = np.arange(3)``:

   .. code-block:: python

      chunk = np.empty(chunk_shape)
      chunk[sub] = value[out]

4. Query ``IndexChunkMapper.whole_chunks_indexer()`` on each axis. If a chunk is a whole
   chunk along *all* axes, it's wholly selected. If it's selected by
   ``IndexChunkMapper.chunks_indexer()`` but missing on one or more axes from
   ``whole_chunks_indexer()``, it's partially selected.

5. Cross the information above with each chunk that is not yet in memory
   (``chunk_states == 0``),

6. If the chunk is wholly selected along all axes, simply append it to
   ``chunk_values`` and flip the corresponding point in ``chunk_states`` to the new
   index in chunk_values. Again you need to consider if ``sub`` is trivial or not:

   .. code-block:: python

      for i in range(ndim):
          out[i], _ = mappers[i].chunk_submap(chunk_idx[i], chunk_idx[i] + 1, shift=True)
      chunk_states[chunk_idx] = len(chunk_values)

      if is_trivial(sub):
          chunk = value[out]
      else:
          chunk = np.empty(chunk_shape)
          chunk[sub] = value[out]

      chunk_values.append(chunk)

7. If the chunk is only partially selected along one or more axes, we need to first load
   it from the base array *and then* update it. Much like in ``__getitem__()`` we loaded
   many chunks at once with a single ``base_getitem()`` call, we're going to do the same
   here, calling ``fill_hyperspace()`` and passing as obstacles the list of the chunks
   that are already in memory, *plus the list of the chunks that are not in memory but
   are wholly selected by the index*. ``fill_hyperspace()`` will generate the
   ~complement of this map, which is all the chunks that are not already in memory and
   only partially selected by the ``__setitem__()`` index.

   This will result in slabs of the base array loaded at once, which we now need to cut
   them into individual chunks, append them to chunk_values, and flip the point in
   chunk_states like we did in the previous step.

   .. code-block:: python

      slab = base_getitem[rect_start[i]:rect_stop[i] for i in range(ndim)]
      for sub in cartesian_product(
          [
              np.arange(rect_start[i], rect_stop[i], chunk_size[i])
              for i in range(ndim)
          ]
      ):
          chunk_values.append(slab[sub])

   Once these chunks are loaded, we can proceed to update them like in step 2.

8. If the chunk is partially selected along one or more axes AND it's full of
   ``fill_value`` (``chunk_states == -1``), then

   a. create an actual full chunk of the correct shape
   b. append it to ``chunk_values`` and update the matching point in ``chunk_states``
   c. update the chunk as normal as in step 2.


``resize()`` algorithm
----------------------

1. When you shrink and drop an entire chunk as a consequence, if the chunk is in memory
   (``chunk_states > 0``), dereference it by setting ``chunk_values[i]`` to None.

2. When you enlarge and create a brand new chunk as a consequence, set
   ``chunk_states`` to -1. Don't create any entry in ``chunk_values``.

3. Edge chunks that are not exactly divisible by the ``chunk_size`` in either the old
   or the new shape which are not exactly divisible by the ``chunk_size`` in the new
   shape need special treatment:

   a. If a chunk was not in memory (``chunk_states == 0``), load it from the base array.
      Do so in entire slabs at a time, much like in ``__getitem__()`` or
      ``__setitem__()``.
   b. Resize each edge chunk; if enlarging, fill the new area with ``fill_value``.

4. Finally, keep track of the original shape. When the user calls ``changes()``, yield
   None for any chunk that has been deleted over any number of ``resize()`` operations,
   including edge chunks that changed in shape.


``as_type()`` algorithm
-----------------------

``StagedChangesArray.as_type()`` is performed eagerly for the chunks that are already
in memory in ``chunk_values`` and lazily for those that aren't, meaning that they are
converted on the fly by ``base_getitem()``.

The method raises an internal flag which later causes ``changes()`` to yield all chunks
in the array, not just the ones impacted by ``__setitem___()`` or ``resize()``.


``copy()`` algorithm
--------------------

``StagedChangesArray.copy()`` performs a Copy-on-Write, meaning that the old and new
object will share all the chunks in memory, until one of the two objects modifies a
chunk. Only at that point, the individual chunk will be copied and the two
StagedChangesArray will diverge.


``refill()`` algorithm
----------------------

``StagedChangesArray.refill()`` calls ``copy()``; then it changes the ``fill_value``,
which impacts all current and future full chunks; finally it iterates on
``chunk_values`` and replaces all points equal to the old ``fill_value`` with the new
one. If and only if a chunk contains at least one point equal to the wold ``fill_value``
the CoW view created by ``copy()`` is replaced by an actual deep copy.

Finally, it overrides ``base_getitem()`` so that any new chunks that are returned by
it have the old ``fill_value`` replaced with the new one
The method raises an internal flag which later causes ``changes()`` to yield all chunks
in the array, not just the ones impacted by ``__setitem___()`` or ``resize()``.


``full()`` algorithm
--------------------
This static method creates a new ``StagedChangesArray`` with all chunks filled with
``fill_value`` - meaning that the whole ``chunk_states`` is set to -1 and
``chunk_values`` is empty. As it will never interact with the base array,
``base_getitem()`` is a dummy that never gets called.
