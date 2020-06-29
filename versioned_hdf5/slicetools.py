from ndindex import Slice, Tuple, ndindex

from itertools import product
import math

# TODO: Move this into ndindex
def split_slice(s, chunk):
    """
    Split a slice into multiple slices along 0:chunk, chunk:2*chunk, etc.

    Yields tuples, (i, slice), where i is the chunk that should be sliced.
    """
    start, stop, step = s.args
    for i in range(math.floor(start/chunk), math.ceil(stop/chunk)):
        yield i, s.as_subindex(Slice(i*chunk, (i + 1)*chunk))

def as_subchunks(idx, shape, chunk):
    """
    Split an index `idx` on an array of shape `shape` into subchunks assuming
    a chunk size of `chunk`.

    Yields tuples `(c, index)`, where `c` is an index for the chunk that
    should be sliced, and `index` is an index into that chunk giving the
    elements of `idx` that are included in it (`c` and `index` are both
    ndindex indices).

    That is to say, for each `(c, index)` pair yielded, `a[c][index]` will
    give those elements of `a[idx]` that are part of the `c` chunk.

    Note that this only yields those indices that are nonempty.

    >>> from versioned_hdf5.slicetools import as_subchunks
    >>> idx = (slice(5, 15), 0)
    >>> shape = (20, 20)
    >>> chunk = (10, 10)
    >>> for c, index in as_subchunks(idx, shape, chunk):
    ...     print(c)
    ...     print('    ', index)
    Tuple(slice(0, 10, None), slice(0, 10, None))
        Tuple(slice(5, 10, 1), 0)
    Tuple(slice(10, 20, None), slice(0, 10, None))
        Tuple(slice(0, 5, 1), 0)

    """
    idx = ndindex(idx)
    for c in split_chunks(shape, chunk):
        index = idx.as_subindex(c)
        if not index.isempty(chunk):
            yield (c, index)


# TODO: Should this go in ndindex?
def split_chunks(shape, chunk):
    """
    Yield a set of ndindex indices for chunks over shape

    For example, if a has shape (10, 20) and is chunked into chunks of shape
    (5, 5). If the shape is not a multiple of the chunk size, some chunks will
    be truncated.

    >>> from versioned_hdf5.slicetools import split_chunks
    >>> for i in split_chunks((10, 19), (5, 5)):
    ...     print(i)
    Tuple(slice(0, 5, None), slice(0, 5, None))
    Tuple(slice(0, 5, None), slice(5, 10, None))
    Tuple(slice(0, 5, None), slice(10, 15, None))
    Tuple(slice(0, 5, None), slice(15, 19, None))
    Tuple(slice(5, 10, None), slice(0, 5, None))
    Tuple(slice(5, 10, None), slice(5, 10, None))
    Tuple(slice(5, 10, None), slice(10, 15, None))
    Tuple(slice(5, 10, None), slice(15, 19, None))

    """
    if len(shape) != len(chunk):
        raise ValueError("chunks shape must equal the array shape")
    if len(shape) == 0:
        raise NotImplementedError("Scalar datasets")

    d = [math.ceil(i/c) for i, c in zip(shape, chunk)]
    for c in product(*[range(i) for i in d]):
        # c = (0, 0, 0), (0, 0, 1), ...
        yield Tuple(*[Slice(chunk_size*i, min(chunk_size*(i + 1), n)) for
    n, chunk_size,
                      i in zip(shape, chunk, c)])

def spaceid_to_slice(space):
    from h5py import h5s

    sel_type = space.get_select_type()

    if sel_type == h5s.SEL_ALL:
        return Tuple()
    elif sel_type == h5s.SEL_HYPERSLABS:
        slices = []
        starts, strides, counts, blocks = space.get_regular_hyperslab()
        for _start, _stride, count, block in zip(starts, strides, counts, blocks):
            start = _start
            if not (block == 1 or count == 1):
                raise NotImplementedError("Nontrivial blocks are not yet supported")
            end = _start + (_stride*(count - 1) + 1)*block
            stride = _stride if block == 1 else 1
            slices.append(Slice(start, end, stride))
        return Tuple(*slices)
    elif sel_type == h5s.SEL_NONE:
        return Tuple(Slice(0, 0),)
    else:
        raise NotImplementedError("Point selections are not yet supported")
