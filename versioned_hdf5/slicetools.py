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


def as_subchunks(idx, shape, chunks):
    """
    Split an index `idx` on an array of shape `shape` into subchunks assuming
    a chunk size of `chunks`.

    Yields tuples `(c, index)`, where `c` is an index for the chunks that
    should be sliced, and `index` is an index into that chunk giving the
    elements of `idx` that are included in it (`c` and `index` are both
    ndindex indices).

    That is to say, for each `(c, index)` pair yielded, `a[c][index]` will
    give those elements of `a[idx]` that are part of the `c` chunk.

    Note that this only yields those indices that are nonempty.

    >>> from versioned_hdf5.slicetools import as_subchunks
    >>> idx = (slice(5, 15), 0)
    >>> shape = (20, 20)
    >>> chunks = (10, 10)
    >>> for c, index in as_subchunks(idx, shape, chunks):
    ...     print(c)
    ...     print('    ', index)
    Tuple(slice(0, 10, 1), slice(0, 10, 1))
        Tuple(slice(5, 10, 1), 0)
    Tuple(slice(10, 20, 1), slice(0, 10, 1))
        Tuple(slice(0, 5, 1), 0)

    """
    idx = ndindex(idx)
    for c in split_chunks(shape, chunks):
        try:
            index = idx.as_subindex(c)
        except ValueError:
            continue

        if not index.isempty(chunks):
            yield (c, index)


# TODO: Should this go in ndindex?
def split_chunks(shape, chunks):
    """
    Yield a set of ndindex indices for chunks over shape

    If the shape is not a multiple of the chunk size, some chunks will be
    truncated.

    For example, if a has shape (10, 19) and is chunked into chunks
    of shape (5, 5):

    >>> from versioned_hdf5.slicetools import split_chunks
    >>> for i in split_chunks((10, 19), (5, 5)):
    ...     print(i)
    Tuple(slice(0, 5, 1), slice(0, 5, 1))
    Tuple(slice(0, 5, 1), slice(5, 10, 1))
    Tuple(slice(0, 5, 1), slice(10, 15, 1))
    Tuple(slice(0, 5, 1), slice(15, 19, 1))
    Tuple(slice(5, 10, 1), slice(0, 5, 1))
    Tuple(slice(5, 10, 1), slice(5, 10, 1))
    Tuple(slice(5, 10, 1), slice(10, 15, 1))
    Tuple(slice(5, 10, 1), slice(15, 19, 1))

    """
    if len(shape) != len(chunks):
        raise ValueError("chunks dimensions must equal the array dimensions")
    if len(shape) == 0:
        # chunk_size = 1
        yield Tuple(Slice(0))
    else:
        if len(shape) != len(chunks):
            raise ValueError("chunks dimensions must equal the array dimensions")

    d = [math.ceil(i/c) for i, c in zip(shape, chunks)]
    if 0 in d:
        yield Tuple(*[Slice(0, bool(i)*chunk_size, 1) for i, chunk_size in zip(d, chunks)]).expand(shape)
    for c in product(*[range(i) for i in d]):
        # c = (0, 0, 0), (0, 0, 1), ...
        yield Tuple(*[Slice(chunk_size*i, min(chunk_size*(i + 1), n), 1) for n, chunk_size,
                      i in zip(shape, chunks, c)])


def spaceid_to_slice(space):
    """
    Convert an h5py spaceid object into an ndindex index

    The resulting index is always a Tuple index.
    """

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
