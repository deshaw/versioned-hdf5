from ndindex import Slice, Tuple

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

# TODO: Should this go in ndindex?
def split_chunks(shape, chunks):
    """
    Yield a set of ndindex indices for chunks over shape

    For example, if a has shape (10, 20) and is chunked into chunks of shape
    (5, 5).

    >>> from versioned_hdf5.slicetools import split_chunks
    >>> for i in split_chunks((10, 20), (5, 5)):
    ...     print(i)
    Tuple(slice(0, 5, None), slice(0, 5, None))
    Tuple(slice(0, 5, None), slice(5, 10, None))
    Tuple(slice(0, 5, None), slice(10, 15, None))
    Tuple(slice(0, 5, None), slice(15, 20, None))
    Tuple(slice(5, 10, None), slice(0, 5, None))
    Tuple(slice(5, 10, None), slice(5, 10, None))
    Tuple(slice(5, 10, None), slice(10, 15, None))
    Tuple(slice(5, 10, None), slice(15, 20, None))

    """
    if len(shape) != len(chunks):
        raise ValueError("chunks shape must equal the array shape")
    if len(shape) == 0:
        raise NotImplementedError("Scalar datasets")

    d = [math.ceil(i/c) for i, c in zip(shape, chunks)]
    for c in product(*[range(i) for i in d]):
        # c = (0, 0, 0), (0, 0, 1), ...
        yield Tuple(*[Slice(chunk_size*i, chunk_size*(i + 1)) for chunk_size,
                      i in zip(chunks, c)])

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
