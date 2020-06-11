from ndindex import Slice, Tuple

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

def split_chunks(shape, chunks):
    if len(shape) != len(chunks):
        raise ValueError("chunks shape must equal the array shape")
    if len(shape) == 0:
        raise NotImplementedError("Scalar datasets")

    if any(i != j for i, j in zip(shape[1:], chunks[1:])):
        raise NotImplementedError("Chunking over any dimension other than the first is not yet implemented")

    chunk_size = chunks[0]

    for i in range(math.ceil(shape[0]/chunk_size)):
        yield Tuple(Slice(chunk_size*i, chunk_size*(i + 1)))

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
