import math

# Helper functions to workaround slices not being hashable
def s2t(s):
    if isinstance(s, tuple):
        return tuple(s2t(i) for i in s)
    return (s.start, s.stop)

def t2s(t):
    if isinstance(t[0], tuple):
        return tuple(slice(*i) for i in t)
    return slice(*t)

def split_slice(s, chunk):
    """
    Split a slice into multiple slices along 0:chunk, chunk:2*chunk, etc.

    Yields tuples, (i, slice), where i is the chunk that should be sliced.
    """
    start, stop, step = s.start, s.stop, s.step
    if any(i < 0 for i in [start, stop, step]):
        raise NotImplementedError("slices with negative values are not yet supported")
    for i in range(math.floor(start/chunk), math.ceil(stop/chunk)):
        if i == 0:
            new_start = start
        elif i*chunk < start:
            new_start = start - i*chunk
        else:
            new_start = (i*chunk - start) % step
            if new_start:
                new_start = step - new_start
        new_stop = min(stop - i*chunk, chunk)
        new_step = step
        yield i, slice(new_start, new_stop, new_step)

def slice_size(s):
    """
    Give the maximum size of an array axis sliced by slice s

    The true size could be smaller if the slice extends beyond the bounds of
    the array.

    """
    start, stop, step = s.start, s.stop, s.step
    if step == None:
        step = 1
    if start == None:
        start = 0
    return len(range(start, stop, step))
