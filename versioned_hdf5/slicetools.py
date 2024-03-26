from functools import lru_cache

from ndindex import Integer, Slice, Tuple


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
        for start, stride, count, block in zip(starts, strides, counts, blocks):
            slices.append(hyperslab_to_slice(start, stride, count, block))
        return Tuple(*slices)
    elif sel_type == h5s.SEL_NONE:
        return Tuple(
            Slice(0, 0),
        )
    else:
        raise NotImplementedError("Point selections are not yet supported")


@lru_cache(2048)
def hyperslab_to_slice(start, stride, count, block):
    if not (block == 1 or count == 1):
        raise NotImplementedError("Nontrivial blocks are not yet supported")
    end = start + (stride * (count - 1) + 1) * block
    stride = stride if block == 1 else 1
    return Slice(start, end, stride)


def to_slice_tuple(index: Tuple) -> Tuple:
    """Make an ndindex.Tuple into a ndindex.Tuple of slices.

    ndindex.Integer doesn't support the same methods that ndindex.Slice does. This function ensures
    that indices which are ndindex.Integer are converted into ndindex.Slice first, so that we can
    use a common API.

    Parameters
    ----------
    index : Tuple
        Tuple index to convert. Slice dimensions are left as-is; Tuple dimensions are converted to
        single-element slices.

    Returns
    -------
    Tuple
        Tuple of Slice instance; Integer dimensions are converted to single-element Slice instances.
    """
    result = []
    for dim in index.args:
        if isinstance(dim, Integer):
            result.append(Slice(dim.raw, dim.raw + 1))
        elif isinstance(dim, Slice):
            result.append(dim)
        else:
            raise TypeError(f"Cannot convert type of {dim} to a Slice.")

    return Tuple(*result)
