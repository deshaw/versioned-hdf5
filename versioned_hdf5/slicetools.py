from functools import lru_cache
from typing import Iterator, Optional, Union

import numpy as np
from ndindex import ChunkSize, Integer, Slice, Tuple
from ndindex.ndindex import NDIndex


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


def index_relative_to(index: Tuple, other: Tuple) -> Tuple:
    """Compute the index relative to the start of another index.

    Parameters
    ----------
    index : Tuple
        A Tuple index
    other : Tuple
        Another Tuple index

    Returns
    -------
    Tuple
        The first index, adjusted to be relative to the second

    Examples
    --------
        index:          ---------------

        other:     --------------------------


                   ^    ^              ^
                   index starts at other[5]
                   index stops at other[20]

        thus index_relative_to(index, other) -> Tuple(Slice(5, 20))
    """
    relative = []
    for dim1, dim2 in zip(index.args, other.args):
        relative.append(Slice(dim1.start - dim2.start, dim1.stop - dim2.start))
    return Tuple(*relative)


def to_raw_index(rchunk: Tuple, relative_index: Tuple) -> Tuple:
    """Convert a relative virtual index to a raw slice.

    The use case for this function is if you have a raw chunk and a corresponding virtual chunk,
    and you want to get a subchunk of the virtual chunk and its corresponding raw indices,
    you'll first need to compute the relative indices of the virtual chunk; then you'd use
    to_raw_index to get the raw indices of the subchunk.

    Parameters
    ----------
    rchunk : Tuple
        Raw index defining the boundaries of a chunk of the raw dataset
    relative_index : Tuple
        Arbitrary slice of a dataset; indices are relative to the start of the
        a chunk. Since this is relative to the start of a chunk, it is neither a virtual
        nor a raw index.

    Returns
    -------
    Tuple
        Slice of the raw dataset computed using the relative index

    Examples
    --------
        rchunk:   Slice(5, 25)
        relative_index: Slice(3, 5)
        to_raw_index(rchunk, relative_index) -> Tuple(Slice(8, 10))

        The call to `to_raw_index` should be read as "Return elements 3-5 of rchunk."
    """
    raw_indices = []
    for rdim, relative_dim in zip(rchunk.args, relative_index.args):
        offset = rdim.start
        if not (
            rdim.start <= offset + relative_dim.start <= rdim.stop
            and rdim.start <= offset + relative_dim.stop <= rdim.stop
        ):
            raise ValueError(
                f"Cannot convert elements {relative_index} to a raw index "
                f"without exceeding the bounds of the raw chunk: {rchunk}"
            )

        raw_indices.append(
            Slice(relative_dim.start + offset, relative_dim.stop + offset)
        )
    return Tuple(*raw_indices)


def get_vchunk_overlap(vchunk: NDIndex, index: NDIndex) -> tuple[Tuple, Tuple]:
    """Get the overlap of the index with the vchunk.

    Here, ``index`` represents an arbitrary slice which may or may not overlap
    a virtual chunk ``vchunk_data``, defined by the index ``vchunk``.

      +---index.as_subindex(vchunk)----+  <-- Referenced with respect to beginning of vchunk
      |                                |
      +------------ vchunk ------------+
      |                                |
    +------------ arr -------------------------+
    |                                          |
    +-+-------- index -----------------+-------+
      |                                |
      +----vchunk.as_subindex(index)---+  <-- Referenced with respect to beginning of index




    Examples
    --------
    Assume we are trying to write an array ``arr`` to an index ``index``. Then for
    a given virtual slice ``vchunk``, ``get_vchunk_overlap(vchunk, index)``
    returns the indices in ``arr`` that overlap with the virtual slice, and the indices
    of the virtual chunk to write to the new virtual slice.

    arr_overlap, relative_overlap = get_vchunk_overlap(vchunk, index)

    Elements of arr that overlap:
        arr[arr_overlap.raw]

    Elements of the virtual chunk ``vchunk_data`` to overwrite:
        vchunk_data[relative_overlap.raw]

    Parameters
    ----------
    vchunk : NDIndex
        Chunk of virtual data
    index : NDIndex
        Arbitrary index in the virtual dataset

    Returns
    -------
    tuple(Tuple, Tuple)
        (Indices of ``arr`` which overlap, indices of ``vchunk_data`` which overlap)
    """
    return vchunk.as_subindex(index), index.as_subindex(vchunk)


def get_shape(index: Tuple, shape: Optional[tuple[int, ...]] = None) -> tuple[int, ...]:
    """Get the size of the index along each dimension.

    If the shape has any indefinite boundaries e.g. NoneType, Ellipsis,
    these will be defined by `shape`.

    Parameters
    ----------
    index : Tuple
        Index which defines a shape

    shape : Optional[tuple[int, ...]]
        Shape to resolve any indefinite boundaries on. If None,
        an exception will be raised if there are any unresolved
        boundaries in `index`

    Returns
    -------
    tuple[int, ...]
        Shape of the region defined by index, after applying the
        index to an array of shape `shape`
    """
    if shape is not None:
        index = index.expand(shape)

    return tuple([len(dim) for dim in index.args])


def partition(
    obj: Union[np.ndarray, Tuple],
    chunks: Union[int, tuple[int, ...], ChunkSize],
) -> Iterator[Tuple]:
    """Break an array or a Tuple of slices into chunks of the given chunk size.

    Parameters
    ----------
    obj : Union[np.ndarray, Tuple]
        Array or Tuple index to partition
    chunks : Union[int, tuple[int, ...], ChunkSize]
        If this is an int, this is the size of each partitioned chunk.
        If it is a tuple of ints or a ChunkSize, consider the indices of the
        object the shape of the chunks. Multidimensional chunks should supply
        a tuple giving the chunk size in each dimension.

    Returns
    -------
    Iterator[Tuple]
        A list of slices of arr that make up the chunks
    """
    if isinstance(obj, np.ndarray):
        index = Tuple(*[Slice(0, dim) for dim in obj.shape])
        shape = obj.shape
    else:
        index = to_slice_tuple(obj)
        shape = tuple(dim.stop for dim in index.args)

    if isinstance(chunks, (int, np.integer)):
        chunks = (chunks,)
    elif isinstance(chunks, ChunkSize):
        chunks = tuple(chunks)

    yield from ChunkSize(chunks).as_subchunks(index, shape)
