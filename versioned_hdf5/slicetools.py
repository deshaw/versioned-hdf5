from functools import lru_cache
from typing import Dict

import numpy as np
from ndindex import Slice, Tuple


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


class AppendChunk:
    def __init__(
        self,
        target_vindex: Tuple,
        target_rindex: Slice,
        array: np.ndarray,
        extant_vindex: Tuple,
        extant_rindex: Slice,
    ):
        self.target_vindex = target_vindex
        self.target_rindex = target_rindex
        self.array = array
        self.extant_vindex = extant_vindex
        self.extant_rindex = extant_rindex

    def get_concatenated_rindex(self) -> Slice:
        return Slice(self.extant_rindex.start, self.target_rindex.stop)
