from h5py import h5p
from h5py._hl.base import phil

from .slicetools import _spaceid_to_slice


def build_data_dict(dcpl: h5p.PropDCID, raw_data_name: str):
    """Build the data_dict of a versioned virtual dataset.

    All virtual datasets created by versioned-hdf5 should have chunks in
    exactly one raw dataset `raw_data_name` in the same file.
    This function blindly assumes this is the case.

    :param dcpl: the dataset creation property list of the versioned dataset
    :param raw_data_name: the name of the corresponding raw dataset
    :return: a dictionary mapping the `Tuple` of the virtual dataset chunk
        to a `Slice` in the raw dataset.
    """
    data_dict: dict = {}

    with phil:
        for j in range(dcpl.get_virtual_count()):
            vspace = dcpl.get_virtual_vspace(j)
            srcspace = dcpl.get_virtual_srcspace(j)

            vspace_slice_tuple = _spaceid_to_slice(vspace.id)
            srcspace_slice_tuple = _spaceid_to_slice(srcspace.id)

            # the slice into the raw_data (srcspace_slice_tuple) is only
            # on the first axis
            data_dict[vspace_slice_tuple] = srcspace_slice_tuple.args[0]

    return data_dict
