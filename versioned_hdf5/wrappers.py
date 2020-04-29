"""
Wrappers of h5py objects that work in memory

Much of this code is modified from code in h5py. See the LICENSE file for the
h5py license.
"""

from h5py import Empty, Dataset, Datatype, Group, h5d, h5i, h5p, h5s, h5t
from h5py._hl.base import guess_dtype, phil
from h5py._hl.dataset import _LEGACY_GZIP_COMPRESSION_VALS
from h5py._hl import filters
from h5py._hl.selections import select
from h5py._hl.vds import VDSmap

import numpy as np

from collections import defaultdict
import math
import posixpath as pp

from .slicetools import s2t, slice_size, split_slice, spaceid_to_slice


class InMemoryGroup(Group):
    def __init__(self, bind):
        self._data = {}
        self._subgroups = {}
        self.chunk_size = defaultdict(type(None))
        self.compression = defaultdict(type(None))
        self.compression_opts = defaultdict(type(None))
        super().__init__(bind)

    # Based on Group.__repr__
    def __repr__(self):
        if not self:
            r = u"<Closed InMemoryGroup>"
        else:
            namestr = (
                '"%s"' % self.name
            ) if self.name is not None else u"(anonymous)"
            r = '<InMemoryGroup %s (%d members)>' % (namestr, len(self))

        return r

    def __getitem__(self, name):
        parts = name.split('/')
        if len(parts) > 1:
            return self.__getitem__(parts[0])['/'.join(parts[1:])]

        if name in self._data:
            return self._data[name]
        if name in self._subgroups:
            return self._subgroups[name]

        res = super().__getitem__(name)
        if isinstance(res, Group):
            self._subgroups[name] = self.__class__(res.id)
            return self._subgroups[name]
        elif isinstance(res, Dataset):
            self._data[name] = InMemoryDataset(res.id)
            return self._data[name]
        else:
            raise NotImplementedError(f"Cannot handle {type(res)!r}")

    def __setitem__(self, name, obj):
        parts = name.split('/')
        if len(parts) > 1:
            self[parts[0]]['/'.join(parts[1:])] = obj
            return

        if isinstance(obj, Dataset):
            self._data[name] = InMemoryDataset(obj.id)
        elif isinstance(obj, Group):
            self._subgroups[name] = InMemoryGroup(obj.id)
        elif isinstance(obj, InMemoryGroup):
            self._subgroups[name] = obj
        else:
            self._data[name] = InMemoryArrayDataset(name, np.asarray(obj))

    def __delitem__(self, name):
        if name in self._data:
            del self._data[name]

    def create_group(self, name, track_order=None):
        g = super().create_group(name, track_order=track_order)
        group = type(self)(g.id)
        self._subgroups[name.split('/')[0]] = group
        return group

    def create_dataset(self, name, **kwds):
        *path, data_name = name.split('/')
        self.create_group('/'.join(path))
        data = _make_new_dset(**kwds)
        chunk_size = kwds.get('chunks')
        if isinstance(chunk_size, tuple):
            if len(chunk_size) > 1:
                raise NotImplementedError("Multiple dimensions")
            chunk_size = chunk_size[0]
        if chunk_size is True:
            raise NotImplementedError("auto-chunking is not yet supported")
        self.chunk_size[name] = chunk_size
        self.compression[name] = kwds.get('compression')
        self.compression_opts[name] = kwds.get('compression_opts')
        self[name] = data
        return self[name]

    def __iter__(self):
        names = list(self._data) + list(self._subgroups)
        for i in super().__iter__():
            if i in names:
                names.remove(i)
            yield i
        for i in names:
            yield i

    def datasets(self):
        res = self._data.copy()

        def _get(name, item):
            if name in res:
                return
            if isinstance(item, (Dataset, InMemoryArrayDataset, np.ndarray)):
                res[name] = item

        self.visititems(_get)

        return res

    def visititems(self, func):
        self._visit('', func)

    def _visit(self, prefix, func):
        for name in self:
            func(prefix + name, self[name])
            if isinstance(self[name], InMemoryGroup):
                self[name]._visit(name + '/', func)

    #TODO: override other relevant methods here

# Based on h5py._hl.dataset.make_new_dset(), except it doesn't actually create
# the dataset, it just canonicalizes the arguments. See the LICENSE file for
# the h5py license.
def _make_new_dset(shape=None, dtype=None, data=None, chunks=None,
                  compression=None, shuffle=None, fletcher32=None,
                  maxshape=None, compression_opts=None, fillvalue=None,
                  scaleoffset=None, track_times=None, external=None,
                  track_order=None, dcpl=None):
    """ Return a new low-level dataset identifier """

    # Convert data to a C-contiguous ndarray
    if data is not None and not isinstance(data, Empty):
        # normalize strings -> np.dtype objects
        if dtype is not None:
            _dtype = np.dtype(dtype)
        else:
            _dtype = None

        # if we are going to a f2 datatype, pre-convert in python
        # to workaround a possible h5py bug in the conversion.
        is_small_float = (_dtype is not None and
                          _dtype.kind == 'f' and
                          _dtype.itemsize == 2)
        data = np.asarray(data, order="C",
                             dtype=(_dtype if is_small_float
                                    else guess_dtype(data)))

    # Validate shape
    if shape is None:
        if data is None:
            if dtype is None:
                raise TypeError("One of data, shape or dtype must be specified")
            data = Empty(dtype)
        shape = data.shape
    else:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if data is not None and (np.product(shape, dtype=np.ulonglong) != np.product(data.shape, dtype=np.ulonglong)):
            raise ValueError("Shape tuple is incompatible with data")

    if isinstance(maxshape, int):
        maxshape = (maxshape,)

    # Validate chunk shape
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)
    # The original make_new_dset errors here if the shape is less than the
    # chunk size, but we avoid doing that as we cannot change the chunk size
    # for a dataset for any version once it is created. See #34.

    if isinstance(dtype, Datatype):
        # Named types are used as-is
        tid = dtype.id
        dtype = tid.dtype  # Following code needs this
    else:
        # Validate dtype
        if dtype is None and data is None:
            dtype = np.dtype("=f4")
        elif dtype is None and data is not None:
            dtype = data.dtype
        else:
            dtype = np.dtype(dtype)
        tid = h5t.py_create(dtype, logical=1)

    # Legacy
    if any((compression, shuffle, fletcher32, maxshape, scaleoffset)) and chunks is False:
        raise ValueError("Chunked format required for given storage options")

    # Legacy
    if compression is True:
        if compression_opts is None:
            compression_opts = 4
        compression = 'gzip'

    # Legacy
    if compression in _LEGACY_GZIP_COMPRESSION_VALS:
        if compression_opts is not None:
            raise TypeError("Conflict in compression options")
        compression_opts = compression
        compression = 'gzip'
    dcpl = filters.fill_dcpl(
        dcpl or h5p.create(h5p.DATASET_CREATE), shape, dtype,
        chunks, compression, compression_opts, shuffle, fletcher32,
        maxshape, scaleoffset, external)

    if fillvalue is not None:
        fillvalue = np.array(fillvalue)
        dcpl.set_fill_value(fillvalue)

    if track_times in (True, False):
        dcpl.set_obj_track_times(track_times)
    elif track_times is not None:
        raise TypeError("track_times must be either True or False")
    if track_order == True:
        dcpl.set_attr_creation_order(
            h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
    elif track_order == False:
        dcpl.set_attr_creation_order(0)
    elif track_order is not None:
        raise TypeError("track_order must be either True or False")

    if maxshape is not None:
        maxshape = tuple(m if m is not None else h5s.UNLIMITED for m in maxshape)


    if isinstance(data, Empty):
        raise NotImplementedError("Empty datasets")
    return data

class InMemoryDataset(Dataset):
    """
    Class that looks like a h5py.Dataset but is backed by a versioned dataset

    The versioned dataset can be modified, which performs modifications
    in-memory only.
    """
    def __init__(self, bind, **kwargs):
        # Hold a reference to the original bind so h5py doesn't invalidate the id
        # XXX: We need to handle deallocation here properly when our object
        # gets deleted or closed.
        self.orig_bind = bind
        super().__init__(InMemoryDatasetID(bind.id), **kwargs)
        self._attrs = dict(super().attrs)

    @property
    def chunks(self):
        return (self.id.chunk_size,)

    @property
    def attrs(self):
        return self._attrs

class InMemoryArrayDataset:
    """
    Class that looks like a h5py.Dataset but is backed by an array
    """
    def __init__(self, name, array):
        self.name = name
        self._array = array
        self.attrs = {}

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, array):
        self._array = array

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def ndim(self):
        return len(self._array.shape)

    def __getitem__(self, item):
        return self.array.__getitem__(item)

    def __setitem__(self, item, value):
        self.array.__setitem__(item, value)

    def __len__(self):
        return self.len()

    def len(self):
        """
        Length of the first axis
        """
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Attempt to take len() of scalar dataset")
        return shape[0]

    def __array__(self, dtype=None):
        return self.array

    def __repr__(self):
        name = pp.basename(pp.normpath(self.name))
        namestr = '"%s"' % (name if name != '' else '/')
        return '<InMemoryArrayDataset %s: shape %s, type "%s">' % (
                namestr, self.shape, self.dtype.str
            )

    def __iter__(self):
        """ Iterate over the first axis.  TypeError if scalar.

        BEWARE: Modifications to the yielded data are *NOT* written to file.
        """
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Can't iterate over a scalar dataset")
        for i in range(shape[0]):
            yield self[i]

    def resize(self, size, axis=None):
        if axis is not None:
            if not (axis >=0 and axis < self.ndim):
                raise ValueError("Invalid axis (0 to %s allowed)" % (self.ndim-1))
            try:
                newlen = int(size)
            except TypeError:
                raise TypeError("Argument must be a single int if axis is specified")
            size = list(self.shape)
            size[axis] = newlen

        size = tuple(size)
        if len(size) > 1:
            raise NotImplementedError("More than one dimension is not yet supported")
        if size[0] > self.shape[0]:
            self.array = np.concatenate((self.array,
                                         np.zeros(size[0] - self.shape[0],
                                                  dtype=self.dtype)))
        else:
            self.array = self.array[:size[0]]

class InMemoryDatasetID(h5d.DatasetID):
    def __init__(self, _id):
        # super __init__ is handled by DatasetID.__cinit__ automatically
        self.data_dict = {}
        with phil:
            sid = self.get_space()
            self._shape = sid.get_simple_extent_dims()

        dcpl = self.get_create_plist()
        # Same as dataset.get_virtual_sources
        virtual_sources = [
                VDSmap(dcpl.get_virtual_vspace(j),
                       dcpl.get_virtual_filename(j),
                       dcpl.get_virtual_dsetname(j),
                       dcpl.get_virtual_srcspace(j))
                for j in range(dcpl.get_virtual_count())]

        slice_map = {s2t(spaceid_to_slice(i.vspace)): spaceid_to_slice(i.src_space)
                     for i in virtual_sources}
        if any(len(i) != 1 for i in slice_map) or any(len(i) != 1 for i in slice_map.values()):
            raise NotImplementedError("More than one dimension is not yet supported")

        slice_map = {i[0]: j[0] for i, j in slice_map.items()}
        fid = h5i.get_file_id(self)
        g = Group(fid)
        self.chunk_size = g[virtual_sources[0].dset_name].attrs['chunk_size']
        for i in range(math.ceil(self.shape[0]/self.chunk_size)):
            for t in slice_map:
                r = range(*t)
                if i*self.chunk_size in r:
                    self.data_dict[i] = slice_map[t]

    def set_extent(self, shape):
        if len(shape) > 1:
            raise NotImplementedError("More than one dimension is not yet supported")

        old_shape = self.shape
        data_dict = self.data_dict
        chunk_size = self.chunk_size
        if shape[0] < old_shape[0]:
            for i in list(data_dict):
                if (i + 1)*chunk_size > shape[0]:
                    if i*chunk_size >= shape[0]:
                        del data_dict[i]
                    else:
                        if isinstance(data_dict[i], slice):
                            # Non-chunk multiple
                            a = self._read_chunk(i)
                        else:
                            a = data_dict[i]
                        data_dict[i] = a[:shape[0] - i*chunk_size]
        elif shape[0] > old_shape[0]:
            quo, rem = divmod(shape[0], chunk_size)
            if old_shape[0] % chunk_size != 0:
                i = max(data_dict)
                if isinstance(data_dict[i], slice):
                    a = self._read_chunk(i)
                else:
                    a = data_dict[i]
                assert a.shape[0] == old_shape % chunk_size
                if i == quo:
                    data_dict[i] = np.concatenate([a, np.zeros((rem -
                        a.shape[0]), dtype=self.dtype)])
                else:
                    data_dict[i] = np.concatenate([a, np.zeros((chunk_size -
                        a.shape[0],), dtype=self.dtype)])
            if rem != 0 and quo not in data_dict:
                # Zeros along the chunks are added in the for loop below, but
                # we have to add a sub-chunk zeros here
                data_dict[quo] = np.zeros((rem,), dtype=self.dtype)
            for i in range(math.ceil(old_shape[0]/chunk_size), quo):
                data_dict[i] = np.zeros((chunk_size,), dtype=self.dtype)
        self.shape = shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, size):
        self._shape = size

    def _read_chunk(self, i, mtype=None, dxpl=None):
        # Based on Dataset.__getitem__
        s = slice(i*self.chunk_size, (i+1)*self.chunk_size)
        selection = select(self.shape, (s,), dsid=self)

        assert selection.nselect != 0

        a = np.ndarray(selection.mshape, self.dtype, order='C')

        # Read the data into the array a
        mspace = h5s.create_simple(selection.mshape)
        fspace = selection.id
        super().read(mspace, fspace, a, mtype, dxpl=dxpl)
        return a

    def write(self, mspace, fspace, arr_obj, mtype=None, dxpl=None):
        if mtype is not None:
            raise NotImplementedError("mtype != None")
        mslice = spaceid_to_slice(mspace)
        fslice = spaceid_to_slice(fspace)
        if len(fslice) > 1 or len(self.shape) > 1:
            raise NotImplementedError("More than one dimension is not yet supported")
        data_dict = self.data_dict
        arr = arr_obj[mslice]
        if np.isscalar(arr):
            arr = arr.reshape((1,))

        if fslice == ():
            fslice = (slice(0, arr_obj.shape[0], 1),)
        # Chunks that are modified
        N0 = 0
        for i, s_ in split_slice(fslice[0], chunk=self.chunk_size):
            if isinstance(self.data_dict[i], slice):
                a = self._read_chunk(i, mtype=mtype, dxpl=dxpl)
                data_dict[i] = a

            N = N0 + slice_size(s_)
            data_dict[i][s_] = arr[N0:N]
            N0 = N

        return data_dict

    def read(self, mspace, fspace, arr_obj, mtype=None, dxpl=None):
        mslice = spaceid_to_slice(mspace)
        fslice = spaceid_to_slice(fspace)
        if len(fslice) > 1 or len(self.shape) > 1:
            raise NotImplementedError("More than one dimension is not yet supported")
        data_dict = self.data_dict
        arr = arr_obj[mslice]
        if np.isscalar(arr):
            arr = arr.reshape((1,))

        if fslice == ():
            fslice = (slice(0, arr_obj.shape[0], 1),)
        # Chunks that are modified
        N0 = 0
        for i, s_ in split_slice(fslice[0], chunk=self.chunk_size):
            if isinstance(self.data_dict[i], slice):
                a = self._read_chunk(i, mtype=mtype, dxpl=dxpl)
                data_dict[i] = a

            N = N0 + slice_size(s_)
            arr[N0:N] = data_dict[i][s_]
            N0 = N
