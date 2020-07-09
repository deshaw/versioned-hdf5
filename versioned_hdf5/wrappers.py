"""
Wrappers of h5py objects that work in memory

Much of this code is modified from code in h5py. See the LICENSE file for the
h5py license.
"""

from h5py import Empty, Dataset, Datatype, Group, h5d, h5i, h5p, h5s, h5t, h5r
from h5py._hl.base import guess_dtype, with_phil, phil
from h5py._hl.dataset import _LEGACY_GZIP_COMPRESSION_VALS
from h5py._hl import filters
from h5py._hl.selections import select, guess_shape
from h5py._hl.vds import VDSmap

from ndindex import Tuple, Slice, ndindex

import numpy as np

from collections import defaultdict
import math
import posixpath as pp

from .backend import DEFAULT_CHUNK_SIZE
from .slicetools import split_slice, spaceid_to_slice, as_subchunks

_groups = {}
class InMemoryGroup(Group):
    def __new__(cls, bind):
        # Make sure each group only corresponds to one InMemoryGroup instance.
        # Otherwise a new instance would lose track of any datasets or
        # subgroups created in the old one.
        # TODO: Garbage collect closed groups.
        if bind in _groups:
            return _groups[bind]
        obj = super().__new__(cls)
        obj._initialized = False
        _groups[bind] = obj
        return obj

    def __init__(self, bind):
        if self._initialized:
            return
        self._data = {}
        self._subgroups = {}
        self.chunks = defaultdict(type(None))
        self.compression = defaultdict(type(None))
        self.compression_opts = defaultdict(type(None))
        self._parent = None
        self._initialized = True
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
        dirname, basename = pp.split(name)
        if dirname:
            return self.__getitem__(dirname)[basename]

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
        dirname, basename = pp.split(name)
        if dirname:
            self[dirname][basename] = obj
            return

        if isinstance(obj, Dataset):
            self._data[name] = InMemoryDataset(obj.id)
        elif isinstance(obj, Group):
            self._subgroups[name] = InMemoryGroup(obj.id)
        elif isinstance(obj, InMemoryGroup):
            self._subgroups[name] = obj
        elif isinstance(obj, InMemoryArrayDataset):
            self._data[name] = obj
        else:
            self._data[name] = InMemoryArrayDataset(name, np.asarray(obj))

    def __delitem__(self, name):
        if name in self._data:
            del self._data[name]

    @property
    def parent(self):
        if self._parent is None:
            return super().parent
        return self._parent

    @parent.setter
    def parent(self, p):
        self._parent = p

    def create_group(self, name, track_order=None):
        if name.startswith('/'):
            raise ValueError("Root level groups cannot be created inside of versioned groups")
        group = type(self)(
            super().create_group(name, track_order=track_order).id)
        g = group
        n = name
        while n:
            dirname, basename = pp.split(n)
            if not dirname:
                parent = self
            else:
                parent = type(self)(g.parent.id)
            parent._subgroups[basename] = g
            g.parent = parent
            g = parent
            n = dirname
        return group

    def create_dataset(self, name, **kwds):
        dirname, data_name = pp.split(name)
        if dirname and dirname not in self:
            self.create_group(dirname)
        data = _make_new_dset(**kwds)
        shape = data.shape
        if 'fillvalue' in kwds:
            data = InMemoryArrayDataset(name, data, fillvalue=kwds['fillvalue'])
        chunks = kwds.get('chunks')
        if chunks in [True, None]:
            if len(shape) == 1:
                chunks = (DEFAULT_CHUNK_SIZE,)
            else:
                raise NotImplementedError("chunks must be specified for multi-dimensional datasets")
        if isinstance(chunks, int) and not isinstance(chunks, bool):
            chunks = (chunks,)
        if len(shape) != len(chunks):
            raise ValueError("chunks shape must equal the array shape")
        if len(shape) == 0:
            raise NotImplementedError("Scalar datasets")
        self.chunks[name] = chunks
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

    def __contains__(self, item):
        dirname, data_name = pp.split(item)
        if dirname not in ['', '/']:
            return dirname in self and data_name in self[dirname]
        for i in self:
            if i == item:
                return True
        return False

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
            func(pp.join(prefix, name), self[name])
            if isinstance(self[name], InMemoryGroup):
                self[name]._visit(pp.join(prefix, name), func)

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
        return tuple(self.id.chunks)

    @property
    def attrs(self):
        return self._attrs

    def __array__(self, dtype=None):
        return self.__getitem__((), new_dtype=dtype)

    @with_phil
    def __getitem__(self, args, new_dtype=None):
        """ Read a slice from the HDF5 dataset.

        Takes slices and recarray-style field names (more than one is
        allowed!) in any order.  Obeys basic NumPy rules, including
        broadcasting.

        Also supports:

        * Boolean "mask" array indexing
        """
        # This boilerplate code is based on h5py.Dataset.__getitem__
        args = args if isinstance(args, tuple) else (args,)

        if new_dtype is None:
            new_dtype = getattr(self._local, 'astype', None)

        # Sort field names from the rest of the args.
        names = tuple(x for x in args if isinstance(x, str))

        if names:
            # Read a subset of the fields in this structured dtype
            if len(names) == 1:
                names = names[0]  # Read with simpler dtype of this field
            args = tuple(x for x in args if not isinstance(x, str))
            return self.fields(names, _prior_dtype=new_dtype)[args]

        if new_dtype is None:
            new_dtype = self.dtype
        mtype = h5t.py_create(new_dtype)

        # === Special-case region references ====

        if len(args) == 1 and isinstance(args[0], h5r.RegionReference):

            obj = h5r.dereference(args[0], self.id)
            if obj != self.id:
                raise ValueError("Region reference must point to this dataset")

            sid = h5r.get_region(args[0], self.id)
            mshape = guess_shape(sid)
            if mshape is None:
                # 0D with no data (NULL or deselected SCALAR)
                return Empty(new_dtype)
            out = np.empty(mshape, dtype=new_dtype)
            if out.size == 0:
                return out

            sid_out = h5s.create_simple(mshape)
            sid_out.select_all()
            self.id.read(sid_out, sid, out, mtype)
            return out

        # === END CODE FROM h5py.Dataset.__getitem__ ===

        idx = ndindex(args).reduce(self.shape)

        arr = np.ndarray(idx.newshape(self.shape), new_dtype, order='C')

        for c, index in as_subchunks(idx, self.shape, self.chunks):
            if isinstance(self.id.data_dict[c], (slice, Slice, tuple, Tuple)):
                a = self.id._read_chunk(c.raw, mtype=mtype, dxpl=self._dxpl)
                self.id.data_dict[c] = a

            if self.id.data_dict[c].size != 0:
                arr_idx = c.as_subindex(idx)
                arr[arr_idx.raw] = self.id.data_dict[c][index.raw]

        return arr

    @with_phil
    def __setitem__(self, args, val):
        """ Write to the HDF5 dataset from a Numpy array.

        NumPy's broadcasting rules are honored, for "simple" indexing
        (slices and integers).  For advanced indexing, the shapes must
        match.
        """
        # This boilerplate code is based on h5py.Dataset.__setitem__
        args = args if isinstance(args, tuple) else (args,)

        # Sort field indices from the slicing
        names = tuple(x for x in args if isinstance(x, str))
        args = tuple(x for x in args if not isinstance(x, str))

        # Generally we try to avoid converting the arrays on the Python
        # side.  However, for compound literals this is unavoidable.
        vlen = h5t.check_vlen_dtype(self.dtype)
        if vlen is not None and vlen not in (bytes, str):
            try:
                val = np.asarray(val, dtype=vlen)
            except ValueError:
                try:
                    val = np.array([np.array(x, dtype=vlen)
                                       for x in val], dtype=self.dtype)
                except ValueError:
                    pass
            if vlen == val.dtype:
                if val.ndim > 1:
                    tmp = np.empty(shape=val.shape[:-1], dtype=object)
                    tmp.ravel()[:] = [i for i in val.reshape(
                        (np.product(val.shape[:-1], dtype=np.ulonglong), val.shape[-1]))]
                else:
                    tmp = np.array([None], dtype=object)
                    tmp[0] = val
                val = tmp
        elif self.dtype.kind == "O" or \
          (self.dtype.kind == 'V' and \
          (not isinstance(val, np.ndarray) or val.dtype.kind != 'V') and \
          (self.dtype.subdtype == None)):
            if len(names) == 1 and self.dtype.fields is not None:
                # Single field selected for write, from a non-array source
                if not names[0] in self.dtype.fields:
                    raise ValueError("No such field for indexing: %s" % names[0])
                dtype = self.dtype.fields[names[0]][0]
                cast_compound = True
            else:
                dtype = self.dtype
                cast_compound = False

            val = np.asarray(val, dtype=dtype.base, order='C')
            if cast_compound:
                val = val.view(np.dtype([(names[0], dtype)]))
                val = val.reshape(val.shape[:len(val.shape) - len(dtype.shape)])
        else:
            val = np.asarray(val, order='C')

        # Check for array dtype compatibility and convert
        if self.dtype.subdtype is not None:
            shp = self.dtype.subdtype[1]
            valshp = val.shape[-len(shp):]
            if valshp != shp:  # Last dimension has to match
                raise TypeError("When writing to array types, last N dimensions have to match (got %s, but should be %s)" % (valshp, shp,))
            mtype = h5t.py_create(np.dtype((val.dtype, shp)))
            mshape = val.shape[0:len(val.shape)-len(shp)]

        # Make a compound memory type if field-name slicing is required
        elif len(names) != 0:

            mshape = val.shape

            # Catch common errors
            if self.dtype.fields is None:
                raise TypeError("Illegal slicing argument (not a compound dataset)")
            mismatch = [x for x in names if x not in self.dtype.fields]
            if len(mismatch) != 0:
                mismatch = ", ".join('"%s"'%x for x in mismatch)
                raise ValueError("Illegal slicing argument (fields %s not in dataset type)" % mismatch)

            # Write non-compound source into a single dataset field
            if len(names) == 1 and val.dtype.fields is None:
                subtype = h5t.py_create(val.dtype)
                mtype = h5t.create(h5t.COMPOUND, subtype.get_size())
                mtype.insert(self._e(names[0]), 0, subtype)

            # Make a new source type keeping only the requested fields
            else:
                fieldnames = [x for x in val.dtype.names if x in names] # Keep source order
                mtype = h5t.create(h5t.COMPOUND, val.dtype.itemsize)
                for fieldname in fieldnames:
                    subtype = h5t.py_create(val.dtype.fields[fieldname][0])
                    offset = val.dtype.fields[fieldname][1]
                    mtype.insert(self._e(fieldname), offset, subtype)

        # Use mtype derived from array (let DatasetID.write figure it out)
        else:
            mtype = None


        # === END CODE FROM h5py.Dataset.__setitem__ ===

        idx = ndindex(args).reduce(self.shape)

        val = np.broadcast_to(val, idx.newshape(self.shape))

        for c, index in as_subchunks(idx, self.shape, self.chunks):
            if isinstance(self.id.data_dict[c], (slice, Slice, tuple, Tuple)):
                a = self.id._read_chunk(c.raw, mtype=mtype, dxpl=self._dxpl)
                self.id.data_dict[c] = a

            if self.id.data_dict[c].size != 0:
                val_idx = c.as_subindex(idx)
                self.id.data_dict[c][index.raw] = val[val_idx.raw]

class InMemoryArrayDataset:
    """
    Class that looks like a h5py.Dataset but is backed by an array
    """
    def __init__(self, name, array, fillvalue=None):
        self.name = name
        self._array = array
        self.attrs = {}
        self.fillvalue = fillvalue or array.dtype.type()

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
            self.array = np.concatenate((self.array, np.full(size[0] -
                                                             self.shape[0], self.fillvalue,
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

        slice_map = {spaceid_to_slice(i.vspace): spaceid_to_slice(i.src_space)
                     for i in virtual_sources}

        # slice_map = {i.args[0]: j.args[0] for i, j in slice_map.items()}
        fid = h5i.get_file_id(self)
        g = Group(fid)
        self.chunks = tuple(g[virtual_sources[0].dset_name].attrs['chunks'])

        for s in slice_map:
            src_idx = slice_map[s]
            if isinstance(src_idx, Tuple):
                # The pointers to the raw data should only be slices, since
                # the raw data chunks are extended in the first dimension
                # only.
                assert src_idx != Tuple()
                assert len(src_idx.args) == len(self.chunks)
                assert all(i.reduce() == Slice(0, j, 1) for i, j in
                    zip(src_idx.args[1:], self.chunks[1:])), (src_idx, self.chunks)
                src_idx = src_idx.args[0]
            assert isinstance(src_idx, Slice)
            self.data_dict[s] = src_idx

        fillvalue_a = np.empty((1,), dtype=self.dtype)
        dcpl.get_fill_value(fillvalue_a)
        self.fillvalue = fillvalue_a[0]

    def set_extent(self, shape):
        raise NotImplementedError("resizing is not yet implemented")

        old_shape = self.shape
        if old_shape[1:] != shape[1:]:
            raise NotImplementedError("Resizing is currently only supported in the first dimension")
        data_dict = self.data_dict
        chunks = self.chunks
        chunk_size = chunks[0]
        if shape[0] < old_shape[0]:
            for i in list(data_dict):
                if (i + 1)*chunk_size > shape[0]:
                    if i*chunk_size >= shape[0]:
                        del data_dict[i]
                    else:
                        if isinstance(data_dict[i], (Slice, slice, tuple, Tuple)):
                            # Non-chunk multiple
                            a = self._read_chunk(i)
                        else:
                            a = data_dict[i]
                        data_dict[i] = a[:shape[0] - i*chunk_size]
        elif shape[0] > old_shape[0]:
            quo, rem = divmod(shape[0], chunk_size)
            if old_shape[0] % chunk_size != 0:
                i = max(data_dict)
                if isinstance(data_dict[i], (Slice, slice, tuple, Tuple)):
                    a = self._read_chunk(i)
                else:
                    a = data_dict[i]
                assert a.shape[0] == old_shape[0] % chunk_size

                if i == quo:
                    data_dict[i] = np.concatenate([a, np.full((rem -
                        a.shape[0],) + chunks[1:], self.fillvalue, dtype=self.dtype)])
                else:
                    data_dict[i] = np.concatenate([a, np.full((chunk_size -
                        a.shape[0],) + chunks[1:], self.fillvalue, dtype=self.dtype)])
            if rem != 0 and quo not in data_dict:
                # fillvalue along the chunks are added in the for loop below, but
                # we have to add a sub-chunk fillvalues here
                data_dict[quo] = np.full((rem,) + chunks[1:], self.fillvalue, dtype=self.dtype)
            for i in range(math.ceil(old_shape[0]/chunk_size), quo):
                data_dict[i] = np.full(chunks, self.fillvalue, dtype=self.dtype)
        self.shape = shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, size):
        self._shape = size

    def _read_chunk(self, chunk_idx, mtype=None, dxpl=None):
        # Based on Dataset.__getitem__
        selection = select(self.shape, chunk_idx, dsid=self)

        assert selection.nselect != 0

        a = np.ndarray(selection.mshape, self.dtype, order='C')

        # Read the data into the array a
        mspace = h5s.create_simple(selection.mshape)
        fspace = selection.id
        super().read(mspace, fspace, a, mtype, dxpl=dxpl)
        return a

    def write(self, mspace, fspace, arr_obj, mtype=None, dxpl=None):
        raise NotImplementedError("Writing to an InMemoryDataset other than via __setitem__")

    def read(self, mspace, fspace, arr_obj, mtype=None, dxpl=None):
        raise NotImplementedError("Reading from an InMemoryDataset other than via __getitem__")
