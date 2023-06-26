import numpy as np
from h5py import VirtualLayout, VirtualSource, h5s
from ndindex import Slice, ndindex, Tuple, ChunkSize

from .hashtable import Hashtable

DEFAULT_CHUNK_SIZE = 2**12

def normalize_dtype(dtype):
    return np.array([], dtype=dtype).dtype

def get_chunks(shape, dtype, chunk_size):
    # TODO: Implement this
    if len(shape) > 1:
        raise NotImplementedError
    return (chunk_size,)

def initialize(f):
    import datetime
    from .versions import TIMESTAMP_FMT

    version_data = f.create_group('_version_data')
    versions = version_data.create_group('versions')
    versions.create_group('__first_version__')
    versions.attrs['current_version'] = '__first_version__'
    ts = datetime.datetime.now(datetime.timezone.utc)
    versions['__first_version__'].attrs['timestamp'] = ts.strftime(TIMESTAMP_FMT)


def create_base_dataset(f, name, *, shape=None, data=None, dtype=None,
    chunks=True, compression=None, compression_opts=None, fillvalue=None):

    # Validate shape (based on h5py._hl.dataset.make_new_dset)
    if shape is None:
        if data is None:
            if dtype is None:
                raise TypeError("One of data, shape or dtype must be specified")
            raise NotImplementedError("empty datasets are not yet implemented")
        shape = data.shape
    else:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if data is not None and (np.product(shape, dtype=np.ulonglong) != np.product(data.shape, dtype=np.ulonglong)):
            raise ValueError("Shape tuple is incompatible with data")

    ndims = len(shape)
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)
    if chunks in [True, None]:
        if ndims <= 1:
            chunks = (DEFAULT_CHUNK_SIZE,)
        else:
            raise NotImplementedError("chunks must be specified for multi-dimensional datasets")
    group = f['_version_data'].create_group(name)

    if dtype is None:
        # https://github.com/h5py/h5py/issues/1474
        dtype = data.dtype
    dtype = normalize_dtype(dtype)
    if dtype.metadata and ('vlen' in dtype.metadata or 'h5py_encoding' in dtype.metadata):
        # h5py string dtype
        # (https://h5py.readthedocs.io/en/2.10.0/strings.html). Setting the
        # fillvalue in this case doesn't work
        # (https://github.com/h5py/h5py/issues/941).
        if fillvalue not in [0, '', b'', None]:
            raise ValueError("Non-default fillvalue not supported for variable length strings")
        fillvalue = None
    dataset = group.create_dataset('raw_data', shape=(0,) + chunks[1:],
                                   chunks=tuple(chunks), maxshape=(None,) + chunks[1:],
                                   dtype=dtype, compression=compression,
                                   compression_opts=compression_opts,
                                   fillvalue=fillvalue)
    dataset.attrs['chunks'] = chunks
    return write_dataset(f, name, data, chunks=chunks)


def write_dataset(f, name, data, chunks=None, dtype=None, compression=None,
                  compression_opts=None, fillvalue=None):

    if name not in f['_version_data']:
        return create_base_dataset(f, name, data=data, dtype=dtype,
                                   chunks=chunks, compression=compression,
                                   compression_opts=compression_opts, fillvalue=fillvalue)

    ds = f['_version_data'][name]['raw_data']
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)
    if chunks is None:
        chunks = tuple(ds.attrs['chunks'])
    else:
        if chunks != tuple(ds.attrs['chunks']):
            raise ValueError("Chunk size specified but doesn't match already existing chunk size")

    if dtype is not None:
        if dtype != ds.dtype:
            raise ValueError("dtype specified but doesn't match already existing dtype")

    if compression and compression != ds.compression or compression_opts and compression_opts != ds.compression_opts:
        raise ValueError("Compression options can only be specified for the first version of a dataset")
    if fillvalue is not None and fillvalue != ds.fillvalue:
        dtype = ds.dtype
        if dtype.metadata and ('vlen' in dtype.metadata or 'h5py_encoding' in dtype.metadata):
            # Variable length string dtype. The ds.fillvalue will be None in
            # this case (see create_virtual_dataset() below)
            pass
        else:
            raise ValueError(f"fillvalues do not match ({fillvalue} != {ds.fillvalue})")
    if data.dtype != ds.dtype:
        raise ValueError(f"dtypes do not match ({data.dtype} != {ds.dtype})")
    # TODO: Handle more than one dimension
    old_shape = ds.shape
    slices = {}
    slices_to_write = {}
    chunk_size = chunks[0]

    with Hashtable(f, name) as hashtable:
        if len(data.shape) != 0:
            for s in ChunkSize(chunks).indices(data.shape):
                idx = hashtable.largest_index
                data_s = data[s.raw]
                raw_slice = Slice(idx*chunk_size, idx*chunk_size + data_s.shape[0])
                data_hash = hashtable.hash(data_s)
                raw_slice2 = hashtable.setdefault(data_hash, raw_slice)
                if raw_slice2 == raw_slice:
                    slices_to_write[raw_slice] = s
                slices[s] = raw_slice2

            ds.resize((old_shape[0] + len(slices_to_write)*chunk_size,) + chunks[1:])
            for raw_slice, s in slices_to_write.items():
                # idx = raw_slice.expand(ds.shape[:1] + s.newshape(data.shape)[1:])
                data_s = data[s.raw]
                idx = Tuple(raw_slice, *[slice(0, i) for i in data_s.shape[1:]])
                ds[idx.raw] = data[s.raw]
    return slices

def write_dataset_chunks(f, name, data_dict, shape=None):
    """
    data_dict should be a dictionary mapping chunk_size index to either an
    array for that chunk, or a slice into the raw data for that chunk

    """
    if name not in f['_version_data']:
        raise NotImplementedError("Use write_dataset() if the dataset does not yet exist")

    ds = f['_version_data'][name]['raw_data']
    chunks = tuple(ds.attrs['chunks'])
    chunk_size = chunks[0]

    if shape is None:
        shape = tuple(max(c.args[i].stop for c in data_dict) for i in
                      range(len(chunks)))
    # all_chunks = list(ChunkSize(chunks).indices(shape))
    # for c in data_dict:
    #     if c not in all_chunks:
    #         raise ValueError(f"data_dict contains extra chunks ({c})")

    with Hashtable(f, name) as hashtable:
        slices = {i: None for i in data_dict}
        data_to_write = {}
        for chunk, data_s in data_dict.items():
            if not isinstance(data_s, (slice, tuple, Tuple, Slice)) and data_s.dtype != ds.dtype:
                raise ValueError(f"dtypes do not match ({data_s.dtype} != {ds.dtype})")

            idx = hashtable.largest_index
            if isinstance(data_s, (slice, tuple, Tuple, Slice)):
                slices[chunk] = ndindex(data_s)
            else:
                raw_slice = Slice(idx*chunk_size, idx*chunk_size + data_s.shape[0])
                data_hash = hashtable.hash(data_s)
                raw_slice2 = hashtable.setdefault(data_hash, raw_slice)
                if raw_slice2 == raw_slice:
                    data_to_write[raw_slice] = data_s
                slices[chunk] = raw_slice2

    assert None not in slices.values()
    old_shape = ds.shape
    ds.resize((old_shape[0] + len(data_to_write)*chunk_size,) + chunks[1:])
    for raw_slice, data_s in data_to_write.items():
        c = (raw_slice.raw,) + tuple(slice(0, i) for i in data_s.shape[1:])
        ds[c] = data_s
    return slices

def create_virtual_dataset(f, version_name, name, shape, slices, attrs=None, fillvalue=None):
    from h5py._hl.selections import select
    from h5py._hl.vds import VDSmap

    raw_data = f['_version_data'][name]['raw_data']
    raw_data_shape = raw_data.shape
    slices = {c: s.reduce() for c, s in slices.items()}

    if len(raw_data) == 0:
        shape = ()
        layout = VirtualLayout((1,), dtype=raw_data.dtype)
        vs = VirtualSource('.', name=raw_data.name, shape=(1,), dtype=raw_data.dtype)
        layout[0] = vs[()]
    else:
        # Chunks in the raw dataset are expanded along the first dimension only.
        # Since the chunks are pointed to by virtual datasets, it doesn't make
        # sense to expand the chunks in the raw dataset along multiple dimensions
        # (the true layout of the chunks in the raw dataset is irrelevant).
        for c, s in slices.items():
            if len(c.args[0]) != len(s):
                raise ValueError(f"Inconsistent slices dictionary ({c.args[0]}, {s})")

        # h5py 3.3 changed the VirtualLayout code so that it no longer uses
        # sources. See https://github.com/h5py/h5py/pull/1905.
        layout = VirtualLayout(shape, dtype=raw_data.dtype)
        layout_has_sources = hasattr(layout, 'sources')
        if not layout_has_sources:
            from h5py import _selector
            layout._src_filenames.add(b'.')
            space = h5s.create_simple(shape)
            selector = _selector.Selector(space)

        for c, s in slices.items():
            if c.isempty():
                continue
            # idx = Tuple(s, *Tuple(*[slice(0, i) for i in shape[1:]]).as_subindex(Tuple(*c.args[1:])).args)
            S = [Slice(0, len(c.args[i])) for i in range(1, len(shape))]
            idx = Tuple(s, *S)
            # assert c.newshape(shape) == vs[idx.raw].shape, (c, shape, s)

            # This is equivalent to
            #
            # layout[c.raw] = vs[idx.raw]
            #
            # but faster because vs[idx.raw] does a deepcopy(vs), which is
            # slow. We need different versions for h5py 2 and 3 because the
            # virtual sources code was rewritten.
            if not layout_has_sources:
                key = idx.raw
                vs_sel = select(raw_data.shape, key, dataset=None)

                sel = selector.make_selection(c.raw)
                layout.dcpl.set_virtual(
                    sel.id, b'.', raw_data.name.encode('utf-8'), vs_sel.id
                )

            else:
                vs_sel = select(raw_data_shape, idx.raw, None)
                layout_sel = select(shape, c.raw, None)
                layout.sources.append(VDSmap(layout_sel.id,
                                       '.',
                                       raw_data.name,
                                       vs_sel.id))

    dtype = raw_data.dtype
    if dtype.metadata and ('vlen' in dtype.metadata or 'h5py_encoding' in dtype.metadata):
        # Variable length string dtype
        # (https://h5py.readthedocs.io/en/2.10.0/strings.html). Setting the
        # fillvalue in this case doesn't work
        # (https://github.com/h5py/h5py/issues/941).
        if fillvalue not in [0, '', b'', None]:
            raise ValueError("Non-default fillvalue not supported for variable length strings")
        fillvalue = None

    virtual_data = f['_version_data/versions'][version_name].create_virtual_dataset(name, layout, fillvalue=fillvalue)

    if attrs:
        for k, v in attrs.items():
            virtual_data.attrs[k] = v
    virtual_data.attrs['raw_data'] = raw_data.name
    virtual_data.attrs['chunks'] = raw_data.chunks
    return virtual_data
