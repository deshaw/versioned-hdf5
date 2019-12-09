from h5py import VirtualLayout, VirtualSource

CHUNK_SIZE = 2**20


def get_chunks(shape):
    # TODO: Implement this
    if len(shape) > 1:
        raise NotImplementedError
    return (CHUNK_SIZE,)

def initialize(f):
    f.create_group('_version_data/raw_data')

def create_base_dataset(f, name, *, shape=None, data=None):
    if data is not None and shape is not None:
        raise ValueError("Only one of data or shape should be passed")
    if shape is None:
        shape = data.shape
    ds = f['/_version_data/raw_data'].create_dataset(name, shape=shape, data=data,
                                            chunks=get_chunks(shape),
                                            maxshape=(None,)*len(shape))

    return ds

def write_dataset(f, name, data):
    if name not in f['/_version_data/raw_data']:
        create_base_dataset(f, name, data=data)
        return

    ds = f['/_version_data/raw_data'][name]
    # TODO: Handle more than one dimension
    old_shape = ds.shape
    ds.resize((old_shape[0] + data.shape[0],))
    ds[old_shape[0]:] = data
    return ds

def create_virtual_dataset(f, name, shape, indices):
    layout = VirtualLayout(shape)
    vs = VirtualSource(f['_version_data/raw_data'][name])

    for i, idx in enumerate(indices):
        # TODO: This needs to handle more than one dimension
        layout[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] = vs[idx*CHUNK_SIZE:(idx+1)*CHUNK_SIZE]

    virtual_data = f.create_virtual_dataset(name, layout)
    return virtual_data
