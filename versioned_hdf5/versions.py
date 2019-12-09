CHUNK_SIZE = 2**20


def get_chunks(shape):
    # TODO: Implement this
    if len(shape) > 1:
        raise NotImplementedError
    return (CHUNK_SIZE,)

def initialize(f):
    f.create_group('_version_data')

def create_base_dataset(f, name, *, shape=None, data=None):
    if data is not None and shape is not None:
        raise ValueError("Only one of data or shape should be passed")
    if shape is None:
        shape = data.shape
    ds = f['/_version_data'].create_dataset(name, shape=shape, data=data,
                                            chunks=get_chunks(shape),
                                            maxshape=(None,)*len(shape))

    return ds

def write_dataset(f, name, data):
    if name not in f['/_version_data']:
        create_base_dataset(f, name, data=data)
        return
