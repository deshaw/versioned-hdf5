from uuid import uuid4
from collections import defaultdict

from .backend import write_dataset, write_dataset_chunks, create_virtual_dataset
from .wrappers import InMemoryGroup, InMemoryDataset

def create_version_group(f, version_name, prev_version=None):
    versions = f['_version_data/versions']

    if prev_version == '':
        prev_version = '__first_version__'
    elif prev_version is None:
        prev_version = versions.attrs['current_version']

    if version_name is None:
        version_name = str(uuid4())

    if version_name in versions:
        raise ValueError(f"There is already a version with the name {version_name}")
    if prev_version not in versions:
        raise ValueError(f"Previous version {prev_version!r} not found")

    group = InMemoryGroup(versions.create_group(version_name).id)
    group.attrs['prev_version'] = prev_version
    group.attrs['committed'] = False

    # Copy everything over from the previous version
    prev_group = versions[prev_version]

    def _get(name, item):
        group[name] = item

    prev_group.visititems(_get)
    return group

def commit_version(version_group, datasets, *,
                   make_current=True, chunk_size=None,
                   compression=None, compression_opts=None):
    """
    Create a new version

    prev_version should be a pre-existing version name, None, or ''
    If it is None, it defaults to the current version. If it is '', it creates
    a version with no parent version.

    datasets should be a dictionary mapping {path: dataset}, where `dataset`
    is either a numpy array, or a dictionary mapping {chunk_index:
    data_or_slice}, where `data_or_slice` is either an array or a slice
    pointing into the raw data for that chunk.

    If make_current is True, the new version will be set as the current version.

    Returns the group for the new version.
    """
    if 'committed' not in version_group.attrs:
        raise ValueError("version_group must be a group created by create_version_group()")
    if version_group.attrs['committed']:
        raise ValueError("This version group has already been committed")
    #f = version_group.file
    version_name = version_group.name.rsplit('/', 1)[1]
    #versions = f['_version_data/versions']
    versions = version_group.parent
    f = versions.parent.parent

    chunk_size = chunk_size or defaultdict(type(None))
    compression = compression or defaultdict(type(None))
    compression_opts = compression_opts or defaultdict(type(None))

    if make_current:
        old_current = versions.attrs['current_version']
        versions.attrs['current_version'] = version_name

    try:
        for name, data in datasets.items():
            
            #print(f"name={name}, data={data}")

            if isinstance(data, InMemoryDataset):
                data = data.id.data_dict
            if isinstance(data, dict):
                if chunk_size[name] is not None:
                    raise NotImplementedError("Specifying chunk size with dict data")
                slices = write_dataset_chunks(f, name, data)
            else:
                slices = write_dataset(f, name, data,
                                       chunk_size=chunk_size[name], compression=compression[name],
                                       compression_opts=compression_opts[name])
            create_virtual_dataset(f, version_name, name, slices)
        version_group.attrs['committed'] = True
    except Exception:
        del versions[version_name]
        if make_current:
            versions.attrs['current_version'] = old_current
        raise

def get_nth_previous_version(f, version_name, n):
    versions = f['_version_data/versions']
    if version_name not in versions:
        raise IndexError(f"Version {version_name!r} not found")

    version = version_name
    for i in range(n):
        version = versions[version].attrs['prev_version']

        # __first_version__ is a meta-version and should not be returnable
        if version == '__first_version__':
            raise IndexError(f"{version_name!r} has fewer than {n} versions before it")

    return version

def set_current_version(f, version_name):
    versions = f['_version_data/versions']
    if version_name not in versions:
        raise ValueError(f"Version {version_name!r} not found")

    versions.attrs['current_version'] = version_name

def all_versions(f, *, include_first=False):
    """
    Return a generator that iterates all versions by name

    If include_first is True, it will include '__first_version__'.

    Note that the order of the versions is completely arbitrary.
    """
    versions = f['_version_data/versions']
    for version in versions:
        if version == '__first_version__':
            if include_first:
                yield version
        else:
            yield version
