from uuid import uuid4

from .backend import write_dataset, write_dataset_chunks, create_virtual_dataset

# TODO: Allow version_name to be a version group
def create_version(f, version_name, datasets, prev_version=None, *,
                   make_current=True):
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

    group = versions.create_group(version_name)
    group.attrs['prev_version'] = prev_version
    if make_current:
        versions.attrs['current_version'] = version_name

    for name, data in datasets.items():
        if isinstance(data, dict):
            slices = write_dataset_chunks(f, name, data)
        else:
            slices = write_dataset(f, name, data)
        create_virtual_dataset(f, version_name, name, slices)

    return group

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
