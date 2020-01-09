from contextlib import contextmanager
from uuid import uuid4

import numpy

from .backend import write_dataset, create_virtual_dataset

def create_version(f, version_name, prev_version, datasets):
    if version_name is None:
        version_name = str(uuid4())

    versions = f['_version_data/versions']
    if version_name in versions:
        raise ValueError(f"There is already a version with the name {version_name}")
    if prev_version and prev_version not in versions:
        raise ValueError(f"Previous version {prev_version!r} not found")

    group = versions.create_group(version_name)
    group.attrs['version_name'] = version_name
    group.attrs['prev_version'] = prev_version

    for name, data in datasets.items():
        slices = write_dataset(f, name, data)
        create_virtual_dataset(f, version_name, name, slices)

    return group
