# Design of the versioned HDF5

The basic idea will be a copy-on-write system, inspired by git as well as modern
filesystems such as APFS and Btrfs. Copy-on-write is a good fit whenever data
should be completely immutable. In a copy-on-write system, any modifications
to a piece of data produce a new copy of the data, leaving the original
intact. Any references to the original will continue to point to it. The new
data may be a full physical copy, or be implemented as a shallow copy that
references the common parts of the old data. Since all objects are immutable,
this is an implementation detail that does not affect the high-level usage of
the system. Thus, the implementation is free to do this or not do this
depending on the efficiency tradeoffs (as a general rule, sharing data saves
space but increases the time needed to access the data).

The fundamental primitive of the system will be HDF5 chunks. In HDF5, a
dataset is split into multiple chunks. Each chunk is of equal size, which is
configurable, although some chunks may not be completely full.

When a new version of a dataset is produced, which is a modification of a
dataset from an existing version, all chunks that are modified are copied.
This will be implemented using HDF5 virtual datasets. HDF5 virtual datasets
can reference arbitrary subsets of a dataset, however, for our implementation,
we will only reference full chunks. This will greatly simplify the process of
producing an index into the final dataset. This is analogous to a
copy-on-write filesystem that works at the block level. The most appropriate
chunk size can be chosen to fit the specific application.

Chunk data will be hashed and stored in a hash table. This will enable two
chunks that have identical data to only be stored once in the virtual
filesystem, and referenced multiple times. This can also be used to verify
data integrity and partially enforce immutability (see below).

Every dataset in the system will be part of a snapshot or "version". In order
for a modification to be saved, it must be added to a new version. Versions
will be ordered as a directed acyclic graph, where each version references a
previous version that it is based on (similar to commits in git, except
without merge commits). The ordering of versions is not necessary to read the
data for a given version, since each version metadata will point to a dataset
object which itself references its corresponding chunks. However, the ordering
of versions is useful for other operations, such as listing all known
versions.

Since every version is a virtual dataset in HDF5, metadata for the
versions can be stored in the HDF5 metadata for the dataset. This will include
a timestamp and an optional name.

## Example API Code

The HDF5 virtual dataset and chunk primitives will be invisible to the
end-user. The end user will access a particular version of an array using
getitem on a `version` attribute.

```py
with versioned_hdf5.open('data.hdf5', 'r') as data:
    data.version[-100].array # The version 100 steps ago
    data.version['name'].array # Access a version by name
    data.version[datetime.datetime(...)].array # Access a version by timestamp
    data.current_version # A shorthand for data.version[-1]
```

This will return a numpy array. To create a new version, a new version of the
array will be passed to a function

```py
with versioned_hdf5.open('data.hdf5', '+') as data:
    data.current_version.new_version(new_array,
                                     name='name', # optional
                                     )
```

The library will then automatically take care of chunking the array and
storing only the new chunks. This is achieved by checking for existing chunks
in the hash table, and only storing them if they are not already there.

As an alternate API, the array can be mutated in-place under a context
manager, which would create a version on exit. This would work similar to
"staging" in git.

```py
with data.current_version.stage_version(name='name') as array:
    # array is mutated here
    array[100] = ...
# On context manager exit, array is saved as a new version of data
```

## Open questions

- How can we enforce immutability on the underlying data? One way would be to
  require that any interaction with the data goes through the library, but
  this means that the library must expose a sufficient API to do anything the
  end-user would want to do.

- Since versions can be branching, do we need to support any kind of notion of
  "merge" versions, which would be versions that have more than one parent
  version? If so, what sort of tooling is required around this? Multiple
  parent versions can also complicate version indexing syntax and logic.

- Storing each duplicated chunk can be inefficient if chunks are mostly
  similar. Is it sufficient to manage this by being careful about selecting
  the chunk size for the target application? Can we manage this via HDF5's
  built-in compression? Will it be necessary to build an additional
  abstraction where chunks can be referenced as shallow diffs of each other,
  similar to git's pack abstraction.

## Relevant References

- Virtual Datasets
  - h5py documentation: https://h5py.readthedocs.io/en/stable/vds.html
  - HD5F specification: https://support.hdfgroup.org/HDF5/docNewFeatures/NewFeaturesVirtualDatasetDocs.html

- Git implementation details:
  - The "Git internals" chapter of the Git book (describes basic details of git objects)
    https://git-scm.com/book/en/v2/Git-Internals-Git-Objects
  - "Git internals" by John Britton (talk version of the same information)
    https://www.youtube.com/watch?v=lG90LZotrpo
  - "Unpacking git packfiles" by Aditya Mukerjee
    https://codewords.recurse.com/issues/three/unpacking-git-packfiles
