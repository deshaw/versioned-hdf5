# Design

Versioned-hdf5 is built as a wrapper on top of h5py. The basic idea behind the
design is that versioned-hdf5 is a
[copy-on-write](https://en.wikipedia.org/wiki/Copy-on-write) system, inspired
by git as well as modern filesystems such as APFS and Btrfs. Copy-on-write is
a good fit whenever data should be completely immutable. In a copy-on-write
system, any modification to a piece of data produces a new copy of the data,
leaving the original intact. Any references to the original will continue to
point to it.

This is implemented using two key HDF5 primitives: chunks and virtual
datasets.

In HDF5, datasets are split into multiple chunks. Each chunk is of equal size,
which is configurable, although some chunks may not be completely full. A
chunk is the smallest part of a dataset that HDF5 operates on. Whenever a
subset of a dataset is to be read, the entire chunk containing that dataset is
read into memory. Picking an optimal chunk size is a nontrivial task, and
depends on things such as the size of your L1 cache and the typical shape of
your dataset. Furthermore, in versioned-hdf5 a chunk is the smallest amount of
data that is stored only once across versions if it has not changed. If the
chunk size is too small, it would affect performance, as operations would
require reading and writing more chunks, but if it is too large, it would make
the resulting versioned file unnecessarily large, as changing even a single
element of a chunk requires rewriting the entire chunk. Versioned-hdf5 does
not presently contain any logic for automatically picking a chunk size. The
[pytables
documentation](https://www.pytables.org/usersguide/optimization.html) has some
tips on picking an optimal chunk size.

[Virtual datasets](http://docs.h5py.org/en/stable/vds.html) are a special kind
of dataset that reference data from other datasets in a seamless way. The data
from each part of a virtual dataset comes from another dataset. HDF5 does this
seamlessly, so that a virtual dataset appears to be a normal dataset.

The basic design of versioned-hdf5 is this: whenever a dataset is created for
the first time (the first version containing the dataset), it is split into
chunks. The data in each chunk is hashed and stored in a hash table. The
unique chunks are then appended into to a `raw_data` dataset corresponding to
the dataset. Finally, a virtual dataset is made that references the
corresponding chunks in the raw dataset to recreate the original dataset. When
later versions modify this dataset, each modified chunk is appended to the raw
dataset, and a new virtual dataset is created pointing to corresponding
chunks.

For example, say we start with the first version, `version_1`, and create a
dataset `my_dataset` with `n` chunks. The dataset chunks will be written into the
raw dataset, and the final virtual dataset will point to those chunks.

```{graphviz}
digraph g {
graph [
rankdir = "LR"
];
node [
fontsize = "16"
];
edge [
];
"dataset (version_1)" [
label = "my_dataset (version_1)|<f0>CHUNK 0|<f1>CHUNK 1|<fdot>...|<fn>CHUNK n"
shape = "record"
];
"raw_data" [
label = "raw_data|<f0>CHUNK 0|<f1>CHUNK 1|<fdot>...|<fn>CHUNK n"
shape = "record"
];
"dataset (version_1)":f0 -> "raw_data":f0 [];
"dataset (version_1)":f1 -> "raw_data":f1 [];
"dataset (version_1)":fdot -> "raw_data":fdot [];
"dataset (version_1)":fn -> "raw_data":fn [];
}
```

If we then create a version `version_2` based off `version_1`, and modify only
data contained in CHUNK 1, that new data will be appended to the raw dataset,
and the resulting virtual dataset for `version_2` will look like this:

```{graphviz}
digraph g {
graph [
rankdir = "LR"
];
node [
fontsize = "16"
];
edge [
];
"dataset (version_1)" [
label = "my_dataset (version_1)|<f0>CHUNK 0|<f1>CHUNK 1|<fdot>...|<fn>CHUNK n"
shape = "record"
];
"dataset (version_2)" [
label = "my_dataset (version_2)|<f0>CHUNK 0|<f1>CHUNK 1|<fdot>...|<fn>CHUNK n"
shape = "record"
];
"raw_data" [
label = "raw_data|<f0>CHUNK 0|<f1>CHUNK 1|<fdot>...|<fn>CHUNK n|<fn1>CHUNK n+1"
shape = "record"
];
"dataset (version_1)":f0 -> "raw_data":f0 [];
"dataset (version_1)":f1 -> "raw_data":f1 [];
"dataset (version_1)":fdot -> "raw_data":fdot [];
"dataset (version_1)":fn -> "raw_data":fn [];
"raw_data":f0 -> "dataset (version_2)":f0 [dir=back];
"dataset (version_2)":fdot -> "raw_data":fdot [];
"dataset (version_2)":f1 -> "raw_data":fn1 [];
"dataset (version_2)":fn -> "raw_data":fn [];
}
```

Since both versions 1 and 2 of `my_dataset` have identical data in chunks other than
CHUNK 1, they both point to the exact same data in `raw_data`. Thus, the
underlying HDF5 file only stores the data in version 1 of `my_dataset` once, and
only the modified chunks from `version_2`'s `my_dataset` are stored on top of that.

All extra metadata, such as attributes, is stored on the virtual dataset.
Since virtual datasets act exactly like real datasets and operate at the HDF5
level, each version is a real group in the HDF5 file that is exactly that
version. However, these groups should be treated as read-only, and you should
never access them outside of the versioned-hdf5 API (see below).

## HDF5 File Layout

Inside of the HDF5 file, there is a special `_versioned_data` group that holds
all the internal data for versioned-hdf5. This group contains a `versions`
group, which contains groups for each version that has been created. It also
contains a group for each dataset that exists in a version. These groups each
contain two datasets, `hash_table`, and `raw_data`.

For example, consider a versioned-hdf5 file that contains two versions,
`version1`, and `version2`, with datasets `data1` and `data2`. Suppose also
that `data1` exists in both versions and `data2` only exists in `version2`.
The HDF5 layout would look like this

```
/_versioned_data/
├── data1/
│   ├── hash_table
│   └── raw_data
│
├── data2/
│   ├── hash_table
│   └── raw_data
│
└── versions/
    ├── __first_version__/
    │
    ├── version1/
    │   └── data1
    │
    └── version2/
        ├── data1
        └── data2
```

`__first_version__` is an empty group that exists only for internal
bookkeeping purposes (see below).

## Submodule Organization

The versioned-hdf5 code is split into four layers, the backend, the versions,
the h5py wrappers, and the top-level API.

### Backend

The backend layer is the bottommost layer. It is the only layer that does
actual dataset writes to HDF5. It deals with the splitting of chunks from the
versioned dataset and creation of the virtual datasets that compromise the
version groups. The relevant modules are `versioned_hdf5.backend` and
`versioned_hdf5.hashtable`.

`versioned_hdf5.backend.write_dataset()` takes a dataset (or array) and writes
it to the raw data for the given dataset. The data in each chunk of the
dataset is SHA256 hashed, and the hash is looked up in the hashtable dataset.
If it already exists in the raw data, that chunk in the raw data is reused.

To enable a check that the reused chunk matches the data that the user is
intending to write, set the following environment variable:
`ENABLE_CHUNK_REUSE_VALIDATION = 1`. This option is enabled during tests, but
is disabled by default for better performance.

The hashtable maps `SHA256 hash -> (start, stop)` where `(start, stop)` gives
a slice range for the chunk in the raw dataset (chunks in the `raw_data`
dataset are concatenated along the first axis only). All chunks that do not
exist in the hashtable already are appended to the raw dataset and added to
the hashtable. `versioned_hdf5.backend.write_dataset_chunks()` works
similarly, except instead of taking a dataset as input, it takes an dictionary
mapping chunks. This allows the higher levels of the API to only pass in the
chunks of an existing dataset that have been modified (see below).

`versioned_hdf5.backend.create_virtual_dataset()` creates a virtual dataset
in the version group pointing to corresponding chunks in the raw dataset.
`versioned_hdf5.backend` also has various functions for initializing a
dataset the first time it is created in a version.

`versioned_hdf5.hashtable` contains a `Hashtable` object that wraps the
hashtable dataset in HDF5 as a dict-like object.

### Versions

Each version is stored as a subgroup of the `_versioned_data/versions/` group.
The group contains attributes that reference the previous version, as well as
metadata like the timestamp when the version was created. Consequently, the
versions form a DAG. However, the reference to the previous version is only
used by the top-level API that allows traversing versions. Each version group
is self-contained, containing only virtual datasets that point only to the
respective raw datasets.

Versioned-hdf5 also keeps track of the "current version", which is used only
to allow previous version to not be specified when creating a new version
(this information is stored on the attributes of the
`_versioned_data/versions` group). If a version does not have a previous
version, its previous version is the special empty `__first_version__` version
group.

`versioned_hdf5.versions` contains functions to create a version group, commit
a version, and access and manipulate versions. The main function here is
`versioned_hdf5.versions.commit_version()`, which is called with all the
datasets that should be committed to the new version when the
`VersionedHDF5File.stage_version()` context manager exits.

### h5py Wrappers

One minor issue with the copy-on-write idea is that HDF5 does not have a native
way to make virtual datasets read-only. If you modify a virtual dataset, it
will also modify the dataset that it points to. In our design, this would
modify all other versions of a dataset pointing to the same raw data chunks.

Hence, versioned-hdf5 provides wrappers to the various h5py objects that
implement the proper copy-on-write semantics. Versioned HDF5 files should only
be interacted with via the versioned-hdf5 library. Writing to a versioned
dataset directly with h5py or another HDF5 wrapper library may lead to data
corruption, as common data is shared between versions.

The objects for this layer all live in `versioned_hdf5.wrappers`. The primary
objects are

`InMemoryGroup`: This is the object returned by the
`VersionedHDF5File.stage_version()` context manager. It acts like an
`h5py.Group` object, but all data is stored in memory. This is done efficiently
so that only data that is modified is actually read in from the file. This
object is also used for any subgroups of the version group. The primary
purpose of this object is to keep track of what has been modified while a
version is being staged. Once the `stage_version()` context manager exits,
this object is passed to `commit_version()` (see above), which extracts the
relevant information about what datasets exist in the new version and how they
relate to previous versions, if there are any.

`InMemoryArrayDataset`: This objects acts like a `h5py.Dataset`, but wraps a
NumPy array in memory. This object is used whenever a dataset is created for
the first time.

`InMemoryDataset`: This objects acts like a `h5py.Dataset`. It is used
whenever a dataset in a version already exists from a previous version. This
object stores only those chunks of the dataset in memory that are actually
read in or modified. This is not only more memory efficient, but it allows
passing only the modified chunks as arrays to the backend. The remaining
chunks will then automatically point to the chunks in the raw data that they
pointed to in the previous version, without needing to re-hash the data.

One challenge with this design is that `InMemoryDataset` represents a single
dataset that is broken up into chunks, which live in the raw dataset and may
not be contiguous. The
[ndindex](https://quansight-labs.github.io/ndindex/) library is used to
manage translation of indices on the dataset to and from the chunked data.
ndindex is also used throughout versioned-hdf5 to store and manipulate slice
and other index objects, as it is more convenient than using the raw index
types. For example, in the backend, we need to store slices in a dictionary.
The default Python `slice` object is not hashable until Python 3.12, which
makes this annoying to do. The ndindex index objects are all hashable.
The ndindex library was initially created for versioned-hdf5, in order to make
index manipulation possible as well as allowing code that passes indices around
to become much cleaner.

`InMemoryDataset` is implemented as a fairly thin wrapper around
`StagedChangesArray`, which is abstracted from h5py and holds all the modified
chunks in memory, presenting them as a numpy-like array, until they are ready to
be flushed to disk. For details, read [Staging changes in memory](staged_changes).

These wrapper objects all try to emulate the h5py API as closely as possible,
so that the user can use them just as they would the real h5py objects. Any
discrepancy between h5py and versioned-hdf5 semantics should be considered a
bug in versioned-hdf5.

### Top-level API

The top-level API consists of one object, {any}`VersionedHDF5File`. This object
allows accessing versions via getitem, like
`VersionedHDF5File(f)[version_name]`. The primary use of this object, however,
is the `stage_version()` method, which is a context manager that returns a group
for a new version. The way to make a new version is

```py
import h5py
from versioned_hdf5 import VersionedHDF5File

f = h5py.File(...)
file = VersionedHDF5File(f)

# new_version and prev_version are strings corresponding the the version names
# for the new and previous versions
with file.stage_version(new_version, prev_version) as g:
    g['dataset'][0] = 1 # Modify a dataset from prev_version
    g['dataset'].resize(...) # Resize a dataset from prev_version
    g.create_dataset('dataset2', ...) # Create a new dataset
    g.create_group('new_group') # Create a new subgroup
```

Inside of the context manager, the group `g` will look exactly like the
previous version `prev_version`, but modifications to it will not actually
modify `prev_version`. Rather, they will stage changes for the new version
`new_version`. When the context manager exits, whatever the state of the
version group `g` is will be written as `new_version`. Any data chunks from
`prev_version` that were not modified will be reused as described above.

Once a version is committed (after the context manager exits), it should be
treated as read-only. The versioned-hdf5 objects have some safeguards to
prevent accidentally writing to existing versioned data, but the underlying
h5py has no such safeguards, since there are no notions of read-only datasets
in HDF5 itself, so these safeguards should not be relied on.
