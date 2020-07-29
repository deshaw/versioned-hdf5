# Design

Versioned-hdf5 is built as a wrapper on top of h5py. The basic idea behind the
design is that versioned-hdf5 is a copy-on-write system, inspired by git as
well as modern filesystems such as APFS and Btrfs. Copy-on-write is a good fit
whenever data should be completely immutable. In a copy-on-write system, any
modifications to a piece of data produce a new copy of the data, leaving the
original intact. Any references to the original will continue to point to it.

This is implemented using two key HDF5 primitives: chunks and virtual
datasets.

In HDF5, datasets are split into multiple chunks. Each chunk is of equal size,
which is configurable, although some chunks may not be completely full. A
chunk is the smallest part of a dataset that HDF5 operates on. Whenever a
subset of a dataset is to be read, the entire chunk containing that dataset is
read into memory. Picking an optimal chunk size is a nontrivial task, and
depends on things such as the size of your L1 cache, and the typical shape of
your dataset. Furthermore, in versioned-hdf5 a chunk is the smallest amount of
data that is stored only once across versions if it has not changed. If the
chunk size is too small, it could affect performance, as all operations must
operate on chunks, but if it is too large, it could make the resulting
versioned file large, as even a change of a single element requires rewriting
the entire chunk. Versioned-hdf5 does not presently contain any logic for
automatically picking a chunk size. The [pytables
documentation](https://www.pytables.org/usersguide/optimization.html) has some
tips on picking an optimal chunk size.

Virtual datasets are a special kind of dataset that reference data from other
datasets in a seamless way. The data from each part of a virtual dataset comes
from another dataset. HDF5 does this seamlessly, so that a virtual dataset
appears to be a normal dataset.

The basic design of versioned-hdf5 is this. Whenever a dataset is created for
the first time (the first version containing the dataset), it is split into
chunks. The data in each chunk is hashed and stored in a hash table. The
unique chunks are then appended into to a single raw_data dataset. The raw
dataset stores each chunk along the first dimension only. Finally, a virtual
dataset is made that references the corresponding chunks in the raw dataset to
recreate the original dataset. When later versions modify this dataset, each
modified chunk is appended to the raw dataset, and a new virtual dataset is
created pointing to corresponding chunks.

For example, say we start with the first version, `version 1`, and create a
dataset `dataset` with `n` chunks. The dataset chunks will be written into the
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
"dataset (version 1)" [
label = "data (version 1)|<f0>CHUNK 0|<f1>CHUNK 1|<fdot>...|<fn>CHUNK n"
shape = "record"
];
"raw_data" [
label = "raw_data|<f0>CHUNK 0|<f1>CHUNK 1|<fdot>...|<fn>CHUNK n"
shape = "record"
];
"dataset (version 1)":f0 -> "raw_data":f0 [];
"dataset (version 1)":f1 -> "raw_data":f1 [];
"dataset (version 1)":fdot -> "raw_data":fdot [];
"dataset (version 1)":fn -> "raw_data":fn [];
}
```

If we then create a version `version 2` based off `version 1`, and modify only
data contained in chunk 2, that new data will be appended to the raw dataset,
and the resulting virtual dataset for version 2 will look like this:

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
"dataset (version 1)" [
label = "data (version 1)|<f0>CHUNK 0|<f1>CHUNK 1|<fdot>...|<fn>CHUNK n"
shape = "record"
];
"dataset (version 2)" [
label = "data (version 2)|<f0>CHUNK 0|<f1>CHUNK 1|<fdot>...|<fn>CHUNK n"
shape = "record"
];
"raw_data" [
label = "raw_data|<f0>CHUNK 0|<f1>CHUNK 1|<fdot>...|<fn>CHUNK n|<fn1>CHUNK n+1"
shape = "record"
];
"dataset (version 1)":f0 -> "raw_data":f0 [];
"dataset (version 1)":f1 -> "raw_data":f1 [];
"dataset (version 1)":fdot -> "raw_data":fdot [];
"dataset (version 1)":fn -> "raw_data":fn [];
"raw_data":f0 -> "dataset (version 2)":f0 [dir=back];
"dataset (version 2)":fdot -> "raw_data":fdot [];
"dataset (version 2)":f1 -> "raw_data":fn1 [];
"dataset (version 2)":fn -> "raw_data":fn [];
}
```

Since both versions 1 and 2 of `dataset` have identical data in chunks other
than chunk 2, they both point to the exact same data in the raw data. Thus,
the underlying HDF5 file only stores the data in version 1 of `dataset` once,
and only the modified chunk in version 2 is stored on top of that.

## HDF5 File Layout

Inside of the HDF5 file, there is
a special `_versioned_data` group that holds all the internal data for
versioned-hdf5. This group contains a `versions` group, which contains groups
for each version that has been created. It also contains a group for each
dataset that exists in a version. These groups each contain two groups,
`hash_table`, and `raw_data`. For example, consider a versioned-hdf5 file that
contains two versions, `version1`, and `version2`, with datasets `data1` and
`data2`. Suppose also that `data1` exists in both versions and `data2` only
exists in `version2`. The HDF5 layout would look like this

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
    ├── version1/
    │   └── data1
    │
    └── version2/
        ├── data1
        └── data2
```

## Submodule Organization

The versioned-hdf5 code is split into three layers.

### Backend

The backend layer is the bottom most layer. It is the only layer that does
actual writes to HDF5. It deals with the splitting of chunks from the
versioned dataset, and creation of the virtual datasets that compromise the
version groups. The relevant modules are `versioned_hdf5.backend`,
`versioned_hdf5.hashtable`, and `versioned_hdf5.versions`.

`versioned_hdf5.backend.write_dataset` takes a dataset (or array) and
writes it to the raw data for the given dataset. The data in each chunk of the
dataset is SHA256 hashed, and the hash is looked up in the hashtable dataset
and if it already exists in the raw data, that chunk in the raw data is
reused. The hashtable maps `SHA256 hash -> (start, stop)` where `(start,
stop)` gives a slice range for the chunk in the raw dataset. All chunks that
do not exist in the hashtable already are appended to the raw dataset and
added to the hashtable. `versioned_hdf5.backend.write_dataset_chunks` works
similarly, except instead of taking a dataset as input, it takes an dictionary
mapping chunks. This allows the higher levels of the API to only pass in the
chunks of an existing dataset that have been modified.

`versioned_hdf5.backend.create_virtual_dataset` creates a virtual dataset
in the version group pointing to corresponding chunks in the raw dataset.
`versioned_hdf5.backend` also has various functions for initializing a
dataset the first time it is created in a version.

`versioned_hdf5.hashtable` contains a `Hashtable` object that wraps the
hashtable dataset in HDF5 as a dict-like object.

`versioned_hdf5.versions` contains functions to create a version group, commit
a version, and access and manipulate versions. The main function here is
`versioned_hdf5.versions.commit_version`, which is called when the
`VersionedHDF5File.stage_version` context manager exits with all the datasets
that should be committed to the new version.

### h5py Wrappers

One minor issue with the copy-on-write idea is that HDF5 does have a native
way to make virtual datasets read-only. If you modify a virtual dataset, it
will also modify the dataset that it points to. In our design, this would
modify all other versions of a dataset pointing to the same raw data chunks.

Hence, versioned-hdf5 provides wrappers to the various h5py objects that
implement the proper copy-on-write semantics. Versioned HDF5 files should only
be interacted with via the versioned-hdf5 library. Writing to a versioned
dataset directly with h5py or another HDF5 wrapper library may lead to data
corruption, as common data is shared between versions.

The objects for this layer all live in `versioned_hdf5.wrappers`.

`InMemoryGroup`: This is the object returned by the
`VersionedHDF5File.stage_version` context manager. It acts like an
`h5py.Group` object, but all data is stored in memory. This is done efficiency
so that only data that is modified is actually read in from the file. This
object is also used for any subgroups that are used in the versioned file. The
primary purpose of this object is to keep track of what has been modified
while a version is being staged. Once the `stage_version` context manager
exits, this object is passed to `commit_version` (see above), which extracts
the relevant information about what datasets exist in the new version and how
they relate to previous versions, if there are any.

`InMemoryArrayDataset`: This objects acts like a `h5py.Dataset`, but wraps a
NumPy array in memory. This object is used whenever a dataset is created for
the first time.

`InMemoryDataset`: This objects acts like a `h5py.Dataset`. This object is
used whenever a dataset in a version already exists from a previous version.
This object stores only those chunks of the dataset in memory that are
actually read in or modified. This is not only more memory efficient, but it
allows passing only the modified chunks as arrays to the backend. The
remaining chunks will then automatically point to the chunks in the raw data
that they pointed to in the previous version, without needing to re-hash the
data. One challenge with this design is that InMemoryDataset represents a
single dataset that is broken up into chunks, which live in the raw dataset
and may not be contiguous. The
[ndindex](https://quansight.github.io/ndindex/index.html) library is used to
manage translation of indices on the dataset to and from the chunked data.
ndindex is also used throughout versioned_hdf5 to store and manipulate slice
and other index objects, as it is more convenient than using the raw index
types. For example, in the backend, we need to store slices in a dictionary.
The default Python `slice` object is not hashable, which makes this annoying
to do. The ndindex index objects are all hashable. The ndindex library was
created in order to make index manipulation in versioned-hdf5 possible, as
well as allowing code that passes indices around to become much cleaner.

These wrapper objects all try to emulate the h5py API as closely as possible,
so that the user can use them just as they would the real h5py objects. Any
discrepancy between h5py and VersionedHDF5 semantics should be considered a
bug in versioned-hdf5.

### Top-level API

The top-level API consists of one object, `VersionedHDF5File`. This object
allows accessing versions via getitem, like
`VersionedHDF5File(f)[version_name]`. The primary use of this object, however,
is the `stage_version` method, which is a context manager that returns a group
for a new version. The way to make a new version is

```py
f = h5py.File(...)
file = VersionedHDF5File(f)
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
`prev_version` that was not modified will be reused as described above.

Once a version is committed (after the context manager exits), it should be
treated as read-only. The versioned-hdf5 objects have some safeguards to
prevent accidentally writing to existing versioned data, but the underlying
h5py has no such safeguards, since there are no notions of read-only datasets
in HDF5 itself, so these safeguards should not be relied on.
