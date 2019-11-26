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
producing an index into the final dataset.

<!-- Copy on write -->

<!-- Chunk based -->

<!-- Chunk storage dataset vs. Reference storage -->

Every dataset in the system will be part of a snapshot or "version". In order
for a modification to be saved, it must be added to a new version. Versions
will be ordered as a directed acyclic graph, where each version references a
previous version that it is based on (similar to commits in git, except
without merge commits). The ordering of versions is not necessary to read the
data for a given version, since each version metadata will point to a dataset
object which itself references its corresponding chunks. However, the ordering
of versions is useful for other operations, such as listing all known
versions.

## Open questions

- How can we enforce immutability on the underlying data? One way would be to
  require that any interaction with the data goes through the library, but
  this means that the library must expose a sufficient API to do anything the
  end-user would want to do.
