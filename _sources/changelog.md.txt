Versioned HDF5 Change Log
=========================

## 1.3.3 (2021-07-02)

## Minor Changes

- Fix a regression that prevented indexing a dataset with a mask from working.

## 1.3.2 (2021-07-28)

## Minor Changes

- Improve the performance of reading a reading a dataset that hasn't been
  written to (e.g., reading from an already committed version).

- Fix Versioned HDF5 to work with h5py 3.3.

- Fix an issue that would occur when using np.datetime64 objects for
  timestamps when the fractional part of the second was exactly 0.

## 1.3.1 (2021-05-20)

## Minor Changes

- Avoid some unnecessary precomputation in the hashtable object. This improves
  the performance for files that have many versions in them.

## 1.3 (2021-05-07)

## Major Changes

- Support h5py 3. h5py 2.10 is also still officially supported.

- Add functionality to replay versions. This allows mutating old versions.

- Add new helper functions {any}`delete_version` and {any}`modify_metadata`
  to delete a version and modify metadata on a dataset that must be the same
  across versions, such as chunk size or dtype.

- Add helper function {any}`recreate_dataset` for more advanced version
  replaying functionality.

## Minor Changes

- Disallow accessing vfile[version] before version has been committed. Doing
  so previously could lead to inconsistencies.

- Disallow accessing non-versioned groups from VersionedHDF5File.

- Better error messages from VersionedHDF5File when the underlying file is
  closed.

- Remove codecov.

- Some improvements to the benchmarking suite.

## 1.2.6 (2021-04-20)

## Minor Changes

- Fix a bug where chunks could be deleted from a dataset.

- Workaround an upstream h5py bug in the tests.

## 1.2.5 (2021-04-15)

## Minor Changes

- Fix a bug where attrs could be deleted from a dataset.

## 1.2.4 (2021-04-08)

## Major Changes

- Many improvements to performance throughout the library, particularly for
  datasets with many chunks, and for looking up versions by timestamp. This
  also sets up potential future performance improvements by automatically
  converting sparse staged datasets to fully in-memory.

- Add some additional benchmarks to the benchmark suite.

## 1.2.3 (2021-02-25)

### Minor Changes

- Fix the length of str dtype not being maintained from the fillvalue.

## 1.2.2 (2021-02-04)

### Minor Changes

- Many improvements to performance throughout the library.

- Added a benchmarking suite using [airspeed
  velocity](https://asv.readthedocs.io/en/stable/). Graphs of the benchmarks
  can be viewed at <https://deshaw.github.io/versioned-hdf5/benchmarks/>.

- Versioned HDF5 now depends on ndindex version 1.5.1 or greater.

## 1.2.1 (2020-12-30)

### Minor Changes

- Python 3.6 support has been dropped. The lowest version of Python now
  supported is 3.7.
- Fix creating a completely empty sparse dataset
- Use ndindex.ChunkSize internally. This is the beginning of an overhaul that
  improves the performance of many operations. ndindex 1.5 or newer is now
  required.

## 1.2 (2020-11-17)

### Major Changes

- Add support for sparse datasets (`data=None`).
- Store the chunks on an attribute of the dataset.
- versioned-hdf5 is currently pinned to `h5py<3`. h5py 3 support will be added
  in a future version.
- `VersionedHDF5File[timestamp]` now returns the closest version before
  `timestamp` if there is no version at `timestamp`.

## 1.1 (2020-09-15)

### Major Changes

* Added support for shape-0 datasets.
* Fix a memory leak where in-memory datasets would not be garbage collected.
* Added support for empty datasets (size 0).
* Make sure versioned data is read-only after closing and reopening the file.
* Allow deleting groups and datasets between versions. Note that currently
  dataset metadata cannot change between versions, even if they are deleted in
  between.

### Minor Changes

* Most tests now use a temporary directory instead of writing a file in the
  current directory.
* Fix logic for handling trailing slashes with `in`.
* Automatically create intermediate groups when creating a dataset.
* Make indices that should give a scalar object do so instead of giving a
  shape () array.

## 1.0 (2020-08-03)

### Major Changes

* First release of Versioned-HDF5.
