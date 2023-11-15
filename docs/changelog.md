Versioned HDF5 Change Log
=========================

## 1.6.0 (2023-11-15)

### Major Changes

- String lengths are now included with variable-length string array hashes. This
  solves an issue with hashing uniqueness with string arrays.

### Minor Changes

- Fixed an error message directly passing a file name to VersionedHDF5File.
- Removed all posixpath "pp" aliases, to prevent conflict with prettyprint debug
  command.
- Fix deprecated CI parameter.

## 1.5.0 (2023-10-19)

### Major Changes

- Fixed an issue with rebuilding hashtables; new `DATA_VERSION` released

### Minor Changes

- Rewrote a test for improved maintainability

## 1.4.3 (2023-10-05)

### Minor Changes

- Fix to avoid writing to temporary dataset during `delete_versions`, reducing
  the chance for out-of-memory errors.
- Fix to `_rebuild_hashtables` to allow for rebuilding hashtables of datasets
  nested arbitrarily deeply inside groups.
- A sensible default chunk size is now guessed by default if none is specified
  when creating 1D datasets.

## 1.4.2 (2023-09-13)

### Minor Changes

- Fix to prevent a quadratic runtime in `delete_versions`

## 1.4.1 (2023-09-13)

### Minor Changes

- Fix an issue where calling `delete_versions` on a dataset with a variable
  length string dtype before using `h5repack` on it would corrupt the file

## 1.4.0 (2023-09-11)

### Major Changes

- Fix an issue where hashes of object dtypes were not being correctly computed
  ([#257](https://github.com/deshaw/versioned-hdf5/pull/257)). This change
  modifies how the hashes are computed for object dtypes.

### Minor Changes

- Remove old rever-based version release system in favor of GitHub workflows.

## 1.3.14 (2023-07-27)

### Major Changes

- Fix issues with the latest version of h5py.

### Minor Changes

- Fix an issue where empty datasets deleted with `delete_versions()` could
  result in corrupted versions.

- Add info log showing stats about data upon exiting stage_version context.

- Drop support for Python 3.7.

- Remove all Travis CI configuration.

- Various minor cleanups to the test suite.

## 1.3.13 (2023-01-30)

### Minor Changes

- Fix the setting of `prev_version` in `delete_versions()`.

## 1.3.12 (2022-08-17)

### Minor Changes

- Fix variable length strings in _recreate_virtual_dataset. Note that for h5py
  3.7.0 they cause a segfault without the upstream fix
  [h5py/h5py#2111](https://github.com/h5py/h5py/pull/2111).

## 1.3.11 (2022-08-02)

### Minor Changes

- Improve the performance of creating versions in h5py 3.

## 1.3.10 (2022-06-16)

### Minor Changes

- Fix deleting the current version.
- Properly handle fillvalue-only datasets in `delete_versions()`.

## 1.3.9 (2022-04-28)

### Minor Changes

- Fix `delete_versions()` with nested groups.
- Fix `delete_versions()` with variable length strings.

## 1.3.8 (2022-03-29)

### Major Changes

- Versioning wrappers are now skipped when loading a file in read-only mode.
  This leads to better performance in these cases.

### Minor Changes

- Fix the test suite with newer versions of pytest.

## 1.3.7 (2022-01-27)

### Major Changes
- `delete_version` has been renamed to `delete_versions`, and now takes a list
  of versions to delete. The old `delete_version` is kept intact for backwards
  compatibility.
- `delete_versions` (née `delete_version`) is now much faster.

## 1.3.6 (2021-10-19)

### Minor Changes

- Create hashtable datasets with lzf compression enabled.

## 1.3.5 (2021-09-30)

### Minor Changes

- Fix a bug with the hashtable introduced in 1.3.4.

## 1.3.4 (2021-09-27)

### Minor Changes

- Make the hashtable dataset much smaller for datasets with no data or version
  history.

## 1.3.3 (2021-07-02)

### Minor Changes

- Fix a regression that prevented indexing a dataset with a mask from working.

## 1.3.2 (2021-07-28)

### Minor Changes

- Improve the performance of reading a reading a dataset that hasn't been
  written to (e.g., reading from an already committed version).

- Fix Versioned HDF5 to work with h5py 3.3.

- Fix an issue that would occur when using np.datetime64 objects for
  timestamps when the fractional part of the second was exactly 0.

## 1.3.1 (2021-05-20)

### Minor Changes

- Avoid some unnecessary precomputation in the hashtable object. This improves
  the performance for files that have many versions in them.

## 1.3 (2021-05-07)

### Major Changes

- Support h5py 3. h5py 2.10 is also still officially supported.

- Add functionality to replay versions. This allows mutating old versions.

- Add new helper functions {any}`delete_version` and {any}`modify_metadata`
  to delete a version and modify metadata on a dataset that must be the same
  across versions, such as chunk size or dtype.

- Add helper function {any}`recreate_dataset` for more advanced version
  replaying functionality.

### Minor Changes

- Disallow accessing vfile[version] before version has been committed. Doing
  so previously could lead to inconsistencies.

- Disallow accessing non-versioned groups from VersionedHDF5File.

- Better error messages from VersionedHDF5File when the underlying file is
  closed.

- Remove codecov.

- Some improvements to the benchmarking suite.

## 1.2.6 (2021-04-20)

### Minor Changes

- Fix a bug where chunks could be deleted from a dataset.

- Workaround an upstream h5py bug in the tests.

## 1.2.5 (2021-04-15)

### Minor Changes

- Fix a bug where attrs could be deleted from a dataset.

## 1.2.4 (2021-04-08)

### Major Changes

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
