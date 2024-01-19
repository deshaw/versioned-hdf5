project(
  'versioned-hdf5',
  version: run_command(['python', '-m', 'setuptools_scm'], check: true).stdout().strip(),
)

py = import('python').find_installation()
py.install_sources(
  [
    'versioned_hdf5/__init__.py',
    'versioned_hdf5/api.py',
    'versioned_hdf5/backend.py',
    'versioned_hdf5/hashtable.py',
    'versioned_hdf5/replay.py',
    'versioned_hdf5/slicetools.py',
    'versioned_hdf5/versions.py',
    'versioned_hdf5/wrappers.py',
  ],
  subdir: 'versioned_hdf5',
)
