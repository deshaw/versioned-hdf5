project(
  'versioned-hdf5',
  'c',
  'cpp',
  'cython',
  version: run_command(['python', '-m', 'setuptools_scm'], check: true).stdout().strip(),
)

py = import('python').find_installation(pure: false)
cy = meson.get_compiler('cython')

subdir('versioned_hdf5')
