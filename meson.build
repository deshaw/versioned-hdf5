project(
  'versioned-hdf5',
  'c',
  'cpp',
  'cython',
  version: run_command(['python', '-m', 'setuptools_scm'], check: true).stdout().strip(),
  meson_version: '>=1.1',
)

py = import('python').find_installation(pure: false)
cy = meson.get_compiler('cython')

compiled_deps = [dependency('hdf5')]

subdir('versioned_hdf5')
