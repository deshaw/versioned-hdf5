project(
  'versioned-hdf5',
  'c',
  'cython',
  version: run_command([find_program('python'), '-m', 'setuptools_scm'], check: true).stdout().strip(),
  meson_version: '>=1.2',
)

py = import('python').find_installation(pure: false)
cy = meson.get_compiler('cython')

compiled_deps = [dependency('hdf5')]

subdir('versioned_hdf5')
