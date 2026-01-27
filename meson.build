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

if host_machine.system() == 'windows'
  cc = meson.get_compiler('c')
  compiled_deps += cc.find_library('zlib', dirs: get_option('zlib_path'))
  compiled_deps += cc.find_library('shlwapi')
endif

subdir('versioned_hdf5')
