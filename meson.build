project(
  'versioned-hdf5',
  'c',
  'cpp',
  'cython',
  version: run_command(['python', '-m', 'setuptools_scm'], check: true).stdout().strip(),
)

py = import('python').find_installation(pure: false)
cy = meson.get_compiler('cython')

other_deps = []
hdf5_dep = dependency('hdf5')

if host_machine.system() == 'windows'
  cc = meson.get_compiler('c')
  other_deps += cc.find_library('zlib', dirs: get_option('zlib_libs'))
  other_deps += cc.find_library('shlwapi')
endif

compiled_deps = [hdf5_dep] + other_deps

subdir('versioned_hdf5')
