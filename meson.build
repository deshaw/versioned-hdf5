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
mpi_dep = dependency('mpi', required: false)

# In conda, this is installed as "ompi.pc"
if not mpi_dep.found()
  message('Could not find `mpi` with pkg-config; using pkg-config to look for `ompi`...')
  mpi_dep = dependency('ompi', required: false)
endif


# pkg-config hasn't been able to find mpi (.pc file not generated?)
if not mpi_dep.found()
  # Windows uses msmpi
  if host_machine.system() == 'windows'
    cc = meson.get_compiler('c')
    mpi_dep = cc.find_library('msmpi', dirs: get_option('msmpi_libs'))

    other_deps += cc.find_library('zlib', dirs: get_option('zlib_libs'))
    other_deps += cc.find_library('shlwapi')

  else
    # Try declaring dependency with mpicc on linux and macos
    message('Could not find `ompi` with pkg-config; manually setting the include and link args with mpicc output...')

    mpicc = find_program('mpicc')
    mpicc_cflags = run_command(mpicc, '--showme:compile').stdout().strip().split()
    mpicc_ldflags = run_command(mpicc, '--showme:link').stdout().strip().split()

    mpi_dep = declare_dependency(
        link_args : mpicc_ldflags,
        include_directories : mpicc_cflags,
    )
  endif
endif

compiled_deps = [
  mpi_dep,
  hdf5_dep,
] + other_deps


subdir('versioned_hdf5')
