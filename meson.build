project(
  'versioned-hdf5',
  'c',
  'cpp',
  'cython',
  version: run_command([find_program('python'), '-m', 'setuptools_scm'], check: true).stdout().strip(),
  meson_version: '>=1.2',
  default_options: [
    'cpp_std=c++17'
  ],
)

py = import('python').find_installation(pure: false)
cy = meson.get_compiler('cython')

# Declare every Cython module in this project compatible with a free-threaded
# interpreter (Py_mod_gil = Py_MOD_GIL_NOT_USED). This must be a project-wide
# argument so that it reaches all five extension modules, including
# 'slicetools', which does not use the shared 'cython_args' list below.
# Cython <3.1 does not know the directive; the 'mindeps' environment pins 3.0.10.
if cy.version().version_compare('>=3.1.0')
    add_project_arguments('-X', 'freethreading_compatible=True', language: 'cython')
endif

compiled_deps = [dependency('hdf5')]

subdir('versioned_hdf5')
