py.install_sources(
    [
        '__init__.py',
        'api.py',
        'backend.py',
        'hashtable.py',
        'replay.py',
        'versions.py',
        'wrappers.py',
    ],
    subdir: 'versioned_hdf5',
)

# Adapted from https://numpy.org/doc/2.1/reference/random/examples/cython/meson.build.html
# Note:
# numpy 2.x -> site-packages/numpy/_core/include
# numpy 1.x -> site-packages/numpy/core/include
npy_include_path = run_command(
    py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true
).stdout().strip()

compiled_deps = [
    dependency('hdf5'),
    dependency('mpi'),
]

py.extension_module(
    'slicetools',
    'slicetools.pyx',
    install: true,
    subdir: 'versioned_hdf5',
    dependencies: compiled_deps,
    include_directories: [npy_include_path],
    cython_args: ['--cplus'],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'subchunk_map',
    # FIXME This is a symlink. Can't find a way to convince Meson to compile
    # with Cython pure-python .py files
    # (read: https://cython.readthedocs.io/en/latest/src/tutorial/pure.html)
    'subchunk_map.pyx',
    install: true,
    subdir: 'versioned_hdf5',
)
