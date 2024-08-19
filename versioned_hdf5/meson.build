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
