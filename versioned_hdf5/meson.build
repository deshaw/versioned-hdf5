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
    [
        'slicetools.pyx',
    ],
    install: true,
    subdir: 'versioned_hdf5',
    dependencies: compiled_deps,
    cython_args: ['--cplus'],
    override_options : ['cython_language=cpp'],
)
