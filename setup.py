import setuptools
import _versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="versioned-hdf5",
    version=_versioneer.get_version(),
    cmdclass=_versioneer.get_cmdclass(),
    author="Quansight",
    description="Versioned HDF5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deshaw/versioned-hdf5",
    packages=['versioned_hdf5', 'versioned_hdf5.tests'],
    license="BSD",
    install_requires=[
        "h5py<3",
        "numpy",
        "ndindex>=1.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
