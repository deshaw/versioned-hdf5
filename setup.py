import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="versioned-hdf5",
    version="0.0.1",
    author="Quansight",
    description="Versioned HDF5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quansight/versioned-hdf5",
    packages=['versioned_hdf5', 'versioned_hdf5.tests'],
    license="MIT",
    install_requires=[
        "h5py",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
