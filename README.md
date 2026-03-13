# versioned-hdf5

Versioned HDF5 provides a versioned abstraction on top of `h5py`.

See the documentation at https://deshaw.github.io/versioned-hdf5/ and [learn more about Versioned HDF5](https://www.deshaw.com/library/desco-quansight-introducing-versioned-hdf5) in our introductory blog post.

## Binary wheels on PyPi

This project does not distribute wheels, due to complexity of sharing the libhdf5 C library with
h5py. Please either use conda or compile both h5py and versioned-hdf5 from sources.
Read more at https://deshaw.github.io/versioned-hdf5/master/installation.html.

## History

This was created by the [D. E. Shaw group](https://www.deshaw.com/) in conjunction with [Quansight](https://www.quansight.com/).

<p align="center">
    <a href="https://www.deshaw.com">
       <img src="https://www.deshaw.com/assets/logos/blue_logo_417x125.png" alt="D. E. Shaw Logo" height="75" >
    </a>
</p>

## License

This project is released under a [BSD-3-Clause license](https://github.com/deshaw/versioned-hdf5/blob/master/LICENSE).

We love contributions! Before you can contribute, please sign and submit this [Contributor License Agreement (CLA)](https://www.deshaw.com/oss/cla).
This CLA is in place to protect all users of this project.
