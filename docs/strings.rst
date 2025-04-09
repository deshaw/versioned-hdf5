Strings support
===============

Support status and requirements
-------------------------------

versioned-hdf5 supports all data types that h5py supports. This includes:

- `fixed-width bytes arrays
  <https://docs.h5py.org/en/stable/strings.html#storing-strings>`_
  (``dtype=h5py.string_dtype(length=n)`` or ``dtype="S"``);
- `variable-width object strings
  <https://docs.h5py.org/en/stable/strings.html#storing-strings>`_
  (``dtype=h5py.string_dtype()``);
- `native NumPy variable-width strings
  <https://docs.h5py.org/en/stable/strings.html#numpy-variable-width-strings>`_
  (``dtype="T"``, a.k.a. StringDType, a.k.a. NpyStrings).

StringDType support requires:

- NumPy >=2.0
- h5py >=3.14
- versioned-hdf5 >=2.1

Just like in h5py, StringDType should be preferred unless your code needs to run on
older versions of the above dependencies. Also just like in h5py, StringDType produces
the exact same data on disk as object strings, so it is possible to write as StringDType
and then read it back as object strings (e.g. due to older dependencies versions),
and vice versa.


Don't do this! (a.k.a. the short version)
-----------------------------------------

When using StringDType, you should always change the dtype of the dataset *before*
slicing.

This pattern is slow in h5py and *extremely* slow in versioned-hdf5::

    for i in range(ds.shape[0]):
        x = ds[i, :].astype("T")  # DON'T DO THIS
        y = f(x)  # StringDType -> StringDType
        ds[i, :] = y

you should instead write::

    for i in range(ds.shape[0]):
        x = ds.astype("T")[i, :]  # DO THIS! astype first, __getitem__ second
        y = f(x)
        ds[i, :] = y

To understand why, continue to the following section.


Performance anti-patterns (a.k.a. the long version)
---------------------------------------------------

As a general rule, versioned-hdf5 and h5py are interchangeable once you open a (staged)
dataset. StringDType support is opt-in in versioned-hdf5 just like it is in h5py.

versioned-hdf5 implements support by swapping the dtype of staged changes whenever
there is a change in the dtype requested by

- ``Dataset.__getitem__`` (always returns object-type strings, like h5py);
- ``Dataset.astype("T").__getitem__`` (returns NpyStrings, like h5py);
- ``Dataset.__setitem__`` (accepts both object-type strings and NpyStrings, like h5py).

One caveat of this design is that it makes it very expensive to mix accesses in
object dtype and StringDType. Consider this pattern::

    for i in range(ds.shape[0]):
        x = ds[i, :].astype("T")  # DON'T DO THIS
        y = f(x)
        ds[i, :] = y

where f is a function that accepts an array with StringDType and returns the same.
The above pattern is already suboptimal in h5py, because it causes to:

1. (i=0, 1, 2, ...)

  a. read object strings from the dataset;
  b. convert them in memory with NumPy to StringDType;
  c. write natively in StringDType.
  d. go back to step 1.a for next iteration.

However, in versioned-hdf5 it's going to be even slower:

1. (i=0, 2, 4, ...)

  a. read object strings from the dataset;
  b. convert them in memory with NumPy to StringDType;
  c. stage the changes as StringDType. This triggers a conversion of previous
     staged changes from object strings to StringDType;

2. (i=1, 3, 5, ...)

  a. at the next iteration, read object strings from the dataset again; however staged
     changes are now in StringDType and will be converted back to object strings;
  b. go back to step 1.b;

3. (commit)

   when you finally commit your changes, if the last action wasn't to write in
   StringDType (e.g. because you don't always execute the write-back), the whole
   set of staged changes will be written as object strings.

To avoid this inefficiency, in both h5py and versioned-hdf5 you should always first
change dtype and then slice::

    for i in range(ds.shape[0]):
        x = ds.astype("T")[i, :]  # DO THIS
        y = f(x)
        ds[i, :] = y

In h5py, the above does the following:

1. (i=0, 1, 2, ...)

  a. read natively from hdf5 into a NumPy StringDType array, without any intermediate
     conversion through object strings;
  b. write to hdf5 natively in StringDType;
  c. go back to step 1.a for next iteration.

And in versioned-hdf5:

1. (i=0, 1, 2, ...)

  a. read natively into NumPy StringDType
  b. stage changes in memory as StringDType, without conversion;
  c. go back to step 1.a for next iteration.

2. (commit) write to hdf5 natively in StringDType.
