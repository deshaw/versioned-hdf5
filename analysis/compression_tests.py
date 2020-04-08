import os
import sys
sys.path.append('..')
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
from versioned_hdf5 import VersionedHDF5File
from generate_data_deterministic import TestVersionedDatasetPerformance as TVDP

path = "/home/melissa/projects/versioned-hdf5/analysis" # change this as necessary

# auxiliary code to format file sizes 
def format_size(size):
    suffixes = ['B', 'KB', 'MB', 'GB']
    i = 0
    while size >= 1024 and i < len(suffixes)-1:
        size = size/1024
        i += 1
    return f"{size:.2f} {suffixes[i]}"

testname = "test_large_fraction_changes_sparse"

num_transactions_1 = [50, 100, 500, 1000, 5000, 10000, 20000]
exponents = [14]
compression = [None, "gzip", "lzf"]
#compression_opts = []

# Lossless compression filters
# ----------------------------
# - GZIP filter ("gzip")
#    Available with every installation of HDF5, so it’s best where portability is required. Good compression, moderate speed. compression_opts sets the compression level and may be an integer from 0 to 9, default is 4.
# - LZF filter ("lzf")
#    Available with every installation of h5py (C source code also available). Low to moderate compression, very fast. No options.
# - SZIP filter ("szip")
#    Patent-encumbered filter used in the NASA community. Not available with all installations of HDF5 due to legal reasons. Consult the HDF5 docs for filter options.
#
# Custom compression filters
# --------------------------
# In addition to the compression filters listed above, compression filters can be dynamically loaded by the underlying HDF5 library. This is done by passing a filter number to Group.create_dataset() as the compression parameter. The compression_opts parameter will then be passed to this filter.

# Create files and set up dictionary with test info.
tests_complete = []
for c in compression:
    for p in exponents:
        for n in num_transactions_1:
            chunk_size = 2**p
            name = f"{testname}_{n}_{p}_{c}"
            filename = os.path.join(path, f"{name}.h5")
            print("File with\n" \
                  f"- {n} transactions\n" \
                  f"- chunk size 2**{p}\n"\
                  f"- compression filter {c}\n")
            try:
                h5pyfile = h5py.File(filename, 'r')
                print("already exists - unable to compute creation time.")
                t = 0
            except:
                print("not available. Creating new file.")
                t0 = time.time()
                TVDP().test_large_fraction_changes_sparse(n, name, chunk_size, c)
                t = time.time()-t0
                h5pyfile = h5py.File(filename, 'r')
            data = VersionedHDF5File(h5pyfile)
            tests_complete.append(dict(num_transactions=n,
                                       chunk_size=chunk_size,
                                       compression=c,
                                       filename=filename,
                                       h5pyfile=h5pyfile,
                                       data=data,
                                       t_write=t))

for test in tests_complete:
    test['size'] = os.path.getsize(test['filename'])
    test['size_label'] = format_size(test['size'])

n = len(num_transactions_1)
for test in tests_complete[-n:]:
    lengths = []
    total_size = 0
    for vname in test['data']._versions:
        if vname != '__first_version__':
            version = test['data'][vname]
            group_key = list(version.keys())[0]
            lengths.append(len(version[group_key]['val']))
            total_size += len(version[group_key]['val'])
    test['theoretical_sizes'] = 24*total_size
    #print(f"Maximum array size for file with {test['num_transactions']} transactions: {max(lengths)}")
t_sizes_1 = [test['theoretical_sizes'] for test in tests_complete[-n:]]

# Removing some irrelevant info from the dictionary 
test_large_fraction_changes_sparse = []
for test in tests_complete:
    test_large_fraction_changes_sparse.append(dict((k, test[k]) for k in ['num_transactions', 'filename', 'size', 'size_label', 't_write']))

filesizes_1 = np.array([test['size'] for test in test_large_fraction_changes_sparse])
sizelabels_1 = np.array([test['size_label'] for test in test_large_fraction_changes_sparse])

fig_large_fraction_changes = plt.figure(figsize=(14,10))
plt.plot(num_transactions_1, t_sizes_1, 'o--', ms=5, color='magenta', label="Theoretical file size")

for i in range(len(compression)):
    plt.plot(num_transactions_1, filesizes_1[i*n:(i+1)*n], '*--', ms=12, label=f"Compression={compression[i]}")

plt.xlabel("Transactions")
plt.title("test_large_fraction_changes_sparse")
plt.legend()
plt.yticks(filesizes_1[-n:], sizelabels_1[-n:])
plt.show()

fig_large_fraction_changes_log = plt.figure(figsize=(14,10))
plt.loglog(num_transactions_1, t_sizes_1, 'o--', ms=5, color='magenta', label="Theoretical file size")

for i in range(len(compression)):
    plt.loglog(num_transactions_1, filesizes_1[i*n:(i+1)*n], '*--', ms=12, label=f"Compression={compression[i]}")

plt.xlabel("Transactions")
plt.title("test_large_fraction_changes_sparse")
plt.legend()
plt.yticks(filesizes_1[-n:], sizelabels_1[-n:])
plt.show()

t_write_1 = np.array([test['t_write'] for test in test_large_fraction_changes_sparse])

fig_large_fraction_changes_times = plt.figure()
for i in range(len(compression)):
    plt.plot(num_transactions_1, t_write_1[i*n:(i+1)*n], 'o--', ms=8, label=f"Compression={compression[i]}")
plt.xlabel("Transactions")
plt.title("test_large_fraction_changes_sparse - creation times in seconds")
plt.legend()
plt.xticks(num_transactions_1)
plt.show()

with open("test_large_fraction_changes_sparse_14_compression.pickle", "wb") as pickle_out:
    pickle.dump(test_large_fraction_changes_sparse, pickle_out)

for test in tests_complete:
    test['h5pyfile'].close()
