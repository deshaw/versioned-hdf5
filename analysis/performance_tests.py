import os
import sys
sys.path.append('..')
import json
import h5py
import time
from versioned_hdf5 import VersionedHDF5File
from generate_data_deterministic import TestVersionedDatasetPerformance as TVDP

# auxiliary code to format file sizes 
def format_size(size):
    """
    Auxiliary function to convert bytes to a more readable
    human format.
    """
    suffixes = ['B', 'KB', 'MB', 'GB']
    i = 0
    while size >= 1024 and i < len(suffixes)-1:
        size = size/1024
        i += 1
    return f"{size:.2f} {suffixes[i]}"
    

class PerformanceTests:
    
    def __init__(self, **kwargs):
        pass

    def _setoptions(self, options):
        keys = options.keys()
        if "path" in keys:
            self.path = options["path"]
        else:
            self.path = "."
        if "num_transactions" in keys:
            self.num_transactions = options["num_transactions"]
        else:
            self.num_transactions = []
        if "exponents" in keys:
            self.exponents = options["exponents"]
        else:
            self.exponents = []
        if "compression" in keys:
            self.compression = options["compression"]
        else:
            self.compression = []
        if "verbose" in keys:
            self.verbose = options["verbose"]
        else:
            self.verbose = False
            
    def create_files(self):        
        tests = []
        for c in self.compression:
            for p in self.exponents:
                for n in self.num_transactions:
                    chunk_size = 2**p
                    name = f"{self.testname}_{n}_{p}_{c}"
                    filename = os.path.join(self.path, f"{name}.h5")
                    if self.verbose:
                        print(name)
                        print("File with\n" \
                              f"- {n} transactions\n" \
                              f"- chunk size 2**{p}\n"\
                              f"- compression filter {c}")
                    try:
                        h5pyfile = h5py.File(filename, 'r')
                        if self.verbose:
                            print("already exists - unable to compute creation time.")
                        t = 0
                    except:
                        if self.verbose:
                            print("not available. Creating new file.")
                        t0 = time.time()
                        self.testfun(n, name, chunk_size, c)
                        t = time.time()-t0
                        h5pyfile = h5py.File(filename, 'r')
                    data = VersionedHDF5File(h5pyfile)
                    tests.append(dict(num_transactions=n,
                                      chunk_size=chunk_size,
                                      compression=c,
                                      filename=filename,
                                      h5pyfile=h5pyfile,
                                      data=data,
                                      t_write=t))

        for test in tests:
            test['size'] = os.path.getsize(test['filename'])
            test['size_label'] = format_size(test['size'])

        nt = len(self.num_transactions)
        for test in tests[-nt:]:
            lengths = []
            total_size = 0
            for vname in test['data']._versions:
                if vname != '__first_version__':
                    version = test['data'][vname]
                    group_key = list(version.keys())[0]
                    lengths.append(len(version[group_key]['val']))
                    total_size += len(version[group_key]['val'])
            test['theoretical_sizes'] = 24*total_size
            test['h5pyfile'].close()        

        # Removing some irrelevant info from the dictionary 
        summary =[]
        for test in tests:
            summary.append(dict((k, test[k]) for k in ['num_transactions', 'filename', 'size', 'size_label', 't_write', 'chunk_size', 'compression']))
            
        self.tests = tests
        return summary

    def save(self, summary):
        with open(f"{self.testname}.json", "w") as json_out:
            json.dump(summary, json_out)

        
class test_large_fraction_changes_sparse(PerformanceTests):

    def __init__(self, **kwargs):
        self.testname = "test_large_fraction_changes_sparse"
        self.testfun = TVDP().test_large_fraction_changes_sparse
        super()._setoptions(options=kwargs)
        
    def create_files(self):
        return super().create_files()

    def save(self, summary):
        super().save(summary)


class test_small_fraction_changes_sparse(PerformanceTests):

    def __init__(self, **kwargs):
        self.testname = "test_small_fraction_changes_sparse"
        self.testfun = TVDP().test_small_fraction_changes_sparse
        super()._setoptions(options=kwargs)

    def create_files(self):
        return super().create_files()

    def save(self, summary):
        super().save(summary)

        
class test_mostly_appends_sparse(PerformanceTests):

    def __init__(self, **kwargs):
        self.testname = "test_mostly_appends_sparse"
        self.testfun = TVDP().test_mostly_appends_sparse
        super()._setoptions(options=kwargs)

    def create_files(self):
        return super().create_files()
        
    def save(self, summary):
        super().save(summary)


class test_mostly_appends_dense(PerformanceTests):

    def __init__(self, **kwargs):
        self.testname = "test_mostly_appends_dense"
        self.testfun = TVDP().test_mostly_appends_dense
        super()._setoptions(options=kwargs)

    def create_files(self):
        return super().create_files()

    def save(self, summary):
        super().save(summary)

        
if __name__ == "__main__":
    
    tests = [test_large_fraction_changes_sparse,
             test_small_fraction_changes_sparse,
             test_mostly_appends_sparse,
             test_mostly_appends_dense]

    for test in tests:
        testcase = test()
        summary = testcase.create_files()
        testcase.save(summary) 
