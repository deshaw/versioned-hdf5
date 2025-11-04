import json
import os

import h5py
from generate_data import TestVersionedDatasetPerformance as TVDP  # noqa: N817

from versioned_hdf5 import VersionedHDF5File


# auxiliary code to format file sizes
def format_size(size):
    """
    Auxiliary function to convert bytes to a more readable
    human format.
    """
    suffixes = ["B", "KB", "MB", "GB"]
    i = 0
    while size >= 1024 and i < len(suffixes) - 1:
        size = size / 1024
        i += 1
    return f"{size:.2f} {suffixes[i]}"


class PerformanceTests:
    testname: str

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

    def create_files(self, versions=True):
        tests = []
        msg = ""
        for c in self.compression:
            for p in self.exponents:
                for n in self.num_transactions:
                    chunk_size = 2**p
                    if versions:
                        name = f"{self.testname}_{n}_{p}_{c}"
                    else:
                        name = f"{self.testname}_{n}_{p}_{c}_no_versions"
                    filename = os.path.join(self.path, f"{name}.h5")
                    msg += (
                        f"File with {n} transactions, chunk size 2**{p} "
                        f"and compression filter {c}"
                    )
                    try:
                        h5pyfile = h5py.File(filename, "r")
                        msg += " exists - unable to compute creation time.\n"
                        t = 0
                    except Exception:
                        msg += " not available. Creating new file.\n"
                        t = self.testfun(
                            n,
                            name,
                            chunk_size,
                            c,
                            versions=versions,
                            deterministic=True,
                        )
                        h5pyfile = h5py.File(filename, "r")
                    if versions:
                        data = VersionedHDF5File(h5pyfile)
                        tests.append(
                            dict(
                                num_transactions=n,
                                chunk_size=chunk_size,
                                compression=c,
                                filename=filename,
                                h5pyfile=h5pyfile,
                                data=data,
                                t_write=t,
                            )
                        )
                    else:
                        tests.append(
                            dict(
                                num_transactions=n,
                                chunk_size=chunk_size,
                                compression=c,
                                filename=filename,
                                h5pyfile=h5pyfile,
                                t_write=t,
                            )
                        )

        for test in tests:
            test["size"] = os.path.getsize(test["filename"])
            test["size_label"] = format_size(test["size"])

        if versions:
            nt = len(self.num_transactions)
            for test in tests[-nt:]:
                lengths = []
                total_size = 0
                for vname in test["data"]._versions:
                    if vname != "__first_version__":
                        version = test["data"][vname]
                        group_key = list(version.keys())[0]
                        lengths.append(len(version[group_key]["val"]))
                        total_size += len(version[group_key]["val"])
                test["theoretical_sizes"] = 24 * total_size

        # Removing some irrelevant info from the dictionary
        summary = []
        for test in tests:
            summary.append(
                dict(
                    (k, test[k])
                    for k in [
                        "num_transactions",
                        "filename",
                        "size",
                        "size_label",
                        "t_write",
                        "chunk_size",
                        "compression",
                    ]
                )
            )
            test["h5pyfile"].close()

        self.tests = tests
        return summary, msg

    def save(self, summary, filename):
        with open(f"{filename}.json", "w") as json_out:
            json.dump(summary, json_out)


class TestLargeFractionChangesSparse(PerformanceTests):
    def __init__(self, **kwargs):
        self.testname = "test_large_fraction_changes_sparse"
        self.testfun = TVDP().test_large_fraction_changes_sparse
        super()._setoptions(options=kwargs)

    def create_files(self, versions=True):
        return super().create_files(versions=versions)

    def save(self, summary, filename):
        super().save(summary, filename)


class TestSmallFractionChangesSparse(PerformanceTests):
    def __init__(self, **kwargs):
        self.testname = "test_small_fraction_changes_sparse"
        self.testfun = TVDP().test_small_fraction_changes_sparse
        super()._setoptions(options=kwargs)

    def create_files(self, versions=True):
        return super().create_files(versions=versions)

    def save(self, summary, filename):
        super().save(summary, filename)


class TestMostlyAppendsSparse(PerformanceTests):
    def __init__(self, **kwargs):
        self.testname = "test_mostly_appends_sparse"
        self.testfun = TVDP().test_mostly_appends_sparse
        super()._setoptions(options=kwargs)

    def create_files(self, versions=True):
        return super().create_files(versions=versions)

    def save(self, summary, filename):
        super().save(summary, filename)


class TestMostlyAppendsDense(PerformanceTests):
    def __init__(self, **kwargs):
        self.testname = "test_mostly_appends_dense"
        self.testfun = TVDP().test_mostly_appends_dense
        super()._setoptions(options=kwargs)

    def create_files(self, versions=True):
        return super().create_files(versions=versions)

    def save(self, summary, filename):
        super().save(summary, filename)


class TestLargeFractionConstantSparse(PerformanceTests):
    def __init__(self, **kwargs):
        self.testname = "test_large_fraction_constant_sparse"
        self.testfun = TVDP().test_large_fraction_constant_sparse
        super()._setoptions(options=kwargs)

    def create_files(self, versions=True):
        return super().create_files(versions=versions)

    def save(self, summary, filename):
        super().save(summary, filename)


if __name__ == "__main__":
    tests = [
        TestMostlyAppendsDense,
        TestSmallFractionChangesSparse,
        TestLargeFractionChangesSparse,
        TestLargeFractionConstantSparse,
        TestMostlyAppendsSparse,
    ]

    for test in tests:
        testcase = test(
            num_transactions=[2],
            exponents=[
                12,
            ],
            compression=[
                None,
            ],
        )
        summary, msg = testcase.create_files(versions=True)
        testcase.save(summary, f"{testcase.testname}")
        summary, msg = testcase.create_files(versions=False)
        testcase.save(summary, f"{testcase.testname}_no_versions")
