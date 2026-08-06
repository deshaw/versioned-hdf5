"""Benchmarks for StagedChangesArray, in isolation from the rest of the library.

Every slab here is a tiny NumPy array and there is deliberately no h5py anywhere in
this module: the point is to measure the cost of creating and executing the ``*Plan``
objects, which is dominated by per-chunk index arithmetic, and not the memory
bandwidth of the transfers.
"""

from __future__ import annotations

from contextlib import suppress
from functools import cache

import numpy as np

with suppress(ImportError):  # Allow asv-compare vs. older releases
    from versioned_hdf5.staged_changes import StagedChangesArray

#: Deliberately not divisible by the chunk size, so that every benchmark exercises the
#: edge chunks code paths
SHAPE = (301, 302)
CHUNK_SIZE = (3, 3)
DTYPE = "i2"

#: Index that selects the whole array
ALL = (slice(None), slice(None))

#: How the chunks are laid out on the slabs before a benchmark runs
STATES = [
    # Empty array; every chunk points to the full slab (a.k.a. the fill_value).
    # This is what backs a brand new InMemorySparseDataset.
    "full",
    # Every chunk lies on a single read-only base slab, complete with hash table.
    # This is what backs an InMemoryDataset that has just been staged, where the base
    # slab is the raw_data dataset on disk.
    "base",
    # Every chunk has been copied over from the base slab to a single staged slab
    "staged",
    # Every chunk lies on a staged slab of its own, as it happens when the user
    # writes to the chunks one by one. Worst case for the plans, which must
    # partition their transfers by slab.
    "fragmented",
]

#: (index, initial state) pairs for __getitem__ and __setitem__.
#: This is deliberately not the full cross product of indices and states, which would
#: be far too slow to run; the first five cases vary the index on the most typical
#: state, whereas the last two vary the state on the most demanding index.
CASES = {
    # 1 chunk out of 400, wholly selected
    "one_chunk": ((slice(0, 10), slice(0, 10)), "base"),
    # 1 chunk out of 400, partially selected. __setitem__ must first copy the chunk
    # from the base slab onto a brand new staged slab.
    "one_point": ((0, 0), "base"),
    # All 400 chunks, wholly selected. __setitem__ never reads the base slab.
    "all": (ALL, "base"),
    # All 400 chunks, partially selected. Worst case for __setitem__, which must
    # first copy every single chunk from the base slab onto the staged slabs.
    "step": ((slice(None, None, 2), slice(None, None, 2)), "base"),
    # 60 chunks, selected with an advanced (fancy) index along axis 0
    "fancy": ((np.array([3, 55, 194]), slice(None)), "base"),
    # All 400 chunks of an array that is entirely full of the fill_value
    "all_full": (ALL, "full"),
    # All 400 chunks of an array where each chunk lies on a staged slab of its own
    "all_fragmented": (ALL, "fragmented"),
}

#: Subset of CASES for __getitem__, which unlike __setitem__ makes no distinction
#: between wholly and partially selected chunks; the cases that only differ by that
#: are omitted to keep the runtime of the module in check.
GETITEM_CASES = ["one_chunk", "all", "fancy", "all_full", "all_fragmented"]
assert not set(GETITEM_CASES) - set(CASES)

#: New shapes for resize()
RESIZES = {
    # Append one row along axis 0; the typical pattern of a time series. This enlarges
    # the edge chunks along axis 0, which must be physically filled with fill_value.
    "append": (SHAPE[0] + 1, SHAPE[1]),
    # Double the size along both axes, which also enlarges the edge chunks
    "enlarge": (SHAPE[0] * 2, SHAPE[1] * 2),
    # Halve the size along both axes, which dereferences the chunks beyond the edge
    "shrink": (SHAPE[0] // 2, SHAPE[1] // 2),
}

#: What has been staged before commit() runs
COMMIT_SCENARIOS = [
    # Nothing was modified; commit() must not write anything
    "no_changes",
    # A single chunk was modified
    "one_chunk",
    # Every chunk was modified and every staged chunk is unique
    "all_new",
    # Every chunk was rewritten with the contents it already has on the base slab;
    # every staged chunk is deduplicated away and nothing is written
    "all_duplicate",
    # Every chunk was modified separately, so each lies on a staged slab of its own
    "fragmented",
    # Same staged changes as "all_new", but the base slab also carries the chunks of
    # 9 previous versions, which are no longer referenced by the virtual array.
    # commit() scans their hashes too - it may resuscitate a chunk that was deleted
    # versions ago - so the deduplication table is 10x larger, even though the amount
    # of data that is read, hashed, and written is the same as in "all_new".
    "obsolete_base",
]

#: Number of chunks on the base slab that are left over from previous versions in the
#: "obsolete_base" scenario, on top of the 400 that the virtual array points to
N_OBSOLETE_CHUNKS = 3600


def _commit_random(shape: tuple[int, ...], seed: int) -> StagedChangesArray:
    """Return a StagedChangesArray covering shape, with all of its chunks on a single
    base slab complete with hash table, built by staging random data and committing it
    """
    rng = np.random.default_rng(seed)
    arr = StagedChangesArray.from_array(
        rng.random(shape), chunk_size=CHUNK_SIZE, as_base_slabs=False
    )
    # Consolidate the staged slabs into a single base slab, complete with hashes
    # This is unlike as_base_slabs, which would not compute the hashes.
    arr.commit()
    assert arr.n_base_slabs == 1
    return arr


@cache
def _base_slab(
    n_obsolete_chunks: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (slab, hash_table, slab_indices, slab_offsets) of a single base slab
    that covers the whole of SHAPE with unique chunks, optionally preceded by
    n_obsolete_chunks more unique chunks that the virtual array doesn't point to.

    This mimics the raw_data dataset of an InMemoryDataset, whose hash table is loaded
    from disk alongside it and which accumulates the chunks of all the versions that
    came before. The slab is read-only and its hash table is never updated, so both
    are memoized and shared by all the benchmarks in this module.
    """
    tmp = _commit_random(SHAPE, seed=42)
    (slab,) = tmp.base_slabs
    (hash_table,) = tmp.base_hash_tables
    assert slab is not None
    assert hash_table is not None
    slab_indices, slab_offsets = tmp.slab_indices, tmp.slab_offsets

    if n_obsolete_chunks:
        # The chunks left behind by the previous versions. They are all unique, so
        # they never actually deduplicate anything; they only make the hash table
        # that commit() has to scan larger. Prepend them, as the chunks of the
        # current version are the most recently appended ones.
        obsolete = _commit_random(
            (n_obsolete_chunks * CHUNK_SIZE[0], CHUNK_SIZE[1]), seed=43
        )
        (obsolete_slab,) = obsolete.base_slabs
        (obsolete_hash_table,) = obsolete.base_hash_tables
        assert obsolete_slab is not None
        assert obsolete_hash_table is not None
        slab = np.concatenate([obsolete_slab, slab])
        hash_table = np.concatenate([obsolete_hash_table, hash_table])
        slab_offsets = slab_offsets + obsolete_slab.shape[0]

    slab.flags.writeable = False
    return slab, hash_table, slab_indices, slab_offsets


def make_array(state: str, n_obsolete_chunks: int = 0) -> StagedChangesArray:
    """Create a StagedChangesArray with the chunks laid out as described by STATES"""
    assert state in STATES

    if state in ("full", "fragmented"):
        arr = StagedChangesArray.full(SHAPE, chunk_size=CHUNK_SIZE, dtype=DTYPE)
        if state == "fragmented":
            # This is the same state you get by looping on the chunks and setting each
            # of them to a different value with a separate __setitem__ call, only much
            # faster to build. Values must be unique so that commit() can't
            # deduplicate the chunks, and must differ from the fill_value for the
            # same reason.
            n = np.prod(arr.n_chunks)
            arr.slabs += [
                np.full(CHUNK_SIZE, float(i + 1), dtype=DTYPE) for i in range(n)
            ]
            arr.hash_tables += [None] * n
            arr.slab_indices[()] = np.arange(1, n + 1).reshape(arr.n_chunks)
        return arr

    slab, hash_table, slab_indices, slab_offsets = _base_slab(n_obsolete_chunks)
    arr = StagedChangesArray(
        shape=SHAPE,
        chunk_size=CHUNK_SIZE,
        base_slabs=[slab],
        # __init__ modifies these in place
        slab_indices=slab_indices.copy(),
        slab_offsets=slab_offsets.copy(),
        base_hash_tables=[hash_table],
    )
    if state == "staged":
        arr.load()
    elif state != "base":
        raise AssertionError("unreachable")
    return arr


def make_committable(scenario: str) -> StagedChangesArray:
    """Create a StagedChangesArray with staged changes as described by
    COMMIT_SCENARIOS, ready to be committed
    """
    if scenario == "fragmented":
        return make_array("fragmented")

    n_obsolete_chunks = N_OBSOLETE_CHUNKS if scenario == "obsolete_base" else 0
    arr = make_array("base", n_obsolete_chunks=n_obsolete_chunks)
    if scenario == "no_changes":
        pass
    elif scenario == "one_chunk":
        arr[:10, :10] = 42.0
    elif scenario in ("all_new", "obsolete_base"):
        arr[ALL] = np.random.default_rng(0).random(SHAPE)
    elif scenario == "all_duplicate":
        arr[ALL] = arr[ALL]
    else:
        raise AssertionError("unreachable")

    return arr


class _MutatingBenchmark:
    """Common settings for benchmarks that alter the state of the StagedChangesArray
    and so need setup() to run again before every call.

    asv runs setup() once per sample and then calls the benchmark ``number`` times,
    skipping setup() altogether during the warmup phase; this forces one call per
    sample and no warmup. ``asv run --quick`` does the same globally.
    """

    number = 1
    warmup_time = 0


class TimeGetItem:
    """Benchmark GetItemPlan creation and execution"""

    params = [GETITEM_CASES]
    param_names = ["case"]

    def setup(self, case):
        self.idx, state = CASES[case]
        self.arr = make_array(state)

    def time_getitem_plan(self, case):
        self.arr._getitem_plan(self.idx)

    def time_getitem(self, case):
        # Internally calls _getitem_plan() and then executes the plan
        self.arr[self.idx]


class TimeSetItem(_MutatingBenchmark):
    """Benchmark SetItemPlan creation and execution"""

    params = [list(CASES)]
    param_names = ["case"]

    def setup(self, case):
        self.idx, state = CASES[case]
        self.arr = make_array(state)
        self.value = np.asarray(self.arr[self.idx]) + 1.0

    def time_setitem_plan(self, case):
        self.arr._setitem_plan(self.idx)

    def time_setitem(self, case):
        # Internally calls _setitem_plan() and then executes the plan
        self.arr[self.idx] = self.value


class TimeResize(_MutatingBenchmark):
    """Benchmark ResizePlan creation and execution"""

    params = [list(RESIZES)]
    param_names = ["resize"]

    def setup(self, resize):
        self.arr = make_array("base")
        self.shape = RESIZES[resize]

    def time_resize_plan(self, resize):
        self.arr._resize_plan(self.shape)

    def time_resize(self, resize):
        # Internally calls _resize_plan() and then executes the plan
        self.arr.resize(self.shape)


class TimeLoad(_MutatingBenchmark):
    """Benchmark LoadPlan creation and execution.

    Only the chunks that lie on the base slabs are loaded, so this is a no-op for all
    the states other than "base".
    """

    params = [["base", "staged"]]
    param_names = ["state"]

    def setup(self, state):
        self.arr = make_array(state)

    def time_load_plan(self, state):
        self.arr._load_plan()

    def time_load(self, state):
        # Internally calls _load_plan() and then executes the plan
        self.arr.load()


class TimeChanges:
    """Benchmark ChangesPlan creation and execution.

    Chunks that lie on the base slabs are yielded as slices, whereas the staged ones
    are yielded as numpy arrays; these are the two states that matter here.
    """

    params = [["base", "staged"]]
    param_names = ["state"]

    def setup(self, state):
        self.arr = make_array(state)

    def time_changes_plan(self, state):
        self.arr._changes_plan()

    def time_changes(self, state):
        # Internally calls _changes_plan() and then executes the plan
        for _ in self.arr.changes():
            pass


class TimeCommit(_MutatingBenchmark):
    """Benchmark HashPlan creation and execution, and the whole of commit()"""

    params = [COMMIT_SCENARIOS]
    param_names = ["scenario"]

    def setup(self, scenario):
        self.arr = make_committable(scenario)

    def time_hash_plan(self, scenario):
        self.arr._hash_plan()

    def time_calc_hashes(self, scenario):
        # Internally calls _hash_plan() then executes the plan
        self.arr._calc_hashes()

    def time_commit(self, scenario):
        # Internally calls _calc_hashes(), then calls
        # _commit_plan() and executes the plan
        self.arr.commit()


class TimeCommitPlan:
    """Benchmark CommitPlan creation, which deduplicates the staged chunks against the
    hashes of the base and full chunks. Unlike the other benchmarks in TimeCommit, it
    requires all the hashes to be already up to date.
    """

    params = [COMMIT_SCENARIOS]
    param_names = ["scenario"]

    def setup(self, scenario):
        self.arr = make_committable(scenario)
        self.arr._calc_hashes()

    def time_commit_plan(self, scenario):
        self.arr._commit_plan()
