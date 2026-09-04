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

# Deliberately not divisible by the chunk size, so that every benchmark exercises the
# edge chunks code paths
SHAPE = (4001, 4002)  # 122 MiB
CHUNK_SIZES = {
    "8 kiB": (32, 32),  # (126, 126) chunks -> 15_876 chunks
    "512 kiB": (256, 256),  # (16, 16) chunks -> 256 chunks
}
DTYPE = "f8"

# Index that selects the whole array
ALL = (slice(None), slice(None))

# How the chunks are laid out on the slabs before a benchmark runs
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
    # No base slabs; every chunk lies on a staged slab that is a view of the original
    # dataset. Edge slabs may be smaller than the chunk size.
    "from_array",
]

# (index, initial state) pairs for __getitem__ and __setitem__.
# This is deliberately not the full cross product of indices and states, which would
# be far too slow to run; the first five cases vary the index on the most typical
# state, whereas the last two vary the state on the most demanding index.
CASES = {
    # Select one to a few whole chunks
    "few_chunks": ((slice(256), slice(256)), "base"),
    # Single chunk, partially selected. __setitem__ must first copy the chunk
    # from the base slab onto a brand new staged slab.
    "one_point": ((0, 0), "base"),
    # All 400 chunks, wholly selected. __setitem__ never reads the base slab.
    "all": (ALL, "base"),
    # All 400 chunks, partially selected. Worst case for __setitem__, which must
    # first copy every single chunk from the base slab onto the staged slabs.
    "step": ((slice(None, None, 2), slice(None, None, 2)), "base"),
    # Three rows of points, selected with an advanced (fancy) index along
    "fancy": ((np.array([230, 520, 1000]), slice(None)), "base"),
    # All 400 chunks of an array that is entirely full of the fill_value
    "all_full": (ALL, "full"),
    # All 400 chunks of an array where each chunk lies on a staged slab of its own
    "all_fragmented": (ALL, "fragmented"),
}

# Subset of CASES for __getitem__, which unlike __setitem__ makes no distinction
# between wholly and partially selected chunks; the cases that only differ by that
# are omitted to keep the runtime of the module in check.
GETITEM_CASES = ["few_chunks", "all", "fancy", "all_full", "all_fragmented"]
assert not set(GETITEM_CASES) - set(CASES)

# New shapes for resize()
RESIZES = {
    # Append one row along axis 0; the typical pattern of a time series. This enlarges
    # the edge chunks along axis 0, which must be physically filled with fill_value.
    "append_row": (SHAPE[0] + 1, SHAPE[1]),
    # Same as above, but append one column. This changes things when the
    # StagedChangesArray's base slabs were created by from_array.
    "append_col": (SHAPE[0], SHAPE[1] + 1),
    # Double the size along both axes, which also enlarges the edge chunks
    "enlarge": (SHAPE[0] * 2, SHAPE[1] * 2),
    # Halve the size along both axes, which dereferences the chunks beyond the edge
    "shrink": (SHAPE[0] // 2, SHAPE[1] // 2),
}

# What has been staged before commit() runs
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
    # No base slabs; every chunk lies on a staged slab that is a view of the original
    # dataset. Edge slabs may be smaller than the chunk size.
    "from_array",
]

#: Number of chunks on the base slab that are left over from previous versions in the
#: "obsolete_base" scenario, on top of the 400 that the virtual array points to
N_OBSOLETE_CHUNKS = 3600


def _commit_random(
    shape: tuple[int, ...], chunk_size: tuple[int, ...], seed: int
) -> StagedChangesArray:
    """Return a StagedChangesArray covering shape, with all of its chunks on a single
    base slab complete with hash table, built by staging random data and committing it
    """
    rng = np.random.default_rng(seed)
    arr = StagedChangesArray.from_array(
        rng.random(shape), chunk_size=chunk_size, as_base_slabs=False
    )
    # Consolidate the staged slabs into a single base slab, complete with hashes
    # This is unlike as_base_slabs, which would not compute the hashes.
    arr.commit()
    assert arr.n_base_slabs == 1
    return arr


@cache
def buffer() -> np.ndarray:
    """Return a writeable buffer of random data, of shape SHAPE.

    This mimics the _buffer attribute of an InMemoryArrayDataset, the input to
    StagedChangesArray.from_array(as_base_slabs=False) in
    InMemoryArrayDatasetWrapper.resize().
    """
    return np.random.default_rng(0).random(SHAPE).astype(DTYPE)


@cache
def base_slab(
    chunk_size: tuple[int, ...],
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
    tmp = _commit_random(SHAPE, chunk_size, seed=42)
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
            shape=(n_obsolete_chunks * chunk_size[0], *chunk_size[1:]),
            chunk_size=chunk_size,
            seed=43,
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


def make_array(
    state: str, chunk_size: tuple[int, ...], n_obsolete_chunks: int = 0
) -> StagedChangesArray:
    """Create a StagedChangesArray with the chunks laid out as described by STATES"""
    assert state in STATES

    if state == "from_array":
        return StagedChangesArray.from_array(
            buffer(), chunk_size=chunk_size, fill_value=0, as_base_slabs=False
        )

    if state == "full":
        return StagedChangesArray.full(SHAPE, chunk_size=chunk_size, dtype=DTYPE)

    if state == "fragmented":
        # Array has been modified through 256 individual __setitem__ calls that
        # hit different chunks; each call creates a separate slab.
        arr = StagedChangesArray.full(SHAPE, chunk_size=chunk_size, dtype=DTYPE)
        for r in range(1, 4000, 256):
            for c in range(2, 4000, 256):
                arr[r, c] = r + c
        return arr

    slab, hash_table, slab_indices, slab_offsets = base_slab(
        chunk_size, n_obsolete_chunks
    )
    arr = StagedChangesArray(
        shape=SHAPE,
        chunk_size=chunk_size,
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


def make_committable(scenario: str, chunk_size: tuple[int, ...]) -> StagedChangesArray:
    """Create a StagedChangesArray with staged changes as described by
    COMMIT_SCENARIOS, ready to be committed
    """
    if scenario in ("fragmented", "from_array"):
        return make_array(scenario, chunk_size)

    n_obsolete_chunks = N_OBSOLETE_CHUNKS if scenario == "obsolete_base" else 0
    arr = make_array("base", chunk_size, n_obsolete_chunks=n_obsolete_chunks)
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

    params = [GETITEM_CASES, list(CHUNK_SIZES)]
    param_names = ["case", "chunk_size"]

    def setup(self, case: str, chunk_size: str) -> None:
        self.idx, state = CASES[case]
        self.arr = make_array(state, CHUNK_SIZES[chunk_size])

    def time_getitem_plan(self, case: str, chunk_size: str) -> None:
        self.arr._getitem_plan(self.idx)

    def time_getitem(self, case: str, chunk_size: str) -> None:
        # Internally calls _getitem_plan() and then executes the plan
        self.arr[self.idx]


class TimeSetItem(_MutatingBenchmark):
    """Benchmark SetItemPlan creation and execution"""

    params = [list(CASES), list(CHUNK_SIZES)]
    param_names = ["case", "chunk_size"]

    def setup(self, case: str, chunk_size: str) -> None:
        self.idx, state = CASES[case]
        self.arr = make_array(state, CHUNK_SIZES[chunk_size])
        self.value = np.asarray(self.arr[self.idx]) + 1.0

    def time_setitem_plan(self, case: str, chunk_size: str) -> None:
        self.arr._setitem_plan(self.idx, copy=False)

    def time_setitem(self, case: str, chunk_size: str) -> None:
        # Internally calls _setitem_plan() and then executes the plan
        self.arr[self.idx] = self.value


class TimeResize(_MutatingBenchmark):
    """Benchmark ResizePlan creation and execution.

    For the "from_array" state, enlarging also deep-copies the trimmed staged slabs,
    as in the resize() of an InMemoryArrayDataset.
    """

    params = [list(RESIZES), ["base", "from_array"], list(CHUNK_SIZES)]
    param_names = ["resize", "state", "chunk_size"]

    def setup(self, resize: str, state: str, chunk_size: str) -> None:
        self.arr = make_array(state, CHUNK_SIZES[chunk_size])
        self.shape = RESIZES[resize]

    def time_resize_plan(self, resize: str, state: str, chunk_size: str) -> None:
        self.arr._resize_plan(self.shape, copy=False)

    def time_resize(self, resize: str, state: str, chunk_size: str) -> None:
        # Internally calls _resize_plan() and then executes the plan
        self.arr.resize(self.shape)


class TimeFromArray:
    """Benchmark StagedChangesArray.from_array().

    This is the entry point of the resize() of an InMemoryArrayDataset, which calls
    from_array(as_base_slabs=False) and then resize(). With as_base_slabs=False the
    staged slabs are views of the input buffer; only the trimmed edge slabs are
    deep-copied, and only later, upon resize() or first write.
    """

    params = [["base", "staged"], list(CHUNK_SIZES)]
    param_names = ["as_base_slabs", "chunk_size"]

    def setup(self, as_base_slabs: str, chunk_size: str) -> None:
        self.arr = buffer()

    def time_from_array(self, as_base_slabs: str, chunk_size: str) -> None:
        _ = StagedChangesArray.from_array(
            self.arr,
            chunk_size=CHUNK_SIZES[chunk_size],
            as_base_slabs=as_base_slabs == "base",
        )


class TimeLoad(_MutatingBenchmark):
    """Benchmark LoadPlan creation and execution.

    Only the chunks that lie on the base slabs are loaded, so this is a no-op for all
    the states other than "base".
    """

    params = [["base", "staged"], list(CHUNK_SIZES)]
    param_names = ["state", "chunk_size"]

    def setup(self, state: str, chunk_size: str) -> None:
        self.arr = make_array(state, CHUNK_SIZES[chunk_size])

    def time_load_plan(self, state: str, chunk_size: str) -> None:
        self.arr._load_plan(copy=False)

    def time_load(self, state: str, chunk_size: str) -> None:
        # Internally calls _load_plan() and then executes the plan
        self.arr.load()


class TimeChanges:
    """Benchmark ChangesPlan creation and execution.

    Chunks that lie on the base slabs are yielded as slices, whereas the staged ones
    are yielded as numpy arrays; these are the two states that matter here.
    """

    params = [["base", "staged"], list(CHUNK_SIZES)]
    param_names = ["state", "chunk_size"]

    def setup(self, state: str, chunk_size: str) -> None:
        self.arr = make_array(state, CHUNK_SIZES[chunk_size])

    def time_changes_plan(self, state: str, chunk_size: str) -> None:
        self.arr._changes_plan()

    def time_changes(self, state: str, chunk_size: str) -> None:
        # Internally calls _changes_plan() and then executes the plan
        for _ in self.arr.changes():
            pass


class TimeCommit(_MutatingBenchmark):
    """Benchmark HashPlan creation and execution, and the whole of commit()"""

    params = [COMMIT_SCENARIOS, list(CHUNK_SIZES)]
    param_names = ["scenario", "chunk_size"]

    def setup(self, scenario: str, chunk_size: str) -> None:
        self.arr = make_committable(scenario, CHUNK_SIZES[chunk_size])

    def time_hash_plan(self, scenario: str, chunk_size: str) -> None:
        self.arr._hash_plan()

    def time_calc_hashes(self, scenario: str, chunk_size: str) -> None:
        # Internally calls _hash_plan() then executes the plan
        self.arr._calc_hashes()

    def time_commit(self, scenario: str, chunk_size: str) -> None:
        # Internally calls _calc_hashes(), then calls
        # _commit_plan() and executes the plan
        self.arr.commit()


class TimeCommitPlan:
    """Benchmark CommitPlan creation, which deduplicates the staged chunks against the
    hashes of the base and full chunks. Unlike the other benchmarks in TimeCommit, it
    requires all the hashes to be already up to date.
    """

    params = [COMMIT_SCENARIOS, list(CHUNK_SIZES)]
    param_names = ["scenario", "chunk_size"]

    def setup(self, scenario: str, chunk_size: str) -> None:
        self.arr = make_committable(scenario, CHUNK_SIZES[chunk_size])
        self.arr._calc_hashes()

    def time_commit_plan(self, scenario: str, chunk_size: str) -> None:
        self.arr._commit_plan(copy=False)
