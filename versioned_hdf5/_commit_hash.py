"""Pure-Python fallback for C++ ChunkHash/ChunkLoc/ChunkHashMap/ChunkLocMap.

Matches the API of _commit_hash.pxd.
Used when versioned_hdf5.staged_changes runs uncompiled (cython.compiled is False).
"""

from __future__ import annotations

from typing import NamedTuple


class ChunkHash(NamedTuple):
    """4-word SHA256 digest"""

    h0: int
    h1: int
    h2: int
    h3: int


class ChunkLoc(NamedTuple):
    slab_idx: int
    slab_offset: int


class ChunkHashMap(dict[ChunkHash, ChunkLoc]):
    def count(self, key: ChunkHash) -> int:  # API compatibility with std::unordered_map
        return 1 if key in self else 0  # pragma: no cover


class ChunkLocMap(dict[ChunkLoc, ChunkLoc]):
    def count(self, key: ChunkLoc) -> int:
        return 1 if key in self else 0  # pragma: no cover
