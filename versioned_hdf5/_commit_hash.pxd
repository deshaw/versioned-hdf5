# C++ declarations backing staged_changes.CommitPlan.
# See _commit_hash.hpp for the C++ implementations.
#
# This pxd grants Cython access to two std::unordered_map instantiations:
#
#   ChunkHashMap : {ChunkHash(sha256 4x uint64)    : ChunkLoc(slab_idx, slab_offset)}
#   ChunkLocMap  : {ChunkLoc(slab_idx, slab_offset): ChunkLoc(slab_idx, slab_offset)}

from libc.stdint cimport uint64_t
from libcpp.unordered_map cimport unordered_map

cdef extern from "_commit_hash.hpp" namespace "versioned_hdf5":
    # 4-word SHA256 digest used as a map key.
    # No std::array hashing is needed on the Cython side.
    cdef cppclass ChunkHash:
        ChunkHash() except +
        ChunkHash(uint64_t, uint64_t, uint64_t, uint64_t) except +
        ChunkHash(const ChunkHash&) except +
        bint operator==(const ChunkHash&) except +

    cdef cppclass ChunkHashHash:
        ChunkHashHash() except +

    # (slab_idx, slab_offset) pair
    cdef cppclass ChunkLoc:
        uint64_t slab_idx
        uint64_t slab_offset
        ChunkLoc() except +
        ChunkLoc(uint64_t, uint64_t) except +
        ChunkLoc(const ChunkLoc&) except +
        bint operator==(const ChunkLoc&) except +

    cdef cppclass ChunkLocHash:
        ChunkLocHash() except +


# Concrete map typedefs cimported by staged_changes.
ctypedef unordered_map[ChunkHash, ChunkLoc, ChunkHashHash] ChunkHashMap
ctypedef unordered_map[ChunkLoc, ChunkLoc, ChunkLocHash] ChunkLocMap
