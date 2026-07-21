// Custom hash functors and key types for the C++ std::unordered_map used by
// staged_changes.CommitPlan.
//
// C++ std::hash has no specialization for std::array, std::pair, or std::tuple,
// so we provide trivial custom hashers here. This keeps the Cython side free of
// C++ boilerplate: staged_changes only has to declare the instantiated
// unordered_map typedefs.
//
// This header has no dependencies beyond the C++ standard library.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <utility>

namespace versioned_hdf5 {

// Key for the {sha256 -> (slab_idx, slab_offset)} map.
// A SHA256 digest is 4 x uint64 = 32 bytes. The digest is already a
// cryptographically strong, uniformly distributed hash, so we can feed any one
// of its words straight into std::hash<uint64_t> instead of hashing all four.
struct ChunkHash {
    std::array<uint64_t, 4> words;

    ChunkHash() = default;
    ChunkHash(uint64_t h0, uint64_t h1, uint64_t h2, uint64_t h3) noexcept {
        words[0] = h0;
        words[1] = h1;
        words[2] = h2;
        words[3] = h3;
    }

    bool operator==(const ChunkHash& other) const noexcept {
        return words == other.words;
    }
};

struct ChunkHashHash {
    std::size_t operator()(const ChunkHash& key) const noexcept {
        // words[0] is uniformly distributed (it's a SHA256 word), so a single
        // pass through std::hash<uint64_t> is sufficient and collision resistant.
        return std::hash<uint64_t>{}(key.words[0]);
    }
};

// Key for the {(slab_idx, slab_offset) -> (slab_idx, slab_offset)} map.
struct ChunkLoc {
    uint64_t slab_idx;
    uint64_t slab_offset;

    ChunkLoc() = default;
    ChunkLoc(uint64_t idx, uint64_t off) noexcept
        : slab_idx(idx), slab_offset(off) {}

    bool operator==(const ChunkLoc& other) const noexcept {
        return slab_idx == other.slab_idx && slab_offset == other.slab_offset;
    }
};

struct ChunkLocHash {
    std::size_t operator()(const ChunkLoc& key) const noexcept {
        std::size_t h = std::hash<uint64_t>{}(key.slab_offset);
        return h ^ (std::hash<uint64_t>{}(key.slab_idx) << 1u);
    }
};

}  // namespace versioned_hdf5
