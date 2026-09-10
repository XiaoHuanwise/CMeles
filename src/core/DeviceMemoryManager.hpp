/// @file DeviceMemoryManager.hpp
/// @brief Unified OCCA memory management, abstracting backend memory model
/// differences.
///
/// Reference: docs/tech_docs/occa.md

#pragma once

#include <occa.hpp>

#include "common/Types.hpp"

/// @brief Unified OCCA memory management interface.
///
/// Wraps the backend differences between unified memory space (Serial/OpenMP)
/// and separate memory space (CUDA/HIP/OpenCL/Metal) backends:
///
///  - unified space: `wrapMemory` wraps existing host pointers (zero-copy)
///  - separate space: `malloc` allocates device memory (explicit copies needed)
class DeviceMemoryManager {
public:
    /// @brief Construct from a device reference.
    ///
    /// Queries `hasSeparateMemorySpace()` at construction time; the result
    /// is immutable for the lifetime of this object.
    explicit DeviceMemoryManager(occa::device &device);

    /// @brief Return the underlying `occa::device` reference.
    occa::device &device() {
        return device_;
    }

    /// @brief Unified data creation.
    ///
    /// If `hasSeparateMemorySpace()` is true, allocates device memory via
    /// `device.malloc`, optionally initializing from $src$.
    /// Otherwise, wraps the host pointer via `device.wrapMemory`.
    ///
    /// @param src     Host source pointer (may be `nullptr` for uninitialised
    /// allocation).
    /// @param entries Number of elements of type `Real`.
    occa::memory wrapOrMalloc(const Real *src, occa::dim_t entries);

    /// @brief Zero-initialised unified data creation.
    ///
    /// Always uses `device.malloc` since there is no host pointer to wrap.
    occa::memory wrapOrMalloc(occa::dim_t entries);

    /// @brief Device-to-host copy.
    ///
    /// If `hasSeparateMemorySpace()` is true, performs `o_data.copyTo(dst,
    /// ...)`. Otherwise no-op (same address space).
    void copyToHost(occa::memory &o_data, Real *dst, occa::dim_t entries);

    /// @brief Host-to-device copy.
    ///
    /// If `hasSeparateMemorySpace()` is true, performs `o_data.copyFrom(src,
    /// ...)`. Otherwise no-op (same address space).
    void copyFromHost(occa::memory &o_data, const Real *src,
                      occa::dim_t entries);

    // ---- int-typed variants (for topology/adjacency arrays) ----

    /// @brief Unified data creation for \c int arrays (e.g. element-face
    ///        connectivity, face neighbour indices).
    ///
    /// Same semantics as the \c Real overloads: separate memory space uses
    /// `malloc<int>`, unified space wraps the host pointer via
    /// `wrapMemory<int>` (zero-copy).
    ///
    /// @param src     Host source pointer (may be `nullptr` for uninitialised
    /// allocation).
    /// @param entries Number of elements of type `int`.
    occa::memory wrapOrMallocInt(const int *src, occa::dim_t entries);

    /// @brief Device-to-host copy for \c int arrays.
    ///
    /// If `hasSeparateMemorySpace()` is true, performs `o_data.copyTo(dst,
    /// ...)`. Otherwise no-op (same address space).
    void copyToHost(occa::memory &o_data, int *dst, occa::dim_t entries);

    /// @brief Host-to-device copy for \c int arrays.
    ///
    /// If `hasSeparateMemorySpace()` is true, performs `o_data.copyFrom(src,
    /// ...)`. Otherwise no-op (same address space).
    void copyFromHost(occa::memory &o_data, const int *src,
                      occa::dim_t entries);

    /// @brief Whether the backend uses a separate memory space.
    bool hasSeparateMemorySpace() const {
        return has_separate_;
    }

private:
    occa::device &device_;
    bool has_separate_;
};
