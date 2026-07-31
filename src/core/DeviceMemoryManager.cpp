/// @file DeviceMemoryManager.cpp
/// @brief Implementation of DeviceMemoryManager.

#include "DeviceMemoryManager.hpp"

#include <vector>

DeviceMemoryManager::DeviceMemoryManager(occa::device &device)
    : device_(device),
      has_separate_(device.hasSeparateMemorySpace())
{
}

// ---------------------------------------------------------------------------
// wrapOrMalloc (from host pointer)
// ---------------------------------------------------------------------------

occa::memory DeviceMemoryManager::wrapOrMalloc(
    const Real *src, occa::dim_t entries)
{
    if (has_separate_) {
        // GPU backend: allocate device memory on the device.
        // If src is provided, use it to initialise the allocation.
        return src
            ? device_.malloc<Real>(entries, src)
            : device_.malloc<Real>(entries);
    } else {
        // CPU backend: wrap existing host pointer.
        // Caller must ensure src lifetime exceeds all kernel launches.
        return device_.wrapMemory<Real>(src, entries);
    }
}

// ---------------------------------------------------------------------------
// wrapOrMalloc (zero-initialised)
// ---------------------------------------------------------------------------

occa::memory DeviceMemoryManager::wrapOrMalloc(occa::dim_t entries)
{
    // No host pointer to wrap; allocate fresh memory and explicitly zero it.
    // OCCA's malloc does not guarantee zero-initialisation, so we zero-fill
    // via a host-side buffer regardless of the backend.
    std::vector<Real> zeros(static_cast<std::size_t>(entries), Real(0));
    occa::memory mem = device_.malloc<Real>(entries, zeros.data());
    return mem;
}

// ---------------------------------------------------------------------------
// copyToHost
// ---------------------------------------------------------------------------

void DeviceMemoryManager::copyToHost(
    occa::memory &o_data, Real *dst, occa::dim_t entries)
{
    if (has_separate_) {
        o_data.copyTo(dst, entries);
    }
    // Unified space: same address space, no copy needed.
}

// ---------------------------------------------------------------------------
// copyFromHost
// ---------------------------------------------------------------------------

void DeviceMemoryManager::copyFromHost(
    occa::memory &o_data, const Real *src, occa::dim_t entries)
{
    if (has_separate_) {
        o_data.copyFrom(src, entries);
    }
    // Unified space: same address space, no copy needed.
}
