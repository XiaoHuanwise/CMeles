/// @file MathOps.cpp
/// @brief Implementation of MathOps — supplementary math kernel wrappers.

#include "math/MathOps.hpp"

#include <vector>

#include "common/KernelProps.hpp"

// ============================================================================
// Construction
// ============================================================================

MathOps::MathOps(occa::device &device, DeviceMemoryManager &mem,
                 const std::string &oklDir)
    : device_(device), mem_(mem), oklDir_(oklDir) {
}

// ============================================================================
// Kernel compilation (lazy, cached)
// ============================================================================

occa::kernel MathOps::buildKernel(const std::string &file,
                                  const std::string &name) {
    occa::json props;
#ifdef USE_FLOAT_PRECISION
    props["defines/Real"] = "float";
#else
    props["defines/Real"] = "double";
#endif
    props["defines/TILE_SIZE"] = tileSize_;
    cmeles::finaliseKernelProps(props, device_);

    return device_.buildKernel(oklDir_ + "/" + file, name, props);
}

void MathOps::setTileSize(int tileSize) {
    tileSize_ = tileSize;
    vmul_     = occa::kernel();
    countNaN_ = occa::kernel();
}

// ============================================================================
// Element-wise
// ============================================================================

void MathOps::vmul(occa::dim_t n, occa::memory &x, occa::memory &y,
                   occa::memory &z) {
    if (!vmul_.isInitialized())
        vmul_ = buildKernel("elemwise.okl", "vmul");
    vmul_(static_cast<int>(n), x, y, z);
}

// ============================================================================
// Diagnostics
// ============================================================================

void MathOps::ensureNanScratch(occa::dim_t groups) {
    if (nanPartialCap_ >= groups)
        return;

    // Allocate directly via device (not wrapOrMalloc) — scratch is
    // device-resident and fully overwritten by the kernel, so no zero-fill
    // is needed (same rationale as Blas::ensureScratch).
    o_nanPartial_  = device_.malloc<Real>(groups);
    nanPartialCap_ = groups;
}

Real MathOps::countNaN(occa::dim_t n, occa::memory &x) {
    if (n == 0)
        return Real(0);

    if (!countNaN_.isInitialized())
        countNaN_ = buildKernel("nan_check.okl", "countNaN");

    const occa::dim_t groups = (n + tileSize_ - 1) / tileSize_;
    ensureNanScratch(groups);
    countNaN_(static_cast<int>(groups), static_cast<int>(n), x, o_nanPartial_);

    // Final combine on the host: the partial count is small relative to the
    // scanned field and this is already the synchronisation point.
    std::vector<Real> partials(static_cast<std::size_t>(groups));
    o_nanPartial_.copyTo(partials.data());
    Real count = Real(0);
    for (const Real p : partials) {
        count += p;
    }
    return count;
}
