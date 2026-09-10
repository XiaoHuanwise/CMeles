/// @file MathOps.cpp
/// @brief Implementation of MathOps — supplementary math kernel wrappers.

#include "math/MathOps.hpp"

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
