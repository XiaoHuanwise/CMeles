/// @file MathOps.hpp
/// @brief Supplementary math kernels (element-wise ops, etc.) beyond
///        standard BLAS 1-3.
///
/// These kernels live in `src/math/okl/` and share the same OCCA compile
/// conventions as the BLAS kernels (`Real`/`TILE_SIZE` JIT defines).

#pragma once

#include <occa.hpp>

#include <string>

#include "common/KernelProps.hpp"
#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"

/// @brief Supplementary OCCA math kernels.
class MathOps {
public:
    MathOps(occa::device &device, DeviceMemoryManager &mem,
            const std::string &oklDir = OCCA_OKL_DIR);
    ~MathOps() = default;

    /// Rebuild affected kernels with a new tile size (recompiles via JIT).
    void setTileSize(int tileSize);

    // ---- Element-wise ------------------------------------------------------

    /// z = x .* y  (Hadamard / element-wise product)
    void vmul(occa::dim_t n, occa::memory &x, occa::memory &y, occa::memory &z);

    // ---- Diagnostics -------------------------------------------------------

    /// @brief Count NaN entries in \p x (self-inequality test $x \neq x$).
    ///
    /// Synchronises the device: the kernel writes one NaN count per tile
    /// group and the host sums the partials. Intended for once-per-step
    /// sanity checks, not per-RHS-evaluation hot paths.
    /// @return The number of NaN entries (0 when \p n is 0).
    Real countNaN(occa::dim_t n, occa::memory &x);

private:
    occa::kernel buildKernel(const std::string &file, const std::string &name);

    /// Grow the NaN-partial scratch to at least \p groups entries (lazy).
    void ensureNanScratch(occa::dim_t groups);

    occa::device &device_;
    DeviceMemoryManager &mem_;
    std::string oklDir_;
    int tileSize_ = cmeles::DefaultTileSize;

    occa::kernel vmul_;
    occa::kernel countNaN_;

    // NaN-count partials (device-resident, grown on demand; fully
    // overwritten by each launch, so no zero-fill is needed).
    occa::memory o_nanPartial_;
    occa::dim_t nanPartialCap_ = 0;
};
