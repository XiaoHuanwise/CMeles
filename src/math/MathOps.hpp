/// @file MathOps.hpp
/// @brief Supplementary math kernels (element-wise ops, etc.) beyond
///        standard BLAS 1-3.
///
/// These kernels live in `src/math/okl/` and share the same OCCA compile
/// conventions as the BLAS kernels (`Real`/`TILE_SIZE` JIT defines).

#pragma once

#include <occa.hpp>

#include <string>

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

private:
    occa::kernel buildKernel(const std::string &file, const std::string &name);

    occa::device &device_;
    DeviceMemoryManager &mem_;
    std::string oklDir_;
    int tileSize_ = 256;

    occa::kernel vmul_;
};
