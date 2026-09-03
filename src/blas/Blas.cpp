/// @file Blas.cpp
/// @brief Implementation of the Blas class — kernel compilation, argument
///        binding, execution, and multi-level reduction orchestration.

#include "blas/Blas.hpp"
#include "common/KernelProps.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// ============================================================================
// Construction
// ============================================================================

Blas::Blas(occa::device &device, DeviceMemoryManager &mem,
           const std::string &oklDir)
    : device_(device), mem_(mem), oklDir_(oklDir)
{
}

// ============================================================================
// Kernel compilation (lazy, cached)
// ============================================================================

occa::kernel Blas::buildKernel(const std::string &file, const std::string &name)
{
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

void Blas::setTileSize(int tileSize)
{
    tileSize_ = tileSize;
    // Invalidate all cached kernels so they are recompiled on next use.
    scal_      = occa::kernel();
    axpy_      = occa::kernel();
    copy_      = occa::kernel();
    dot_       = occa::kernel();
    nrm2_      = occa::kernel();
    asum_      = occa::kernel();
    sumReduce_ = occa::kernel();
    gemv_      = occa::kernel();
    ger_       = occa::kernel();
    gemm_      = occa::kernel();
}

// ============================================================================
// Scratch buffer management (reduction pipeline)
// ============================================================================

void Blas::ensureScratch(occa::dim_t capacity)
{
    if (scratchCap_ >= capacity)
        return;

    // Allocate directly via device (not wrapOrMalloc) — scratch is
    // device-resident and fully overwritten by kernels, so no zero-fill needed.
    o_partial_a_ = device_.malloc<Real>(capacity);
    o_partial_b_ = device_.malloc<Real>(capacity);
    scratchCap_  = capacity;
}

// ============================================================================
// BLAS-1 element-wise
// ============================================================================

void Blas::scal(occa::dim_t n, Real alpha, occa::memory &x)
{
    if (!scal_.isInitialized())
        scal_ = buildKernel("blas1.okl", "scal");
    scal_(static_cast<int>(n), alpha, x);
}

void Blas::axpy(occa::dim_t n, Real alpha, occa::memory &x, occa::memory &y)
{
    if (!axpy_.isInitialized())
        axpy_ = buildKernel("blas1.okl", "axpy");
    axpy_(static_cast<int>(n), alpha, x, y);
}

void Blas::copy(occa::dim_t n, occa::memory &x, occa::memory &y)
{
    if (!copy_.isInitialized())
        copy_ = buildKernel("blas1.okl", "copy");
    copy_(static_cast<int>(n), x, y);
}

// ============================================================================
// Multi-level reduction pipeline
// ============================================================================

template <typename FinalOp>
Real Blas::finalizeReduction(occa::dim_t count, FinalOp finalOp)
{
    // Level-1+: device-side ping-pong sumReduce
    occa::memory *src = &o_partial_a_;
    occa::memory *dst = &o_partial_b_;

    while (count > kHostReduceMax)
    {
        const occa::dim_t next = (count + tileSize_ - 1) / tileSize_;
        ensureScratch(next);
        sumReduce_(static_cast<int>(count), *src, *dst);
        std::swap(src, dst);
        count = next;
    }

    // Level-2: copy remaining partials to host and sum.
    // Use raw occa::memory::copyTo (not DeviceMemoryManager::copyToHost)
    // because scratch was allocated via device_.malloc, not wrapOrMalloc,
    // so the DeviceMemoryManager no-op path does not apply.
    std::vector<Real> host(static_cast<std::size_t>(count));
    src->copyTo(host.data());

    Real acc = Real(0);
    for (occa::dim_t i = 0; i < count; ++i)
    {
        acc += host[i];
    }
    finalOp(acc);
    return acc;
}

// ---- Reduction entry points ----

Real Blas::dot(occa::dim_t n, occa::memory &x, occa::memory &y)
{
    if (n == 0)
        return Real(0);

    if (!dot_.isInitialized())
        dot_ = buildKernel("reduce.okl", "dot");
    if (!sumReduce_.isInitialized())
        sumReduce_ = buildKernel("reduce.okl", "sumReduce");

    const occa::dim_t count0 = (n + tileSize_ - 1) / tileSize_;
    ensureScratch(count0);
    dot_(static_cast<int>(n), x, y, o_partial_a_);

    return finalizeReduction(count0, [](Real & /*acc*/) { /* identity */ });
}

Real Blas::nrm2(occa::dim_t n, occa::memory &x)
{
    if (n == 0)
        return Real(0);

    if (!nrm2_.isInitialized())
        nrm2_ = buildKernel("reduce.okl", "nrm2");
    if (!sumReduce_.isInitialized())
        sumReduce_ = buildKernel("reduce.okl", "sumReduce");

    const occa::dim_t count0 = (n + tileSize_ - 1) / tileSize_;
    ensureScratch(count0);
    nrm2_(static_cast<int>(n), x, o_partial_a_);

    return finalizeReduction(count0, [](Real &acc) { acc = std::sqrt(acc); });
}

Real Blas::asum(occa::dim_t n, occa::memory &x)
{
    if (n == 0)
        return Real(0);

    if (!asum_.isInitialized())
        asum_ = buildKernel("reduce.okl", "asum");
    if (!sumReduce_.isInitialized())
        sumReduce_ = buildKernel("reduce.okl", "sumReduce");

    const occa::dim_t count0 = (n + tileSize_ - 1) / tileSize_;
    ensureScratch(count0);
    asum_(static_cast<int>(n), x, o_partial_a_);

    return finalizeReduction(count0, [](Real & /*acc*/) { /* identity */ });
}

// ============================================================================
// BLAS-2
// ============================================================================

void Blas::gemv(occa::dim_t m, occa::dim_t n, Real alpha, occa::memory &A,
                occa::memory &x, Real beta, occa::memory &y)
{
    if (!gemv_.isInitialized())
        gemv_ = buildKernel("blas2.okl", "gemv");
    gemv_(static_cast<int>(m), static_cast<int>(n), alpha, beta, A, x, y);
}

void Blas::ger(occa::dim_t m, occa::dim_t n, Real alpha, occa::memory &x,
               occa::memory &y, occa::memory &A)
{
    if (!ger_.isInitialized())
        ger_ = buildKernel("blas2.okl", "ger");
    ger_(static_cast<int>(m), static_cast<int>(n), alpha, x, y, A);
}

// ============================================================================
// BLAS-3
// ============================================================================

void Blas::gemm(occa::dim_t m, occa::dim_t n, occa::dim_t k, Real alpha,
                occa::memory &A, occa::memory &B, Real beta, occa::memory &C)
{
    // Guard against int32 overflow in the 1-D flattened index m*n.
    // The 1-D gemm uses int idx; reject cases where m*n exceeds INT_MAX.
    if (m > 0 && n > 0 &&
        static_cast<std::size_t>(m) * static_cast<std::size_t>(n) >
            static_cast<std::size_t>(INT_MAX))
    {
        throw std::runtime_error(
            "gemm: m*n exceeds INT_MAX — use gemmBlocked for large matrices");
    }

    if (!gemm_.isInitialized())
        gemm_ = buildKernel("gemm.okl", "gemm");
    gemm_(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha,
          beta, A, B, C);
}
