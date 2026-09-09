/// @file Blas.hpp
/// @brief OCCA-backed BLAS 1-3 kernels with uniform backend paths.
///
/// All kernels operate on device-resident data through `occa::memory`
/// handles.  Element-wise and matrix operations are fire-and-forget;
/// reductions synchronise the device and return a host scalar.
///
/// @note `Real` and `TILE_SIZE` are injected as JIT defines via
///       `occa::json` properties so that the same .okl source works
///       across float/double and tile-size configurations.

#pragma once

#include <occa.hpp>

#include <string>

#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"

/// @brief OCCA-backed BLAS 1-3 kernels over \c Real with uniform backend paths.
class Blas
{
public:
    /// @brief Construct a Blas instance bound to a device and memory manager.
    /// @param device  OCCA device (lifetime must exceed this object).
    /// @param mem     Device memory manager (lifetime must exceed this object).
    /// @param oklDir  Directory containing the .okl source files.
    ///                 Defaults to the `OCCA_OKL_DIR` compile definition.
    Blas(occa::device &device, DeviceMemoryManager &mem,
         const std::string &oklDir = OCCA_OKL_DIR);

    ~Blas() = default;

    /// @brief Change tile size and recompile affected kernels via JIT.
    /// @param tileSize New tile size (must be a compile-time constant for
    /// @tile).
    void setTileSize(int tileSize);

    // ---- BLAS-1 element-wise -----------------------------------------------

    /// x = alpha * x
    void scal(occa::dim_t n, Real alpha, occa::memory &x);

    /// y = alpha * x + y
    void axpy(occa::dim_t n, Real alpha, occa::memory &x, occa::memory &y);

    /// y = x
    void copy(occa::dim_t n, occa::memory &x, occa::memory &y);

    // ---- BLAS-1 reductions (synchronise + return host scalar) --------------

    /// @return x^T y
    Real dot(occa::dim_t n, occa::memory &x, occa::memory &y);

    /// @return ||x||_2
    Real nrm2(occa::dim_t n, occa::memory &x);

    /// @return ||x||_1  (sum of absolute values)
    Real asum(occa::dim_t n, occa::memory &x);

    /// @return max|x|  (infinity norm of x)
    Real amax(occa::dim_t n, occa::memory &x);

    // ---- BLAS-2 ------------------------------------------------------------

    /// y = alpha * A * x + beta * y
    /// @param A m×n row-major matrix
    void gemv(occa::dim_t m, occa::dim_t n, Real alpha, occa::memory &A,
              occa::memory &x, Real beta, occa::memory &y);

    /// A = alpha * x * y^T + A
    /// @param A m×n row-major matrix (in/out)
    void ger(occa::dim_t m, occa::dim_t n, Real alpha, occa::memory &x,
             occa::memory &y, occa::memory &A);

    // ---- BLAS-3 ------------------------------------------------------------

    /// C = alpha * A * B + beta * C
    /// @param A m×k row-major
    /// @param B k×n row-major
    /// @param C m×n row-major (in/out)
    void gemm(occa::dim_t m, occa::dim_t n, occa::dim_t k, Real alpha,
              occa::memory &A, occa::memory &B, Real beta, occa::memory &C);

private:
    /// Build (or return cached) kernel from a .okl file.
    occa::kernel buildKernel(const std::string &file, const std::string &name);

    /// Multi-level reduction pipeline: run Level-0 kernel (which writes
    /// partials into `o_partial_a_`), then ping-pong `sumReduce` until
    /// ≤ kHostReduceMax, then host-finalise.
    /// @tparam FinalOp  void(Real &acc) — e.g. acc = std::sqrt(acc) for nrm2
    template <typename FinalOp>
    Real finalizeReduction(occa::dim_t count, FinalOp finalOp);

    /// Multi-level max-reduction pipeline: run amax (which writes partial
    /// maxima into `o_partial_a_`), then ping-pong `maxReduce` until
    /// ≤ kHostReduceMax, then host-finalise with max.
    Real finalizeMaxReduction(occa::dim_t count);

    /// Grow scratch buffers to at least `capacity` elements (lazy / reuse).
    void ensureScratch(occa::dim_t capacity);

    occa::device &device_;
    DeviceMemoryManager &mem_;
    std::string oklDir_;
    int tileSize_ = 256;

    /// Threshold for switching from device-side multi-level reduce to host sum.
    static constexpr occa::dim_t kHostReduceMax = 4096;

    // Cached kernel handles (lazy-compiled on first use).
    occa::kernel scal_;
    occa::kernel axpy_;
    occa::kernel copy_;
    occa::kernel dot_;
    occa::kernel nrm2_;
    occa::kernel asum_;
    occa::kernel amax_;
    occa::kernel sumReduce_;
    occa::kernel maxReduce_;
    occa::kernel gemv_;
    occa::kernel ger_;
    occa::kernel gemm_;

    // Reduction scratch (device-resident, grown on demand, ping-pong pair).
    occa::memory o_partial_a_;
    occa::memory o_partial_b_;
    occa::dim_t scratchCap_ = 0;
};
