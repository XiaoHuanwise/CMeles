/// @file StepperBase.hpp
/// @brief CRTP base class and shared types for the time-marching module.
///
/// The stepper family uses *compile-time* polymorphism (CRTP) per the
/// project's static-polymorphism guideline: the concrete stepper type is
/// resolved at the solver's dispatch point
/// (runCompressibleFlowSolver), so the per-step advance() call carries no
/// virtual dispatch.
///
/// Shared context held by the base: the OCCA device / memory manager, the
/// right-hand-side callable $\mathcal{R}(u)$ (boundary conditions live
/// inside the DG residual, so they are re-applied at every RK stage), the
/// positivity-preserving limiter hook (identity by default, invoked after
/// every stage state — ported from the reference prototype), the vector
/// update kernels of okl/time_update.okl and a Blas instance for the
/// axpy/copy/reduction glue.
///
/// Reference: docs/tech_docs/time_marching/ and the PyTorch prototype
/// .stepper.py (algorithmic reference only; the C++ implementation adapts
/// the structure to OCCA device buffers).

#pragma once

#include <occa.hpp>

#include <functional>
#include <string>
#include <vector>

#include "blas/Blas.hpp"
#include "common/KernelProps.hpp"
#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "time/TimeTypes.hpp"

/// @brief CRTP base for all time steppers.
///
/// Derived classes implement:
///   - `Real advanceImpl(occa::memory o_u, Real t, Real dt)` — advance the
///     state in place by one step and return the physical dt consumed
///     (fixed-dt schemes: dt itself; adaptive schemes: the accepted step),
///   - `int orderImpl() const` and `const char *nameImpl() const`.
template <class Derived> class StepperBase
{
public:
    /// @param device  OCCA device (lifetime must exceed this object).
    /// @param mem     Device memory manager (lifetime must exceed this
    ///                object).
    /// @param rhs     Right-hand side $\mathcal{R}(u)$.
    /// @param nDof    Length of the ODE vector.
    /// @param oklDir  Directory containing the .okl kernel sources.
    StepperBase(occa::device &device, DeviceMemoryManager &mem, RhsFunction rhs,
                occa::dim_t nDof, const std::string &oklDir = OCCA_OKL_DIR)
        : device_(device), mem_(mem), rhs_(std::move(rhs)), nDof_(nDof),
          oklDir_(oklDir), blas_(device, mem, oklDir)
    {
    }

    /// @brief Advance the state by one time step in place.
    /// @return The physical time step consumed.
    Real advance(occa::memory o_u, Real t, Real dt)
    {
        return static_cast<Derived *>(this)->advanceImpl(o_u, t, dt);
    }

    /// @brief Formal order of the time integrator.
    int order() const
    {
        return static_cast<const Derived *>(this)->orderImpl();
    }

    /// @brief Canonical method name.
    const char *name() const
    {
        return static_cast<const Derived *>(this)->nameImpl();
    }

    /// @brief Install a positivity-preserving limiter (no-op by default).
    void setPositivityLimiter(PositivityLimiter limiter)
    {
        limiter_ = std::move(limiter);
    }

protected:
    // ---- Shared kernel helpers
    // ------------------------------------------------

    /// @brief Build a kernel from time_update.okl with the standard JIT
    ///        props (Real / TILE_SIZE defines).
    occa::kernel buildTimeKernel(const std::string &name) const
    {
        occa::json props;
#ifdef USE_FLOAT_PRECISION
        props["defines/Real"] = "float";
#else
        props["defines/Real"] = "double";
#endif
        props["defines/TILE_SIZE"] = 256;
        cmeles::finaliseKernelProps(props, device_);
        return device_.buildKernel(oklDir_ + "/time_update.okl", name, props);
    }

    /// @brief u_np1 = u_n + dt * res_n (per-DOF aliasing is safe).
    void eulerUpdate(occa::memory &o_un, Real dt, occa::memory &o_res,
                     occa::memory &o_unp1)
    {
        explicitEulerUpdate_(static_cast<int>(nDof_), dt, o_un, o_res, o_unp1);
    }

    /// @brief u_new = alpha * u_n + beta * u_temp (per-DOF aliasing safe).
    void convexCombine(Real alpha, Real beta, occa::memory &o_un,
                       occa::memory &o_utemp, occa::memory &o_unew)
    {
        sspConvexCombine_(static_cast<int>(nDof_), alpha, beta, o_un, o_utemp,
                          o_unew);
    }

    /// @brief out = c0*x0 + c1*x1 + c2*x2 + c3*x3 (zero-coefficient terms
    ///        may alias any valid buffer).
    void vecCombine4(Real c0, occa::memory &o_x0, Real c1, occa::memory &o_x1,
                     Real c2, occa::memory &o_x2, Real c3, occa::memory &o_x3,
                     occa::memory &o_out)
    {
        vecCombine4_(static_cast<int>(nDof_), c0, o_x0, c1, o_x1, c2, o_x2, c3,
                     o_x3, o_out);
    }

    /// @brief Apply the positivity limiter to a stage state (no-op unless
    ///        installed).
    void applyLimiter(occa::memory &o_u)
    {
        if (limiter_)
        {
            limiter_(o_u);
        }
    }

    /// @brief $\|x\|_\infty$ over \p n entries (host reduction; grows the
    ///        scratch lazily, so it also serves stacked dual-time states).
    Real infNorm(occa::memory &o_x, occa::dim_t n)
    {
        if (o_abs_.size() < n)
        {
            o_abs_ = mem_.wrapOrMalloc(n);
            absHost_.resize(n);
        }
        occa::memory oxa = o_x;
        absInto_(static_cast<int>(n), oxa, o_abs_);
        if (mem_.hasSeparateMemorySpace())
        {
            occa::memory oa = o_abs_;
            mem_.copyToHost(oa, absHost_.data(), n);
        }
        else
        {
            const Real *src = o_abs_.ptr<Real>();
            std::copy(src, src + static_cast<std::size_t>(n), absHost_.begin());
        }
        Real m = Real(0);
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
        {
            m = std::max(m, absHost_[i]);
        }
        return m;
    }

    // ---- Shared context
    // --------------------------------------------------------

    occa::device &device_;
    DeviceMemoryManager &mem_;
    RhsFunction rhs_;
    occa::dim_t nDof_;
    std::string oklDir_;
    Blas blas_;
    PositivityLimiter limiter_;

    // Lazy-compiled kernels (built on first use by the derived classes).
    occa::kernel explicitEulerUpdate_;
    occa::kernel sspConvexCombine_;
    occa::kernel rkStageCombine_;
    occa::kernel rkWeightedSum_;
    occa::kernel rkFinalUpdate_;
    occa::kernel rkScaledErrorSq_;
    occa::kernel vecCombine4_;
    occa::kernel absInto_;
    bool kernelsBuilt_ = false;

    /// @brief Lazily compile the shared kernels (idempotent).
    void ensureKernels()
    {
        if (kernelsBuilt_)
        {
            return;
        }
        explicitEulerUpdate_ = buildTimeKernel("explicitEulerUpdate");
        sspConvexCombine_    = buildTimeKernel("sspConvexCombine");
        rkStageCombine_      = buildTimeKernel("rkStageCombine");
        rkWeightedSum_       = buildTimeKernel("rkWeightedSum");
        rkFinalUpdate_       = buildTimeKernel("rkFinalUpdate");
        rkScaledErrorSq_     = buildTimeKernel("rkScaledErrorSq");
        vecCombine4_         = buildTimeKernel("vecCombine4");
        absInto_             = buildTimeKernel("absInto");
        kernelsBuilt_        = true;
    }

private:
    occa::memory o_abs_;        ///< |x| scratch for the inf-norm reduction.
    std::vector<Real> absHost_; ///< Host mirror / wrap source of o_abs_.
};
