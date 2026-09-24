/// @file StepperBase.hpp
/// @brief Abstract base class and shared types for the time-marching module.
///
/// The stepper family uses ordinary virtual inheritance: the polymorphic
/// surface is coarse — advance() is called once per physical time step and
/// all stage looping is internal to the steppers — so the virtual dispatch
/// cost is negligible next to the per-stage device kernels, while a single
/// non-template solver controller avoids per-stepper template
/// instantiation (compile time and binary size).
///
/// The base carries only what is shared by several steppers: the OCCA
/// device / memory manager / Blas context, the right-hand-side callable
/// $\mathcal{R}(u)$ (boundary conditions live inside the DG residual, so
/// they are re-applied at every RK stage), the shared kernel builder, and
/// the Euler-update kernel. Steppers own the further kernels they launch
/// and build them lazily on first use, so a stepper never JIT-compiles a
/// kernel it does not launch.
///
/// Reference: docs/tech_docs/time_marching/ and the PyTorch prototype
/// .stepper.py (algorithmic reference only; the C++ implementation adapts
/// the structure to OCCA device buffers).

#pragma once

#include <occa.hpp>

#include <string>

#include "blas/Blas.hpp"
#include "common/KernelProps.hpp"
#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "time/TimeTypes.hpp"

/// @brief Abstract base for all time steppers.
///
/// Derived classes implement:
///   - `Real advance(occa::memory o_u, Real t, Real dt)` — advance the
///     state in place by one step and return the physical dt consumed
///     (fixed-dt schemes: dt itself; adaptive schemes: the accepted step),
///   - `int order() const` and `const char *name() const`.
class StepperBase {
public:
    virtual ~StepperBase() = default;

    /// @brief Advance the state by one time step in place.
    /// @return The physical time step consumed.
    virtual Real advance(occa::memory o_u, Real t, Real dt) = 0;

    /// @brief Formal order of the time integrator.
    virtual int order() const = 0;

    /// @brief Canonical method name.
    virtual const char *name() const = 0;

    /// @brief Install a post-stage state filter (e.g. the MR-WENO limiter).
    ///
    /// Filtering contract — binding for every stepper, current and future:
    ///
    ///   1. Every stage state that will subsequently be used as data by an
    ///      RHS evaluation, and the state published to the caller at the
    ///      end of advance(), must have passed through the filter first
    ///      (an SSP framework requirement: the limiter acts on the state
    ///      between the stage update and the next RHS evaluation).
    ///
    ///   2. States internal to an implicit solve iteration (dual-time
    ///      pseudo iterates) must NOT be filtered. The limiter acts as a
    ///      projection on the converged stage state instead:
    ///      $u^{n+1} = \pi(u^*)$ where $u^*$ solves the smooth equation
    ///      $G(u^*; u^n) = 0$ that contains no limiter. Embedding the
    ///      projection inside the iteration would (a) make the iterated
    ///      operator piecewise discontinuous (WENO weights / troubled
    ///      flags switch at O(1) magnitude when iterates cross the KXRCF
    ///      threshold), stalling the residual in a limit cycle, and
    ///      (b) fake convergence (the projection pushes states back into
    ///      the admissible set, lowering the residual norm without solving
    ///      the equation). The equation data ($u^n$, $R(u^n)$, $u^{n-1}$)
    ///      always comes from published — hence already filtered — states,
    ///      so the implicit residual sees limited physics by construction.
    ///
    /// An empty filter (the default) disables limiting.
    void setPostStageFilter(StateFilter filter) {
        postStageFilter_ = std::move(filter);
    }

protected:
    /// @param device  OCCA device (lifetime must exceed this object).
    /// @param mem     Device memory manager (lifetime must exceed this
    ///                object).
    /// @param rhs     Right-hand side $\mathcal{R}(u)$.
    /// @param nDof    Length of the ODE vector.
    /// @param oklDir  Directory containing the .okl kernel sources.
    StepperBase(occa::device &device, DeviceMemoryManager &mem, RhsFunction rhs,
                occa::dim_t nDof, const std::string &oklDir = OCCA_OKL_DIR)
        : device_(device), mem_(mem), rhs_(std::move(rhs)), nDof_(nDof),
          oklDir_(oklDir), blas_(device, mem, oklDir) {
    }

    /// @brief Build a kernel from time_update.okl with the standard JIT
    ///        props (Real / TILE_SIZE defines).
    occa::kernel buildTimeKernel(const std::string &name) const {
        occa::json props;
#ifdef USE_FLOAT_PRECISION
        props["defines/Real"] = "float";
#else
        props["defines/Real"] = "double";
#endif
        props["defines/TILE_SIZE"] = cmeles::DefaultTileSize;
        cmeles::finaliseKernelProps(props, device_);
        return device_.buildKernel(oklDir_ + "/time_update.okl", name, props);
    }

    /// @brief u_np1 = u_n + dt * res_n (per-DOF aliasing is safe); shared
    ///        by the explicit steppers and the RK initial-step heuristic.
    void eulerUpdate(occa::memory &o_un, Real dt, occa::memory &o_res,
                     occa::memory &o_unp1) {
        if (!explicitEulerUpdate_.isInitialized()) {
            explicitEulerUpdate_ = buildTimeKernel("explicitEulerUpdate");
        }
        explicitEulerUpdate_(static_cast<int>(nDof_), dt, o_un, o_res, o_unp1);
    }

    /// @brief Apply the installed post-stage filter to \p o_state (a no-op
    ///        when no filter is installed).
    ///
    /// Steppers call this at the stage-state write-back points defined by
    /// the setPostStageFilter() contract.
    void applyPostStageFilter(occa::memory o_state) {
        if (postStageFilter_) {
            postStageFilter_(o_state);
        }
    }

    // ---- Shared context
    // --------------------------------------------------------

    occa::device &device_;
    DeviceMemoryManager &mem_;
    RhsFunction rhs_;
    occa::dim_t nDof_;
    std::string oklDir_;
    Blas blas_;

private:
    /// Shared Euler-update kernel (lazily built by eulerUpdate()).
    occa::kernel explicitEulerUpdate_;

    /// Post-stage state filter (limiter); empty = no limiting.
    StateFilter postStageFilter_;
};
