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
};
