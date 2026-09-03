/// @file CompressibleFlowSolver.hpp
/// @brief Top-level solver controller: owns the computation flow and the
///        lifetimes of all modules (device, memory manager, DG field,
///        time stepper).
///
/// The stepper is owned through the abstract StepperBase interface (one
/// virtual advance() call per physical time step — negligible next to the
/// per-stage device kernels, and it keeps the controller non-template).
/// The single dispatch point is runCompressibleFlowSolver, which switches
/// on [time_marching] method and constructs the matching stepper factory.
///
/// Flow: setup the DG field -> apply the initial condition
/// (uniform free stream or exprtk expressions) -> march to t_final with
/// either the fixed configured dt or the CFL estimate
/// $\Delta t = \mathrm{CFL} \min_K h_K/((2N{+}1)\Lambda_{\max,K})$ (clamped
/// to the remaining time) -> report progress and the final residual norm.

#pragma once

#include <algorithm>
#include <functional>
#include <memory>

#include <occa.hpp>

#include "blas/Blas.hpp"
#include "common/Types.hpp"
#include "config/Config.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "dg/DgField.hpp"
#include "time/StepperBase.hpp"

namespace solver_detail
{
/// Defined in ExprInitialCondition.cpp: evaluates the configured
/// expressions and projects them onto the field.
void applyExprICFromConfig(DgField &field, const Config &cfg);

/// Defined in CompressibleFlowSolver.cpp: overrides the OpenMP thread
/// count (no-op for threads <= 0).
void applyOmpThreads(int threads);
} // namespace solver_detail

/// @brief Run a complete computation for the configured time-marching
///        method — the single runtime-to-compile-time dispatch point
///        (defined in CompressibleFlowSolver.cpp).
/// @return Process exit code (0 on success).
int runCompressibleFlowSolver(const Config &cfg);

/// @brief Controller for one complete compressible-flow computation.
class CompressibleFlowSolver
{
public:
    /// @brief Factory invoked after the DG field is set up; binds the
    ///        residual to the field and constructs the stepper.
    using StepperFactory = std::function<std::unique_ptr<StepperBase>(
        occa::device &, DeviceMemoryManager &, const RhsFunction &,
        occa::dim_t)>;

    /// @param cfg     Configuration (kept by value; the field references it).
    /// @param factory Stepper construction callback.
    CompressibleFlowSolver(Config cfg, StepperFactory factory)
        : cfg_(std::move(cfg)), makeStepper_(std::move(factory))
    {
    }

    /// @brief Deterministic teardown in reverse dependency order: the
    ///        stepper and the DG field own the occa::memory /
    ///        occa::kernel handles, the Blas and the memory manager sit on
    ///        the device. Member destruction would release them in this
    ///        order anyway; the explicit sequence also drops the device
    ///        context before the members, keeping the GPU context alive
    ///        only as long as needed (relevant for iGPU/OpenCL sessions).
    ~CompressibleFlowSolver()
    {
        stepper_.reset();
        field_.reset();
        blas_.reset();
        mem_.reset();
        device_.free();
    }

    /// @brief Run the full computation. Member order guarantees the
    ///        destruction order stepper -> field -> blas -> memory manager
    ///        -> device.
    /// @return 0 on success.
    int run();

    // ---- Test accessors ----------------------------------------------------

    const Config &config() const
    {
        return cfg_;
    }
    DgField &field()
    {
        return *field_;
    }
    StepperBase &stepper()
    {
        return *stepper_;
    }
    occa::device &device()
    {
        return device_;
    }
    DeviceMemoryManager &mem()
    {
        return *mem_;
    }
    Real time() const noexcept
    {
        return time_;
    }
    int stepCount() const noexcept
    {
        return steps_;
    }

private:
    /// @brief Apply the configured initial condition to field_->o_u().
    void applyInitialCondition();

    /// @brief Time step: the fixed configured dt (if > 0), otherwise the
    ///        CFL estimate, clamped to the remaining time.
    Real computeDt(Real t);

    Config cfg_;
    StepperFactory makeStepper_;

    occa::device device_;
    std::unique_ptr<DeviceMemoryManager> mem_;
    std::unique_ptr<Blas> blas_;
    std::unique_ptr<DgField> field_;
    std::unique_ptr<StepperBase> stepper_;

    Real time_         = Real(0);
    int steps_         = 0;
    int printInterval_ = 100;
};

// ----------------------------------------------------------------------------
// Implementation (header-only)
// ----------------------------------------------------------------------------

inline void CompressibleFlowSolver::applyInitialCondition()
{
    switch (cfg_.icType())
    {
        case IcType::Uniform:
            field_->applyFreeStreamInitialCondition(field_->o_u());
            break;
        case IcType::Expr:
            solver_detail::applyExprICFromConfig(*field_, cfg_);
            break;
    }
}

inline Real CompressibleFlowSolver::computeDt(Real t)
{
    const Real remaining = cfg_.timeFinal() - t;
    Real dt = (cfg_.timeDt() > Real(0))
                  ? cfg_.timeDt()
                  : field_->estimateDt(field_->o_u(), cfg_.timeCfl());
    return std::min(dt, remaining);
}

inline int CompressibleFlowSolver::run()
{
    // OCCA's OpenMP backend emits `#pragma omp parallel for` with no
    // num_threads clause, so the runtime ICV governs every kernel; set it
    // before the first parallel region (no-op for the non-OpenMP backends).
    solver_detail::applyOmpThreads(cfg_.occaThreads());
    // platform_id selects the OpenCL/dpcpp platform, device_id the device
    // inside it (also the CUDA/HIP device id); Serial/OpenMP ignore both.
    device_ = occa::device({{"mode", cfg_.occaMode()},
                            {"platform_id", cfg_.occaPlatform()},
                            {"device_id", cfg_.occaDevice()}});
    mem_    = std::make_unique<DeviceMemoryManager>(device_);
    blas_   = std::make_unique<Blas>(device_, *mem_);

    field_ = std::make_unique<DgField>(cfg_, device_, *mem_);
    field_->setup();

    const occa::dim_t nDof = static_cast<occa::dim_t>(field_->numElements()) *
                             field_->numVars() * field_->numModes();
    DgField *f             = field_.get();
    RhsFunction rhs        = [f](occa::memory u, occa::memory res) {
        f->computeRHS(u, res);
    };
    stepper_ = makeStepper_(device_, *mem_, rhs, nDof);

    applyInitialCondition();
    device_.finish();

    std::cout << "CMeles: method=" << stepper_->name()
              << " order=" << stepper_->order()
              << " elems=" << field_->numElements() << " dofs=" << nDof
              << " T_final=" << cfg_.timeFinal() << "\n";

    time_  = Real(0);
    steps_ = 0;
    while (time_ < cfg_.timeFinal() - Real(1e-12) * cfg_.timeFinal())
    {
        const Real dt    = computeDt(time_);
        const Real taken = stepper_->advance(field_->o_u(), time_, dt);
        time_ += taken;
        ++steps_;

        if (steps_ % printInterval_ == 0)
        {
            occa::memory ou = field_->o_u(), ores = field_->o_res();
            rhs(ou, ores);
            const Real resNorm = blas_->nrm2(nDof, ores);
            std::cout << "  step " << steps_ << ": t = " << time_
                      << ", dt = " << taken << ", |res|_2 = " << resNorm
                      << "\n";
        }
        if (taken <= Real(0))
        {
            std::cout << "CMeles: zero time step at t = " << time_
                      << " — aborting\n";
            return 1;
        }
    }

    occa::memory ou = field_->o_u(), ores = field_->o_res();
    rhs(ou, ores);
    const Real resNorm = blas_->nrm2(nDof, ores);
    std::cout << "CMeles: finished t = " << time_ << " in " << steps_
              << " steps, |res|_2 = " << resNorm << "\n";
    device_.finish();
    return 0;
}
