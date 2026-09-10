/// @file CompressibleFlowSolver.hpp
/// @brief Top-level solver controller: owns the computation flow and the
///        lifetimes of all modules (device, memory manager, DG field,
///        time stepper).
///
/// The stepper is selected by the time module's factory
/// (makeStepperFactory, src/time/StepperFactory.cpp) from the
/// [time_marching] configuration and owned through the abstract
/// StepperBase interface — one virtual advance() call per physical time
/// step, negligible next to the per-stage device kernels.
///
/// Flow: setup the DG field -> apply the initial condition
/// (uniform free stream or exprtk expressions) -> march to t_final with
/// either the fixed configured dt or the CFL estimate
/// $\Delta t = \mathrm{CFL} \min_K h_K/((2N{+}1)\Lambda_{\max,K})$ (clamped
/// to the remaining time) -> report progress and the final residual norm.

#pragma once

#include <memory>

#include <occa.hpp>

#include "blas/Blas.hpp"
#include "common/Types.hpp"
#include "config/Config.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "dg/DgField.hpp"
#include "time/StepperBase.hpp"

namespace solver_detail {
/// Defined in CompressibleFlowSolver.cpp: overrides the OpenMP thread
/// count (no-op for threads <= 0).
void applyOmpThreads(int threads);
} // namespace solver_detail

/// @brief Run a complete computation for the configured time-marching
///        method (defined in CompressibleFlowSolver.cpp).
/// @return Process exit code (0 on success).
int runCompressibleFlowSolver(const Config &cfg);

/// @brief Controller for one complete compressible-flow computation.
class CompressibleFlowSolver {
public:
    /// @param cfg Configuration (kept by value; the field references it).
    explicit CompressibleFlowSolver(Config cfg);

    /// @brief Deterministic teardown in reverse dependency order: the
    ///        stepper and the DG field own the occa::memory /
    ///        occa::kernel handles, the Blas and the memory manager sit on
    ///        the device. Member destruction would release them in this
    ///        order anyway; the explicit sequence also drops the device
    ///        context before the members, keeping the GPU context alive
    ///        only as long as needed (relevant for iGPU/OpenCL sessions).
    ~CompressibleFlowSolver();

    /// @brief Run the full computation. Member order guarantees the
    ///        destruction order stepper -> field -> blas -> memory manager
    ///        -> device.
    /// @return 0 on success.
    int run();

    // ---- Test accessors ----------------------------------------------------

    const Config &config() const {
        return cfg_;
    }
    DgField &field() {
        return *field_;
    }
    StepperBase &stepper() {
        return *stepper_;
    }
    occa::device &device() {
        return device_;
    }
    DeviceMemoryManager &mem() {
        return *mem_;
    }
    Real time() const noexcept {
        return time_;
    }
    int stepCount() const noexcept {
        return steps_;
    }

private:
    /// @brief Apply the configured initial condition to field_->o_u().
    void applyInitialCondition();

    /// @brief Time step: the fixed configured dt (if > 0), otherwise the
    ///        CFL estimate, clamped to the remaining time.
    Real computeDt(Real t);

    Config cfg_;

    occa::device device_;
    std::unique_ptr<DeviceMemoryManager> mem_;
    std::unique_ptr<Blas> blas_;
    std::unique_ptr<DgField> field_;
    std::unique_ptr<StepperBase> stepper_;

    Real time_         = Real(0);
    int steps_         = 0;
    int printInterval_ = 100;
};
