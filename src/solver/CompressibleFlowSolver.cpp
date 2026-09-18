/// @file CompressibleFlowSolver.cpp
/// @brief Implementation of the solver controller and the
///        runCompressibleFlowSolver entry point. The stepper itself is
///        selected by the time module's factory (makeStepperFactory).

#include "CompressibleFlowSolver.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ExprInitialCondition.hpp"
#include "common/Log.hpp"
#include "common/Timer.hpp"
#include "io/FieldOutput.hpp"
#include "time/StepperFactory.hpp"

namespace solver_detail {
void applyOmpThreads(int threads) {
    if (threads <= 0) {
        return;
    }
#ifdef _OPENMP
    // libgomp snapshots OMP_NUM_THREADS in a loader-time constructor (it is
    // pulled in as a dependency of libocca), so setenv(3) inside the program
    // is too late. The runtime API is the only reliable in-process override;
    // it governs the `#pragma omp parallel for` regions OCCA emits (they
    // carry no num_threads clause).
    omp_set_num_threads(threads);
#else
    // No OpenMP in this build: the env var still governs any OpenMP runtime
    // loaded later through OCCA's shared library.
    setenv("OMP_NUM_THREADS", std::to_string(threads).c_str(), 1);
#endif
}
} // namespace solver_detail

int runCompressibleFlowSolver(const Config &cfg) {
    CompressibleFlowSolver solver(cfg);
    return solver.run();
}

// ----------------------------------------------------------------------------
// CompressibleFlowSolver
// ----------------------------------------------------------------------------

CompressibleFlowSolver::CompressibleFlowSolver(Config cfg)
    : cfg_(std::move(cfg)) {
}

CompressibleFlowSolver::~CompressibleFlowSolver() {
    output_.reset();
    stepper_.reset();
    field_.reset();
    blas_.reset();
    mem_.reset();
    device_.free();
}

void CompressibleFlowSolver::applyInitialCondition() {
    switch (cfg_.icType()) {
        case IcType::Uniform:
            field_->applyFreeStreamInitialCondition(field_->o_u());
            break;
        case IcType::Expr:
            solver_detail::applyExprICFromConfig(*field_, cfg_);
            break;
    }
}

Real CompressibleFlowSolver::computeDt(Real t) {
    const Real remaining = cfg_.timeFinal() - t;
    Real dt = (cfg_.timeDt() > Real(0))
                  ? cfg_.timeDt()
                  : field_->estimateDt(field_->o_u(), cfg_.timeCfl());
    return std::min(dt, remaining);
}

int CompressibleFlowSolver::run() {
    cmeles::Log::setLevel(cfg_.logLevel());
    logInterval_ = cfg_.logInterval();
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
    stepper_ = makeStepperFactory(cfg_)(device_, *mem_, rhs, nDof);

    applyInitialCondition();
    device_.finish();

    // Output stays completely dormant unless [output] enable is true.
    if (cfg_.outputEnable()) {
        outputInterval_ = cfg_.outputInterval();
        output_         = std::make_unique<FieldOutput>(cfg_, *field_);
        output_->write(0, time_);
    }

    CMES_LOG_NORMAL << "CMeles: method=" << stepper_->name()
                    << " order=" << stepper_->order()
                    << " elems=" << field_->numElements() << " dofs=" << nDof
                    << " T_final=" << cfg_.timeFinal();

    cmeles::Timer loopTimer;
    cmeles::Timer intervalTimer;

    time_  = Real(0);
    steps_ = 0;
    loopTimer.start();
    intervalTimer.start();
    while (time_ < cfg_.timeFinal() - Real(1e-12) * cfg_.timeFinal()) {
        const Real dt    = computeDt(time_);
        const Real taken = stepper_->advance(field_->o_u(), time_, dt);
        time_ += taken;
        ++steps_;

        // Per-step NaN sweep of the solution: a NaN residual inside any
        // stage contaminates the updated state immediately, so one sweep
        // per physical step (instead of per RHS evaluation) catches the
        // blow-up without serialising every stage.
        const Real nNaN = field_->countSolutionNaN();
        if (nNaN > Real(0)) {
            CMES_LOG_NORMAL << "CMeles: " << nNaN
                            << " NaN entries in the solution at step " << steps_
                            << ", t = " << time_ << " — aborting";
            throw std::runtime_error(
                "CompressibleFlowSolver::run: NaN detected in the solution");
        }

        if (steps_ % logInterval_ == 0) {
            occa::memory ou = field_->o_u(), ores = field_->o_res();
            rhs(ou, ores);
            // nrm2 returns a host value, so this is also the device sync
            // point that makes the interval wall time trustworthy on the
            // asynchronous backends.
            const Real resNorm = blas_->nrm2(nDof, ores);
            if (!std::isfinite(resNorm)) {
                CMES_LOG_NORMAL << "CMeles: residual is not finite at step "
                                << steps_ << ", t = " << time_ << " — aborting";
                return 1;
            }
            const double wall = intervalTimer.elapsed();
            CMES_LOG_NORMAL << "  step " << steps_ << ": t = " << time_
                            << ", dt = " << taken << ", |res|_2 = " << resNorm
                            << ", wall = " << wall << " s ("
                            << wall / logInterval_ << " s/step)";
            intervalTimer.start();
        }
        if (taken <= Real(0)) {
            CMES_LOG_NORMAL << "CMeles: zero time step at t = " << time_
                            << " — aborting";
            return 1;
        }
        if (output_ && steps_ % outputInterval_ == 0) {
            output_->write(steps_, time_);
        }
    }
    const double loopWall = loopTimer.elapsed();

    occa::memory ou = field_->o_u(), ores = field_->o_res();
    rhs(ou, ores);
    const Real resNorm = blas_->nrm2(nDof, ores);
    if (!std::isfinite(resNorm)) {
        CMES_LOG_NORMAL << "CMeles: final residual is not finite after "
                        << steps_ << " steps — aborting";
        return 1;
    }
    // Final state (skipped when the last interval output already covers it).
    if (output_ && steps_ % outputInterval_ != 0) {
        output_->write(steps_, time_);
    }
    CMES_LOG_NORMAL << "CMeles: finished t = " << time_ << " in " << steps_
                    << " steps, |res|_2 = " << resNorm
                    << ", loop wall time = " << loopWall << " s ("
                    << loopWall / steps_ << " s/step)";
    device_.finish();
    return 0;
}
