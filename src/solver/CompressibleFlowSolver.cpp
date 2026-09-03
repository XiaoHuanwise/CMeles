/// @file CompressibleFlowSolver.cpp
/// @brief The single runtime-to-compile-time dispatch point: switches on
///        the configured time-marching method and instantiates the matching
///        CompressibleFlowSolver<Stepper>.
///
/// Five solver instantiations: EulerStepper, SspRk3Stepper,
/// RungeKuttaStepper (tableau selected by data), DualStepper<BackwardEuler>,
/// DualStepper<DitrStepper> (variant selected by data).

#include "CompressibleFlowSolver.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "time/DualStepper.hpp"
#include "time/ImplicitResidual.hpp"
#include "time/RungeKuttaStepper.hpp"
#include "time/SimpleExplicitStepper.hpp"

namespace solver_detail
{
void applyOmpThreads(int threads)
{
    if (threads <= 0)
    {
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

namespace
{
/// @brief Physical-stepper factory: (device, mem, rhs, nDof) -> PhyStepper.
template <class PhyStepper>
using PhyFactory = std::function<PhyStepper(
    occa::device &, DeviceMemoryManager &, const RhsFunction &, occa::dim_t)>;

/// @brief Run the solver with a default-constructed simple stepper.
template <class Stepper> int runWith(const Config &cfg)
{
    CompressibleFlowSolver<Stepper> solver(
        cfg, [](occa::device &device, DeviceMemoryManager &mem,
                const RhsFunction &rhs, occa::dim_t nDof) {
            return std::make_unique<Stepper>(device, mem, rhs, nDof);
        });
    return solver.run();
}

/// @brief Run the solver with an adaptive embedded RK stepper.
int runWithRk(const Config &cfg)
{
    const ButcherTable *table = butcherTableForMethod(cfg.timeMethod());
    if (table == nullptr)
    {
        throw std::invalid_argument(
            "runCompressibleFlowSolver: method is not an embedded RK pair");
    }
    RungeKuttaStepper::Params params;
    params.rtol = cfg.timeRtol();
    params.atol = cfg.timeAtol();

    const ButcherTable &tab = *table;
    CompressibleFlowSolver<RungeKuttaStepper> solver(
        cfg, [&tab, params](occa::device &device, DeviceMemoryManager &mem,
                            const RhsFunction &rhs, occa::dim_t nDof) {
            return std::make_unique<RungeKuttaStepper>(device, mem, rhs, nDof,
                                                       tab, params);
        });
    return solver.run();
}

/// @brief Run the solver with dual time stepping.
template <class PhyStepper>
int runWithDual(const Config &cfg, PhyFactory<PhyStepper> makePhy)
{
    const ButcherTable *pseudoTable =
        butcherTableForMethod(cfg.timePseudoMethod());
    if (pseudoTable == nullptr)
    {
        throw std::invalid_argument(
            "runCompressibleFlowSolver: pseudo_method must be an embedded "
            "RK pair");
    }

    typename DualStepper<PhyStepper>::Params params;
    params.atol           = cfg.timeAtol();
    params.rtol           = cfg.timeRtol();
    params.maxPseudoSteps = cfg.timeMaxPseudoSteps();
    params.pseudoFixedDt  = cfg.timePseudoDt();
    params.allowReject    = cfg.timePseudoReject();
    params.decoupled      = cfg.timeDualDecoupled();
    // Loose pseudo-local accuracy: the dual-time convergence criterion
    // governs, and a large initial pseudo step keeps the iteration count
    // sane (the Hairer heuristic otherwise starts far below the stability
    // limit). In single precision the extra-loose tolerance keeps the
    // pseudo step near the stability limit — the rounding-floor-limited
    // error estimate otherwise stalls the residual reduction.
    params.rkParams.rtol =
        std::is_same<Real, float>::value ? Real(1e-2) : Real(1e-3);
    params.rkParams.atol = params.rkParams.rtol;

    const ButcherTable &tab = *pseudoTable;
    CompressibleFlowSolver<DualStepper<PhyStepper>> solver(
        cfg,
        [&tab, params, makePhy](occa::device &device, DeviceMemoryManager &mem,
                                const RhsFunction &rhs, occa::dim_t nDof) {
            return std::make_unique<DualStepper<PhyStepper>>(
                device, mem, rhs, nDof, tab, params,
                makePhy(device, mem, rhs, nDof));
        });
    return solver.run();
}

/// @brief DITR variant for the configured method.
DitrStepper::Variant ditrVariant(const Config &cfg)
{
    switch (cfg.timeMethod())
    {
        case TimeMethod::DitrU2R1:
            return DitrStepper::Variant::U2R1;
        case TimeMethod::DitrU3R1:
            return DitrStepper::Variant::U3R1;
        default:
            return DitrStepper::Variant::U2R2;
    }
}
} // namespace

int runCompressibleFlowSolver(const Config &cfg)
{
    switch (cfg.timeMethod())
    {
        case TimeMethod::Euler:
            return runWith<EulerStepper>(cfg);
        case TimeMethod::SspRk3:
            return runWith<SspRk3Stepper>(cfg);
        case TimeMethod::Rk32:
        case TimeMethod::Rk54:
        case TimeMethod::SspRk221:
        case TimeMethod::SspRk321:
        case TimeMethod::SspRk332:
        case TimeMethod::SspRk432:
            return runWithRk(cfg);
        case TimeMethod::BackwardEuler:
            return runWithDual<BackwardEulerStepper>(
                cfg, [](occa::device &device, DeviceMemoryManager &mem,
                        const RhsFunction &rhs, occa::dim_t nDof) {
                    return BackwardEulerStepper(device, mem, rhs, nDof);
                });
        case TimeMethod::DitrU2R2:
        case TimeMethod::DitrU2R1:
        case TimeMethod::DitrU3R1:
            return runWithDual<DitrStepper>(cfg, [cfg](occa::device &device,
                                                       DeviceMemoryManager &mem,
                                                       const RhsFunction &rhs,
                                                       occa::dim_t nDof) {
                return DitrStepper(device, mem, rhs, nDof, ditrVariant(cfg));
            });
        default:
            throw std::invalid_argument(
                "runCompressibleFlowSolver: unknown time method");
    }
}
