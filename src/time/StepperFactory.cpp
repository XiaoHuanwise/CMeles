/// @file StepperFactory.cpp
/// @brief Implementation of the config-driven stepper factory.

#include "StepperFactory.hpp"

#include <stdexcept>

#include "config/Config.hpp"
#include "time/ButcherTable.hpp"
#include "time/DualStepper.hpp"
#include "time/ImplicitResidual.hpp"
#include "time/RungeKuttaStepper.hpp"
#include "time/SimpleExplicitStepper.hpp"

namespace
{
/// @brief Dual-time parameters shared by the Backward-Euler and DITR arms.
DualStepper::Params dualParams(const Config &cfg)
{
    DualStepper::Params params;
    params.atol           = cfg.timeAtol();
    params.rtol           = cfg.timeRtol();
    params.maxPseudoSteps = cfg.timeMaxPseudoSteps();
    params.pseudoFixedDt  = cfg.timePseudoDt();
    params.allowReject    = cfg.timePseudoReject();
    params.decoupled      = cfg.timeDualDecoupled();
    // Loose pseudo-local accuracy: the dual-time convergence criterion
    // governs, and a large initial pseudo step keeps the iteration count
    // sane (the Hairer heuristic otherwise starts far below the stability
    // limit). Looser than ~1e-3 destabilises the pseudo iteration in
    // single precision (observed as NaN on the vortex dual-time runs);
    // tighter costs more pseudo iterations. Override through
    // [time_marching] pseudo_rtol / pseudo_atol.
    params.rkParams.rtol =
        cfg.timePseudoRtol() > Real(0) ? cfg.timePseudoRtol() : Real(1e-3);
    params.rkParams.atol = cfg.timePseudoAtol() > Real(0)
                               ? cfg.timePseudoAtol()
                               : params.rkParams.rtol;
    return params;
}

/// @brief Butcher tableau of the pseudo stepper (shared by the dual arms).
const ButcherTable &pseudoTable(const Config &cfg)
{
    const ButcherTable *table = butcherTableForMethod(cfg.timePseudoMethod());
    if (table == nullptr)
    {
        throw std::invalid_argument(
            "makeStepperFactory: pseudo_method must be an embedded RK pair");
    }
    return *table;
}

/// @brief DITR variant for the configured method.
DitrResidual::Variant ditrVariant(const Config &cfg)
{
    switch (cfg.timeMethod())
    {
        case TimeMethod::DitrU2R1:
            return DitrResidual::Variant::U2R1;
        case TimeMethod::DitrU3R1:
            return DitrResidual::Variant::U3R1;
        default:
            return DitrResidual::Variant::U2R2;
    }
}
} // namespace

StepperFactory makeStepperFactory(const Config &cfg)
{
    switch (cfg.timeMethod())
    {
        case TimeMethod::Euler:
            return [](occa::device &device, DeviceMemoryManager &mem,
                      const RhsFunction &rhs, occa::dim_t nDof) {
                return std::make_unique<EulerStepper>(device, mem, rhs, nDof);
            };
        case TimeMethod::SspRk3:
            return [](occa::device &device, DeviceMemoryManager &mem,
                      const RhsFunction &rhs, occa::dim_t nDof) {
                return std::make_unique<SspRk3Stepper>(device, mem, rhs, nDof);
            };
        case TimeMethod::Rk32:
        case TimeMethod::Rk54:
        case TimeMethod::SspRk221:
        case TimeMethod::SspRk321:
        case TimeMethod::SspRk332:
        case TimeMethod::SspRk432:
        {
            const ButcherTable *table = butcherTableForMethod(cfg.timeMethod());
            if (table == nullptr)
            {
                throw std::invalid_argument(
                    "makeStepperFactory: method is not an embedded RK pair");
            }
            RungeKuttaStepper::Params params;
            params.rtol = cfg.timeRtol();
            params.atol = cfg.timeAtol();

            const ButcherTable &tab = *table;
            return [&tab, params](occa::device &device,
                                  DeviceMemoryManager &mem,
                                  const RhsFunction &rhs, occa::dim_t nDof) {
                return std::make_unique<RungeKuttaStepper>(device, mem, rhs,
                                                           nDof, tab, params);
            };
        }
        case TimeMethod::BackwardEuler:
        {
            // Decoupled per-stage pseudo stepping requires a two-stage
            // residual; Backward Euler is single-stage (the stage residuals
            // would otherwise throw at the first pseudo step).
            if (cfg.timeDualDecoupled())
            {
                throw std::invalid_argument(
                    "makeStepperFactory: dual_decoupled requires a DITR "
                    "method");
            }
            const ButcherTable &tab          = pseudoTable(cfg);
            const DualStepper::Params params = dualParams(cfg);
            return
                [&tab, params](occa::device &device, DeviceMemoryManager &mem,
                               const RhsFunction &rhs, occa::dim_t nDof) {
                    return std::make_unique<DualStepper>(
                        device, mem, rhs, nDof, tab, params,
                        std::make_unique<BackwardEulerResidual>(device, mem,
                                                                rhs, nDof));
                };
        }
        case TimeMethod::DitrU2R2:
        case TimeMethod::DitrU2R1:
        case TimeMethod::DitrU3R1:
        {
            const ButcherTable &tab             = pseudoTable(cfg);
            const DualStepper::Params params    = dualParams(cfg);
            const DitrResidual::Variant variant = ditrVariant(cfg);
            return [&tab, params,
                    variant](occa::device &device, DeviceMemoryManager &mem,
                             const RhsFunction &rhs, occa::dim_t nDof) {
                return std::make_unique<DualStepper>(
                    device, mem, rhs, nDof, tab, params,
                    std::make_unique<DitrResidual>(device, mem, rhs, nDof,
                                                   variant));
            };
        }
        default:
            throw std::invalid_argument(
                "makeStepperFactory: unknown time method");
    }
}
