/// @file DualStepper.hpp
/// @brief Dual time stepping: implicit physical step driven to convergence
///        by an adaptive embedded-RK pseudo stepper.
///
/// One physical step $u^n \to u^{n+1}$ solves $\mathcal{F}(u^{n+1}) = 0$ of
/// the owned temporal residual (BackwardEulerResidual or DitrResidual) by
/// marching the pseudo-time ODE $du/d\tau = \mathcal{F}(u)$ until
/// $$ \|\mathcal{F}\|_\infty / \max_{k \le 5} \|\mathcal{F}_0\|_\infty
///    \le \mathrm{rtol} \quad\text{or}\quad
///    \|\mathcal{F}\|_\infty \le \mathrm{atol}, $$
/// where the reference norm is the running maximum over the first
/// REF_STEP = 5 pseudo steps (the "magic number from fluent" in the
/// prototype). The state initial guess is $u^n$ (both stages for DITR).
///
/// Coupled mode advances the stacked state $[u^{n+c_2}, u^{n+1}]$ with a
/// single pseudo stepper; decoupled mode (DITR only) advances each stage
/// with its own pseudo stepper and refreshes the other stage's residual
/// after every half-iteration. $u^{n-1}$ and $\theta = \Delta t^{n-1} /
/// \Delta t^n$ are maintained internally for U3R1.
///
/// Reference: docs/tech_docs/time_marching/implicit_time_marching.md and
/// .stepper.py DualStepper.

#pragma once

#include <occa.hpp>

#include <memory>
#include <string>

#include "ImplicitResidual.hpp"
#include "RungeKuttaStepper.hpp"
#include "StepperBase.hpp"

/// @brief Dual time-stepping driver: owns the temporal residual and the
///        pseudo steppers, advances one physical step per advance() call.
class DualStepper : public StepperBase
{
public:
    /// @brief Dual-time parameters.
    struct Params
    {
        Real atol          = Real(1e-6); ///< Absolute convergence tolerance.
        Real rtol          = Real(1e-6); ///< Relative convergence tolerance.
        int maxPseudoSteps = 100;
        Real pseudoFixedDt = Real(0); ///< Fixed pseudo step; <= 0 adaptive.
        bool allowReject   = false;   ///< Retry rejected pseudo steps.
        bool decoupled     = false;   ///< Per-stage pseudo steppers (DITR).
        RungeKuttaStepper::Params rkParams{}; ///< Pseudo stepper tolerances.
    };

    /// @param residual  Physical temporal residual (owned; its stage count
    ///                  sizes the stacked pseudo state).
    DualStepper(occa::device &device, DeviceMemoryManager &mem, RhsFunction rhs,
                occa::dim_t nDof, const ButcherTable &pseudoTable,
                Params params, std::unique_ptr<TemporalResidual> residual,
                const std::string &oklDir = OCCA_OKL_DIR);

    /// @brief One physical step: initialise the guess, march pseudo time to
    ///        convergence, publish $u^{n+1}$. Returns \p dt.
    Real advance(occa::memory o_u, Real t, Real dt) override;

    /// @brief Order of the physical scheme (Backward Euler: 1, DITR: 2).
    int order() const override
    {
        return nStages_ == 1 ? 1 : 2;
    }
    const char *name() const override
    {
        return nStages_ == 1 ? "be" : "ditr";
    }

private:
    static constexpr int kRefStep = 5;

    /// @brief One pseudo step (fixed or adaptive) on \p ps.
    void pseudoStep(RungeKuttaStepper &ps);

    /// @brief Prototype convergence test (converged when false).
    bool notConverged(Real fNorm, Real f0Norm) const;

    /// @brief $\|f\|_\infty$ of a pseudo stepper's current residual.
    Real residualNorm(const RungeKuttaStepper &ps, occa::dim_t n);

    Params params_;
    std::unique_ptr<TemporalResidual> residual_;
    const int nStages_; ///< Stacked stages of the implicit state.

    /// @brief Pseudo steppers. Coupled DITR works on the stacked 2N state;
    ///        in decoupled mode both work on N entries. The c2 stepper is
    ///        only used in decoupled mode.
    RungeKuttaStepper pseudo_;
    RungeKuttaStepper pseudoC2_;

    occa::memory o_uNew_;  ///< Implicit state guess (stacked for DITR).
    occa::memory o_uNc2_;  ///< Decoupled stage $u^{n+c_2}$.
    occa::memory o_uN1_;   ///< Decoupled stage $u^{n+1}$.
    occa::memory o_Rn_;    ///< $R(u^n)$.
    occa::memory o_Rnew0_; ///< Decoupled current $R(u^{n+c_2})$.
    occa::memory o_Rnew1_; ///< Decoupled current $R(u^{n+1})$.
    occa::memory o_uPrev_; ///< $u^{n-1}$ (U3R1); refreshed every step.

    Real dtPrev_ = Real(0); ///< Previous physical step (for theta).
};
