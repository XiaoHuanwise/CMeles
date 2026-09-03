/// @file DualStepper.hpp
/// @brief Dual time stepping: implicit physical step driven to convergence
///        by an adaptive embedded-RK pseudo stepper.
///
/// One physical step $u^n \to u^{n+1}$ solves $\mathcal{F}(u^{n+1}) = 0$ of
/// the physical stepper (BackwardEulerStepper or DitrStepper) by marching
/// the pseudo-time ODE $du/d\tau = \mathcal{F}(u)$ until
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

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "RungeKuttaStepper.hpp"
#include "StepperBase.hpp"

template <class PhyStepper>
class DualStepper : public StepperBase<DualStepper<PhyStepper>>
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

    /// @param phy  Constructed physical stepper (owns the temporal residual).
    DualStepper(occa::device &device, DeviceMemoryManager &mem, RhsFunction rhs,
                occa::dim_t nDof, const ButcherTable &pseudoTable,
                Params params, PhyStepper phy,
                const std::string &oklDir = OCCA_OKL_DIR);

    /// @brief One physical step: initialise the guess, march pseudo time to
    ///        convergence, publish $u^{n+1}$. Returns \p dt.
    Real advanceImpl(occa::memory o_u, Real t, Real dt);

    /// @brief Order of the physical scheme (Backward Euler: 1, DITR: 2).
    int orderImpl() const
    {
        return kPhyStages == 1 ? 1 : 2;
    }
    const char *nameImpl() const
    {
        return kPhyStages == 1 ? "be" : "ditr";
    }

private:
    static constexpr int kPhyStages = PhyStepper::nStages();
    static constexpr int kRefStep   = 5;

    /// @brief One pseudo step (fixed or adaptive) on \p ps.
    void pseudoStep(RungeKuttaStepper &ps)
    {
        if (params_.pseudoFixedDt > Real(0))
        {
            ps.stepFixed(params_.pseudoFixedDt);
        }
        else
        {
            ps.stepPseudo(params_.allowReject);
        }
    }

    /// @brief Prototype convergence test (converged when false).
    bool notConverged(Real fNorm, Real f0Norm) const
    {
        return (fNorm / f0Norm > params_.rtol) && (fNorm > params_.atol);
    }

    /// @brief $\|f\|_\infty$ of a pseudo stepper's current residual.
    Real residualNorm(const RungeKuttaStepper &ps, occa::dim_t n)
    {
        occa::memory f = ps.f();
        return this->infNorm(f, n);
    }

    Params params_;
    PhyStepper phy_;

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

// ----------------------------------------------------------------------------
// Implementation (template: header-only)
// ----------------------------------------------------------------------------

template <class PhyStepper>
DualStepper<PhyStepper>::DualStepper(occa::device &device,
                                     DeviceMemoryManager &mem, RhsFunction rhs,
                                     occa::dim_t nDof,
                                     const ButcherTable &pseudoTable,
                                     Params params, PhyStepper phy,
                                     const std::string &oklDir)
    : StepperBase<DualStepper<PhyStepper>>(device, mem, std::move(rhs), nDof,
                                           oklDir),
      params_(params), phy_(std::move(phy)),
      // Coupled: the pseudo state is the stacked implicit state; decoupled:
      // each stage is advanced independently (N entries per stepper).
      pseudo_(device, mem, nullptr, params.decoupled ? nDof : nDof * kPhyStages,
              pseudoTable, params.rkParams, oklDir),
      pseudoC2_(device, mem, nullptr, nDof, pseudoTable, params.rkParams,
                oklDir)
{
    if (params_.decoupled && kPhyStages != 2)
    {
        throw std::invalid_argument(
            "DualStepper: decoupled mode requires a DITR physical stepper");
    }
    this->ensureKernels();

    o_uNew_  = mem.wrapOrMalloc(nDof * kPhyStages);
    o_uNc2_  = mem.wrapOrMalloc(nDof);
    o_uN1_   = mem.wrapOrMalloc(nDof);
    o_Rn_    = mem.wrapOrMalloc(nDof);
    o_Rnew0_ = mem.wrapOrMalloc(nDof);
    o_Rnew1_ = mem.wrapOrMalloc(nDof);
    o_uPrev_ = mem.wrapOrMalloc(nDof);
}

template <class PhyStepper>
Real DualStepper<PhyStepper>::advanceImpl(occa::memory o_u, Real /*t*/, Real dt)
{
    // U3R1 needs u^{n-1} (kept from the previous step) and theta.
    if constexpr (kPhyStages == 2)
    {
        if (phy_.needsPrev())
        {
            phy_.setTheta(dtPrev_ > Real(0) ? dtPrev_ / dt : Real(1));
        }
    }

    // R(u^n) is required by the DITR residuals.
    if constexpr (kPhyStages == 2)
    {
        this->rhs_(o_u, o_Rn_);
    }

    Real f0Norm = Real(0);
    Real fNorm  = Real(0);
    int cnt     = 0;

    if (params_.decoupled)
    {
        // ---- Decoupled DITR: two per-stage pseudo steppers ----------------
        if constexpr (kPhyStages == 2)
        {
            this->blas_.copy(this->nDof_, o_u, o_uNc2_);
            this->blas_.copy(this->nDof_, o_u, o_uN1_);
            this->blas_.copy(this->nDof_, o_Rn_, o_Rnew0_);
            this->blas_.copy(this->nDof_, o_Rn_, o_Rnew1_);

            occa::memory oUN = o_u, oUP = o_uPrev_, oRN = o_Rn_;
            occa::memory oUN1 = o_uN1_, oRN1 = o_Rnew1_, oRN0 = o_Rnew0_;
            PhyStepper &phy = phy_;
            pseudo_.setRhs([&phy, oUN, oRN0, oRN, dt](occa::memory x,
                                                      occa::memory F) mutable {
                phy.temporalResidualStageN1(x, oRN0, oUN, oRN, dt, F);
            });
            pseudoC2_.setRhs([&phy, oUN1, oRN1, oUN, oRN, oUP,
                              dt](occa::memory x, occa::memory F) mutable {
                phy.temporalResidualStageC2(x, oUN1, oRN1, oUN, oRN, oUP, dt,
                                            F);
            });

            pseudoC2_.setState(o_uNc2_, dt);
            pseudo_.setState(o_uN1_, dt);

            f0Norm = residualNorm(pseudo_, this->nDof_);
            fNorm  = f0Norm;
            while (notConverged(fNorm, f0Norm) && cnt < params_.maxPseudoSteps)
            {
                pseudoStep(pseudoC2_);
                occa::memory sC2 = pseudoC2_.state();
                this->blas_.copy(this->nDof_, sC2, o_uNc2_);
                this->rhs_(o_uNc2_, o_Rnew0_);
                pseudoStep(pseudo_);
                occa::memory sN1 = pseudo_.state();
                this->blas_.copy(this->nDof_, sN1, o_uN1_);
                this->rhs_(o_uN1_, o_Rnew1_);
                ++cnt;
                fNorm = residualNorm(pseudo_, this->nDof_);
                if (cnt <= kRefStep)
                {
                    f0Norm = std::max(f0Norm, fNorm);
                }
            }
        }
    }
    else
    {
        // ---- Coupled: one pseudo stepper on the stacked state -------------
        const auto stackedN = this->nDof_ * kPhyStages;
        if constexpr (kPhyStages == 1)
        {
            this->blas_.copy(this->nDof_, o_u, o_uNew_);
        }
        else
        {
            occa::memory stage0 = o_uNew_.slice(0, this->nDof_);
            occa::memory stage1 = o_uNew_.slice(this->nDof_, this->nDof_);
            this->blas_.copy(this->nDof_, o_u, stage0);
            this->blas_.copy(this->nDof_, o_u, stage1);
        }

        occa::memory oUN = o_u, oUP = o_uPrev_, oRN = o_Rn_;
        PhyStepper &phy = phy_;
        pseudo_.setRhs(
            [&phy, oUN, oUP, oRN, dt](occa::memory x, occa::memory F) mutable {
                if constexpr (kPhyStages == 1)
                {
                    phy.temporalResidual(x, oUN, dt, F);
                }
                else
                {
                    phy.temporalResidual(x, oUN, oRN, oUP, dt, F);
                }
            });

        pseudo_.setState(o_uNew_, dt);

        f0Norm = residualNorm(pseudo_, stackedN);
        fNorm  = f0Norm;
        while (notConverged(fNorm, f0Norm) && cnt < params_.maxPseudoSteps)
        {
            pseudoStep(pseudo_);
            ++cnt;
            fNorm = residualNorm(pseudo_, stackedN);
            if (cnt <= kRefStep)
            {
                f0Norm = std::max(f0Norm, fNorm);
            }
        }
    }

    if (notConverged(fNorm, f0Norm) && cnt >= params_.maxPseudoSteps)
    {
        std::cout << "DualStepper[" << nameImpl()
                  << "]: pseudo stepper hit max steps (|F|_inf = " << fNorm
                  << ")\n";
    }

    // Publish: u^{n-1} <- u^n, u^{n+1} <- the marched pseudo state. The
    // pseudo steppers own copies of the state (setState copies the guess
    // in), so the result must be read back from them: the coupled state is
    // the stacked [u^{n+c_2}, u^{n+1}] (publish the last stage), the
    // decoupled stage-n+1 stepper holds u^{n+1} directly.
    this->blas_.copy(this->nDof_, o_u, o_uPrev_);
    occa::memory pseudoState = pseudo_.state();
    if (params_.decoupled || kPhyStages == 1)
    {
        this->blas_.copy(this->nDof_, pseudoState, o_u);
    }
    else
    {
        occa::memory stage1 = pseudoState.slice(this->nDof_, this->nDof_);
        this->blas_.copy(this->nDof_, stage1, o_u);
    }
    dtPrev_ = dt;
    return dt;
}
