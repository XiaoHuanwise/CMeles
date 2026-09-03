/// @file DualStepper.cpp
/// @brief Implementation of the dual time-stepping driver.

#include "DualStepper.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

DualStepper::DualStepper(occa::device &device, DeviceMemoryManager &mem,
                         RhsFunction rhs, occa::dim_t nDof,
                         const ButcherTable &pseudoTable, Params params,
                         std::unique_ptr<TemporalResidual> residual,
                         const std::string &oklDir)
    : StepperBase(device, mem, std::move(rhs), nDof, oklDir), params_(params),
      residual_(std::move(residual)), nStages_(residual_->nStages()),
      // Coupled: the pseudo state is the stacked implicit state; decoupled:
      // each stage is advanced independently (N entries per stepper).
      pseudo_(device, mem, nullptr, params_.decoupled ? nDof : nDof * nStages_,
              pseudoTable, params_.rkParams, oklDir),
      pseudoC2_(device, mem, nullptr, nDof, pseudoTable, params_.rkParams,
                oklDir)
{
    if (params_.decoupled && nStages_ != 2)
    {
        throw std::invalid_argument(
            "DualStepper: decoupled mode requires a DITR residual");
    }
    ensureKernels();

    o_uNew_  = mem.wrapOrMalloc(nDof * nStages_);
    o_uNc2_  = mem.wrapOrMalloc(nDof);
    o_uN1_   = mem.wrapOrMalloc(nDof);
    o_Rn_    = mem.wrapOrMalloc(nDof);
    o_Rnew0_ = mem.wrapOrMalloc(nDof);
    o_Rnew1_ = mem.wrapOrMalloc(nDof);
    o_uPrev_ = mem.wrapOrMalloc(nDof);
}

void DualStepper::pseudoStep(RungeKuttaStepper &ps)
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

bool DualStepper::notConverged(Real fNorm, Real f0Norm) const
{
    return (fNorm / f0Norm > params_.rtol) && (fNorm > params_.atol);
}

Real DualStepper::residualNorm(const RungeKuttaStepper &ps, occa::dim_t n)
{
    occa::memory f = ps.f();
    return infNorm(f, n);
}

Real DualStepper::advance(occa::memory o_u, Real /*t*/, Real dt)
{
    // U3R1 needs u^{n-1} (kept from the previous step) and theta.
    if (residual_->needsPrev())
    {
        residual_->setTheta(dtPrev_ > Real(0) ? dtPrev_ / dt : Real(1));
    }

    // R(u^n) is required by the two-stage residuals.
    if (nStages_ == 2)
    {
        rhs_(o_u, o_Rn_);
    }

    Real f0Norm = Real(0);
    Real fNorm  = Real(0);
    int cnt     = 0;

    if (params_.decoupled)
    {
        // ---- Decoupled DITR: two per-stage pseudo steppers ----------------
        blas_.copy(nDof_, o_u, o_uNc2_);
        blas_.copy(nDof_, o_u, o_uN1_);
        blas_.copy(nDof_, o_Rn_, o_Rnew0_);
        blas_.copy(nDof_, o_Rn_, o_Rnew1_);

        occa::memory oUN = o_u, oUP = o_uPrev_, oRN = o_Rn_;
        occa::memory oUN1 = o_uN1_, oRN1 = o_Rnew1_, oRN0 = o_Rnew0_;
        TemporalResidual &res = *residual_;
        pseudo_.setRhs([&res, oUN1, oRN0, oUN, oRN,
                        dt](occa::memory x, occa::memory F) mutable {
            res.temporalResidualStageN1(x, oRN0, oUN, oRN, dt, F);
        });
        pseudoC2_.setRhs([&res, oUN1, oRN1, oUN, oRN, oUP,
                          dt](occa::memory x, occa::memory F) mutable {
            res.temporalResidualStageC2(x, oUN1, oRN1, oUN, oRN, oUP, dt, F);
        });

        pseudoC2_.setState(o_uNc2_, dt);
        pseudo_.setState(o_uN1_, dt);

        f0Norm = residualNorm(pseudo_, nDof_);
        fNorm  = f0Norm;
        while (notConverged(fNorm, f0Norm) && cnt < params_.maxPseudoSteps)
        {
            pseudoStep(pseudoC2_);
            occa::memory sC2 = pseudoC2_.state();
            blas_.copy(nDof_, sC2, o_uNc2_);
            rhs_(o_uNc2_, o_Rnew0_);
            pseudoStep(pseudo_);
            occa::memory sN1 = pseudo_.state();
            blas_.copy(nDof_, sN1, o_uN1_);
            rhs_(o_uN1_, o_Rnew1_);
            ++cnt;
            fNorm = residualNorm(pseudo_, nDof_);
            if (cnt <= kRefStep)
            {
                f0Norm = std::max(f0Norm, fNorm);
            }
        }
    }
    else
    {
        // ---- Coupled: one pseudo stepper on the stacked state -------------
        const auto stackedN = nDof_ * nStages_;
        if (nStages_ == 1)
        {
            blas_.copy(nDof_, o_u, o_uNew_);
        }
        else
        {
            occa::memory stage0 = o_uNew_.slice(0, nDof_);
            occa::memory stage1 = o_uNew_.slice(nDof_, nDof_);
            blas_.copy(nDof_, o_u, stage0);
            blas_.copy(nDof_, o_u, stage1);
        }

        occa::memory oUN = o_u, oUP = o_uPrev_, oRN = o_Rn_;
        TemporalResidual &res = *residual_;
        pseudo_.setRhs(
            [&res, oUN, oUP, oRN, dt](occa::memory x, occa::memory F) mutable {
                res.temporalResidual(x, oUN, oRN, oUP, dt, F);
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
        std::cout << "DualStepper[" << name()
                  << "]: pseudo stepper hit max steps (|F|_inf = " << fNorm
                  << ")\n";
    }

    // Publish: u^{n-1} <- u^n, u^{n+1} <- the marched pseudo state. The
    // pseudo steppers own copies of the state (setState copies the guess
    // in), so the result must be read back from them: the coupled state is
    // the stacked [u^{n+c_2}, u^{n+1}] (publish the last stage), the
    // decoupled stage-n+1 stepper holds u^{n+1} directly.
    blas_.copy(nDof_, o_u, o_uPrev_);
    occa::memory pseudoState = pseudo_.state();
    if (params_.decoupled || nStages_ == 1)
    {
        blas_.copy(nDof_, pseudoState, o_u);
    }
    else
    {
        occa::memory stage1 = pseudoState.slice(nDof_, nDof_);
        blas_.copy(nDof_, stage1, o_u);
    }
    dtPrev_ = dt;
    return dt;
}
