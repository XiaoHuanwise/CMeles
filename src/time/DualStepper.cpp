/// @file DualStepper.cpp
/// @brief Implementation of the dual time-stepping driver.

#include "DualStepper.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "common/Log.hpp"

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
              pseudoTable, params_.rkParams, oklDir) {
    if (params_.decoupled && nStages_ != 2) {
        throw std::invalid_argument(
            "DualStepper: decoupled mode requires a DITR residual");
    }

    // Buffers are allocated per mode/residual: coupled stages the guess in
    // o_uNew_, decoupled keeps the two stage guesses plus their refreshed
    // residuals and constructs the second pseudo stepper. o_Rn_ feeds the
    // two-stage residuals; o_uPrev_ is needed by U3R1 only — the U2/BE
    // residuals read the u-prev operand at zero coefficient, so advance()
    // aliases a live buffer into it instead.
    if (nStages_ == 2) {
        o_Rn_ = mem.wrapOrMalloc(nDof);
    }
    if (residual_->needsPrev()) {
        o_uPrev_ = mem.wrapOrMalloc(nDof);
    }
    if (params_.decoupled) {
        pseudoC2_ = std::make_unique<RungeKuttaStepper>(
            device, mem, nullptr, nDof, pseudoTable, params_.rkParams, oklDir);
        o_uNc2_  = mem.wrapOrMalloc(nDof);
        o_uN1_   = mem.wrapOrMalloc(nDof);
        o_Rnew0_ = mem.wrapOrMalloc(nDof);
        o_Rnew1_ = mem.wrapOrMalloc(nDof);
    } else {
        o_uNew_ = mem.wrapOrMalloc(nDof * nStages_);
    }
}

DualStepper::~DualStepper() {
    CMES_LOG_DETAIL << "DualStepper[" << name() << "] " << advances_
                    << " physical steps, total pseudo steps = "
                    << totalPseudoSteps_;
}

void DualStepper::pseudoStep(RungeKuttaStepper &ps) {
    if (params_.pseudoFixedDt > Real(0)) {
        ps.stepFixed(params_.pseudoFixedDt);
    } else {
        ps.stepPseudo(params_.allowReject);
    }
}

bool DualStepper::notConverged(Real fNorm, Real f0Norm) const {
    // A NaN/Inf residual norm must never pass as converged — a diverged
    // pseudo march would otherwise publish garbage silently. It runs to
    // maxPseudoSteps (warning) and fails loudly in the solver's checks.
    if (!std::isfinite(fNorm)) {
        return true;
    }
    return (fNorm / f0Norm > params_.rtol) && (fNorm > params_.atol);
}

void DualStepper::setPseudoStateFilter(StateFilter filter) {
    // Post-stage slots of the pseudo steppers: stage states and the
    // candidate state of every pseudo RK attempt pass through the filter
    // (positivity-type continuous projections only — the filtering
    // contract keeps WENO-type limiters out of the pseudo iteration).
    pseudo_.setPostStageFilter(filter);
    if (pseudoC2_) {
        pseudoC2_->setPostStageFilter(filter);
    }
}

const char *DualStepper::pseudoNormTag() const {
    switch (params_.pseudoNorm) {
        case 1:
            return "|F|_2";
        case 2:
            return "|F|_mean";
        default:
            return "|F|_inf";
    }
}

Real DualStepper::residualNorm(const RungeKuttaStepper &ps, occa::dim_t n) {
    occa::memory f = ps.f();
    // Convergence norm of the pseudo residual ([time_marching]
    // pseudo_norm). All variants keep the per-DOF O(1) scale of the
    // infinity norm, so the rtol/atol semantics carry over unchanged:
    //   inf  — single worst DOF (strictest; a point singularity such as
    //          the four-state corner of the 2-D Riemann problem blocks
    //          the criterion long after the bulk has converged),
    //   l2   — RMS level sqrt(mean f^2), the standard CFD residual
    //          monitor (SU2 / Fluent scaled residuals), robust to
    //          isolated outlier DOFs,
    //   mean — mean |f| (L1 average), similar robustness, looser.
    switch (params_.pseudoNorm) {
        case 1:
            return sqrt(blas_.dot(n, f, f) / Real(n));
        case 2:
            return blas_.asum(n, f) / Real(n);
        default:
            return blas_.amax(n, f);
    }
}

Real DualStepper::advance(occa::memory o_u, Real /*t*/, Real dt) {
    // U3R1 needs u^{n-1} (kept from the previous step) and theta. The first
    // step has no u^{n-1} yet, so it runs with the U2R1 start-up
    // reconstruction; theta is irrelevant there (kept at 1).
    if (residual_->needsPrev()) {
        const bool startup = dtPrev_ <= Real(0);
        residual_->setStartup(startup);
        residual_->setTheta(startup ? Real(1) : dtPrev_ / dt);
    }

    // R(u^n) is required by the two-stage residuals.
    if (nStages_ == 2) {
        rhs_(o_u, o_Rn_);
    }

    Real f0Norm = Real(0);
    Real fNorm  = Real(0);
    int cnt     = 0;

    if (params_.decoupled) {
        // ---- Decoupled DITR: two per-stage pseudo steppers ----------------
        blas_.copy(nDof_, o_u, o_uNc2_);
        blas_.copy(nDof_, o_u, o_uN1_);
        blas_.copy(nDof_, o_Rn_, o_Rnew0_);
        blas_.copy(nDof_, o_Rn_, o_Rnew1_);

        occa::memory oUN = o_u, oRN = o_Rn_;
        // Zero-coefficient u-prev operand: a live buffer for the U2
        // variants (no o_uPrev_ allocation), the true u^{n-1} for U3R1.
        occa::memory oUP =
            residual_->needsPrev() ? occa::memory(o_uPrev_) : o_u;
        occa::memory oUN1 = o_uN1_, oRN1 = o_Rnew1_, oRN0 = o_Rnew0_;
        TemporalResidual &res = *residual_;
        pseudo_.setRhs([&res, oUN1, oRN0, oUN, oRN,
                        dt](occa::memory x, occa::memory F) mutable {
            res.temporalResidualStageN1(x, oRN0, oUN, oRN, dt, F);
        });
        pseudoC2_->setRhs([&res, oUN1, oRN1, oUN, oRN, oUP,
                           dt](occa::memory x, occa::memory F) mutable {
            res.temporalResidualStageC2(x, oUN1, oRN1, oUN, oRN, oUP, dt, F);
        });

        pseudoC2_->setState(o_uNc2_, dt);
        pseudo_.setState(o_uN1_, dt);

        f0Norm        = residualNorm(pseudo_, nDof_);
        fNorm         = f0Norm;
        Real fMinStag = fNorm;
        int stag      = 0;
        while (notConverged(fNorm, f0Norm) && cnt < params_.maxPseudoSteps) {
            pseudoStep(*pseudoC2_);
            occa::memory sC2 = pseudoC2_->state();
            blas_.copy(nDof_, sC2, o_uNc2_);
            rhs_(o_uNc2_, o_Rnew0_);
            pseudoStep(pseudo_);
            occa::memory sN1 = pseudo_.state();
            blas_.copy(nDof_, sN1, o_uN1_);
            rhs_(o_uN1_, o_Rnew1_);
            ++cnt;
            fNorm = residualNorm(pseudo_, nDof_);
            if (cnt <= kRefStep) {
                f0Norm = std::max(f0Norm, fNorm);
            }
            if (params_.stagnationExit) {
                // Improvement = a drop below 0.98 x best-so-far. The 0.98/10
                // pair is mechanism-level, not tuned: it separates the
                // frozen phase (0%/iter) from the slow tail (~3%/iter) —
                // see Params::stagnationExit for the sensitivity note.
                if (fNorm < Real(0.98) * fMinStag) {
                    fMinStag = fNorm;
                    stag     = 0;
                } else if (++stag >= 10) {
                    CMES_LOG_DETAIL << "DualStepper[" << name()
                                    << "] stagnation exit after " << cnt
                                    << " pseudo steps";
                    break;
                }
            }
        }
    } else {
        // ---- Coupled: one pseudo stepper on the stacked state -------------
        const auto stackedN = nDof_ * nStages_;
        if (nStages_ == 1) {
            blas_.copy(nDof_, o_u, o_uNew_);
        } else {
            occa::memory stage0 = o_uNew_.slice(0, nDof_);
            occa::memory stage1 = o_uNew_.slice(nDof_, nDof_);
            blas_.copy(nDof_, o_u, stage0);
            blas_.copy(nDof_, o_u, stage1);
        }

        occa::memory oUN = o_u, oRN = o_Rn_;
        // Zero-coefficient u-prev operand (see the decoupled branch).
        occa::memory oUP =
            residual_->needsPrev() ? occa::memory(o_uPrev_) : o_u;
        TemporalResidual &res = *residual_;
        pseudo_.setRhs(
            [&res, oUN, oUP, oRN, dt](occa::memory x, occa::memory F) mutable {
                res.temporalResidual(x, oUN, oRN, oUP, dt, F);
            });

        pseudo_.setState(o_uNew_, dt);

        f0Norm        = residualNorm(pseudo_, stackedN);
        fNorm         = f0Norm;
        Real fMinStag = fNorm;
        int stag      = 0;
        while (notConverged(fNorm, f0Norm) && cnt < params_.maxPseudoSteps) {
            pseudoStep(pseudo_);
            ++cnt;
            fNorm = residualNorm(pseudo_, stackedN);
            if (cnt <= kRefStep) {
                f0Norm = std::max(f0Norm, fNorm);
            }
            if (params_.stagnationExit) {
                // Improvement = a drop below 0.98 x best-so-far. The 0.98/10
                // pair is mechanism-level, not tuned: it separates the
                // frozen phase (0%/iter) from the slow tail (~3%/iter) —
                // see Params::stagnationExit for the sensitivity note.
                if (fNorm < Real(0.98) * fMinStag) {
                    fMinStag = fNorm;
                    stag     = 0;
                } else if (++stag >= 10) {
                    CMES_LOG_DETAIL << "DualStepper[" << name()
                                    << "] stagnation exit after " << cnt
                                    << " pseudo steps";
                    break;
                }
            }
        }
    }

    if (notConverged(fNorm, f0Norm) && cnt >= params_.maxPseudoSteps) {
        CMES_LOG_NORMAL << "DualStepper[" << name()
                        << "]: pseudo stepper hit max steps ("
                        << pseudoNormTag() << " = " << fNorm << ")";
    }

    ++advances_;
    totalPseudoSteps_ += cnt;
    CMES_LOG_DETAIL << "DualStepper[" << name() << "] step " << advances_
                    << ": pseudo_iters=" << cnt;

    // Publish: u^{n-1} <- u^n (U3R1 only), u^{n+1} <- the marched pseudo
    // state. The pseudo steppers own copies of the state (setState copies
    // the guess in), so the result must be read back from them: the coupled
    // state is the stacked [u^{n+c_2}, u^{n+1}] (publish the last stage),
    // the decoupled stage-n+1 stepper holds u^{n+1} directly.
    if (residual_->needsPrev()) {
        blas_.copy(nDof_, o_u, o_uPrev_);
        dtPrev_ = dt;
    }
    occa::memory pseudoState = pseudo_.state();
    if (params_.decoupled || nStages_ == 1) {
        blas_.copy(nDof_, pseudoState, o_u);
    } else {
        occa::memory stage1 = pseudoState.slice(nDof_, nDof_);
        blas_.copy(nDof_, stage1, o_u);
    }

    // Filter contract (implicit family): project the converged stage state
    // once, OUTSIDE the implicit solve — $u^{n+1} = \pi(u^*)$ with $u^*$ the
    // solution of the limiter-free temporal residual. The pseudo iterates
    // above deliberately carry no WENO-type limiter: embedding such a
    // projection in the iteration would stall the residual in a limit
    // cycle and fake convergence (rationale in
    // StepperBase::setPostStageFilter). Continuous positivity projections
    // installed via setPseudoStateFilter are the one exception and ride
    // the pseudo steppers' own post-stage slots. The data of the NEXT
    // physical step — R(u^n), the temporal-residual operands and u^{n-1}
    // for U3R1 (copied above from the already-filtered o_u) — therefore
    // sees limited physics by construction. The decoupled mode's internal
    // u^{n+c_2} belongs to the implicit solve as well and receives the
    // same treatment (positivity filter only inside, projection here).
    applyPostStageFilter(o_u);
    return dt;
}
