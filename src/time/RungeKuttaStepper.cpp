/// @file RungeKuttaStepper.cpp
/// @brief Implementation of the generic embedded Runge-Kutta stepper.

#include "RungeKuttaStepper.hpp"

#include <algorithm>
#include <cmath>

RungeKuttaStepper::RungeKuttaStepper(occa::device &device,
                                     DeviceMemoryManager &mem, RhsFunction rhs,
                                     occa::dim_t nDof,
                                     const ButcherTable &table, Params params,
                                     const std::string &oklDir)
    : StepperBase(device, mem, std::move(rhs), nDof, oklDir), table_(table),
      params_(params) {
    errorExponent_ = Real(1) / Real(table_.errorEstimatorOrder + 1);

    const int s       = table_.nStages;
    const auto nDofSz = static_cast<std::size_t>(nDof_);

    // Flatten the tableau (A row-major s x s, B length s, E length s + 1).
    aFlat_.resize(static_cast<std::size_t>(s) * static_cast<std::size_t>(s));
    bFlat_.resize(static_cast<std::size_t>(s));
    eFlat_.resize(static_cast<std::size_t>(s) + 1);
    for (int i = 0; i < s; ++i) {
        for (int j = 0; j < s; ++j) {
            aFlat_[static_cast<std::size_t>(i * s + j)] =
                table_.A[static_cast<std::size_t>(i)]
                        [static_cast<std::size_t>(j)];
        }
        bFlat_[static_cast<std::size_t>(i)] =
            table_.B[static_cast<std::size_t>(i)];
        eFlat_[static_cast<std::size_t>(i)] =
            table_.E[static_cast<std::size_t>(i)];
    }
    eFlat_[static_cast<std::size_t>(s)] = table_.E[static_cast<std::size_t>(s)];

    o_A_ = mem_.wrapOrMalloc(aFlat_.data(), aFlat_.size());
    o_B_ = mem_.wrapOrMalloc(bFlat_.data(), bFlat_.size());
    o_E_ = mem_.wrapOrMalloc(eFlat_.data(), eFlat_.size());

    // Stage-derivative buffer K has s + 1 rows (FSAL row for f(u_new)).
    o_K_       = mem_.wrapOrMalloc(nDofSz * (static_cast<std::size_t>(s) + 1));
    o_u_       = mem_.wrapOrMalloc(nDof_);
    o_uNew_    = mem_.wrapOrMalloc(nDof_);
    o_uStage_  = mem_.wrapOrMalloc(nDof_);
    o_f_       = mem_.wrapOrMalloc(nDof_);
    o_fNew_    = mem_.wrapOrMalloc(nDof_);
    o_err_     = mem_.wrapOrMalloc(nDof_);
    o_scratch_ = mem_.wrapOrMalloc(nDof_);

    // Per-DOF controller state (local mode only). In the coupled dual-time
    // mode nDof covers the stacked stages, so each stage row carries its own
    // step sizes (prototype: pseudo_dt = empty_like(F)); in decoupled mode
    // every pseudo stepper owns one array of length N.
    if (params_.localDt) {
        o_dt_        = mem_.wrapOrMalloc(nDof_);
        o_sigma_     = mem_.wrapOrMalloc(nDof_);
        o_sigmaPrev_ = mem_.wrapOrMalloc(nDof_);
    }
}

void RungeKuttaStepper::setRhs(RhsFunction rhs) {
    rhs_ = std::move(rhs);
}

void RungeKuttaStepper::rkStep(occa::memory &o_u, occa::memory &o_uNew,
                               occa::memory &o_fNew, bool useLocalDt) {
    ensureRkKernels();
    if (useLocalDt) {
        ensureLocalKernels();
    }
    const int n    = static_cast<int>(nDof_);
    const int s    = table_.nStages;
    const auto nSz = static_cast<std::size_t>(nDof_);

    // K[0] = f.
    occa::memory k0 = o_K_.slice(0, nDof_);
    blas_.copy(nDof_, o_f_, k0);

    // Same Butcher recurrence with a scalar dt or a per-DOF dt array; only
    // the launched kernels differ.
    for (int stage = 1; stage < s; ++stage) {
        if (useLocalDt) {
            rkStageCombineLocalDt_(n, o_dt_, stage, s, o_A_, o_u, o_K_,
                                   o_uStage_);
        } else {
            rkStageCombine_(n, dt_, stage, s, o_A_, o_u, o_K_, o_uStage_);
        }
        occa::memory kS =
            o_K_.slice(nSz * static_cast<std::size_t>(stage), nSz);
        rhs_(o_uStage_, kS);
    }

    if (useLocalDt) {
        rkFinalUpdateLocalDt_(n, o_dt_, s, o_B_, o_u, o_K_, o_uNew);
    } else {
        rkFinalUpdate_(n, dt_, s, o_B_, o_u, o_K_, o_uNew);
    }
    rhs_(o_uNew_, o_fNew);

    // FSAL row: K[s] = f_new (consumed by the error estimate).
    occa::memory kLast = o_K_.slice(nSz * static_cast<std::size_t>(s), nSz);
    blas_.copy(nDof_, o_fNew_, kLast);
}

Real RungeKuttaStepper::computeErrorNorm(Real rmsU, Real rmsUnew) {
    ensureRkKernels();
    const int n = static_cast<int>(nDof_);
    rkWeightedSum_(n, dt_, table_.nStages + 1, o_E_, o_K_, o_err_);

    // sigma = RMS(err) / (atol + max(RMS(u), RMS(u_new)) * rtol), floored
    // at 1e-14 (prototype).
    const Real scale = params_.atol + std::max(rmsU, rmsUnew) * params_.rtol;
    Real sigma = blas_.nrm2(nDof_, o_err_) / (std::sqrt(Real(nDof_)) * scale);
    return std::max(sigma, Real(1e-14));
}

void RungeKuttaStepper::updateDt(Real sigma, bool accepted, bool wasRejected) {
    Real factor = params_.safety * std::pow(sigma, -kAlpha * errorExponent_) *
                  std::pow(errorNormPrev_, kBeta * errorExponent_);
    if (accepted) {
        factor = (sigma == Real(0)) ? params_.maxFactor
                                    : std::min(params_.maxFactor, factor);
        if (wasRejected) {
            factor = std::min(factor, Real(1));
        }
    } else {
        factor = std::max(params_.minFactor, factor);
    }

    dt_ *= factor;
    dt_ = std::min(dt_, params_.maxGrowth * dtInit_);
    dt_ = std::max(dt_, params_.minStep);
    dt_ = std::min(dt_, maxStep_);
}

Real RungeKuttaStepper::computeErrorNormLocal() {
    ensureLocalKernels();
    const int n = static_cast<int>(nDof_);
    rkWeightedSumLocalDt_(n, o_dt_, table_.nStages + 1, o_E_, o_K_, o_err_);
    rkErrorNormLocal_(n, params_.rtol, params_.atol, o_u_, o_uNew_, o_err_,
                      o_sigma_);
    // Accept test of the prototype's per-DOF controller: max sigma < 1.
    return blas_.amax(nDof_, o_sigma_);
}

void RungeKuttaStepper::updateDtLocal(bool accepted, bool wasRejected) {
    ensureLocalKernels();
    const int n = static_cast<int>(nDof_);
    rkDtUpdateLocal_(n, -kAlpha * errorExponent_, kBeta * errorExponent_,
                     params_.safety, params_.minFactor, params_.maxFactor,
                     params_.maxGrowth * dtInit_, params_.minStep, maxStep_,
                     (accepted && wasRejected) ? 1 : 0, o_sigma_, o_sigmaPrev_,
                     o_dt_);
}

Real RungeKuttaStepper::dtMinLocal() {
    return blas_.amin(nDof_, o_dt_);
}

Real RungeKuttaStepper::selectInitialStep(occa::memory &o_u) {
    // Hairer's heuristic (prototype _select_initial_step_sdt).
    const Real rmsU  = blas_.nrm2(nDof_, o_u) / std::sqrt(Real(nDof_));
    const Real rmsF  = blas_.nrm2(nDof_, o_f_) / std::sqrt(Real(nDof_));
    const Real scale = params_.atol + rmsU * params_.rtol;

    const Real d0 = rmsU / scale;
    const Real d1 = rmsF / scale;
    const Real h0 = (d0 < Real(1e-5) || d1 < Real(1e-5)) ? Real(1e-6)
                                                         : Real(0.01) * d0 / d1;

    // Trial step: f1 = R(u + h0 * f).
    eulerUpdate(o_u, h0, o_f_, o_uStage_);
    rhs_(o_uStage_, o_scratch_);

    // d2 = RMS((f1 - f0) / scale) / h0.
    blas_.copy(nDof_, o_scratch_, o_err_);
    blas_.axpy(nDof_, Real(-1), o_f_, o_err_);
    const Real d2 =
        blas_.nrm2(nDof_, o_err_) / (std::sqrt(Real(nDof_)) * scale * h0);

    const Real h1 =
        (d1 <= Real(1e-15) && d2 <= Real(1e-15))
            ? std::max(Real(1e-6), h0 * Real(1e-3))
            : std::pow(Real(0.01) / std::max(d1, d2), errorExponent_);
    return std::min(std::min(Real(100) * h0, h1), maxStep_);
}

Real RungeKuttaStepper::advance(occa::memory o_u, Real /*t*/, Real dtMax) {
    if (!stateValid_) {
        rhs_(o_u, o_f_);
        maxStep_    = dtMax;
        dt_         = std::min(selectInitialStep(o_u), dtMax);
        dtInit_     = dt_;
        stateValid_ = true;
    }
    maxStep_ = std::min(maxStep_, dtMax);
    dt_      = std::min(dt_, maxStep_);

    bool wasRejected = false;
    Real sigma       = Real(1);
    Real taken       = dt_;
    for (;;) {
        taken           = dt_;
        const Real rmsU = blas_.nrm2(nDof_, o_u) / std::sqrt(Real(nDof_));
        rkStep(o_u, o_uNew_, o_fNew_, /*useLocalDt=*/false);
        sigma = computeErrorNorm(rmsU, blas_.nrm2(nDof_, o_uNew_) /
                                           std::sqrt(Real(nDof_)));
        if (sigma < Real(1)) {
            updateDt(sigma, /*accepted=*/true, wasRejected);
            break;
        }
        // Guard against an endless loop when the step has bottomed out at
        // minStep: the attempt just made already ran at the floor step size,
        // so commit it as-is — the smallest-error option among the steps the
        // controller would otherwise force through, and retrying would repeat
        // the identical attempt forever.
        if (dt_ <= params_.minStep) {
            break;
        }
        updateDt(sigma, /*accepted=*/false, wasRejected);
        wasRejected = true;
    }

    blas_.copy(nDof_, o_uNew_, o_u);
    blas_.copy(nDof_, o_fNew_, o_f_);
    errorNormPrev_ = sigma;
    return taken;
}

void RungeKuttaStepper::setState(occa::memory &o_u0, Real phyDtCap) {
    blas_.copy(nDof_, o_u0, o_u_);
    rhs_(o_u_, o_f_);
    maxStep_ = phyDtCap;
    dt_      = std::min(selectInitialStep(o_u_), phyDtCap);
    dtInit_  = dt_;
    if (params_.localDt) {
        // Uniform fill with the scalar heuristic (prototype set_new_state):
        // per-DOF divergence comes from the PI updates, not the seed.
        ensureLocalKernels();
        const int n = static_cast<int>(nDof_);
        fillReal_(n, dt_, o_dt_);
        fillReal_(n, Real(1), o_sigmaPrev_);
    }
    errorNormPrev_ = Real(1);
    stateValid_    = true;
}

void RungeKuttaStepper::stepPseudo(bool allowReject) {
    if (params_.localDt) {
        // Per-DOF controller (prototype step(), non-cell branch): one PI
        // factor per DOF, accept iff max sigma < 1. The dt update runs on
        // every attempt (factor clamps apply unconditionally, unlike the
        // scalar controller); sigma_prev is committed from the final
        // attempt only.
        bool wasRejected = false;
        for (;;) {
            rkStep(o_u_, o_uNew_, o_fNew_, /*useLocalDt=*/true);
            const bool accepted = computeErrorNormLocal() < Real(1);
            updateDtLocal(accepted, wasRejected);
            if (accepted) {
                break;
            }
            // Single-attempt mode (prototype step(reject=False)): the
            // attempt is unconditionally accepted, but the controller still
            // adapts dt for the next pseudo step.
            if (!allowReject) {
                break;
            }
            // Guard against an endless loop when the step has bottomed out
            // at minStep
            if (dtMinLocal() <= params_.minStep) {
                break;
            }
            wasRejected = true;
        }

        blas_.copy(nDof_, o_sigma_, o_sigmaPrev_);
        blas_.copy(nDof_, o_uNew_, o_u_);
        blas_.copy(nDof_, o_fNew_, o_f_);
        return;
    }

    bool wasRejected = false;
    Real sigma       = Real(1);
    for (;;) {
        const Real rmsU = blas_.nrm2(nDof_, o_u_) / std::sqrt(Real(nDof_));
        rkStep(o_u_, o_uNew_, o_fNew_, /*useLocalDt=*/false);
        sigma = computeErrorNorm(rmsU, blas_.nrm2(nDof_, o_uNew_) /
                                           std::sqrt(Real(nDof_)));
        if (sigma < Real(1)) {
            updateDt(sigma, /*accepted=*/true, wasRejected);
            break;
        }
        // Single-attempt mode (prototype step(reject=False)): the attempt is
        // unconditionally accepted, but the controller still adapts dt for
        // the next pseudo step.
        if (!allowReject) {
            updateDt(sigma, /*accepted=*/false, wasRejected);
            break;
        }
        // Guard against an endless loop when the step has bottomed out at
        // minStep: the attempt just made already ran at the floor step size,
        // so commit it as-is (retrying would repeat the identical attempt
        // forever).
        if (dt_ <= params_.minStep) {
            break;
        }
        updateDt(sigma, /*accepted=*/false, wasRejected);
        wasRejected = true;
    }

    blas_.copy(nDof_, o_uNew_, o_u_);
    blas_.copy(nDof_, o_fNew_, o_f_);
    errorNormPrev_ = sigma;
}

void RungeKuttaStepper::stepFixed(Real dt) {
    dt_ = dt;
    rkStep(o_u_, o_uNew_, o_fNew_, /*useLocalDt=*/false);
    blas_.copy(nDof_, o_uNew_, o_u_);
    blas_.copy(nDof_, o_fNew_, o_f_);
}
