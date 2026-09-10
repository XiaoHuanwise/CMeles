/// @file RungeKuttaStepper.hpp
/// @brief Generic embedded Runge-Kutta stepper with adaptive step control.
///
/// One class serves all six embedded pairs: the Butcher tableau is runtime
/// data (uploaded once, read by the OKL kernels), so the variants differ by
/// construction argument, not by code.
///
/// Ported from the prototype's RungeKuttaStepper. The scalar-dt controller
/// of `step_single_dt` (RMS-scaled error, PI factor) is the single
/// controller here, since CMeles advances a global dt; the prototype's
/// per-DOF controller exists only for its local-dt tensor mode. The
/// prototype's `step(reject=False)` semantics (one attempt, forced accept,
/// dt still updated by the PI factor) are preserved in stepPseudo() for
/// the dual time-stepping driver.
///
/// Controller constants (prototype / docs): SAFETY = 0.9, MIN_FACTOR = 0.2,
/// MAX_FACTOR = 5, MAX_GROWTH = 10, ALPHA = 0.7, BETA = 0.4.

#pragma once

#include "ButcherTable.hpp"
#include "StepperBase.hpp"

class RungeKuttaStepper : public StepperBase {
public:
    /// @brief Tolerances and step bounds.
    struct Params {
        Real rtol    = Real(1e-6);
        Real atol    = Real(1e-6);
        Real minStep = Real(10) * RealEpsilon;
        Real maxStep = std::numeric_limits<Real>::infinity();
    };

    RungeKuttaStepper(occa::device &device, DeviceMemoryManager &mem,
                      RhsFunction rhs, occa::dim_t nDof,
                      const ButcherTable &table, Params params,
                      const std::string &oklDir = OCCA_OKL_DIR);

    /// @brief Overload with default tolerances.
    RungeKuttaStepper(occa::device &device, DeviceMemoryManager &mem,
                      RhsFunction rhs, occa::dim_t nDof,
                      const ButcherTable &table,
                      const std::string &oklDir = OCCA_OKL_DIR)
        : RungeKuttaStepper(device, mem, std::move(rhs), nDof, table, Params(),
                            oklDir) {
    }

    // ---- StepperBase interface ----

    /// @brief One adaptive physical step in place on \p o_u (internal
    ///        accept/reject; \p dt is the step-size cap). Returns the
    ///        accepted step.
    Real advance(occa::memory o_u, Real t, Real dtMax) override;

    int order() const override {
        return table_.order;
    }
    const char *name() const override {
        return table_.name;
    }

    // ---- Stateful pseudo-stepping API (dual time stepping) ----

    /// @brief Rebind the right-hand side (the temporal residual changes
    ///        every physical step).
    void setRhs(RhsFunction rhs);

    /// @brief Install a state $u_0$ and (re)initialise the step size via
    ///        the Hairer heuristic, capped by the physical step
    ///        (= set_new_state in the prototype).
    void setState(occa::memory &o_u0, Real phyDtCap);

    /// @brief One pseudo step with the PI controller. With
    ///        \p allowReject false the attempt is unconditionally accepted
    ///        (single attempt), matching step(reject=False).
    void stepPseudo(bool allowReject);

    /// @brief One fixed-step pseudo step without control (= step_set_sdt).
    void stepFixed(Real dt);

    /// @brief Current internal state $u$ (pseudo stepping).
    const occa::memory &state() const {
        return o_u_;
    }
    /// @brief Current right-hand-side evaluation $f = R(u)$.
    const occa::memory &f() const {
        return o_f_;
    }
    /// @brief Current (controlled) step size.
    Real dt() const noexcept {
        return dt_;
    }
    /// @brief Last accepted error norm.
    Real errorNorm() const noexcept {
        return errorNormPrev_;
    }

    // PI controller constants (prototype).
    static constexpr Real kSafety    = Real(0.9);
    static constexpr Real kMinFactor = Real(0.2);
    static constexpr Real kMaxFactor = Real(5);
    static constexpr Real kMaxGrowth = Real(10);
    static constexpr Real kAlpha     = Real(0.7);
    static constexpr Real kBeta      = Real(0.4);

private:
    /// @brief One Butcher step: stages from (u, f), writes u_new and
    ///        f_new = R(u_new) (FSAL); K[0..s] filled. u is not modified.
    void rkStep(occa::memory &o_u, occa::memory &o_uNew, occa::memory &o_fNew);

    /// @brief RMS-scaled error norm sigma of the last attempt (K, dt).
    Real computeErrorNorm(Real rmsU, Real rmsUnew);

    /// @brief Hairer initial-step heuristic (= _select_initial_step_sdt).
    Real selectInitialStep(occa::memory &o_u);

    /// @brief Advance the internal step-size controller after an attempt
    ///        with error norm \p sigma.
    void updateDt(Real sigma, bool accepted, bool wasRejected);

    /// @brief Lazily compile the Butcher kernels (idempotent).
    void ensureRkKernels() {
        if (rkKernelsBuilt_) {
            return;
        }
        rkStageCombine_ = buildTimeKernel("rkStageCombine");
        rkWeightedSum_  = buildTimeKernel("rkWeightedSum");
        rkFinalUpdate_  = buildTimeKernel("rkFinalUpdate");
        rkKernelsBuilt_ = true;
    }

    const ButcherTable &table_;
    Params params_;
    Real errorExponent_;

    // Flattened tableau uploads (host sources kept as members: wrapOrMalloc
    // aliases them on unified backends).
    std::vector<Real> aFlat_, bFlat_, eFlat_;
    occa::memory o_A_, o_B_, o_E_;

    // Stepper state.
    occa::memory o_u_;      ///< Internal state (pseudo stepping).
    occa::memory o_uNew_;   ///< Candidate state of an attempt.
    occa::memory o_uStage_; ///< Stage states.
    occa::memory o_f_;      ///< Current f = R(u) (FSAL across steps).
    occa::memory o_fNew_;   ///< Candidate f (kept on accept).
    occa::memory o_K_;      ///< Stage derivatives, (s + 1) x nDof.
    occa::memory o_err_;    ///< Embedded error vector.
    occa::memory o_scratch_;

    Real dt_            = Real(0);
    Real dtInit_        = Real(0);
    Real maxStep_       = std::numeric_limits<Real>::infinity();
    Real errorNormPrev_ = Real(1);
    bool stateValid_    = false;

    // Butcher kernels (lazily built by ensureRkKernels()).
    occa::kernel rkStageCombine_;
    occa::kernel rkWeightedSum_;
    occa::kernel rkFinalUpdate_;
    bool rkKernelsBuilt_ = false;
};
