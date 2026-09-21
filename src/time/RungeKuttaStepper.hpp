/// @file RungeKuttaStepper.hpp
/// @brief Generic embedded Runge-Kutta stepper with adaptive step control.
///
/// One class serves all six embedded pairs: the Butcher tableau is runtime
/// data (uploaded once, read by the OKL kernels), so the variants differ by
/// construction argument, not by code.
///
/// Ported from the prototype's RungeKuttaStepper. Two step-size controllers
/// exist, selected by Params::localDt:
///
///  - Global (localDt = false, the default): the scalar controller of the
///    prototype's `step_single_dt` — RMS-scaled error norm, scalar PI
///    factor. Serves both the explicit adaptive RK arm (advance()) and the
///    dual-time pseudo stepper.
///  - Local (localDt = true): the per-DOF controller of the prototype's
///    `step()` — one pseudo step size per modal coefficient
///    $u_{lkj}$ (PyFR-style local adaptive pseudo time stepping). Pseudo
///    stepping only: it drives stepPseudo() of the dual time-stepping
///    driver; advance() keeps the global controller (a per-DOF physical
///    step would be local time stepping, which is out of scope).
///
/// The prototype's `step(reject=False)` semantics (one attempt, forced
/// accept, dt still updated by the PI factor) are preserved in stepPseudo()
/// for both controllers.
///
/// Controller constants (prototype / docs): SAFETY = 0.9, MIN_FACTOR = 0.2,
/// MAX_FACTOR = 5, MAX_GROWTH = 10, ALPHA = 0.7, BETA = 0.4. SAFETY /
/// MIN_FACTOR / MAX_FACTOR / MAX_GROWTH live in Params and are configurable
/// through the [time_marching] keys safety / min_factor / max_factor /
/// max_growth; ALPHA and BETA remain compile-time constants.

#pragma once

#include "ButcherTable.hpp"
#include "StepperBase.hpp"

class RungeKuttaStepper : public StepperBase {
public:
    /// @brief Tolerances, step bounds and PI-controller knobs.
    struct Params {
        Real rtol    = Real(1e-6);
        Real atol    = Real(1e-6);
        Real minStep = Real(10) * RealEpsilon;
        Real maxStep = std::numeric_limits<Real>::infinity();

        // PI-controller knobs (formerly class constants), configurable
        // through [time_marching] safety / min_factor / max_factor /
        // max_growth (see updateDt).
        Real safety    = Real(0.9);
        Real minFactor = Real(0.2);
        Real maxFactor = Real(5);
        Real maxGrowth = Real(10);

        /// @brief Per-DOF (modal-coefficient-level) adaptive pseudo step
        ///        sizes instead of a single scalar dt config key
        ///        [time_marching] pseudo_dt_mode = "local"). Pseudo
        ///        stepping only; advance() always uses the global
        ///        controller.
        bool localDt = false;
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
    ///        accepted step. Always the global controller — the local-dt
    ///        mode is a pseudo-stepping feature and is ignored here.
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
    ///        (= set_new_state in the prototype). In local-dt mode the
    ///        per-DOF dt array is filled uniformly with that scalar
    ///        (divergence comes from the PI updates) and sigma_prev is
    ///        reset to one.
    void setState(occa::memory &o_u0, Real phyDtCap);

    /// @brief One pseudo step with the PI controller (global or local,
    ///        per Params::localDt). With \p allowReject false the attempt
    ///        is unconditionally accepted (single attempt), matching
    ///        step(reject=False).
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
    /// @brief Current (controlled) step size (global controller).
    Real dt() const noexcept {
        return dt_;
    }
    /// @brief Last accepted error norm (global controller).
    Real errorNorm() const noexcept {
        return errorNormPrev_;
    }
    /// @brief Per-DOF step-size array (local-dt mode; pseudo stepping).
    const occa::memory &localDt() const {
        return o_dt_;
    }

    // PI controller exponents (compile-time; the gain/clamp knobs live in
    // Params so they can be configured per run).
    static constexpr Real kAlpha = Real(0.7);
    static constexpr Real kBeta  = Real(0.4);

private:
    /// @brief One Butcher step: stages from (u, f), writes u_new and
    ///        f_new = R(u_new) (FSAL); K[0..s] filled. u is not modified.
    ///        Launches the scalar-dt or per-DOF-dt kernels per \p
    ///        useLocalDt; the recurrence is identical. advance() and
    ///        stepFixed() always pass false (local dt is a pseudo-stepping
    ///        mode and their dt is scalar).
    void rkStep(occa::memory &o_u, occa::memory &o_uNew, occa::memory &o_fNew,
                bool useLocalDt);

    /// @brief RMS-scaled error norm sigma of the last attempt (K, dt).
    Real computeErrorNorm(Real rmsU, Real rmsUnew);

    /// @brief Per-DOF normalised error of the last attempt into o_sigma_
    ///        (floor 1e-14); returns $\max_d \sigma_d$ for the accept test.
    Real computeErrorNormLocal();

    /// @brief Hairer initial-step heuristic (= _select_initial_step_sdt).
    Real selectInitialStep(occa::memory &o_u);

    /// @brief Advance the internal scalar step-size controller after an
    ///        attempt with error norm \p sigma.
    void updateDt(Real sigma, bool accepted, bool wasRejected);

    /// @brief Advance the per-DOF step-size controller (one rkDtUpdateLocal
    ///        launch; factor clamps apply on every attempt).
    void updateDtLocal(bool accepted, bool wasRejected);

    /// @brief $\min_d dt_d$ of the per-DOF array (Blas::amin).
    Real dtMinLocal();

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

    /// @brief Lazily compile the local-dt controller kernels (idempotent).
    void ensureLocalKernels() {
        if (localKernelsBuilt_) {
            return;
        }
        fillReal_              = buildTimeKernel("fillReal");
        rkStageCombineLocalDt_ = buildTimeKernel("rkStageCombineLocalDt");
        rkWeightedSumLocalDt_  = buildTimeKernel("rkWeightedSumLocalDt");
        rkFinalUpdateLocalDt_  = buildTimeKernel("rkFinalUpdateLocalDt");
        rkErrorNormLocal_      = buildTimeKernel("rkErrorNormLocal");
        rkDtUpdateLocal_       = buildTimeKernel("rkDtUpdateLocal");
        localKernelsBuilt_     = true;
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

    // Local-dt controller state (allocated only when params_.localDt).
    occa::memory o_dt_;        ///< Per-DOF pseudo step sizes.
    occa::memory o_sigma_;     ///< Per-DOF normalised error of the attempt.
    occa::memory o_sigmaPrev_; ///< Per-DOF error of the last commit.

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

    // Local-dt controller kernels (lazily built by ensureLocalKernels()).
    occa::kernel fillReal_;
    occa::kernel rkStageCombineLocalDt_;
    occa::kernel rkWeightedSumLocalDt_;
    occa::kernel rkFinalUpdateLocalDt_;
    occa::kernel rkErrorNormLocal_;
    occa::kernel rkDtUpdateLocal_;
    bool localKernelsBuilt_ = false;
};
