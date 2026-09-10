/// @file ImplicitResidual.hpp
/// @brief Implicit temporal-residual builders: Backward Euler and DITR.
///
/// These classes do not advance the state — they build the temporal
/// residual $\mathcal{F} = \mathcal{G}(u^{n+1})$ whose root is the implicit
/// step; a DualStepper drives them to convergence through pseudo time
/// (the role of ImplicitStepper in the .stepper.py prototype: one unified
/// residual signature, with the single-stage scheme ignoring the
/// multi-stage arguments).
///
/// Backward Euler:
///
/// $$ \mathcal{F} = \frac{u^n - u^{n+1}}{\Delta t} + \mathcal{R}(u^{n+1}) $$
///
/// DITR (deferred-implicit time reconstruction, see
/// docs/tech_docs/time_marching/implicit_time_marching.md) reconstructs the
/// stage values $u^{n+c_2}$ and $u^{n+1}$ with the reconstruction
/// coefficients $a$ and $d$:
///
/// $$ \begin{aligned} F_0 &= \frac{a_0 u^{n-1} + a_1 u^n + a_2 u^{n+1} - u^{n+c_2}}{\Delta t} + d_1 R^n + d_2 R^{n+1} \\ F_1 &= \frac{u^n - u^{n+1}}{\Delta t} + b_1 R^n + b_2 R^{n+c_2} + b_3 R^{n+1} \end{aligned} $$
///
/// with the quadrature weights
///
/// $$b = [0, \frac12 - \frac{1}{6 c_2}, \frac{1}{6 c_2 (1 - c_2)}, \frac12 - \frac{1}{6 (1 - c_2)}]$$
///
/// and the coupling preconditioner $F_0 \mathrel{+}= \beta F_1$
///
/// $$\mathbf{P} = \begin{bmatrix} I & \beta I \\ 0 & I \end{bmatrix}$$
///
/// ($\beta = 1$).
/// The U2R2/U2R1/U3R1 variants differ only in the coefficient values
/// ($a_0 = 0$ and $d_1 = 0$ except where noted; U3R1 additionally uses
/// $\theta = \Delta t^{n-1} / \Delta t^n$ and $u^{n-1}$).
///
/// Coupled mode stacks the two stages into one $2 N_{dof}$ state and one
/// pseudo stepper; decoupled mode advances each stage with its own pseudo
/// stepper using the stage residuals below (the preconditioner is folded
/// into the stage-0 expression).

#pragma once

#include <occa.hpp>

#include <stdexcept>
#include <string>

#include "blas/Blas.hpp"
#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "time/TimeTypes.hpp"

/// @brief Abstract implicit temporal residual driven to its root by
///        dual time stepping.
///
/// Derived classes implement the coupled residual with one unified
/// signature: Backward Euler (\p nStages == 1) ignores \p o_Rn and
/// \p o_uPrev; DITR (\p nStages == 2) consumes them. The decoupled
/// stage residuals only exist for two-stage schemes and default to an
/// error on single-stage ones.
class TemporalResidual {
public:
    virtual ~TemporalResidual() = default;

    /// @brief Number of stacked implicit stages (size of the pseudo state).
    virtual int nStages() const = 0;

    /// @brief Whether $u^{n-1}$ is required (DITR U3R1 only).
    virtual bool needsPrev() const = 0;

    /// @brief Set $\theta = \Delta t^{n-1} / \Delta t^n$; no-op except DITR
    ///        U3R1 (rebuilds the reconstruction coefficients).
    virtual void setTheta(Real /*theta*/) {
    }

    /// @brief Coupled residual of the stacked implicit state.
    /// @param o_uNew  Implicit guess: $u^{n+1}$ (single stage) or the
    ///                stacked $[u^{n+c_2}, u^{n+1}]$ (two stages).
    /// @param o_u     Current state $u^n$.
    /// @param o_Rn    $R(u^n)$ (two stages only; ignored by Backward Euler).
    /// @param o_uPrev $u^{n-1}$ (DITR U3R1 only; ignored otherwise).
    /// @param dt      Physical time step $\Delta t$.
    /// @param o_F     Output residual, stacked for two stages (with the
    ///                coupling preconditioner applied).
    virtual void temporalResidual(occa::memory &o_uNew, occa::memory &o_u,
                                  occa::memory &o_Rn, occa::memory &o_uPrev,
                                  Real dt, occa::memory &o_F) = 0;

    /// @brief Decoupled stage-$n{+}c_2$ residual (DITR only).
    virtual void temporalResidualStageC2(occa::memory & /*o_uNc2*/,
                                         occa::memory & /*o_uN1*/,
                                         occa::memory & /*o_Rn1*/,
                                         occa::memory & /*o_u*/,
                                         occa::memory & /*o_Rn*/,
                                         occa::memory & /*o_uPrev*/,
                                         Real /*dt*/, occa::memory & /*o_F0*/) {
        throw std::logic_error(
            "temporalResidualStageC2: requires a two-stage residual (DITR)");
    }

    /// @brief Decoupled stage-$n{+}1$ residual (DITR only).
    virtual void temporalResidualStageN1(occa::memory & /*o_uN1*/,
                                         occa::memory & /*o_Rnc2*/,
                                         occa::memory & /*o_u*/,
                                         occa::memory & /*o_Rn*/, Real /*dt*/,
                                         occa::memory & /*o_F1*/) {
        throw std::logic_error(
            "temporalResidualStageN1: requires a two-stage residual (DITR)");
    }

protected:
    /// @param device  OCCA device (lifetime must exceed this object).
    /// @param mem     Device memory manager (lifetime must exceed this
    ///                object).
    /// @param rhs     Right-hand side $\mathcal{R}(u)$.
    /// @param nDof    Length of the ODE vector.
    /// @param oklDir  Directory containing the .okl kernel sources.
    TemporalResidual(occa::device &device, DeviceMemoryManager &mem,
                     RhsFunction rhs, occa::dim_t nDof,
                     const std::string &oklDir);

    /// @brief out = c0*x0 + c1*x1 + c2*x2 + c3*x3 over nDof entries.
    void combine4(Real c0, occa::memory &o_x0, Real c1, occa::memory &o_x1,
                  Real c2, occa::memory &o_x2, Real c3, occa::memory &o_x3,
                  occa::memory &o_out);

    /// @brief out = c0*x0 + ... + c4*x4 over nDof entries.
    void combine5(Real c0, occa::memory &o_x0, Real c1, occa::memory &o_x1,
                  Real c2, occa::memory &o_x2, Real c3, occa::memory &o_x3,
                  Real c4, occa::memory &o_x4, occa::memory &o_out);

    /// @brief out = c0*x0 + ... + c6*x6 over nDof entries.
    void combine7(Real c0, occa::memory &o_x0, Real c1, occa::memory &o_x1,
                  Real c2, occa::memory &o_x2, Real c3, occa::memory &o_x3,
                  Real c4, occa::memory &o_x4, Real c5, occa::memory &o_x5,
                  Real c6, occa::memory &o_x6, occa::memory &o_out);

    occa::device &device_;
    DeviceMemoryManager &mem_;
    RhsFunction rhs_;
    occa::dim_t nDof_;
    std::string oklDir_;
    Blas blas_;

private:
    occa::kernel vecCombine4_;
    occa::kernel vecCombine5_;
    occa::kernel vecCombine7_;
};

/// @brief Backward Euler temporal residual.
class BackwardEulerResidual : public TemporalResidual {
public:
    BackwardEulerResidual(occa::device &device, DeviceMemoryManager &mem,
                          RhsFunction rhs, occa::dim_t nDof,
                          const std::string &oklDir = OCCA_OKL_DIR);

    /// @brief $\mathcal{F} = (u^n - u^{n+1})/\Delta t + R(u^{n+1})$;
    ///        \p o_Rn and \p o_uPrev are ignored (single stage).
    void temporalResidual(occa::memory &o_uNew, occa::memory &o_u,
                          occa::memory &o_Rn, occa::memory &o_uPrev, Real dt,
                          occa::memory &o_F) override;

    /// @brief Number of stacked implicit stages (size of the pseudo state).
    int nStages() const override {
        return 1;
    }
    /// @brief Whether $u^{n-1}$ is required (DITR U3R1 only).
    bool needsPrev() const override {
        return false;
    }

private:
    occa::memory o_R_; ///< R(u^{n+1}) scratch.
};

/// @brief DITR temporal residual (U2R2 / U2R1 / U3R1).
class DitrResidual : public TemporalResidual {
public:
    /// @brief DITR reconstruction variant. U = number of solution values in
    ///        the reconstruction stencil, R = number of residual values (the
    ///        quadrature weights themselves are shared by all variants).
    enum class Variant {
        U2R2 = 0, ///< 2 u-values + 2 residuals: cubic Hermite reconstruction.
        U2R1 = 1, ///< 2 u-values + 1 residual.
        U3R1 = 2  ///< 3 u-values (incl. u_prev) + 1 residual.
    };

    DitrResidual(occa::device &device, DeviceMemoryManager &mem,
                 RhsFunction rhs, occa::dim_t nDof, Variant variant,
                 Real c2 = Real(0.5), Real beta = Real(1),
                 const std::string &oklDir = OCCA_OKL_DIR);

    /// @brief Set $\theta = \Delta t^{n-1} / \Delta t^n$ (U3R1 only;
    ///        rebuilds the reconstruction coefficients).
    void setTheta(Real theta) override;

    // ---- Coupled residuals (stacked 2*N_dof states) ----

    /// @brief Both stage residuals of the stacked state
    ///        $u_{new} = [u^{n+c_2}, u^{n+1}]$, with the preconditioner
    ///        applied: $F = [F_0 + \beta F_1, F_1]$.
    void temporalResidual(occa::memory &o_uNew2N, occa::memory &o_u,
                          occa::memory &o_Rn, occa::memory &o_uPrev, Real dt,
                          occa::memory &o_F2N) override;

    // ---- Decoupled per-stage residuals ----

    /// @brief Stage $n + c_2$ residual with the preconditioner folded in;
    ///        \p o_uN1 / \p o_Rn1 are the *current* values of the $n+1$
    ///        stage.
    void temporalResidualStageC2(occa::memory &o_uNc2, occa::memory &o_uN1,
                                 occa::memory &o_Rn1, occa::memory &o_u,
                                 occa::memory &o_Rn, occa::memory &o_uPrev,
                                 Real dt, occa::memory &o_F0) override;

    /// @brief Stage $n+1$ residual; \p o_Rnc2 is the *current* residual of
    ///        the $n+c_2$ stage.
    void temporalResidualStageN1(occa::memory &o_uN1, occa::memory &o_Rnc2,
                                 occa::memory &o_u, occa::memory &o_Rn, Real dt,
                                 occa::memory &o_F1) override;

    int nStages() const override {
        return 2;
    }
    bool needsPrev() const override {
        return variant_ == Variant::U3R1;
    }
    Variant variant() const {
        return variant_;
    }

private:
    /// @brief Recompute a/d from (variant, c2, theta).
    void rebuildCoefficients();

    Variant variant_;
    Real c2_;
    Real beta_;
    Real theta_ = Real(1);
    Real a_[3]  = {0, 0, 0};    ///< Reconstruction coefficients $a$.
    Real b_[4]  = {0, 0, 0, 0}; ///< Quadrature weights $b$.
    Real d_[3]  = {0, 0, 0};    ///< Reconstruction coefficients $d$.

    occa::memory o_Rnc2_; ///< R(u^{n+c_2}) scratch.
    occa::memory o_Rn1_;  ///< R(u^{n+1}) scratch.
};
