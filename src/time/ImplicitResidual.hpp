/// @file ImplicitResidual.hpp
/// @brief Implicit temporal-residual builders: Backward Euler and DITR.
///
/// These classes do not advance the state — they build the temporal
/// residual $\mathcal{F} = \mathcal{G}(u^{n+1})$ whose root is the implicit
/// step; a DualStepper drives them to convergence through pseudo time.
///
/// Backward Euler:
/// $$ \mathcal{F} = \frac{u^n - u^{n+1}}{\Delta t} + \mathcal{R}(u^{n+1}) $$
///
/// DITR (deferred-implicit time reconstruction, see
/// docs/tech_docs/time_marching/implicit_time_marching.md) reconstructs the
/// stage values $u^{n+c_2}$ and $u^{n+1}$ with the reconstruction
/// coefficients $a$ and $d$:
/// $$
/// \begin{aligned}
///   F_0 &= \frac{a_0 u^{n-1} + a_1 u^n + a_2 u^{n+1} - u^{n+c_2}}
///              {\Delta t} + d_1 R^n + d_2 R^{n+1} \\
///   F_1 &= \frac{u^n - u^{n+1}}{\Delta t}
///          + b_1 R^n + b_2 R^{n+c_2} + b_3 R^{n+1}
/// \end{aligned}
/// $$
/// with the quadrature weights $b = [0, \frac12 - \frac{1}{6 c_2},
/// \frac{1}{6 c_2 (1 - c_2)}, \frac12 - \frac{1}{6 (1 - c_2)}]$ and the
/// coupling preconditioner $F_0 \mathrel{+}= \beta F_1$ ($\mathbf{P} =
/// \begin{bmatrix} I & \beta I \\ 0 & I \end{bmatrix}$, $\beta = 1$).
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

#include <string>

#include "blas/Blas.hpp"
#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "time/TimeTypes.hpp"

/// @brief Shared context of the temporal-residual builders (composition,
///        no polymorphism needed — the DualStepper knows the static type).
class TemporalResidualBase
{
protected:
    TemporalResidualBase(occa::device &device, DeviceMemoryManager &mem,
                         RhsFunction rhs, occa::dim_t nDof,
                         const std::string &oklDir);

    /// @brief out = c0*x0 + c1*x1 + c2*x2 + c3*x3 over nDof entries.
    void combine4(Real c0, occa::memory &o_x0, Real c1, occa::memory &o_x1,
                  Real c2, occa::memory &o_x2, Real c3, occa::memory &o_x3,
                  occa::memory &o_out);

    occa::device &device_;
    DeviceMemoryManager &mem_;
    RhsFunction rhs_;
    occa::dim_t nDof_;
    std::string oklDir_;
    Blas blas_;

private:
    occa::kernel vecCombine4_;
};

/// @brief Backward Euler temporal residual.
class BackwardEulerStepper : public TemporalResidualBase
{
public:
    BackwardEulerStepper(occa::device &device, DeviceMemoryManager &mem,
                         RhsFunction rhs, occa::dim_t nDof,
                         const std::string &oklDir = OCCA_OKL_DIR);

    /// @brief $\mathcal{F} = (u^n - u^{n+1})/\Delta t + R(u^{n+1})$.
    void temporalResidual(occa::memory &o_uNew, occa::memory &o_u, Real dt,
                          occa::memory &o_F);

    /// @brief Number of stacked implicit stages (size of the pseudo state).
    static constexpr int nStages()
    {
        return 1;
    }
    /// @brief Whether $u^{n-1}$ is required (U3R1 only).
    static constexpr bool needsPrev()
    {
        return false;
    }

private:
    occa::memory o_R_; ///< R(u^{n+1}) scratch.
};

/// @brief DITR temporal residual (U2R2 / U2R1 / U3R1).
class DitrStepper : public TemporalResidualBase
{
public:
    /// @brief DITR reconstruction variant.
    enum class Variant
    {
        U2R2 = 0, ///< 2nd-order reconstruction, 2nd-order quadrature.
        U2R1 = 1, ///< 2nd-order reconstruction, 1st-order quadrature.
        U3R1 = 2  ///< 3rd-order reconstruction, 1st-order quadrature.
    };

    DitrStepper(occa::device &device, DeviceMemoryManager &mem, RhsFunction rhs,
                occa::dim_t nDof, Variant variant, Real c2 = Real(0.5),
                Real beta = Real(1), const std::string &oklDir = OCCA_OKL_DIR);

    /// @brief Set $\theta = \Delta t^{n-1} / \Delta t^n$ (U3R1 only;
    ///        rebuilds the reconstruction coefficients).
    void setTheta(Real theta);

    // ---- Coupled residuals (stacked 2*N_dof states) ----

    /// @brief Both stage residuals of the stacked state
    ///        $u_{new} = [u^{n+c_2}, u^{n+1}]$, with the preconditioner
    ///        applied: $F = [F_0 + \beta F_1, F_1]$.
    void temporalResidual(occa::memory &o_uNew2N, occa::memory &o_u,
                          occa::memory &o_Rn, occa::memory &o_uPrev, Real dt,
                          occa::memory &o_F2N);

    // ---- Decoupled per-stage residuals ----

    /// @brief Stage $n + c_2$ residual with the preconditioner folded in;
    ///        \p o_uN1 / \p o_Rn1 are the *current* values of the $n+1$
    ///        stage.
    void temporalResidualStageC2(occa::memory &o_uNc2, occa::memory &o_uN1,
                                 occa::memory &o_Rn1, occa::memory &o_u,
                                 occa::memory &o_Rn, occa::memory &o_uPrev,
                                 Real dt, occa::memory &o_F0);

    /// @brief Stage $n+1$ residual; \p o_Rnc2 is the *current* residual of
    ///        the $n+c_2$ stage.
    void temporalResidualStageN1(occa::memory &o_uN1, occa::memory &o_Rnc2,
                                 occa::memory &o_u, occa::memory &o_Rn, Real dt,
                                 occa::memory &o_F1);

    static constexpr int nStages()
    {
        return 2;
    }
    constexpr bool needsPrev() const
    {
        return variant_ == Variant::U3R1;
    }
    Variant variant() const
    {
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
