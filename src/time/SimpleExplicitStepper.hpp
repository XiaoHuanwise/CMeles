/// @file SimpleExplicitStepper.hpp
/// @brief Fixed-step explicit steppers: Forward Euler and SSPRK3.
///
/// SSPRK3 in the Shu-Osher form (SSP coefficient $c = 1$, three stages,
/// three RHS evaluations, 3x storage), see
/// docs/tech_docs/time_marching/explicit_time_marching.md Section 2.2:
///
/// $$ \begin{aligned} u^{(1)} &= u^n + \Delta t \, \mathcal{R}(u^n) \\ u^{(2)} &= \tfrac{3}{4} u^n + \tfrac{1}{4} \big(u^{(1)} + \Delta t \, \mathcal{R}(u^{(1)})\big) \\ u^{n+1} &= \tfrac{1}{3} u^n + \tfrac{2}{3} \big(u^{(2)} + \Delta t \, \mathcal{R}(u^{(2)})\big) \end{aligned} $$
///
/// The expanded form maps onto the two documented kernels: the Euler
/// update builds each bracketed increment, the convex combination performs
/// the SSP averages. The positivity limiter (if installed) is applied to
/// every stage state including the output, matching the prototype.

#pragma once

#include "StepperBase.hpp"

/// @brief Forward Euler stepper (order 1).
class EulerStepper : public StepperBase {
public:
    EulerStepper(occa::device &device, DeviceMemoryManager &mem,
                 RhsFunction rhs, occa::dim_t nDof,
                 const std::string &oklDir = OCCA_OKL_DIR);

    Real advance(occa::memory o_u, Real t, Real dt) override;

    int order() const override {
        return 1;
    }
    const char *name() const override {
        return "euler";
    }

private:
    occa::memory o_res_; ///< Stage residual buffer.
};

/// @brief Third-order SSP Runge-Kutta stepper (Shu-Osher form).
class SspRk3Stepper : public StepperBase {
public:
    SspRk3Stepper(occa::device &device, DeviceMemoryManager &mem,
                  RhsFunction rhs, occa::dim_t nDof,
                  const std::string &oklDir = OCCA_OKL_DIR);

    Real advance(occa::memory o_u, Real t, Real dt) override;

    int order() const override {
        return 3;
    }
    const char *name() const override {
        return "ssprk3";
    }

private:
    /// @brief u_new = alpha * u_n + beta * u_temp (per-DOF aliasing safe);
    ///        the kernel is built lazily on first use.
    void convexCombine(Real alpha, Real beta, occa::memory &o_un,
                       occa::memory &o_utemp, occa::memory &o_unew) {
        if (!sspConvexCombine_.isInitialized()) {
            sspConvexCombine_ = buildTimeKernel("sspConvexCombine");
        }
        sspConvexCombine_(static_cast<int>(nDof_), alpha, beta, o_un, o_utemp,
                          o_unew);
    }

    occa::memory o_res_;            ///< Stage residual buffer ($k_i$).
    occa::memory o_u1_;             ///< First stage state $u^{(1)}$.
    occa::memory o_u2_;             ///< Second stage state $u^{(2)}$.
    occa::kernel sspConvexCombine_; ///< SSP averaging kernel (lazy).
};
