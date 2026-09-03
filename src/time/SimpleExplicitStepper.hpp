/// @file SimpleExplicitStepper.hpp
/// @brief Fixed-step explicit steppers: Forward Euler and SSPRK3.
///
/// SSPRK3 in the Shu-Osher form (SSP coefficient $c = 1$, three stages,
/// three RHS evaluations, 3x storage), see
/// docs/tech_docs/time_marching/explicit_time_marching.md Section 2.2:
/// $$
/// \begin{aligned}
///   u^{(1)} &= u^n + \Delta t \, \mathcal{R}(u^n) \\
///   u^{(2)} &= \tfrac{3}{4} u^n + \tfrac{1}{4}
///              \big(u^{(1)} + \Delta t \, \mathcal{R}(u^{(1)})\big) \\
///   u^{n+1} &= \tfrac{1}{3} u^n + \tfrac{2}{3}
///              \big(u^{(2)} + \Delta t \, \mathcal{R}(u^{(2)})\big)
/// \end{aligned}
/// $$
/// The expanded form maps onto the two documented kernels: the Euler
/// update builds each bracketed increment, the convex combination performs
/// the SSP averages. The positivity limiter (if installed) is applied to
/// every stage state including the output, matching the prototype.

#pragma once

#include "StepperBase.hpp"

/// @brief Forward Euler stepper (order 1).
class EulerStepper : public StepperBase<EulerStepper>
{
public:
    EulerStepper(occa::device &device, DeviceMemoryManager &mem,
                 RhsFunction rhs, occa::dim_t nDof,
                 const std::string &oklDir = OCCA_OKL_DIR);

    Real advanceImpl(occa::memory o_u, Real t, Real dt);

    static constexpr int orderImpl()
    {
        return 1;
    }
    static constexpr const char *nameImpl()
    {
        return "euler";
    }

private:
    occa::memory o_res_; ///< Stage residual buffer.
};

/// @brief Third-order SSP Runge-Kutta stepper (Shu-Osher form).
class SspRk3Stepper : public StepperBase<SspRk3Stepper>
{
public:
    SspRk3Stepper(occa::device &device, DeviceMemoryManager &mem,
                  RhsFunction rhs, occa::dim_t nDof,
                  const std::string &oklDir = OCCA_OKL_DIR);

    Real advanceImpl(occa::memory o_u, Real t, Real dt);

    static constexpr int orderImpl()
    {
        return 3;
    }
    static constexpr const char *nameImpl()
    {
        return "ssprk3";
    }

private:
    occa::memory o_res_; ///< Stage residual buffer ($k_i$).
    occa::memory o_u1_;  ///< First stage state $u^{(1)}$.
    occa::memory o_u2_;  ///< Second stage state $u^{(2)}$.
};
