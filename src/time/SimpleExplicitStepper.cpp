/// @file SimpleExplicitStepper.cpp
/// @brief Implementation of the fixed-step explicit steppers.

#include "SimpleExplicitStepper.hpp"

// ============================================================================
// EulerStepper
// ============================================================================

EulerStepper::EulerStepper(occa::device &device, DeviceMemoryManager &mem,
                           RhsFunction rhs, occa::dim_t nDof,
                           const std::string &oklDir)
    : StepperBase(device, mem, std::move(rhs), nDof, oklDir)
{
    ensureKernels();
    o_res_ = mem_.wrapOrMalloc(nDof_);
}

Real EulerStepper::advance(occa::memory o_u, Real /*t*/, Real dt)
{
    // u^{n+1} = u^n + dt * R(u^n); the per-DOF kernels make the in-place
    // update (output aliasing the input) safe.
    rhs_(o_u, o_res_);
    eulerUpdate(o_u, dt, o_res_, o_u);
    applyLimiter(o_u);
    return dt;
}

// ============================================================================
// SspRk3Stepper
// ============================================================================

SspRk3Stepper::SspRk3Stepper(occa::device &device, DeviceMemoryManager &mem,
                             RhsFunction rhs, occa::dim_t nDof,
                             const std::string &oklDir)
    : StepperBase(device, mem, std::move(rhs), nDof, oklDir)
{
    ensureKernels();
    o_res_ = mem_.wrapOrMalloc(nDof_);
    o_u1_  = mem_.wrapOrMalloc(nDof_);
    o_u2_  = mem_.wrapOrMalloc(nDof_);
}

Real SspRk3Stepper::advance(occa::memory o_u, Real /*t*/, Real dt)
{
    // Stage 1: u1 = u^n + dt * R(u^n).
    rhs_(o_u, o_res_);
    eulerUpdate(o_u, dt, o_res_, o_u1_);
    applyLimiter(o_u1_);

    // Stage 2: u2 = 3/4 u^n + 1/4 (u1 + dt * R(u1)). The bracketed
    // increment overwrites u1 (per-DOF aliasing is safe).
    rhs_(o_u1_, o_res_);
    eulerUpdate(o_u1_, dt, o_res_, o_u1_);
    convexCombine(Real(3) / Real(4), Real(1) / Real(4), o_u, o_u1_, o_u2_);
    applyLimiter(o_u2_);

    // Stage 3: u^{n+1} = 1/3 u^n + 2/3 (u2 + dt * R(u2)), written in place
    // over u^n (the convex combination reads both inputs per DOF before
    // writing).
    rhs_(o_u2_, o_res_);
    eulerUpdate(o_u2_, dt, o_res_, o_u2_);
    convexCombine(Real(1) / Real(3), Real(2) / Real(3), o_u, o_u2_, o_u);
    applyLimiter(o_u);
    return dt;
}
