/// @file ImplicitResidual.cpp
/// @brief Implementation of the Backward Euler and DITR temporal residuals.

#include "ImplicitResidual.hpp"

#include "common/KernelProps.hpp"

#include <stdexcept>

// ============================================================================
// TemporalResidual
// ============================================================================

TemporalResidual::TemporalResidual(occa::device &device,
                                   DeviceMemoryManager &mem, RhsFunction rhs,
                                   occa::dim_t nDof, const std::string &oklDir)
    : device_(device), mem_(mem), rhs_(std::move(rhs)), nDof_(nDof),
      oklDir_(oklDir), blas_(device, mem, oklDir)
{
    occa::json props;
#ifdef USE_FLOAT_PRECISION
    props["defines/Real"] = "float";
#else
    props["defines/Real"] = "double";
#endif
    props["defines/TILE_SIZE"] = 256;
    cmeles::finaliseKernelProps(props, device_);
    vecCombine4_ =
        device_.buildKernel(oklDir_ + "/time_update.okl", "vecCombine4", props);
}

void TemporalResidual::combine4(Real c0, occa::memory &o_x0, Real c1,
                                occa::memory &o_x1, Real c2, occa::memory &o_x2,
                                Real c3, occa::memory &o_x3,
                                occa::memory &o_out)
{
    vecCombine4_(static_cast<int>(nDof_), c0, o_x0, c1, o_x1, c2, o_x2, c3,
                 o_x3, o_out);
}

// ============================================================================
// BackwardEulerResidual
// ============================================================================

BackwardEulerResidual::BackwardEulerResidual(occa::device &device,
                                             DeviceMemoryManager &mem,
                                             RhsFunction rhs, occa::dim_t nDof,
                                             const std::string &oklDir)
    : TemporalResidual(device, mem, std::move(rhs), nDof, oklDir)
{
    o_R_ = mem_.wrapOrMalloc(nDof_);
}

void BackwardEulerResidual::temporalResidual(occa::memory &o_uNew,
                                             occa::memory &o_u,
                                             occa::memory & /*o_Rn*/,
                                             occa::memory & /*o_uPrev*/,
                                             Real dt, occa::memory &o_F)
{
    // F = (u_n - u_new)/dt + R(u_new).
    rhs_(o_uNew, o_R_);
    combine4(Real(1) / dt, o_u, -Real(1) / dt, o_uNew, Real(1), o_R_, Real(0),
             o_R_, o_F);
}

// ============================================================================
// DitrResidual
// ============================================================================

DitrResidual::DitrResidual(occa::device &device, DeviceMemoryManager &mem,
                           RhsFunction rhs, occa::dim_t nDof, Variant variant,
                           Real c2, Real beta, const std::string &oklDir)
    : TemporalResidual(device, mem, std::move(rhs), nDof, oklDir),
      variant_(variant), c2_(c2), beta_(beta)
{
    // Quadrature weights b = [0, b1, b2, b3].
    b_[0] = Real(0);
    b_[1] = Real(0.5) - Real(1) / (Real(6) * c2);
    b_[2] = Real(1) / (Real(6) * c2 * (Real(1) - c2));
    b_[3] = Real(0.5) - Real(1) / (Real(6) * (Real(1) - c2));
    rebuildCoefficients();

    o_Rnc2_ = mem_.wrapOrMalloc(nDof_);
    o_Rn1_  = mem_.wrapOrMalloc(nDof_);
}

void DitrResidual::rebuildCoefficients()
{
    const Real c2 = c2_, c22 = c2 * c2, c23 = c22 * c2;
    switch (variant_)
    {
        case Variant::U2R2:
            a_[0] = Real(0);
            a_[1] = Real(1) - (Real(3) * c22 - Real(2) * c23);
            a_[2] = Real(3) * c22 - Real(2) * c23;
            d_[0] = Real(0);
            d_[1] = c2 - Real(2) * c22 + c23;
            d_[2] = -c22 + c23;
            break;
        case Variant::U2R1:
            a_[0] = Real(0);
            a_[1] = Real(1) - (Real(2) * c2 - c22);
            a_[2] = Real(2) * c2 - c22;
            d_[0] = Real(0);
            d_[1] = Real(0);
            d_[2] = c22 - c2;
            break;
        case Variant::U3R1:
        {
            const Real th = theta_, th2 = th * th;
            const Real tp1 = th + Real(1), tp12 = tp1 * tp1;
            const Real c2m1 = c2 - Real(1), c2m12 = c2m1 * c2m1;
            a_[0] = -(c2 * c2m12) / (th * tp12);
            a_[1] = ((th + c2) * c2m12) / th;
            a_[2] = c2 *
                    (-th2 * c2 + Real(2) * th2 - th * c22 + Real(3) * th -
                     Real(2) * c22 + Real(3) * c2) /
                    tp12;
            d_[0] = Real(0);
            d_[1] = Real(0);
            d_[2] = (c2 * (th + c2) * c2m1) / tp1;
            break;
        }
    }
}

void DitrResidual::setTheta(Real theta)
{
    theta_ = theta;
    rebuildCoefficients();
}

void DitrResidual::temporalResidual(occa::memory &o_uNew2N, occa::memory &o_u,
                                    occa::memory &o_Rn, occa::memory &o_uPrev,
                                    Real dt, occa::memory &o_F2N)
{
    const auto nDof     = nDof_;
    occa::memory o_uNc2 = o_uNew2N.slice(0, nDof);
    occa::memory o_uN1  = o_uNew2N.slice(nDof, nDof);
    occa::memory o_F0   = o_F2N.slice(0, nDof);
    occa::memory o_F1   = o_F2N.slice(nDof, nDof);

    rhs_(o_uNc2, o_Rnc2_);
    rhs_(o_uN1, o_Rn1_);

    // F0 = (a0 u_prev + a1 u_n + a2 u_n1 - u_nc2)/dt + d1 R_n + d2 R_n1.
    combine4(a_[1] / dt, o_u, a_[2] / dt, o_uN1, -Real(1) / dt, o_uNc2, d_[1],
             o_Rn, o_F0);
    if (a_[0] != Real(0))
    {
        blas_.axpy(nDof_, a_[0] / dt, o_uPrev, o_F0);
    }
    blas_.axpy(nDof_, d_[2], o_Rn1_, o_F0);

    // F1 = (u_n - u_n1)/dt + b1 R_n + b2 R_nc2 + b3 R_n1.
    combine4(Real(1) / dt, o_u, -Real(1) / dt, o_uN1, b_[2], o_Rnc2_, b_[3],
             o_Rn1_, o_F1);
    blas_.axpy(nDof_, b_[1], o_Rn, o_F1);

    // Preconditioner: F0 += beta * F1.
    blas_.axpy(nDof_, beta_, o_F1, o_F0);
}

void DitrResidual::temporalResidualStageC2(
    occa::memory &o_uNc2, occa::memory &o_uN1, occa::memory &o_Rn1,
    occa::memory &o_u, occa::memory &o_Rn, occa::memory &o_uPrev, Real dt,
    occa::memory &o_F0)
{
    rhs_(o_uNc2, o_Rnc2_);

    // F0 = ((a1 + beta) u_n + (a2 - beta) u_n1 - u_nc2)/dt
    //      + (d1 + beta b1) R_n + beta b2 R_nc2 + (d2 + beta b3) R_n1,
    // plus the a0 u_prev / dt term of U3R1.
    combine4((a_[1] + beta_) / dt, o_u, (a_[2] - beta_) / dt, o_uN1,
             -Real(1) / dt, o_uNc2, (d_[1] + beta_ * b_[1]), o_Rn, o_F0);
    if (a_[0] != Real(0))
    {
        blas_.axpy(nDof_, a_[0] / dt, o_uPrev, o_F0);
    }
    blas_.axpy(nDof_, beta_ * b_[2], o_Rnc2_, o_F0);
    blas_.axpy(nDof_, (d_[2] + beta_ * b_[3]), o_Rn1, o_F0);
}

void DitrResidual::temporalResidualStageN1(occa::memory &o_uN1,
                                           occa::memory &o_Rnc2,
                                           occa::memory &o_u,
                                           occa::memory &o_Rn, Real dt,
                                           occa::memory &o_F1)
{
    rhs_(o_uN1, o_Rn1_);

    // F1 = (u_n - u_n1)/dt + b1 R_n + b2 R_nc2 + b3 R_n1.
    combine4(Real(1) / dt, o_u, -Real(1) / dt, o_uN1, b_[2], o_Rnc2, b_[3],
             o_Rn1_, o_F1);
    blas_.axpy(nDof_, b_[1], o_Rn, o_F1);
}
