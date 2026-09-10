/// @file Riemann.cpp
/// @brief Host-side reference implementation of the Riemann solver module.
///
/// These functions mirror the OKL kernels in okl/llf.okl; tests compare
/// the device output against this host baseline.

#include "Riemann.hpp"

#include <cmath>

EulerState primitiveFromConserved(const Real q[4], Real gamma) {
    const Real rho = q[0];
    const Real u   = q[1] / rho;
    const Real v   = q[2] / rho;
    const Real ke  = Real(0.5) * rho * (u * u + v * v);
    const Real p   = (gamma - Real(1)) * (q[3] - ke);
    const Real a   = std::sqrt(gamma * p / rho);
    return {rho, u, v, p, a};
}

void computePhysicalFlux(const Real q[4], Real gamma, Real f[4]) {
    const EulerState st = primitiveFromConserved(q, gamma);
    const Real rho      = st.rho;
    const Real u        = st.u;
    const Real v        = st.v;
    const Real p        = st.p;

    f[0] = rho * u;
    f[1] = rho * u * u + p;
    f[2] = rho * u * v;
    f[3] = u * (q[3] + p);
}

void rotateState(const Real q[4], Real n_unit_x, Real n_unit_y, Real q_rot[4]) {
    const Real rho = q[0];
    const Real u   = q[1] / rho;
    const Real v   = q[2] / rho;

    const Real U = n_unit_x * u + n_unit_y * v;  // normal velocity
    const Real V = -n_unit_y * u + n_unit_x * v; // tangential velocity

    q_rot[0] = rho;
    q_rot[1] = rho * U;
    q_rot[2] = rho * V;
    q_rot[3] = q[3];
}

void rotateFluxBack(const Real flux_rot[4], Real n_unit_x, Real n_unit_y,
                    Real flux[4]) {
    const Real frhoU = flux_rot[1];
    const Real frhoV = flux_rot[2];

    flux[0] = flux_rot[0];                         // rho
    flux[1] = n_unit_x * frhoU - n_unit_y * frhoV; // rho u
    flux[2] = n_unit_y * frhoU + n_unit_x * frhoV; // rho v
    flux[3] = flux_rot[3];                         // E
}

void computeLLFFlux(const Real qL[4], const Real qR[4], Real gamma,
                    Real flux[4]) {
    // Physical fluxes of both sides.
    Real fL[4], fR[4];
    computePhysicalFlux(qL, gamma, fL);
    computePhysicalFlux(qR, gamma, fR);

    // Local maximum wave speed alpha = max(|U_L|+a_L, |U_R|+a_R).
    const EulerState stL = primitiveFromConserved(qL, gamma);
    const EulerState stR = primitiveFromConserved(qR, gamma);
    const Real alpha =
        std::max(std::abs(stL.u) + stL.a, std::abs(stR.u) + stR.a);

    for (int k = 0; k < kNumVars2D; ++k) {
        flux[k] =
            Real(0.5) * (fL[k] + fR[k]) - Real(0.5) * alpha * (qR[k] - qL[k]);
    }
}

void computeFaceFlux(const Real qL[4], const Real qR[4], Real n_unit_x,
                     Real n_unit_y, Real gamma, Real flux[4]) {
    // Rotate both states into the local normal-tangent frame.
    Real qL_rot[4], qR_rot[4];
    rotateState(qL, n_unit_x, n_unit_y, qL_rot);
    rotateState(qR, n_unit_x, n_unit_y, qR_rot);

    // 1-D Riemann problem in the local frame.
    Real flux_rot[4];
    computeLLFFlux(qL_rot, qR_rot, gamma, flux_rot);

    // Rotate the flux back to physical space.
    rotateFluxBack(flux_rot, n_unit_x, n_unit_y, flux);
}
