/// @file Riemann.hpp
/// @brief Numerical flux (Riemann solver) family for the inviscid Euler
///        equations, plus the face-local rotation used by all fluxes.
///
/// Reference: docs/tech_docs/riemann_solver.md.
///
/// This header provides the host-side reference implementation. The same
/// formulas are reproduced in the OCCA kernel `okl/llf.okl`, which is
/// included by the DG surface-integral kernel at JIT-compile time and
/// selected by the `FLUX_TYPE` JIT define (see DgField::buildKernel). The
/// host implementation doubles as the correctness baseline for tests.
///
/// Conventions (2D, 4 conserved variables):
///   q = (rho, rho u, rho v, E)
///   f_c(q) = (rho u, rho u^2 + p, rho u v, u (E + p))
///
/// The face rotation is independent of any particular flux: every Riemann
/// solver (LLF today, Roe/FVS later) works in the local normal-tangent
/// frame, so rotateState / rotateFluxBack are shared helpers, not part of
/// the LLF.

#pragma once

#include <array>

#include "common/Types.hpp"

/// @brief Number of conserved variables for 2D Euler.
inline constexpr int kNumVars2D = 4;

/// @brief Primitive variables extracted from a conserved state.
struct EulerState
{
    Real rho; ///< Density.
    Real u;   ///< x-velocity (or normal velocity in local frame).
    Real v;   ///< y-velocity (or tangential velocity in local frame).
    Real p;   ///< Pressure.
    Real a;   ///< Sound speed $a = \sqrt{\gamma p / \rho}$.
};

/// @brief Extract primitive variables from a conserved state.
///
/// Uses the ideal-gas relation $E = p/(\gamma-1) + \frac12 \rho (u^2+v^2)$.
///
/// @param q      Conserved state, size 4.
/// @param gamma  Ratio of specific heats.
EulerState primitiveFromConserved(const Real q[4], Real gamma);

/// @brief Compute the physical (convective) flux of a single state.
///
/// @param q      Conserved state, size 4.
/// @param gamma  Ratio of specific heats.
/// @param[out] f Physical flux, size 4.
void computePhysicalFlux(const Real q[4], Real gamma, Real f[4]);

/// @brief Rotate a conserved state into the local normal-tangent frame.
///
/// The unit normal $\mathbf{n}$ and unit tangent $\mathbf{t} =
/// (-n_y, n_x)$ define an orthonormal basis. Velocities are projected onto
/// it:
///
/// $$
///   U = u n_x + v n_y, \qquad V = -u n_y + v n_x,
/// $$
///
/// so that $\tilde q = (\rho, \rho U, \rho V, E)$ is the state in the frame
/// where the face normal is the local $x$ axis (the 1-D Riemann problem).
/// See docs/tech_docs/mesh_and_geometry.md, Section 1.3.
///
/// @param q            Conserved state, size 4.
/// @param n_unit_x     Unit normal x-component (normalised).
/// @param n_unit_y     Unit normal y-component (normalised).
/// @param[out] q_rot   Rotated state, size 4.
void rotateState(const Real q[4], Real n_unit_x, Real n_unit_y, Real q_rot[4]);

/// @brief Rotate a face flux back from the local frame to physical space.
///
/// Mass and energy components are scalars invariant under rotation; the
/// momentum components are rotated back with $\mathbf{R}^T$:
///
/// $$
///   f_{\rho u} = n_x \tilde f_{\rho U} - n_y \tilde f_{\rho V},
///   \qquad f_{\rho v} = n_y \tilde f_{\rho U} + n_x \tilde f_{\rho V}.
/// $$
///
/// @param flux_rot     Flux in the local frame, size 4.
/// @param n_unit_x     Unit normal x-component (normalised).
/// @param n_unit_y     Unit normal y-component (normalised).
/// @param[out] flux    Flux in physical space, size 4.
void rotateFluxBack(const Real flux_rot[4], Real n_unit_x, Real n_unit_y,
                    Real flux[4]);

/// @brief Local Lax-Friedrichs (Rusanov) numerical flux in the *local*
///        normal-tangent frame.
///
/// $$
///   \hat F^{LLF}(q_L, q_R) = \frac12 (f_L + f_R)
///     - \frac12 \alpha (q_R - q_L), \quad
///   \alpha = \max(|U_L| + a_L, |U_R| + a_R)
/// $$
///
/// where $U$ is the normal velocity in the local frame and $a$ the sound
/// speed. Both input states must already be rotated into the local frame.
///
/// @param qL     Left conserved state (local frame), size 4.
/// @param qR     Right conserved state (local frame), size 4.
/// @param gamma  Ratio of specific heats.
/// @param[out] flux  Numerical flux in the local frame, size 4.
void computeLLFFlux(const Real qL[4], const Real qR[4], Real gamma,
                    Real flux[4]);

/// @brief Face numerical flux in *physical* space: rotate both states to
///        the local frame, apply the flux, rotate the result back.
///
/// This is the entry point used by the DG surface-integral kernel
/// (`computeFaceFlux` in surface_integral.okl), which feeds it the
/// physical-state values on each face quadrature point.
///
/// @param qL     Left conserved state (physical space), size 4.
/// @param qR     Right conserved state (physical space), size 4.
/// @param n_unit_x  Unit normal x-component.
/// @param n_unit_y  Unit normal y-component.
/// @param gamma  Ratio of specific heats.
/// @param[out] flux  Numerical flux in physical space, size 4.
void computeFaceFlux(const Real qL[4], const Real qR[4], Real n_unit_x,
                     Real n_unit_y, Real gamma, Real flux[4]);
