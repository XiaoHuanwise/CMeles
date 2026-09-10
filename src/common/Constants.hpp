/// @file Constants.hpp
/// @brief System-level constants shared across CMeles modules.
///
/// Governing-equation and cross-module constants live here so that every
/// module sees a single definition (module-private tuning constants stay
/// in their owning headers). The OKL kernels receive them as JIT defines
/// (`N_VARS` / `NQ_MAX` / `NQ2_MAX` / `NMODES_MAX`) through the kernel
/// properties, because thread-local arrays in .okl sources must be sized
/// by compile-time constants; see docs/tech_docs/occa.md for the capacity
/// table and the silent-overflow caveat.

#pragma once

/// @brief Number of conserved variables of the 2D Euler system
///        $[\rho, \rho u, \rho v, E]$.
inline constexpr int kNumVars2D = 4;

/// @brief Max quadrature points per direction / per face (cap of $N_q$).
///        6 covers the $N + 2$ anti-aliasing margin for the max order
///        $N = 4$; $N + 1$ already integrates the linear (mass/projection)
///        terms exactly (Gauss-Legendre: $n$ points are exact to degree
///        $2n - 1$).
inline constexpr int kMaxQuadPts1D = 6;

/// @brief Max tensor-product quadrature points per element (cap of
///        $N_q^2$).
inline constexpr int kMaxQuadPts2D = kMaxQuadPts1D * kMaxQuadPts1D;

/// @brief Max number of modal coefficients (cap of $N_{\text{modes}}$).
///        $N_{\text{modes}} = (N+1)(N+2)/2 \le 16$ covers polynomial
///        orders up to $N = 4$ (15 modes).
inline constexpr int kMaxModes = 16;
