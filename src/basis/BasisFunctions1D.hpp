/// @file BasisFunctions1D.hpp
/// @brief Pre-computed 1D basis functions using normalized Legendre
/// polynomials,
///        Gauss-Legendre quadrature, and Vandermonde matrices.
///
/// Reference: docs/tech_docs/basis_functions.md, Section 1.

#pragma once

#include <Eigen/Dense>
#include <occa.hpp>

#include "common/Types.hpp"

class DeviceMemoryManager;

/// @brief Pre-computed 1D basis functions (normalized Legendre polynomials),
///        quadrature data, and Vandermonde matrices.
///
/// All pre-computation happens at construction time. The class serves as
/// a data provider for BasisFunctions2D.
///
/// All 2D matrices use row-major storage to match OCCA's memory layout,
/// so that `wrapMemory` directly exposes data to OCCA kernels without
/// transposition.
class BasisFunctions1D
{
public:
    /// @brief Construct from polynomial order $N$ and number of quadrature
    /// points $N_q$.
    /// @param N  Maximum polynomial order (inclusive), $N \ge 0$.
    /// @param Nq Number of Gauss-Legendre quadrature points, $N_q \ge 1$.
    BasisFunctions1D(int N, int Nq);

    // ---- Accessors ----

    /// @brief Maximum polynomial order.
    int order() const noexcept
    {
        return N_;
    }

    /// @brief Number of Gauss-Legendre quadrature points.
    int numPoints() const noexcept
    {
        return Nq_;
    }

    /// @brief Gauss-Legendre quadrature nodes on $[-1, 1]$, size $N_q$.
    const VectorXr &points() const noexcept
    {
        return points_;
    }

    /// @brief Gauss-Legendre quadrature weights, size $N_q$.
    const VectorXr &weights() const noexcept
    {
        return weights_;
    }

    /// @brief Vandermonde matrix $V$ of size $N_q \times (N+1)$.
    ///        $V(i, j) = \tilde{P}_j(\xi_i)$, where $\xi_i$ is the $i$-th
    ///        quadrature point.
    const MatrixXr &
    vandermonde() const noexcept
    {
        return V_;
    }

    /// @brief Vandermonde derivative matrix $dV$ of size $N_q \times (N+1)$.
    ///        $dV(i, j) = \tilde{P}'_j(\xi_i)$, derivative w.r.t. the reference
    ///        coordinate.
    const MatrixXr &
    vandermondeDerivative() const noexcept
    {
        return dV_;
    }

    // ---- Static utilities (reusable, no instance needed) ----

    /// @brief Evaluate normalized Legendre polynomials $\tilde{P}_0 \dots
    /// \tilde{P}_N$
    ///        and their derivatives $\tilde{P}'_0 \dots \tilde{P}'_N$ at a
    ///        single point $x$.
    ///
    /// Uses the normalized three-term recurrence (Section 1.1 of
    /// basis_functions.md).
    ///
    /// $$
    ///   \tilde{P}_{n+1}(x) = \frac{\sqrt{(2n+1)(2n+3)} \cdot x \,
    ///   \tilde{P}_n(x)
    ///                         - n \sqrt{\frac{2n+3}{2n-1}} \cdot
    ///                         \tilde{P}_{n-1}(x)}{n+1}
    /// $$
    ///
    /// $$
    ///   \tilde{P}'_{n+1}(x) = \sqrt{(2n+1)(2n+3)} \cdot \tilde{P}_n(x)
    ///                         + \sqrt{\frac{2n+3}{2n-1}} \cdot
    ///                         \tilde{P}'_{n-1}(x)
    /// $$
    ///
    /// @param x   Evaluation point in $[-1, 1]$.
    /// @param N   Maximum polynomial order (inclusive), $N \ge 0$.
    /// @param[out] P   Values $\tilde{P}_n(x)$, size $N+1$.
    /// @param[out] dP  Derivatives $\tilde{P}'_n(x)$, size $N+1$.
    static void evalPolynomials(Real x, int N, VectorXr &P,
                                VectorXr &dP);

    /// @brief Compute $N_q$-point Gauss-Legendre quadrature via Newton
    /// iteration.
    ///
    /// Uses the classic (non-normalized) Legendre polynomials $P_{N_q}(x)$
    /// to find the roots. Initial guesses from the Tricomi asymptotic formula:
    ///
    /// @f[
    ///   x_i^{(0)} = \cos\left(\pi \frac{4i + 3}{4N_q + 2}\right)
    /// @f]
    ///
    /// Weights: $w_i = 2 / \bigl((1 - x_i^2) \cdot [P'_{N_q}(x_i)]^2\bigr)$.
    ///
    /// @param Nq           Number of quadrature points, $N_q \ge 1$.
    /// @param[out] points  Quadrature nodes on $[-1, 1]$, size $N_q$.
    /// @param[out] weights Corresponding weights, size $N_q$.
    static void computeGaussLegendre(int Nq, VectorXr &points,
                                     VectorXr &weights);

    // ---- OCCA device memory ----

    /// @brief Allocate device memory for all pre-computed data via
    ///        DeviceMemoryManager.
    ///
    /// After this call, the `o_*` accessors return valid `occa::memory` handles
    /// suitable for passing to OCCA kernels.
    ///
    /// @param mgr  Device memory manager for the active OCCA device.
    void allocateDeviceMemory(DeviceMemoryManager &mgr);

    /// @brief Device memory handle for quadrature nodes.
    occa::memory o_points() const
    {
        return o_points_;
    }

    /// @brief Device memory handle for quadrature weights.
    occa::memory o_weights() const
    {
        return o_weights_;
    }

    /// @brief Device memory handle for Vandermonde matrix $V$.
    occa::memory o_V() const
    {
        return o_V_;
    }

    /// @brief Device memory handle for Vandermonde derivative matrix $dV$.
    occa::memory o_dV() const
    {
        return o_dV_;
    }

private:
    int N_;  ///< Maximum polynomial order.
    int Nq_; ///< Number of quadrature points per dimension.

    VectorXr points_;  ///< Gauss-Legendre nodes.
    VectorXr weights_; ///< Gauss-Legendre weights.

    /// Vandermonde matrix, $N_q \times (N+1)$, row-major.
    MatrixXr V_;

    /// Vandermonde derivative matrix, $N_q \times (N+1)$, row-major.
    MatrixXr dV_;

    occa::memory o_points_;  ///< Device copy of quadrature nodes.
    occa::memory o_weights_; ///< Device copy of quadrature weights.
    occa::memory o_V_;       ///< Device copy of Vandermonde matrix.
    occa::memory o_dV_;      ///< Device copy of Vandermonde derivative matrix.

    void buildVandermonde();
    void buildVandermondeDerivative();
};
