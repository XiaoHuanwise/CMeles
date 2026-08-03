/// @file BasisFunctions2D.hpp
/// @brief Pre-computed 2D basis functions on the tensor-product reference
///        element $[-1,1]^2$, using Pascal-triangle (full polynomial $P_N$)
///        ordering.
///
/// Reference: docs/tech_docs/basis_functions.md, Section 2.

#pragma once

#include <Eigen/Dense>
#include <occa.hpp>
#include <utility>

#include "common/Types.hpp"

class BasisFunctions1D;
class DeviceMemoryManager;

/// @brief Pre-computed 2D basis functions on the tensor-product reference
///        element $[-1,1]^2$, using the Pascal-triangle (full polynomial $P_N$)
///        ordering.
///
/// 2D basis functions are tensor products of 1D normalized Legendre
/// polynomials:
///
/// $$
///   \phi_{ij}(r, s) = \tilde{P}_i(r) \cdot \tilde{P}_j(s), \quad i + j \le N.
/// $$
///
/// They are stored in Pascal-triangle (row) order: functions with the same
/// total degree $p = i + j$ are grouped together. The linear index $l$ is given
/// by
///
/// $$
///   l = \frac{p(p+1)}{2} + j, \quad p = i + j.
/// $$
///
/// All pre-computation happens at construction time. The class holds a
/// non-owning const reference to a BasisFunctions1D instance whose lifetime
/// must exceed its own.
///
/// All 2D matrices use row-major storage to match OCCA's memory layout,
/// so that `wrapMemory` directly exposes data to OCCA kernels without
/// transposition.
class BasisFunctions2D
{
public:
    /// @brief Construct from 1D basis functions and polynomial order $N$.
    /// @param basis1D  1D basis (must be configured with order $\ge N$).
    /// @param N        2D polynomial order ($i+j \le N$), $N \ge 0$.
    ///
    /// @pre basis1D.order() >= N
    /// @pre basis1D.numPoints() >= 1
    BasisFunctions2D(const BasisFunctions1D &basis1D, int N);

    // ---- Accessors ----

    /// @brief 2D polynomial order $N$ (total degree $i+j \le N$).
    int order() const noexcept
    {
        return N_;
    }

    /// @brief Number of basis functions $N_{\text{base}} = (N+1)(N+2)/2$.
    int numBases() const noexcept
    {
        return N_base_;
    }

    /// @brief Total number of 2D quadrature points $N_q^2$.
    int numPoints() const noexcept
    {
        return numPoints1D_ * numPoints1D_;
    }

    /// @brief Number of quadrature points per dimension $N_q$.
    int numPoints1D() const noexcept
    {
        return numPoints1D_;
    }

    /// @brief Access to the underlying 1D basis.
    const BasisFunctions1D &basis1D() const noexcept
    {
        return *basis1D_;
    }

    // ---- Pascal-triangle index mapping ----

    /// @brief Forward map: $(i, j) \to l$.
    ///
    /// Computes $l = p(p+1)/2 + j$, where $p = i + j$.
    ///
    /// @param i  $r$-direction polynomial order.
    /// @param j  $s$-direction polynomial order.
    /// @return   Linear index $l \in [0, N_{\text{base}})$.
    ///
    /// @pre  $i + j \le N$.
    int linearIndex(int i, int j) const;

    /// @brief Inverse map: $l \to (i_l, j_l)$.
    ///
    /// Computes
    /// $p = \lfloor (\sqrt{8l + 1} - 1) / 2 \rfloor$,
    /// $j = l - p(p+1)/2$,
    /// $i = p - j$.
    ///
    /// @param l  Linear index in $[0, N_{\text{base}})$.
    /// @return   Pair $(i_l, j_l)$.
    std::pair<int, int> indexPair(int l) const;

    /// @brief Pre-computed $(i_l, j_l)$ pairs for all bases, shape
    /// $N_{\text{base}} \times 2$.
    const Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> &indexMap()
        const noexcept
    {
        return ij_map_;
    }

    // ---- Quadrature data ----

    /// @brief 2D tensor-product quadrature points, shape $N_q^2 \times 2$.
    ///
    /// Stored in column-major order: $k = j \cdot N_q + i$, where $i$ indexes
    /// the $r$-direction column and $j$ indexes the $s$-direction row.
    /// Row $k$ stores the point $(r_i, s_j)$.
    const MatrixX2r &quadraturePoints() const noexcept
    {
        return quad_points_;
    }

    /// @brief 2D tensor-product quadrature weights, size $N_q^2$.
    ///
    /// $W_k = w_i \cdot w_j$, where $k = j \cdot N_q + i$.
    const VectorXr &quadratureWeights() const noexcept
    {
        return quad_weights_;
    }

    // ---- Vandermonde matrices ----

    /// @brief 2D Vandermonde matrix, size $N_q^2 \times N_{\text{base}}$.
    ///
    /// $V_{2D}(k, l) = \tilde{P}_{i_l}(r_i) \cdot \tilde{P}_{j_l}(s_j)$,
    /// where $k = j \cdot N_q + i$ (column-major grid order).
    const MatrixXr &vandermonde() const noexcept
    {
        return V2D_;
    }

    /// @brief $r$-direction derivative Vandermonde, size $N_q^2 \times
    /// N_{\text{base}}$.
    ///
    /// $dV_{2D,r}(k, l) = \tilde{P}'_{i_l}(r_i) \cdot \tilde{P}_{j_l}(s_j)$.
    const MatrixXr &vandermondeDerivativeR() const noexcept
    {
        return dV2D_r_;
    }

    /// @brief $s$-direction derivative Vandermonde, size $N_q^2 \times
    /// N_{\text{base}}$.
    ///
    /// $dV_{2D,s}(k, l) = \tilde{P}_{i_l}(r_i) \cdot \tilde{P}'_{j_l}(s_j)$.
    const MatrixXr &vandermondeDerivativeS() const noexcept
    {
        return dV2D_s_;
    }

    // ---- Modal-nodal transforms ----

    /// @brief Modal to nodal: $u = V_{2D} \cdot \hat{u}$.
    ///
    /// Converts modal coefficients to nodal values at quadrature points.
    ///
    /// @param u_hat  Modal coefficients, size $N_{\text{base}}$.
    /// @return       Nodal values at quadrature points, size $N_q^2$.
    VectorXr modalToNodal(const VectorXr &u_hat) const;

    /// @brief Nodal to modal ($L^2$ projection on reference element):
    ///        $\hat{u} = V_{2D}^T \, W \, u$.
    ///
    /// Uses the orthonormal property of the normalized Legendre basis:
    /// the mass matrix $M = I$ on the reference element $[-1,1]^2$, so
    /// modal coefficients are obtained directly via the discrete $L^2$
    /// projection.
    ///
    /// @note For physical elements with non-constant Jacobian, the per-element
    ///       mass matrix $$M = V_{2D}^T \, \operatorname{diag}(|J| \odot W) \,
    ///       V_{2D}$$
    ///  must be inverted separately.
    ///
    /// @param u  Nodal values at quadrature points, size $N_q^2$.
    /// @return   Modal coefficients, size $N_{\text{base}}$.
    VectorXr nodalToModal(const VectorXr &u) const;

    // ---- Sum-factorization (approach 1: flat l iteration) ----

    /// @brief Sum-factorized $r$-direction gradient.
    ///
    /// Computes, for each basis $l$ with $(i_l, j_l)$:
    ///
    /// $$
    ///   R_l = \sum_i \tilde{P}'_{i_l}(r_i)
    ///        \left[ \sum_j G(r_i, s_j) \, \tilde{P}_{j_l}(s_j) \right]
    /// $$
    ///
    /// The two-step decomposition achieves $O(N_q^2 N + N^2 N_q)$ complexity
    /// instead of the naive $O(N_q^2 N^2)$.
    ///
    /// @param G  Field at quadrature points, stored as $N_q \times N_q$ matrix
    ///           where $G(j, i) = G(r_i, s_j)$: row $j$ indexes the
    ///           $s$-direction, column $i$ indexes the $r$-direction.
    /// @return   Gradient coefficients for each basis, size $N_{\text{base}}$.
    VectorXr sumFactorR(const MatrixXrCol &G) const;

    /// @brief Sum-factorized $s$-direction gradient.
    ///
    /// Computes, for each basis $l$ with $(i_l, j_l)$:
    ///
    /// $$
    ///   R_l = \sum_i \tilde{P}_{i_l}(r_i)
    ///        \left[ \sum_j G(r_i, s_j) \, \tilde{P}'_{j_l}(s_j) \right]
    /// $$
    ///
    /// @param G  Field at quadrature points, stored as $N_q \times N_q$ matrix
    ///           where $G(j, i) = G(r_i, s_j)$: row $j$ indexes the
    ///           $s$-direction, column $i$ indexes the $r$-direction.
    /// @return   Gradient coefficients for each basis, size $N_{\text{base}}$.
    VectorXr sumFactorS(const MatrixXrCol &G) const;

    // ---- OCCA device memory ----

    /// @brief Allocate device memory for all pre-computed data via
    ///        DeviceMemoryManager.
    ///
    /// After this call, the `o_*` accessors return valid `occa::memory` handles
    /// suitable for passing to OCCA kernels.
    ///
    /// @param mgr  Device memory manager for the active OCCA device.
    void allocateDeviceMemory(DeviceMemoryManager &mgr);

    /// @brief Device memory handle for 2D quadrature points.
    occa::memory o_quadraturePoints() const
    {
        return o_quad_points_;
    }

    /// @brief Device memory handle for 2D quadrature weights.
    occa::memory o_quadratureWeights() const
    {
        return o_quad_weights_;
    }

    /// @brief Device memory handle for 2D Vandermonde matrix $V_{2D}$.
    occa::memory o_V2D() const
    {
        return o_V2D_;
    }

    /// @brief Device memory handle for $r$-derivative Vandermonde $dV_{2D,r}$.
    occa::memory o_dV2D_r() const
    {
        return o_dV2D_r_;
    }

    /// @brief Device memory handle for $s$-derivative Vandermonde $dV_{2D,s}$.
    occa::memory o_dV2D_s() const
    {
        return o_dV2D_s_;
    }

private:
    const BasisFunctions1D *basis1D_; ///< Non-owning pointer to 1D basis.

    int N_;           ///< 2D polynomial order (total degree $\le N$).
    int N_base_;      ///< Number of basis functions $N_{\text{base}} =
                      ///< (N+1)(N+2)/2$.
    int numPoints1D_; ///< $N_q$ from 1D basis.

    /// Pascal-triangle index map,
    /// shape $N_{\text{base}} \times 2$: $(i_l,j_l)$ for each $l$.
    Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> ij_map_;

    /// Quadrature points, shape $N_q^2 \times 2$, row-major.
    MatrixX2r quad_points_;
    VectorXr quad_weights_; ///< Quadrature weights, size $N_q^2$.

    /// 2D Vandermonde matrix, $N_q^2 \times N_{\text{base}}$, row-major.
    MatrixXr V2D_;

    /// $r$-derivative Vandermonde, $N_q^2 \times N_{\text{base}}$, row-major.
    MatrixXr dV2D_r_;

    /// $s$-derivative Vandermonde, $N_q^2 \times N_{\text{base}}$, row-major.
    MatrixXr dV2D_s_;

    occa::memory o_quad_points_;  ///< Device copy of quadrature points.
    occa::memory o_quad_weights_; ///< Device copy of quadrature weights.
    occa::memory o_V2D_;          ///< Device copy of 2D Vandermonde matrix.
    occa::memory o_dV2D_r_; ///< Device copy of $r$-derivative Vandermonde.
    occa::memory o_dV2D_s_; ///< Device copy of $s$-derivative Vandermonde.

    void buildIndexMap();
    void buildQuadratureData();
    void buildVandermonde2D();
    void buildVandermondeDerivatives2D();
};
