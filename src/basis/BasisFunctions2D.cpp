/// @file BasisFunctions2D.cpp
/// @brief Implementation of 2D basis functions, Pascal-triangle indexing,
///        tensor-product quadrature, and sum-factorization.

#include "BasisFunctions2D.hpp"
#include "BasisFunctions1D.hpp"
#include "core/DeviceMemoryManager.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

// ============================================================================
// Constructor
// ============================================================================

BasisFunctions2D::BasisFunctions2D(const BasisFunctions1D &basis1D, int N)
    : basis1D_(&basis1D), N_(N),
      N_base_((N + 1) * (N + 2) / 2),
      numPoints1D_(basis1D.numPoints())
{
    if (N < 0) {
        throw std::invalid_argument("BasisFunctions2D: N must be >= 0");
    }
    if (basis1D.order() < N) {
        throw std::invalid_argument(
            "BasisFunctions2D: basis1D.order() must be >= N");
    }

    buildIndexMap();
    buildQuadratureData();
    buildVandermonde2D();
    buildVandermondeDerivatives2D();
}

// ============================================================================
// Pascal-triangle index mapping
// ============================================================================

void BasisFunctions2D::buildIndexMap()
{
    ij_map_.resize(N_base_, 2);

    int l = 0;
    for (int p = 0; p <= N_; ++p) {
        for (int j = 0; j <= p; ++j) {
            int i = p - j;
            ij_map_(l, 0) = i;  // $r$-direction order
            ij_map_(l, 1) = j;  // $s$-direction order
            ++l;
        }
    }
}

int BasisFunctions2D::linearIndex(int i, int j) const
{
    int p = i + j;
    return p * (p + 1) / 2 + j;
}

std::pair<int, int> BasisFunctions2D::indexPair(int l) const
{
    // $p = \lfloor (\sqrt{8l + 1} - 1) / 2 \rfloor$
    int p = static_cast<int>(std::floor((std::sqrt(Real(8) * l + Real(1)) - Real(1)) / Real(2)));
    int j = l - p * (p + 1) / 2;
    int i = p - j;
    return {i, j};
}

// ============================================================================
// Tensor-product quadrature data (column-major grid order)
// ============================================================================

void BasisFunctions2D::buildQuadratureData()
{
    int Nq2 = numPoints1D_ * numPoints1D_;
    quad_points_.resize(Nq2, 2);
    quad_weights_.resize(Nq2);

    const auto &pts1D  = basis1D_->points();
    const auto &wts1D  = basis1D_->weights();

    // Column-major grid order: $k = j \cdot N_q + i$,
    // where $i$ = $r$-direction index, $j$ = $s$-direction index.
    // The point stored at index $k$ is $(r_i, s_j)$.
    for (int j = 0; j < numPoints1D_; ++j) {
        for (int i = 0; i < numPoints1D_; ++i) {
            int k = j * numPoints1D_ + i;
            quad_points_(k, 0) = pts1D(i);  // $r_i$
            quad_points_(k, 1) = pts1D(j);  // $s_j$
            quad_weights_(k) = wts1D(i) * wts1D(j);
        }
    }
}

// ============================================================================
// 2D Vandermonde matrix (column-major grid order)
// ============================================================================

void BasisFunctions2D::buildVandermonde2D()
{
    int Nq2 = numPoints1D_ * numPoints1D_;
    V2D_.resize(Nq2, N_base_);

    const auto &V1D = basis1D_->vandermonde();

    // For each basis $l$ with $(i_l, j_l)$, the 2D Vandermonde column is the
    // column-major flattening of the outer product:
    //   $V1D.col(i_l) \cdot V1D.col(j_l)^\top$
    // Column-major flattening: $(i, j) \to k = j \cdot N_q + i$.
    //
    // $V_{2D}(k, l) = \tilde{P}_{i_l}(r_i) \cdot \tilde{P}_{j_l}(s_j)$
    for (int l = 0; l < N_base_; ++l) {
        int i_l = ij_map_(l, 0);  // $r$-direction order
        int j_l = ij_map_(l, 1);  // $s$-direction order

        // Loop rows first (column-major inner loop over i = row index)
        for (int j = 0; j < numPoints1D_; ++j) {
            Real V1D_sj = V1D(j, j_l);  // $\tilde{P}_{j_l}(s_j)$, constant across i
            for (int i = 0; i < numPoints1D_; ++i) {
                int k = j * numPoints1D_ + i;
                V2D_(k, l) = V1D(i, i_l) * V1D_sj;
            }
        }
    }
}

// ============================================================================
// 2D Vandermonde derivative matrices (column-major grid order)
// ============================================================================

void BasisFunctions2D::buildVandermondeDerivatives2D()
{
    int Nq2 = numPoints1D_ * numPoints1D_;
    dV2D_r_.resize(Nq2, N_base_);
    dV2D_s_.resize(Nq2, N_base_);

    const auto &V1D  = basis1D_->vandermonde();
    const auto &dV1D = basis1D_->vandermondeDerivative();

    for (int l = 0; l < N_base_; ++l) {
        int i_l = ij_map_(l, 0);  // $r$-direction order
        int j_l = ij_map_(l, 1);  // $s$-direction order

        for (int j = 0; j < numPoints1D_; ++j) {
            Real V1D_sj  = V1D(j, j_l);   // $\tilde{P}_{j_l}(s_j)$
            Real dV1D_sj = dV1D(j, j_l);  // $\tilde{P}'_{j_l}(s_j)$
            for (int i = 0; i < numPoints1D_; ++i) {
                int k = j * numPoints1D_ + i;
                dV2D_r_(k, l) = dV1D(i, i_l) * V1D_sj;   // $r$-derivative
                dV2D_s_(k, l) = V1D(i, i_l) * dV1D_sj;   // $s$-derivative
            }
        }
    }
}

// ============================================================================
// Modal-nodal transforms
// ============================================================================

VectorXr BasisFunctions2D::modalToNodal(const VectorXr &u_hat) const
{
    // $u = V_{2D} \cdot \hat{u}$
    assert(u_hat.size() == N_base_ &&
           "modalToNodal: u_hat must have size N_base");
    return V2D_ * u_hat;
}

VectorXr BasisFunctions2D::nodalToModal(const VectorXr &u) const
{
    int Nq2 = numPoints1D_ * numPoints1D_;
    assert(u.size() == Nq2 &&
           "nodalToModal: u must have size Nq^2");

    // $\hat{u} = V_{2D}^T \, W \, u$
    //          $= V_{2D}^T \cdot (u \odot w\_\text{quad})$
    VectorXr u_weighted = u.array() * quad_weights_.array();
    return V2D_.transpose() * u_weighted;
}

// ============================================================================
// Sum-factorization: approach 1 (flat l iteration)
// ============================================================================

VectorXr BasisFunctions2D::sumFactorR(const MatrixXrCol &G) const
{
    // $G$ is $N_q \times N_q$: $G(j, i) = G(r_i, s_j)$
    // $i$ = $r$-direction index (col), $j$ = $s$-direction index (row)
    assert(G.rows() == numPoints1D_ && G.cols() == numPoints1D_ &&
           "sumFactorR: G must be Nq x Nq");

    const auto &V1D  = basis1D_->vandermonde();
    const auto &dV1D = basis1D_->vandermondeDerivative();

    VectorXr R(N_base_);
    R.setZero();

    // Temporary storage for $s$-direction contraction result per basis:
    // $\text{temp}(i) = \sum_j V1D(j, j_l) \cdot G(j, i)$
    VectorXr temp(numPoints1D_);

    for (int l = 0; l < N_base_; ++l) {
        int i_l = ij_map_(l, 0);  // $r$-direction order
        int j_l = ij_map_(l, 1);  // $s$-direction order

        // Step 1 ($s$-direction):
        // $\text{temp}(i) = \sum_j \tilde{P}_{j_l}(s_j) \cdot G(r_i, s_j)$
        //   $= G.col(i)^\top \cdot V1D.col(j_l)$  for each $i$
        for (int i = 0; i < numPoints1D_; ++i) {
            temp(i) = V1D.col(j_l).dot(G.col(i));
        }

        // Step 2 ($r$-direction):
        // $R(l) = \sum_i \tilde{P}'_{i_l}(r_i) \cdot \text{temp}(i)$
        //   $= dV1D.col(i_l)^\top \cdot \text{temp}$
        R(l) = dV1D.col(i_l).dot(temp);
    }

    return R;
}

VectorXr BasisFunctions2D::sumFactorS(const MatrixXrCol &G) const
{
    assert(G.rows() == numPoints1D_ && G.cols() == numPoints1D_ &&
           "sumFactorS: G must be Nq x Nq");

    const auto &V1D  = basis1D_->vandermonde();
    const auto &dV1D = basis1D_->vandermondeDerivative();

    VectorXr R(N_base_);
    R.setZero();

    VectorXr temp(numPoints1D_);

    for (int l = 0; l < N_base_; ++l) {
        int i_l = ij_map_(l, 0);  // $r$-direction order
        int j_l = ij_map_(l, 1);  // $s$-direction order

        // Step 1 ($s$-direction, with derivative):
        // $\text{temp}(i) = \sum_j \tilde{P}'_{j_l}(s_j) \cdot G(r_i, s_j)$
        for (int i = 0; i < numPoints1D_; ++i) {
            temp(i) = dV1D.col(j_l).dot(G.col(i));
        }

        // Step 2 ($r$-direction, with basis values):
        // $R(l) = \sum_i \tilde{P}_{i_l}(r_i) \cdot \text{temp}(i)$
        R(l) = V1D.col(i_l).dot(temp);
    }

    return R;
}

// ---------------------------------------------------------------------------
// OCCA device memory allocation
// ---------------------------------------------------------------------------

void BasisFunctions2D::allocateDeviceMemory(DeviceMemoryManager &mgr)
{
    o_quad_points_  = mgr.wrapOrMalloc(quad_points_.data(),  quad_points_.size());
    o_quad_weights_ = mgr.wrapOrMalloc(quad_weights_.data(), quad_weights_.size());
    o_V2D_          = mgr.wrapOrMalloc(V2D_.data(),          V2D_.size());
    o_dV2D_r_       = mgr.wrapOrMalloc(dV2D_r_.data(),       dV2D_r_.size());
    o_dV2D_s_       = mgr.wrapOrMalloc(dV2D_s_.data(),       dV2D_s_.size());
}
