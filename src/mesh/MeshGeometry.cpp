/// @file MeshGeometry.cpp
/// @brief Implementation of MeshGeometry.

#include "MeshGeometry.hpp"

#include <cmath>
#include <stdexcept>

#include "Mesh.hpp"
#include "basis/BasisFunctions2D.hpp"
#include "core/DeviceMemoryManager.hpp"

namespace
{
/// @brief Face-to-element reference coordinate maps.
///
/// Table 2.5 of mesh_and_geometry.md. Each table is keyed by the
/// *querying element's own* local face number: kFaceRefLeft is the
/// element's own face map with its counter-clockwise traversal ($t$
/// increasing along the edge direction $A \to B$); kFaceRefRight is the
/// right element's own face map traversed clockwise, i.e. its own map
/// with $t \to -t$ (the right element traverses the same physical edge
/// against its own counter-clockwise direction).
///
/// | f | own face as left elem. (CCW) | own face as right elem. (CW) |
/// |:-:|:------------------------------:|:----------------------------:|
/// | 0 | $(-1, -t)$                    | $(-1, +t)$                   |
/// | 1 | $(t, -1)$                     | $(-t, -1)$                   |
/// | 2 | $(+1, t)$                     | $(+1, -t)$                   |
/// | 3 | $(-t, +1)$                    | $(+t, +1)$                   |
///
/// In an all-quadrilateral mesh $f_R = (f_L + 2) \bmod 4$ (opposite
/// faces), which is why an earlier version could key the right map by
/// $f_L$. Triangle (degenerate quadrilateral) meshes break this pairing
/// — e.g. along a split diagonal the shared edge has $(f_L, f_R) = (0,
/// 1)$ — so the right map must be keyed by the right element's own
/// local face number, exactly what face_elements_(F, 3) stores.
constexpr FaceRefMap kFaceRefLeft[4] = {
    {Real(0), Real(-1), Real(-1), Real(0)}, // f=0: r=-1,   s=-t
    {Real(1), Real(0), Real(0), Real(-1)},  // f=1: r=t,    s=-1
    {Real(0), Real(1), Real(1), Real(0)},   // f=2: r=+1,   s=t
    {Real(-1), Real(0), Real(0), Real(1)},  // f=3: r=-t,   s=+1
};

constexpr FaceRefMap kFaceRefRight[4] = {
    {Real(0), Real(-1), Real(1), Real(0)},  // f=0: r=-1,   s=+t
    {Real(-1), Real(0), Real(0), Real(-1)}, // f=1: r=-t,   s=-1
    {Real(0), Real(1), Real(-1), Real(0)},  // f=2: r=+1,   s=-t
    {Real(1), Real(0), Real(0), Real(1)},   // f=3: r=+t,   s=+1
};
} // namespace

MeshGeometry::MeshGeometry(const Mesh &mesh, const BasisFunctions2D &basis)
    : mesh_(&mesh), basis_(&basis)
{
    N_elem_ = mesh.numElements();
    N_face_ = mesh.numFaces();
    Nq_     = basis.numPoints1D();
    Nq2_    = basis.numPoints();
    N_base_ = basis.numBases();

    buildElementGeometry();
    buildFaceGeometry();
}

// ---------------------------------------------------------------------------
// Per-element geometry
// ---------------------------------------------------------------------------

void MeshGeometry::buildElementGeometry()
{
    const MatrixX2r &meshVerts = mesh_->vertices();
    const auto &elemVerts      = mesh_->elementVertices();
    const MatrixX2r &qp        = basis_->quadraturePoints();
    const VectorXr &qw         = basis_->quadratureWeights();
    const MatrixXr &V2D        = basis_->vandermonde();

    elem_vertices_.resize(N_elem_, 8);
    absJ_.resize(N_elem_ * Nq2_);
    Jinv11_.resize(N_elem_ * Nq2_);
    Jinv12_.resize(N_elem_ * Nq2_);
    Jinv21_.resize(N_elem_ * Nq2_);
    Jinv22_.resize(N_elem_ * Nq2_);
    lambdaWJ_.resize(N_elem_ * Nq2_);
    minv_.resize(N_elem_ * N_base_ * N_base_);
    jacobian_constant_.resize(N_elem_);
    minv_diag_.resize(N_elem_);

    // Parallelogram condition: P1 + P3 == P2 + P4 (within tolerance).
    is_orthogonal_ = true;

    for (int e = 0; e < N_elem_; ++e)
    {
        // (1) Vertex coordinates into the 8-slot row.
        Real x[4], y[4];
        for (int k = 0; k < 4; ++k)
        {
            const int v                  = elemVerts(e, k);
            x[k]                         = meshVerts(v, 0);
            y[k]                         = meshVerts(v, 1);
            elem_vertices_(e, 2 * k)     = x[k];
            elem_vertices_(e, 2 * k + 1) = y[k];
        }

        // (2) Bilinear coefficients (Section 2.4 of mesh_and_geometry.md):
        //     x(r,s) = a0 + a1 r + a2 s + a3 rs, y likewise with b_i.
        const Real a0 = (x[0] + x[1] + x[2] + x[3]) / Real(4);
        const Real a1 = (-x[0] + x[1] + x[2] - x[3]) / Real(4);
        const Real a2 = (-x[0] - x[1] + x[2] + x[3]) / Real(4);
        const Real a3 = (x[0] - x[1] + x[2] - x[3]) / Real(4);
        const Real b0 = (y[0] + y[1] + y[2] + y[3]) / Real(4);
        const Real b1 = (-y[0] + y[1] + y[2] - y[3]) / Real(4);
        const Real b2 = (-y[0] - y[1] + y[2] + y[3]) / Real(4);
        const Real b3 = (y[0] - y[1] + y[2] - y[3]) / Real(4);
        (void)a0;
        (void)b0;

        // (3) Per-quadrature-point Jacobian quantities.
        //     Point order k = j * Nq + i matches BasisFunctions2D.
        const size_t eOff = static_cast<size_t>(e) * Nq2_;
        for (int q = 0; q < Nq2_; ++q)
        {
            const Real r    = qp(q, 0);
            const Real s    = qp(q, 1);
            const Real dxdr = a1 + a3 * s;
            const Real dxds = a2 + a3 * r;
            const Real dydr = b1 + b3 * s;
            const Real dyds = b2 + b3 * r;
            const Real detJ = dxdr * dyds - dxds * dydr;
            if (!(detJ > Real(0)))
            {
                throw std::runtime_error(
                    "MeshGeometry: non-positive Jacobian determinant in "
                    "element " +
                    std::to_string(e));
            }
            absJ_[eOff + q]     = detJ;
            Jinv11_[eOff + q]   = dyds / detJ;
            Jinv12_[eOff + q]   = -dxds / detJ;
            Jinv21_[eOff + q]   = -dydr / detJ;
            Jinv22_[eOff + q]   = dxdr / detJ;
            lambdaWJ_[eOff + q] = qw(q) * detJ;
        }

        // (4) Orthogonal (constant-Jacobian) detection: the parallelogram
        //     condition P1 + P3 == P2 + P4 (affine map, |J| constant).
        //     Tolerance relative to the coordinate scale of the element.
        const Real tol   = Real(1e3) * RealEpsilon *
                           (std::abs(x[0]) + std::abs(x[1]) + std::abs(x[2]) +
                            std::abs(x[3]) + std::abs(y[0]) + std::abs(y[1]) +
                            std::abs(y[2]) + std::abs(y[3]) + Real(1));
        const Real condX = (x[0] + x[2]) - (x[1] + x[3]);
        const Real condY = (y[0] + y[2]) - (y[1] + y[3]);
        if (std::abs(condX) > tol || std::abs(condY) > tol)
        {
            is_orthogonal_ = false;
        }

        // (5) Inverse mass matrix: M = V2D^T diag(Lambda_wJ) V2D.
        //     Row scaling: diag(Lambda) V2D.
        Eigen::Map<const VectorXr> lam(lambdaWJ_.data() + eOff, Nq2_);
        MatrixXr M =
            (lam.asDiagonal() * V2D).transpose() * V2D; // N_base x N_base
        MatrixXr Minv = M.ldlt().solve(MatrixXr::Identity(N_base_, N_base_));
        Eigen::Map<MatrixXr>(minv_.data() +
                                 static_cast<size_t>(e) * N_base_ * N_base_,
                             N_base_, N_base_) = Minv;

        // (6) Constant-Jacobian quantities (valid regardless of
        //     is_orthogonal_; consumed only when it is true). For an affine
        //     map |J| is constant; sample it at the first quadrature point.
        jacobian_constant_(e) = absJ_[eOff];
        minv_diag_(e)         = Real(1) / absJ_[eOff];
    }
}

// ---------------------------------------------------------------------------
// Per-face geometry
// ---------------------------------------------------------------------------

void MeshGeometry::buildFaceGeometry()
{
    const MatrixX2r &meshVerts = mesh_->vertices();
    const auto &faceElems      = mesh_->faceElements();

    face_normals_.resize(N_face_, 2);
    face_jac_.resize(N_face_);

    for (int F = 0; F < N_face_; ++F)
    {
        // Edge direction is defined by the left element's counter-clockwise
        // traversal, A -> B (increasing face coordinate t).
        const int KL        = faceElems(F, 0);
        const int fL        = faceElems(F, 1);
        const auto [vA, vB] = mesh_->faceVertices(KL, fL);

        // Edge vector e = (ex, ey); normal n = (ey, -ex) is the edge
        // rotated clockwise by 90 degrees, pointing from K_L to K_R.
        const Real ex       = meshVerts(vB, 0) - meshVerts(vA, 0);
        const Real ey       = meshVerts(vB, 1) - meshVerts(vA, 1);
        face_normals_(F, 0) = ey;
        face_normals_(F, 1) = -ex;

        // Face Jacobian |J_f| = |e| / 2 (half the physical edge length).
        face_jac_(F) = std::sqrt(ex * ex + ey * ey) / Real(2);
    }
}

// ---------------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------------

void MeshGeometry::allocateDeviceMemory(DeviceMemoryManager &mgr)
{
    o_vertices_ = mgr.wrapOrMalloc(
        elem_vertices_.data(), static_cast<occa::dim_t>(elem_vertices_.size()));
    o_absJ_        = mgr.wrapOrMalloc(absJ_.data(), absJ_.size());
    o_Jinv11_      = mgr.wrapOrMalloc(Jinv11_.data(), Jinv11_.size());
    o_Jinv12_      = mgr.wrapOrMalloc(Jinv12_.data(), Jinv12_.size());
    o_Jinv21_      = mgr.wrapOrMalloc(Jinv21_.data(), Jinv21_.size());
    o_Jinv22_      = mgr.wrapOrMalloc(Jinv22_.data(), Jinv22_.size());
    o_lambdaWJ_    = mgr.wrapOrMalloc(lambdaWJ_.data(), lambdaWJ_.size());
    o_minv_        = mgr.wrapOrMalloc(minv_.data(), minv_.size());
    o_faceNormals_ = mgr.wrapOrMalloc(
        face_normals_.data(), static_cast<occa::dim_t>(face_normals_.size()));
    o_faceJac_ = mgr.wrapOrMalloc(face_jac_.data(), face_jac_.size());

    const auto &faceElems = mesh_->faceElements();
    const auto &elemFaces = mesh_->elementFaces();
    const auto &faceTypes = mesh_->faceTypes();

    // Row-major Eigen matrices have non-contiguous columns; copy each
    // column into a contiguous int buffer before wrapping. The buffers are
    // members (face_kl_ etc.) so that the wrapMemory handles stay valid on
    // unified memory backends (zero-copy aliases the host pointer).
    face_kl_     = faceElems.col(0);
    face_fl_     = faceElems.col(1);
    face_kr_     = faceElems.col(2);
    face_fr_     = faceElems.col(3);
    o_faceKL_    = mgr.wrapOrMallocInt(face_kl_.data(), N_face_);
    o_faceFL_    = mgr.wrapOrMallocInt(face_fl_.data(), N_face_);
    o_faceKR_    = mgr.wrapOrMallocInt(face_kr_.data(), N_face_);
    o_faceFR_    = mgr.wrapOrMallocInt(face_fr_.data(), N_face_);
    o_faceTypes_ = mgr.wrapOrMallocInt(faceTypes.data(), N_face_);
    o_elemFaces_ = mgr.wrapOrMallocInt(
        elemFaces.data(), static_cast<occa::dim_t>(elemFaces.size()));
}

// ---------------------------------------------------------------------------
// Face-to-element reference coordinate maps
// ---------------------------------------------------------------------------

const FaceRefMap &faceRefMapLeft(int f)
{
    return kFaceRefLeft[f];
}

const FaceRefMap &faceRefMapRight(int f)
{
    return kFaceRefRight[f];
}
