/// @file MeshGeometry.hpp
/// @brief Per-element and per-face geometric pre-computation for the
///        bilinear-quadrilateral mesh framework, plus OCCA device memory.
///
/// Reference: docs/tech_docs/mesh_and_geometry.md, Sections 1.4 and 2.4.
///
/// The volume quadrature point order matches BasisFunctions2D exactly:
/// $k = j \cdot N_q + i$ (column-major grid order, $i$ indexes the
/// $r$-direction, $j$ the $s$-direction). All per-element arrays are
/// element-major with the $N_q^2$ points of each element contiguous, so
/// that kernels parallelised per element (volume integral, gradient,
/// RHS assembly) read one contiguous block per element.
///
/// The Jacobian determinant is computed per quadrature point directly from
/// the four partial derivatives (bilinear coefficients), which is
/// mathematically equivalent to the edge-vector cross-product expansion of
/// the documentation but unambiguous to implement. The formula holds for
/// both quadrilaterals and collapsed (triangular) elements without any
/// branching.

#pragma once

#include <Eigen/Dense>
#include <occa.hpp>

#include "common/Types.hpp"

class Mesh;
class BasisFunctions2D;
class DeviceMemoryManager;

/// @brief Per-element and per-face geometric pre-computation.
class MeshGeometry
{
public:
    /// @brief Compute all geometric quantities.
    /// @param mesh  Mesh topology (lifetime must exceed this object).
    /// @param basis 2D basis functions defining the quadrature points and
    ///              the Vandermonde matrix (lifetime must exceed this
    ///              object).
    MeshGeometry(const Mesh &mesh, const BasisFunctions2D &basis);

    // ---- Sizes ----

    int numElements() const noexcept
    {
        return N_elem_;
    }
    int numFaces() const noexcept
    {
        return N_face_;
    }

    /// @brief Number of quadrature points per element, $N_q^2$.
    int numPointsPerElement() const noexcept
    {
        return Nq2_;
    }

    /// @brief Number of 2D basis functions $N_{\text{base}}$.
    int numBases() const noexcept
    {
        return N_base_;
    }

    // ---- Per-element geometry (host side) ----

    /// @brief Per-element vertex coordinates, shape
    ///        $N_{\text{elem}} \times 8$: $[x_1, y_1, x_2, y_2, x_3, y_3,
    ///        x_4, y_4]$ per row (row-major).
    const Eigen::Matrix<Real, Eigen::Dynamic, 8, Eigen::RowMajor> &vertices()
        const noexcept
    {
        return elem_vertices_;
    }

    /// @brief Jacobian determinant $|J(r_q, s_q)|$, size
    ///        $N_{\text{elem}} \cdot N_q^2$ (element-major).
    const VectorXr &absJacobian() const noexcept
    {
        return absJ_;
    }

    /// @brief Inverse Jacobian components, each of size
    ///        $N_{\text{elem}} \cdot N_q^2$ (element-major).
    ///
    /// Component layout ($\mathbf{J}^{-1}_{ij}$ in the convention
    /// $\mathbf{J}_{11}=\partial x/\partial r$):
    ///   Jinv11 = $\partial y/\partial s / |J|$ = $\partial r/\partial x$,
    ///   Jinv12 = $-\partial x/\partial s / |J|$ = $\partial r/\partial y$,
    ///   Jinv21 = $-\partial y/\partial r / |J|$ = $\partial s/\partial x$,
    ///   Jinv22 = $\partial x/\partial r / |J|$ = $\partial s/\partial y$.
    ///
    /// These are the components of the inverse-map Jacobian
    /// $\mathbf{J}^{-1} = \partial(r, s)/\partial(x, y)$ (each row is the
    /// gradient of a reference coordinate w.r.t. the physical ones).
    /// Gradient transformation uses the column convention
    /// $\begin{bmatrix} u_x \\ u_y \end{bmatrix} =
    /// \mathbf{J}^{-T} \begin{bmatrix} u_r \\ u_s \end{bmatrix}$ — the
    /// cross components swap: read them as (Jinv11, Jinv21, Jinv12, Jinv22)
    /// when multiplying a column of reference derivatives (see
    /// docs/tech_docs/mesh_and_geometry.md, Section 2.4).
    ///
    /// Stored as four separate arrays instead of an interleaved one so
    /// that the per-point loops vectorise cleanly with `-O3 -march=native`.
    const VectorXr &Jinv11() const noexcept
    {
        return Jinv11_;
    }
    const VectorXr &Jinv12() const noexcept
    {
        return Jinv12_;
    }
    const VectorXr &Jinv21() const noexcept
    {
        return Jinv21_;
    }
    const VectorXr &Jinv22() const noexcept
    {
        return Jinv22_;
    }

    /// @brief Weighted quadrature weights $\Lambda_{wJ} = w_q \, |J_K|$,
    ///        size $N_{\text{elem}} \cdot N_q^2$ (element-major).
    const VectorXr &lambdaWJ() const noexcept
    {
        return lambdaWJ_;
    }

    /// @brief Inverse mass matrix $\mathbf{M}^{-1}$ of each element,
    ///        size $N_{\text{elem}} \cdot N_{\text{base}}^2$, element-major
    ///        with the $N_{\text{base}} \times N_{\text{base}}$ block of
    ///        each element in row-major order.
    ///
    /// $$\mathbf{M} = \mathbf{V}_{2D}^T
    /// \operatorname{diag}(\Lambda_{wJ}) \, \mathbf{V}_{2D}$$.
    const VectorXr &massMatrixInverse() const noexcept
    {
        return minv_;
    }

    // ---- Per-face geometry (host side) ----

    /// @brief Un-normalised face normal $\hat{n} = (e_y, -e_x)$ for each
    ///        face, shape $N_{\text{face}} \times 2$. $|\hat{n}|$ equals the
    ///        physical edge length; the normal points from the left element
    ///        $K_L$ to the right element $K_R$.
    const MatrixX2r &faceNormals() const noexcept
    {
        return face_normals_;
    }

    /// @brief Face Jacobian $|J_f| = |\vec{e}| / 2$ (half the physical edge
    ///        length), size $N_{\text{face}}$.
    const VectorXr &faceJacobians() const noexcept
    {
        return face_jac_;
    }

    // ---- Orthogonal (constant-Jacobian) simplification ----

    /// @brief Whether every element satisfies the parallelogram condition
    ///        $P_1 + P_3 = P_2 + P_4$ (within tolerance).
    ///
    /// When true, the bilinear map degenerates to an affine map, the
    /// Jacobian determinant is constant per element and the mass matrix is
    /// diagonal: $\mathbf{M} = |J_e| \, \mathbf{I}$ (see
    /// parallelogram_assumption.md). The per-point arrays are still filled
    /// for data completeness; the constant-quantity accessors below provide
    /// the cheap path.
    bool isOrthogonal() const noexcept
    {
        return is_orthogonal_;
    }

    /// @brief Per-element constant Jacobian determinant $|J_e|$ (only valid
    ///        when isOrthogonal()), size $N_{\text{elem}}$.
    const VectorXr &jacobianConstant() const noexcept
    {
        return jacobian_constant_;
    }

    /// @brief Per-element diagonal inverse mass matrix $1 / |J_e|$ (only
    ///        valid when isOrthogonal()), size $N_{\text{elem}}$.
    const VectorXr &massMatrixInverseDiagonal() const noexcept
    {
        return minv_diag_;
    }

    // ---- OCCA device memory ----

    /// @brief Allocate device memory for all pre-computed data via
    ///        DeviceMemoryManager.
    ///
    /// After this call, the `o_*` accessors return valid `occa::memory`
    /// handles suitable for passing to OCCA kernels.
    void allocateDeviceMemory(DeviceMemoryManager &mgr);

    /// @name Device handles (see allocateDeviceMemory for sizes).
    /// @{
    occa::memory o_vertices() const
    {
        return o_vertices_;
    }
    occa::memory o_absJ() const
    {
        return o_absJ_;
    }
    occa::memory o_Jinv11() const
    {
        return o_Jinv11_;
    }
    occa::memory o_Jinv12() const
    {
        return o_Jinv12_;
    }
    occa::memory o_Jinv21() const
    {
        return o_Jinv21_;
    }
    occa::memory o_Jinv22() const
    {
        return o_Jinv22_;
    }
    occa::memory o_lambdaWJ() const
    {
        return o_lambdaWJ_;
    }
    occa::memory o_minv() const
    {
        return o_minv_;
    }
    occa::memory o_faceNormals() const
    {
        return o_faceNormals_;
    }
    occa::memory o_faceJac() const
    {
        return o_faceJac_;
    }
    occa::memory o_faceTypes() const
    {
        return o_faceTypes_;
    }
    occa::memory o_faceKL() const
    {
        return o_faceKL_;
    }
    occa::memory o_faceFL() const
    {
        return o_faceFL_;
    }
    occa::memory o_faceKR() const
    {
        return o_faceKR_;
    }
    occa::memory o_faceFR() const
    {
        return o_faceFR_;
    }
    occa::memory o_elemFaces() const
    {
        return o_elemFaces_;
    }
    /// @}

private:
    const Mesh *mesh_;              ///< Non-owning pointer to the topology.
    const BasisFunctions2D *basis_; ///< Non-owning pointer to the basis.

    int N_elem_ = 0; ///< Number of elements.
    int N_face_ = 0; ///< Number of faces.
    int Nq_     = 0; ///< Quadrature points per dimension.
    int Nq2_    = 0; ///< Quadrature points per element, $N_q^2$.
    int N_base_ = 0; ///< Number of basis functions.

    Eigen::Matrix<Real, Eigen::Dynamic, 8, Eigen::RowMajor> elem_vertices_;
    VectorXr absJ_;
    VectorXr Jinv11_, Jinv12_, Jinv21_, Jinv22_;
    VectorXr lambdaWJ_;
    VectorXr minv_;
    MatrixX2r face_normals_;
    VectorXr face_jac_;

    bool is_orthogonal_ = false;
    VectorXr jacobian_constant_;
    VectorXr minv_diag_;

    occa::memory o_vertices_, o_absJ_;
    occa::memory o_Jinv11_, o_Jinv12_, o_Jinv21_, o_Jinv22_;
    occa::memory o_lambdaWJ_, o_minv_;
    occa::memory o_faceNormals_, o_faceJac_;
    occa::memory o_faceTypes_, o_faceKL_, o_faceFL_, o_faceKR_, o_faceFR_;
    occa::memory o_elemFaces_;

    /// @brief Per-element geometry: bilinear coefficients, per-point
    ///        Jacobian quantities and the inverse mass matrix.
    void buildElementGeometry();

    /// @brief Per-face geometry: normals and face Jacobians.
    void buildFaceGeometry();
};

/// @brief Coefficients of the affine map from face coordinate
///        $t \in [-1, 1]$ to the element reference coordinates:
///        $r = c_r \, t + d_r$, $s = c_s \, t + d_s$.
///
/// Table 2.5 of mesh_and_geometry.md. $t$ increases along the face edge
/// vector $A \to B$ (fixed by the left element's counter-clockwise
/// traversal); see faceRefMapLeft/faceRefMapRight for how each element
/// queries its own face map.
struct FaceRefMap
{
    Real cr, dr; ///< $r = c_r t + d_r$.
    Real cs, ds; ///< $s = c_s t + d_s$.
};

/// @brief Face-to-element reference coordinate map of the element's own
///        local face \p f, queried by the face's left element
///        (counter-clockwise traversal: $t$ along $A \to B$).
const FaceRefMap &faceRefMapLeft(int f);

/// @brief Face-to-element reference coordinate map of the right element's
///        own local face \p f, traversed clockwise ($t$ reversed w.r.t.
///        its own counter-clockwise convention). \p f must be the face
///        number in the right element itself (face_elements_(F, 3)), not
///        the left one: the quad-only pairing $f_R = (f_L + 2) \bmod 4$
///        breaks on triangle (degenerate quad) meshes.
const FaceRefMap &faceRefMapRight(int f);
