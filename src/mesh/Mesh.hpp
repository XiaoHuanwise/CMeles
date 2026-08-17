/// @file Mesh.hpp
/// @brief Mesh topology: global vertices, element-vertex connectivity,
///        faces (element edges) and two-sided adjacency.
///
/// Reference: docs/tech_docs/mesh_and_geometry.md, Section 2.5.
///
/// All elements use the unified quadrilateral framework: four vertices
/// $P_1 \dots P_4$ ordered counter-clockwise in reference space
/// $(r, s) \in [-1, 1]^2$. Triangles are represented as degenerate
/// quadrilaterals with $P_3 = P_4$ (vertex collapse); their collapsed
/// face (local face 3) has element-face slot $-1$ and takes no part in
/// face assembly.

#pragma once

#include <Eigen/Dense>
#include <utility>

#include "common/Types.hpp"

/// @brief Mesh topology: global vertices, element-vertex connectivity,
///        faces (element edges) and two-sided adjacency.
///
/// The constructor builds the face list from the raw element-vertex
/// connectivity by matching shared edges (see buildFaces).
class Mesh
{
public:
    /// @brief Face type. Only Interior/Boundary are distinguished for now;
    ///        the boundary-condition module will extend this enum later.
    enum class FaceType : int
    {
        Interior = 0,
        Boundary = 1
    };

    /// @brief Construct from raw topology and build the face list.
    /// @param vertices         Global vertex coordinates, shape
    ///                         $N_{\text{vert}} \times 2$, row-major.
    /// @param elementVertices  Element-vertex connectivity, shape
    ///                         $N_{\text{elem}} \times 4$, row-major. Row $e$
    ///                         lists the global indices of $P_1 \dots P_4$
    ///                         (counter-clockwise; triangles satisfy
    ///                         `(e,2) == (e,3)`).
    ///
    /// @pre All elements are consistently oriented (counter-clockwise).
    ///      Mismatched edge directions are rejected with an error.
    Mesh(const MatrixX2r &vertices,
         const Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>
             &elementVertices);

    // ---- Sizes ----

    int numVertices() const noexcept
    {
        return N_vert_;
    }
    int numElements() const noexcept
    {
        return N_elem_;
    }
    int numFaces() const noexcept
    {
        return N_face_;
    }

    // ---- Topology data ----

    /// @brief Global vertex coordinates, shape $N_{\text{vert}} \times 2$,
    ///        row-major.
    const MatrixX2r &vertices() const noexcept
    {
        return vertices_;
    }

    /// @brief Element-vertex connectivity, shape $N_{\text{elem}} \times 4$,
    ///        row-major. Triangles satisfy `(e,2) == (e,3)`.
    const Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> &elementVertices()
        const noexcept
    {
        return elem_verts_;
    }

    /// @brief Element local-face to global-face mapping, shape
    ///        $N_{\text{elem}} \times 4$. The collapsed face of a triangle
    ///        (local face 3) holds $-1$.
    const Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> &elementFaces()
        const noexcept
    {
        return elem_faces_;
    }

    /// @brief Per-face adjacency, shape $N_{\text{face}} \times 4$; row
    ///        $= (K_L, f_L, K_R, f_R)$. Boundary faces have $K_R = -1$.
    const Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> &faceElements()
        const noexcept
    {
        return face_elements_;
    }

    /// @brief Per-face type (underlying int of FaceType), size
    /// $N_{\text{face}}$.
    const Eigen::VectorXi &faceTypes() const noexcept
    {
        return face_types_;
    }

    // ---- Queries ----

    /// @brief Whether element \p elem is a triangle ($P_3 = P_4$).
    bool isTriangle(int elem) const noexcept
    {
        return elem_verts_(elem, 2) == elem_verts_(elem, 3);
    }

    /// @brief Whether face \p face is a boundary face ($K_R = -1$).
    bool isBoundaryFace(int face) const noexcept
    {
        return face_elements_(face, 2) == -1;
    }

    /// @brief Directed endpoint pair $(v_A, v_B)$ of local face \p f of
    ///        element \p elem; the direction $A \to B$ is the direction of
    ///        increasing face coordinate $t$.
    ///
    /// Table (Section 2.5 of mesh_and_geometry.md):
    ///   f=0 (left,  $r=-1$): $P_4 \to P_1$
    ///   f=1 (bottom,$s=-1$): $P_1 \to P_2$
    ///   f=2 (right, $r=+1$): $P_2 \to P_3$
    ///   f=3 (top,   $s=+1$): $P_3 \to P_4$
    std::pair<int, int> faceVertices(int elem, int f) const;

private:
    int N_vert_ = 0; ///< Number of global vertices.
    int N_elem_ = 0; ///< Number of elements.
    int N_face_ = 0; ///< Number of faces (edges) in the face list.

    MatrixX2r vertices_;
    Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> elem_verts_;
    Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> elem_faces_;
    Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> face_elements_;
    Eigen::VectorXi face_types_;

    /// @brief Build the face list by matching shared edges.
    ///
    /// The first element claiming an edge becomes its left element $K_L$;
    /// the second claim (traversed in the opposite direction) becomes the
    /// right element $K_R$; unclaimed edges are boundary faces with
    /// $K_R = -1$. Collapsed (zero-length) edges of triangles are skipped.
    void buildFaces();
};
