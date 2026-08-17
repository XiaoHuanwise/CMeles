/// @file StructuredMeshGenerator.hpp
/// @brief Structured mesh generator: rectangular / parallelogram domains
///        of quadrilaterals, optionally split into triangles.
///
/// The generator produces raw topology (global vertices + element-vertex
/// connectivity) and hands it to the Mesh constructor, which builds the
/// face adjacency. This is the entry point for mesh creation until the
/// HDF5 mesh reader is implemented by the I/O module; the reader will
/// provide the same raw topology to the Mesh constructor.

#pragma once

#include "Mesh.hpp"

/// @brief Structured mesh generator.
class StructuredMeshGenerator
{
public:
    /// @brief Generation parameters.
    struct Params
    {
        int nx  = 1;       ///< Number of elements in the $x$ direction.
        int ny  = 1;       ///< Number of elements in the $y$ direction.
        Real x0 = Real(0); ///< Origin $x$ coordinate.
        Real y0 = Real(0); ///< Origin $y$ coordinate.
        Real dx = Real(1); ///< Element size in the $x$ direction.
        Real dy = Real(1); ///< Element size in the $y$ direction.
        /// Shear parameter: vertex $(i, j)$ is placed at
        /// $(x_0 + i\,dx + j\,\text{skewX}, \, y_0 + j\,dy)$. A non-zero
        /// value produces a parallelogram-shaped domain; every element is
        /// still a parallelogram (constant Jacobian).
        Real skewX = Real(0);
        /// If true, each quadrilateral is split along the $P_1 \to P_3$
        /// diagonal into two triangles (degenerate quadrilaterals with
        /// $P_3 = P_4$).
        bool splitTriangles = false;
    };

    /// @brief Generate a structured mesh.
    /// @param p Generation parameters.
    /// @return  Mesh with faces/adjacency built.
    static Mesh generate(const Params &p);
};
