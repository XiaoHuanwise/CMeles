/// @file StructuredMeshGenerator.cpp
/// @brief Implementation of StructuredMeshGenerator.

#include "StructuredMeshGenerator.hpp"

#include <stdexcept>

Mesh StructuredMeshGenerator::generate(const Params &p)
{
    if (p.nx < 1 || p.ny < 1)
    {
        throw std::runtime_error(
            "StructuredMeshGenerator: nx and ny must be >= 1");
    }
    if (!(p.dx > Real(0)) || !(p.dy > Real(0)))
    {
        throw std::runtime_error(
            "StructuredMeshGenerator: dx and dy must be positive");
    }

    const int nx = p.nx, ny = p.ny;
    const int nVerts = (nx + 1) * (ny + 1);

    // Global vertex indexing: v(i, j) = j * (nx + 1) + i.
    MatrixX2r vertices(nVerts, 2);
    for (int j = 0; j <= ny; ++j)
    {
        for (int i = 0; i <= nx; ++i)
        {
            const int v    = j * (nx + 1) + i;
            vertices(v, 0) = p.x0 + Real(i) * p.dx + Real(j) * p.skewX;
            vertices(v, 1) = p.y0 + Real(j) * p.dy;
        }
    }

    const int nElem = p.splitTriangles ? 2 * nx * ny : nx * ny;
    Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> elemVerts(nElem, 4);

    if (!p.splitTriangles)
    {
        // Quadrilaterals, counter-clockwise: (v00, v10, v11, v01).
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                const int e   = j * nx + i;
                const int v00 = j * (nx + 1) + i;
                const int v10 = v00 + 1;
                const int v11 = v00 + (nx + 1) + 1;
                const int v01 = v00 + (nx + 1);
                elemVerts.row(e) << v00, v10, v11, v01;
            }
        }
    }
    else
    {
        // Two triangles per quadrilateral, split along the P1 -> P3
        // diagonal. Both triangles have their collapsed edge (P3 = P4) at
        // local face 3, matching the general convention.
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                const int e0  = 2 * (j * nx + i);
                const int v00 = j * (nx + 1) + i;
                const int v10 = v00 + 1;
                const int v11 = v00 + (nx + 1) + 1;
                const int v01 = v00 + (nx + 1);
                // Lower triangle: (v00, v10, v11, v11).
                elemVerts.row(e0) << v00, v10, v11, v11;
                // Upper triangle: (v00, v11, v01, v01).
                elemVerts.row(e0 + 1) << v00, v11, v01, v01;
            }
        }
    }

    return Mesh(vertices, elemVerts);
}
