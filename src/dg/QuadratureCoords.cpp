/// @file QuadratureCoords.cpp
/// @brief Implementation of the bilinear element map and
///        quadraturePhysicalCoords.

#include "dg/QuadratureCoords.hpp"

#include "basis/BasisFunctions2D.hpp"
#include "dg/DgField.hpp"
#include "mesh/MeshGeometry.hpp"

MatrixX2r bilinearPhysicalCoords(
    const Eigen::Matrix<Real, Eigen::Dynamic, 8, Eigen::RowMajor> &vertices,
    const MatrixX2r &pts) {
    const Eigen::Index nElem = vertices.rows();
    const Eigen::Index m     = pts.rows();

    MatrixX2r xy(nElem * m, 2);
    for (Eigen::Index e = 0; e < nElem; ++e) {
        const Real x1 = vertices(e, 0), y1 = vertices(e, 1);
        const Real x2 = vertices(e, 2), y2 = vertices(e, 3);
        const Real x3 = vertices(e, 4), y3 = vertices(e, 5);
        const Real x4 = vertices(e, 6), y4 = vertices(e, 7);
        for (Eigen::Index k = 0; k < m; ++k) {
            const Real r = pts(k, 0), s = pts(k, 1);
            const Real phi1  = (Real(1) - r) * (Real(1) - s) / Real(4);
            const Real phi2  = (Real(1) + r) * (Real(1) - s) / Real(4);
            const Real phi3  = (Real(1) + r) * (Real(1) + s) / Real(4);
            const Real phi4  = (Real(1) - r) * (Real(1) + s) / Real(4);
            xy(e * m + k, 0) = phi1 * x1 + phi2 * x2 + phi3 * x3 + phi4 * x4;
            xy(e * m + k, 1) = phi1 * y1 + phi2 * y2 + phi3 * y3 + phi4 * y4;
        }
    }
    return xy;
}

MatrixX2r quadraturePhysicalCoords(const DgField &field) {
    return bilinearPhysicalCoords(field.geometry().vertices(),
                                  field.basis().quadraturePoints());
}
