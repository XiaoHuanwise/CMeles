/// @file QuadratureCoords.cpp
/// @brief Implementation of quadraturePhysicalCoords.

#include "dg/QuadratureCoords.hpp"

#include "basis/BasisFunctions2D.hpp"
#include "dg/DgField.hpp"
#include "mesh/MeshGeometry.hpp"

MatrixX2r quadraturePhysicalCoords(const DgField &field) {
    const int N_elem = field.numElements();
    const int Nq2    = field.numQuadPoints();

    const auto &e8      = field.geometry().vertices(); // N_elem x 8
    const MatrixX2r &qp = field.basis().quadraturePoints();

    MatrixX2r xy(static_cast<Eigen::Index>(N_elem * Nq2), 2);
    for (int e = 0; e < N_elem; ++e) {
        const Real x1 = e8(e, 0), y1 = e8(e, 1);
        const Real x2 = e8(e, 2), y2 = e8(e, 3);
        const Real x3 = e8(e, 4), y3 = e8(e, 5);
        const Real x4 = e8(e, 6), y4 = e8(e, 7);
        for (int q = 0; q < Nq2; ++q) {
            const Real r = qp(q, 0), s = qp(q, 1);
            const Real phi1 = (Real(1) - r) * (Real(1) - s) / Real(4);
            const Real phi2 = (Real(1) + r) * (Real(1) - s) / Real(4);
            const Real phi3 = (Real(1) + r) * (Real(1) + s) / Real(4);
            const Real phi4 = (Real(1) - r) * (Real(1) + s) / Real(4);
            xy(static_cast<Eigen::Index>(e) * Nq2 + q, 0) =
                phi1 * x1 + phi2 * x2 + phi3 * x3 + phi4 * x4;
            xy(static_cast<Eigen::Index>(e) * Nq2 + q, 1) =
                phi1 * y1 + phi2 * y2 + phi3 * y3 + phi4 * y4;
        }
    }
    return xy;
}
