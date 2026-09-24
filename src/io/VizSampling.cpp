/// @file VizSampling.cpp
/// @brief Implementation of the boundary-inclusive visualization grid
///        helpers.

#include "io/VizSampling.hpp"

#include <Eigen/QR>

#include <stdexcept>

namespace viz {

VectorXr extendedAbscissae(const VectorXr &gauss) {
    VectorXr ext(gauss.size() + 2);
    ext(0)                       = Real(-1);
    ext.segment(1, gauss.size()) = gauss;
    ext(ext.size() - 1)          = Real(1);
    return ext;
}

MatrixX2r tensorPoints(const VectorXr &a) {
    const Eigen::Index n = a.size();
    MatrixX2r pts(n * n, 2);
    for (Eigen::Index j = 0; j < n; ++j) {
        for (Eigen::Index i = 0; i < n; ++i) {
            const Eigen::Index k = j * n + i;
            pts(k, 0)            = a(i);
            pts(k, 1)            = a(j);
        }
    }
    return pts;
}

namespace {
/// @brief Design matrix of the bilinear span $\{1, r, s, rs\}$ at the
///        given reference points.
MatrixXr bilinearDesign(const MatrixX2r &pts) {
    MatrixXr D(pts.rows(), 4);
    for (Eigen::Index k = 0; k < pts.rows(); ++k) {
        D(k, 0) = Real(1);
        D(k, 1) = pts(k, 0);
        D(k, 2) = pts(k, 1);
        D(k, 3) = pts(k, 0) * pts(k, 1);
    }
    return D;
}
} // namespace

MatrixX2r extendedCoordsFromQuadratureCoords(const MatrixX2r &quadXY,
                                             const MatrixX2r &quadPts,
                                             const MatrixX2r &extPts) {
    const Eigen::Index nQ    = quadPts.rows();
    const Eigen::Index nElem = quadXY.rows() / nQ;
    if (nElem * nQ != quadXY.rows() || nElem == 0) {
        throw std::invalid_argument(
            "extendedCoordsFromQuadratureCoords: quadXY rows are not a "
            "positive multiple of the quadrature point count");
    }

    // One decomposition shared by all elements and both coordinates: the
    // quadrature samples of a bilinear map are consistent, so the
    // least-squares solution IS the exact coefficient set.
    const Eigen::ColPivHouseholderQR<MatrixXr> qr(bilinearDesign(quadPts));
    if (qr.rank() < 4) {
        throw std::invalid_argument(
            "extendedCoordsFromQuadratureCoords: quadrature point set is "
            "degenerate for the bilinear fit");
    }
    const MatrixXr Dext = bilinearDesign(extPts);

    const Eigen::Index mExt = extPts.rows();
    MatrixX2r xy(nElem * mExt, 2);
    for (Eigen::Index e = 0; e < nElem; ++e) {
        const MatrixXr coeff          = qr.solve(quadXY.middleRows(e * nQ, nQ));
        xy.middleRows(e * mExt, mExt) = Dext * coeff;
    }
    return xy;
}

} // namespace viz
