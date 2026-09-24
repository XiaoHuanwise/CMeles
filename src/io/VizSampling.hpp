/// @file VizSampling.hpp
/// @brief Boundary-inclusive visualization sampling grid shared by the
///        direct writer and the offline checkpoint converter.
///
/// The visualization nodes of an element are the tensor grid
/// $\{-1, \xi_1, \dots, \xi_{N_q}, +1\}^2$ built from the interior
/// Gauss-Legendre abscissae plus the reference-element endpoints. The
/// interior Gauss points alone stop $\sim 7\%$ of a cell short of each
/// boundary (for $N_q = 4$), so a disconnected sub-grid over them leaves
/// background-colored gaps between elements. Evaluating the DG polynomial
/// on the extended grid closes every element up to its interfaces.
///
/// The values at the extended nodes are evaluated from the element's OWN
/// modal coefficients — an exact evaluation of the element polynomial,
/// NOT an interpolation across elements — so the DG discontinuity is
/// preserved as a color jump exactly at the interfaces. The quadrature
/// data is the tensor subset $(i, j) \in [1, N_q]^2$ of the extended
/// grid, so nothing is lost and the archive data remains recoverable by
/// strided extraction.

#pragma once

#include <Eigen/Dense>

#include "common/Types.hpp"

namespace viz {

/// @brief Extended 1-D abscissae $[-1, \xi_1, \dots, \xi_{N_q}, +1]$.
///
/// @param gauss Interior Gauss-Legendre nodes on $[-1, 1]$, size $N_q$.
/// @return      Size $N_q + 2$ vector: the endpoints around \p gauss.
VectorXr extendedAbscissae(const VectorXr &gauss);

/// @brief Tensor points of a per-direction 1-D point set, $n^2 \times 2$.
///
/// Row $k = j \cdot n + i$ stores $(r_i, s_j)$ — the same column-major
/// grid flattening as BasisFunctions2D::quadraturePoints(), so with the
/// extended abscissae the interior Gauss points occupy tensor indices
/// $(i, j) \in [1, N_q]^2$ of the resulting grid.
MatrixX2r tensorPoints(const VectorXr &abscissae1D);

/// @brief Extended-grid physical coordinates recovered from the
///        quadrature-point coordinates alone (offline converter path).
///
/// The element map is bilinear and the tensor Gauss points are unisolvent
/// for the span $\{1, r, s, rs\}$, so a per-element least-squares fit of
/// the four bilinear coefficients reproduces the map exactly (round-off
/// only) without the element vertices. Old checkpoints therefore convert
/// without any format change.
///
/// @param quadXY  Quadrature coordinates, $N_{\text{elem}} \cdot N_q^2 \times 2$
///                (mesh.h5 x/y ordering, row $e \cdot N_q^2 + q$).
/// @param quadPts Reference quadrature points, $N_q^2 \times 2$.
/// @param extPts  Reference extended points, $(N_q + 2)^2 \times 2$.
/// @return        $N_{\text{elem}} \cdot (N_q + 2)^2 \times 2$ extended
///                coordinates in the same row layout.
MatrixX2r extendedCoordsFromQuadratureCoords(const MatrixX2r &quadXY,
                                             const MatrixX2r &quadPts,
                                             const MatrixX2r &extPts);

} // namespace viz
