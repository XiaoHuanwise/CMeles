/// @file QuadratureCoords.hpp
/// @brief Bilinear reference-to-physical element mapping and the physical
///        coordinates of the volume quadrature points.

#pragma once

#include <Eigen/Dense>

#include "common/Types.hpp"

class DgField;

/// @brief Bilinear reference-to-physical map for every element at the
///        given reference points.
///
/// The map uses the standard $Q_1$ shape functions; triangles are
/// collapsed quadrilaterals ($P_3 = P_4$), for which the formula holds
/// without branching.
///
/// @param vertices N_elem x 8 matrix, one row $[x_1, y_1, \dots, x_4, y_4]$
///                 per element (MeshGeometry::vertices()).
/// @param pts      Reference points, $m \times 2$; row $k = (r_k, s_k)$.
/// @return         $N_{\text{elem}} \cdot m \times 2$ matrix; row
///                 $e \cdot m + k$ holds the physical $(x, y)$.
MatrixX2r bilinearPhysicalCoords(
    const Eigen::Matrix<Real, Eigen::Dynamic, 8, Eigen::RowMajor> &vertices,
    const MatrixX2r &pts);

/// @brief Physical coordinates of all volume quadrature points via the
///        bilinear element map.
///
/// @return $N_{\text{elem}} \cdot N_q^2 \times 2$ matrix; row
///         $e \cdot N_q^2 + q$ holds $(x, y)$ of quadrature point $q$ of
///         element $e$ (the same ordering as the nodal initial condition
///         and the HDF5 quadrature-point output).
MatrixX2r quadraturePhysicalCoords(const DgField &field);
