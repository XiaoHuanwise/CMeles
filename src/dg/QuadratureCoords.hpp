/// @file QuadratureCoords.hpp
/// @brief Physical coordinates of the volume quadrature points.

#pragma once

#include "common/Types.hpp"

class DgField;

/// @brief Physical coordinates of all volume quadrature points via the
///        bilinear element map.
///
/// @return $N_{\text{elem}} \cdot N_q^2 \times 2$ matrix; row
///         $e \cdot N_q^2 + q$ holds $(x, y)$ of quadrature point $q$ of
///         element $e$ (the same ordering as the nodal initial condition
///         and the HDF5 quadrature-point output).
MatrixX2r quadraturePhysicalCoords(const DgField &field);
