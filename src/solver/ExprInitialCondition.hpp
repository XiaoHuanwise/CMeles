/// @file ExprInitialCondition.hpp
/// @brief Expression-based initial conditions (exprtk) and quadrature-point
///        coordinate mapping.
///
/// The [initial_condition] section provides the primitive variables
/// (rho, u, v, p) as exprtk expressions of the built-in symbols x, y, t
/// (t = 0) and gamma plus any numeric constants from
/// [initial_condition.symbols]. The expressions are evaluated at the volume
/// quadrature points, converted to conserved variables
/// $E = p/(\gamma-1) + \tfrac12 \rho (u^2 + v^2)$ and projected onto the
/// modal coefficients through DgField::setInitialConditionNodal.
///
/// The four expressions are compiled independently — no cross references
/// between them.

#pragma once

#include <map>
#include <string>

#include "common/Types.hpp"

class Config;
class DgField;

namespace solver_detail
{
/// @brief Evaluate the configured [initial_condition] expressions and
///        project them onto the field (Config-aware entry point).
void applyExprICFromConfig(DgField &field, const Config &cfg);
} // namespace solver_detail

/// @brief Physical coordinates of all volume quadrature points via the
///        bilinear element map.
///
/// @return $N_{\text{elem}} \cdot N_q^2 \times 2$ matrix; row
///         $e \cdot N_q^2 + q$ holds $(x, y)$ of quadrature point $q$ of
///         element $e$ (the same ordering as the nodal initial condition).
///         Also the coordinate source of the future HDF5 quadrature-point
///         output.
MatrixX2r quadraturePhysicalCoords(const DgField &field);
