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

class DgField;

/// @brief Physical coordinates of all volume quadrature points via the
///        bilinear element map.
///
/// @return $N_{\text{elem}} \cdot N_q^2 \times 2$ matrix; row
///         $e \cdot N_q^2 + q$ holds $(x, y)$ of quadrature point $q$ of
///         element $e$ (the same ordering as the nodal initial condition).
MatrixX2r quadraturePhysicalCoords(const DgField &field);

/// @brief Evaluate the primitive-variable expressions at the quadrature
///        points and project them onto the modal coefficients of
///        field.o_u().
///
/// @throws std::runtime_error with the exprtk diagnostic when an
///         expression fails to compile.
void applyExprInitialCondition(DgField &field, Real gamma,
                               const std::string &rhoExpr,
                               const std::string &uExpr,
                               const std::string &vExpr,
                               const std::string &pExpr,
                               const std::map<std::string, Real> &symbols);
