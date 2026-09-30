/// @file ExprBoundaryCondition.hpp
/// @brief Expression-evaluated Dirichlet boundary states (exprtk).
///
/// Boundary faces resolved to [bc] type "expr" take their exterior state
/// from the [bc] rho / u / v / p expressions evaluated at every
/// boundary-face quadrature point (x, y) and the current physical time t.
/// The expressions are compiled once; refresh(t) re-evaluates them and
/// uploads the conserved ghost states to the field, so the solver refreshes
/// once per physical time step (stage-level time refinement is unnecessary:
/// the refresh error of a boundary datum moving at speed O(1) over one step
/// is dt, far below the face size).
///
/// The interior of this class (exprtk state) is hidden behind an opaque
/// State pointer so exprtk.hpp is compiled in exactly one translation unit.

#pragma once

#include <memory>

#include "common/Types.hpp"

class Config;
class DgField;

namespace solver_detail {

/// @brief Compiled [bc] Dirichlet expressions bound to a field's boundary
///        points; call refresh(t) before each time step.
class ExprBoundaryCondition {
public:
    /// @brief Compile the [bc] rho/u/v/p expressions.
    ///
    /// @throws std::runtime_error with the exprtk diagnostic when an
    ///         expression fails to compile.
    ExprBoundaryCondition(const Config &cfg, DgField &field);

    ~ExprBoundaryCondition();

    ExprBoundaryCondition(const ExprBoundaryCondition &)            = delete;
    ExprBoundaryCondition &operator=(const ExprBoundaryCondition &) = delete;

    /// @brief Evaluate the expressions at physical time \p t and upload the
    ///        conserved ghost states
    ///        $E = p/(\gamma-1) + \tfrac12 \rho (u^2 + v^2)$.
    void refresh(Real t);

private:
    /// Opaque exprtk state (symbol table + compiled expressions).
    struct State;

    DgField &field_;
    std::unique_ptr<State> state_;
};

} // namespace solver_detail
