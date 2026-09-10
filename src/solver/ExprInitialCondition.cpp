/// @file ExprInitialCondition.cpp
/// @brief Implementation of the expression-based initial condition.
///
/// exprtk.hpp is included only in this translation unit to isolate its
/// compilation cost from the rest of the project.

#include "ExprInitialCondition.hpp"

#include <stdexcept>
#include <string>

#include <exprtk.hpp>

#include "config/Config.hpp"
#include "dg/DgField.hpp"
#include "dg/QuadratureCoords.hpp"

namespace {
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
                               const std::map<std::string, Real> &symbols) {
    // Symbol storage must outlive the expressions (exprtk variables are
    // references into this state).
    Real x = 0, y = 0, t = 0, gam = gamma;
    std::map<std::string, Real> symValues = symbols;

    exprtk::symbol_table<Real> symTab;
    symTab.add_variable("x", x);
    symTab.add_variable("y", y);
    symTab.add_variable("t", t);
    symTab.add_variable("gamma", gam);
    for (auto &[name, value] : symValues) {
        symTab.add_variable(name, value);
    }
    symTab.add_constants(); // pi, epsilon, inf

    auto compile = [&](const std::string &exprStr) {
        auto expr = std::make_shared<exprtk::expression<Real>>();
        expr->register_symbol_table(symTab);
        exprtk::parser<Real> parser;
        if (!parser.compile(exprStr, *expr)) {
            throw std::runtime_error(
                "applyExprInitialCondition: failed to compile '" + exprStr +
                "': " + parser.error());
        }
        return expr;
    };

    const auto exRho = compile(rhoExpr);
    const auto exU   = compile(uExpr);
    const auto exV   = compile(vExpr);
    const auto exP   = compile(pExpr);

    const int N_elem = field.numElements();
    const int N_vars = field.numVars();
    const int Nq2    = field.numQuadPoints();

    const MatrixX2r xy = quadraturePhysicalCoords(field);
    VectorXr nodal(static_cast<Eigen::Index>(N_elem * N_vars * Nq2));
    for (int e = 0; e < N_elem; ++e) {
        for (int q = 0; q < Nq2; ++q) {
            x              = xy(static_cast<Eigen::Index>(e) * Nq2 + q, 0);
            y              = xy(static_cast<Eigen::Index>(e) * Nq2 + q, 1);
            const Real rho = exRho->value();
            const Real uu  = exU->value();
            const Real vv  = exV->value();
            const Real p   = exP->value();
            const Real E =
                p / (gam - Real(1)) + Real(0.5) * rho * (uu * uu + vv * vv);
            const Eigen::Index base =
                (static_cast<Eigen::Index>(e) * N_vars + 0) * Nq2 + q;
            nodal[base]           = rho;
            nodal[base + Nq2]     = rho * uu;
            nodal[base + 2 * Nq2] = rho * vv;
            nodal[base + 3 * Nq2] = E;
        }
    }

    field.setInitialConditionNodal(field.o_u(), nodal);
}
} // namespace

namespace solver_detail {
void applyExprICFromConfig(DgField &field, const Config &cfg) {
    applyExprInitialCondition(field, cfg.gamma(), cfg.icRho(), cfg.icU(),
                              cfg.icV(), cfg.icP(), cfg.icSymbols());
}
} // namespace solver_detail
