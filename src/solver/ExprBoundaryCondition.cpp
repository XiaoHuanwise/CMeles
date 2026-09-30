/// @file ExprBoundaryCondition.cpp
/// @brief Implementation of the expression-evaluated Dirichlet boundary.
///
/// exprtk.hpp is included only in this translation unit to isolate its
/// compilation cost from the rest of the project (same policy as
/// ExprInitialCondition.cpp).

#include "ExprBoundaryCondition.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <exprtk.hpp>

#include "config/Config.hpp"
#include "dg/DgField.hpp"

namespace {

/// @brief exprtk state shared by the four compiled expressions.
struct ExprState {
    // Symbol storage must outlive the expressions (exprtk variables are
    // references into this state).
    Real x   = 0;
    Real y   = 0;
    Real t   = 0;
    Real gam = Real(1.4);

    exprtk::symbol_table<Real> symTab;
    std::array<std::shared_ptr<exprtk::expression<Real>>, 4> ex; // rho,u,v,p
};

} // namespace

namespace solver_detail {

struct ExprBoundaryCondition::State {
    ExprState expr;
};

ExprBoundaryCondition::ExprBoundaryCondition(const Config &cfg, DgField &field)
    : field_(field), state_(std::make_unique<State>()) {
    ExprState &st = state_->expr;

    st.gam = cfg.gamma();
    st.symTab.add_variable("x", st.x);
    st.symTab.add_variable("y", st.y);
    st.symTab.add_variable("t", st.t);
    st.symTab.add_variable("gamma", st.gam);
    st.symTab.add_constants(); // pi, epsilon, inf

    const std::string *sources[4] = {&cfg.bcRho(), &cfg.bcU(), &cfg.bcV(),
                                     &cfg.bcP()};
    const char *names[4]          = {"rho", "u", "v", "p"};
    for (int k = 0; k < 4; ++k) {
        auto expr = std::make_shared<exprtk::expression<Real>>();
        expr->register_symbol_table(st.symTab);
        exprtk::parser<Real> parser;
        if (!parser.compile(*sources[k], *expr)) {
            throw std::runtime_error(
                "ExprBoundaryCondition: failed to compile the [bc] '" +
                std::string(names[k]) + "' expression '" + *sources[k] +
                "': " + parser.error());
        }
        st.ex[static_cast<std::size_t>(k)] = std::move(expr);
    }
}

ExprBoundaryCondition::~ExprBoundaryCondition() = default;

void ExprBoundaryCondition::refresh(Real t) {
    ExprState &st = state_->expr;
    st.t          = t;

    const int nBf  = field_.numBoundaryFaces();
    const int Nq   = field_.numPoints1D();
    const Real gm1 = st.gam - Real(1);

    const std::vector<Real> &pts = field_.bcFacePoints();
    std::vector<Real> ghost(static_cast<std::size_t>(nBf) * Nq * 4, Real(0));
    for (int bf = 0; bf < nBf; ++bf) {
        for (int i = 0; i < Nq; ++i) {
            const std::size_t pt = (static_cast<std::size_t>(bf) * Nq +
                                    static_cast<std::size_t>(i)) *
                                   2;
            st.x                 = pts[pt + 0];
            st.y                 = pts[pt + 1];

            const Real rho = st.ex[0]->value();
            const Real uu  = st.ex[1]->value();
            const Real vv  = st.ex[2]->value();
            const Real p   = st.ex[3]->value();
            const Real E   = p / gm1 + Real(0.5) * rho * (uu * uu + vv * vv);

            Real *g = ghost.data() + (static_cast<std::size_t>(bf) * Nq +
                                      static_cast<std::size_t>(i)) *
                                         4;
            g[0]    = rho;
            g[1]    = rho * uu;
            g[2]    = rho * vv;
            g[3]    = E;
        }
    }
    field_.uploadBcGhost(ghost);
}

} // namespace solver_detail
