/// @file test_bc.cpp
/// @brief Boundary-condition integration tests: the slip (reflection) wall
///        and the expression (Dirichlet) boundary.
///
/// Cases:
///   1. Slip-wall neutrality: a uniform flow parallel to slip walls (y_low /
///      y_high) with free ends (x_low / x_high) must stay exactly constant —
///      the mirrored ghost produces zero wall-normal mass flux and the
///      tangential flux is the physical one of the constant state.
///   2. Expression-Dirichlet neutrality: a uniform field with constant [bc]
///      expressions on every side must stay exactly constant (evaluator +
///      ghost upload + kernel path).
///   3. Analytic-reference neutrality (riemann_target = "expr"):
///      uniform subsonic flow, all walls riemann with constant-valued
///      analytic expressions — the reference collapses onto the trace and
///      the field stays exactly constant.
///   4. Free-avg neutrality: uniform subsonic flow, all walls free-avg —
///      the constant mode of a constant field is the field itself, so the
///      ghost collapses onto the trace and the field stays constant.
///   5. Double-Mach reflection smoke: the Woodward & Colella setup (Mach 10
///      oblique shock, post-shock inflow left, moving-shock Dirichlet top,
///      split floor: post-shock inflow ahead of the shock foot / slip wall
///      behind it) on a coarse grid over a short time — completes with
///      positive density and pressure.

#include <Eigen/Dense>
#include <occa.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "basis/BasisFunctions2D.hpp"
#include "common/Types.hpp"
#include "config/Config.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "dg/DgField.hpp"
#include "mesh/MeshGeometry.hpp"
#include "solver/CompressibleFlowSolver.hpp"

namespace {

constexpr bool kSinglePrecision = std::is_same<Real, float>::value;
/// Uniform-state drift tolerance: round-off at double precision, a few
/// ulps of the state scale at single precision.
constexpr Real kUniformTol = kSinglePrecision ? Real(1e-4) : Real(1e-12);

int g_failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        ++g_failures;
        std::cout << "  [FAIL] " << what << "\n";
    } else {
        std::cout << "  [ok]   " << what << "\n";
    }
}

/// @brief Read the modal coefficients back to the host.
std::vector<Real> readModal(DgField &field, DeviceMemoryManager &mem) {
    const int n = field.numElements() * field.numVars() * field.numModes();
    std::vector<Real> u(static_cast<std::size_t>(n));
    occa::memory ou = field.o_u();
    if (mem.hasSeparateMemorySpace()) {
        mem.copyToHost(ou, u.data(), n);
    } else {
        const Real *src = ou.ptr<Real>();
        std::memcpy(u.data(), src, static_cast<std::size_t>(n) * sizeof(Real));
    }
    return u;
}

/// @brief Run a toml config through the solver; false if the run fails.
bool runSolver(const std::string &toml,
               CompressibleFlowSolver **out = nullptr) {
    const std::string path = "/tmp/cmeles_test_bc.toml";
    std::FILE *f           = std::fopen(path.c_str(), "w");
    std::fputs(toml.c_str(), f);
    std::fclose(f);
    const Config cfg(path);
    std::remove(path.c_str());

    static std::unique_ptr<CompressibleFlowSolver> solver;
    solver = std::make_unique<CompressibleFlowSolver>(cfg);
    if (solver->run() != 0) {
        return false;
    }
    if (out != nullptr) {
        *out = solver.get();
    }
    return true;
}

// ----------------------------------------------------------------------------
// 1. Slip-wall neutrality on a uniform parallel flow
// ----------------------------------------------------------------------------

bool testSlipWallUniform() {
    std::cout << "Slip wall: uniform parallel flow\n";
    // Uniform flow (rho, u, v, p) = (1.2, 0.7, 0, 1.3); y walls slip (the
    // flow is tangential to them), x walls free (constant state, neutral).
    const std::string toml =
        "[basis]\norder = 3\nnq = 4\n\n[mesh]\nnx = 12\nny = 12\nx0 = 0.0\n"
        "y0 = 0.0\ndx = 0.08\ndy = 0.08\n\n[bc]\ntype = \"free\"\n\n"
        "[bc.y_low]\ntype = \"slip\"\n\n[bc.y_high]\ntype = \"slip\"\n\n"
        "[flow]\nrho = 1.2\nu = 0.7\nv = 0.0\np = 1.3\n\n[time_marching]\n"
        "method = \"ssprk3\"\ncfl = 0.2\nt_final = 0.05\n";
    CompressibleFlowSolver *solver = nullptr;
    if (!runSolver(toml, &solver)) {
        check(false, "run completes");
        return false;
    }
    check(true, "run completes");

    DgField &field            = solver->field();
    const std::vector<Real> u = readModal(field, solver->mem());
    // Expected modal coefficients: the constant mode of the orthonormal
    // basis is phi_0 = 1/2 (integral 2 over the reference element), so a
    // constant field c has modal [2c, 0, 0, ...].
    const Real cons[4] = {Real(1.2), Real(1.2) * Real(0.7), Real(0),
                          Real(1.3) / Real(0.4) +
                              Real(0.5) * Real(1.2) * Real(0.49)};
    Real err           = Real(0);
    for (int e = 0; e < field.numElements(); ++e) {
        for (int var = 0; var < 4; ++var) {
            const Real *c = u.data() + (static_cast<std::size_t>(e) * 4 + var) *
                                           field.numModes();
            err           = std::max(err, std::abs(c[0] - 2 * cons[var]));
            for (int m = 1; m < field.numModes(); ++m) {
                err = std::max(err, std::abs(c[m]));
            }
        }
    }
    check(err <= kUniformTol,
          "field stays exactly constant (err " + std::to_string(err) + ")");
    return err <= kUniformTol;
}

// ----------------------------------------------------------------------------
// 2. Expression-Dirichlet neutrality on a uniform field
// ----------------------------------------------------------------------------

bool testExprBoundaryUniform() {
    std::cout << "Expression boundary: uniform Dirichlet\n";
    const std::string toml =
        "[basis]\norder = 3\nnq = 4\n\n[mesh]\nnx = 12\nny = 12\nx0 = 0.0\n"
        "y0 = 0.0\ndx = 0.08\ndy = 0.08\n\n[bc]\ntype = \"expr\"\n"
        "rho = \"1.1\"\nu = \"0.0\"\nv = \"0.5\"\np = \"0.9\"\n\n"
        "[flow]\nrho = 1.1\nu = 0.0\nv = 0.5\np = 0.9\n\n[time_marching]\n"
        "method = \"ssprk3\"\ncfl = 0.2\nt_final = 0.05\n";
    CompressibleFlowSolver *solver = nullptr;
    if (!runSolver(toml, &solver)) {
        check(false, "run completes");
        return false;
    }
    check(true, "run completes");

    DgField &field            = solver->field();
    const std::vector<Real> u = readModal(field, solver->mem());
    // Constant field c -> modal [2c, 0, ...] (phi_0 = 1/2, see above).
    const Real cons[4] = {Real(1.1), Real(0), Real(1.1) * Real(0.5),
                          Real(0.9) / Real(0.4) +
                              Real(0.5) * Real(1.1) * Real(0.25)};
    Real err           = Real(0);
    for (int e = 0; e < field.numElements(); ++e) {
        for (int var = 0; var < 4; ++var) {
            const Real *c = u.data() + (static_cast<std::size_t>(e) * 4 + var) *
                                           field.numModes();
            err           = std::max(err, std::abs(c[0] - 2 * cons[var]));
            for (int m = 1; m < field.numModes(); ++m) {
                err = std::max(err, std::abs(c[m]));
            }
        }
    }
    check(err <= kUniformTol,
          "field stays exactly constant (err " + std::to_string(err) + ")");
    return err <= kUniformTol;
}

// ----------------------------------------------------------------------------
// 3. Double-Mach reflection smoke
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Analytic-reference (riemann_target = "expr") neutrality
// ----------------------------------------------------------------------------

bool testRiemannExprUniform() {
    std::cout << "Riemann expr target: uniform subsonic flow\n";
    // Uniform flow (rho, u, v, p) = (1.2, 0.7, 0.0, 1.3), all walls
    // riemann with the "expr" target whose analytic expressions evaluate
    // to the very same constants. The reference then collapses onto the
    // trace on every (subsonic) wall, so the field must stay exactly
    // constant.
    const std::string toml =
        "[basis]\norder = 3\nnq = 4\n\n[mesh]\nnx = 12\nny = 12\nx0 = 0.0\n"
        "y0 = 0.0\ndx = 0.08\ndy = 0.08\n\n[bc]\ntype = \"riemann\"\n"
        "riemann_target = \"expr\"\nrho = \"1.2\"\nu = \"0.7\"\n"
        "v = \"0.0\"\np = \"1.3\"\n\n[flow]\nrho = 1.2\nu = 0.7\nv = 0.0\n"
        "p = 1.3\n\n[time_marching]\nmethod = \"ssprk3\"\ncfl = 0.2\n"
        "t_final = 0.05\n";
    CompressibleFlowSolver *solver = nullptr;
    if (!runSolver(toml, &solver)) {
        check(false, "run completes");
        return false;
    }
    check(true, "run completes");

    DgField &field            = solver->field();
    const std::vector<Real> u = readModal(field, solver->mem());
    const Real cons[4]        = {Real(1.2), Real(1.2) * Real(0.7), Real(0),
                                 Real(1.3) / Real(0.4) +
                                     Real(0.5) * Real(1.2) * Real(0.49)};
    Real err                  = Real(0);
    for (int e = 0; e < field.numElements(); ++e) {
        for (int var = 0; var < 4; ++var) {
            const Real *c = u.data() + (static_cast<std::size_t>(e) * 4 + var) *
                                           field.numModes();
            err           = std::max(err, std::abs(c[0] - 2 * cons[var]));
            for (int m = 1; m < field.numModes(); ++m) {
                err = std::max(err, std::abs(c[m]));
            }
        }
    }
    check(err <= kUniformTol,
          "field stays exactly constant (err " + std::to_string(err) + ")");
    return err <= kUniformTol;
}

// ----------------------------------------------------------------------------
// Zero-gradient-from-cell-mean (free-avg) neutrality
// ----------------------------------------------------------------------------

bool testFreeAvgUniform() {
    std::cout << "Free-avg boundary: uniform subsonic flow\n";
    // Uniform flow (rho, u, v, p) = (1.2, 0.7, 0.0, 1.3), all walls
    // free-avg. The constant mode of a constant field IS the field, so
    // the ghost collapses onto the trace exactly like the free boundary
    // and the field must stay exactly constant.
    const std::string toml =
        "[basis]\norder = 3\nnq = 4\n\n[mesh]\nnx = 12\nny = 12\nx0 = 0.0\n"
        "y0 = 0.0\ndx = 0.08\ndy = 0.08\n\n[bc]\ntype = \"free-avg\"\n\n"
        "[flow]\nrho = 1.2\nu = 0.7\nv = 0.0\n"
        "p = 1.3\n\n[time_marching]\nmethod = \"ssprk3\"\ncfl = 0.2\n"
        "t_final = 0.05\n";
    CompressibleFlowSolver *solver = nullptr;
    if (!runSolver(toml, &solver)) {
        check(false, "run completes");
        return false;
    }
    check(true, "run completes");

    DgField &field            = solver->field();
    const std::vector<Real> u = readModal(field, solver->mem());
    const Real cons[4]        = {Real(1.2), Real(1.2) * Real(0.7), Real(0),
                                 Real(1.3) / Real(0.4) +
                                     Real(0.5) * Real(1.2) * Real(0.49)};
    Real err                  = Real(0);
    for (int e = 0; e < field.numElements(); ++e) {
        for (int var = 0; var < 4; ++var) {
            const Real *c = u.data() + (static_cast<std::size_t>(e) * 4 + var) *
                                           field.numModes();
            err           = std::max(err, std::abs(c[0] - 2 * cons[var]));
            for (int m = 1; m < field.numModes(); ++m) {
                err = std::max(err, std::abs(c[m]));
            }
        }
    }
    check(err <= kUniformTol,
          "field stays exactly constant (err " + std::to_string(err) + ")");
    return err <= kUniformTol;
}

/// @brief Double-Mach data: post-shock (left of the front) / pre-shock
///        states of the Mach 10 oblique shock. The front makes 60 degrees
///        with the x-axis, x_s = 1/6 + y*tan(pi/6) = 1/6 + y/sqrt(3), and
///        translates rigidly at 10*(cos(pi/6), -sin(pi/6)); along any
///        fixed horizontal line the tilted front sweeps at
///        10/sin(pi/3) = 20/sqrt(3) (translation + front-slope
///        contribution of the downward drift), so the boundary
///        expressions classify a point by
///        $$ x - 1/6 - \frac{20 t}{\sqrt{3}} < \frac{y}{\sqrt{3}}. $$
///        The post-shock velocity 8.25*(cos(pi/6), -sin(pi/6)) is
///        perpendicular to the front (normal-shock consistency).
bool testDoubleMachSmoke() {
    std::cout << "Double Mach reflection smoke\n";
    const std::string front   = "(x - 1/6 - 20*t/sqrt(3)) < y/sqrt(3)";
    const std::string icFront = "(x - 1/6) < y/sqrt(3)";
    const std::string toml =
        std::string(
            "[basis]\norder = 2\nnq = 3\n\n[mesh]\nnx = 96\nny = 24\n") +
        "x0 = 0.0\ny0 = 0.0\ndx = 0.041666666666666664\ndy = "
        "0.041666666666666664\n\n[bc]\ntype = \"free\"\n"
        "rho = \"" +
        front + " ? 8.0 : 1.4\"\n" + "u = \"" + front +
        " ? 8.25*cos(pi/6) : 0.0\"\n" + "v = \"" + front +
        " ? -8.25*sin(pi/6) : 0.0\"\n" + "p = \"" + front +
        " ? 116.5 : 1.0\"\n\n" +
        "[bc.x_low]\ntype = \"expr\"\n\n"
        "[bc.y_high]\ntype = \"expr\"\n\n"
        "[bc.y_low]\ntype = \"slip\"\nsplit_x = 0.16666666666666666\n"
        "split_type = \"expr\"\n\n"
        "[flow]\nrho = 1.4\nu = 0.0\nv = 0.0\np = 1.0\n\n"
        "[time_marching]\nmethod = \"ssprk3\"\ncfl = 0.1\nt_final = 0.02\n\n"
        "[initial_condition]\ntype = \"expr\"\n" +
        "rho = \"" + icFront + " ? 8.0 : 1.4\"\n" + "u = \"" + icFront +
        " ? 8.25*cos(pi/6) : 0.0\"\n" + "v = \"" + icFront +
        " ? -8.25*sin(pi/6) : 0.0\"\n" + "p = \"" + icFront +
        " ? 116.5 : 1.0\"\n\n"
        "[limiter]\ntype = \"mr-weno\"\n";
    CompressibleFlowSolver *solver = nullptr;
    if (!runSolver(toml, &solver)) {
        check(false, "run completes");
        return false;
    }
    check(true, "run completes");

    DgField &field            = solver->field();
    const std::vector<Real> u = readModal(field, solver->mem());
    const int N_elem          = field.numElements();
    const int N_modes         = field.numModes();
    const int Nq2             = field.numQuadPoints();
    const MatrixXr &V         = field.basis().vandermonde();

    Real rhoMin = Real(1e300), rhoMax = Real(-1e300), pMin = Real(1e300);
    for (int e = 0; e < N_elem; ++e) {
        VectorXr qn[4];
        for (int var = 0; var < 4; ++var) {
            Eigen::Map<const VectorXr> uh(
                u.data() + (static_cast<std::size_t>(e) * 4 + var) * N_modes,
                N_modes);
            qn[var] = V * uh;
        }
        for (int q = 0; q < Nq2; ++q) {
            const Real rho = qn[0](q);
            const Real uu  = qn[1](q) / rho;
            const Real vv  = qn[2](q) / rho;
            const Real p   = (Real(1.4) - Real(1)) *
                             (qn[3](q) - Real(0.5) * rho * (uu * uu + vv * vv));
            rhoMin         = std::min(rhoMin, rho);
            rhoMax         = std::max(rhoMax, rho);
            pMin           = std::min(pMin, p);
        }
    }
    // By t = 0.02 the incident shock has just begun interacting with the
    // wall; states stay bracketed by [pre-shock undershoot, post-shock
    // compression + mild reflection overshoot].
    check(rhoMin > Real(0.5) && pMin > Real(0.05),
          "density/pressure positive (min rho " + std::to_string(rhoMin) +
              ", min p " + std::to_string(pMin) + ")");
    check(rhoMax < Real(40.0),
          "no wall pile-up (max rho " + std::to_string(rhoMax) + ")");
    return rhoMin > Real(0.5) && pMin > Real(0.05) && rhoMax < Real(40.0);
}

} // namespace

int main() {
    bool ok = true;
    ok &= testSlipWallUniform();
    ok &= testExprBoundaryUniform();
    ok &= testRiemannExprUniform();
    ok &= testFreeAvgUniform();
    ok &= testDoubleMachSmoke();
    std::cout << (ok ? "ALL BC TESTS PASSED\n" : "BC TESTS FAILED\n");
    return ok ? 0 : 1;
}
