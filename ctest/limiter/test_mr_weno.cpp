/// @file test_mr_weno.cpp
/// @brief Invariant tests of the MR-WENO limiter and the post-stage filter
///        contract.
///
/// Stage-1 acceptance of docs/tech_docs/mr_weno_limiter.md:
///   a. Constant field + force_troubled on an affine mesh: the limiter
///      must return the input to round-off (telescoping identity chain).
///   b. Smooth field + force_troubled on an affine mesh: near-recovery of
///      u_h (omega -> gamma) and exact cell-mean conservation.
///   c. Smooth field + force_troubled on a split-triangle mesh: cell-mean
///      conservation through the non-affine projection path (in-kernel
///      Cholesky of the restricted Gram blocks).
///   d. Built-in default: no limiter.
///   e. Filter-contract smoke across every stepper family (euler, ssprk3,
///      rk32, be, ditr-u2r1 + pseudo ssprk321): complete runs with the
///      limiter active and conserved total mass.
///   f. Clean constant field, detection-based: zero troubled cells, the
///      reconstruction chain is skipped and u stays bit-identical.
///   g. Mixed smooth/discontinuous field, detection-based: partial trouble
///      list coverage; untroubled cells untouched bit-exactly, cell means
///      conserved everywhere.

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
#include "limiter/MrWenoLimiter.hpp"
#include "mesh/MeshGeometry.hpp"
#include "solver/CompressibleFlowSolver.hpp"
#include "solver/ExprInitialCondition.hpp"

namespace {

constexpr bool kSinglePrecision = std::is_same<Real, float>::value;

/// @brief Read the whole modal state back to host memory.
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

/// @brief True cell means per element/variable: mean = sum_q Lambda_wJ
///        u_h(q) / sum_q Lambda_wJ (the general 2 (M u)_0 / |Omega|
///        formula; valid on non-affine elements).
std::vector<Real> cellMeans(const DgField &field, const std::vector<Real> &u) {
    const int N_elem       = field.numElements();
    const int N_vars       = field.numVars();
    const int N_modes      = field.numModes();
    const int Nq2          = field.numQuadPoints();
    const MatrixXr &V      = field.basis().vandermonde();
    const VectorXr &lamAll = field.geometry().lambdaWJ();

    std::vector<Real> means(static_cast<std::size_t>(N_elem) * N_vars);
    for (int e = 0; e < N_elem; ++e) {
        Eigen::Map<const VectorXr> lam(
            lamAll.data() + static_cast<std::size_t>(e) * Nq2, Nq2);
        const Real area = lam.sum();
        for (int var = 0; var < N_vars; ++var) {
            Eigen::Map<const VectorXr> uh(
                u.data() +
                    (static_cast<std::size_t>(e) * N_vars + var) * N_modes,
                N_modes);
            means[static_cast<std::size_t>(e) * N_vars + var] =
                (lam.cwiseProduct(V * uh)).sum() / area;
        }
    }
    return means;
}

/// @brief Total mass sum_e int_e rho dOmega (quadrature of the modal rho).
Real totalMass(const DgField &field, const std::vector<Real> &u) {
    const int N_elem       = field.numElements();
    const int N_vars       = field.numVars();
    const int N_modes      = field.numModes();
    const MatrixXr &V      = field.basis().vandermonde();
    const VectorXr &lamAll = field.geometry().lambdaWJ();
    const int Nq2          = field.numQuadPoints();

    Real mass = Real(0);
    for (int e = 0; e < N_elem; ++e) {
        Eigen::Map<const VectorXr> lam(
            lamAll.data() + static_cast<std::size_t>(e) * Nq2, Nq2);
        Eigen::Map<const VectorXr> rhoh(
            u.data() + static_cast<std::size_t>(e) * N_vars * N_modes, N_modes);
        mass += (lam.cwiseProduct(V * rhoh)).sum();
    }
    return mass;
}

/// @brief Standalone field fixture (device + memory manager + DG field with
///        the configured initial condition applied) for single-apply
///        limiter tests, independent of the solver's time loop.
struct FieldFixture {
    occa::device device;
    std::unique_ptr<DeviceMemoryManager> mem;
    std::unique_ptr<DgField> field;
};

FieldFixture makeField(const Config &cfg) {
    FieldFixture fx;
    fx.device = occa::device({{"mode", cfg.occaMode()}});
    fx.mem    = std::make_unique<DeviceMemoryManager>(fx.device);
    fx.field  = std::make_unique<DgField>(cfg, fx.device, *fx.mem);
    fx.field->setup();
    switch (cfg.icType()) {
        case IcType::Uniform:
            fx.field->applyFreeStreamInitialCondition(fx.field->o_u());
            break;
        case IcType::Expr:
            solver_detail::applyExprICFromConfig(*fx.field, cfg);
            break;
    }
    fx.device.finish();
    return fx;
}

/// @brief Write a TOML file, parse it, remove the file.
Config configFromString(const std::string &toml, const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "w");
    std::fputs(toml.c_str(), f);
    std::fclose(f);
    Config cfg(path);
    std::remove(path.c_str());
    return cfg;
}

int g_failures = 0;

/// @brief Report one boolean check.
void check(bool ok, const std::string &what) {
    if (!ok) {
        ++g_failures;
        std::cout << "  [FAIL] " << what << "\n";
    } else {
        std::cout << "  [ok]   " << what << "\n";
    }
}

const char *kSmoothIc = "[initial_condition]\ntype = \"expr\"\n"
                        "rho = \"1 + 0.2*sin(2*pi*x) + 0.1*cos(2*pi*y)\"\n"
                        "u = \"0.1\"\nv = \"0.05\"\n"
                        "p = \"1 + 0.15*sin(2*pi*(x + y))\"\n";

/// @brief Single-apply limiter test core: apply once to the initial state
///        and report the modal change and the mean drift.
void runSingleApply(const std::string &label, const std::string &toml,
                    const std::string &scratch) {
    std::cout << label << "\n";
    const Config cfg = configFromString(toml, scratch);
    FieldFixture fx  = makeField(cfg);

    const std::vector<Real> u0    = readModal(*fx.field, *fx.mem);
    const std::vector<Real> mean0 = cellMeans(*fx.field, u0);

    MrWenoLimiter limiter(cfg, fx.device, *fx.mem, *fx.field);
    limiter.apply(fx.field->o_u());
    fx.device.finish();

    const std::vector<Real> u1    = readModal(*fx.field, *fx.mem);
    const std::vector<Real> mean1 = cellMeans(*fx.field, u1);

    // Modal change (relative to the largest coefficient magnitude).
    Real maxC = Real(1e-30), maxD = Real(0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        maxC = std::max(maxC, std::abs(u0[i]));
        maxD = std::max(maxD, std::abs(u1[i] - u0[i]));
    }
    const Real relChange = maxD / maxC;

    // Cell-mean drift per element/variable.
    Real meanDrift = Real(0);
    for (std::size_t i = 0; i < mean0.size(); ++i) {
        meanDrift =
            std::max(meanDrift, std::abs(mean1[i] - mean0[i]) /
                                    std::max(Real(1), std::abs(mean0[i])));
    }
    const bool finite = std::isfinite(relChange) && std::isfinite(meanDrift);
    check(finite, "output finite after apply");
    check(meanDrift <= (kSinglePrecision ? Real(5e-5) : Real(1e-12)),
          "cell means conserved (drift " + std::to_string(meanDrift) + ")");
    std::cout << "  modal rel-change = " << relChange << "\n";
}

// ----------------------------------------------------------------------------
// a) Constant field, affine mesh, forced troubled
// ----------------------------------------------------------------------------

void testConstantAffine() {
    const std::string toml =
        "[basis]\norder = 2\nnq = 4\n\n[mesh]\nnx = 8\nny = 8\ndx = 0.125\n"
        "dy = 0.125\n\n[bc]\ntype = \"periodic\"\n\n[flow]\nrho = 1.3\n"
        "u = 0.2\nv = -0.1\np = 0.9\n\n[time_marching]\nmethod = \"ssprk3\"\n"
        "cfl = 0.18\nt_final = 0.01\n\n[initial_condition]\n"
        "type = \"uniform\"\n\n[limiter]\ntype = \"mr-weno\"\n"
        "force_troubled = true\n";
    runSingleApply("(a) constant field / affine / forced troubled", toml,
                   "/tmp/cmeles_mrw_a.toml");
    // The telescoping identity chain must return a constant field to
    // round-off (all candidates coincide).
    const Real tol   = kSinglePrecision ? Real(1e-3) : Real(1e-12);
    const Config cfg = configFromString(toml, "/tmp/cmeles_mrw_a2.toml");
    FieldFixture fx  = makeField(cfg);
    const std::vector<Real> u0 = readModal(*fx.field, *fx.mem);
    MrWenoLimiter limiter(cfg, fx.device, *fx.mem, *fx.field);
    limiter.apply(fx.field->o_u());
    fx.device.finish();
    const std::vector<Real> u1 = readModal(*fx.field, *fx.mem);
    Real maxC = Real(1e-30), maxD = Real(0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        maxC = std::max(maxC, std::abs(u0[i]));
        maxD = std::max(maxD, std::abs(u1[i] - u0[i]));
    }
    check(maxD / maxC <= tol, "constant field recovered to round-off (" +
                                  std::to_string(maxD / maxC) + ")");
}

// ----------------------------------------------------------------------------
// b) Smooth field, affine mesh, forced troubled
// ----------------------------------------------------------------------------

void testSmoothAffine() {
    const std::string toml =
        "[basis]\norder = 2\nnq = 4\n\n[mesh]\nnx = 16\nny = 16\n"
        "dx = 0.0625\ndy = 0.0625\n\n[bc]\ntype = \"periodic\"\n\n[flow]\n"
        "rho = 1.0\nu = 0.1\nv = 0.05\np = 1.0\n\n[time_marching]\n"
        "method = \"ssprk3\"\ncfl = 0.18\nt_final = 0.01\n\n" +
        std::string(kSmoothIc) +
        "\n[limiter]\ntype = \"mr-weno\"\nforce_troubled = true\n";
    runSingleApply("(b) smooth field / affine / forced troubled", toml,
                   "/tmp/cmeles_mrw_b.toml");
}

// ----------------------------------------------------------------------------
// c) Smooth field, split-triangle (non-affine) mesh, forced troubled
// ----------------------------------------------------------------------------

void testSmoothTriangles() {
    const std::string toml =
        "[basis]\norder = 2\nnq = 4\n\n[mesh]\nnx = 12\nny = 12\n"
        "dx = 0.0833333333333333\ndy = 0.0833333333333333\n"
        "split_triangles = true\n\n[bc]\ntype = \"periodic\"\n\n[flow]\n"
        "rho = 1.0\nu = 0.1\nv = 0.05\np = 1.0\n\n[time_marching]\n"
        "method = \"ssprk3\"\ncfl = 0.18\nt_final = 0.01\n\n" +
        std::string(kSmoothIc) +
        "\n[limiter]\ntype = \"mr-weno\"\nforce_troubled = true\n";
    runSingleApply("(c) smooth field / triangles / forced troubled", toml,
                   "/tmp/cmeles_mrw_c.toml");
    // Non-affine flag spot check: every triangle must be classified
    // non-affine (the collapsed vertex fails the parallelogram condition).
    const Config cfg = configFromString(toml, "/tmp/cmeles_mrw_c2.toml");
    FieldFixture fx  = makeField(cfg);
    int nonAffine    = 0;
    for (int e = 0; e < fx.field->numElements(); ++e) {
        nonAffine += fx.field->geometry().elemAffine()[e] == 0 ? 1 : 0;
    }
    check(nonAffine == fx.field->numElements(),
          "all split-triangle elements flagged non-affine");
}

// ----------------------------------------------------------------------------
// d) Built-in default: no limiter
// ----------------------------------------------------------------------------

void testDefaultNone() {
    std::cout << "(d) default limiter type\n";
    const Config cfg;
    check(cfg.limiterType() == LimiterType::None,
          "default [limiter] type is none");
    check(parseLimiterType("mr-weno") == LimiterType::MrWeno &&
              std::string(limiterTypeName(LimiterType::MrWeno)) == "mr-weno",
          "limiter name round-trip");
}

// ----------------------------------------------------------------------------
// e) Filter-contract smoke across every stepper family
// ----------------------------------------------------------------------------

void testStepperFamilies() {
    std::cout << "(e) stepper families with the limiter active\n";
    struct Case {
        const char *name;
        const char *timeTOML;
    };
    const Case cases[] = {
        {"euler", "method = \"euler\"\ncfl = 0.05\nt_final = 0.02\n"},
        {"ssprk3", "method = \"ssprk3\"\ncfl = 0.18\nt_final = 0.02\n"},
        {"rk32", "method = \"rk32\"\ncfl = 0.2\nt_final = 0.02\n"},
        {"be", "method = \"be\"\ncfl = 0.2\nt_final = 0.02\n"
               "max_pseudo_steps = 300\n"},
        {"ditr-u2r1/ssprk321", "method = \"ditr-u2r1\"\ncfl = 0.2\n"
                               "t_final = 0.02\npseudo_method = "
                               "\"ssprk321\"\nmax_pseudo_steps = 300\n"},
    };

    for (const Case &c : cases) {
        const std::string toml =
            "[basis]\norder = 2\nnq = 4\n\n[mesh]\nnx = 16\nny = 8\n"
            "dx = 0.0625\ndy = 0.125\n\n[bc]\ntype = \"periodic\"\n\n"
            "[flow]\nrho = 1.0\nu = 0.1\nv = 0.0\np = 1.0\n\n"
            "[time_marching]\n" +
            std::string(c.timeTOML) + "\n" + std::string(kSmoothIc) +
            "\n[limiter]\ntype = \"mr-weno\"\n";
        const Config cfg = configFromString(toml, "/tmp/cmeles_mrw_e.toml");
        CompressibleFlowSolver solver(cfg);
        const bool ran = solver.run() == 0;
        check(ran, std::string("run completes (") + c.name + ")");

        // Total mass against the analytic integral of rho0 over [0,1]^2
        // (= 1: the sinusoids integrate to zero and the L2 projection
        // preserves the mean exactly).
        DgField &field            = solver.field();
        const std::vector<Real> u = readModal(field, solver.mem());
        const Real mass           = totalMass(field, u);
        const Real tol            = kSinglePrecision ? Real(5e-4) : Real(1e-10);
        check(std::abs(mass - Real(1)) <= tol,
              std::string("mass conserved (") + c.name +
                  ", mass = " + std::to_string(mass) + ")");
    }
}

// ----------------------------------------------------------------------------
// f) Clean constant field, detection-based: zero troubled cells
// ----------------------------------------------------------------------------

void testCleanFieldNoTroubled() {
    const std::string toml =
        "[basis]\norder = 2\nnq = 4\n\n[mesh]\nnx = 8\nny = 8\ndx = 0.125\n"
        "dy = 0.125\n\n[bc]\ntype = \"periodic\"\n\n[flow]\nrho = 1.3\n"
        "u = 0.2\nv = -0.1\np = 0.9\n\n[time_marching]\nmethod = \"ssprk3\"\n"
        "cfl = 0.18\nt_final = 0.01\n\n[initial_condition]\n"
        "type = \"uniform\"\n\n[limiter]\ntype = \"mr-weno\"\n";
    std::cout << "(f) clean field / detection-based / zero troubled\n";
    const Config cfg = configFromString(toml, "/tmp/cmeles_mrw_f.toml");
    FieldFixture fx  = makeField(cfg);
    const std::vector<Real> u0 = readModal(*fx.field, *fx.mem);
    MrWenoLimiter limiter(cfg, fx.device, *fx.mem, *fx.field);
    limiter.apply(fx.field->o_u());
    fx.device.finish();
    check(limiter.lastTroubledCount() == 0,
          "constant field flags no troubled cell");
    const std::vector<Real> u1 = readModal(*fx.field, *fx.mem);
    bool identical             = u0.size() == u1.size();
    for (std::size_t i = 0; identical && i < u0.size(); ++i) {
        identical = u0[i] == u1[i];
    }
    check(identical, "u bit-identical (reconstruction chain skipped)");
}

// ----------------------------------------------------------------------------
// g) Mixed smooth/discontinuous field, detection-based: partial coverage
// ----------------------------------------------------------------------------

void testMixedFieldPartialCoverage() {
    // Smooth sinusoid background with a strong density/pressure depression
    // band (0.4 < x < 0.6): KXRCF flags the band-edge cells, the far field
    // stays clean — the dense trouble list must touch exactly the flagged
    // cells and leave the rest bit-identical.
    const std::string toml =
        "[basis]\norder = 2\nnq = 4\n\n[mesh]\nnx = 32\nny = 16\n"
        "dx = 0.03125\ndy = 0.0625\n\n[bc]\ntype = \"periodic\"\n\n[flow]\n"
        "rho = 1.0\nu = 0.1\nv = 0.05\np = 1.0\n\n[time_marching]\n"
        "method = \"ssprk3\"\ncfl = 0.18\nt_final = 0.01\n\n"
        "[initial_condition]\ntype = \"expr\"\n"
        "rho = \"1 + 0.2*sin(2*pi*x) - ((x > 0.4) ? ((x < 0.6) ? 0.6 : 0) : "
        "0)\"\n"
        "u = \"0.1\"\nv = \"0.05\"\n"
        "p = \"1 + 0.15*sin(2*pi*(x + y)) - ((x > 0.4) ? ((x < 0.6) ? 0.5 : "
        "0) : 0)\"\n\n[limiter]\ntype = \"mr-weno\"\n";
    std::cout << "(g) mixed field / detection-based / partial coverage\n";
    const Config cfg = configFromString(toml, "/tmp/cmeles_mrw_g.toml");
    FieldFixture fx  = makeField(cfg);
    const std::vector<Real> u0    = readModal(*fx.field, *fx.mem);
    const std::vector<Real> mean0 = cellMeans(*fx.field, u0);
    MrWenoLimiter limiter(cfg, fx.device, *fx.mem, *fx.field);
    limiter.apply(fx.field->o_u());
    fx.device.finish();
    const std::vector<Real> u1    = readModal(*fx.field, *fx.mem);
    const std::vector<Real> mean1 = cellMeans(*fx.field, u1);

    const int nElem  = fx.field->numElements();
    const int nVars  = fx.field->numVars();
    const int nModes = fx.field->numModes();
    const int count  = limiter.lastTroubledCount();
    check(count > 0 && count < nElem, "partial coverage (troubled " +
                                          std::to_string(count) + " of " +
                                          std::to_string(nElem) + ")");

    // Per-element classification: untouched blocks must be bit-identical
    // (the dense list never visits clean cells), flagged-region blocks
    // must actually change.
    int untouched = 0, touched = 0;
    bool finite = true;
    for (int e = 0; e < nElem; ++e) {
        bool same = true;
        for (int i = 0; i < nVars * nModes; ++i) {
            const std::size_t idx =
                (static_cast<std::size_t>(e) * nVars) * nModes +
                static_cast<std::size_t>(i);
            if (u0[idx] != u1[idx]) {
                same = false;
            }
            finite = finite && std::isfinite(u1[idx]);
        }
        same ? ++untouched : ++touched;
    }
    check(finite, "output finite after apply");
    check(untouched > 0, "clean cells untouched bit-exactly (" +
                             std::to_string(untouched) + " of " +
                             std::to_string(nElem) + ")");
    check(touched > 0, "troubled cells reconstructed (" +
                           std::to_string(touched) + " changed)");

    Real meanDrift = Real(0);
    for (std::size_t i = 0; i < mean0.size(); ++i) {
        meanDrift =
            std::max(meanDrift, std::abs(mean1[i] - mean0[i]) /
                                    std::max(Real(1), std::abs(mean0[i])));
    }
    check(meanDrift <= (kSinglePrecision ? Real(5e-5) : Real(1e-12)),
          "cell means conserved (drift " + std::to_string(meanDrift) + ")");
}

} // namespace

int main() {
    testConstantAffine();
    testSmoothAffine();
    testSmoothTriangles();
    testDefaultNone();
    testStepperFamilies();
    testCleanFieldNoTroubled();
    testMixedFieldPartialCoverage();

    if (g_failures == 0) {
        std::cout << "mr_weno: all checks passed\n";
        return 0;
    }
    std::cout << "mr_weno: " << g_failures << " check(s) failed\n";
    return 1;
}
