/// @file test_sod.cpp
/// @brief Two-dimensional Sod shock tube with the MR-WENO limiter:
///        end-to-end acceptance across the SSPRK3 and DITR-U2R1(+SSPRK321
///        pseudo) time-marching paths.
///
/// Setup: periodic domain [0, 2] x [0, 1] with the discontinuity at
/// x = 1.0, run to t = 0.2. The periodic seam (x = 0 = 2) carries a second
/// (mirrored) Sod Riemann problem; at t = 0.2 the two wave systems have
/// not yet interacted (they collide at t ~ 0.42), so the exact solution
/// everywhere is the superposition of the two independent Riemann
/// problems, computed here by an embedded exact Sod solver.
///
/// Acceptance criteria (per time-marching configuration):
///   - the run completes with a finite residual (the solver NaN-aborts
///     otherwise);
///   - total mass equals the analytic 1.125 to round-off (the limiter is
///     conservative and the periodic domain is closed);
///   - density and pressure are positive at all quadrature points;
///   - the density stays within physical bounds (non-oscillatory);
///   - the primary shock position matches the exact solution within two
///     cells;
///   - the L1 density error against the exact solution stays small.

#include <Eigen/Dense>
#include <occa.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
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
constexpr Real kGamma           = Real(1.4);

// Sod left/right states.
constexpr Real kRhoL = Real(1.0), kPL = Real(1.0);
constexpr Real kRhoR = Real(0.125), kPR = Real(0.1);

int g_failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        ++g_failures;
        std::cout << "  [FAIL] " << what << "\n";
    } else {
        std::cout << "  [ok]   " << what << "\n";
    }
}

// ----------------------------------------------------------------------------
// Exact 1-D Sod Riemann solution (primitive variables, self-similar
// coordinate xi = (x - x0) / t).
// ----------------------------------------------------------------------------

struct Prim {
    Real rho, u, p;
};

/// @brief f_K(p) of the two-shock/two-rarefaction Newton iteration (Toro,
///        ch. 4): exact shock branch, not the weak-shock linearisation.
Real fK(Real p, Real rhoK, Real pK) {
    if (p > pK) {
        return (p - pK) *
               std::sqrt(Real(2) /
                         ((kGamma + Real(1)) * rhoK *
                          (p + (kGamma - Real(1)) / (kGamma + Real(1)) * pK)));
    }
    const Real cK = std::sqrt(kGamma * pK / rhoK);
    return Real(2) * cK / (kGamma - Real(1)) *
           (std::pow(p / pK, (kGamma - Real(1)) / (Real(2) * kGamma)) -
            Real(1));
}

/// @brief Solve the star state of the Sod problem (Newton on p*).
void solveStar(Real &pStar, Real &uStar) {
    pStar = Real(0.3);
    for (int it = 0; it < 60; ++it) {
        const Real f  = fK(pStar, kRhoL, kPL) + fK(pStar, kRhoR, kPR);
        const Real dp = std::max(pStar * Real(1e-8), Real(1e-12));
        const Real fP = fK(pStar + dp, kRhoL, kPL) + fK(pStar + dp, kRhoR, kPR);
        const Real df = (fP - f) / dp;
        const Real step = f / df;
        pStar -= step;
        if (pStar <= Real(0)) {
            pStar = Real(0.01);
        }
        if (std::abs(step) < Real(1e-14) * pStar) {
            break;
        }
    }
    uStar = Real(0.5) * (fK(pStar, kRhoR, kPR) - fK(pStar, kRhoL, kPL));
}

/// @brief Exact primitive state at self-similar coordinate \p xi.
Prim exactSod(Real xi) {
    static Prim star = [] {
        Real ps, us;
        solveStar(ps, us);
        return Prim{Real(0), us, ps};
    }();
    const Real pStar = star.p;
    const Real uStar = star.u;

    const Real cL = std::sqrt(kGamma * kPL / kRhoL);
    const Real cR = std::sqrt(kGamma * kPR / kRhoR);
    // Left rarefaction: a*_L from the isentropic relation.
    const Real aStarL =
        cL * std::pow(pStar / kPL, (kGamma - Real(1)) / (Real(2) * kGamma));
    // Right shock speed (Toro): W = uR + cR * sqrt(...).
    const Real shockSpeed =
        cR * std::sqrt((kGamma + Real(1)) / (Real(2) * kGamma) * (pStar / kPR) +
                       (kGamma - Real(1)) / (Real(2) * kGamma));
    // Star densities.
    const Real rhoStarL = kRhoL * std::pow(pStar / kPL, Real(1) / kGamma);
    const Real rhoStarR =
        kRhoR * (pStar / kPR + (kGamma - Real(1)) / (kGamma + Real(1))) /
        ((kGamma - Real(1)) / (kGamma + Real(1)) * (pStar / kPR) + Real(1));

    if (xi <= -cL) {
        return Prim{kRhoL, Real(0), kPL};
    }
    if (xi <= uStar - aStarL) {
        // Inside the left rarefaction fan.
        const Real u   = Real(2) / (kGamma + Real(1)) * (cL + xi);
        const Real a   = cL - (kGamma - Real(1)) / Real(2) * u;
        const Real rho = kRhoL * std::pow(a / cL, Real(2) / (kGamma - Real(1)));
        const Real p =
            kPL * std::pow(a / cL, Real(2) * kGamma / (kGamma - Real(1)));
        return Prim{rho, u, p};
    }
    if (xi <= uStar) {
        return Prim{rhoStarL, uStar, pStar};
    }
    if (xi <= shockSpeed) {
        return Prim{rhoStarR, uStar, pStar};
    }
    return Prim{kRhoR, Real(0), kPR};
}

/// @brief Exact rho on the periodic ring at (x, t): the primary problem
///        centred at x = 1 and its mirror at the seam x = 0 = 2; before
///        wave interaction the nearest-centre assignment is consistent
///        (both give the same constant state in the overlap).
Real exactRhoRing(Real x, Real t) {
    if (x >= Real(0.5) && x <= Real(1.5)) {
        return exactSod((x - Real(1)) / t).rho;
    }
    const Real s = (x < Real(0.5)) ? x : x - Real(2);
    return exactSod(-s / t).rho;
}

// ----------------------------------------------------------------------------
// Run helpers
// ----------------------------------------------------------------------------

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

/// @brief Write the Sod configuration (two wave systems: periodic ring of
///        length 2 with the primary discontinuity at x = 1).
Config makeSodConfig(const std::string &method, const std::string &pseudo,
                     int order, Real cfl, const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "w");
    std::fprintf(f,
                 "[basis]\norder = %d\nnq = 4\n\n[mesh]\nnx = 64\nny = 16\n"
                 "x0 = 0.0\ny0 = 0.0\ndx = %.12g\ndy = %.12g\n\n"
                 "[bc]\ntype = \"periodic\"\n\n[riemann]\nflux = \"llf\"\n\n"
                 "[gas]\ngamma = 1.4\n\n[flow]\nrho = 1.0\nu = 0.0\nv = 0.0\n"
                 "p = 1.0\n\n[time_marching]\nmethod = \"%s\"\ncfl = %.12g\n"
                 "t_final = 0.2\nmax_pseudo_steps = 400",
                 order, 2.0 / 64.0, 1.0 / 16.0, method.c_str(), cfl);
    if (!pseudo.empty()) {
        std::fprintf(f, "\npseudo_method = \"%s\"", pseudo.c_str());
    }
    std::fprintf(f, "\n\n[initial_condition]\ntype = \"expr\"\n"
                    "rho = \"x < 1.0 ? 1.0 : 0.125\"\nu = \"0\"\nv = \"0\"\n"
                    "p = \"x < 1.0 ? 1.0 : 0.1\"\n\n[limiter]\n"
                    "type = \"mr-weno\"\n\n[occa]\nmode = \"Serial\"\n");
    std::fclose(f);
    Config cfg(path);
    std::remove(path.c_str());
    return cfg;
}

/// @brief Run one Sod configuration and apply the acceptance checks.
/// \p pBoundLo is the pressure lower bound of the non-oscillatory check
/// (P2 recovers the physical range closely; P1 allows a wider
/// undershoot).
void runSodCase(const std::string &label, const std::string &method,
                const std::string &pseudo, int order, Real cfl, Real pBoundLo) {
    std::cout << label << "\n";
    const Config cfg =
        makeSodConfig(method, pseudo, order, cfl, "/tmp/cmeles_sod.toml");
    CompressibleFlowSolver solver(cfg);
    if (solver.run() != 0) {
        check(false, "run completes");
        return;
    }
    check(true, "run completes");

    DgField &field            = solver.field();
    const std::vector<Real> u = readModal(field, solver.mem());
    const int N_elem          = field.numElements();
    const int N_vars          = field.numVars();
    const int N_modes         = field.numModes();
    const int Nq2             = field.numQuadPoints();
    const MatrixXr &V         = field.basis().vandermonde();
    const VectorXr &lamAll    = field.geometry().lambdaWJ();
    const auto &e8            = field.geometry().vertices();

    const Real tEnd = cfg.timeFinal();

    // --- Quadrature-point states: positivity, bounds, and cell means.
    Real mass   = Real(0);
    Real rhoMin = Real(1e300), rhoMax = Real(-1e300);
    Real pMin = Real(1e300);
    std::vector<Real> cellRho(static_cast<std::size_t>(N_elem));
    std::vector<Real> xc(static_cast<std::size_t>(N_elem));
    for (int e = 0; e < N_elem; ++e) {
        Eigen::Map<const VectorXr> lam(
            lamAll.data() + static_cast<std::size_t>(e) * Nq2, Nq2);
        VectorXr qn[4];
        for (int var = 0; var < 4; ++var) {
            Eigen::Map<const VectorXr> uh(
                u.data() +
                    (static_cast<std::size_t>(e) * N_vars + var) * N_modes,
                N_modes);
            qn[var] = V * uh;
        }
        Real area = lam.sum();
        for (int q = 0; q < Nq2; ++q) {
            const Real rho = qn[0](q);
            const Real uu  = qn[1](q) / rho;
            const Real vv  = qn[2](q) / rho;
            const Real p   = (kGamma - Real(1)) *
                             (qn[3](q) - Real(0.5) * rho * (uu * uu + vv * vv));
            rhoMin         = std::min(rhoMin, rho);
            rhoMax         = std::max(rhoMax, rho);
            pMin           = std::min(pMin, p);
        }
        cellRho[e] = (lam.cwiseProduct(qn[0])).sum() / area;
        mass += (lam.cwiseProduct(qn[0])).sum();
        xc[e] = (e8(e, 0) + e8(e, 2) + e8(e, 4) + e8(e, 6)) / Real(4);
    }

    const Real massTol = kSinglePrecision ? Real(5e-4) : Real(1e-10);
    check(std::abs(mass - Real(1.125)) <= massTol,
          "total mass = 1.125 (got " + std::to_string(mass) + ")");
    check(rhoMin > Real(0) && pMin > Real(0),
          "density/pressure positive (min rho " + std::to_string(rhoMin) +
              ", min p " + std::to_string(pMin) + ")");
    check(rhoMin >= Real(0.10) && rhoMax <= Real(1.05),
          "density in [0.10, 1.05] (got [" + std::to_string(rhoMin) + ", " +
              std::to_string(rhoMax) + "])");
    check(pMin >= pBoundLo && pMin <= Real(1.05),
          "pressure non-oscillatory (min p " + std::to_string(pMin) + ")");

    // --- y-averaged density profile binned by element centroid column
    //     (centroids sit at (c + 0.5) h, so plain floor bins correctly).
    const Real h = Real(2) / Real(64);
    std::vector<Real> prof(64, Real(0));
    std::vector<int> cnt(64, 0);
    for (int e = 0; e < N_elem; ++e) {
        int col = static_cast<int>(std::floor(xc[e] / h + Real(1e-9)));
        col     = std::max(0, std::min(63, col));
        prof[col] += cellRho[e];
        ++cnt[col];
    }
    for (int c = 0; c < 64; ++c) {
        if (cnt[c] > 0) {
            prof[c] /= Real(cnt[c]);
        }
    }

    // --- Primary shock position: crossing of the mid value between the
    //     post-shock star state (rho*_R, right of the contact) and the
    //     undisturbed right state, searched in (1.2, 1.5); linear
    //     interpolation between the bracketing column centres. xi = 1.3
    //     samples the right star region (between u* = 0.927 and the shock
    //     speed 1.75).
    const Prim starR = exactSod(Real(1.3));
    const Real mid   = Real(0.5) * (starR.rho + kRhoR);
    const Real xExact =
        Real(1) + Real(0.2) *
                      std::sqrt((kGamma + Real(1)) / (Real(2) * kGamma) *
                                    (starR.p / kPR) +
                                (kGamma - Real(1)) / (Real(2) * kGamma)) *
                      std::sqrt(kGamma * kPR / kRhoR);
    Real xShock = Real(-1);
    for (int c = 0; c < 63; ++c) {
        const Real xc0 = (c + Real(0.5)) * h, xc1 = (c + Real(1.5)) * h;
        if (xc0 >= Real(1.2) && xc1 <= Real(1.5) && prof[c] >= mid &&
            prof[c + 1] < mid) {
            const Real fr = (prof[c] - mid) / (prof[c] - prof[c + 1]);
            xShock        = xc0 + fr * h;
            break;
        }
    }
    const Real posTol = Real(2) * h;
    check(xShock > Real(0) && std::abs(xShock - xExact) <= posTol,
          "shock position within 2 cells (exact " + std::to_string(xExact) +
              ", got " + std::to_string(xShock) + ")");

    // --- L1 density error against the exact ring solution (column
    //     centroids, area-weighted).
    Real l1 = Real(0);
    for (int c = 0; c < 64; ++c) {
        l1 += std::abs(prof[c] - exactRhoRing((c + Real(0.5)) * h, tEnd));
    }
    l1 *= h / Real(2);
    const Real l1Tol = kSinglePrecision ? Real(0.1) : Real(0.05);
    check(l1 <= l1Tol, "L1 density error <= " + std::to_string(l1Tol) +
                           " (got " + std::to_string(l1) + ")");
}

} // namespace

int main() {
    runSodCase("Sod / ssprk3 / P2", "ssprk3", "", 2, Real(0.18), Real(0.09));
    runSodCase("Sod / ssprk3 / P1", "ssprk3", "", 1, Real(0.3), Real(0.05));
    runSodCase("Sod / ditr-u2r1 + ssprk321 / P2", "ditr-u2r1", "ssprk321", 2,
               Real(0.5), Real(0.09));

    if (g_failures == 0) {
        std::cout << "sod: all checks passed\n";
        return 0;
    }
    std::cout << "sod: " << g_failures << " check(s) failed\n";
    return 1;
}
