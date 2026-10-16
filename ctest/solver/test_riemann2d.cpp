/// @file test_riemann2d.cpp
/// @brief Two-dimensional Riemann problem (Lax & Liu 1998 configuration 3,
///        ORIGINAL paper convention: split at (0.8, 0.8), T = 0.8) with the
///        MR-WENO limiter under the analytic-reference characteristic
///        zero-gradient-from-cell-mean boundary condition (free-avg).
///
/// Setup: [0, 1]^2 with the four-quadrant initial data
///   LL = (0.138,  1.206, 1.206, 0.029)
///   LR = (0.5323, 0,     1.206, 0.3  )
///   UR = (1.5,    0,     0,     1.5  )
///   UL = (0.5323, 1.206, 0,     0.3  )
/// (rho, u, v, p). All four quadrant-interface 1-D Riemann problems are
/// single pure shocks (Schulz-Rinne class "S, S, S, S"; the star state
/// equals the heavy quadrant exactly, by construction of the data);
/// everything converges toward the heavy upper-right quadrant and the
/// interaction rolls the shear layers into the classic double spiral.
///
/// Boundary handling: free-avg — the exterior state is the interior
/// element's constant mode (the cell mean), the finite-volume edge-pad.
/// Unlike the pointwise free trace, whose high-order oscillations feed
/// the wall flux and pile up unboundedly in the indicator-blind wall
/// band (rho ~ 68 by t = 0.8 on this setup), the constant mode carries
/// no oscillation and the wave systems exit cleanly.
///
/// Acceptance criteria:
///   - the run completes with a finite residual;
///   - density/pressure positivity and a loose non-oscillatory bound;
///   - the exact x <-> y symmetry of the configuration (UL <-> LR, LL/UR
///     self-symmetric) is preserved by the scheme + limiter + boundary
///     data on a square mesh: the cell-mean field equals its transpose up
///     to the chaotic amplification of transposed-pair rounding
///     differences (measured max ~0.21 at t = 0.8 on 96^2 P2).
/// The triangle variant (P1, 32x32) covers the full chain (non-affine
/// projection-path limiter + analytic boundary on split triangulations);
/// the free-BC neutrality of quiescent walls is checked exactly by the
/// confined-bump case below.

#include <Eigen/Dense>
#include <occa.hpp>

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

int g_failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        ++g_failures;
        std::cout << "  [FAIL] " << what << "\n";
    } else {
        std::cout << "  [ok]   " << what << "\n";
    }
}

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

/// @brief Run one configuration of the 2-D Riemann problem and verify.
/// \p symmetric enables the x <-> y transpose check (square quad mesh).
/// Boundary: free-avg (cell-mean zero-gradient) on all walls.
void runRiemann2d(const std::string &label, int order, int n, Real cfl,
                  Real tFinal, bool splitTriangles, bool symmetric) {
    std::cout << label << "\n";
    const char *tri = splitTriangles ? "\nsplit_triangles = true" : "";
    const std::string toml =
        std::string("") + "[basis]\norder = " + std::to_string(order) +
        "\nnq = " + std::to_string(order + 1) +
        "\n\n[mesh]\nnx = " + std::to_string(n) +
        "\nny = " + std::to_string(n) +
        "\nx0 = 0.0\ny0 = 0.0\ndx = " + std::to_string(1.0 / n) +
        "\ndy = " + std::to_string(1.0 / n) + tri +
        "\n\n[bc]\ntype = \"free-avg\"\n"
        "\n[flow]\nrho = 1.5\nu = 0.0\nv = 0.0\n"
        "p = 1.5\n\n[time_marching]\nmethod = \"ssprk3\"\ncfl = " +
        std::to_string(static_cast<double>(cfl)) +
        "\nt_final = " + std::to_string(static_cast<double>(tFinal)) +
        "\n\n[initial_condition]\ntype = \"expr\"\n"
        "rho = \"(y < 0.8) ? ((x < 0.8) ? 0.138 : 0.5323) : "
        "((x < 0.8) ? 0.5323 : 1.5)\"\n"
        "u = \"(x < 0.8) ? 1.206 : 0.0\"\n"
        "v = \"(y < 0.8) ? 1.206 : 0.0\"\n"
        "p = \"(y < 0.8) ? ((x < 0.8) ? 0.029 : 0.3) : "
        "((x < 0.8) ? 0.3 : 1.5)\"\n\n[limiter]\ntype = \"mr-weno\"\n"
        "positivity = true\n";

    const std::string path = "/tmp/cmeles_riemann2d.toml";
    std::FILE *f           = std::fopen(path.c_str(), "w");
    std::fputs(toml.c_str(), f);
    std::fclose(f);
    const Config cfg(path);
    std::remove(path.c_str());

    CompressibleFlowSolver solver(cfg);
    if (solver.run() != 0) {
        check(false, "run completes");
        return;
    }
    check(true, "run completes");

    DgField &field            = solver.field();
    const std::vector<Real> u = readModal(field, solver.mem());
    const int N_elem          = field.numElements();
    const int N_modes         = field.numModes();
    const int Nq2             = field.numQuadPoints();
    const MatrixXr &V         = field.basis().vandermonde();
    const VectorXr &lamAll    = field.geometry().lambdaWJ();
    const auto &e8            = field.geometry().vertices();

    // --- Quadrature-point states: positivity, bounds, cell means.
    Real mass   = Real(0);
    Real rhoMin = Real(1e300), rhoMax = Real(-1e300), pMin = Real(1e300);
    std::vector<Real> cellRho(static_cast<std::size_t>(N_elem));
    std::vector<Real> xc(static_cast<std::size_t>(N_elem));
    std::vector<Real> yc(static_cast<std::size_t>(N_elem));
    for (int e = 0; e < N_elem; ++e) {
        Eigen::Map<const VectorXr> lam(
            lamAll.data() + static_cast<std::size_t>(e) * Nq2, Nq2);
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
        cellRho[e] = (lam.cwiseProduct(qn[0])).sum();
        mass += cellRho[e];
        xc[e] = (e8(e, 0) + e8(e, 2) + e8(e, 4) + e8(e, 6)) / Real(4);
        yc[e] = (e8(e, 1) + e8(e, 3) + e8(e, 5) + e8(e, 7)) / Real(4);
    }

    // --- Mass: the characteristic boundary is open (outflow through the
    //     walls once the waves arrive), so no closed-form balance exists —
    //     finiteness only. Exact conservation of the closed domain is
    //     covered by the periodic DG field tests; the free-BC neutrality
    //     is verified exactly by the confined-bump case below.
    check(mass > Real(0) && std::isfinite(mass), "mass finite/positive");
    check(rhoMin > Real(0) && pMin > Real(0),
          "density/pressure positive (min rho " + std::to_string(rhoMin) +
              ", min p " + std::to_string(pMin) + ")");
    // Loose physical bracket across the quad/triangle variants: by
    // T = 0.8 most compression has exited through the near walls and the
    // field lives in [0.06, 1.8] on the 96^2 P2 reference run; coarse
    // resolutions mix the contacts more (undershoot below the untouched
    // LL value 0.138) — keep the wider margin.
    check(rhoMin >= Real(0.03) && rhoMax <= Real(3.0),
          "density in [0.03, 3.0] (got [" + std::to_string(rhoMin) + ", " +
              std::to_string(rhoMax) + "])");

    // --- x <-> y symmetry: bin the cell means on the square grid and
    //     compare the field with its transpose.
    if (symmetric) {
        const Real h = Real(1) / Real(n);
        std::vector<Real> a(static_cast<std::size_t>(n) * n, Real(0));
        std::vector<int> cnt(static_cast<std::size_t>(n) * n, 0);
        bool binned = true;
        for (int e = 0; e < N_elem; ++e) {
            const int c = static_cast<int>(std::floor(xc[e] / h + Real(1e-9)));
            const int r = static_cast<int>(std::floor(yc[e] / h + Real(1e-9)));
            if (c < 0 || c >= n || r < 0 || r >= n) {
                binned = false;
                break;
            }
            a[static_cast<std::size_t>(r) * n + c] += cellRho[e];
            ++cnt[static_cast<std::size_t>(r) * n + c];
        }
        Real asym = Real(0);
        if (binned) {
            for (int r = 0; r < n; ++r) {
                for (int c = 0; c < n; ++c) {
                    const Real ar = a[static_cast<std::size_t>(r) * n + c];
                    const Real at = a[static_cast<std::size_t>(c) * n + r];
                    asym          = std::max(asym, std::abs(ar - at));
                }
            }
        }
        // Tolerance: configuration 3's double spiral is a chaotic
        // amplifier — transposed cell pairs meet identical math but
        // different tensor-quadrature summation orders, so 1e-16-level
        // rounding differences grow exponentially in time: ~3e-6 at
        // t = 0.15 (measured), reaching max ~0.21 by the full T = 0.8.
        // The fast t_final = 0.05 regression window stays orders below a
        // systematic asymmetry bug's O(1) signature (a y-biased flux,
        // asymmetric boundary data).
        const Real symTol = kSinglePrecision ? Real(1e-2) : Real(1e-3);
        check(binned && asym <= symTol,
              "x<->y symmetry (asymmetry " + std::to_string(asym) + ")");
    }
}

} // namespace

// ----------------------------------------------------------------------------
// Free-boundary neutrality: confined bump on quiescent walls
// ----------------------------------------------------------------------------

namespace {
/// @brief Exact free-BC conservation check: a Gaussian density bump
///        confined to the domain centre on an otherwise quiescent field.
///        The walls stay in their constant state until the bump's waves
///        arrive (fastest speed ~1.3, travel <= 0.13 by t = 0.1, while the
///        bump reaches r ~ 0.25 and the walls sit 0.5 away), so the net
///        boundary flux is exactly zero and the total mass equals the
///        analytic integral of the initial field to round-off.
void runFreeBoundaryNeutrality() {
    std::cout << "Free BC neutrality (confined bump)\n";
    const std::string toml =
        "[basis]\norder = 1\nnq = 4\n\n[mesh]\nnx = 32\nny = 32\nx0 = 0.0\n"
        "y0 = 0.0\ndx = 0.03125\ndy = 0.03125\n\n[bc]\ntype = \"free\"\n\n"
        "[flow]\nrho = 1.0\nu = 0.0\nv = 0.0\np = 1.0\n\n[time_marching]\n"
        "method = \"ssprk3\"\ncfl = 0.3\nt_final = 0.1\n\n[initial_condition]\n"
        "type = \"expr\"\n"
        "rho = \"1 + 0.5*exp(-((x - 0.5)^2 + (y - 0.5)^2)/0.01)\"\n"
        "u = \"0\"\nv = \"0\"\np = \"1\"\n\n[limiter]\ntype = \"mr-weno\"\n";
    const std::string path = "/tmp/cmeles_freebc.toml";
    std::FILE *f           = std::fopen(path.c_str(), "w");
    std::fputs(toml.c_str(), f);
    std::fclose(f);
    const Config cfg(path);
    std::remove(path.c_str());

    CompressibleFlowSolver solver(cfg);
    if (solver.run() != 0) {
        check(false, "run completes");
        return;
    }
    check(true, "run completes");

    DgField &field            = solver.field();
    const std::vector<Real> u = readModal(field, solver.mem());
    const int N_elem          = field.numElements();
    const int N_modes         = field.numModes();
    const int Nq2             = field.numQuadPoints();
    const MatrixXr &V         = field.basis().vandermonde();
    const VectorXr &lamAll    = field.geometry().lambdaWJ();

    Real mass = Real(0);
    for (int e = 0; e < N_elem; ++e) {
        Eigen::Map<const VectorXr> lam(
            lamAll.data() + static_cast<std::size_t>(e) * Nq2, Nq2);
        Eigen::Map<const VectorXr> rhoh(
            u.data() + static_cast<std::size_t>(e) * 4 * N_modes, N_modes);
        mass += (lam.cwiseProduct(V * rhoh)).sum();
    }

    // Analytic initial mass by composite Simpson (smooth bump: ~1e-12).
    const int nS = 1000;
    const Real h = Real(1) / Real(nS);
    auto rho0    = [](Real x, Real y) {
        return Real(1) +
               Real(0.5) * std::exp(-(((x - Real(0.5)) * (x - Real(0.5)) +
                                       (y - Real(0.5)) * (y - Real(0.5))) /
                                      Real(0.01)));
    };
    Real m0 = Real(0);
    for (int i = 0; i <= nS; ++i) {
        const Real wx = (i == 0 || i == nS) ? Real(1)
                        : (i % 2 == 1)      ? Real(4)
                                            : Real(2);
        for (int j = 0; j <= nS; ++j) {
            const Real wy = (j == 0 || j == nS) ? Real(1)
                            : (j % 2 == 1)      ? Real(4)
                                                : Real(2);
            m0 += wx * wy * rho0(i * h, j * h);
        }
    }
    m0 = m0 * h * h / Real(9);

    const Real tol = kSinglePrecision ? Real(5e-4) : Real(1e-10);
    check(std::abs(mass - m0) <= tol * std::max(Real(1), m0),
          "mass exactly conserved with quiescent walls (M0 " +
              std::to_string(m0) + ", got " + std::to_string(mass) + ")");
}
} // namespace

int main() {
    // t_final = 0.05 keeps the regression a smoke test (~40 s): the
    // grid-misaligned (0.8*64 = 51.2) projected discontinuity's
    // positivity path is covered and the wall-adjacent shock feet are
    // already active. The full presentation time T = 0.8 of the ORIGINAL
    // Lax & Liu convention (split at (0.8, 0.8)) is covered by the
    // production runs (config/riemann2d.toml); the pointwise free
    // variant would pile up unboundedly instead (archived negative
    // result).
    runRiemann2d("2D Riemann (Lax-Liu 3) / quad / P2 / analytic riemann bc", 2,
                 64, Real(0.1), Real(0.05), false, true);
    runRiemann2d("2D Riemann (Lax-Liu 3) / triangles / P1 / analytic riemann bc",
                 1, 32, Real(0.1), Real(0.05), true, false);
    runFreeBoundaryNeutrality();

    if (g_failures == 0) {
        std::cout << "riemann2d: all checks passed\n";
        return 0;
    }
    std::cout << "riemann2d: " << g_failures << " check(s) failed\n";
    return 1;
}
