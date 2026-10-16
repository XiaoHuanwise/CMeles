/// @file test_positivity.cpp
/// @brief Invariant tests of the Zhang & Shu positivity-preserving scaling
///        limiter and its global stepper integration.
///
/// Acceptance of docs/tech_docs/positivity_limiter.md:
///   a. Adversarial modal field (negative density / internal energy at
///      quadrature and face points): after apply() every volume Gauss and
///      face Gauss point satisfies rho > 0 and e > 0 with margin, cell
///      means are conserved, total mass is conserved, injected elements
///      are flagged 1 and clean elements flagged 0 / untouched.
///   b. Smooth field: theta == 1 everywhere — u bit-identical, flags 0.
///   c. Coupled dual-time stacked state (2N): each half projected
///      independently, clean half bit-identical.
///   d. Filter-contract smoke across stepper families with
///      positivity = true (pure and chained with mr-weno): complete runs,
///      conserved mass — exercises the pseudo-state filter plumbing of the
///      implicit family.
///   e. Dual-time strong shock (double-Mach IC on a small mesh): the
///      be method survives the first physical step that previously NaN'd
///      without the positivity limiter.

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
#include "limiter/PositivityLimiter.hpp"
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

/// @brief Write a host modal vector into the field state.
void writeModal(DgField &field, DeviceMemoryManager &mem,
                const std::vector<Real> &u) {
    occa::memory ou = field.o_u();
    if (mem.hasSeparateMemorySpace()) {
        mem.copyFromHost(ou, u.data(), static_cast<occa::dim_t>(u.size()));
    } else {
        Real *dst = ou.ptr<Real>();
        std::memcpy(dst, u.data(), u.size() * sizeof(Real));
    }
}

/// @brief True cell means per element/variable (exact quadrature formula).
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

/// @brief Total mass = sum over elements of the quadrature integral of rho.
Real totalMass(const DgField &field, const std::vector<Real> &u) {
    const int N_elem       = field.numElements();
    const int N_vars       = field.numVars();
    const int N_modes      = field.numModes();
    const int Nq2          = field.numQuadPoints();
    const MatrixXr &V      = field.basis().vandermonde();
    const VectorXr &lamAll = field.geometry().lambdaWJ();

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

/// @brief Host mirror of the kernel's admissibility walk: the minimum of
///        rho / rhoBar and e / eBar over every volume Gauss point and face
///        Gauss trace point of every element.
void worstAdmissibleRatios(DgField &field, DeviceMemoryManager &mem,
                           const std::vector<Real> &u, Real &worstRho,
                           Real &worstInt) {
    const int N_elem        = field.numElements();
    const int N_vars        = field.numVars();
    const int N_modes       = field.numModes();
    const int Nq2           = field.numQuadPoints();
    const int N_q           = field.numPoints1D();
    const MeshGeometry &geo = field.geometry();

    // Face-point tables (device -> host copies; layout mirrors the kernel).
    const int nFaceInfo = geo.numFaces();
    std::vector<Real> vface(static_cast<std::size_t>(8) * N_q * N_modes);
    std::vector<int> elemFaces(static_cast<std::size_t>(N_elem) * 4);
    std::vector<int> faceKL(nFaceInfo), faceFL(nFaceInfo), faceKR(nFaceInfo),
        faceFR(nFaceInfo);
    occa::memory oVface  = field.o_faceVandermonde();
    occa::memory oEFaces = geo.o_elemFaces();
    occa::memory oFKL    = geo.o_faceKL();
    occa::memory oFFL    = geo.o_faceFL();
    occa::memory oFKR    = geo.o_faceKR();
    occa::memory oFFR    = geo.o_faceFR();
    // Unified backends: copyToHost is a no-op (the destination is never
    // the wrapped source) — read through the host alias instead.
    if (mem.hasSeparateMemorySpace()) {
        mem.copyToHost(oVface, vface.data(),
                       static_cast<occa::dim_t>(vface.size()));
        mem.copyToHost(oEFaces, elemFaces.data(),
                       static_cast<occa::dim_t>(elemFaces.size()));
        mem.copyToHost(oFKL, faceKL.data(), nFaceInfo);
        mem.copyToHost(oFFL, faceFL.data(), nFaceInfo);
        mem.copyToHost(oFKR, faceKR.data(), nFaceInfo);
        mem.copyToHost(oFFR, faceFR.data(), nFaceInfo);
    } else {
        std::memcpy(vface.data(), oVface.ptr<Real>(),
                    vface.size() * sizeof(Real));
        std::memcpy(elemFaces.data(), oEFaces.ptr<int>(),
                    elemFaces.size() * sizeof(int));
        std::memcpy(faceKL.data(), oFKL.ptr<int>(), nFaceInfo * sizeof(int));
        std::memcpy(faceFL.data(), oFFL.ptr<int>(), nFaceInfo * sizeof(int));
        std::memcpy(faceKR.data(), oFKR.ptr<int>(), nFaceInfo * sizeof(int));
        std::memcpy(faceFR.data(), oFFR.ptr<int>(), nFaceInfo * sizeof(int));
    }

    const MatrixXr &V = field.basis().vandermonde();
    worstRho          = Real(1e300);
    worstInt          = Real(1e300);
    for (int e = 0; e < N_elem; ++e) {
        const Real *uE  = &u[static_cast<std::size_t>(e) * N_vars * N_modes];
        const auto mean = [V, uE, N_modes, e, &field, N_vars,
                           Nq2](int var) -> Real {
            Eigen::Map<const VectorXr> lam(field.geometry().lambdaWJ().data() +
                                               static_cast<std::size_t>(e) *
                                                   Nq2,
                                           Nq2);
            Eigen::Map<const VectorXr> uh(
                uE + static_cast<std::size_t>(var) * N_modes, N_modes);
            return (lam.cwiseProduct(V * uh)).sum() / lam.sum();
        };
        const Real mRho = mean(0), mMx = mean(1), mMy = mean(2), mE = mean(3);
        const Real mInt = mE - Real(0.5) * (mMx * mMx + mMy * mMy) / mRho;

        const auto probe = [&](const Real *Vrow) {
            Real rh = 0, mx = 0, my = 0, en = 0;
            for (int m = 0; m < N_modes; ++m) {
                const Real v = Vrow[m];
                rh += v * uE[m];
                mx += v * uE[N_modes + m];
                my += v * uE[2 * N_modes + m];
                en += v * uE[3 * N_modes + m];
            }
            const Real intE = en - Real(0.5) * (mx * mx + my * my) / rh;
            worstRho        = std::min(worstRho, rh / mRho);
            worstInt        = std::min(worstInt, intE / mInt);
        };

        for (int q = 0; q < Nq2; ++q) {
            probe(&V(static_cast<Eigen::Index>(q), 0));
        }
        for (int f = 0; f < 4; ++f) {
            const int F = elemFaces[static_cast<std::size_t>(e) * 4 + f];
            if (F < 0) {
                continue;
            }
            const int side = (e == faceKL[static_cast<std::size_t>(F)]) ? 0 : 1;
            const int fOwn = (side == 0) ? faceFL[static_cast<std::size_t>(F)]
                                         : faceFR[static_cast<std::size_t>(F)];
            const Real *Vf =
                vface.data() +
                (static_cast<std::size_t>(side * 4 + fOwn)) * N_q * N_modes;
            for (int i = 0; i < N_q; ++i) {
                probe(Vf + static_cast<std::size_t>(i) * N_modes);
            }
        }
    }
}

/// @brief Standalone field fixture (device + memory manager + DG field with
///        the configured initial condition applied).
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

void check(bool ok, const std::string &what) {
    if (!ok) {
        ++g_failures;
        std::cout << "  [FAIL] " << what << "\n";
    } else {
        std::cout << "  [ok]   " << what << "\n";
    }
}

// ----------------------------------------------------------------------------
// a) Adversarial modal field: negative density / internal energy at points
// ----------------------------------------------------------------------------

void testAdversarial() {
    const std::string toml =
        "[basis]\norder = 2\nnq = 3\n\n[mesh]\nnx = 8\nny = 8\ndx = 0.125\n"
        "dy = 0.125\n\n[bc]\ntype = \"periodic\"\n\n[flow]\nrho = 1.2\n"
        "u = 0.0\nv = 0.0\np = 1.0\n\n[time_marching]\nmethod = \"ssprk3\"\n"
        "cfl = 0.18\nt_final = 0.01\n\n[initial_condition]\n"
        "type = \"uniform\"\n\n[limiter]\npositivity = true\n";
    std::cout << "(a) adversarial field / admissibility + conservation\n";
    const Config cfg = configFromString(toml, "/tmp/cmeles_pos_a.toml");
    FieldFixture fx  = makeField(cfg);

    const int nElem  = fx.field->numElements();
    const int nVars  = fx.field->numVars();
    const int nModes = fx.field->numModes();

    // Inject violent high-order modes into a middle band of elements: a
    // large linear mode on rho (drives rho negative at cell ends) and one
    // on rho_u (drives the internal energy negative via |m|^2/(2 rho)).
    std::vector<Real> u0 = readModal(*fx.field, *fx.mem);
    int nInjected        = 0;
    for (int e = nElem / 4; e < 3 * nElem / 4; ++e) {
        Real *uE = &u0[static_cast<std::size_t>(e) * nVars * nModes];
        uE[1] += Real(3.6);          // rho, linear-x mode
        uE[nModes + 1] += Real(3.0); // rho_u, linear-x mode
        ++nInjected;
    }
    writeModal(*fx.field, *fx.mem, u0);

    const std::vector<Real> mean0 = cellMeans(*fx.field, u0);
    const Real mass0              = totalMass(*fx.field, u0);

    PositivityLimiter limiter(cfg, fx.device, *fx.mem, *fx.field);
    limiter.apply(fx.field->o_u());
    fx.device.finish();

    const std::vector<Real> u1    = readModal(*fx.field, *fx.mem);
    const std::vector<Real> mean1 = cellMeans(*fx.field, u1);
    const Real mass1              = totalMass(*fx.field, u1);

    // Admissibility at every volume + face point (relative to the means).
    Real worstRho = 0, worstInt = 0;
    worstAdmissibleRatios(*fx.field, *fx.mem, u1, worstRho, worstInt);
    std::cout << std::scientific << "  worst rho/rhoBar = " << worstRho
              << ", worst e/eBar = " << worstInt << std::defaultfloat << "\n";
    check(worstRho > Real(0.5e-10) && worstInt > Real(0.5e-10),
          "all volume/face points admissible");

    // Cell means and total mass conserved.
    Real meanDrift = Real(0);
    for (std::size_t i = 0; i < mean0.size(); ++i) {
        meanDrift =
            std::max(meanDrift, std::abs(mean1[i] - mean0[i]) /
                                    std::max(Real(1), std::abs(mean0[i])));
    }
    check(meanDrift <= (kSinglePrecision ? Real(5e-5) : Real(1e-12)),
          "cell means conserved (drift " + std::to_string(meanDrift) + ")");
    check(std::abs(mass1 - mass0) <=
              std::max(Real(1), std::abs(mass0)) *
                  (kSinglePrecision ? Real(5e-5) : Real(1e-12)),
          "total mass conserved");

    // Flags: injected elements scaled (1), clean elements untouched (0),
    // and the clean elements' modal blocks bit-identical.
    std::vector<int> flags;
    limiter.readFlags(flags);
    int scaled = 0, inadmissible = 0;
    bool flagsOk = true, cleanUntouched = true, allFinite = true;
    for (int e = 0; e < nElem; ++e) {
        if (flags[static_cast<std::size_t>(e)] == 1) {
            ++scaled;
        }
        if (flags[static_cast<std::size_t>(e)] == 2) {
            ++inadmissible;
        }
        const bool injected = e >= nElem / 4 && e < 3 * nElem / 4;
        flagsOk = flagsOk &&
                  (flags[static_cast<std::size_t>(e)] == (injected ? 1 : 0));
        if (!injected) {
            for (int i = 0; i < nVars * nModes; ++i) {
                const std::size_t idx =
                    static_cast<std::size_t>(e) * nVars * nModes + i;
                cleanUntouched = cleanUntouched && u0[idx] == u1[idx];
                allFinite      = allFinite && std::isfinite(u1[idx]);
            }
        } else {
            for (int i = 0; i < nVars * nModes; ++i) {
                const std::size_t idx =
                    static_cast<std::size_t>(e) * nVars * nModes + i;
                allFinite = allFinite && std::isfinite(u1[idx]);
            }
        }
    }
    check(inadmissible == 0, "no element flagged mean-inadmissible");
    check(scaled == nInjected, "exactly the injected elements scaled (" +
                                   std::to_string(scaled) + " of " +
                                   std::to_string(nInjected) + ")");
    check(flagsOk, "flag pattern matches injection band");
    check(cleanUntouched, "clean elements bit-identical");
    check(allFinite, "output finite");
}

// ----------------------------------------------------------------------------
// b) Smooth field: theta == 1, bit-identical no-op
// ----------------------------------------------------------------------------

void testSmoothNoOp() {
    const std::string toml =
        "[basis]\norder = 2\nnq = 3\n\n[mesh]\nnx = 8\nny = 8\ndx = 0.125\n"
        "dy = 0.125\n\n[bc]\ntype = \"periodic\"\n\n[flow]\nrho = 1.0\n"
        "u = 0.1\nv = 0.05\np = 1.0\n\n[time_marching]\nmethod = \"ssprk3\"\n"
        "cfl = 0.18\nt_final = 0.01\n\n[initial_condition]\ntype = \"expr\"\n"
        "rho = \"1 + 0.2*sin(2*pi*x) + 0.1*cos(2*pi*y)\"\nu = \"0.1\"\n"
        "v = \"0.05\"\np = \"1 + 0.15*sin(2*pi*(x + y))\"\n\n[limiter]\n"
        "positivity = true\n";
    std::cout << "(b) smooth field / bit-identical no-op\n";
    const Config cfg = configFromString(toml, "/tmp/cmeles_pos_b.toml");
    FieldFixture fx  = makeField(cfg);
    const std::vector<Real> u0 = readModal(*fx.field, *fx.mem);

    PositivityLimiter limiter(cfg, fx.device, *fx.mem, *fx.field);
    limiter.apply(fx.field->o_u());
    fx.device.finish();

    const std::vector<Real> u1 = readModal(*fx.field, *fx.mem);
    bool identical             = u0.size() == u1.size();
    for (std::size_t i = 0; identical && i < u0.size(); ++i) {
        identical = u0[i] == u1[i];
    }
    check(identical, "u bit-identical (theta == 1 everywhere)");
    std::vector<int> flags;
    limiter.readFlags(flags);
    bool allZero = true;
    for (int f : flags) {
        allZero = allZero && f == 0;
    }
    check(allZero, "all flags 0");
}

// ----------------------------------------------------------------------------
// c) Coupled dual-time stacked state (2N): halves projected independently
// ----------------------------------------------------------------------------

void testStackedState() {
    const std::string toml =
        "[basis]\norder = 2\nnq = 3\n\n[mesh]\nnx = 8\nny = 8\ndx = 0.125\n"
        "dy = 0.125\n\n[bc]\ntype = \"periodic\"\n\n[flow]\nrho = 1.2\n"
        "u = 0.0\nv = 0.0\np = 1.0\n\n[time_marching]\nmethod = \"ssprk3\"\n"
        "cfl = 0.18\nt_final = 0.01\n\n[initial_condition]\n"
        "type = \"uniform\"\n\n[limiter]\npositivity = true\n";
    std::cout << "(c) stacked 2N state / halves independent\n";
    const Config cfg = configFromString(toml, "/tmp/cmeles_pos_c.toml");
    FieldFixture fx  = makeField(cfg);
    const int n =
        fx.field->numElements() * fx.field->numVars() * fx.field->numModes();

    // First half: adversarial field (limited reference computed on the
    // field itself); second half: clean field.
    std::vector<Real> half = readModal(*fx.field, *fx.mem);
    for (int e = 16; e < 48; ++e) {
        half[static_cast<std::size_t>(e) * 4 * 6 + 1] += Real(3.6);
        half[static_cast<std::size_t>(e) * 4 * 6 + 6 + 1] += Real(3.0);
    }
    writeModal(*fx.field, *fx.mem, half);

    // Reference: limit the field directly, then restore.
    PositivityLimiter limiter(cfg, fx.device, *fx.mem, *fx.field);
    limiter.apply(fx.field->o_u());
    fx.device.finish();
    const std::vector<Real> limited = readModal(*fx.field, *fx.mem);

    std::vector<Real> clean(static_cast<std::size_t>(n), Real(0));
    for (int e = 0; e < 64; ++e) {
        clean[static_cast<std::size_t>(e) * 4 * 6]     = Real(1.2);
        clean[static_cast<std::size_t>(e) * 4 * 6 + 3] = Real(2.5);
    }

    // Stacked buffer [adversarial | clean], allocated through the manager
    // (unified backends wrap, separate backends malloc+upload).
    std::vector<Real> stacked;
    stacked.insert(stacked.end(), half.begin(), half.end());
    stacked.insert(stacked.end(), clean.begin(), clean.end());
    occa::memory o_stacked =
        fx.mem->wrapOrMalloc(stacked.data(), static_cast<occa::dim_t>(2 * n));

    limiter.apply(o_stacked);
    fx.device.finish();

    std::vector<Real> out(static_cast<std::size_t>(2 * n), Real(0));
    if (fx.mem->hasSeparateMemorySpace()) {
        fx.mem->copyToHost(o_stacked, out.data(), 2 * n);
    } else {
        const Real *src = o_stacked.ptr<Real>();
        std::memcpy(out.data(), src,
                    static_cast<std::size_t>(2 * n) * sizeof(Real));
    }

    bool firstOk = true, secondOk = true;
    for (int i = 0; i < n; ++i) {
        firstOk  = firstOk && out[static_cast<std::size_t>(i)] ==
                                  limited[static_cast<std::size_t>(i)];
        secondOk = secondOk && out[static_cast<std::size_t>(n + i)] ==
                                   clean[static_cast<std::size_t>(i)];
    }
    check(firstOk, "stacked first half matches the single-field limiting");
    check(secondOk, "stacked second half (clean) bit-identical");
}

// ----------------------------------------------------------------------------
// d) Filter-contract smoke across stepper families with positivity on
// ----------------------------------------------------------------------------

void testStepperFamilies() {
    std::cout << "(d) stepper families with positivity active\n";
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
        {"ditr-u2r1/ssprk321",
         "method = \"ditr-u2r1\"\ncfl = 0.2\nt_final = 0.02\npseudo_method "
         "= \"ssprk321\"\nmax_pseudo_steps = 300\n"},
    };
    for (const Case &c : cases) {
        const std::string toml =
            "[basis]\norder = 2\nnq = 4\n\n[mesh]\nnx = 16\nny = 8\n"
            "dx = 0.0625\ndy = 0.125\n\n[bc]\ntype = \"periodic\"\n\n"
            "[flow]\nrho = 1.0\nu = 0.1\nv = 0.0\np = 1.0\n\n"
            "[time_marching]\n" +
            std::string(c.timeTOML) +
            "\n[initial_condition]\ntype = \"expr\"\n"
            "rho = \"1 + 0.2*sin(2*pi*x) + 0.1*cos(2*pi*y)\"\nu = \"0.1\"\n"
            "v = \"0.05\"\np = \"1 + 0.15*sin(2*pi*(x + y))\"\n\n[limiter]\n"
            "type = \"mr-weno\"\npositivity = true\n";
        const Config cfg = configFromString(toml, "/tmp/cmeles_pos_d.toml");
        CompressibleFlowSolver solver(cfg);
        const bool ran = solver.run() == 0;
        check(ran, std::string("run completes (mr-weno+positivity, ") + c.name +
                       ")");

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
// e) Dual-time strong shock: the double-Mach IC that NaN'd at step 1
//    without the positivity limiter
// ----------------------------------------------------------------------------

void testDualTimeStrongShock() {
    std::cout << "(e) dual-time strong shock (double-Mach IC)\n";
    const std::string toml =
        "[basis]\norder = 2\nnq = 3\n\n[mesh]\nnx = 24\nny = 6\n"
        "dx = 0.16666666666666666\ndy = 0.16666666666666666\n\n[bc]\n"
        "type = \"free\"\nrho = \"(x - 1/6 - 20*t/sqrt(3)) < y/sqrt(3) ? 8.0 "
        ": 1.4\"\nu = \"(x - 1/6 - 20*t/sqrt(3)) < y/sqrt(3) ? "
        "8.25*cos(pi/6) : 0.0\"\nv = \"(x - 1/6 - 20*t/sqrt(3)) < "
        "y/sqrt(3) ? -8.25*sin(pi/6) : 0.0\"\np = \"(x - 1/6 - "
        "20*t/sqrt(3)) < y/sqrt(3) ? 116.5 : 1.0\"\n\n[bc.x_low]\n"
        "type = \"expr\"\n\n[bc.y_high]\ntype = \"expr\"\n\n[bc.y_low]\n"
        "type = \"slip\"\nsplit_x = 0.16666666666666666\nsplit_type = "
        "\"expr\"\n\n[riemann]\nflux = \"llf\"\n\n[gas]\ngamma = 1.4\n\n"
        "[flow]\nrho = 1.4\nu = 0.0\nv = 0.0\np = 1.0\n\n[time_marching]\n"
        "method = \"be\"\ncfl = 0.18\nt_final = 0.004\nmax_pseudo_steps = "
        "300\n\n[initial_condition]\ntype = \"expr\"\n"
        "rho = \"(x - 1/6) < y/sqrt(3) ? 8.0 : 1.4\"\nu = \"(x - 1/6) < "
        "y/sqrt(3) ? 8.25*cos(pi/6) : 0.0\"\nv = \"(x - 1/6) < y/sqrt(3) ? "
        "-8.25*sin(pi/6) : 0.0\"\np = \"(x - 1/6) < y/sqrt(3) ? 116.5 : "
        "1.0\"\n\n[limiter]\ntype = \"mr-weno\"\npositivity = true\n";
    const Config cfg = configFromString(toml, "/tmp/cmeles_pos_e.toml");
    CompressibleFlowSolver solver(cfg);
    const bool ran = solver.run() == 0;
    check(ran, "be + positivity survives the strong-shock start-up");

    DgField &field            = solver.field();
    const std::vector<Real> u = readModal(field, solver.mem());
    bool finite               = true;
    for (Real v : u) {
        finite = finite && std::isfinite(v);
    }
    check(finite, "final state finite");
}

} // namespace

int main() {
    testAdversarial();
    testSmoothNoOp();
    testStackedState();
    testStepperFamilies();
    testDualTimeStrongShock();

    if (g_failures == 0) {
        std::cout << "positivity limiter: all tests passed\n";
        return 0;
    }
    std::cout << "positivity limiter: " << g_failures << " FAILURES\n";
    return 1;
}
