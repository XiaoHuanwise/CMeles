/// @file test_dg_periodic.cpp
/// @brief Correctness tests for the periodic boundary pairing.
///
/// Key checks:
///   1. Pairing topology: on a structured mesh with periodic boundaries every
///      boundary face is paired (FaceType::Periodic), the K_R/f_R slots hold
///      the partner face's own left element/local face, and each pair
///      traverses its shared periodic edge in opposite directions under the
///      domain translation (the t2 = -t1 property the pseudo-interior
///      representation relies on).
///   2. Constant (uniform-flow) solution preservation on a periodic domain:
///      the residual of a uniform free stream must vanish.
///   3. Conservation of a doubly-periodic trigonometric density field: the
///      total mass rate sum_e int_K res rho dx must vanish to machine
///      precision — every interior *and* periodic face flux cancels pairwise,
///      which is the strongest validation of the periodic pairing.
///   4. Advection source sanity: the density residual matches the L2
///      projection of the analytic -(u d/dx + v d/dy) rho source (loose
///      tolerance: the LLF face dissipation of the non-polynomial field is
///      O(h^{p+1}) but not zero).
///   5. Uniform-flow preservation on a split-triangle periodic mesh.

#include <Eigen/Dense>
#include <occa.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "basis/BasisFunctions2D.hpp"
#include "common/Types.hpp"
#include "config/Config.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "dg/DgField.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/MeshGeometry.hpp"
#include "mesh/StructuredMeshGenerator.hpp"

namespace
{
const Real tol = Real(1e3) * RealEpsilon;

static occa::json tryMakeDevice(const std::string &mode)
{
    try
    {
        occa::json props;
        props["mode"] = mode;
        if (mode == "OpenCL")
        {
            props["platform_id"] = 0;
            props["device_id"]   = 0;
        }
        occa::device dev(props);
        if (dev.mode() != mode)
        {
            dev.free();
            return occa::json();
        }
        dev.free();
        return props;
    }
    catch (const std::exception &e)
    {
        std::cout << "  (skip " << mode << ": " << e.what() << ")\n";
        return occa::json();
    }
}

static void readResult(DeviceMemoryManager &mem, const occa::memory &o_data,
                       Real *dst, occa::dim_t entries)
{
    if (mem.hasSeparateMemorySpace())
    {
        occa::memory o = o_data;
        mem.copyToHost(o, dst, entries);
    }
    else
    {
        const Real *src = o_data.ptr<Real>();
        std::memcpy(dst, src, static_cast<std::size_t>(entries) * sizeof(Real));
    }
}

/// @brief Build a periodic test configuration (nx x ny grid of unit cells).
static Config makeConfig(int order, int nq, int nx, int ny, bool splitTriangles,
                         bool periodic)
{
    const std::string path = "test_dg_periodic_scratch.toml";
    std::FILE *f           = std::fopen(path.c_str(), "w");
    std::fprintf(f,
                 "[basis]\norder = %d\nnq = %d\n\n[mesh]\nnx = %d\nny = %d\n"
                 "x0 = 0.0\ny0 = 0.0\ndx = 1.0\ndy = 1.0\n"
                 "split_triangles = %s\n\n[bc]\ntype = \"%s\"\n\n"
                 "[riemann]\nflux = \"llf\"\n\n[gas]\ngamma = 1.4\n\n[flow]\n"
                 "rho = 1.0\nu = 0.5\nv = -0.2\np = 1.0\nmach = 0.1\n\n"
                 "[time_marching]\nmethod = \"ssprk3\"\ncfl = 0.2\n"
                 "t_final = 1.0\n\n[occa]\nmode = \"Serial\"\n",
                 order, nq, nx, ny, splitTriangles ? "true" : "false",
                 periodic ? "periodic" : "farfield");
    std::fclose(f);
    Config cfg(path);
    std::remove(path.c_str());
    return cfg;
}

static Real maxAbsResidual(DeviceMemoryManager &mem, DgField &field)
{
    const int nEvm = field.numElements() * field.numVars() * field.numModes();
    std::vector<Real> res(nEvm);
    readResult(mem, field.o_res(), res.data(), nEvm);
    Real maxAbs = Real(0);
    for (Real r : res)
    {
        maxAbs = std::max(maxAbs, std::abs(r));
    }
    return maxAbs;
}

static Real residualScale(const Config &cfg)
{
    return std::max(std::abs(cfg.flowRho() * cfg.flowU()),
                    std::abs(cfg.flowRho() * cfg.flowV())) +
           Real(1);
}
} // namespace

// ---------------------------------------------------------------------------
// 1. Pairing topology
// ---------------------------------------------------------------------------

static bool testPairingTopology()
{
    std::cout << "Test 1: periodic pairing topology\n";
    const int nx = 4, ny = 5;
    const Mesh mesh = StructuredMeshGenerator::generate(
        StructuredMeshGenerator::Params{nx, ny, 0, 0, 1, 1, 0, false});

    // Before pairing: boundary faces per the plain mesh.
    int nBoundaryBefore = 0;
    for (int F = 0; F < mesh.numFaces(); ++F)
    {
        nBoundaryBefore += mesh.isBoundaryFace(F) ? 1 : 0;
    }
    if (nBoundaryBefore != 2 * (nx + ny))
    {
        std::cout << "  FAIL: expected " << 2 * (nx + ny)
                  << " boundary faces, found " << nBoundaryBefore << "\n";
        return false;
    }

    Mesh paired = mesh;
    paired.applyPeriodicPairing(Real(nx), Real(ny));

    // Every former boundary face must now be Periodic with a valid partner.
    int nPeriodic = 0, nBoundaryAfter = 0;
    bool ok = true;
    for (int F = 0; F < paired.numFaces(); ++F)
    {
        if (paired.isPeriodicFace(F))
        {
            ++nPeriodic;
            const int KR = paired.faceElements()(F, 2);
            const int fR = paired.faceElements()(F, 3);
            if (KR < 0 || fR < 0)
            {
                std::cout << "  FAIL: periodic face " << F
                          << " has invalid partner slots\n";
                ok = false;
            }
        }
        nBoundaryAfter += paired.isBoundaryFace(F) ? 1 : 0;
    }
    if (nPeriodic != 2 * (nx + ny) || nBoundaryAfter != 0)
    {
        std::cout << "  FAIL: periodic = " << nPeriodic << " (expect "
                  << 2 * (nx + ny) << "), boundary = " << nBoundaryAfter
                  << " (expect 0)\n";
        ok = false;
    }

    // Pair structure: the partner relation must be symmetric and the two
    // faces must traverse the shared periodic edge in opposite directions
    // under the domain translation (A1 + T = B2, B1 + T = A2), which is the
    // t2 = -t1 property behind the pseudo-interior representation.
    const Real Lx = Real(nx), Ly = Real(ny);
    const Real eps = Real(1e-10);
    for (int F = 0; F < paired.numFaces() && ok; ++F)
    {
        if (!paired.isPeriodicFace(F))
        {
            continue;
        }
        const int KL = paired.faceElements()(F, 0);
        const int fL = paired.faceElements()(F, 1);
        const int KR = paired.faceElements()(F, 2);
        const int fR = paired.faceElements()(F, 3);

        // Find the partner face: the face whose left element is KR and
        // whose left local face is fR.
        int P = -1;
        for (int G = 0; G < paired.numFaces(); ++G)
        {
            if (paired.isPeriodicFace(G) && paired.faceElements()(G, 0) == KR &&
                paired.faceElements()(G, 1) == fR)
            {
                P = G;
                break;
            }
        }
        if (P < 0)
        {
            std::cout << "  FAIL: no partner face for periodic face " << F
                      << "\n";
            ok = false;
            break;
        }

        // Symmetry: the partner's partner must be F itself.
        if (paired.faceElements()(P, 2) != KL ||
            paired.faceElements()(P, 3) != fL)
        {
            std::cout << "  FAIL: pairing not symmetric at face " << F << "\n";
            ok = false;
            break;
        }

        // Opposite traversal under one of the four domain translations:
        // A1 + T = B2 and B1 + T = A2 (reversed traversal => t2 = -t1).
        const auto [a1, b1] = paired.faceVertices(KL, fL);
        const auto [a2, b2] = paired.faceVertices(KR, fR);
        const Real A1[2]    = {mesh.vertices()(a1, 0), mesh.vertices()(a1, 1)};
        const Real B1[2]    = {mesh.vertices()(b1, 0), mesh.vertices()(b1, 1)};
        const Real A2[2]    = {mesh.vertices()(a2, 0), mesh.vertices()(a2, 1)};
        const Real B2[2]    = {mesh.vertices()(b2, 0), mesh.vertices()(b2, 1)};
        auto dist           = [](const Real p[2], Real qx, Real qy) {
            const Real dx = p[0] - qx, dy = p[1] - qy;
            return std::sqrt(dx * dx + dy * dy);
        };
        const Real translations[4][2] = {{Lx, 0}, {-Lx, 0}, {0, Ly}, {0, -Ly}};
        bool matched                  = false;
        for (const auto &t : translations)
        {
            if (dist(A1, B2[0] - t[0], B2[1] - t[1]) < eps &&
                dist(B1, A2[0] - t[0], A2[1] - t[1]) < eps)
            {
                matched = true;
                break;
            }
        }
        if (!matched)
        {
            std::cout << "  FAIL: pair (" << F << "," << P
                      << ") does not match under a domain translation\n";
            ok = false;
            break;
        }
    }
    std::cout << "  periodic faces = " << nPeriodic << " (all paired, "
              << (ok ? "orientation OK" : "orientation FAIL") << ")\n";
    return ok;
}

// ---------------------------------------------------------------------------
// 2. Uniform-flow solution preservation on a periodic domain
// ---------------------------------------------------------------------------

static bool testPeriodicUniformFlow(const Config &cfg, const std::string &label)
{
    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mem(device);
    DgField field(cfg, device, mem);
    field.setup();

    field.applyFreeStreamInitialCondition(field.o_u());
    device.finish();
    field.computeRHS(field.o_u(), field.o_res());
    device.finish();

    const Real maxAbs = maxAbsResidual(mem, field);
    const Real scale  = residualScale(cfg);
    std::cout << "  [" << label << "] max |res| = " << maxAbs << " (scale "
              << scale << ")\n";
    return maxAbs <= tol * scale;
}

// ---------------------------------------------------------------------------
// 3+4. Trigonometric advection: conservation + analytic source
// ---------------------------------------------------------------------------

static bool testTrigAdvection()
{
    std::cout << "Test 3: doubly-periodic trigonometric advection\n";
    const int nx = 8, ny = 8;
    const Config cfg = makeConfig(2, 4, nx, ny, false, true);
    const Real gamma = cfg.gamma();
    const Real u0    = cfg.flowU();
    const Real v0    = cfg.flowV();
    const Real p0    = cfg.flowP();
    const Real Lx = Real(nx), Ly = Real(ny);
    const Real kx = Real(2) * Real(M_PI) / Lx;
    const Real ky = Real(2) * Real(M_PI) / Ly;

    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mem(device);
    DgField field(cfg, device, mem);
    field.setup();

    const int N_elem  = field.numElements();
    const int N_vars  = field.numVars();
    const int N_modes = field.numModes();
    const int Nq2     = field.numQuadPoints();

    // Nodal initial condition: rho = 1 + 0.1 sin(kx x) + 0.1 cos(ky y),
    // uniform velocity and pressure (smooth and doubly periodic).
    const auto &e8      = field.geometry().vertices();
    const MatrixX2r &qp = field.basis().quadraturePoints();
    std::vector<Real> xs(N_elem * Nq2), ys(N_elem * Nq2);
    VectorXr nodal(N_elem * N_vars * Nq2);
    for (int e = 0; e < N_elem; ++e)
    {
        const Real x1 = e8(e, 0), y1 = e8(e, 1);
        const Real x2 = e8(e, 2), y2 = e8(e, 3);
        const Real x3 = e8(e, 4), y3 = e8(e, 5);
        const Real x4 = e8(e, 6), y4 = e8(e, 7);
        for (int q = 0; q < Nq2; ++q)
        {
            const Real r = qp(q, 0), s = qp(q, 1);
            const Real phi1 = (Real(1) - r) * (Real(1) - s) / Real(4);
            const Real phi2 = (Real(1) + r) * (Real(1) - s) / Real(4);
            const Real phi3 = (Real(1) + r) * (Real(1) + s) / Real(4);
            const Real phi4 = (Real(1) - r) * (Real(1) + s) / Real(4);
            const Real x    = phi1 * x1 + phi2 * x2 + phi3 * x3 + phi4 * x4;
            const Real y    = phi1 * y1 + phi2 * y2 + phi3 * y3 + phi4 * y4;
            const Real rho  = Real(1) + Real(0.1) * std::sin(kx * x) +
                              Real(0.1) * std::cos(ky * y);
            const Real E =
                p0 / (gamma - Real(1)) + Real(0.5) * rho * (u0 * u0 + v0 * v0);
            const int base            = (e * N_vars + 0) * Nq2;
            xs[e * Nq2 + q]           = x;
            ys[e * Nq2 + q]           = y;
            nodal(base + q)           = rho;
            nodal(base + Nq2 + q)     = rho * u0;
            nodal(base + 2 * Nq2 + q) = rho * v0;
            nodal(base + 3 * Nq2 + q) = E;
        }
    }
    field.setInitialConditionNodal(field.o_u(), nodal);
    device.finish();

    field.computeRHS(field.o_u(), field.o_res());
    device.finish();

    const int nEvm = N_elem * N_vars * N_modes;
    std::vector<Real> res(nEvm);
    readResult(mem, field.o_res(), res.data(), nEvm);

    const MatrixXr &V      = field.basis().vandermonde();
    const VectorXr &lamAll = field.geometry().lambdaWJ();

    // (3) Conservation: total mass rate = sum_e sum_q lambda_q res_nodal
    // must vanish to machine precision (all face fluxes cancel pairwise,
    // including the periodic pairs).
    Real massRate = Real(0), massScale = Real(0);
    for (int e = 0; e < N_elem; ++e)
    {
        Eigen::Map<const VectorXr> lam(lamAll.data() + e * Nq2, Nq2);
        Eigen::Map<const MatrixXr> rhat(res.data() + (e * N_vars + 0) * N_modes,
                                        N_modes, 1);
        const VectorXr rn   = V * rhat;
        const VectorXr rhon = Eigen::Map<const VectorXr>(
            nodal.data() + (e * N_vars + 0) * Nq2, Nq2);
        massRate += lam.dot(rn);
        massScale += lam.dot(rhon);
    }
    std::cout << "  total mass rate = " << massRate << " (mass ~ " << massScale
              << ")\n";
    const bool conserved = std::abs(massRate) <= tol * massScale;

    // (4) Analytic source: d(rho)/dt = -(u0 dx + v0 dy) rho projected onto
    // the modal space; loose tolerance (LLF dissipation of the
    // non-polynomial field is small but non-zero).
    Real err2 = Real(0), src2 = Real(0);
    for (int e = 0; e < N_elem; ++e)
    {
        Eigen::Map<const VectorXr> lam(lamAll.data() + e * Nq2, Nq2);
        Eigen::Map<const MatrixXr> minv(
            field.geometry().massMatrixInverse().data() + e * N_modes * N_modes,
            N_modes, N_modes);
        VectorXr srcNodal(Nq2);
        for (int q = 0; q < Nq2; ++q)
        {
            const Real x = xs[e * Nq2 + q], y = ys[e * Nq2 + q];
            srcNodal(q) = -u0 * Real(0.1) * kx * std::cos(kx * x) +
                          v0 * Real(0.1) * ky * std::sin(ky * y);
        }
        const VectorXr srcModal =
            minv * (V.transpose() * (lam.asDiagonal() * srcNodal));
        Eigen::Map<const VectorXr> rhat(res.data() + (e * N_vars + 0) * N_modes,
                                        N_modes);
        err2 += (rhat - srcModal).squaredNorm();
        src2 += srcModal.squaredNorm();
    }
    // Modal coefficients are unweighted: mass-weight them for a physical norm.
    // On the orthogonal unit grid M = |J| I with |J| = 1/4, so the weighting
    // cancels in the relative measure.
    const Real relErr = std::sqrt(err2 / src2);
    std::cout << "  source relative L2 error = " << relErr << "\n";
    return conserved && relErr <= Real(0.2);
}

// ---------------------------------------------------------------------------
// 5. Uniform-flow preservation on a split-triangle periodic mesh
// ---------------------------------------------------------------------------

static bool testTrianglePeriodic(const Config &cfg, const std::string &label)
{
    return testPeriodicUniformFlow(cfg, label);
}

// ---------------------------------------------------------------------------
// Device backend sweep
// ---------------------------------------------------------------------------

static bool runOnBackend(const std::string &mode, occa::json props)
{
    if (props.isNull())
    {
        return true;
    }
    std::cout << "\n===== DG periodic backend [" << mode
              << "] =====" << std::endl;
    const Config cfg = makeConfig(2, 4, 4, 4, false, true);
    occa::device device(props);
    DeviceMemoryManager mem(device);
    DgField field(cfg, device, mem);
    try
    {
        field.setup();
    }
    catch (const std::exception &e)
    {
        // tryMakeDevice already confirmed the backend is usable, so a
        // setup failure here is a real defect, not an unavailable device.
        std::cout << "  (FAILED " << mode << ": setup failed — " << e.what()
                  << ")\n";
        return false;
    }

    field.applyFreeStreamInitialCondition(field.o_u());
    device.finish();
    field.computeRHS(field.o_u(), field.o_res());
    device.finish();

    const Real maxAbs = maxAbsResidual(mem, field);
    const Real scale  = residualScale(cfg);
    std::cout << "  max |res| = " << maxAbs << " (scale " << scale << ")\n";
    return maxAbs <= tol * scale;
}

int main()
{
    bool ok = true;
    ok &= testPairingTopology();

    std::cout << "Test 2: uniform-flow preservation (periodic)\n";
    ok &= testPeriodicUniformFlow(makeConfig(2, 4, 4, 5, false, true), "quads");

    ok &= testTrigAdvection();

    std::cout << "Test 5: uniform-flow preservation (split triangles)\n";
    ok &= testTrianglePeriodic(makeConfig(2, 4, 4, 5, true, true), "triangles");

    ok &= runOnBackend("Serial", tryMakeDevice("Serial"));
    ok &= runOnBackend("OpenMP", tryMakeDevice("OpenMP"));
    ok &= runOnBackend("OpenCL", tryMakeDevice("OpenCL"));

    std::cout << (ok ? "ALL DG PERIODIC TESTS PASSED\n"
                     : "DG PERIODIC TESTS FAILED\n");
    return ok ? 0 : 1;
}
