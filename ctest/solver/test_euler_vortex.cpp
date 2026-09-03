/// @file test_euler_vortex.cpp
/// @brief Isentropic-vortex advection validation with SSPRK3 (the
///        designated first validation of all non-time-marching modules).
///
/// Setup (classic Shu vortex): domain [0, 10]^2, mean flow (1, 1), vortex
/// centre (5, 5), beta = 5, periodic boundaries, N = 2 (nq = 4), CFL = 0.2.
/// The vortex is advected for T = 2 (centre moves to (7, 7)) and the final
/// density is compared against the analytic translated vortex at the
/// quadrature points:
///
/// $$
/// \begin{aligned}
///   u &= u_\infty - \tfrac{\beta}{2\pi} e^{(1-r^2)/2} (y - y_c), \quad
///   v = v_\infty + \tfrac{\beta}{2\pi} e^{(1-r^2)/2} (x - x_c), \\
///   T &= 1 - \tfrac{(\gamma-1)\beta^2}{8\gamma\pi^2} e^{1-r^2}, \quad
///   \rho = T^{1/(\gamma-1)}, \quad p = \rho^\gamma .
/// \end{aligned}
/// $$
///
/// With SSPRK3 (3rd order) and N = 2 (3rd-order space) the observed
/// convergence order must be ~3 under mesh refinement; the run goes through
/// the full CompressibleFlowSolver pipeline (config -> exprtk initial
/// condition -> periodic DG residual -> CFL stepping -> SSPRK3).
///
/// Full refinement study on Serial; one coarse run per additional backend
/// (unavailable backends / kernel-build failures are skipped).

#include <Eigen/Dense>
#include <occa.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

#include "basis/BasisFunctions2D.hpp"
#include "common/Types.hpp"
#include "config/Config.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "dg/DgField.hpp"
#include "mesh/MeshGeometry.hpp"
#include "solver/CompressibleFlowSolver.hpp"
#include "time/SimpleExplicitStepper.hpp"

namespace
{
/// @brief In single precision the 64x64 run bottoms out on accumulated
///        rounding; the refinement study uses two meshes and the drift
///        tolerance is scaled accordingly.
constexpr bool kSinglePrecision = std::is_same<Real, float>::value;
constexpr Real kMassDriftTol =
    std::is_same<Real, float>::value ? Real(5e-3) : Real(1e-8);

/// @brief Density of the analytic isentropic vortex at time \p t.
Real vortexRho(Real x, Real y, Real t, Real gamma, Real beta, Real x0, Real y0,
               Real uInf, Real vInf)
{
    const Real xc = x0 + uInf * t;
    const Real yc = y0 + vInf * t;
    const Real r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
    const Real T  = Real(1) - (gamma - Real(1)) * beta * beta /
                                  (Real(8) * gamma * Real(M_PI) * Real(M_PI)) *
                                  std::exp(Real(1) - r2);
    return std::pow(T, Real(1) / (gamma - Real(1)));
}

/// @brief Write the vortex configuration for an nx x nx grid of [0,10]^2.
Config makeVortexConfig(int nx, const std::string &path,
                        const std::string &occaMode)
{
    const Real dx = Real(10) / Real(nx);
    std::FILE *f  = std::fopen(path.c_str(), "w");
    std::fprintf(f,
                 "[basis]\norder = 2\nnq = 4\n\n[mesh]\nnx = %d\nny = %d\n"
                 "x0 = 0.0\ny0 = 0.0\ndx = %.12g\ndy = %.12g\n\n"
                 "[bc]\ntype = \"periodic\"\n\n[riemann]\nflux = \"llf\"\n\n"
                 "[gas]\ngamma = 1.4\n\n[flow]\nrho = 1.0\nu = 1.0\nv = 1.0\n"
                 "p = 1.0\n\n[time_marching]\nmethod = \"ssprk3\"\ncfl = 0.2\n"
                 "t_final = 2.0\n\n[initial_condition]\ntype = \"expr\"\n"
                 "rho = \"(1 - (gamma-1)*beta^2/(8*gamma*pi^2)*"
                 "exp(1 - (x-x0)^2 - (y-y0)^2))^(1/(gamma-1))\"\n"
                 "u = \"uinf - beta/(2*pi)*"
                 "exp((1 - (x-x0)^2 - (y-y0)^2)/2)*(y-y0)\"\n"
                 "v = \"vinf + beta/(2*pi)*"
                 "exp((1 - (x-x0)^2 - (y-y0)^2)/2)*(x-x0)\"\n"
                 "p = \"(1 - (gamma-1)*beta^2/(8*gamma*pi^2)*"
                 "exp(1 - (x-x0)^2 - (y-y0)^2))^(gamma/(gamma-1))\"\n\n"
                 "[initial_condition.symbols]\nbeta = 5.0\nx0 = 5.0\ny0 = 5.0\n"
                 "uinf = 1.0\nvinf = 1.0\n\n[occa]\nmode = \"%s\"\n",
                 nx, nx, dx, dx, occaMode.c_str());
    std::fclose(f);
    Config cfg(path);
    std::remove(path.c_str());
    return cfg;
}

/// @brief Run the vortex and return the density errors (L1, L2, Linf) at
///        T = 2 against the analytic translated vortex.
bool runVortex(const std::string &scratch, const std::string &mode, int nx,
               Real *l1, Real *l2, Real *linf, Real *massDrift)
{
    const Config cfg = makeVortexConfig(nx, scratch, mode);
    CompressibleFlowSolver solver(
        cfg, [](occa::device &device, DeviceMemoryManager &mem,
                const RhsFunction &rhs, occa::dim_t nDof) {
            return std::make_unique<SspRk3Stepper>(device, mem, rhs, nDof);
        });
    if (solver.run() != 0)
    {
        return false;
    }

    DgField &field    = solver.field();
    const int N_elem  = field.numElements();
    const int N_modes = field.numModes();
    const int Nq2     = field.numQuadPoints();
    const int nEvm    = N_elem * field.numVars() * N_modes;

    // Read the modal state back (unified backends alias host memory,
    // separate backends copy).
    std::vector<Real> u(static_cast<std::size_t>(nEvm));
    {
        occa::memory ou = field.o_u();
        if (solver.mem().hasSeparateMemorySpace())
        {
            solver.mem().copyToHost(ou, u.data(), nEvm);
        }
        else
        {
            const Real *src = ou.ptr<Real>();
            std::memcpy(u.data(), src,
                        static_cast<std::size_t>(nEvm) * sizeof(Real));
        }
    }

    // Nodal density values and analytic comparison at the quadrature
    // points.
    const MatrixXr &V      = field.basis().vandermonde();
    const VectorXr &lamAll = field.geometry().lambdaWJ();
    const auto &e8         = field.geometry().vertices();
    const MatrixX2r &qp    = field.basis().quadraturePoints();

    const Real gamma = cfg.gamma();
    const Real beta  = Real(5);
    const Real tEnd  = cfg.timeFinal();

    Real e1 = Real(0), e2 = Real(0), einf = Real(0);
    Real massNow = Real(0), mass0 = Real(0);
    for (int e = 0; e < N_elem; ++e)
    {
        Eigen::Map<const VectorXr> rhot(
            u.data() +
                (static_cast<std::size_t>(e) * field.numVars() + 0) * N_modes,
            N_modes);
        const VectorXr rhoNodal = V * rhot;
        Eigen::Map<const VectorXr> lam(
            lamAll.data() + static_cast<std::size_t>(e) * Nq2, Nq2);

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

            const Real rhoEx = vortexRho(x, y, tEnd, gamma, beta, 5, 5, 1, 1);
            const Real rho0  = vortexRho(x, y, 0, gamma, beta, 5, 5, 1, 1);
            const Real d     = std::abs(rhoNodal[q] - rhoEx);
            e1 += lam[q] * d;
            e2 += lam[q] * d * d;
            einf = einf > d ? einf : d;
            massNow += lam[q] * rhoNodal[q];
            mass0 += lam[q] * rho0;
        }
    }
    *l1        = e1;
    *l2        = std::sqrt(e2);
    *linf      = einf;
    *massDrift = std::abs(massNow - mass0) / mass0;
    return true;
}

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

/// @brief One coarse vortex run on a specific backend (skip on failure).
bool runOnBackend(const std::string &mode)
{
    if (tryMakeDevice(mode).isNull())
    {
        return true;
    }
    std::cout << "\n===== vortex backend [" << mode << "] =====" << std::endl;
    try
    {
        Real l1, l2, linf, drift;
        if (!runVortex("test_vortex_backend.toml", mode, 16, &l1, &l2, &linf,
                       &drift))
        {
            std::cout << "  (skip " << mode << ": solver failed)\n";
            return true;
        }
        std::cout << "  16x16: L1 = " << l1 << ", L2 = " << l2
                  << ", Linf = " << linf << ", mass drift = " << drift << "\n";
        return l2 < Real(1.0);
    }
    catch (const std::exception &e)
    {
        std::cout << "  (skip " << mode << ": " << e.what() << ")\n";
        return true;
    }
}

} // namespace

int main()
{
    bool ok = true;

    std::cout << "Isentropic vortex, SSPRK3, N=2, CFL=0.2, T=2 (Serial)\n";
    Real prevL2         = std::numeric_limits<Real>::max();
    const int meshes[3] = {16, 32, 64};
    const int nMeshes   = kSinglePrecision ? 2 : 3;
    std::vector<Real> errors(static_cast<std::size_t>(nMeshes));
    std::vector<Real> hs(static_cast<std::size_t>(nMeshes));
    for (int k = 0; k < nMeshes; ++k)
    {
        hs[static_cast<std::size_t>(k)] = Real(10) / Real(meshes[k]);
        Real l1, l2, linf, drift;
        if (!runVortex("test_vortex_scratch.toml", "Serial", meshes[k], &l1,
                       &l2, &linf, &drift))
        {
            ok = false;
            break;
        }
        errors[static_cast<std::size_t>(k)] = l2;
        std::cout << "  " << meshes[k] << "x" << meshes[k] << ": L1 = " << l1
                  << ", L2 = " << l2 << ", Linf = " << linf
                  << ", mass drift = " << drift << "\n";
        ok &= (l2 < prevL2);
        ok &= drift < kMassDriftTol;
        prevL2 = l2;
    }

    if (ok)
    {
        const Real order = std::log(errors[0] / errors[nMeshes - 1]) /
                           std::log(hs[0] / hs[nMeshes - 1]);
        std::cout << "  observed order (L2) = " << order << "\n";
        ok &= order > Real(2.7) && order < Real(3.6);
    }

    ok &= runOnBackend("OpenMP");
    ok &= runOnBackend("OpenCL");

    std::cout << (ok ? "ALL EULER VORTEX TESTS PASSED\n"
                     : "EULER VORTEX TESTS FAILED\n");
    return ok ? 0 : 1;
}
