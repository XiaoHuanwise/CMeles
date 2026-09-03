/// @file VortexCommon.hpp
/// @brief Shared helpers of the isentropic-vortex tests: analytic vortex,
///        configuration writer, error norms and the generic solver runner.
///
/// Setup (classic Shu vortex): domain [0, 10]^2, mean flow (1, 1), vortex
/// centre (5, 5), beta = 5, periodic boundaries, N = 2 (nq = 4). The
/// analytic solution advects the vortex with the mean flow.

#pragma once

#include <Eigen/Dense>
#include <occa.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "basis/BasisFunctions2D.hpp"
#include "common/Types.hpp"
#include "config/Config.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "dg/DgField.hpp"
#include "mesh/MeshGeometry.hpp"
#include "solver/CompressibleFlowSolver.hpp"
#include "time/StepperBase.hpp"

namespace vortex
{

/// @brief Options of one vortex run.
struct Options
{
    int nx               = 16;        ///< Mesh resolution (nx x nx).
    std::string method   = "ssprk3";  ///< Time-marching method.
    Real dt              = Real(0);   ///< Fixed dt; <= 0 uses the CFL estimate.
    Real cfl             = Real(0.2); ///< CFL number.
    Real tFinal          = Real(2);   ///< Final time.
    Real rtol            = Real(1e-6); ///< Adaptive / dual-time tolerance.
    Real atol            = Real(1e-6);
    int maxPseudoSteps   = 100; ///< Dual-time cap.
    std::string occaMode = "Serial";
};

/// @brief Density of the analytic isentropic vortex at time \p t.
inline Real vortexRho(Real x, Real y, Real t, Real gamma, Real beta, Real x0,
                      Real y0, Real uInf, Real vInf)
{
    const Real xc = x0 + uInf * t;
    const Real yc = y0 + vInf * t;
    const Real r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
    const Real T  = Real(1) - (gamma - Real(1)) * beta * beta /
                                  (Real(8) * gamma * Real(M_PI) * Real(M_PI)) *
                                  std::exp(Real(1) - r2);
    return std::pow(T, Real(1) / (gamma - Real(1)));
}

/// @brief Write the vortex configuration and parse it.
inline Config makeConfig(const Options &opt, const std::string &path)
{
    const Real dx = Real(10) / Real(opt.nx);
    std::FILE *f  = std::fopen(path.c_str(), "w");
    std::fprintf(f,
                 "[basis]\norder = 2\nnq = 4\n\n[mesh]\nnx = %d\nny = %d\n"
                 "x0 = 0.0\ny0 = 0.0\ndx = %.12g\ndy = %.12g\n\n"
                 "[bc]\ntype = \"periodic\"\n\n[riemann]\nflux = \"llf\"\n\n"
                 "[gas]\ngamma = 1.4\n\n[flow]\nrho = 1.0\nu = 1.0\nv = 1.0\n"
                 "p = 1.0\n\n[time_marching]\nmethod = \"%s\"\ncfl = %.12g\n"
                 "dt = %.12g\nt_final = %.12g\nrtol = %.12g\natol = %.12g\n"
                 "max_pseudo_steps = %d\n\n[initial_condition]\n"
                 "type = \"expr\"\n"
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
                 opt.nx, opt.nx, dx, dx, opt.method.c_str(), opt.cfl, opt.dt,
                 opt.tFinal, opt.rtol, opt.atol, opt.maxPseudoSteps,
                 opt.occaMode.c_str());
    std::fclose(f);
    Config cfg(path);
    std::remove(path.c_str());
    return cfg;
}

/// @brief Density errors (L1, L2, Linf) and relative mass drift of the
///        final state against the analytic vortex at tFinal.
struct Errors
{
    Real l1        = Real(0);
    Real l2        = Real(0);
    Real linf      = Real(0);
    Real massDrift = Real(0);
    int steps      = 0;
};

/// @brief Run one configured solver and evaluate the errors.
inline Errors runAndMeasure(
    const Config &cfg, const CompressibleFlowSolver::StepperFactory &factory)
{
    CompressibleFlowSolver solver(cfg, factory);
    Errors err;
    if (solver.run() != 0)
    {
        err.l2 = std::numeric_limits<Real>::max();
        return err;
    }
    err.steps = solver.stepCount();

    DgField &field    = solver.field();
    const int N_elem  = field.numElements();
    const int N_modes = field.numModes();
    const int Nq2     = field.numQuadPoints();
    const int nEvm    = N_elem * field.numVars() * N_modes;

    // Read the modal state back.
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

    const MatrixXr &V      = field.basis().vandermonde();
    const VectorXr &lamAll = field.geometry().lambdaWJ();
    const auto &e8         = field.geometry().vertices();
    const MatrixX2r &qp    = field.basis().quadraturePoints();

    const Real gamma = cfg.gamma();
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

            const Real rhoEx = vortexRho(x, y, tEnd, gamma, 5, 5, 5, 1, 1);
            const Real rho0  = vortexRho(x, y, 0, gamma, 5, 5, 5, 1, 1);
            const Real d     = std::abs(rhoNodal[q] - rhoEx);
            e1 += lam[q] * d;
            e2 += lam[q] * d * d;
            einf = einf > d ? einf : d;
            massNow += lam[q] * rhoNodal[q];
            mass0 += lam[q] * rho0;
        }
    }
    err.l1        = e1;
    err.l2        = std::sqrt(e2);
    err.linf      = einf;
    err.massDrift = std::abs(massNow - mass0) / mass0;
    return err;
}

/// @brief Default factory for simple steppers (Euler / SSPRK3).
template <class Stepper> CompressibleFlowSolver::StepperFactory simpleFactory()
{
    return [](occa::device &device, DeviceMemoryManager &mem,
              const RhsFunction &rhs, occa::dim_t nDof) {
        return std::make_unique<Stepper>(device, mem, rhs, nDof);
    };
}

} // namespace vortex
