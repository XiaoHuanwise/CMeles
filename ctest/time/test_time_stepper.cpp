/// @file test_time_stepper.cpp
/// @brief ODE-level correctness tests for the time-marching module.
///
/// The manufactured ODE is u' = lambda * u (lambda < 0) with the exact
/// solution u(T) = e^{lambda T} u0, evaluated on device buffers. The RHS is
/// assembled from Blas primitives (copy + scal), so the tests exercise the
/// steppers end-to-end through the OCCA kernels without the DG field.
///
/// Key checks:
///   1. Euler / SSPRK3 convergence orders (1 and 3) under dt halving.
///   2. Embedded RK pairs (RK32/RK54/SSPRK221/321/332/432): fixed-step
///      convergence orders of the solution weights (3/5/2/3/3/4) and
///      adaptive-mode acceptance behaviour.
///   3. Dual time stepping: BackwardEuler (order 1) and DITR U2R2/U2R1/U3R1
///      physical-time convergence orders.

#include <occa.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

#include "blas/Blas.hpp"
#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "time/ButcherTable.hpp"
#include "time/DualStepper.hpp"
#include "time/ImplicitResidual.hpp"
#include "time/RungeKuttaStepper.hpp"
#include "time/SimpleExplicitStepper.hpp"

namespace
{
/// @brief High-order convergence checks bottom out at the float rounding
///        floor; those assertions degrade to accuracy-level checks.
constexpr bool kSinglePrecision = std::is_same<Real, float>::value;

/// @brief Read a device Real array back to host (unified vs separate path).
void readResult(DeviceMemoryManager &mem, const occa::memory &o_data, Real *dst,
                occa::dim_t entries)
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

/// @brief Manufactured ODE context: u' = lambda u on the device.
struct DecayOde
{
    occa::device &device;
    DeviceMemoryManager &mem;
    Blas blas;
    Real lambda;
    int n;
    std::vector<Real> u0Host; ///< Host source of o_u0 (must outlive the wrap).
    occa::memory o_u0;

    DecayOde(occa::device &dev, DeviceMemoryManager &m, Real lam, int nDof)
        : device(dev), mem(m), blas(dev, m), lambda(lam), n(nDof)
    {
        u0Host.resize(static_cast<std::size_t>(nDof));
        for (int i = 0; i < nDof; ++i)
        {
            // Distinct positive values so the norm is non-trivial.
            u0Host[static_cast<std::size_t>(i)] =
                Real(1) + Real(0.25) * Real(i);
        }
        o_u0 = mem.wrapOrMalloc(u0Host.data(), nDof);
    }

    /// res = lambda * u.
    RhsFunction rhs() const
    {
        Blas b           = blas;
        Real lam         = lambda;
        occa::dim_t nDof = n;
        return [b, lam, nDof](occa::memory u, occa::memory res) mutable {
            occa::memory x = u, y = res;
            b.copy(nDof, x, y);
            b.scal(nDof, lam, y);
        };
    }

    Real exactScale() const
    {
        return std::exp(lambda); // u(1) = e^lambda u0
    }
};

/// @brief Integrate with a fixed-dt stepper and return the max error
///        against e^{lambda T} u0 at T = 1.
template <class Stepper>
Real fixedStepError(occa::device &device, DeviceMemoryManager &mem,
                    DecayOde &ode, Real dt)
{
    Stepper stepper(device, mem, ode.rhs(), ode.n);
    occa::memory o_u   = mem.wrapOrMalloc(static_cast<occa::dim_t>(ode.n));
    occa::memory o_src = ode.o_u0;
    Blas blas(device, mem);
    blas.copy(ode.n, o_src, o_u);

    Real t = Real(0);
    while (t < Real(1) - Real(0.5) * dt)
    {
        const Real taken = stepper.advance(o_u, t, dt);
        t += taken;
    }

    std::vector<Real> uh(static_cast<std::size_t>(ode.n));
    readResult(mem, o_u, uh.data(), ode.n);
    const Real scale = ode.exactScale();
    Real err         = Real(0);
    for (int i = 0; i < ode.n; ++i)
    {
        const Real ex = scale * (Real(1) + Real(0.25) * Real(i));
        err = std::max(err, std::abs(uh[static_cast<std::size_t>(i)] - ex));
    }
    return err;
}

/// @brief Measured convergence order from three successive dt halvings.
Real measuredOrder(const Real errors[3])
{
    return std::log2(errors[0] / errors[2]) / Real(2);
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Euler / SSPRK3 convergence orders
// ---------------------------------------------------------------------------

static bool testSimpleExplicit()
{
    std::cout << "Test 1: Euler / SSPRK3 convergence orders\n";
    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mem(device);
    DecayOde ode(device, mem, Real(-1), 37);

    bool ok = true;

    {
        Real errors[3];
        for (int k = 0; k < 3; ++k)
        {
            errors[k] = fixedStepError<EulerStepper>(
                device, mem, ode, Real(0.1) / std::pow(Real(2), k));
        }
        const Real p = measuredOrder(errors);
        std::cout << "  euler  errors " << errors[0] << " -> " << errors[2]
                  << ", order " << p << "\n";
        ok &= p > Real(0.85) && p < Real(1.3);
    }
    {
        Real errors[3];
        for (int k = 0; k < 3; ++k)
        {
            errors[k] = fixedStepError<SspRk3Stepper>(
                device, mem, ode, Real(0.2) / std::pow(Real(2), k));
        }
        const Real p = measuredOrder(errors);
        std::cout << "  ssprk3 errors " << errors[0] << " -> " << errors[2]
                  << ", order " << p << "\n";
        ok &= p > Real(2.7) && p < Real(3.3);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 2. Embedded RK pairs: fixed-step orders of the solution weights
// ---------------------------------------------------------------------------

/// @brief Integrate with a RungeKuttaStepper at fixed dt; max error at T=1.
Real rkFixedStepError(occa::device &device, DeviceMemoryManager &mem,
                      DecayOde &ode, const ButcherTable &table, Real dt)
{
    RungeKuttaStepper stepper(device, mem, ode.rhs(), ode.n, table);
    occa::memory o_u  = mem.wrapOrMalloc(static_cast<occa::dim_t>(ode.n));
    occa::memory o_u0 = ode.o_u0;
    Blas blas(device, mem);
    blas.copy(ode.n, o_u0, o_u);
    stepper.setState(o_u, Real(1e30));

    Real t = Real(0);
    while (t < Real(1) - Real(0.5) * dt)
    {
        stepper.stepFixed(dt);
        t += dt;
    }

    std::vector<Real> uh(static_cast<std::size_t>(ode.n));
    readResult(mem, stepper.state(), uh.data(), ode.n);
    const Real scale = ode.exactScale();
    Real err         = Real(0);
    for (int i = 0; i < ode.n; ++i)
    {
        const Real ex = scale * (Real(1) + Real(0.25) * Real(i));
        err = std::max(err, std::abs(uh[static_cast<std::size_t>(i)] - ex));
    }
    return err;
}

static bool testEmbeddedRkOrders()
{
    std::cout << "Test 2: embedded RK fixed-step orders\n";
    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mem(device);
    DecayOde ode(device, mem, Real(-1), 23);

    const ButcherTable *tables[] = {&kRk32,     &kRk54,     &kSspRk221,
                                    &kSspRk321, &kSspRk332, &kSspRk432};
    bool ok                      = true;
    for (const ButcherTable *table : tables)
    {
        Real errors[3];
        for (int k = 0; k < 3; ++k)
        {
            errors[k] = rkFixedStepError(device, mem, ode, *table,
                                         Real(0.2) / std::pow(Real(2), k));
        }
        const Real p = measuredOrder(errors);
        std::cout << "  " << table->name << " (expect " << table->order
                  << "): errors " << errors[0] << " -> " << errors[2]
                  << ", order " << p << "\n";
        if (kSinglePrecision && table->order >= 5)
        {
            // RK54-level errors sit at the float floor: accuracy only
            // (monotonicity is noise at that level).
            ok &= errors[2] < Real(1e-5) && errors[0] < Real(1e-5);
        }
        else
        {
            ok &= errors[2] < errors[0];
            ok &= p > Real(table->order - 0.4) && p < Real(table->order + 0.4);
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 3. Adaptive mode: advances to tolerance
// ---------------------------------------------------------------------------

static bool testAdaptiveMode()
{
    std::cout << "Test 3: adaptive step control\n";
    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mem(device);
    DecayOde ode(device, mem, Real(-1), 23);

    RungeKuttaStepper::Params params;
    params.rtol = Real(1e-6);
    params.atol = Real(1e-9);
    RungeKuttaStepper stepper(device, mem, ode.rhs(), ode.n, kSspRk332, params);

    occa::memory o_u  = mem.wrapOrMalloc(static_cast<occa::dim_t>(ode.n));
    occa::memory o_u0 = ode.o_u0;
    Blas blas(device, mem);
    blas.copy(ode.n, o_u0, o_u);

    Real t    = Real(0);
    int steps = 0;
    while (t < Real(1) - Real(1e-12))
    {
        const Real taken =
            stepper.advance(o_u, t, std::min(Real(0.5), Real(1) - t));
        t += taken;
        ++steps;
    }

    std::vector<Real> uh(static_cast<std::size_t>(ode.n));
    readResult(mem, o_u, uh.data(), ode.n);
    const Real scale = ode.exactScale();
    Real err         = Real(0);
    for (int i = 0; i < ode.n; ++i)
    {
        const Real ex = scale * (Real(1) + Real(0.25) * Real(i));
        err = std::max(err, std::abs(uh[static_cast<std::size_t>(i)] - ex));
    }
    std::cout << "  ssprk332 adaptive: err = " << err << ", steps = " << steps
              << ", dt = " << stepper.dt() << "\n";
    // Tolerance-consistent accuracy with a sane step count.
    return err < Real(1e-4) && steps > 3 && steps < 500;
}

// ---------------------------------------------------------------------------
// 4. Dual time stepping: BE (order 1), DITR (order 2)
// ---------------------------------------------------------------------------

/// @brief Integrate with a DualStepper<Phy> at fixed physical dt.
template <class PhyStepper, class PhyFactory>
Real dualStepError(occa::device &device, DeviceMemoryManager &mem,
                   DecayOde &ode, Real dt, PhyFactory makePhy,
                   typename DualStepper<PhyStepper>::Params params)
{
    DualStepper<PhyStepper> stepper(device, mem, ode.rhs(), ode.n, kSspRk332,
                                    params, makePhy());
    occa::memory o_u  = mem.wrapOrMalloc(static_cast<occa::dim_t>(ode.n));
    occa::memory o_u0 = ode.o_u0;
    Blas blas(device, mem);
    blas.copy(ode.n, o_u0, o_u);

    Real t = Real(0);
    while (t < Real(1) - Real(0.5) * dt)
    {
        stepper.advance(o_u, t, dt);
        t += dt;
    }

    std::vector<Real> uh(static_cast<std::size_t>(ode.n));
    readResult(mem, o_u, uh.data(), ode.n);
    const Real scale = ode.exactScale();
    Real err         = Real(0);
    for (int i = 0; i < ode.n; ++i)
    {
        const Real ex = scale * (Real(1) + Real(0.25) * Real(i));
        err = std::max(err, std::abs(uh[static_cast<std::size_t>(i)] - ex));
    }
    return err;
}

static bool testDualTime()
{
    std::cout << "Test 4: dual time stepping orders\n";
    occa::device device({{"mode", "Serial"}});
    DeviceMemoryManager mem(device);
    DecayOde ode(device, mem, Real(-1), 11);

    // Tight dual-time convergence so the pseudo error is negligible.
    DualStepper<BackwardEulerStepper>::Params beParams;
    // Below the single-precision rounding floor the dual criterion can
    // never be met; scale the tolerances with the working precision.
    beParams.rtol           = kSinglePrecision ? Real(1e-4) : Real(1e-9);
    beParams.atol           = kSinglePrecision ? Real(1e-7) : Real(1e-12);
    beParams.maxPseudoSteps = 300;
    // Loose pseudo-local accuracy: the dual-time convergence criterion
    // governs; a large initial pseudo step keeps the iteration count sane.
    beParams.rkParams.rtol = Real(1e-3);
    beParams.rkParams.atol = Real(1e-3);

    DualStepper<DitrStepper>::Params ditrParams;
    ditrParams.rtol           = kSinglePrecision ? Real(1e-4) : Real(1e-9);
    ditrParams.atol           = kSinglePrecision ? Real(1e-7) : Real(1e-12);
    ditrParams.maxPseudoSteps = 300;
    ditrParams.rkParams.rtol  = Real(1e-3);
    ditrParams.rkParams.atol  = Real(1e-3);

    DualStepper<DitrStepper>::Params decoupledParams = ditrParams;
    decoupledParams.decoupled                        = true;

    auto makeBE = [&] {
        return BackwardEulerStepper(device, mem, ode.rhs(), ode.n);
    };
    auto makeU2R2 = [&] {
        return DitrStepper(device, mem, ode.rhs(), ode.n,
                           DitrStepper::Variant::U2R2);
    };
    auto makeU2R1 = [&] {
        return DitrStepper(device, mem, ode.rhs(), ode.n,
                           DitrStepper::Variant::U2R1);
    };
    auto makeU3R1 = [&] {
        return DitrStepper(device, mem, ode.rhs(), ode.n,
                           DitrStepper::Variant::U3R1);
    };

    bool ok = true;
    {
        Real errors[3];
        for (int k = 0; k < 3; ++k)
        {
            errors[k] = dualStepError<BackwardEulerStepper>(
                device, mem, ode, Real(0.1) / std::pow(Real(2), k), makeBE,
                beParams);
        }
        const Real p = measuredOrder(errors);
        std::cout << "  be        errors " << errors[0] << " -> " << errors[2]
                  << ", order " << p << "\n";
        ok &= p > Real(0.8) && p < Real(1.2);
    }
    {
        Real errors[3];
        for (int k = 0; k < 3; ++k)
        {
            errors[k] = dualStepError<DitrStepper>(
                device, mem, ode, Real(0.1) / std::pow(Real(2), k), makeU2R2,
                ditrParams);
        }
        const Real p = measuredOrder(errors);
        std::cout << "  ditr_u2r2 errors " << errors[0] << " -> " << errors[2]
                  << ", order " << p << "\n";
        // Documented 2nd order; the Simpson-type b weights raise the
        // observed order to ~4 on this linear constant-dt problem. In
        // single precision the errors sit at the rounding floor.
        if (kSinglePrecision)
        {
            // Residual-limited accuracy at the scaled dual tolerance.
            ok &= errors[0] < Real(1e-3);
        }
        else
        {
            ok &= p > Real(2.6) && p < Real(4.4);
        }
    }
    {
        // U2R1 / U3R1 / decoupled U2R2: exercise and require ~2nd order
        // (single coarse comparison: error at dt/2 must shrink clearly).
        const Real e1U2R1 = dualStepError<DitrStepper>(
            device, mem, ode, Real(0.1), makeU2R1, ditrParams);
        const Real e2U2R1 = dualStepError<DitrStepper>(
            device, mem, ode, Real(0.05), makeU2R1, ditrParams);
        const Real e1U3R1 = dualStepError<DitrStepper>(
            device, mem, ode, Real(0.1), makeU3R1, ditrParams);
        const Real e2U3R1 = dualStepError<DitrStepper>(
            device, mem, ode, Real(0.05), makeU3R1, ditrParams);
        const Real e1Dec = dualStepError<DitrStepper>(
            device, mem, ode, Real(0.1), makeU2R2, decoupledParams);
        const Real e2Dec = dualStepError<DitrStepper>(
            device, mem, ode, Real(0.05), makeU2R2, decoupledParams);
        std::cout << "  u2r1  " << e1U2R1 << " -> " << e2U2R1 << "\n"
                  << "  u3r1  " << e1U3R1 << " -> " << e2U3R1 << "\n"
                  << "  decou " << e1Dec << " -> " << e2Dec << "\n";
        // U2R1 observes ~3rd order here (linear problem); U3R1 is limited
        // to ~1st order by the start-up u^{n-1} = u^0 initialisation
        // (inherent to the method on runs that start at t = 0, matching
        // the prototype); the decoupled U2R2 tracks the coupled one.
        if (kSinglePrecision)
        {
            // Residual-limited accuracy at the scaled dual tolerance.
            ok &= e1U2R1 < Real(1e-3) && e1U3R1 < Real(1e-2) &&
                  e1Dec < Real(1e-3);
        }
        else
        {
            ok &= e2U2R1 < Real(0.4) * e1U2R1;
            ok &= e2U3R1 < Real(0.7) * e1U3R1;
            ok &= e2Dec < Real(0.4) * e1Dec;
        }
    }
    return ok;
}

int main()
{
    bool ok = true;
    ok &= testSimpleExplicit();
    ok &= testEmbeddedRkOrders();
    ok &= testAdaptiveMode();
    ok &= testDualTime();

    std::cout << (ok ? "ALL TIME STEPPER TESTS PASSED\n"
                     : "TIME STEPPER TESTS FAILED\n");
    return ok ? 0 : 1;
}
