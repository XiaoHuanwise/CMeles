/// @file test_basis.cpp
/// @brief Numerical verification tests for BasisFunctions1D and
/// BasisFunctions2D.

#include "basis/BasisFunctions1D.hpp"
#include "basis/BasisFunctions2D.hpp"
#include "common/Types.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

// ---------------------------------------------------------------------------
// Helper: relative error
// ---------------------------------------------------------------------------
static Real relErr(Real computed, Real expected)
{
    if (std::abs(expected) < RealEpsilon)
        return std::abs(computed);
    return std::abs(computed - expected) / std::abs(expected);
}

// ---------------------------------------------------------------------------
// Test 1: Gauss-Legendre quadrature nodes and weights against known values
// ---------------------------------------------------------------------------
bool testGaussLegendre()
{
    std::cout << "\n=== Test 1: Gauss-Legendre quadrature ===\n";

    // Known values from SciPy / standard tables
    struct
    {
        int Nq;
        VectorXr points;
        VectorXr weights;
    } ref[] = {
        {1, VectorXr::Zero(1), VectorXr::Ones(1) * Real(2)},
        {2,
         (VectorXr(2) << Real(-0.5773502691896257), Real(0.5773502691896257))
             .finished(),
         (VectorXr(2) << Real(1), Real(1)).finished()},
        {3,
         (VectorXr(3) << Real(-0.7745966692414834), Real(0), Real(0.7745966692414834))
             .finished(),
         (VectorXr(3) << Real(0.5555555555555556), Real(0.8888888888888888),
          Real(0.5555555555555556))
             .finished()},
        {4,
         (VectorXr(4) << Real(-0.8611363115940526), Real(-0.3399810435848563),
          Real(0.3399810435848563), Real(0.8611363115940526))
             .finished(),
         (VectorXr(4) << Real(0.3478548451374538), Real(0.6521451548625461),
          Real(0.6521451548625461), Real(0.3478548451374538))
             .finished()},
    };

    bool allPass          = true;
    constexpr Real tol    = Real(1e3) * RealEpsilon;

    for (const auto &r : ref)
    {
        BasisFunctions1D basis(/*N=*/0, r.Nq); // N=0 is enough for quad test
        Real maxErrPts = Real(0), maxErrWts = Real(0);

        for (int i = 0; i < r.Nq; ++i)
        {
            maxErrPts =
                std::max(maxErrPts, std::abs(basis.points()(i) - r.points(i)));
            maxErrWts = std::max(maxErrWts,
                                 std::abs(basis.weights()(i) - r.weights(i)));
        }

        bool pass = (maxErrPts < tol) && (maxErrWts < tol);
        std::cout << "  Nq=" << r.Nq << "  pts_err=" << std::scientific
                  << maxErrPts << "  wts_err=" << maxErrWts << "  "
                  << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass)
            allPass = false;
    }

    return allPass;
}

// ---------------------------------------------------------------------------
// Test 2: Vandermonde orthogonality check
// ---------------------------------------------------------------------------
bool testVandermondeOrthogonality()
{
    std::cout << "\n=== Test 2: Vandermonde orthogonality ===\n";

    bool allPass          = true;
    constexpr Real tol    = Real(1e3) * RealEpsilon;

    for (int N = 1; N <= 5; ++N)
    {
        int Nq = N + 2; // sufficiently many quadrature points
        BasisFunctions1D basis(N, Nq);

        // Check orthonormality of normalized Legendre polynomials:
        // sum_k w_k * tilde{P}_i(xi_k) * tilde{P}_j(xi_k) = delta_{ij}
        MatrixXrCol I_ref = MatrixXrCol::Identity(N + 1, N + 1);
        MatrixXrCol M = basis.vandermonde().transpose() *
                        basis.weights().asDiagonal() * basis.vandermonde();

        Real orthoErr = (M - I_ref).norm() / (N + Real(1));
        bool pass     = (orthoErr < tol);
        std::cout << "  N=" << N << "  Nq=" << Nq
                  << "  |V^T W V - I|=" << std::scientific << orthoErr << "  "
                  << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass)
            allPass = false;
    }

    return allPass;
}

// ---------------------------------------------------------------------------
// Test 3: Vandermonde derivative consistency check
// ---------------------------------------------------------------------------
bool testVandermondeDerivative()
{
    std::cout
        << "\n=== Test 3: Vandermonde derivative (finite difference) ===\n";

    bool allPass          = true;
    // Use precision-adapted step size to balance truncation and round-off errors.
    // For double: h ≈ 1e-5, for float: h ≈ 5e-3.
    const Real h   = std::cbrt(RealEpsilon);
    // Tolerance: O(h^2) truncation with safety factor for high-order polynomials.
    const Real tol = h * h * Real(1e3);  // ~1e-7 for double, ~2e-2 for float

    for (int N = 1; N <= 5; ++N)
    {
        int Nq = N + 2;
        BasisFunctions1D basis(N, Nq);

        Real maxErr = Real(0);
        for (int i = 0; i < Nq; ++i)
        {
            Real xi = basis.points()(i);

            // Use finite difference to approximate P'_j(xi)
            VectorXr P_plus, dP_plus;
            BasisFunctions1D::evalPolynomials(xi + h, N, P_plus, dP_plus);
            VectorXr P_minus, dP_minus;
            BasisFunctions1D::evalPolynomials(xi - h, N, P_minus, dP_minus);

            for (int j = 1; j <= N; ++j)
            {
                Real fd_deriv = (P_plus(j) - P_minus(j)) / (Real(2) * h);
                Real exact    = basis.vandermondeDerivative()(i, j);
                Real err      = std::abs(fd_deriv - exact);
                maxErr        = std::max(maxErr, err);
            }
        }

        bool pass = (maxErr < tol);
        std::cout << "  N=" << N << "  Nq=" << Nq
                  << "  max_FD_err=" << std::scientific << maxErr << "  "
                  << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass)
            allPass = false;
    }

    return allPass;
}

// ---------------------------------------------------------------------------
// Test 4: Pascal-triangle index mapping (round-trip)
// ---------------------------------------------------------------------------
bool testPascalTriangleMapping()
{
    std::cout << "\n=== Test 4: Pascal-triangle index mapping ===\n";

    bool allPass = true;

    for (int N = 0; N <= 5; ++N)
    {
        BasisFunctions1D basis1D(N, N + 1);
        BasisFunctions2D basis2D(basis1D, N);

        int N_base   = basis2D.numBases();
        int expected = (N + 1) * (N + 2) / 2;

        bool sizesOk = (N_base == expected);

        // Verify round-trip: l -> (i,j) -> l
        bool roundtripOk = true;
        for (int l = 0; l < N_base; ++l)
        {
            auto [i, j] = basis2D.indexPair(l);
            int l2      = basis2D.linearIndex(i, j);
            if (l2 != l)
            {
                roundtripOk = false;
                break;
            }
        }

        // Verify each (i,j) satisfies i+j <= N and maps correctly
        bool allPairsOk = true;
        const auto &ij  = basis2D.indexMap();
        for (int l = 0; l < N_base; ++l)
        {
            int i = ij(l, 0), j = ij(l, 1);
            if (i + j > N || basis2D.linearIndex(i, j) != l)
            {
                allPairsOk = false;
                break;
            }
        }

        bool pass = sizesOk && roundtripOk && allPairsOk;
        std::cout << "  N=" << N << "  N_base=" << N_base << " (expected "
                  << expected << ")"
                  << "  roundtrip=" << (roundtripOk ? "OK" : "FAIL") << "  "
                  << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass)
            allPass = false;
    }

    return allPass;
}

// ---------------------------------------------------------------------------
// Test 5: 2D Vandermonde orthogonality
// ---------------------------------------------------------------------------
bool testVandermonde2DOrthogonality()
{
    std::cout << "\n=== Test 5: 2D Vandermonde orthogonality ===\n";

    bool allPass          = true;
    constexpr Real tol    = Real(1e3) * RealEpsilon;

    for (int N = 1; N <= 3; ++N)
    {
        int Nq = N + 2;
        BasisFunctions1D basis1D(N, Nq);
        BasisFunctions2D basis2D(basis1D, N);

        // M = V2D^T * W * V2D should equal identity
        const auto &V2D = basis2D.vandermonde();
        const auto &W   = basis2D.quadratureWeights();
        int N_base      = basis2D.numBases();

        MatrixXrCol M     = V2D.transpose() * W.asDiagonal() * V2D;
        MatrixXrCol I_ref = MatrixXrCol::Identity(N_base, N_base);

        Real orthoErr = (M - I_ref).norm() / static_cast<Real>(N_base);
        bool pass     = (orthoErr < tol);

        std::cout << "  N=" << N << "  Nq=" << Nq << "  Nq^2=" << Nq * Nq
                  << "  N_base=" << N_base
                  << "  |V^T W V - I|/N_base=" << std::scientific << orthoErr
                  << "  " << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass)
            allPass = false;
    }

    return allPass;
}

// ---------------------------------------------------------------------------
// Test 6: Modal-nodal round-trip
// ---------------------------------------------------------------------------
bool testModalNodalRoundtrip()
{
    std::cout << "\n=== Test 6: Modal-nodal round-trip ===\n";

    bool allPass          = true;
    constexpr Real tol    = Real(1e3) * RealEpsilon;

    for (int N = 1; N <= 3; ++N)
    {
        int Nq = N + 2;
        BasisFunctions1D basis1D(N, Nq);
        BasisFunctions2D basis2D(basis1D, N);

        // Random modal coefficients
        VectorXr u_hat = VectorXr::Random(basis2D.numBases());

        // Forward: u = V2D * u_hat
        VectorXr u = basis2D.modalToNodal(u_hat);

        // Check against direct computation
        VectorXr u_direct = basis2D.vandermonde() * u_hat;
        Real fwdErr       = (u - u_direct).norm() / u.norm();

        // Backward: u_hat_proj = V2D^T * W * u
        VectorXr u_hat_proj = basis2D.nodalToModal(u);

        // For orthonormal basis, should recover u_hat exactly
        Real rttErr = (u_hat - u_hat_proj).norm() / u_hat.norm();

        bool pass = (fwdErr < tol) && (rttErr < tol);
        std::cout << "  N=" << N << "  fwd_err=" << std::scientific << fwdErr
                  << "  rtt_err=" << rttErr << "  " << (pass ? "PASS" : "FAIL")
                  << "\n";
        if (!pass)
            allPass = false;
    }

    return allPass;
}

// ---------------------------------------------------------------------------
// Test 7: Sum-factorization consistency with Vandermonde
// ---------------------------------------------------------------------------
bool testSumFactorization()
{
    std::cout << "\n=== Test 7: Sum-factorization consistency ===\n";

    bool allPass          = true;
    constexpr Real tol    = Real(1e3) * RealEpsilon;

    for (int N = 1; N <= 3; ++N)
    {
        int Nq = N + 2;
        BasisFunctions1D basis1D(N, Nq);
        BasisFunctions2D basis2D(basis1D, N);

        // Random field at quadrature points (column-major: G(j,i) = G(r_i, s_j))
        MatrixXrCol G = MatrixXrCol::Random(Nq, Nq);

        // --- r-gradient ---
        // Sum-factorized
        VectorXr R_r_sf = basis2D.sumFactorR(G);

        // Direct: R_r = dV2D_r^T * G_flat
        const auto &dV2D_r = basis2D.vandermondeDerivativeR();
        const auto &dV2D_s = basis2D.vandermondeDerivativeS();

        // Build G_flat in the same column-major-flattened-by-dV2D order: k = j*Nq + i
        VectorXr G_flat(Nq * Nq);
        for (int j = 0; j < Nq; ++j)
        {
            for (int i = 0; i < Nq; ++i)
            {
                G_flat(j * Nq + i) = G(j, i);
            }
        }
        VectorXr R_r_vdm = dV2D_r.transpose() * G_flat;

        Real errR =
            (R_r_sf - R_r_vdm).norm() / std::max(RealEpsilon, R_r_vdm.norm());

        // --- s-gradient ---
        VectorXr R_s_sf  = basis2D.sumFactorS(G);
        VectorXr R_s_vdm = dV2D_s.transpose() * G_flat;

        Real errS =
            (R_s_sf - R_s_vdm).norm() / std::max(RealEpsilon, R_s_vdm.norm());

        bool pass = (errR < tol) && (errS < tol);
        std::cout << "  N=" << N << "  Nq=" << Nq
                  << "  r_err=" << std::scientific << errR << "  s_err=" << errS
                  << "  " << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass)
            allPass = false;
    }

    return allPass;
}

// ============================================================================
// Main
// ============================================================================
int main()
{
    std::cout << std::setprecision(4) << std::scientific;
    std::cout << "Basis Functions Verification Suite\n";

    bool ok = true;
    ok &= testGaussLegendre();
    ok &= testVandermondeOrthogonality();
    ok &= testVandermondeDerivative();
    ok &= testPascalTriangleMapping();
    ok &= testVandermonde2DOrthogonality();
    ok &= testModalNodalRoundtrip();
    ok &= testSumFactorization();

    std::cout << "\n===========================\n";
    std::cout << (ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    std::cout << "===========================\n";

    return ok ? 0 : 1;
}
