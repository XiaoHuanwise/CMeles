/// @file BasisFunctions1D.cpp
/// @brief Implementation of 1D basis functions, quadrature, and Vandermonde matrices.

#include "BasisFunctions1D.hpp"
#include "core/DeviceMemoryManager.hpp"

#include <cmath>
#include <stdexcept>

BasisFunctions1D::BasisFunctions1D(int N, int Nq) : N_(N), Nq_(Nq) {
    if (N < 0) {
        throw std::invalid_argument("BasisFunctions1D: N must be >= 0");
    }
    if (Nq < 1) {
        throw std::invalid_argument("BasisFunctions1D: Nq must be >= 1");
    }

    computeGaussLegendre(Nq_, points_, weights_);
    V_.resize(Nq_, N_ + 1);
    dV_.resize(Nq_, N_ + 1);

    buildVandermonde();
    buildVandermondeDerivative();
}

// ---------------------------------------------------------------------------
// Static utility: evalPolynomials
// ---------------------------------------------------------------------------

void BasisFunctions1D::evalPolynomials(Real x, int N, VectorXr &P,
                                       VectorXr &dP) {
    P.resize(N + 1);
    dP.resize(N + 1);

    // Initial values for normalized Legendre polynomials
    P(0)  = Real(1) / std::sqrt(Real(2));
    dP(0) = Real(0);

    if (N >= 1) {
        P(1)  = std::sqrt(Real(3) / Real(2)) * x;
        dP(1) = std::sqrt(Real(3) / Real(2));
    }

    // Compute P[n+1] and dP[n+1] using three-term recurrence for n = 1..N-1.
    //
    // Normalized Legendre value recurrence:
    //
    // $$ \tilde{P}_{n+1}(x) = \frac{\sqrt{(2n+1)(2n+3)} \cdot x \, \tilde{P}_n(x) - n \sqrt{\frac{2n+3}{2n-1}} \cdot \tilde{P}_{n-1}(x)}{n+1} $$
    //
    // Derivation: substitute $P_n = \sqrt{2/(2n+1)} \, \tilde{P}_n$ into the
    // classic recurrence $(n+1)P_{n+1} = (2n+1)x P_n - n P_{n-1}$.
    for (int n = 1; n <= N - 1; ++n) {
        P(n + 1) =
            (std::sqrt((Real(2) * n + Real(1)) * (Real(2) * n + Real(3))) * x *
                 P(n) -
             n * std::sqrt((Real(2) * n + Real(3)) / (Real(2) * n - Real(1))) *
                 P(n - 1)) /
            (n + Real(1));

        // Normalized Legendre derivative recurrence:
        //
        // $$ \tilde{P}'_{n+1}(x) = \sqrt{(2n+1)(2n+3)} \cdot \tilde{P}_n(x) + \sqrt{\frac{2n+3}{2n-1}} \cdot \tilde{P}'_{n-1}(x) $$
        //
        dP(n + 1) =
            std::sqrt((Real(2) * n + Real(1)) * (Real(2) * n + Real(3))) *
                P(n) +
            std::sqrt((Real(2) * n + Real(3)) / (Real(2) * n - Real(1))) *
                dP(n - 1);
    }
}

// ---------------------------------------------------------------------------
// Static utility: computeGaussLegendre
// ---------------------------------------------------------------------------

void BasisFunctions1D::computeGaussLegendre(int Nq, VectorXr &points,
                                            VectorXr &weights) {
    points.resize(Nq);
    weights.resize(Nq);

    constexpr Real tol    = RealEpsilon;
    constexpr int maxIter = 50;

    // For odd Nq, the middle point is x = 0; compute floor((Nq+1)/2) roots
    // in (-1, 0) and use symmetry to fill the rest.
    int halfN = (Nq + 1) / 2;

    for (int i = 0; i < halfN; ++i) {
        // Initial guess from the asymptotic formula (Tricomi):
        //
        // $$ x_k \approx \cos\!\left(\pi \frac{4k + 3}{4N_q + 2}\right) $$
        //
        // This gives the (Nq - k) root in descending order.
        // For i = 0 we target the root closest to +1; for i = halfN-1
        // we target the root around 0 (or 0 itself if Nq is odd).
        Real x =
            std::cos(M_PI * (Real(4) * i + Real(3)) / (Real(4) * Nq + Real(2)));

        // Newton iteration on the classic Legendre polynomial $P_{N_q}(x)$
        for (int iter = 0; iter < maxIter; ++iter) {
            // Evaluate $P_{N_q}(x)$ and $P_{N_q-1}(x)$ via classic Legendre recurrence
            Real Pk   = Real(1); // P_0(x)
            Real Pk_1 = Real(0); // P_{-1}(x) = 0 (placeholder before iteration)

            for (int n = 0; n < Nq; ++n) {
                // $P_{n+1} = ((2n+1) \, x \, P_n - n \, P_{n-1}) / (n+1)$
                Real Pk_next = ((Real(2) * n + Real(1)) * x * Pk - n * Pk_1) /
                               (n + Real(1));
                Pk_1         = Pk;
                Pk           = Pk_next;
            }
            // After loop: $Pk = P_{N_q}(x), \, Pk\_1 = P_{N_q-1}(x)$

            // $P'_{N_q}(x) = N_q \, (x \, P_{N_q} - P_{N_q-1}) / (x^2 - 1)$
            Real dP = Nq * (x * Pk - Pk_1) / (x * x - Real(1));

            Real dx = Pk / dP;
            x -= dx;

            if (std::abs(dx) < tol) {
                break;
            }
        }

        // Store root and its negative symmetric counterpart.
        // Roots are computed in descending order (most positive first).
        // Place positive root at upper end, negative root at lower end.
        points(Nq - 1 - i) = x;  // positive half (descending order)
        points(i)          = -x; // negative half (ascending order)

        // Weight: $w_i = 2 / \bigl((1 - x_i^2) \cdot [P'_{N_q}(x_i)]^2\bigr)$
        Real Pk   = Real(1);
        Real Pk_1 = Real(0);
        for (int n = 0; n < Nq; ++n) {
            Real Pk_next =
                ((Real(2) * n + Real(1)) * x * Pk - n * Pk_1) / (n + Real(1));
            Pk_1 = Pk;
            Pk   = Pk_next;
        }
        Real dP = Nq * (x * Pk - Pk_1) / (x * x - Real(1));

        Real w              = Real(2) / ((Real(1) - x * x) * dP * dP);
        weights(Nq - 1 - i) = w;
        weights(i)          = w;
    }
}

// ---------------------------------------------------------------------------
// Pre-computation: Vandermonde matrix
// ---------------------------------------------------------------------------

void BasisFunctions1D::buildVandermonde() {
    V_.resize(Nq_, N_ + 1);

    VectorXr P, dP;
    for (int i = 0; i < Nq_; ++i) {
        evalPolynomials(points_(i), N_, P, dP);
        V_.row(i) = P.transpose();
    }
}

// ---------------------------------------------------------------------------
// Pre-computation: Vandermonde derivative matrix
// ---------------------------------------------------------------------------

void BasisFunctions1D::buildVandermondeDerivative() {
    dV_.resize(Nq_, N_ + 1);

    VectorXr P, dP;
    for (int i = 0; i < Nq_; ++i) {
        evalPolynomials(points_(i), N_, P, dP);
        dV_.row(i) = dP.transpose();
    }
}

// ---------------------------------------------------------------------------
// OCCA device memory allocation
// ---------------------------------------------------------------------------

void BasisFunctions1D::allocateDeviceMemory(DeviceMemoryManager &mgr) {
    o_points_  = mgr.wrapOrMalloc(points_.data(), points_.size());
    o_weights_ = mgr.wrapOrMalloc(weights_.data(), weights_.size());
    o_V_       = mgr.wrapOrMalloc(V_.data(), V_.size());
    o_dV_      = mgr.wrapOrMalloc(dV_.data(), dV_.size());
}
