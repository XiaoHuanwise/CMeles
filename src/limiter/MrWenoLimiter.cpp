/// @file MrWenoLimiter.cpp
/// @brief Implementation of the MR-WENO limiter.

#include "MrWenoLimiter.hpp"

#include <cmath>
#include <stdexcept>

#include "basis/BasisFunctions2D.hpp"
#include "common/KernelProps.hpp"
#include "config/Config.hpp"
#include "dg/DgField.hpp"
#include "mesh/MeshGeometry.hpp"

// ============================================================================
// Construction
// ============================================================================

MrWenoLimiter::MrWenoLimiter(const Config &cfg, occa::device &device,
                             DeviceMemoryManager &mem, DgField &field,
                             const std::string &oklDir)
    : device_(device), mem_(mem), field_(field), oklDir_(oklDir) {
    N_elem_  = field.numElements();
    N_face_  = field.geometry().numFaces();
    N_modes_ = field.numModes();
    N_q_     = field.numPoints1D();
    N_q2_    = field.numQuadPoints();
    N_       = field.basis().order();

    gammaGas_  = cfg.gamma();
    ckc_       = cfg.limiterCkc();
    shockNorm_ = static_cast<int>(cfg.limiterShockNorm());
    gammaLow_  = cfg.limiterGammaLow();
    gammaHigh_ = cfg.limiterGammaHigh();
    epsW_      = cfg.limiterEpsilon();
    force_     = cfg.limiterForceTroubled() ? 1 : 0;
    // KXRCF exponent (Fu & Shu): R = (min(N, 2) + 1) / 2 — P1 takes 1,
    // P2 and above take 1.5.
    hPow_ = (Real(std::min(N_, 2)) + Real(1)) / Real(2);

    if (N_modes_ > kMaxModes) {
        throw std::invalid_argument(
            "MrWenoLimiter: modal count exceeds the kernel capacity");
    }

    buildDerivativeTables();

    troubled_.assign(static_cast<std::size_t>(N_elem_), 0);
    o_troubled_ = mem_.wrapOrMallocInt(troubled_.data(),
                                       static_cast<occa::dim_t>(N_elem_));
    troubleList_.assign(static_cast<std::size_t>(N_elem_), 0);
    o_troubleList_ = mem_.wrapOrMallocInt(troubleList_.data(),
                                          static_cast<occa::dim_t>(N_elem_));
    u_lin_.assign(static_cast<std::size_t>(N_elem_) * N_vars_ * 3, Real(0));
    o_uLin_     = mem_.wrapOrMalloc(u_lin_.data(),
                                    static_cast<occa::dim_t>(u_lin_.size()));
    o_faceJump_ = mem_.wrapOrMalloc(static_cast<occa::dim_t>(N_face_));

    kxrcfFaceJump_  = buildKernel("kxrcf_face_jump.okl", "kxrcfFaceJump");
    kxrcfFlag_      = buildKernel("kxrcf_flag.okl", "kxrcfFlag");
    mrWenoLinStage_ = buildKernel("mr_weno_lin_stage.okl", "mrWenoLinStage");
    mrWenoLimit_    = buildKernel("mr_weno_limit.okl", "mrWenoLimit");
}

// ============================================================================
// Derivative Vandermonde tables
// ============================================================================

void MrWenoLimiter::buildDerivativeTables() {
    const BasisFunctions2D &basis = field_.basis();
    const MatrixXr &qp            = basis.quadraturePoints(); // Nq2 x 2
    const auto &ij                = basis.indexMap();         // Nm x 2

    // Derivatives of the normalised Legendre polynomials
    // $\tilde{P}_n(x) = \sqrt{(2n+1)/2}\, P_n(x)$ at a given argument,
    // via the k-times differentiated three-term recurrence
    //
    /// $$ (n+1)\, P_{n+1}^{(k)} = (2n+1)\,\big(x\, P_n^{(k)} + k\, P_n^{(k-1)}\big) - n\, P_{n-1}^{(k)}. $$
    ///
    /// Fills P[k][n] = $\tilde{P}_n^{(k)}(x)$ for k, n <= N (N <= 4).
    const int N             = N_;
    auto legendreDerivTable = [N](Real x, Real(&P)[5][5]) {
        for (int k = 0; k <= N; ++k) {
            for (int n = 0; n <= N; ++n) {
                P[k][n] = Real(0);
            }
        }
        P[0][0] = Real(1);
        if (N >= 1) {
            P[0][1] = x;
            P[1][1] = Real(1);
        }
        for (int n = 1; n < N; ++n) {
            for (int k = 0; k <= N; ++k) {
                Real v = (Real(2 * n + 1)) * (x * P[k][n]);
                if (k >= 1) {
                    v += (Real(2 * n + 1)) * (Real(k) * P[k - 1][n]);
                }
                v -= (Real(n)) * P[k][n - 1];
                P[k][n + 1] = v / Real(n + 1);
            }
        }
        // Normalise: $\tilde{P}_n^{(k)} = \sqrt{(2n+1)/2}\, P_n^{(k)}$.
        for (int k = 0; k <= N; ++k) {
            for (int n = 0; n <= N; ++n) {
                P[k][n] *= std::sqrt((Real(2 * n + 1)) / Real(2));
            }
        }
    };

    // Tables for every multi-index (a1, a2), 1 <= a1 + a2 <= N, flattened
    // as derivV[slot][q][l] with slot = d(d+1)/2 + a1, d = a1 + a2.
    // The d = 0 row is skipped, so the buffer holds N_modes - 1 tables
    // (slots 1 .. N_modes - 1; slot 0 is dead padding that keeps the
    // indexing branch-free — one extra Nq2 x Nm block).
    deriv_v_.assign(static_cast<std::size_t>(N_modes_) * N_q2_ * N_modes_,
                    Real(0));
    for (int q = 0; q < N_q2_; ++q) {
        Real Pr[5][5], Ps[5][5];
        legendreDerivTable(qp(q, 0), Pr);
        legendreDerivTable(qp(q, 1), Ps);
        for (int l = 0; l < N_modes_; ++l) {
            const int il = ij(l, 0);
            const int jl = ij(l, 1);
            for (int a1 = 0; a1 <= N; ++a1) {
                for (int a2 = (a1 == 0) ? 1 : 0; a1 + a2 <= N; ++a2) {
                    const int d    = a1 + a2;
                    const int slot = d * (d + 1) / 2 + a1;
                    deriv_v_[static_cast<std::size_t>(slot) * N_q2_ * N_modes_ +
                             static_cast<std::size_t>(q) * N_modes_ + l] =
                        Pr[a1][il] * Ps[a2][jl];
                }
            }
        }
    }

    o_derivV_ = mem_.wrapOrMalloc(deriv_v_.data(),
                                  static_cast<occa::dim_t>(deriv_v_.size()));
}

// ============================================================================
// Kernel compilation
// ============================================================================

occa::kernel MrWenoLimiter::buildKernel(const std::string &file,
                                        const std::string &name) {
    occa::json props;
#ifdef USE_FLOAT_PRECISION
    props["defines/Real"] = "float";
#else
    props["defines/Real"] = "double";
#endif
    props["defines/TILE_SIZE"]  = cmeles::DefaultTileSize;
    // Tile of the list-driven reconstruction kernel: its loop bound is
    // the SHORT troubled list, so the outer block count must track the
    // troubled-cell count (1 work unit per cell on the unified backends;
    // a 64-thread wavefront per block on the GPU backends). A 256-tile
    // would collapse a few hundred troubled cells into a handful of
    // outer blocks and serialize the heavy work on that many threads.
    props["defines/TROUBLE_TILE_SIZE"] =
        mem_.hasSeparateMemorySpace() ? 64 : 1;
    props["defines/N_VARS"]     = kNumVars2D;
    props["defines/NQ_MAX"]     = kMaxQuadPts1D;
    props["defines/NQ2_MAX"]    = kMaxQuadPts2D;
    props["defines/NMODES_MAX"] = kMaxModes;
    cmeles::finaliseKernelProps(props, device_);

    return device_.buildKernel(oklDir_ + "/" + file, name, props);
}

// ============================================================================
// Application
// ============================================================================

void MrWenoLimiter::apply(occa::memory o_u) {
    // P0: the hierarchy contains only the constant — nothing to limit.
    if (N_ == 0) {
        return;
    }

    const MeshGeometry &geo = field_.geometry();

    // 1. Per-face signed density-jump integrals (dummy values on boundary
    //    and degenerate faces; consumed only for interior/periodic faces).
    kxrcfFaceJump_(N_face_, N_vars_, N_modes_, N_q_, o_u,
                   field_.o_faceVandermonde(), field_.o_faceWeights(),
                   geo.o_faceKL(), geo.o_faceFL(), geo.o_faceKR(),
                   geo.o_faceFR(), geo.o_faceJac(), o_faceJump_);

    // 2. Per-element modified KXRCF test -> troubled flags.
    kxrcfFlag_(N_elem_, N_vars_, N_modes_, N_q_, N_q2_, hPow_, ckc_, shockNorm_,
               force_, o_u, field_.basis().o_V2D(), geo.o_lambdaWJ(),
               field_.o_faceVandermonde(), geo.o_elemFaces(), geo.o_faceKL(),
               geo.o_faceFL(), geo.o_faceKR(), geo.o_faceFR(),
               geo.o_faceTypes(), geo.o_faceNormals(), o_faceJump_,
               geo.o_vertices(), gammaGas_, o_troubled_);

    // 3. Compact the troubled flags into a dense ascending index list.
    //    Host side on purpose: the unified backends alias the flag
    //    buffer into troubled_ (zero-copy), and OCCA 2.0's @atomic is
    //    not portable (the OpenCL backend silently drops it), so a
    //    device-side compaction is not an option. A clean field
    //    short-circuits the whole reconstruction chain.
    mem_.copyToHost(o_troubled_, troubled_.data(),
                    static_cast<occa::dim_t>(N_elem_));
    int nTroubled = 0;
    for (int e = 0; e < N_elem_; ++e) {
        if (troubled_[e] != 0) {
            troubleList_[nTroubled++] = e;
        }
    }
    last_troubled_count_ = nTroubled;
    if (nTroubled == 0) {
        return;
    }
    mem_.copyFromHost(o_troubleList_, troubleList_.data(),
                      static_cast<occa::dim_t>(nTroubled));

    // 4. Snapshot every element's linear modes (each troubled cell reads
    //    its neighbours' PRE-limit linear modes — order-free by
    //    construction), then reconstruct exactly the troubled cells
    //    from the dense list: one thread slot per troubled cell, no
    //    idle slots, no load imbalance.
    mrWenoLinStage_(N_elem_, N_vars_, N_modes_, o_u, o_uLin_);
    mrWenoLimit_(nTroubled, N_, N_vars_, N_modes_, N_q_, N_q2_, gammaLow_,
                 gammaHigh_, epsW_, gammaGas_, o_u, o_uLin_, o_troubleList_,
                 geo.o_elemFaces(), geo.o_faceKL(), geo.o_faceFL(),
                 geo.o_faceKR(), geo.o_faceFR(), geo.o_faceNormals(),
                 geo.o_vertices(), field_.basis().o_V2D(),
                 field_.o_faceVandermonde(), geo.o_lambdaWJ(), geo.o_Jinv11(),
                 geo.o_Jinv12(), geo.o_Jinv21(), geo.o_Jinv22(),
                 geo.o_elemAffine(), o_derivV_);
}
