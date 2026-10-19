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

namespace {

/// @brief In-place Cholesky inversion of an SPD matrix (row-major,
///        n <= 10). Arithmetically identical twin of the Cholesky that
///        used to live in mr_weno_limit.okl (removed with the host
///        Gram precomputation) — keep the operation order in sync so
///        the staged blocks stay bit-identical to the former in-kernel
///        computation.
void cholInvertHost(const int n, Real *A, Real *Ainv) {
    for (int j = 0; j < n; ++j) {
        Real d = A[j * n + j];
        for (int k = 0; k < j; ++k) {
            d -= A[j * n + k] * A[j * n + k];
        }
        A[j * n + j] = std::sqrt(d);
        for (int i = j + 1; i < n; ++i) {
            Real s = A[i * n + j];
            for (int k = 0; k < j; ++k) {
                s -= A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = s / A[j * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        A[i * n + i] = Real(1) / A[i * n + i];
        for (int j = 0; j < i; ++j) {
            Real s = Real(0);
            for (int k = j; k < i; ++k) {
                s += A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] = -s * A[i * n + i];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            Real s = Real(0);
            for (int k = i; k < n; ++k) {
                s += A[k * n + i] * A[k * n + j];
            }
            Ainv[i * n + j] = s;
            Ainv[j * n + i] = s;
        }
    }
}

} // namespace

// ============================================================================
// Construction
// ============================================================================

MrWenoLimiter::MrWenoLimiter(const Config &cfg, occa::device &device,
                             DeviceMemoryManager &mem, DgField &field,
                             const std::string &oklDir)
    : device_(device), mem_(mem), field_(field), oklDir_(oklDir) {
    N_elem_  = field.numElements();
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

    // Driver mode of the reconstruction kernel, resolved once:
    // - list-driven keeps the (static-scheduled) OpenMP parallel-for
    //   balanced — the dense list holds only heavy work units — and its
    //   flag readback is a zero-copy no-op on unified backends;
    // - full-field early-exit needs no host readback at all: on
    //   separate-memory backends every copyToHost drains the command
    //   queue (a per-stage pipeline sync), while the clean-cell exits
    //   are absorbed by the block-level oversubscription of the GPU.
    const WenoDriveMode drive = cfg.limiterWenoDriveMode();
    listDriven_ =
        drive == WenoDriveMode::List ||
        (drive == WenoDriveMode::Auto && !mem_.hasSeparateMemorySpace());

    if (N_modes_ > kMaxModes) {
        throw std::invalid_argument(
            "MrWenoLimiter: modal count exceeds the kernel capacity");
    }

    buildDerivativeTables();
    buildGramTables();

    troubled_.assign(static_cast<std::size_t>(N_elem_), 0);
    o_troubled_ = mem_.wrapOrMallocInt(troubled_.data(),
                                       static_cast<occa::dim_t>(N_elem_));
    troubleList_.assign(static_cast<std::size_t>(N_elem_), 0);
    o_troubleList_ = mem_.wrapOrMallocInt(troubleList_.data(),
                                          static_cast<occa::dim_t>(N_elem_));
    u_lin_.assign(static_cast<std::size_t>(N_elem_) * N_vars_ * 3, Real(0));
    o_uLin_ = mem_.wrapOrMalloc(u_lin_.data(),
                                static_cast<occa::dim_t>(u_lin_.size()));

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
// Inverse Gram tables (non-affine projection support)
// ============================================================================

void MrWenoLimiter::buildGramTables() {
    const MeshGeometry &geo           = field_.geometry();
    const MatrixXr &V                 = field_.basis().vandermonde();
    const VectorXr &lamAll            = geo.lambdaWJ();
    const Eigen::VectorXi &elemAffine = geo.elemAffine();

    // Per-element stride: sum of the squared block sizes over l = 0..N-1.
    int stride = 0;
    for (int ell = 0; ell < N_; ++ell) {
        const int Nl = (ell + 1) * (ell + 2) / 2;
        stride += Nl * Nl;
    }

    // Dense slots over the non-affine elements only; affine elements
    // keep -1 and never touch the block buffer.
    minv_idx_.assign(static_cast<std::size_t>(N_elem_), -1);
    minv_blk_.clear();
    for (int e = 0; e < N_elem_; ++e) {
        if (elemAffine[e] != 0) {
            continue;
        }
        minv_idx_[static_cast<std::size_t>(e)] =
            static_cast<int>(minv_blk_.size());
        minv_blk_.resize(minv_blk_.size() + static_cast<std::size_t>(stride),
                         Real(0));
        Real *dst = minv_blk_.data() + minv_idx_[static_cast<std::size_t>(e)];
        int off   = 0;
        for (int ell = 0; ell < N_; ++ell) {
            const int Nl = (ell + 1) * (ell + 2) / 2;
            // Same assembly product order as the former in-kernel loop:
            // s += lambdaWJ * V2D[q][i] * V2D[q][j], q ascending.
            Real A[100], Ainv[100];
            for (int i = 0; i < Nl; ++i) {
                for (int j = 0; j < Nl; ++j) {
                    Real s = Real(0);
                    for (int q = 0; q < N_q2_; ++q) {
                        s += lamAll[static_cast<std::size_t>(e) * N_q2_ +
                                    static_cast<std::size_t>(q)] *
                             V(q, i) * V(q, j);
                    }
                    A[i * Nl + j] = s;
                }
            }
            cholInvertHost(Nl, A, Ainv);
            for (int i = 0; i < Nl * Nl; ++i) {
                dst[off + i] = Ainv[i];
            }
            off += Nl * Nl;
        }
    }
    if (minv_blk_.empty()) {
        // All elements affine: the buffer is never read, but a zero-size
        // device allocation is not well defined — stage one dummy entry.
        minv_blk_.assign(1, Real(0));
    }

    o_minvBlk_ = mem_.wrapOrMalloc(minv_blk_.data(),
                                   static_cast<occa::dim_t>(minv_blk_.size()));
    o_minvIdx_ = mem_.wrapOrMallocInt(
        minv_idx_.data(), static_cast<occa::dim_t>(minv_idx_.size()));
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
    props["defines/TILE_SIZE"] = cmeles::DefaultTileSize;
    // Tile of the reconstruction kernel: its loop bound is the SHORT
    // troubled list (list-driven mode) or the full element count with a
    // flag-guard early exit (full-field mode — clean cells are free, so
    // the bound length no longer matters for the block count). Either
    // way the tile must stay small: 1 work unit per cell on the unified
    // backends; a 64-thread wavefront per block on the GPU backends.
    // A 256-tile over a few hundred troubled cells would collapse the
    // outer block count to a handful and serialize the heavy work on
    // that many threads.
    props["defines/TROUBLE_TILE_SIZE"] = mem_.hasSeparateMemorySpace() ? 64 : 1;
    props["defines/WENO_LIST_DRIVEN"]  = listDriven_ ? 1 : 0;
    props["defines/N_VARS"]            = kNumVars2D;
    props["defines/NQ_MAX"]            = kMaxQuadPts1D;
    props["defines/NQ2_MAX"]           = kMaxQuadPts2D;
    props["defines/NMODES_MAX"]        = kMaxModes;
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

    // 1. Per-element modified KXRCF test -> troubled flags. The per-face
    //    signed density-jump integrals of the inflow faces are evaluated
    //    inline by the same kernel (fused from the former face-parallel
    //    kxrcfFaceJump kernel and its o_faceJump_ staging buffer).
    kxrcfFlag_(N_elem_, N_vars_, N_modes_, N_q_, N_q2_, hPow_, ckc_, shockNorm_,
               force_, o_u, field_.basis().o_V2D(), geo.o_lambdaWJ(),
               field_.o_faceVandermonde(), geo.o_elemFaces(), geo.o_faceKL(),
               geo.o_faceFL(), geo.o_faceKR(), geo.o_faceFR(),
               geo.o_faceTypes(), geo.o_faceNormals(), field_.o_faceWeights(),
               geo.o_faceJac(), geo.o_vertices(), gammaGas_, o_troubled_);

    // 3. Resolve the troubled set for the reconstruction kernel.
    //    List-driven (host compaction, on purpose): the unified backends
    //    alias the flag buffer into troubled_ (zero-copy), and OCCA 2.0's
    //    @atomic is not portable (the OpenCL backend silently drops it),
    //    so a device-side compaction is not an option. A clean field
    //    short-circuits the whole reconstruction chain.
    //    Full-field: no readback at all — the kernel guards every clean
    //    cell with a single flag load (linStage then also runs on a clean
    //    field; its full-mesh linear-mode copy is far cheaper than the
    //    per-stage queue-drain a readback would cost).
    int nTroubled = 0;
    if (listDriven_) {
        mem_.copyToHost(o_troubled_, troubled_.data(),
                        static_cast<occa::dim_t>(N_elem_));
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
    }

    // 4. Snapshot every element's linear modes (each troubled cell reads
    //    its neighbours' PRE-limit linear modes — order-free by
    //    construction), then reconstruct: one work unit per troubled cell
    //    in list mode (dense list, no idle slots), or one per element
    //    with a flag-guard early exit in full mode (troubleList is then
    //    an unused placeholder).
    mrWenoLinStage_(N_elem_, N_vars_, N_modes_, o_u, o_uLin_);
    const occa::memory troublePtr = listDriven_ ? o_troubleList_ : o_troubled_;
    mrWenoLimit_(listDriven_ ? nTroubled : N_elem_, N_, N_vars_, N_modes_, N_q_,
                 N_q2_, gammaLow_, gammaHigh_, epsW_, gammaGas_, o_u, o_uLin_,
                 troublePtr, o_troubled_, geo.o_elemFaces(), geo.o_faceKL(),
                 geo.o_faceFL(), geo.o_faceKR(), geo.o_faceFR(),
                 geo.o_faceNormals(), geo.o_vertices(), field_.basis().o_V2D(),
                 field_.o_faceVandermonde(), geo.o_lambdaWJ(), geo.o_Jinv11(),
                 geo.o_Jinv12(), geo.o_Jinv21(), geo.o_Jinv22(),
                 geo.o_elemAffine(), o_minvIdx_, o_minvBlk_, o_derivV_);
}
