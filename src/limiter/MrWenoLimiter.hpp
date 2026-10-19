/// @file MrWenoLimiter.hpp
/// @brief Multi-resolution WENO (MR-WENO) discontinuity limiter.
///
/// Reference: docs/tech_docs/mr_weno_limiter.md (full specification);
/// Zhu, Qiu & Shu 2019 (JCP 407:109105) and Zhu, Shu & Qiu 2021
/// (CiCP 29:1030).
///
/// Pipeline of one apply() call:
///   1. kxrcfFlag — per-element modified KXRCF troubled-cell test
///      (Fu & Shu exponent, Zhu & Shu min normalisation), with the
///      inflow-face density-jump integrals evaluated inline;
///   2. troubled-list resolution — list-driven mode ([limiter]
///      weno_drive_mode = "list"/"auto-on-CPU): host compaction of the
///      flags into a dense ascending index list (a clean field
///      short-circuits the remaining steps); full-field mode ("full"/
///      "auto-on-GPU): no readback at all, the reconstruction kernel
///      early-exits on the per-element flag;
///   3. mrWenoLinStage — snapshot of every element's linear modes;
///   4. mrWenoLimit — reconstruction of the troubled cells: face-normal
///      characteristic decomposition, per-component scalar MR-WENO
///      (hierarchical candidates -> telescoping recursion -> WENO-Z
///      weights), edge-length weighted average.
///
/// Stencil: only the direct face neighbours' linear modal coefficients
/// (for the lowest-order smoothness indicator), read from the staged
/// snapshot; every troubled cell is processed independently from its
/// neighbours' *pre-limit* state, so the kernel has no ordering
/// dependence and no write conflicts.

#pragma once

#include <occa.hpp>

#include <string>
#include <vector>

#include "common/Constants.hpp"
#include "common/Types.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "limiter/LimiterBase.hpp"

class Config;
class DgField;

/// @brief MR-WENO limiter: detection + reconstruction on the DG field.
class MrWenoLimiter : public LimiterBase {
public:
    /// @param cfg     Configuration ([limiter], [gas], [basis] sections).
    /// @param device  OCCA device (lifetime must exceed this object).
    /// @param mem     Device memory manager (lifetime must exceed this
    ///                object).
    /// @param field   Fully set-up DG field providing all geometry/basis
    ///                device handles and the element/var/mode dimensions.
    /// @param oklDir  Directory containing the limiter .okl sources.
    MrWenoLimiter(const Config &cfg, occa::device &device,
                  DeviceMemoryManager &mem, DgField &field,
                  const std::string &oklDir = OCCA_OKL_DIR);

    /// @brief Detect troubled cells and limit them in place in \p o_u.
    void apply(occa::memory o_u) override;

    /// @brief Cells flagged troubled by the last apply() call (0 when the
    ///        field is clean — the reconstruction chain is then skipped
    ///        entirely). Maintained by the list-driven path only; the
    ///        full-field mode never reads the flags back, so the value
    ///        stays at its last list-mode count.
    int lastTroubledCount() const {
        return last_troubled_count_;
    }

private:
    /// @brief Build a limiter kernel with the standard JIT props
    ///        (Real / TILE_SIZE / capacity caps).
    occa::kernel buildKernel(const std::string &file, const std::string &name);

    /// @brief Build the multi-index derivative Vandermonde tables of the
    ///        smoothness indicator (host side) and upload them.
    ///
    /// Table layout: derivV[slot(a1, a2)][q][l] with the r/s derivative
    /// orders (a1, a2), $1 \le a_1 + a_2 \le N$, and
    /// $\mathrm{slot}(a_1, a_2) = \tfrac{d(d+1)}{2} + a_1$ with
    /// $d = a_1 + a_2$ (the Pascal-triangle slot of the multi-index;
    /// $d = 0$ is skipped, so exactly $N_{\text{modes}} - 1$ tables exist).
    /// The first-order tables (d = 1) duplicate basis.dV2D_r/dV2D_s by
    /// construction — kept here so the kernel reads a single buffer with a
    /// uniform indexing.
    void buildDerivativeTables();

    /// @brief Precompute the inverse Gram blocks $M_l^{-1}$ of every
    ///        non-affine element (host side, once — the blocks depend
    ///        only on the static element geometry) and upload them.
    ///
    /// Layout: minvBlk[slot(e) + off_l + i*Nl + j] with the per-element
    /// slot from minvIdx (dense over the non-affine elements, -1 for
    /// affine) and the same cumulative block offsets the kernel
    /// reconstructs (block $l$ holds $N_l^2$ entries,
    /// $N_l = (l+1)(l+2)/2$). The assembly product order and the
    /// Cholesky inversion mirror the former in-kernel computation
    /// operation-for-operation, so the staged blocks are bit-identical
    /// to what the kernel used to compute per apply.
    void buildGramTables();

    // ---- Context ----
    occa::device &device_;
    DeviceMemoryManager &mem_;
    DgField &field_;
    std::string oklDir_;

    // ---- Dimensions (snapshots from the field) ----
    int N_       = 0; ///< Polynomial order.
    int N_elem_  = 0;
    int N_vars_  = kNumVars2D;
    int N_modes_ = 0;
    int N_q_     = 0;
    int N_q2_    = 0;

    // ---- Parameters ([limiter] / [gas], by value) ----
    Real gammaGas_  = Real(1.4);
    Real ckc_       = Real(1);
    int shockNorm_  = 0; ///< ShockNorm::Min/Max/Mean as int.
    Real gammaLow_  = Real(0.01);
    Real gammaHigh_ = Real(0.99);
    Real epsW_      = Real(1e-6);
    int force_      = 0;
    Real hPow_      = Real(1); ///< KXRCF exponent R = (min(N,2)+1)/2.

    /// List-driven reconstruction (host compaction per apply) vs
    /// full-field early-exit (zero host readback). Resolved once in the
    /// constructor from [limiter] weno_drive_mode and the memory-space
    /// kind of the device backend (see \p WenoDriveMode).
    bool listDriven_ = true;

    // ---- Host storage of uploaded data (wrap sources on unified
    // backends must outlive the occa handles) ----
    std::vector<Real> deriv_v_;    ///< Flattened derivative Vandermonde tables.
    std::vector<Real> minv_blk_;   ///< Precomputed inverse Gram blocks (dense
                                   ///< over the non-affine elements).
    std::vector<int> minv_idx_;    ///< Element -> minv_blk_ slot (-1 affine).
    std::vector<int> troubled_;    ///< Troubled-cell flags (0/1).
    std::vector<int> troubleList_; ///< Dense ascending troubled indices.
    std::vector<Real> u_lin_;      ///< Linear-mode snapshot (staging source).

    // ---- Device buffers ----
    occa::memory o_derivV_;
    occa::memory o_minvBlk_;
    occa::memory o_minvIdx_;
    occa::memory o_troubled_;
    occa::memory o_troubleList_;
    occa::memory o_uLin_;

    // ---- Kernels ----
    occa::kernel kxrcfFlag_;
    occa::kernel mrWenoLinStage_;
    occa::kernel mrWenoLimit_;

    /// Troubled cells found by the last apply() call.
    int last_troubled_count_ = 0;
};
