/// @file PositivityLimiter.hpp
/// @brief Zhang & Shu positivity-preserving scaling limiter.
///
/// Reference: docs/tech_docs/positivity_limiter.md (full specification);
/// X. Zhang & C.-W. Shu, "On positivity-preserving high order discontinuous
/// Galerkin schemes for compressible Euler equations on rectangular
/// meshes", J. Comput. Phys. 229 (2010) 8918-8941.
///
/// Element-local conservative projection toward the cell mean: the
/// high-order modes of all conserved variables are scaled by a single
/// factor theta in [0, 1] until every volume Gauss point and face Gauss
/// trace point of every element satisfies rho >= eps*rhoBar and
/// e >= eps*eBar. Unlike the MR-WENO limiter this is a CONTINUOUS
/// (piecewise-linear in the state) convex projection with no troubled-cell
/// flags or WENO weights, which is why it may also ride the dual-time
/// pseudo iterations (see StepperBase::setPseudoStateFilter): it cannot
/// inject O(1) discontinuities into the iterated operator.

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

/// @brief Positivity-preserving scaling limiter on the DG field.
class PositivityLimiter : public LimiterBase {
public:
    /// @param cfg     Configuration ([limiter], [basis] sections).
    /// @param device  OCCA device (lifetime must exceed this object).
    /// @param mem     Device memory manager (lifetime must exceed this
    ///                object).
    /// @param field   Fully set-up DG field providing the basis/geometry
    ///                device handles and the element/var/mode dimensions.
    /// @param oklDir  Directory containing the limiter .okl sources.
    PositivityLimiter(const Config &cfg, occa::device &device,
                      DeviceMemoryManager &mem, DgField &field,
                      const std::string &oklDir = OCCA_OKL_DIR);

    /// @brief Project \p o_u onto the admissible set in place.
    ///
    /// Accepts either the field-length modal vector or the coupled
    /// dual-time stacked state [stage0 | stage1] (2x the field length) —
    /// the halves are projected independently.
    void apply(occa::memory o_u) override;

    /// @brief Copy out the per-element action flags of the last apply():
    ///        0 = untouched (theta == 1), 1 = scaled (theta < 1),
    ///        2 = mean inadmissible (cannot be repaired by scaling).
    ///
    /// Diagnostics/tests only — apply() itself never syncs.
    void readFlags(std::vector<int> &flags);

private:
    /// @brief Build the kernel with the standard JIT props.
    occa::kernel buildKernel();

    // ---- Context ----
    occa::device &device_;
    DeviceMemoryManager &mem_;
    DgField &field_;
    std::string oklDir_;

    // ---- Dimensions (snapshots from the field) ----
    int N_elem_  = 0;
    int N_vars_  = kNumVars2D;
    int N_modes_ = 0;
    int N_q_     = 0;
    int N_q2_    = 0;

    /// Relative positivity margin eps ([limiter] positivity_epsilon):
    /// point values are kept at rho >= eps*rhoBar, e >= eps*eBar.
    Real eps_rel_ = Real(1e-10);

    // ---- Host storage of the action flags ----
    std::vector<int> flags_;

    // ---- Device buffers / kernels ----
    occa::memory o_flags_;
    occa::kernel positivityScale_;
};
