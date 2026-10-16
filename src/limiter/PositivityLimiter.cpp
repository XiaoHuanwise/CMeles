/// @file PositivityLimiter.cpp
/// @brief Implementation of the Zhang & Shu positivity scaling limiter.

#include "PositivityLimiter.hpp"

#include <cstring>

#include "basis/BasisFunctions2D.hpp"
#include "common/KernelProps.hpp"
#include "config/Config.hpp"
#include "dg/DgField.hpp"
#include "mesh/MeshGeometry.hpp"

PositivityLimiter::PositivityLimiter(const Config &cfg, occa::device &device,
                                     DeviceMemoryManager &mem, DgField &field,
                                     const std::string &oklDir)
    : device_(device), mem_(mem), field_(field), oklDir_(oklDir) {
    N_elem_  = field.numElements();
    N_modes_ = field.numModes();
    N_q_     = field.numPoints1D();
    N_q2_    = field.numQuadPoints();

    eps_rel_ = cfg.limiterPositivityEpsilon();

    flags_.assign(static_cast<std::size_t>(N_elem_), 0);
    o_flags_ =
        mem_.wrapOrMallocInt(flags_.data(), static_cast<occa::dim_t>(N_elem_));

    positivityScale_ = buildKernel();
}

occa::kernel PositivityLimiter::buildKernel() {
    occa::json props;
#ifdef USE_FLOAT_PRECISION
    props["defines/Real"] = "float";
#else
    props["defines/Real"] = "double";
#endif
    props["defines/TILE_SIZE"] = cmeles::DefaultTileSize;
    cmeles::finaliseKernelProps(props, device_);

    return device_.buildKernel(oklDir_ + "/positivity_scale.okl",
                               "positivityScale", props);
}

void PositivityLimiter::apply(occa::memory o_u) {
    // P0: the hierarchy contains only the constant mode (the mean itself)
    // — nothing to scale.
    if (N_modes_ <= 1) {
        return;
    }

    const MeshGeometry &geo = field_.geometry();
    const auto n = static_cast<occa::dim_t>(N_elem_) * N_vars_ * N_modes_;

    const auto entries = static_cast<occa::dim_t>(o_u.length());
    if (entries == 2 * n) {
        // Coupled dual-time stacked state [stage0 | stage1]: each half is a
        // full field; project them independently.
        positivityScale_(N_elem_, N_vars_, N_modes_, N_q_, N_q2_, eps_rel_,
                         o_u.slice(0, n), field_.basis().o_V2D(),
                         geo.o_lambdaWJ(), field_.o_faceVandermonde(),
                         geo.o_elemFaces(), geo.o_faceKL(), geo.o_faceFL(),
                         geo.o_faceKR(), geo.o_faceFR(), o_flags_);
        positivityScale_(N_elem_, N_vars_, N_modes_, N_q_, N_q2_, eps_rel_,
                         o_u.slice(n, n), field_.basis().o_V2D(),
                         geo.o_lambdaWJ(), field_.o_faceVandermonde(),
                         geo.o_elemFaces(), geo.o_faceKL(), geo.o_faceFL(),
                         geo.o_faceKR(), geo.o_faceFR(), o_flags_);
    } else {
        positivityScale_(N_elem_, N_vars_, N_modes_, N_q_, N_q2_, eps_rel_, o_u,
                         field_.basis().o_V2D(), geo.o_lambdaWJ(),
                         field_.o_faceVandermonde(), geo.o_elemFaces(),
                         geo.o_faceKL(), geo.o_faceFL(), geo.o_faceKR(),
                         geo.o_faceFR(), o_flags_);
    }
}

void PositivityLimiter::readFlags(std::vector<int> &flags) {
    flags.assign(static_cast<std::size_t>(N_elem_), 0);
    if (mem_.hasSeparateMemorySpace()) {
        mem_.copyToHost(o_flags_, flags.data(),
                        static_cast<occa::dim_t>(N_elem_));
    } else {
        // Unified backends: copyToHost is a no-op (the caller's vector is
        // never the wrapped host source) — read through the host alias.
        const int *src = o_flags_.ptr<int>();
        std::memcpy(flags.data(), src,
                    static_cast<std::size_t>(N_elem_) * sizeof(int));
    }
}
