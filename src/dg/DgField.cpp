/// @file DgField.cpp
/// @brief Implementation of DgField: pre-computation, device buffers and
///        the residual pipeline for the 2D inviscid Euler equations.

#include "dg/DgField.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "basis/BasisFunctions1D.hpp"
#include "basis/BasisFunctions2D.hpp"
#include "common/KernelProps.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/MeshGeometry.hpp"
#include "mesh/StructuredMeshGenerator.hpp"

// ============================================================================
// Construction / setup
// ============================================================================

DgField::DgField(const Config &cfg, occa::device &device,
                 DeviceMemoryManager &mem, const std::string &oklDir)
    : cfg_(cfg), device_(device), mem_(mem), oklDir_(oklDir) {
}

DgField::~DgField() = default;

void DgField::setup() {
    // Reference data (pre-computation on the host).
    basis1D_ = std::make_unique<BasisFunctions1D>(cfg_.polynomialOrder(),
                                                  cfg_.quadratureOrder());
    basis2D_ =
        std::make_unique<BasisFunctions2D>(*basis1D_, cfg_.polynomialOrder());

    mesh_ = std::make_unique<Mesh>(
        StructuredMeshGenerator::generate(StructuredMeshGenerator::Params{
            cfg_.meshNx(), cfg_.meshNy(), cfg_.meshX0(), cfg_.meshY0(),
            cfg_.meshDx(), cfg_.meshDy(), cfg_.meshShear(),
            cfg_.meshSplitTriangles()}));
    // Periodic boundaries: pair the boundary faces into pseudo-interior
    // faces before the geometry is built, so the K_R/f_R adjacency columns
    // uploaded by MeshGeometry carry the partner elements.
    if (cfg_.bcType() == BcType::Periodic) {
        mesh_->applyPeriodicPairing(cfg_.meshNx() * cfg_.meshDx(),
                                    cfg_.meshNy() * cfg_.meshDy());
    }
    geo_ = std::make_unique<MeshGeometry>(*mesh_, *basis2D_);

    N_modes_ = basis2D_->numBases();
    N_q_     = basis2D_->numPoints1D();
    N_q2_    = basis2D_->numPoints();

    // Upload the reference data to the device.
    basis1D_->allocateDeviceMemory(mem_);
    basis2D_->allocateDeviceMemory(mem_);
    geo_->allocateDeviceMemory(mem_);

    buildFaceQuadratureData();
    allocateDeviceMemory();
    buildKernels();
}

// ============================================================================
// Sizes / accessors
// ============================================================================

int DgField::numElements() const noexcept {
    return geo_ ? geo_->numElements() : 0;
}
int DgField::numModes() const noexcept {
    return N_modes_;
}
int DgField::numQuadPoints() const noexcept {
    return N_q2_;
}
int DgField::numPoints1D() const noexcept {
    return N_q_;
}

const Mesh &DgField::mesh() const noexcept {
    return *mesh_;
}
const MeshGeometry &DgField::geometry() const noexcept {
    return *geo_;
}
const BasisFunctions2D &DgField::basis() const noexcept {
    return *basis2D_;
}

// ============================================================================
// Face quadrature data
// ============================================================================

void DgField::buildFaceQuadratureData() {
    // Face Vandermonde: Vface[side][f][i][mode] = phi_mode(r(t_i), s(t_i))
    // with (r, s) from faceRefMapLeft(f) for side 0 (keyed by the left
    // element's own face number fL) and faceRefMapRight(f) for side 1
    // (keyed by the right element's own face number fR), t_i the i-th
    // face Gauss point (the same N_q Gauss-Legendre points as the volume
    // quadrature).
    //
    // phi_mode(r, s) = Ptilde_{i_mode}(r) * Ptilde_{j_mode}(s), where
    // Ptilde_k(x) is the k-th normalised Legendre polynomial. For each
    // candidate argument x in {t_i, -t_i, 1, -1} we reuse the 1D
    // Vandermonde rows (t_i) and the boundary values Ptilde_k(+-1).
    const int N  = cfg_.polynomialOrder();
    const int Nq = N_q_;

    const MatrixXr &V1D = basis1D_->vandermonde(); // Nq x (N+1)
    // Normalised Legendre boundary values: Ptilde_k(1) = sqrt((2k+1)/2),
    // Ptilde_k(-1) = (-1)^k * Ptilde_k(1).
    VectorXr P1(N + 1), Pm1(N + 1);
    for (int k = 0; k <= N; ++k) {
        P1(k)  = std::sqrt((Real(2) * k + Real(1)) / Real(2));
        Pm1(k) = (k % 2 == 0) ? P1(k) : -P1(k);
    }

    // Evaluate the k-th normalised Legendre polynomial at a candidate
    // argument x: x == t_i (V1D row), x == -t_i (parity), x == 1, x == -1.
    auto evalP = [&](Real x, int k) -> Real {
        if (x == Real(1)) {
            return P1(k);
        }
        if (x == Real(-1)) {
            return Pm1(k);
        }
        // x is one of the Gauss points (or its negation): find the row.
        const VectorXr &pts = basis1D_->points();
        for (int i = 0; i < Nq; ++i) {
            if (std::abs(x - pts(i)) < Real(1e-12) * (Real(1) + std::abs(x))) {
                return V1D(i, k);
            }
            if (std::abs(x + pts(i)) < Real(1e-12) * (Real(1) + std::abs(x))) {
                // Ptilde_k(-t_i) = (-1)^k Ptilde_k(t_i)
                return (k % 2 == 0) ? V1D(i, k) : -V1D(i, k);
            }
        }
        throw std::runtime_error(
            "DgField::buildFaceQuadratureData: unexpected reference coord");
    };

    const int N_modes = N_modes_;
    face_vandermonde_.assign(static_cast<size_t>(2) * 4 * Nq * N_modes,
                             Real(0));
    const auto &ij = basis2D_->indexMap();

    for (int side = 0; side < 2; ++side) {
        for (int f = 0; f < 4; ++f) {
            const FaceRefMap &map =
                (side == 0) ? faceRefMapLeft(f) : faceRefMapRight(f);
            for (int i = 0; i < Nq; ++i) {
                const Real t = basis1D_->points()(i);
                const Real r = map.cr * t + map.dr;
                const Real s = map.cs * t + map.ds;
                for (int l = 0; l < N_modes; ++l) {
                    const Real Pr = evalP(r, ij(l, 0));
                    const Real Ps = evalP(s, ij(l, 1));
                    face_vandermonde_[((side * 4 + f) * Nq + i) * N_modes + l] =
                        Pr * Ps;
                }
            }
        }
    }

    // Face quadrature weights (1D Gauss-Legendre weights). The buffers are
    // members (face_vandermonde_, face_weights_) so that the wrapMemory
    // handles stay valid on unified memory backends (zero-copy aliases the
    // host pointer).
    const VectorXr &w1d = basis1D_->weights();
    face_weights_.assign(Nq, Real(0));
    for (int i = 0; i < Nq; ++i) {
        face_weights_[i] = w1d(i);
    }

    o_faceVandermonde_ =
        mem_.wrapOrMalloc(face_vandermonde_.data(), face_vandermonde_.size());
    o_faceWeights_ =
        mem_.wrapOrMalloc(face_weights_.data(), face_weights_.size());
}

// ============================================================================
// Device memory
// ============================================================================

void DgField::allocateDeviceMemory() {
    const occa::dim_t nEvm =
        static_cast<occa::dim_t>(geo_->numElements()) * N_vars_ * N_modes_;
    const occa::dim_t nFaceEvm =
        static_cast<occa::dim_t>(geo_->numFaces()) * N_vars_ * 2 * N_modes_;

    o_u_        = mem_.wrapOrMalloc(nEvm);
    o_res_      = mem_.wrapOrMalloc(nEvm);
    o_faceFlux_ = mem_.wrapOrMalloc(nFaceEvm);
    // Scratch for the residual pipeline: R_vol is contracted by the fused
    // assembleRHS together with the gathered face flux.
    o_volScratch_ = mem_.wrapOrMalloc(nEvm);
    // Reserved for the viscous path: gradients at quadrature points,
    // N_elem * N_vars * 2 * N_q^2. Allocated but never written in this stage.
    const occa::dim_t nGrad =
        static_cast<occa::dim_t>(geo_->numElements()) * N_vars_ * 2 * N_q2_;
    o_gradU_ = mem_.wrapOrMalloc(nGrad);

    // Time-step estimation scratch (element count; the host mirror is the
    // wrap source on unified backends).
    dt_elem_.assign(static_cast<std::size_t>(geo_->numElements()), Real(0));
    o_dtElem_ = mem_.wrapOrMalloc(dt_elem_.data(),
                                  static_cast<occa::dim_t>(dt_elem_.size()));
}

// ============================================================================
// Kernel compilation
// ============================================================================

occa::kernel DgField::buildKernel(const std::string &file,
                                  const std::string &name) {
    occa::json props;
#ifdef USE_FLOAT_PRECISION
    props["defines/Real"] = "float";
#else
    props["defines/Real"] = "double";
#endif
    props["defines/TILE_SIZE"] = tileSize_;
    // JIT define selecting the Riemann flux family. The preprocessor
    // removes the unused branches, so the compiled binary contains no
    // runtime flux-type branch (see surface_integral.okl).
    props["defines/FLUX_TYPE"] = cfg_.fluxTypeInt();
    props["defines/FLUX_LLF"]  = fluxTypeValue(FluxType::Llf);
    props["defines/FLUX_ROE"]  = fluxTypeValue(FluxType::Roe);
    props["defines/FLUX_ROEE"] = fluxTypeValue(FluxType::RoeE);
    props["defines/FLUX_STEGERWARMING"] =
        fluxTypeValue(FluxType::StegerWarming);
    props["defines/FLUX_VANLEER"] = fluxTypeValue(FluxType::VanLeer);
    // Search path for `#include` of shared device functions (llf.okl).
    props["okl/include_paths"].asArray().array().push_back(oklDir_);
    cmeles::finaliseKernelProps(props, device_);

    return device_.buildKernel(oklDir_ + "/" + file, name, props);
}

void DgField::buildKernels() {
    initModeCoeffs_  = buildKernel("init.okl", "initModeCoeffs");
    volumeIntegral_  = buildKernel("volume_integral.okl", "volumeIntegral");
    computeFaceFlux_ = buildKernel("surface_integral.okl", "computeFaceFlux");
    assembleRHS_     = buildKernel("assemble_rhs.okl", "assembleRHS");
    estimateDt_      = buildKernel("estimate_dt.okl", "estimateDt");
    // computeGradient_ is reserved (viscous path); the kernel source is not
    // built in this stage.
}

// ============================================================================
// Viscous path (reserved skeleton)
// ============================================================================

void DgField::computeGradient(occa::memory o_u, occa::memory o_grad) {
    // Reserved skeleton for the viscous path. The inviscid residual does
    // not need gradients, so this is not implemented in the current stage.
    (void)o_u;
    (void)o_grad;
    throw std::runtime_error(
        "DgField::computeGradient: viscous path not implemented yet");
}

// ============================================================================
// Kernel accessors
// ============================================================================

occa::kernel DgField::kernelInitModeCoeffs() const {
    return initModeCoeffs_;
}
occa::kernel DgField::kernelVolumeIntegral() const {
    return volumeIntegral_;
}
occa::kernel DgField::kernelComputeFaceFlux() const {
    return computeFaceFlux_;
}
occa::kernel DgField::kernelAssembleRHS() const {
    return assembleRHS_;
}
occa::kernel DgField::kernelEstimateDt() const {
    return estimateDt_;
}

// ============================================================================
// Initial conditions
// ============================================================================

void DgField::applyFreeStreamInitialCondition(occa::memory o_u) {
    // Free-stream conserved state from the configuration. Stored in a
    // member so that the wrapped handle stays valid on unified memory
    // backends (zero-copy) and on asynchronous GPU backends.
    const Real rho = cfg_.flowRho();
    const Real uu  = cfg_.flowU();
    const Real vv  = cfg_.flowV();
    const Real p   = cfg_.flowP();
    const Real gam = cfg_.gamma();
    const Real E = p / (gam - Real(1)) + Real(0.5) * rho * (uu * uu + vv * vv);
    free_state_  = {rho, rho * uu, rho * vv, E};

    occa::memory o_free = mem_.wrapOrMalloc(free_state_.data(), 4);
    initModeCoeffs_(static_cast<int>(geo_->numElements()), N_vars_, N_modes_,
                    N_q2_, o_free, geo_->o_lambdaWJ(), basis2D_->o_V2D(),
                    geo_->o_minv(), o_u);
}

void DgField::setInitialConditionNodal(occa::memory o_u,
                                       const VectorXr &nodal) {
    // Host-side L2 projection per element:
    //   u_hat = Minv * (V^T * diag(Lambda) * q_nodal)
    // per element and per conserved variable.
    const int N_elem  = geo_->numElements();
    const int Nq2     = N_q2_;
    const int N_modes = N_modes_;
    if (nodal.size() != N_elem * N_vars_ * Nq2) {
        throw std::invalid_argument(
            "DgField::setInitialConditionNodal: wrong nodal size");
    }

    const MatrixXr &V = basis2D_->vandermonde();
    VectorXr uhat(N_elem * N_vars_ * N_modes);

    for (int e = 0; e < N_elem; ++e) {
        const Eigen::Map<const MatrixXr> minv(geo_->massMatrixInverse().data() +
                                                  static_cast<size_t>(e) *
                                                      N_modes * N_modes,
                                              N_modes, N_modes);
        Eigen::Map<const VectorXr> lam(
            geo_->lambdaWJ().data() + static_cast<size_t>(e) * Nq2, Nq2);

        for (int var = 0; var < N_vars_; ++var) {
            Eigen::Map<const VectorXr> qn(
                nodal.data() + (static_cast<size_t>(e) * N_vars_ + var) * Nq2,
                Nq2);
            // b = V^T (Lambda qn)
            const VectorXr b = V.transpose() * (lam.asDiagonal() * qn);
            Eigen::Map<VectorXr>(uhat.data() +
                                     (static_cast<size_t>(e) * N_vars_ + var) *
                                         N_modes,
                                 N_modes) = minv * b;
        }
    }

    // Copy the modal coefficients into the caller's buffer. Use the OCCA
    // memory API directly (not DeviceMemoryManager::copyFromHost, which is
    // a no-op on unified-memory backends) so that the data is written on
    // both Serial/OpenMP and separate-memory backends.
    o_u.copyFrom(uhat.data(), uhat.size());
}

// ============================================================================
// Residual pipeline
// ============================================================================

void DgField::computeRHS(occa::memory o_u, occa::memory o_res) {
    const int N_elem = geo_->numElements();
    const int N_face = geo_->numFaces();

    // 1. Volume integral of the convective flux.
    volumeIntegral_(N_elem, N_vars_, N_modes_, N_q2_, o_u, basis2D_->o_V2D(),
                    basis2D_->o_dV2D_r(), basis2D_->o_dV2D_s(),
                    geo_->o_lambdaWJ(), geo_->o_Jinv11(), geo_->o_Jinv12(),
                    geo_->o_Jinv21(), geo_->o_Jinv22(), cfg_.gamma(),
                    o_volScratch_);

    // 2. Face flux: per-face accumulated numerical flux. The free-stream
    //    state provides the exterior state on boundary faces (KR = -1).
    //    Stored in a member (free_state_) so the wrapped handle stays valid
    //    on unified memory backends and asynchronous GPU backends.
    const Real rho = cfg_.flowRho();
    const Real uu  = cfg_.flowU();
    const Real vv  = cfg_.flowV();
    const Real p   = cfg_.flowP();
    const Real gam = cfg_.gamma();
    const Real E = p / (gam - Real(1)) + Real(0.5) * rho * (uu * uu + vv * vv);
    free_state_  = {rho, rho * uu, rho * vv, E};
    occa::memory o_free = mem_.wrapOrMalloc(free_state_.data(), 4);

    computeFaceFlux_(N_face, N_vars_, N_modes_, N_q_, o_u, o_faceVandermonde_,
                     o_faceWeights_, geo_->o_faceKL(), geo_->o_faceFL(),
                     geo_->o_faceKR(), geo_->o_faceFR(), geo_->o_faceNormals(),
                     geo_->o_faceJac(), o_free, gam, o_faceFlux_);

    // 3. Assembly: res = M^-1 (R_vol - G), with the surface contribution G
    //    gathered in-place from the face-flux slots inside the kernel
    //    (fused gather; no R_surf scratch round-trip).
    assembleRHS_(N_elem, N_vars_, N_modes_, o_volScratch_, o_faceFlux_,
                 geo_->o_elemFaces(), geo_->o_faceKL(), geo_->o_faceKR(),
                 geo_->o_minv(), o_res);
}

Real DgField::estimateDt(occa::memory o_u, Real cfl) {
    const int N_elem = geo_->numElements();
    estimateDt_(N_elem, N_vars_, N_modes_, N_q2_, cfg_.polynomialOrder(),
                cfg_.gamma(), cfl, o_u, basis2D_->o_V2D(), geo_->o_vertices(),
                o_dtElem_);

    // Host min-reduction (the estimate synchronises the loop anyway). On
    // unified backends o_dtElem_ aliases dt_elem_ directly.
    if (mem_.hasSeparateMemorySpace()) {
        occa::memory o = o_dtElem_;
        mem_.copyToHost(o, dt_elem_.data(), N_elem);
    }
    Real dt = std::numeric_limits<Real>::max();
    for (Real d : dt_elem_) {
        dt = std::min(dt, d);
    }
    return dt;
}
