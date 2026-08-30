/// @file DgField.hpp
/// @brief Discontinuous Galerkin field module: computes the semi-discrete
///        right-hand side $\mathcal{R}(u_f)$ of the 2D inviscid Euler
///        equations.
///
/// Reference: docs/tech_docs/governed_equations_and_DG_field.md.
///
/// The module holds the mesh, geometry and basis pre-computation, the
/// device-resident field arrays and the OCCA kernels that assemble
///
/// $$
///   \mathcal{R}(u_f) = M^{-1}\big(R_{\text{vol}} - R_{\text{surf}}\big),
/// $$
///
/// where $R_{\text{vol}}$ is the volume integral of the convective flux
/// and $R_{\text{surf}}$ the surface integral of the numerical flux
/// (Riemann solver). The viscous path is not implemented yet:
/// `computeGradient` is reserved as an interface skeleton.
///
/// Data layout (see the documentation):
///   u_f: [elem][var][mode]  (element-major, EVM)
///   quadrature points:      k = j * N_q + i  (column-major grid order)
///   face points:            one set of N_q Gauss-Legendre points per face
///
/// The flux family is selected at *JIT compile time* via the `FLUX_TYPE`
/// define (an integer FluxType), so the kernel contains no runtime
/// branching on the flux type — the preprocessor removes the unused
/// branches before the binary is produced.

#pragma once

#include <occa.hpp>

#include <memory>
#include <string>

#include "common/Types.hpp"
#include "config/Config.hpp"
#include "core/DeviceMemoryManager.hpp"
#include "riemann/Riemann.hpp"

class BasisFunctions1D;
class BasisFunctions2D;
class Mesh;
class MeshGeometry;

/// @brief Discontinuous Galerkin field for the 2D inviscid Euler equations.
class DgField
{
public:
    /// @brief Construct the field module.
    /// @param cfg     Configuration (basis order, mesh, flux, gas).
    /// @param device  OCCA device (lifetime must exceed this object).
    /// @param mem     Device memory manager (lifetime must exceed this object).
    /// @param oklDir  Directory containing the .okl source files
    ///                (defaults to OCCA_OKL_DIR).
    DgField(const Config &cfg, occa::device &device, DeviceMemoryManager &mem,
            const std::string &oklDir = OCCA_OKL_DIR);

    ~DgField();

    /// @brief Build the mesh/geometry/basis and allocate all device memory.
    /// Must be called once before any computation.
    void setup();

    // ---- Sizes ----

    int numElements() const noexcept;
    int numVars() const noexcept
    {
        return kNumVars2D;
    }
    int numModes() const noexcept;
    int numQuadPoints() const noexcept;
    int numPoints1D() const noexcept;

    // ---- Reference data access ----

    const Mesh &mesh() const noexcept;
    const MeshGeometry &geometry() const noexcept;
    const BasisFunctions2D &basis() const noexcept;

    // ---- Device memory ----

    /// @brief Field (modal coefficients), $N_e \cdot N_v \cdot N_m$.
    occa::memory o_u() const
    {
        return o_u_;
    }

    /// @brief Residual, $N_e \cdot N_v \cdot N_m$.
    occa::memory o_res() const
    {
        return o_res_;
    }

    /// @brief Per-face accumulated modal flux,
    ///        $N_f \cdot N_v \cdot 2 \cdot N_m$ (side 0 = left, 1 = right).
    occa::memory o_faceFlux() const
    {
        return o_faceFlux_;
    }

    /// @brief Face Vandermonde matrices,
    ///        $2 \cdot 4 \cdot N_q \cdot N_m$ (side, local face, face point,
    ///        mode).
    occa::memory o_faceVandermonde() const
    {
        return o_faceVandermonde_;
    }

    /// @brief Face quadrature weights, size $N_q$.
    occa::memory o_faceWeights() const
    {
        return o_faceWeights_;
    }

    /// @brief Reserved gradient array (viscous path), unused in this stage.
    occa::memory o_gradU() const
    {
        return o_gradU_;
    }

    // ---- Initial conditions ----

    /// @brief Project a constant free-stream state into modal coefficients
    ///        via the per-element L2 projection (initModeCoeffs kernel).
    ///
    /// The free-stream state comes from the configuration ([flow] section).
    void applyFreeStreamInitialCondition(occa::memory o_u);

    /// @brief Project nodal values into modal coefficients on the host
    ///        (Eigen). Intended for tests and user-defined initial
    ///        conditions; the result is copied to \p o_u.
    ///
    /// @param nodal  Nodal values at quadrature points, size
    ///               $N_e \cdot N_v \cdot N_q^2$ (element-major, per-element
    ///               layout k = j*N_q + i).
    void setInitialConditionNodal(occa::memory o_u, const VectorXr &nodal);

    // ---- Residual ----

    /// @brief Compute the semi-discrete right-hand side.
    ///
    /// Full pipeline: volume integral -> face flux -> gather -> assemble:
    /// $$
    ///   \mathrm{res} = M^{-1}\big(R_{\text{vol}} - R_{\text{surf}}\big)
    /// $$
    ///
    /// @param o_u    Input modal coefficients.
    /// @param o_res  Output residual (may alias a distinct buffer).
    void computeRHS(occa::memory o_u, occa::memory o_res);

    // ---- Viscous path (reserved skeleton) ----

    /// @brief Reserved interface skeleton for the viscous gradient
    ///        computation. Not implemented in this stage; calling it throws.
    void computeGradient(occa::memory o_u, occa::memory o_grad);

    // ---- Kernel access (for tests) ----

    occa::kernel kernelInitModeCoeffs() const;
    occa::kernel kernelVolumeIntegral() const;
    occa::kernel kernelComputeFaceFlux() const;
    occa::kernel kernelGatherSurfaceRHS() const;
    occa::kernel kernelAssembleRHS() const;

private:
    // ---- Kernels ----

    /// @brief Build (or return cached) an OCCA kernel with the standard
    ///        JIT props: Real, TILE_SIZE and the FLUX_TYPE define used to
    ///        prune the unused flux branches at compile time.
    occa::kernel buildKernel(const std::string &file, const std::string &name);

    void buildKernels();

    // ---- Pre-computation ----

    /// @brief Build the face Vandermonde matrices and upload them.
    ///
    /// Vface[side][f][i][mode] = $\phi_{mode}(r(t_i), s(t_i))$ with
    /// (r, s) given by faceRefMapLeft(f) for side 0 (keyed by the left
    /// element's own face number) and faceRefMapRight(f) for side 1
    /// (keyed by the right element's own face number), $t_i$ the i-th
    /// face Gauss point. size 2 * 4 * N_q * N_modes.
    void buildFaceQuadratureData();

    /// @brief Allocate all device buffers (fields, face flux, gradients,
    ///        face Vandermonde, weights).
    void allocateDeviceMemory();

    // ---- Members ----

    const Config &cfg_;
    occa::device &device_;
    DeviceMemoryManager &mem_;
    std::string oklDir_;
    int tileSize_ = 256;

    int N_vars_  = kNumVars2D;
    int N_modes_ = 0;
    int N_q_     = 0;
    int N_q2_    = 0;

    std::unique_ptr<BasisFunctions1D> basis1D_;
    std::unique_ptr<BasisFunctions2D> basis2D_;
    std::unique_ptr<Mesh> mesh_;
    std::unique_ptr<MeshGeometry> geo_;

    // Persistent host copies of data wrapped on the device. On unified
    // memory backends wrapMemory aliases these buffers (zero-copy), so they
    // must outlive the occa::memory handles — i.e. be members rather than
    // temporaries local to buildFaceQuadratureData / computeRHS.
    std::vector<Real> face_vandermonde_;
    std::vector<Real> face_weights_;
    std::vector<Real> free_state_;

    // Device buffers.
    occa::memory o_u_;
    occa::memory o_res_;
    occa::memory o_faceFlux_;
    occa::memory o_gradU_; ///< Reserved for the viscous path.
    occa::memory o_faceVandermonde_;
    occa::memory o_faceWeights_;

    // Scratch buffers for the residual pipeline (R_vol and R_surf are both
    // needed simultaneously in assembleRHS).
    occa::memory o_volScratch_;
    occa::memory o_surfScratch_;

    // Cached kernels (lazy-compiled by buildKernels).
    occa::kernel initModeCoeffs_;
    occa::kernel computeGradient_; ///< Reserved skeleton.
    occa::kernel volumeIntegral_;
    occa::kernel computeFaceFlux_;
    occa::kernel gatherSurfaceRHS_;
    occa::kernel assembleRHS_;
};
