/// @file SolutionWriter.cpp
/// @brief Implementation of the strategy A output writer.

#include "io/SolutionWriter.hpp"

#include <cstdio>
#include <filesystem>

#include "basis/BasisFunctions2D.hpp"
#include "common/Constants.hpp"
#include "config/Config.hpp"
#include "dg/DgField.hpp"
#include "dg/QuadratureCoords.hpp"
#include "io/HDF5Writer.hpp"

namespace {
/// @brief Zero-padded solution file name solution_<step>.h5.
std::string solutionFileName(long step) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "solution_%07ld.h5", step);
    return std::string(buf);
}
} // namespace

SolutionWriter::SolutionWriter(const Config &cfg, const DgField &field)
    : field_(field), gamma_(cfg.gamma()),
      polynomialOrder_(cfg.polynomialOrder()),
      directory_(cfg.outputDirectory()) {
    nElem_  = field.numElements();
    nModes_ = field.numModes();
    nq2_    = field.numQuadPoints();
    nTot_   = static_cast<size_t>(nElem_) * static_cast<size_t>(nq2_);

    modal_.assign(static_cast<size_t>(nElem_) * kNumVars2D *
                      static_cast<size_t>(nModes_),
                  Real(0));
    nodal_.resize(nq2_);
    for (auto *buf : {&rho_, &rhoU_, &rhoV_, &energy_, &u_, &v_, &p_}) {
        buf->assign(nTot_, Real(0));
    }

    // Contiguous per-component coordinates (the MatrixX2r columns of
    // quadraturePhysicalCoords are strided and not directly writable).
    const MatrixX2r xy = quadraturePhysicalCoords(field);
    coordsX_.resize(nTot_);
    coordsY_.resize(nTot_);
    for (size_t k = 0; k < nTot_; ++k) {
        coordsX_[k] = xy(static_cast<Eigen::Index>(k), 0);
        coordsY_[k] = xy(static_cast<Eigen::Index>(k), 1);
    }
}

void SolutionWriter::write(long step, Real time) {
    // Device -> host copy. Plain occa::memory::copyTo is used on purpose:
    // DeviceMemoryManager::copyToHost is a no-op on unified-memory
    // backends and would leave the buffer untouched there.
    field_.o_u().copyTo(modal_.data(), static_cast<occa::dim_t>(modal_.size()));

    // Modal -> quadrature-point projection per element and variable:
    // nodal = V * u_hat with V the (Nq2 x N_modes) Vandermonde matrix.
    const MatrixXr &V                        = field_.basis().vandermonde();
    std::vector<Real> *conserved[kNumVars2D] = {&rho_, &rhoU_, &rhoV_,
                                                &energy_};
    for (int e = 0; e < nElem_; ++e) {
        for (int var = 0; var < kNumVars2D; ++var) {
            const Eigen::Map<const VectorXr> uhat(
                modal_.data() + (static_cast<size_t>(e) * kNumVars2D +
                                 static_cast<size_t>(var)) *
                                    static_cast<size_t>(nModes_),
                nModes_);
            nodal_.noalias() = V * uhat;
            Real *dst = conserved[var]->data() + static_cast<size_t>(e) * nq2_;
            for (int q = 0; q < nq2_; ++q) {
                dst[q] = nodal_[q];
            }
        }
    }

    // Primitive variables from the conserved ones:
    // $u = \rho u / \rho$, $v = \rho v / \rho$, $p = (\gamma - 1)(E - \tfrac{1}{2} \rho (u^2 + v^2))$.
    for (size_t k = 0; k < nTot_; ++k) {
        u_[k] = rhoU_[k] / rho_[k];
        v_[k] = rhoV_[k] / rho_[k];
        p_[k] = (gamma_ - Real(1)) *
                (energy_[k] -
                 Real(0.5) * rho_[k] * (u_[k] * u_[k] + v_[k] * v_[k]));
    }

    const std::filesystem::path path =
        std::filesystem::path(directory_) / solutionFileName(step);
    HDF5Writer writer(path.string());
    writer.writeAttribute("time", time);
    writer.writeAttribute("step", static_cast<int>(step));
    writer.writeAttribute("order", polynomialOrder_);
    const std::vector<Real> *soa[] = {&rho_, &rhoU_, &rhoV_, &energy_,
                                      &u_,   &v_,    &p_};
    const char *const names[] = {"rho", "rho_u", "rho_v", "E", "u", "v", "p"};
    for (int i = 0; i < 7; ++i) {
        writer.writeDataset(names[i], soa[i]->data(), nTot_);
    }
    writer.writeGroupDataset("mesh", "x", coordsX_.data(), nTot_);
    writer.writeGroupDataset("mesh", "y", coordsY_.data(), nTot_);
    writer.flush();
}
