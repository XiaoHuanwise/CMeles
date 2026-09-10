/// @file CheckpointWriter.cpp
/// @brief Implementation of the strategy B checkpoint writer.

#include "io/CheckpointWriter.hpp"

#include <cstdio>
#include <filesystem>

#include "common/Constants.hpp"
#include "config/Config.hpp"
#include "dg/DgField.hpp"
#include "dg/QuadratureCoords.hpp"
#include "io/HDF5Writer.hpp"

namespace {
/// Conserved-variable names in the canonical order [rho, rho u, rho v, E].
const char *const kVarNames[kNumVars2D] = {"rho", "rho_u", "rho_v", "E"};

/// @brief Zero-padded checkpoint file name checkpoint_<step>.h5.
std::string checkpointFileName(long step) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "checkpoint_%07ld.h5", step);
    return std::string(buf);
}
} // namespace

CheckpointWriter::CheckpointWriter(const Config &cfg, const DgField &field)
    : field_(field), gamma_(cfg.gamma()),
      polynomialOrder_(cfg.polynomialOrder()),
      numPoints1D_(field.numPoints1D()),
      elementType_(cfg.meshSplitTriangles() ? "tri" : "quad"),
      directory_(cfg.outputDirectory()) {
    nElem_  = field.numElements();
    nModes_ = field.numModes();
    nq2_    = field.numQuadPoints();
    nTot_   = static_cast<size_t>(nElem_) * static_cast<size_t>(nq2_);

    modal_.assign(static_cast<size_t>(nElem_) * kNumVars2D *
                      static_cast<size_t>(nModes_),
                  Real(0));
    perVar_.assign(static_cast<size_t>(nElem_) * static_cast<size_t>(nModes_),
                   Real(0));

    const MatrixX2r xy = quadraturePhysicalCoords(field);
    coordsX_.resize(nTot_);
    coordsY_.resize(nTot_);
    for (size_t k = 0; k < nTot_; ++k) {
        coordsX_[k] = xy(static_cast<Eigen::Index>(k), 0);
        coordsY_[k] = xy(static_cast<Eigen::Index>(k), 1);
    }
}

void CheckpointWriter::writeMesh() {
    const std::filesystem::path path =
        std::filesystem::path(directory_) / "mesh.h5";
    HDF5Writer writer(path.string());
    // Reconstruction metadata: the offline converter rebuilds the basis
    // (and with it the Vandermonde matrix) from these attributes alone.
    writer.writeAttribute("dim", 2);
    writer.writeAttribute("N_elem", nElem_);
    writer.writeAttribute("order", polynomialOrder_);
    writer.writeAttribute("N_q", numPoints1D_);
    writer.writeAttribute("N_modes", nModes_);
    writer.writeAttribute("element_type", elementType_);
    writer.writeDataset("x", coordsX_.data(), nTot_);
    writer.writeDataset("y", coordsY_.data(), nTot_);
    writer.flush();
}

void CheckpointWriter::write(long step, Real time) {
    // Device -> host copy of the raw modal coefficients (plain copyTo:
    // valid on both unified and separate memory backends).
    field_.o_u().copyTo(modal_.data(), static_cast<occa::dim_t>(modal_.size()));

    const std::filesystem::path path =
        std::filesystem::path(directory_) / checkpointFileName(step);
    HDF5Writer writer(path.string());
    writer.writeAttribute("time", time);
    writer.writeAttribute("step", static_cast<int>(step));
    writer.writeAttribute("gamma", gamma_);
    for (int var = 0; var < kNumVars2D; ++var) {
        // Restage [e][var][m] into the per-variable (N_e, N_modes)
        // row-major block expected by the converter.
        for (int e = 0; e < nElem_; ++e) {
            const Real *src =
                modal_.data() + (static_cast<size_t>(e) * kNumVars2D +
                                 static_cast<size_t>(var)) *
                                    static_cast<size_t>(nModes_);
            Real *dst = perVar_.data() +
                        static_cast<size_t>(e) * static_cast<size_t>(nModes_);
            for (int m = 0; m < nModes_; ++m) {
                dst[m] = src[m];
            }
        }
        writer.writeDataset2D(kVarNames[var], perVar_.data(),
                              static_cast<size_t>(nElem_),
                              static_cast<size_t>(nModes_));
    }
    writer.flush();
}
